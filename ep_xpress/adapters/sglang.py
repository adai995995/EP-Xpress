"""Experimental eager BF16 adapter for the pinned SGLang source revision.

The patch under adapters/sglang registers this as its own A2A backend. The
communication library itself does not depend on SGLang or its global state.
"""

from collections import deque
from dataclasses import dataclass
import logging
import os
from typing import NamedTuple

import torch
import torch.distributed as dist

from ep_xpress import Config, Context
from ep_xpress.adapters.sglang_config import (
    contiguous_expert_map,
    select_active_channels,
    validate_valid_count_runner,
    validate_runner_config,
)
from sglang.srt.layers.dp_attention import get_global_dp_buffer_len, get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher, DispatchOutputFormat
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel


_contexts = {}
_logger = logging.getLogger(__name__)


def close_contexts():
    """Close shared contexts on every EP rank, after all forwards complete.

    Call before destroying process groups. No distributed work is performed
    by an atexit handler, since peer processes may already have terminated.
    """
    for context in list(_contexts.values()):
        context.close()
    _contexts.clear()


class HostSMDispatchOutput(NamedTuple):
    """Standard expert input with a GPU count for the EP-Xpress runner."""

    hidden_states: object
    hidden_states_scale: object
    topk_output: object
    valid_count: object
    hidden_states_pre_quant: object = None

    @property
    def format(self):
        return DispatchOutputFormat.STANDARD


@dataclass(frozen=True)
class _Ticket:
    handle: object
    tag: object
    rows: int


class HostEPDispatcher(BaseDispatcher):
    """Standard/Triton input contract with EP-Xpress communication.

    SGLang's StandardCombineInput has no handle field. The eager adapter
    therefore accepts exactly one outstanding operation per layer; an
    immutable operation ticket survives until its matching combine. TBO/SBO
    must use a future interface carrying tickets through expert execution.
    """

    def __init__(self, moe_runner_config):
        super().__init__()
        parallel = get_parallel()
        size = parallel.moe_ep_size
        if not (parallel.tp_size == parallel.attn_dp_size == size):
            raise ValueError("EP-Xpress requires attention DP = TP = EP")
        validate_runner_config(moe_runner_config, world_size=size, bf16=torch.bfloat16)
        self._group = parallel.moe_ep_group.cpu_group
        if str(dist.get_backend(self._group)) != "gloo":
            raise ValueError("EP-Xpress requires a Gloo metadata group")
        if dist.get_world_size(self._group) != size:
            raise ValueError("EP-Xpress metadata group does not match the EP group")
        if dist.get_rank(self._group) != parallel.moe_ep_rank:
            raise ValueError("EP-Xpress requires matching framework and metadata rank ordering")
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._world_size = size
        transport = os.environ.get("EP_XPRESS_TRANSPORT", "host_ce")
        diagnostics = os.environ.get("EP_XPRESS_DIAGNOSTICS", "0")
        if diagnostics not in ("0", "1"):
            raise ValueError("EP_XPRESS_DIAGNOSTICS must be 0 or 1")
        decode_workspace = os.environ.get("EP_XPRESS_DECODE_WORKSPACE", "1")
        fused_decode = os.environ.get("EP_XPRESS_FUSED_DECODE", "1")
        if decode_workspace not in ("0", "1") or fused_decode not in ("0", "1"):
            raise ValueError("EP_XPRESS_DECODE_WORKSPACE and EP_XPRESS_FUSED_DECODE must be 0 or 1")
        self._reuse_decode_workspace = transport == "host_sm" and decode_workspace == "1"
        self._fused_decode = transport == "host_sm" and fused_decode == "1"
        self._config = Config(
            hidden_size=moe_runner_config.hidden_size,
            topk=moe_runner_config.top_k,
            max_tokens_per_rank=int(os.environ.get("EP_XPRESS_MAX_TOKENS", "16384")),
            chunk_bytes=int(os.environ.get("EP_XPRESS_CHUNK_BYTES", str(2 << 20))),
            slots=int(os.environ.get("EP_XPRESS_SLOTS", "2")),
            channels=int(os.environ.get("EP_XPRESS_CHANNELS", "4")),
            host_pool_budget_bytes=int(os.environ.get("EP_XPRESS_HOST_BUDGET_BYTES", str(512 << 20))),
            transport=transport,
            fatal_errors=transport == "host_sm",
            diagnostics=diagnostics == "1",
            combine_frame_rows=(int(os.environ.get("EP_XPRESS_COMBINE_FRAME_ROWS", "16"))
                                if transport == "host_sm" else 0),
            decode_max_tokens=(int(os.environ.get("EP_XPRESS_DECODE_MAX_TOKENS", "64"))
                               if transport == "host_sm" else 0),
        )
        adaptive = os.environ.get("EP_XPRESS_ADAPTIVE_CHANNELS", "1")
        valid_count = os.environ.get("EP_XPRESS_VALID_COUNT", "1")
        if adaptive not in ("0", "1") or valid_count not in ("0", "1"):
            raise ValueError("EP_XPRESS_ADAPTIVE_CHANNELS and EP_XPRESS_VALID_COUNT must be 0 or 1")
        self._adaptive_channels = adaptive == "1"
        self._use_valid_count = transport == "host_sm" and valid_count == "1"
        if self._use_valid_count:
            validate_valid_count_runner(moe_runner_config)
        self._prefill_channels = int(os.environ.get("EP_XPRESS_PREFILL_CHANNELS", "0"))
        self._decode_channels = int(os.environ.get("EP_XPRESS_DECODE_CHANNELS", "0"))
        for value in (self._prefill_channels, self._decode_channels):
            if not 0 <= value <= self._config.channels:
                raise ValueError("phase channel override must be 0 (auto) or within configured channels")
        self._expert_map = tuple(contiguous_expert_map(moe_runner_config.num_experts, size))
        self._context_key = (self._group, self._device, self._config, self._expert_map)
        self._resolved_context = None
        self._layer_id = moe_runner_config.layer_id
        self._sequence = 0
        self._tickets = deque()
        self._failed = False
        # Per-layer storage avoids aliasing another layer's residual/input.
        # Values are borrowed until this layer reuses the same exact shape;
        # SGLang queues all consumers on its forward stream before that reuse.
        self._decode_workspaces = {}

    def _context(self):
        # Lazy creation avoids allocating CUDA resources during weight setup.
        # All EP ranks execute the same layer sequence in the supported path.
        if self._resolved_context is not None:
            return self._resolved_context
        # Bootstrap itself allocates/registers memory and synchronizes once.
        # Guard only this first resolution; normal calls use the cached context
        # and the transport's existing per-submission capture check.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EP-Xpress currently requires eager execution")
        if self._context_key not in _contexts:
            _contexts[self._context_key] = Context(
                group=self._group,
                expert_to_rank=list(self._expert_map),
                config=self._config,
                device=self._device,
            )
            if self._config.transport == "host_sm":
                _logger.info(
                    "EP-Xpress host_sm channels: requested=%d effective=%d",
                    self._config.channels,
                    _contexts[self._context_key].effective_channels,
                )
        self._resolved_context = _contexts[self._context_key]
        return self._resolved_context

    def _check_eager(self):
        if self._failed:
            raise RuntimeError("EP-Xpress dispatcher failed; restart the worker group")
        # The transport validates capture before enqueueing. Avoid repeating
        # the same CUDA query here on every dispatch and combine.

    def dispatch(self, hidden_states, topk_output):
        self._check_eager()
        if self._tickets:
            raise RuntimeError("Combine the previous layer operation before dispatching again")
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError("EP-Xpress requires standard top-k IDs and weights")
        stream = torch.cuda.current_stream(self._device)
        tag = (f"layer/{self._layer_id}/operation/{self._sequence}"
               if self._config.diagnostics or self._config.transport == "host_ce" else None)
        try:
            dispatch_options = {}
            if self._config.transport == "host_sm":
                # The pinned scheduler computes this sum once per forward,
                # after MAX_LEN/SUM_LEN padding. Reuse that exact CPU bound
                # instead of validating and summing all rank counts per layer.
                # Each source's input capacity is still checked by the core.
                capacity = get_global_dp_buffer_len()
                is_extend = get_is_extend_in_batch()
                dispatch_options["recv_capacity"] = capacity
                dispatch_options["active_channels"] = select_active_channels(
                    capacity, is_extend=is_extend,
                    allocated=self._context().effective_channels,
                    adaptive=self._adaptive_channels,
                    prefill_override=self._prefill_channels,
                    decode_override=self._decode_channels,
                )
                # All ranks derive fusion from existing global CPU metadata.
                # Local token counts only select a local workspace shape.
                small_decode = (not is_extend and
                                0 <= capacity <= self._config.decode_max_tokens and
                                self._config.decode_max_tokens > 0)
                fused = small_decode and self._fused_decode
                dispatch_options["fused_combine"] = fused
                if small_decode and self._reuse_decode_workspace:
                    key = (capacity, hidden_states.shape[0], fused)
                    workspace = self._decode_workspaces.get(key)
                    if workspace is None:
                        workspace = self._context().create_workspace(
                            recv_capacity=capacity, local_tokens=hidden_states.shape[0],
                            fused_combine=fused,
                        )
                        # Bound storage for dynamic serving batches. Outstanding
                        # tensors retain their stream lifetimes on eviction.
                        if len(self._decode_workspaces) == 4:
                            self._decode_workspaces.pop(next(iter(self._decode_workspaces)))
                        self._decode_workspaces[key] = workspace
                    dispatch_options["workspace"] = workspace
            batch = self._context().dispatch_async(
                hidden_states,
                topk_output.topk_ids,
                topk_output.topk_weights,
                tag=tag,
                stream=stream,
                **dispatch_options,
            ).wait(stream=stream)
            rows = batch.hidden.shape[0]
            if self._config.transport == "host_ce" and batch.valid_count != rows:
                raise ValueError("The eager adapter requires unpadded dispatch output")
            # host_sm returns fixed capacity. Padding carries IDs=-1 and zero
            # weights; combine consumes only valid rows using its GPU handle.
            self._tickets.append(_Ticket(batch.handle, tag, rows))
            self._sequence += 1
            output_type = HostSMDispatchOutput if self._use_valid_count else StandardDispatchOutput
            return output_type(
                **({"valid_count": batch.valid_count} if self._use_valid_count else {}),
                hidden_states=batch.hidden,
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(
                    topk_weights=batch.weights,
                    topk_ids=batch.local_ids,
                    router_logits=batch.hidden.new_empty((rows, 0), dtype=torch.float32),
                ),
            )
        except Exception:
            self._failed = True
            raise

    def combine(self, combine_input):
        self._check_eager()
        if not self._tickets:
            raise RuntimeError("EP-Xpress combine requires a matching dispatch")
        if not isinstance(combine_input, StandardCombineInput):
            raise TypeError("EP-Xpress requires StandardCombineInput")
        ticket = self._tickets[0]
        stream = torch.cuda.current_stream(self._device)
        try:
            output = self._context().combine_async(
                combine_input.hidden_states,
                ticket.handle,
                tag=ticket.tag,
                stream=stream,
                weights_applied=True,
            ).wait(stream=stream)
        except Exception:
            self._failed = True
            raise
        self._tickets.popleft()
        return output
