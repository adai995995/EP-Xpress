"""Direct GIN between nodes, GPU-initiated host staging within each node.

Only bootstrap/teardown exchange CPU metadata. Per-operation routes, counts and
inverse maps stay on GPU. All ranks enqueue phases in the same order. Buffers
are reclaimed at the next phase's device barrier, after prior readers finish.
"""

from dataclasses import asdict, dataclass
import ctypes
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import uuid

import torch

from .comm import NCCLCommHandle
from .context import Context, DispatchBatch, RouteHandle
from .host_sm import _MappedArena, _allocate_producer_arenas
from . import gin_native


class MultiNodeWorkspace:
    """Fixed-capacity storage borrowed until the next dispatch into it.

    Enqueue every consumer on the combine/wait stream before reusing storage.
    Graphs retain their workspaces and must be replayed serially per context.
    """

    def __init__(self, context, capacity, tokens, fused):
        self.context_id = context._id
        self.recv_capacity, self.local_tokens = capacity, tokens
        self.fused_combine = fused  # API compatibility; this path always reduces once
        self.busy = False
        self.generation = 0
        self.consumers = []
        self.allocation_stream = torch.cuda.current_stream(context.device)
        h, k = context.config.hidden_size, context.config.topk
        options = {"device": context.device}
        self.hidden = torch.empty((capacity, h), dtype=torch.bfloat16, **options)
        self.local_ids = torch.empty((capacity, k), dtype=torch.int64, **options)
        self.weights = torch.empty((capacity, k), dtype=torch.float32, **options)
        self.inverse = torch.empty((capacity, 2), dtype=torch.int64, **options)
        self.count = torch.empty(1, dtype=torch.int32, **options)
        self.expected_ranks = torch.empty((tokens, context.world_size), dtype=torch.uint8, **options)
        self.output = torch.empty((tokens, h), dtype=torch.bfloat16, **options)

    def record_stream(self, stream):
        for value in (self.hidden, self.local_ids, self.weights, self.inverse,
                      self.count, self.expected_ranks, self.output):
            value.record_stream(stream)


@dataclass
class _Route:
    inverse: object
    count: object
    expected_ranks: object
    workspace: object
    active_channels: int
    generation: int


class _Pending:
    def __init__(self, value, event, workspace, device):
        self.value, self.event, self.workspace, self.device = value, event, workspace, device
        self.generation = workspace.generation

    def wait(self, stream=None, timeout=None):
        if timeout is not None:
            raise ValueError("GIN wait is a device stream dependency; no host timeout")
        if self.workspace.generation != self.generation:
            raise RuntimeError("borrowed workspace ticket expired")
        consumer = stream or torch.cuda.current_stream(self.device)
        if consumer.device != self.device:
            raise ValueError("completion stream belongs to another device")
        consumer.wait_event(self.event)
        self.workspace.record_stream(consumer)
        if consumer not in self.workspace.consumers:
            self.workspace.consumers.append(consumer)
        return self.value

    @property
    def done(self):
        return self.event.query()


class MultiNodeContext(Context):
    """BF16 dynamic EP using one borrowed NCCL communicator.

    The initial direct implementation uses batch completion barriers and bounded
    source-partitioned registered buffers. It makes no zero-copy or overlap claim.
    It does not reuse routing decisions across operations or graph replays.
    """

    def __init__(self, nccl_comm, expert_to_rank, config, device=None, shared_directory=None):
        if not isinstance(nccl_comm, NCCLCommHandle):
            raise TypeError("Context requires an upper-layer NCCLCommHandle")
        self.nccl_comm_handle = nccl_comm
        self.rank, self.world_size = nccl_comm.rank, nccl_comm.world_size
        if not 2 <= self.world_size <= 256:
            raise ValueError("GIN supports 2 through 256 ranks")
        self.device = nccl_comm.device
        if device is not None:
            chosen = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            if chosen.type != "cuda" or (chosen.index is not None and chosen != self.device):
                raise ValueError("Context and NCCL communicator must use the same CUDA device")
        self.config = config
        self.expert_to_rank = tuple(expert_to_rank)
        if (not self.expert_to_rank or config.topk > len(self.expert_to_rank) or
            any(isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < self.world_size
                for r in self.expert_to_rank)):
            raise ValueError("invalid expert placement or topk")
        if config.diagnostics:
            raise NotImplementedError("host-ring diagnostics do not describe the GIN transport")
        self._id = uuid.uuid4().hex
        self._counter = 0
        self._closed = False
        self._lock = threading.RLock()
        self._cuda_graph_enabled = False
        self._capture_ready = False
        self._outstanding = set()
        self._root = None
        self._arenas = []
        self._runtime = None
        self.effective_channels = config.channels
        with torch.cuda.device(self.device):
            extension = gin_native.load()
            self._work_stream = torch.cuda.Stream(device=self.device)
            self._last_stream = torch.cuda.current_stream(self.device)
            # boot_id is shared by containers on one host, unlike container hostname.
            boot = Path("/proc/sys/kernel/random/boot_id")
            identity = boot.read_text().strip() if boot.exists() else socket.gethostname()
            signature = (identity, asdict(config), self.expert_to_rank,
                         str(torch.cuda.get_device_properties(self.device).uuid))
            signatures = nccl_comm.all_gather_object(signature)
            if any(s[1:3] != signature[1:3] for s in signatures):
                raise ValueError("ranks must use matching configuration and expert placement")
            if len({s[3] for s in signatures}) != self.world_size:
                raise ValueError("each rank must use a distinct CUDA device")
            self.local_ranks = tuple(i for i, s in enumerate(signatures) if s[0] == identity)
            if len(self.local_ranks) == self.world_size:
                raise ValueError("internode_transport=gin requires multiple hosts")
            local_indices = {rank: i for i, rank in enumerate(self.local_ranks)}
            local_map = [local_indices.get(i, -1) for i in range(self.world_size)]
            h, k, t = config.hidden_size, config.topk, config.max_tokens_per_rank
            host_bytes = config.gin_producer_bytes(len(self.local_ranks))
            gpu_bytes = config.gin_registered_bytes(self.world_size)
            if self.world_size * t >= (1 << 31):
                raise ValueError("aggregate token capacity must fit int32")
            if (host_bytes * len(self.local_ranks) > config.host_pool_budget_bytes or
                    gpu_bytes > config.gin_gpu_budget_bytes):
                raise ValueError(f"GIN staging requires {host_bytes * len(self.local_ranks)} node host bytes "
                                 f"and {gpu_bytes} GPU bytes per rank; increase budgets or reduce token capacity")
            leader = self.local_ranks[0]
            # The root is node-local even if the optional parent is shared storage.
            root = tempfile.mkdtemp(prefix="ep-xpress-gin-", dir=shared_directory or "/dev/shm") if self.rank == leader else None
            roots = nccl_comm.all_gather_object(root)
            self._root = Path(roots[leader])
            _allocate_producer_arenas(self._root, self.rank, {"hybrid": host_bytes})
            nccl_comm.barrier()
            pointers = [0] * self.world_size
            for rank in self.local_ranks:
                arena = _MappedArena(self._root / f"hybrid-{rank}.arena", host_bytes)
                self._arenas.append(arena)
                pointers[rank] = arena.address
            local_experts, counts = [], [0] * self.world_size
            for owner in self.expert_to_rank:
                local_experts.append(counts[owner])
                counts[owner] += 1
            symbol = ctypes.cast(nccl_comm._library.lib.ncclAllGather, ctypes.c_void_p).value
            self._runtime = extension.GinRuntime(nccl_comm.get(), symbol, h, k, t,
                config.gin_contexts, config.timeout_seconds, config.gin_gpu_budget_bytes,
                config.host_pool_budget_bytes, local_map, pointers, list(self.expert_to_rank), local_experts)
            self.transport_info = dict(self._runtime.info())
            if (self.transport_info["gpu_bytes"] != gpu_bytes or
                    self.transport_info["host_bytes_per_producer"] != host_bytes):
                raise RuntimeError("native GIN staging layout differs from Config budgeting")
            nccl_comm.barrier()

    def _stream(self, stream, direct):
        if self._closed:
            raise RuntimeError("Context is closed")
        producer = stream or torch.cuda.current_stream(self.device)
        if producer.device != self.device:
            raise ValueError("producer stream belongs to another device")
        capturing = torch.cuda.is_current_stream_capturing()
        if self._cuda_graph_enabled and not direct:
            raise RuntimeError("CUDA Graph context requires direct-stream operations")
        if capturing:
            if not (direct and self._cuda_graph_enabled and self._capture_ready and
                    producer == torch.cuda.current_stream(self.device)):
                raise RuntimeError("prepare_graph_capture and direct capture stream are required")
            return producer, True
        execution = producer if direct else self._work_stream
        for dependency in (self._last_stream, producer):
            if execution != dependency:
                execution.wait_stream(dependency)
        self._last_stream = execution
        self._capture_ready = False
        return execution, False

    def _tensor(self, value, dtype, shape, name):
        if not isinstance(value, torch.Tensor) or value.device != self.device:
            raise ValueError(name + " must be a tensor on the context device")
        if value.dtype != dtype or tuple(value.shape) != tuple(shape):
            raise ValueError(name + " has invalid shape or dtype")
        if value.requires_grad:
            raise ValueError("EP-Xpress accepts inference tensors only")
        return value.contiguous()

    def create_workspace(self, recv_capacity, local_tokens, *, fused_combine=False):
        for n, limit in ((recv_capacity, self.world_size * self.config.max_tokens_per_rank),
                         (local_tokens, self.config.max_tokens_per_rank)):
            if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= limit:
                raise ValueError("invalid workspace capacity")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("allocate workspaces before capture")
            return MultiNodeWorkspace(self, recv_capacity, local_tokens, fused_combine)

    def _dispatch(self, x, topk_ids, topk_weights, tag=None, stream=None, *, recv_capacity=None,
                  active_channels=None, fused_combine=False, workspace=None, direct=False):
        with self._lock, torch.cuda.device(self.device):
            execution, capturing = self._stream(stream, direct)
            n = x.shape[0]
            capacity = ((workspace.recv_capacity if isinstance(workspace, MultiNodeWorkspace)
                         else self.world_size * self.config.max_tokens_per_rank)
                        if recv_capacity is None else recv_capacity)
            if n > self.config.max_tokens_per_rank:
                raise ValueError("local token capacity exceeded")
            if len(self._outstanding) >= self.config.max_pending_operations:
                raise RuntimeError("combine earlier handles before submitting more dispatches")
            if workspace is None:
                if capturing:
                    raise RuntimeError("capture requires a warmed workspace")
                with torch.cuda.stream(execution):
                    workspace = self.create_workspace(capacity, n, fused_combine=fused_combine)
            if (not isinstance(workspace, MultiNodeWorkspace) or
                    workspace.context_id != self._id or workspace.recv_capacity != capacity or
                    workspace.local_tokens != n or workspace.busy):
                raise ValueError("workspace ownership, capacity or lifetime mismatch")
            if capturing and workspace.generation == 0:
                raise RuntimeError("warm the workspace before capture")
            if not capturing:
                for consumer in [workspace.allocation_stream] + workspace.consumers:
                    if consumer != execution:
                        execution.wait_stream(consumer)
                workspace.record_stream(execution)
            workspace.consumers.clear()
            with torch.cuda.stream(execution):
                hidden = self._tensor(x, torch.bfloat16, (n, self.config.hidden_size), "hidden")
                if topk_ids.dtype not in (torch.int32, torch.int64):
                    raise ValueError("topk_ids must be int32 or int64")
                ids = self._tensor(topk_ids, topk_ids.dtype, (n, self.config.topk), "topk_ids").to(torch.int64)
                weights = self._tensor(topk_weights, torch.float32, (n, self.config.topk), "topk_weights")
                self._runtime.dispatch(hidden.data_ptr(), ids.data_ptr(), weights.data_ptr(), n, capacity,
                    workspace.hidden.data_ptr(), workspace.local_ids.data_ptr(), workspace.weights.data_ptr(),
                    workspace.inverse.data_ptr(), workspace.count.data_ptr(), workspace.expected_ranks.data_ptr(),
                    execution.cuda_stream)
                if not capturing:
                    for value in (hidden, ids, weights):
                        value.record_stream(execution)
            self._counter += 1
            workspace.generation += 1
            workspace.busy = True
            route = _Route(workspace.inverse, workspace.count, workspace.expected_ranks,
                           workspace, active_channels or self.effective_channels, workspace.generation)
            handle = RouteHandle(self._id, self._counter, route, n, capacity, tag)
            self._outstanding.add(handle.operation)
            batch = DispatchBatch(workspace.hidden, workspace.local_ids, workspace.weights, handle, workspace.count)
            if direct:
                if not capturing:
                    workspace.consumers.append(execution)
                return batch
            event = torch.cuda.Event()
            event.record(execution)
            return _Pending(batch, event, workspace, self.device)

    def dispatch_on_stream(self, x, topk_ids, topk_weights, tag=None, stream=None, **options):
        return self._dispatch(x, topk_ids, topk_weights, tag, stream, direct=True, **options)

    def dispatch_async(self, x, topk_ids, topk_weights, tag=None, stream=None, **options):
        return self._dispatch(x, topk_ids, topk_weights, tag, stream, direct=False, **options)

    def _combine(self, partial, handle, tag, stream, weights_applied, direct):
        with self._lock, torch.cuda.device(self.device):
            if (not isinstance(handle, RouteHandle) or handle._context_id != self._id or
                    handle._consumed or handle.operation not in self._outstanding):
                raise ValueError("combine requires an unconsumed handle from this context")
            if not weights_applied:
                raise ValueError("expert partials must already contain routing weights")
            if tag != handle.tag:
                raise ValueError("combine tag differs from dispatch")
            execution, capturing = self._stream(stream, direct)
            plan, n, capacity = handle._plan, handle.local_token_count, handle.received_count
            workspace = plan.workspace
            if workspace.generation != plan.generation:
                raise RuntimeError("dispatch workspace was reused before combine")
            with torch.cuda.stream(execution):
                partial = self._tensor(partial, torch.bfloat16, (capacity, self.config.hidden_size), "partial")
                self._runtime.combine(partial.data_ptr(), plan.inverse.data_ptr(), plan.count.data_ptr(),
                    plan.expected_ranks.data_ptr(), workspace.output.data_ptr(), n, capacity, execution.cuda_stream)
                if not capturing:
                    partial.record_stream(execution)
                    workspace.record_stream(execution)
                    if execution not in workspace.consumers:
                        workspace.consumers.append(execution)
            handle._consumed = True
            self._outstanding.remove(handle.operation)
            workspace.busy = False
            if direct:
                return workspace.output
            event = torch.cuda.Event()
            event.record(execution)
            return _Pending(workspace.output, event, workspace, self.device)

    def combine_on_stream(self, partial, handle, tag=None, stream=None, weights_applied=True):
        return self._combine(partial, handle, tag, stream, weights_applied, True)

    def combine_async(self, partial, handle, tag=None, stream=None, weights_applied=True):
        return self._combine(partial, handle, tag, stream, weights_applied, False)

    def enable_cuda_graph(self):
        with self._lock:
            if self._counter or self._closed or self._cuda_graph_enabled:
                raise RuntimeError("enable CUDA Graph once, before submitting operations")
            self._cuda_graph_enabled = True

    def prepare_graph_capture(self):
        with self._lock, torch.cuda.device(self.device):
            if not self._cuda_graph_enabled or self._closed or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("an enabled graph context outside capture is required")
            self._last_stream.synchronize()
            self._capture_ready = True

    def prepare_graph_replay(self, stream=None):
        with self._lock, torch.cuda.device(self.device):
            if not self._cuda_graph_enabled or self._closed or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("an enabled graph context outside capture is required")
            execution = stream or torch.cuda.current_stream(self.device)
            if execution.device != self.device:
                raise ValueError("replay stream belongs to another device")
            if execution != self._last_stream:
                execution.wait_stream(self._last_stream)
            self._last_stream = execution
            self._capture_ready = False

    def check_errors(self):
        self._last_stream.synchronize()

    def close(self):
        with self._lock, torch.cuda.device(self.device):
            if self._closed:
                return
            if self._outstanding:
                raise RuntimeError("combine outstanding handles before closing")
            self._last_stream.synchronize()
            self.nccl_comm_handle.barrier()
            self._runtime.close()
            for arena in self._arenas:
                arena.close()
            self.nccl_comm_handle.barrier()
            if self.rank == self.local_ranks[0]:
                shutil.rmtree(self._root)
            self._closed = True
