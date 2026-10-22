"""Framework-independent eager EP context and operation-scoped handles."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
import uuid

import torch
import torch.distributed as dist

from .config import Config
from . import native
from .ring import SharedRing
from .routing import build_routes
from .transport import exchange


@dataclass
class RouteHandle:
    """A single dispatch's inverse map. It cannot be reused for another route."""
    _context_id: str
    operation: int
    _plan: object = field(repr=False)
    local_token_count: int
    received_count: int
    tag: object = None
    _consumed: bool = field(default=False, repr=False)


@dataclass
class DispatchBatch:
    hidden: torch.Tensor
    local_ids: torch.Tensor
    weights: torch.Tensor
    handle: RouteHandle
    valid_count: int


class Pending:
    """Completion ticket. wait() may block the host for dynamic eager metadata."""
    def __init__(self, future, device):
        self._future = future
        self._device = device

    def wait(self, stream=None, timeout=None):
        value, completed, _stats = self._future.result(timeout=timeout)
        with torch.cuda.device(self._device):
            consumer = stream or torch.cuda.current_stream(self._device)
            if consumer.device != self._device:
                raise ValueError("completion stream must belong to the context device")
            consumer.wait_event(completed)
            tensors = (value.hidden, value.local_ids, value.weights) if isinstance(value, DispatchBatch) else (value,)
            for tensor in tensors:
                tensor.record_stream(consumer)
        return value

    @property
    def done(self):
        if not self._future.done():
            return False
        if self._future.exception() is not None:
            return True
        return self._future.result()[1].query()

    @property
    def metrics(self):
        """Return scalar operation metrics after host progress completes."""
        return dict(self._future.result()[2])


class Context:
    """One long-lived EP group, one device per process, shared host arenas.

    All ranks must construct contexts and submit dispatch/combine in the same
    order. group must be Gloo: it carries bootstrap and route metadata only.
    Activation and expert-result payloads always use the host-CE transport.
    """

    def __init__(self, group, expert_to_rank, config: Config, device=None, shared_directory=None):
        if not dist.is_initialized():
            raise RuntimeError("initialize a Gloo process group before constructing Context")
        if str(dist.get_backend(group)) != "gloo":
            raise ValueError("Context requires a Gloo control group")
        if not torch.cuda.is_available():
            raise RuntimeError("Context requires CUDA; routing and ring tests are CPU-only")
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if not 1 <= self.world_size <= 63:
            raise ValueError("supported EP group size is 1 through 63")
        self.config = config
        self.expert_to_rank = tuple(expert_to_rank)
        if not self.expert_to_rank or any(isinstance(r, bool) or not isinstance(r, int)
                                           or r < 0 or r >= self.world_size for r in self.expert_to_rank):
            raise ValueError("expert_to_rank must contain valid EP group ranks")
        if config.topk > len(self.expert_to_rank):
            raise ValueError("topk cannot exceed the number of experts")
        chosen = torch.cuda.current_device() if device is None else device
        if isinstance(chosen, int):
            chosen = torch.device("cuda", chosen)
        chosen = torch.device(chosen)
        if chosen.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if chosen.index is None else chosen.index)
        if config.pinned_working_set_bytes(self.world_size) > config.host_pool_budget_bytes:
            raise ValueError("shared rings and receive packs exceed host_pool_budget_bytes")
        # JIT compilation is a one-time setup step and may exceed the operation
        # timeout. PyTorch serializes concurrent builds through its cache lock.
        # Compile before creating the timeout-bounded communication group.
        with torch.cuda.device(self.device):
            native.load()
        # Isolate metadata ordering from framework collectives and bound failed
        # peer waits independently of the caller's potentially long timeout.
        members = dist.get_process_group_ranks(group) if group is not None else list(range(dist.get_world_size()))
        self.group = dist.new_group(ranks=members, backend="gloo",
                                    timeout=timedelta(seconds=config.timeout_seconds),
                                    use_local_synchronization=True)
        group = self.group
        self._id = uuid.uuid4().hex
        self._closed = False
        self._failure = None
        self._counter = 0
        self._lock = threading.Lock()
        self._capacity = threading.BoundedSemaphore(config.max_pending_operations)
        self._rings = {"dispatch": [], "combine": []}
        self._registered = []
        self._pack_pool = {}
        self._executor = None
        self._root = None
        self._healthy = False

        # Host identity is compared in memory only and never stored in reports.
        signature = (socket.gethostname(), asdict(config), self.expert_to_rank,
                     str(torch.cuda.get_device_properties(self.device).uuid))
        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=group)
        if any(item[:3] != signature[:3] for item in signatures):
            raise ValueError("all EP ranks must be on one host with matching configuration and expert placement")
        if len({str(item[3]) for item in signatures}) != self.world_size:
            raise ValueError("each EP rank must use a distinct CUDA device")

        leader = dist.get_global_rank(group, 0) if group is not None else 0
        with torch.cuda.device(self.device):
            self._work_stream = torch.cuda.Stream(device=self.device)
            self._export_stream = torch.cuda.Stream(device=self.device)
            self._import_stream = torch.cuda.Stream(device=self.device)

        root_message = [None, None]
        if self.rank == 0:
            try:
                base = Path(shared_directory) if shared_directory else Path("/dev/shm")
                if not base.is_dir():
                    raise ValueError("a shared-memory directory is required")
                available = os.statvfs(base)
                if available.f_bavail * available.f_frsize < config.pool_bytes(self.world_size) + (1 << 20):
                    raise ValueError("insufficient shared-memory space for configured ring pool")
                root = Path(tempfile.mkdtemp(prefix="ep-xpress-", dir=base))
                self._root = root
                for phase in self._rings:
                    for peer in range(self.world_size):
                        ring = SharedRing(root, peer, phase, config.slots, config.slot_bytes, create=True)
                        ring.close()
                root_message[0] = str(root)
            except Exception:
                root_message[1] = "shared ring initialization failed; check shared-memory capacity and permissions"
        dist.broadcast_object_list(root_message, src=leader, group=group)
        if root_message[1]:
            if self.rank == 0 and self._root:
                shutil.rmtree(self._root, ignore_errors=True)
            raise RuntimeError(root_message[1])
        self._root = Path(root_message[0])
        init_error = None
        try:
            for phase, rings in self._rings.items():
                for peer in range(self.world_size):
                    rings.append(SharedRing(self._root, peer, phase, config.slots, config.slot_bytes))
                address = rings[self.rank].payload_address(0)
                with torch.cuda.device(self.device):
                    native.register_host(address, config.slots * config.slot_bytes)
                self._registered.append(address)
            for peer in range(self.world_size):
                if peer != self.rank:
                    self._pack_pool[peer] = [torch.empty((config.chunk_rows, config.hidden_size),
                                                        dtype=torch.bfloat16, pin_memory=True)
                                             for _ in range(config.slots)]
        except Exception:
            init_error = "ring mapping or CUDA host registration failed"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, init_error, group=group)
        if any(errors):
            self._release_resources()
            raise RuntimeError(next(error for error in errors if error))
        self._healthy = True
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ep-xpress-progress")

    def _check_aborted(self):
        if self._failure is not None:
            raise RuntimeError("EP context is in a failed state") from self._failure
        if any(ring.aborted() for rings in self._rings.values() for ring in rings):
            raise RuntimeError("a peer aborted the EP context")

    def _abort(self, error):
        self._failure = error
        for rings in self._rings.values():
            if len(rings) > self.rank:
                try:
                    rings[self.rank].abort()
                except Exception:
                    pass

    def _metadata(self, operation, phase, tag, payload, error=None):
        try:
            tag_bytes = json.dumps(tag, sort_keys=True, separators=(",", ":")).encode()
            tag_hash = hashlib.sha256(tag_bytes).hexdigest()
        except (TypeError, ValueError):
            error = "tag must be JSON-serializable"
            tag_hash = None
        item = {"operation": operation, "phase": phase, "tag": tag_hash,
                "error": error, "payload": payload}
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, item, group=self.group)
        if any(entry["error"] for entry in gathered):
            raise ValueError(next(entry["error"] for entry in gathered if entry["error"]))
        if any((entry["operation"], entry["phase"], entry["tag"]) !=
               (operation, phase, tag_hash) for entry in gathered):
            raise ValueError("EP ranks submitted inconsistent operation order or tags")
        return [entry["payload"] for entry in gathered]

    def _submit(self, function, inputs, stream, tag, handle=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Context is closed")
            self._check_aborted()
            if handle is not None:
                if not isinstance(handle, RouteHandle) or handle._context_id != self._id:
                    raise ValueError("handle belongs to another context")
                if handle._consumed:
                    raise ValueError("handle has already been consumed by combine")
            if not self._capacity.acquire(blocking=False):
                raise RuntimeError("pending operation capacity exceeded; wait for earlier tickets")
            try:
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("EP-Xpress eager backend cannot be captured in a CUDA Graph")
                    producer = stream or torch.cuda.current_stream(self.device)
                    if producer.device != self.device:
                        raise ValueError("producer stream must belong to the context device")
                    ready = torch.cuda.Event()
                    ready.record(producer)
                    for tensor in inputs:
                        if isinstance(tensor, torch.Tensor) and tensor.device == self.device:
                            tensor.record_stream(self._work_stream)
                self._counter += 1
                operation = self._counter
                if operation >= 1 << 31:
                    raise RuntimeError("operation generation exhausted; recreate the context")
                future = self._executor.submit(self._execute, function, inputs, ready,
                                               operation, tag, handle)
                if handle is not None:
                    handle._consumed = True
                future.add_done_callback(lambda _future: self._capacity.release())
                return Pending(future, self.device)
            except BaseException:
                self._capacity.release()
                raise

    def _execute(self, function, inputs, ready, operation, tag, handle):
        began = time.monotonic()
        try:
            self._check_aborted()
            with torch.cuda.device(self.device), torch.cuda.stream(self._work_stream), torch.no_grad():
                self._work_stream.wait_event(ready)
                value, stats = function(inputs, operation, tag, handle)
                completed = torch.cuda.Event()
                completed.record(self._work_stream)
                stats["host_operation_seconds"] = time.monotonic() - began
                return value, completed, stats
        except BaseException as error:
            self._abort(error)
            raise

    def dispatch_async(self, x, topk_ids, topk_weights, tag=None, stream=None):
        return self._submit(self._dispatch, (x, topk_ids, topk_weights), stream, tag)

    def _dispatch(self, inputs, operation, tag, _handle):
        x, ids, weights = inputs
        payload, error = None, None
        try:
            if not all(isinstance(tensor, torch.Tensor) for tensor in inputs):
                raise ValueError("dispatch inputs must be tensors")
            if x.device != self.device or x.dtype != torch.bfloat16 or x.ndim != 2:
                raise ValueError("x must be a BF16 matrix on the context device")
            if not x.is_contiguous() or x.requires_grad:
                raise ValueError("x must be contiguous and inference-only")
            if x.shape[1] != self.config.hidden_size or x.shape[0] > self.config.max_tokens_per_rank:
                raise ValueError("x exceeds configured token capacity or hidden size")
            expected = (x.shape[0], self.config.topk)
            if tuple(ids.shape) != expected or tuple(weights.shape) != expected:
                raise ValueError("top-k tensors must have shape [local_tokens, topk]")
            if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
                raise ValueError("top-k ids must be integer and weights must be FP32")
            if ids.device != self.device or weights.device != self.device or weights.requires_grad:
                raise ValueError("top-k tensors must be inference tensors on the context device")
            payload = (ids.cpu().tolist(), weights.cpu().tolist())
        except ValueError as exc:
            error = str(exc)
        metadata = self._metadata(operation, "dispatch", tag, payload, error)
        plan = build_routes([item[0] for item in metadata], [item[1] for item in metadata],
                            self.expert_to_rank, self.world_size, self.config.topk)
        received, stats = exchange(self, x, plan.dispatch_dest_masks[self.rank],
                                   plan.dispatch_recv_indices[self.rank], operation, "dispatch")
        parts = []
        for source, indices in enumerate(plan.dispatch_recv_indices[self.rank]):
            if not indices:
                continue
            if source == self.rank:
                index = torch.tensor(indices, dtype=torch.int64, device=self.device)
                parts.append(x.index_select(0, index))
            else:
                parts.append(received[source])
        hidden = torch.cat(parts, dim=0) if parts else torch.empty(
            (0, self.config.hidden_size), dtype=torch.bfloat16, device=self.device)
        count = len(plan.rows[self.rank])
        local_ids = torch.tensor(plan.local_ids[self.rank], dtype=torch.int64,
                                 device=self.device).reshape(count, self.config.topk)
        weights_out = torch.tensor(plan.weights[self.rank], dtype=torch.float32,
                                   device=self.device).reshape(count, self.config.topk)
        handle = RouteHandle(self._id, operation, plan, x.shape[0], count, tag)
        stats["logical_remote_rows"] = sum(len(indices) for source, indices in
                                            enumerate(plan.dispatch_recv_indices[self.rank])
                                            if source != self.rank)
        return DispatchBatch(hidden, local_ids, weights_out, handle, count), stats

    def combine_async(self, partial, handle, tag=None, stream=None, weights_applied=True):
        if weights_applied is not True:
            raise ValueError("combine accepts already-weighted rank partials only")
        return self._submit(self._combine, (partial,), stream, tag, handle)

    def _combine(self, inputs, operation, tag, handle):
        partial = inputs[0]
        error = None
        if (not isinstance(partial, torch.Tensor) or partial.device != self.device
            or partial.dtype != torch.bfloat16 or partial.ndim != 2
            or tuple(partial.shape) != (handle.received_count, self.config.hidden_size)
            or not partial.is_contiguous() or partial.requires_grad):
            error = "partial must be contiguous BF16 [received_count, hidden_size] on the context device"
        metadata = self._metadata(operation, "combine", tag, handle.operation, error)
        if any(dispatch_id != handle.operation for dispatch_id in metadata):
            raise ValueError("combine handles refer to different dispatch operations")
        plan = handle._plan
        masks = [0 if owner == self.rank else 1 << owner for owner, _token in plan.rows[self.rank]]
        received, stats = exchange(self, partial, masks, plan.combine_recv_indices[self.rank],
                                   operation, "combine")
        shape = (handle.local_token_count, self.config.hidden_size)
        remote = torch.zeros(shape, dtype=torch.float32, device=self.device)
        local = torch.zeros(shape, dtype=torch.bfloat16, device=self.device)
        for expert_rank in range(self.world_size):
            token_ids = plan.combine_token_indices[self.rank][expert_rank]
            if not token_ids:
                continue
            indices = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
            if expert_rank == self.rank:
                rows = torch.tensor(plan.combine_recv_indices[self.rank][expert_rank],
                                     dtype=torch.int64, device=self.device)
                local.index_copy_(0, indices, partial.index_select(0, rows))
            else:
                native.add_partial(remote, received[expert_rank], indices, validate_indices=False)
        output = native.finish_combine(remote, local)
        stats["logical_remote_rows"] = sum(len(indices) for expert, indices in
                                            enumerate(plan.combine_recv_indices[self.rank])
                                            if expert != self.rank)
        return output, stats

    def _release_resources(self):
        for stream_name in ("_work_stream", "_export_stream", "_import_stream"):
            stream = getattr(self, stream_name, None)
            if stream is not None:
                stream.synchronize()
        with torch.cuda.device(self.device):
            for address in self._registered:
                native.unregister_host(address)
        self._registered.clear()
        self._pack_pool.clear()
        for rings in self._rings.values():
            for ring in rings:
                ring.close()
            rings.clear()
        if self.rank == 0 and self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)

    def close(self):
        """Collective on healthy contexts; drain work before releasing storage."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        aborted = self._failure is not None or any(
            ring.aborted() for rings in self._rings.values() for ring in rings)
        try:
            if self._healthy and not aborted:
                dist.barrier(group=self.group)
        finally:
            try:
                self._release_resources()
            finally:
                dist.destroy_process_group(self.group)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self._abort(exc_value)
        self.close()
