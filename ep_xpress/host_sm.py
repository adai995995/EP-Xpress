"""GPU-initiated host staging with a CPU bootstrap and bounded shared queues.

No per-operation route tensor is read by the CPU. The public tensors have a
caller-provided upper-bound capacity, with padding IDs -1 and a device count.
All ranks must enqueue the same phase and handle order. CUDA Graph capture is
not supported. GPU stream dependencies and queue credits remain necessary.
"""

from dataclasses import asdict, dataclass
from datetime import timedelta
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import threading
import uuid

import torch
import torch.distributed as dist

from . import native
from .context import Context, DispatchBatch, Pending, RouteHandle


_ERRORS = {
    1: "invalid expert ID, routing weight or inverse map",
    2: "receive capacity exceeded",
    3: "GPU transport timed out",
    4: "a peer aborted the GPU transport",
    5: "ranks submitted different tags or dispatch handles",
}


class _MappedArena:
    """One private shared-file mapping; registration is managed by the context."""

    def __init__(self, path, size, channels=1):
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 32:
            raise ValueError("mapped queue channels must be in [1, 32]")
        if size % channels or (size // channels) % 4096:
            raise ValueError("mapped queue channel extents must be page aligned")
        self.channels = channels
        self.channel_stride = size // channels
        self.data = None
        self.fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            info = os.fstat(self.fd)
            if info.st_size != size or info.st_mode & 0o077:
                raise ValueError("mapped queue size or permissions are invalid")
            self.data = mmap.mmap(self.fd, size)
            self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.data))
            self.size = size
        except BaseException:
            self.close()
            raise

    def abort_code(self):
        # The first naturally aligned word is written only by the owning GPU.
        # This is an asynchronous error observation, not a completion fence.
        for channel in range(self.channels):
            gpu_error, cpu_abort = struct.unpack_from("=QQ", self.data, channel * self.channel_stride)
            if gpu_error or cpu_abort:
                return gpu_error or 4
        return 0

    def request_abort(self):
        # A separate CPU-owned word avoids CPU/GPU competing stores.
        for channel in range(self.channels):
            struct.pack_into("=Q", self.data, channel * self.channel_stride + 8, 1)

    def set_fatal_errors(self):
        """Set the immutable bootstrap flag in every lane before GPU use."""
        for channel in range(self.channels):
            struct.pack_into("=Q", self.data, channel * self.channel_stride + 16, 1)

    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@dataclass(frozen=True)
class _DeviceRoute:
    inverse: torch.Tensor
    count: torch.Tensor


def _operation_identity(phase, tag, dispatch_operation=None):
    """Fingerprint small host API metadata; never includes route tensors."""
    payload = json.dumps((phase, tag, dispatch_operation), sort_keys=True,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


class _DevicePending(Pending):
    """Enqueued CUDA completion; wait inserts a stream dependency only."""

    def __init__(self, context, value, completed):
        self._context = context
        self._value = value
        self._completed = completed
        self._device = context.device
        self._waited = False

    def wait(self, stream=None, timeout=None):
        if timeout is not None:
            raise ValueError("host_sm wait is nonblocking; use Context.check_errors for a host fence")
        self._context._check_aborted()
        with torch.cuda.device(self._device):
            consumer = stream or torch.cuda.current_stream(self._device)
            if consumer.device != self._device:
                raise ValueError("completion stream must belong to the context device")
            consumer.wait_event(self._completed)
            if isinstance(self._value, DispatchBatch):
                tensors = (self._value.hidden, self._value.local_ids, self._value.weights,
                           self._value.valid_count, self._value.handle.inverse_map)
            else:
                tensors = (self._value,)
            for tensor in tensors:
                tensor.record_stream(consumer)
        with self._context._lock:
            if not self._waited:
                self._context._unwaited -= 1
                self._waited = True
        return self._value

    @property
    def done(self):
        return self._completed.query()

    @property
    def metrics(self):
        # Byte counters require device instrumentation. Do not report inferred
        # values as measured PCIe traffic or reuse host_ce progress statistics.
        return {"transport": "host_sm", "channels": self._context.effective_channels}


class HostSMContext(Context):
    """Mapped-host backend selected by Context(config.transport='host_sm').

    Setup validates the group and placement once, maps all producer queues on
    each GPU, and creates device expert-placement tables. Dispatch and combine
    enqueue GPU work without a Gloo exchange or CPU routing/count readback.
    Call check_errors() at an experiment or application completion boundary.
    """

    def __init__(self, group, expert_to_rank, config, device=None, shared_directory=None):
        if not dist.is_initialized() or str(dist.get_backend(group)) != "gloo":
            raise ValueError("host_sm requires an initialized Gloo control group")
        if not torch.cuda.is_available():
            raise RuntimeError("host_sm requires CUDA")
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        if not 1 <= self.world_size <= 63:
            raise ValueError("supported EP group size is 1 through 63")
        self.config = config
        self.expert_to_rank = tuple(expert_to_rank)
        if not self.expert_to_rank or any(isinstance(r, bool) or not isinstance(r, int) or
                                         r < 0 or r >= self.world_size for r in self.expert_to_rank):
            raise ValueError("expert_to_rank must contain valid EP group ranks")
        if config.topk > len(self.expert_to_rank):
            raise ValueError("topk cannot exceed the number of experts")
        chosen = torch.cuda.current_device() if device is None else device
        chosen = torch.device("cuda", chosen) if isinstance(chosen, int) else torch.device(chosen)
        if chosen.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if chosen.index is None else chosen.index)
        if self.world_size * config.max_tokens_per_rank > (1 << 31) - 1:
            raise ValueError("total receive capacity must fit int32")
        with torch.cuda.device(self.device):
            native.load()
            self._channel_capacity = native.host_sm_max_channels()
        members = dist.get_process_group_ranks(group) if group is not None else list(range(dist.get_world_size()))
        self.group = dist.new_group(ranks=members, backend="gloo",
                                    timeout=timedelta(seconds=config.timeout_seconds),
                                    use_local_synchronization=True)
        self._id = uuid.uuid4().hex
        self._closed = False
        self._healthy = False
        self._failure = None
        self._counter = 0
        self._epochs = {"dispatch": 0, "combine": 0}
        self._lock = threading.RLock()
        self._unwaited = 0
        self._arenas = {"dispatch": [], "combine": []}
        self._registered = []
        self._pointers = {}
        self._root = None
        self._work_stream = None
        try:
            self._bootstrap(shared_directory)
        except BaseException:
            try:
                self._release_resources()
            finally:
                dist.destroy_process_group(self.group)
                self._closed = True
            raise

    def _bootstrap(self, shared_directory):
        signature = (socket.gethostname(), asdict(self.config), self.expert_to_rank,
                     str(torch.cuda.get_device_properties(self.device).uuid), self._channel_capacity)
        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(item[:3] != signature[:3] for item in signatures):
            raise ValueError("EP ranks must share one host, configuration and expert placement")
        if len({item[3] for item in signatures}) != self.world_size:
            raise ValueError("each EP rank must use a distinct CUDA device")
        # A cooperative grid must fit every device. This is bootstrap metadata,
        # never a per-layer collective. A device lacking cooperative launch uses
        # the explicitly reported single-lane path across the entire group.
        self.effective_channels = min(self.config.channels, *(max(1, item[4]) for item in signatures))
        self._channel_chunk_rows = self.config.channel_chunk_rows(self.effective_channels)
        with torch.cuda.device(self.device):
            self._arena_sizes = {phase: native.host_sm_arena_bytes(
                self.config.hidden_size, topk, self.config.slots, self._channel_chunk_rows,
                channels=self.effective_channels)
                for phase, topk in (("dispatch", self.config.topk), ("combine", 0))}
        if any(self._arena_sizes[phase] != self.config.mapped_arena_bytes(topk, self.effective_channels)
               for phase, topk in (("dispatch", self.config.topk), ("combine", 0))):
            raise RuntimeError("mapped queue allocation ABI mismatch")
        self._pool_bytes = sum(self._arena_sizes.values()) * self.world_size
        if self._pool_bytes > self.config.host_pool_budget_bytes:
            raise ValueError("negotiated mapped host queues exceed host_pool_budget_bytes")
        root_message = [None, None]
        if self.rank == 0:
            try:
                base = Path(shared_directory) if shared_directory else Path("/dev/shm")
                available = os.statvfs(base)
                if available.f_bavail * available.f_frsize < self._pool_bytes + (1 << 20):
                    raise ValueError("insufficient shared-memory space")
                self._root = Path(tempfile.mkdtemp(prefix="ep-xpress-sm-", dir=base))
                for phase, size in self._arena_sizes.items():
                    for peer in range(self.world_size):
                        fd = os.open(self._root / f"{phase}-{peer}.arena",
                                     os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                        try:
                            os.ftruncate(fd, size)
                            # Reserve backing storage now; avoid a later SIGBUS
                            # when registering a sparse file on a full tmpfs.
                            os.posix_fallocate(fd, 0, size)
                        finally:
                            os.close(fd)
                root_message[0] = str(self._root)
            except Exception:
                root_message[1] = "mapped queue allocation failed; check shared-memory capacity and permissions"
        leader = dist.get_global_rank(self.group, 0)
        dist.broadcast_object_list(root_message, src=leader, group=self.group)
        if root_message[1]:
            raise RuntimeError(root_message[1])
        self._root = Path(root_message[0])
        init_error = None
        try:
            with torch.cuda.device(self.device):
                self._work_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(self._work_stream):
                    for phase, size in self._arena_sizes.items():
                        pointers = []
                        for peer in range(self.world_size):
                            arena = _MappedArena(self._root / f"{phase}-{peer}.arena", size,
                                                 channels=self.effective_channels)
                            self._arenas[phase].append(arena)
                            if peer == self.rank and self.config.fatal_errors:
                                # Bootstrap-only flag: native errors also trap,
                                # surfacing at the framework's existing CUDA
                                # output fence even after the final MoE layer.
                                arena.set_fatal_errors()
                            pointers.append(native.register_mapped_host(arena.address, size))
                            self._registered.append(arena.address)
                        self._pointers[phase] = torch.tensor(pointers, dtype=torch.int64, device=self.device)
                    offsets = [0] * self.world_size
                    local_ids = []
                    for rank in self.expert_to_rank:
                        local_ids.append(offsets[rank])
                        offsets[rank] += 1
                    self._expert_rank = torch.tensor(self.expert_to_rank, dtype=torch.int32, device=self.device)
                    self._expert_local = torch.tensor(local_ids, dtype=torch.int32, device=self.device)
                    self._status = torch.zeros(1, dtype=torch.int32, device=self.device)
                self._work_stream.synchronize()
        except Exception:
            init_error = "mapped queue registration or CUDA initialization failed"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, init_error, group=self.group)
        if any(errors):
            raise RuntimeError(next(error for error in errors if error))
        self._healthy = True

    def _check_aborted(self):
        if self._failure is not None:
            raise RuntimeError("EP context is in a failed state") from self._failure
        for arenas in self._arenas.values():
            for arena in arenas:
                code = arena.abort_code()
                if code:
                    error = RuntimeError("host_sm: " + _ERRORS.get(code, "unknown GPU transport error"))
                    self._abort(error)
                    raise error

    def _abort(self, error):
        self._failure = error
        for arenas in self._arenas.values():
            if len(arenas) > self.rank:
                arenas[self.rank].request_abort()

    def check_errors(self):
        """Wait for enqueued work and reject a sticky GPU error.

        This explicit host fence belongs at a validation/request boundary, not
        between MoE layers. Submission also polls published host abort words
        without synchronizing CUDA, so a failed context cannot be reused.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Context is closed")
            with torch.cuda.device(self.device):
                self._work_stream.synchronize()
                code = int(self._status.item())
            if code:
                error = RuntimeError("host_sm: " + _ERRORS.get(code, "unknown GPU transport error"))
                self._abort(error)
                raise error
            self._check_aborted()

    check = check_errors

    def _begin(self, phase, stream):
        if self._closed:
            raise RuntimeError("Context is closed")
        self._check_aborted()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("host_sm eager backend cannot be captured in a CUDA Graph")
        if self._unwaited >= self.config.max_pending_operations:
            raise RuntimeError("pending ticket capacity exceeded; wait on earlier tickets")
        producer = stream or torch.cuda.current_stream(self.device)
        if producer.device != self.device:
            raise ValueError("producer stream must belong to the context device")
        self._epochs[phase] += 1
        self._counter += 1
        if self._epochs[phase] >= 1 << 31:
            raise RuntimeError("operation generation exhausted; recreate the context")
        self._work_stream.wait_stream(producer)
        return self._epochs[phase]

    def _finish(self, value):
        completed = torch.cuda.Event()
        completed.record(self._work_stream)
        self._unwaited += 1
        return _DevicePending(self, value, completed)

    def _validate_dispatch(self, x, ids, weights):
        if not all(isinstance(t, torch.Tensor) for t in (x, ids, weights)):
            raise ValueError("dispatch inputs must be tensors")
        if (x.device != self.device or x.dtype != torch.bfloat16 or x.ndim != 2 or
                not x.is_contiguous() or x.requires_grad):
            raise ValueError("x must be contiguous inference BF16 [tokens, hidden] on the context device")
        if x.shape[1] != self.config.hidden_size or x.shape[0] > self.config.max_tokens_per_rank:
            raise ValueError("x exceeds configured token capacity or hidden size")
        shape = (x.shape[0], self.config.topk)
        if tuple(ids.shape) != shape or tuple(weights.shape) != shape:
            raise ValueError("top-k tensors must have shape [local_tokens, topk]")
        if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
            raise ValueError("top-k IDs must be integer and weights must be FP32")
        if ids.device != self.device or weights.device != self.device or weights.requires_grad:
            raise ValueError("top-k tensors must be inference tensors on the context device")

    def dispatch_async(self, x, topk_ids, topk_weights, tag=None, stream=None, recv_capacity=None):
        """Enqueue routed dispatch without reading routes or counts on the CPU.

        recv_capacity is a CPU-known upper bound on received rows, normally the
        sum of scheduler-known input rows across ranks. If omitted, use the
        configured group maximum. A too-small bound poisons the context; it
        never silently drops rows. Output padding has IDs -1 and zero weights.
        """
        with self._lock, torch.cuda.device(self.device):
            try:
                self._validate_dispatch(x, topk_ids, topk_weights)
                capacity = self.world_size * self.config.max_tokens_per_rank if recv_capacity is None else recv_capacity
                if (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0 or
                        capacity > self.world_size * self.config.max_tokens_per_rank):
                    raise ValueError("recv_capacity must be an integer within the configured group capacity")
                identity = _operation_identity("dispatch", tag)
                epoch = self._begin("dispatch", stream)
                for tensor in (x, topk_ids, topk_weights):
                    tensor.record_stream(self._work_stream)
                with torch.cuda.stream(self._work_stream), torch.no_grad():
                    ids = topk_ids.contiguous()
                    weights = topk_weights.contiguous()
                    hidden = torch.empty((capacity, self.config.hidden_size), dtype=torch.bfloat16, device=self.device)
                    local_ids = torch.empty((capacity, self.config.topk), dtype=torch.int64, device=self.device)
                    weights_out = torch.empty((capacity, self.config.topk), dtype=torch.float32, device=self.device)
                    inverse = torch.empty((capacity, 2), dtype=torch.int64, device=self.device)
                    count = torch.empty(1, dtype=torch.int32, device=self.device)
                    native.host_sm_dispatch(
                        x, ids, weights, self._expert_rank, self._expert_local, self._pointers["dispatch"],
                        self.rank, epoch, self.config.slots, self._channel_chunk_rows,
                        max(1, math.ceil(self.config.timeout_seconds * 1000)),
                        hidden, local_ids, weights_out, inverse, count, self._status, identity=identity,
                        channels=self.effective_channels)
                    handle = RouteHandle(self._id, self._counter, _DeviceRoute(inverse, count),
                                         x.shape[0], capacity, tag)
                    return self._finish(DispatchBatch(hidden, local_ids, weights_out, handle, count))
            except BaseException as error:
                self._abort(error)
                raise

    def combine_async(self, partial, handle, tag=None, stream=None, weights_applied=True):
        with self._lock, torch.cuda.device(self.device):
            # Pure local API misuse is rejected before an epoch is submitted;
            # callers can correct the handle without poisoning healthy peers.
            if weights_applied is not True:
                raise ValueError("combine accepts already-weighted rank partials only")
            if not isinstance(handle, RouteHandle) or handle._context_id != self._id:
                raise ValueError("handle belongs to another context")
            if handle._consumed:
                raise ValueError("handle has already been consumed by combine")
            try:
                if (not isinstance(partial, torch.Tensor) or partial.device != self.device or
                        partial.dtype != torch.bfloat16 or partial.ndim != 2 or
                        tuple(partial.shape) != (handle.received_count, self.config.hidden_size) or
                        not partial.is_contiguous() or partial.requires_grad):
                    raise ValueError("partial must be contiguous inference BF16 [receive_capacity, hidden] on the context device")
                identity = _operation_identity("combine", tag, handle.operation)
                epoch = self._begin("combine", stream)
                for tensor in (partial, handle._plan.inverse, handle._plan.count):
                    tensor.record_stream(self._work_stream)
                with torch.cuda.stream(self._work_stream), torch.no_grad():
                    output = torch.empty((handle.local_token_count, self.config.hidden_size),
                                         dtype=torch.bfloat16, device=self.device)
                    remote = torch.empty((self.world_size, handle.local_token_count, self.config.hidden_size),
                                         dtype=torch.bfloat16, device=self.device)
                    native.host_sm_combine(
                        partial, handle._plan.inverse, handle._plan.count, self._pointers["combine"],
                        self.rank, epoch, self.config.slots, self._channel_chunk_rows,
                        max(1, math.ceil(self.config.timeout_seconds * 1000)), output, remote, self._status,
                        identity=identity, channels=self.effective_channels)
                    handle._consumed = True
                    return self._finish(output)
            except BaseException as error:
                self._abort(error)
                raise

    def _release_resources(self):
        # Every GPU uses its process's registered virtual mappings; never unpin
        # or unmap them while that process's transport stream can still access.
        try:
            with torch.cuda.device(self.device):
                if self._work_stream is not None:
                    self._work_stream.synchronize()
                for address in self._registered:
                    native.unregister_host(address)
                self._registered.clear()
            self._pointers.clear()
            for arenas in self._arenas.values():
                for arena in arenas:
                    arena.close()
                arenas.clear()
        finally:
            # A fatal CUDA trap requires a worker restart and can make drain or
            # unregister fail. Unlink still prevents persistent named tmpfs
            # leaks; existing mappings remain alive until the workers exit.
            if self.rank == 0 and self._root is not None:
                shutil.rmtree(self._root, ignore_errors=True)

    def close(self):
        """Drain local GPU work, rendezvous if healthy, then release mappings."""
        with self._lock:
            if self._closed:
                return
            try:
                if self._failure is None and self._healthy:
                    self.check_errors()
            except BaseException as error:
                self._abort(error)
                raise
            finally:
                self._closed = True
                try:
                    if self._healthy and self._failure is None:
                        dist.barrier(group=self.group)
                finally:
                    try:
                        self._release_resources()
                    finally:
                        dist.destroy_process_group(self.group)
