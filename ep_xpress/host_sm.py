"""GPU-initiated host staging with a CPU bootstrap and bounded shared queues.

No per-operation route tensor is read by the CPU. The public tensors have a
caller-provided upper-bound capacity, with padding IDs -1 and a device count.
All ranks must enqueue the same phase and handle order. CUDA Graph capture is
not supported. GPU stream dependencies and queue credits remain necessary.
"""

from dataclasses import asdict, dataclass
from datetime import timedelta
import copy
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
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 32 * 63:
            raise ValueError("mapped queue lanes must be in [1, 2016]")
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


def _allocate_producer_arenas(root, rank, arena_sizes):
    """Allocate and fault this producer's pages under its worker NUMA policy.

    Call before any peer maps or registers these files. The application must
    bind each worker's memory policy to its intended node before context setup;
    CPU affinity alone does not establish a strict allocation policy.
    """
    for phase, size in arena_sizes.items():
        fd = os.open(root / f"{phase}-{rank}.arena",
                     os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            os.ftruncate(fd, size)
            # Reserve storage as the producer, not the coordinator. A rank-0
            # fallocate would already place tmpfs pages before peer first touch.
            os.posix_fallocate(fd, 0, size)
            with mmap.mmap(fd, size) as data:
                ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(data)), 0, size)
        finally:
            os.close(fd)


@dataclass(frozen=True)
class _DeviceRoute:
    inverse: torch.Tensor
    count: torch.Tensor
    owner_rows: torch.Tensor
    owner_counts: torch.Tensor
    active_channels: int
    expected_ranks: object = None
    local_rows: object = None
    workspace: object = None


class HostSMWorkspace:
    """Explicitly borrowed, fixed-shape storage for sequential inference.

    Create with ``HostSMContext.create_workspace``. Dispatch batches, combine
    outputs and pending tickets backed by this workspace are valid only until
    its next dispatch. Enqueue all output consumers on the stream passed to
    combine's ``wait`` before that next dispatch. Clone retained outputs, or
    use a different workspace for operations whose lifetimes overlap. The
    default API without a workspace continues to return independent storage.
    """

    def __init__(self, context, recv_capacity, local_tokens, fused_combine):
        self._context_id = context._id
        self.recv_capacity = recv_capacity
        self.local_tokens = local_tokens
        self.fused_combine = fused_combine
        self._generation = 0
        self._busy = False
        self._consumer_streams = []
        hidden, topk = context.config.hidden_size, context.config.topk
        options = {"device": context.device}
        self.hidden = torch.empty((recv_capacity, hidden), dtype=torch.bfloat16, **options)
        self.local_ids = torch.empty((recv_capacity, topk), dtype=torch.int64, **options)
        self.weights = torch.empty((recv_capacity, topk), dtype=torch.float32, **options)
        self.inverse = torch.empty((recv_capacity, 2), dtype=torch.int64, **options)
        self.count = torch.empty(1, dtype=torch.int32, **options)
        self.owner_rows = torch.empty((context.world_size, recv_capacity), dtype=torch.int32, **options)
        self.owner_counts = torch.empty(context.world_size, dtype=torch.int32, **options)
        self.output = torch.empty((local_tokens, hidden), dtype=torch.bfloat16, **options)
        self.remote = (None if fused_combine else torch.empty(
            (context.world_size, local_tokens, hidden), dtype=torch.bfloat16, **options))
        self.expected_ranks = (torch.empty(local_tokens, dtype=torch.int64, **options)
                               if fused_combine else None)
        self.local_rows = (torch.empty(local_tokens, dtype=torch.int32, **options)
                           if fused_combine else None)

    def _acquire(self, context, producer):
        # These dependencies protect prior expert/output consumers, including
        # a stream different from the producer supplied for this dispatch.
        for consumer in self._consumer_streams:
            if consumer != producer:  # _begin already ordered this producer.
                context._work_stream.wait_stream(consumer)
        self._consumer_streams.clear()
        self._generation += 1
        self._busy = True

    def _combined(self, consumer):
        if consumer not in self._consumer_streams:
            self._consumer_streams.append(consumer)
        self._busy = False


def _operation_identity(phase, tag, dispatch_operation=None):
    """Fingerprint small host API metadata; never includes route tensors."""
    payload = json.dumps((phase, tag, dispatch_operation), sort_keys=True,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


class _DevicePending(Pending):
    """Enqueued CUDA completion; wait inserts a stream dependency only."""

    def __init__(self, context, value, completed, active_channels, workspace=None):
        self._context = context
        self._value = value
        self._completed = completed
        self._device = context.device
        self._waited = False
        self._active_channels = active_channels
        self._workspace = (value.handle._plan.workspace
                           if isinstance(value, DispatchBatch) else workspace)
        self._workspace_generation = (self._workspace._generation
                                      if self._workspace is not None else None)

    def _check_workspace(self):
        if (self._workspace is not None and
                self._workspace._generation != self._workspace_generation):
            raise RuntimeError("borrowed workspace ticket expired when the workspace was reused")

    def wait(self, stream=None, timeout=None):
        if timeout is not None:
            raise ValueError("host_sm wait is nonblocking; use Context.check_errors for a host fence")
        self._check_workspace()
        # Submission and wait stay free of host queue-header polling. Observe
        # asynchronous transport errors at the explicit check_errors boundary.
        if self._context._failure is not None:
            raise RuntimeError("EP context is in a failed state") from self._context._failure
        with torch.cuda.device(self._device):
            consumer = stream or torch.cuda.current_stream(self._device)
            if consumer.device != self._device:
                raise ValueError("completion stream must belong to the context device")
            consumer.wait_event(self._completed)
            if isinstance(self._value, DispatchBatch):
                tensors = (self._value.hidden, self._value.local_ids, self._value.weights,
                           self._value.valid_count, self._value.handle.inverse_map,
                           self._value.handle._plan.owner_rows, self._value.handle._plan.owner_counts,
                           self._value.handle._plan.expected_ranks, self._value.handle._plan.local_rows)
            else:
                tensors = (self._value,)
            for tensor in tensors:
                if tensor is not None:
                    tensor.record_stream(consumer)
        with self._context._lock:
            if self._workspace is not None and not isinstance(self._value, DispatchBatch):
                self._workspace._combined(consumer)
            if not self._waited:
                self._context._unwaited -= 1
                self._waited = True
        return self._value

    @property
    def done(self):
        self._check_workspace()
        return self._completed.query()

    @property
    def metrics(self):
        # Byte counters require device instrumentation. Do not report inferred
        # values as measured PCIe traffic or reuse host_ce progress statistics.
        return {"transport": "host_sm", "channels": self._active_channels,
                "allocated_channels": self._context.effective_channels}


class HostSMContext(Context):
    """Mapped-host backend selected by Context(config.transport='host_sm').

    Setup validates the group and placement once, maps all producer queues on
    each GPU, and creates device expert-placement tables. Dispatch and combine
    enqueue GPU work without a Gloo exchange or CPU routing/count readback.
    Protocol diagnostics and tag fingerprints are opt-in at construction;
    dynamic tensor bounds, queue credits and stream ordering remain required.
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
        self._timeout_ms = max(1, math.ceil(config.timeout_seconds * 1000))
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
        self._max_receive_capacity = self.world_size * config.max_tokens_per_rank
        if self._max_receive_capacity > (1 << 31) - 1:
            raise ValueError("total receive capacity must fit int32")
        with torch.cuda.device(self.device):
            native.load()
            self._channel_capacity = native.host_sm_max_channels(self.world_size)
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
        self._mailbox_tokens = config.decode_max_tokens
        if self._mailbox_tokens:
            self._arenas["mailbox"] = []
        self._registered = []
        self._pointers = {}
        self._prepared = None
        self._root = None
        self._work_stream = None
        self._combine_profile = None
        self._combine_profile_last = None
        self._combine_profile_label = None
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
        # The dispatch and pairwise combine grids must fit every device. This
        # is bootstrap metadata, never a per-layer collective. Only world=1
        # can use the single-CTA fallback without cooperative launch support.
        if self.world_size > 1 and any(item[4] < 1 for item in signatures):
            raise ValueError("rank-pair combine requires cooperative all-resident launch support")
        self.effective_channels = min(self.config.channels, *(max(1, item[4]) for item in signatures))
        self._channel_chunk_rows = self.config.channel_chunk_rows(self.effective_channels)
        self._combine_chunk_rows = self.config.combine_chunk_rows(self.effective_channels, self.world_size)
        self._queue_peers = {"dispatch": 1, "combine": self.world_size}
        with torch.cuda.device(self.device):
            self._arena_sizes = {phase: native.host_sm_arena_bytes(
                self.config.hidden_size, topk, self.config.slots,
                self._channel_chunk_rows if phase == "dispatch" else self._combine_chunk_rows,
                channels=self.effective_channels, queue_peers=self._queue_peers[phase])
                for phase, topk in (("dispatch", self.config.topk), ("combine", 0))}
        if any(self._arena_sizes[phase] != self.config.mapped_arena_bytes(
                   topk, self.effective_channels, queue_peers=self._queue_peers[phase])
               for phase, topk in (("dispatch", self.config.topk), ("combine", 0))):
            raise RuntimeError("mapped queue allocation ABI mismatch")
        if self.config.decode_max_tokens:
            with torch.cuda.device(self.device):
                self._arena_sizes["mailbox"] = native.host_sm_mailbox_bytes(
                    self.config.hidden_size, self.config.decode_max_tokens, self.world_size)
            if self._arena_sizes["mailbox"] != self.config.mailbox_arena_bytes(self.world_size):
                raise RuntimeError("mapped decode mailbox allocation ABI mismatch")
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
                root_message[0] = str(self._root)
            except Exception:
                root_message[1] = "mapped queue allocation failed; check shared-memory capacity and permissions"
        leader = dist.get_global_rank(self.group, 0)
        dist.broadcast_object_list(root_message, src=leader, group=self.group)
        if root_message[1]:
            raise RuntimeError(root_message[1])
        self._root = Path(root_message[0])
        allocation_error = None
        try:
            # Dispatch, rank-pair combine and decode mailbox files are all
            # indexed by payload producer, including queues with many readers.
            _allocate_producer_arenas(self._root, self.rank, self._arena_sizes)
        except Exception:
            allocation_error = "producer queue allocation failed; check shared-memory capacity and permissions"
        allocation_errors = [None] * self.world_size
        dist.all_gather_object(allocation_errors, allocation_error, group=self.group)
        if any(allocation_errors):
            raise RuntimeError(next(error for error in allocation_errors if error))
        # Bootstrap-only barrier above: every producer has reserved and touched
        # its complete allocation before another rank can cudaHostRegister it.
        init_error = None
        try:
            with torch.cuda.device(self.device):
                self._work_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(self._work_stream):
                    for phase, size in self._arena_sizes.items():
                        pointers = []
                        for peer in range(self.world_size):
                            arena = _MappedArena(self._root / f"{phase}-{peer}.arena", size,
                                                 channels=(1 if phase == "mailbox" else
                                                           self.effective_channels * self._queue_peers[phase]))
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
                    self._prepared = native.prepare_host_sm(
                        self._expert_rank, self._expert_local, self._pointers["dispatch"],
                        self._pointers["combine"], self._status, self.rank,
                        self.config.hidden_size, self.config.topk, self.config.slots,
                        self._channel_chunk_rows, self._combine_chunk_rows,
                        self._timeout_ms, self.effective_channels,
                        diagnostics=self.config.diagnostics, fatal_errors=self.config.fatal_errors,
                        mailbox_pointers=self._pointers.get("mailbox"),
                        mailbox_tokens=self.config.decode_max_tokens,
                        combine_warp_specialization=self.config.combine_warp_specialization)
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
        between MoE layers. Host queue headers are inspected only at explicit
        validation/diagnostic boundaries, never on normal submission or wait.
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

    def start_combine_profile(self, max_operations=1024):
        """Start bounded, opt-in GPU diagnostics at a request boundary.

        Allocation and zero initialization happen here, not in combine. Each
        accepted operation gets independent storage until stop_combine_profile
        completes. The session never overwrites older samples; excess combines
        run without instrumentation and increment dropped_operations.
        """
        with self._lock, torch.cuda.device(self.device):
            if self._closed:
                raise RuntimeError("Context is closed")
            if self.config.combine_warp_specialization:
                raise ValueError("combine phase profiling requires combine_warp_specialization=False; "
                                 "its counters describe the sequential pair-queue kernel")
            self._check_aborted()
            if self._combine_profile is not None:
                raise RuntimeError("stop the current combine profile before starting another")
            if (isinstance(max_operations, bool) or not isinstance(max_operations, int) or
                    not 1 <= max_operations <= 65536):
                raise ValueError("max_operations must be an integer in [1, 65536]")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("combine diagnostics cannot start during CUDA Graph capture")
            fields = native.host_sm_profile_fields()
            if not fields or len(set(fields)) != len(fields):
                raise RuntimeError("invalid native combine diagnostic schema")
            lanes = self.effective_channels * self.world_size
            shape = (max_operations, lanes, len(fields))
            storage_bytes = max_operations * lanes * len(fields) * 8
            if storage_bytes * 2 + 4 > 64 << 20:
                raise ValueError("combine profile GPU and pinned readout buffers exceed the 64 MiB budget")
            with torch.cuda.stream(self._work_stream):
                counters = torch.zeros(shape, dtype=torch.int64, device=self.device)
            readout = torch.empty(shape, dtype=torch.int64, pin_memory=True)
            status_readout = torch.empty(1, dtype=torch.int32, pin_memory=True)
            # Finish initialization before the caller starts its measured
            # request/trace; this fence is outside the per-layer critical path.
            self._work_stream.synchronize()
            self._combine_profile = {
                "fields": fields, "max_operations": max_operations,
                "counters": counters, "readout": readout, "status_readout": status_readout,
                "operations": [], "dropped_operations": 0,
            }
            self._combine_profile_label = None
            self._combine_profile_last = None

    def set_combine_profile_label(self, label):
        """Snapshot JSON-compatible CPU metadata once per forward, if active.

        Labels may describe local phase, input rows and forward index. They do
        not infer a global phase for IDLE ranks. Tensors are not accepted.
        Returns False when diagnostics are off without serializing the label.
        """
        with self._lock:
            if self._combine_profile is None:
                return False
            self._combine_profile_label = json.loads(json.dumps(label, allow_nan=False))
            return True

    def _combine_profile_slice(self, handle, tag, epoch):
        session = self._combine_profile
        if session is None:
            return None, None
        index = len(session["operations"])
        if index >= session["max_operations"]:
            session["dropped_operations"] += 1
            return None, None
        record = {
            "index": index, "operation": self._counter, "epoch": epoch,
            "dispatch_operation": handle.operation,
            "tag": copy.deepcopy(tag), "label": self._combine_profile_label,
            "local_tokens": handle.local_token_count,
            "received_capacity": handle.received_count,
            "active_channels": handle.active_channels, "submitted": False,
            "counter_rows": [{"channel": channel, "peer": peer}
                             for channel in range(handle.active_channels)
                             for peer in range(self.world_size)],
        }
        session["operations"].append(record)
        return session["counters"][index, :handle.active_channels * self.world_size], record

    def stop_combine_profile(self):
        """Finish capture and bulk-read GPU counters at a completed boundary.

        Copies are queued on the transport stream, followed by one stream
        synchronization. No Gloo operation is involved. The report retains raw
        per-(channel, peer) rows in native field order; wait counters overlap phase
        counters, and timings are diagnostic kernel residency, not wire time.
        Each operation's counter_rows describes its own active grid. Inactive
        allocated channels are omitted from that operation's raw_counters.
        Fatal CUDA failures may prevent readout and require worker restart.
        """
        with self._lock, torch.cuda.device(self.device):
            if self._combine_profile is None:
                if self._combine_profile_last is None:
                    raise RuntimeError("no combine profile session has been started")
                return copy.deepcopy(self._combine_profile_last)
            session = self._combine_profile
            self._combine_profile = None
            self._combine_profile_label = None
            used = len(session["operations"])
            # The local session reference retains every GPU slice until all
            # selected operations and their readback have completed.
            with torch.cuda.stream(self._work_stream):
                session["readout"][:used].copy_(session["counters"][:used], non_blocking=True)
                session["status_readout"].copy_(self._status, non_blocking=True)
            self._work_stream.synchronize()
            result = {
                "schema_version": 3, "transport": "host_sm", "rank": self.rank,
                "world_size": self.world_size, "queue_layout": "rank_pair",
                "channels": self.effective_channels, "fields": session["fields"],
                "counter_rows": [{"channel": channel, "peer": peer}
                                 for channel in range(self.effective_channels)
                                 for peer in range(self.world_size)],
                "row_layout": "raw_counters[i] follows operations[i].counter_rows; top-level rows describe allocation",
                "max_operations": session["max_operations"], "captured_operations": used,
                "dropped_operations": session["dropped_operations"],
                "status": int(session["status_readout"].tolist()[0]),
                "host_failed": self._failure is not None,
                "operations": session["operations"],
                "raw_counters": [session["readout"][index, :operation["active_channels"] * self.world_size].tolist()
                                 for index, operation in enumerate(session["operations"])],
                "scope": "combine transport kernels only; excludes initialization and final reduction",
                "timing": {
                    "unit": "nanoseconds for fields ending in _ns",
                    "clock_scope": "GPU globaltimer; raw timestamps are not assumed comparable across devices",
                    "phase_fields": session["fields"][3:12],
                    "idle_fields": [field for field in session["fields"] if field.startswith("idle_") and field.endswith("_ns")],
                    "phase_rule": "nine disjoint phase durations sum to elapsed_ns per (channel, peer) CTA",
                    "completion_phase_order": session["fields"][3:12],
                    "idle_rule": "no-progress wait categories are exclusive subsets of elapsed time; they overlap the phase view",
                    "interpretation": "instrumented GPU instructions and CTA/peer waiting, not pure PCIe transfer time",
                    "missing_first_event": -1,
                },
                "payload_bytes": "requested payload load/store bytes; excludes metadata/control and is not a hardware PCIe traffic counter",
            }
            self._combine_profile_last = result
            return copy.deepcopy(result)

    def read_combine_profile(self):
        """Read the completed CPU report without synchronizing or reading CUDA."""
        with self._lock:
            if self._combine_profile is not None:
                raise RuntimeError("stop the combine profile at a completed boundary before reading it")
            if self._combine_profile_last is None:
                raise RuntimeError("no completed combine profile is available")
            return copy.deepcopy(self._combine_profile_last)

    def _begin(self, phase, stream):
        if self._closed:
            raise RuntimeError("Context is closed")
        if self._failure is not None:
            raise RuntimeError("EP context is in a failed state") from self._failure
        # The caller's producer stream can be capturing even though the
        # separate native transport stream is not; keep both boundary guards.
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

    def _finish(self, value, active_channels, workspace=None):
        completed = torch.cuda.Event()
        completed.record(self._work_stream)
        self._unwaited += 1
        return _DevicePending(self, value, completed, active_channels, workspace=workspace)

    def create_workspace(self, recv_capacity, local_tokens, *, fused_combine=False):
        """Allocate exact-shape, explicitly borrowed dispatch/combine storage.

        The caller owns the cache and its memory bound; this method does not
        retain a pool in the context. Use only for sequential inference where
        the previous output's consumers have been enqueued before reuse. No
        CUDA event query, CPU count readback or stream fence is required.
        """
        with self._lock, torch.cuda.device(self.device):
            if self._closed:
                raise RuntimeError("Context is closed")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("workspace allocation cannot be captured in a CUDA Graph")
            self._validate_workspace_shape(recv_capacity, local_tokens, fused_combine)
            with torch.cuda.stream(self._work_stream), torch.no_grad():
                return HostSMWorkspace(self, recv_capacity, local_tokens, fused_combine)

    def _validate_workspace_shape(self, capacity, local_tokens, fused_combine):
        if (isinstance(capacity, bool) or not isinstance(capacity, int) or
                not 0 <= capacity <= self._max_receive_capacity):
            raise ValueError("recv_capacity must be an integer within the configured group capacity")
        if (isinstance(local_tokens, bool) or not isinstance(local_tokens, int) or
                not 0 <= local_tokens <= self.config.max_tokens_per_rank):
            raise ValueError("local_tokens must be an integer within the configured rank capacity")
        if not isinstance(fused_combine, bool):
            raise ValueError("fused_combine must be a bool")
        if fused_combine and (not self.config.decode_max_tokens or
                              local_tokens > self.config.decode_max_tokens):
            raise ValueError("fused combine requires local tokens within the configured decode mailbox capacity")

    def _validate_dispatch(self, x, ids, weights):
        # Keep only the public contract and CPU-known allocation bounds here.
        # The prepared native entry validates each operation's dynamic tensor
        # device/dtype/layout/shape once, before it enqueues any kernel.
        if not all(isinstance(t, torch.Tensor) for t in (x, ids, weights)):
            raise ValueError("dispatch inputs must be tensors")
        if x.ndim != 2 or x.shape[0] > self.config.max_tokens_per_rank:
            raise ValueError("x must be [tokens, hidden] within the configured token capacity")
        if x.requires_grad or weights.requires_grad:
            raise ValueError("dispatch accepts inference tensors only")

    def dispatch_async(self, x, topk_ids, topk_weights, tag=None, stream=None, recv_capacity=None,
                       active_channels=None, workspace=None, fused_combine=False):
        """Enqueue routed dispatch without reading routes or counts on the CPU.

        recv_capacity is a CPU-known upper bound on received rows, normally the
        sum of scheduler-known input rows across ranks. If omitted, use the
        configured group maximum. A too-small bound poisons the context; it
        never silently drops rows. Output padding has IDs -1 and zero weights.

        active_channels is a CPU-known launch width in [1, effective_channels].
        All ranks must pass the same width for this dispatch. None uses every
        allocated channel. The selected width is retained by the handle and
        reused by combine; queue layout and frame sizes remain unchanged.

        workspace explicitly borrows fixed-shape storage created by
        create_workspace. Its previous combine must have been waited, and all
        output consumers must already be enqueued, before another dispatch.
        fused_combine selects bounded decode mailboxes independently of storage
        reuse. All ranks must agree on that mode for the same operation.
        """
        with self._lock, torch.cuda.device(self.device):
            try:
                self._validate_dispatch(x, topk_ids, topk_weights)
                capacity = ((workspace.recv_capacity if isinstance(workspace, HostSMWorkspace)
                             else self._max_receive_capacity) if recv_capacity is None else recv_capacity)
                self._validate_workspace_shape(capacity, x.shape[0], fused_combine)
                if workspace is not None:
                    if not isinstance(workspace, HostSMWorkspace) or workspace._context_id != self._id:
                        raise ValueError("workspace belongs to another context")
                    if (workspace.recv_capacity, workspace.local_tokens, workspace.fused_combine) != (
                            capacity, x.shape[0], fused_combine):
                        raise ValueError("workspace shape or fused combine mode does not match dispatch")
                    if workspace._busy:
                        raise RuntimeError("wait on the previous workspace combine before reusing it")
                if fused_combine and self._combine_profile is not None:
                    raise ValueError("fused combine does not support combine profiling")
                channels = self.effective_channels if active_channels is None else active_channels
                if (isinstance(channels, bool) or not isinstance(channels, int) or
                        not 1 <= channels <= self.effective_channels):
                    raise ValueError("active_channels must be an integer within the allocated channel count")
                identity = _operation_identity("dispatch", tag) if self.config.diagnostics else 0
                epoch = self._begin("dispatch", stream)
                if workspace is not None:
                    workspace._acquire(self, stream or torch.cuda.current_stream(self.device))
                for tensor in (x, topk_ids, topk_weights):
                    tensor.record_stream(self._work_stream)
                with torch.cuda.stream(self._work_stream), torch.no_grad():
                    ids = topk_ids.contiguous()
                    weights = topk_weights.contiguous()
                    if workspace is None:
                        hidden = torch.empty((capacity, self.config.hidden_size), dtype=torch.bfloat16, device=self.device)
                        local_ids = torch.empty((capacity, self.config.topk), dtype=torch.int64, device=self.device)
                        weights_out = torch.empty((capacity, self.config.topk), dtype=torch.float32, device=self.device)
                        inverse = torch.empty((capacity, 2), dtype=torch.int64, device=self.device)
                        count = torch.empty(1, dtype=torch.int32, device=self.device)
                        owner_rows = torch.empty((self.world_size, capacity), dtype=torch.int32, device=self.device)
                        owner_counts = torch.empty(self.world_size, dtype=torch.int32, device=self.device)
                        expected_ranks = (torch.empty(x.shape[0], dtype=torch.int64, device=self.device)
                                          if fused_combine else None)
                        local_rows = (torch.empty(x.shape[0], dtype=torch.int32, device=self.device)
                                      if fused_combine else None)
                    else:
                        hidden, local_ids, weights_out = workspace.hidden, workspace.local_ids, workspace.weights
                        inverse, count = workspace.inverse, workspace.count
                        owner_rows, owner_counts = workspace.owner_rows, workspace.owner_counts
                        expected_ranks, local_rows = workspace.expected_ranks, workspace.local_rows
                    decode_options = ({"expected_ranks": expected_ranks, "local_rows": local_rows}
                                      if fused_combine else {})
                    self._prepared.dispatch(x, ids, weights, epoch, hidden, local_ids,
                                            weights_out, inverse, count, owner_rows, owner_counts,
                                            active_channels=channels, identity=identity, **decode_options)
                    handle = RouteHandle(self._id, self._counter,
                                         _DeviceRoute(inverse, count, owner_rows, owner_counts, channels,
                                                      expected_ranks, local_rows, workspace),
                                         x.shape[0], capacity, tag)
                    return self._finish(DispatchBatch(hidden, local_ids, weights_out, handle, count), channels)
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
                if not isinstance(partial, torch.Tensor) or partial.requires_grad:
                    raise ValueError("partial must be an inference tensor")
                fused_combine = handle._plan.expected_ranks is not None
                workspace = handle._plan.workspace
                if fused_combine and self._combine_profile is not None:
                    raise ValueError("fused combine does not support combine profiling")
                identity = _operation_identity("combine", tag, handle.operation) if self.config.diagnostics else 0
                epoch = self._begin("combine", stream)
                for tensor in (partial, handle._plan.inverse, handle._plan.count,
                               handle._plan.owner_rows, handle._plan.owner_counts,
                               handle._plan.expected_ranks, handle._plan.local_rows):
                    if tensor is not None:
                        tensor.record_stream(self._work_stream)
                with torch.cuda.stream(self._work_stream), torch.no_grad():
                    if workspace is None:
                        output = torch.empty((handle.local_token_count, self.config.hidden_size),
                                             dtype=torch.bfloat16, device=self.device)
                        remote = (None if fused_combine else torch.empty(
                            (self.world_size, handle.local_token_count, self.config.hidden_size),
                            dtype=torch.bfloat16, device=self.device))
                    else:
                        output, remote = workspace.output, workspace.remote
                    profile, profile_record = self._combine_profile_slice(handle, tag, epoch)
                    decode_options = ({"expected_ranks": handle._plan.expected_ranks,
                                       "local_rows": handle._plan.local_rows} if fused_combine else {})
                    self._prepared.combine(partial, handle._plan.inverse, handle._plan.count,
                                           handle._plan.owner_rows, handle._plan.owner_counts,
                                           epoch, output, remote, active_channels=handle.active_channels,
                                           identity=identity, profile=profile, **decode_options)
                    if profile_record is not None:
                        profile_record["submitted"] = True
                    handle._consumed = True
                    if workspace is None:
                        return self._finish(output, handle.active_channels)
                    return self._finish(output, handle.active_channels, workspace=workspace)
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
                # An unfinished session is intentionally not read implicitly at
                # shutdown. Stop it explicitly to retain its diagnostic report.
                self._combine_profile = None
                self._combine_profile_label = None
                # Release the prepared object's retained device tensors only
                # after draining transport, and before unregistering mappings.
                self._prepared = None
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
