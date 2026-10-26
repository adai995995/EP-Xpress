"""GPU-initiated host staging with a CPU bootstrap and bounded shared queues.

No per-operation route tensor is read by the CPU. The public tensors have a
caller-provided upper-bound capacity, with padding IDs -1 and a device count.
All ranks must enqueue the same phase and handle order. Serialized CUDA Graph
execution is opt-in. GPU stream dependencies and queue credits remain necessary.
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
    return_rows: object = None


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
        self._recorded_streams = []
        self._allocation_stream = context._work_stream
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
        self.return_rows = (torch.empty((context.world_size, context.config.decode_max_tokens),
                                      dtype=torch.int32, **options)
                            if fused_combine and context._reuse_return_index else None)

    def _record_stream(self, stream, *, capturing=False):
        if capturing:
            # Graph callers retain storage for the entire graph lifetime.
            return
        if stream not in self._recorded_streams:
            if stream != self._allocation_stream:
                stream.wait_stream(self._allocation_stream)
            for tensor in (self.hidden, self.local_ids, self.weights, self.inverse, self.count,
                           self.owner_rows, self.owner_counts, self.output, self.remote,
                           self.expected_ranks, self.local_rows, self.return_rows):
                if tensor is not None:
                    tensor.record_stream(stream)
            self._recorded_streams.append(stream)

    def _acquire(self, context, producer, execution=None, *, capturing=False):
        # These dependencies protect prior expert/output consumers, including
        # a stream different from the producer supplied for this dispatch.
        execution = context._work_stream if execution is None else execution
        for consumer in self._consumer_streams:
            if not capturing and consumer != producer and consumer != execution:
                execution.wait_stream(consumer)
        self._consumer_streams.clear()
        self._generation += 1
        self._busy = True

    def _combined(self, consumer, *, capturing=False):
        if not capturing and consumer not in self._consumer_streams:
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
                           self._value.handle._plan.expected_ranks, self._value.handle._plan.local_rows,
                           self._value.handle._plan.return_rows)
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
            self._decode_packet_rows = native.load().host_sm_packet_rows()
            self._dispatch_banks = native.load().host_sm_dispatch_banks()
            self._dispatch_roles = native.load().host_sm_dispatch_roles()
            self._reuse_return_index = native.load().host_sm_reuse_return_index()
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
        self._last_stream = None
        self._cuda_graph_enabled = False
        self._capture_ready = False
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
                     str(torch.cuda.get_device_properties(self.device).uuid), self._channel_capacity,
                     native.load().__name__)
        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(item[:3] != signature[:3] or item[5] != signature[5] for item in signatures):
            raise ValueError("EP ranks must share one host, configuration, native build and expert placement")
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
                   topk, self.effective_channels, queue_peers=self._queue_peers[phase],
                   dispatch_banks=self._dispatch_banks if phase == "dispatch" else 0)
               for phase, topk in (("dispatch", self.config.topk), ("combine", 0))):
            raise RuntimeError("mapped queue allocation ABI mismatch")
        if self.config.decode_max_tokens:
            with torch.cuda.device(self.device):
                self._arena_sizes["mailbox"] = native.host_sm_mailbox_bytes(
                    self.config.hidden_size, self.config.decode_max_tokens, self.world_size)
            if self._arena_sizes["mailbox"] != self.config.mailbox_arena_bytes(
                    self.world_size, packet_rows=self._decode_packet_rows):
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
                        combine_warp_specialization=self.config.combine_warp_specialization,
                        decode_single_frame=self.config.decode_single_frame)
                self._work_stream.synchronize()
                self._last_stream = self._work_stream
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
                (getattr(self, "_last_stream", None) or self._work_stream).synchronize()
                code = int(self._status.item())
            if code:
                error = RuntimeError("host_sm: " + _ERRORS.get(code, "unknown GPU transport error"))
                self._abort(error)
                raise error
            self._check_aborted()

    check = check_errors

    def prepare_decode_profile(self, max_samples_per_channel=1024, *, stage="combine"):
        """Install bounded decode dispatch or fused-combine counters before the first submission.

        Optional diagnostic specialization, including Graph replay. Start/stop
        at quiescent request boundaries only; never overlap submission from
        another thread. Normal contexts allocate no storage and use no timers.
        """
        with self._lock, torch.cuda.device(self.device):
            if self._closed or self._counter or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare decode profiling before submissions, outside capture")
            if getattr(self, "_decode_profile", None) is not None:
                raise RuntimeError("decode profiling is already prepared")
            if (type(max_samples_per_channel) is not int or
                    not 1 <= max_samples_per_channel <= 4096):
                raise ValueError("max_samples_per_channel must be in [1, 4096]")
            if stage not in ("dispatch", "combine"):
                raise ValueError("decode profile stage must be dispatch or combine")
            fields = list(native.load().host_sm_decode_profile_fields(stage == "dispatch"))
            with torch.cuda.stream(self._work_stream):
                roles = 2 if (stage == "combine" and
                              (os.environ.get("EP_XPRESS_DECODE_SPLIT_TX_RX", "0") == "1" or
                               self._decode_packet_rows != 0)) else 1
                if stage == "dispatch":
                    roles = self._dispatch_roles
                storage = torch.zeros((max_samples_per_channel + 1,
                                       self.effective_channels * roles, len(fields)),
                                      dtype=torch.int64, device=self.device)
                self._prepared.prepare_decode_profile(storage, stage == "dispatch")
            self._work_stream.synchronize()
            self._decode_profile = {"storage": storage, "fields": fields,
                                    "max_samples": max_samples_per_channel, "active": False, "stage": stage}

    def start_decode_profile(self):
        """Reset and enable previously captured counters at a completed boundary."""
        with self._lock, torch.cuda.device(self.device):
            state = getattr(self, "_decode_profile", None)
            if self._closed or state is None or state["active"] or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("decode profiling must be prepared, inactive and outside capture")
            torch.cuda.synchronize(self.device)
            state["storage"].zero_()
            state["storage"][0, 0, 1].fill_(1)
            torch.cuda.synchronize(self.device)
            state["active"] = True

    def stop_decode_profile(self):
        """Read counters once after work completes; no per-layer CPU reads."""
        with self._lock, torch.cuda.device(self.device):
            state = getattr(self, "_decode_profile", None)
            if self._closed or state is None or not state["active"] or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("decode profiling must be active and outside capture")
            torch.cuda.synchronize(self.device)
            state["storage"][0, 0, 1].zero_()
            raw = state["storage"].cpu().tolist()
            state["active"] = False
            samples = []
            for channel, header in enumerate(raw[0]):
                for i in range(min(header[0], state["max_samples"])):
                    samples.append({"channel": channel, "values": raw[i + 1][channel]})
            if "ctrl0_ready_poll_ns" in state["fields"]:
                tx = ["tx_route_local_ns", "tx_credit_wait_ns", "tx_metadata_ns", "tx_payload_ns", "tx_publish_ns"]
                controls = {f"ctrl{g}": [f"ctrl{g}_{f}_ns" for f in (
                    "ready_poll", "descriptor_alloc", "metadata", "metadata_commit", "stage_wait",
                    "ack_publish", "ack_drain", "other")] for g in range(2)}
                rx = [f"rx_{f}_ns" for f in ("stage_wait", "payload", "completion", "ack_drain_wait", "other")]
                return {
                    "schema_version": 3, "stage": "dispatch_pipeline", "rank": self.rank,
                    "cta_roles": "independent_tx_rx" if self._dispatch_roles == 2 else "combined",
                    "role_index": "For split mode, physical CTA below active channels is TX; remaining CTAs are RX.",
                    "fields": state["fields"], "samples": samples,
                    "dropped_samples": sum(row[2] for row in raw[0]), "status": int(self._status.item()),
                    "phase_fields": [], "role_phase_fields": {"tx": tx, **controls, "rx": rx},
                    "timing": "Concurrent role-leader timelines; never sum roles as wall time. "
                        "TX phases partition tx_end_offset_ns; control/RX phases partition their "
                        "end-minus-begin offsets. No additional synchronization barriers. "
                        "TX/payload phases are leader instruction intervals; publication/completion "
                        "include other-thread stragglers and fences. READY polling includes host "
                        "flag accesses, votes and loop overhead, not only remote lateness. "
                        "RX stage wait includes metadata preparation and scheduling; ACK drain wait "
                        "starts once both control groups complete incoming reads/ACK publication. "
                        "Only eligible pipeline decode operations are sampled; prefill/fallback excluded. "
                        "Prepared but disabled kernels retain instrumentation resources. "
                        "GPU clocks are local; record writes excluded.",
                }
            if state["stage"] == "dispatch":
                phases = ["route_local_ns", "send_metadata_ns", "send_payload_ns",
                          "publish_fence_ready_ns", "peer_poll_ns", "receive_metadata_select_ns",
                          "receive_payload_ns", "ack_publish_ns", "ack_drain_check_ns", "other_control_ns"]
            else:
                phases = ["send_metadata_ns", "send_payload_ns", "publish_fence_ready_ns",
                          "receive_metadata_select_ns", "peer_poll_ns", "receive_payload_ns",
                          "ack_publish_ns", "idle_final_drain_ns", "idle_wait_credit_ns", "other_control_ns"]
            return {
                "schema_version": 2, "stage": state["stage"],
                "rank": self.rank, "fields": state["fields"],
                "samples": samples, "dropped_samples": sum(row[2] for row in raw[0]),
                "status": int(self._status.item()),
                "phase_fields": phases,
                "scope": state["stage"] + " decode only; per-CTA eager and Graph samples",
                "dispatch_note": "Dispatch metadata includes host IDs/weights and compact indices; "
                                 "idle durations overlap phases. Disabled dispatch diagnostics still "
                                 "execute diagnostic barriers; use ordinary contexts for performance.",
                "timing": "Disjoint phase durations sum to elapsed_ns. peer_poll_ns includes READY "
                          "checks and progress-loop overhead. Dispatch payload excludes routing IDs/weights; "
                          "combine payload includes reduction/output and local-only tokens. Dispatch "
                          "ack_drain_check_ns counts checks/completion, whereas idle_* fields overlap phases. "
                          "Combine idle_final_drain_ns is its explicit ACK-check phase. "
                          "Includes instrumentation barriers; not pure PCIe time. GPU clocks are local. " +
                          ("Packet mode samples transport only: send_metadata includes credit checks; "
                           "receive_payload excludes the separate HBM reduction; preparation and reduction "
                           "kernels must be included separately from a GPU timeline."
                           if state["stage"] == "combine" and self._decode_packet_rows else ""),
            }

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
            self._check_aborted()
            if getattr(self, "_cuda_graph_enabled", False):
                raise RuntimeError("operation profiling is unavailable in CUDA Graph mode")
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
            "timing_layout": "overlapping_warp_groups" if self.config.combine_warp_specialization else "sequential_cta",
            "kind": "combine", "index": index, "operation": self._counter, "epoch": epoch,
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
            last = getattr(self, "_last_stream", None)
            if last is not None and last != self._work_stream:
                self._work_stream.wait_stream(last)
            with torch.cuda.stream(self._work_stream):
                session["readout"][:used].copy_(session["counters"][:used], non_blocking=True)
                session["status_readout"].copy_(self._status, non_blocking=True)
            self._work_stream.synchronize()
            result = {
                "schema_version": 4, "transport": "host_sm", "rank": self.rank,
                "world_size": self.world_size, "queue_layout": "rank_pair",
                "warp_specialized": self.config.combine_warp_specialization,
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
                "raw_counters": [session["readout"][index, :len(operation["counter_rows"])].tolist()
                                 for index, operation in enumerate(session["operations"])],
                "scope": "non-fused dispatch and combine transport kernels (fused decode/idle excluded); excludes initialization and final reduction; dispatch uses sequential CTA phase schema",
                "timing": {
                    "unit": "nanoseconds for fields ending in _ns",
                    "clock_scope": "GPU globaltimer; raw timestamps are not assumed comparable across devices",
                    "phase_fields": session["fields"][3:12],
                    "idle_fields": [field for field in session["fields"] if field.startswith("idle_") and field.endswith("_ns")],
                    "phase_rule": ("Tx/Rx group phase times overlap; no sum-to-elapsed invariant; idle fields are group wait phases"
                                   if self.config.combine_warp_specialization else
                                   "nine disjoint phase durations sum to elapsed_ns per (channel, peer) CTA"),
                    "completion_phase_order": session["fields"][3:12],
                    "idle_rule": ("credit/drain belong to Tx, READY belongs to Rx; they overlap across groups"
                                  if self.config.combine_warp_specialization else
                                  "no-progress wait categories are exclusive subsets of elapsed time; they overlap the phase view"),
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

    def enable_cuda_graph(self):
        """Opt into serialized direct-stream eager/capture/replay execution.

        Call once before dispatch. Prepare each capture with
        ``prepare_graph_capture`` after warming its workspaces, and call
        ``prepare_graph_replay`` immediately before every graph replay. Keep
        the context, workspaces and external static inputs alive until graphs
        are destroyed; never run graphs sharing this context concurrently.
        All ranks must execute the same operation and graph-bucket order.
        """
        with self._lock, torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("enable CUDA Graph outside capture")
            if self._closed or self._counter or self._cuda_graph_enabled:
                raise RuntimeError("enable CUDA Graph once, before the first submission")
            if self.config.diagnostics or self._combine_profile is not None:
                raise ValueError("CUDA Graph mode requires operation diagnostics off")
            self._work_stream.synchronize()
            with torch.cuda.stream(self._work_stream):
                self._prepared.enable_device_epochs()
            self._work_stream.synchronize()
            self._cuda_graph_enabled = True

    def prepare_graph_capture(self):
        """Drain eager work once outside capture; never resets protocol state.

        Warm every workspace shape before this boundary. The capture must use
        only direct-stream operations and already allocated workspaces. CPU
        handles describe the recording; routes/counts are recomputed on replay.
        """
        with self._lock, torch.cuda.device(self.device):
            if not self._cuda_graph_enabled or self._closed or self._failure is not None:
                raise RuntimeError("an enabled, healthy graph context is required")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_graph_capture must run outside capture")
            (self._last_stream or self._work_stream).synchronize()
            self._capture_ready = True

    def prepare_graph_replay(self, stream=None):
        """Order one replay with prior eager/replay work without a host fence.

        Invoke before updating shared static inputs and graph.replay() on the
        same stream. Output consumers must be enqueued before another operation
        reuses workspace; cross-stream input writes need this dependency too.
        This notification is needed because replay bypasses all Python methods.
        """
        with self._lock, torch.cuda.device(self.device):
            if not self._cuda_graph_enabled or self._closed or self._failure is not None:
                raise RuntimeError("an enabled, healthy graph context is required")
            execution = stream or torch.cuda.current_stream(self.device)
            if execution.device != self.device or torch.cuda.is_current_stream_capturing():
                raise ValueError("replay must be submitted outside capture on the context device")
            previous = self._last_stream or self._work_stream
            if execution != previous:
                execution.wait_stream(previous)
            self._last_stream = execution
            self._capture_ready = False

    def _begin(self, phase, stream, *, same_stream=False):
        if self._closed:
            raise RuntimeError("Context is closed")
        if self._failure is not None:
            raise RuntimeError("EP context is in a failed state") from self._failure
        # The caller's producer stream can be capturing even though the
        # separate native transport stream is not; keep both boundary guards.
        capturing = torch.cuda.is_current_stream_capturing()
        graph_mode = getattr(self, "_cuda_graph_enabled", False)
        if graph_mode and not same_stream:
            raise RuntimeError("CUDA Graph mode requires direct-stream operations")
        if capturing and not (graph_mode and self._capture_ready):
            raise RuntimeError("prepare_graph_capture is required before CUDA Graph capture")
        if self._unwaited >= self.config.max_pending_operations:
            raise RuntimeError("pending ticket capacity exceeded; wait on earlier tickets")
        producer = stream or torch.cuda.current_stream(self.device)
        if producer.device != self.device:
            raise ValueError("producer stream must belong to the context device")
        self._capturing_submission = capturing
        if capturing:
            if producer != torch.cuda.current_stream(self.device):
                raise ValueError("capture operations must use the current capture stream")
            # Recording must neither wait on a noncapturing stream nor change
            # the stream of the last executed work. Native epochs live on GPU.
            self._counter += 1
            return 1
        self._capture_ready = False
        self._epochs[phase] += 1
        self._counter += 1
        if self._epochs[phase] >= 1 << 31:
            raise RuntimeError("operation generation exhausted; recreate the context")
        execution = producer if same_stream else self._work_stream
        previous = getattr(self, "_last_stream", None) or self._work_stream
        # Queue epochs are shared across APIs. Order only stream transitions;
        # consecutive direct submissions already have stream ordering.
        if previous != execution:
            execution.wait_stream(previous)
        if producer != execution and producer != previous:
            execution.wait_stream(producer)
        self._last_stream = execution
        return 1 if graph_mode else self._epochs[phase]

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

    def dispatch_on_stream(self, x, topk_ids, topk_weights, tag=None, stream=None,
                           recv_capacity=None, active_channels=None, workspace=None,
                           fused_combine=False):
        """Enqueue dispatch directly on the consumer stream and return its batch.

        This does not block the CPU. Consume the batch on this stream; callers
        using another stream must establish its dependency and tensor lifetimes.
        Shared queue transitions remain ordered with the asynchronous API.
        Capture requires enable_cuda_graph and the documented lifecycle hooks.
        """
        return self._dispatch(x, topk_ids, topk_weights, tag, stream, recv_capacity,
                              active_channels, workspace, fused_combine, same_stream=True)

    def dispatch_async(self, x, topk_ids, topk_weights, tag=None, stream=None, recv_capacity=None,
                       active_channels=None, workspace=None, fused_combine=False):
        """Enqueue on the communication stream; wait on the returned ticket.

        See ``_dispatch`` for capacity, workspace and channel contracts.
        """
        return self._dispatch(x, topk_ids, topk_weights, tag, stream, recv_capacity,
                              active_channels, workspace, fused_combine)

    def _dispatch(self, x, topk_ids, topk_weights, tag=None, stream=None, recv_capacity=None,
                  active_channels=None, workspace=None, fused_combine=False, *, same_stream=False):
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
                channels = self.effective_channels if active_channels is None else active_channels
                if (isinstance(channels, bool) or not isinstance(channels, int) or
                        not 1 <= channels <= self.effective_channels):
                    raise ValueError("active_channels must be an integer within the allocated channel count")
                identity = _operation_identity("dispatch", tag) if self.config.diagnostics else 0
                execution = (stream or torch.cuda.current_stream(self.device)) if same_stream else self._work_stream
                epoch = (self._begin("dispatch", execution, same_stream=True) if same_stream
                         else self._begin("dispatch", stream))
                capturing = getattr(self, "_capturing_submission", False)
                if capturing and (workspace is None or workspace._generation == 0):
                    raise RuntimeError("CUDA Graph capture requires a warmed, retained workspace")
                if workspace is not None:
                    workspace._acquire(self, stream or torch.cuda.current_stream(self.device), execution,
                                       capturing=capturing)
                    if same_stream:
                        workspace._record_stream(execution, capturing=getattr(self, "_capturing_submission", False))
                for tensor in (x, topk_ids, topk_weights):
                    tensor.record_stream(execution)
                with torch.cuda.stream(execution), torch.no_grad():
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
                        return_rows = (torch.empty((self.world_size, self.config.decode_max_tokens),
                                                   dtype=torch.int32, device=self.device)
                                       if fused_combine and self._reuse_return_index else None)
                    else:
                        hidden, local_ids, weights_out = workspace.hidden, workspace.local_ids, workspace.weights
                        inverse, count = workspace.inverse, workspace.count
                        owner_rows, owner_counts = workspace.owner_rows, workspace.owner_counts
                        expected_ranks, local_rows = workspace.expected_ranks, workspace.local_rows
                        return_rows = workspace.return_rows
                    decode_options = ({"expected_ranks": expected_ranks, "local_rows": local_rows,
                                       "return_rows": return_rows}
                                      if fused_combine else {})
                    handle = RouteHandle(self._id, self._counter,
                                         _DeviceRoute(inverse, count, owner_rows, owner_counts, channels,
                                                      expected_ranks, local_rows, workspace, return_rows),
                                         x.shape[0], capacity, tag)
                    dispatch_profile, dispatch_record = ((None, None) if fused_combine else
                                                         self._combine_profile_slice(handle, tag, epoch))
                    if dispatch_record is not None:
                        dispatch_record["kind"] = "dispatch"
                        dispatch_record["timing_layout"] = "sequential_cta"
                        dispatch_record["counter_rows"] = [{"channel": c} for c in range(channels)]
                        dispatch_profile = dispatch_profile[:channels]
                    self._prepared.dispatch(x, ids, weights, epoch, hidden, local_ids,
                                            weights_out, inverse, count, owner_rows, owner_counts,
                                            active_channels=channels, identity=identity, profile=dispatch_profile, **decode_options)
                    if dispatch_record is not None:
                        dispatch_record["submitted"] = True
                    batch = DispatchBatch(hidden, local_ids, weights_out, handle, count)
                    return batch if same_stream else self._finish(batch, channels)
            except BaseException as error:
                self._abort(error)
                raise

    def combine_on_stream(self, partial, handle, tag=None, stream=None, weights_applied=True):
        """Enqueue combine directly on its consumer stream and return the tensor.

        Enqueue every consumer of borrowed output on this stream before the
        workspace is reused. For cross-stream consumers use explicit ordering
        and lifetime management, or use ``combine_async(...).wait(stream=...)``.
        """
        return self._combine(partial, handle, tag, stream, weights_applied, same_stream=True)

    def combine_async(self, partial, handle, tag=None, stream=None, weights_applied=True):
        return self._combine(partial, handle, tag, stream, weights_applied)

    def _combine(self, partial, handle, tag=None, stream=None, weights_applied=True, *, same_stream=False):
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
                identity = _operation_identity("combine", tag, handle.operation) if self.config.diagnostics else 0
                execution = (stream or torch.cuda.current_stream(self.device)) if same_stream else self._work_stream
                epoch = (self._begin("combine", execution, same_stream=True) if same_stream
                         else self._begin("combine", stream))
                if same_stream and workspace is not None:
                    workspace._record_stream(execution, capturing=getattr(self, "_capturing_submission", False))
                # Workspace storage is registered once per execution stream.
                # The expert output is external and still needs its lifetime recorded.
                inputs = ((partial,) if same_stream and workspace is not None else
                          (partial, handle._plan.inverse, handle._plan.count,
                           handle._plan.owner_rows, handle._plan.owner_counts,
                           handle._plan.expected_ranks, handle._plan.local_rows, handle._plan.return_rows))
                for tensor in inputs:
                    if tensor is not None:
                        tensor.record_stream(execution)
                with torch.cuda.stream(execution), torch.no_grad():
                    if workspace is None:
                        output = torch.empty((handle.local_token_count, self.config.hidden_size),
                                             dtype=torch.bfloat16, device=self.device)
                        remote = (None if fused_combine else torch.empty(
                            (self.world_size, handle.local_token_count, self.config.hidden_size),
                            dtype=torch.bfloat16, device=self.device))
                    else:
                        output, remote = workspace.output, workspace.remote
                    profile, profile_record = ((None, None) if fused_combine else
                                               self._combine_profile_slice(handle, tag, epoch))
                    decode_options = ({"expected_ranks": handle._plan.expected_ranks,
                                       "local_rows": handle._plan.local_rows,
                                       "return_rows": handle._plan.return_rows} if fused_combine else {})
                    self._prepared.combine(partial, handle._plan.inverse, handle._plan.count,
                                           handle._plan.owner_rows, handle._plan.owner_counts,
                                           epoch, output, remote, active_channels=handle.active_channels,
                                           identity=identity, profile=profile, **decode_options)
                    if profile_record is not None:
                        profile_record["submitted"] = True
                    handle._consumed = True
                    if same_stream:
                        if workspace is not None:
                            workspace._combined(execution, capturing=getattr(self, "_capturing_submission", False))
                        return output
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
                last = getattr(self, "_last_stream", None)
                if last is not None and last != self._work_stream:
                    last.synchronize()
                if self._work_stream is not None:
                    self._work_stream.synchronize()
                # An unfinished session is intentionally not read implicitly at
                # shutdown. Stop it explicitly to retain its diagnostic report.
                self._combine_profile = None
                self._combine_profile_label = None
                self._decode_profile = None
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
