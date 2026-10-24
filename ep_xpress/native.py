"""Lazy CUDA helpers for the host-staged transport.

Importing this module never compiles code. Call :func:`load` explicitly or use an
operator on a CUDA-capable host with PyTorch and its matching CUDA toolkit.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading

import torch

_extension = None
_load_lock = threading.Lock()


def load(*, verbose: bool = False):
    """Build and load the bundled extension, using the PyTorch extension cache.

    No download or peer-device setup is performed. ``TORCH_EXTENSIONS_DIR``,
    ``TORCH_CUDA_ARCH_LIST`` and ``MAX_JOBS`` retain their PyTorch meanings.
    """
    global _extension
    if _extension is not None:
        return _extension
    with _load_lock:
        if _extension is not None:
            return _extension
        if not torch.cuda.is_available():
            raise RuntimeError("EP-Xpress native operators require a CUDA device")
        from torch.utils.cpp_extension import CUDA_HOME, load as build_extension

        if CUDA_HOME is None:
            raise RuntimeError("A CUDA toolkit is required to build EP-Xpress")
        source_dir = Path(__file__).resolve().parent / "csrc"
        sources = [source_dir / name for name in ("bindings.cpp", "kernels.cu", "host_sm.cu")]
        # Build-time tuning only: no environment lookup on the transport path.
        # Keep scalar copies as the default until model-level gains are repeatable.
        read_unroll = os.environ.get("EP_XPRESS_PREFILL_READ_UNROLL", "0")
        if read_unroll not in {"0", "1", "2", "4", "8"}:
            raise ValueError("EP_XPRESS_PREFILL_READ_UNROLL must be 0, 1, 2, 4, or 8")
        write_unroll = os.environ.get("EP_XPRESS_PREFILL_WRITE_UNROLL", "0")
        if write_unroll not in {"0", "1", "2", "4", "8"}:
            raise ValueError("EP_XPRESS_PREFILL_WRITE_UNROLL must be 0, 1, 2, 4, or 8")
        dispatch_read_unroll = os.environ.get("EP_XPRESS_DISPATCH_READ_UNROLL", "0")
        if dispatch_read_unroll not in {"0", "1", "2", "4", "8"}:
            raise ValueError("EP_XPRESS_DISPATCH_READ_UNROLL must be 0, 1, 2, 4, or 8")
        cuda_flags = ["-O3", "-lineinfo", f"-DEP_XPRESS_PREFILL_READ_UNROLL={read_unroll}",
                      f"-DEP_XPRESS_PREFILL_WRITE_UNROLL={write_unroll}",
                      f"-DEP_XPRESS_DISPATCH_READ_UNROLL={dispatch_read_unroll}"]
        digest = hashlib.sha256()
        for source in sources + sorted(source_dir.glob("*.h")):
            digest.update(source.read_bytes())
        digest.update(torch.__version__.encode())
        digest.update(str(torch.version.cuda).encode())
        digest.update(" ".join(cuda_flags).encode())
        # Avoid oversized parallel compiler jobs by default; honor user settings.
        os.environ.setdefault("MAX_JOBS", "1")
        _extension = build_extension(
            name="ep_xpress_cuda_" + digest.hexdigest()[:12],
            sources=[str(source) for source in sources],
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=verbose,
        )
    return _extension


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > (1 << 63) - 1:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _matrix(tensor, name: str, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous two-dimensional tensor")


def register_host(address: int, nbytes: int) -> None:
    """Pin an existing host mapping for portable CUDA copy-engine access.

    The caller owns the mapping and must retain it until all transfers complete
    and :func:`unregister_host` has returned. This call does not allocate memory.
    Register each mapping once per process, including in receiving processes.
    """
    address = _integer(address, "address", 1)
    nbytes = _integer(nbytes, "nbytes", 1)
    load().register_host(address, nbytes)


def unregister_host(address: int) -> None:
    """Unpin a registered mapping after its users have completed.

    This function does not insert stream dependencies or perform an explicit
    device synchronization. The caller is responsible for completion ordering.
    """
    load().unregister_host(_integer(address, "address", 1))


def copy_d2h_async(
    dst_address: int,
    src: torch.Tensor,
    row_start: int,
    row_count: int,
    stream: int,
) -> None:
    """Enqueue a contiguous BF16 row slice into a registered host mapping.

    ``stream`` is a ``torch.cuda.Stream.cuda_stream`` handle on ``src.device``.
    The destination range is checked against this process's registrations.
    Source allocator lifetime is recorded on that stream. The caller must order
    source writes before this copy, avoid overwriting either buffer while it is
    in flight, and publish host readiness only after its CUDA event completes.
    """
    _matrix(src, "src", torch.bfloat16)
    dst_address = _integer(dst_address, "dst_address", 1)
    row_start = _integer(row_start, "row_start")
    row_count = _integer(row_count, "row_count")
    stream = _integer(stream, "stream")
    if row_start > src.shape[0] or row_count > src.shape[0] - row_start:
        raise ValueError("row slice is outside src")
    load().copy_d2h_async(dst_address, src, row_start, row_count, stream)
    if src.numel() and row_count:
        src.record_stream(torch.cuda.ExternalStream(stream, device=src.device))


def add_partial(
    accum: torch.Tensor,
    partial: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> None:
    """Add BF16 rows to an FP32 accumulator on the current PyTorch stream.

    Indices must be unique within this call and belong to ``[0, accum.size(0))``.
    Calls writing the same accumulator must be serialized on one stream or
    explicitly ordered by the caller; this kernel does not use atomic adds.

    The default index validation copies indices to the host and synchronizes
    that copy. Set ``validate_indices=False`` only for immutable route indices
    whose bounds and uniqueness were already validated. This trusted fast path
    has no host synchronization and can be captured in a CUDA graph.
    """
    _matrix(accum, "accum", torch.float32)
    _matrix(partial, "partial", torch.bfloat16)
    if not isinstance(token_indices, torch.Tensor):
        raise TypeError("token_indices must be a tensor")
    if token_indices.dtype != torch.int64:
        raise TypeError("token_indices must have dtype torch.int64")
    if token_indices.device != accum.device or partial.device != accum.device:
        raise ValueError("all tensors must be on the same CUDA device")
    if token_indices.ndim != 1 or not token_indices.is_contiguous():
        raise ValueError("token_indices must be a contiguous vector")
    if partial.shape[1] != accum.shape[1]:
        raise ValueError("partial and accum must have matching hidden dimensions")
    if token_indices.numel() != partial.shape[0]:
        raise ValueError("token_indices length must match the number of partial rows")
    if not isinstance(validate_indices, bool):
        raise TypeError("validate_indices must be a bool")
    if validate_indices:
        indices = token_indices.detach().cpu().tolist()
        if any(index < 0 or index >= accum.shape[0] for index in indices):
            raise ValueError("token_indices contains an out-of-bounds index")
        if len(set(indices)) != len(indices):
            raise ValueError("token_indices must be unique within one partial")
    load().add_partial(accum, partial, token_indices)


def finish_combine(remote: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Round the FP32 remote sum to BF16, add local BF16, and return BF16.

    The intermediate BF16 rounding is intentional: it defines the same result
    contract for host and device reduction backends. Launches on the current
    PyTorch stream and performs no host synchronization.
    """
    _matrix(remote, "remote", torch.float32)
    _matrix(local, "local", torch.bfloat16)
    if remote.device != local.device or remote.shape != local.shape:
        raise ValueError("remote and local must have the same shape and device")
    return load().finish_combine(remote, local)


def register_mapped_host(address: int, nbytes: int) -> int:
    """Register shared host pages and return this GPU's mapped device address.

    Every participating process must register its mappings. The caller retains
    mappings and all device-pointer tensors until the transport stream drains.
    Only bootstrap calls this function; registration is not a per-layer action.
    """
    return load().register_mapped_host(
        _integer(address, "address", 1), _integer(nbytes, "nbytes", 1)
    )


def host_sm_max_channels(world: int = 1) -> int:
    """Cooperative transport occupancy on the current GPU; zero means unavailable.

    Context construction agrees on the minimum capability across the group and
    accounts for ``channels * world`` combine CTAs. Only world=1 supports the
    single-CTA fallback when cooperative launch is unavailable.
    """
    return load().host_sm_max_channels(_integer(world, "world", 1))


def host_sm_profile_fields():
    """Ordered int64 per-pair-CTA diagnostic fields for optional combine capture.

    Timing fields ending in ``_ns`` use GPU globaltimer nanoseconds. Phase
    durations and wait subsets must be interpreted using the native schema;
    overlapping counters must not be summed into a wall-clock duration.
    """
    return list(load().host_sm_profile_fields())


def host_sm_arena_bytes(hidden: int, topk: int, slots: int, chunk_rows: int,
                        channels: int = 1, queue_peers: int = 1) -> int:
    """Bytes for all lanes of one producer; topk=0 selects combine storage.

    chunk_rows is the per-queue capacity, not the producer's total payload budget.
    Dispatch uses queue_peers=1; combine requires queue_peers=world. Flattened
    queue index is channel * queue_peers + peer. Each queue has independent
    page-aligned storage and control.
    """
    return load().host_sm_arena_bytes(
        _integer(hidden, "hidden", 1), _integer(topk, "topk"),
        _integer(slots, "slots", 1), _integer(chunk_rows, "chunk_rows", 1),
        _integer(channels, "channels", 1), _integer(queue_peers, "queue_peers", 1),
    )


def host_sm_mailbox_bytes(hidden: int, tokens: int, world: int) -> int:
    """Bytes in one producer's bounded decode return mailbox arena."""
    return load().host_sm_mailbox_bytes(
        _integer(hidden, "hidden", 1), _integer(tokens, "tokens", 1),
        _integer(world, "world", 1),
    )


def prepare_host_sm(
    expert_rank, expert_local, dispatch_pointers, combine_pointers, status,
    rank, hidden, topk, slots, dispatch_rows, combine_rows, timeout_ms, channels,
    *, diagnostics=False, fatal_errors=False, mailbox_pointers=None, mailbox_tokens=0,
    combine_warp_specialization=False,
):
    """Validate and cache immutable GPU transport state during bootstrap.

    The returned operator retains the placement, pointer-table and status
    tensors. They must not be mutated or resized while it is alive; registered
    host arenas remain the context's responsibility. Preparation reads placement
    and pointer tables once on the host. Dispatch/combine never repeat that read
    or the cooperative occupancy query. Each call still validates dynamic
    tensor shapes, dtype/device/layout and epoch before enqueuing work.

    Diagnostics are opt-in: abort/timeout polling, identity fingerprints, finite
    route weights and duplicate expert checks are omitted by default. Route
    bounds, capacity checks, sticky device status and the ready/ACK protocol are
    always active. With diagnostics off, a missing peer requires external job
    cancellation. Fatal contexts trap on a device error and omit dispatch's
    otherwise error-only count-reset kernel.

    Prepared dispatch also fills int32 ``owner_rows[world, capacity]`` and
    ``owner_counts[world]``. These operation-scoped tensors retain the indices
    used by combine without rescanning the inverse map. Both prepared methods
    accept ``active_channels``: zero uses the allocated width, otherwise the
    value must be in [1, channels]. The caller uses the same active width for
    every rank and for the matching combine. Arena strides and frame capacities
    remain those fixed at preparation time.

    Optional mailbox_pointers int64[world] enables bounded decode combine.
    Prepared dispatch may then fill expected_ranks int64[local_tokens] and
    local_rows int32[local_tokens]. Passing that matching state to combine
    selects receive-and-reduce fusion with remote=None. Mailbox tokens bound
    each source rank's token capacity; their host storage remains owned by the
    context. Omitting these arguments preserves the rank-pair queue path.

    combine_warp_specialization=True gives prepared pair queues independent
    128-thread send/receive groups. It preserves the grid and frame layout,
    leaves fused mailbox combine unchanged; opt-in profiling records overlapping group times.
    The default False keeps the sequential pair kernel and its profiler.
    """
    if not all(isinstance(value, bool) for value in
               (diagnostics, fatal_errors, combine_warp_specialization)):
        raise TypeError("diagnostics, fatal_errors and combine_warp_specialization must be bools")
    return load().prepare_host_sm(
        expert_rank, expert_local, dispatch_pointers, combine_pointers, status,
        _integer(rank, "rank"), _integer(hidden, "hidden", 1),
        _integer(topk, "topk", 1), _integer(slots, "slots", 1),
        _integer(dispatch_rows, "dispatch_rows", 1),
        _integer(combine_rows, "combine_rows", 1),
        _integer(timeout_ms, "timeout_ms", 1), _integer(channels, "channels", 1),
        diagnostics=diagnostics, fatal_errors=fatal_errors,
        mailbox_pointers=mailbox_pointers,
        mailbox_tokens=_integer(mailbox_tokens, "mailbox_tokens"),
        combine_warp_specialization=combine_warp_specialization,
    )


def host_sm_dispatch(
    x, ids, weights, expert_rank, expert_local, pointers, rank, epoch, slots,
    chunk_rows, timeout_ms, output, output_ids, output_weights, inverse, count, status,
    *, identity=0, channels=1,
) -> None:
    """Enqueue GPU routing and mapped-host dispatch on the current CUDA stream.

    All tensors stay on this GPU. Output tensors have a fixed upper-bound row
    capacity; the kernel writes the exact row count to the device-only `count`.
    Input expert IDs may be int32 or int64; output IDs remain int64.
    Padding has zero payload/weights and -1 expert IDs. `status` is a sticky
    int32[1] error word initially zero. The caller must check it before accepting
    experimental results and destroy the context after any error. No host read,
    collective, copy-engine payload transfer or peer access occurs here.

    Native pointer tensors are trusted bootstrap data, not arbitrary user input.
    Their arenas must have the exact size returned by host_sm_arena_bytes and
    remain registered and alive through completion. Calls sharing arenas must be
    serialized, with monotonically increasing positive epochs on every rank.
    All ranks must use the same channels and per-lane chunk_rows. More than one
    lane requires a cooperatively resident grid supported by every GPU.
    """
    load().host_sm_dispatch(
        x, ids, weights, expert_rank, expert_local, pointers, rank, epoch, slots,
        chunk_rows, timeout_ms, output, output_ids, output_weights, inverse, count, status,
        _integer(identity, "identity"), _integer(channels, "channels", 1),
    )


def host_sm_combine(
    partial, inverse, count, pointers, rank, epoch, slots, chunk_rows, timeout_ms,
    output, remote_accum, status, *, identity=0, channels=1, profile=None,
) -> None:
    """Enqueue inverse-routed mapped-host combine without a CPU count read.

    Remote BF16 partials are staged in rank-indexed BF16 planes
    [world, local_tokens, hidden]. The owning GPU sums ranks in ascending order
    in FP32, rounds to BF16, and adds the local BF16 partial. `partial`
    already includes routing weights. Arena and status lifetime rules are the
    same as host_sm_dispatch; combine uses separate arenas allocated with
    topk=0 and queue_peers=world. chunk_rows is the per-pair capacity.
    Optional ``profile`` is a CUDA int64 [channels * world, field_count] output whose
    columns follow host_sm_profile_fields(). None selects the uninstrumented
    kernel variant. The caller retains its storage through completion and
    reads it only at an explicit diagnostic boundary. Row index is
    channel * world + peer, including the local-bypass peer CTA.
    """
    load().host_sm_combine(
        partial, inverse, count, pointers, rank, epoch, slots, chunk_rows,
        timeout_ms, output, remote_accum, status, _integer(identity, "identity"),
        _integer(channels, "channels", 1), profile,
    )
