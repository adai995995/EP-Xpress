"""Upper-layer NCCL handles for host-SM bootstrap, never token transport.

Handles borrow a communicator and retain its owner. Close EP contexts before
explicitly destroying the owner's communicator/process group. Bootstrap calls
are collective, synchronous, and must not overlap other uses of that comm.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import pickle
import weakref

import torch
import torch.distributed as dist


class _NCCLLibrary:
    """Use the same DSO as the owner; never search for another NCCL version."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            "ncclCommUserRank": [pointer, ctypes.POINTER(integer)],
            "ncclCommCount": [pointer, ctypes.POINTER(integer)],
            "ncclCommCuDevice": [pointer, ctypes.POINTER(integer)],
            "ncclAllGather": [pointer, pointer, ctypes.c_size_t, integer, pointer, pointer],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, integer
        self.lib.ncclGetErrorString.argtypes = [integer]
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p

    def check(self, result):
        if result != 0:
            message = self.lib.ncclGetErrorString(result).decode(errors="replace")
            raise RuntimeError(f"NCCL bootstrap failed: {message}")

    def query(self, name, pointer):
        value = ctypes.c_int()
        self.check(getattr(self.lib, name)(pointer, ctypes.byref(value)))
        return value.value


class NCCLCommHandle:
    """Borrowed raw ``ncclComm_t`` plus owner, group ordering and CUDA device.

    Construct with :meth:`borrow` or :func:`get_nccl_comm_handle`. No destructor
    calls ncclCommDestroy/Abort. Keeping the owner alive cannot protect against
    an explicit upstream destroy; the application must order shutdown.
    """

    @classmethod
    def borrow(cls, pointer, *, owner, library_path, group=None):
        """Wrap an existing comm using its owner's NCCL DSO.

        ``group`` is optional for standalone raw comms. When supplied, it must
        describe the *same ordered ranks*, not just have the same size. Rank
        and size are checked here; the caller guarantees membership identity.
        The communicator must use NCCL's normal blocking host API mode.
        """
        if isinstance(pointer, ctypes.c_void_p):
            pointer = pointer.value
        if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0:
            raise ValueError("a non-null NCCL communicator pointer is required")
        if owner is None:
            raise ValueError("a borrowed communicator requires a lifetime owner")
        result = cls()
        result._owner = owner
        result._pointer = pointer
        result._library = _NCCLLibrary(library_path)
        result.group = group
        result.rank = result._library.query("ncclCommUserRank", pointer)
        result.world_size = result._library.query("ncclCommCount", pointer)
        result.device = torch.device("cuda", result._library.query("ncclCommCuDevice", pointer))
        if not 0 <= result.rank < result.world_size:
            raise ValueError("invalid NCCL communicator rank or size")
        if group is not None:
            result.validate_group(group)
        return result

    def get(self):
        """Return the borrowed raw pointer without a collective or CUDA call."""
        return self._pointer

    def validate_group(self, group):
        if not dist.is_initialized():
            raise RuntimeError("initialize the supplied process group first")
        if (dist.get_rank(group), dist.get_world_size(group)) != (self.rank, self.world_size):
            raise ValueError("NCCL communicator and process group ranks do not match")
        if self.group is not None and self.group is not group:
            if dist.get_process_group_ranks(group) != dist.get_process_group_ranks(self.group):
                raise ValueError("NCCL communicator and process group membership do not match")

    def _all_gather(self, tensor):
        # Initialization/shutdown only. Allocate and enqueue on one current
        # stream so tensor uploads precede the raw NCCL call. The caller's
        # subsequent CPU copy provides the completion fence.
        output = torch.empty((self.world_size, tensor.numel()), dtype=tensor.dtype, device=self.device)
        datatype = {torch.uint8: 1, torch.int64: 4}[tensor.dtype]
        self._library.check(self._library.lib.ncclAllGather(
            tensor.data_ptr(), output.data_ptr(), tensor.numel(), datatype,
            self._pointer, torch.cuda.current_stream(self.device).cuda_stream))
        return output

    def all_gather_object(self, value):
        """Trusted-rank bootstrap metadata only; never use in dispatch/combine.

        As with torch.distributed object collectives, peers must be trusted
        because deserialization uses pickle. Aggregate padding is capped at
        16 MiB. These raw calls do not create PyTorch Work objects: the process
        group watchdog timeout does not bound them. The application must bound
        worker startup/shutdown and restart failed workers; this handle never
        aborts a borrowed communicator.
        """
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("NCCL bootstrap must finish before CUDA Graph capture")
            sizes = self._all_gather(torch.tensor([len(payload)], dtype=torch.int64,
                                                  device=self.device)).cpu().view(-1).tolist()
            width = max(sizes)
            if min(sizes) <= 0 or width * self.world_size > 16 << 20:
                raise ValueError("NCCL bootstrap metadata exceeds the 16 MiB aggregate limit")
            data = torch.zeros(width, dtype=torch.uint8, device=self.device)
            data[:len(payload)].copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
            gathered = self._all_gather(data).cpu()
        return [pickle.loads(bytes(row[:size].tolist())) for row, size in zip(gathered, sizes)]

    def barrier(self):
        """Collective teardown rendezvous, outside the token data path."""
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("NCCL teardown cannot run during CUDA Graph capture")
            self._all_gather(torch.zeros(1, dtype=torch.uint8, device=self.device)).cpu()


_handles = weakref.WeakValueDictionary()
# Remember warmup independently of handle lifetime: rank-local GC must not
# turn the next lookup into an unmatched collective. Weak keys retain no owner.
_initialized_groups = weakref.WeakKeyDictionary()


def get_nccl_comm_handle(group, device=None):
    """Borrow PyTorch's NCCL communicator, as in DeepEP's reuse path.

    This setup-time call is collective on first use per group/device. It initializes the
    process group's lazy communicator before querying ``_comm_ptr()``. Older
    PyTorch builds without this API must pass an explicitly borrowed handle
    (for example their framework's PyNccl communicator) or use legacy Gloo
    bootstrap. EP-Xpress does not silently create another NCCL communicator.
    """
    if not dist.is_initialized():
        raise RuntimeError("initialize the NCCL process group first")
    group = dist.group.WORLD if group is None else group
    if str(dist.get_backend(group)) != "nccl":
        raise ValueError("get_nccl_comm_handle requires an NCCL process group")
    chosen = torch.cuda.current_device() if device is None else device
    chosen = torch.device("cuda", chosen) if isinstance(chosen, int) else torch.device(chosen)
    if chosen.type != "cuda":
        raise ValueError("NCCL communicator device must be CUDA")
    device = torch.device("cuda", torch.cuda.current_device() if chosen.index is None else chosen.index)
    key = (group, device)
    if key in _handles:
        return _handles[key]
    backend = group._get_backend(device)
    if not callable(getattr(backend, "_comm_ptr", None)):
        raise RuntimeError("PyTorch does not expose _comm_ptr(); pass NCCLCommHandle.borrow() "
                           "from the framework or use legacy Gloo bootstrap")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("obtain the NCCL communicator before CUDA Graph capture")
        if device not in _initialized_groups.get(group, ()):
            dist.all_reduce(torch.zeros(1, device=device), group=group)
            torch.cuda.current_stream(device).synchronize()
            _initialized_groups.setdefault(group, set()).add(device)
        # libtorch_cuda's symbol lookup uses the same linked NCCL DSO as this
        # ProcessGroup, including wheel-vendored NCCL installations.
        handle = NCCLCommHandle.borrow(backend._comm_ptr(), owner=group, group=group,
            library_path=Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so")
    if handle.device != device:
        raise ValueError("the process group's NCCL communicator is bound to another CUDA device")
    _handles[key] = handle
    return handle
