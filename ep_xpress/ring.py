"""File-backed, single-producer host ring with process-shared control locks.

Only CPU control-plane operations use locks. A producer must complete its CUDA
write before publishing; a consumer must complete its CUDA read before ACKing.
These methods deliberately make no assumptions about GPU-to-CPU atomics.
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import re
import struct
import threading
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path


class RingError(RuntimeError):
    """A shared ring could not be opened or its control state is invalid."""


class RingAbortedError(RingError):
    """The operation owning this ring has been aborted."""


_MAGIC = b"EPXRING1"
_HEADER = struct.Struct("<8sQQQ")  # magic, slots, slot_bytes, aborted
_SLOT = struct.Struct("<QqQQQ")  # state, generation, expected, ack, rows
_FREE, _WRITING, _READY = 0, 1, 2
_STATES = {0: "FREE", 1: "WRITING", 2: "READY"}
_MAX_I64 = (1 << 63) - 1
_MAX_MASK = (1 << 63) - 1


def _bounded_int(value, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} is outside the supported range")
    return int(value)


class SharedRing:
    """A bounded rank/phase ring whose immutable slots support multiple readers.

    All processes must open the same shared filesystem directory. The directory
    must be private to a communicator epoch. Creation never overwrites an
    existing ring. Sequences increase monotonically for each slot; an ACK for a
    previous generation is rejected. A duplicate ACK remains harmless after
    the slot becomes FREE, until the next generation claims it.

    CUDA registration and any exported memoryviews must be released before
    calling ``close``. Data mappings are writable for CUDA registration, so
    correctness also requires callers to obey the WRITING/READY ownership rule.
    """

    def __init__(
        self, root: Path, rank: int, phase: str, slots: int, slot_bytes: int,
        create: bool = False,
    ):
        self.rank = _bounded_int(rank, "rank", 0, 62)
        self.slots = _bounded_int(slots, "slots", 1, (1 << 31) - 1)
        self.slot_bytes = _bounded_int(slot_bytes, "slot_bytes", 4096, _MAX_I64)
        if self.slot_bytes % 4096:
            raise ValueError("slot_bytes must be a multiple of 4096")
        if not isinstance(phase, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", phase):
            raise ValueError("phase must be a short alphanumeric identifier")
        if self.slots * self.slot_bytes > _MAX_I64:
            raise ValueError("total ring storage is outside the supported range")
        self.phase = phase
        self._lock = threading.RLock()
        self._closed = False
        self._control_fd = self._data_fd = None
        self._control = self.data = None
        control_size = _HEADER.size + self.slots * _SLOT.size
        data_size = self.slots * self.slot_bytes
        root = Path(root)
        control_path = root / f"rank-{self.rank}-{phase}.control"
        data_path = root / f"rank-{self.rank}-{phase}.data"
        created = []
        try:
            if create:
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            self._control_fd = os.open(control_path, flags, 0o600)
            if create:
                created.append(control_path)
            fcntl.flock(self._control_fd, fcntl.LOCK_EX)
            try:
                self._data_fd = os.open(data_path, flags, 0o600)
                if create:
                    created.append(data_path)
                    os.fchmod(self._control_fd, 0o600)
                    os.fchmod(self._data_fd, 0o600)
                    os.ftruncate(self._control_fd, control_size)
                    os.ftruncate(self._data_fd, data_size)
                if os.fstat(self._control_fd).st_size != control_size:
                    raise RingError("ring control dimensions do not match")
                if os.fstat(self._data_fd).st_size != data_size:
                    raise RingError("ring payload dimensions do not match")
                for fd in (self._control_fd, self._data_fd):
                    if os.fstat(fd).st_mode & 0o077:
                        raise RingError("ring storage permissions must be private")
                self._control = mmap.mmap(self._control_fd, control_size)
                self.data = mmap.mmap(self._data_fd, data_size)
                if create:
                    _HEADER.pack_into(self._control, 0, _MAGIC, self.slots, self.slot_bytes, 0)
                    for slot in range(self.slots):
                        self._write(slot, (_FREE, -1, 0, 0, 0))
                else:
                    magic, actual_slots, actual_bytes, aborted = _HEADER.unpack_from(self._control)
                    if (magic, actual_slots, actual_bytes) != (_MAGIC, self.slots, self.slot_bytes):
                        raise RingError("ring control header does not match")
                    if aborted not in (0, 1):
                        raise RingError("ring abort state is invalid")
            finally:
                fcntl.flock(self._control_fd, fcntl.LOCK_UN)
        except (OSError, ValueError, RingError) as exc:
            self.close()
            for path in created:
                try:
                    path.unlink()
                except OSError:
                    pass
            if isinstance(exc, RingError):
                raise exc from None
            raise RingError("unable to initialize shared ring storage") from None

    @contextmanager
    def _guard(self, check_abort: bool = True):
        with self._lock:
            if self._closed:
                raise RingError("ring is closed")
            fcntl.flock(self._control_fd, fcntl.LOCK_EX)
            try:
                if check_abort and _HEADER.unpack_from(self._control)[3]:
                    raise RingAbortedError("ring has been aborted")
                yield
            finally:
                fcntl.flock(self._control_fd, fcntl.LOCK_UN)

    def _slot_index(self, slot: int) -> int:
        return _bounded_int(slot, "slot", 0, self.slots - 1)

    def _read(self, slot):
        values = _SLOT.unpack_from(self._control, _HEADER.size + slot * _SLOT.size)
        if values[0] not in _STATES:
            raise RingError("ring slot state is invalid")
        return values

    def _write(self, slot, values):
        _SLOT.pack_into(self._control, _HEADER.size + slot * _SLOT.size, *values)

    def payload_address(self, slot: int) -> int:
        """Return a host pointer to a slot, valid until this mapping is closed."""
        slot = self._slot_index(slot)
        with self._guard():
            return ctypes.addressof(ctypes.c_char.from_buffer(self.data, slot * self.slot_bytes))

    def payload_view(self, slot: int) -> memoryview:
        slot = self._slot_index(slot)
        with self._guard():
            start = slot * self.slot_bytes
            return memoryview(self.data)[start:start + self.slot_bytes]

    def try_claim(self, slot: int, sequence: int, expected_mask: int, nrows: int) -> bool:
        slot = self._slot_index(slot)
        sequence = _bounded_int(sequence, "sequence", 0, _MAX_I64)
        expected_mask = _bounded_int(expected_mask, "expected_mask", 1, _MAX_MASK)
        nrows = _bounded_int(nrows, "nrows", 0, _MAX_I64)
        with self._guard():
            state, previous, _, _, _ = self._read(slot)
            if state != _FREE:
                return False
            if sequence <= previous:
                raise ValueError("sequence must increase when reusing a slot")
            self._write(slot, (_WRITING, sequence, expected_mask, 0, nrows))
            return True

    def publish(self, slot: int, sequence: int) -> None:
        slot = self._slot_index(slot)
        sequence = _bounded_int(sequence, "sequence", 0, _MAX_I64)
        with self._guard():
            state, generation, expected, acked, nrows = self._read(slot)
            if generation != sequence:
                raise ValueError("cannot publish a different slot generation")
            if state != _WRITING:
                raise ValueError("only a WRITING slot can be published")
            self._write(slot, (_READY, generation, expected, acked, nrows))

    def ready(self, slot: int, sequence: int):
        slot = self._slot_index(slot)
        sequence = _bounded_int(sequence, "sequence", 0, _MAX_I64)
        with self._guard():
            state, generation, _, _, nrows = self._read(slot)
            return nrows if state == _READY and generation == sequence else None

    def ack(self, slot: int, sequence: int, consumer_rank: int) -> None:
        slot = self._slot_index(slot)
        sequence = _bounded_int(sequence, "sequence", 0, _MAX_I64)
        consumer_rank = _bounded_int(consumer_rank, "consumer_rank", 0, 62)
        bit = 1 << consumer_rank
        with self._guard():
            state, generation, expected, acked, nrows = self._read(slot)
            if generation != sequence:
                raise ValueError("cannot acknowledge a different slot generation")
            if not expected & bit:
                raise ValueError("consumer does not belong to this slot")
            if state == _FREE and acked == expected:
                return
            if state != _READY:
                raise ValueError("only a READY slot can be acknowledged")
            acked |= bit
            state = _FREE if acked == expected else _READY
            self._write(slot, (state, generation, expected, acked, nrows))

    def snapshot(self, slot: int) -> dict:
        slot = self._slot_index(slot)
        with self._guard(check_abort=False):
            state, sequence, expected, acked, nrows = self._read(slot)
            return dict(state=_STATES[state], sequence=sequence,
                        expected_mask=expected, ack_mask=acked, nrows=nrows)

    def abort(self) -> None:
        with self._guard(check_abort=False):
            _HEADER.pack_into(self._control, 0, _MAGIC, self.slots, self.slot_bytes, 1)

    def aborted(self) -> bool:
        with self._guard(check_abort=False):
            return bool(_HEADER.unpack_from(self._control)[3])

    def close(self) -> None:
        """Release mappings and handles; exported views must be released first."""
        with self._lock:
            if self._closed:
                return
            # Closing the payload first lets BufferError leave the object usable.
            if self.data is not None:
                self.data.close()
                self.data = None
            if self._control is not None:
                self._control.close()
                self._control = None
            for name in ("_data_fd", "_control_fd"):
                fd = getattr(self, name)
                if fd is not None:
                    os.close(fd)
                    setattr(self, name, None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
