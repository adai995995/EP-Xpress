"""CPU-side bootstrap and API contracts; CUDA replay covers device execution."""

import os
import struct
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from ep_xpress import Config, Context
from ep_xpress.host_sm import HostSMContext, _MappedArena, _operation_identity


def test_context_factory_runs_selected_initializer_once():
    config = Config(128, 2, 32, transport="host_sm")
    with patch.object(HostSMContext, "__init__", return_value=None) as initialize:
        context = Context(group=None, expert_to_rank=(0, 1), config=config)
    assert isinstance(context, HostSMContext)
    assert isinstance(context, Context)
    initialize.assert_called_once_with(group=None, expert_to_rank=(0, 1), config=config)


def test_abort_flags_have_distinct_writers_and_remain_sticky(tmp_path):
    path = tmp_path / "queue.arena"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, 4096)
    finally:
        os.close(fd)
    arena = _MappedArena(path, 4096)
    try:
        assert arena.abort_code() == 0
        arena.request_abort()
        assert arena.abort_code() == 4
        assert struct.unpack_from("=Q", arena.data, 0)[0] == 0
        struct.pack_into("=Q", arena.data, 0, 2)
        arena.request_abort()
        assert arena.abort_code() == 2
    finally:
        arena.close()
    arena.close()  # Bootstrap cleanup can safely repeat after partial failure.


def test_mapped_queue_rejects_wrong_extent_and_public_permissions(tmp_path):
    path = tmp_path / "queue.arena"
    path.write_bytes(bytes(4096))
    path.chmod(0o600)
    with pytest.raises(ValueError, match="size or permissions"):
        _MappedArena(path, 8192)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="size or permissions"):
        _MappedArena(path, 4096)


def test_operation_identity_is_stable_but_detects_handle_and_phase_mismatch():
    tag = {"layer": 4, "request": "test"}
    same_tag = {"request": "test", "layer": 4}
    expected = _operation_identity("combine", tag, 7)
    assert expected == _operation_identity("combine", same_tag, 7)
    assert expected != _operation_identity("combine", tag, 8)
    assert expected != _operation_identity("dispatch", tag, 7)
    assert 0 <= expected < 1 << 63
