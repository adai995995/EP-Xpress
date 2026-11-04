"""Fail-fast communicator contract checks; no CUDA execution."""

import ctypes
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, NCCLCommHandle


@pytest.mark.parametrize("pointer", [None, 0, -1, True, ctypes.c_void_p()])
def test_borrow_rejects_null_or_invalid_pointer_before_loading(pointer):
    with patch("ep_xpress.comm._NCCLLibrary") as library:
        with pytest.raises(ValueError, match="non-null"):
            NCCLCommHandle.borrow(pointer, owner=object(), library_path="unused")
        library.assert_not_called()


def test_borrow_requires_lifetime_owner():
    with pytest.raises(ValueError, match="lifetime owner"):
        NCCLCommHandle.borrow(1, owner=None, library_path="unused")


def test_owner_retained_and_nccl_rank_and_device_queried():
    owner = SimpleNamespace()
    with patch("ep_xpress.comm._NCCLLibrary") as library:
        library.return_value.query.side_effect = [1, 4, 2]
        handle = NCCLCommHandle.borrow(ctypes.c_void_p(42), owner=owner, library_path="owner-dso")
        assert handle.get() == 42 and handle._owner is owner
        assert (handle.rank, handle.world_size, handle.device) == (1, 4, torch.device("cuda", 2))


def test_raw_pointer_is_not_accepted_as_context_handle():
    with pytest.raises(TypeError, match="NCCLCommHandle"):
        Context(None, [0], Config(128, 1, 8, transport="host_sm"), nccl_comm=42)


def test_host_ce_does_not_silently_switch_its_control_plane():
    with pytest.raises(ValueError, match="host_sm"):
        Context(None, [0], Config(128, 1, 8), nccl_comm=object())
