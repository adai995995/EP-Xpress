"""CPU-side bootstrap and API contracts; CUDA replay covers device execution."""

import os
import struct
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from ep_xpress import Config, Context, native
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


def test_multilane_abort_and_fatal_flags_cover_every_queue(tmp_path):
    path = tmp_path / "queue.arena"
    path.write_bytes(bytes(4 * 4096))
    path.chmod(0o600)
    arena = _MappedArena(path, 4 * 4096, channels=4)
    try:
        arena.set_fatal_errors()
        assert all(struct.unpack_from("=Q", arena.data, lane * 4096 + 16)[0] == 1
                   for lane in range(4))
        struct.pack_into("=Q", arena.data, 3 * 4096, 3)
        assert arena.abort_code() == 3
        arena.request_abort()
        assert all(struct.unpack_from("=Q", arena.data, lane * 4096 + 8)[0] == 1
                   for lane in range(4))
        assert struct.unpack_from("=Q", arena.data, 3 * 4096)[0] == 3
    finally:
        arena.close()


@pytest.mark.parametrize("capacities, expected", [([16, 8, 16], 4), ([0], 1), ([16, 2, 8], 2)])
def test_bootstrap_negotiates_group_min_then_checks_actual_budget(capacities, expected):
    context = object.__new__(HostSMContext)
    context.rank = 0
    context.world_size = len(capacities)
    context.config = Config(128, 2, 32, transport="host_sm", chunk_bytes=7 * 256,
                            host_pool_budget_bytes=1)
    context.expert_to_rank = tuple(range(context.world_size))
    context.device = torch.device("cuda", 0)
    context.group = object()
    context._channel_capacity = capacities[0]

    def gather(result, signature, group):
        result[:] = [(*signature[:3], f"gpu-{rank}", cap) for rank, cap in enumerate(capacities)]

    def arena_size(hidden, topk, slots, chunk_rows, channels, queue_peers):
        assert queue_peers == (1 if topk else context.world_size)
        assert chunk_rows == context.config.combine_chunk_rows(expected, queue_peers)
        assert channels == expected
        return context.config.mapped_arena_bytes(topk, channels, queue_peers=queue_peers)

    with patch("ep_xpress.host_sm.dist.all_gather_object", side_effect=gather), \
            patch("ep_xpress.host_sm.torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid="gpu-0")), \
            patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.native.host_sm_arena_bytes", side_effect=arena_size), \
            patch("ep_xpress.host_sm.dist.broadcast_object_list") as broadcast:
        with pytest.raises(ValueError, match="negotiated mapped host queues exceed"):
            context._bootstrap(None)
    assert context.effective_channels == expected
    assert context._combine_chunk_rows == context.config.combine_chunk_rows(expected, len(capacities))
    assert context._pool_bytes == len(capacities) * (
        context.config.mapped_arena_bytes(2, expected) +
        context.config.mapped_arena_bytes(0, expected, queue_peers=len(capacities)))
    broadcast.assert_not_called()  # No storage is created after a budget failure.


def test_pair_grid_rejects_missing_cooperative_support_before_allocation():
    context = object.__new__(HostSMContext)
    context.rank, context.world_size = 0, 2
    context.config = Config(128, 2, 32, transport="host_sm")
    context.expert_to_rank = (0, 1)
    context.device = torch.device("cuda", 0)
    context.group = object()
    context._channel_capacity = 8

    def gather(result, signature, group):
        result[:] = [(*signature[:3], f"gpu-{rank}", cap) for rank, cap in enumerate((8, 0))]

    with patch("ep_xpress.host_sm.dist.all_gather_object", side_effect=gather), \
            patch("ep_xpress.host_sm.torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid="gpu-0")), \
            patch("ep_xpress.host_sm.native.host_sm_arena_bytes") as allocate:
        with pytest.raises(ValueError, match="cooperative all-resident"):
            context._bootstrap(None)
    allocate.assert_not_called()


def test_pair_queue_abort_headers_cover_more_than_32_lanes(tmp_path):
    lanes = 4 * 9
    path = tmp_path / "pair-queue.arena"
    path.write_bytes(bytes(lanes * 4096))
    path.chmod(0o600)
    arena = _MappedArena(path, lanes * 4096, channels=lanes)
    try:
        arena.set_fatal_errors()
        arena.request_abort()
        assert all(struct.unpack_from("=QQ", arena.data, lane * 4096 + 8) == (1, 1)
                   for lane in range(lanes))
        struct.pack_into("=Q", arena.data, (lanes - 1) * 4096, 3)
        assert struct.unpack_from("=Q", arena.data, (lanes - 1) * 4096)[0] == 3
    finally:
        arena.close()


def test_native_pair_allocator_and_capacity_pass_world_to_extension():
    with patch("ep_xpress.native.load") as load:
        load.return_value.host_sm_max_channels.return_value = 16
        load.return_value.host_sm_arena_bytes.return_value = 4096 * 32
        assert native.host_sm_max_channels(8) == 16
        assert native.host_sm_arena_bytes(128, 0, 2, 3, channels=4, queue_peers=8) == 4096 * 32
        load.return_value.host_sm_max_channels.assert_called_once_with(8)
        load.return_value.host_sm_arena_bytes.assert_called_once_with(128, 0, 2, 3, 4, 8)
