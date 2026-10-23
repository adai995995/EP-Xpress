"""CPU-side bootstrap and API contracts; CUDA replay covers device execution."""

import os
import struct
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

torch = pytest.importorskip("torch")

from ep_xpress import Config, Context, native
from ep_xpress.context import DispatchBatch, RouteHandle
from ep_xpress.host_sm import HostSMContext, _DevicePending, _DeviceRoute, _MappedArena, _operation_identity


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


@pytest.mark.parametrize("diagnostics", [False, True])
def test_context_uses_prepared_operations_and_hashes_tags_only_when_enabled(diagnostics):
    context = object.__new__(HostSMContext)
    context.config = Config(4, 2, 3, transport="host_sm", chunk_bytes=1024,
                            diagnostics=diagnostics)
    context.device = torch.device("cuda", 0)
    context.world_size = 2
    context.effective_channels = 4
    context._max_receive_capacity = 6
    context._id = "prepared-context"
    context._counter = 1
    context._lock = threading.RLock()
    context._work_stream = object()
    context._prepared = Mock()
    context._begin = Mock(return_value=1)
    context._finish = Mock(side_effect=lambda value, channels: value)
    context._abort = Mock()
    context._combine_profile = None
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    # Unserializable metadata is intentionally allowed when diagnostics are
    # disabled: normal submissions do not inspect tags or perform JSON/hash.
    tag = {"layer": 1} if diagnostics else object()
    empty = torch.empty

    def cpu_empty(shape, **kwargs):
        kwargs.pop("device", None)
        return empty(shape, **kwargs)

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch.object(torch.Tensor, "record_stream"), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty), \
            patch("ep_xpress.host_sm._operation_identity", wraps=_operation_identity) as identity:
        batch = context.dispatch_async(x, ids, weights, tag=tag, recv_capacity=5)
        result = context.combine_async(batch.hidden, batch.handle, tag=tag)

    dispatch = context._prepared.dispatch.call_args
    combine = context._prepared.combine.call_args
    assert dispatch.args[:4] == (x, ids, weights, 1)
    assert dispatch.args[4] is batch.hidden
    assert dispatch.args[7] is batch.handle.inverse_map
    assert dispatch.args[8] is batch.valid_count
    assert dispatch.args[9] is batch.handle._plan.owner_rows
    assert dispatch.args[10] is batch.handle._plan.owner_counts
    assert dispatch.kwargs["active_channels"] == 4
    assert tuple(batch.handle._plan.owner_rows.shape) == (2, 5)
    assert tuple(batch.handle._plan.owner_counts.shape) == (2,)
    assert batch.handle._plan.owner_rows.dtype == batch.handle._plan.owner_counts.dtype == torch.int32
    assert combine.args[0] is batch.hidden
    assert combine.args[1] is batch.handle.inverse_map
    assert combine.args[2] is batch.valid_count
    assert combine.args[3] is batch.handle._plan.owner_rows
    assert combine.args[4] is batch.handle._plan.owner_counts
    assert combine.args[5] == 1
    assert combine.args[6] is result
    assert combine.kwargs["active_channels"] == 4
    assert tuple(result.shape) == (3, 4)
    assert batch.handle._consumed
    assert combine.kwargs["profile"] is None
    if diagnostics:
        assert identity.call_count == 2
        assert dispatch.kwargs["identity"] == _operation_identity("dispatch", tag)
        assert combine.kwargs["identity"] == _operation_identity("combine", tag, 1)
    else:
        identity.assert_not_called()
        assert dispatch.kwargs["identity"] == combine.kwargs["identity"] == 0
    context._abort.assert_not_called()


def test_active_width_and_owner_lists_follow_each_handle_in_reverse_combine_order():
    context = object.__new__(HostSMContext)
    context.config = Config(4, 2, 3, transport="host_sm", chunk_bytes=1024, channels=32)
    context.device = torch.device("cuda", 0)
    context.world_size, context.effective_channels = 2, 32
    context._max_receive_capacity = 6
    context._id, context._counter = "active-context", 0
    context._closed, context._failure, context._unwaited = False, None, 0
    context._epochs = {"dispatch": 0, "combine": 0}
    context._lock = threading.RLock()
    context._work_stream = Mock()
    context._prepared = Mock()
    context._finish = Mock(side_effect=lambda value, channels: value)
    context._abort = Mock()
    context._combine_profile = None
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    producer = SimpleNamespace(device=context.device)
    empty, recorded = torch.empty, []

    def cpu_empty(shape, **kwargs):
        kwargs.pop("device", None)
        return empty(shape, **kwargs)

    def record_stream(tensor, stream):
        recorded.append((id(tensor), stream))

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.is_current_stream_capturing", return_value=False), \
            patch.object(torch.Tensor, "record_stream", new=record_stream), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty):
        batches = [context.dispatch_async(x, ids, weights, stream=producer,
                                          recv_capacity=5, active_channels=channels)
                   for channels in (8, 1, 32)]
        assert [batch.handle.operation for batch in batches] == [1, 2, 3]
        assert [batch.handle.active_channels for batch in batches] == [8, 1, 32]
        for batch in reversed(batches):
            context.combine_async(batch.hidden, batch.handle, stream=producer)

    assert context._epochs == {"dispatch": 3, "combine": 3}
    assert [call.kwargs["active_channels"] for call in context._prepared.dispatch.call_args_list] == [8, 1, 32]
    assert [call.kwargs["active_channels"] for call in context._prepared.combine.call_args_list] == [32, 1, 8]
    assert len({id(batch.handle._plan.owner_rows) for batch in batches}) == 3
    for epoch, (batch, call) in enumerate(zip(reversed(batches), context._prepared.combine.call_args_list), 1):
        assert call.args[3] is batch.handle._plan.owner_rows
        assert call.args[4] is batch.handle._plan.owner_counts
        assert call.args[5] == epoch
        assert (id(batch.handle._plan.owner_rows), context._work_stream) in recorded
        assert (id(batch.handle._plan.owner_counts), context._work_stream) in recorded
        assert batch.handle._consumed
    context._abort.assert_not_called()


@pytest.mark.parametrize("channels", [0, 33, True, 1.5])
def test_invalid_active_width_rejected_before_epoch_or_allocation(channels):
    context = object.__new__(HostSMContext)
    context.config = Config(4, 2, 3, transport="host_sm", chunk_bytes=1024, channels=32)
    context.device = torch.device("cuda", 0)
    context.effective_channels, context._max_receive_capacity = 32, 6
    context._lock, context._begin, context._abort = threading.RLock(), Mock(), Mock()
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.empty") as allocate:
        with pytest.raises(ValueError, match="active_channels"):
            context.dispatch_async(x, ids, weights, active_channels=channels)
    context._begin.assert_not_called()
    allocate.assert_not_called()


def test_pending_preserves_owner_tensor_stream_lifetimes_and_operation_width():
    context = SimpleNamespace(device=torch.device("cuda", 0), effective_channels=32,
                              _failure=None, _lock=threading.RLock(), _unwaited=1)
    route = _DeviceRoute(torch.empty((5, 2), dtype=torch.int64),
                         torch.empty(1, dtype=torch.int32),
                         torch.empty((2, 5), dtype=torch.int32),
                         torch.empty(2, dtype=torch.int32), 1)
    handle = RouteHandle("pending-context", 1, route, 3, 5)
    batch = DispatchBatch(torch.empty((5, 4), dtype=torch.bfloat16),
                          torch.empty((5, 2), dtype=torch.int64),
                          torch.empty((5, 2), dtype=torch.float32), handle, route.count)
    completed, consumer = object(), Mock(device=context.device)
    pending = _DevicePending(context, batch, completed, 1)
    recorded = []

    def record_stream(tensor, stream):
        recorded.append((id(tensor), stream))

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch.object(torch.Tensor, "record_stream", new=record_stream):
        assert pending.wait(stream=consumer) is batch
        assert pending.wait(stream=consumer) is batch
    assert context._unwaited == 0
    assert (id(route.owner_rows), consumer) in recorded
    assert (id(route.owner_counts), consumer) in recorded
    assert all(call.args == (completed,) for call in consumer.wait_event.call_args_list)
    assert pending.metrics == {"transport": "host_sm", "channels": 1, "allocated_channels": 32}


def _workspace_context():
    context = object.__new__(HostSMContext)
    context.config = Config(4, 2, 3, transport="host_sm", chunk_bytes=1024,
                            channels=2, decode_max_tokens=3)
    context.device = torch.device("cuda", 0)
    context.world_size, context.effective_channels = 2, 2
    context._max_receive_capacity = 6
    context._id, context._counter = "workspace-context", 0
    context._closed, context._failure, context._unwaited = False, None, 0
    context._epochs = {"dispatch": 0, "combine": 0}
    context._lock = threading.RLock()
    context._work_stream = Mock()
    context._prepared = Mock()
    context._abort = Mock()
    context._combine_profile = None
    return context


@pytest.mark.parametrize("fused", [False, True])
def test_borrowed_workspace_reuses_tensors_after_combine_wait_with_stream_dependencies(fused):
    context = _workspace_context()
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    producer, consumer = Mock(device=context.device), Mock(device=context.device)
    empty = torch.empty

    def cpu_empty(shape, **kwargs):
        kwargs.pop("device", None)
        return empty(shape, **kwargs)

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.is_current_stream_capturing", return_value=False), \
            patch("ep_xpress.host_sm.torch.cuda.Event", side_effect=lambda: Mock()), \
            patch.object(torch.Tensor, "record_stream"), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty) as allocate:
        workspace = context.create_workspace(5, 3, fused_combine=fused)
        allocated = allocate.call_count
        first = context.dispatch_async(x, ids, weights, stream=producer, workspace=workspace,
                                       fused_combine=fused)
        first_batch = first.wait(stream=producer)
        first_combine = context.combine_async(first_batch.hidden, first_batch.handle, stream=producer)
        # Merely enqueueing combine is insufficient: output consumers must be
        # attached before the explicitly borrowed storage can be overwritten.
        epochs = context._epochs.copy()
        with pytest.raises(RuntimeError, match="previous workspace combine"):
            context.dispatch_async(x, ids, weights, stream=producer, workspace=workspace,
                                   fused_combine=fused)
        assert context._epochs == epochs
        output = first_combine.wait(stream=consumer)
        assert output is workspace.output
        first_combine.wait(stream=producer)
        context._work_stream.wait_stream.reset_mock()
        second = context.dispatch_async(x, ids, weights, stream=producer, workspace=workspace,
                                        fused_combine=fused)
        second_batch = second.wait(stream=producer)
        second_combine = context.combine_async(second_batch.hidden, second_batch.handle, stream=producer)
        assert second_combine.wait(stream=producer) is output
        assert allocate.call_count == allocated  # No subsequent tensor allocations.
        assert first_batch.hidden is second_batch.hidden is workspace.hidden
        assert first_batch.handle is not second_batch.handle
        assert first_batch.handle._consumed and second_batch.handle._consumed
        assert first_batch.handle.inverse_map is second_batch.handle.inverse_map
        assert first_batch.handle._plan.owner_rows is second_batch.handle._plan.owner_rows
        assert any(call.args == (consumer,) for call in context._work_stream.wait_stream.call_args_list)
        with pytest.raises(RuntimeError, match="expired"):
            first.wait(stream=producer)
        with pytest.raises(RuntimeError, match="expired"):
            _ = first_combine.done

    dispatch, combine = context._prepared.dispatch.call_args, context._prepared.combine.call_args
    assert combine.args[7] is workspace.remote
    if fused:
        assert workspace.remote is None
        assert dispatch.kwargs["expected_ranks"] is combine.kwargs["expected_ranks"] is workspace.expected_ranks
        assert dispatch.kwargs["local_rows"] is combine.kwargs["local_rows"] is workspace.local_rows
    else:
        assert tuple(workspace.remote.shape) == (2, 3, 4)
        assert "expected_ranks" not in dispatch.kwargs and "expected_ranks" not in combine.kwargs


def test_workspace_rejects_shape_mode_and_owner_before_submitting():
    context = _workspace_context()
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    empty = torch.empty

    def cpu_empty(shape, **kwargs):
        kwargs.pop("device", None)
        return empty(shape, **kwargs)

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.is_current_stream_capturing", return_value=False), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty), \
            patch.object(context, "_begin") as begin:
        workspace = context.create_workspace(5, 3, fused_combine=True)
        with pytest.raises(ValueError, match="shape or fused combine mode"):
            context.dispatch_async(x, ids, weights, workspace=workspace)
        with pytest.raises(ValueError, match="shape or fused combine mode"):
            context.dispatch_async(x, ids, weights, recv_capacity=6, workspace=workspace, fused_combine=True)
        workspace._context_id = "another-context"
        with pytest.raises(ValueError, match="another context"):
            context.dispatch_async(x, ids, weights, workspace=workspace, fused_combine=True)
        with pytest.raises(ValueError, match="rank capacity"):
            context.create_workspace(5, 4, fused_combine=True)
        begin.assert_not_called()


def test_fused_combine_without_workspace_keeps_operation_owned_output():
    context = _workspace_context()
    x = torch.zeros((3, 4), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.ones((3, 2), dtype=torch.float32)
    producer = Mock(device=context.device)
    empty = torch.empty

    def cpu_empty(shape, **kwargs):
        kwargs.pop("device", None)
        return empty(shape, **kwargs)

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.is_current_stream_capturing", return_value=False), \
            patch("ep_xpress.host_sm.torch.cuda.Event", side_effect=lambda: Mock()), \
            patch.object(torch.Tensor, "record_stream"), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty):
        batches = [context.dispatch_async(x, ids, weights, stream=producer, recv_capacity=5,
                                          fused_combine=True).wait(stream=producer)
                   for _ in range(2)]
        outputs = [context.combine_async(batch.hidden, batch.handle, stream=producer).wait(stream=producer)
                   for batch in reversed(batches)]
        assert outputs[0] is not outputs[1]
        assert batches[0].handle._plan.expected_ranks is not batches[1].handle._plan.expected_ranks
        assert all(batch.handle._plan.workspace is None for batch in batches)
        assert all(call.args[7] is None for call in context._prepared.combine.call_args_list)
        assert [call.args[5] for call in context._prepared.combine.call_args_list] == [1, 2]
