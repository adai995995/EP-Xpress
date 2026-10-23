"""Bounded combine diagnostics API contracts without a CUDA device."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from ep_xpress.host_sm import HostSMContext


FIELDS = ["begin_ns", "end_ns", "elapsed_ns"] + [f"phase_{i}_ns" for i in range(9)] + [
    "idle_wait_ready_ns", "idle_wait_credit_ns", "payload_written_bytes", "status",
]


class Stream:
    def __init__(self):
        self.fences = 0

    def synchronize(self):
        self.fences += 1


@contextmanager
def cpu_context(channels=4, fields=FIELDS, world=1):
    context = object.__new__(HostSMContext)
    context.config = SimpleNamespace(combine_warp_specialization=False)
    context.rank = 0
    context.world_size = world
    context.device = torch.device("cuda", 0)
    context.effective_channels = channels
    context._lock = threading.RLock()
    context._closed = False
    context._failure = None
    context._arenas = {"dispatch": [], "combine": []}
    context._work_stream = Stream()
    context._status = torch.tensor([0], dtype=torch.int32)
    context._combine_profile = None
    context._combine_profile_last = None
    context._combine_profile_label = None
    context._counter = 1
    zeros, empty = torch.zeros, torch.empty

    def cpu_zeros(shape, **kwargs):
        kwargs.pop("device", None)
        return zeros(shape, **kwargs)

    def cpu_empty(shape, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(shape, **kwargs)

    with patch("ep_xpress.host_sm.torch.cuda.device", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.stream", side_effect=lambda _: nullcontext()), \
            patch("ep_xpress.host_sm.torch.cuda.is_current_stream_capturing", return_value=False), \
            patch("ep_xpress.host_sm.torch.zeros", side_effect=cpu_zeros), \
            patch("ep_xpress.host_sm.torch.empty", side_effect=cpu_empty), \
            patch("ep_xpress.host_sm.native.host_sm_profile_fields", return_value=list(fields)):
        yield context


def handle(operation=1, channels=4):
    return SimpleNamespace(operation=operation, local_token_count=3, received_count=8,
                           active_channels=channels)


def test_disabled_profile_does_not_inspect_labels_or_allocate():
    with cpu_context() as context, patch("ep_xpress.host_sm.torch.zeros") as allocate:
        assert context.set_combine_profile_label(object()) is False
        assert context._combine_profile_slice(handle(), "layer-0", 1) == (None, None)
        allocate.assert_not_called()
        assert context._work_stream.fences == 0
        with pytest.raises(RuntimeError, match="no completed"):
            context.read_combine_profile()


def test_bounded_capture_preserves_samples_and_reads_once():
    with cpu_context() as context:
        context.start_combine_profile(max_operations=2)
        assert context._work_stream.fences == 1
        label = {"phase": "EXTEND", "forward_index": 0, "local_rows": 3}
        context.set_combine_profile_label(label)
        label["phase"] = "mutated"
        first, metadata = context._combine_profile_slice(handle(), "layer-0", 1)
        first.fill_(7)
        metadata["submitted"] = True
        context.set_combine_profile_label({"phase": "IDLE", "forward_index": 1})
        second, _ = context._combine_profile_slice(handle(2), "layer-1", 2)
        second.fill_(9)
        assert context._combine_profile_slice(handle(3), "layer-2", 3) == (None, None)
        assert context._work_stream.fences == 1  # Per-operation selection never fences.
        context._status.fill_(3)
        result = context.stop_combine_profile()
        assert context._work_stream.fences == 2
        assert result["captured_operations"] == 2
        assert result["dropped_operations"] == 1
        assert result["status"] == 3
        assert result["operations"][0]["label"]["phase"] == "EXTEND"
        assert result["operations"][1]["label"]["phase"] == "IDLE"
        assert result["raw_counters"][0] == [[7] * len(FIELDS)] * 4
        assert result["raw_counters"][1] == [[9] * len(FIELDS)] * 4
        result["operations"].clear()
        assert len(context.read_combine_profile()["operations"]) == 2
        assert context.stop_combine_profile()["captured_operations"] == 2
        assert context._work_stream.fences == 2


def test_sessions_are_explicit_and_reusable_without_stale_counters():
    with cpu_context() as context:
        context.start_combine_profile(1)
        with pytest.raises(RuntimeError, match="stop the current"):
            context.start_combine_profile(1)
        with pytest.raises(RuntimeError, match="completed boundary"):
            context.read_combine_profile()
        with pytest.raises(TypeError):
            context.set_combine_profile_label({"tensor": torch.tensor(1)})
        sample, _ = context._combine_profile_slice(handle(), None, 1)
        sample.fill_(11)
        context.stop_combine_profile()
        context.start_combine_profile(1)
        sample, metadata = context._combine_profile_slice(handle(), None, 2)
        assert sample.count_nonzero().item() == 0
        assert metadata["label"] is None
        result = context.stop_combine_profile()
        assert result["dropped_operations"] == 0
        assert context._work_stream.fences == 4


@pytest.mark.parametrize("limit", [0, -1, True, 65537, 1.5])
def test_capture_limits_reject_invalid_sizes_before_allocation(limit):
    with cpu_context() as context, patch("ep_xpress.host_sm.torch.zeros") as allocate:
        with pytest.raises(ValueError, match="max_operations"):
            context.start_combine_profile(limit)
        allocate.assert_not_called()


def test_combined_gpu_and_pinned_storage_is_bounded():
    with cpu_context(channels=32) as context, patch("ep_xpress.host_sm.torch.zeros") as allocate:
        with pytest.raises(ValueError, match="64 MiB budget"):
            context.start_combine_profile(65536)
        allocate.assert_not_called()
        assert context._combine_profile is None


def test_pair_profile_rows_are_explicit_and_budget_counts_every_peer():
    with cpu_context(channels=2, world=3) as context:
        context.start_combine_profile(1)
        sample, _ = context._combine_profile_slice(handle(channels=2), None, 1)
        assert sample.shape == (6, len(FIELDS))
        report = context.stop_combine_profile()
        assert report["schema_version"] == 4
        assert report["queue_layout"] == "rank_pair"
        assert report["channels"] == 2
        assert report["world_size"] == 3
        assert report["counter_rows"] == [
            {"channel": channel, "peer": peer} for channel in range(2) for peer in range(3)]
    with cpu_context(channels=32, world=8) as context, \
            patch("ep_xpress.host_sm.torch.zeros") as allocate:
        with pytest.raises(ValueError, match="64 MiB budget"):
            context.start_combine_profile(2048)
        allocate.assert_not_called()


def test_profile_counters_only_include_the_operation_active_grid():
    with cpu_context(channels=8, world=2) as context:
        context.start_combine_profile(3)
        for index, channels in enumerate((8, 1, 4)):
            sample, metadata = context._combine_profile_slice(handle(index, channels), None, index + 1)
            assert sample.shape == (channels * 2, len(FIELDS))
            sample.fill_(index + 1)
            metadata["submitted"] = True
        report = context.stop_combine_profile()
    assert [len(rows) for rows in report["raw_counters"]] == [16, 2, 8]
    for index, (operation, rows) in enumerate(zip(report["operations"], report["raw_counters"])):
        assert len(operation["counter_rows"]) == len(rows)
        assert rows == [[index + 1] * len(FIELDS)] * len(rows)
        assert operation["counter_rows"] == [
            {"channel": channel, "peer": peer}
            for channel in range(operation["active_channels"]) for peer in range(2)]
