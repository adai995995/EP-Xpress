"""Native mapped-host protocol checks; distributed tests exercise peer traffic."""

from contextlib import contextmanager
import ctypes
import mmap
import struct

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import native

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")]


@pytest.fixture(scope="module", autouse=True)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("The CUDA toolkit is unavailable")
    native.load()


@contextmanager
def arenas(world=1, hidden=8, topk=2, slots=2, chunk_rows=1, channels=1):
    mappings, addresses, pointers = [], [], []
    nbytes = native.host_sm_arena_bytes(hidden, topk, slots, chunk_rows, channels=channels)
    try:
        for _ in range(world):
            mapping = mmap.mmap(-1, nbytes)
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            pointer = native.register_mapped_host(address, nbytes)
            mappings.append(mapping)
            addresses.append(address)
            pointers.append(pointer)
        yield mappings, torch.tensor(pointers, dtype=torch.int64, device="cuda")
    finally:
        torch.cuda.synchronize()
        for address in addresses:
            native.unregister_host(address)
        for mapping in mappings:
            mapping.close()


def dispatch(pointers, *, ids=None, weights=None, capacity=5, epoch=1, status=None,
             timeout_ms=1000, expert_rank=None, x=None, slots=2, chunk_rows=1,
             channels=1):
    if ids is None:
        ids = torch.tensor([[0, 1], [-1, -1], [1, -1]], device="cuda")
    if weights is None:
        weights = torch.tensor([[0.25, 0.75], [float("nan"), float("inf")],
                                [1.0, float("nan")]], device="cuda")
    if x is None:
        x = torch.arange(ids.shape[0] * 8, dtype=torch.float32, device="cuda").reshape(-1, 8).bfloat16()
    if expert_rank is None:
        expert_rank = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    expert_local = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    output = torch.empty(capacity, 8, dtype=torch.bfloat16, device="cuda")
    output_ids = torch.empty(capacity, 2, dtype=torch.int64, device="cuda")
    output_weights = torch.empty(capacity, 2, dtype=torch.float32, device="cuda")
    inverse = torch.empty(capacity, 2, dtype=torch.int64, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    if status is None:
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
    native.host_sm_dispatch(x, ids, weights, expert_rank, expert_local, pointers,
                           0, epoch, slots, chunk_rows, timeout_ms, output, output_ids,
                           output_weights, inverse, count, status, channels=channels)
    return output, output_ids, output_weights, inverse, count, status


def test_host_sm_local_bypass_padding_and_epoch_reuse():
    with arenas() as (_, pointers), arenas(topk=0) as (_, combine_pointers):
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        for epoch in (1, 2, 3):
            output, ids, weights, inverse, count, status = dispatch(pointers, epoch=epoch, status=status)
            assert status.item() == 0
            assert count.item() == 2
            assert ids.cpu().tolist() == [[0, 1], [1, -1], [-1, -1], [-1, -1], [-1, -1]]
            assert weights[:2].cpu().tolist() == [[0.25, 0.75], [1.0, 0.0]]
            assert inverse[:2].cpu().tolist() == [[0, 0], [0, 2]]
            assert torch.all(output[2:] == 0)
            result = torch.empty(3, 8, dtype=torch.bfloat16, device="cuda")
            scratch = torch.empty(1, 3, 8, dtype=torch.bfloat16, device="cuda")
            native.host_sm_combine(output, inverse, count, combine_pointers,
                                  0, epoch, 2, 1, 1000, result, scratch, status)
            assert status.item() == 0
            expected = torch.arange(24, dtype=torch.float32).reshape(3, 8).bfloat16()
            expected[1] = 0
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["route", "capacity"])
def test_host_sm_reports_sticky_errors(failure):
    with arenas() as (mappings, pointers):
        kwargs = {"capacity": 0} if failure == "capacity" else {
            "ids": torch.tensor([[-2, 1], [-1, -1], [1, -1]], device="cuda")}
        *_, count, status = dispatch(pointers, **kwargs)
        expected = 2 if failure == "capacity" else 1
        assert status.item() == expected
        assert count.item() == 0
        assert struct.unpack_from("<Q", mappings[0], 0)[0] == expected
        *_, count, status = dispatch(pointers, epoch=2, status=status)
        assert status.item() == expected
        assert count.item() == 0


def test_host_sm_missing_peer_times_out_without_hanging_device():
    with arenas(world=2) as (mappings, pointers):
        *_, count, status = dispatch(
            pointers, timeout_ms=20,
            expert_rank=torch.tensor([1, 1], dtype=torch.int32, device="cuda"))
        assert status.item() == 3
        assert count.item() == 0
        assert struct.unpack_from("<Q", mappings[0], 0)[0] == 3
        # Timeout exits normally, leaving unrelated CUDA work usable.
        assert torch.ones(8, device="cuda").sum().item() == 8


def test_host_sm_cpu_abort_word_is_observed():
    with arenas() as (mappings, pointers):
        struct.pack_into("<Q", mappings[0], 8, 1)
        *_, status = dispatch(pointers)
        assert status.item() == 4


def test_host_sm_rejects_bad_metadata_before_launch():
    with arenas() as (_, pointers):
        with pytest.raises(RuntimeError, match="ids.*dtype"):
            dispatch(pointers, ids=torch.zeros(3, 2, dtype=torch.float32, device="cuda"))
        with pytest.raises(RuntimeError, match="hidden"):
            dispatch(pointers, x=torch.zeros(3, 7, dtype=torch.bfloat16, device="cuda"))
        with pytest.raises(RuntimeError, match="epoch"):
            dispatch(pointers, epoch=0)


def test_host_sm_fatal_errors_surface_at_existing_cuda_synchronization():
    # A trap invalidates its CUDA context, so isolate this test in a fresh
    # process. No check_errors/status read is allowed between launch and fence.
    import subprocess
    import sys
    from pathlib import Path
    code = r'''
import ctypes
import mmap
import os
import runpy
import struct
import sys
import torch
from ep_xpress import native
helpers = runpy.run_path(sys.argv[1])
size = native.host_sm_arena_bytes(8, 2, 2, 1)
mapping = mmap.mmap(-1, size)
address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
ptr = native.register_mapped_host(address, size)
struct.pack_into("<Q", mapping, 16, 1)
pointers = torch.tensor([ptr], dtype=torch.int64, device="cuda")
ids = torch.tensor([[-2, 1], [-1, -1], [1, -1]], device="cuda")
try:
    helpers["dispatch"](pointers, ids=ids)
    torch.cuda.synchronize()
except RuntimeError:
    print("EXPECTED_TRANSPORT_TRAP", flush=True)
    os._exit(0)
os._exit(7)
'''
    completed = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve())],
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "EXPECTED_TRANSPORT_TRAP" in completed.stdout
    assert torch.ones(2, device="cuda").sum().item() == 2


def test_host_sm_rejects_mismatched_terminal_identity():
    # Simulate a peer's already-published terminal frame. The mismatch must
    # fail even when neither producer has remote payloads to transfer.
    with arenas(world=2) as (mappings, pointers):
        slot = 128
        struct.pack_into("<i", mappings[1], slot + 512, -1)
        struct.pack_into("<Q", mappings[1], slot + 520, 123)
        struct.pack_into("<Q", mappings[1], slot, (1 << 32) | 1)
        *_, count, status = dispatch(pointers)
        assert status.item() == 5
        assert count.item() == 0


@pytest.mark.parametrize("channels,tokens,slots", [
    (4, tokens, slots) for tokens in (1, 3, 4, 5, 17) for slots in (1, 2)
] + [(3, 5, 2)])
def test_host_sm_multichannel_partition_tails_and_gpu_valid_count(channels, tokens, slots):
    if native.host_sm_max_channels() < channels:
        pytest.skip(f"device cannot safely run {channels} cooperative transport channels")
    # Native chunk_rows is per channel; include a non-power-of-two grid too.
    with arenas(slots=slots, chunk_rows=2, channels=channels) as (_, pointers), \
            arenas(topk=0, slots=slots, chunk_rows=2, channels=channels) as (_, combine_pointers):
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        for epoch, rows in enumerate((tokens, max(0, tokens - 1)), start=1):
            x = torch.arange(rows * 8, dtype=torch.float32, device="cuda").reshape(rows, 8).bfloat16()
            ids = torch.tensor([0, 1], dtype=torch.int32, device="cuda").expand(rows, 2).clone()
            weights = torch.tensor([0.25, 0.75], device="cuda").expand(rows, 2).clone()
            padding = (torch.arange(rows, device="cuda") + epoch).remainder(3) == 0
            ids[padding] = -1
            weights[padding] = float("nan")
            output, out_ids, out_weights, inverse, count, status = dispatch(
                pointers, ids=ids, weights=weights, x=x, capacity=rows + 3,
                epoch=epoch, status=status, slots=slots, chunk_rows=2, channels=channels)
            assert status.item() == 0
            active = (~padding).nonzero().flatten()
            received = count.item()
            assert received == active.numel()
            order = inverse[:received, 1].argsort()
            torch.testing.assert_close(inverse[:received, 1][order], active, rtol=0, atol=0)
            assert (inverse[:received, 0] == 0).all()
            torch.testing.assert_close(output[:received][order], x[active], rtol=0, atol=0)
            torch.testing.assert_close(out_ids[:received][order], ids[active].long(), rtol=0, atol=0)
            torch.testing.assert_close(out_weights[:received][order], weights[active], rtol=0, atol=0)
            assert (output[received:] == 0).all()
            assert (out_ids[received:] == -1).all()
            assert (out_weights[received:] == 0).all()

            result = torch.empty(rows, 8, dtype=torch.bfloat16, device="cuda")
            scratch = torch.empty(1, rows, 8, dtype=torch.bfloat16, device="cuda")
            native.host_sm_combine(output, inverse, count, combine_pointers,
                                  0, epoch, slots, 2, 1000, result, scratch, status,
                                  channels=channels)
            assert status.item() == 0
            expected = x.clone()
            expected[padding] = 0
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_host_sm_rejects_channels_above_safe_cooperative_capacity():
    unsafe_channels = max(2, native.host_sm_max_channels() + 1)
    # Invalid launch metadata must fail before any arena pointer is accessed;
    # allocating only one channel also detects accidental unsafe launches.
    with arenas() as (_, pointers):
        with pytest.raises(RuntimeError, match="channel|cooperative"):
            dispatch(pointers, channels=unsafe_channels)
    assert torch.ones(2, device="cuda").sum().item() == 2


def test_host_sm_error_in_last_channel_aborts_the_operation_and_stays_sticky():
    channels = 4
    if native.host_sm_max_channels() < channels:
        pytest.skip("device cannot safely run four cooperative transport channels")
    with arenas(slots=1, chunk_rows=2, channels=channels) as (_, pointers):
        ids = torch.tensor([0, 1], dtype=torch.int32, device="cuda").expand(17, 2).clone()
        ids[-1, 0] = -2  # A late error while earlier partitions can finish first.
        weights = torch.full((17, 2), 0.5, device="cuda")
        *_, count, status = dispatch(pointers, ids=ids, weights=weights, capacity=20,
                                    slots=1, chunk_rows=2, channels=channels)
        assert status.item() == 1
        assert count.item() == 0
        ids[-1, 0] = 0
        *_, count, status = dispatch(pointers, ids=ids, weights=weights, capacity=20,
                                    epoch=2, status=status, slots=1, chunk_rows=2,
                                    channels=channels)
        assert status.item() == 1
        assert count.item() == 0
        assert torch.ones(2, device="cuda").sum().item() == 2


_PROFILE_PHASES = (
    "route_local_ns", "send_metadata_ns", "send_payload_ns", "publish_fence_ready_ns",
    "peer_poll_ns", "receive_metadata_select_ns", "receive_payload_ns",
    "ack_publish_ns", "other_control_ns",
)
_PROFILE_WAITS = (
    "idle_wait_ready_ns", "idle_wait_credit_ns", "idle_wait_both_ns",
    "idle_final_drain_ns", "idle_other_ns",
)


def _checked_profile(profile, status=0):
    fields = native.host_sm_profile_fields()
    assert len(set(fields)) == len(fields)
    rows = [dict(zip(fields, values)) for values in profile.cpu().tolist()]
    for row in rows:
        assert row["status"] == status
        assert row["elapsed_ns"] == row["end_ns"] - row["begin_ns"] > 0
        assert sum(row[field] for field in _PROFILE_PHASES) == row["elapsed_ns"]
        assert all(row[field] >= 0 for field in _PROFILE_PHASES + _PROFILE_WAITS)
        # Wait durations classify idle iterations inside the phase durations;
        # they are a second view of the same time, not additional elapsed time.
        assert sum(row[field] for field in _PROFILE_WAITS) <= row["elapsed_ns"]
        assert 0 <= row["completion_phase"] < len(_PROFILE_PHASES)
        completed = row["idle_iterations"] + row["progress_iterations"]
        assert completed <= row["loop_iterations"]
        if status == 0:
            assert completed == row["loop_iterations"]
        for field in ("credit_blocked_progress_iterations", "ready_missing_progress_iterations"):
            assert 0 <= row[field] <= row["progress_iterations"]
        for field in ("first_publish_elapsed_ns", "first_receive_elapsed_ns",
                      "first_payload_publish_elapsed_ns", "first_payload_receive_elapsed_ns"):
            assert row[field] == -1 or 0 <= row[field] <= row["elapsed_ns"]
        assert 0 <= row["last_progress_elapsed_ns"] <= row["elapsed_ns"]
        assert row["payload_written_bytes"] == row["send_payload_rows"] * 8 * 2
        assert row["payload_read_bytes"] == row["receive_payload_rows"] * 8 * 2
    return rows


def _profile_combine_inputs(pointers, tokens):
    # Padding deliberately contains nonzero data and invalid inverse entries.
    # Only the device valid count may decide which input rows participate.
    partial = torch.arange((tokens + 3) * 8, device="cuda").reshape(tokens + 3, 8).bfloat16()
    inverse = torch.full((tokens + 3, 2), -1, dtype=torch.int64, device="cuda")
    inverse[:tokens, 0] = 0
    inverse[:tokens, 1] = torch.arange(tokens, device="cuda").flip(0)
    count = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    output = torch.full((tokens + 1, 8), 42, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(pointers.numel(), tokens + 1, 8, dtype=torch.bfloat16, device="cuda")
    status = torch.zeros(1, dtype=torch.int32, device="cuda")
    return partial, inverse, count, output, scratch, status


@pytest.mark.parametrize("channels,tokens", [(1, 0), (1, 1), (1, 17), (4, 0), (4, 1), (4, 17)])
def test_host_sm_combine_profile_preserves_local_outputs_and_slot_reuse(channels, tokens):
    if channels > 1 and native.host_sm_max_channels() < channels:
        pytest.skip(f"device cannot safely run {channels} cooperative transport channels")
    with arenas(topk=0, slots=1, chunk_rows=2, channels=channels) as (_, pointers):
        partial, inverse, count, output, scratch, status = _profile_combine_inputs(pointers, tokens)
        expected = torch.zeros_like(output)
        expected[:tokens] = partial[:tokens].flip(0)
        profile = torch.empty(channels, len(native.host_sm_profile_fields()), dtype=torch.int64, device="cuda")
        for epoch in (1, 2, 3):
            profile.fill_(-777)
            native.host_sm_combine(partial, inverse, count, pointers, 0, epoch, 1, 2, 1000,
                                   output, scratch, status, channels=channels,
                                   profile=None if epoch == 1 else profile)
            assert status.item() == 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            if epoch == 1:
                assert (profile == -777).all().item()
                continue
            rows = _checked_profile(profile)
            assert [row["local_rows"] for row in rows] == [
                tokens * (channel + 1) // channels - tokens * channel // channels
                for channel in range(channels)]
            for row in rows:
                assert row["send_payload_rows"] == row["receive_payload_rows"] == 0
                assert row["send_frames"] == row["terminal_sent"] == 1
                assert row["receive_frames"] == row["terminal_received"] == 0
                assert row["first_publish_elapsed_ns"] >= 0
                assert row["first_receive_elapsed_ns"] == -1
                assert row["first_payload_publish_elapsed_ns"] == -1
                assert row["first_payload_receive_elapsed_ns"] == -1


@pytest.mark.parametrize("waiting", ["ready", "both", "drain"])
def test_host_sm_combine_profile_accounts_missing_peer_waits_and_bytes(waiting):
    tokens = 3 if waiting == "both" else 0
    with arenas(world=2, topk=0, slots=1, chunk_rows=2) as (mappings, pointers):
        partial, inverse, count, output, scratch, status = _profile_combine_inputs(pointers, tokens)
        inverse[:tokens, 0] = 1
        if waiting == "drain":
            # The peer has finished sending, but never ACKs our terminal frame.
            slot = 128
            struct.pack_into("<i", mappings[1], slot + 512, -1)
            struct.pack_into("<Q", mappings[1], slot + 520, 0)
            struct.pack_into("<Q", mappings[1], slot, (1 << 32) | 1)
        profile = torch.empty(1, len(native.host_sm_profile_fields()), dtype=torch.int64, device="cuda")
        native.host_sm_combine(partial, inverse, count, pointers, 0, 1, 1, 2, 20,
                               output, scratch, status, profile=profile)
        assert status.item() == 3
        row = _checked_profile(profile, status=3)[0]
        field = "idle_final_drain_ns" if waiting == "drain" else f"idle_wait_{waiting}_ns"
        assert row[field] > 0
        assert row["idle_iterations"] > 0
        assert row["local_rows"] == row["receive_payload_rows"] == 0
        assert row["send_payload_rows"] == (2 if waiting == "both" else 0)
        assert row["payload_written_bytes"] == (32 if waiting == "both" else 0)
        assert row["send_frames"] == 1
        assert row["terminal_sent"] == (0 if waiting == "both" else 1)
        assert row["terminal_received"] == (1 if waiting == "drain" else 0)
        assert torch.ones(2, device="cuda").sum().item() == 2


def test_host_sm_combine_profile_records_invalid_inverse_failure():
    with arenas(topk=0) as (_, pointers):
        partial, inverse, count, output, scratch, status = _profile_combine_inputs(pointers, 1)
        inverse[0, 0] = -1
        profile = torch.empty(1, len(native.host_sm_profile_fields()), dtype=torch.int64, device="cuda")
        native.host_sm_combine(partial, inverse, count, pointers, 0, 1, 2, 1, 1000,
                               output, scratch, status, profile=profile)
        assert status.item() == 1
        row = _checked_profile(profile, status=1)[0]
        assert row["local_rows"] == row["send_payload_rows"] == row["receive_payload_rows"] == 0
        assert row["first_payload_publish_elapsed_ns"] == row["first_payload_receive_elapsed_ns"] == -1
        assert torch.ones(2, device="cuda").sum().item() == 2


@pytest.mark.parametrize("invalid", ["dtype", "device", "dimensions", "fields", "channels", "contiguity"])
def test_host_sm_combine_rejects_invalid_profile_before_launch(invalid):
    with arenas(topk=0) as (_, pointers):
        partial, inverse, count, output, scratch, status = _profile_combine_inputs(pointers, 1)
        fields = len(native.host_sm_profile_fields())
        options = {"dtype": torch.int64, "device": "cuda"}
        shape = (1, fields)
        if invalid == "dtype":
            options["dtype"] = torch.float32
        elif invalid == "device":
            options["device"] = "cpu"
        elif invalid == "dimensions":
            shape = (fields,)
        elif invalid == "fields":
            shape = (1, fields - 1)
        elif invalid == "channels":
            shape = (2, fields)
        profile = (torch.empty(1, fields * 2, **options)[:, ::2] if invalid == "contiguity"
                   else torch.empty(shape, **options))
        with pytest.raises(RuntimeError, match="profile"):
            native.host_sm_combine(partial, inverse, count, pointers, 0, 1, 2, 1, 1000,
                                   output, scratch, status, profile=profile)
        assert status.item() == 0
        assert (output == 42).all().item()  # Init must not run after validation fails.
