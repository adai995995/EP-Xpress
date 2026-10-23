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
