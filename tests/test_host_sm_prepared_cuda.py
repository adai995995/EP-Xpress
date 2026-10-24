"""Focused prepared transport checks for selective readers and GPU owner lists."""

import ctypes
import mmap
import os
import struct
import tempfile
import time

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required")]


def _routes(epoch, source):
    lengths = {1: (17, 0, 0), 2: (9, 7, 5), 3: (0, 3, 0), 4: (5, 0, 7)}
    routes = []
    for row in range(lengths[epoch][source]):
        if epoch in (1, 3):
            routes.append([2, 3])  # Same destination, distinct experts.
        else:
            owner = (source + row) % 3
            routes.append([owner * 2, ((owner + 1) % 3) * 2 + 1])
    return routes


def _prepared_worker(rank, directory, barrier, hidden, chunk_rows, offset):
    torch.cuda.set_device(rank)
    world, channels, slots, capacity = 3, 2, 2, 32
    mappings, addresses, pointers = {}, [], {}
    prepared = None
    try:
        for phase, topk, queue_peers in (("dispatch", 2, 1), ("combine", 0, world)):
            size = native.host_sm_arena_bytes(hidden, topk, slots, chunk_rows,
                                             channels=channels, queue_peers=queue_peers)
            mapped = []
            mappings[phase] = []
            for peer in range(world):
                fd = os.open(os.path.join(directory, f"{phase}-{peer}"), os.O_RDWR)
                mapping = mmap.mmap(fd, size)
                os.close(fd)
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                mapped.append(native.register_mapped_host(address, size))
                addresses.append(address)
                mappings[phase].append(mapping)
            pointers[phase] = torch.tensor(mapped, dtype=torch.int64, device="cuda")
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        prepared = native.prepare_host_sm(
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device="cuda"),
            pointers["dispatch"], pointers["combine"], status, rank, hidden, 2,
            slots, chunk_rows, chunk_rows, 10000, channels, diagnostics=True)
        for epoch, active in ((1, 1), (2, 2), (3, 1), (4, 2)):
            routes = _routes(epoch, rank)
            n = len(routes)
            ids = torch.tensor(routes, dtype=torch.int64, device="cuda").reshape(n, 2)
            weights = torch.tensor([0.25, 0.75], device="cuda").expand(n, 2).contiguous()
            values = torch.arange(n, dtype=torch.float32, device="cuda") + rank * 16 + epoch
            pattern = (torch.arange(hidden, device="cuda") % 8).float()
            x = (values[:, None] + pattern).bfloat16()
            storage = torch.empty(capacity * hidden + offset, dtype=torch.bfloat16, device="cuda")
            output = storage[offset:].view(capacity, hidden)
            output_ids = torch.empty(capacity, 2, dtype=torch.int64, device="cuda")
            output_weights = torch.empty(capacity, 2, dtype=torch.float32, device="cuda")
            inverse = torch.empty(capacity, 2, dtype=torch.int64, device="cuda")
            count = torch.empty(1, dtype=torch.int32, device="cuda")
            owner_rows = torch.empty(world, capacity, dtype=torch.int32, device="cuda")
            owner_counts = torch.empty(world, dtype=torch.int32, device="cuda")
            torch.cuda.synchronize()
            barrier.wait(timeout=60)
            if epoch == 1 and rank == 2:
                # Rank 2 is not a payload reader. Rank 0 must publish every
                # frame through two slots before rank 2 launches its kernel.
                row_stride = ((16 + 2 * 12 + hidden * 2 + 15) // 16) * 16
                expected_frames = (17 + chunk_rows - 1) // chunk_rows
                slot_stride = ((640 + chunk_rows * row_stride + 127) // 128) * 128
                deadline = time.monotonic() + 5
                progress = 0
                while time.monotonic() < deadline:
                    sequences = [struct.unpack_from("<Q", mappings["dispatch"][0],
                                                    128 + slot * slot_stride)[0]
                                 for slot in range(slots)]
                    progress = max(((seq & 0xffffffff) for seq in sequences if seq >> 32 == epoch), default=0)
                    if progress >= expected_frames:
                        break
                    time.sleep(0.005)
                assert progress >= expected_frames, "unrelated rank blocked payload-slot reuse"
            prepared.dispatch(x, ids, weights, epoch, output, output_ids, output_weights,
                              inverse, count, owner_rows, owner_counts,
                              active_channels=active, identity=epoch)
            torch.cuda.synchronize()
            assert status.item() == 0
            expected = {}
            for source in range(world):
                for token, experts in enumerate(_routes(epoch, source)):
                    if any(expert // 2 == rank for expert in experts):
                        expected[(source, token)] = source * 16 + epoch + token
            valid = count.item()
            actual_inverse = inverse[:valid].cpu().tolist()
            assert len(actual_inverse) == len(expected)
            assert {tuple(row) for row in actual_inverse} == set(expected)
            for row, key in enumerate(actual_inverse):
                torch.testing.assert_close(output[row].cpu(),
                    (expected[tuple(key)] + pattern.cpu()).bfloat16(), rtol=0, atol=0)
            covered = []
            for owner, length in enumerate(owner_counts.cpu().tolist()):
                indices = owner_rows[owner, :length].cpu().tolist()
                assert len(indices) == sum(row[0] == owner for row in actual_inverse)
                assert all(0 <= row < valid and actual_inverse[row][0] == owner for row in indices)
                covered.extend(indices)
            assert sorted(covered) == list(range(valid))
            # Simulate local weighted expert output; padding remains zero.
            partial = (output.float() * (output_weights * (output_ids >= 0)).sum(1, keepdim=True)).bfloat16()
            result = torch.empty(n, hidden, dtype=torch.bfloat16, device="cuda")
            remote = torch.empty(world, n, hidden, dtype=torch.bfloat16, device="cuda")
            prepared.combine(partial, inverse, count, owner_rows, owner_counts,
                             epoch, result, remote, active_channels=active, identity=epoch)
            torch.cuda.synchronize()
            assert status.item() == 0
            torch.testing.assert_close(result, x, rtol=0, atol=0)
            barrier.wait(timeout=60)
        # Empty operations still publish diagnostic identity in their separate
        # terminal. A mismatched rank must not silently succeed without frames.
        empty = torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
        empty_ids = torch.empty(0, 2, dtype=torch.int64, device="cuda")
        empty_weights = torch.empty(0, 2, dtype=torch.float32, device="cuda")
        prepared.dispatch(empty, empty_ids, empty_weights, 5, output, output_ids,
                          output_weights, inverse, count, owner_rows, owner_counts,
                          active_channels=1, identity=6 if rank == 1 else 5)
        torch.cuda.synchronize()
        assert status.item() in (4, 5)
        barrier.wait(timeout=60)
    finally:
        torch.cuda.synchronize()
        prepared = None
        for address in addresses:
            native.unregister_host(address)
        for phase_mappings in mappings.values():
            for mapping in phase_mappings:
                mapping.close()


@pytest.mark.parametrize("hidden,chunk_rows,offset", [
    (8, 2, 0), (2048, 7, 0), (264, 3, 0), (258, 3, 0), (2048, 7, 2),
])
def test_prepared_selective_readers_owner_lists_and_active_channel_reuse(
        hidden, chunk_rows, offset, monkeypatch):
    monkeypatch.setenv("EP_XPRESS_DISPATCH_READ_UNROLL", "4")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    if native.host_sm_max_channels(3) < 2:
        pytest.skip("insufficient cooperative residency for the focused test")
    with tempfile.TemporaryDirectory(prefix="ep-xpress-prepared-test-", dir="/dev/shm") as directory:
        for phase, topk, peers in (("dispatch", 2, 1), ("combine", 0, 3)):
            size = native.host_sm_arena_bytes(hidden, topk, 2, chunk_rows, channels=2, queue_peers=peers)
            for rank in range(3):
                with open(os.path.join(directory, f"{phase}-{rank}"), "wb") as output:
                    os.posix_fallocate(output.fileno(), 0, size)
        context = torch.multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        torch.multiprocessing.spawn(_prepared_worker, args=(directory, barrier, hidden, chunk_rows, offset), nprocs=3, join=True)
