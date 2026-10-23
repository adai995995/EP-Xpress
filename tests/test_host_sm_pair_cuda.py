"""Focused pair-queue lifetime/progress checks, requiring three CUDA GPUs."""

import ctypes
import mmap
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required")]


def _pair_worker(rank, directory, absent, barrier):
    torch.cuda.set_device(rank)
    world, channels, slots, chunk_rows, hidden, capacity = 3, 2, 1, 2, 8, 20
    size = native.host_sm_arena_bytes(hidden, 0, slots, chunk_rows,
                                     channels=channels, queue_peers=world)
    mappings, addresses, mapped = [], [], []
    try:
        for peer in range(world):
            fd = os.open(os.path.join(directory, str(peer)), os.O_RDWR)
            mapping = mmap.mmap(fd, size)
            os.close(fd)
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            mapped.append(native.register_mapped_host(address, size))
            addresses.append(address)
            mappings.append(mapping)
        pointers = torch.tensor(mapped, dtype=torch.int64, device="cuda")
        fields = native.host_sm_profile_fields()
        profile = torch.empty(channels * world, len(fields), dtype=torch.int64, device="cuda")
        for epoch in ((1,) if absent else (1, 2)):
            lengths = (17, 13, 0 if epoch == 1 else 1)
            n = lengths[rank]
            values = torch.arange(n, dtype=torch.float32, device="cuda") + rank * 64 + epoch
            partial = values[:, None].expand(n, hidden).contiguous().bfloat16()
            inverse = torch.empty(n, 2, dtype=torch.int64, device="cuda")
            inverse[:, 0] = (torch.arange(n, device="cuda") + rank) % world
            inverse[:, 1] = torch.arange(n, device="cuda")
            count = torch.tensor([n], dtype=torch.int32, device="cuda")
            output = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
            scratch = torch.empty(world, capacity, hidden, dtype=torch.bfloat16, device="cuda")
            status = torch.zeros(1, dtype=torch.int32, device="cuda")
            torch.cuda.synchronize()
            barrier.wait(timeout=60)
            if not (absent and rank == 2):
                native.host_sm_combine(
                    partial, inverse, count, pointers, rank, epoch, slots, chunk_rows,
                    1000 if absent else 10000, output, scratch, status,
                    channels=channels, profile=profile)
                torch.cuda.synchronize()
                if absent:
                    assert status.item() == 3
                    # Rank 2 never consumes or ACKs anything. The rank 0/1
                    # pair must nevertheless drain more than one slot and its
                    # terminal before the unrelated pair's bounded timeout.
                    peer = 1 - rank
                    records = profile.cpu().tolist()
                    total_sent = 0
                    for channel in range(channels):
                        record = dict(zip(fields, records[channel * world + peer]))
                        assert record["status"] == 0, record
                        assert record["terminal_sent"] == record["terminal_received"] == 1
                        total_sent += record["send_frames"]
                    assert total_sent > channels * slots
                    expected = torch.zeros(capacity, hidden, dtype=torch.bfloat16)
                    for row in range(lengths[peer]):
                        if (row + peer) % world == rank:
                            expected[row] = peer * 64 + epoch + row
                    torch.testing.assert_close(scratch[peer].cpu(), expected, rtol=0, atol=0)
                else:
                    assert status.item() == 0
                    local = torch.zeros(capacity, hidden, dtype=torch.bfloat16)
                    remote = torch.zeros(capacity, hidden, dtype=torch.float32)
                    for source, length in enumerate(lengths):
                        for row in range(length):
                            if (row + source) % world == rank:
                                value = torch.tensor(source * 64 + epoch + row).bfloat16()
                                if source == rank:
                                    local[row] = value
                                else:
                                    remote[row] += value.float()
                    expected = (remote.bfloat16().float() + local.float()).bfloat16()
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            barrier.wait(timeout=60)
    finally:
        torch.cuda.synchronize()
        for address in addresses:
            native.unregister_host(address)
        for mapping in mappings:
            mapping.close()


@pytest.mark.parametrize("absent", [False, True], ids=["reuse-and-empty", "unrelated-peer-absent"])
def test_pair_queue_independent_progress_and_reuse(absent):
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    if native.host_sm_max_channels(3) < 2:
        pytest.skip("insufficient cooperative residency for the focused test")
    size = native.host_sm_arena_bytes(8, 0, 1, 2, channels=2, queue_peers=3)
    # Use the same RAM-backed mapping type as the real context. Docker's
    # default /tmp overlay filesystem does not support CUDA host registration.
    with tempfile.TemporaryDirectory(prefix="ep-xpress-pair-test-", dir="/dev/shm") as directory:
        for rank in range(3):
            with open(os.path.join(directory, str(rank)), "wb") as output:
                os.posix_fallocate(output.fileno(), 0, size)
        context = torch.multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        torch.multiprocessing.spawn(_pair_worker, args=(directory, absent, barrier),
                                    nprocs=3, join=True)
