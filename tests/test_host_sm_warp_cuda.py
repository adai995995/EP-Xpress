"""Focused Tx/Rx subgroup progress and legacy-rounding checks on three GPUs."""

import ctypes
import mmap
import os
import random
import struct
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required")]


def _rows(case, producer, frame_rows):
    if case in (1, 5):
        # Long, asymmetric opposite directions; one completely empty rank.
        lengths = ((0, 4 * frame_rows + 7, 0),
                   (3 * frame_rows + 5, 9, 0), (0, 0, 0))[producer]
    elif case == 3:
        lengths = tuple((3, 0, 5)[producer] if peer == producer else 0
                        for peer in range(3))
    elif case == 4:
        lengths = (0, 0, 0)
    else:
        lengths = tuple(4 * frame_rows + 7 - (producer + peer) % 3
                        for peer in range(3))
    rows = [(peer, token) for peer, length in enumerate(lengths)
            for token in range(length)]
    # Dispatch compaction is not owner-ordered. Lists must address scattered
    # rows, including the same original token contributed by several ranks.
    random.Random(7100 + case * 17 + producer).shuffle(rows)
    return rows


def _sizes(hidden, frame_rows, slots=2):
    return {
        "dispatch": native.host_sm_arena_bytes(hidden, 1, slots, 4, channels=2),
        "prepared": native.host_sm_arena_bytes(
            hidden, 0, slots, frame_rows, channels=2, queue_peers=3),
        "legacy": native.host_sm_arena_bytes(
            hidden, 0, slots, frame_rows, channels=2, queue_peers=3),
    }


def _assert_drained(mapping, rank, epoch, active, hidden, frame_rows,
                    owner_counts, exact_frames, slots=2):
    # Inspect protocol state only after the kernel completes. Slot geometry is
    # shared with the existing arena ABI: 128-byte queue and 640-byte slot head.
    row_stride = ((16 + hidden * 2 + 15) // 16) * 16
    slot_stride = ((640 + frame_rows * row_stride + 127) // 128) * 128
    channel_stride = len(mapping) // (2 * 3)
    for channel in range(2):
        for peer in range(3):
            base = (channel * 3 + peer) * channel_stride
            assert struct.unpack_from("<QQQ", mapping, base) == (0, 0, 0)
            records = []
            for slot in range(slots):
                offset = base + 128 + slot * slot_stride
                ready, ack = struct.unpack_from("<QQ", mapping, offset)
                assert ready == ack, (rank, channel, peer, slot, ready, ack)
                records.append((ready, struct.unpack_from("<i", mapping, offset + 512)[0]))
            if channel >= active or peer == rank:
                continue
            latest, rows = max(records)
            assert latest >> 32 == epoch and rows == -1, (rank, channel, peer, records)
            if exact_frames:
                count = owner_counts[peer]
                length = count * (channel + 1) // active - count * channel // active
                frames = (length + frame_rows - 1) // frame_rows + 1
                assert latest & 0xffffffff == frames, (rank, channel, peer, frames, records)


def _warp_worker(rank, directory, frame_rows, hidden, slots, offset, barrier):
    torch.cuda.set_device(rank)
    world, channels = 3, 2
    local_tokens = 4 * frame_rows + 7
    capacity = world * local_tokens + 4
    mappings, addresses, pointers = {}, [], {}
    prepared = None
    try:
        for phase, size in _sizes(hidden, frame_rows, slots).items():
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
            torch.arange(world, dtype=torch.int32, device="cuda"),
            torch.zeros(world, dtype=torch.int32, device="cuda"),
            pointers["dispatch"], pointers["prepared"], status, rank, hidden, 1,
            slots, 4, frame_rows, 10000, channels, diagnostics=True,
            combine_warp_specialization=True)
        for epoch, active in ((1, 1), (2, 2), (3, 1), (4, 2), (5, 1), (6, 2)):
            rows = _rows(epoch, rank, frame_rows)
            inverse = torch.full((capacity, 2), -1, dtype=torch.int64, device="cuda")
            if rows:
                inverse[:len(rows)] = torch.tensor(rows, dtype=torch.int64, device="cuda")
            owner_rows = torch.full((world, capacity), -1, dtype=torch.int32, device="cuda")
            owner_counts_cpu = []
            for peer in range(world):
                indices = [i for i, (owner, _) in enumerate(rows) if owner == peer]
                owner_counts_cpu.append(len(indices))
                if indices:
                    owner_rows[peer, :len(indices)] = torch.tensor(indices, dtype=torch.int32, device="cuda")
            owner_counts = torch.tensor(owner_counts_cpu, dtype=torch.int32, device="cuda")
            count = torch.tensor([len(rows)], dtype=torch.int32, device="cuda")
            torch.manual_seed(8200 + epoch * 13 + rank)
            # Nonexact sums exercise the remote BF16 rounding before local add.
            # Hidden=258 also leaves a one-BF16-pair tail for a 128-thread group.
            partial = (torch.randn(capacity, hidden, device="cuda") * 31).bfloat16()
            if offset:
                shifted = torch.empty(partial.numel() + offset, dtype=partial.dtype, device="cuda")
                shifted[offset:].copy_(partial.flatten())
                partial = shifted[offset:].view_as(partial)
            reference = torch.empty(local_tokens, hidden, dtype=torch.bfloat16, device="cuda")
            result = torch.empty_like(reference)
            reference_remote = torch.empty(world, local_tokens, hidden, dtype=torch.bfloat16, device="cuda")
            remote = torch.empty(reference_remote.numel() + offset, dtype=torch.bfloat16,
                                 device="cuda")[offset:].view_as(reference_remote)
            torch.cuda.synchronize()
            barrier.wait(timeout=60)
            native.host_sm_combine(
                partial, inverse, count, pointers["legacy"], rank, epoch,
                slots, frame_rows, 10000, reference, reference_remote, status,
                identity=epoch, channels=active)
            torch.cuda.synchronize()
            assert status.item() == 0
            _assert_drained(mappings["legacy"][rank], rank, epoch, active,
                            hidden, frame_rows, owner_counts_cpu, False, slots)
            barrier.wait(timeout=60)
            fields = native.host_sm_profile_fields()
            profile = (torch.zeros((active * world, len(fields)), dtype=torch.int64,
                                   device="cuda") if epoch % 2 == 0 else None)
            prepared.combine(partial, inverse, count, owner_rows, owner_counts,
                epoch, result, remote, active_channels=active, identity=epoch, profile=profile)
            torch.cuda.synchronize()
            assert status.item() == 0
            if profile is not None:
                counters = profile.cpu()
                assert (counters[:, fields.index("status")] == 0).all()
                expected = sum(n for peer, n in enumerate(owner_counts_cpu) if peer != rank) * hidden * 2
                assert counters[:, fields.index("payload_written_bytes")].sum().item() == expected
                assert (counters[:, fields.index("elapsed_ns")] > 0).all()
            torch.testing.assert_close(result.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0)
            torch.testing.assert_close(remote.view(torch.int16), reference_remote.view(torch.int16), rtol=0, atol=0)
            _assert_drained(mappings["prepared"][rank], rank, epoch, active,
                            hidden, frame_rows, owner_counts_cpu, True, slots)
            barrier.wait(timeout=60)
    finally:
        torch.cuda.synchronize()
        prepared = None
        for address in addresses:
            native.unregister_host(address)
        for phase_mappings in mappings.values():
            for mapping in phase_mappings:
                mapping.close()


@pytest.mark.parametrize(
    "frame_rows,hidden,slots,offset",
    [(4, 258, 2, 0), (256, 258, 2, 0), (16, 2048, 2, 0),
     (16, 264, 4, 0), (16, 2048, 2, 2), (3, 264, 2, 0)],
    ids=["scalar-tail", "subgroup-metadata-tail", "vector-model-width",
         "vector-tail-four-slots", "unaligned-output", "unaligned-frame"],
)
def test_prefill_warp_pair_legacy_rounding_and_slot_reuse(frame_rows, hidden, slots, offset, monkeypatch):
    # Spawned workers import native afresh and exercise the opt-in vector build.
    monkeypatch.setenv("EP_XPRESS_PREFILL_READ_UNROLL", "4")
    monkeypatch.setenv("EP_XPRESS_PREFILL_WRITE_UNROLL", "4")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    if native.host_sm_max_channels(3) < 2:
        pytest.skip("insufficient cooperative residency for the focused test")
    with tempfile.TemporaryDirectory(prefix="ep-xpress-warp-test-", dir="/dev/shm") as directory:
        for phase, size in _sizes(hidden, frame_rows, slots).items():
            for rank in range(3):
                with open(os.path.join(directory, f"{phase}-{rank}"), "wb") as output:
                    os.posix_fallocate(output.fileno(), 0, size)
        context = torch.multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        torch.multiprocessing.spawn(_warp_worker, args=(directory, frame_rows, hidden, slots, offset, barrier),
                                    nprocs=3, join=True)
