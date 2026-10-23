"""Focused decode mailbox lifetime, routing and legacy-rounding comparison."""

import ctypes
import mmap
import os
from pathlib import Path
import tempfile
import time

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required")]


def _routes(case, rank):
    if case == 0:
        return [[2, 3, -1]] * (5 if rank == 0 else 0)
    if case == 3:
        return []
    lengths = (16, 16, 16) if case == 4 else ((9, 3, 0) if case == 1 else (4, 8, 6))
    result = []
    for token in range(lengths[rank]):
        if token % 5 == 0:
            result.append([-1, -1, -1])
        elif token % 5 == 1:
            result.append([rank * 2, rank * 2 + 1, -1])
        else:
            result.append([0, 2, 4])
    return result


def _mailbox_worker(rank, directory, barrier):
    torch.cuda.set_device(rank)
    world, hidden, topk, channels, capacity, tokens = 3, 64, 3, 2, 32, 16
    mappings, addresses, pointers = [], [], {}
    prepared = None
    try:
        for phase, size in _sizes(hidden, topk, channels, world, tokens).items():
            mapped = []
            for peer in range(world):
                fd = os.open(os.path.join(directory, f"{phase}-{peer}"), os.O_RDWR)
                mapping = mmap.mmap(fd, size)
                os.close(fd)
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                mapped.append(native.register_mapped_host(address, size))
                mappings.append(mapping)
                addresses.append(address)
            pointers[phase] = torch.tensor(mapped, dtype=torch.int64, device="cuda")
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        prepared = native.prepare_host_sm(
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device="cuda"),
            pointers["dispatch"], pointers["combine"], status, rank, hidden, topk,
            2, 2, 2, 10000, channels, diagnostics=True,
            mailbox_pointers=pointers["mailbox"], mailbox_tokens=tokens)
        combine_epoch = 0
        # The middle group holds two dispatched layouts concurrently and
        # combines them in reverse order with fresh operation epochs.
        for cases in ((0,), (1, 2), (3,), (4,)):
            handles = []
            for case in cases:
                routes = _routes(case, rank)
                n = len(routes)
                ids = torch.tensor(routes, dtype=torch.int64, device="cuda").reshape(n, topk)
                weights = torch.ones(n, topk, dtype=torch.float32, device="cuda")
                x = torch.randn(n, hidden, dtype=torch.bfloat16, device="cuda")
                output = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
                output_ids = torch.empty(capacity, topk, dtype=torch.int64, device="cuda")
                output_weights = torch.empty(capacity, topk, dtype=torch.float32, device="cuda")
                inverse = torch.empty(capacity, 2, dtype=torch.int64, device="cuda")
                count = torch.empty(1, dtype=torch.int32, device="cuda")
                owner_rows = torch.empty(world, capacity, dtype=torch.int32, device="cuda")
                owner_counts = torch.empty(world, dtype=torch.int32, device="cuda")
                expected = torch.empty(n, dtype=torch.int64, device="cuda")
                local = torch.empty(n, dtype=torch.int32, device="cuda")
                prepared.dispatch(x, ids, weights, case + 1, output, output_ids,
                    output_weights, inverse, count, owner_rows, owner_counts,
                    active_channels=1 + case % 2, identity=case + 1,
                    expected_ranks=expected, local_rows=local)
                torch.cuda.synchronize()
                assert status.item() == 0
                masks = [sum(1 << owner for owner in {e // 2 for e in row if e >= 0}) for row in routes]
                assert expected.cpu().tolist() == masks
                for token, compact in enumerate(local.cpu().tolist()):
                    if masks[token] & (1 << rank):
                        assert inverse[compact].cpu().tolist() == [rank, token]
                    else:
                        assert compact == -1
                torch.manual_seed(1900 + case * 17 + rank)
                # Random BF16 values exercise cancellation and intermediate
                # BF16 rounding, rather than an exactly representable identity.
                partial = (torch.randn(capacity, hidden, device="cuda") * 31).bfloat16()
                handles.append((case, n, partial, inverse, count, owner_rows,
                                owner_counts, expected, local))
            for case, n, partial, inverse, count, owner_rows, owner_counts, expected, local in reversed(handles):
                combine_epoch += 1
                reference = torch.empty(n, hidden, dtype=torch.bfloat16, device="cuda")
                remote = torch.empty(world, n, hidden, dtype=torch.bfloat16, device="cuda")
                result = torch.empty_like(reference)
                prepared.combine(partial, inverse, count, owner_rows, owner_counts,
                    combine_epoch, reference, remote, active_channels=1 + case % 2,
                    identity=combine_epoch)
                torch.cuda.synchronize()
                assert status.item() == 0
                barrier.wait(timeout=60)
                if case == 0 and rank == 2:
                    # A nonparticipant has no receive/ACK obligation. The two
                    # communicating ranks must finish before rank 2 launches.
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and not all(
                        Path(directory, f"finished-{peer}").exists() for peer in (0, 1)
                    ):
                        time.sleep(0.005)
                    assert all(Path(directory, f"finished-{peer}").exists() for peer in (0, 1))
                prepared.combine(partial, inverse, count, owner_rows, owner_counts,
                    combine_epoch, result, None, active_channels=1 + case % 2,
                    identity=combine_epoch, expected_ranks=expected, local_rows=local)
                torch.cuda.synchronize()
                assert status.item() == 0
                torch.testing.assert_close(result.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0)
                if case == 0:
                    Path(directory, f"finished-{rank}").touch()
                barrier.wait(timeout=60)
        # A duplicate READY epoch could expose old data; reject before launch.
        with pytest.raises(RuntimeError, match="epochs must increase"):
            prepared.combine(partial, inverse, count, owner_rows, owner_counts,
                combine_epoch, result, None, expected_ranks=expected, local_rows=local)
    finally:
        torch.cuda.synchronize()
        prepared = None
        for address in addresses:
            native.unregister_host(address)
        for mapping in mappings:
            mapping.close()


def _sizes(hidden, topk, channels, world, tokens):
    return {
        "dispatch": native.host_sm_arena_bytes(hidden, topk, 2, 2, channels=channels),
        "combine": native.host_sm_arena_bytes(hidden, 0, 2, 2, channels=channels, queue_peers=world),
        "mailbox": native.host_sm_mailbox_bytes(hidden, tokens, world),
    }


def test_mailbox_sparse_readers_reuse_reverse_handles_and_legacy_rounding():
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    if native.host_sm_max_channels(3) < 2:
        pytest.skip("insufficient cooperative residency")
    with tempfile.TemporaryDirectory(prefix="ep-xpress-mailbox-test-", dir="/dev/shm") as directory:
        for phase, size in _sizes(64, 3, 2, 3, 16).items():
            for rank in range(3):
                with open(os.path.join(directory, f"{phase}-{rank}"), "wb") as output:
                    os.posix_fallocate(output.fileno(), 0, size)
        context = torch.multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        torch.multiprocessing.spawn(_mailbox_worker, args=(directory, barrier), nprocs=3, join=True)
