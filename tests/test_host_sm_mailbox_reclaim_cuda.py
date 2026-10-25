"""Mailbox lifetime without dispatch, host rendezvous or per-call GPU waits."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required")]


def _reclaim_worker(rank, init_file, tokens):
    import torch.distributed as dist
    world, hidden = 3, 258
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=world)
    # Force more than 32 live packet tasks per CTA even when the build requests
    # extra packet channels (the native override clamps to prepared capacity).
    channels = 1 if tokens > 128 else 4
    context = Context(None, list(range(world)), Config(
        hidden, 1, world * tokens, transport="host_sm", channels=channels,
        decode_max_tokens=tokens, fatal_errors=True), device=rank)
    stream = torch.cuda.current_stream()
    context._last_stream = stream
    packet_rows = native.load().host_sm_packet_rows()
    max_packet_tasks = 0

    def selected(producer, destination, token, step):
        # Empty routes, skipped mailbox generations and changing row counts.
        if step % 8 in (0, 1):
            # Only rank zero receives. The other ranks can finish their local
            # result and reach the next publication before its delayed reader.
            return producer != 0 and destination == 0 and (token + step) % 3 != 2
        return (producer != destination and step % 7 != 3 and
                (producer * 3 + destination + token + step) % 5 < 3)

    prepared = []
    try:
        # Prepare all inputs first: H2D tensor construction must not hide a
        # missing credit check by serializing the actual combine submissions.
        for step in range(32):
            inverse, owner_lists, values = [], [[] for _ in range(world)], []
            for destination in range(world):
                for token in range(tokens):
                    if selected(rank, destination, token, step):
                        owner_lists[destination].append(len(inverse))
                        inverse.append([destination, token])
                        values.append(rank + token + step + 1)
            rows = len(inverse)
            if packet_rows:
                max_packet_tasks = max(max_packet_tasks,
                    len({(d, t // packet_rows) for d, t in inverse}))
            owner_rows = torch.zeros((world, rows), dtype=torch.int32, device="cuda")
            for destination, indices in enumerate(owner_lists):
                if indices:
                    owner_rows[destination, :len(indices)] = torch.tensor(indices, device="cuda")
            partial = torch.tensor(values, dtype=torch.bfloat16, device="cuda")[:, None].expand(rows, hidden).contiguous()
            inverse = torch.tensor(inverse, dtype=torch.int64, device="cuda").reshape(rows, 2)
            counts = torch.tensor([len(x) for x in owner_lists], dtype=torch.int32, device="cuda")
            count = torch.tensor([rows], dtype=torch.int32, device="cuda")
            masks, expected = [], []
            for token in range(tokens):
                peers = [p for p in range(world) if selected(p, rank, token, step)]
                masks.append(sum(1 << p for p in peers))
                expected.append(sum(p + token + step + 1 for p in peers))
            masks = torch.tensor(masks, dtype=torch.int64, device="cuda")
            local_rows = torch.full((tokens,), -1, dtype=torch.int32, device="cuda")
            output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
            expected = torch.tensor(expected, dtype=torch.bfloat16, device="cuda")[:, None].expand(tokens, hidden)
            prepared.append((partial, inverse, count, owner_rows, counts, masks, local_rows, output, expected))
        if tokens > 128 and packet_rows:
            assert channels == 1 and max_packet_tasks > 32
        torch.cuda.synchronize()
        dist.barrier()
        # Delays rotate between ranks. Do not insert a dispatch or rendezvous
        # between calls: a fast producer can reach reuse before a reader ACKs.
        for step, (partial, inverse, count, owner_rows, counts, masks, local_rows, output, _) in enumerate(prepared):
            if step % 8 == 0 and rank == 0:
                torch.cuda._sleep(5_000_000)
            elif step % 8 not in (0, 1) and step % world == rank:
                torch.cuda._sleep(500_000)
            context._prepared.combine(partial, inverse, count, owner_rows, counts,
                step + 1, output, None, active_channels=1 + step % channels,
                expected_ranks=masks, local_rows=local_rows)
        stream.synchronize()
        for *_, output, expected in prepared:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        # Each rank drains only its local work; healthy collective close keeps
        # all mappings pinned until every remote reader has also completed.
        context.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("tokens", [7, 129])
def test_deferred_mailbox_reuse_with_lagging_readers(tokens):
    if (os.environ.get("EP_XPRESS_DECODE_DEFER_RECLAIM") != "1" and
            os.environ.get("EP_XPRESS_DECODE_PACKET_ROWS", "0") == "0"):
        pytest.skip("enable deferred mailbox reclamation")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-reclaim-") as root:
        torch.multiprocessing.spawn(_reclaim_worker,
            args=(os.path.join(root, "rendezvous"), tokens), nprocs=3, join=True)
