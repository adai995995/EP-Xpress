"""Focused stream transitions and borrowed-workspace lifetime validation."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs are required")]


def _worker(rank, init_file, fused):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    context = None
    try:
        context = Context(None, [0, 0, 1, 1], Config(
            128, 2, 8, transport="host_sm", channels=2, chunk_bytes=4096,
            decode_max_tokens=8, timeout_seconds=30), device=rank)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        workspace = context.create_workspace(8, 4, fused_combine=fused)
        outputs, expected = [], []
        # Switch APIs and producers, reuse borrowed storage, and queue delayed
        # consumers on another stream before returning to a direct producer.
        for step, mode in enumerate(("async", "direct0", "direct0", "direct1", "async", "direct0")):
            stream = streams[1 if mode == "direct1" else 0]
            with torch.cuda.stream(stream):
                x = torch.full((4, 128), step + rank + 1, dtype=torch.bfloat16, device="cuda")
                routes = ([[0, 1], [2, 3], [0, 2], [-1, -1]] if step % 2 == 0 else
                          [[2, 3], [0, 1], [1, 3], [0, -1]])
                ids = torch.tensor(routes, dtype=torch.int64, device="cuda")
                weights = torch.full((4, 2), 0.5, device="cuda")
                options = dict(stream=stream, workspace=workspace, fused_combine=fused,
                               active_channels=1 + step % 2)
                if mode == "async":
                    batch = context.dispatch_async(x, ids, weights, **options).wait(stream=stream)
                else:
                    batch = context.dispatch_on_stream(x, ids, weights, **options)
                partial = (batch.hidden.float() * (batch.weights * (batch.local_ids >= 0)).sum(1)[:, None]).bfloat16()
                reference = x.float() * (ids >= 0).sum(1)[:, None] * 0.5
                if mode == "async":
                    pending = context.combine_async(partial, batch.handle, stream=stream)
                    consumer = streams[1]
                    with torch.cuda.stream(consumer):
                        out = pending.wait(stream=consumer)
                        torch.cuda._sleep(2_000_000)
                        outputs.append(out.clone())
                    expected.append(reference)
                else:
                    outputs.append(context.combine_on_stream(partial, batch.handle, stream=stream).clone())
                    expected.append(reference)
        # close must drain direct work before unregistering mapped host pages.
        context.close()
        context = None
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.float(), reference, rtol=0, atol=0)
    finally:
        if context is not None:
            context.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fused", [False, True])
def test_direct_async_transitions_and_cross_stream_workspace_consumers(fused):
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-stream-test-") as root:
        torch.multiprocessing.spawn(_worker, args=(os.path.join(root, "rendezvous"), fused), nprocs=2, join=True)
