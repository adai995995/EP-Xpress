"""Changing routes, bucket switches and eager/graph reuse of host queues."""
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
    context = Context(None, [0, 0, 1, 1], Config(
        128, 2, 8, transport="host_sm", channels=2, chunk_bytes=4096,
        decode_max_tokens=8, fatal_errors=True, combine_warp_specialization=True), device=rank)
    context.enable_cuda_graph()
    capture_stream = torch.cuda.Stream()
    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, saved, results = {}, [], []

    def inputs(n, step):
        x = torch.full((n, 128), rank + step + 1, dtype=torch.bfloat16, device="cuda")
        patterns = ([[0, 1], [2, 3], [0, 2], [-1, -1]],
                    [[2, 3], [0, 1], [1, 3], [0, -1]])
        ids = torch.tensor([patterns[step % 2][i % 4] for i in range(n)],
                           dtype=torch.int64, device="cuda").reshape(n, 2)
        weights = torch.full((n, 2), 0.5, device="cuda")
        return x, ids, weights

    def forward(data, workspace, channels, fused):
        batch = context.dispatch_on_stream(*data, workspace=workspace,
            active_channels=channels, fused_combine=fused)
        partial = (batch.hidden.float() *
                   (batch.weights * (batch.local_ids >= 0)).sum(1)[:, None]).bfloat16()
        return context.combine_on_stream(partial, batch.handle)

    try:
        # Five retained buckets exceed the adapter's former four-entry cache;
        # the empty-rank bucket verifies zero-row kernels and remote experts.
        for key in range(5):
            n = key if rank == 0 else key + 1
            with torch.cuda.stream(capture_stream):
                data = inputs(n, 0)
                workspace = context.create_workspace(2 * key + 1, n, fused_combine=fused)
                forward(data, workspace, 1 + key % 2, fused)
            context.prepare_graph_capture()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                out = forward(data, workspace, 1 + key % 2, fused)
            graphs[key] = (graph, data, out)
            saved.append(workspace)
        # Capture must not consume a generation. Every replay computes fresh
        # routing from mutated GPU IDs, including local dedup and absent ranks.
        for step, key in enumerate((0, 4, 1, 3, 0, 2, 4, 1)):
            stream = replay_streams[step % 2]
            with torch.cuda.stream(stream):
                graph, data, output = graphs[key]
                context.prepare_graph_replay(stream)
                current = inputs(data[0].shape[0], step + 1)
                for static, value in zip(data, current):
                    static.copy_(value)
                graph.replay()
                reference = current[0].float() * (current[1] >= 0).sum(1)[:, None] * 0.5
                results.append((output.clone(), reference))
                # Eager prefill uses pair queues and the same device counters.
                eager = inputs(4, step)
                workspace = context.create_workspace(8, 4)
                result = forward(eager, workspace, 2, False)
                results.append((result.clone(), eager[0].float() * (eager[1] >= 0).sum(1)[:, None] * 0.5))
                saved.append(workspace)
        torch.cuda.synchronize()
        for actual, expected in results:
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
        for graph, _, _ in graphs.values():
            graph.reset()
        graphs.clear()
        context.check_errors()
    finally:
        context.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fused", [False, True])
def test_graph_dynamic_routing_bucket_switch_and_eager_prefill(fused):
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-graph-test-") as root:
        torch.multiprocessing.spawn(_worker, args=(os.path.join(root, "rendezvous"), fused), nprocs=2, join=True)
