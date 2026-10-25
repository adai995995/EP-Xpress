"""Changing routes, bucket switches and eager/graph reuse of host queues."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs are required")]


def _worker(rank, init_file, fused, single_frame=False, chunk_bytes=4096, id_bits=64, profile=False, world=2, hidden=128, wide=False):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world)
    context = Context(None, [i // 2 for i in range(world * 2)], Config(
        hidden, 2, 64 if wide else 8, transport="host_sm", channels=2, chunk_bytes=chunk_bytes,
        decode_max_tokens=64 if wide else 8, fatal_errors=True, combine_warp_specialization=True,
        decode_single_frame=single_frame), device=rank)
    if profile:
        context.prepare_decode_profile(max_samples_per_channel=3,
                                       stage=profile if isinstance(profile, str) else "combine")
    context.enable_cuda_graph()
    capture_stream = torch.cuda.Stream()
    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, saved, results = {}, [], []

    def inputs(n, step):
        x = torch.full((n, hidden), rank + step + 1, dtype=torch.bfloat16, device="cuda")
        patterns = ([[0, 1], [2, 3], [0, 2], [-1, -1]],
                    [[2, 3], [0, 1], [1, 3], [0, -1]])
        if world == 3:
            patterns = ([[0, 1], [2, 4], [0, 4], [-1, -1]],
                        [[4, 5], [0, 3], [1, 5], [2, -1]])
        ids = torch.tensor([patterns[step % 2][i % 4] for i in range(n)],
                           dtype=torch.int32 if id_bits == 32 else torch.int64, device="cuda").reshape(n, 2)
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
            base = (0, 33, 45, 63, 7)[key] if wide else key
            n = base if rank == 0 else base + 1
            with torch.cuda.stream(capture_stream):
                data = inputs(n, 0)
                workspace = context.create_workspace(base + (world - 1) * (base + 1), n, fused_combine=fused)
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
        if profile:
            context.start_decode_profile()
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
                workspace = context.create_workspace(world * 4, 4)
                result = forward(eager, workspace, 2, False)
                results.append((result.clone(), eager[0].float() * (eager[1] >= 0).sum(1)[:, None] * 0.5))
                saved.append(workspace)
        # Exact configured source bound, maximally skewed token ownership.
        # The other rank still participates with zero local rows. One channel
        # exercises the full frame; small-frame configurations use the ring.
        if single_frame:
            with torch.cuda.stream(replay_streams[0]):
                bound = 64 if wide else 8
                edge = inputs(bound if rank == 0 else 0, 3)
                workspace = context.create_workspace(bound, edge[0].shape[0], fused_combine=True)
                output = forward(edge, workspace, 1, True)
                results.append((output.clone(), edge[0].float() * (edge[1] >= 0).sum(1)[:, None] * 0.5))
                saved.append(workspace)
        torch.cuda.synchronize()
        for actual, expected in results:
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
        if profile:
            report = context.stop_decode_profile()
            assert report["status"] == 0 and report["dropped_samples"] > 0
            assert report["samples"]
            for sample in report["samples"]:
                values = dict(zip(report["fields"], sample["values"]))
                assert values["elapsed_ns"] == sum(values[k] for k in report["phase_fields"])
                assert values["status"] == 0 and values["epoch"] > 0
            # A new session resets counters without recapturing the graph.
            context.start_decode_profile()
            context.prepare_graph_replay()
            graphs[1][0].replay()
            report = context.stop_decode_profile()
            assert report["dropped_samples"] == 0 and len(report["samples"]) == 2
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


@pytest.mark.parametrize("chunk_bytes, id_bits", [(4096, 32), (256, 64)])
def test_single_frame_graph_and_small_frame_fallback(chunk_bytes, id_bits):
    """Both protocols support fresh routes, empty ranks, skew and slot reuse."""
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-single-frame-test-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, chunk_bytes, id_bits),
            nprocs=2, join=True)


@pytest.mark.parametrize("stage", ["combine", "dispatch"])
def test_decode_profile_graph_replay_reset_and_overflow(stage):
    """Bounded counters survive changed routes and do not overwrite on overflow."""
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-profile-test-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, 4096, 32, stage),
            nprocs=2, join=True)


@pytest.mark.parametrize("hidden", [2048, 258])
def test_indexed_dispatch_multi_peer_graph_reuse(hidden):
    """Multiple ACK owners, real route changes, empty ranks and scalar tails."""
    if torch.cuda.device_count() < 3:
        pytest.skip("three CUDA GPUs are required")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-indexed-test-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, 131072, 64, False, 3, hidden),
            nprocs=3, join=True)


def test_indexed_dispatch_multi_warp_route_tail():
    """Index ballots span more than one warp with a partial tail and skew."""
    if torch.cuda.device_count() < 3:
        pytest.skip("three CUDA GPUs are required")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-indexed-tail-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, 262144, 32, False, 3, 258, True),
            nprocs=3, join=True)
