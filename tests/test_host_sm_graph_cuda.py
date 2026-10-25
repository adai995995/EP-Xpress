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
        routes = [patterns[step % 2][i % 4] for i in range(n)]
        if world > 3:
            # Multiple queued peers per control warp; destinations change every
            # replay, including same-rank dedup, absent routes and zero readers.
            routes = []
            for i in range(n):
                peer = (rank + step + i) % world
                routes.append(([-1, -1] if i % 4 == 3 else
                               [2 * peer, 2 * peer + 1] if i % 4 == 0 else
                               [2 * peer, 2 * ((peer + 1) % world) + 1]))
        ids = torch.tensor(routes,
                           dtype=torch.int32 if id_bits == 32 else torch.int64, device="cuda").reshape(n, 2)
        weights = torch.full((n, 2), 0.5, device="cuda")
        if world > 3:
            # Distinguish rows/lanes and route weights so premature reuse or a
            # metadata/payload mismatch cannot hide behind constant payloads.
            rows = torch.arange(n, device="cuda")
            columns = torch.arange(hidden, device="cuda")
            x = ((rank + step + 1 + rows[:, None] % 7 + columns[None, :] % 4) * 8).to(torch.bfloat16)
            weights[:, 0] = 0.25 + ((rows + step) % 2) * 0.5
            weights[:, 1] = 1.0 - weights[:, 0]
        return x, ids, weights

    def reference(data):
        return data[0].float() * (data[2] * (data[1] >= 0)).sum(1)[:, None]

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
                results.append((output.clone(), reference(current)))
                # Eager prefill uses pair queues and the same device counters.
                eager = inputs(4, step)
                workspace = context.create_workspace(world * 4, 4)
                result = forward(eager, workspace, 2, False)
                results.append((result.clone(), reference(eager)))
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
                results.append((output.clone(), reference(edge)))
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
                if report["schema_version"] == 3:
                    for role, fields in report["role_phase_fields"].items():
                        begin = 0 if role == "tx" else values[role + "_begin_offset_ns"]
                        assert sum(values[k] for k in fields) == values[role + "_end_offset_ns"] - begin
                    assert values["rx_payload_bytes"] == hidden * 2 * sum(
                        values[f"ctrl{g}_payload_rows"] for g in range(2))
                    assert values["end_ns"] - values["begin_ns"] == values["elapsed_ns"]
                else:
                    assert values["elapsed_ns"] == sum(values[k] for k in report["phase_fields"])
                assert values["status"] == 0 and values["epoch"] > 0
            # A new session resets counters without recapturing the graph.
            context.start_decode_profile()
            context.prepare_graph_replay()
            graphs[1][0].replay()
            report = context.stop_decode_profile()
            roles = 2 if (profile == "combine" and
                          (os.environ.get("EP_XPRESS_DECODE_SPLIT_TX_RX") == "1" or
                           os.environ.get("EP_XPRESS_DECODE_PACKET_ROWS", "0") != "0")) else 1
            assert report["dropped_samples"] == 0 and len(report["samples"]) == 2 * roles
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


@pytest.mark.parametrize("hidden,id_bits", [(2048, 32), (258, 64)])
def test_dispatch_pipeline_multi_peer_stage_reuse(hidden, id_bits):
    """Each control group reuses two stages across dynamic routes and epochs."""
    if os.environ.get("EP_XPRESS_DECODE_DISPATCH_PIPELINE") != "1":
        pytest.skip("enable the experimental decode dispatch pipeline")
    if torch.cuda.device_count() < 8:
        pytest.skip("eight CUDA GPUs are required")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-pipeline-test-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, 262144, id_bits, False, 8, hidden, True),
            nprocs=8, join=True)


def test_dispatch_pipeline_profile_graph_reset_and_overflow():
    """Role timelines survive changing routes, Graph replay, reset and overflow."""
    if os.environ.get("EP_XPRESS_DECODE_DISPATCH_PIPELINE") != "1":
        pytest.skip("enable dispatch pipeline")
    if torch.cuda.device_count() < 5:
        pytest.skip("five CUDA GPUs are required")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-pipeline-profile-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), True, True, 262144, 64, "dispatch", 5, 258),
            nprocs=5, join=True)
