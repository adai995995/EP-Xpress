"""Fused mailbox scheduling preserves dynamic reduction and buffer lifetimes."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="four CUDA GPUs are required")]


def _worker(rank, init_file, hidden, diagnostics=False):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=4)
    context = Context(None, [i // 2 for i in range(8)], Config(
        hidden, 4, 8, transport="host_sm", channels=2, chunk_bytes=262144,
        decode_max_tokens=8, decode_single_frame=True, fatal_errors=True, diagnostics=diagnostics), device=rank)
    if not diagnostics:
        context.enable_cuda_graph()
    stream = torch.cuda.Stream()
    factors = (1.0, 100.0, -100.0, 0.01)

    def inputs(n, step):
        values = torch.arange(n * hidden, device="cuda", dtype=torch.float32).reshape(n, hidden)
        x = (torch.sin(values * 0.13 + step + rank) * 3).bfloat16()
        routes = []
        for i in range(n):
            # Four contributors, deduplicated experts, empty routes and partial tails.
            cases = [[0, 2, 4, 6], [2, 3, 6, -1], [-1] * 4,
                     [0, 1, 4, 7], [6, 4, 2, 0]]
            routes.append(cases[(i + step) % len(cases)])
        ids = torch.tensor(routes, device="cuda", dtype=torch.int64).reshape(n, 4)
        weights = torch.tensor([0.17, 0.23, 0.29, 0.31], device="cuda").expand(n, 4).contiguous()
        return x, ids, weights

    def reference(data):
        x, ids, weights = data
        parts = []
        for peer in range(4):
            weight = (weights * ((ids >= 0) & (ids // 2 == peer))).sum(1)
            parts.append((x.float() * weight[:, None] * factors[peer]).bfloat16())
        remote = torch.zeros_like(x, dtype=torch.float32)
        for peer in range(4):
            if peer != rank:
                remote += parts[peer].float()
        return (remote.bfloat16().float() + parts[rank].float()).bfloat16()

    def forward(data, workspace):
        batch = context.dispatch_on_stream(*data, workspace=workspace,
            active_channels=2, fused_combine=True)
        weights = (batch.weights * (batch.local_ids >= 0)).sum(1)
        partial = (batch.hidden.float() * weights[:, None] * factors[rank]).bfloat16()
        # Stagger actual producer completion to exercise partial READY masks.
        # Delay stays on the GPU, including during graph replay.
        torch.cuda._sleep(200000 * (rank % 3))
        return context.combine_on_stream(partial, batch.handle)

    graph = None
    try:
        n = rank + 1
        if diagnostics:
            # Operation diagnostics are intentionally incompatible with Graphs.
            # Exercise their identity/status barriers on serialized eager calls.
            results, retained_workspaces = [], []
            for step in range(6):
                with torch.cuda.stream(stream):
                    current = inputs(0 if rank == 0 and step % 2 else n, step)
                    workspace = context.create_workspace(10, current[0].shape[0], fused_combine=True)
                    results.append((forward(current, workspace).clone(), reference(current)))
                    retained_workspaces.append(workspace)
            torch.cuda.synchronize()
            for actual, expected in results:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            context.check_errors()
            return
        with torch.cuda.stream(stream):
            data = inputs(n, 0)
            workspace = context.create_workspace(10, n, fused_combine=True)
            forward(data, workspace)
        context.prepare_graph_capture()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward(data, workspace)
        results = []
        retained_workspaces = [workspace]
        for step in range(1, 7):
            with torch.cuda.stream(stream):
                fresh = inputs(n, step)
                for static, value in zip(data, fresh):
                    static.copy_(value)
                context.prepare_graph_replay(stream)
                graph.replay()
                results.append((output.clone(), reference(fresh)))
                # Alternate captured generations with an eager, empty-owner shape.
                eager = inputs(0 if rank == 0 else 2, step)
                eager_workspace = context.create_workspace(6, eager[0].shape[0], fused_combine=True)
                results.append((forward(eager, eager_workspace).clone(), reference(eager)))
                retained_workspaces.append(eager_workspace)
        torch.cuda.synchronize()
        for actual, expected in results:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        context.check_errors()
    finally:
        if graph is not None:
            graph.reset()
        context.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("hidden", [2048, 258, 6144])
def test_ready_prefetch_dynamic_graph_order_and_fallback(hidden):
    """Vector staging, scalar tails, and the bounded shared-memory fallback."""
    if os.environ.get("EP_XPRESS_DECODE_READY_PREFETCH") != "1":
        pytest.skip("enable ready-peer prefetch")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-ready-prefetch-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), hidden), nprocs=4, join=True)


@pytest.mark.parametrize("hidden,diagnostics", [(2048, False), (258, True)])
def test_control_warp_dynamic_graph_order_and_ack(hidden, diagnostics):
    """One-warp polling preserves dynamic masks, exact sums and slot lifetime."""
    if os.environ.get("EP_XPRESS_DECODE_CONTROL_WARP") != "1":
        pytest.skip("enable decode control warp")
    if os.environ.get("EP_XPRESS_DECODE_READY_PREFETCH", "0") != "0":
        pytest.skip("exercise direct full-CTA reception")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-control-warp-") as root:
        torch.multiprocessing.spawn(_worker,
            args=(os.path.join(root, "rendezvous"), hidden, diagnostics), nprocs=4, join=True)
