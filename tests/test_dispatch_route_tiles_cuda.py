"""Dynamic route tails and full receive stages, independent of model shapes."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, native

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 5,
    reason="five CUDA GPUs are required")]


def _worker(rank, rendezvous, topk):
    import torch.distributed as dist
    from datetime import timedelta
    world, hidden, rows = 5, 258, 64
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    # Noncontiguous ownership; the transport must consult the live expert map.
    owners = [(i * 3) % world for i in range(world * 40)]
    experts = [[i for i, owner in enumerate(owners) if owner == r] for r in range(world)]
    try:
        ctx = Context(None, owners, Config(
            hidden, topk, rows, transport="host_sm", channels=2,
            chunk_bytes=262144, decode_max_tokens=rows,
            decode_single_frame=True, combine_warp_specialization=True,
            diagnostics=False, fatal_errors=True), device=rank)
        ctx.enable_cuda_graph()
        stream = torch.cuda.Stream()

        def inputs(step):
            # Encode every source rank and row across columns using exact
            # BF16 values 1/2, so swapped addresses cannot hide behind parity.
            tags = rank * rows + torch.arange(rows, device="cuda")
            bits = torch.arange(hidden, device="cuda") % (world * rows - 1).bit_length()
            x = (1 + ((tags[:, None] >> bits[None, :]) & 1)).bfloat16()
            ids = []
            for row in range(rows):
                if step == 2 and (rank == 0 or (row + rank) % 5 == 0):
                    ids.append([-1] * topk)
                else:
                    # Full 32-row stages from multiple producers, followed
                    # by mixed destinations, absent routes and skew changes.
                    ids.append([experts[(1 + step) % world if step % 2 == 0 else
                                        (rank + row + step + k // 8) % world][k]
                                for k in range(topk)])
            ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
            weights = ((torch.arange(rows, device="cuda")[:, None] +
                        torch.arange(topk, device="cuda")[None, :] + step) % 2 + 1).float() / 16
            return x, ids, weights

        def forward(data, workspace):
            batch = ctx.dispatch_on_stream(*data, workspace=workspace,
                                           active_channels=2, fused_combine=True)
            partial = (batch.hidden.float() *
                       (batch.weights * (batch.local_ids >= 0)).sum(1)[:, None]).bfloat16()
            return ctx.combine_on_stream(partial, batch.handle)

        graph = None
        try:
            workspace = ctx.create_workspace(world * rows, rows, fused_combine=True)
            with torch.cuda.stream(stream):
                data = inputs(0)
                forward(data, workspace)
                forward(data, workspace)
            torch.cuda.synchronize()
            ctx.prepare_graph_capture()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward(data, workspace)
            saved = []
            for step in (0, 1, 2, 3, 4, 1):
                with torch.cuda.stream(stream):
                    ctx.prepare_graph_replay(stream)
                    current = inputs(step)
                    for static, value in zip(data, current):
                        static.copy_(value)
                    graph.replay()
                    expected = current[0].float() * (
                        current[2] * (current[1] >= 0)).sum(1)[:, None]
                    saved.append((output.clone(), expected))
            torch.cuda.synchronize()
            failure = None
            try:
                for output, expected in saved:
                    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)
            except AssertionError as error:
                failure = str(error)
            failures = [None] * world
            dist.all_gather_object(failures, failure)
            assert not any(failures), failures
        finally:
            if graph is not None:
                graph.reset()
            ctx.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("topk", [3, 33])
def test_route_tiles_full_stages_and_non_power_of_two_topk(topk):
    if os.environ.get("EP_XPRESS_DECODE_DISPATCH_PIPELINE") != "1":
        pytest.skip("enable dispatch pipeline")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit unavailable")
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-route-tiles-") as root:
        torch.multiprocessing.spawn(_worker, args=(os.path.join(root, "rendezvous"), topk),
                                    nprocs=5, join=True)
