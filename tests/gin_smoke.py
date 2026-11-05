"""Focused multi-node dynamic routing/lifetime check (run with torchrun).

Requires GDAKI, two hosts and the same NCCL device headers/runtime. This is a
correctness check, not a communication or model performance benchmark.
"""

import argparse
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from ep_xpress import Config, Context
from ep_xpress.comm import get_nccl_comm_handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    mapping = [(e * 5 + 1) % world for e in range(world * 3)]
    # Multiplication by 5 permutes the tested 4/16 rank groups. Arbitrary maps
    # including ranks with no experts are legal; no contiguous partition assumption.
    local_globals = torch.tensor([e for e, owner in enumerate(mapping) if owner == rank], device=device)
    config = Config(hidden_size=args.hidden, topk=4, max_tokens_per_rank=64,
                    transport="host_sm", internode_transport="gin", timeout_seconds=60)
    ctx = Context(get_nccl_comm_handle(dist.group.WORLD), mapping, config)
    if args.graph:
        ctx.enable_cuda_graph()
    stream = torch.cuda.current_stream(device)

    def inputs(n, epoch, skew=False):
        gen = torch.Generator(device=device).manual_seed(9001 + 29 * rank + 97 * epoch)
        x = torch.randn((n, args.hidden), generator=gen, device=device).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(len(mapping), generator=gen, device=device)[:4]
                           for _ in range(n)]) if n else torch.empty((0, 4), dtype=torch.int64, device=device)
        if skew and n:
            ids[:, :3] = torch.tensor([e for e, owner in enumerate(mapping) if owner == 0][:3], device=device)
        if n:
            ids[-1, -1] = -1
        w = torch.rand((n, 4), generator=gen, device=device)
        w = torch.where(ids >= 0, w, 0)
        return x, ids, w

    def expert(batch):
        valid = batch.local_ids >= 0
        if local_globals.numel():
            scale = (local_globals[batch.local_ids.clamp(min=0)] + 1).float() / 128
        else:
            scale = torch.zeros_like(batch.weights)
        factor = torch.where(valid, scale * batch.weights, 0).sum(dim=1, keepdim=True)
        return (batch.hidden.float() * factor).to(torch.bfloat16)

    def reference(x, ids, weights):
        owners = torch.tensor(mapping, device=device)[ids.clamp(min=0)]
        result = torch.zeros_like(x, dtype=torch.float32)
        for peer in range(world):
            coefficient = torch.where((ids >= 0) & (owners == peer),
                                     (ids + 1).float() / 128 * weights, 0).sum(1, keepdim=True)
            result += (x.float() * coefficient).to(torch.bfloat16).float()
        return result.to(torch.bfloat16)

    def check(x, ids, weights, batch, result):
        torch.testing.assert_close(result, reference(x, ids, weights), rtol=0, atol=0)
        # Independently verify destination deduplication and per-slot expert mapping.
        gathered = [None] * world
        dist.all_gather_object(gathered, (x.cpu(), ids.cpu(), weights.cpu()))
        count = batch.valid_count.item()
        rows = batch.handle.inverse_map[:count].cpu().tolist()
        assert len(set(map(tuple, rows))) == count, "duplicate token/rank dispatch"
        expected = {(src, token) for src, (_, route, _) in enumerate(gathered)
                    for token, route_row in enumerate(route.tolist())
                    if any(e >= 0 and mapping[e] == rank for e in route_row)}
        assert set(map(tuple, rows)) == expected, (rank, set(map(tuple, rows)), expected)
        local_map = {e: i for i, e in enumerate([e for e, p in enumerate(mapping) if p == rank])}
        for row, (src, token) in enumerate(rows):
            sx, si, sw = gathered[src]
            torch.testing.assert_close(batch.hidden[row].cpu(), sx[token], rtol=0, atol=0)
            want_ids = torch.tensor([local_map.get(e, -1) for e in si[token].tolist()])
            torch.testing.assert_close(batch.local_ids[row].cpu(), want_ids, rtol=0, atol=0)
            torch.testing.assert_close(batch.weights[row].cpu(), torch.where(want_ids >= 0, sw[token], 0), rtol=0, atol=0)
        assert (batch.local_ids[count:] == -1).all().item()

    # Empty ranks, all empty, balanced and skewed changing routes, mixed async/direct.
    for epoch in range(7):
        n = 0 if epoch == 0 or (epoch == 1 and rank % 3 == 0) else 1 + (rank * 3 + epoch) % 11
        x, ids, weights = inputs(n, epoch, skew=epoch == 3)
        if args.graph or epoch % 2 == 0:
            batch = ctx.dispatch_on_stream(x, ids, weights, recv_capacity=world * 16, stream=stream)
            result = ctx.combine_on_stream(expert(batch), batch.handle, stream=stream)
        else:
            batch = ctx.dispatch_async(x, ids, weights, recv_capacity=world * 16, stream=stream).wait(stream)
            result = ctx.combine_async(expert(batch), batch.handle, stream=stream).wait(stream)
        check(x, ids, weights, batch, result)
        if rank == 0:
            print(json.dumps({"stage": "eager", "epoch": epoch, "correct": True}), flush=True)

    # Operation-scoped inverse maps must survive another dispatch's staging reuse.
    first, second = inputs(3, 31), inputs(6, 32, skew=True)
    b0 = ctx.dispatch_on_stream(*first, recv_capacity=world * 6)
    b1 = ctx.dispatch_on_stream(*second, recv_capacity=world * 6)
    y0 = ctx.combine_on_stream(expert(b0), b0.handle)
    y1 = ctx.combine_on_stream(expert(b1), b1.handle)
    check(*first, b0, y0)
    check(*second, b1, y1)

    if args.graph:
        n = 5
        x, ids, weights = inputs(n, 10)
        ws = ctx.create_workspace(world * n, n)
        for _ in range(2):
            batch = ctx.dispatch_on_stream(x, ids, weights, workspace=ws, recv_capacity=world * n)
            result = ctx.combine_on_stream(expert(batch), batch.handle)
        ctx.prepare_graph_capture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            batch = ctx.dispatch_on_stream(x, ids, weights, workspace=ws, recv_capacity=world * n)
            result = ctx.combine_on_stream(expert(batch), batch.handle)
        for epoch in range(8):
            ctx.prepare_graph_replay()
            for dst, src in zip((x, ids, weights), inputs(n, 20 + epoch, skew=epoch % 2 == 0)):
                dst.copy_(src)
            graph.replay()
            check(x, ids, weights, batch, result)
        del graph
    ctx.close()
    print(json.dumps({"stage": "PASS", "rank": rank, "graph": args.graph,
                      "transport": ctx.transport_info}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
