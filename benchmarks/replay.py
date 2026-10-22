"""Deterministic multi-GPU dispatch/combine correctness and timing replay.

Run with torchrun --standalone --nproc-per-node=N --module benchmarks.replay.
The synthetic expert function is not a model or an expert-GEMM benchmark.
"""

import argparse
from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist

from ep_xpress import Config, Context


def inputs(rank, world, tokens, hidden, case, iteration, device):
    count = tokens
    if case == "empty":
        count = 0 if rank == iteration % world else max(tokens - rank, 1)
    if case == "all_empty":
        count = 0
    x = ((torch.arange(count * hidden, device=device).reshape(count, hidden) + 7 * rank)
         .remainder(29).float() / 128).to(torch.bfloat16)
    rows = []
    for token in range(count):
        row = []
        for slot in range(4):
            owner = rank if case == "local" else (0 if case == "hotspot" else
                    (rank + token + slot // 2 + iteration) % world)
            row.append(owner * 4 + slot)
        rows.append(row)
    ids = torch.tensor(rows, dtype=torch.int64, device=device).reshape(count, 4)
    weights = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float32,
                           device=device).expand(count, 4).contiguous()
    return x, ids, weights


def expert_partial(batch, rank):
    present = batch.local_ids >= 0
    factors = (rank * 4 + batch.local_ids + 1).float() / 8
    coefficient = torch.where(present, factors * batch.weights, 0).sum(dim=1)
    return (batch.hidden.float() * coefficient[:, None]).to(torch.bfloat16).contiguous()


def reference(x, ids, weights, rank, world):
    remote = torch.zeros_like(x, dtype=torch.float32)
    local = torch.zeros_like(x)
    for expert_rank in range(world):
        coefficient = torch.where(ids // 4 == expert_rank,
                                  (ids + 1).float() / 8 * weights, 0).sum(dim=1)
        partial = (x.float() * coefficient[:, None]).to(torch.bfloat16)
        if expert_rank == rank:
            local = partial
        else:
            remote.add_(partial.float())
    return (remote.to(torch.bfloat16).float() + local.float()).to(torch.bfloat16)


def run(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    config = Config(hidden_size=args.hidden, topk=4, max_tokens_per_rank=args.tokens,
                    chunk_bytes=args.chunk_bytes, slots=args.slots,
                    timeout_seconds=args.timeout)
    totals = []
    cases = args.cases.split(",")
    with Context(dist.group.WORLD, [owner for owner in range(world) for _ in range(4)],
                 config, device=device) as context:
        for iteration in range(args.iterations):
            for case in cases:
                x, ids, weights = inputs(rank, world, args.tokens, args.hidden, case, iteration, device)
                torch.cuda.synchronize()
                dist.barrier()
                start = time.perf_counter()
                dispatch = context.dispatch_async(x, ids, weights, tag=[iteration, case])
                batch = dispatch.wait()
                partial = expert_partial(batch, rank)
                combine = context.combine_async(partial, batch.handle, tag=[iteration, case])
                output = combine.wait()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                expected = reference(x, ids, weights, rank, world)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert batch.valid_count == batch.hidden.shape[0]
                # Independently reconstruct receiver rows without using RoutePlan.
                expected_rows = []
                for source in range(world):
                    source_x, source_ids, _ = inputs(source, world, args.tokens, args.hidden,
                                                      case, iteration, device)
                    expected_rows.append(source_x[(source_ids // 4 == rank).any(dim=1)])
                expected_hidden = torch.cat(expected_rows)
                torch.testing.assert_close(batch.hidden, expected_hidden, rtol=0, atol=0)
                record = {"case": case, "iteration": iteration,
                          "elapsed_ms": elapsed * 1000,
                          "dispatch": dispatch.metrics, "combine": combine.metrics}
                gathered = [None] * world
                dist.all_gather_object(gathered, record)
                if rank == 0:
                    totals.append({"case": case, "iteration": iteration,
                                   "max_rank_ms": max(item["elapsed_ms"] for item in gathered),
                                   "d2h_bytes": sum(item[phase]["d2h_bytes"] for item in gathered
                                                    for phase in ("dispatch", "combine")),
                                   "h2d_bytes": sum(item[phase]["h2d_bytes"] for item in gathered
                                                    for phase in ("dispatch", "combine")),
                                   "max_active_exports": max(item[phase]["max_active_exports"]
                                                             for item in gathered for phase in ("dispatch", "combine"))})
                if args.check_handles:
                    try:
                        context.combine_async(partial, batch.handle)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("consumed handle was accepted")

        # Keep two dispatch handles alive and combine them in reverse order.
        if args.check_handles:
            x, ids, weights = inputs(rank, world, args.tokens, args.hidden, "mixed", 0, device)
            first = context.dispatch_async(x, ids, weights, tag="first")
            second = context.dispatch_async(x, ids, weights, tag="second")
            first_batch, second_batch = first.wait(), second.wait()
            for batch in (second_batch, first_batch):
                output = context.combine_async(expert_partial(batch, rank), batch.handle).wait()
                torch.testing.assert_close(output, reference(x, ids, weights, rank, world), rtol=0, atol=0)

    if rank == 0:
        print(json.dumps({"status": "passed", "kind": "synthetic_correctness_replay",
                          "world_size": world, "hidden_size": args.hidden,
                          "tokens_per_rank": args.tokens, "slots": args.slots,
                          "chunk_bytes": args.chunk_bytes, "graph": "eager",
                          "precision": "bf16", "transport": "host_ce",
                          "results": totals}, indent=2), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--chunk-bytes", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--cases", default="mixed,empty,local,hotspot,all_empty")
    parser.add_argument("--check-handles", action="store_true")
    args = parser.parse_args()
    unknown = set(args.cases.split(",")) - {"mixed", "empty", "local", "hotspot", "all_empty"}
    if unknown:
        parser.error("unknown replay case")
    run(args)


if __name__ == "__main__":
    main()
