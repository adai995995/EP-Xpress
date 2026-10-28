"""Prepared BF16 EP primitives, with independent GPU event timing.

Run using torchrun. Routes change on every submission, outside timed regions.
This is a communication benchmark, not model inference or physical PCIe telemetry.
NCCL AG/RS are explicitly raw collective references, without EP layout processing.
"""
import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from ep_xpress import Config, Context


TRANSPORT_OPTIONS = (
    "EP_XPRESS_HOST_READ64", "EP_XPRESS_DISPATCH_SPLIT",
    "EP_XPRESS_DISPATCH_DIRECT", "EP_XPRESS_DISPATCH_RETIRE",
    "EP_XPRESS_PACKET_PIPELINE", "EP_XPRESS_RING_SPLIT",
    "EP_XPRESS_ROUTE_PACKED", "EP_XPRESS_RX_FENCE",
    "EP_XPRESS_READY_BATCH", "EP_XPRESS_PACKET_RECV_LOCAL",
    "EP_XPRESS_DISPATCH_DEFER_RECLAIM", "EP_XPRESS_REUSE_RETURN_INDEX",
    "EP_XPRESS_DISPATCH_WRITE_UNROLL", "EP_XPRESS_DECODE_PACKET_ROWS",
    "EP_XPRESS_DECODE_PACKET_CHANNELS", "EP_XPRESS_DECODE_DISPATCH_INDEXED",
    "EP_XPRESS_DECODE_DISPATCH_PIPELINE", "EP_XPRESS_DECODE_DESCRIPTOR_BATCH",
    "EP_XPRESS_DECODE_PUBLISH_BATCH", "EP_XPRESS_DECODE_PEER_BATCH",
    "EP_XPRESS_PREFILL_READ_UNROLL", "EP_XPRESS_PREFILL_WRITE_UNROLL",
    "EP_XPRESS_DISPATCH_READ_UNROLL", "EP_XPRESS_COMBINE_LAST_DATA",
)


def summary(values):
    ordered = sorted(values)
    return {"median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values), "p95_us": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))]}


def bind_producer_numa(device):
    """Bind bootstrap first-touch to this GPU's CPU node, outside timed work."""
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    bus = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id",
        "--format=csv,noheader", "--id=" + uuid], text=True).strip().lower()
    domain, bus_id, function = bus.split(":")
    pci = f"{int(domain, 16):04x}:{bus_id}:{function}"
    node = int((Path("/sys/bus/pci/devices") / pci / "numa_node").read_text())
    if node < 0:
        raise RuntimeError("GPU NUMA node is unknown; explicitly use --cpu-numa off to omit binding")
    cpus = set()
    for part in (Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(",")):
        bounds = list(map(int, part.split("-")))
        cpus.update(range(bounds[0], bounds[-1] + 1))
    cpus &= os.sched_getaffinity(0)
    if not cpus:
        raise RuntimeError("GPU NUMA node has no CPU in the process's allowed affinity")
    os.sched_setaffinity(0, cpus)
    return node


def verify_producer_pages(context, node):
    entries = {int(parts[0], 16): parts for line in Path("/proc/self/numa_maps").read_text().splitlines()
               if (parts := line.split())}
    report = {}
    for phase, arenas in context._arenas.items():
        arena = arenas[context.rank]
        fields = entries.get(arena.address, [])
        pages = {int(p[1:p.index("=")]): int(p.split("=")[1]) for p in fields
                 if p.startswith("N") and "=" in p and p[1:p.index("=")].isdigit()}
        if node is not None and (set(pages) != {node} or pages[node] * os.sysconf("SC_PAGE_SIZE") != arena.size):
            raise RuntimeError("producer arena pages do not match the requested NUMA placement")
        report[phase] = {"bytes": arena.size, "node_pages": pages}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--cpu-numa", choices=["auto", "off"], default="auto")
    parser.add_argument("--nccl-reference", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (min(args.tokens) < 1 or args.iterations < 1 or args.warmup < 1 or
            args.topk < 1 or args.experts < 1 or args.topk > args.experts or args.hidden < 2 or args.hidden % 2):
        parser.error("positive sizes, even hidden and topk <= experts are required")
    for key in ("NCCL_P2P_DISABLE", "NCCL_NVLS_ENABLE"):
        expected = "1" if key == "NCCL_P2P_DISABLE" else "0"
        if os.environ.get(key) != expected:
            parser.error(f"set {key}={expected} for the host-staged comparison")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    producer_node = bind_producer_numa(rank) if args.cpu_numa == "auto" else None
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    if dist.get_rank() != rank or int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("this benchmark requires one host and one process per local GPU")
    if args.experts % world:
        raise ValueError("experts must divide evenly across ranks")
    nccl = dist.new_group(backend="nccl") if args.nccl_reference else None
    results = []
    config = Config(args.hidden, args.topk, max(args.tokens), transport="host_sm",
                    channels=args.channels, chunk_bytes=4 << 20, combine_frame_rows=16,
                    decode_max_tokens=min(256, max(args.tokens)), decode_single_frame=True,
                    host_pool_budget_bytes=1 << 30, fatal_errors=True)
    context = Context(None, [i // (args.experts // world) for i in range(args.experts)], config, device=rank)
    placement = {"rank": rank, "requested_node": producer_node,
                 "arenas": verify_producer_pages(context, producer_node)}
    placements = [None] * world
    dist.all_gather_object(placements, placement)
    context.enable_cuda_graph()
    for tokens in args.tokens:
        stream = torch.cuda.Stream()
        context._last_stream = stream
        fused = tokens <= config.decode_max_tokens
        workspace = context.create_workspace(world * tokens, tokens, fused_combine=fused)
        decode_options = ({"expected_ranks": workspace.expected_ranks,
                           "local_rows": workspace.local_rows,
                           "return_rows": workspace.return_rows} if fused else {})
        x = torch.full((tokens, args.hidden), rank + 1, dtype=torch.bfloat16, device="cuda")
        ids = torch.empty((tokens, args.topk), dtype=torch.int32, device="cuda")
        weights = torch.full((tokens, args.topk), 1 / args.topk, device="cuda")
        partial = torch.empty_like(workspace.hidden)
        output = torch.empty_like(x)
        traces, volumes = [], []
        for step in range(8):
            generator = torch.Generator().manual_seed(7100 + rank * 37 + step)
            routes = torch.stack([torch.randperm(args.experts, generator=generator)[:args.topk]
                                  for _ in range(tokens)]).int()
            if step % 4 == 1 and args.topk <= args.experts // world:
                routes = torch.arange(args.topk).expand(tokens, -1).int() + (step % world) * (args.experts // world)
            if step % 4 == 2:
                routes[(torch.arange(tokens) + rank) % 3 == 0] = -1
            owners = routes.long() // (args.experts // world)
            remote = (owners[:, :, None] == torch.arange(world)[None, None, :]).any(1)
            remote[:, rank] = False
            volumes.append([int(remote.any(1).sum()) * args.hidden * 2,
                            int(remote.sum()) * args.hidden * 2])
            traces.append(routes.cuda())
        gathered_volumes = [None] * world
        dist.all_gather_object(gathered_volumes, volumes)

        def dispatch():
            context._prepared.dispatch(x, ids, weights, 1, workspace.hidden, workspace.local_ids,
                workspace.weights, workspace.inverse, workspace.count, workspace.owner_rows,
                workspace.owner_counts, active_channels=context.effective_channels,
                **decode_options)

        def combine():
            context._prepared.combine(partial, workspace.inverse, workspace.count,
                workspace.owner_rows, workspace.owner_counts, 1, output, workspace.remote,
                active_channels=context.effective_channels, expected_ranks=workspace.expected_ranks,
                local_rows=workspace.local_rows, return_rows=workspace.return_rows)

        def expert():
            # Synthetic weighted contribution, deliberately outside both timers.
            partial.copy_((workspace.hidden.float() *
                (workspace.weights * (workspace.local_ids >= 0)).sum(1)[:, None]).bfloat16())

        graphs = []
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ids.copy_(traces[0])
            dispatch(); expert(); combine()
        stream.synchronize()
        dist.barrier()
        if args.graph:
            for operation in (dispatch, expert, combine):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    operation()
                graphs.append(graph)
            operations = [g.replay for g in graphs]
        else:
            operations = [dispatch, expert, combine]
        phases = {"dispatch": [], "combine": []}
        samples = []
        with torch.cuda.stream(stream):
            for iteration in range(args.warmup + args.iterations):
                step = iteration % len(traces)
                ids.copy_(traces[step])
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                events[0].record(); operations[0](); events[1].record()
                operations[1]()
                events[2].record(); operations[2](); events[3].record()
                if iteration >= args.warmup:
                    samples.append((step, events))
        stream.synchronize()
        context.check_errors()
        # Input values are exact integers and weights are powers of two for topk8.
        valid = (ids >= 0).sum(1).float() / args.topk
        torch.testing.assert_close(output.float(), x.float() * valid[:, None], rtol=.01, atol=.02)
        for step, events in samples:
            phases["dispatch"].append(events[0].elapsed_time(events[1]) * 1000)
            phases["combine"].append(events[2].elapsed_time(events[3]) * 1000)
        all_phases = [None] * world
        dist.all_gather_object(all_phases, phases)
        row = {"tokens_per_rank": tokens, "channels": context.effective_channels,
               "path": "bounded_decode" if fused else "generic_ring",
               "dispatch": summary([max(p["dispatch"][i] for p in all_phases) for i in range(args.iterations)]),
               "combine": summary([max(p["combine"][i] for p in all_phases) for i in range(args.iterations)]),
               "rank_samples_us": all_phases}
        row["dispatch_payload_GBps_aggregate"] = {
            "host_write": statistics.mean(sum(v[step][0] for v in gathered_volumes) /
                max(p["dispatch"][i] for p in all_phases) / 1000 for i, (step, _) in enumerate(samples)),
            "host_read": statistics.mean(sum(v[step][1] for v in gathered_volumes) /
                max(p["dispatch"][i] for p in all_phases) / 1000 for i, (step, _) in enumerate(samples))}
        # Each routed remote rank returns one combined BF16 row to its source.
        # Logical bytes exclude packet padding, descriptors and local HBM work.
        combine_bandwidth = statistics.mean(sum(v[step][1] for v in gathered_volumes) /
            max(p["combine"][i] for p in all_phases) / 1000 for i, (step, _) in enumerate(samples))
        row["combine_payload_GBps_aggregate"] = {
            "host_write": combine_bandwidth, "host_read": combine_bandwidth}
        for graph in graphs:
            graph.reset()
        if nccl is not None:
            dense = torch.empty((world * tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
            reduced = torch.empty_like(x)
            dense.fill_(1)
            stream.wait_stream(torch.cuda.current_stream())
            references = {}
            for name, operation in (("raw_all_gather", lambda: dist.all_gather_into_tensor(dense, x, group=nccl)),
                                    ("raw_reduce_scatter", lambda: dist.reduce_scatter_tensor(reduced, dense, group=nccl))):
                with torch.cuda.stream(stream):
                    for _ in range(args.warmup): operation()
                stream.synchronize(); dist.barrier()
                graph = None
                if args.graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream): operation()
                times = []
                with torch.cuda.stream(stream):
                    for _ in range(args.iterations):
                        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        begin.record()
                        graph.replay() if graph is not None else operation()
                        end.record(); times.append((begin, end))
                stream.synchronize()
                local = [a.elapsed_time(b) * 1000 for a, b in times]
                peers = [None] * world; dist.all_gather_object(peers, local)
                references[name] = summary([max(p[i] for p in peers) for i in range(args.iterations)])
                if graph is not None: graph.reset()
            row["nccl_collective_reference"] = references
        results.append(row)
        if rank == 0:
            print(json.dumps({"tokens_per_rank": tokens, "dispatch": row["dispatch"], "combine": row["combine"]}), flush=True)
    context.close()
    if rank == 0:
        report = {"kind": "prepared_dynamic_route_communication", "graph": args.graph,
                  "warmup": args.warmup, "iterations": args.iterations,
                  "transport_options": {key: os.environ[key] for key in TRANSPORT_OPTIONS
                      if key in os.environ},
                  "producer_numa": placements, "world": world, "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
                  "timing": "same-device events; maximum rank duration per sample; preparation/reduction included; synthetic expert excluded",
                  "bytes": "logical payload host writes/reads, summed over ranks; excludes metadata, not hardware PCIe bandwidth",
                  "nccl_scope": "raw BF16 AG/RS only; excludes routes and EP output layout; full-model NCCL remains the performance baseline",
                  "results": results}
        with open(args.output, "w") as target:
            json.dump(report, target, indent=2)
        print(json.dumps({"status": "passed", "shapes": len(results)}))
    if nccl is not None: dist.destroy_process_group(nccl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
