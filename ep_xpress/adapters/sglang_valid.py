"""GPU-count-aware helpers for the pinned SGLang BF16 Triton runner.

Workspace shapes remain bounded by scheduler capacity. Only valid routed rows
are aligned, activated, and reduced. The tuned SGLang GEMMs and CUDA SiLU
arithmetic are retained; the GPU count is never converted to a Python scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['CAP'])
def _align_small(Ids, Count, Sorted, Experts, Padded, CAP,
                 TOPK: tl.constexpr, E: tl.constexpr, EP: tl.constexpr,
                 BM: tl.constexpr, R: tl.constexpr, B: tl.constexpr):
    rows = tl.arange(0, R)
    experts = tl.arange(0, EP)
    n = tl.load(Count) * TOPK
    ids = tl.load(Ids + rows, rows < n, other=-1)
    match = (ids[:, None] == experts[None, :]) & (rows[:, None] < n) & (experts[None, :] < E)
    positions = tl.cumsum(match.to(tl.int32), axis=0) - 1
    counts = tl.sum(match.to(tl.int32), axis=0)
    padded = tl.cdiv(counts, BM) * BM
    starts = tl.cumsum(padded, axis=0) - padded
    tl.store(Padded, tl.sum(padded, axis=0))
    tl.store(Sorted + starts[None, :] + positions, rows[:, None], match)
    tail = tl.arange(0, BM)
    tl.store(Sorted + starts[None, :] + counts[None, :] + tail[:, None],
             CAP * TOPK, (tail[:, None] < padded[None, :] - counts[None, :]) & (experts[None, :] < E))
    blocks = tl.arange(0, B)
    tl.store(Experts + starts[None, :] // BM + blocks[:, None], experts[None, :],
             (blocks[:, None] < padded[None, :] // BM) & (experts[None, :] < E))


@triton.jit(do_not_specialize=['CAP'])
def _route_tile_counts(Ids, Count, Counts, CAP,
                       TOPK: tl.constexpr, E: tl.constexpr,
                       EP: tl.constexpr, HP: tl.constexpr,
                       BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    n = tl.minimum(tl.maximum(tl.load(Count), 0), CAP) * TOPK
    if tile * BLOCK < n:
        rows = tile * BLOCK + tl.arange(0, BLOCK)
        ids = tl.load(Ids + rows, rows < n, other=-1)
        # Check the original dtype before narrowing: an out-of-range int64 id
        # must not wrap into a valid local expert. The extra histogram bin
        # absorbs invalid ids and the final partial tile.
        valid = (rows < n) & (ids >= 0) & (ids < E)
        bins = tl.where(valid, ids, E).to(tl.int32)
        counts = tl.histogram(bins, HP)
        es = tl.arange(0, HP)
        tl.store(Counts + tile * EP + es, counts, es < E)


@triton.jit(do_not_specialize=['CAP', 'TILES'])
def _route_tile_prefix(Count, Counts, Offsets, Padded,
                       CAP, TOPK: tl.constexpr,
                       E: tl.constexpr, EP: tl.constexpr, BM: tl.constexpr,
                       BLOCK: tl.constexpr, TILES,
                       PREFIX_TILES: tl.constexpr):
    n = tl.minimum(tl.maximum(tl.load(Count), 0), CAP) * TOPK
    active_tiles = tl.cdiv(n, BLOCK)
    tiles = tl.arange(0, PREFIX_TILES)
    es = tl.arange(0, EP)
    if TILES <= PREFIX_TILES:
        counts = tl.load(
            Counts + tiles[:, None] * EP + es[None, :],
            (tiles[:, None] < active_tiles) & (es[None, :] < E), other=0,
        )
        totals = tl.sum(counts, axis=0)
        padded = tl.cdiv(totals, BM) * BM
        starts = tl.cumsum(padded, axis=0) - padded
        prefixes = tl.cumsum(counts, axis=0) - counts
        # Also write row zero for an empty batch. The last metadata row holds
        # each expert's unpadded end; neither row requires a CPU count read.
        tl.store(
            Offsets + tiles[:, None] * EP + es[None, :],
            starts[None, :] + prefixes,
            (tiles[:, None] < tl.maximum(active_tiles, 1)) & (es[None, :] < E),
        )
    else:
        # Bound the per-CTA scan tensor for larger scheduler capacities. This
        # loops over the small tile histogram, never over the routing table.
        totals = tl.zeros((EP,), tl.int32)
        for base in range(0, active_tiles, PREFIX_TILES):
            counts = tl.load(
                Counts + (base + tiles[:, None]) * EP + es[None, :],
                (base + tiles[:, None] < active_tiles) & (es[None, :] < E), other=0,
            )
            totals += tl.sum(counts, axis=0)
        padded = tl.cdiv(totals, BM) * BM
        starts = tl.cumsum(padded, axis=0) - padded
        tl.store(Offsets + es, starts, es < E)
        cursor = starts
        for base in range(0, active_tiles, PREFIX_TILES):
            counts = tl.load(
                Counts + (base + tiles[:, None]) * EP + es[None, :],
                (base + tiles[:, None] < active_tiles) & (es[None, :] < E), other=0,
            )
            prefixes = tl.cumsum(counts, axis=0) - counts
            tl.store(
                Offsets + (base + tiles[:, None]) * EP + es[None, :],
                cursor[None, :] + prefixes,
                (base + tiles[:, None] < active_tiles) & (es[None, :] < E),
            )
            cursor += tl.sum(counts, axis=0)
    tl.store(Offsets + TILES * EP + es, starts + totals, es < E)
    tl.store(Padded, tl.sum(padded, axis=0))


@triton.jit
def _route_maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=['CAP', 'TILES'])
def _scatter_route_tiles(Ids, Count, Offsets, Sorted, Experts,
                         CAP, TOPK: tl.constexpr,
                         E: tl.constexpr, EP: tl.constexpr, BM: tl.constexpr,
                         BLOCK: tl.constexpr, TILES):
    tile = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if tile < TILES:
        n = tl.minimum(tl.maximum(tl.load(Count), 0), CAP) * TOPK
        if tile * BLOCK < n:
            rows = tile * BLOCK + offsets
            ids = tl.load(Ids + rows, rows < n, other=-1)
            valid = (rows < n) & (ids >= 0) & (ids < E)
            local_ids = tl.where(valid, ids, E)
            if E * BLOCK < 2147483647:
                local_ids = local_ids.to(tl.int32)
            else:
                local_ids = local_ids.to(tl.int64)
            # Including the source position in the key makes ties stable.
            # Sorting only 256 routes avoids an [all routes, experts] match
            # tensor and lets independent input tiles scatter concurrently.
            keys = tl.sort(local_ids * BLOCK + offsets, descending=False)
            sorted_experts = keys // BLOCK
            source_rows = tile * BLOCK + keys % BLOCK
            previous = tl.gather(sorted_experts, tl.maximum(offsets - 1, 0), axis=0)
            run_start = tl.where((offsets == 0) | (sorted_experts != previous), offsets, 0)
            run_start = tl.associative_scan(run_start, 0, _route_maximum)
            valid_sorted = sorted_experts < E
            tile_start = tl.load(
                Offsets + tile * EP + sorted_experts, valid_sorted, other=0,
            )
            positions = tile_start + offsets - run_start
            tl.store(Sorted + positions, source_rows, valid_sorted)
    else:
        # Expert padding occupies disjoint positions from all valid scatters.
        expert = tile - TILES
        start = tl.load(Offsets + expert)
        end = tl.load(Offsets + TILES * EP + expert)
        count = end - start
        own_padded = tl.cdiv(count, BM) * BM
        tail = tl.arange(0, BM)
        tl.store(Sorted + end + tail, CAP * TOPK, tail < own_padded - count)
        for block in range(0, tl.cdiv(own_padded, BM), BLOCK):
            indices = block + offsets
            tl.store(Experts + start // BM + indices, expert, indices < own_padded // BM)


def align_valid_routes(topk_ids, valid_count, block_size, num_experts):
    capacity, topk = topk_ids.shape
    numel = capacity * topk
    maximum = numel + num_experts * (block_size - 1)
    sorted_ids = torch.empty((maximum,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((triton.cdiv(maximum, block_size),), dtype=torch.int32, device=topk_ids.device)
    padded = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    if numel <= 1024:
        _align_small[(1,)](
            topk_ids, valid_count, sorted_ids, expert_ids, padded,
            capacity, topk, num_experts, triton.next_power_of_2(num_experts),
            block_size, triton.next_power_of_2(max(1, numel)),
            triton.next_power_of_2(max(1, triton.cdiv(numel, block_size))),
            num_warps=4,
        )
    else:
        route_block = 256
        tiles = triton.cdiv(numel, route_block)
        expert_block = triton.next_power_of_2(num_experts)
        # One bounded scan geometry per expert layout, independent of scheduler
        # capacity. CAP/TILES remain actual bounds, including padding sentinels.
        prefix_tiles = max(1, 8192 // expert_block)
        counts = torch.empty((tiles, expert_block), dtype=torch.int32, device=topk_ids.device)
        offsets = torch.empty((tiles + 1, expert_block), dtype=torch.int32, device=topk_ids.device)
        _route_tile_counts[(tiles,)](
            topk_ids, valid_count, counts, capacity, topk, num_experts,
            expert_block, triton.next_power_of_2(num_experts + 1), route_block,
            num_warps=4,
        )
        _route_tile_prefix[(1,)](
            valid_count, counts, offsets, padded, capacity, topk,
            num_experts, expert_block, block_size, route_block, tiles, prefix_tiles,
            num_warps=8 if prefix_tiles * expert_block >= 4096 else 4,
        )
        _scatter_route_tiles[(tiles + num_experts,)](
            topk_ids, valid_count, offsets, sorted_ids, expert_ids,
            capacity, topk, num_experts, expert_block, block_size, route_block, tiles,
            num_warps=4,
        )
    return sorted_ids, expert_ids, padded


@triton.jit
def _sum_valid(Input, Output, Ids, Count, H: tl.constexpr, TOPK: tl.constexpr,
               SCALE: tl.constexpr, BLOCK: tl.constexpr, ROUND_BEFORE_SCALE: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(Count):
        h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        accum = tl.zeros((BLOCK,), tl.float32)
        # Keep the original sequential FP32 top-k sum and a single BF16 cast.
        # Negative local ids were omitted by alignment and have no GEMM output.
        for k in range(TOPK):
            local = tl.load(Ids + row * TOPK + k) >= 0
            value = tl.load(Input + (row * TOPK + k) * H + h,
                            local & (h < H), other=0).to(tl.float32)
            accum += value
        if ROUND_BEFORE_SCALE:
            accum = accum.to(Input.dtype.element_ty).to(tl.float32)
        tl.store(Output + row * H + h, accum * SCALE, h < H)


def sum_valid_routes(partials, output, topk_ids, valid_count, scaling_factor, round_before_scale=False):
    capacity, topk, hidden = partials.shape
    _sum_valid[(capacity, triton.cdiv(hidden, 512))](
        partials, output, topk_ids, valid_count, hidden, topk, scaling_factor, 512, round_before_scale,
        enable_fp_fusion=False,
    )


def fused_experts_valid_count(dispatch_output, quant_info, runner_config):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

    # This adapter is deliberately limited to Qwen3 BF16 SiLU. Its constructor
    # validates the static runner contract once; quantization is rejected by
    # SGLang's integration hook before model construction.
    if quant_info.fuse_swiglu_interleaved:
        raise ValueError("EP-Xpress valid-count runner requires noninterleaved SiLU weights")
    return fused_experts_impl(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_weights=dispatch_output.topk_output.topk_weights,
        topk_ids=dispatch_output.topk_output.topk_ids,
        b1=quant_info.b13,
        b2=quant_info.b2,
        inplace=runner_config.inplace,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        filter_expert=True,
        valid_count=dispatch_output.valid_count,
    )
