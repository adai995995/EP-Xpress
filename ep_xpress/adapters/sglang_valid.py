"""GPU-count-aware helpers for the pinned SGLang BF16 Triton runner.

Workspace shapes remain bounded by scheduler capacity. Only valid routed rows
are aligned, activated, and reduced. The tuned SGLang GEMMs and CUDA SiLU
arithmetic are retained; the GPU count is never converted to a Python scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _align_small(Ids, Count, Sorted, Experts, Padded, CAP: tl.constexpr,
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


@triton.jit
def _count_expert(Ids, Count, Counts, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    n = tl.load(Count) * TOPK
    offset = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.int32)
    for start in range(0, n, BLOCK):
        rows = start + offset
        ids = tl.load(Ids + rows, rows < n, other=-1)
        total += tl.sum((ids == expert).to(tl.int32), axis=0)
    tl.store(Counts + expert, total)


@triton.jit
def _scatter_expert(Ids, Count, Counts, Sorted, Experts, Padded,
                    CAP: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr,
                    EP: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    es = tl.arange(0, EP)
    counts = tl.load(Counts + es, es < E, other=0)
    padded = tl.cdiv(counts, BM) * BM
    start = tl.sum(tl.where(es < expert, padded, 0), axis=0)
    own_count = tl.sum(tl.where(es == expert, counts, 0), axis=0)
    own_padded = tl.cdiv(own_count, BM) * BM
    if expert == 0:
        tl.store(Padded, tl.sum(padded, axis=0))
    n = tl.load(Count) * TOPK
    offsets = tl.arange(0, BLOCK)
    cursor = tl.full((), 0, tl.int32)
    for base in range(0, n, BLOCK):
        rows = base + offsets
        ids = tl.load(Ids + rows, rows < n, other=-1)
        match = ids == expert
        positions = tl.cumsum(match.to(tl.int32), axis=0) - 1
        tl.store(Sorted + start + cursor + positions, rows, match)
        cursor += tl.sum(match.to(tl.int32), axis=0)
    tail = tl.arange(0, BM)
    tl.store(Sorted + start + own_count + tail, CAP * TOPK, tail < own_padded - own_count)
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
        counts = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
        _count_expert[(num_experts,)](topk_ids, valid_count, counts, topk, 1024)
        _scatter_expert[(num_experts,)](
            topk_ids, valid_count, counts, sorted_ids, expert_ids, padded,
            capacity, topk, num_experts, triton.next_power_of_2(num_experts), block_size, 1024,
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
