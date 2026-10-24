"""Dependency-free validation for the experimental SGLang adapter."""


def validate_server_args(args, hf_config):
    """Fail closed outside the first integration's explicitly supported scope."""
    if getattr(args, "nnodes", 1) != 1:
        raise ValueError("EP-Xpress requires a single host")
    if getattr(args, "pp_size", 1) != 1:
        raise ValueError("EP-Xpress does not support pipeline parallelism")
    tp = getattr(args, "tp_size", 1)
    if not getattr(args, "enable_dp_attention", False) or args.dp_size != tp:
        raise ValueError("EP-Xpress requires attention DP = TP = EP")
    if getattr(args, "ep_size", tp) != tp or getattr(args, "moe_dp_size", 1) != 1:
        raise ValueError("EP-Xpress requires attention DP = TP = EP and MoE DP = 1")
    if getattr(args, "dwdp_size", 1) != 1:
        raise ValueError("EP-Xpress does not support dense-weight data parallelism")
    if getattr(args, "moe_runner_backend", None) != "triton":
        raise ValueError("EP-Xpress requires --moe-runner-backend triton")
    if getattr(args, "dtype", None) not in ("bfloat16", "bf16"):
        raise ValueError("EP-Xpress requires an explicit --dtype bfloat16")
    if getattr(args, "quantization", None) or getattr(hf_config, "quantization_config", None):
        raise ValueError("EP-Xpress currently supports unquantized BF16 experts only")
    for name in (
        "enable_eplb", "enable_waterfill", "enable_lora",
        "enable_two_batch_overlap", "enable_single_batch_overlap",
        "enable_fused_moe_sum_all_reduce", "enable_torch_compile",
        "speculative_algorithm", "ep_num_redundant_experts", "ep_join_mode",
        "elastic_ep_backend", "enable_memory_saver", "enable_pdmux",
    ):
        if getattr(args, name, None):
            raise ValueError(f"EP-Xpress does not support {name}")
    if getattr(hf_config, "architectures", None) != ["Qwen3MoeForCausalLM"]:
        raise ValueError("The initial EP-Xpress adapter supports Qwen3MoeForCausalLM only")
    if getattr(hf_config, "n_shared_experts", 0) or getattr(hf_config, "num_shared_experts", 0):
        raise ValueError("EP-Xpress does not support shared experts")


def validate_runner_config(config, *, world_size, bf16):
    if config.params_dtype != bf16:
        raise ValueError("EP-Xpress requires BF16 activations and expert weights")
    if config.num_fused_shared_experts:
        raise ValueError("EP-Xpress does not support fused shared experts")
    if config.no_combine or config.apply_router_weight_on_input:
        raise ValueError("EP-Xpress requires weighted, locally combined expert output")
    if config.num_experts != world_size * config.num_local_experts:
        raise ValueError("EP-Xpress requires an even, contiguous expert partition")
    if config.use_tp_all_gather_activation:
        raise ValueError("EP-Xpress cannot be combined with TP activation all-gather")


def contiguous_expert_map(num_experts, world_size):
    if world_size < 1 or num_experts < 1 or num_experts % world_size:
        raise ValueError("num_experts must be positive and divisible by world_size")
    local = num_experts // world_size
    return [expert // local for expert in range(num_experts)]


def receive_capacity_from_scheduler(counts, *, world_size, max_tokens_per_rank):
    """Bound rank-deduplicated rows with existing CPU forward metadata.

    SGLang publishes padded per-DP-rank counts before the forward. Each source
    token can appear at most once on a receiving rank, regardless of top-k.
    No routing tensor or device count is copied to the host here.
    """
    if not isinstance(counts, (list, tuple)) or len(counts) != world_size:
        raise ValueError("host_sm requires scheduler token counts for every EP rank")
    if any(type(count) is not int or not 0 <= count <= max_tokens_per_rank
           for count in counts):
        raise ValueError("scheduler token counts exceed the configured per-rank capacity")
    return sum(counts)


def validate_valid_count_runner(config):
    """Validate the static scope of the GPU-count-aware BF16 runner once."""
    if getattr(config, "activation", "silu") != "silu" or not getattr(config, "is_gated", True):
        raise ValueError("EP-Xpress valid-count runner requires gated SiLU")
    if any(getattr(config, name, None) is not None for name in (
        "gemm1_alpha", "gemm1_clamp_limit", "swiglu_limit",
    )):
        raise ValueError("EP-Xpress valid-count runner does not support activation modifiers")


def select_active_channels(global_capacity, *, is_extend, allocated, adaptive,
                           prefill_override=0, decode_override=0):
    """Select identically on all ranks from existing global scheduler metadata.

    A phase override requests a fixed count (clamped to cooperative capacity).
    Auto uses one channel per token in prefill, one per eight global tokens in
    decode. Local token counts and local forward mode must not enter this choice.
    """
    override = prefill_override if is_extend else decode_override
    if override:
        return min(allocated, override)
    if not adaptive:
        return allocated
    tokens_per_channel = 1 if is_extend else 8
    return max(1, min(allocated, (global_capacity + tokens_per_channel - 1) // tokens_per_channel))
