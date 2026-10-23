# Experimental SGLang integration

This adapter targets SGLang v0.5.21 at source revision
`e00930c5489053f26d86b179cee0d087f846acbb`. It registers the independent
`ep_xpress` A2A backend and reuses SGLang's Standard/Triton expert runner.
The EP-Xpress core has no SGLang dependency.

## Supported contract

- One Linux host, NVIDIA CUDA, BF16 activations and unquantized BF16 experts.
- `Qwen3MoeForCausalLM`, evenly partitioned contiguous experts.
- Attention DP = TP = EP; MoE TP = 1, MoE DP = 1, pipeline parallelism = 1.
- Eager prefill and decode. Both CUDA Graph phases are disabled by the patch.
- No shared experts, EPLB, redundant experts, LoRA, speculative decoding,
  Waterfill, TBO, SBO, Torch Compile, elastic EP, memory saver, or DWDP.
- One outstanding operation per layer. All EP ranks must follow the same layer
  sequence, including ranks with zero input or zero received tokens.

These restrictions belong to this first adapter, not the core EP interface.
Patch application and mocked contract tests do not establish end-to-end model
correctness or performance. Compare outputs and throughput against the same
eager SGLang configuration. For tuning an already validated path, prioritize
that full-model comparison; use focused standalone checks when an interface
or protocol change introduces a concrete correctness or progress risk.

## Install

In the environment used by the SGLang worker processes, install EP-Xpress and
the pinned SGLang checkout. From the EP-Xpress repository:

```sh
python -m pip install --no-build-isolation -e .
python adapters/sglang/apply_patch.py "$SGLANG_SOURCE" --check
python adapters/sglang/apply_patch.py "$SGLANG_SOURCE"
python -m pip install --no-build-isolation -e "$SGLANG_SOURCE/python"
```

`SGLANG_SOURCE` is a local source directory supplied by the operator. The
installer checks the exact Git revision, refuses to overwrite changes in its
target files, and recognizes an already applied patch. It does not download
dependencies, models, or an alternative SGLang revision.

All worker processes need a common shared-memory filesystem and enough free
space for the configured rings. In containers, provision IPC/shared-memory
capacity explicitly; the usual 64 MiB container default is too small for the
eight-rank example below once ring metadata is included.

Example launch after setting `MODEL_PATH` to a supported BF16 checkpoint:

```sh
export EP_XPRESS_MAX_TOKENS=16384
export EP_XPRESS_CHUNK_BYTES=2097152
export EP_XPRESS_SLOTS=2
export EP_XPRESS_CHANNELS=4
export EP_XPRESS_HOST_BUDGET_BYTES=536870912
export EP_XPRESS_TRANSPORT=host_sm

NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --dtype bfloat16 \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention \
  --moe-a2a-backend ep_xpress --moe-runner-backend triton \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce
```

`EP_XPRESS_TRANSPORT=host_sm` selects GPU kernels that read and write mapped
pinned host staging. `host_ce` selects the original CPU-planned CE transport
and remains the default when this variable is unset. Neither EP payload path
calls NCCL. The NCCL settings constrain other framework collectives and do not
physically disable NVLink. Keep those settings, GPU count, model, precision,
expert runner and eager mode identical in the `--moe-a2a-backend none` baseline.
`EP_XPRESS_CHANNELS` requests parallel host-SM channels. The group negotiates
the common cooperative-launch capacity once at initialization. Multi-rank
operation requires cooperative launch support; the single-device path alone
supports a noncooperative single-block fallback. The payload budget is split
across allocated channels, with row rounding and lane metadata included in the
actual host-pool check. Active channels can vary per operation without resizing
these rings. By default, `EP_XPRESS_ADAPTIVE_CHANNELS=1` selects:

- Prefill: `min(effective_channels, max(1, global_capacity))`.
- Decode: `min(effective_channels, max(1, ceil(global_capacity / 8)))`.

Both inputs are already shared scheduler metadata: `get_global_dp_buffer_len()`
and `get_is_extend_in_batch()`. The latter reports whether any rank is extending;
local token counts and local forward modes never choose the channel count. No
new metadata collective or GPU-to-CPU count copy is introduced. Combine reuses
the count stored in its matching dispatch handle.

Set `EP_XPRESS_ADAPTIVE_CHANNELS=0` to use all effective channels in both phases.
`EP_XPRESS_PREFILL_CHANNELS` and `EP_XPRESS_DECODE_CHANNELS` default to `0`, which
uses the policy above. A positive phase override takes precedence over the
adaptive switch, must not exceed `EP_XPRESS_CHANNELS`, and is clamped to the
negotiated cooperative capacity. Set the same options on every rank.

Model measurements and channel-count/prefill/decode tradeoffs are
reported in [the performance comparison](../../docs/model-performance.md).
Set the maximum tokens to cover the largest per-rank forward; the core fails
explicitly if capacity is exceeded. These capacities are starting points,
not measured optimal values. `EP_XPRESS_HOST_BUDGET_BYTES` limits the transport's
pinned working set: mapped rings for `host_sm`, plus reusable private receive
packs for `host_ce`. Device staging and framework allocations are additional.

## Data ownership and lifecycle

The adapter sends global top-k IDs to the core, then passes the returned local
IDs directly to Standard/Triton. It never calls `StandardDispatcher`, which
would remap the IDs a second time. Expert outputs are already weighted and
locally summed; `combine_async(..., weights_applied=True)` only reduces across
ranks. No top-k weight is applied a second time.

For `host_sm`, the adapter sizes each receive tensor from SGLang's existing
`get_global_dp_buffer_len()`: the scheduler computes the sum of padded rank
counts once per forward in the pinned revision. Rank deduplication bounds
each destination by this sum. It does not copy routes or receive counts to the
CPU. The actual valid count and inverse map remain on the GPU. A missing or
invalid scheduler bound fails explicitly.

`EP_XPRESS_VALID_COUNT=1` is the default for `host_sm`. The adapter carries the
GPU count into its BF16 Triton runner. Alignment only includes valid local
routes, omitting capacity-only rows and the `-1` expert bucket. The original
SGLang GEMMs retain their tuning and BF16 intermediates. The original CUDA SiLU
arithmetic reads the count and local int64 IDs directly, avoiding the old ID
conversion and skipping unused rows. Local reduction skips nonlocal slots and
uses FP32 top-k accumulation; weights remain applied once in the down GEMM.
The count-aware path supports gated SiLU without activation modifiers or fused
interleaved SiLU weights, and ordinary route-major GEMMs without TMA.

This keeps workspace allocations and launch upper bounds based on scheduler
capacity. GPU checks skip inactive work; the CPU never reads the count to resize
tensors or select an expert-kernel launch size. It does not claim dynamically
smaller allocations or that every launched CTA performs useful work. Set
`EP_XPRESS_VALID_COUNT=0` to use the previous Standard/Triton expert path for an
ablation; unused IDs remain `-1`, weights and activations are zero, and combine
still reads only the valid prefix. `host_ce` continues to use the original runner.
The adapter enables fatal CUDA errors for this backend: a detected route or
capacity failure aborts the worker's CUDA context and surfaces at SGLang's
normal output synchronization. This also rejects a failed final MoE layer without
adding a per-layer CPU fence. Restart the worker group after such a failure.
The default `EP_XPRESS_DIAGNOSTICS=0` disables tag/fingerprint validation,
duplicate-expert and finite-weight checks, and peer-abort/timeout polling.
Set it to `1` for those protocol diagnostics. With diagnostics disabled, a
missing peer requires external job cancellation. The fatal inference path
omits dispatch's otherwise error-only count-reset kernel. Ready/ACK, epochs,
memory bounds, stream dependencies, and combine reduction remain active.

The patch classifies this backend as completing its own combine, so Qwen3's
existing local-router forward skips the outer post-expert reduction. That
SGLang predicate is historically called `is_deepep_class_backend`; EP-Xpress
uses its own factory, core library and Standard runner, not DeepEP kernels.

The context and host pool are shared by compatible layers on a process.
Per-operation immutable tickets retain the handle until the matching combine.
The first adapter rejects a second dispatch before combine; it does not silently
overwrite a handle. Zero-capacity ranks skip GEMM but still execute combine.
`host_sm` ranks with nonzero capacity and zero valid rows still enter the expert
runner and matching combine. The default count-aware kernels exit without
processing those rows; the adapter never reads the device count to skip a phase.

For controlled worker shutdown, call
`ep_xpress.adapters.sglang.close_contexts()` on every EP rank after forwards
finish and before destroying process groups. Shutdown is collective. Do not
invoke it from just one rank or an `atexit` callback. After an operation failure,
restart the worker group rather than continuing with unmatched operations.

## Validation checklist

1. Use focused checks for a changed protocol or interface. For the count-aware
   runner, `python tests/check_sglang_valid_count.py` compares route coverage and
   BF16 output bits with the original runner for zero, partial, and full counts.
2. Start the patched SGLang with a supported checkpoint and eager mode.
3. Check generated output against the same Standard/Triton baseline with fixed
   requests, seeds and precision. BF16 reduction order can change rounding;
   include layer-output error checks rather than text equality alone.
4. Exercise uneven per-rank input, zero-token ranks, repeated decode steps and
   a prefill close to configured capacity.
5. Prioritize full-model latency against the default NCCL backend with matching
   GPU configuration and transport restrictions. Profile communication only when
   needed to explain a measured gap; standalone kernel timings are not model
   performance. See the performance report for measured results and limitations.
