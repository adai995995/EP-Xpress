# Two-node SGLang EP16 validation — 2026-11-05

The initial direct-GIN implementation completed full Qwen3-30B-A3B-Instruct-2507
inference on two hosts with eight NVIDIA H20D GPUs each. Intra-node EP-Xpress
payloads used mapped host memory; inter-node payloads used GPU-resident GDAKI/GDR
windows. The default SGLang NCCL backend was measured on the same 16 GPUs.

Both modes disabled GPU P2P, NVLS, MNNVL, CollNet, custom all-reduce and
FlashInfer all-reduce fusion. IB/RoCE, GDR, SHM and `NCCL_CUMEM_ENABLE=1` were
enabled. Communication logs selected SHM and NET/IB/GDRDMA, with no P2P or NVLS
payload channels. These settings disable transport paths, not physical links.

## Complete-model measurements

Each entry is the arithmetic mean of three completed measurements after 16
warmup batches per workload. Every request had 512 input tokens and 32 output
tokens. Prompt hashes matched across modes. Prefix/radix caching was disabled.
There was no competing GPU workload during the accepted run.

| Concurrent requests | Backend | Mean request TTFT (ms) | Steady Decode TPS | Full-batch completion (ms) |
| ---: | --- | ---: | ---: | ---: |
| 1 | Default NCCL | 139.07 | 52.81 | 726.27 |
| 1 | EP-Xpress host + GIN | 101.25 | 68.34 | 555.03 |
| 32 | Default NCCL | 809.00 | 1520.89 | 1542.60 |
| 32 | EP-Xpress host + GIN | 531.66 | 2009.30 | 1042.18 |

Single-request TTFT decreased 27.2% and Decode TPS increased 29.4%. At 32
concurrent requests, mean request TTFT decreased 34.3% and aggregate Decode TPS
increased 32.1%. This is an end-to-end result for this model and topology, not
a general hardware-independent speedup claim or a communication-only benchmark.

Observed minimum–maximum ranges across the three samples:

| Concurrent requests | Backend | Mean request TTFT range (ms) | Decode TPS range |
| ---: | --- | ---: | ---: |
| 1 | Default NCCL | 109.76–165.54 | 52.73–52.87 |
| 1 | EP-Xpress host + GIN | 99.73–103.24 | 68.06–68.49 |
| 32 | Default NCCL | 800.09–822.55 | 1518.28–1523.41 |
| 32 | EP-Xpress host + GIN | 497.96–552.29 | 2002.59–2017.85 |

TTFT is measured at the HTTP client and includes request admission, scheduling,
prefill and the first output token. The 32-request figure averages per-request
TTFT within a batch. Last-request TTFT averaged 893.95 ms for NCCL and 549.85 ms
for EP-Xpress. Steady Decode TPS is the aggregate progress between the last
request's first token and the first request's completion, interpolating
cumulative token counts. It excludes the prefill/admission period.

The same round-robin scheduler assigned two requests per DP rank in the
32-request cases, but actual prefill splits were not identical. Of the 16 ranks,
NCCL processed both requests together on 9/11/13 ranks across the three trials;
EP-Xpress did so on 15/14/10 ranks. Other ranks processed two separate prefill
batches. Consequently, the TTFT difference includes scheduler effects and must
not be attributed entirely to faster transport kernels. Decode rates had much
less variation. No frozen routing table or synthetic route replay was used in
these model measurements.

## Matched configuration

- SGLang v0.5.21 source revision
  `e00930c5489053f26d86b179cee0d087f846acbb`, with the same adapter patch.
- TP = EP = attention DP = 16, pipeline size 1; BF16 weights/activations,
  Triton experts and FlashInfer attention. Both used the default available
  Triton expert configuration for this GPU/model shape.
- Full decode CUDA Graph with local buckets 1, 2, 4; eager prefill.
- Context length 4096, maximum running requests 64, maximum total tokens 32768,
  memory fraction 0.5, unchunked prefill, round-robin scheduling, seed 42.
- PyTorch 2.13.0+cu130, CUDA toolkit 13.0, NCCL runtime/device headers 2.30.7,
  Triton 3.7.1, FlashInfer 0.6.18. Versions matched on both hosts.
- The candidate requested four GIN contexts. Source capacity was 4096 tokens;
  registered staging was 832,569,344 bytes per GPU, with 151,523,328 host bytes
  per producer. Eight producer arenas used about 1.13 GiB per host. Expert
  tensors, KV cache and NCCL internal allocations are additional.

Both modes used the same network interface selection, GDR policy, NUMA worker
binding and compiler settings. The earlier single-node NCCL archive was not
used. An earlier run with CUMEM disabled and an incomplete run interrupted by
an SSH monitoring failure were excluded. The final accepted run completed all
six timed samples in each mode and both output checks.

## Correctness and release checks

The direct GIN/GDR probe first verified payloads across two ranks and then all
16 ranks. The library's focused EP16 check verified dispatch destinations,
rank deduplication, top-k slot weights, arbitrary expert placement, empty and
skewed routes, inverse maps, and FP32 rank-ordered combine against a reference.
It also covered two outstanding dispatch handles and eight Graph replays with
changing route values. A clean export of the staged source passed this check;
it did not depend on unrelated local profiling or benchmark modifications.

Both complete-model runs returned `2 + 2 = 4` and `Paris` for the same two
untimed sanity prompts. These checks supplement the exact transport check;
they are not a model accuracy evaluation.

The implementation remains opt-in and retains batch-wide completion/reuse
barriers. Per-peer streaming credits and computation/communication overlap are
future work, not features inferred from the measured gain. Other models,
consumer GPU GDR support and other NCCL releases need separate validation.

Machine identifiers, addresses, deployment paths, access information and raw
logs remain outside the repository. The [sanitized measurement data](multinode-performance.json)
contains aggregate values, ranges, public software/model metadata, prompt hashes
and transport source hashes. See [setup and protocol details](multinode-gin.md).
