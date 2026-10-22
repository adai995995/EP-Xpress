# Full-model inference comparison

These experiments run the complete Qwen3-30B-A3B-Instruct-2507 model through
SGLang v0.5.21. They are distinct from the synthetic expert function in the
standalone transport replay.

## Matched configuration

- Eight H20D GPUs on one host; BF16 unquantized model weights and activations.
- TP = EP = attention DP = 8, with DP LM head, Triton experts, and FlashInfer
  attention. Prefill and decode CUDA Graph are both disabled.
- Python 3.12, PyTorch 2.13.0+cu130, CUDA toolkit 13.0.
- SGLang baseline uses `--moe-a2a-backend none`; EP-Xpress uses `ep_xpress`.
  The baseline includes SGLang's native gather/reduction path and remaining
  framework collectives; it is not a standalone NCCL all-to-all benchmark.
- GPU peer transport, NVLS, MNNVL, CollNet, and IB are disabled. Custom
  all-reduce and FlashInfer all-reduce fusion are disabled in both runs.
  NCCL logs identify shared-memory transport, and hardware NVLink counters
  show zero payload increments over each measured run.
- Synthetic token-ID requests: 512 input tokens and 32 generated tokens,
  greedy decoding, ignored EOS, disabled radix cache. Request hashes match
  across configurations. Concurrent batch sizes are 1, 8, and 32.
- Each shape has eight warmup batches generating four tokens, then three
  measured batches generating 32 tokens. Values below are medians.

These controls establish a comparison under eager execution. They do not
represent the fastest available SGLang configuration or a CUDA Graph baseline.
Identical inputs do not by themselves guarantee identical later expert routes.

## Performance baseline

The sole performance baseline is SGLang's default NCCL backend path,
`--moe-a2a-backend none`. The candidate is EP-Xpress with
`--moe-a2a-backend ep_xpress` and `EP_XPRESS_TRANSPORT=host_sm`.
The library's `host_ce` implementation is retained for functional regression
testing; it is not a performance comparator.

## GPU-initiated results

Measured on 2026-10-23. The configuration, input-hash, hardware, source-base,
and transport checks passed. The saved NCCL run was reused; the candidate run
used EP-Xpress commit `9ad5e28`. There is no speedup over the NCCL baseline.

| Concurrent requests | NCCL batch latency (s) | EP-Xpress batch latency (s) | EP-Xpress / NCCL | NCCL output tokens/s | EP-Xpress output tokens/s |
|---|---:|---:|---:|---:|---:|
| 1 | 3.065 | 4.861 | 1.586 | 10.439 | 6.583 |
| 8 | 2.969 | 10.759 | 3.624 | 86.226 | 23.794 |
| 32 | 3.421 | 29.570 | 8.643 | 299.317 | 34.629 |

Observed batch-latency ranges over the three repetitions were 3.054–3.083,
2.956–2.970, and 3.400–3.454 seconds for NCCL; and 4.813–4.865,
10.706–10.808, and 29.354–29.670 seconds for EP-Xpress. These are observed
ranges, not confidence intervals.

| Concurrent requests | NCCL mean TTFT (ms) | EP-Xpress mean TTFT (ms) | NCCL last TTFT (ms) | EP-Xpress last TTFT (ms) |
|---|---:|---:|---:|---:|
| 1 | 190.314 | 1858.477 | 190.314 | 1858.477 |
| 8 | 209.710 | 5744.011 | 210.145 | 7585.583 |
| 32 | 602.041 | 20324.784 | 629.398 | 26480.826 |

| Concurrent requests | NCCL per-request TPOT (ms) | EP-Xpress per-request TPOT (ms) | NCCL steady-window TPOT (ms) | EP-Xpress steady-window TPOT (ms) |
|---|---:|---:|---:|---:|
| 1 | 92.741 | 96.358 | 92.741 | 96.358 |
| 8 | 88.978 | 161.060 | 88.978 | 101.555 |
| 32 | 91.001 | 301.412 | 90.628 | 101.342 |

Batch latency is HTTP-client time until every request finishes. Mean TTFT is
the mean across requests in each repetition; last TTFT is when the final
request receives its first token. Per-request TPOT averages each request's
time from its first to last token divided by the subsequent token count.
Steady-window TPOT uses only the interval from the final first token to the
first completed request, normalized by concurrent requests and tokens emitted
in that window. It excludes the staggered-start period and should not replace
per-request TPOT when assessing user latency. Each table reports the median
of its respective per-repetition statistic.

Most of the batch-latency gap occurs before the final request's first token;
TTFT includes scheduling and prefill. These measurements alone do not isolate
communication time, PCIe bandwidth utilization, or a communication fraction.

The current transport launches one CUDA thread block per rank and phase,
processes rows serially within that block, and interleaves bounded sends and
receives. This establishes bounded progress and GPU-side routing, but does
not establish link saturation. Parallel routing, multiple communication
channels, and separate prefill/decode tuning are optimization candidates;
their contribution to the observed gap still requires profiling.

Generated outputs were checked separately against the NCCL baseline. All four
short greedy prompts matched in text and token IDs (29 generated tokens).
Chosen-token log probabilities over the common prefixes differed by up to
0.219 (mean absolute difference 0.0147). This small check does not prove
numerical equivalence or identical routes for the performance requests;
reduction order and BF16 rounding can differ.

Deployment addresses, credentials, model paths, and raw logs are deliberately
excluded. Experiment records remain in the operator's private artifact store.
