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

## CPU reference transport

The `host_ce` implementation is slower than this NCCL baseline. Its full-model
cost includes CPU route readback/planning, Gloo object metadata exchange, CPU
packing, and transport progress; it is a functional reference, not a measured
performance improvement.

| Concurrent requests | NCCL batch latency (ms) | host_ce batch latency (ms) | NCCL decode TPOT (ms) | host_ce decode TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 3065.4 | 11087.8 | 92.74 | 317.64 |
| 8 | 2969.0 | 17292.8 | 88.98 | 378.22 |
| 32 | 3421.1 | 33074.1 | 90.63 | 414.57 |

Batch latency is HTTP-client time until every request finishes. Decode TPOT
uses the benchmark's steady-state token interval; it is not communication
kernel time. TTFT also includes scheduling and prefill. These measurements
alone do not establish PCIe bandwidth utilization or a communication fraction.

Four short natural-language sanity prompts produced identical greedy text and
token IDs between `host_ce` and NCCL. The maximum selected-token log-probability
difference over 29 common-prefix tokens was 0.219. This limited check does not
prove numerical equivalence; reduction order and BF16 rounding can differ.

Deployment addresses, credentials, model paths, and raw logs are deliberately
excluded. Experiment records remain in the operator's private artifact store.
