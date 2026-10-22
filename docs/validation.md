# Initial validation

Validation date: 2026-10-23. These results establish reference-implementation
correctness, not a production throughput or model-inference speedup claim.

## Environment scope

- Linux, CUDA toolkit 12.9, PyTorch 2.13.0+cu129, Python 3.12.
- NVIDIA H20D GPUs; independent runs with two and eight ranks.
- BF16 payloads, host-CE transport, eager execution, two slots per producer and
  phase. The library used no GPU peer-memory API.
- Native CUDA tests compiled and executed the bundled extension.

This does not establish compatibility or performance on consumer GPUs, other
PyTorch/CUDA combinations, or under CUDA Graph execution.

## Completed checks

| Check | Result |
|---|---|
| Full pytest suite, including native CUDA | 42 passed, 35 subtests passed |
| SGLang patch applied to a temporary copy of the pinned source files | Passed |
| Two-rank replay, H=128, up to 17 tokens per rank, 1 KiB chunks | Five cases over two iterations passed |
| Eight-rank replay, H=128, up to 17 tokens per rank, 1 KiB chunks | Five cases over two iterations passed |
| Eight-rank replay, H=2048, up to 257 tokens per rank, 64 KiB chunks | Mixed, uneven/empty, and hotspot cases passed |
| Independent live handles combined in reverse dispatch order | Passed on two and eight ranks |
| Two-rank replay with one slot and mixed/empty routes | Passed; active exports remained bounded by one |
| Mismatched tags, invalid routes, missing metadata participant | Errors propagated within the configured test bounds |
| Public-content gate, including deleted historical content tests | Passed |
| Offline wheel build without fetching dependencies | Passed |

Replay compares received activations and final outputs with an independent
synthetic-expert reference using exact BF16 equality. Cases include local-only
traffic and all-empty operations, where measured activation/result D2H and H2D
bytes are zero. Multi-chunk cases exercised both ring slots and repeated reuse.

The tests do not infer concurrent PCIe utilization from the number of active
slots. A timeline/counter study is needed to quantify actual overlap and link
utilization.

## Still to validate

- Full SGLang model startup, numerical agreement, generated outputs, and
  matched-baseline end-to-end latency.
- Actual consumer-card operation and topology sensitivity.
- Transport throughput under simultaneous expert GEMM and large realistic
  route traces.
- Alternative SM transport, CPU reduction, and CUDA Graph support; these modes
  are not implemented in this version.

No raw deployment logs, machine identifiers, model locations, or private
environment configuration are included in this report.
