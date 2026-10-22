# Validation

Validation date: 2026-10-23. These results establish reference-implementation
correctness, not a production throughput or model-inference speedup claim.

## GPU-initiated backend

The `host_sm` implementation was compiled and exercised with CUDA toolkit
13.0, PyTorch 2.13.0+cu130, Python 3.12, and the same H20D hardware. It uses
GPU loads/stores to registered shared host memory, not peer-memory transport.

| Check | Result |
|---|---|
| Full suite including both native CUDA transports | 61 passed, 43 subtests passed; one optional source-patch test skipped remotely |
| Local adapter tests with the pinned SGLang source available | All 11 passed, including patch application |
| Two ranks, H=128, up to 17 tokens per rank, 1 KiB chunks, two slots | Seven cases over two iterations passed |
| Eight ranks with the same small shape and two slots | Seven cases over three iterations passed |
| Deliberately delay rank 7 by 20 ms before each phase | Eight-rank results remained exact; no overwrite/deadlock observed |
| Eight ranks, H=2048, up to 257 tokens per rank, 64 KiB chunks | Mixed, uneven/empty and hotspot cases passed over two iterations |
| Two ranks, one slot, delayed receiver and repeated multi-chunk traffic | All seven cases passed over two iterations |
| Live dispatch handles combined in reverse order; reject a consumed handle | Passed |
| Capacity overflow, invalid active IDs/weights, mismatched tags and handles | Both ranks observed an error in each two-rank test |
| Missing peer in a native operation | Device timeout returned control without hanging unrelated CUDA work |
| Fatal error in an isolated CUDA process | Ordinary CUDA synchronization reported the failure without a status readback |
| Host-CE functional regression with two ranks | Seven cases over two iterations and reverse-order handles passed |

The seven routing cases include mixed traffic, uneven/empty ranks, local-only
traffic, hotspots, all-empty operations, partial padding, and all-padding.
The replay checks device inverse maps, exact BF16 activations and final output,
and zero-filled receive padding. Artificially delayed runs validate progress,
not communication performance. SM byte counters are not yet implemented.

## Host-CE reference validation

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

## Full-model measurements and remaining scope

Full SGLang startup and matched eager model runs with one, eight, and 32
concurrent requests are documented in [the model comparison](model-performance.md).
The sole performance baseline is SGLang's native NCCL backend. The current
GPU-initiated implementation is slower in all three shapes. Four short
generated-output checks matched token IDs but do not replace a layer-wise
numerical error study.

Remaining validation and implementation work:

- Broader model correctness and layer-output numerical checks.
- Actual consumer-card operation and topology sensitivity.
- Transport throughput under simultaneous expert GEMM and large realistic
  route traces.
- SM full-duplex tuning, CPU reduction, and CUDA Graph support. The latter
  two modes remain unimplemented.

No raw deployment logs, machine identifiers, model locations, or private
environment configuration are included in this report.
