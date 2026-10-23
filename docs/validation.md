# Validation

Validation date: 2026-10-23. These results establish reference-implementation
correctness, not a production throughput or model-inference speedup claim.

## GPU-initiated backend

The `host_sm` implementation was compiled and exercised with CUDA toolkit
13.0, PyTorch 2.13.0+cu130, Python 3.12, and the same H20D hardware. It uses
GPU loads/stores to registered shared host memory, not peer-memory transport.

| Check | Result |
|---|---|
| Full suite including both native CUDA transports and optional combine diagnostics | 104 passed, 43 subtests passed; one optional source-patch test skipped remotely |
| Local adapter tests with the pinned SGLang source available | All 11 passed, including patch application |
| Native four-channel partition tails, 1/3/4/5/17 tokens and one/two slots | Passed, including padding, int32 IDs, valid counts, and a shorter subsequent operation |
| Native three-channel grid and rejection above the safe cooperative capacity | Passed; invalid launch leaves CUDA usable |
| Late error in the final channel | Sticky error and zero valid count observed |
| Two ranks, H=128, one token per rank, four channels | Seven cases and independent reverse-order handles passed over two iterations |
| Eight ranks, H=128, up to 17 tokens per rank, 1 KiB chunks, four channels, two slots | Seven cases over three iterations passed |
| Deliberately delay rank 7 by 20 ms before each phase | Eight-rank results remained exact; no overwrite/deadlock observed |
| Eight ranks, H=2048, up to 257 tokens per rank, 64 KiB chunks, four channels | Mixed, uneven/empty and hotspot cases passed over two iterations with int32 IDs |
| Two ranks, one slot, four channels, delayed receiver and repeated multi-chunk traffic | All seven cases passed over two iterations |
| Two ranks, 1,025 tokens, 2 MiB chunks and three channels | Mixed, uneven/empty and hotspot cases plus reverse-order handles passed |
| Eight ranks, 16 channels, H=2048, up to 257 tokens per rank | Mixed, uneven/empty and hotspot cases plus reverse-order handles passed over two iterations |
| Eight ranks, 16 channels, one token per rank, one slot, rank 7 delayed 20 ms | All seven cases and reverse-order handles passed over two iterations |
| Live dispatch handles combined in reverse order; reject a consumed handle | Passed |
| Capacity overflow, invalid active IDs/weights, mismatched tags and handles | Both ranks observed an error in each two-rank test |
| Missing peer in a native operation | Device timeout returned control without hanging unrelated CUDA work |
| Fatal error in an isolated CUDA process | Ordinary CUDA synchronization reported the failure without a status readback |
| Host-CE functional regression with two ranks | Seven cases over two iterations and reverse-order handles passed |

The seven routing cases include mixed traffic, uneven/empty ranks, local-only
traffic, hotspots, all-empty operations, partial padding, and all-padding.
The replay checks device inverse maps, exact BF16 activations and final output,
and zero-filled receive padding. Artificially delayed runs validate progress,
not communication performance. Optional SM combine counters measure requested
payload read/write bytes, excluding metadata/control and local bypass. They
are software counters, not measured PCIe link traffic.

Optional native diagnostics passed local-only and empty C1/C4 output equality,
repeated epochs, exact phase closure, idle-wait subset bounds, missing-peer
timeouts, invalid inverse maps, and prelaunch buffer validation. CPU API tests
cover bounded capture, dropped samples, immutable labels, repeated sessions,
and a single bulk-readback fence at stop. Diagnostic phases include additional
CTA synchronization; instrumented timings require an overhead control run.

Eight-rank diagnostic replay also passed at 16 channels: H=2048 with 257-token
multi-chunk mixed/empty/hotspot cases, and H=128 with 17 tokens, one slot, all
seven routing cases, and a 20 ms delayed peer. Both used two iterations and
reverse-order handle checks. Requested read/write bytes matched independent
route reconstruction and balanced globally per operation. Full-model capture
validated 64 rank reports and 233,472 CTA records; the measured overhead and
phase results are documented in [the model comparison](model-performance.md).

The distributed checks require the stated effective channel count, rather than
silently accepting a single-channel fallback. All eight devices reported a
safe cooperative limit of 32 channels. Bootstrap tests also exercise mixed
device limits, unsupported cooperative launch, actual allocation budgets, and
per-channel abort flags. Multi-handle tests use different lengths, routes,
weights, and payloads before combining in reverse order.

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
GPU-initiated implementation remains slower in all three shapes, at both four
and 16 channels. Both settings passed four short generated-output checks with
matching token IDs, but these do not replace a layer-wise numerical error study.

Remaining validation and implementation work:

- Broader model correctness and layer-output numerical checks.
- Actual consumer-card operation and topology sensitivity.
- Transport throughput under simultaneous expert GEMM and large realistic
  route traces.
- SM full-duplex tuning, CPU reduction, and CUDA Graph support. The latter
  two modes remain unimplemented.

No raw deployment logs, machine identifiers, model locations, or private
environment configuration are included in this report.
