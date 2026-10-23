# Validation

## Independent Tx/Rx warp groups for pair combine

The prepared pair-combine option gives each 256-thread CTA four sending and
four receiving warps, with separate shared metadata and group barriers. The
frame ABI, CTA grid, final reduction and decode mailbox implementation are
unchanged. The CUDA extension compiles successfully; all eight validation
devices report a safe cooperative capacity of 32 channels at world size eight.

Two focused three-GPU parameter cases passed for 4-row and 256-row frames.
Each exercises six operation epochs with one/two active channels and two
slots, covering asymmetric bidirectional traffic, an empty rank, local-only
and all-empty operations, repeated multi-frame reuse and a hidden-size tail.
Outputs and remote scratch match the legacy raw combine bit for bit. All
published slots are acknowledged on completion, including terminal frames;
prepared-path frame counts match the owner-list partition. Diagnostics are
enabled in these bounded protocol checks; full-model timing uses the normal
diagnostics-disabled setting.

Python syntax, three focused local Config/adapter cases, and whitespace checks
pass. The broader native suite was not repeated. Legacy phase counters are
explicitly unavailable for the new concurrent kernel; disabling the option
profiles the old sequential kernel, not the new execution.

## Decode workspace and fused mailbox combine

Runtime `4356b6f` adds explicitly borrowed decode workspaces, a bounded host
mailbox with fused receive/reduction, and independently configurable combine
frame rows for the rank-pair path. The CUDA extension compiled successfully.

Five focused checks passed in the offline CUDA environment:

- One three-GPU protocol check covers sparse contributors, an unrelated rank
  delayed until its peers finish, empty inputs, exact mailbox capacity,
  changing active widths, repeated reuse and reverse-order dispatch handles.
  Fused outputs match the original rank-pair combine bit for bit. Reusing a
  combine submission epoch is rejected before launch.
- Four CPU interface cases (including a two-mode parameterization) cover
  borrowed storage reuse, consumer-stream dependencies, expired tickets,
  incompatible workspaces and operation-owned fused outputs.

Three focused local Config/adapter cases, Python syntax, and whitespace checks
also pass. The GPU check enables protocol diagnostics; full-model timing uses
the normal diagnostics-disabled setting. The broader historical suites were
not rerun. These checks establish the changed protocol and storage contract;
two full-model candidates completed batches 1 and 32 with three repetitions
each. Matched inputs, settings, hardware, source and host-only transport checks
pass. All four short output checks match 29 token IDs against default NCCL; see
the [decode experiment](model-performance.md#decode-workspace-and-fused-receive-experiment)
for measured performance and numerical/attribution limits.

## GPU owner lists, selective dispatch, and count-aware experts

Runtime `7b05797` builds return-owner lists during dispatch, lets each handle
retain its active channel width, and reclaims dispatch payload slots after only
actual readers acknowledge them. The SGLang adapter can pass its GPU receive
count into the BF16 Triton expert path without a CPU scalar readback.

Focused validation covered the new protocol and numerical contract:

- A three-GPU check delayed an unrelated rank while two other ranks recycled
  two slots across nine frames. Repeated active widths 1/2/1/2, owner-list
  coverage, mixed and empty routes, weighted combine, and empty-operation
  diagnostic identity checks passed.
- Pending-handle lifetime and per-operation active-grid profile checks passed.
  Seven prepared-operation and active-width interface cases passed after their
  small-hidden-size fixtures specified an appropriately small chunk budget.
- Nine GPU expert comparisons covered zero, partial, and full valid counts,
  capacities 1/32/129/256, int64 expert IDs, and GPU int32 counts. Route coverage
  and BF16 outputs matched the original runner with zero numeric tolerance.
- The CUDA extension compiled successfully. Focused local configuration and
  adapter checks, Python syntax, and patch application checks passed.

Full-model inference completed at batches 1 and 32 with three repetitions per
shape, followed by one environment-only valid-count ablation. Both use the same
patched SGLang source as the default NCCL baseline. Source, input, configuration,
hardware and host-only transport checks pass; all four short sanity prompts
match text and all 29 token IDs. See the
[owner-list experiment](model-performance.md#owner-list-and-sparse-work-experiment)
for timings, prefill split variability and numerical limits.

These checks target changed code; the broader historical suite below was not
rerun. Payload credit independence does not remove the final all-peer terminal
handshake, memory publication fences, or CUDA stream dependencies. Expert
workspaces and launch upper bounds remain capacity-sized even when kernels
skip inactive rows.

## Prepared transport revision

Runtime `282437b` moves static native checks to initialization and defaults
protocol diagnostics to off. The CUDA extension compiled in full-model
SGLang startup. Focused local checks passed for the diagnostics configuration,
reuse of SGLang's global receive-capacity bound, device-resident receive counts,
and adapter handle/error handling. Python syntax and whitespace checks passed.
The broader historical native and fault-injection suites below were not rerun
for this change; their timeout/identity results do not describe the new default
with protocol diagnostics disabled.

Full-model channel-sweep results, transport evidence and output-sanity limits
are recorded in the
[prepared transport comparison](model-performance.md#prepared-transport-and-channel-count-experiment).
All four configurations completed batches 1 and 32 with three timed repetitions
per shape; each matched all 29 tokens in four short output-sanity prompts.

## Rank-pair combine revision

The pair-queue CUDA extension compiled successfully. Thirty-one focused
bootstrap, profile-layout, local-combine and bounded-wait checks passed. Two
additional three-GPU protocol checks passed using RAM-backed shared mappings:

- Independent pair progress: rank 2 launches no consumer and sends no ACK.
  Ranks 0 and 1 still exchange multiple frames, reuse a one-slot queue, copy
  the expected payload, and drain their mutual terminal frames before the
  absent pair times out.
- Successful mixed-destination combine across repeated epochs, with uneven
  counts, empty input/channel partitions, and exact output checks.

An initial version of these two test fixtures used a container overlay file
that CUDA could not register. Switching the fixture to the same `/dev/shm`
mapping type used by the runtime resolved that setup failure. The checks
below describe prior revisions and were not all repeated for this change.

Validation date: 2026-10-23. These results establish reference-implementation
correctness, not a production throughput or model-inference speedup claim.

### Subsequent critical-path cleanup

Runtimes `6ecedd7` and `d3410eb` compiled and completed full-model SGLang
inference on eight GPUs at batches 1 and 32, following the performance-first validation
policy. The broad native suite above was not rerun for these changes. Matched
configuration, inputs, hardware, source and host-only transport checks passed;
all six timed batches in each candidate run completed without reported runtime
errors. Four short greedy output checks matched all 29 generated token IDs
against default NCCL.
The end-to-end results and their numerical limits are recorded in the
[latest model comparison](model-performance.md#rank-pair-combine-and-critical-path-cleanup).

CPU CI identified a missing capture rejection before first lazy Context
initialization. Revision `daec161` restores that bootstrap-only guard; its
targeted adapter check and public CPU CI passed. It does not alter the timed
steady-state path. The final `d3410eb` model rerun includes that guard and removal
of the adapter's duplicate shape check; the core shape check remains.

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
The sole performance baseline is SGLang's native NCCL backend. The earlier
four- and 16-channel implementations were slower at all three shapes. The
latest owner-list/count-aware run observes 5.77% lower latency at batch 1 and
approximately equal latency at batch 32 (+0.23%), with substantial batch-32
variation. Disabling the count-aware expert path slows both shapes. Batch 8 was
not remeasured. Short generated-output checks match
token IDs, but do not replace a layer-wise numerical error study.

Remaining validation and implementation work:

- Broader model correctness and layer-output numerical checks.
- Actual consumer-card operation and topology sensitivity.
- Transport throughput under simultaneous expert GEMM and large realistic
  route traces.
- SM full-duplex tuning, CPU reduction, and CUDA Graph support. The latter
  two modes remain unimplemented.

No raw deployment logs, machine identifiers, model locations, or private
environment configuration are included in this report.
