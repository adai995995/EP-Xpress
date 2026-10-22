# Architecture and ownership

## Boundary

The inference framework owns the router, expert weights/GEMM, request scheduling,
and KV cache. EP-Xpress owns communication layouts, host storage, data movement,
completion, inverse routing, and cross-rank reduction. The core uses PyTorch
Tensor and distributed bootstrap but never imports SGLang.

`Config` defines capacity, chunk size, ring depth, memory budget, and timeout.
`Context` owns persistent arenas and one CUDA device. `RouteHandle` identifies
one dispatch and its inverse mapping. `Pending` retains completion state.
Compatible layers reuse the same context; live handles remain independent.

## Dispatch

1. Wait for the caller stream's input event on the worker stream.
2. Copy top-k metadata to CPU and exchange it on an isolated Gloo group.
3. Build stable source/token row maps and deduplicate destination ranks.
4. Copy each relevant source chunk once to its registered shared host slot.
5. After D2H completion, publish the slot with its generation and reader mask.
6. Each target CPU gathers selected rows into an independent pinned pack,
   returns its ACK, and submits H2D into local contiguous receive storage.
7. Return local expert IDs, unchanged weights, activation rows, and a handle.

The reference planner exchanges full top-k metadata and materializes Python
lists. It has significant CPU and memory overhead; the asynchronous API does
not remove those costs. CPU packing can add host DRAM traffic. A later GPU
planner or SM gather backend can replace these pieces behind the same contract.

## Ring protocol

Each producer has separate dispatch and combine rings. A slot transitions:

```text
FREE -> WRITING -> READY -> all expected readers ACK -> FREE
```

The producer is the sole payload writer. READY data is immutable. Sequence
numbers combine the context operation generation and chunk index; modulo slot
selection does not identify an operation. Header updates use process-shared
file locking. GPU host-memory atomics are not assumed.

Readiness is published only after a CUDA completion event. A receiver ACKs
after its CPU pack is independent of the source slot; its pack cannot be reused
until H2D completes. This separates shared-slot lifetime from receive-buffer
lifetime. No slot is held until expert computation ends.

The progress loop drains receive/ACK work even when send slots are full. Slow
readers create bounded backpressure rather than overwrite. Both phases have
separate arenas. All ranks, including empty ranks, enter metadata exchange.

One background worker serializes operations within a context. Separate CUDA
streams allow transport overlap inside an operation. This release does not
expose per-chunk expert-compute scheduling and does not claim arbitrary
cross-layer or multi-batch overlap.

## Combine and numerical contract

The expert runner produces BF16 `[R, H]` rank partials after applying routing
weights exactly once and summing experts resident on the same rank. Combine
does not multiply weights again.

Remote partials return along the handle's inverse map. The source GPU adds
them in ascending rank order into FP32, rounds that remote sum to BF16, adds
the local BF16 partial, and rounds the output to BF16. This explicit intermediate
rounding supports controlled comparisons with future host-reduction backends.
It can differ from another framework's reduction order; model-level numerical
validation remains necessary.

Native accumulation kernels use unique token indices for each rank's partial,
so they need no inter-thread atomics. Calls on a shared accumulator are ordered
on one stream. Context only uses the trusted fast path after route validation.

## Memory and failure handling

Shared rings are allocated in one private directory with private files. Only
each rank's export arenas are CUDA-registered in that process; receiving CPUs
read peer mappings and use their own pinned packs for H2D. No process imports
another GPU's memory.

The host budget counts the shared arena and reusable private packs for all
ranks. Output/device staging and Python route metadata are outside that budget.
Freed pinned allocator caches may outlive a context.

Each context creates a dedicated Gloo metadata group with the configured
timeout. Stream events order GPU work; metadata operations are host blocking.
Ranks must construct overlapping contexts and submit operations consistently.
Local invalid submission can cause peers to time out; a failed communicator is
not recoverable in this release.

Peer abort flags terminate ring polling. Cleanup drains CUDA streams before
unregistering host memory, closing mappings, and deleting temporary files.
Healthy shutdown is collective. CUDA-driver failures may still require process
termination; no in-process recovery is promised.

## Measuring changes

Hold GPU count/model, routing trace, dtype, expert runner, Graph mode, and inputs
constant. Record actual D2H/H2D bytes, CPU pack bytes, slot pressure, control
time, and end-to-end time. A smaller GPU communication-kernel percentage alone
does not show improvement when work moves to CE or CPU.

NCCL is a comparison baseline, not this release's payload transport. Route
planner, transport engine, CPU packing, reduction placement, and chunk/slot
changes should be measured as separate factors when possible.
