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

## Transport selection

`host_ce` is the CPU reference path described below. `host_sm` uses CUDA
kernels to access mapped host memory; its protocol is described separately.
Selection is explicit at context creation and is never a silent fallback.
Neither mode enables GPU peer access or CUDA Graph execution.

## Host-CE dispatch

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
not remove those costs. CPU packing can add host DRAM traffic. The SM backend
replaces these pieces while retaining token-routing semantics.

## Host-CE ring protocol

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

## Host-SM dispatch and progress

Initialization creates separate shared host arenas for every producer and
phase. Every process registers all mappings with `cudaHostRegisterMapped` and
obtains its own device addresses. These addresses refer to CPU memory, not
peer GPU allocations. Expert placement is uploaded once. Gloo is used for
bootstrap and collective healthy teardown, not for dispatch/combine metadata.

Each operation launches one CUDA CTA per rank. It validates GPU top-k IDs,
constructs a destination rank mask, and exports each remote activation row
once into a shared chunk. Selected experts on the same rank share that row.
The chunk contains source token indices, routing metadata, and BF16 payloads.
The local-rank contribution bypasses host memory. Receivers inspect headers,
read only selected payload rows, and compact them into local GPU buffers.

The caller supplies a CPU-known receive capacity, such as the sum of scheduler
token counts. The exact received row count stays in a CUDA tensor, together
with an inverse `[source rank, source token]` map. Unused receive rows are
initialized to zero with expert IDs `-1`. The framework runs experts against
this padded shape and must mask these IDs. No device count is read by the CPU
to allocate an exact output shape. Combine consumes the saved inverse and
count directly on the GPU.

Each slot has a producer-written generation and one consumer-written ACK per
peer. A generation combines an operation epoch with a chunk number. Payload
writers issue system fences before publishing; readers finish their loads
before acknowledging. Synchronization uses naturally aligned, single-writer
loads/stores with system acquire/release semantics, not mapped-host atomic
read-modify-write operations. The implementation follows CUDA's mapped-memory
access constraints: [CUDA memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

Each frame also carries a stable fingerprint of its phase, optional API tag,
and, for combine, originating dispatch operation. Peers check that identity on
the GPU, including terminal frames. This detects inconsistent layer tags and
handle order without a CPU collective. The fingerprint hashes only small API
metadata; GPU routing IDs and weights are never read back for this check.

The producer reuses a slot only after all peer ACKs match its previous
generation. Every peer acknowledges each header, including when its rank is
absent from the payload mask. An explicit terminal frame covers empty inputs.
The CTA alternates producer work with receiver progress instead of waiting
inside a blocked producer. Before returning, it consumes peer terminal frames
and drains its own slot ACKs. This prevents one operation from overwriting a
chunk that a delayed peer still needs.

`Pending.wait()` inserts an event dependency on the consuming stream without
waiting for device routing on the CPU. Input/output tensors and mapped arenas
remain alive until their transport work completes. GPU status is sticky;
invalid routes, insufficient receive capacity, or a timeout poison the context.
`check()` is an explicit synchronizing validation boundary. CPU-readable abort
words allow later submissions to reject a failure without reading GPU tensors.
The SGLang adapter enables `fatal_errors`: a GPU transport error traps after
publishing the abort flag, so the framework's normal output synchronization
also catches a failure in its final layer. This requires worker restart. The
standalone default keeps errors recoverable for inspection, but the failed
communication context still must be destroyed.

This first implementation has deliberate limits: one CTA limits bandwidth,
all-peer header ACKs generate control traffic, and each operation drains its
queues before returning. Double slots provide bounded buffering; they do not
establish overlap with expert GEMM or arbitrary cross-layer concurrency.
Send and receive instruction streams within one rank are interleaved, not yet
assigned independent warps or CTAs for full-duplex throughput tuning.
Fixed-capacity padding also has an expert-runner cost. These costs must be
measured separately from removed CPU route planning and object collectives.

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

The SM path receives remote partials into rank-indexed BF16 GPU scratch planes,
then sums ranks in ascending order. Arrival order does not change the result.
This scratch storage is additional device memory, separate from the bounded
host rings. Routing weights are not applied again during combine.

## Memory and failure handling

Shared rings are allocated in one private directory with private files. In
`host_ce`, only
each rank's export arenas are CUDA-registered in that process; receiving CPUs
read peer mappings and use their own pinned packs for H2D. No process imports
another GPU's memory.

The host budget counts shared arenas and, for `host_ce`, reusable private packs
for all ranks. Device staging and Python route metadata are outside that budget.
Freed pinned allocator caches may outlive a context.

Each context creates a dedicated Gloo control group with the configured
timeout. Stream events order GPU work; metadata operations are host blocking.
Per-operation Gloo metadata exchange applies only to `host_ce`.
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
