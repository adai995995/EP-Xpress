# Architecture and ownership

## Boundary

The inference framework owns the router, expert weights/GEMM, request scheduling,
and KV cache. EP-Xpress owns communication layouts, host storage, data movement,
completion, inverse routing, and cross-rank reduction. The core uses PyTorch
Tensor and distributed bootstrap but never imports SGLang.

`Config` defines capacity, chunk size, channel count, ring depth, memory budget, and timeout.
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

Each rank partitions its input among independent communication channels, with
one CUDA CTA per channel. Within a batch, threads validate GPU top-k IDs and
construct destination masks in parallel. A block prefix calculation compacts
selected rows, followed by flattened copies across all threads. Each remote
activation is exported once; selected experts on the same rank share that row.
Host chunks store mask, token-index, expert-ID, weight, and payload arrays
separately so receivers can fetch metadata in batches instead of making one
thread read a small host record and synchronize the CTA for every token.
The local-rank contribution bypasses host memory. Receivers compact selected
rows into GPU output buffers using device-memory reservations. Output row
order can vary across channels; the saved inverse map preserves token identity.
Input expert IDs may be int32 or int64 without an intermediate cast kernel.

Parallel initialization kernels clear output padding and combine scratch.
After communication, a parallel combine kernel performs deterministic
rank-order reduction. These kernels are stream ordered, so communication
blocks do not also serialize full-buffer initialization and final reduction.

The channel count is agreed during bootstrap using the minimum safe kernel
occupancy across the group. A channel is a logical work partition, not a
reserved SM. Dispatch launches `channels` CTAs; combine launches
`channels * world_size` CTAs, including one local-only CTA per channel. CUDA
determines their placement, and several CTAs may reside on one SM. For
multi-rank contexts, cooperative launch support is required even with one
channel. The common channel limit accounts for the full combine grid and both
its ordinary and diagnostic kernel occupancy. A single-device cooperative
launch ensures the bounded grid can be resident; persistent workers must not
rely on queued blocks eventually being scheduled for progress. Only a
one-rank, one-channel context can use an ordinary one-block launch. There is
no multi-device cooperative launch, GPU peer memory, TMA, or per-layer CPU
capability exchange. See [CUDA cooperative launch requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).

For dispatch, `chunk_bytes` is divided across channels, rounding up to whole
rows. Combine divides it across both channels and peers. This keeps the
payload budget approximately constant as queues are split, although row
rounding, queue headers, and page alignment add overhead. Each queue has its
own page-aligned arena and ring slots. Actual negotiated allocation is checked
against the host-memory budget. Producer work batches contain at most 256 rows
per published frame, even when a queue has room for more; the configured chunk
size is an upper bound.

The caller supplies a CPU-known receive capacity, such as the sum of scheduler
token counts. The exact received row count stays in a CUDA tensor, together
with an inverse `[source rank, source token]` map. Unused receive rows are
initialized to zero with expert IDs `-1`. The framework runs experts against
this padded shape and must mask these IDs. No device count is read by the CPU
to allocate an exact output shape. Combine consumes the saved inverse and
count directly on the GPU.

Each dispatch channel's slot has a producer-written generation and one
consumer-written ACK per peer. A generation combines an operation epoch with
a chunk number. Payload writers issue system fences before publishing;
readers finish their loads before acknowledging. Synchronization uses
naturally aligned, single-writer loads/stores with system acquire/release
semantics, not mapped-host atomic read-modify-write operations. The
implementation follows CUDA's mapped-memory access constraints:
[CUDA memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

Each frame also carries a stable fingerprint of its phase, optional API tag,
and, for combine, originating dispatch operation. A receiver checks this
operation identity on the first frame from each peer and channel, including
when that first frame is an empty operation's terminal. Generation and
sequence checks still apply to every frame. The identity is constant within
the operation, so later frames do not repeat the host fingerprint read. This
detects inconsistent layer tags and handle order without a CPU collective.
The fingerprint hashes only small API metadata; GPU routing IDs and weights
are never read back for this check.

The dispatch producer reuses a slot only after all peer ACKs match its
previous generation. Every peer acknowledges each header, including when its
rank is absent from the payload mask. An explicit terminal frame covers empty
inputs. The CTA alternates producer work with receiver progress instead of
waiting inside a blocked producer. Before returning, it consumes peer terminal
frames and drains its own slot ACKs. This prevents one operation from
overwriting a chunk that a delayed peer still needs.

## Host-SM combine queues

Combine uses separate source-owned rings indexed by `(channel, destination
rank)`. Each directed queue has one producer and one consumer. Its slot has a
producer-written generation and a single ACK word written only by that
destination. A source can reuse a slot as soon as that consumer has copied
its payload to local GPU scratch and acknowledged the generation. Other
ranks neither inspect that payload nor acknowledge its slot.

```text
GPU A sender -> host Q[A, B, channel] -> GPU B receiver
GPU B sender -> host Q[B, A, channel] -> GPU A receiver
```

One CTA on each GPU owns both directions of a `(channel, peer)` pair. Each
pair has an independent input cursor, outgoing credit, and incoming sequence.
If its outgoing ring is full, that CTA still checks and drains incoming data.
Another pair's payload or ACK cannot block its cursor. CTAs for other peers
advance independently, subject to shared GPU and PCIe resources. The local
peer CTA copies local partials directly to GPU output without host traffic.

Terminal frames remain necessary even for empty pairs. A pair CTA completes
after sending and receiving its terminals and draining its own outgoing ACKs.
The operation completes only after all CTAs finish, then the final reduction
runs in stream order. Thus queue credit is pair-local, while operation
completion still requires every peer to enter the operation. This release
does not expose early per-token completion or overlap expert GEMM with
individual returned chunks. Source-indexed scratch and deterministic
rank-order reduction are unchanged by queue arrival order.

This split applies to combine only. Dispatch still shares one host payload
among destinations and retains all-peer header ACKs. A future multicast
reclamation scheme could count only the actual readers of each payload, but
that is not implemented here. Such a count must be tied to a generation and
advance only after each reader finishes; it cannot assume that atomic
decrements to mapped host memory are coherent across GPUs. Per-reader,
single-writer completion records can provide that logical count without a
shared cross-GPU atomic counter.

`Pending.wait()` inserts an event dependency on the consuming stream without
waiting for device routing on the CPU. Input/output tensors and mapped arenas
remain alive until their transport work completes. GPU status is sticky;
invalid routes, insufficient receive capacity, or a timeout poison the context.
`check()` is an explicit synchronizing validation boundary. Normal submission
and `wait()` do not scan mapped host error headers. GPU workers poll mapped
abort headers on the first loop iteration and every 256 iterations thereafter;
local GPU status checks and per-frame ready/credit checks remain. This reduces
health-check traffic but can delay peer-abort observation by up to 255
completed loop iterations; it is not a wall-clock response guarantee.
Bounds checks, publication fences, ACK ordering, and stream
dependencies still protect payload access and buffer lifetime. CPU-readable
abort words are inspected at explicit validation or diagnostic boundaries.
Once a failure is observed, later submissions reject the failed context.
The SGLang adapter enables `fatal_errors`: a GPU transport error traps after
publishing the abort flag, so the framework's normal output synchronization
also catches a failure in its final layer. This requires worker restart. The
standalone default keeps errors recoverable for inspection, but the failed
communication context still must be destroyed.

Static CPU configuration is resolved outside repeated layer calls: the
adapter retains group information, expert placement, and its resolved
context, while the context caches timeout conversion and native occupancy
limits. Dynamic tensor shapes, scheduler-provided capacity, operation
generations, and handle ownership are still checked when each operation is
submitted. These metadata checks do not read routing tensors back to the CPU.

A follow-up audit identified remaining control work that is not yet removed:

- Dispatch launches a final kernel whose only job is to zero the valid count
  after an error. Normal execution does no useful work in that kernel. Removing
  it requires preserving the standalone nonfatal error contract or making the
  SGLang fatal-error mode explicit at the native launch boundary.
- Native entry points still check fixed parameters and look up cached launch
  capacity under a mutex. GPU attribute and occupancy queries are already
  cached. A prepared native context could validate fixed parameters once while
  keeping dynamic bounds and tensor lifetime checks on submission.
- Each layer recomputes scheduler receive capacity from existing CPU counts.
  Sharing that result across layers requires a reliable forward identifier;
  caching by a mutable list's address would be incorrect.
- Operation identity is still serialized and hashed on the CPU for each call.
  A numeric operation identity could avoid that work but must preserve matching
  across ranks and independently live dispatch handles.
- GPU route preparation still checks duplicate expert IDs and finite weights.
  A trusted-input mode would need an explicit contract before removing those
  checks; expert-index bounds also protect memory access.

The dispatch loop's barrier before reading shared error state also separates
the previous iteration's reads from the next iteration's updates. Sampling
health checks does not make that barrier redundant. Ready/ACK fences, stream
events and `record_stream` likewise carry data dependencies, rather than
serving only as diagnostics. The adapter's duplicate expert-output shape check
has been removed; the core retains the check for both transports.

Remaining limits include dispatch's all-peer header ACK traffic and draining
each operation's queues before returning. Double slots and multiple channels
provide buffering and parallel progress; they do not establish overlap with
expert GEMM or arbitrary cross-layer concurrency. Dispatch interleaves sends
and receives within each channel CTA; combine does so within each pair CTA,
rather than assigning dedicated sender/receiver blocks. Each combine pair
also scans its channel's inverse-route range to select its own rows.
Fixed-capacity padding has an expert-runner cost. These costs must be measured
separately from removed CPU route planning and collectives.

The design borrows GPU layout preparation, channel partitioning, and batched
publication from [DeepEP v1.2.1 layout](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/csrc/kernels/layout.cu)
and [intranode kernels](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/csrc/kernels/intranode.cu).
The implementation here uses shared host memory and its own flow control;
DeepEP's peer-memory addresses and hardware assumptions are not imported.

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
