# Architecture and ownership

## Boundary

The inference framework owns the router, expert weights/GEMM, request scheduling,
and KV cache. EP-Xpress owns communication layouts, host storage, data movement,
completion, inverse routing, and cross-rank reduction. The core uses PyTorch
Tensor and distributed bootstrap but never imports SGLang.

`Config` defines capacity, chunk size, channel count, ring depth, memory budget, and timeout.
`Context` owns persistent arenas and one CUDA device. `RouteHandle` identifies
one dispatch and its inverse mapping. `Pending` retains completion state.
Compatible layers reuse the same context; ordinary live handles remain
independent. Explicit workspaces provide a separate borrowed-storage contract
for sequential inference.

## Transport selection

`host_ce` is the CPU reference path described below. `host_sm` uses CUDA
kernels to access mapped host memory; its protocol is described separately.
Selection is explicit at context creation and is never a silent fallback.
Neither mode enables GPU peer access. CUDA Graph execution requires explicit
opt-in on the host-SM direct-stream path described below.

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
At each receive reservation, the prepared path also appends output-row indices
to a device-resident list for that row's source rank. Local bypass rows enter
the current rank's list. This creates the return-owner index once during
dispatch, without a later sort, route readback, or metadata collective. The
handle retains `owner_rows: int32[world_size, receive_capacity]` and
`owner_counts: int32[world_size]`, adding
`4 * world_size * (receive_capacity + 1)` bytes of device storage per live
handle. Only valid list entries are written; initialization clears the counts.
Input expert IDs may be int32 or int64 without an intermediate cast kernel.

Parallel initialization kernels clear output padding and pair-queue combine scratch.
For pair queues, a parallel combine kernel after communication performs
deterministic rank-order reduction. These kernels are stream ordered, so queue
communication blocks do not also serialize full-buffer initialization and final
reduction. The bounded decode mailbox path below fuses receive and reduction.

The allocated channel count is agreed during bootstrap using the minimum safe
kernel occupancy across the group. A channel is a logical work partition, not
a reserved SM. Each dispatch may choose a CPU-known `active_channels` in
`[1, effective_channels]`; all ranks must agree on that choice. Omitting it uses
the full allocation. The handle carries the active width into combine.
Dispatch launches `active_channels` CTAs; pair-queue combine launches
`active_channels * world_size` CTAs, including one local-only CTA per channel. CUDA
determines their placement, and several CTAs may reside on one SM. For
multi-rank contexts, cooperative launch support is required even with one
channel. The common channel limit accounts for the full combine grid and both
its ordinary and diagnostic kernel occupancy. A single-device cooperative
launch ensures the bounded grid can be resident; persistent workers must not
rely on queued blocks eventually being scheduled for progress. Only a
one-rank, one-channel context can use an ordinary one-block launch. There is
no multi-device cooperative launch, GPU peer memory, TMA, or per-layer CPU
capability exchange. See [CUDA cooperative launch requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).

For dispatch, `chunk_bytes` is divided across allocated channels, rounding up
to whole rows. With the default `combine_frame_rows=0`, pair-queue combine
divides it across both allocated channels and peers. A positive
`combine_frame_rows` in `[1, 256]` instead fixes the row capacity of each pair's
frame independently of channel count. This lets prefill adjust parallelism
without shrinking each frame. The original split keeps the payload budget
approximately constant as queues are split; fixed rows instead make combine
storage grow with allocated channels. Row rounding, queue headers and page
alignment also contribute to the allocation.
Changing an operation's active width changes its work partition and launch
only: arena strides, row capacity per frame, and reserved memory stay fixed. Each queue has its
own page-aligned arena and ring slots. Actual negotiated allocation is checked
against the host-memory budget. Producer work batches contain at most 256 rows
per published frame, even when a queue has room for more; the configured chunk
size is an upper bound.

The caller supplies a CPU-known receive capacity, such as the sum of scheduler
token counts. The exact received row count stays in a CUDA tensor, together
with an inverse `[source rank, source token]` map and the prepared path's owner
lists. Unused receive rows are initialized to zero with expert IDs `-1`.
An expert runner can consume the device count to skip padded work, or mask the
unused rows when processing the full capacity. No device count is read by the
CPU to allocate an exact output shape. Prepared combine consumes each owner's
row list and count directly on the GPU; the inverse map supplies the original
token index for each listed row.

Each dispatch channel's payload slot has a producer-written generation and
one consumer-written ACK per peer. Prepared dispatch additionally uses
producer-written READY notifications indexed by `(slot, reader)`. A generation
combines an operation epoch with a frame number. Payload writers issue system
fences before publishing; readers finish their loads before acknowledging.
Synchronization uses naturally aligned, single-writer loads/stores with system acquire/release
semantics, not mapped-host atomic read-modify-write operations. The
implementation follows CUDA's mapped-memory access constraints:
[CUDA memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

With diagnostics enabled, payload frames also carry a stable fingerprint of
their phase, optional API tag and, for combine, originating dispatch operation.
Prepared dispatch checks the first received payload identity and the separate
terminal identity; even an empty operation therefore validates its tag.
Combine and legacy raw dispatch check the first frame, including a terminal
when the pair or operation has no payload. The identity stays constant within
the operation. This detects inconsistent tags and handle order without a CPU
collective; GPU routing IDs and weights are never read back for this check.

For each published dispatch frame, the producer takes the union of its remote
destination masks and sends a READY notification only to those readers. The
payload is still written once. Each reader polls its notifications across the
bounded slots, accepts a current-epoch sequence different from its own ACK,
reads the selected rows, and ACKs that sequence. A nonreader never inspects the
frame's mutable metadata or payload. Readers do not require a contiguous
sequence of every producer frame, so a skipped destination cannot lose its
place when the producer reuses a slot.

The producer retains each slot's last reader mask in CTA shared memory and
reuses the slot after exactly those readers have acknowledged its generation.
Per-reader single-writer records provide the logical remaining-reader count;
no cross-GPU atomic decrement on host memory is required. Notification words
follow the payload slots in each dispatch channel arena. The reserved suffix
uses `8 * (63 * slots + 64)` bytes before page rounding, covering per-reader
notifications plus terminal READY/ACK words at the maximum supported rank
count. Terminal identity uses the existing arena header.

After all its payload slots have been released, a producer publishes a
separate terminal READY for the operation. Every peer consumes and ACKs this
terminal, including ranks that read no payload. The CTA continues receiving
while its outgoing payload or terminal is waiting. It returns only after
consuming every peer's terminal and draining its own terminal ACKs. This final
handshake protects epoch reuse and empty-input completion; it does not put
unrelated peers back into payload-slot credits.

The direct raw native dispatch API retains the earlier all-peer header ACK
protocol and in-ring terminal frames. The Context and SGLang adapter use the
prepared protocol described above. Both paths retain epoch checks, bounded
queues, publication fences, and stream-lifetime ordering.

## Host-SM combine queues

The general combine path uses separate source-owned rings indexed by `(channel, destination
rank)`. Each directed queue has one producer and one consumer. Its slot has a
producer-written generation and a single ACK word written only by that
destination. A source can reuse a slot as soon as that consumer has copied
its payload to local GPU scratch and acknowledged the generation. Other
ranks neither inspect that payload nor acknowledge its slot.

```text
GPU A sender -> host Q[A, B, channel] -> GPU B receiver
GPU B sender -> host Q[B, A, channel] -> GPU A receiver
```

One CTA on each GPU owns both directions of a `(channel, peer)` pair. In the
prepared path, that peer's owner list is partitioned among active channels.
The CTA reads the next bounded set of row indices directly, then obtains their
original token indices from inverse. It does not scan other owners' rows or
rescan an already inspected suffix when a host frame is smaller than a tile.
The direct raw native combine API retains its inverse-scan implementation for
callers that do not supply owner lists. Each pair has an independent input
cursor, outgoing credit, and incoming sequence.
If its outgoing ring is full, that CTA still checks and drains incoming data.
Another pair's payload or ACK cannot block its cursor. CTAs for other peers
advance independently, subject to shared GPU and PCIe resources. The local
peer CTA copies local partials directly to GPU output without host traffic.

Core `Config.combine_warp_specialization` defaults to `False`; this retains
the sequential send/receive loop. Setting it to `True` selects a prepared
owner-list kernel with four Tx warps and four Rx warps in the same 256-thread
remote-peer CTA. The SGLang adapter enables it by default through
`EP_XPRESS_COMBINE_WARP_SPECIALIZATION=1`. The CTA grid, active channel count,
frame capacity and fused decode mailbox kernel are unchanged.

Tx and Rx have separate shared metadata arrays and separate named barriers
covering their respective 128 threads. Tx waits only for its destination
credit, publishes payload and terminal frames, then drains its outgoing ACKs.
Rx independently polls incoming READY, copies the payload into remote scratch
and acknowledges the consumed generation. Every payload writer completes a
system fence before the Tx group barrier and leader READY publication. Every
reader finishes before the Rx group barrier and leader ACK publication. A
full-CTA barrier joins the groups only after both directions complete or
observe an error. Local-only CTAs retain their device copy. Frame metadata
capacity remains 256 rows, processed in strides by each 128-thread group.

The phase profiler currently instruments only the sequential pair kernel.
Warp specialization rejects profile captures explicitly; selecting the old
path for profiling does not measure the new kernel. This execution split
permits concurrent communication progress but does not itself demonstrate
PCIe full-duplex utilization or overlap with expert computation.

Terminal frames remain necessary even for empty pairs. A pair CTA completes
after sending and receiving its terminals and draining its own outgoing ACKs.
The operation completes only after all CTAs finish, then the final reduction
runs in stream order. Thus queue credit is pair-local, while operation
completion still requires every peer to enter the operation. This release
does not expose early per-token completion or overlap expert GEMM with
individual returned chunks. Source-indexed scratch and deterministic
rank-order reduction are unchanged by queue arrival order.

Combine keeps one payload queue per directed rank pair, while prepared
dispatch shares a payload frame among its selected readers. Their queue
layouts differ, but both reclaim payload slots using only actual consumers.
Neither path currently exposes per-token completion to expert execution.

## Bounded decode mailboxes

`Config.decode_max_tokens` defaults to `0`, leaving this path disabled. A
positive value in `[1, 256]` allocates a separate host mailbox arena per
producer, with a fixed slot for every `(destination rank, source token)`.
All processes map these CPU pages just as they map the general queues. One
producer's allocation is page-rounded from
`128 + world_size * decode_max_tokens * align128(32 + hidden_size * 2)` bytes.
It is included in the group host budget and checked against the native ABI at
bootstrap. Device route/output storage remains outside that host budget.

The caller selects `dispatch_async(..., fused_combine=True)` consistently on
all ranks. Dispatch writes two additional device arrays for each original
source token: its target-rank mask and its compact local row, or `-1` if there
is no local contribution. This adds no CPU route readback or metadata
collective. The matching handle retains these arrays through combine.

Fused combine launches one resident CTA per active channel, rather than one
per channel and peer. Each CTA first publishes its partition of outgoing rows
from the GPU owner lists into the destination's fixed token mailboxes. Payload
writers fence before publishing READY for the current combine epoch. There
is no wait for previous credits at this point: the preceding fused operation
drained all of its published ACKs before returning on the same transport stream.

The CTA then handles a partition of local source tokens. It waits only for
contributors present in each token's saved mask, loads their host payloads,
and accumulates them in ascending remote-rank order in FP32. It rounds the
remote sum to BF16, adds the local BF16 partial if present, and rounds the
output to BF16. This is the existing pair-queue arithmetic order, including
the intermediate rounding. A dropped token produces zero. The receiver
publishes each contributor's ACK only after all payload reads complete.

Finally, each CTA drains only the ACKs for the contributions it published.
Mailbox combine has no terminal frames or ACKs from unrelated ranks. Its
output is produced in this one kernel, without clearing or writing
rank-indexed remote GPU scratch and without a separate final-reduction launch.
Dispatch still uses its required all-peer terminal handshake. Thus this change
does not remove the operation ordering contract or provide early token-level
completion to the expert runner.

The native prepared operator requires increasing fused combine epochs and
a serialized execution sequence. A CUDA event orders mailbox stream changes;
consecutive submissions on one stream require no transition event. An epoch belongs to the combine
submission, not to its dispatch handle; independently allocated handles may
still be combined in a consistent reverse order across ranks. GPU bounds,
READY/ACK ordering and optional identity diagnostics remain active. Existing
prefill phase profiling does not instrument fused mailbox operations. They
continue on their normal path and are excluded from the diagnostic capture;
profiling never silently changes fusion or transport selection.

The experimental `EP_XPRESS_DECODE_PEER_BATCH` native build setting changes
only the mailbox receive payload schedule. Zero retains scalar reads. Values
1, 2 and 4 load a bounded group of remote-rank positions as 128-bit vectors
before consuming those values. One thread owns eight consecutive BF16 hidden
elements. Mailbox payloads have a 16-byte-aligned base; hidden widths not
divisible by eight, or shifted GPU input/output views without 16-byte
alignment, use the original scalar loop. Local partial reads and final output
stores retain the same 128-bit ownership. Host vector reads keep the same
cache-volatile policy on mailbox reuse.

Accumulation still visits remote ranks in ascending order, including zero
additions for absent remote contributors. Intermediate BF16 rounding and the
local BF16 addition remain unchanged. There is no new host work, allocation,
scratch plane, synchronization, or change to send/READY/ACK lifetimes. The
setting is read once at extension load and participates in the build cache
key; it does not specialize kernels on per-layer token counts. Channel
selection is independent. This option remains experimental until full-model
measurements establish a repeatable benefit.

## Borrowed communication workspaces

`Context.create_workspace(recv_capacity, local_tokens, fused_combine=...)` is a
host-SM API for allocating exact-shape dispatch tensors, inverse/owner lists,
device counts and combine output once. The legacy mode also allocates remote
scratch; fused mode instead allocates the contributor masks and local-row
array. `dispatch_async(..., workspace=workspace, fused_combine=...)` uses those
tensors without repeated tensor allocations. This storage is for communication;
expert GEMM intermediates remain the runner's responsibility.

Workspace batches, outputs and tickets are borrowed until its next dispatch.
For the asynchronous API, the previous combine must have been waited and all
output consumers enqueued on the wait stream before reuse. With the direct
API, enqueue those consumers on the combine stream before the next dispatch. Retaining output
across reuse requires a clone or a different workspace. A new dispatch
invalidates the old workspace-backed tickets; an unconsumed route cannot be
overwritten. Generic calls that omit a workspace continue to allocate
independent storage and support outstanding handles.

Reuse orders the transport stream after prior consumer streams using GPU
dependencies only. The usual same-producer case uses the dependency already
inserted at submission, without adding another wait event. Cross-stream
consumers receive the additional required dependencies. There is no host event
query, CPU count readback or stream synchronization in workspace acquisition.
Tensor stream lifetimes also protect storage when a caller evicts a workspace.

The SGLang adapter enables reuse only for globally selected small decode
operations. Each layer owns a FIFO cache of at most four exact
`(global capacity, local token count, fusion mode)` shapes. Layers share the
transport context but not these output buffers, avoiding cross-layer residual
aliases. Cache misses allocate once; eviction removes the oldest inserted
shape, and cache hits do not change that order. Dynamic GPU valid counts still
control meaningful rows within the scheduler capacity.

## Submission and diagnostics

`Pending.wait()` inserts an event dependency on the consuming stream without
waiting for device routing on the CPU. Input/output tensors and mapped arenas
remain alive until their transport work completes. GPU status is sticky;
invalid route indices or insufficient receive capacity poison the context.
`check()` is an explicit synchronizing validation boundary. Normal submission
and `wait()` do not scan mapped host error headers.

`Config.diagnostics` defaults to `False` for `host_sm`. The prepared path omits
CPU tag serialization/hashing, GPU operation fingerprint publication/checking,
peer-abort and timeout polling, duplicate expert detection, and finite-weight
checks. Unique top-k IDs and finite weights remain input requirements. Enabling
this option restores those diagnostics; abort/timeout polling then runs on the
first loop iteration and every 256 iterations. Diagnostics do not change queue
ready/ACK, epochs, payload bounds or publication/lifetime synchronization.

With diagnostics disabled, a missing or failed peer can leave a kernel waiting;
external job cancellation is required. A synchronizing `check()` or `close()`
cannot impose a host timeout on such a kernel. The SGLang adapter enables
`fatal_errors`: a detected local GPU transport error traps, so ordinary model
output synchronization surfaces it. This requires worker restart. The fatal
inference path omits dispatch's otherwise error-only count-reset kernel;
nonfatal contexts retain it for the standalone failure-inspection contract.

Bootstrap constructs `PreparedHostSM`, validates static placement, pointer
layout, rank/world, queue sizes and cooperative launch capacity, and retains
its CUDA tensors. It caches addresses, slot/channel strides, timeout conversion
and launch configuration. Per-operation native submission does not repeat the
capacity mutex/map lookup. It retains only checks for changing tensor shapes,
dtype/device/layout, epoch and bounds, with one device-guard scope. Placement
and pointer tables are read back only once during bootstrap; routes and valid
counts remain on the GPU throughout submission.

The adapter reads SGLang's `get_global_dp_buffer_len()`, already computed once
per forward after MAX_LEN/SUM_LEN padding, instead of validating/summing rank
counts in every layer. It caches the resolved Context and omits diagnostic tag
strings unless diagnostics are enabled. Each source still checks its local
input token limit; the receiving Context checks the aggregate capacity bound.
The core validates expert-output shape without duplicating it in the adapter.

The dispatch loop's barrier before reading shared error state also separates
the previous iteration's reads from the next iteration's updates. Disabling
health checks does not make that barrier redundant. Ready/ACK fences, stream
events and `record_stream` likewise carry required data dependencies. The
producer and native work-stream capture guards check different streams and
remain in place for ordinary eager contexts. Opt-in Graph contexts instead
use GPU generations and the capture/replay lifecycle described below.

Remaining limits include draining published payload ACKs before returning and
dispatch's all-peer terminal handshake. Double slots and multiple active channels
provide buffering and parallel progress; they do not establish overlap with
expert GEMM or arbitrary cross-layer concurrency. Dispatch interleaves sends
and receives within each channel CTA. Prepared pair-queue combine can assign
independent Tx/Rx warp groups within each remote-peer CTA; the core default retains
the sequential loop. Neither configuration uses dedicated sender/receiver
blocks or pipelines expert computation. The owner-list path
removes combine's repeated inverse scans, but still performs per-frame
metadata, fence, and ACK work. Mailbox combine replaces those frames with
per-token notifications and still waits for each actual contributor. Reused
communication buffers do not eliminate expert workspace allocation or every
Python submission object. Fixed-capacity layouts and expert runners
that ignore the device valid count can also add work. These costs must be
measured separately from removed CPU route planning and collectives.

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

The SM pair-queue path receives remote partials into rank-indexed BF16 GPU
scratch planes, then sums ranks in ascending order. The mailbox path loads
those partials directly from host memory in the same arithmetic order. Arrival
order does not change either result. Pair-queue scratch is additional device
memory, separate from bounded host arenas. Routing weights are not applied
again during combine.

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


## Runtime route capacities and prefill diagnostics

Large-route alignment passes exact `CAP` and `TILES` as non-specializing Triton
runtime scalars. The prefix scan uses one bounded static geometry per expert
layout, with a runtime loop for larger tile counts. Allocation bounds, masks,
and padding sentinels still use the actual capacity. Small-route geometry uses
power-of-two buckets; dtype, expert layout and GEMM block size can still require
initial compilation. This removes exact-capacity specialization, not all JIT.

Opt-in capture schema 4 records a `kind` and `timing_layout` per operation.
Dispatch retains sequential CTA phase accounting. Prefill warp combine records
Tx and Rx group timings independently using their existing named-barrier groups,
without a cross-group phase barrier. Credit check/wait and final credit drain
belong to Tx; READY check/wait belongs to Rx. Their durations overlap, so sums
across groups or CTAs are not request latency. Payload intervals include GPU
instructions and diagnostic barriers; publish and ACK include system fences.
Byte counters count requested payload accesses, not physical PCIe transactions.
Fused decode and idle operations are excluded. Initialization, final GPU
reduction and diagnostic readback are outside the transport phase intervals.

The normal template specialization omits group timers and profiling barriers.
Profile storage is allocated at an explicit request boundary and read back only
after capture. Measurements with these diagnostics must be paired with an
uninstrumented run; end-to-end performance uses diagnostics disabled.


## Prefill payload read scheduling

Prepared warp-specialized combine can read aligned host payloads as 16-byte
vectors. Each Rx thread issues a bounded group of independent loads into
registers before storing them to the existing per-peer GPU scratch plane.
The payload uses the same cache-volatile system-memory policy as the scalar
path; READY acquire, reader completion and ACK ordering remain unchanged.
Widths and source/destination addresses that do not satisfy vector alignment
use the scalar path, including contiguous tensor views with shifted storage.
Masked final iterations do not access the next ring slot. This optimization
changes neither the wire layout nor the BF16 reduction order.

`EP_XPRESS_PREFILL_READ_UNROLL` is read once when the native extension loads.
Values 1, 2, 4 and 8 select vector load expansion; the default 0 selects scalar
reads. Vector reads remain opt-in until model-level gains are repeatable.
The build setting is included in the extension cache identity. It cannot be
changed in an already-loaded process. It affects prepared warp combine only,
not dispatch or fused decode. Queue depth remains `Config.slots`, independent
of combine frame rows and channel count. Deeper queues increase the total host
pool and must pass the existing initialization budget check. They can hide
transient consumer lag, but cannot increase the sustained link bandwidth.


The corresponding send path is independently selectable with
`EP_XPRESS_PREFILL_WRITE_UNROLL` (default 0; vector expansions 1, 2, 4, 8).
It gathers aligned vectors from the owner-indexed GPU rows, then issues volatile
128-bit host stores. Every writer still executes its system fence and joins
the Tx barrier before READY publication. Unaligned inputs or payloads use the
scalar path. This setting also participates in the extension cache identity;
it changes no token routing, frame format or ACK/credit semantics.

### Selected dispatch payload reads

`EP_XPRESS_DISPATCH_READ_UNROLL` independently controls dispatch's host reads.
The default is zero (scalar); values 1, 2, 4 and 8 select 128-bit vector reads
with that many independent iterations expanded before HBM stores. The setting
is compiled into the extension and included in its cache key. Dispatch uses
this setting in both prefill and decode; the fused decode combine is unchanged.

The receiver first performs the existing rank-mask selection and compaction.
Vector addresses then use the selected wire-row list, so a receiver reads only
its own rows even when a shared frame contains other destinations. Output rows
retain their compact indices, inverse mappings and owner lists. Hidden sizes,
frame payload addresses or output views that lack 16-byte alignment use the
original scalar loop. Tail iterations never read beyond selected payload rows.
READY/acquire, cache-invalidating host reads, reader completion and ACK release
retain their original ordering. Queue capacity, multicast reader sets and
terminal handshakes are unchanged.

### Optional last-data marker in warp combine

`EP_XPRESS_COMBINE_LAST_DATA=1` lets the prepared warp-combine sender attach
end-of-queue to its last payload frame. The default zero retains the separate
empty terminal for ablation. This is a build-time setting included in the
extension cache key; set it before starting all workers. Legacy sequential
pair combine, selective dispatch and fused decode mailbox protocols are
unchanged.

The existing signed row-count word encodes LAST as `-(payload_rows + 1)`.
Thus `-1` still identifies an empty terminal, and a full or partial last data
frame has a value below `-1`. Positive values identify nonfinal payload frames;
zero is invalid on a published frame. The warp receiver accepts both formats,
decodes and checks the payload length, and retains LAST in thread-local state
while consuming the frame. LAST never bypasses its payload reads.

Every payload writer still fences before READY publication. Every receiver
still finishes its reads and the existing system fence/group barrier before
ACK. Tx drains all published slots, including the last data frame, before
returning. Unused slots from older epochs remain protected by their matching
READY/ACK values. Empty and local-only operations still notify remote queues.
No pool-size, slot-count, channel-count or floating-point ordering change is
required. Workers must use the same supported native version; older warp
receivers reject negative payload-bearing row counts.

Diagnostic `terminal_sent` and `terminal_received` counters in warp combine
count end markers, including markers attached to data. They must not be
interpreted as counts of empty frames. Payload byte/row counts include the
last data frame normally; frame counters count it once.


### Direct-stream eager submission

`HostSMContext.dispatch_on_stream` and `combine_on_stream` enqueue on the
specified stream (current stream by default) and return tensors directly.
They do not synchronize the CPU or produce completion tickets. Consume the
results on that stream. Cross-stream consumers need explicit dependency and
tensor-lifetime management; the asynchronous API remains the convenient choice
when a completion must be waited on by another stream.

A context orders transitions between direct streams and its communication
stream, since all submissions share queue epochs. Consecutive direct operations
on one stream need no event handshake. Workspace reuse still waits for recorded
previous consumers; first use on a new stream records its tensor lifetimes and
orders allocation-stream work. `check_errors`, profile readout and destruction
also drain direct work before inspecting or releasing mapped storage. The
READY/ACK and credit protocols are unchanged.

The SGLang adapter exposes this as `EP_XPRESS_DECODE_SAME_STREAM=1` for decode
within the configured small-token capacity. Its default remains off; the measured `host_sm_bf16_ep8_same_stream.sh` preset
enables it for the documented workload. This is an eager optimization,
separate from the opt-in GPU-epoch Graph mode below.


### Serialized CUDA Graph execution

`enable_cuda_graph()` opts a fresh host-SM context into GPU-owned dispatch and
combine generations. Both eager and captured work use direct-stream APIs in
this mode. Dispatch and pair combine increment their counter in the existing
initialization kernel; fused mailbox combine adds a one-thread generation
kernel because it has no initialization pass. All transport CTAs then read the
same generation. Counters are retained for the context lifetime, never reset
at capture, and trap at the existing 31-bit protocol limit. Payload arithmetic,
deduplication, multicast readers, release/acquire and ACK drains are unchanged.

Warm each fixed workspace shape, then call `prepare_graph_capture()` outside
capture. This startup boundary drains prior eager operations. Recording neither
advances device generations nor inserts waits on unrelated, noncapturing
streams. Each graph recomputes its device route maps, counts and masks from the
current input tensors. Call `prepare_graph_replay(stream)` before updating shared static inputs and
each replay to order it after prior eager/graph work and to track the actual
execution stream for later eager submissions and shutdown. This is a GPU stream
dependency on a stream change, not a host fence or per-layer collective.

The caller must retain all static input storage, workspaces and the context
through graph destruction. A graph output is borrowed until its workspace is
reused; enqueue consumers before the next operation and clone retained outputs.
Concurrent graph submissions sharing one context are unsupported. The SGLang
adapter retains decode workspaces for all selected buckets, integrates the two
lifecycle hooks in its full-graph backend, and keeps eager prefill on the same
ordered context. CUDA Graphs preserve changing top-k routing; they do not cache
a previous token's expert assignment.


## Bounded single-frame decode dispatch

`Config(decode_single_frame=True, decode_max_tokens=...)` opts a host-SM
context into a bounded dispatch protocol. The option is included in the existing
bootstrap configuration exchange, so ranks with different protocol settings fail
before arena allocation. No new metadata collective or host read of routing is
added to dispatch. The SGLang adapter exposes it as
`EP_XPRESS_DECODE_SINGLE_FRAME=1` before context initialization.

Selection requires fused decode and
`ceil(decode_max_tokens / active_channels) <= min(dispatch_frame_rows, 256)`.
Both operands use group-wide configured limits, not the rank's live row count,
batch-size labels, observed routing distribution or model identity. Empty ranks
select the same protocol as populated ranks. Larger configured bounds and all
non-fused operations retain the selective ring. The active width and fused mode
remain collective submission contracts, as with the existing mailbox API.

Each producer/channel publishes one frame, including a negative row count for
an empty frame. A nonempty frame is also final: consumers ACK it and finish that
source without a separate terminal notification. Payload remains rank-deduplicated
and shared among selected readers; all peers inspect frame metadata, and only
selected peers read token payload. The one metadata/payload ACK per consumer
protects reuse. This is not removal of all-peer completion or of required memory
ordering. System fences remain on publish and ACK.

Each operation drains its outgoing frame before returning, so the next operation
can reuse slot zero without an initial credit poll. The drain compares ACKs
against the known generation instead of rereading the producer READY word.
General-ring/fixed-frame transitions retain that same lifetime guarantee. This
revision does not defer ACKs across calls, change buffer ownership, or introduce
concurrent graph replays. Invalid routes and capacity errors remain checked.

Fixed capacity reserves storage; it does not imply fixed expert routing or that
all reserved rows are valid. Every graph replay still reads the current top-k IDs
and generates fresh inverse maps, owner lists and GPU receive counts.


In SGLang automatic-channel mode, fused decode raises the active width to at
least `ceil(decode_max_tokens / min(dispatch_frame_rows, 256))` when that width
fits the allocated cooperative grid. This is a resource-bound decision shared
by every rank, not a special case for an observed benchmark batch. Explicit
channel overrides remain authoritative; insufficient grid capacity retains the
ring fallback. The same width is saved in the handle for combine.


## Bounded decode publication batching

The optional `EP_XPRESS_DECODE_PUBLISH_BATCH` build flag accepts 1, 2, 4, 8 or 16.
Default 1 retains the original per-row publication schedule. Larger values store
up to that many mailbox addresses in CTA shared memory while all 256 threads
write the outgoing payloads. Every writing thread executes a system fence before
a CTA barrier; only then may individual lanes publish the corresponding READY
words. All publication lanes join before descriptors are reused. A partial batch
is flushed at the last actual outgoing row, before trailing empty peers are
scanned, including under sparse or skewed routes. Its last peer is derived from
the current GPU owner counts and this CTA partition.
No timer or future-token arrival is needed to flush it.

This batches visibility notifications, not data ownership: each mailbox still
has one producer and one ACK writer. No extra payload copy, CPU routing, host
atomic or cross-rank collective is introduced. A batch may include different
destinations owned by this CTA, but sending never waits for their ACKs; it adds
no new credit dependency. All-resident launch and the existing final outgoing
ACK drain retain safe reuse across serialized eager/Graph calls. Receiving keeps
the original 256-thread CTA, contributor masks, scalar/vector fallback and
arithmetic order. No receive resources are reserved for a separate send group.

Counts and addresses come from each dispatch's current GPU owner lists. The
fixed batch bound is unrelated to a benchmark concurrency or model identity.
Fewer fences can improve throughput, while buffering notifications can delay
the earliest READY and add descriptor overhead; the option therefore remains
explicit, with an ordinary per-row path available.

Decode mailbox profiling is prepared once before submissions/capture. Captured
kernels retain that allocation; a GPU enable word is changed only at synchronized
request boundaries. Independent per-channel bounded sequences record epochs,
capacities, active channels and disjoint phase times. Samples never overwrite
older samples on overflow. Normal kernels instantiate profiling off, so there
is no diagnostic flag read, counter write, timer or added profile barrier in
production contexts. Concurrent replay/submission remains unsupported, as for
the underlying mailbox protocol.

## Bounded decode descriptor tiles

`EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32` enables an optional schedule for fused
mailbox combine; default 0 retains the existing path. Each CTA first computes
its per-peer intervals from current GPU owner counts. A prefix over these
intervals defines a flat send sequence with the same peer/row order as before.
Up to 32 lanes prepare row indices and destination addresses together. Tiles
are streamed, so shared storage does not grow with token capacity or route
skew. No descriptors or routes are reused across invocations.

All 256 threads still write each payload. Publication uses the independent
`EP_XPRESS_DECODE_PUBLISH_BATCH` bound: all writers fence, synchronize, and then
descriptor lanes publish READY. The last partial batch is published immediately.
Every lane joins before tile storage can be overwritten. Reception is unchanged:
all actual contributors must be ready, the full CTA reads/reduces in the original
arithmetic order, and ACK publication follows the last host read.

After reception, each CTA reconstructs bounded descriptor tiles from the same
immutable owner lists. Each active lane checks one outgoing mailbox ACK and
remembers when it has completed; completed ACKs are not polled again. The CTA
exits a tile only after all its actual ACKs arrive. There is still a final drain
before return and no unrelated rank obligation, host atomic, additional payload
copy, or credit bypass. Tiling reduces metadata barriers and serialized checks;
it cannot remove a remote consumer's genuine delay. All-resident launch and
serialized eager/Graph workspace lifetime requirements remain unchanged.


## Direct decode receiver indices and batched ACK commits

The optional `EP_XPRESS_DECODE_DISPATCH_INDEXED=1` build setting extends bounded
single-frame dispatch. All ranks check the same native build at bootstrap;
mixed build/protocol settings fail before communication starts. The default
remains the original mask-scanning path.

For each channel, let `R = ceil(decode_max_tokens / active_channels)`. After
reserving room for `R` BF16 payload rows, each destination needs a 64-bit count
and at most `R` descriptors, each packing the original source token index and
its compact wire row. This index region uses otherwise unused slot capacity;
it does not overlap the fixed ID/weight arrays or the reserved payload prefix.
The region is naturally aligned, and its entire extent must fit within the
existing slot stride. When the payload suffix is too small and at least two
slots were allocated, the single-frame protocol instead borrows the idle second
slot's payload area. It preserves that slot's READY/ACK header so the general
ring can later reuse it. The entire index list must fit in that payload area;
no additional host allocation is needed. All quantities come from configured
group-wide bounds. If neither extent fits, all ranks use the existing scan protocol,
including empty ranks. General prefill is unchanged and can reuse the same
storage only after the preceding operation has drained its ACKs.

A sender warp builds one destination's list from the current GPU rank masks,
using warp ballots over bounded tiles. It does not transfer a second payload
copy per destination. Each destination appears once per source token, regardless
of how many experts on that destination were selected. Counts, descriptors,
IDs, weights and payload are all covered by the existing writer system fences,
CTA barrier and frame READY release. Every replay rebuilds the lists; there is
no CPU route readback or reuse of a previous route.

A receiver acquires the producer READY and reads only its own count and indices.
It reserves compact GPU rows and builds the same inverse/owner lists without a
host mask scan or receiver CUB compaction. A zero count explicitly completes that
producer's metadata frame. It is not permission to ignore publication or ACK.

All frames found ready in one progress iteration are consumed independently.
After their last payload/metadata reads, all CTA readers fence and synchronize
once, then peer lanes release their separate ACK words in parallel. No new
all-rank commit word or host atomic RMW is introduced. A missing READY is left
for a later iteration and never needed to fill a commit batch. Final outgoing
ACK drain remains mandatory, including metadata-only readers; returning early
would allow the next layer or Graph replay to overwrite a live index region,
including one placed in the second slot.
The tradeoff is that an early frame's ACK may wait for other already-ready
frames in the same local batch. This is measured rather than assumed beneficial.

`prepare_decode_profile(stage="dispatch")` installs optional bounded per-channel
records before capture. The report separates host ID/weight processing from
payload reads and distinguishes final ACK checks from READY polling. Idle-loop
classification overlaps the ten disjoint phase totals. Generic eager diagnostic
schemas retain their previous phase partition. Normal contexts do not allocate
these counters or execute diagnostic timers/barriers. Diagnostic samples are
not a substitute for ordinary full-model TTFT and Decode TPS measurements.


## Experimental decode dispatch prefetch pipeline

Status: CUDA compilation, focused dynamic-route checks and same-machine
full-model comparisons completed; see the [performance report](model-performance.md#decode-dispatch-early-publication-and-double-stages-2026-10-25). The build flag `EP_XPRESS_DECODE_DISPATCH_PIPELINE=1`
requires `EP_XPRESS_DECODE_DISPATCH_INDEXED=1`; both remain opt-in. This does not
change prefill, combine, active channels, arena allocation or routing weights.

Eligibility is derived exclusively from group-wide configuration: bounded
single-frame dispatch, a fitting index region and no more than 32 configured
source rows per active channel. Larger bounds or insufficient space use the
existing indexed or general-ring path. Empty local inputs do not change the
wire protocol. The native build identity is already included in bootstrap.

The sender writes the same deduplicated payload once and constructs each
reader's compact descriptor list. After every writer fences and the CTA joins,
it release-stores `(epoch << 32) | (count + 1)` into each reader's aligned 64-bit
index header. An acquire of this word establishes both readiness and count;
there is no separate host count load. Count zero is encoded as one, so an
empty current-generation frame is distinguishable from uninitialized storage.
The original slot READY is retained for transitions back to the general ring.

A 320-thread CTA has two control warps and eight payload RX warps. Initial
routing, local bypass and TX publication complete before the receive pipeline;
this version does not overlap the local TX phase with RX. The control warps
own disjoint incoming peers by rank parity. Each has two shared-memory stages
and polls its pending peers concurrently; an unavailable peer cannot block a
ready peer in that group. After reserving output/owner ranges and reading all
row descriptors, control publishes the descriptor stage immediately. It then
reads IDs/weights and writes metadata/inverse maps concurrently with RX payload
reads. The second slot allows preparation of another peer while the first slot
is still being consumed; the two slots do not duplicate payload storage.

The 256-thread RX group consumes ready stages fairly, retaining the vectorized
host payload reader and scalar-tail fallback. Stage states are:

- Control-owned: descriptors may be replaced.
- Descriptor-ready: RX may read payload; descriptors/output ranges are immutable,
  while control may still be processing metadata.
- Payload-read-complete: all RX threads have fenced and joined; control may ACK
  only after its own metadata readers have also fenced and joined.
- Finished: no incoming stage or outgoing ACK remains in that group.

Shared stage transfers use CTA acquire/release operations. Only the RX group
participates in its 256-thread named barrier; control warps use warp barriers.
Descriptor publication follows the descriptor writers' warp barrier and uses
CTA release/acquire. Each metadata reader fences and joins its control warp
before that warp can examine completion or reuse either slot on its next loop.
Every payload reader fences before the RX leader release-publishes completion.
The control leader acquires completion before publishing the remote ACK. Thus
early RX completion alone cannot release the host buffer: ACK joins metadata
and payload completion. Slots are reaped independently and only by their owning
control warp. The other control group
and RX can keep progressing during that ACK store. No ACK depends on unrelated
peer completion, and no mutable host atomic counter is introduced. Count-only
readers ACK without a payload transfer. Each group also drains its actual
outgoing ACK words before operation completion and arena reuse.

The relevant ordering primitives follow NVIDIA's
[PTX memory and barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).
Source review is not a substitute for running the CUDA validation below.

The pipeline deliberately retains per-frame reader fences. Their cost, the
extra control-warps' register footprint, shared handoff latency and a possible
imbalance between parity groups can offset overlap gains. No numerical speedup
is assumed. The old nine/ten sequential phase counters cannot describe these
overlapping groups. Prepared decode profiling now selects a separate role-based
schema for this kernel, rather than profiling the old sequential schedule.
Combine sampling and ordinary CUDA timelines remain available.

Completed focused validation covers dynamic Graph routes and eager-prefill transitions,
int32/int64 IDs, empty/skewed ranks, scalar payload tails, two stages per control
warp refilled across multiple peers, and transitions between eligible/fallback channel
counts. Complete SGLang inference was compared against the previous indexed
implementation and default NCCL on identical GPUs with P2P/NVLink disabled,
reporting single-request and 32-request TTFT and Decode TPS. These checks cover
the measured configuration; the specialization remains opt-in.

## Ready-peer decode combine prefetch

The opt-in build setting `EP_XPRESS_DECODE_READY_PREFETCH=1` retains the full
256-thread fused mailbox receiver. For each token, it reads the current
GPU-produced expected-rank mask and polls only remote contributors that have
not yet been loaded. Two warp ballots cover the supported rank mask; a CTA
barrier makes acquired READY state visible to every payload reader.

If all contributors are ready before any partial prefetch, the receiver uses
the existing direct host-read/reduce path. Otherwise, it copies ready peers'
BF16 payloads into per-peer CTA shared memory while missing peers continue
producing their results. Staging retains `EP_XPRESS_DECODE_PEER_BATCH` load
grouping: issue each group of independent host reads before storing their
results to shared memory. A loaded-mask prevents duplicate reads. After all
expected payloads are present, it reduces them in ascending rank order,
including the existing absent-rank zero additions, then performs the original
remote BF16 rounding and local BF16 addition. Scalar tails use the same order.
No arrival-order accumulation or reuse of a previous token's route is allowed.

Staging requires `world_size * hidden_size * 2 <= 40 KiB` per CTA; larger shapes
use the original wait/read path. This is a static shape-based resource bound,
not an assumption about top-k, balance, prompt shape or future routes. No HBM
scratch or additional CPU routing operation is introduced. The allocation is
fixed at launch/capture, while readiness and route masks remain GPU-dynamic.

The existing payload fences, output completion, ACK publication and final
outgoing ACK drain are retained. Staging does not release a sender's slot early.
The all-TX-before-RX schedule is also retained, so this overlaps ready-peer reads
with remote producer progress rather than claiming simultaneous local TX/RX or
a dedicated polling warp. Shared-memory traffic and occupancy are costs to
measure against any reduction in exposed READY waiting. The option defaults
to zero; see the performance report for the evaluated configuration.


## Decode combine control-warp polling

`EP_XPRESS_DECODE_CONTROL_WARP=1` moves fused mailbox READY and outgoing ACK
polling to the first warp. It is a build-time opt-in; the default remains zero.
READY lanes derive pointers from this call's GPU expected-rank mask. Each lane
can poll two peers, covering all supported ranks through 63, and stops polling
a contributor after observing its current epoch. ACK lanes similarly retire
actual outgoing descriptors. There is no cached routing assumption or all-rank
completion requirement.

The other seven warps wait at one CTA rendezvous. Once every required READY
has been acquired, that rendezvous transfers visibility to the full 256-thread
receiver. The existing ascending-rank FP32 reduction, remote BF16 rounding and
local addition are unchanged. After payload consumption, all reader fences,
ACK publication and final outgoing ACK drain remain. The optimization reduces
barrier participation during polling; it cannot remove the dependency on a
late actual contributor or make the PCIe control-word load itself local.

When operation diagnostics are disabled, the identity-check block and its two
CTA barriers are omitted. Diagnostics retain those barriers when enabled.
Dynamic-route bounds checks, status propagation, epoch acquire/release and
buffer-lifetime synchronization are retained. The helper never invokes a
CTA-wide profiling barrier from within its warp; opt-in phase sampling remains
at uniform CTA boundaries.

Keep `EP_XPRESS_DECODE_READY_PREFETCH=0` for the evaluated direct receive path.
If both experimental options are enabled, eligible staging continues to use
its existing polling loop; control-warp READY applies to direct/fallback reads,
and outgoing ACK draining still uses the control warp. This combination has
not been evaluated here. Neither switch changes prefill transport scheduling.


## Dispatch pipeline role diagnostics

Preparing `stage="dispatch"` with the pipeline build selects a separate CUDA
specialization and schema 3. Normal contexts have no diagnostic storage, clocks
or counter accesses. A GPU enable flag captured by Graphs controls bounded
samples; reset/readback remains at synchronized request boundaries. Profile
preparation checks both ID-width specializations' resident capacity. Fallback
and eager-prefill kernels remain ordinary and are deliberately not sampled.

Thread zero records the TX prefix, control-warp leaders record their respective
receive/ACK timelines, and the RX leader records its own timeline. Each role's
phases partition that role's duration. TX subdivision is leader instruction
residence: it introduces no additional rendezvous, so publication includes
writer stragglers. Similarly RX completion includes its original reader fence
and barriers; RX payload time alone is not full-group completion or wire time.
The existing final CTA barrier precedes record emission. Final record writes
are excluded from elapsed time.

Each control role splits READY poll, descriptor/output allocation, IDs/weights,
metadata fence/join, occupied-stage wait, ACK publication, and final
outgoing ACK drain. With early descriptor publication, publication falls in the
descriptor interval, and metadata completion remains in `metadata_commit_ns`.
Poll/missing-round counts distinguish immediately available
notifications from rounds with none ready; they are not a count of all missing
peers. Empty notifications and received payload rows are counted separately.
After finishing incoming work, each control leader publishes a diagnostic-only
shared flag. RX can then distinguish waiting for an incoming stage from waiting
while only outgoing ACK drain remains. The flags never control transport progress.

Role intervals overlap. Never add control, RX and TX totals as a wall-time
breakdown, subtract timestamps across GPUs, or apply their percentages directly
to an uninstrumented trace. Choosing the latest-finishing CTA per rank/operation
identifies a local critical tail, not a globally synchronized causal trace.
Prepared but disabled kernels retain profiling resources. Enable/disable trials
measure active counter perturbation; an ordinary context is also required to
assess the behavior of the production specialization.


### Bounded decode control records and receive completion batches

The optional `EP_XPRESS_DECODE_CONTROL_RECORDS` layout appends bounded records
behind the legacy receiver index region. Each receiver row holds one 64-bit
source-token/wire descriptor, followed by top-k 64-bit words packing a signed
32-bit local expert ID and the original FP32 weight bits. Top-k positions remain
unchanged; nonlocal experts use -1. Producer-side mapping reads the current GPU
routing tensors on every invocation. Existing writer fences publish records and
the deduplicated payload before generation/count READY. The receiver's route
loads no longer depend on a host-loaded wire index. Metadata and payload readers
still both finish and fence before ACK. The extra allocation is included in the
group-wide index-space eligibility check; fallback never depends on local live
row counts. The single-frame operation drains ACKs before another operation may
reuse borrowed ring-slot storage.

`EP_XPRESS_DECODE_ACK_BATCH` applies only to bounded decode mailbox combine.
Send completion precedes receive processing, and every returned token has a
distinct mailbox, so delaying a small completion batch does not consume credit
needed for another send in the same operation. The full CTA retains the original
rank-ordered arithmetic and payload reads. Non-final rows keep a token-boundary
barrier to protect shared reader masks/local-row state; final rows perform an
all-reader system fence and join before parallel per-reader ACK release, then
join before reusing completion masks. The final partial batch always flushes.
An error aborts the operation rather than falsely acknowledging incomplete data.
No host atomic, global ACK or route cache is introduced. This is batching of
fences and individual ACK publication, not a shared bitmap wire protocol.

The control-record writer assigns consecutive route words to warp lanes using
an ephemeral per-warp compact row list. It does not serialize all top-k stores
on a token's single lane. The scratch list is bounded by the pipeline's source
row limit and is rebuilt from the current route ballot, with warp joins before
consumption and reuse.

Receive completion batching can only amortize fences when one CTA owns more
than one local output row. With the adapter's balanced eight-rank decode rule
(one channel per eight global tokens), batch 32 has four local rows and four
channels: every nonempty CTA flushes a one-row tail even with ACK_BATCH=4.
Changing that would require a different scheduling/completion design. The flag
intentionally does not reduce the receiver grid or add a cross-CTA barrier.

### Bounded decode combine: balanced sends and independent progress

Three build-time experiments require `EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32`:

- `EP_XPRESS_DECODE_BALANCED_SEND=1` prefixes the live GPU owner counts across
  all remote destinations, then divides the concatenated row range across
  channels. Per-channel row counts differ by at most one. A descriptor maps a
  global row ordinal back to its peer and owner-list index. Unequal/empty peers
  and changing token counts use the same rule; no CPU planning or route cache
  is introduced. Receive token ownership is unchanged.
- `EP_XPRESS_DECODE_SPLIT_TX_RX=1` launches two full 256-thread CTAs per logical
  channel: one sends, the other receives and reduces. Unlike splitting a CTA's
  warps, this retains all receiver lanes. The cooperative launch and occupancy
  bound account for both roles, including the single-logical-channel case.
  There is no grid barrier between sending and receiving. Each rank still
  completes its local sends and receives before the kernel returns.
- `EP_XPRESS_DECODE_DEFER_RECLAIM=1` requires independent TX/RX. Outgoing ACKs
  no longer gate the end of combine. Before overwriting a mailbox, the sender
  reads that mailbox's last published generation from producer-owned GPU state
  and waits for its matching host ACK. The generation cache is initialized once
  with the arenas and updated after each publication, avoiding a host READY
  read on reuse. This uses the last actual publication, not `epoch - 1`: dynamic
  routing can leave a mailbox unused for many operations. All pending checks in
  a bounded publication batch progress concurrently. Initial READY/ACK are zero.

The deferred variant retains the single-bank mailbox layout. It separates
completion of the local result from reclamation of outgoing host slots; it does
not create an extra copy or imply unlimited buffering. A slow reader can still
stall reuse of its specific mailbox. Only the producer writes READY/payload;
only the designated destination writes ACK. Distinct CTAs own disjoint outgoing
rows, and subsequent invocations remain serialized on the transport stream.
Native callers must initialize a fresh, quiescent arena together with its
producer state, use one prepared producer state per mailbox arena, and provide
non-overlapping partial/output storage. Resetting only the GPU generation cache
while readers still use an old arena is invalid. The context supplies these
lifetimes and does not reset generation state between replays.
Every payload writer still executes a system fence before READY publication;
every payload reader completes its reads before its ACK. The established
ascending-rank reduction and BF16 rounding sequence are unchanged.

Progress does not rely on a blocked TX CTA releasing resources for RX. All TX
and RX CTAs are admitted together by a cooperative launch, with capacity checked
at initialization. An earlier invocation's reader can finish and ACK while
another rank is already waiting to reuse that slot. The required healthy
collective `Context.close()` drains local work and rendezvous before unregistering
mapped arenas. Native callers must provide equivalent distributed lifetime
management: local result completion alone is not proof that remote readers are
finished. No CPU work or collective is added between model layers.

The independent-role profiler allocates twice the channel storage. Physical CTA
indices below the operation's active channel count are TX; subsequent active
indices are RX. Each CTA's phase fields partition its own elapsed interval;
concurrent role durations must not be added as operation latency. Deferred
sender credit checks are reported in `idle_wait_credit_ns`, including the check
and barrier cost even if the first poll succeeds. Production runs allocate no
profiling storage and use the uninstrumented specialization.

These settings are opt-in pending end-to-end evidence. Full receiver width is
preserved while publication size can be varied independently. A larger batch
amortizes fences but delays the first READY; a smaller batch publishes earlier
but adds fences. Neither one-token workloads nor one particular route pattern
is baked into the transport's partitioning rule.

### Experimental decode packets and early host-slot reclamation

`EP_XPRESS_DECODE_PACKET_ROWS=2`, `4`, or `8` selects a separate build-time
combine transport; zero retains the existing mailbox implementation. Disable
the legacy balanced-send, split-TX/RX, deferred-reclaim and READY-prefetch options
when selecting packets. Dispatch, expert placement, token deduplication, routing
weights and prefill pair queues are unchanged. This is a measured experiment,
not a default performance policy or a batch-size-specific fast path.

One packet represents `(producer, destination, original-token tile, bank)`.
Its payload has fixed row positions within the tile and a single READY/ACK
pair. Only live rows cross PCIe; holes consume reserved arena space but no
payload traffic. Partial tiles publish immediately. The receiver already knows
the current contributor masks from dispatch and does not fetch a host count
or perform a metadata collective to discover packet membership.

A GPU preparation kernel rebuilds row indexes and packet masks from the current
owner lists. Each peer compacts its nonempty send/receive tasks on the GPU.
Transport CTAs partition the concatenated live task lists, including empty and
skewed peers. The lists are rebuilt for every invocation, including Graph
replays; task completion order may vary without changing reduction arithmetic.
No Python route inspection, per-call allocation, CPU planning, or host barrier
is added. Preparation still scans the configured bounded token index, so its
cost is part of the end-to-end comparison rather than assumed negligible.

The transport launches one full 256-thread TX CTA and one full 256-thread RX
CTA per active channel, all cooperatively resident. A control warp polls up to
32 assigned tasks and selects an available task without waiting for the other
tasks in that window. RX stages the chosen peer's live payload into persistent,
rank-private HBM. Every payload reader completes before a single ACK releases
the whole packet. A subsequent stream-ordered GPU kernel reduces staged data
in ascending rank order, rounds the remote sum to BF16, then adds the local
BF16 result exactly as the legacy path does. Absent peers are never read from
scratch. Host slots can be reused before this local reduction completes.

Lookahead is bounded: a CTA drains its current window before scanning the next.
The window-retirement barrier protects the exhausted/error selection before
the control warp can overwrite shared selection state for the next window.
There is no claim of global work-conserving scheduling. Transport completion
still joins local TX and RX before the separate reduction launch; this does not
overlap the next expert computation with unfinished outgoing payload writes.

Each packet has two payload banks selected by operation epoch. The producer
retains each bank's last *actual* publication in GPU memory. Before reuse it
requires an ACK for that exact generation; skipping routes or operation epochs
does not turn `epoch - 2` into a valid credit rule. No outgoing final ACK drain
is performed. All READY/ACK release/acquire operations, writer/reader fences,
stream ordering and distributed arena lifetime requirements remain. Bounded
buffers can still apply backpressure if a reader is late; double buffering
does not remove this correctness dependency.

Bootstrap queries the compiled packet layout, checks its Python/native size
agreement, includes both banks in the host memory budget, and allocates GPU
metadata and scratch once. `Config.mailbox_arena_bytes(world, packet_rows=...)`
can estimate this optional arena; its default and `Config.pool_bytes` describe
the default mailbox. Packet scratch requires `world * decode_max_tokens *
hidden_size * 2` bytes of HBM per prepared context, in addition to integer
indexes, task lists and generation state. Each arena and producer generation
state must share a lifetime; native callers must not attach multiple independent
prepared producers to the same arena or alias partial/output storage.

The optional combine profiler samples transport CTAs only. Sending metadata
includes credit polling; receive polling includes ready-task selection. Packet
payload receive time excludes the later HBM reduction. Frame counters count
nonempty packets, whereas row/byte counters count actual live payload. Preparation
and final reduction are separate kernels and must be included from a model GPU
timeline when assessing latency. Instrumented CTA times are not additive across
concurrent roles or ranks. Production uses the uninstrumented specialization.

`EP_XPRESS_DECODE_PACKET_CHANNELS=4/8/16/32` optionally selects packet-only
parallelism, clamped to the prepared cooperative channel capacity. Zero inherits
the caller's active channels. This setting does not change dispatch channels or
the token routing rule. A full TX and RX CTA are reserved per selected channel;
empty assigned lists finish immediately. More CTAs are a measurement variable,
not a guaranteed improvement for small payloads.

The vector packet implementation issues up to four independent host loads
before their dependent HBM stores, uses one aligned 128-bit host store instead
of four strided 32-bit stores, and reuses rank-ordered vector reduction on the
staged HBM layout. Scalar/unaligned cases retain a bounded fallback. Neither
transfer width nor packet capacity changes which rows are routed or weighted.


### Independent bounded dispatch and delayed bank reclamation

`EP_XPRESS_DISPATCH_SPLIT=1` launches two cooperative resident CTA sets for
bounded indexed dispatch. Physical CTAs `[0, channels)` publish local source
rows; `[channels, 2*channels)` receive their channel's peer rows. Every RX CTA
retains 256 payload threads and two descriptor-control warps. Output-count and
owner-list reservations remain GPU-local atomics. RX starts without waiting for
local TX. The ordinary completion of both CTA sets orders subsequent expert
computation. Occupancy negotiation includes both sets and the instrumented
specialization; scheduling never relies on a nonresident block making progress.

`EP_XPRESS_DISPATCH_DEFER_RECLAIM=1` additionally separates local result
completion from outgoing host-buffer reclamation. A physical channel owns the
original fallback ring plus two full decode extents. Decode chooses a bank by
its GPU operation epoch and stores each bank's **last actual published sequence**
in producer-only HBM. Before the next overwrite, its TX control lanes wait for
all readers of that publication to ACK. RX CTAs continue progressing independently.
After incoming metadata and payload have joined and their ACKs have been issued,
RX can finish without draining this operation's outgoing ACKs. All necessary
writer fences, metadata/payload joins and remote ACKs remain.

Count-only readers still acknowledge their control records: zero payload does
not make a mutable READY/count record safe to overwrite. Credits cover the
whole bank, including any borrowed index payload in its second slot. Eager
fallback uses a different extent; Graph buckets, active channel counts, empty
ranks, skipped routes and prefill/decode transitions cannot alias live decode
storage. Using `epoch - 2` as a substitute for the last publication is incorrect
when a bank/channel was not used in intervening operations. Healthy context
shutdown synchronizes local work and rendezvous with all ranks before unmapping
host memory. There is no per-operation CPU collective.

The pinned dispatch allocation grows by a factor of three for this opt-in build;
combine arenas are unchanged. The native allocation query and
`Config.mapped_arena_bytes(..., dispatch_banks=2)` agree, and bootstrap validates
the total against the host-pool budget. Static `Config.pool_bytes()` without
compiled-option arguments remains the default-layout estimate.

### Per-handle return indices

`EP_XPRESS_REUSE_RETURN_INDEX=1` requires packet combine. Each fused route handle,
or its explicitly borrowed workspace, owns an `int32[world, decode_max_tokens]`
map from source rank/token to compact expert-output row. Dispatch clears this
bounded map in its existing initialization kernel and fills it when it writes
inverse metadata, including generic fallback dispatch. Packet preparation reads
this map directly instead of clearing a context-wide map and traversing
`owner_rows -> inverse -> token` again. It still constructs the current live
packet masks and task lists; no routing decision is cached between invocations.

The map is independent for outstanding handles, retained across Graph capture,
and recorded on consumer streams. It is not valid to store it only in the context:
a later dispatch could then silently replace an earlier uncombined handle's
routes. Local rows remain excluded from remote sends, and reduction keeps the
existing rank order and weight application.

### Dispatch stores and interpretation of counters

`EP_XPRESS_DISPATCH_WRITE_UNROLL=1/2/4` uses aligned 128-bit payload stores and
bounded instruction unrolling in the pipeline TX role. Row widths or pointers
unsuitable for 16-byte vector access retain the existing 32-bit transfer path;
this does not add support for two-byte-offset BF16 views. Sparse and partial
batches retain exact bounds.
It does not change routing, frame size, channel count, READY granularity or ACK
ownership. These knobs must be compared independently before selecting a policy.

Split dispatch profiling records physical TX and RX CTAs separately. Active
channel count is stored with each record; the RX set starts at that count.
`tx_credit_wait_ns` isolates delayed-bank checks from route/local work. TX,
control and RX timelines overlap, and role durations must not be added as if
sequential. Instrumented kernels are separate from performance measurements.


`EP_XPRESS_HOST_READ64=1` separately replaces the legacy pair of 32-bit loads
used for aligned 64-bit host metadata with `ld.global.cv.u64`. Descriptor and ID
words are immutable after their matching READY; the acquisition of that READY
still uses the system-acquire helper. All metadata offsets retain natural
8-byte alignment. This reduces load instructions without changing the wire
format, synchronization ownership or payload. PCIe transaction counts are not
inferred from instruction width alone.
