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
one serialized CUDA transport stream. An epoch belongs to the combine
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
The previous combine must have been waited, and all consumers of its output
must have been enqueued on the wait stream before reuse. Retaining output
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
remain in place because CUDA Graph execution is not implemented.

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
