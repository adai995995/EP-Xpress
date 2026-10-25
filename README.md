# EP-Xpress

An independent expert-parallel communication library for single-node MoE
inference through host memory, without GPU peer access or NVLink payloads.

**Status: experimental BF16 inference implementation.** Two selectable transports
share the dispatch/combine interface: `host_ce` uses CPU routing and copy
engines; `host_sm` runs routing and mapped-host communication on the GPU.
The `host_sm` direct-stream API supports opt-in serialized CUDA Graph replay;
`host_ce` remains eager. See the SGLang adapter for the supported decode scope.
End-to-end performance must be measured against the intended inference framework and workload.

## Implemented

- Token routing with one activation row per destination rank, even when several
  selected experts are on that rank. The router and expert GEMM stay outside
  the library.
- Source-shared host payloads, CE D2H/H2D, and CPU gather into reusable pinned
  receive packs in `host_ce`. The `host_sm` path uses GPU loads/stores to
  registered shared host pages, with GPU routing, destination deduplication,
  publication, acknowledgments, and inverse routing.
- No remote GPU pointers, GPU IPC, peer copies, or RDMA in either transport.
- Bounded host rings with configurable slot count, immutable published chunks,
  generation IDs, consumer ACKs, and simultaneous send/receive progress.
  General prepared `host_sm` dispatch shares each producer's payload among its actual
  readers and reclaims it using only their ACKs. Combine uses independent
  queues for each directed rank pair and channel, with GPU owner row lists
  generated during dispatch. Optional warp specialization gives prepared pair
  queues four sending warps and four receiving warps within the same CTA.
  An optional bounded decode path receives and
  reduces directly from per-token host mailboxes in one combine kernel.
- Opt-in bounded decode dispatch uses one final frame per source/channel,
  removing a separate terminal exchange. Selected ranks still share token
  payload; all frame-metadata readers acknowledge reuse. Selection depends on
  configured capacity and frame size, with the general ring as fallback.
  The SGLang automatic policy derives the minimum width from those bounds.
- Explicitly borrowed decode workspaces reuse communication tensors. Generic
  calls without a workspace retain operation-owned outputs and route storage.
- Operation-scoped inverse-route handles, already-weighted rank partials,
  deterministic rank-order FP32 remote accumulation, and BF16 results.
- A dedicated Gloo control group with a bounded timeout. In `host_ce` it
  carries initialization and route metadata. In `host_sm` it is used for
  setup and shutdown only. GPUs process routing IDs and weights and transfer
  them through host queues without CPU route planning; receive counts remain
  device-resident. The caller preserves operation order and supplies capacity bounds.
- An experimental adapter patch for SGLang v0.5.21, kept separate from the core.

The `host_ce` reference may export unused rows within a nonempty source chunk.
Reported CE D2H bytes include those rows; ideal routing byte counts are not
reported as actual transfer volume. Optional SM combine diagnostics count
requested payload loads/stores; these are not hardware PCIe traffic counters.
A host copy is shared, but each destination still consumes its own PCIe read
bandwidth.

## Requirements and install

The CUDA path requires Linux, NVIDIA GPUs supporting BF16, a matching CUDA
toolkit with `nvcc`, PyTorch, a C++ compiler, and Ninja. All ranks must share a
host-memory filesystem, normally `/dev/shm`. Allocate sufficient shared memory
when using containers. One process owns one GPU.

```sh
python -m pip install --no-build-isolation -e .
```

The bundled extension builds lazily through PyTorch's extension cache. Set
`TORCH_CUDA_ARCH_LIST`, `TORCH_EXTENSIONS_DIR`, and `MAX_JOBS` as appropriate
for the target system. Importing `ep_xpress` does not compile or initialize CUDA.
For offline installation, provision PyTorch and build tools beforehand, then
use `python -m pip install --no-deps --no-build-isolation -e .`.

## API

Initialize a Gloo process group and set the current CUDA device before creating
a context. Every EP rank must submit operations in the same order. Compatible
model layers should share a long-lived context and its memory budget.

```python
from ep_xpress import Config, Context

config = Config(hidden_size=2048, topk=4, max_tokens_per_rank=4096)
with Context(group=ep_group, expert_to_rank=expert_owners, config=config) as ep:
    batch = ep.dispatch_async(x, topk_ids, topk_weights, tag="layer-0").wait()
    # Runner applies each routing weight exactly once and sums experts on this
    # rank. partial has shape [batch.valid_count, hidden_size], dtype BF16.
    partial = expert_runner(batch.hidden, batch.local_ids, batch.weights)
    output = ep.combine_async(partial, batch.handle, tag="layer-0").wait()
```

`x` is contiguous BF16 `[T, H]`; IDs are integer `[T, K]`; weights are FP32
`[T, K]`, all on the local device. `local_ids` uses `-1` for experts on other
ranks. Active routing weights are retained and are not renormalized.
Input IDs also accept `-1` as a dropped expert slot. An all-`-1` row represents
a padding or dropped token: dispatch omits it and combine returns a zero row
at its original source position, preserving the input token count.
Weights paired with input `-1` IDs are ignored and canonicalized to zero,
including nonfinite padding values. Active weights must remain finite.

`Pending.wait(stream)` can block the host while eager metadata and dynamic
shapes are resolved in `host_ce`. It then inserts an event dependency into the
consumer stream. It is not a promise of CPU-free progress or Graph compatibility.
Operation submission is bounded; each context serializes its background
operations, while each operation progresses D2H and H2D independently.

For GPU-initiated communication, select `transport="host_sm"` and provide a
CPU-known bound for the number of received rows:

```python
config = Config(hidden_size=2048, topk=4, max_tokens_per_rank=4096,
                transport="host_sm", channels=4)
with Context(group=ep_group, expert_to_rank=expert_owners, config=config) as ep:
    batch = ep.dispatch_async(x, topk_ids, topk_weights,
                              recv_capacity=global_token_bound).wait()
    partial = expert_runner(batch.hidden, batch.local_ids, batch.weights)
    output = ep.combine_async(partial, batch.handle).wait()
    ep.check()  # Explicit validation boundary; synchronizes outside the hot path.
```

Here `batch.valid_count` stays on the GPU. Receive tensors have the requested
capacity; unused rows have expert IDs `-1`, zero weights, and zero activations.
The expert runner must either honor the GPU valid count or support these
masked rows. `wait()` inserts a CUDA event
dependency without waiting for GPU routing on the CPU. Without an explicit
bound, the context reserves `world_size * max_tokens_per_rank` rows, which can
be wasteful for decode. An undersized bound is an error, never silent truncation.
The SGLang adapter obtains its bound from existing scheduler metadata.

Host-SM arenas are allocated and initialized by their payload producer at
context setup, including rank-pair combine queues and decode mailboxes. Bind
each worker's NUMA memory policy to its intended GPU-local node before creating
the context. The library follows that existing policy; it does not infer a node
from rank numbers or change process affinity. All producers finish allocation
before any peer registers the shared pages. This initialization collective adds
no synchronization to dispatch or combine. Verify physical placement with
`/proc/<pid>/numa_maps`; CPU affinity alone is not proof of page placement.
Producer-local placement is not an established general speedup. A batch-32
regression prompted a follow-up investigation of NUMA placement, dynamic
prefill batches and first-use kernel compilation; see the
[profiling investigation](docs/model-performance.md#numa-regression-investigation).

For sequential, fixed-shape decode, the SM context also supports explicit
workspace reuse and receive/reduce fusion:

```python
import torch

# Configure decode_max_tokens > 0 when constructing this host_sm context.
forward_stream = torch.cuda.current_stream()
workspace = ep.create_workspace(global_token_bound, x.shape[0], fused_combine=True)
batch = ep.dispatch_async(x, topk_ids, topk_weights, workspace=workspace,
                          fused_combine=True, stream=forward_stream).wait(stream=forward_stream)
partial = expert_runner(batch.hidden, batch.local_ids, batch.weights)
output = ep.combine_async(partial, batch.handle, stream=forward_stream).wait(stream=forward_stream)
# Enqueue every consumer of output on forward_stream before reusing workspace.
```

Workspace-backed batches, outputs and tickets are borrowed until that
workspace's next dispatch. Wait on its previous combine and enqueue all output
consumers before reuse; clone any output that must survive reuse. A workspace
has an exact receive capacity, local token count and fusion mode. Different
outstanding operations need different workspaces. The default non-workspace API
keeps independent storage. Fusion and reuse are independently selectable.

`Config(decode_max_tokens=N)` reserves mapped return mailboxes for up to `N`
source tokens per rank; `0` disables them, and supported positive values are
1 through 256. `fused_combine=True` makes dispatch retain each source token's
contributor mask and local row on the GPU. Combine then reads only actual remote
contributors, accumulates in the existing numerical order, and writes output
directly. It does not allocate remote scratch or launch its separate clear and
final-reduction kernels. Only published contributions require ACKs; this
combine path has no all-peer terminal frames. General dispatch retains its
terminal handshake; bounded single-frame dispatch folds completion into its
only frame. Larger operations use the existing pair queues.

`EP_XPRESS_DECODE_PUBLISH_BATCH=8` opts into bounded READY publication batching
for fused decode. It keeps full-CTA reception and the existing ACK protocol.
The default `1` retains per-row publication; this is a throughput/latency tradeoff,
not a universal speedup. Set it before extension loading or worker startup.

`EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32` additionally prepares outgoing row and
mailbox descriptors in bounded GPU tiles and checks each tile's outgoing ACKs
in parallel. Default `0` retains the original descriptor/check schedule. Both
paths retain full-CTA reception and, by default, wait for every actual reader
before return. The separate deferred-reclamation experiment below changes the
slot-lifetime rule.
Routes are rebuilt from current GPU owner lists on each invocation; the tile
size is independent of token count and does not limit supported routing shapes.

`EP_XPRESS_DECODE_CONTROL_WARP=1` optionally assigns fused decode READY and ACK
drain polling to one warp while retaining all 256 payload threads. Only actual
dynamic contributors are waited on; the CTA joins once after polling. Diagnostic
identity-check barriers are omitted when diagnostics are disabled. Required
reader fences, ACKs and rank-order reduction remain. Use with ready prefetch off;
see the [protocol](docs/architecture.md#decode-combine-control-warp-polling).

`EP_XPRESS_DECODE_READY_PREFETCH=1` experimentally stages ready peers' decode
payloads while other contributors are pending. It preserves the 256-thread
receiver, grouped peer reads, rank-order reduction and ACK drain. All-ready
calls read directly; shapes exceeding 40 KiB of CTA staging use the existing
path. It is disabled by default; see the [architecture](docs/architecture.md#ready-peer-decode-combine-prefetch)
and [model measurements](docs/model-performance.md#ready-peer-combine-prefetch-2026-10-25)
for scope and results.

`EP_XPRESS_DECODE_DISPATCH_INDEXED=1` opts bounded single-frame dispatch into
GPU-generated receiver index lists and batched ACK publication (default `0`).
The payload is still written once for all selected readers. Receivers read their
own compact indices instead of scanning every transmitted token's rank mask.
All completed reads share one fence/barrier before separate ACK words are
published in parallel; an unavailable peer never holds up a commit batch.
The index lists borrow unused frame space or the idle second slot's payload area. Selection uses
group-wide capacity bounds and falls back to the original protocol when space
is insufficient; no assumption about live routing or a particular model is used.

Experimental, validated on the configuration in the [performance report](docs/model-performance.md#decode-dispatch-early-publication-and-double-stages-2026-10-25): `EP_XPRESS_DECODE_DISPATCH_PIPELINE=1`
requires indexed dispatch and adds two control warps beside a full 256-thread
receiver group. A receiver-specific 64-bit READY contains generation and count;
each control warp has two descriptor stages and publishes payload work before
reading route metadata. ACK joins metadata and payload completion independently
per peer. This
specialization requires at most 32 configured source rows per active channel
and a fitting descriptor region; other shapes retain the existing protocol.
It is disabled by default. Optional role-based dispatch counters describe the
overlapping control and RX timelines; sequential phase counters do not apply.
See the performance report for version-specific end-to-end measurements.

Two independent experimental build options retain the existing defaults:

- `EP_XPRESS_DECODE_CONTROL_RECORDS=1` requires the dispatch pipeline and writes
  receiver-specific contiguous control records: a payload/source descriptor plus
  packed local expert IDs and unchanged FP32 weights. The receiver addresses
  metadata by row ordinal instead of first loading a wire index. Payload remains
  one producer copy shared by its readers. Extra control storage is bounded by
  configured capacity; insufficient space selects the existing fallback on all
  ranks. No routing decisions are cached between replays.
- `EP_XPRESS_DECODE_ACK_BATCH=4` batches decode combine receive completion for up
  to four consecutive token rows (accepted values: 1, 2, 4, 8; default 1). All 256
  threads still receive and reduce each token in the original order. Every reader
  fences before the batch's individual ACK words are published in parallel. A
  token-boundary barrier protects shared metadata, and a partial final batch is
  flushed. This reduces fence/join frequency, not the number of ACK words or the
  READY dependencies. The general pair-queue combine path is unchanged. Batching
  needs multiple local token rows per channel: a balanced eight-rank batch of 32
  with four channels per rank has only one row per CTA and cannot amortize fences
  this way. This option does not reduce channels to manufacture a larger batch.

`Config(combine_frame_rows=N)` sets rows per pair-queue frame independently of
channel count, for `N` in 1 through 256. The default `0` retains the original
split of `chunk_bytes` across channels and peers. Larger fixed frames consume
more host memory as the allocated channel count grows; the host budget covers
both queues and optional mailboxes. Dispatch frame sizing is unchanged.

`Config(combine_warp_specialization=True)` lets prepared pair-queue combine
advance sending and receiving in independent four-warp groups within each
256-thread remote-peer CTA. Local copies retain the whole CTA. The core default
is `False`, which retains the sequential
send/receive implementation. The SGLang adapter enables this option by default
with `EP_XPRESS_COMBINE_WARP_SPECIALIZATION=1`. It changes neither the grid,
channel count, frame capacity nor fused decode execution. Separate group
metadata and barriers preserve READY/ACK ordering while permitting concurrent
progress; this alone establishes neither measured bidirectional PCIe bandwidth
nor overlap with expert computation.

Normal submission and `wait()` do not scan host queue error headers. Use
`check()` at a completed experiment/request boundary to inspect asynchronous
errors. `Config(diagnostics=False)` is the default for `host_sm`: tag hashing,
operation fingerprints, peer-abort/timeout polling, duplicate expert detection,
and finite-weight diagnostics are disabled. Inputs must still satisfy the
unique top-k and finite-weight contract. Set `diagnostics=True` to investigate
protocol errors; absent peers otherwise require external job cancellation.
Required ready/credit checks, payload bounds, and slot-lifetime ordering remain
in the transport. Bootstrap creates a prepared native operator that validates
and caches placement, queue layout and launch capacity once. Only changing
tensor shapes/layouts and bounds are checked on each operation. The adapter
reuses SGLang's CPU capacity already computed once per forward.

The SM backend batches routing and metadata access. `channels` is the allocated
number of logical work partitions, not a count of reserved SMs. A dispatch may
supply `active_channels` from 1 through `ep.effective_channels`; all ranks must
use the same value. Omitting it uses the full allocation. The handle retains
this width for combine. Dispatch launches one CUDA block per active channel;
pair-queue combine launches one block per `(active channel, peer)`, including a local-only
block for the current rank. Four active channels on eight ranks therefore
launch four dispatch blocks and 32 combine blocks per GPU. Arena addresses and
frame sizes stay fixed when the active width changes. CUDA places those blocks
on SMs according to resource availability. Fused mailbox combine instead
launches one block per active channel.

`channels` defaults to four. Initialization negotiates a common limit from
dispatch and combine kernel occupancy, including the optional diagnostic
kernel, and exposes it as `ep.effective_channels`. Multi-rank `host_sm`
requires cooperative launch support so the whole communication grid can be
resident. Each pair-queue combine block progresses both directions of its rank pair:
only that pair's receiver acknowledges its outgoing slots. Prepared dispatch
publishes per-reader READY notifications for the union of a frame's target
ranks, so unrelated ranks do not hold up payload-slot reuse. A separate terminal
handshake with every peer protects operation completion and epoch reuse. The
legacy raw native dispatch API retains its all-peer header ACK protocol.
The sole performance baseline is SGLang's default NCCL backend; see
[the protocol description](docs/architecture.md) and
[full-model results](docs/model-performance.md).

Each handle can be combined once and belongs to its context. Zero-input ranks
still receive work; zero-receive ranks still participate in combine. Healthy
`close()` is collective and must run before destroying the caller's process
groups. After failure, restart the worker group instead of reusing the context.

### Optional combine diagnostics

For `host_sm`, start and stop a bounded capture at completed request boundaries:

```python
ep.start_combine_profile(max_operations=64)
# Submit the workload normally. An optional CPU-only label can be set once
# per forward with ep.set_combine_profile_label({"phase": "EXTEND"}).
run_workload(ep)
report = ep.stop_combine_profile()
```

This operation capture supports eager dispatch and pair-queue combine, including
warp-specialized combine. It excludes fused decode and CUDA Graph mode; use the
separate mailbox capture below for that path. For warp-specialized combine,
Tx/Rx phase times overlap and must not be summed as latency.

The start call preallocates counters; the stop call bulk-copies them after one
transport-stream fence. No per-layer CPU readback or collective is added.
Excess operations run uninstrumented and increment `dropped_operations`.
`read_combine_profile()` returns the completed report without another fence.
Schema version 4 contains native field names, operation labels, and
`active_channels * world_size` rows per operation. Each operation records its
own `active_channels` and `counter_rows`, which map its raw rows to
`(channel, peer)` pairs, including local-only rows. Top-level `channels` and
`counter_rows` describe the allocation; inactive rows are omitted from each
operation's raw counters. The capture size must fit the 64 MiB combined GPU-counter and pinned-readout
budget; reduce `max_operations` for larger grids. Diagnostic timing adds CTA
barriers and must be compared with an uninstrumented run before interpreting
its overhead.

For sequential transport, nine exclusive phases partition each block's elapsed time: local routing,
send metadata, send payload, publication/fence, peer polling, receive metadata,
receive payload, ACK publication, and other control. A second view classifies
iterations with no cursor or frame progress as waiting for ready, credit, both,
or final drain. This view overlaps the phase totals and must not be added to
them. Init/final-reduction kernels require separate trace measurement. Raw GPU
timestamps are device-local; block times must not be summed as wall latency.
Use `benchmarks.replay --transport host_sm --profile-combine` to validate the
counter invariants and requested bytes against independently generated routes.

### Optional decode dispatch/combine diagnostics

Prepare fixed diagnostic storage before the first submission or Graph capture:

```python
ep.prepare_decode_profile(max_samples_per_channel=1024, stage="dispatch")
# stage="combine" (the default) samples fused mailbox combine instead.
# Optionally enable Graph mode, warm/capture buckets, and finish prior work.
ep.start_decode_profile()
run_workload(ep)
report = ep.stop_decode_profile()
```

The capture uses the actual decode transport kernel and works across dynamic routes,
Graph bucket changes and serialized eager/Graph execution. Each channel owns a
bounded sample sequence; overflow increments `dropped_samples` without overwriting
samples. A fresh start resets counters without recapturing the graphs. Start/stop
synchronize only at explicit, quiescent request boundaries; no other thread may
submit work during these boundaries. No per-layer CPU readback is introduced.

For the sequential/mailbox schema, `phase_fields` partition each sampled CTA's `elapsed_ns`. READY
polling is `peer_poll_ns`; `receive_payload_ns` includes reduction and output
stores; `idle_final_drain_ns` includes all final ACK checks, even successful first
checks. These are instrumented GPU durations, not PCIe wire times. This mailbox
schema does not use the pair-queue profiler's overlapping idle view. GPU clocks
are local, and concurrent CTA/rank durations must not be added as wall latency.
Normal contexts do not prepare this storage and launch an uninstrumented kernel
specialization. Use a separate ordinary context/process for headline performance.

With the decode dispatch pipeline enabled, the same API returns schema 3:
`role_phase_fields` describes TX, control warp 0, control warp 1, and RX separately.
TX partitions `tx_end_offset_ns`; each other role partitions its own end-minus-begin
offset. These timelines overlap and must never be summed into CTA wall time.
`phase_fields` is empty to prevent accidental sequential attribution. The report
separates descriptor allocation, IDs/weights, metadata publication, READY polling,
occupied-stage waiting, ACK publication/drain, RX payload and RX completion.
RX waiting after both control groups finish incoming work is classified separately
as `rx_ack_drain_wait_ns`. Pipeline-ineligible calls are not sampled.

Pipeline instrumentation adds no synchronization barriers and retains the original
schedule. It timestamps role leaders; payload instruction intervals exclude
other-thread stragglers subsequently observed at completion barriers. Prepared
but disabled counters still use the diagnostic specialization's resources, so
compare enabled/disabled runs and an ordinary context separately. Counter readback
occurs only after the request. See the architecture for interpretation limits.

## Validation

Pure routing, ring, configuration, adapter-contract, and public-content tests
run on a CPU-only machine. CUDA tests skip when CUDA/PyTorch is unavailable.

```sh
python -m pip install pytest
python -m pytest -q
```

On a CUDA development machine, first run native operator tests, then the
independent distributed replay:

```sh
python -m pytest tests/test_native_cuda.py tests/test_host_sm_native_cuda.py -q
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module benchmarks.replay --iterations 2 --check-handles
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module benchmarks.replay --iterations 2 --check-handles
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module benchmarks.replay --transport host_sm --channels 4 --require-channels 4 \
  --iterations 2 --check-handles
```

The replay exercises mixed and changing routes, duplicate destination ranks,
local-only routing, hotspots, empty ranks, all-empty operations, slot wrap,
independent outstanding handles, and exact numerical reference checks. Its
synthetic expert function is **not a model performance benchmark**. Timing
includes eager control overhead. See [full-model measurements](docs/model-performance.md)
for the separate SGLang comparison and its limits.

See [initial validation results](docs/validation.md) for completed two-rank and
eight-rank checks and their limits. Failure propagation can be exercised with:

```sh
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module benchmarks.failure_replay
```

Use [the SGLang integration instructions](adapters/sglang/README.md) for the
experimental framework adapter. Patch validation and mocked contracts do not
replace full-model validation.

## Design boundaries

Prefill and decode share routing and numerical semantics. Bounded decode can
select the mailbox combine kernel while prefill retains pair queues. Neither
path overlaps expert GEMM with partially returned chunks. Current options
reject unimplemented CPU reduce and Graph modes instead of silently falling back.

The host budget covers shared rings plus reusable private pinned packs across
the EP group. Device receive buffers, route metadata, allocator caches, and
framework allocations are additional. The initial library does not implement
training backward/autograd, cross-node communication, expert rebalancing, or
communication/GEMM tile scheduling.

See [architecture and ownership](docs/architecture.md) for the protocol,
numerical contract, and implementation limits.

## Public repository hygiene

Machine addresses, SSH settings, credentials, model locations, deployment
scripts, raw logs, build artifacts, and private experiment records belong
outside this repository. Example commands use operator-supplied paths.

```sh
git config user.name 'EP-Xpress Contributors'
git config user.email 'contributors@users.noreply.github.com'
git config core.hooksPath .githooks
python tools/check_public_content.py --worktree --staged --history
```

The hooks check staged contents and public commit identity before committing,
then check all reachable history before pushing, including content later
deleted. Optional organization-specific literal fragments can be supplied
through `EP_XPRESS_PRIVATE_PATTERNS_FILE`, pointing to a JSON file outside this
repository. Findings report rule names and paths without echoing secret values.
For persistent local checks, set `git config epxpress.privatePatternsFile
"$PRIVATE_PATTERN_FILE"`; this setting stays in local Git configuration and is
not published. An environment override takes precedence.

Dispatch diagnostics split GPU routing/local copies, send metadata, payload writes,
fence/READY publication, peer polling, receive metadata, payload reads, ACK
publication, final ACK checks and other control work. IDs/weights are counted
as receive metadata. Dispatch `idle_*` durations overlap its phase view; do not
add them to phase totals. Unlike combine, a prepared dispatch diagnostic kernel
still executes its diagnostic barriers while recording is disabled. Always use
a context without prepared diagnostics for performance comparisons.

Additional bounded-decode scheduling experiments (all default off):
`EP_XPRESS_DECODE_BALANCED_SEND=1` balances the concatenated live remote send
rows across channels; `EP_XPRESS_DECODE_SPLIT_TX_RX=1` uses independently resident
full CTAs for send and receive; `EP_XPRESS_DECODE_DEFER_RECLAIM=1` waits for slot
credit at reuse instead of draining outgoing ACKs before returning the result.
All require `EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32`; deferred reclamation also
requires split TX/RX. Healthy collective context close remains mandatory for
mapped-memory lifetime. See [the protocol and progress contract](docs/architecture.md#bounded-decode-combine-balanced-sends-and-independent-progress)
and [measured model results](docs/model-performance.md) before selecting a setting.
