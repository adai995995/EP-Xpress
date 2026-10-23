# EP-Xpress

An independent expert-parallel communication library for single-node MoE
inference through host memory, without GPU peer access or NVLink payloads.

**Status: experimental BF16 eager implementation.** Two selectable transports
share the dispatch/combine interface: `host_ce` uses CPU routing and copy
engines; `host_sm` runs routing and mapped-host communication on the GPU.
Neither transport supports CUDA Graph capture. End-to-end performance must be
measured against the intended inference framework and workload.

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
  Prepared `host_sm` dispatch shares each producer's payload among its actual
  readers and reclaims it using only their ACKs. Combine uses independent
  queues for each directed rank pair and channel, with GPU owner row lists
  generated during dispatch.
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
combine launches one block per `(active channel, peer)`, including a local-only
block for the current rank. Four active channels on eight ranks therefore
launch four dispatch blocks and 32 combine blocks per GPU. Arena addresses and
frame sizes stay fixed when the active width changes. CUDA places those blocks
on SMs according to resource availability.

`channels` defaults to four. Initialization negotiates a common limit from
dispatch and combine kernel occupancy, including the optional diagnostic
kernel, and exposes it as `ep.effective_channels`. Multi-rank `host_sm`
requires cooperative launch support so the whole communication grid can be
resident. Each combine block progresses both directions of its rank pair:
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

The start call preallocates counters; the stop call bulk-copies them after one
transport-stream fence. No per-layer CPU readback or collective is added.
Excess operations run uninstrumented and increment `dropped_operations`.
`read_combine_profile()` returns the completed report without another fence.
Schema version 3 contains native field names, operation labels, and
`active_channels * world_size` rows per operation. Each operation records its
own `active_channels` and `counter_rows`, which map its raw rows to
`(channel, peer)` pairs, including local-only rows. Top-level `channels` and
`counter_rows` describe the allocation; inactive rows are omitted from each
operation's raw counters. The capture size must fit the 64 MiB combined GPU-counter and pinned-readout
budget; reduce `max_operations` for larger grids. Diagnostic timing adds CTA
barriers and must be compared with an uninstrumented run before interpreting
its overhead.

Nine exclusive phases partition each block's elapsed time: local routing,
send metadata, send payload, publication/fence, peer polling, receive metadata,
receive payload, ACK publication, and other control. A second view classifies
iterations with no cursor or frame progress as waiting for ready, credit, both,
or final drain. This view overlaps the phase totals and must not be added to
them. Init/final-reduction kernels require separate trace measurement. Raw GPU
timestamps are device-local; block times must not be summed as wall latency.
Use `benchmarks.replay --transport host_sm --profile-combine` to validate the
counter invariants and requested bytes against independently generated routes.

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

Prefill and decode share routing and combine semantics. Future implementations
may select different kernels and progress strategies without changing those
semantics. Current options deliberately reject unimplemented CPU reduce and
Graph modes instead of silently falling back.

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
