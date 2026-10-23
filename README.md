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
- Bounded per-producer dispatch/combine rings, configurable slot count,
  immutable published chunks, generation IDs, idempotent consumer ACKs,
  backpressure, and simultaneous send/receive progress.
- Operation-scoped inverse-route handles, already-weighted rank partials,
  deterministic rank-order FP32 remote accumulation, and BF16 results.
- A dedicated Gloo control group with a bounded timeout. In `host_ce` it
  carries initialization and route metadata. In `host_sm` it is used for
  setup and shutdown only; routing IDs, weights, and receive counts stay on
  the GPU. Operation identity and capacity bounds are supplied by the caller.
- An experimental adapter patch for SGLang v0.5.21, kept separate from the core.

The `host_ce` reference may export unused rows within a nonempty source chunk.
Reported CE D2H bytes include those rows; ideal routing byte counts are not
reported as actual transfer volume. SM payload counters are not implemented.
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
The expert runner must support these masked rows. `wait()` inserts a CUDA event
dependency without waiting for GPU routing on the CPU. Without an explicit
bound, the context reserves `world_size * max_tokens_per_rank` rows, which can
be wasteful for decode. An undersized bound is an error, never silent truncation.
The SGLang adapter obtains its bound from existing scheduler metadata.

The SM backend batches routing and metadata access and assigns one CUDA block
to each independent communication channel. `channels` defaults to four; the
context negotiates a safe common limit during initialization and exposes it as
`ep.effective_channels`. Multiple channels require cooperative launch support
on every GPU. The group caps the requested count at the common occupancy
limit, falling back to one channel when cooperative launch is unavailable.
This changes parallelism, not the selected transport. Each channel
interleaves sends and receives, and every peer still acknowledges every chunk
header. These control costs remain measurable. The sole performance baseline
is SGLang's default NCCL backend; see [the protocol description](docs/architecture.md)
and [full-model results](docs/model-performance.md).

Each handle can be combined once and belongs to its context. Zero-input ranks
still receive work; zero-receive ranks still participate in combine. Healthy
`close()` is collective and must run before destroying the caller's process
groups. After failure, restart the worker group instead of reusing the context.

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
