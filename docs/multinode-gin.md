# Direct GIN across hosts

`Config(transport="host_sm", internode_transport="gin", ...)` selects an
experimental BF16 multi-node implementation. Local peers exchange payloads
through GPU-mapped host memory. Remote peers exchange GPU-resident registered
buffers using NCCL GDAKI. No GPU peer pointer or NVLink payload is used by these
kernels. The single-node transport remains the default.

The initial [two-node EP16 validation and model results](multinode-performance.md)
cover dynamic routing, CUDA Graph replay and a matched default-NCCL comparison.

The caller supplies its initialized `NCCLCommHandle`. EP-Xpress creates a device
communication resource on that handle; it does not create another NCCL or Gloo
process group. The library retains the communicator owner. Close all contexts
collectively before the application destroys its communicator.

## Data path and ownership

1. GPU routing deduplicates each token's target ranks. The core supports an
   arbitrary expert-to-global-rank map, including ranks without experts. A
   destination receives one activation row even when multiple selected experts
   reside there. Local expert IDs retain the input top-k slot ordering.
2. A producer writes each local multicast activation once into its host arena.
   Local readers independently read that row through PCIe. Each remote
   destination gets one GIN put from the registered GPU record. Remote network
   multicast is **not** implemented; deduplication is per destination rank.
3. Remote counts, receive prefixes, valid row count and inverse routing stay on
   GPU. Rows are compacted into caller-bounded output storage. Unused rows have
   IDs `-1`, zero weights and zero activations. Dynamic route values are read
   anew on every dispatch, including every CUDA Graph replay.
4. Experts produce one already-weighted BF16 partial per received row. Combine
   writes the partial to the original token's slot: host memory for local
   origins, GDR for remote origins. Source rank/token indices come from the
   dispatch handle, rather than another route exchange.
5. The origin accumulates expected contributors in ascending global-rank order
   using FP32, then rounds once to BF16. Weights are not applied a second time.

The initial protocol uses **batch completion barriers**, not per-token ACKs.
The entry barrier establishes that prior readers are finished before a staging
buffer is overwritten. The publication barrier establishes that both payloads
and counts are visible before receiving/reducing. The barriers cover every GIN
context; a signal on context 0 alone cannot order puts on another context/QP.
Host writers execute system-scope fences before publication. Receivers use
volatile loads from mapped host storage.

These are full-batch barriers across the EP group. This implementation does not
yet overlap expert computation with remote receives, use per-peer credits, or
remove the batch straggler dependency. It establishes a functional direct
transport and a measurable baseline, not a claim of parity with DeepEP's tuned
LL/HT kernels. The existing single-node ring/packet tuning knobs do not select
different algorithms inside this multi-node path.

## Runtime and resource requirements

- Linux, CUDA toolkit, BF16-capable NVIDIA GPUs and a GDAKI-capable IB/RoCE
  fabric with GDR. Direct GIN requires full connectivity across selected NICs.
- NCCL device headers and the runtime must match exactly and be at least
  2.30.7. The tested runtime is 2.30.7. Other versions are not validated by that
  result. `EP_XPRESS_NCCL_ROOT` may identify the installed NCCL include/lib root.
- The extension verifies that it resolves the same NCCL library as the borrowed
  communicator. It fails if GDAKI is unavailable; it does not substitute TCP or
  CPU proxy payload movement.
- The entire EP group currently has 2–256 ranks. Each rank owns one distinct
  GPU. The global token capacity must fit int32. All ranks submit phases and
  graph buckets in the same order.
- Containers on each host share `/dev/shm` (or an explicitly shared local
  directory), have sufficient locked-memory limits, and expose the RDMA devices.
  Apply the intended CPU NUMA memory policy before constructing the context:
  each producer first-touches its own pages.

The transport reserves full-batch staging. This differs from the single-node
bounded rings. With `P` global ranks, `L` ranks on this host, `T` source token
capacity, hidden width `H`, and top-k `K`, a packed record is
`align128(2*H + 8*K + 4)` bytes. Dispatch reserves one TX and `P` RX partitions;
combine reserves `P*T` TX and `P*T` RX rows per GPU. A host producer reserves
`T` dispatch records and `L*T` combine rows. Node host memory is the sum over
its `L` producers. No activation buffer is allocated on the CPU per operation.

Inspect the exact rounded budgets before allocating:

```python
config.gin_registered_bytes(world_size=16)  # per GPU, not model/expert workspace
config.pool_bytes(world_size=16, local_world_size=8)  # per host
```

`gin_gpu_budget_bytes` bounds registered staging per GPU;
`host_pool_budget_bytes` bounds all producer arenas on that host. Initialization
checks both budgets and the native layout. `gin_contexts` controls the number
of network contexts requested, not the number of SMs. `transport_info` reports
the actual allocation. Receive capacity must cover every token/rank pair that
can reach that rank; capacity overflow traps rather than corrupting storage.

## API and CUDA Graphs

Use the existing dispatch/combine interface. Pass `recv_capacity` using a
CPU-known upper bound; the exact count is `batch.valid_count`, an int32 CUDA
tensor. Supply BF16 hidden/partial tensors, int32 or int64 top-k IDs, and FP32
weights. IDs `-1` denote inactive slots. Nonfinite weights or invalid IDs fail
the routing contract. Autograd is not supported.

Borrowed workspaces are valid until their next dispatch. Enqueue all consumers
on the combine/wait stream before reuse, or retain independent storage. Keep
handles distinct for concurrent outstanding operations; each handle is consumed
once. The context serializes its transport staging across streams.

For Graphs, call `enable_cuda_graph()` before the first operation. Warm fixed
workspaces, call `prepare_graph_capture()`, and capture the direct-stream API.
Call `prepare_graph_replay()` immediately before input updates and each replay.
Keep the context and workspaces alive and serialize graphs sharing a context.
Routing, counts and network barrier generations are device state, not captured
CPU route values. Host-ring diagnostic/profiling APIs are not implemented for
this transport. `Config.graph_mode` retains its existing eager-only setting;
Graph capture is enabled through this explicit API.

## SGLang integration and comparison

Use the pinned adapter and `--moe-a2a-backend ep_xpress`. Set:

```sh
export EP_XPRESS_TRANSPORT=host_sm
export EP_XPRESS_INTERNODE_TRANSPORT=gin
export EP_XPRESS_MAX_TOKENS=4096
export EP_XPRESS_HOST_BUDGET_BYTES=4294967296
export EP_XPRESS_GIN_GPU_BUDGET_BYTES=2147483648
export EP_XPRESS_GIN_CONTEXTS=4
export EP_XPRESS_CUDA_GRAPH=1
export NCCL_CUMEM_ENABLE=1
```

Those capacities are examples, not universal model sizing. The adapter still
requires attention DP = TP = EP, contiguous expert partitioning, Triton BF16
experts, and the other restrictions in the adapter README. Launch the same
model on every node with SGLang's `--nnodes`, `--node-rank`, and
`--dist-init-addr`; TP/EP/DP sizes describe the global group.

Set `NCCL_CUMEM_ENABLE=1` **before** SGLang initializes the supplied communicator;
the framework otherwise defaults this to 0 for this backend. It enables the
VMM allocations required for GIN symmetric windows, not GPU P2P payloads.
Use the same setting in the NCCL baseline. Changing it after the communicator
exists cannot retroactively enable device API support.

For both candidate and default NCCL baseline, set `NCCL_P2P_DISABLE=1`,
`NCCL_NVLS_ENABLE=0`, `NCCL_MNNVL_ENABLE=0`, `NCCL_COLLNET_ENABLE=0`,
`NCCL_IB_DISABLE=0`, and `NCCL_SHM_DISABLE=0`. Keep GDR enabled and select the
same working NICs, socket interface and GDR policy. GIN requires
`NCCL_IB_MERGE_NICS=0`. Disable SGLang custom all-reduce/FlashInfer all-reduce
fusion as well so they do not bypass NCCL's P2P controls. Environment controls
select communication paths; they do not physically power down NVLink.

The default NCCL baseline is `--moe-a2a-backend none` in the pinned adapter.
Keep hardware, topology, framework, dtype, expert runner, Graph buckets,
requests, actual scheduler batches and warmup coverage matched. A single-node
archived NCCL result is not a valid baseline for this multi-node configuration.
Report TTFT separately from steady Decode TPS and exclude compilation/warmup.

The focused distributed check is `tests/gin_smoke.py`, launched with `torchrun`
on at least two hosts. `--graph` also checks changing routing values across
replays. It validates exact dispatch destinations/weights, same-rank dedup,
empty and skewed routes, eager reuse and ordered FP32 combine against a reference.
It is not an end-to-end performance measurement.

References: [NCCL 2.30.7 device API](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/deviceapi.html)
and [DeepEP](https://github.com/deepseek-ai/DeepEP). This implementation uses
the NCCL device API directly and does not vendor DeepEP kernels.
