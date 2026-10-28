# Host read progress experiments

These build-time options default to `0` and accept only `0` or `1`. Set them
before constructing the native extension and its contexts; every rank must use
the same native build. They add no CPU routing or per-invocation environment
inspection to the transport path.

| Option | Scope | Requirement |
| --- | --- | --- |
| `EP_XPRESS_ROUTE_PACKED` | Compact shared metadata in bounded dispatch | `EP_XPRESS_DISPATCH_DIRECT=1` |
| `EP_XPRESS_RX_FENCE` | Narrow receive-worker fence scope at audited pipeline sites | Existing group joins and completion handoffs retained |
| `EP_XPRESS_READY_BATCH` | Up to two ready payload tasks per full-width receive group | Direct dispatch and packet pipeline enabled |
| `EP_XPRESS_PACKET_RECV_LOCAL` | Receiver-owned host storage for packet combine | Packet transport enabled (`EP_XPRESS_DECODE_PACKET_ROWS>0`) |

The existing requirements for direct dispatch and packet pipelining still apply;
see [host address and progress experiments](host-address-pipeline.md). Routing is
rebuilt from the current GPU inputs on every invocation. These options preserve
BF16 payloads, exact FP32 route weights, rank deduplication, necessary ACKs and
the separate deterministic rank-order combine reduction.

## Shared route words

Packed dispatch stores each route as one 64-bit source-owned word: the low
32 bits contain the signed global expert ID and the high 32 bits contain the
FP32 weight bits. All destination readers use the same source copy. There is no
per-destination metadata replication or conversion to local expert IDs on the
sender. Receivers perform their existing local expert mapping.

The public input may still use int32 or int64 IDs. Prepared transport already
limits the expert-table size to `INT32_MAX`, and route validation checks the
original input ID before narrowing. Thus every valid expert ID fits; `-1`
padding retains its meaning. The arena allocation and offsets are unchanged:
the old ID region holds the packed words and the separate weight region is
unused for this path. Unbounded dispatch retains its existing format.

## Receive completion

The receive-fence option replaces per-worker system fences with block fences
only at the audited dispatch metadata/payload and packet-pipeline receive
completion sites. Transmit system fences, group joins, shared completion
release/acquire and system-scope ACK publication remain. The generic ring is
outside this change. See the [ordering argument and preconditions](host-rx-ordering.md)
before extending this option to another transfer mechanism or consumer.

## Ready payload batches

Bounded dispatch and packet-pipeline receive workers may select up to two stages
that are already READY. The complete 256-thread payload group issues independent
loads from both selected tasks before consuming their values and writing HBM.
With only one ready task, it uses the existing single-task copy; it never waits
for a second task to fill a batch. Transmit scheduling is unchanged.

The value buffer is bounded. Stage metadata remains immutable until all selected
reads and local writes have joined, after which each stage is released through
its own existing completion/ACK protocol. No worker references a stage after
releasing it. A slower second task can delay completion of the pair; batching
does not promise a benefit for every route density or NUMA topology.

## Receiver-owned combine slots

For a logical packet `(producer, destination, tile, bank)`, receiver-local mode
uses the destination's host arena and the slot indexed by `(producer, tile,
bank)`. Both endpoints derive that same address. The sender's generation table
continues to track its logical destination/tile/bank, so changing active channel
counts or Graph bucket sizes does not redefine credit ownership.

Arena sizes, bank count and matching-epoch checks are unchanged. No CPU relay or
additional payload copy is introduced. Physical NUMA locality requires the
existing per-rank arena placement to be configured and verified; the flag changes
slot ownership, not the operating system's page policy. Arena-level error headers
remain rank-owned and outside the packet slots. Dispatch keeps its source-owned
multicast layout.

Consult [model performance](model-performance.md) for separately measured
configurations, correctness coverage and workload limits. Logical payload rates
and code-level concurrency are not measurements of physical PCIe utilization.
