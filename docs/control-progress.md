# Cached readiness and receive polling

These build-time experiments default to zero and accept only `0` or `1`.
They apply to the bounded dispatch and packet-combine pipelines. The generic
ring keeps its existing behavior. All ranks must use the same native build.

| Option | Effect |
| --- | --- |
| `EP_XPRESS_READY_CACHE` | Retain and prioritize confirmed dispatch READY observations until the current task is consumed |
| `EP_XPRESS_PACKET_READY_CACHE` | Independently enable the same scheduling for packet READY and previous-bank TX credit |
| `EP_XPRESS_RX_POLL_WARP` | Let the first complete receive-worker warp retry empty shared-stage scans while the other payload warps wait at the handoff |

These options do not cache routing across invocations, change BF16 precision,
remove rank deduplication, or reduce the 256-thread payload group. They add no
CPU inspection or communication to the per-layer path. See
[model performance](model-performance.md) for separately measured configurations.

## Readiness belongs to one task and generation

Dispatch assigns one pending source peer to each control lane. A successful
system-scope acquire supplies that epoch's notification, including its current
bitmap. With caching enabled, a lane retains this value even when another ready
peer is selected first. Before issuing new host reads, the whole control warp
checks for confirmed pending tasks. If any exist, it consumes one without
waiting for unconfirmed peers to load host memory. Only an empty confirmed set
triggers parallel polling of the remaining pending peers. Empty notifications
remain meaningful and still receive ACKs.

The pending set is finite: consuming confirmed tasks removes them, so this
priority cannot indefinitely starve an unconfirmed peer. It does not wait for a
whole peer group to become ready. The initial acquire and all stage retirement
work remain mandatory.

The producer cannot overwrite the corresponding slot before this receiver's
ACK. Therefore a confirmed notification remains valid until it is consumed.
The cache is a kernel-local register: Graph replay, a different handle and the
next routing invocation all start with an unconfirmed observation. No token ID,
route bitmap or weight from a previous invocation is reused.

When `EP_XPRESS_PACKET_READY_CACHE=1`, packet combine applies the same rule
to READY on RX and the previous bank's credit on TX. The separate switch allows
dispatch scheduling to be measured without changing packet scheduling. Each lane clears its cached availability when it loads a new batch
of task descriptors, including batches after the first 32 lane-owned tasks.
The selected stage already owns copies of its task, mask and slot pointer, so
new lane descriptors cannot change an outstanding stage. Different task IDs
must continue to identify distinct slots within a generation.

The ordering chain remains: system acquire, control-warp handoff, shared-stage
release, receive acquire/group handoff, payload completion, existing join and
ACK publication. A cached observation never substitutes for the first acquire. Packet task
submission explicitly joins the control warp after selecting the acquiring
lane and before lane zero publishes the shared stage. The indexed dispatch
path likewise joins before other lanes read host descriptors; direct slots
already join when building their shared row indices. Ballot and shuffle alone
are not used as a memory-ordering bridge. Empty polls add no such barrier.
This follows the distinction between register exchange and memory ordering in
[NVIDIA’s warp primitive documentation](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

## Park idle receive warps at the handoff

Previously an empty shared-stage scan took the entire payload group through
repeated named barriers. With warp polling enabled, only the first full
32-thread receive warp repeats that scan. Lane zero selects work; a full-mask
shuffle gives all 32 lanes the same continue/exit decision. The other seven
warps wait at the existing 256-thread handoff barrier.

Ready work, completion or error releases the polling warp into that handoff.
All 256 workers then observe the selected stage and participate in the ordinary
payload copy. The completion fence/join, stage DONE publication and the second
join protecting stage reuse are unchanged. TX retains its existing polling
behavior. Control, metadata and ACK warps do not wait on the payload-group
barrier and can continue making progress while the payload workers are parked.

The optional two-ready-task selection and independent metadata completion use
their existing state predicates. An empty scan alone is never a completion
condition. Code review of these combinations is distinct from GPU validation
of a particular selected configuration.

## Diagnostic interpretation

Dispatch `poll_rounds` continues to count scheduling/polling rounds; with cached
readiness it is not a count of host loads. Role counters describe overlapping
timelines and must not be summed as a single kernel latency. Waiting for a
shared stage includes upstream readiness and descriptor supply; it is not a
measurement of physical PCIe transfer time. Neither option promises a gain
when independent ready tasks or empty polling rounds are uncommon.
