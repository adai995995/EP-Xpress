# Host address and progress experiments

These build-time options are independent experiments. All default to zero.
They preserve BF16, current GPU routing on every invocation, rank deduplication,
one source payload write, and the existing deterministic combine reduction.
They do not enable GPU P2P, NVLink, CPU routing or CPU reduction.

| Option | Effect | Requirement |
| --- | --- | --- |
| `EP_XPRESS_DISPATCH_DIRECT=1` | Source slots and epoch/bitmap notification replace bounded dispatch descriptors | Indexed dispatch pipeline; control records disabled |
| `EP_XPRESS_DISPATCH_RETIRE=1` | Independent metadata and ACK warps | Direct slots |
| `EP_XPRESS_PACKET_PIPELINE=1` | Two packet descriptors, independent control/commit warp and 256 payload workers | Packet combine enabled |
| `EP_XPRESS_RING_SPLIT=1` | Independent producer/receiver CTAs for generic selective dispatch | Cooperative all-resident launch |

## Bounded dispatch addressing

For an agreed active channel count C, source token t belongs to channel t % C,
at slot floor(t / C). This rule does not depend on another rank's token count.
The allocated per-channel capacity remains ceil(max_tokens / C), at most 32
slots for this path. Ranks with fewer or zero tokens still publish the epoch.

The low 32 notification bits identify the current token slots requested by one
destination rank; the high bits contain the operation epoch. A zero bitmap is
a valid empty notification. The receiver computes the slot indices locally and
compacts its GPU output without fetching a host descriptor. Metadata and payload
addresses use the same source slot. Holes are neither written nor read. Expert
IDs and weights are written once per remotely routed token, not per destination.
The arena allocation and payload precision are unchanged.

ACK still protects the notification, metadata and payload until all readers have
finished. A new operation cannot overwrite the previous generation's storage.
Changing routes, active channels, graph buckets and eager fallback does not reuse
an old route plan. Larger operations retain the generic ring protocol.

## Dispatch role ownership

The optional independent-role RX CTA has two descriptor-control warps, two
metadata warps, one ACK warp and eight payload warps. Each control warp owns two
stages. The full 256-thread payload group is preserved.

Stage state is published with release ordering. The payload and metadata groups
fence their own accesses, join their lanes, then independently set completion
bits using shared-memory atomics. The ACK owner acquires both bits, publishes
one host ACK, then releases the stage to its descriptor owner. No host atomic
read-modify-write is used. No reader accesses a stage after releasing its bit.
The final outgoing ACK drain is retained.

## Packet and generic progress

Packet control polls up to 32 current tasks, prepares a free descriptor, and
commits completed descriptors while 256 other threads copy payload. Two local
descriptors permit the next task's readiness/credit work to overlap the current
copy. RX ACK follows the host read and HBM staging; the existing separate
rank-ordered reduction only reads private HBM. This does not promise concurrent
payload copies from two packets or unlimited lookahead.

The generic selective ring optionally launches C producer CTAs and C receiver
CTAs. Only producers own slot-reader masks and terminal publication. Only
receivers consume remote notifications and ACK payload/terminal generations.
The cooperative residency bound accounts for both roles, including C=1.
Receiving no longer waits for the same CTA's local send. Peer payload handling
within a receiver CTA remains sequential.

## Measurement boundaries

Use ordinary SGLang TTFT/Decode TPS and CUDA timelines for these ablations.
Sequential packet phase counters are rejected for the overlapping packet
pipeline; sequential generic-ring counters are rejected for split ring launches.
Dispatch's existing control/RX timelines do not separately time the new metadata
and ACK warps. Diagnostic recording remains separate from normal runs.
Do not interpret waiting intervals or logical payload rates as physical PCIe
bandwidth. Consult `model-performance.md` for measured configurations and limits.
