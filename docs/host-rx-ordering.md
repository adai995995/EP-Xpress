# Receive completion and host-slot reuse

`EP_XPRESS_RX_FENCE=1` is a receive-only fence-scope experiment. It replaces the
per-worker system fence with a block fence at the audited completion sites.
The default retains the original system fence. This note records the protocol
argument; performance and CUDA validation are recorded separately.

## Scope

The audited sites are bounded dispatch payload completion, bounded dispatch
metadata completion, and packet-pipeline receive completion. They use ordinary
synchronous `.cv` loads from mapped host memory and ordinary stores to this
GPU's output/workspace. The named 256-thread payload barrier, metadata warp
joins, shared-stage release/acquire operations and system-scope ACK publication
remain unchanged. The generic ring is outside this experiment.

Transmit workers still execute the system fence before publishing READY. Error
publication retains its existing fences. The change does not enable relaxed
READY or ACK loads/stores, reuse on error, or host atomic read-modify-write.

## Ordering argument

For every receive worker, the relevant events are:

```text
host payload/metadata load
    -> local HBM store and block fence
    -> payload-group barrier or metadata-warp join
    -> leader releases shared completion
    -> ACK owner acquires shared completion
    -> ACK owner stores ACK with release.sys
    -> producer loads matching ACK with acquire.sys
    -> producer may overwrite that host generation
```

The payload group retains all 256 participating threads. Its named barrier
joins every worker before the leader marks completion. The block fence is not
a replacement for that join. The shared-state release/acquire pair then passes
completion to the control owner, which performs the existing system-scope ACK.
Thus the producer's reuse is ordered after every worker's host reads, although
those workers do not individually publish anything to a different GPU.

The ordinary metadata path joins all lanes before its control warp returns to
the stage-retirement loop. The optional independent metadata path releases its
completion bit only after that join. Its ACK owner requires both metadata and
payload bits. A completed payload alone never authorizes reuse while metadata
is still being read. No participant accesses that stage generation after
releasing its completion.

The receive-side HBM writes have a different consumer: later kernels on the
ordered execution stream. Transport kernel completion, and the existing event
ordering when streams change, protect that consumer. Publishing a host ACK
does not declare the final combine reduction complete; it only releases the
source host storage. Packet reduction still reads the private HBM copy in its
existing deterministic rank order.

This argument uses PTX's transitive synchronization rules. The barrier orders
ordinary memory operations within its participants; CTA-scope stage handoff
and system-scope ACK handoff connect those participants to the producer.
See [PTX memory synchronization and causality](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-synchronization),
[named barriers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar),
and [warp barriers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync).

## Preconditions that must remain true

- Every expected payload/metadata reader participates in its existing join.
- The ACK owner waits for every relevant completion, including both dispatch
  roles when independent metadata processing is enabled.
- Shared-stage lifetime and host generation/credit checks are unchanged.
- Receiving uses ordinary loads whose values are consumed before completion.
  Future `cp.async`, TMA, copy-engine or other asynchronous transfers require
  their own completion waits; a block barrier alone does not complete them.
- Output buffers and handles retain their existing stream/lifetime ownership.
  No new concurrent consumer may read the output before transport completion.
- READY/ACK cache and scope semantics are unchanged. A fence is not a cache
  invalidation substitute; the existing host-load cache policy remains active.

## Focused validation

Exercise vector and scalar payloads, changing and empty routes, integer ID
widths, repeated host-generation reuse, Graph bucket switches, eager fallback,
and multiple outstanding handles. Include independent metadata completion if
that option is combined with this experiment. Compare exact routing/weights and
combined outputs with the existing reference. These checks can expose mistakes
in the implementation; they do not replace the ordering argument above.
