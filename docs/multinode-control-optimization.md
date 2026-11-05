# Direct GIN control experiments - 2026-11-05

All performance candidates from this round were reverted. Several improved
Decode throughput, but none established non-regression of ordinary 32-request
TTFT alongside that gain. The retained implementation keeps the original four
publication/reuse barriers and fixes a separate receive-count ownership race.
No hybrid forwarding or new selectable performance mode was added.

The [subsequent work-distribution profiling](multinode-work-balance.md) separates
context assignment, route volume and expert computation, and records the rejected
task-interleaving experiment. It retains the same production transport.

## Final complete-model verification

This is a new matched two-node allocation with eight NVIDIA H20D GPUs per host.
The NCCL baseline was rebuilt for it. Full Qwen3-30B-A3B-Instruct-2507 inference
used SGLang v0.5.21, TP = EP = attention DP = 16, BF16, Triton experts,
FlashInfer attention, full decode Graph buckets 1/2/4, and eager prefill.
Each request had 512 input tokens and 32 output tokens. Prefix caching was off.
Both backends disabled NVLink/GPU P2P paths and retained GDR over the same
RoCE NICs. See the [initial configuration details](multinode-performance.md#matched-configuration)
and [sanitized data](multinode-control-optimization.json) for versions, settings,
prompt hashes, source hashes, sample counts and observed ranges.

| Requests | Backend | Mean request TTFT (ms) | Steady Decode TPS |
| ---: | --- | ---: | ---: |
| 1 | Default NCCL | 140.29 | 52.50 |
| 1 | EP-Xpress before experiments | 103.32 | 68.08 |
| 1 | Retained EP-Xpress, count fix only | 100.56 | 68.07 |
| 32 | Default NCCL | 748.79 | 1512.75 |
| 32 | EP-Xpress before experiments | 457.62 | 2001.10 |
| 32 | Retained EP-Xpress, count fix only | 453.96 | 1999.38 |

The retained result is a verification of the original performance level, not
a new speedup claim. NCCL and the retained version have three measurements
after 16 warmup batches per workload. The original EP-Xpress reference has six
measurements from two independent server starts. All accepted timing batches
completed without another GPU workload. Background computation was stopped
for experiments and restored afterward.

TTFT includes admission, scheduling, prefill and first-token production.
Steady Decode TPS measures aggregate progress from the last first-token arrival
to the first request completion, with token-count interpolation. It is not
full-request throughput. Synthetic prompt IDs drive the real model router;
no route table is frozen or replayed as a substitute for inference.

## Rejected candidates

| Candidate | Single TTFT (ms) | Single Decode TPS | 32-request TTFT (ms) | 32-request Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Original reference | 103.32 | 68.08 | 457.62 | 2001.10 |
| Peer publication, fixed peer receive quotas | 258.98 | 83.69 | 951.52 | 2247.56 |
| Bounded batched posting, pooled reception | 170.42 | 78.53 | 745.09 | 2122.75 |
| Pooled reception, one QP per peer | 113.41 | 78.96 | 715.76 | 2308.82 |
| Active QP publication for dispatch and combine | 101.77 | 76.98 | 484.85 | 2217.94 |
| Active QP publication for combine only | 101.91 | 71.95 | 517.48 | 2121.57 |
| Active QP publication for dispatch only | 102.42 | 74.81 | 482.16 | 2162.00 |
| Parallel count reads and inline count sends | 101.42 | 69.98 | 481.68 | 2068.93 |

The both-phase active-QP result has six samples across two starts; the other
candidates have three. That candidate improved Decode TPS by about 13.1% for
one request and 10.8% for 32 requests, while ordinary 32-request mean TTFT was
about 6.0% higher. The observed TTFT ranges overlap, so these small samples do
not prove an intrinsic prefill-kernel regression. They also do not establish
the requested end-to-end non-regression; the candidate was not retained.

Normal prefill admission differed between runs. For example, the first original
32-request run grouped both requests together on 9/2/6 of the 16 ranks across
the three trials; the active-QP run did so on 3/12/16 ranks. Other ranks executed
two prefill batches. Comparing average prefill kernel durations without this
context would mix different token counts.

## Matched-admission diagnostic

A separate diagnostic held requests until each DP rank had its specified inputs,
and then resumed inference. Actual per-rank input membership and local token
shapes matched in all 12 paired cases. One warm repetition per shape was
excluded; the table averages the remaining three repetitions. Routing remained
dynamic. These release-relative times are not ordinary HTTP TTFT.

| Shape | Original release TTFT (ms) | Active-QP release TTFT (ms) | Original max-rank prefill GPU time (ms) | Active-QP max-rank prefill GPU time (ms) |
| --- | ---: | ---: | ---: | ---: |
| Two requests per rank, one prefill | 493.74 | 481.89 | 487.37 | 475.70 |
| Two requests per rank, split into two prefills | 415.79 | 419.76 | 480.71 | 482.67 |
| Uneven request placement, including idle ranks | 549.73 | 545.46 | 588.29 | 586.77 |

The fixed-shape results are broadly comparable and show why a larger ordinary
TTFT cannot be assigned entirely to transport. They are diagnostic evidence,
not a replacement for the normal-request acceptance criteria. An initial
admission probe failed on shared-file plan visibility after the ordinary
measurements had completed. Those diagnostic samples were excluded. The corrected
probe supplied the same plan at process startup and completed both versions.

## What profiling established

- In the original 32-request decode trace, the four barrier kernels averaged
  about 161.0 us per layer out of 221.6 us for the measured communication kernels.
  These intervals include payload completion and rank arrival skew; they are
  not 161 us of pure barrier-instruction cost. The both-phase active-QP version
  reduced the corresponding barrier total to 87.7 us and the communication
  kernel total to 186.5 us, with new publication/wait kernels included.
- Fixed receive quotas per peer can strand most of the receive grid when one
  producer supplies the bulk of the rows. A pooled receive row space avoids
  this scheduling restriction for arbitrary skewed routes.
- Collapsing each peer to one QP preserved simple publication ordering but
  reduced bulk posting parallelism. For the same single-request 512-token
  prefill, the active producer's mean dispatch-send kernel rose from about
  197.7 us to 807.7 us in the pooled single-QP candidate. Active-QP masks
  restored the multi-QP result to about 198.6 us.
- Reacquiring readiness in every combine payload tile was expensive. The
  active rank's single-prefill reduction interval, including its new waits,
  reached about 2.26 ms versus about 205 us for the original reduction alone.
  The original publication wait was a separate interval, so this is not a
  pure payload-bandwidth comparison. Acquiring completion once before pooled
  reduction restored the reduction kernel to about 204 us.
- Concurrent independent count-header reads reduced the decode receive-prefix
  kernel from about 11.0 us to 2.9 us. This isolated kernel win did not establish
  simultaneous improvement of both ordinary-request metrics.

All trace figures come from separate profiling runs, averaged per kernel per
rank and then across ranks unless an active rank is specified. They are not
summed over GPUs or substituted for unprofiled model latency. The 100 ms NIC
counter samples did not establish sustained port saturation, but CNP events
were observed and short bursts can be missed. Congestion has not been ruled out.

## Retained correctness fix and validation

After the entry barrier, a faster producer may publish its new receive count
before a slower receiver executes its local initialization. Clearing both send
and receive counts at that point can erase the new count. Initialization now
clears only locally owned send counts. Every remote producer publishes its
count, including zero; the existing publication barrier orders all receive
count and payload reads. No synchronization was removed by the retained fix.

The focused two-host EP16 check passed on all 16 ranks with hidden width 2050
and the optional sparse 257-token case. It verified exact BF16 results against
an independent reference, rank deduplication, arbitrary expert placement,
empty/nonempty transitions, two outstanding handles and eight Graph replays
with changing routes. Full-model verification also completed the untimed outputs
`2 + 2 = 4` and `Paris`. These are transport checks, not a model accuracy study.

## Implications for the next experiment

Direct GIN remains the multi-node mode. DeepEP's strong per-context publication,
signal shadows and parallel receive scheduling are useful design references,
but shortening a synchronization path must preserve bulk concurrency and pass
normal end-to-end tests. Removing buffer-reuse barriers requires an explicit
credit/ownership protocol with progress guarantees; simply deleting waits is
not valid. Before adding host-based hybrid forwarding, measure actual per-node
fan-out and network pressure at finer resolution and account for the extra
GPU-to-host-to-GPU hop. Hybrid forwarding is not an automatic solution to the
observed control waits.

References: [NCCL device API](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/deviceapi.html),
[DeepEP communication handle](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/comm/handle.cuh),
[DeepEP hybrid dispatch](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh).
No DeepEP kernels were copied into the repository. Machine identifiers, access
information, deployment paths, raw traces and rejected experimental source
remain outside the public repository.
