# Direct GIN work distribution profiling

This follow-up measured real-model route distributions, context assignment and
barrier intervals. Interleaving send tasks across contexts did not establish an
end-to-end improvement and was reverted. The production transport remains the
count-corrected implementation at `7dba5f8`; no new runtime mode was added.

## Matched complete-model comparison

The configuration matches the [preceding experiment](multinode-control-optimization.md):
two hosts with eight H20D GPUs each, Qwen3-30B-A3B-Instruct-2507, SGLang v0.5.21,
BF16, TP = EP = attention DP = 16, Triton experts, full decode CUDA Graph,
512 input tokens and 32 output tokens. NVLink/GPU P2P paths were disabled;
GDR remained enabled. Real model routing was recomputed at every step.

The default NCCL and retained-reference measurements are reused from the same
allocation and settings. Each row is the arithmetic mean of three batches after
16 warmups per workload. Decode TPS is the aggregate steady output rate, not
full-request throughput; TTFT includes admission and scheduling.

| Concurrent requests | Backend | Mean request TTFT (ms) | Decode TPS |
| ---: | --- | ---: | ---: |
| 1 | Archived default NCCL | 140.29 | 52.50 |
| 1 | Retained EP-Xpress reference | 100.56 | 68.07 |
| 1 | Task interleaving, rejected | 99.27 | 68.20 |
| 32 | Archived default NCCL | 748.79 | 1512.75 |
| 32 | Retained EP-Xpress reference | 453.96 | 1999.38 |
| 32 | Task interleaving, rejected | 487.94 | 2016.20 |

The candidate distributed dispatch token/peer pairs across CTAs, rotating peer
order per token to preserve multiple contexts per peer for larger batches.
Combine rows were interleaved across CTAs instead of assigning the first 32
rows to context zero. Receive parallelism, barriers, weights and rank-ordered
FP32 combine were unchanged. This was a general task assignment rule rather
than a special route or model-shape case.

Decode improved only about 0.2% and 0.8%, while 32-request mean TTFT increased
about 7.5%. The observed TTFT ranges overlap (reference 422.62–484.01 ms,
candidate 467.04–503.13 ms), and admission can differ, so these samples do not
prove an intrinsic kernel regression. They do not establish the required
end-to-end benefit either. The change was reverted, without adding a switch
for the rejected mapping. All six timed batches and the two untimed output
checks completed before access to one host was lost during cleanup.

## What the separate diagnostic run showed

The diagnostic version records GPU-resident counters during real inference,
including Graph replay. Host export occurs after the profiler stops. It is not
the version used for the timing table. Additional statistics kernels and timing
instructions perturb execution, so diagnostic times must not be treated as
unprofiled performance or exact network RTT.

| Workload and phase | Dispatch remote data across QP 0/1/2/3 | Combine remote data across QP 0/1/2/3 |
| --- | --- | --- |
| Single-request prefill | Approximately 25% each | 31.4% / 27.4% / 22.6% / 18.6% |
| Single-request decode | 100% / 0% / 0% / 0% | 100% / 0% / 0% / 0% |
| 32-request prefill | Approximately 25% each | Approximately 25% each |
| 32-request decode | 49.9% / 50.1% / 0% / 0% | 100% / 0% / 0% / 0% |

Thus context skew is real, but correcting it alone was not enough to improve
the model. The counters also distinguish other kinds of imbalance:

- At 32-request decode, the busiest rank received about 1.38 times the mean
  remote dispatch rows. Sending ranks varied less: the maximum was about
  1.05 times the mean. These are byte-volume indicators for equal-size rows,
  not measurements of congestion or available NIC bandwidth.
- The busiest expert had about 3.51 times the mean token selections. This is
  different from deduplicated rank traffic. The two expert GEMM kernels together
  averaged 31.34 us per layer across ranks, with rank means ranging from 28.03
  to 34.11 us. Routing, scatter, activation and attention are excluded.
- In single-request inference only one DP rank originated input. Concentrated
  combine reception is consequently expected; it cannot be made independent
  of the final destination merely by changing CTA assignment.

The 32-request decode barrier measurements were:

| Interval | Mean CTA interval (us) | Mean slowest notification submission (us) | Mean slowest reciprocal wait (us) |
| --- | ---: | ---: | ---: |
| Dispatch reuse | 37.16 | 4.63 | 32.39 |
| Dispatch publication | 35.47 | 3.59 | 31.66 |
| Combine reuse | 46.37 | 3.72 | 42.44 |
| Combine publication | 37.97 | 3.59 | 34.17 |

The four intervals total about 157 us per layer on this diagnostic. Maxima can
belong to different threads, so the last two columns are not an additive
breakdown. Waits include rank arrival skew, remote operation visibility and
control progress. They are not all removable synchronization overhead.
The current protocol has batch-wide publication/reuse barriers, not independent
per-peer credits or per-token ACK queues.

The recorded dispatch traffic matrix equals the transpose of combine traffic,
and sums of per-context counts match remote row totals. The profile contains
48 prefill layer calls and 1536 decode layer calls per rank in each workload;
each layer has two observed expert GEMM kernels. These are actual executed
calls, which need not equal the number of output tokens requested by the client.

## Follow-up boundary

The next isolated candidate distinguishes data publication from reader
quiescence. Publication must continue to order every data context. After that
publication has completed, reuse needs confirmation that prior readers have
finished, which may use a single control context. A further candidate uses
producer-owned mapped host records for local control while retaining GIN for
remote notifications; it posts remote notifications before polling local peers.

Both candidates compiled against the installed headers/runtime, but neither
has completed two-host correctness or model-performance validation. They remain
private experimental sources and are not shipped. Per-peer buffer reclamation,
shared ready-task scheduling and hybrid forwarding require subsequent validation;
this report does not claim those features are implemented.

The [sanitized data](multinode-work-balance.json) records distributions, per-rank
expert timings, arithmetic means, ranges, sample counts and source/prompt hashes.
Raw traces, access information, machine identifiers, deployment paths and rejected
experimental implementations remain outside the public repository.
