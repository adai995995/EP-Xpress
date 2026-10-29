# Full-model inference comparison

These experiments run the complete Qwen3-30B-A3B-Instruct-2507 model through
SGLang v0.5.21. They are distinct from the synthetic expert function in the
standalone transport replay.

The latest investigation is in
[independent dispatch scheduling and host access width](#independent-dispatch-scheduling-and-host-access-width).
Earlier sections describe earlier runtime revisions and their measurements.

## Matched configuration

- Eight H20D GPUs on one host; BF16 unquantized model weights and activations.
- TP = EP = attention DP = 8, with DP LM head, Triton experts, and FlashInfer
  attention. Prefill and decode CUDA Graph are both disabled.
- Python 3.12, PyTorch 2.13.0+cu130, CUDA toolkit 13.0.
- SGLang baseline uses `--moe-a2a-backend none`; EP-Xpress uses `ep_xpress`.
  The baseline includes SGLang's native gather/reduction path and remaining
  framework collectives; it is not a standalone NCCL all-to-all benchmark.
- GPU peer transport, NVLS, MNNVL, CollNet, and IB are disabled. Custom
  all-reduce and FlashInfer all-reduce fusion are disabled in both runs.
  NCCL logs identify shared-memory transport, and hardware NVLink counters
  show zero payload increments over each measured run.
- Synthetic token-ID requests: 512 input tokens and 32 generated tokens,
  greedy decoding, ignored EOS, disabled radix cache. Request hashes match
  across configurations. Concurrent batch sizes are 1, 8, and 32.
- The initial channel experiments use eight warmup batches generating four
  tokens, then three measured batches generating 32 tokens. The later rank-pair
  comparison uses 16 output tokens per warmup batch in both backends. Values
  below are medians unless specified otherwise.

These controls establish a comparison under eager execution. They do not
represent the fastest available SGLang configuration or a CUDA Graph baseline.
Identical inputs do not by themselves guarantee identical later expert routes.

## Performance baseline

The sole performance baseline is SGLang's default NCCL backend path,
`--moe-a2a-backend none`. The candidate is EP-Xpress with
`--moe-a2a-backend ep_xpress` and `EP_XPRESS_TRANSPORT=host_sm`.
The library's `host_ce` implementation is retained for functional regression
testing; it is not a performance comparator.

## Profiling and parallel implementation

Before changing the runtime, a separate full-model trace reproduced the slow
single-block implementation. On the active rank at batch 1, the prefill GPU
forward window was 1,799.408 ms: dispatch occupied 458.054 ms and combine
1,327.937 ms, together 99.25% of the window. At batch 32, the corresponding
per-rank medians were 25,555.241 ms for the forward window, 9,583.585 ms for
dispatch, and 15,922.336 ms for combine; the median path time fraction was 99.73%.

Those monolithic kernels included routing, host loads/stores, initialization,
reduction, and peer waiting. Their duration is not pure PCIe transfer time.
The trace localizes the bottleneck to this path; it does not separately measure
the contribution of every serial instruction sequence or PCIe link saturation.

Runtime commit `215eb33` implements:

- Batched GPU route validation, destination masks, prefix calculations, and
  compacted copies, with up to 256 rows per work batch.
- Separate host arrays for masks, token indices, IDs, weights, and payloads,
  allowing parallel metadata reads rather than a host read and block barrier
  for every token.
- Multiple independent communication channels, each progressed by one block
  interleaving bounded sends and receives. Multi-block operation uses a
  cooperatively resident grid, bounded by every participating device's capacity.
- Parallel initialization and final deterministic combine reduction outside
  the communication grid, plus direct int32/int64 routing-ID support.

There is still no per-layer Gloo metadata collective or GPU routing-count
readback. CPU-side launch/handle management and caller-supplied capacity bounds
remain. The default requests four channels; the model experiment verified
`effective=4` on all eight ranks. These changes were measured together and are
not an ablation of the individual techniques.

## Four-channel full-model results

Measured on 2026-10-23. Both backends were run afresh with the matched settings
above; configuration, input-hash, hardware, source, and transport checks passed.
The candidate used runtime commit `215eb33`. It remains slower than NCCL.

| Concurrent requests | NCCL batch latency (s) | EP-Xpress batch latency (s) | EP-Xpress / NCCL | NCCL output tokens/s | EP-Xpress output tokens/s |
|---|---:|---:|---:|---:|---:|
| 1 | 3.055 | 3.344 | 1.095 | 10.474 | 9.568 |
| 8 | 2.990 | 4.282 | 1.432 | 85.629 | 59.779 |
| 32 | 3.401 | 6.850 | 2.014 | 301.066 | 149.496 |

Observed batch-latency ranges over the three repetitions were 3.034–3.091,
2.982–3.052, and 3.397–3.404 seconds for NCCL; and 3.316–3.391,
4.276–4.316, and 6.831–6.981 seconds for EP-Xpress. These are observed
ranges, not confidence intervals.

| Concurrent requests | NCCL mean TTFT (ms) | EP-Xpress mean TTFT (ms) | NCCL last TTFT (ms) | EP-Xpress last TTFT (ms) |
|---|---:|---:|---:|---:|
| 1 | 190.013 | 329.123 | 190.013 | 329.123 |
| 8 | 210.318 | 1202.257 | 211.096 | 1203.032 |
| 32 | 545.603 | 2969.157 | 595.878 | 3707.160 |

| Concurrent requests | NCCL per-request TPOT (ms) | EP-Xpress per-request TPOT (ms) | NCCL steady-window TPOT (ms) | EP-Xpress steady-window TPOT (ms) |
|---|---:|---:|---:|---:|
| 1 | 92.424 | 97.265 | 92.424 | 97.265 |
| 8 | 89.390 | 99.656 | 89.390 | 99.656 |
| 32 | 92.188 | 125.146 | 90.896 | 102.398 |

Batch latency is HTTP-client time until every request finishes. Mean TTFT is
the mean across requests in each repetition; last TTFT is when the final
request receives its first token. Per-request TPOT averages each request's
time from its first to last token divided by the subsequent token count.
Steady-window TPOT uses only the interval from the final first token to the
first completed request, normalized by concurrent requests and tokens emitted
in that window. It excludes the staggered-start period and should not replace
per-request TPOT when assessing user latency. Each table reports the median
of its respective per-repetition statistic.

For implementation history only, the preceding single-block runtime had
batch latencies of 4.861, 10.759, and 29.570 seconds at the same three shapes.
Four channels with the batched implementation reduce those latencies by about
31.2%, 60.2%, and 76.8%. This is an improvement over the previous implementation;
the performance baseline remains the freshly measured NCCL path above.

## Four-channel timeline after optimization

Profiling was collected separately after unprofiled performance measurement,
using the same 512-token inputs and eight output tokens. GPU work is assigned
to forwards through runtime launch correlation and checked against exported
GPU forward annotations. All traces resolved. The denominator is the union of
each rank's first-to-last GPU operation windows for the indicated phase, not
HTTP duration or CPU annotation time. IDLE ranks are reported separately;
their collective waiting is not assigned an inferred global phase.

| Concurrent requests | Prefill NCCL kernel union (ms) | Prefill EP path kernel union (ms) | NCCL window time fraction | EP path window time fraction |
|---|---:|---:|---:|---:|
| 1 | 17.104 | 202.066 | 15.27% | 92.95% |
| 8 | 85.117 | 748.905 | 72.89% | 64.61% |
| 32 | 389.167 | 2883.021 | 82.61% | 97.72% |

Batch 1 uses the single active rank. Larger batches report per-rank medians;
medians of components and ratios need not add or divide exactly. The NCCL
column covers recognized framework collectives; the EP column includes
initialization, transport, and finalization kernels. Both include peer waiting
and are not pure PCIe-copy time or bandwidth-utilization measurements.

Prefill scheduling differs even under matched settings: NCCL batch 32 has two
EXTEND steps per rank in this diagnostic, while the candidate has one or two.
The table aggregates the actual observed prefill windows for each workload;
it does not claim identical per-layer route traces or a fixed-step microbenchmark.
At batch 8, substantial gaps on some candidate ranks lower the time fraction
while the absolute communication duration remains much higher than NCCL.

At batch 32, the candidate's median dispatch transport time is 931.432 ms and
combine transport time is 1,951.195 ms. The medians of all initialization and
finalization components sum to only about 3.5 ms. Nearly all measured EP kernel
time remains in transport/progress; separate initialization and final reduction
are small. Transport duration can include waits for late peer launches or
framework scheduling as well as host-memory service and flow control. The
transport still uses all-peer header acknowledgments, serial peer traversal
within each channel, and drains queues at each operation boundary. A timeline
alone cannot distinguish their individual cost from memory-service latency.

## Sixteen-channel follow-up

The four-channel trace motivated a bounded channel-count experiment using the
same runtime commit and the existing `EP_XPRESS_CHANNELS=16` option. Eight-rank
correctness replay passed before the model run, and all eight model workers
reported 16 effective channels. This candidate reused the fresh NCCL baseline
above; source, image, hardware, model/request settings, and host-only transport
checks passed. NVLink payload increments were zero in this run as well.

| Concurrent requests | NCCL batch latency (s) | EP, 4 channels (s) | EP, 16 channels (s) | 16-channel latency increase over NCCL |
|---|---:|---:|---:|---:|
| 1 | 3.055 | 3.344 | 3.404 | 11.4% |
| 8 | 2.990 | 4.282 | 3.740 | 25.1% |
| 32 | 3.401 | 6.850 | 4.929 | 44.9% |

Sixteen-channel latency ranges over three repetitions were 3.372–3.411,
3.738–3.751, and 4.824–4.997 seconds. Its output throughput medians were
9.401, 68.453, and 207.730 tokens/s.

| Concurrent requests | Mean TTFT, 16 channels (ms) | Last TTFT, 16 channels (ms) | Per-request TPOT, 16 channels (ms) | Steady-window TPOT, 16 channels (ms) |
|---|---:|---:|---:|---:|
| 1 | 211.803 | 211.803 | 102.961 | 102.961 |
| 8 | 531.599 | 532.071 | 103.515 | 103.515 |
| 32 | 1329.368 | 1657.691 | 117.981 | 107.183 |

Higher channel count improves measured large-batch prefill and end-to-end
latency, but increases steady-window decode time. At batch 1, the shorter TTFT
does not offset slower decoding. The library default remains four channels;
16 is an explicit measured alternative, not a universal optimum or an
implemented automatic prefill/decode policy. At H=2048 and a 2 MiB total
payload budget, each channel holds 128 rows with four channels and 32 rows with
16, so this experiment changes both parallelism and message granularity.
The previous single-block batch-32 latency of 29.570 seconds falls to 4.929
seconds, about a 6.0x implementation improvement, while remaining 44.9% slower
than the NCCL baseline.

The sixteen-channel traces also resolved. Median prefill measurements are:

| Concurrent requests | GPU forward-window union (ms) | EP path kernel union (ms) | EP path time fraction | Dispatch transport (ms) | Combine transport (ms) | Initialization + finalization (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 125.899 | 59.632 | 47.37% | 14.786 | 43.971 | 0.875 |
| 8 | 372.916 | 348.499 | 93.46% | 65.117 | 281.604 | 0.965 |
| 32 | 1337.494 | 1266.692 | 94.36% | 272.751 | 988.444 | 3.578 |

For batch 32, ranks have one or two EXTEND steps (four of each); the window
median lies between these schedule groups and is not a representative single
forward. Components are medians of per-rank totals and do not necessarily sum
to the median union. Nearly all EP kernel time still belongs to transport,
with combine the largest component. These are kernel-duration measurements
including waiting; they do not identify achieved PCIe bandwidth or prove that
all of the remaining time is data movement.

Both channel configurations were checked separately against NCCL. All four short
greedy prompts matched in text and token IDs (29 generated tokens). Chosen-token log
probabilities over the common prefixes differed by up to 0.219 (mean absolute
difference 0.0147). This small check does not prove numerical equivalence or
identical routes for the performance requests; reduction order and BF16
rounding can differ.

## Combine phase diagnostics

The optional instrumentation in revision `5bf0cfe` was measured with the same
eight GPUs, model, BF16 kernels, EP/TP/DP=8, eager execution, two slots, 2 MiB
chunk budget, and 16 communication channels. NVLink counter increments were
zero, and the remaining framework NCCL connections used SHM without P2P, NET,
or NVLS. The performance baseline remains SGLang's default NCCL backend from
the preceding comparison.

These diagnostic requests contain 512 input and **eight output tokens**, so
their generation times must not replace the 32-output-token performance
results above. Batch sizes are one and 32. Each shape uses identical input IDs
in the order plain/plain/trace-off/trace-on/trace-on/trace-off/plain/plain;
"on" enables native combine counters and "off" retains ordinary kernels.
The single request is pinned to DP rank 0. Batch 32 verifies four requests per
rank, although asynchronous admission can still produce different prefill
batch splits. Both on trials transfer identical total requested combine
payload bytes, with per-operation global writes equal to reads.

Counters are allocated before the request and copied back after profiling
stops. No CPU counter readback or additional collective runs between layers.
All 64 rank captures passed completeness and mode checks. The 32 instrumented
captures contributed 233,472 CTA records, which passed status, phase-closure,
and wait-subset checks. All 64 Chrome traces resolved GPU operations to their
local forwards.

### Measurement definitions and perturbation

For every combine operation, the analysis selects its longest-resident channel
CTA, then sums that CTA's phase times across layers on each rank. Concurrent
channel durations are never added as latency. This closely follows transport
kernel residency but excludes counter stores, launch boundaries, and separate
initialization/final-reduction kernels. It is not end-to-end critical-path time.

Nine disjoint phases partition that CTA duration. A separate view classifies
completed loop iterations with **no cursor or frame progress**. Ready-only,
credit-only, both-missing, final-drain, and other idle categories are exclusive
within this second view. They overlap the nine phases and cannot be added to
the phase totals. A blocked producer that still receives useful data is not
classified as idle. The categories describe snapshots, not an independently
isolated causal cost of ready or ACK.

| Diagnostic request | Plain generation median (ms) | Trace, counters off (ms) | Trace, counters on (ms) |
|---|---:|---:|---:|
| Batch 1 | 966.341 | 1110.190 | 1118.335 |
| Batch 32 | 2469.697 | 2597.372 | 2552.491 |

For batch 1, the counters-on generation median is 0.73% above counters-off;
Chrome-measured combine transport is about 0.71% higher in prefill and 6.76%
higher in decode. Small-message phase proportions are therefore approximate.
The trace-off generation median is 14.89% above the pooled plain median, but
plain repeats also drift from about 934 to 1001 ms. These are observed differences,
not isolated overhead estimates. Batch 32's
negative on/off difference is **not** evidence of a speedup or negative
instrumentation overhead: one on trial has a different prefill batch split.
The other on trial and both off trials have two EXTEND steps per rank, but two
repeats still do not establish a precise perturbation bound.

### Single request: receiving payload versus control

Rank 0 owns the request and receives the combined partials. Across its 48
prefill combines, the selected-CTA total is 44.69--44.85 ms. The following
numbers are the range across the two instrumented repeats, rounded:

| Rank-0 phase | Prefill accumulated time (ms) | Fraction of selected-CTA time |
|---|---:|---:|
| Host payload reads | 38.37--38.52 | 85.9% |
| ACK publication, including fence | 2.26 | 5.0% |
| Local routing/copy | 1.38--1.39 | 3.1% |
| Peer polling | 1.03 | 2.3% |
| Receive metadata/selection | 0.97--0.98 | 2.2% |
| Remaining stages | 0.67--0.69 | about 1.5% |

Rank 0 has no complete no-progress iterations in these captures. Its requested
prefill payload reads total 443.734 MiB across channels and layers. The measured
read phase includes GPU instructions, host-memory service, and CTA barriers;
it is not pure PCIe wire time and does not prove link saturation. Publication
fences also account for completion effects that may outlast payload stores.

Across eight decode forwards, rank 0's selected-CTA combine total is about
37.09 ms. Payload reads account for 26.2%, ACK publication 25.8%, peer polling
20.2%, and receive metadata/selection 13.0%. Control work matters much more for
small messages. In the ordinary trace, combine transport totals about 35.32 ms,
or 4.41 ms per decode forward across all 48 layers.

The other seven ranks are locally IDLE but still execute experts and combine.
Their apparent combine waiting is much larger: about 85% of their selected-CTA
residency has no progress. This does not add 85% to request latency. Explicit
operation identities align their work with rank 0: coarse start spreads are
about 0.50 ms per prefill combine and 0.72 ms per decode combine. Early ranks
can wait while the request-owning rank is still completing earlier work. In
prefill they also spend about 40.5--41.4 ms per rank draining the final ACKs,
overlapping the receiver's payload processing. Summing these waits would
double-count concurrent activity.

### Batch 32: stalled progress dominates kernel residency

The two diagnostic trials produce different prefill splits; local IDLE
forwards are retained separately. Within each trial the table reports the
**median of per-rank fractions**, so column entries need not sum to 100%.

| Phase within active prefill combines | First on trial | Second on trial |
|---|---:|---:|
| Host payload reads | 17.1% | 11.4% |
| Host payload writes | 7.2% | 6.2% |
| Peer polling | 60.6% | 67.6% |
| ACK publication | 2.5% | 2.2% |
| Other control | 9.8% | 11.0% |
| Remaining routing/metadata/publication phases | about 2.9% | about 2.3% |

The overlapping no-progress view is:

| No-progress category | First on trial | Second on trial |
|---|---:|---:|
| Ready missing, credit available | 16.4% | 17.2% |
| Credit missing, ready available | 0.13% | 0.14% |
| Both ready and credit missing | 43.7% | 46.3% |
| Final drain | 0.12% | 0.25% |
| **All no-progress categories, median per rank** | **62.3%** | **70.5%** |

The total is computed per rank before taking its median, not by adding category
medians. Selected-CTA prefill totals have rank medians of 962 and 1179 ms. The
ordinary traces show combine transport medians of 1143 and 1232 ms; initialization
and final reduction together cost only about 2.65 ms per rank over prefill.
Final reduction arithmetic is not the large component in this implementation.

Decode shows a different stall pattern: 70.1--73.0% no-progress residency, mostly
ready-only wait (68.0--71.8%), with no credit-only or both-missing idle iterations.
Its coarse rank-start spread is around 0.61 ms per combine. Prefill's start-spread
median is 0.49--0.54 ms, much shorter than its roughly 12--20 ms median active
combine kernels. This motivates investigating progress inside the transport
protocol; the traces do not isolate arrival skew's causal contribution or
exclude its amplification by subsequent backpressure. These spreads use the shared exported Chrome time
base and matching operation identities, with unquantified clock-alignment
uncertainty; raw GPU globaltimer values are never subtracted across devices.

### Consequences for the next optimization at that revision

Deduplication and source-shared payloads primarily reduce dispatch traffic.
Combine returns distinct already-weighted partials from each participating
expert rank; those results cannot be deduplicated as identical activations.
Lower byte counts do not compensate automatically for inefficient host reads,
frequent small control transactions, or stalled queues.

The measured producer reuses a slot only after every peer has acknowledged it,
including peers that did not select its payload. A CTA processes ready peers
in sequence. These are concrete places to test destination-aware reclamation
and more independent receive progress, while preserving safe shared-slot
lifetime. For small decode messages, reducing empty-channel/control work and
understanding rank arrival skew are separate priorities. For single-request
prefill, the host-read phase dominates measured selected-CTA residency.

The data establishes these mechanisms and their observed residency; it does
not assign all both-missing waits to ACK latency, establish achieved physical
PCIe bandwidth, or predict the speedup of removing a wait. An optimization must
be validated again against the same default NCCL full-model baseline.

## Rank-pair combine and critical-path cleanup

Measured on 2026-10-23 with runtime `d3410eb`. The model, eight GPUs, BF16,
TP/EP/DP=8, expert/attention kernels, and eager execution remain as specified
above. SGLang is pinned to `e00930c5489053f26d86b179cee0d087f846acbb`
(v0.5.21 with the adapter patch). Both the default NCCL baseline and candidate
were run afresh for batches 1 and 32. Each shape has eight warmup batches
generating **16 tokens**, followed by three timed batches with 512 input and
32 output tokens. No profiler or combine counters are enabled during timing.

### What changed

Combine now has a separate directed queue for each source/destination rank
pair in each channel. Only that queue's destination acknowledges its slots.
A block progresses both outgoing and incoming traffic with one peer, so a
full outgoing queue does not prevent it from receiving. Unrelated peers no
longer gate that pair's slot reuse. The whole combine operation still needs
its required results and pair terminal frames before it can finish.

At 16 channels and EP8, the combine grid grows from 16 to 128 blocks per GPU,
including the self/local blocks. Channel count is not a fixed SM count. The
per-pair chunk row limit is divided by the world size to keep the aggregate
payload budget approximately fixed. The actual combine arena increases from
4,259,840 to 4,718,592 bytes per rank because headers and page rounding grow.
This is therefore not an isolated ACK-policy ablation: queue layout, scheduling
parallelism, and per-queue granularity also change.

The first pair-queue implementation exposed substantial repeated control
work. The final timed runtime additionally:

- Removes Python host-header error scans from normal submission and
  `Pending.wait()`. Scanning remains at explicit `check_errors()` or diagnostic
  boundaries. These were health checks, separate from the GPU ready/ACK
  protocol; they were not payload copies or GPU completion fences.
- Samples mapped-host abort flags and timeout clocks on the first loop iteration
  and every 256 iterations thereafter, instead of every iteration. Error
  observation can consequently be delayed; this is an iteration cadence,
  not a wall-clock response bound.
- Replaces read-only `atomicAdd(status, 0)` operations with relaxed device
  status loads, and checks the immutable operation fingerprint on the first
  received frame of each peer/channel rather than every frame.
- Caches the adapter's resolved Context and timeout conversion, removes the
  adapter's redundant per-call capture query, consolidates native combine
  capacity validation, and limits token-index validation to its destination
  block. Sequence/epoch checks, ready/ACK ordering, bounds, and tensor/handle
  lifetime dependencies remain.

Dispatch still shares a source payload for multicast; its queue reclamation
has not been changed to actual-reader acknowledgments in this revision.
Combine partials are directed to one destination, so they do not require a
multicast reference counter. The final runtime also includes `daec161`'s
capture guard at first Context initialization and `d3410eb`'s removal of the
adapter's duplicate expert-output shape check; the core still validates it.
The measured runtime SHA is `d3410eb`.

### Full-model result

| Concurrent requests | NCCL batch latency (s) | EP-Xpress batch latency (s) | EP-Xpress latency increase | NCCL output tokens/s | EP-Xpress output tokens/s |
|---|---:|---:|---:|---:|---:|
| 1 | 3.090 | 3.108 | 0.61% | 10.357 | 10.295 |
| 32 | 3.391 | 3.954 | 16.59% | 301.943 | 258.979 |

Observed three-repeat batch-latency ranges are 3.030–3.096 seconds for NCCL
and 3.108–3.121 for EP-Xpress at batch 1; at batch 32 they are 3.385–3.456
and 3.932–4.019 seconds respectively. These ranges are not confidence
intervals. Batch 1 is close to NCCL and does not establish a speedup. Batch 32
remains slower than NCCL.

| Concurrent requests | NCCL mean TTFT (ms) | EP-Xpress mean TTFT (ms) | NCCL per-request TPOT (ms) | EP-Xpress per-request TPOT (ms) |
|---|---:|---:|---:|---:|
| 1 | 192.189 | 196.213 | 93.465 | 93.951 |
| 32 | 470.762 | 821.056 | 93.972 | 100.161 |

TTFT and TPOT first average across requests in each repetition, then take
the median over three repetitions. TPOT uses each request's elapsed time
between its first and last received token divided by the number of tokens
after its first streaming chunk. It is not the median of individual token
intervals. The larger remaining relative gap at batch 32 is in time to first
token; the 6.59% per-request TPOT increase is smaller. These client timings
do not identify a specific transport bottleneck or PCIe utilization.

For implementation history, the first pair-queue runtime (`ebaedcf`) measured
4.144 and 5.012 seconds at batches 1 and 32. The control-path cleanup reduces
these to 3.108 and 3.954 seconds, or 25.0% and 21.1%. Multiple changes were
measured together; these reductions cannot all be attributed to Python error
scanning. That intermediate run completed all six timed batches, but its
runner lost its output connection afterward and required separately recovered
transport evidence. It is not the validated final comparison above.

The preceding cleanup run (`6ecedd7`, before the cold-init guard and duplicate
adapter-check removal) measured 3.117 and 3.918 seconds. The final rerun's
small changes have no demonstrated independent performance benefit.

The earlier source-shared 16-channel results were 3.404 and 4.929 seconds.
The current medians are 8.7% and 19.8% lower. This is historical context with
a different warmup length, not a fresh ablation or the performance baseline.
The fresh default NCCL comparison is the main result.

### Evidence and limits

The final comparison passed configuration, input-hash, source, hardware, and
transport checks with no warnings or errors. All eight candidate ranks used
16 effective channels. NCCL logs show SHM transport with no P2P, NET, or NVLS
connections; all eight GPUs have zero NVLink payload counter increments in
both runs. The candidate's EP payload itself uses GPU loads/stores to mapped
host memory. Both full-model runs completed without reported runtime errors.

Four additional short natural-language prompts produced identical greedy
text and all 29 generated token IDs across the two backends. Maximum common-
prefix log-probability difference was 0.218722. This is limited output sanity,
not proof of layer-wise numerical equivalence or identical expert routes
throughout the timed requests. No new phase profiling was collected for this
revision, so older communication-time fractions must not be assigned to it.

Deployment addresses, credentials, model paths, and raw logs are deliberately
excluded. Experiment records remain in the operator's private artifact store.

## Prepared transport and channel-count experiment

Runtime `282437b` moves fixed native validation into `PreparedHostSM`
construction. The object retains placement and queue tensors and caches their
addresses, strides, dimensions and cooperative launch capacity. The adapter
reuses SGLang's already-computed global DP buffer length. No SGLang source
patch change is needed for that getter in the pinned revision.

Protocol diagnostics default to off. Normal submission does not hash tags,
and the kernels skip mapped-host abort/timeout polling, operation fingerprints,
duplicate-expert checks and finite-weight diagnostics. Dynamic tensor/index
bounds, ready/ACK, epochs, required barriers and stream lifetime dependencies
remain. The SGLang fatal-error path skips dispatch's error-only finalization
kernel. These changes were made together, so a model result cannot isolate the
benefit of each check or launch removed.

The experiment uses a fresh default NCCL baseline and the same eight GPUs,
BF16 model, TP/EP/DP=8, eager execution, 512 input/32 output tokens, eight
16-token warmup batches, and three timed repetitions per shape. Both sides
disable GPU P2P and NVLink transports. Timing is unprofiled full-model HTTP
latency. Protocol diagnostics are explicitly disabled in candidate runs.

In the pinned SGLang implementation, the default backend's MoE input gather
depends on DP padding. Eager prefill and the uneven one-request decode use
`SUM_LEN`: each rank zeros a global buffer, writes its own segment and uses
AllReduce to assemble the input. Balanced 32-request decode uses `MAX_LEN`
and AllGather. With this run's `boundary_reduction=rs+rsv`, the output boundary
uses `reduce_scatterv`, implemented by grouped NCCL Reduce calls to each root.
It would be inaccurate to call every phase AllGather plus a single native
NCCL ReduceScatter. EP-Xpress replaces this MoE boundary with local routing,
targeted dispatch, local experts and directed combine. This code-path account
does not determine either backend's physical PCIe byte count.

### Full-model results

All four candidates use the same runtime `282437b`. The fresh baseline was
measured before the serial candidate sweep. The following values are in
milliseconds; brackets show the minimum and maximum of three repetitions.
The change column compares full-batch completion latency with default NCCL.

| Backend / channels / chunk | Concurrent requests | Completion median [min, max] | Change vs NCCL | Mean TTFT | Mean per-request TPOT |
|---|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3114.857 [3062.530, 3127.222] | — | 194.775 | 94.193 |
| EP-Xpress / 8 / 2 MiB | 1 | 3039.172 [3002.523, 3045.518] | -2.43% | 191.185 | 91.867 |
| EP-Xpress / 16 / 2 MiB | 1 | 3043.810 [3040.791, 3063.358] | -2.28% | 193.269 | 92.002 |
| EP-Xpress / 32 / 2 MiB | 1 | 3056.821 [3001.140, 3057.351] | -1.86% | 192.205 | 92.374 |
| EP-Xpress / 32 / 4 MiB | 1 | 3045.453 [3042.591, 3058.332] | -2.23% | 191.061 | 92.074 |
| Default NCCL | 32 | 3439.637 [3402.790, 3456.911] | — | 546.432 | 93.723 |
| EP-Xpress / 8 / 2 MiB | 32 | 4100.810 [4037.064, 4159.558] | +19.22% | 964.735 | 101.206 |
| EP-Xpress / 16 / 2 MiB | 32 | 3839.446 [3838.758, 3857.310] | +11.62% | 810.555 | 97.788 |
| EP-Xpress / 32 / 2 MiB | 32 | 3758.110 [3755.210, 3783.087] | +9.26% | 680.794 | 99.699 |
| EP-Xpress / 32 / 4 MiB | 32 | 3611.143 [3603.167, 3671.255] | +4.99% | 568.159 | 98.127 |

TTFT and per-request TPOT are first averaged over requests within a repetition,
then reported as the median of three repetitions. Per-request TPOT measures
the first-to-last streamed-token interval divided by subsequent token count.
It is not isolated GPU decode time, and its median need not add to the
completion-latency median with the TTFT median.

C32/4-MiB is the best tested 32-request configuration, 5.95% faster than the
cleaned-up C16/2-MiB candidate but still 4.99% slower than default NCCL. At
fixed 2 MiB, increasing C16 to C32 reduces completion time by 2.12% and mean
TTFT by 16.01%; per-request TPOT increases by 1.95%. Increasing the chunk
budget from 2 to 4 MiB at C32 reduces completion time by another 3.91%.
This supports tuning parallelism together with frame granularity; it does
not show that more CTAs alone solve the remaining overhead. The single-request
changes are small, and these three repetitions are insufficient to establish
a robust improvement across runs or hardware.

All candidate comparisons passed configuration, input-hash, source, hardware
and transport validation with no warnings or errors. Every rank used the
requested effective channel count. NCCL connection logs show SHM only
(120 baseline records, 60 per candidate), with no P2P, NET or NVLS records;
NVLink payload counters remain unchanged on all eight GPUs in every run.
Connection counts are not traffic measurements. The candidate's EP payload
uses mapped host memory directly. The four short sanity prompts match text
and all 29 output token IDs for every candidate; maximum common-prefix
log-probability difference is 0.218722. This is limited output sanity, not
layer-wise numerical equivalence. All full-model runs completed without
reported runtime errors; no new kernel-phase profile was collected.

To reproduce the best tested 32-request candidate's transport settings, use
`EP_XPRESS_TRANSPORT=host_sm`, `EP_XPRESS_CHANNELS=32`,
`EP_XPRESS_CHUNK_BYTES=4194304`, `EP_XPRESS_SLOTS=2` and
`EP_XPRESS_DIAGNOSTICS=0`, together with the matched model and host-only
transport configuration above. This is a measured configuration for this
hardware/workload, not a new portable default.

### Interpreting additional CTAs

The channel parameter controls both phases and partitions the payload budget:

| Channels | Chunk budget | Dispatch CTAs/GPU | Combine CTAs/GPU | Dispatch rows/slot | Combine rows/pair/slot | Total mapped host allocation |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 2 MiB | 8 | 64 | 64 | 8 | 70,254,592 B |
| 16 | 2 MiB | 16 | 128 | 32 | 4 | 72,876,032 B |
| 32 | 2 MiB | 32 | 256 | 16 | 2 | 77,594,624 B |
| 32 | 4 MiB | 32 | 256 | 32 | 4 | 145,752,064 B |

These are configured layouts for EP8, H=2048 and top-k=8, including two slots,
headers and page rounding. Allocation counts unique shared host storage across
all eight ranks and both phases, not each process's mappings of that storage.
All eight ranks used these effective channel counts in the completed runs.
CTAs are blocks, not reserved SMs. The fixed-2-MiB sweep changes both
parallelism and message granularity; it is not a pure CTA-count ablation.
C32/4-MiB retains C16/2-MiB's row granularity but doubles mapped allocation.
The cooperative grid must fit the kernel's active-block capacity across the
device, which is why bootstrap checks occupancy instead of treating the CTA
count as unlimited ([CUDA execution API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)).

### Why fewer payload bytes need not make this implementation faster

Rank deduplication is useful but does not imply an order-of-magnitude traffic
advantage. As a hypothetical uniform, distinct top-k example for 128 experts
on eight ranks, the probability that one rank receives a token is
`q = 1 - C(112, 8) / C(128, 8) = 0.66742`. A token reaches an expected 5.339
ranks, including 4.672 remote ranks. That is about 33.3% fewer remote destination
payloads than delivery to all seven remote ranks. Real model routes need not
be uniform; these numbers are an analytical example, not measured routing.

For a token with `d > 0` remote destinations and a BF16 row size `S=4096 B`,
source-shared dispatch writes `S` bytes to host memory and destinations read
`d*S`; combine writes and reads `d` distinct partials, adding `2*d*S`. Thus
logical payload traffic across all participating GPUs, summing their host
payload writes and reads for this token, is `(1 + 3*d)*S`, roughly 61.5
kB/token/layer in the uniform example. This describes the current algorithm,
not a universal lower bound. It excludes headers, fences, padding and local GPU traffic;
it is neither a physical PCIe counter nor a measured NCCL byte comparison.

There are concrete costs beyond payload traffic:

- Every combine peer CTA scans its channel's inverse-route range independently.
  For a remote-peer CTA, a scan examines up to 256 rows but sends at most the
  per-pair slot limit. The self/local CTA bypasses host memory and has a
  256-row limit.
  When more rows match, it resumes after the last emitted match, rereading the
  already-scanned tail in the next iteration. With four-row slots and a long,
  roughly uniform range, a 256-row scan can find about 32 matches and emit only
  four. This repeats work beyond the initial scan by all eight peer CTAs.
- More channels create more terminal frames and control lanes even for empty
  partitions. With one global decode token, at most one channel in each
  participating rank has a payload; increasing channels does not divide that
  single token into additional useful transfers.
- Each small frame still publishes ready, receives ACK and fences memory.
  At fixed total budget, C16 to C32 halves combine payload from 16 to 8 KiB per
  full slot. More parallel blocks can therefore coexist with more control
  operations per payload byte.
- Dispatch still traverses peers within a channel and waits for all peers to
  acknowledge each shared chunk header. Its reclamation policy is different
  from the independent pair queues now used by combine.
- The adapter gives the Triton expert runner a tensor sized to the global
  receive-capacity bound. Received rows are compacted, but unused rows remain
  padding with expert IDs of -1 and zero weights. The runner still sizes
  temporary buffers and some route/reduction work from that tensor shape and
  top-k. This does not mean every padded row executes an expert GEMM; it means
  communication deduplication has not also removed all capacity-sized expert
  preparation and output work.

For that revision, the next algorithmic priorities were to build GPU row-index
lists by return owner once and consume those lists directly, reclaim dispatch payloads using
only their actual readers, and choose different active channel counts for
prefill and decode. Building return lists while dispatch creates inverse rows
could avoid a separate route-planning pass without adding CPU route readbacks.
Separate dispatch/combine channel settings would also distinguish the needs
of the two phases. The expert adapter/runner could consume a GPU valid count
or compact layout without introducing a CPU count readback. Another candidate
is assigning separate send/receive warp groups inside a block: the current
block interleaves send and receive phases,
and double buffering alone does not establish simultaneous payload movement
in both directions. Such a change must preserve independent progress and
publication/slot-lifetime ordering. Buffer reuse and fewer eager launches
remain useful, but older profiles found initialization/final reduction small relative to transport;
that historical result does not identify the current revision's bottleneck.


## Owner-list and sparse-work experiment

Runtime `7b05797` implements four changes in the prepared host-SM path:

1. Dispatch builds GPU row-index lists by return rank. Combine consumes these
   lists directly instead of scanning inverse maps separately for each peer or
   rescanning a suffix that did not fit the previous frame.
2. Each handle carries an active channel count. The SGLang adapter uses common
   scheduler metadata to select prefill and decode widths without a new
   collective or CPU count readback. Explicit phase overrides are available.
3. Dispatch publishes per-reader READY notifications and waits only for actual
   readers when reusing a payload slot. All-peer terminal handshakes remain at
   operation completion; publication fences and stream dependencies remain.
4. An optional BF16 Triton expert path consumes the GPU receive count, omits
   invalid routes from alignment, and skips unused activation/reduction rows.
   It retains the original tuned GEMMs and CUDA SiLU arithmetic. Workspace
   allocations and launch upper bounds remain based on receive capacity.

The same eight-GPU, model, dtype, eager-mode, 512-input/32-output workload and
host-only restrictions above apply. Both backends use the same updated pinned
SGLang patch. Eight warmup batches generate 16 tokens; each measured shape has
three repetitions. The candidate allocates 32 channels, two slots and a 4-MiB
chunk budget with diagnostics disabled. Adaptive mode uses 32 active prefill
channels for these inputs and `ceil(global_capacity / 8)` decode channels,
bounded to 1 through 32. Frame sizes do not change with the active width.

### Initial all-four comparison

Times below are milliseconds. Brackets are the observed three-repeat range.

| Backend | Concurrent requests | Completion median [min, max] | Change vs NCCL | Mean TTFT | Mean per-request TPOT |
|---|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3070.566 [3030.910, 3074.277] | — | 191.544 | 92.869 |
| EP-Xpress, all four enabled | 1 | 2893.403 [2890.425, 2971.019] | -5.77% | 185.236 | 87.365 |
| Default NCCL | 32 | 3434.193 [3397.490, 3462.598] | — | 441.536 | 95.502 |
| EP-Xpress, all four enabled | 32 | 3441.940 [3370.109, 3687.773] | +0.23% | 592.161 | 94.153 |

Single-request completion improves in this run. The 32-request result is
approximately tied with NCCL, with a wider candidate range; it does not show a
robust batch-32 speedup. Mean TTFT varies substantially across repetitions:
434.321–575.995 ms for NCCL and 450.438–763.841 ms for EP-Xpress. The median last
request's TTFT differs by 6.79%, smaller than the difference in mean TTFT.
Server logs confirm that request arrival produced different prefill splits
across repetitions and backends: some ranks processed four requests together,
while others split them as 1+3, 2+2, or 3+1. These HTTP metrics include scheduling
and split prefills; they do not isolate expert alignment or communication-kernel time.

Configuration, input hashes, source provenance, hardware and host-only transport
validation passed without warnings or errors. All ranks negotiated 32 allocated
channels. NCCL logs show SHM only: 120 baseline records and 60 candidate records,
with no P2P, NET or NVLS records. Hardware NVLink payload increments are zero on
all eight GPUs. The EP payload itself uses mapped host memory. These are
transport checks, not a PCIe bandwidth measurement. All four short sanity
prompts match text and all 29 token IDs; the maximum common-prefix log-probability
difference is 0.218722. This limited output check does not establish layer-wise
numerical equivalence. No new kernel-phase profiling was collected.


### Valid-count ablation

A second EP-only run uses exactly the same source manifest, SGLang patch and
settings, changing only `EP_XPRESS_VALID_COUNT=1` to `0`. It therefore retains
owner lists, selective dispatch and adaptive channels, but uses the previous
capacity-based Standard/Triton expert path. It is compared with the same fresh
NCCL baseline above.

| Valid-count path | Concurrent requests | Completion median [min, max] | Mean TTFT | Mean per-request TPOT |
|---|---:|---:|---:|---:|
| Enabled | 1 | 2893.403 [2890.425, 2971.019] | 185.236 | 87.365 |
| Disabled | 1 | 3088.813 [3066.236, 3090.977] | 193.927 | 93.380 |
| Enabled | 32 | 3441.940 [3370.109, 3687.773] | 592.161 | 94.153 |
| Disabled | 32 | 3583.773 [3565.736, 3593.134] | 526.910 | 98.579 |

Disabling the GPU-count expert path increases completion latency by 6.75% at
batch 1 and 4.12% at batch 32 relative to the enabled run. The path remains
enabled by default. The lower batch-32 mean TTFT with the path disabled does
not isolate alignment cost: server logs show different prefill splits in this
run as well. The complete decode interval and end-to-end result favor keeping
it enabled. Configuration/source/hardware/transport validation again passes
without warnings or errors; output token/text checks match, and NVLink payload
counters remain unchanged. No additional native regression suite or kernel
profiling was run for this environment-only ablation.

These measurements support the combined implementation and the count-aware
expert path on the tested workload. They do not independently measure the
contribution of owner lists, selective ACKs, or adaptive channels. The batch-32
result remains near parity with NCCL, not a demonstrated robust speedup. The
large count-aware alignment still uses one CTA per expert and scans valid IDs
per expert; tile-parallel alignment is a remaining implementation opportunity,
not part of the tested runtime. Dense frame destination unions can also include
all peers, so actual-reader ACKs need not remove a reader in every frame.


## Decode workspace and fused receive experiment

Runtime `4356b6f`, with source snapshot `5c208da`, adds three changes:

1. The SGLang adapter caches exact-shape communication workspaces per layer,
   with up to four entries. Reuse avoids repeated communication tensor
   allocations; expert GEMM workspaces are outside this change.
2. Bounded decode writes direct-address host mailboxes and fuses receive with
   deterministic reduction. Combine uses one kernel instead of clear,
   communication and finalize kernels, without a remote device scratch tensor.
   Actual contributors publish and acknowledge each returned token; this
   combine path needs no unrelated-pair terminal frames. Dispatch still uses
   its terminal handshake and required stream/publication dependencies remain.
3. `combine_frame_rows` configures pair-queue frame capacity independently of
   allocated channel count. The first candidate uses 16 rows instead of the
   previous 4, while retaining 32 allocated channels and a 4-MiB dispatch budget.

The full model, eight GPUs, BF16, TP/DP/EP8, Triton experts, eager execution and
host-only restrictions are unchanged. A fresh default-NCCL baseline and the
candidate each run batches 1 and 32 with 512 input and 32 output tokens, eight
16-token warmup batches and three timed repetitions. Adaptive channels and the
GPU-count expert path stay enabled; decode mailbox capacity is 64. No profiler
is active during timing.

### Sixteen-row combine frames

Times are milliseconds. Brackets show the three-repeat range, not confidence
intervals. TPOT below is the median of each batch's mean per-request TPOT.

| Backend | Concurrent requests | Completion median [min, max] | Change vs NCCL | Mean TTFT | Per-request TPOT |
|---|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3096.493 [3064.832, 3134.372] | — | 193.482 | 93.643 |
| EP-Xpress, fused decode, 16 rows | 1 | 2833.628 [2806.105, 2860.894] | -8.49% | 183.971 | 85.470 |
| Default NCCL | 32 | 3408.966 [3395.272, 3440.669] | — | 461.489 | 94.841 |
| EP-Xpress, fused decode, 16 rows | 32 | 3551.251 [3308.553, 3557.894] | +4.17% | 705.898 | 91.681 |

The single-request end-to-end result improves. At batch 32, per-request TPOT
falls by 3.33%, but a longer first-token window offsets that reduction. Last
TTFT is 558.294 ms for NCCL and 810.862 ms for the candidate; the candidate's
three values range from 569.152 to 813.846 ms. These HTTP metrics include
arrival, scheduling and split prefills; they do not isolate a frame-size cost.
Steady-window TPOT after the last first token falls from 92.332 to 89.432 ms.
This candidate does not demonstrate a batch-32 completion speedup.

Configuration, input hashes, source and hardware checks pass without warnings
or errors. NCCL logs contain SHM connections only: 120 for the baseline and
60 for remaining framework collectives in the candidate; P2P, NET and NVLS
records are zero. NVLink payload counters stay unchanged on all eight GPUs.
EP payloads use mapped host memory. These checks do not measure PCIe bandwidth.
Four short greedy output checks match text and all 29 token IDs; maximum
common-prefix log-probability difference is 0.218722. The focused native
mailbox test separately matches the old combine path bit for bit. Neither
check establishes equivalence for every full-model route or input.

Server logs show different prefill partitions. In the two slower 16-row
candidate repetitions, every rank runs two nonempty prefills, commonly 2+2
requests; the faster repetition includes four ranks accepting all four of
their requests together. NCCL also runs two nonempty prefills on every rank
in two repetitions, commonly 1+3. Split composition differs, but the presence
of an extra prefill alone cannot explain the candidate regression. A subsequent
environment-only four-row comparison keeps both decode optimizations enabled.


### Four-row frame follow-up

A second EP-only run keeps the exact same source snapshot and all settings,
changing only `EP_XPRESS_COMBINE_FRAME_ROWS=16` to `4`. Both decode optimizations
remain enabled; the comparator is the fresh NCCL baseline above.

| Backend | Concurrent requests | Completion median [min, max] | Change vs NCCL | Mean TTFT | Per-request TPOT |
|---|---:|---:|---:|---:|---:|
| EP-Xpress, fused decode, 4 rows | 1 | 2850.916 [2845.297, 2868.363] | -7.93% | 185.098 | 85.990 |
| EP-Xpress, fused decode, 4 rows | 32 | 3345.226 [3337.085, 3379.145] | -1.87% | 587.434 | 88.933 |

The observed batch-32 completion improvement is modest. Last TTFT remains
7.39% above NCCL (599.550 vs 558.294 ms), while steady-window decode TPOT is
4.00% lower (88.637 vs 92.332 ms). Per-request TPOT is 6.23% lower, but this
metric also reflects differences in the individual first-token times. The
single-request steady-window TPOT reduction is 8.17%.

All three four-row repetitions have the same prefill split: one rank runs
1+3 requests and the others accept four together. This differs from both the
baseline and the 16-row run. Consequently the full difference between the two
EP runs cannot be assigned to frame size; three repetitions are also not a
broad workload study. The 16-row setting is not supported as a general default
by these results. The measured four-row configuration is retained for this
workload, with independent frame sizing available for further tuning.

The adapter's fallback frame setting is therefore returned to `0`, preserving
the original budget-derived size unless explicitly overridden. At the tested
32 channels, eight ranks, H=2048 and 4-MiB chunk budget, that resolves to the
same four rows. This post-experiment default-only change does not alter either
run's explicit environment setting or CUDA implementation; it was checked
locally, without repeating the model runs.

Source, configuration, hardware, input and transport validation again passes
without warnings or errors. NVLink payload increments stay zero; remaining
NCCL traffic uses SHM only. Output sanity again matches all four texts and 29
tokens, with the same maximum log-probability delta. Both experiments complete
all requested repetitions without reported runtime errors. No phase profiling
or independent workspace-versus-fusion ablation was collected. These results
support the combined changes on the tested host-only eager workload, not a
claim of maximum PCIe utilization or performance on consumer GPUs.


## Independent send and receive warp experiment

Runtime `3f5f644` changes prepared prefill pair-combine execution. Each remote-peer
256-thread CTA uses four warps for sending and four for receiving, with separate
shared metadata and 128-thread named barriers. Local copies retain the whole CTA.
Each group can progress its own direction while the other waits for pair-local
READY or ACK. Publication fences,
slot credits, terminal handshakes and the final CTA join remain necessary. This
is concurrent progress within the kernel, not a measurement of PCIe full-duplex
bandwidth or overlap with expert computation.

The experiment retains four-row combine frames, 32 allocated channels, two
slots and the 4-MiB dispatch chunk budget. All eight ranks negotiate 32 channels;
adaptive channel selection is unchanged. The decode workspace cache and fused
mailbox combine remain enabled with capacity 64; the new option does not replace
the decode kernel or change its reduction order. Prefill retains the existing
remote scratch and final reduction kernels.

A fresh default-NCCL baseline and EP-Xpress use the same complete model, eight
GPUs, BF16, TP/DP/EP8, Triton experts and eager execution described above. Both
disable GPU peer transport and NVLink-dependent paths. Each shape uses 512 input
and 32 output tokens, eight 16-token warmup batches and three timed repetitions.
No profiler is active. The candidate explicitly sets
`EP_XPRESS_COMBINE_WARP_SPECIALIZATION=1` and `EP_XPRESS_COMBINE_FRAME_ROWS=4`.

### Full-model comparison

Times are milliseconds. Brackets show the observed three-repeat range.
Steady-window TPOT uses the interval from the last request's first token to
the first request finishing: batch size times elapsed time, divided by observed
token progress in that interval. It is distinct from the mean per-request TPOT
used in the preceding section.

| Backend | Concurrent requests | Completion median [min, max] | Change vs NCCL | Last TTFT | Steady-window TPOT |
|---|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3086.775 [3060.693, 3098.547] | — | 191.741 | 93.383 |
| EP-Xpress, separate Tx/Rx warps | 1 | 2847.997 [2843.384, 2862.120] | -7.74% | 184.891 | 85.903 |
| Default NCCL | 32 | 3399.117 [3367.655, 3404.603] | — | 560.912 | 91.176 |
| EP-Xpress, separate Tx/Rx warps | 32 | 3503.642 [3323.057, 3573.695] | +3.08% | 728.857 | 89.096 |

Single-request completion is lower than default NCCL. At batch 32 the decode
window is 2.28% shorter per token, but the longer first-token window offsets this
gain and the median completion is slower. These differences compare the whole
EP-Xpress implementation with NCCL; they are not the incremental benefit of
splitting sending and receiving warps. Decode keeps the previously fused path.

The batch-32 candidate's last TTFT values are 573.303, 728.857 and 822.177 ms.
Logs show different nonempty prefill distributions across ranks: some ranks
accept all four requests together, while others split them as 1+3. The number
of split ranks does not monotonically track TTFT. HTTP first-token timing
includes arrival, scheduling and split prefills, so these three repetitions do
not isolate kernel overlap, link utilization or the cause of the variation.

Source hashes, configuration, hardware and matching-input checks pass without
warnings or errors. NCCL logs show SHM only: 120 records for the baseline and
60 for the candidate's remaining framework collectives, with zero P2P, NET or
NVLS records. Hardware NVLink payload increments are zero on all eight GPUs;
EP payloads use mapped host memory. All four short sanity prompts match text
and all 29 token IDs; maximum common-prefix log-probability difference is
0.218722. The separate focused CUDA checks match legacy combine outputs and
remote scratch bit for bit, including repeated slot reuse. These are bounded
correctness checks, not proof of equivalence for every full-model input.

### Same-source warp-switch ablation

A subsequent EP-only run keeps runtime `3f5f644`, its source manifest and every
workload setting, changing only `EP_XPRESS_COMBINE_WARP_SPECIALIZATION` from
`1` to `0`. It retains four-row frames, channel selection, workspace reuse and
fused decode. The performance baseline remains the fresh default-NCCL run
above; this switch comparison examines the new prefill execution choice.

| Pair-combine execution | Concurrent requests | Completion median [min, max] | Last TTFT | Steady-window TPOT | Mean per-request TPOT |
|---|---:|---:|---:|---:|---:|
| Sequential, switch off | 1 | 2850.161 [2807.027, 2888.896] | 184.505 | 85.985 | 85.985 |
| Separate Tx/Rx warps, switch on | 1 | 2847.997 [2843.384, 2862.120] | 184.891 | 85.903 | 85.903 |
| Sequential, switch off | 32 | 3596.668 [3571.360, 3623.921] | 827.313 | 89.815 | 92.016 |
| Separate Tx/Rx warps, switch on | 32 | 3503.642 [3323.057, 3573.695] | 728.857 | 89.096 | 89.865 |

The single-request medians differ by only 0.08%, with overlapping ranges;
this experiment shows no material additional single-request benefit from the
warp split. The batch-32 median is 2.59% lower with the split, but first-token
timing and prefill partitions vary between these sequentially run experiments.
Decode uses the same fused kernel in both, so its small timing differences are
not evidence of a decode implementation improvement. These measurements do not
establish a stable incremental warp-split speedup. In particular, the enabled
candidate remains 3.08% slower than default NCCL at batch 32 in the paired run.

The switch-off run also passes configuration, matching-input, source and
host-only transport checks. All ranks retain 32 channels and NVLink counters
remain unchanged. Its four short sanity outputs match the switch-on run's
text, all 29 token IDs and compared log probabilities exactly. No additional
functional suite or kernel-phase profiling was run for this environment-only
ablation. Both EP runs finish all requested repetitions without reported
runtime errors; the adapter keeps the new path enabled, with the explicit
switch available for further comparisons.

## Expert route alignment and NUMA inspection

Before changing routing alignment, runtime `a570d86` was profiled through the
complete model at batches 1 and 32. Warp-specialized prefill combine, four-row
frames, 32 allocated channels, decode workspace reuse and fused mailbox combine
remained enabled. This inspection uses the same eight GPUs, BF16 model, EP8,
eager execution and host-only transport restrictions as the preceding section.
Each shape has eight 16-token warmup batches, followed by a short eight-token
diagnostic trace. The later unprofiled performance run uses 32 output tokens.

### Current execution profile

CPU/GPU traces were collected from all eight ranks for both shapes. GPU
operations are assigned to forwards using launch correlations and annotations;
CPU annotation timestamps are not used as GPU execution windows. Each entry
below is an average forward on each rank followed by the median across the
applicable ranks. Times are milliseconds and cover all model layers in that
forward, not a single MoE layer. Component times use GPU interval unions.

| Shape / local phase | Ranks | Forwards per rank | GPU forward window | Dispatch | Expert alignment | Expert GEMM | Combine | Observed GPU gaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 1, active prefill | 1 | 1 | 113.664 | 9.976 | 0.256 | 2.964 | 20.808 | 74.648 |
| Batch 1, active decode | 1 | 8 | 101.054 | 2.767 | 0.062 | 0.569 | 1.868 | 93.465 |
| Batch 1, locally idle ranks | 7 | 9 | 101.487 | 79.387 | 0.093 | 0.979 | 19.684 | 1.157 |
| Batch 32, prefill | 8 | 2 | 359.028 | 86.813 | 2.894 | 14.971 | 99.886 | 143.945 |
| Batch 32, decode | 8 | 8 | 103.655 | 39.401 | 0.158 | 1.939 | 20.192 | 39.243 |

Expert alignment is approximately 0.8% of the batch-32 prefill diagnostic
window. It contains redundant work, but this trace does not identify it as the
dominant cost. Dispatch, combine and gaps between recorded GPU operations are
larger. The locally idle ranks still execute expert-parallel work; their long
communication intervals can include waiting for the active rank and cannot be
interpreted as high useful PCIe utilization.

Torch profiler perturbs submission timing. Observed gaps mean no recorded GPU
operation within the attributed forward window; they do not establish whether
CPU scheduling, launch submission or dependencies caused the gaps. Transport
intervals include waiting, and fused mailbox combine includes reduction.
Consequently this table is not an unprofiled end-to-end communication fraction
or a hardware PCIe bandwidth measurement. Other model kernels are omitted from
the component columns, and independent rank medians need not add exactly.

### Shared host-memory placement

The host has two NUMA nodes with four GPUs attached to each. GPU worker memory
policies are bound to their corresponding nodes. However, the 24 shared host
arenas contain 39,816 resident 4-KiB pages, totaling 155.53125 MiB, all on node 0.
Before/after snapshots agree. The 192 worker mappings are deduplicated by
backing-file identity and mapped extent; every view agrees and covers the full
arena, so the shared pages are not counted eight times.

The current allocator creates and reserves all arena backing files from rank
0. Worker binding and CUDA host registration do not by themselves give each
shared arena a different physical placement. This observation establishes an
asymmetry in host-memory placement, not its latency penalty. Placement remains
unchanged in this experiment to isolate the routing implementation change. A
future placement experiment can allocate producer-owned buffers on the relevant
NUMA node at initialization, while accounting for cross-node multicast readers.

### Tiled expert alignment

Runtime `d8bb75a` replaces the large-input expert alignment with three GPU kernels:
256-route tile histograms, a bounded prefix scan of the small histogram table,
and stable tile scatter with expert padding. Route IDs are read twice across
all experts instead of twice per expert. Sorting within each tile and prefix
offsets preserve source-row order within every expert. The small alignment
kernel, GEMMs, numerical precision, routing weights, transport and decode fusion
are unchanged. The new path adds one kernel launch and tile metadata allocation;
it introduces no GPU-count readback or host routing decisions.

### Full-model comparison after alignment replacement

The unprofiled run compares runtime `d8bb75a` with a newly measured default
NCCL baseline. Both use the matched eight-GPU configuration above, with 512
input tokens, 32 output tokens, eight 16-token warmup batches and three timed
repetitions per shape. The EP candidate retains four-row combine frames, 32
allocated channels, adaptive active channels, separate prefill Tx/Rx warps,
decode workspace reuse and fused mailbox combine. NUMA placement is unchanged.
NCCL is measured first and EP-Xpress second; this is not an interleaved,
randomized comparison.

All latency values are milliseconds. Completion is the client-observed time
until all requests finish. Last TTFT is when the last request receives its first
token. Steady-window TPOT uses the interval from that point to the first request
finishing, normalized by observed token progress and batch size. Mean request
TPOT instead averages each request's decode interval. Ranges are the observed
three repetitions, not confidence intervals.

| Backend | Concurrent requests | Completion median [min, max] | Last TTFT | Steady-window TPOT | Mean request TPOT | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3063.074 [3048.605, 3081.160] | 190.282 | 92.667 | 92.667 | 10.447 |
| EP-Xpress | 1 | 2846.474 [2812.844, 2851.764] | 185.941 | 85.819 | 85.819 | 11.242 |
| Default NCCL | 32 | 3396.269 [3367.625, 3430.748] | 568.421 | 91.613 | 93.400 | 301.507 |
| EP-Xpress | 32 | 3310.950 [3274.705, 3327.890] | 580.689 | 87.843 | 88.868 | 309.277 |

EP-Xpress reduces completion latency by 7.07% for one request and 2.51% for 32
requests in this run; corresponding output throughput increases are 7.61% and
2.58%. The batch-32 last TTFT is still 2.16% higher than NCCL, and mean request
TTFT is 559.463 versus 499.840 ms, 11.93% higher. The candidate's advantage is
in subsequent decoding, with steady-window TPOT 4.12% lower at batch 32.
The measurements do not establish a prefill latency win.

Equal submitted requests do not produce identical scheduler batches. At batch
32, NCCL splits each rank's four requests into two prefill groups on 8, 6 and 4
ranks across the three repetitions; EP-Xpress does so on 1, 4 and 2 ranks.
All repetitions still contain 32 requests. Arrival and scheduling differences
therefore remain part of the client-observed result.

These are whole-backend comparisons against NCCL, not the isolated speedup of
the alignment replacement. Compared with the preceding warp-specialized EP
run, single-request completion is essentially unchanged (2847.997 versus
2846.474 ms). Batch-32 completion decreases from 3503.642 to 3310.950 ms, but
the previous run has substantially different first-token timing and request
partitioning. The small alignment share in the pre-change profile, unchanged
small decode path, and timing variation prevent attributing that entire
difference to the new alignment. No isolated before/after alignment kernel
timing or repeated interleaved old/new model comparison was performed.

Source manifests match all 47 tracked files before and after the run; the
pinned SGLang patch is unchanged. Input hashes, critical settings and all eight
effective channel counts match the intended configuration. NCCL transport logs
show SHM with no P2P, NET or NVLS records, and all eight GPUs have zero NVLink
payload increments in both runs. A read-only post-run NUMA snapshot again finds
all 24 host arenas on node 0. Both experiment containers exit after validation.

The focused alignment checks pass all 15 stable-layout cases, and all nine
BF16 expert-output checks match the original SGLang runner bit for bit. Four
short full-model sanity prompts match text and all 29 generated token IDs
against NCCL. Their common-prefix log probabilities differ by up to 0.218722
(mean absolute difference 0.014670); this is a limited output sanity check,
not a claim of full-model numerical equivalence or identical timing-run routes.

## Producer-local host-arena allocation

The preceding inspection found every shared arena on NUMA node 0. Runtime
`d7dc52d` moves allocation from the rank-0 coordinator to each payload producer.
Rank 0 creates the private shared directory; each producer creates, reserves
and zero-initializes its own dispatch, combine and decode-mailbox files under
its existing worker memory policy. An initialization-only collective completes
before any worker maps and registers its peers' files. Allocation errors are
collected at this same boundary, before any registered mapping can be used.

Combine files contain separate destination queues, but their payload owner is
still the producer in the filename. Mailboxes follow the same ownership rule.
The library does not derive NUMA nodes from rank numbers or override affinity.
Applications must configure their workers' memory policies before creating a
context; a tmpfs mount policy can also affect allocation. Physical page location
is checked separately from process binding.

This change adds no dispatch/combine synchronization or per-layer NUMA calls.
The communication kernels, frame sizes, channel selection, expert alignment,
workspace reuse and numerical operations are unchanged. Producer-local memory
does not make cross-node consumers local; those readers and their ACK writes
still reach the producer's memory node.

### Measured page placement

The pre-warmup snapshot identifies eight workers, 24 unique arenas and 192
mapping views. Each arena is mapped by all eight ranks, and all views agree on
full resident-page coverage. The check uses physical node page counts, not
the worker's `bind` label. Every dispatch, combine and mailbox arena for
producers 0 through 3 is on node 0; every arena for producers 4 through 7 is on
node 1, matching the configured GPU/worker topology.
Post-timing snapshots confirm the same per-producer placement and full coverage.

| Physical page placement | Node 0 | Node 1 | Total |
|---|---:|---:|---:|
| Previous coordinator allocation | 155.53125 MiB | 0 MiB | 155.53125 MiB |
| Producer allocation, before warmup | 77.765625 MiB | 77.765625 MiB | 155.53125 MiB |

Each producer owns 2,144 dispatch pages, 2,304 combine pages and 529 mailbox
pages, all 4 KiB. Consequently each node has 19,908 pages. The inspection is
outside the performance timing window and does not sample every memory access.

### Full-model result

Both backends use the preceding eight-GPU, BF16, TP/DP/EP8 eager configuration,
512 input tokens and 32 output tokens, with eight 16-token warmup batches and
three timed repetitions at batches 1 and 32. Four-row frames, 32 allocated
channels, adaptive channel selection, valid-count experts, decode workspace
reuse, fused decode and prefill Tx/Rx warp separation remain unchanged.
The default NCCL baseline is measured freshly, first, followed by EP-Xpress.
Neither performance run uses a profiler. Physical page checks happen outside
warmup and timing. The latency definitions are the same as the preceding
alignment comparison; all times below are milliseconds.

| Backend | Concurrent requests | Completion median [min, max] | Last TTFT | Steady-window TPOT | Mean request TPOT | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Default NCCL | 1 | 3105.647 [3040.094, 3119.601] | 192.871 | 93.957 | 93.957 | 10.304 |
| Producer-local EP-Xpress | 1 | 2821.935 [2818.582, 2823.066] | 184.600 | 85.072 | 85.072 | 11.340 |
| Default NCCL | 32 | 3405.740 [3372.632, 3426.461] | 553.176 | 92.154 | 94.626 | 300.669 |
| Producer-local EP-Xpress | 32 | 5222.785 [4187.206, 5240.674] | 2478.962 | 89.333 | 90.849 | 196.064 |

Single-request completion is 9.14% lower than fresh NCCL, but batch-32
completion is 53.35% higher and its output throughput is 34.79% lower. The
batch-32 last TTFT increases from 553.176 to 2478.962 ms, while steady-window
decode TPOT remains 3.06% lower than NCCL. The large regression is therefore
observed before all requests receive their first token, not in the steady
decode window. TTFT includes scheduling, so it is not pure prefill kernel time.

The previous all-node-0 EP run completes in 2846.474 ms at batch 1 and
3310.950 ms at batch 32. The later run's medians differ by
-0.86% and +57.74%, respectively; this is not an isolated causal estimate of
NUMA placement. The single-request ranges overlap, so this
does not establish a material incremental single-request benefit. The new
batch-32 range is entirely slower than the previous EP range, whereas fresh
NCCL batch-32 completion changes by only +0.28%. The current producer-local
implementation is retained as the measured experiment; these results do not
support treating it as a general performance improvement for this workload.

Both backends' three single-request repetitions execute on DP ranks 6, 7 and
0, respectively. Batch-32 prefill partitions still vary. NCCL splits four
requests into two local groups on 8, 8 and 5 ranks; EP-Xpress does so on 8, 5
and 8 ranks. The three EP last-TTFT values are 1429.537, 2478.962 and 2489.123
ms. Fewer local prefill groups alone therefore do not explain the regression.
These are sequential runs with three repetitions, not a randomized,
interleaved old/new allocation ablation; identical inputs do not guarantee
identical later routing or scheduler execution.

The confirmed change is physical producer-page placement. A plausible next
question is how it changes cross-node GPU reads, writes and queue progress;
producer-local allocation only localizes the producer side. This run does not
separate those costs or establish that remote reads are the cause. No new
kernel-phase profiling or additional placement experiments were performed in
that run. The follow-up investigation below revisits its interpretation.

Both runs pass matched-input, configuration, source and host-only transport
checks. All 47 tracked files match the frozen source before and after; the
pinned SGLang patch is unchanged. Critical settings and input hashes also match
the preceding alignment experiment, as do the GPU identities, PCIe topology,
container image and model configuration. All eight ranks report 32 effective
channels. NCCL logs contain SHM and no P2P, NET or NVLS channel records, and
all eight GPUs have zero NVLink payload increments. Four short output prompts
match text and all 29 generated token IDs; common-prefix logprob differences
remain at maximum 0.218722 and mean 0.014670. This is a limited sanity check,
not full-model numerical equivalence. Experiment containers exit after the run.

## NUMA regression investigation

The preceding 57.74% batch-32 difference does not by itself establish a NUMA
transport regression. A follow-up profiles both exact allocator revisions:
`3e9f1d4` with all arenas on node 0, and `620fe82` with producer-local arenas.
Both include the same tiled expert-alignment kernels. Their only runtime
source difference is host-arena allocation; the older pre-alignment revision
is not used as the control.

The comparison retains the same eight GPUs, SGLang patch, model, BF16 precision,
eager execution, four-row frames, 32 allocated channels, adaptive channels,
prefill Tx/Rx warp separation and fused decode. Each service receives eight
16-token warmup batches, followed by the same 32 concurrent 512-token requests
with eight diagnostic output tokens. CPU and GPU tracing is enabled only for
the diagnostic requests. Physical-page checks confirm all 155.53125 MiB on
node 0 for the control and 77.765625 MiB on each node for producer allocation,
with all 192 mapping views agreeing. Both runs pass source and host-only
transport checks, including zero NVLink payload increments.
The producer run additionally inspects compiler-cache filenames, modification
times and small metadata files outside the request's Python execution path;
this observer is not present in the initial all-node-0 run.

### Long gaps also occur with the all-node-0 allocator

Both runs execute two prefill forwards. The trace shows long gaps before the
first use of the tiled alignment kernels in each forward. These gaps are
not accounted for by the recorded local dispatch/combine kernel intervals.
Their host or dependency cause is not identified by GPU gaps alone.

| Diagnostic observation | All node 0 | Producer-local |
|---|---:|---:|
| Client last TTFT, ms | 2578.558 | 2514.051 |
| First scatter's preceding GPU gap, across ranks/forwards, ms | 726.792–758.054 | 738.375–748.150 |

These are profiled diagnostic timings, not replacement end-to-end performance
results. Both diagnostic runs contain long gaps. The earlier large latency
difference is not reflected in a comparable increase of recorded prefill
communication-kernel durations in this comparison.

The submitted input IDs match, but scheduler partitions differ. Only some
rank-local prefill shapes overlap, and even those do not establish identical
peer workloads or expert routing. Dispatch/combine durations include queue
waiting and synchronization; they are not measurements of raw PCIe read/write
latency. This trace therefore cannot establish a small residual NUMA benefit
or penalty, nor separate payload transfer from polling inside those kernels.

### Dynamic capacities and compilation coverage

The large alignment path specializes all three kernels on `CAP`, with the
prefix and scatter kernels also specializing on `TILES`. The adapter obtains
its receive capacity from the scheduler's global token bound. Equal submitted
requests can have different capacities when they arrive in different prefill
groups. For example, the initial diagnostic runs split the 32 requests into
15 plus 17 globally in the all-node-0 run, and 20 plus 12 in the producer run.
The same model and request count therefore do not imply the same compiled
specializations.

The private runner starts each service with a fresh, per-worker Triton cache
in tmpfs. Eight warmup batches cover the shapes actually encountered, not all
possible later partitions. Kernel GPU execution time excludes CPU compilation;
the earlier alignment percentage was measured on the pre-change implementation
and cannot establish the first-use cost of the new tiled kernels.

To distinguish backend compilation from disk-cache lookup and CUDA module
loading, a second diagnostic uses Triton's compilation listener, CPU trace
ranges, JIT cache-miss records and first-module-load records. It retains the
actual prefill warp-specialized transport and records no per-launch data on the
JIT cache-hit path. Events stay in memory until profiler stop. Three separate
profile windows submit identical input IDs to the same service, flushing KV
state between windows while retaining compiled kernels. Nested JIT-miss,
compile and module-load durations are reported separately, never added together
or summed across ranks as request latency.

The all-node-0 service alone reproduces the latency jump:

| Profile window | New alignment capacities | Backend compilations per worker | Backend compile interval union per worker, ms | Client last TTFT, ms |
|---|---|---:|---:|---:|
| 0 | None; all calls hit the in-memory cache | 0 | 0 | 555.030 |
| 1 | 5120, 11264 | 6 | 2009.439–2031.682 | 2596.276 |
| 2 | 2560, 13824 | 6 | 1989.777–2019.806 | 2610.269 |

All three windows use identical submitted input IDs and retain the same
all-node-0 page placement. Each cold window compiles three alignment kernels
for two previously unseen capacities on each of eight workers. These are
actual backend compilations according to Triton's `cache_hit=False` listener,
not merely module loads or disk-cache reads. The first window records no JIT
misses, backend compilations or first-module loads. All eight workers report
complete probe installation and no dropped events.

The CPU markers directly account for the long GPU gaps. For rank 0 in window
1's first prefill, the scatter gap is 737.977 ms, of which 737.327 ms overlaps
its backend compile range on the same process timeline. Prefix and histogram
gaps are 167.952 and 49.866 ms, with 167.328 and 49.512 ms covered by their
backend compilation. The second prefill's scatter gap is 758.777 ms, with
758.102 ms covered. In the no-compilation window, this rank's two prefill GPU
windows are 193.244 and 245.497 ms; their largest observed gaps are only 1.881
and 1.257 ms. Workload partitions still differ, so these examples identify
the source of the long gaps rather than an isolated transport speedup.

About two seconds of per-worker CPU compilation accompany the two-second TTFT
jump with no allocator change. The runtime's exact-capacity specialization is
a real first-use serving cost, and the benchmark's warmup did not isolate it
from placement effects. This evidence makes attribution of the earlier
57.74% difference to NUMA alone unsupported. The historical unprofiled runs lack compilation
markers, so their precise per-request compilation cost cannot be reconstructed.

### Producer-local replication and remaining limits

The producer-local service repeats the same three-window diagnostic with the
same private probe and runner. All eight workers again report complete capture
with no dropped events. It also exhibits both a fully cached path and a new
capacity compilation path:

| Allocation | Profile window | New backend compilations per worker | Client last TTFT, ms |
|---|---:|---:|---:|
| All node 0 | 0 | 0 | 555.030 |
| All node 0 | 1 | 6 | 2596.276 |
| All node 0 | 2 | 6 | 2610.269 |
| Producer-local | 0 | 0 | 501.368 |
| Producer-local | 1 | 6 | 2544.629 |
| Producer-local | 2 | 0 | 502.948 |

Producer window 1 introduces capacities 5632 and 10752. Backend compilation
occupies 2017.721–2033.669 ms per worker; first module/launcher initialization
occupies only 1.667–5.404 ms per worker. The latter is not the seconds-scale
cause. Both cached producer windows record no JIT misses or first-module loads.

Both allocations have approximately half-second TTFT with no compilation and
approximately 2.5-second TTFT when the request introduces two new alignment
capacities. The producer-local cached windows do not reproduce the earlier
large latency regression. Different scheduler partitions, limited samples and
profiler overhead prevent interpreting their smaller numerical difference as
a measured NUMA speedup. These are eight-output-token diagnostic requests,
not new unprofiled 32-output-token performance results or a fresh NCCL benchmark.

Window 0 provides a closer warmed comparison: both versions have the same
input hash, per-request DP-rank assignment and per-rank local prefill token
counts. Distinct histogram grid sizes uniquely identify the two prefill groups
on each rank. Their explicit token counts sum to global capacities 7168 and
9216, with matching previously compiled capacities on all eight workers. This
grouping does not rely on aligning different GPUs' clocks or assuming their
forward ordinals identify the same work.

The following values first accumulate the two prefill forwards on each rank,
then report the median and observed range over the eight ranks, in milliseconds:

| Warmed window-0 metric | All node 0 | Producer-local |
|---|---:|---:|
| GPU forward windows | 439.167 [438.741, 439.494] | 385.729 [385.333, 385.944] |
| Dispatch kernel intervals | 171.931 [168.874, 187.628] | 163.830 [161.908, 176.763] |
| Combine kernel intervals | 199.289 [195.904, 206.240] | 154.879 [152.922, 160.340] |
| Observed GPU gaps | 15.867 [10.545, 16.462] | 14.145 [10.317, 16.514] |

Combine intervals are shorter on every rank in this diagnostic sample; the
large regression does not persist after controlling compilation and these
scheduler shapes. This remains one profiled pair across separate service
launches. Actual expert receive counts, complete routing and CPU load are not
matched observations. It establishes neither a stable unprofiled speedup nor
a PCIe bandwidth improvement. Independent component medians need not sum to
the forward median, and other model operations are omitted from the table.
Producer window 2 has different capacities and is not folded into this pair.

The follow-up isolates a dominant CPU compilation cost. It does not separately
measure the remaining cross-NUMA payload reads, writes, READY polling or ACK
costs inside the communication kernels. Peer progress dependencies can propagate
a worker's late kernel submission; longer communication intervals alone would
not identify slower PCIe access. Physical page placement remains verified in
both versions, but that observation does not establish its transport cost.

No library kernel or allocator is changed during this investigation. The next
implementation priority is to make exact `CAP`/`TILES` runtime scalars, prevent
unnecessary scalar specialization, and precompile the finite geometry buckets
that still require static tensor shapes. Actual tensor bounds and padding
sentinels must remain tied to actual capacity. Merely rounding the kernel's
capacity upward without allocating matching storage would violate the bounds
contract. Subsequent NUMA comparisons should record cache-miss coverage and
compare warmed execution before drawing conclusions about placement.

All four diagnostic runs preserve their frozen source manifests and the pinned
SGLang patch; the compiler-marked pair also matches private probe source hashes.
The final two runs retain verified host-only NCCL transport and zero NVLink
payload increments. Their before/after physical-page checks agree. Diagnostic
containers are removed after trace collection, and the current producer-local
runtime is restored. Raw traces and deployment details remain outside this
repository.


## Runtime-capacity repair and current prefill group diagnostics

The route alignment kernels now use non-specializing runtime CAP/TILES and a
bounded prefix geometry independent of exact capacity. A focused CUDA check
switches capacities 129, 5120, 5632, 7168, 9216, 10752, 11264 and 32769 after
warming the first shape: each of the three large-route kernels retains exactly
one in-memory variant for the fixed dtype/expert/block configuration. Fifteen
alignment cases pass, including empty valid counts and crossing the prefix
loop boundary; nine BF16 expert cases remain bitwise equal to the reference.
Initial compilation and other static model configurations remain possible.

The opt-in profiler now measures the actual warp-specialized prefill combine,
plus dispatch, without reverting to the sequential combine implementation.
Fused decode and idle operations continue unchanged and are excluded. Two
focused multi-GPU warp tests pass with instrumentation alternating on/off,
checking numerical results, payload byte counts and drained slots. Full-model
capture completes on all eight workers without dropped operations or errors.
The private runner uses a separate temporary filesystem after a full system
filesystem interrupted an earlier diagnostic build; failed runs are excluded.

For the batch-32 diagnostic, each rank executes 96 dispatches and 96 combines
(two prefill forwards across 48 layers). The median per-rank/trial critical-CTA
sum is 174.906 ms for dispatch and 160.375 ms for combine. The slowest CTA is
selected independently per operation; these sums exclude init/final kernels
and are not HTTP latency. Dispatch payload reads account for 93.403 ms of this
view. Combine Rx reads account for 81.301 ms; Tx credit check/wait accounts for
92.050 ms and overlaps Rx work. These quantities must not be added together.

An alternative aggregation sums work over remote pair CTAs within each rank,
then reports median fractions across ranks/trials. Within measured Tx phases,
credit check/wait is 52.7%, payload writes 19.3%, READY publication/fence 19.2%,
and final drain 7.2%. Within Rx phases, payload reads are 67.8%, READY check/wait
19.5%, and ACK publication/fence 10.6%. Metadata accounts for the remainder.
These are group-work fractions, not fractions of end-to-end latency. The
four-row configuration sends a median 146659 frames per rank in this capture,
including terminals. This motivates a frame-size ablation while holding
channel count fixed. Counter bytes are requested payload accesses, not physical
PCIe traffic. Instrumentation adds barriers; scheduler partitions also vary,
so profiled on/off TTFT cannot isolate instrumentation overhead here.

### Unprofiled full-model results after the repair

The comparison keeps the same eight GPUs, BF16 model, pinned SGLang version,
TP/DP/EP8, Triton experts, FlashInfer attention, eager execution and producer-local
host allocation. All three configurations use identical prompt hashes for
512 input / 32 output tokens, eight warmups per batch size and three measured
repeats. The two EP runs differ only in combine frame rows (4 versus 16);
allocated channels remain 32 and fused decode remains enabled. The NCCL
baseline is the framework's `none` MoE communication backend. All runs verify
SHM transport, no P2P/NVLS/NET channels and zero NVLink payload increments.

Values below are median [min, max]. Decode TPS is the measured aggregate output
rate between all requests receiving their first output and the first request
finishing; prefill is excluded. TTFT includes HTTP and scheduling overhead.

| Batch | Metric | Default NCCL | EP, 4-row frame | EP, 16-row frame |
|---:|---|---:|---:|---:|
| 1 | Mean request TTFT (ms) | 191.927 [191.459, 194.493] | 186.217 [183.962, 187.826] | 187.531 [184.888, 187.984] |
| 1 | Last request TTFT (ms) | 191.927 [191.459, 194.493] | 186.217 [183.962, 187.826] | 187.531 [184.888, 187.984] |
| 1 | Decode TPS | 10.765 [10.652, 10.769] | 11.677 [11.642, 11.839] | 11.669 [11.529, 11.736] |
| 1 | Batch completion (ms) | 3071.167 [3070.628, 3104.803] | 2840.967 [2802.551, 2850.798] | 2844.215 [2826.439, 2877.059] |
| 32 | Mean request TTFT (ms) | 574.743 [433.900, 605.554] | 517.590 [403.367, 534.611] | 424.643 [404.959, 526.324] |
| 32 | Last request TTFT (ms) | 614.007 [542.036, 641.426] | 536.745 [480.571, 545.854] | 483.814 [478.065, 536.449] |
| 32 | Decode TPS | 348.611 [345.112, 350.768] | 355.988 [350.545, 358.791] | 356.903 [354.567, 360.153] |
| 32 | Batch completion (ms) | 3435.291 [3357.356, 3504.043] | 3307.957 [3287.186, 3319.251] | 3265.553 [3236.257, 3288.925] |

The 16-row candidate improves median last-request TTFT by 2.29% at B1 and 21.20%
at B32 versus this NCCL sample. Decode TPS improves by 8.40% and 2.38%,
respectively. Relative to the repaired four-row path, B32 last-request TTFT
falls from 536.745 to 483.814 ms; B1 and decode show no additional clear benefit.
The TTFT ranges of the two EP configurations overlap. Three repetitions and
uncontrolled scheduler partitions do not establish a stable 9.9% frame-only
speedup. There is no after-tuning phase capture establishing a change in raw
PCIe bandwidth. The larger frame is a measured candidate for this workload,
not a universal default; `presets/host_sm_bf16_ep8.sh` beside the adapter README
records its settings without changing the core library's defaults.

Four short greedy full-model sanity prompts produce identical tokens and text
between NCCL and the candidate. Log probabilities are not bitwise identical
(maximum absolute difference 0.219 on the shared generated prefixes); this is
not an exhaustive numerical-equivalence claim. Raw logs, traces, access details
and deployment paths remain outside this repository.


## Vector host reads and queue-depth ablation

Prepared warp combine now offers an opt-in 128-bit cache-volatile host read
path with bounded independent-load expansion. The tested expansion is four;
scalar mode is retained with the same protocol and BF16 reduction order.
Misaligned row widths, frame payloads and output views fall back to scalar
reads. Focused multi-GPU checks cover normal model width, vector tails,
unaligned addresses, empty/asymmetric traffic, profile on/off and slot reuse.
An initial four-case check passed; after the address-alignment guard, three
targeted cases passed. Compiled vector code contains four independent 128-bit
system loads and no local-memory spills. The final vector build uses 86 registers versus 70 in the
scalar comparison build; all eight model workers retain 32 effective channels.

The full-model experiment retains the same eight GPUs, BF16 model, TP/DP/EP8,
expert/attention backends, eager execution and producer-local NUMA allocation
as above. Each configuration uses eight warmups and five timed repetitions
for 512 input / 32 output tokens. All runs verify SHM, zero P2P/NVLS/NET channels
and zero NVLink payload counter increments. Both dispatch and combine use the
selected slot count; the decode mailbox is unchanged. The shared host pool is
347.531 MiB with two slots and 669.531 MiB with four. The experiment uses a
common 1 GiB budget ceiling; actual allocation follows the slot count.

Values below are medians. The final four-slot vector run includes the complete
address-alignment guard. Two-slot exploratory results precede that guard; the
model tensors and payloads are aligned in all runs. The scalar build excludes
the vector branch entirely. Decode TPS uses the same steady-state definition
as above, excluding prefill. TTFT includes HTTP and scheduler effects.

| Batch | Metric | NCCL | Scalar / 2 slots | Vector x4 / 2 slots | Scalar / 4 slots | Vector x4 / 4 slots |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TTFT (ms) | 192.549 | 185.926 | 185.593 | 186.324 | 184.142 |
| 1 | Decode TPS | 10.739 | 11.721 | 11.700 | 11.693 | 11.776 |
| 32 | Mean request TTFT (ms) | 568.963 | 445.947 | 471.654 | 493.518 | 457.488 |
| 32 | Last request TTFT (ms) | 607.648 | 494.604 | 496.833 | 521.048 | 491.762 |
| 32 | Decode TPS | 352.412 | 357.727 | 356.432 | 361.636 | 355.933 |
| 32 | Batch completion (ms) | 3412.189 | 3274.518 | 3235.258 | 3253.878 | 3264.367 |

The final vector/four-slot candidate lowers B32 last-request TTFT by 19.07%
versus NCCL, but only 0.57% versus scalar/two-slot EP. Its B32 decode TPS is
0.50% below the latter. This does not demonstrate a new repeatable end-to-end
improvement. B32 TTFT ranges overlap: scalar/two-slot 493.881–537.590 ms,
vector/two-slot 445.202–519.063 ms, scalar/four-slot 479.356–528.273 ms, and
final vector/four-slot 449.573–511.167 ms. An earlier vector/four-slot run had
470.017 ms TTFT and 363.464 TPS, versus 491.762 ms and 355.933 TPS on the final
run. These are observed ranges, not confidence intervals, and request scheduling
partitions were not fixed.

Consequently the native loader defaults to scalar reads and the existing
preset keeps two slots. `EP_XPRESS_PREFILL_READ_UNROLL=4` enables the experimental
vector path before worker startup. Values 1, 2 and 8 are available tuning
geometries, but were not measured in this model experiment. The setting is
part of the native build cache key and does not specialize on token counts.
Four-slot experiments require a sufficient host-pool budget. No cumulative
credit protocol, dedicated control warp or prefill receive/reduce fusion is
introduced by this change.

Four short greedy prompts match exactly between scalar and vector EP runs,
including token log probabilities. This is a bounded sanity check, not an
exhaustive model-equivalence claim. Private logs, traces, manifests and launch
scripts remain outside the public repository.


### Read-only phase diagnostics at the current frame size

A separate B32 capture fixes 16-row frames and four slots on both sides, with
scalar sends and either scalar reads or vector reads expanded four times.
Each capture verifies all eight workers, two instrumented trials, 96 dispatch
and 96 combine operations per rank, and no dropped records or status errors.
Median requested combine read bytes per rank/trial are identical
(1,832,900,608 bytes). These are software payload counts, not bus counters.

The following values are median per-rank/trial sums of the slowest CTA selected
independently for each operation. Tx and Rx overlap; rows must not be added.

| Diagnostic quantity (ms) | Scalar reads | Vector reads x4 |
|---|---:|---:|
| Dispatch critical-CTA elapsed sum | 182.530 | 239.830 |
| Dispatch payload read component | 92.973 | 94.458 |
| Combine critical-CTA elapsed sum | 161.661 | 138.236 |
| Combine receive payload component | 75.158 | 14.845 |
| Combine Tx credit check/wait | 3.229 | 3.334 |
| Combine Rx READY check/wait | 57.342 | 109.302 |
| Combine Tx final ACK drain | 92.824 | 60.664 |

Across all remote pair CTAs, median receive payload group-work per requested
GiB falls from 9917.488 to 2709.899 ms/GiB. This sums concurrent worker times;
it is not a physical bandwidth estimate. The receive payload fraction of Rx
work falls from 67.99% to 25.53%, while READY check/wait rises from 28.55% to
57.70%. Credit checks/waits occupy only 2.16% and 3.44% of measured Tx work at
four slots, respectively. Final drain remains substantial. Queue capacity can
let producers finish publishing sooner without making consumers finish sooner.

The read optimization works within its measured phase, but the whole pipeline
has not improved proportionally. Dispatch code was unchanged, yet its observed
elapsed/wait time increased, so this is not an isolated bandwidth experiment.
Scheduling, inter-rank arrival and overlap affect phase times; instrumentation
adds group barriers. Neither these counters nor the unprofiled samples establish
PCIe saturation. The next tested candidate therefore vectorizes the sender's
payload copy as well, retaining the same READY/fence/ACK protocol.


### Vectorizing both directions, with the original two-slot capacity

The follow-up applies the same bounded vector-copy scheduling to Tx. GPU
vectors are gathered from the existing owner-indexed rows, then written with
volatile 128-bit host stores; all writers still fence and synchronize before
READY. Read and write expansion have independent build-time switches. Six
focused multi-GPU cases pass with both enabled, covering vector/tail paths,
unaligned source and destination views, non-aligned frame payloads, empty and
asymmetric traffic, and two/four-slot reuse. The normal transport kernel uses
90 registers without spills and all eight model workers retain 32 channels.

The complete model uses the same five-repeat experiment settings as the table
above, with both expansion factors set to four and the original two slots.
No additional shared host memory is required. Values are median [min, max].

| Batch | Metric | NCCL | Scalar EP / 2 slots | Vector read+write x4 / 2 slots |
|---:|---|---:|---:|---:|
| 1 | TTFT (ms) | 192.549 [189.091, 192.903] | 185.926 [184.865, 186.503] | 185.663 [185.017, 187.974] |
| 1 | Decode TPS | 10.739 [10.661, 10.908] | 11.721 [11.617, 11.805] | 11.680 [11.536, 11.891] |
| 32 | Mean request TTFT (ms) | 568.963 [496.359, 626.142] | 445.947 [444.941, 527.542] | 439.373 [363.679, 487.595] |
| 32 | Last request TTFT (ms) | 607.648 [565.040, 640.562] | 494.604 [493.881, 537.590] | 477.385 [443.419, 504.735] |
| 32 | Decode TPS | 352.412 [349.331, 353.596] | 357.727 [355.257, 359.469] | 360.802 [356.362, 361.111] |
| 32 | Batch completion (ms) | 3412.189 [3371.846, 3478.319] | 3274.518 [3247.873, 3305.880] | 3218.068 [3195.644, 3247.953] |

Versus scalar EP at the same two-slot capacity, median B32 last-request TTFT
falls by 3.48% and decode TPS rises by 0.86%.
Versus NCCL, the corresponding changes are -21.44% and +2.38%. B1 is essentially
unchanged versus scalar EP. Four of five matching B32 prompt batches have
lower last-request TTFT, but ranges overlap and scheduler partitions vary.
This is a modest measured candidate, not a demonstrated universal speedup.
Decode's mailbox kernel was not changed, so its small throughput difference
must not be attributed directly to the prefill copy optimization.

The general defaults remain scalar reads/writes and two slots. The separate
`adapters/sglang/presets/host_sm_bf16_ep8_vector.sh` opt-in preset enables both
expansions without increasing memory. The original preset explicitly resets
both switches to zero. No deeper-queue default is adopted. Four short greedy
prompts retain identical tokens/text/logprobs versus scalar EP; versus NCCL,
tokens/text match and logprob differences retain the previously reported bound.


The final two-slot read+write candidate also completes a B32 prefill capture
on all eight ranks with no dropped records or errors. Median critical-CTA
elapsed sums are 166.737 ms for dispatch and 125.234 ms for combine. Within the
selected combine CTAs, payload write/read components are 34.734/19.588 ms,
READY check/wait 87.822 ms, Tx credit check/wait 28.559 ms, final ACK drain
28.359 ms, publication/fence 20.843 ms, and ACK publication 7.717 ms. These
components overlap between Tx and Rx; sums are not request latency. This
capture uses two slots, whereas the read-only diagnostics above use four,
so their differences do not isolate the write optimization alone.

The remaining measured waits motivate examining producer arrival, publication
and buffer-lifetime scheduling before assuming that more CTAs or deeper rings
will improve inference. Prefill still materializes per-peer GPU scratch before
a separate reduction. No change to that dependency structure or to required
memory-ordering operations was made in this experiment.

### Vectorizing selected dispatch reads

The next candidate changes only dispatch payload reads: 128-bit vectors and
four independent expanded iterations, using the existing selected wire-row
indices. It keeps multicast ownership, GPU compaction, two slots, 32 configured
channels, 16-row combine frames, combine read/write expansion four and all
READY/fence/ACK boundaries. Hidden sizes or addresses that are not vector
aligned use the scalar fallback. The independent build-time switch is
`EP_XPRESS_DISPATCH_READ_UNROLL`; its general default remains zero.

Five focused three-GPU cases pass. They cover sparse multicast and same-rank
expert deduplication, a delayed nonreader, four epochs with alternating active
channel counts, empty/asymmetric routes, vector and tail copies, misaligned
frame payloads, non-vector hidden sizes and shifted output storage. The payload
pattern varies across hidden positions. Output/inverse/owner mapping and the
weighted round trip agree exactly with the reference.

The adjacent full-model runs enable or disable only the dispatch read switch;
launch environment differences were checked, and all ten measured input hashes
match. Settings remain BF16, eager, EP8/TP8/DP8, 512 input tokens, 32 output
tokens, eight warmups per batch size and five measured repeats. NCCL numbers
reuse the earlier verified same-configuration run. All runs report SHM, no
P2P/NVLS/network transport channels, and zero available NVLink payload counters.
Values below are median [min, max].

| Batch | Metric | Default NCCL | EP combine-vector / dispatch scalar, rerun | EP combine-vector / dispatch vector x4 |
|---:|---|---:|---:|---:|
| 1 | TTFT (ms) | 192.549 [189.091, 192.903] | 187.298 [184.956, 187.432] | 186.202 [183.735, 188.111] |
| 1 | Decode TPS | 10.739 [10.661, 10.908] | 11.589 [11.518, 11.730] | 11.712 [11.611, 11.839] |
| 32 | Mean request TTFT (ms) | 568.963 [496.359, 626.142] | 410.185 [383.182, 485.525] | 412.489 [333.331, 437.187] |
| 32 | Last request TTFT (ms) | 607.648 [565.040, 640.562] | 459.015 [446.138, 502.974] | 431.148 [381.754, 444.179] |
| 32 | Decode TPS | 352.412 [349.331, 353.596] | 356.913 [349.479, 357.862] | 359.125 [358.481, 361.447] |
| 32 | Batch completion (ms) | 3412.189 [3371.846, 3478.319] | 3270.255 [3225.468, 3274.199] | 3170.257 [3131.883, 3204.697] |

Against the adjacent EP control, B32 last-request TTFT falls 6.07%, and all
five matching prompt batches improve on this metric. Mean request TTFT changes
by +0.56%, effectively unchanged in these variable admission/scheduling runs.
Decode TPS increases only 0.62%; no substantial decode improvement is established.
B1 changes are small and timing ranges overlap. Versus the reused NCCL baseline,
B32 last-request TTFT falls 29.05% and decode TPS rises 1.90%. These are bounded
full-model results, not a general throughput claim or confidence interval.

Four short greedy prompts preserve tokens, text and logprobs exactly versus
both the adjacent control and the preceding combine-vector candidate. Against
NCCL, tokens/text match on all four, with maximum absolute logprob difference
0.218723 on their common generated prefixes. This is a limited output sanity
check, not exhaustive numerical equivalence. The experimental vector preset
now enables dispatch expansion four as well; the original preset explicitly
resets it to zero.

The candidate's ordinary selective dispatch kernels use 72 registers, zero
stack/local storage and four vector loads/stores in their expanded path for
both int32 and int64 expert IDs. All eight profiled workers retain 32 channels.
The diagnostic capture contains two instrumented B32 trials, each with 96
dispatch and 96 combine operations per rank, with no dropped records or status
errors. Median requested payload bytes match the preceding combine-vector
capture: 1,955,778,560 dispatch read bytes and 1,832,900,608 combine read bytes
per rank/trial.

| Diagnostic quantity (ms) | Previous dispatch scalar | Dispatch vector x4 |
|---|---:|---:|
| Dispatch critical-CTA elapsed sum | 166.737 | 92.925 |
| Dispatch selected payload read component | 93.885 | 33.599 |
| Combine critical-CTA elapsed sum | 125.234 | 119.873 |
| Combine receive payload component | 19.588 | 18.577 |
| Combine send payload component | 34.734 | 34.705 |
| Combine publish/fence component | 20.843 | 21.686 |
| Combine READY check/wait | 87.822 | 93.052 |
| Combine Tx credit check/wait | 28.559 | 25.780 |
| Combine final ACK drain | 28.359 | 29.747 |

Each quantity is the median across rank/trial sums of the slowest CTA selected
per operation. Tx/Rx overlap; these rows must not be added or subtracted from
HTTP TTFT. Instrumentation adds synchronization, and the captures are separate
runs. Summed concurrent dispatch payload-read work per requested GiB falls
from 1645.038 to 617.386 ms/GiB; this is not a physical PCIe bandwidth figure.

The remaining combine Rx group-work shares are approximately 53.64% READY
check/wait, 28.27% payload reads and 15.71% ACK publication. Tx shares are
37.31% payload stores, 27.12% publication/fence, 13.69% credit checks/waits and
20.07% final drain. Faster dispatch therefore does not eliminate the combine
producer/publication and buffer-turnover bottleneck. These percentages cannot
by themselves distinguish late expert-compute completion from slow production
or publishing, and no required synchronization was removed.

A separate offline analysis of the exported Chrome traces matches native
(epoch, dispatch-operation) identities and all eight rank-local CUDA launch
correlations. On the declared shared Chrome time base, the per-layer combine
start spread is about 0.360 ms median / 0.850 ms P90 after the change, compared
with 0.368 / 0.853 ms in the preceding capture. This coarse skew persists;
GPU globaltimer values were not subtracted across devices. Cross-device clock
calibration uncertainty is unquantified, so these spreads do not provide an
exact causal decomposition of READY wait. Finer producer milestones or
compute/communication dependency measurements are needed before changing the
publication or final-drain protocol.

### Last-data marker instead of a separate combine terminal

The next isolated change uses the final nonempty warp-combine payload frame to
carry LAST. Empty queues still send a terminal. Payload writers retain their
system fences before READY; readers retain their completion/fence/ACK ordering;
Tx retains its final drain of every published slot. Dispatch, fused decode,
BF16 arithmetic, frame layout size and pool capacity are unchanged. Set
`EP_XPRESS_COMBINE_LAST_DATA=1` before worker startup to enable it; the general
default is zero, and the setting participates in the native cache key.

Six focused CUDA cases pass against the legacy pair-combine reference, with
bitwise-identical output and remote scratch. Tests verify exact frame counts,
LAST payload lengths, matching READY/ACK on every slot, full and partial final
frames, empty/local-only queues, asymmetric traffic, delayed readers, changing
active channel counts across six epochs, and scalar/vector/alignment tails.

Adjacent full-model runs differ only in the LAST switch. They retain vector
read/write expansion four, dispatch read expansion four, two slots, 32 channels,
16-row combine frames and the same host allocation. BF16/eager EP8/TP8/DP8 and
512-input/32-output-token workloads remain fixed, with eight warmups per batch
size and five measured repeats. All ten input hashes match. The NCCL column
reuses the verified same-configuration baseline. All runs retain SHM-only
reported NCCL transport and zero available NVLink payload counters.

Values are median [min, max].

| Batch | Metric | Default NCCL | EP LAST off, adjacent rerun | EP LAST on |
|---:|---|---:|---:|---:|
| 1 | TTFT (ms) | 192.549 [189.091, 192.903] | 187.650 [184.787, 191.975] | 186.428 [184.872, 187.811] |
| 1 | Decode TPS | 10.739 [10.661, 10.908] | 11.648 [11.554, 11.746] | 11.650 [11.634, 11.755] |
| 32 | Mean request TTFT (ms) | 568.963 [496.359, 626.142] | 399.482 [377.849, 437.299] | 365.812 [333.833, 389.439] |
| 32 | Last request TTFT (ms) | 607.648 [565.040, 640.562] | 424.009 [412.293, 444.586] | 404.976 [382.288, 418.397] |
| 32 | Decode TPS | 352.412 [349.331, 353.596] | 359.198 [355.576, 359.712] | 359.932 [359.142, 360.891] |
| 32 | Batch completion (ms) | 3412.189 [3371.846, 3478.319] | 3177.843 [3162.985, 3232.754] | 3149.511 [3121.211, 3164.770] |

B32 last-request TTFT falls 4.49% and mean-request TTFT falls 8.43% against the
adjacent EP control. All five matching input batches improve on last-request
TTFT, but timing ranges overlap and admission partitions remain variable;
these observed ranges are not confidence intervals. Decode TPS differs by
only +0.20%, with no decode kernel change. B1 is essentially unchanged.
Against the reused NCCL baseline, B32 last-request TTFT is 33.35% lower and
decode TPS is 2.13% higher. TTFT improvements must not be described as equivalent
throughput gains.

Four short greedy prompts retain identical text, tokens and logprobs versus
both the adjacent control and the preceding dispatch-vector version. The
experimental `host_sm_bf16_ep8_last_data.sh` preset enables the change on top
of the vector preset. General defaults and both earlier presets retain the
separate terminal. This is a measured optional configuration, not a universal
optimum.

The B32 profile completes all eight workers, four off/on trials (32 rank files),
and 96 captured dispatch plus 96 combine operations per instrumented rank/trial,
with no dropped records or status errors. Effective channels remain 32. The
ordinary warp-combine kernel uses 96 registers (previously 90), with no stack or
local storage and unchanged negotiated channel capacity.

All 344,064 remote queue records in the two instrumented trials satisfy
`send_frames == max(1, ceil(send_payload_rows / 16))`, and each queue sends and
receives exactly one LAST marker. There are 43,426 empty queues. Measured total
frames are 608,077; applying the old separate-terminal rule to these *same*
queue payload lengths would require 908,715 frames, so 300,638 empty-terminal
frames are avoided (33.08%). The old count here is derived from the protocol,
not a separately executed identical-route replay. Payload bytes are unchanged
by attaching LAST.

Comparison with the preceding vector-dispatch capture gives the following
median rank/trial critical-CTA sums. Tx and Rx overlap; rows are not additive.

| Diagnostic quantity (ms) | Previous separate-terminal capture | LAST capture |
|---|---:|---:|
| Dispatch elapsed | 92.925 | 105.385 |
| Dispatch payload reads | 33.599 | 33.460 |
| Combine elapsed | 119.873 | 126.799 |
| Combine payload writes | 34.705 | 33.896 |
| Combine publish/fence | 21.686 | 19.549 |
| Combine payload reads | 18.577 | 18.859 |
| Combine READY check/wait | 93.052 | 93.369 |
| Combine Tx credit check/wait | 25.780 | 10.494 |
| Combine final ACK drain | 29.747 | 50.541 |

These captures do **not** establish a shorter communication kernel. Scheduler
partitions differ: earlier instrumented prefill forwards are uniform across
ranks, while the new capture includes heterogeneous splits and ranks with no
local input in the first forward. Median total requested dispatch/combine
payload bytes match, but matching totals do not imply matching per-frame routes
or arrival times. Diagnostic group barriers also affect scheduling. Credit
wait decreases while final-drain wait increases in the observed breakdown;
this must not be attributed wholly to the protocol change. The defensible
result is fewer control frames and a modest TTFT improvement in the separate,
uninstrumented adjacent comparison.

For prioritization, the preceding Chrome trace's 96 combine operations have
per-rank median cumulative init, transport and finalize GPU durations of
0.950, 119.083 and 1.704 ms. This makes init/finalize fusion a lower priority for
this captured workload than production/publication and arrival dependencies.
It is not a claim that these kernel sums equal request critical-path latency.


## Decode mailbox vector read/reduce/store experiment

The optional `EP_XPRESS_DECODE_PEER_BATCH=2` implementation reads two rank
positions ahead using cache-volatile 128-bit host loads. It retains vector
ownership through local GPU partial loads and final GPU output stores. The
scalar path remains the default (`0`), including all supplied presets. The
receive arithmetic, intermediate BF16 rounding, producer writes, READY/ACK
protocol, workspace capacity and NUMA allocation are unchanged. Widths without
complete vectors and misaligned GPU views fall back to scalar reads/reduction.
No CPU payload relay or new metadata communication is introduced.

Five focused CUDA cases pass with bitwise equality against the existing pair
reference: hidden widths 64, 258, 264 and 2048, plus a shifted 2048-wide GPU
view. Cases retain sparse contributors, an unrelated delayed rank, repeated
mailbox reuse, changing active channels and reverse-order route handles.
The tested vector build uses 40 registers versus the scalar build's 46, with
zero stack/local-memory usage. SASS confirms 128-bit host loads and an aligned
128-bit GPU output store. Other nonzero peer-batch sizes are tuning options,
not measured performance recommendations.

Full-model tests keep the same eight GPUs, Qwen3-30B-A3B-Instruct-2507 BF16,
SGLang v0.5.21, TP8/DP8/EP8, Triton experts, eager execution, 512 input tokens
and 32 output tokens. Both backends disable P2P/NVLink. NCCL was rerun in this
series, rather than borrowed from an earlier measurement. Each shape receives
eight warmups and five timed repeats; all ten input hashes match. The EP
control and final vector candidate use the prior vector-prefill/LAST preset
settings: 32 allocated channels, 16-row frames, two slots and prefill/dispatch
vector expansion four. Automatic decode selection uses one channel at B1
and four at balanced B32. The decode channel override affects both dispatch
and combine, so channel experiments are not isolated combine-only changes.

Values below are median [min, max]. TTFT is the last request's first-token
latency. Decode TPS is aggregate throughput inside the steady decode window,
not maximum sustained serving throughput.

| Implementation | Batch | TTFT (ms) | Steady TPOT (ms) | Decode TPS |
|---|---:|---:|---:|---:|
| Default NCCL | 1 | 191.831 [190.130, 194.547] | 92.538 [92.012, 95.015] | 10.806 [10.525, 10.868] |
| Default NCCL | 32 | 611.968 [559.296, 631.877] | 90.840 [90.504, 91.482] | 352.268 [349.796, 353.575] |
| EP scalar, auto channels | 1 | 185.633 [184.164, 186.045] | 85.095 [84.745, 85.356] | 11.752 [11.716, 11.800] |
| EP scalar, auto channels | 32 | 421.146 [374.875, 438.463] | 88.887 [88.335, 88.975] | 360.008 [359.650, 362.258] |
| EP vector, auto channels | 1 | 184.654 [183.588, 187.664] | 85.417 [84.382, 86.152] | 11.707 [11.607, 11.851] |
| EP vector, auto channels | 32 | 416.945 [381.569, 428.407] | 89.123 [88.471, 90.171] | 359.055 [354.881, 361.702] |

The final auto-channel vector candidate is 0.38% slower than the scalar control
at B1 and 0.26% slower at B32 by median decode TPS, with overlapping observed
ranges. It provides no demonstrated incremental throughput gain. Its B32 TPS
is 1.93% above the new NCCL baseline; that is the existing library advantage,
not a gain attributable to this vector experiment. The scalar control remains
2.20% above NCCL. The TTFT ranges also overlap; this decode experiment is not
evidence of a new prefill optimization. Four short greedy prompts match the
scalar control exactly in text, token IDs and sampled logprobs. This output
check is limited to those prompts, not exhaustive model routing equivalence.

An earlier receive-vector-only candidate left final BF16 output as scattered
scalar stores. At B32 its median TPS was 348.109 with auto/four channels,
359.255 with two channels and 341.623 with eight channels. These results
reject the assumption that more CTAs or vector host loads alone necessarily
improve decode. Completing vector GPU stores restored performance near the
scalar control; this does not establish a large new speedup.

A final two-channel rerun with vector GPU stores gives B32 decode TPS
358.656 [357.842, 359.909], TPOT 89.222 [88.911, 89.425] ms, and last-request
TTFT 430.659 [384.489, 456.272] ms. It does not improve on the scalar control
or the final auto-channel candidate. No new decode preset is promoted.

The final auto-channel candidate was profiled separately after uninstrumented
measurements. All eight B32 traces resolve, each with eight local DECODE
forwards. Per-rank GPU forward windows total about 826 ms. Median observed
device idle fraction is 58.31% (range 36.46% to 73.75%); dispatch kernel
occupancy is 22.75%, and fused mailbox occupancy is 14.35%. These are per-rank
GPU-window fractions, not HTTP latency fractions, and their medians should
not be added. Mailbox time includes waiting and reduction, not pure PCIe
reads. Profiling increases step latency and affects rank scheduling; B1 local
IDLE records are kept separate and never guessed to be global decode. An
earlier eight-channel trace has a very different waiting/idle distribution,
so its communication fraction is not an ablation result for payload traffic.

The observed gaps and rank variation motivate checking eager submission and
rank arrival skew before treating payload bandwidth as the only bottleneck.
The adapter currently immediately waits on each async dispatch/combine while
the core crosses between producer/consumer and transport streams. A future
same-stream synchronous fast path could remove redundant event dependencies
for that use case while retaining the general asynchronous API. Neither that
fast path nor a specific causal CPU bottleneck is established by this trace.


## Direct-stream eager decode submission

The candidate adds `dispatch_on_stream` and `combine_on_stream` and selects
these APIs for eligible SGLang decode with `EP_XPRESS_DECODE_SAME_STREAM=1`.
Consecutive calls use the compute stream's ordering instead of creating a
completion event and Pending object for each operation. Borrowed-workspace
allocator lifetimes are recorded once per execution stream. External inputs
retain lifetime recording. Stream changes, workspace consumers on other
streams, mapped-storage cleanup and GPU READY/ACK dependencies remain ordered.
Native mailbox combine inserts a transition event if its stream changes.
The kernels' payload schedule, reduction arithmetic and routing are unchanged.

The comparison uses the same eight GPUs, Qwen3-30B-A3B-Instruct-2507 BF16,
SGLang v0.5.21, TP8/DP8/EP8, Triton experts, FlashInfer attention, eager execution,
512-token inputs and 32-token outputs. Both backends disable GPU P2P/NVLink,
custom all-reduce and FlashInfer all-reduce fusion. Available NVLink payload
counters remain zero and NCCL logs verify SHM-only transport. Each batch size
receives eight warmups and five timed repeats. New EP, NCCL and the EP async
control are run sequentially on the same machine.

Both EP paths use 32 allocated channels, 16-row prefill combine frames, two
slots, vector prefill/dispatch expansion four, payload-bearing LAST frames,
scalar decode mailbox reads, workspace reuse and fused decode combine. Auto
selection uses one active channel at B1 and four at balanced B32. The only
environment difference between the two EP runs is the same-stream switch.
All ten input hashes match across runs. The NCCL comparisons pass the saved
hardware, model, framework-setting and transport checks.

Values are median [min, max]. TTFT is the last request's first-token latency,
including admission and scheduling. Decode TPS is aggregate throughput in the
steady decode window, not maximum sustained serving throughput.

| Implementation | Batch | TTFT (ms) | Steady TPOT (ms) | Decode TPS |
|---|---:|---:|---:|---:|
| Default NCCL | 1 | 191.031 [188.442, 192.573] | 92.519 [91.858, 92.939] | 10.809 [10.760, 10.886] |
| Default NCCL | 32 | 557.593 [551.518, 622.859] | 91.966 [91.030, 93.776] | 347.954 [341.240, 351.534] |
| EP async control | 1 | 186.922 [183.131, 188.808] | 85.823 [84.247, 86.069] | 11.652 [11.619, 11.870] |
| EP async control | 32 | 397.048 [375.067, 444.563] | 89.843 [89.318, 91.179] | 356.176 [350.957, 358.269] |
| EP direct stream | 1 | 181.343 [179.145, 184.024] | 80.872 [79.739, 82.119] | 12.365 [12.178, 12.541] |
| EP direct stream | 32 | 412.565 [371.392, 437.224] | 83.657 [83.515, 84.501] | 382.516 [378.693, 383.167] |

Relative to the adjacent EP async control, direct-stream decode TPS improves
6.12% at B1 and 7.40% at B32; TPOT falls 5.77% and 6.89%. The five-repeat
observed decode ranges do not overlap. Relative to the fresh NCCL baseline,
decode TPS improves 14.40% and 9.93%. These are separate comparisons: the
NCCL advantage includes previously implemented EP-Xpress optimizations.

B32 last-request TTFT is 3.91% higher than the async control, with overlapping
ranges. Prefill retains its existing asynchronous path, so this experiment
does not establish a prefill improvement. Overall batch completion time is
2.688 s at B1 and 3.001 s at B32, versus NCCL's 3.058 s and 3.403 s.

Twenty-four focused host API checks and two two-GPU stream/workspace cases
pass. The GPU cases cover fused and non-fused combine, repeated workspace
reuse, changes between direct and asynchronous APIs, different producer
streams, delayed consumers on another stream, and close before final host
synchronization. Four greedy full-model prompts match the adjacent EP control
exactly in text, generated token IDs and sampled logprobs. These checks are
bounded correctness evidence, not exhaustive numerical equivalence.

Enable the measured candidate with
`adapters/sglang/presets/host_sm_bf16_ep8_same_stream.sh`, or set
`EP_XPRESS_DECODE_SAME_STREAM=1` on the same host-SM configuration. The generic
adapter default stays off. CUDA Graph remains unsupported: replay-safe device
epochs and graph-lifetime workspaces still need implementation and validation.
Expert-layout fusion, residual epilogue fusion and independent Tx/Rx tuning
are not part of this measured change.


## Serialized full decode CUDA Graph (2026-10-25)

This revision adds opt-in full decode Graph capture/replay. Dispatch and
combine use GPU-owned generations, graph workspaces remain alive, routes are
recomputed from the current token on each replay, and the SGLang full-graph
backend orders replay with eager work. Graph mode also submits eager prefill
on the compute stream. Prefill itself is not captured. Queue READY/ACK and
credit ordering are preserved; no CPU routing or metadata collective is added.

The reference remains SGLang's default NCCL backend with GPU P2P/NVLink paths
disabled on both sides. Both use full decode Graph with local per-DP-rank
buckets `[1, 2, 4, 8]`, eager prefill, eight identical H20D GPUs, BF16
Qwen3-30B-A3B-Instruct-2507, TP=DP=EP=8, Triton experts and FlashInfer attention.
Custom all-reduce and FlashInfer all-reduce fusion are disabled. Prompts have
512 input tokens and 32 generated tokens, B1/B32, eight warmups and five timed
repeats. The host transport retains producer-local NUMA placement, 32 allocated
channels, two slots, 4 MiB payload budget and 16-row prefill combine frames.
Read/write/dispatch unroll remains four and LAST-data remains enabled.

### Graph-mode tuning

The initial scalar decode mailbox uses auto channels (one at B1 and four at
balanced B32). Replacing it with two-peer vector reads changes only the build
setting. Fixed channel counts test the same vector path; prefill is unchanged.
Values below are median steady-state TPOT in milliseconds; smaller is better.

| Candidate | B1 TPOT | B32 TPOT |
| --- | ---: | ---: |
| Scalar / auto | 8.828 | 17.861 |
| Vector / auto | 7.854 | 16.839 |
| Vector / one channel | 7.813 | 22.714 |
| Vector / two channels | 7.987 | 18.995 |

### Final adjacent NCCL Graph comparison

The final candidate keeps vector peer batch two and automatic channels.
Both server configurations, input hashes for all ten timed cases, transport
settings and available NVLink counters were checked. Both use SHM framework
communication with no P2P/NET/NVLS connections in the recorded transport logs;
all available NVLink payload counters remain zero. Values are median [min, max].
TTFT below is the last first-token arrival in a batch; TPS is aggregate steady
state output tokens per second.

| Backend | Batch | TTFT (ms) | TPOT (ms) | Decode TPS | Whole batch (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCCL Graph | 1 | 101.427 [101.010, 102.913] | 6.453 [6.426, 6.476] | 154.958 [154.425, 155.629] | 301.859 [300.713, 302.915] |
| NCCL Graph | 32 | 479.182 [459.855, 535.574] | 8.998 [8.894, 9.096] | 3556.396 [3518.040, 3598.112] | 756.115 [739.167, 811.043] |
| EP-Xpress Graph | 1 | 95.749 [95.209, 96.519] | 7.811 [7.798, 8.051] | 128.031 [124.214, 128.235] | 338.392 [337.440, 345.805] |
| EP-Xpress Graph | 32 | 333.487 [287.789, 334.516] | 16.673 [16.620, 16.887] | 1919.234 [1894.951, 1925.339] | 848.699 [805.793, 850.179] |

**The candidate improves substantially over its earlier eager execution, but
NCCL Graph still wins decode throughput and whole-batch completion time.**
At B1/B32, the candidate's decode throughput is 17.4%/46.0% lower than NCCL
Graph. Its TTFT is 5.6%/30.4% lower, while whole-batch time is 12.1%/12.2%
longer on this 32-output-token workload. The TTFT improvement is a service
metric and must not be presented as an isolated prefill-kernel speedup.

Twenty-four host API checks pass. The final graph tests cover both fused
mailboxes and warp-specialized pair queues on two GPUs: five retained buckets,
changing routing IDs between replays, empty ranks, different replay streams,
and alternating graph decode with eager pair-queue prefill. Thirteen adapter
checks cover patch applicability, parsing and supported configuration bounds.
Four short greedy full-model prompts match the previous eager EP version in
all generated token IDs and text; the 29 sampled common-prefix logprobs have
zero observed difference. These are bounded correctness checks, not exhaustive
numerical equivalence or a proof that every benchmark follows identical routes.

Use `adapters/sglang/presets/host_sm_bf16_ep8_graph.sh` with full decode Graph
and buckets `[1, 2, 4, 8]`. Generic defaults stay eager. The same bucket list
and Graph mode are required in the NCCL comparison. The final scope rejects
PD-Multiplexing alongside the existing overlapping-forward and memory-saver
restrictions; graph contexts execute serially.

### Profiling and scope

Separate eight-step B32 traces of the prior direct-stream eager version and
the initial scalar Graph version show median per-rank observed GPU idle within
forward windows decreasing from 45.5% to 4.1%. In the scalar Graph trace,
dispatch transport occupies 42.4% and fused mailbox combine 26.2% of the forward
window (per-rank medians). Expert GEMM is 10.6%, expert alignment 0.9%.
These are kernel residency intervals including peer waits, GPU work and host
memory accesses, not pure PCIe wire times. The traces use different execution
modes with different profiler overhead; they do not isolate a byte-transfer
speedup or prove the cause of every idle interval.

A separate trace of the final vector candidate has eight decode steps on each
of eight B32 ranks, with eight `cudaGraphLaunch` calls per rank. Median observed
idle is 3.1%; dispatch transport is 45.3%, fused mailbox combine 22.1%, expert
GEMM 11.5% and alignment 1.0% of the forward window. This confirms actual graph
replay and locates the remaining kernel residency, without separating payload
access from READY/ACK/peer waiting inside those kernels.

The remaining decode work should prioritize dispatch's small-message protocol
and mailbox payload/progress efficiency. These traces do not support prioritizing
expert alignment fusion over transport, or assuming that fewer channels improve
performance. More CTAs alone do not remove per-message control dependencies.

The eager-only advantage in earlier sections must not be generalized to a
Graph-enabled NCCL baseline. TTFT includes scheduling/first-token delivery and
is not isolated prefill kernel time. Decode TPS is aggregate output throughput
in the steady window from the last first token to the first request completion;
it is not a maximum sustained serving-throughput benchmark. Repeat ranges are
observations rather than confidence intervals. Graph padding may process more
rows than eager execution, and both backends use the same bucket list.


## Bounded final-frame dispatch and capacity-derived width (2026-10-25)

The opt-in `decode_single_frame` context setting uses one final frame for each
source/channel when the configured per-source decode bound fits a frame. It
removes dispatch's separate terminal notification/ACK phase and its initial
credit poll. Required publication fences, frame ACKs and outgoing-reader drain
remain. Payload is still rank-deduplicated and multicast; all peers consume frame
metadata, while only destinations read token payload. Combine arithmetic and
prefill transport are unchanged.

This is a capacity-based library path, not a branch for a benchmark batch size,
model name or balanced routing. The setting participates in the existing
initialization configuration exchange. Different rank settings fail before
allocation. Empty ranks and skewed local row counts use the same protocol;
insufficient frame/grid capacity selects the general ring on all ranks.

The SGLang adapter's automatic decode width is at least
`ceil(decode_max_tokens / min(dispatch_frame_rows, 256))` when that count fits
the allocated cooperative grid. Explicit channel overrides are honored. The
first candidate retained the old auto width and consequently fell back to the
ring at B1/B8. The final capacity-derived policy uses 2/2/4/8 channels for the
four tested loads, with no batch-specific condition. The library default is off;
the opt-in Graph preset now enables the setting.

### Full-model comparison

Same eight H20D GPUs, BF16 Qwen3-30B-A3B-Instruct-2507, TP=DP=EP=8, Triton
experts, FlashInfer attention, full decode Graph with per-rank buckets [1,2,4,8],
eager prefill, and P2P/NVLink paths disabled on both sides. The baseline is
SGLang's default NCCL backend. Producer-local NUMA placement was verified; all
available NVLink payload counters stayed zero, and transport logs recorded no
P2P/NET/NVLS connections. No CPU trace-analysis job overlapped timed inference.

Each load uses 512 input and 32 output tokens, eight warmups and five timed
repeats. The final candidate reuses the same-session NCCL reference. A separate
same-session EP-Xpress run with the setting disabled checks the feature's effect.
All twenty timed prompt hashes and the benchmark shapes match across the three
runs. Native kernels are unchanged between the disabled run and final policy;
when disabled, the additional policy branch is inactive.

Values below are medians [min,max], in milliseconds or output tokens/second.
TTFT is the last first-token arrival in the batch, not isolated prefill time.

| Backend | Batch | TTFT (ms) | Decode TPOT (ms) | Decode TPS | Batch latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCCL Graph | 1 | 101.859 [100.966, 104.307] | 6.457 [6.433, 6.475] | 154.861 [154.439, 155.456] | 302.143 [301.697, 304.787] |
| NCCL Graph | 8 | 124.712 [124.430, 125.161] | 6.574 [6.505, 6.636] | 1216.884 [1205.571, 1229.913] | 328.869 [326.926, 330.468] |
| NCCL Graph | 32 | 478.317 [463.660, 554.250] | 8.991 [8.892, 9.058] | 3558.986 [3532.894, 3598.900] | 756.678 [740.591, 833.041] |
| NCCL Graph | 64 | 914.722 [905.171, 966.267] | 10.749 [10.691, 10.763] | 5954.112 [5946.137, 5986.277] | 1247.696 [1237.181, 1297.818] |
| EP-Xpress final-frame Graph | 1 | 97.081 [96.216, 100.579] | 7.010 [6.961, 7.063] | 142.652 [141.577, 143.660] | 315.879 [312.094, 317.993] |
| EP-Xpress final-frame Graph | 8 | 105.540 [104.780, 302.993] | 11.384 [11.375, 11.392] | 702.752 [702.235, 703.310] | 458.689 [457.856, 655.928] |
| EP-Xpress final-frame Graph | 32 | 307.203 [288.227, 342.791] | 13.982 [13.915, 14.062] | 2288.636 [2275.676, 2299.611] | 738.334 [717.637, 773.661] |
| EP-Xpress final-frame Graph | 64 | 554.317 [547.998, 589.122] | 14.817 [14.731, 14.883] | 4319.443 [4300.274, 4344.520] | 1014.494 [1006.888, 1045.897] |

### Feature-disabled ablation

| Batch | Disabled decode TPOT (ms) | Enabled decode TPOT (ms) | Disabled TPS | Enabled TPS | TPS change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.837 | 7.010 | 127.592 | 142.652 | +11.8% |
| 8 | 13.654 | 11.384 | 585.926 | 702.752 | +19.9% |
| 32 | 16.802 | 13.982 | 1904.503 | 2288.636 | +20.2% |
| 64 | 17.802 | 14.817 | 3595.108 | 4319.443 | +20.1% |

The final policy improves EP-Xpress decode throughput at all four tested loads,
but NCCL remains faster in decode. The B64 batch completes sooner than NCCL on
this short-output workload because of the existing TTFT advantage; this must
not be called a decode win or attributed entirely to the new dispatch protocol.
The B32 whole-batch difference is small compared with observed repeat spread.

Two focused native Graph cases exercise int32/int64 routes, missing experts,
empty ranks, changing routes, five buckets, two replay streams, eager/Graph and
ring/final-frame reuse, the configured source bound with all eight tokens on one
rank, and a frame too small for the fast path. Analytical BF16 output checks
pass. Initialization tests reject mixed protocol settings before storage.
Local Config/adapter checks pass (one environment-dependent patch check skips).
Four short greedy model prompts match prior Graph EP-Xpress text and token IDs;
all 29 compared logprobs have zero observed difference.

These checks do not establish correctness for every deployment or performance
on other GPU/NUMA topologies. Inputs for the throughput sweep are synthetic;
full model execution is measured, but this is not a production request-length,
routing-skew or sustained-arrival workload study. Bounds and fallback are
correctness constraints, not a claim that the auto width is optimal everywhere.


### Final Graph-only profiling

The final B8/B32 traces include a few eager local-DECODE steps while other DP
ranks are still in prefill. Excluding steps without `cudaGraphLaunch` leaves
63 and 61 rank-steps respectively. Every retained dispatch uses the single-frame
kernel specialization. The table reports the median across ranks of each
rank's mean per replay, in milliseconds:

| Batch | GPU forward window | Dispatch | Fused mailbox combine | Expert GEMM |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11.622 | 4.197 | 3.145 | 0.923 |
| 32 | 13.932 | 4.622 | 3.793 | 1.914 |

The earlier Graph B32 reference had about 7.562 ms dispatch residency per step;
the final trace is about 39% lower. These are separate diagnostic runs, not
isolated PCIe bandwidth measurements. Kernel intervals include peer waits,
metadata, payload access and reduction. Combine remains a major cost, and these
traces do not separate READY waiting from host payload latency or final ACK drain.


## Independent decode groups and Tx/Rx specialization (2026-10-25)

**Rollback status:** The experimental scheduling implementation from commit
`1d28771` has been removed after the throughput regressions below. Runtime
sources are restored exactly to `0c46b29`; `EP_XPRESS_DECODE_WARP_MODE` is no
longer a supported setting. The following measurements and implementation
descriptions are retained as historical evidence for that experimental commit,
not as current API documentation. The rollback was verified by source equality;
no new model benchmark was run for the rollback.


Implemented an optional decode mailbox scheduler with independent thread groups
and a split sending/receiving mode. `EP_XPRESS_DECODE_WARP_MODE=0` retains the
original CTA loop; `1` uses independent groups with sequential send/receive/drain;
`2` assigns 128 threads to Tx/drain and 128 to Rx/reduction. Group widths are
chosen from GPU owner-list counts and token partitions, not benchmark batch
labels. Small partitions use wider groups and larger partitions use more groups.
The choice executes inside the kernel; changed routes/counts do not create a new
CUDA specialization. The build-time mode flag itself requires loading a different
cached extension when changed before startup.

Mailbox addresses, single-writer READY/ACK ownership, publication fences,
ascending-rank arithmetic and final outgoing ACK drain are unchanged. Wide
thread groups use disjoint, non-aligned named hardware barriers and predicate
reductions. Scalar and vector payload paths remain available. This adds no CPU
route read, new collective, deferred credit or extra host arena. Prefill kernels
are unchanged.

### Full-model results

Same eight H20D GPUs, BF16 Qwen3-30B-A3B-Instruct-2507, SGLang v0.5.21,
TP=DP=EP=8, Triton experts, FlashInfer attention, full decode Graph with local
buckets [1,2,4,8] and eager prefill. Both implementations disable P2P, NVLS,
MNNVL, network transports and custom all-reduce. Default SGLang NCCL is the
primary baseline. Every final EP mode uses the same runtime source and other
settings, including single-frame dispatch, peer batch 2, producer-local NUMA,
32 allocated channels and the existing automatic active-channel policy.

Each load has 512 input tokens, 32 output tokens, eight warmups and five timed
repeats. The fresh NCCL run preceded the candidate iterations in the same
session; the final three EP modes were measured sequentially on idle GPUs.
All twenty timed prompt hashes match across backends/modes. Configuration and
transport checks validate all comparisons. All available NVLink payload
counters remained zero; logs show SHM only (60 framework connections for EP,
120 for NCCL), with zero P2P/NET/NVLS. Producer NUMA placement is verified.

Values are median [min,max]. TTFT is the last first-token arrival in a batch,
including scheduling and prefill. Decode TPS counts aggregate output tokens/s.

| Backend | Batch | TTFT (ms) | Decode TPOT (ms) | Decode TPS | Batch latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCCL | 1 | 101.081 [100.846, 102.134] | 6.450 [6.432, 6.474] | 155.040 [154.463, 155.473] | 301.224 [300.835, 301.883] |
| NCCL | 8 | 125.921 [125.297, 210.722] | 6.557 [6.496, 6.622] | 1220.028 [1208.138, 1231.451] | 331.059 [329.161, 413.348] |
| NCCL | 32 | 507.278 [454.292, 550.681] | 9.010 [8.833, 9.059] | 3551.784 [3532.563, 3622.751] | 786.987 [732.238, 824.566] |
| NCCL | 64 | 925.100 [898.375, 995.424] | 10.778 [10.738, 10.832] | 5937.762 [5908.198, 5960.045] | 1259.668 [1230.092, 1329.723] |
| Original CTA (mode 0) | 1 | 96.997 [96.641, 97.419] | 6.998 [6.972, 7.033] | 142.893 [142.185, 143.428] | 314.467 [312.895, 314.814] |
| Original CTA (mode 0) | 8 | 101.132 [100.709, 688.308] | 11.367 [11.355, 11.389] | 703.780 [702.426, 704.563] | 453.308 [452.970, 1040.793] |
| Original CTA (mode 0) | 32 | 292.535 [285.270, 316.709] | 14.002 [13.907, 14.046] | 2285.361 [2278.225, 2301.027] | 720.718 [716.004, 747.154] |
| Original CTA (mode 0) | 64 | 531.654 [527.675, 580.092] | 14.852 [14.752, 14.986] | 4309.232 [4270.642, 4338.539] | 993.992 [985.703, 1037.098] |
| Independent groups (mode 1) | 1 | 97.130 [96.633, 97.487] | 7.271 [7.255, 7.313] | 137.535 [136.744, 137.837] | 322.380 [322.131, 323.864] |
| Independent groups (mode 1) | 8 | 101.184 [100.713, 190.022] | 11.810 [11.796, 11.832] | 677.398 [676.134, 678.191] | 467.524 [466.537, 555.969] |
| Independent groups (mode 1) | 32 | 323.834 [290.751, 334.079] | 14.226 [14.177, 14.335] | 2249.426 [2232.372, 2257.229] | 763.233 [731.778, 772.908] |
| Independent groups (mode 1) | 64 | 527.599 [527.337, 588.580] | 15.069 [14.952, 15.162] | 4247.171 [4220.961, 4280.494] | 995.572 [992.011, 1054.276] |
| Split Tx/Rx groups (mode 2) | 1 | 97.060 [95.779, 98.610] | 7.537 [7.504, 7.557] | 132.673 [132.325, 133.260] | 330.279 [329.670, 332.359] |
| Split Tx/Rx groups (mode 2) | 8 | 102.047 [101.500, 198.560] | 11.731 [11.718, 11.753] | 681.949 [680.686, 682.703] | 466.452 [465.384, 561.382] |
| Split Tx/Rx groups (mode 2) | 32 | 287.900 [286.195, 323.066] | 14.270 [14.255, 14.355] | 2242.409 [2229.182, 2244.770] | 728.888 [724.349, 763.838] |
| Split Tx/Rx groups (mode 2) | 64 | 582.216 [529.288, 591.686] | 15.075 [15.016, 15.187] | 4245.360 [4214.054, 4262.064] | 1049.493 [995.686, 1057.345] |

### Same-source scheduling ablation

| Batch | Mode 0 TPS | Mode 1 TPS | Mode 1 change | Mode 2 TPS | Mode 2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 142.893 | 137.535 | -3.75% | 132.673 | -7.15% |
| 8 | 703.780 | 677.398 | -3.75% | 681.949 | -3.10% |
| 32 | 2285.361 | 2249.426 | -1.57% | 2242.409 | -1.88% |
| 64 | 4309.232 | 4247.171 | -1.44% | 4245.360 | -1.48% |

### Interpretation and validation

The new scheduler remains opt-in; the library and existing Graph preset retain
mode 0. The fixed 32-thread/token prototype regressed strongly (B1/B32/B64 TPS
113.537/1926.054/3666.653). Resource-derived wider groups recovered most of that
loss. Replacing shared-memory group votes with hardware reduction barriers made
only a small difference in the split implementation; synchronization cleanup
alone did not establish a throughput win.

The existing balanced-load channel policy is an important constraint. Local
Graph shapes are 1/1/4/8 and active channel counts are 2/2/4/8 for the four loads.
Thus a nonempty receiving CTA generally has one output token. Independent groups
cannot add cross-token parallelism in that case: mode 1 selects a wide group,
while mode 2 splits a previously 256-thread token operation into a 128-thread Rx
role plus a 128-thread Tx role. This is a code/resource observation consistent
with the timing, not a measured partition of PCIe, READY or ACK latency. Skewed
loads and different channel settings can produce multiple tasks per CTA; those
remain supported, but this report does not claim an unmeasured speedup for them.
TTFT advantages over NCCL largely predate this decode change and must not be
attributed to Tx/Rx overlap. No phase profile was collected in this iteration.

The initial split implementations passed seven focused GPU cases, covering
random BF16 legacy-rounding comparisons, scalar/vector/shifted storage, sparse
contributors, delayed nonparticipants, empty ranks, skewed ownership, reverse
handle use, changing routes, Graph bucket switches and eager/Graph alternation.
After the hardware-barrier change, four focused cases were rerun: aligned
2048-wide and scalar 258-wide mailbox comparisons plus the two Graph/ring-fallback
cases. All passed. Each final full-model mode also matches the prior EP version's
four short greedy outputs/token IDs; all 29 compared prefix logprobs have zero
observed difference. This is bounded correctness evidence, not exhaustive
verification of every route, group size or deployment.

Inputs for the performance sweep are synthetic. Complete model inference is
measured, but production request-length distributions, arrival processes, expert
skew and other GPU/NUMA topologies need their own validation. The mode selection
is deliberately not hard-coded to the benchmark's concurrency values.


## Decode READY publication batching and Graph phase profiling (2026-10-25)

Fused decode now has optional, bounded phase counters that work with serialized
CUDA Graph replay. Preparation precedes capture; start/stop synchronize at
quiescent request boundaries. Production contexts use a separate uninstrumented
specialization. Overflow drops new diagnostic samples and reports the count.

The new build-time `EP_XPRESS_DECODE_PUBLISH_BATCH=8` option batches outgoing
READY publication after every payload writer has executed a system fence.
Partial batches flush before reception. It preserves all 256 receive threads,
current GPU routing/owner lists, exact arithmetic, and final outgoing ACK drain.
There is no CPU routing read or new collective. Default 1 retains per-row
publication; larger batches are an explicit workload-dependent option. The final
implementation flushes a partial batch at its last actual outgoing row, using
current GPU owner counts to avoid trailing empty-peer scans before READY.

### Full-model comparison

Same eight H20D GPUs, BF16 Qwen3-30B-A3B-Instruct-2507, SGLang v0.5.21,
TP=DP=EP=8, Triton experts and FlashInfer attention. Full decode Graph uses local
buckets [1,2,4,8]; prefill stays eager. Producer-local NUMA, 32 allocated channels,
auto active channels, single-frame dispatch, peer batch 2 and all other transport
settings match. Inputs have 512 tokens and request 32 output tokens, with eight
warmups and five timed repeats per load. The initial batch-8 candidate, batch-1
control, default SGLang NCCL and final early-tail batch-8 candidate were measured sequentially on idle GPUs. The tail
correction is compiled only into the batch>1 branch; per-row transport is unchanged.

Both backends disable NVLink payloads and GPU P2P. Available NVLink counters all
remain zero; NCCL transport logs contain SHM only (60 framework connections for
EP, 120 for NCCL), zero P2P/NET/NVLS. EP arena NUMA placement is verified before
and after timing. Resolved model/Graph settings and all twenty timed prompt
hashes match; the comparisons pass the saved-evidence validator.

Values are median [min,max]. TTFT is the last first-token arrival in a batch,
including scheduling/prefill; Decode TPS is aggregate output tokens per second.
No phase counters or profiler are enabled during these performance measurements.

| Backend | Batch | TTFT (ms) | Decode TPOT (ms) | Decode TPS | Batch latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCCL | 1 | 101.250 [100.599, 102.806] | 6.459 [6.433, 6.489] | 154.820 [154.105, 155.451] | 302.216 [300.128, 303.132] |
| NCCL | 8 | 124.767 [124.670, 127.259] | 6.583 [6.510, 6.643] | 1215.275 [1204.364, 1228.868] | 329.207 [328.406, 330.790] |
| NCCL | 32 | 471.876 [460.341, 477.128] | 9.036 [8.885, 9.098] | 3541.446 [3517.077, 3601.451] | 750.249 [738.412, 752.795] |
| NCCL | 64 | 964.458 [903.650, 997.347] | 10.755 [10.680, 10.835] | 5950.447 [5906.967, 5992.256] | 1296.927 [1238.769, 1331.128] |
| EP per-row | 1 | 96.501 [95.820, 100.739] | 7.000 [6.845, 7.060] | 142.857 [141.636, 146.093] | 314.324 [312.302, 314.792] |
| EP per-row | 8 | 100.889 [100.535, 188.756] | 11.392 [11.378, 11.407] | 702.271 [701.298, 703.123] | 454.671 [453.326, 541.490] |
| EP per-row | 32 | 307.883 [289.443, 327.760] | 13.960 [13.856, 14.008] | 2292.230 [2284.398, 2309.426] | 737.907 [720.395, 759.292] |
| EP per-row | 64 | 527.430 [526.425, 531.277] | 14.855 [14.830, 14.942] | 4308.450 [4283.331, 4315.518] | 985.804 [985.133, 988.560] |
| EP batch 8 initial | 1 | 96.749 [96.306, 100.134] | 7.082 [6.988, 7.093] | 141.210 [140.989, 143.109] | 315.953 [313.459, 320.124] |
| EP batch 8 initial | 8 | 105.351 [104.996, 105.735] | 10.705 [10.702, 10.721] | 747.302 [746.216, 747.495] | 437.619 [436.893, 437.770] |
| EP batch 8 initial | 32 | 286.486 [285.098, 289.664] | 12.957 [12.876, 13.014] | 2469.718 [2458.831, 2485.193] | 683.370 [681.410, 689.237] |
| EP batch 8 initial | 64 | 529.404 [524.631, 591.762] | 13.575 [13.458, 13.682] | 4714.667 [4677.520, 4755.410] | 949.232 [943.138, 1009.153] |
| EP batch 8 early tail | 1 | 102.540 [96.395, 103.292] | 7.038 [6.970, 7.049] | 142.087 [141.863, 143.481] | 320.157 [313.722, 321.472] |
| EP batch 8 early tail | 8 | 104.884 [104.456, 105.199] | 10.772 [10.769, 10.784] | 742.666 [741.867, 742.905] | 439.140 [438.518, 439.160] |
| EP batch 8 early tail | 32 | 287.557 [286.407, 315.362] | 12.973 [12.922, 13.078] | 2466.655 [2446.824, 2476.472] | 687.701 [684.243, 714.162] |
| EP batch 8 early tail | 64 | 527.763 [522.476, 533.540] | 13.625 [13.614, 13.720] | 4697.278 [4664.655, 4701.107] | 947.656 [943.505, 954.259] |

| Batch | Final batch 8 Decode TPS change vs per-row | Final batch 8 Decode TPS change vs NCCL |
| --- | ---: | ---: |
| 1 | -0.54% | -8.22% |
| 8 | +5.75% | -38.89% |
| 32 | +7.61% | -30.35% |
| 64 | +9.02% | -21.06% |

The multi-request decode improvement does not close the gap to NCCL. Prefill is
unchanged by the mailbox optimization; multi-request TTFT advantages mostly
predate it, and between-run scheduling variation must not be attributed to decode publication.
The final B1 TTFT median is higher than both the per-row control and NCCL;
no single-request TTFT gain is established. Observed ranges are not confidence
intervals or a production SLA.

### Separate phase measurements (initial batching implementation)

Two instrumented trials per load use the same prompt IDs, with eight requested
output tokens and matched off/on profiler trials. Each rank records 384 fused
combines per trial with no overflow or status error. Only rank-local timestamps
are compared. Per operation, select the CTA whose end timestamp is latest;
the table averages its phase durations across trials (source rank 0 for B1,
all eight ranks for B32). This is a diagnostic view of the last-finishing CTA,
not the sum of concurrent channel/rank durations or global wall latency.

All entries below are microseconds per sampled CTA. READY includes polling and
its synchronization; payload includes local computation, reduction and output
stores; ACK drain includes successful checks and waiting. Phase boundaries add
CTA barriers. These counters do not isolate PCIe wire time, and ACK drain on a
sender can reflect the same receiver delay already visible on another GPU.

| Load | Variant | Elapsed | Send metadata | Send payload | Fence/READY publish | READY poll | Receive payload/reduce | ACK publish | Final ACK drain | Receive metadata + other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | Original | 24.785 | 4.707 | 0.000 | 0.000 | 8.222 | 5.655 | 2.173 | 0.000 | 4.028 |
| B1 | Batch 8 | 27.750 | 4.757 | 0.000 | 0.000 | 11.237 | 5.574 | 2.147 | 0.000 | 4.034 |
| B32 | Original | 75.309 | 10.318 | 5.139 | 17.561 | 10.766 | 5.830 | 3.448 | 15.018 | 7.228 |
| B32 | Batch 8 | 59.989 | 8.826 | 6.231 | 2.928 | 14.618 | 5.907 | 3.493 | 10.654 | 7.332 |

The phase table compares the original path with the initial batching prototype,
not the final early-tail correction. B32 publication fell from 17.561 to 2.928 us,
and last-finishing CTA duration from 75.309 to 59.989 us. Receive payload/reduction
remained approximately 5.8-5.9 us. READY polling increased, so the gain must not be
described as faster payload reads or elimination of all waits.

For B1, receiver READY polling increased from 8.222 to 11.237 us. Inspection found
that the initial prototype deferred its partial-batch publication until after
trailing empty peers. The final correction publishes at the last actual send;
its performance is measured in the separate early-tail full-model rows above.
No final early-tail phase profile was collected; do not substitute the prototype
phase values for a measurement of that final implementation.

### Validation and limits

Three focused GPU cases passed, and all three passed again after the early-tail
correction: dynamic-route Graph replay with bucket changes,
eager interleave, empty/skewed ranks and diagnostic reset/overflow; and aligned
2048-wide plus scalar 258-wide mailbox comparisons against legacy BF16 rounding.
The latter also cover sparse readers, delayed nonparticipants, reverse handle
order and repeated reuse. Four full-model greedy output samples match the
same-source control and prior EP implementation exactly; all 29 compared prefix
logprobs have zero observed difference. These are bounded checks, not exhaustive
validation of every shape, placement or interleaving.

Inputs for the throughput sweep are synthetic token IDs running the complete
model. Production arrival/length distributions, routing skew and other GPU/PCIe
or NUMA topologies need separate evaluation. No routing is cached across decode
steps and no benchmark-specific batch-size branch is introduced. The diagnostic
counter storage is also released at context teardown; that cleanup does not
change the measured inference path.

## Bounded decode descriptors and parallel ACK checks (2026-10-25)

The optional `EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32` schedule prepares outgoing
row/mailbox descriptors in bounded GPU tiles and polls the actual outgoing ACKs
in parallel. It retains all 256 receive threads, current routing masks, the
existing BF16 arithmetic order, all-writer system fences before READY, and the
final ACK drain before workspace reuse. Default 0 keeps the previous schedule.
The candidate's test configuration also sets `EP_XPRESS_DECODE_PUBLISH_BATCH=8`.
Descriptors are recomputed from each invocation's GPU owner lists; tile size is
not a token-capacity limit or a benchmark-specific routing assumption.

The candidate compiles on the development GPUs. Three focused GPU cases pass:
Graph replay with changing routes/buckets, eager interleave and diagnostic
reset/overflow; and vector-width 2048 plus scalar-width 258 mailbox comparisons
against legacy rounding. Mailbox cases now include more than 32 outgoing
descriptors per CTA, partial final tiles, changed route skew/channel counts,
reverse handle consumption, sparse readers and delayed nonparticipants.
The three-case run completed in 83.94 seconds; the only warning concerned the
read-only test-cache directory.

Full-model control and NCCL runs and complete forward traces were collected,
but subsequent GPU process sampling found an independent benchmark sequence
running concurrently with the candidate's timing window. The candidate run was
stopped. The initial NCCL control also had large outliers; it did not have the
same continuous process-ownership audit, so absence of interference cannot be
established retrospectively. These attempts are not accepted as a controlled
performance comparison, and no new TTFT or Decode TPS gain is claimed here.

The controlled rerun below completes the model comparison on an available
machine. Private experiment instrumentation records GPU process ownership
throughout each run and rejects results if another GPU task appears. Timing and
profiling remain separate; raw deployment details stay outside this repository.
The descriptor option remains opt-in because these measurements cover one
hardware topology and a bounded workload set.

### Controlled same-machine model comparison

All three variants ran on the same eight H20D GPUs, split four per NUMA node,
with Qwen3-30B-A3B-Instruct-2507 in BF16, SGLang v0.5.21, TP=DP=EP=8,
Triton experts, FlashInfer attention, eager prefill and full decode CUDA Graph
buckets [1, 2, 4, 8] per rank. They used the same restored offline runtime image
and source revision `6371b75`. NCCL used SGLang's default MoE backend (`none`),
with custom all-reduce and fusion disabled. Both EP variants used publication
batch 8; only descriptor batching changed from 0 to 32. Channels, peer batching,
frame/slot settings, host budget and producer-NUMA placement were held fixed.

Every group completed 8 warmups and 5 timed repetitions per shape: 512 input
and 32 output tokens, with batches 1, 8, 32 and 64. Prompt hashes and resolved
critical settings matched. GPU process ownership was sampled every two seconds:
no external process was observed in 109 control, 55 NCCL and 91 candidate samples.
This cannot exclude sub-period activity or CPU-only interference. Both EP runs
verified producer-local host pages; all groups verified SHM transport with no
P2P/NET/NVLS paths and zero NVLink payload counter increments.

Values below are median [observed minimum, maximum]. TTFT is the time until the
last request in the batch receives its first token. Decode TPS is aggregate
throughput in the steady decode window, not maximum sustained serving throughput.

| Variant | Batch | TTFT (ms) | Decode TPS (tokens/s) | TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| NCCL | 1 | 103.197 [101.343, 104.775] | 154.990 [154.561, 155.797] | 6.452 [6.419, 6.470] |
| NCCL | 8 | 121.903 [121.519, 122.137] | 1217.544 [1206.590, 1230.903] | 6.571 [6.499, 6.630] |
| NCCL | 32 | 438.610 [433.664, 521.860] | 3540.024 [3507.900, 3591.684] | 9.039 [8.909, 9.122] |
| NCCL | 64 | 859.613 [828.333, 930.488] | 5948.362 [5898.773, 5991.135] | 10.759 [10.682, 10.850] |
| EP descriptor 0 | 1 | 96.343 [96.015, 98.055] | 141.703 [141.604, 142.508] | 7.057 [7.017, 7.062] |
| EP descriptor 0 | 8 | 188.983 [101.316, 303.211] | 738.827 [738.247, 739.788] | 10.828 [10.814, 10.836] |
| EP descriptor 0 | 32 | 272.814 [267.411, 312.053] | 2457.320 [2443.246, 2481.691] | 13.022 [12.894, 13.097] |
| EP descriptor 0 | 64 | 495.195 [490.999, 546.586] | 4688.749 [4650.506, 4712.524] | 13.650 [13.581, 13.762] |
| EP descriptor 32 | 1 | 98.527 [97.393, 111.908] | 144.520 [143.875, 145.333] | 6.919 [6.881, 6.950] |
| EP descriptor 32 | 8 | 101.722 [100.983, 111.567] | 809.060 [803.902, 813.451] | 9.888 [9.835, 9.951] |
| EP descriptor 32 | 32 | 273.211 [270.222, 297.169] | 2650.621 [2643.134, 2673.513] | 12.073 [11.969, 12.107] |
| EP descriptor 32 | 64 | 495.994 [493.612, 558.153] | 5041.549 [4996.534, 5083.131] | 12.695 [12.591, 12.809] |

| Batch | Descriptor 32 Decode TPS change vs descriptor 0 | Descriptor 32 Decode TPS change vs NCCL |
| --- | ---: | ---: |
| 1 | +1.99% | -6.76% |
| 8 | +9.51% | -33.55% |
| 32 | +7.87% | -25.12% |
| 64 | +7.52% | -15.24% |

The decode gain is observed at every tested load, but the candidate remains below
NCCL in Decode TPS. B32 and B64 TTFT are essentially unchanged between EP schedules;
their advantage over NCCL predates this decode-only change. The descriptor-0 B8
TTFT ranged from 101.316 to 303.211 ms, so its large TTFT difference from the
candidate must not be attributed to faster decode descriptors. These are five
repeats in one sequential comparison, not confidence intervals or a production SLA.

All four short greedy output samples match between the candidate and control,
including all 29 compared prefix logprobs. The four generated outputs also match
NCCL, but prefix logprobs differ (maximum absolute difference 0.218722, mean
0.0146705). This does not establish numerical equivalence with NCCL or identical
expert routes throughout the synthetic throughput sweep. The implementation
continues to derive descriptors from current GPU owner lists on every invocation;
no prompt, batch-specific route or previous decode route is cached.

### Matched combine phase sampling

Two instrumented trials per load use matched off/on captures, the same repeat-zero
prompt IDs and eight requested output tokens. Every rank recorded 384 combines per
trial without dropped samples or status errors. No external GPU process was
observed in either diagnostic run. For each operation, select the rank-local CTA
with the latest end timestamp, then average across trials (source rank 0 for B1,
all eight ranks for B32). Do not sum concurrent rank/CTA times as wall latency.

The entries are microseconds per sampled CTA. Instrumentation adds barriers;
READY includes polling and synchronization, payload includes reduction and stores,
and ACK drain includes checking as well as waiting. These are not PCIe wire times.

| Load | Descriptors | Elapsed | Send metadata | Send payload | Fence/READY | READY poll | Receive payload/reduce | ACK publish | ACK drain | Receive metadata + other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 0 | 25.620 | 4.634 | 0.000 | 0.000 | 8.626 | 5.642 | 2.130 | 0.000 | 4.588 |
| B1 | 32 | 23.054 | 1.386 | 0.000 | 0.000 | 12.167 | 5.551 | 2.332 | 0.000 | 1.618 |
| B32 | 0 | 62.100 | 8.781 | 6.376 | 3.041 | 14.944 | 5.883 | 3.453 | 10.591 | 9.032 |
| B32 | 32 | 39.418 | 2.699 | 4.623 | 3.357 | 13.395 | 5.831 | 3.520 | 3.507 | 2.487 |

B32 last-finishing CTA residence falls from 62.100 to 39.418 us. Send metadata,
final ACK drain and other control work fall substantially; receive payload/reduce
stays near 5.8-5.9 us. READY polling still costs 13.395 us in the candidate, about
34% of this diagnostic residence, and must not be described as eliminated. B1
READY polling increases while total residence falls: beginning to poll earlier
can expose more waiting even when the whole operation finishes sooner. Separate
instrumented runs do not establish exact cross-rank causality for that change.

### Complete decode forward traces and next bottleneck

Separate PyTorch traces resolve complete CUDA Graph replays for all measured
loads, with validated GPU annotations and no reported missing-node/integrity
issues. The table is the median across active DECODE ranks of each rank's mean
per complete replay. IDLE ranks are retained separately in the private analysis.
Components are unions of kernel intervals within that category; categories can
overlap, so they must not be added to reconstruct forward latency. Profiled
forward time is not HTTP TPOT, and the NCCL column is framework collectives rather
than an identical dispatch/combine interface.

| Batch | Control forward (ms) | Candidate forward (ms) | NCCL forward (ms) | Control combine (ms) | Candidate combine (ms) | Candidate dispatch (ms) | NCCL framework collectives (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.390 | 7.232 | 6.579 | 1.175 | 1.011 | 2.065 | 2.448 |
| 8 | 10.954 | 10.181 | 6.676 | 2.624 | 1.810 | 4.101 | 2.492 |
| 32 | 12.945 | 12.275 | 8.920 | 2.814 | 1.897 | 4.682 | 3.403 |
| 64 | 13.708 | 12.775 | 10.831 | 2.920 | 2.009 | 4.954 | 4.782 |

At B32, dispatch remains approximately 4.68 ms while combine falls from 2.81 to
1.90 ms. Expert GEMM remains approximately 1.92 ms in both EP schedules. The
observed gain is consistent with a localized combine optimization; the remaining
decode gap cannot be solved solely by removing more final ACK checks. First
profile dispatch's control/READY work and host payload accesses separately, then
optimize the dominant part while retaining dynamic readers and safe slot reuse.
Any send/receive overlap change should be justified by those measurements and
must preserve receive parallelism. Increasing CTA count without measuring its
queue and polling costs is not an established improvement.


## Indexed decode dispatch and batched ACK commits (2026-10-25)

The implementation in `3fa4752` adds optional receiver-direct indices and batched
ACK publication for bounded single-frame decode. The recorded candidate manifest
matches all 54 tracked files of that implementation commit. It retains current
GPU token routes, per-rank deduplication, one shared payload write, full-CTA
reception and the mandatory final ACK drain. `EP_XPRESS_DECODE_DISPATCH_INDEXED=1`
enables it; the default remains 0. Receiver indices use unused payload capacity
or the idle second slot's payload area, with one frame ACK protecting both.
Neither the configured source capacity nor the arena allocation is reduced.

Six targeted CUDA cases passed: bounded Graph profiling/reset/overflow,
int32/int64 routing, small-frame fallback, three-rank dynamic Graph/eager reuse,
BF16 vector and scalar-tail widths, empty/skewed ranks, and multi-warp index
ballots with partial tails. These checks verify protocol behavior rather than
provide performance estimates.

### Controlled full-model measurement

Same eight H20D GPUs; SGLang v0.5.21; Qwen3-30B-A3B-Instruct-2507 BF16;
TP=DP=EP=8; Triton experts; FlashInfer attention; full decode Graph with local
buckets 1/2/4/8 and eager prefill. Each load uses 512 input and 32 output tokens,
eight warmups and five measured repeats with identical input hashes. The sole
external baseline is SGLang's default `--moe-a2a-backend none` NCCL path.
Both implementations disable NVLink/P2P/custom all-reduce; transport records
show SHM with zero P2P/NET/NVLS, and available NVLink payload counters have zero
increments. Ownership sampling detected no external GPU process.

EP control and candidate retain 32 allocated/adaptive active channels,
4 MiB chunk budget, two slots, 16-row prefill combine frames, decode source
capacity 64, descriptor batch 32, publish batch 8 and peer batch 2. Host arenas
remain producer-NUMA allocated with equal total page placement on both nodes.
All headline timing uses ordinary contexts without prepared diagnostic counters.

Values below are median [min, max]. TTFT mean averages requests within a repeat;
TTFT last measures the final request's first token. Decode TPS counts output
tokens within the interval after every request has started generating and
before the first request completes. It excludes prefill and is not end-to-end TPS.

| Backend | Batch | Mean TTFT (ms) | Last TTFT (ms) | Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| NCCL | 1 | 101.279 [100.371, 102.513] | 101.279 [100.371, 102.513] | 155.071 [154.181, 155.401] |
| NCCL | 8 | 121.341 [120.931, 143.782] | 121.740 [121.292, 144.156] | 1216.497 [1205.285, 1230.512] |
| NCCL | 32 | 391.222 [374.586, 449.403] | 450.675 [435.678, 484.216] | 3543.945 [3524.285, 3588.787] |
| NCCL | 64 | 821.047 [700.343, 893.274] | 873.281 [842.140, 916.369] | 5957.461 [5906.739, 5999.644] |
| EP control | 1 | 96.867 [96.588, 98.173] | 96.867 [96.588, 98.173] | 144.233 [143.204, 144.432] |
| EP control | 8 | 100.861 [100.390, 101.369] | 101.244 [100.755, 101.744] | 806.785 [801.289, 810.704] |
| EP control | 32 | 247.077 [246.197, 295.941] | 269.730 [267.172, 311.742] | 2645.683 [2638.564, 2668.388] |
| EP control | 64 | 441.813 [425.565, 528.961] | 494.634 [492.702, 550.980] | 5027.234 [5014.218, 5059.895] |
| EP indexed | 1 | 96.905 [96.142, 99.295] | 96.905 [96.142, 99.295] | 150.758 [150.378, 151.090] |
| EP indexed | 8 | 101.652 [101.175, 101.809] | 102.021 [101.543, 102.178] | 832.066 [826.656, 837.735] |
| EP indexed | 32 | 255.183 [246.424, 257.885] | 268.551 [267.108, 283.917] | 2770.741 [2753.737, 2789.169] |
| EP indexed | 64 | 476.529 [426.102, 555.492] | 518.129 [496.329, 562.825] | 5302.553 [5248.867, 5336.196] |

| Batch | Indexed Decode TPS vs control | Indexed Decode TPS vs NCCL |
| --- | ---: | ---: |
| 1 | +4.52% | -2.78% |
| 8 | +3.13% | -31.60% |
| 32 | +4.73% | -21.82% |
| 64 | +5.48% | -10.99% |

Decode TPS improves at all measured loads but remains below NCCL. There is no
stable TTFT improvement from this decode change. B32 mean TTFT increases while
last TTFT is nearly unchanged; B64 mean/last TTFT increase with broad repeat
ranges. The existing TTFT advantage over NCCL must not be attributed to the
new decode indexing. These are sequential five-repeat comparisons, not
confidence intervals or a production workload study.

All four short greedy output samples match the control exactly, including all
29 compared prefix logprobs. NCCL's generated outputs also match, with the same
previously observed logprob differences. This is a limited output sanity check,
not proof of identical routes throughout the throughput sweep. Routes in the
implementation are derived anew from GPU top-k IDs on every invocation.

An initial version restricted indices to unused space in slot zero. With the
unchanged configured source bound of 64 and two active channels, that region
was too small, so B1/B8 correctly fell back to mask scanning and showed no
throughput gain. Borrowing the idle second slot fixes this capacity limitation
without changing routing bounds or recognizing particular benchmark batches.
The final implementation still falls back if neither region fits.

### Dispatch phase sampling

Two instrumented trials per load captured 384 dispatches per rank without
status errors or dropped records. Select the latest-finishing CTA separately
for each rank-local operation, then average across trials: rank 0 for B1, all
eight ranks for B32. The entries below are microseconds per sampled CTA,
including diagnostic barriers. They are not pure PCIe transfer times, and
concurrent CTA or rank times must not be summed as wall latency. B1 rank 0
sends payload but receives no remote dispatch payload; other ranks receive it.

| Phase | B1 control | B1 indexed | B32 control | B32 indexed |
| --- | ---: | ---: | ---: | ---: |
| Elapsed | 47.439 | 42.538 | 103.884 | 96.278 |
| GPU route/local copy | 5.430 | 5.524 | 5.172 | 5.194 |
| Send metadata | 0.451 | 0.597 | 0.446 | 0.766 |
| Payload writes | 1.186 | 1.247 | 1.163 | 1.220 |
| Fence/READY publication | 3.224 | 2.926 | 3.750 | 3.751 |
| READY/frame-count polling | 8.341 | 12.254 | 11.094 | 17.758 |
| Receive indices/IDs/weights | 0.000 | 0.000 | 40.674 | 31.810 |
| Payload reads | 0.000 | 0.000 | 9.028 | 7.994 |
| ACK publication | 15.226 | 3.057 | 18.817 | 5.537 |
| Final ACK checks/completion | 7.366 | 9.711 | 7.172 | 13.418 |
| Other control | 6.215 | 7.222 | 6.568 | 8.828 |

The independent no-progress loop view overlaps those phases. In B32, READY-only
idle time is 0.016 -> 0.019 us, and final-drain idle time is 6.835 -> 18.098 us.
For B1 rank 0, final-drain idle time is 6.407 -> 11.184 us. Credit-only idle
time is zero: the single-frame protocol has no intermediate slot reuse, but
still must drain ACKs before the next call can reuse either payload region.
Do not add these idle times to the phase totals or interpret all READY polling
as time waiting for a late rank. Polling includes the dependent frame-count read.

The intended costs fall: B32 receiver metadata 40.674 -> 31.810 us, ACK publication
18.817 -> 5.537 us. They are partly offset by more READY/count polling, final
ACK checks and no-progress drain residence. Faster local processing can expose
more remote waiting, and committing several completed frames together can delay
an early frame's ACK. These traces show the tradeoff but do not isolate a
cross-rank causal explanation. Total sampled B32 CTA residence falls only
103.884 -> 96.278 us, consistent with a modest end-to-end gain.

Requested payload byte counters are unchanged across both trials and versions:
B1 writes 1,572,864 and reads 7,229,440 bytes; B32 writes 50,331,648 and reads
233,566,208 bytes, aggregated across ranks/channels per trial. These count
requested payload accesses, not actual PCIe bus transactions. The optimization
preserves payload deduplication rather than claiming a new reduction in bytes.

### Complete decode Graph traces

Complete replays were resolved with validated GPU annotations and no reported
missing-node/integrity issues. Values are the median across active DECODE ranks
of each rank's mean per complete replay. IDLE ranks are retained in the private
analysis. Component interval unions may overlap and must not be added to
reconstruct wall time. Profiled GPU forward time is separate from HTTP TTFT/TPOT.

| Batch | EP control forward (ms) | EP indexed forward (ms) | NCCL forward (ms) | Control dispatch (ms) | Indexed dispatch (ms) | Indexed combine (ms) | NCCL framework collectives (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.275 | 6.972 | 6.564 | 2.108 | 1.799 | 1.006 | 2.420 |
| 8 | 10.082 | 9.825 | 6.666 | 4.070 | 3.974 | 1.522 | 2.490 |
| 32 | 12.043 | 11.497 | 8.921 | 4.701 | 4.281 | 1.776 | 3.390 |
| 64 | 12.792 | 12.164 | 10.862 | 4.960 | 4.459 | 1.882 | 4.811 |

At B32, dispatch falls from 4.701 to 4.281 ms and complete forward from
12.043 to 11.497 ms. Expert GEMM stays approximately 1.92 ms. Combine code was
not changed; variations in its measured residence include waiting/arrival
effects and should not be described as a new combine-kernel optimization.

The next priorities are the remaining dependent host metadata transactions
and ACK turnaround, not an unmeasured increase in CTA count. Candidate
experiments include publishing each receiver's count with its READY generation
to avoid a second host read, prefetching several ready peers' indices/IDs/weights
in parallel, and testing earlier ACK release for metadata-only frames while
retaining payload lifetime protection. These are proposed follow-ups, not
implemented or validated capabilities in this revision.


## Decode dispatch control-prefetch pipeline (2026-10-25)

Implementation: `a47ec9d`, opt-in `EP_XPRESS_DECODE_DISPATCH_PIPELINE=1`
with indexed single-frame decode enabled. The baseline is SGLang's default
NCCL backend (`moe-a2a-backend none`). A separate EP control retains indexed
dispatch and batched ACKs with pipeline disabled. This is a fresh same-machine
comparison; no measurements from the preceding machine are mixed into it.

Eight H20D GPUs, TP=DP=EP8, Qwen3-30B-A3B-Instruct-2507 BF16, SGLang v0.5.21,
Triton experts and FlashInfer attention. All groups use full decode CUDA Graphs
with local buckets 1/2/4/8, eager prefill, input 512/output 32 tokens, eight
warmups and five timed repeats per load. Producer NUMA placement, active channel
policy, payload precision and arena capacity are unchanged. All three groups
use the same offline image; all timed prompt hashes match. No external GPU
process was observed by the two-second ownership sampler. NCCL logs contain
only SHM (zero P2P/NET/NVLS); all NVLink counters were available with zero delta.

### Full-model HTTP results

All entries are medians of five repeats. TTFT is the mean across requests in a
batch; last TTFT is also shown because prefill scheduling affects concurrent
requests. Decode TPS is aggregate post-prefill throughput from the benchmark's
steady-state window, not aggregate output tokens divided by whole-request time.

| Batch | Implementation | Mean TTFT (ms) | Last TTFT (ms) | Decode TPS | Whole-batch latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | EP indexed control | 96.939 | 96.939 | 151.109 | 301.941 |
| 1 | EP pipeline | 97.039 | 97.039 | 155.434 | 297.056 |
| 1 | Default NCCL | 101.476 | 101.476 | 154.884 | 301.738 |
| 32 | EP indexed control | 294.807 | 320.729 | 2783.677 | 675.226 |
| 32 | EP pipeline | 295.793 | 319.357 | 2915.332 | 658.202 |
| 32 | Default NCCL | 410.776 | 480.005 | 3551.426 | 757.044 |

Decode TPS improves 2.86% at B1 and 4.73% at B32 over the EP control. Against
NCCL it is +0.35% at B1 and -17.91% at B32: B1 is effectively tied within the
observed variation, not evidence of a robust NCCL win. TTFT does not improve
relative to the EP control. Prefill implementation was unchanged, so the
existing TTFT advantage over NCCL must not be attributed to this decode change.
These sequential five-repeat runs are not confidence intervals or a production
workload study; equal input hashes do not establish identical full routing traces.

| Batch | Implementation | Decode TPS min–max | Mean TTFT min–max (ms) |
| --- | --- | ---: | ---: |
| 1 | EP indexed control | 150.807–151.495 | 95.555–97.602 |
| 1 | EP pipeline | 154.880–156.628 | 95.718–99.580 |
| 1 | Default NCCL | 154.358–155.889 | 100.714–106.701 |
| 32 | EP indexed control | 2753.159–2813.412 | 256.969–345.959 |
| 32 | EP pipeline | 2874.913–2940.681 | 259.377–325.113 |
| 32 | Default NCCL | 3528.217–3599.450 | 390.370–542.817 |

### Complete decode Graph attribution

All complete replays were resolved with validated GPU annotations and no
reported missing-node/integrity issues. Values are the median across active
DECODE ranks of rank-local means over eight complete Graph replays. Timing is
for a full model forward, not one MoE layer. Component interval unions can
overlap and must not be summed as wall time; instrumented Graph time is separate
from the HTTP throughput measurement. The classifier includes the new
`host_sm_dispatch_pipeline_kernel` as dispatch transport.

| Batch | EP control forward (ms) | Pipeline forward (ms) | NCCL forward (ms) | Control dispatch (ms) | Pipeline dispatch (ms) | Control combine (ms) | Pipeline combine (ms) | NCCL collectives (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.942 | 6.777 | 6.622 | 1.799 | 1.517 | 1.015 | 1.111 | 2.364 |
| 32 | 11.514 | 11.022 | 8.910 | 4.297 | 3.590 | 1.760 | 1.967 | 3.390 |

Dispatch falls 15.67% at B1 and 16.44% at B32. At B32, expert GEMM remains
approximately 1.92 ms. Combine residence increases despite unchanged combine
code; differences in arrival/wait timing are a possible explanation, but this
trace does not isolate their cause. Consequently, the dispatch reduction is
partly offset and total forward improves only 2.38% at B1 / 4.27% at B32.
This experiment changes READY/count encoding, metadata scheduling and ACK
handoff together; it cannot assign a separate speedup to each mechanism.

The remaining B32 gap is still concentrated in the communication path: pipeline
dispatch is about 3.59 ms and fused combine about 1.97 ms, while NCCL framework
collectives occupy about 3.39 ms. These are different operation scopes, with
routing/scatter/reduction work fused into EP transport, so this comparison is
not a pure PCIe bandwidth measurement. The initial TX phase still precedes RX,
RX consumes one ready peer payload at a time, and the pipeline retains per-frame
reader fences and final ACK drain. Removing those dependencies requires a
separate protocol/performance evaluation rather than dropping synchronization.

### Correctness and scope

CUDA compilation succeeded. Five focused cases passed in 96.77 seconds:
int32/int64 single-frame/fallback tests, multi-peer Graph reuse with vector and
scalar-tail payloads, and a five-rank changing-route/stage-refill test. These
exercise empty/skewed ranks, eager-prefill transitions, source-bound fallback
and Graph replay with changing routes; they are not exhaustive verification.

All four greedy output samples and 29 compared prefix logprobs match the EP
control exactly. NCCL's generated text/tokens also match all four samples;
prefix logprob deltas remain max 0.218722 / mean 0.0146705. This limited output
sanity check does not prove numerical equivalence with NCCL.

Keep the pipeline opt-in. It requires at most 32 configured source rows per
active channel and fitting index storage; other shapes retain the existing
protocol. Sequential dispatch phase counters are deliberately unsupported in
this mode because groups overlap; complete CUDA timelines remain supported.

## Combine attribution after the dispatch pipeline (2026-10-25)

This is a diagnostic follow-up to the preceding full-model comparison, with
no transport implementation changes. Both groups use implementation `a47ec9d`
and the same deployment/configuration, switching only the dispatch pipeline
flag. Each full-model run uses input 512/output 8 for diagnostic captures,
with eight warmups and counter-off/on/on/off trials at B1 and B32. B1 is pinned
to source rank 0; B32 has four requests on each of eight source ranks. Prompts
match across configurations. Every instrumented trial records 384 combine
operations per rank, with no dropped samples or reported transport errors.
The ownership sampler observed no external GPU process; transport validation
passed with SHM only and zero NVLink counter delta.

### Where the additional combine residence appears

Values below are microseconds per combine invocation, averaged over the two
counter-on trials. For each rank-local operation, select the last-finishing
CTA, then average rank 0 for B1 or all eight ranks equally for B32. Do not sum
concurrent CTA times or subtract clocks on different GPUs. Phase counters
include instrumentation barriers and are diagnostic residence times, not raw
PCIe latency or the uninstrumented end-to-end benchmark.

| Phase (us) | B1 indexed | B1 pipeline | B32 indexed | B32 pipeline |
| --- | ---: | ---: | ---: | ---: |
| Send descriptor preparation | 1.399 | 1.400 | 2.694 | 2.703 |
| Send payload writes | 0.000 | 0.000 | 4.573 | 4.552 |
| Payload fence / READY publication | 0.000 | 0.000 | 3.380 | 3.366 |
| Receive metadata / selection | 1.040 | 1.037 | 0.874 | 0.874 |
| READY checks / progress loop | 10.736 | 12.959 | 9.882 | 13.711 |
| Receive payload / reduction / output | 5.679 | 5.737 | 5.820 | 5.835 |
| ACK publication / reader fence | 2.275 | 2.275 | 3.544 | 3.538 |
| Final ACK drain | 0.000 | 0.000 | 3.584 | 3.508 |
| Other control | 0.582 | 0.583 | 1.615 | 1.614 |
| Total sampled CTA residence | 21.712 | 23.990 | 35.967 | 39.700 |

B32 total residence increases 3.734 us; READY-loop residence increases 3.829 us,
while payload read/reduction increases only 0.014 us and ACK drain decreases
0.075 us. B1 shows the same pattern: +2.278 us total, +2.223 us READY-loop.
Both repeated trials agree: B32 READY-loop means are 9.927/9.836 us before and
13.740/13.682 us after. Missing-READY observations across all B32 CTAs increase
from 57,300/56,599 to 81,966/81,640. These observations distinguish an increase
in unsuccessful readiness checks from a mere shift in phase labels.

The current B32 breakdown is approximately 34.5% READY-loop residence,
14.7% receive payload/reduction, 11.5% payload writes, 8.5% READY publication,
8.9% ACK publication and 8.8% final ACK drain. The READY phase includes host
flag-load latency, collective CTA progress barriers and sleep/poll overhead;
it does not measure pure remote producer delay. Receive payload also includes
reduction/output work. Therefore these counters identify the affected phase
but do not establish PCIe saturation, a particular late producer, or its cause.

B1 rank 0 has no remote sends and therefore no outgoing ACK drain. The other
seven ranks do have outgoing ACK drain, averaging 5.734 -> 5.715 us. A zero
in the B1 source-rank table does not imply that the protocol has eliminated ACKs.

Requested combine payload bytes are identical in every matched trial:
7,229,440 bytes written and read at B1, and 233,566,208 bytes written and read
at B32 (each direction, summed over all ranks/CTAs). B32 sends/receives 57,023
remote partial rows per trial. Equal counts do not prove identical per-token
routes, but exclude an aggregate payload-volume increase in these captures.

Counter-on versus off mean diagnostic TPOT is 6.707 versus 6.656 ms for B1
indexed, 6.519 versus 6.489 ms for B1 pipeline, 12.128 versus 12.154 ms for B32
indexed, and 11.686 versus 11.640 ms for B32 pipeline. These short profiled runs
are instrumentation checks, not replacement throughput results. Counter-off
still uses the diagnostic specialization. The preceding ordinary-kernel
five-repeat results remain the end-to-end comparison.

A consistent explanation is that faster dispatch changes rank arrival skew:
a receiver reaches combine earlier and exposes more of the outstanding peer
work as READY waiting. For example, a receiver arriving 3 us earlier with an
unchanged final peer publication waits 3 us longer without any slower payload
transfer. This is an explanation consistent with the data, not an isolated
cross-rank causal measurement. The observed dispatch-plus-model improvement
and combine residence increase are compatible; combine duration alone is not
a reason to revert the dispatch pipeline.

### Why fewer payload bytes do not guarantee higher throughput

The actual NCCL decode trace contains `AllGather_RING_LL` and
`Reduce_Sum_bf16_RING_LL`. The deployed SGLang variable-size reduce-scatter
implementation groups one `ncclReduce` per source split, returning the reduced
split to its owner. It is incorrect to model this baseline as all-gather
followed by a full all-reduce that broadcasts every final token to every GPU.
There is still dense collective work that sparse routing can avoid.

NCCL's [version-matched LL primitive](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/prims_ll.h)
uses fixed-layout data/flag lines, vector loads, bounded step credits and a
receive/reduce/send pipeline. It also waits and acknowledges consumption, but
amortizes those costs over its regular protocol. SHM logs alone do not identify
copy-engine use, and no copy-engine explanation is needed for this result.

For this diagnostic workload, 57,023 remote partial rows over 384 calls and
32 source tokens correspond to approximately 4.64 remote contributing ranks
per token. That is about 34% fewer remote destinations than seven; top-k 8
across eight ranks does not mean one destination. This is a logical fanout
comparison, not a measurement of NCCL physical PCIe traffic. Host multicast
avoids repeated producer deposits, but every destination still reads its own
copy across PCIe. Flag traffic, read transaction latency, metadata dependencies,
barriers and underfilled transfers are not removed by payload deduplication.

A useful model is payload bytes / effective bandwidth plus exposed dependency
latency and control work, accounting for overlap. Reducing bytes to 0.67x while
achieving 0.5x the effective transfer rate would increase payload time to 1.34x,
even before extra control costs. This is an illustrative ratio, not measured
bandwidth. The normal B32 trace still places dispatch at 3.590 ms and fused
combine at 1.967 ms versus 3.390 ms in NCCL framework collectives; operation
scopes differ because EP fuses routing/scatter/reduction. Expert GEMM remains
about 1.92 ms. The remaining measured gap is concentrated in communication
and its control path, not evidence that sparse routing lacks value.

### Next priorities, grounded in the current workload

1. **Overlap ready-peer reads with waiting for missing peers.** The current
   combine CTA sends all of its outgoing rows, then waits for every actual
   contributor of its current token before reading any of their payloads.
   B32 uses four active channels and four source tokens per rank: each CTA
   already owns one source token. Token lookahead alone cannot fix this case.
   A bounded per-peer readiness mask and prefetch stages could read available
   contributions while other peers are pending, then reduce in the established
   rank order to preserve rounding. Keep the full payload-reader parallelism,
   acquire-before-read, and ACK-after-final-read guarantees. Dynamic routing
   determines the expected mask on each invocation; do not cache old routes.
2. **Advance publication and overlap independent TX/RX work.** The descriptor
   batch and fence batch already exist. The remaining all-TX-before-RX schedule
   can expose the slowest sender. Measure first/final publication and rank-local
   entry timing before choosing bounded TX batches or dedicated control/TX
   workers; do not reduce the 256-thread receiver just to create more groups.
   Increasing channels without useful work adds protocol traffic rather than
   creating payload concurrency.
3. **Separate output readiness from outgoing slot reuse.** Final ACK drain is
   about 8.8% here and did not cause this regression, so it is a lower priority.
   Epoch-separated mailbox storage could defer a drain until that slot is
   actually reused. This needs per-slot generation/credit tracking and bounded
   backpressure across Graph replay, changing routes and eager transitions.
   Deleting ACKs or allowing an early overwrite is not an optimization.

DeepEP provides useful precedents, with hardware and version distinctions:
[v1.2.1 low-latency combine](https://github.com/deepseek-ai/DeepEP/blob/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee/csrc/kernels/internode_ll.cu)
uses batched completion signaling and a three-stage prefetch/consume pipeline;
its [runtime](https://github.com/deepseek-ai/DeepEP/blob/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee/csrc/deep_ep.cpp)
alternates two low-latency buffers. Its receive hook separates issuing transfers
from receiving. These suggest stage ownership, batched control and lifetime
separation, but do not directly implement the proposed host-mailbox protocol.

At the inspected V2.5 revision,
[combine](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/combine.cuh)
assigns independent token work to warps and still has initial/final GPU barriers.
The [EP API](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
models SM demand using fanout and transfer rates and can defer the combine
reduce epilogue to the consumer's wait. Neither version removes necessary
dependencies, and its NVLink/RDMA/TMA path is not a drop-in implementation for
mapped host memory without P2P. The transferable objective is to do useful work
while other data is pending and amortize completion, not to copy its warp count.

## Ready-peer combine prefetch (2026-10-25)

Implementation `98f5ce9` adds opt-in `EP_XPRESS_DECODE_READY_PREFETCH=1` to
fused decode combine. The receiver stages contributors whose READY is visible
while other actual contributors are pending, retaining all 256 payload threads
and the configured cross-peer vector-read grouping. It reduces in the existing
ascending-rank order with the same BF16 rounding/local-add semantics. Per-token
loaded masks are rebuilt each call. No host routing, route cache, early ACK or
relaxed lifetime is introduced. All-ready calls use direct reads; staging over
40 KiB per CTA falls back to the existing path. Prefill kernels are unchanged.

### Matched full-model comparison

A fresh same-machine control retains the preceding dispatch pipeline and sets
only the new prefetch switch to zero. The baseline is SGLang's default NCCL
backend, not host CE. All runs use eight H20D GPUs, TP=DP=EP8, the same
Qwen3-30B-A3B-Instruct-2507 BF16 model, SGLang v0.5.21 source and offline image,
Triton experts, FlashInfer attention, full decode Graphs with local buckets
1/2/4/8, and eager prefill. Input/output lengths are 512/32; each load has eight
warmups and five timed repeats. Decode descriptor/publish/peer batches are
32/8/2, with indexed dispatch and its control-prefetch pipeline enabled.

The EP environments differ only in `EP_XPRESS_DECODE_READY_PREFETCH`. Timed
prompt hashes match. Producer NUMA placement passed before/after profiling.
All transports were SHM, with zero P2P/NET/NVLS records and available NVLink
counters reporting zero delta. The two-second ownership sampler observed no
external GPU process. A source manifest identifies the measured runtime files; those hashes match
the final committed implementation. The remote source tree is synchronized,
so a commit label alone is not used as provenance.

TTFT below is the median of each repeat's request-mean TTFT. Decode TPS is
aggregate post-prefill throughput from the benchmark's steady-state window.
It is not total output tokens divided by whole-request latency.

| Batch | Implementation | Mean TTFT (ms) | Last TTFT (ms) | Decode TPS |
| --- | --- | ---: | ---: | ---: |
| 1 | EP control | 97.235 | 97.235 | 155.745 |
| 1 | EP ready-peer prefetch | 96.995 | 96.995 | 154.805 |
| 1 | Default NCCL | 101.890 | 101.890 | 154.722 |
| 32 | EP control | 265.751 | 288.363 | 2911.245 |
| 32 | EP ready-peer prefetch | 273.695 | 301.226 | 2916.384 |
| 32 | Default NCCL | 445.221 | 490.184 | 3566.185 |

Against the EP control, candidate Decode TPS is -0.60% at B1 and +0.18% at B32.
Mean TTFT is -0.25% at B1 and +2.99% at B32. This does not establish an
end-to-end improvement. B32 TPS ranges overlap almost completely; TTFT varies
substantially within both groups. Against NCCL, candidate TPS is effectively
tied at B1 (+0.05%) and remains 18.22% lower at B32. Its existing TTFT advantage
over NCCL is not evidence that this decode prefetch change improved prefill.
Keep the switch disabled by default.

| Batch | Implementation | Decode TPS min-max | Mean TTFT min-max (ms) |
| --- | --- | ---: | ---: |
| 1 | EP control | 155.482-156.214 | 96.905-102.612 |
| 1 | EP ready-peer prefetch | 153.112-155.325 | 96.367-98.175 |
| 1 | Default NCCL | 154.108-155.269 | 100.917-102.213 |
| 32 | EP control | 2878.198-2928.527 | 257.064-331.137 |
| 32 | EP ready-peer prefetch | 2876.587-2926.147 | 256.974-331.078 |
| 32 | Default NCCL | 3545.957-3613.186 | 426.126-514.476 |

An initial implementation (`daa96ed`) staged one ready peer at a time. It measured
154.194/2918.694 TPS at B1/B32, also without a useful gain. The final candidate
retains grouped independent peer loads before shared stores to preserve the
existing memory-level parallelism. This change did not produce a repeatable
throughput improvement in the measured workload. These sequential bounded
experiments are not confidence intervals or a study of all routing distributions.

### Functional evidence and limits

Three focused CUDA cases passed in 84.97 seconds: vector width 2048, scalar-tail
width 258, and a larger width that exceeds the staging limit. Tests deliberately
delay GPU producers, change route masks including absent and deduplicated
experts, alternate eager calls and Graph replay, and compare against the exact
rank-order BF16 reference with cancellation-sensitive values. After restoring
grouped peer loads, the two affected vector/scalar cases passed again in
74.52 seconds. The unchanged capacity fallback was not redundantly retested.

All four sampled full-model greedy texts/tokens and 29 compared prefix logprobs
match the EP control exactly. All four generated outputs also match NCCL;
NCCL prefix logprob differences remain max 0.218722 / mean 0.0146705. These
limited samples do not establish numerical equivalence for arbitrary workloads.

### Complete decode Graph traces and interpretation

Values are medians across active DECODE ranks of rank-local means over eight
complete Graph replays. GPU annotation/integrity checks passed. Components are
interval unions that can overlap; they are not additive wall-time components
and are separate from the ordinary HTTP timing above.

| Batch | Implementation | GPU forward (ms) | Dispatch transport (ms) | Fused combine (ms) | NCCL collectives (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | EP control | 6.853 | 1.496 | 1.084 | - |
| 1 | EP ready-peer prefetch | 6.807 | 1.524 | 1.145 | - |
| 1 | Default NCCL | 6.634 | - | - | 2.377 |
| 32 | EP control | 11.008 | 3.587 | 1.958 | - |
| 32 | EP ready-peer prefetch | 11.006 | 3.573 | 1.960 | - |
| 32 | Default NCCL | 8.945 | - | - | 3.434 |

B32 combine and total forward are essentially unchanged. B1 combine residence
increases about 5.6%, while other profiled intervals vary; this does not establish
a single-request speedup. The new scheduling opportunity did not translate into
lower measured combine time in this experiment.

This implementation overlaps ready payload reads with outstanding remote
producer work. It does not make the CTA poll while its payload instructions
are running, change sender publication timing, or remove the final contributor
needed for output. It adds shared stores/loads and readiness-mask processing
whenever partial staging occurs. Which of those costs offsets hidden wait time
has not been isolated by these full-model traces; this run does not claim that
READY residence fell or that PCIe bandwidth improved. A further diagnosis would
need staged/all-ready incidence and phase timings before redesigning the
control/payload split. Retain the validated experimental option for that work,
but keep the preceding default path as the deployment choice.


## Decode combine control-warp polling (2026-10-25)

`EP_XPRESS_DECODE_CONTROL_WARP=1` assigns READY and outgoing ACK polling to
one control warp instead of repeatedly joining all eight warps. Lanes retire
observed epochs, with two peer slots per lane covering the supported rank range.
After READY completion, one CTA barrier transfers acquired visibility to all
256 payload readers. Identity-check barriers are omitted only with operation
diagnostics disabled. The dynamic expected-rank mask, reduction order, reader
fences, ACK publication and final drain remain unchanged. Both descriptor-batch
and original ACK schedules support the control warp. Prefill is unchanged;
ready-peer payload prefetch remains disabled.

### Matched full SGLang comparison

The candidate, same-source control with the switch disabled, and default NCCL
were run sequentially on the same eight H20D GPUs. Configuration matches the
preceding experiment: Qwen3-30B-A3B-Instruct-2507 BF16, SGLang v0.5.21,
TP=DP=EP8, Triton experts, FlashInfer attention, full decode CUDA Graphs with
local buckets 1/2/4/8, and eager prefill. Each batch has eight warmups and five
timed repeats, using 512 input and 32 output tokens. EP descriptor/publish/peer
batches are 32/8/2; indexed dispatch and its control-prefetch pipeline are on.

The EP environments differ only in the control-warp flag. Timed prompt hashes,
image, model configuration and framework source match. Runtime source hashes
match the committed implementation. All generated lengths were checked.
Producer NUMA placement passed before/after profiling. The GPU ownership
sampler observed no external process. NCCL logs contain SHM only, with no
P2P/NET/NVLS transport, and all available NVLink counters show zero delta.
Operation diagnostics and phase counters are off during performance timing.

TTFT is the median of repeat-level request-mean TTFT; Decode TPS is aggregate
post-prefill steady-state throughput, not output tokens divided by total latency.

| Batch | Implementation | Mean TTFT (ms) | Last TTFT (ms) | Decode TPS |
| --- | --- | ---: | ---: | ---: |
| 1 | EP control | 96.992 | 96.992 | 155.258 |
| 1 | EP control warp | 97.238 | 97.238 | 156.304 |
| 1 | Default NCCL | 101.994 | 101.994 | 154.959 |
| 32 | EP control | 293.047 | 316.974 | 2929.929 |
| 32 | EP control warp | 291.501 | 315.210 | 2920.720 |
| 32 | Default NCCL | 403.405 | 471.351 | 3568.335 |

Relative to EP control, Decode TPS changes +0.67% at B1 and -0.31% at B32.
Mean TTFT changes +0.25% and -0.53%, respectively. Ranges overlap: this bounded
experiment does not establish an end-to-end gain. Keep the new option disabled
by default. The candidate's B32 TTFT is 27.74% lower than NCCL, while Decode TPS
is 18.15% lower; the pre-existing prefill advantage does not come from this
decode-only change. Sequential repeats are not confidence intervals.

| Batch | Implementation | Decode TPS min-max | Mean TTFT min-max (ms) |
| --- | --- | ---: | ---: |
| 1 | EP control | 154.622-156.277 | 96.008-98.442 |
| 1 | EP control warp | 154.962-156.714 | 94.585-99.591 |
| 1 | Default NCCL | 154.415-155.651 | 99.968-104.259 |
| 32 | EP control | 2906.778-2932.153 | 257.525-320.656 |
| 32 | EP control warp | 2908.556-2945.600 | 263.364-301.436 |
| 32 | Default NCCL | 3505.224-3590.102 | 385.194-428.138 |

### Functional checks and scope

Two focused four-GPU cases passed: width 2048 with diagnostics off covers
changing contributor masks, deduplicated experts, empty ranks, delayed
producers, exact cancellation-sensitive rank-order arithmetic, and alternating
Graph replay/eager calls. Width 258 with diagnostics on checks the scalar path
and identity/status rendezvous over changing eager routes. An initial attempt
to combine diagnostics with Graph mode was rejected by the existing API;
the diagnostic check was corrected to eager execution. No protocol change was
needed. These tests and model runs use descriptor batch 32; the nonbatched ACK
branch and configurations above four/eight ranks respectively were not run.

All four sampled greedy full-model texts/tokens and 29 compared prefix logprobs
match the EP control exactly. All four outputs also match NCCL; prefix logprob
differences are max 0.218722 and mean 0.0146705. This is limited output sanity,
not proof of numerical equivalence for all workloads. Ready-prefetch combined
with control-warp polling remains untested.

### Complete decode Graph traces

These are medians across active DECODE ranks of rank-local means over eight
complete Graph replays. Annotation and trace-integrity checks passed. Component
intervals can overlap and are not additive wall-time components. Trace timings
are separate from uninstrumented HTTP performance; fused communication kernels
include control/routing/reduction work and are not pure PCIe transfer time.

| Batch | Implementation | GPU forward (ms) | Dispatch transport (ms) | Fused combine (ms) | NCCL collectives (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | EP control | 6.789 | 1.554 | 1.116 | - |
| 1 | EP control warp | 6.752 | 1.548 | 1.076 | - |
| 1 | Default NCCL | 6.639 | - | - | 2.389 |
| 32 | EP control | 10.988 | 3.561 | 1.962 | - |
| 32 | EP control warp | 10.947 | 3.558 | 1.927 | - |
| 32 | Default NCCL | 8.931 | - | - | 3.414 |

In this trace, combine decreases 3.60% at B1 and 1.80% at B32. At B32 that is
about 35 microseconds across the entire forward; total forward decreases only
about 40 microseconds (0.37%). This small trace-local change is consistent with
no resolved throughput gain in the ordinary HTTP measurements. It does not
establish the isolated cost of READY, ACK drain or a CTA barrier, since phase
counters were not enabled here.

Dispatch remains about 3.56 ms at B32, and the candidate forward remains
10.95 ms versus NCCL's 8.93 ms. The change removes repeated whole-CTA polling
barriers, but retains host control-word accesses, actual contributor completion,
all outgoing publication before reception, payload reads/reduction and final
lifetime protection. It cannot hide a late actual producer or accelerate the
unchanged dispatch path. The measured result therefore supports keeping the
option available for further work, not claiming that control-warp polling alone
resolves the throughput gap. Further work should identify which dispatch and
producer-readiness dependencies lie on the critical path before changing more
synchronization or sacrificing receive parallelism.


## Fine-grained dispatch pipeline attribution (2026-10-25)

This change adds opt-in diagnostics to the current dispatch pipeline; transport
scheduling, routing, payload arithmetic and ACK lifetimes are unchanged. Schema 3
records TX and three concurrent role-leader timelines instead of treating the
pipeline as the legacy sequential kernel. No new synchronization barriers are
inserted. Phase transitions, protocol counts and diagnostic-only shared completion
flags are recorded; ordinary contexts retain the uninstrumented specialization.
The final implementation timestamps phase transitions only, avoiding repeated
clock reads while polling in the same phase.

### Workload and validation

The full Qwen3-30B-A3B-Instruct-2507 BF16 model runs on eight H20D GPUs using
SGLang v0.5.21, TP=DP=EP8, Triton experts, FlashInfer attention, full decode
Graphs and eager prefill. The current dispatch pipeline remains enabled;
combine control-warp polling and ready-peer prefetch remain disabled. Existing
channel/frame/descriptor/publish/peer settings are unchanged. P2P/NVLink paths
are disabled; NCCL transport logs show SHM only and available NVLink counters
show zero delta. The ownership sampler observed no external GPU process.

Each B1/B32 diagnostic load uses 512-token prompts after eight warmups and then
identical requests in off/on/on/off order, generating eight tokens. B1 is pinned
to DP rank 0; B32 places four requests on each rank. Every on trial records 384
pipeline dispatches per rank, with zero status errors or dropped samples. B1
uses two active channels; B32 uses four. All role partitions and received-byte
identities passed. Prompt hashes and payload/empty-notification totals match
across repetitions and between initial and transition-only instrumentation.
The final runtime source hashes were verified against the recorded manifest.

A focused five-GPU Graph check passed in 69.22 seconds, covering changing routes,
empty/skewed ranks, scalar tails, eager transitions, profile reset and sample
buffer overflow. The transition-only refinement was then validated by the full
model's partition/status/count assertions. No transport optimization is included.

### Ordinary-path reference and probe perturbation

A separate ordinary context, without prepared counters, used eight warmups and
five timed repeats at 512/32 input/output tokens. B1 TTFT/Decode TPS were
96.834 ms / 155.450; B32 were 292.164 ms / 2912.860. Its complete Graph trace
measured B32 dispatch 3.581 ms, fused combine 1.959 ms and forward 11.003 ms,
consistent with the preceding reference. B1 dispatch/forward were 1.525/6.757 ms.
These ordinary timings are separate from the eight-token diagnostic requests;
B1 placement and tracing also differ, so they are not a strict probe A/B.

Prepared-but-disabled counters still use the diagnostic specialization. The
following within-server off/on comparison measures active counter perturbation,
not the total difference from the ordinary kernel:

| Batch | Counter-off TPOT (ms) | Counter-on TPOT (ms) | Increase |
| --- | ---: | ---: | ---: |
| 1 | 6.590 | 7.099 | 7.73% |
| 32 | 11.827 | 12.899 | 9.07% |

The B32 dispatch kernel itself rises from 78.005 to 98.881 us in the matched
trace (+26.76%). The initial per-iteration probe measured 76.989/98.102 us;
transition-only timing did not eliminate observer cost. Thus the detailed values
below characterize instrumented execution and do not precisely partition the
ordinary 3.581 ms. The hotspot ordering remained stable across both probe forms.
Kernel trace durations include entry/record emission; the internal elapsed field
excludes record emission and initial counter setup.

### B32 role timelines

For every rank-local operation, select its latest-finishing CTA. Then average
per rank/trial and across eight ranks and two trials. The control column uses
that CTA's latest-finishing control group; it is not the sum of both groups.
TX, control and RX have separate denominators and overlap after the TX prefix.
No cross-GPU timestamp subtraction or concurrent-duration summation is used.

Sampled CTA residence is 95.997 us: TX prefix 12.120 us, RX role 83.709 us, plus a small start/join remainder.

| TX leader interval | Mean (us) |
| --- | ---: |
| Route/local copy | 5.280 |
| Metadata writes | 0.592 |
| Payload writes | 1.130 |
| Writer fence/READY publication | 5.118 |

| Latest control-group interval | Mean (us) | Share of that role |
| --- | ---: | ---: |
| READY polling | 6.780 | 8.1% |
| Descriptor read/output allocation | 12.108 | 14.6% |
| Expert IDs/weights | 7.725 | 9.3% |
| Metadata fence/stage handoff | 18.204 | 21.9% |
| Occupied-stage waiting | 16.321 | 19.6% |
| ACK publication | 7.006 | 8.4% |
| Outgoing ACK drain | 14.697 | 17.7% |
| Other | 0.358 | 0.4% |

| RX interval | Mean (us) | Share of RX role |
| --- | ---: | ---: |
| Waiting for an available stage | 50.016 | 59.8% |
| Payload instructions | 8.428 | 10.1% |
| Reader fence/completion handoff | 12.176 | 14.5% |
| Waiting after both groups finished incoming work | 13.088 | 15.6% |

RX stage waiting includes delayed READY, descriptor/metadata preparation, ACK
handoff and scheduling; it is not entirely metadata time. Control stage waiting
and RX payload/completion overlap and must not be added. Metadata-commit and RX
completion include system fences, warp/group barriers and handoff, so this run
cannot attribute their entire cost to one fence instruction. Leader payload
intervals exclude other-lane stragglers charged to completion. Nevertheless,
both probes show that payload instructions are a small share of RX residence,
while control preparation, stage supply and completion dominate the measured
schedule. This is not a PCIe bandwidth utilization measurement.

### READY rounds, empty notifications and B1

At B1, the first on trial has 41,243 empty notifications and 1,765 payload-bearing notifications (95.90% empty), with 7,229,440 requested payload-read bytes. 66.03% of control polling rounds observe no ready peer.

At B32, the first on trial has 28,993 empty notifications and 57,023 payload-bearing notifications (33.71% empty), with 233,566,208 requested payload-read bytes. 5.06% of control polling rounds observe no ready peer.

The B1 source rank has no incoming payload: its sampled CTA takes 37.654 us,
including a 12.553 us TX prefix; its latest control group's READY/ACK-publication/drain
intervals are 6.933/5.006/10.520 us. Other ranks,
including nonrecipient routes, average 50.571 us, with 20.905 us READY polling
in the selected control group. Round fractions are counts, not waiting-time
fractions. B1 and B32 therefore have different control bottlenecks.

The B1 captured local shape is one row per rank but uses two channels, leaving
one source-row partition empty. The current adapter raises channel width to
accommodate the configured single-frame source bound. This preserves arbitrary
legal skew but creates empty-channel signaling for the small Graph bucket.
Zero-count notifications are still required by today's completion/reuse protocol;
the measurements do not authorize dropping them without replacing that contract.

### Optimization priorities suggested by the measurements

1. Allow payload reception once descriptors and output locations are available,
   while control reads IDs/weights. Join metadata and payload completion before
   ACK; investigate whether fences can be deferred or shared at that join with
   an explicit memory-order proof. Do not simply delete fences.
2. Give each control group independent next-descriptor staging, so one occupied
   stage does not prevent preparing the next peer. Preserve the 256-thread RX
   group before trying more CTAs or smaller receive groups.
3. Separate local receive readiness from outgoing-buffer reclamation where the
   protocol can enforce credit at the actual next reuse. The measured post-receive
   tail supports examining this dependency but does not prove it is all removable.
4. For B1, investigate group-consistent bucket bounds and batched empty completion
   notifications rather than increasing channel count. Preserve dynamic routing,
   Graph generation safety and fallback interoperability.

The current evidence favors stage/metadata/completion work over payload-vector
micro-tuning. It does not establish a speedup for any proposed change, nor identify
the isolated system-fence cost or prove which rank caused another rank to wait.


## Decode dispatch early publication and double stages (2026-10-25)

The opt-in decode dispatch pipeline now publishes a payload descriptor as soon
as its wire rows and GPU output range are known. The full 256-thread RX group
can copy payload while its owning control warp reads expert IDs/weights and
writes inverse/owner metadata. Each of the two control warps has two independent
shared-memory descriptor slots, allowing preparation of the next peer while
RX consumes the current one. Payload remains a single deduplicated producer
copy in host memory; the additional slots do not duplicate payload buffers.

ACK still joins both readers: metadata lanes system-fence and warp-join before
the owner can reap a slot on its next loop; payload lanes system-fence and
RX-group-join before releasing the completion flag. The owner acquires this
flag before publishing the host ACK. An early payload completion does not
permit early host reuse. Slots are independently reclaimed by their owner;
finished incoming stages and outgoing ACKs are drained before operation exit.
No CPU routing, static route cache, host atomics or extra CTA-wide barrier is
introduced. Configured eligibility, fallback behavior, token/rank deduplication,
weights, combine reduction order, channels, frame sizes and NUMA policy remain
unchanged. This replaces the previous schedule within the existing opt-in
`EP_XPRESS_DECODE_DISPATCH_PIPELINE=1` specialization; the flag remains off by
default.

### Matched full-model comparison

The previous committed schedule, default SGLang NCCL, and candidate ran
sequentially on the same eight H20D GPUs. The model was
Qwen3-30B-A3B-Instruct-2507 BF16 with SGLang v0.5.21, TP=DP=EP8, Triton experts,
FlashInfer attention, full decode CUDA Graphs (local buckets 1/2/4/8), and eager
prefill. Each B1/B32 case used eight warmups and five timed repeats, 512 input
and 32 output tokens, greedy sampling, and cache flush before each request
batch. B32 placed four requests on every DP rank. Detailed phase counters and
operation diagnostics were disabled during timing; ordinary CUDA traces were
collected afterwards using eight output tokens.

EP configurations and runtime environments match exactly. The only changed
runtime source is the dispatch pipeline header. Timed prompt hashes, model
configuration, image and framework source match. Source SHA256 manifests were
verified before launch and against the final runtime implementation. Producer
NUMA placement was verified before timing and before/after tracing. GPU process
sampling found no external workload. P2P/NVLink paths were disabled for all
runs; NCCL logged SHM only, no P2P/NET/NVLS, and all available NVLink counters
had zero delta.

TTFT is the median of repeat-level request-mean TTFT. Decode TPS is aggregate
steady-state throughput after prefill, not total output divided by request
latency. Values below are medians of five repeats; the ranges are observed
min/max, not confidence intervals.

| Batch | Implementation | TTFT (ms) | Decode TPS | Batch completion (ms) |
| --- | --- | ---: | ---: | ---: |
| 1 | Previous EP pipeline | 96.221 | 155.982 | 295.550 |
| 1 | Early publication + double stages | 96.746 | 158.444 | 293.372 |
| 1 | Default NCCL | 101.943 | 154.947 | 302.509 |
| 32 | Previous EP pipeline | 291.541 | 2910.583 | 656.315 |
| 32 | Early publication + double stages | 321.673 | 3139.696 | 651.231 |
| 32 | Default NCCL | 524.257 | 3579.579 | 822.509 |

Relative to the previous EP schedule, B1 Decode TPS improves 1.58% and B32
improves 7.87%. Against default NCCL the candidate is 2.26% higher at B1 but
12.29% lower at B32. B32 candidate and previous-EP Decode TPS ranges do not
overlap in this bounded sequential comparison. This supports a decode gain
on the measured workload, not a general speedup across models or batch sizes.

TTFT does not improve against previous EP: it increases 0.55% at B1 and 10.34%
at B32. B32 candidate TTFT is 38.64% lower than NCCL, but that pre-existing
prefill advantage cannot be credited to this decode-only change. B32 TTFT
ranges are broad and overlap; no causal explanation for the observed increase
was established. Do not dismiss it as noise or claim a prefill gain. With only
32 output tokens, B32 whole-batch completion improves just 0.77%, despite the
7.87% steady-state decode gain.

| Batch | Implementation | TTFT min-max (ms) | Decode TPS min-max |
| --- | --- | ---: | ---: |
| 1 | Previous EP pipeline | 95.915-97.804 | 154.812-156.694 |
| 1 | Early publication + double stages | 96.260-97.873 | 156.992-158.654 |
| 1 | Default NCCL | 100.957-102.583 | 154.634-155.508 |
| 32 | Previous EP pipeline | 256.922-309.724 | 2895.359-2918.603 |
| 32 | Early publication + double stages | 258.017-343.266 | 3084.427-3165.949 |
| 32 | Default NCCL | 452.493-550.190 | 3552.509-3615.184 |

### Ordinary CUDA timeline

Each entry is the median across active decode ranks of the rank-local mean
for eight complete Graph replays. B1 has one active decode rank; B32 has eight.
Kernel intervals within a category are unioned. Categories can overlap, and
these trace timings are distinct from uninstrumented HTTP measurements. No
cross-rank timestamps are subtracted or durations summed.

| Batch | Version | GPU forward (ms) | Dispatch transport (ms) | Combine mailbox (ms) | Expert GEMM (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Previous EP pipeline | 6.759 | 1.548 | 1.087 | 0.774 |
| 1 | Early publication + double stages | 6.662 | 1.495 | 1.049 | 0.773 |
| 32 | Previous EP pipeline | 11.017 | 3.591 | 1.963 | 1.922 |
| 32 | Early publication + double stages | 10.276 | 2.977 | 1.834 | 1.923 |

B32 dispatch falls from 3.591 to 2.977 ms (-17.08%), combine from 1.963 to
1.834 ms, and complete forward from 11.017 to 10.276 ms (-0.741 ms). Expert
GEMM remains approximately 1.923 ms. Thus dispatch savings are not canceled
by a slower combine in this trace. Combine source did not change; its timing
can change with rank arrival and upstream scheduling, so the reduction is not
an independently established combine-kernel speedup. NCCL's B32 forward is
8.958 ms with 3.423 ms of framework NCCL kernels. EP dispatch plus combine
remain substantial, but these categories are not identical collective work
and should not be treated as a pure PCIe bandwidth comparison.

This turn did not repeat the perturbing role-counter experiment. The previous
approximately 60% RX stage-wait fraction cannot be reused as the new fraction,
and the current trace does not isolate which remaining ACK/fence interval
accounts for the residual gap. The measured improvement is consistent with
removing the metadata-before-payload dependency; separate ablations would be
needed to distinguish early-publication gains from double-stage gains.

### Functional validation and limitations

Two focused eight-GPU checks passed in 81.67 seconds. They cover int32 IDs with
2048-wide vector payload reads and int64 IDs with 258-wide scalar reads,
changing per-token routes/weights, rank and row dependent payloads, empty ranks,
skewed ownership, same-rank expert deduplication, multiple peers per control
group, retained Graph buckets, alternating replay streams, eager transitions,
and configured-bound fallback. Every result matches the exact BF16 reference.
The tests deliberately vary payload values and weights so row/metadata mixups
cannot hide behind constant data. No broad functional suite was rerun.

Four greedy full-model outputs and all 29 compared prefix logprobs match the
previous EP schedule exactly. All four generated token sequences also match
NCCL; prefix logprob differences have maximum 0.218722 and mean 0.0146705.
These are limited output checks, not exhaustive numerical equivalence. Larger
world sizes, other models, longer generations and serving arrival distributions
were not measured. The current result establishes a bounded decode improvement
while leaving B32 NCCL parity and the TTFT increase unresolved.


## Decode control records and receive ACK batching (2026-10-25)

The two opt-in changes were tested separately against the previous EP schedule
and a fresh SGLang default NCCL reference. Neither demonstrates an end-to-end
throughput improvement on these workloads. The defaults remain control records
off and ACK batch one; no combined full-model speedup is claimed.

### Implementation and measurement

`EP_XPRESS_DECODE_CONTROL_RECORDS=1` replaces host-indirect route metadata reads
in pipeline dispatch with receiver-specific contiguous records. Records contain
the source/wire descriptor and packed local expert IDs with original FP32 weight
bits, preserving top-k positions. Routing is computed from live GPU tensors on
every call. The final writer spreads consecutive fields across warp lanes using
a bounded ephemeral row list. Payload remains a deduplicated producer copy, and
ACK still joins all metadata and payload readers. The larger control region is
included in group-wide eligibility; insufficient space retains fallback.

`EP_XPRESS_DECODE_ACK_BATCH=4` keeps the full 256-thread mailbox receiver and
rank-ordered arithmetic, but fences and publishes individual ACKs after at most
four local rows. A partial tail flushes; a token boundary barrier prevents shared
reader-mask/local-row overwrite. Necessary publication, receive and reuse
ordering is retained. It does not change the general prefill pair queues.

The matched configuration is eight H20D GPUs, Qwen3-30B-A3B-Instruct-2507 BF16,
SGLang v0.5.21, TP=DP=EP8, Triton experts, FlashInfer, full decode Graphs with
local buckets 1/2/4/8, and eager prefill. Requests use 512 input and 32 output
tokens, eight warmups and five timed repeats per B1/B32 case, greedy sampling,
and cache flush before each batch. B32 places four requests on each DP rank.
Detailed profiling is disabled; only end-to-end performance is measured.

All comparisons verified identical timed prompt hashes, image, framework source,
model configuration and launch arguments. The first three EP ablations use
identical source with one build flag changed at a time. The final records variant
changes only the opt-in writer branch in the pipeline header; the flag-off
control/ACK behavior is unchanged. Final runtime source hashes were verified
against the measured manifest. Each run verified producer NUMA placement, no
external GPU process, NCCL SHM only (no P2P/NET/NVLS), and zero delta for every
available NVLink counter. Startup compilation/Graph capture is excluded.

TTFT is the median of repeat-level request-mean TTFT. Decode TPS is aggregate
steady-state output throughput after prefill. Min/max ranges below are observed
repeat extrema, not confidence intervals.

| Implementation | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Original EP | 96.478 | 158.780 | 291.160 | 3116.728 |
| Default NCCL | 101.720 | 154.816 | 413.752 | 3547.579 |
| Control records, token-lane writer | 96.895 | 158.066 | 265.692 | 3100.529 |
| Control records, parallel writer | 96.019 | 158.493 | 282.196 | 3124.940 |
| Receive ACK batch 4 | 96.273 | 158.274 | 263.947 | 3120.909 |

| Implementation | B1 TTFT range | B1 TPS range | B32 TTFT range | B32 TPS range |
| --- | ---: | ---: | ---: | ---: |
| Original EP | 95.419-97.124 | 158.064-159.189 | 260.858-301.243 | 3106.438-3151.671 |
| Default NCCL | 100.578-103.516 | 154.174-155.111 | 399.590-542.051 | 3522.805-3586.183 |
| Control records, token-lane writer | 95.851-97.698 | 157.800-159.371 | 262.620-309.415 | 3080.181-3136.658 |
| Control records, parallel writer | 95.403-96.714 | 157.013-158.541 | 256.183-309.158 | 3102.573-3142.107 |
| Receive ACK batch 4 | 95.533-97.332 | 157.141-158.894 | 258.116-321.585 | 3107.070-3139.332 |

### Interpretation and validation limits

The final parallel-record writer changes Decode TPS by -0.18% at B1 and +0.26%
at B32 versus original EP. ACK batching changes it by -0.32% and +0.13%.
All candidate/control repeat ranges overlap, so these results do not establish
a performance gain. Final records remain about 11.9% below NCCL at B32 decode.
TTFT medians vary substantially, especially at B32; these decode-only changes
provide no evidence of a causal prefill improvement. No fine-grained phase
profiling was collected in this round, so a particular TX/RX phase cannot be
claimed as the measured cause of the records result.

The ACK ablation exposes a workload applicability limit: the unchanged adaptive
rule selects four channels for 32 global tokens, while each rank returns four
local tokens. Each CTA therefore owns one output row and flushes a one-row tail,
even when ACK_BATCH=4. B1 similarly has at most one local row per CTA. These
shapes cannot amortize a per-token fence with intra-CTA multi-token batching.
This follows from the current scheduler/partition code and agrees with earlier
role counters (B32 local_tokens=4, active_channels=4; B1 1 and 2); those are not
new phase measurements. Wider/skewed native inputs do exercise batch and tail
handling, but no performance benefit on those inputs is claimed. Reducing
channels to force batching would sacrifice receiver parallelism and was not done.

The original token-lane record writer serialized top-k writes within one lane;
the final writer removes that serialization. Nevertheless its full-model result
is still indistinguishable from control. Receiver-local ID translation and fewer
indirect loads trade against additional per-destination control packing/writes;
this tradeoff is an explanation to investigate, not a profiled attribution.

Two focused eight-GPU dynamic-route tests passed in 81.21 s with both initial
features enabled, covering 32/64-bit IDs, vector/scalar hidden widths, empty and
skewed ranks, Graph/eager switching and payload/metadata/ACK reuse. After
parallelizing record writes, one focused vector/dynamic-route test passed in
69.82 s. All EP variants matched control exactly on four short greedy prompts,
including token IDs and 29 common-prefix logprobs. NCCL matched their generated
text; this is a limited output sanity check, not exhaustive numerical validation.
Full-model performance was tested with each feature independently, not both on.


## Balanced decode sends, independent TX/RX and deferred reclamation (2026-10-25)

The first ablation matrix below uses the same runtime source. The original EP schedule remains the
control; the new schedules are independently selectable build-time options.
The unchanged baseline is SGLang's default NCCL backend.

The matched configuration is eight H20D GPUs, Qwen3-30B-A3B-Instruct-2507 BF16,
SGLang v0.5.21, TP=DP=EP8, Triton experts and FlashInfer, eager prefill and full
decode CUDA Graphs with local buckets 1/2/4/8. Each case uses 512 input and 32
output tokens, eight warmups and five measured repeats, with greedy sampling
and cache flush. B1 and B32 have identical timed prompt hashes across variants.
Profiling is disabled during all measurements; startup and capture are excluded.

Every run verified no external GPU process, producer NUMA placement, NCCL SHM
transport without P2P/NET/NVLS, and zero delta for all available NVLink counters.
The image, model configuration, framework source and runtime source manifest
match. All EP launch arguments match; only the listed build settings differ.
Receive width remains 256 threads. Requested channel capacity remains 32, with
the existing global-capacity rule selecting active channels. No batch-specific
route or token behavior is hardcoded.

The balanced variant partitions the concatenated live remote owner lists. The
split variant adds a full resident TX CTA for each unchanged RX CTA. Deferred
reclamation removes outgoing end-of-operation ACK drain, but acquires the last
actual READY generation and waits for its ACK before overwriting that mailbox.
Publication batch is eight unless explicitly shown as four. These changes
affect bounded decode combine, not the prefill pair queues or dispatch.

TTFT is the median of repeat-level request-mean TTFT. Decode TPS is aggregate
steady-state output throughput after prefill. Ranges are observed extrema, not
confidence intervals.

| Implementation | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Default NCCL | 101.816 | 155.068 | 406.694 | 3564.449 |
| Original EP | 96.868 | 158.434 | 257.568 | 3113.953 |
| Balanced sends | 96.519 | 158.403 | 320.521 | 3142.826 |
| Balanced + independent TX/RX | 96.717 | 159.067 | 282.609 | 3129.544 |
| Balanced + independent TX/RX + deferred reclaim | 97.152 | 158.757 | 315.151 | 3159.860 |
| Deferred reclaim, publish batch 4 | 97.265 | 160.073 | 257.329 | 3055.239 |

| Implementation | B1 TTFT range | B1 TPS range | B32 TTFT range | B32 TPS range |
| --- | ---: | ---: | ---: | ---: |
| Default NCCL | 101.422-105.705 | 154.334-155.747 | 403.610-492.670 | 3525.681-3602.254 |
| Original EP | 96.444-97.473 | 157.563-158.884 | 256.973-342.355 | 3091.203-3166.656 |
| Balanced sends | 94.973-99.567 | 158.068-159.182 | 283.610-322.073 | 3127.974-3159.777 |
| Balanced + independent TX/RX | 95.718-97.848 | 157.317-159.238 | 256.481-322.102 | 3106.485-3135.428 |
| Balanced + independent TX/RX + deferred reclaim | 97.075-98.819 | 156.321-159.878 | 257.446-341.976 | 3100.106-3161.380 |
| Deferred reclaim, publish batch 4 | 96.590-104.909 | 158.365-163.240 | 257.142-257.815 | 3031.810-3072.709 |

### Final GPU generation cache and control repeat

The final deferred variant retains the last published generation per mailbox
in a producer-owned GPU array, initialized once and retained across Graph
replays. It needs one host ACK acquire at reuse, without a dependent host READY
read. Unused slots retain zero; absent routes retain the last actual generation.
The extra storage is eight bytes per configured destination/token slot. There
is no per-layer allocation, CPU planning or host rendezvous.

Only the CUDA source changed from the first matrix: runtime additions are gated
by deferred reclamation, and the remaining changes are comments. The opt-in
cache is validated below together with the unchanged flag-off EP schedule,
using the same final runtime source, image, model, prompts and configuration.
All runtime source hashes were checked against the measured manifest. The NCCL
reference above remains unchanged.

| Implementation | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| GPU generation cache + balanced/split/deferred | 96.577 | 159.609 | 293.116 | 3162.134 |
| Original EP repeat | 96.727 | 158.453 | 309.046 | 3138.170 |

| Implementation | B1 TTFT range | B1 TPS range | B32 TTFT range | B32 TPS range |
| --- | ---: | ---: | ---: | ---: |
| GPU generation cache + balanced/split/deferred | 95.850-97.795 | 159.138-160.393 | 257.219-344.603 | 3137.274-3173.252 |
| Original EP repeat | 96.081-99.049 | 158.339-159.457 | 257.360-342.949 | 3113.496-3153.926 |

### Outcome and limits

The final B32 Decode TPS is 1.55% above the first EP control
and 0.76% above the repeated control, but 11.29% below NCCL.
The first control's repeat range overlaps the final candidate. This does not
establish a robust throughput advantage; no scheduling option is promoted to
the default. Publication batch four regresses B32 in this measurement, so it
is not selected. The generation cache barely changes the previous deferred
variant's median, despite removing one dependent host control read.

The sending partition now has a provable at-most-one-row difference between
channels, outgoing ACK drain no longer gates local completion in deferred mode,
and TX/RX can run independently without reducing receiver width. These changes
remove specific structural limitations, but this end-to-end comparison does
not show that they dominate total latency. No new fine-grained model profiling
was collected, so attributing the remaining gap to payload bandwidth, READY
waits, dispatch or expert work would require a separate measurement. TTFT varies
substantially; this decode-focused change does not establish a causal prefill
improvement. No batch-specific channel rule was introduced to fit these two
benchmarks. Existing CTA width and tile sizes are retained.

Two initial eight-GPU dynamic-route checks passed in 79.76 s, covering vector
and scalar hidden widths, int32/int64 routes, empty/skewed ranks, changing
channel widths and Graph/eager reuse. The final runtime passed three focused
checks in 84.06 s: eight-GPU dynamic Graph routing, a direct three-GPU mailbox
reuse test with rotating slow readers and no dispatch/barrier between calls,
and Graph profiling reset/overflow with both roles. The direct test changes
routes, skips generations, includes all-empty operations, changes active
channels and requires exact outputs before collective close. A strengthened
one-way test also passed in 65.94 s: two ranks only send while the destination
is deliberately delayed, allowing the producers to attempt next-generation
reuse before that reader completes. The runtime source was unchanged for this
additional check. Every EP model variant matched the original on four short greedy prompts and 29 common-prefix
logprobs. These are bounded correctness checks, not exhaustive proofs of
numerical equivalence or resilience to failed workers.

## Bounded decode packets, ready-task reception and two-bank credit (2026-10-25)

A separate opt-in combine path groups dynamic token routes into bounded packets,
commits one READY and one ACK per nonempty peer/tile, and stages independently
ready receive tasks in HBM before the original rank-ordered reduction. Two host
banks per packet retain exact per-bank publication generations; only the actual
reader releases credit. Packet rows remain sparse and partial packets flush
immediately. All receive CTAs retain 256 threads. Dispatch and prefill pair
queues are unchanged.

The first two-row implementation uses inherited active channels. The final
vector implementation adds coalesced host stores, four independent receive
loads before dependent HBM stores, and vector reduction of staged data. Its
four-row variants separately compare inherited and eight packet-only channels.
The two-row to four-row comparison changes several factors and does not isolate
packet size. The final channel comparison holds source and all other settings
fixed. A final-source original-EP repeat checks the opt-out path.

Matched full-model configuration: eight H20D GPUs, Qwen3-30B-A3B-Instruct-2507
BF16, SGLang v0.5.21, TP=DP=EP8, Triton experts, FlashInfer, eager prefill and full
decode Graph buckets 1/2/4/8. Each run has 512 input and 32 output tokens, eight
warmups and five measured repeats. Timing excludes startup, capture and profiling;
the diagnostic traces are collected afterwards. Timed prompt hashes match.
Both implementations disable P2P/NVLink communication, verify NCCL SHM without
P2P/NET/NVLS, and observe zero increments in every available NVLink counter.
No external GPU process was observed; producer NUMA placement is verified for
every EP run. The image, model configuration and framework source match. The
fresh NCCL reference was collected before the EP source changes; it does not
invoke the EP extension. The vector channel ablation and original-EP repeat share one source manifest.
A later packet-only window-retirement fix is separately revalidated and retimed;
its recorded runtime source matches the committed files. The fix is compiled
out of the unchanged original-EP path.

TTFT is the median of repeat-level mean request TTFT. Decode TPS is aggregate
steady-state output throughput. Ranges below are observed extrema, not confidence
intervals. TTFT is variable and these decode changes do not establish a causal
prefill speedup.

| Implementation | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Default NCCL | 102.140 | 155.297 | 408.849 | 3576.224 |
| Original EP, publish 8 | 96.535 | 158.323 | 283.511 | 3136.536 |
| Initial two-row packets | 97.332 | 147.097 | 287.815 | 2883.731 |
| Vector four-row packets, eight channels | 96.888 | 156.556 | 266.648 | 3195.791 |
| Vector four-row packets, inherited channels | 97.281 | 150.725 | 273.223 | 3109.955 |
| Final-source original EP repeat | 96.564 | 158.371 | 274.174 | 3119.277 |
| Final window-safe four-row packets, eight channels | 97.389 | 157.185 | 274.658 | 3160.533 |

| Implementation | B1 TPS range | B32 TPS range |
| --- | ---: | ---: |
| Default NCCL | 154.683-155.919 | 3523.191-3592.283 |
| Original EP, publish 8 | 157.438-159.022 | 3108.017-3145.568 |
| Initial two-row packets | 146.236-148.071 | 2865.443-2901.170 |
| Vector four-row packets, eight channels | 155.620-157.510 | 3159.980-3202.895 |
| Vector four-row packets, inherited channels | 150.362-151.184 | 3083.476-3158.026 |
| Final-source original EP repeat | 157.117-158.454 | 3105.292-3138.170 |
| Final window-safe four-row packets, eight channels | 156.535-157.586 | 3127.889-3168.098 |

### Separate diagnostic timeline

Values are medians across active DECODE ranks of each rank's mean per complete
Graph replay. One replay includes all 48 MoE layers. These are source-attributed
GPU intervals, including waiting; they are not PCIe wire time or unprofiled HTTP
TPOT. No GPU clocks or CTA durations are summed across ranks. Packet preparation
and HBM reduction are included explicitly.

| Implementation | B32 GPU forward (ms) | Dispatch transport (ms) | Combine transport/fused (ms) | Packet preparation (ms) | Packet HBM reduction (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original EP, publish 8 | 10.251 | 2.965 | 1.821 | 0.000 | 0.000 |
| Initial two-row packets | 11.106 | 3.040 | 2.116 | 0.121 | 0.316 |
| Vector four-row packets, eight channels | 10.091 | 2.975 | 1.328 | 0.134 | 0.136 |
| Vector four-row packets, inherited channels | 10.301 | 2.996 | 1.553 | 0.133 | 0.131 |
| Final-source original EP repeat | 10.271 | 2.990 | 1.813 | 0.000 | 0.000 |
| Final window-safe four-row packets, eight channels | 10.204 | 2.969 | 1.443 | 0.135 | 0.136 |

The final window-safe four-row/eight-channel candidate achieves 3160.533 B32
Decode TPS: 1.32% above the original-EP repeat and 0.77% above the first control.
The observed repeat ranges overlap, so this is a small measured difference,
not an established robust throughput gain. It remains 11.62% below default NCCL.
B1 throughput is 0.75% below the original-EP repeat. No packet option is enabled
by default. The earlier 3195.791 result predates the final window-retirement
barrier and is retained as development evidence, not the final candidate result.

The initial two-row implementation regresses: independently ready reception
and reduced per-packet signaling do not by themselves pay for staging, extra
kernels and reduced load concurrency. Vector transfers and more independent
packet channels recover much of that cost. The same-source channel ablation
achieves 3195.791 TPS with eight channels versus 3109.955 with inherited channels;
it changes only packet-channel count. The subsequent window-safe retest is a
separate measurement and cannot isolate the barrier cost from run variation.

For the final candidate, the median across active ranks of the complete combine
stage (preparation, transport and reduction together) is 1.721 ms per Decode
step, versus 1.813 ms for the original fused path. The individual component
medians are approximately 0.135 ms preparation, 1.443 ms transport and 0.136 ms
HBM reduction; medians need not add to the median of the total. Dispatch remains
about 2.969 ms and is unchanged by this experiment. Extra preparation/reduction
consumes a substantial part of the transport improvement. The next investigation
should account for dispatch scheduling and these kernel boundaries rather than
assuming fewer READY/ACK records guarantees NCCL-level end-to-end throughput.
These are model GPU-stage observations, not an attribution of dispatch time
specifically to payload, READY or ACK.

The initial packet build passed five focused GPU checks in 102.79 seconds:
lagging readers and consecutive reuse without dispatch/host rendezvous; scalar
width and shifted-storage exact comparison against legacy arithmetic;
eight-GPU changing routes and bucket reuse under Graph; and profile reset/overflow.
The vector/four-row build passed three focused checks in 88.34 seconds:
scalar delayed-reader bank reuse, vector-width bitwise legacy comparison, and
eight-GPU dynamic Graph routes. The final window-retirement fix additionally
passed a 129-token delayed-reader check in 67.92 seconds. That check forces one
channel, asserts more than 32 actual packet tasks per CTA, and queues 32 changing
route invocations without intervening dispatch or CPU rendezvous, exercising
multiple descriptor windows and both payload banks. Tests cover empty/skewed
ranks, skipped generations, partial tiles and changing channel counts. Eight
existing CPU configuration tests also pass. Each measured EP variant matches
the original on four short greedy model prompts and all 29 common-prefix
logprobs. These checks are bounded evidence, not exhaustive correctness or
failure-tolerance proofs. Two- and four-row layouts were exercised; eight-row
tuning was not evaluated in this experiment.

To opt into the measured candidate, use `EP_XPRESS_DECODE_PACKET_ROWS=4` and
`EP_XPRESS_DECODE_PACKET_CHANNELS=8` with the existing bounded fused-decode
configuration. Leave legacy combine balanced/split/deferred/prefetch options
disabled. Setting packet rows/channels back to zero retains the original path.

## Independent dispatch scheduling and host access width

This follow-up evaluates separate dispatch TX/RX CTAs, delayed dispatch-bank
reclamation, per-handle return-index reuse, and host access width independently.
All new build options remain opt-in. The routing contract stays dynamic BF16:
one payload per source token with rank deduplication, current GPU-generated
routes on every invocation, weights applied once, and rank-ordered combine.
No CPU routing collective is added to the operation path.

The model configuration remains eight H20D GPUs, Qwen3-30B-A3B-Instruct-2507,
SGLang v0.5.21, TP/DP/EP 8, Triton experts, FlashInfer, producer-local NUMA
arenas, eager prefill and decode Graph buckets 1/2/4/8. Requests have 512 input
and 32 output tokens, with eight warmups and five measured repeats per batch.
Prefix caching is disabled. Allocated channels remain 32, with adaptive decode
width unless explicitly stated; packet combine uses four-row packets and eight
channels where enabled. Both backends disable NVLink and GPU P2P.

The [archived NCCL baseline](../benchmarks/baselines/sglang_h20d_ep8_bf16_host.json)
combines two matching sessions, ten samples per batch. It records arithmetic
means, sample standard deviations, medians and observed extrema. Later EP-only
changes reuse this record rather than relaunching NCCL. Hardware, software,
model, workload or execution-setting changes invalidate reuse; evidence of
environment drift also requires reassessment. Image, framework source, topology,
server arguments and timed prompt hashes were checked against this baseline.
Each model run verifies exclusive GPU use, producer NUMA pages, SHM transport,
absence of P2P/NET/NVLS channels and zero available NVLink counter deltas.

The following table uses **arithmetic means**, unlike the median tables in
earlier sections. The TTFT values are means of repeat-level mean request TTFT;
Decode TPS is aggregate steady-state output throughput. TTFT varies considerably
at batch 32 and should not be used to assign causality to individual decode
scheduling changes.

| Implementation | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL, 10 samples | 101.801 | 155.051 | 431.710 | 3557.948 |
| Original EP control | 96.549 | 157.968 | 273.852 | 3119.389 |
| Independent dispatch TX/RX | 98.025 | 160.594 | 304.582 | 3147.807 |
| Add delayed dispatch reclamation | 96.928 | 160.529 | 282.975 | 3146.452 |
| Split dispatch + packet combine, reclamation off | 100.828 | 157.497 | 307.529 | 3176.882 |
| Add per-handle return-index reuse | 96.921 | 158.521 | 268.030 | 3180.324 |
| Add 64-bit host metadata loads | 96.879 | 160.283 | 298.537 | 3326.965 |
| Add vector dispatch writes, unroll 4 | 97.356 | 161.017 | 314.996 | 3338.217 |
| Use two active decode dispatch channels | 100.368 | 161.124 | 270.153 | 3364.737 |

Independent dispatch removes the local-TX-before-RX dependency while keeping
256 payload receive threads per RX CTA. Its observed B32 throughput gain is
small, about 0.9%. Delayed reclamation does not improve throughput further and
adds 140,509,184 bytes of pinned dispatch storage in this configuration; it
remains disabled. In the separate diagnostic timeline, dispatch falls from
2.928 to 2.701 ms, but combine rises from 1.816 to 2.053 ms. Deferring the drain
has moved much of the wait rather than removing it from the critical path.

The return index removes the repeated `owner_rows -> inverse -> token` traversal
from packet preparation. Its B32 TPS difference is about 0.1% and the observed
ranges overlap: 3163.16-3204.25 without reuse, 3150.16-3228.03 with reuse. The
diagnostic preparation interval falls from 0.136 to 0.101 ms, partly offset by
dispatch initialization rising from 0.108 to 0.118 ms. This cleanup is useful
structurally but does not account for the main throughput gap.

The 64-bit metadata-load ablation improves B32 TPS by 4.6% relative to index
reuse alone; its observed range is 3304.93-3353.93. In the diagnostic timeline,
dispatch falls from 2.893 to 2.521 ms and the complete GPU forward from 10.111
to 9.692 ms. Expert GEMM and activation intervals remain approximately
1.92 and 1.31 ms. These timings are medians across active DECODE ranks of
per-rank mean complete Graph replays, each covering 48 MoE layers. They include
waiting and must not be interpreted as pure PCIe transfer time. READY remains
a system-acquire load; only aligned descriptor/ID access changes from two
32-bit instructions to one 64-bit instruction. Hardware PCIe transaction
counts are not inferred from this change.

Vector dispatch writes add only 0.3% B32 TPS in their individual ablation,
with overlapping ranges (3304.82-3362.50 versus 3304.93-3353.93). Selecting
two active decode dispatch channels instead of the adaptive four at B32 gives
3364.737 TPS; an independent repeat gives 3367.791 TPS. Packet combine keeps
eight channels in both cases. The second session has B1 TTFT 97.702 ms,
B1 TPS 161.091 and B32 TTFT 269.552 ms. A ten-sample mean over these two
matching candidate sessions is B1 TTFT **99.035 ms**, B1 TPS **161.107**,
B32 TTFT **269.852 ms**, B32 TPS **3366.264**. B32 throughput is 7.9% above
the original EP control and still 5.4% below the archived default NCCL mean.
These results do not establish a twofold speedup or a universal channel policy.

The first two-channel diagnostic has dispatch 2.387 ms, packet transport
1.431 ms, packet preparation 0.101 ms and HBM reduction 0.136 ms per complete
48-layer decode replay; complete GPU forward is 9.556 ms. Dispatch remains
the largest communication kernel interval. Narrower load/store instructions,
dependent metadata reads and per-stage progress still matter despite lower
logical payload volume. The model trace alone does not separate the remaining
payload, READY and credit times. Further optimization should measure that
distinction before removing synchronization or adding CTAs.

### Communication-only shape sweep and limits

The standalone benchmark uses eight ranks, BF16 H=2048, top-k 8, 128 experts,
eight allocated/active communication channels, eight warmups and twenty timed
samples per shape. It replays changing routes including rank deduplication,
hotspots and padding. Both control and candidate verify GPU-local producer
NUMA pages. These are **medians of per-sample maximum rank durations**, not
model TTFT/TPS. The candidate uses the same split/index/wide-access options,
but this sweep fixes eight dispatch channels; it does not use the model's
two-channel setting. The candidate's packet combine uses eight channels.

| Tokens per rank | Control dispatch (us) | Candidate dispatch (us) | Control combine (us) | Candidate combine (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 60.80 | 53.28 | 44.42 | 37.42 |
| 4 | 72.29 | 60.96 | 44.96 | 39.84 |
| 16 | 80.83 | 68.96 | 65.66 | 76.03 |
| 64 | 110.24 | 140.05 | 189.41 | 236.40 |
| 256 | 271.98 | 471.76 | 709.31 | 956.34 |
| 1024 | 1175.71 | 1080.82 | 1171.46 | 1165.86 |
| 4096 | 4371.38 | 3935.44 | 4485.18 | 4468.42 |

The 64/256-token bounded path **regresses**. Larger 1024/4096-token operations
use the generic rings, where independent decode CTAs and packet combine are
not selected; the wider shared metadata helper still applies. At 4096 tokens,
the candidate's logical dispatch host-write/read rates are 31.62/121.66 GB/s
summed over all eight ranks, versus 28.31/109.15 in the control. Combine's
logical host-write/read rate is 96.55 GB/s versus 95.32. These count useful BF16
payload only, exclude metadata/padding/protocol traffic, and are not physical
PCIe utilization measurements. Earlier exploratory runs without verified NUMA
placement are excluded from this comparison. No new NCCL sweep was run.

Use the [small-decode preset](../adapters/sglang/presets/host_sm_bf16_ep8_graph_small_decode.sh)
only as an opt-in reproduction of the measured B1/B32 configuration. It keeps
delayed dispatch reclamation disabled and preserves the necessary READY/ACK
ordering. It does not automatically choose a path for larger batches. The
generic defaults remain unchanged; a general runtime selection policy needs
further shape/routing coverage rather than hard-coding a decision from these
two model batch sizes. The preset's exported EP/NCCL settings were checked
against the measured environment.

Across the ten final candidate samples, B1 TPS spans 160.66-161.72 (sample
standard deviation 0.34), and B32 TPS spans 3340.00-3400.69 (standard deviation
20.09). B1 TTFT spans 96.90-109.88 ms and B32 TTFT 254.86-292.38 ms. These are
observed ranges, not confidence intervals. The candidate's B32 TPS range remains
below the archived NCCL range of 3503.53-3613.55.

Targeted GPU checks cover dynamic/skewed/empty routes, int32/int64 IDs,
H=2048 vector transfers and H=258 scalar fallback, Graph bucket changes,
eager fallback, multiple outstanding handles, and profiler reset/overflow.
The final vector build passes two distributed CUDA cases plus eleven CPU
configuration/workspace cases (13 cases and three subtests, 81.13 seconds).
Both final seven-shape communication sweeps pass numerical checks, including
the large generic-ring path. Every measured EP model variant matches the
original on four short greedy output probes and their common-prefix logprobs.
These are bounded correctness checks, not exhaustive workload or fault-tolerance
coverage. The runtime source manifest is verified before each model launch;
raw traces and deployment details remain outside the public repository.

## Direct dispatch slots and independent progress (2026-10-28)

This follows the small-decode experiment above, on the same eight H20D GPUs,
SGLang 0.5.21, BF16 Qwen3-30B-A3B-Instruct-2507, TP/DP/EP8, Triton experts
and FlashInfer. Prompts contain 512 tokens and outputs contain 32 tokens.
Prefill is eager; decode uses full CUDA Graph buckets 1/2/4/8. There are eight
warmup batches and five measured batches per load and session. Producer NUMA
placement, exclusive GPU use, prompt hashes, output lengths and source manifests
are verified. Both comparisons exclude P2P/NVLink; the archived ten-sample NCCL
mean is reused. No new NCCL benchmark was run.

The allocated channel count remains 32, active model decode dispatch channels
remain two, and combine retains four-row packets on eight channels. Precision,
rank deduplication, per-invocation routing, ACK protection and the separate
rank-ordered HBM reduction are unchanged. See
[the protocol notes](host-address-pipeline.md) for the four opt-in build options.

### Sequential ablations

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL, ten samples | 101.801 | 155.051 | 431.710 | 3557.948 |
| Previous small-decode EP, ten samples | 99.035 | 161.107 | 269.852 | 3366.264 |
| Direct dispatch slots | 97.113 | 162.273 | 294.995 | 3420.192 |
| Direct slots + independent metadata/ACK warps | 97.580 | 160.593 | 294.887 | 3401.959 |
| Direct slots + packet pipeline | 96.223 | 161.711 | 306.417 | 3471.698 |
| Direct slots + packet pipeline + split generic dispatch | 98.410 | 162.346 | 291.759 | 3463.138 |

The independent metadata/ACK experiment is not selected: extra roles preserved
the full payload group but did not improve this model's decode throughput.
The packet pipeline overlaps preparation and commit with one 256-thread payload
group. It improves the B32 mean by 1.5% over direct slots alone in these sessions.
Generic TX/RX splitting has no clear incremental model benefit here. B32 TTFT
varies substantially with scheduling; these means do not establish a prefill
improvement from either new pipeline. Ablation sessions are not pooled as if
they were repeated measurements of one identical executable.

Separate B32 diagnostic replays report the following medians across ranks,
in milliseconds per complete 48-layer decode forward:

| Configuration | GPU forward | Dispatch transport | Packet transport | Packet prepare | HBM reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous small-decode EP | 9.547 | 2.360 | 1.428 | 0.101 | 0.137 |
| Direct slots | 9.424 | 2.240 | 1.423 | 0.101 | 0.137 |
| Direct slots + independent metadata/ACK | 9.483 | 2.278 | 1.458 | 0.101 | 0.137 |
| Direct slots + packet pipeline | 9.306 | 2.241 | 1.331 | 0.101 | 0.136 |
| Direct slots + packet pipeline + split generic dispatch | 9.308 | 2.230 | 1.315 | 0.101 | 0.136 |

Dispatch initialization/finalization remains approximately 0.118 ms. These are
kernel intervals from diagnostic runs, not physical bandwidth or the timed
HTTP request metrics. They support improvements from removing dependent host
descriptor reads and overlapping packet control with payload, but do not
separate the remaining PCIe read latency, READY wait and credit wait.

### Current communication-only sweep

The final source also passes the same seven-shape communication benchmark:
eight ranks, BF16 H=2048, top-k 8, 128 experts, eight active dispatch channels,
eight warmups and twenty measured samples with eight changing route patterns.
Routes include rank duplication, hotspots and padding. Producer NUMA placement
is verified. Values are medians of the maximum rank duration for each sample;
they are neither model latency nor physical PCIe bandwidth.

Here "control" is the previous small-decode configuration, with all four new
options disabled, on the current source. "Candidate" enables direct slots,
packet pipelining and split generic dispatch; independent metadata/ACK is off.
This differs from the earlier sweep's older control. Model dispatch uses two
active decode channels, whereas this sweep uses eight throughout.

| Tokens/rank | Control dispatch (us) | Candidate dispatch (us) | Control combine (us) | Candidate combine (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.14 | 49.79 | 36.66 | 37.79 |
| 4 | 60.50 | 58.34 | 40.29 | 39.54 |
| 16 | 69.73 | 65.34 | 77.60 | 60.16 |
| 64 | 140.16 | 106.58 | 236.22 | 148.96 |
| 256 | 477.98 | 322.40 | 952.99 | 567.23 |
| 1024 | 1088.29 | 1088.67 | 1154.45 | 1172.91 |
| 4096 | 3993.01 | 3270.70 | 4449.86 | 4474.98 |

At 64/256 tokens, bounded dispatch falls by 24.0%/32.5% and combine by
36.9%/40.5%. The 4096-token generic dispatch falls by 18.1%; the 1024-token
case is unchanged. Generic combine is unchanged by these options and varies
by 0.6%-1.6%. The one-token combine median increases by 3.1%. These results
support retaining split generic dispatch as an opt-in option for large
transfers, but do not establish a model TTFT gain. Improvements over the
previous small-decode candidate also do not imply all shapes beat every
older transport configuration or NCCL.

### Final publication-source verification

A fresh control and final candidate were measured using exactly the runtime
source prepared for publication. The source manifest verifies all 45 inputs;
unrelated local development changes are excluded. Each row below is the mean
of five measured batches per load, except the archived ten-sample NCCL row.

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL | 101.801 | 155.051 | 431.710 | 3557.948 |
| Same-day previous EP configuration | 96.742 | 160.820 | 291.224 | 3372.967 |
| Final direct slots + packet pipeline + split generic dispatch | 97.145 | 161.556 | 297.836 | 3469.614 |

The final B32 TPS mean improves by **2.9%** over the same-day control and
remains **2.5% below NCCL**. B1 changes by only +0.5%; its sample ranges overlap.
The same-day control agrees with the previous archived EP TPS mean to within
0.2% at B32. B32 TTFT is 2.3% higher than the same-day control; both sessions
have broad overlapping TTFT ranges, so a TTFT improvement is not established.
Compared with archived NCCL, final B1/B32 TTFT is 4.6%/31.0% lower and B1 TPS
is 4.2% higher. The measured small-decode workload is not evidence of a general
twofold throughput gain.

The final diagnostic replay reduces dispatch transport from 2.374 to 2.237 ms
and packet transport from 1.429 to 1.317 ms per 48-layer decode forward; GPU
forward falls from 9.567 to 9.304 ms. Dispatch is still the larger communication
interval. The shape sweep's larger percentage gains should not be substituted
for this full-model improvement.

Final B1 TPS spans 160.66-161.93 (sample standard deviation 0.51), and B32
spans 3456.07-3489.02 (standard deviation 13.84). Same-day control B32 spans
3350.25-3388.30 (standard deviation 18.88); archived NCCL spans 3503.53-3613.55.
Final B1/B32 TTFT spans 96.01-98.36 / 278.89-327.63 ms. These are observed
ranges, not confidence intervals.

The [direct-slot preset](../adapters/sglang/presets/host_sm_bf16_ep8_graph_direct.sh)
reproduces this opt-in configuration. All 52 exported EP/NCCL settings were
checked against the measured configuration. Independent metadata/ACK remains
off, and all four new options default to zero outside the preset. No routing
pattern is cached or inferred from benchmark outputs. The bounded direct-slot
path supports up to 32 source slots per active channel; larger work retains
the generic ring. A general channel/path policy is still separate work.

Focused eight-rank CUDA checks passed for dynamic/skewed/empty routes,
int32/int64 expert IDs, H=2048 vector and H=258 scalar payloads, Graph bucket
changes, eager fallback, multiple outstanding handles and small-ring wraparound.
Both final seven-shape sweeps pass numerical validation. Every model variant
matches the original four greedy output probes and their common-prefix logprobs.
Live arena snapshots verify producer NUMA placement; monitoring observes no
external GPU processes, no P2P/NET/NVLS transports and zero NVLink counter deltas.
These checks cover the measured contracts, not every possible workload.


## Host read completion and placement ablations (2026-10-28)

Four opt-in changes were measured independently, each enabling only its own
new build option over the direct-slot + packet-pipeline + split-ring preset.
The other new options remained zero. The eight H20D GPUs, SGLang 0.5.21,
BF16 Qwen3-30B-A3B-Instruct-2507, TP/DP/EP8, expert/attention backends,
512-token prompts, 32-token outputs, eager prefill and full decode Graph
buckets 1/2/4/8 remain as above. Each session has eight warmups and five
measured batches per load. No NCCL rerun was performed; the archived ten-sample
mean is reused. Both implementations exclude NVLink and GPU P2P.

Allocated channels remain 32, active model decode dispatch channels two, and
packet combine four rows on eight channels. Routing is rebuilt from current
GPU inputs; rank deduplication, weights, necessary synchronization and the
separate deterministic combine reduction are retained. See
[implementation options](host-read-progress.md) and the
[receive ordering argument](host-rx-ordering.md).

### Independent full-model measurements

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL | 101.801 | 155.051 | 431.710 | 3557.948 |
| Fresh previous-publication EP control | 96.599 | 161.367 | 282.541 | 3461.172 |
| Compact shared route metadata only | 96.796 | 161.713 | 289.444 | 3468.083 |
| Narrow receive fence scope only | 96.681 | 164.724 | 266.170 | 3650.664 |
| Read two already-ready tasks together only | 97.218 | 160.868 | 280.482 | 3484.796 |
| Receiver-owned combine slots only | 96.504 | 161.210 | 296.384 | 3465.584 |

Only the receive-fence change shows a substantial decode improvement in this
first pass: +2.1% at B1 and +5.5% at B32 over the fresh control. The metadata,
ready-batch and placement options change B32 TPS by +0.2%, +0.7% and +0.1%;
they do not establish a consistent model gain and remain off in the selected
configuration. B32 TTFT has broad within-session variation, so these ablations
do not establish a new prefill improvement.

The compact format preserves one shared source copy and combines the global
expert ID and FP32 weight into a 64-bit word; it does not shrink arena
allocations. Ready batching keeps all 256 payload workers and selects only
already-ready stages, but the first task's ACK may wait for the second task.
Receiver-owned combine slots move the existing single host copy to the
receiver's rank-local arena without adding a CPU relay. Live page snapshots
verify rank-owner locality. The snapshot tool's historical label is
`producer_placement`; for receiver-owned combine slots the arena owner is the
receiver, while dispatch remains producer-owned.


The separate B32 diagnostic replay reports median kernel intervals across ranks,
in milliseconds per complete 48-layer decode forward:

| Configuration | GPU forward | Dispatch transport | Packet transport |
| --- | ---: | ---: | ---: |
| Fresh previous-publication control | 9.317 | 2.249 | 1.314 |
| Compact metadata | 9.296 | 2.221 | 1.315 |
| Receive fence | 8.814 | 1.840 | 1.237 |
| Ready payload batch | 9.250 | 2.162 | 1.327 |
| Receiver-owned combine slots | 9.308 | 2.229 | 1.318 |

Dispatch initialization/finalization remains about 0.118 ms, packet preparation
0.101 ms, and the HBM reduction 0.136 ms. Fence scope is a measurable cost in
this synchronous host-read path; removing required ACKs is not necessary to
obtain the observed improvement. Batching reduces the dispatch interval but
slightly increases the packet interval. Receiver-local placement does not
improve packet time on this workload. These traces do not separately measure
physical PCIe bandwidth, READY waits or credit waits, and do not prove how often
the two-ready-stage branch is selected. The ready-batch option therefore remains
an experiment, not a claim of saturated PCIe or generally superior scheduling.


### Validation scope

Each option passes focused distributed CUDA checks before its model run: two
cases per option, plus a small-frame fallback case for receiver-owned slots
(nine cases total). Coverage includes changing/skewed/empty routes, int32/int64
IDs, H=2048 vector and H=258 scalar payloads, repeated stage/epoch reuse, Graph
bucket/channel changes, eager fallback and multiple outstanding handles.
The selected configuration retains the ordinary metadata control path
(`DISPATCH_RETIRE=0`). Combining the new options with independent metadata
retirement or diagnostic-enabled protocol builds is not part of these GPU
checks. The receive ordering proof records the required preconditions separately.

All independent model sessions pass four exact greedy output probes and match
their common-prefix logprobs. Source manifests, framework/image/topology,
prompt hashes, 32 output tokens per timed request, eight warmups and exclusive
GPU use are checked. Arena snapshots verify physical NUMA placement. Transport
logs report SHM with no P2P/NET/NVLS, and NVLink counters remain zero. Raw logs,
private deployment information and credentials are excluded from publication.
These are bounded validation results, not exhaustive workload coverage.


### Final same-source control and repeat

The final control turns all four new options off on the exact publication
runtime source. Its B32 TPS is within 0.3% of the fresh previous-publication
control. A second receive-fence session uses the same runtime source and
configuration as its first session; only those two identical candidate sessions
are pooled below. Each individual session has five samples per load.

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL, ten samples | 101.801 | 155.051 | 431.710 | 3557.948 |
| Final-source EP, all four options off | 96.408 | 161.370 | 270.115 | 3452.218 |
| Receive fence, first session | 96.681 | 164.724 | 266.170 | 3650.664 |
| Receive fence, repeat | 96.914 | 164.495 | 274.252 | 3657.358 |
| Selected receive fence, ten-sample mean | 96.798 | 164.610 | 270.211 | 3654.011 |

The selected B1/B32 decode means improve by **2.0%/5.8%** over the final-source
EP control and by **6.2%/2.7%** over archived NCCL. B1/B32 TTFT is 0.4%/0.04%
higher than the EP control, with overlapping ranges: this change does not
establish a prefill speedup. TTFT remains 4.9%/37.4% lower than archived NCCL,
a benefit already present before the receive-fence change.

Across the ten candidate samples, B1 TPS spans 163.93-165.20 (sample standard
deviation 0.39); B32 spans 3617.95-3685.14 (standard deviation 25.57). B1/B32
TTFT spans 95.49-97.63 / 248.82-324.94 ms. Final-source control B32 TPS spans
3406.73-3487.41, and archived NCCL spans 3503.53-3613.55. These are observed
ranges, not confidence intervals or guarantees on other models/topologies.

Use the [read-progress preset](../adapters/sglang/presets/host_sm_bf16_ep8_graph_read_progress.sh)
to reproduce the selected configuration. It enables only `RX_FENCE` among the
four new options; the remaining three are explicitly off. All four still
default to zero outside this preset. All 56 exported EP/NCCL settings match the
measured configuration, and all 45 runtime source inputs match the tested
publication snapshot. No per-token route is cached, no route distribution is
assumed and no workload-specific output is substituted.


The repeat diagnostic confirms the transport improvement on the same source:
dispatch falls from 2.233 to 1.844 ms and packet transport from 1.326 to
1.237 ms per 48-layer decode forward. GPU forward falls from 9.311 to
8.820 ms. Both final sessions pass the same output, source, placement,
exclusivity and no-P2P/NVLink checks used for the independent ablations.


## Readiness caching and idle receive polling (2026-10-28)

This experiment retains the selected receive-fence configuration and the same
hardware, model, software and workload described above. Both new options default
to zero. Eight warmups precede five timed batches per load in each session.
NCCL is not rerun: the same archived ten-sample baseline is reused. Both paths
exclude NVLink and GPU P2P. The complete 256-thread receive group, per-token
routing, rank deduplication and necessary ACKs remain intact.

### Independent measurements

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL | 101.801 | 155.051 | 431.710 | 3557.948 |
| Fresh previous-publication EP control | 97.079 | 164.956 | 271.325 | 3660.338 |
| Observation caching, first implementation | 96.746 | 163.738 | 265.724 | 3698.883 |
| Receive polling warp only | 96.818 | 165.194 | 273.788 | 3660.120 |
| First cache implementation + receive polling warp | 98.457 | 164.101 | 284.472 | 3695.915 |
| Same-source EP control, both new options off | 96.418 | 164.900 | 305.510 | 3671.399 |

The first cache implementation retains confirmed READY/credit observations but
still polls unconfirmed peers before selecting a task. Its B32 TPS rises 1.1%
against the fresh previous-publication control while B1 falls 0.7%. Receive-warp
polling alone is essentially unchanged at B32. Combining them does not establish
an additional benefit. These small differences are not a general throughput
claim; the same-source control itself changes B32 TPS by +0.3%. B32 TTFT varies
substantially between batches and sessions.

The separate uninstrumented B32 trace reports dispatch/packet transport intervals
of 1.828/1.233 ms for the fresh control, 1.743/1.240 ms for observation caching,
and 1.840/1.221 ms for receive-warp polling, per 48-layer decode forward.
The polling change shifts small costs between transports without reducing the
model's critical path materially.

### Fine dispatch diagnosis

A separate off/on/on/off capture measures the existing dispatch role counters.
It is not pooled with production timing. The baseline capture contains 49,152
CTA samples covering 12,288 rank-local operations, with all eight ranks,
complete split TX/RX groups, zero error status and no dropped samples. Each
role's phase partition is checked independently.

For B32, the rank-local CTA window averages 49.70 microseconds; the last CTA is
always an RX CTA. Its payload leader reports:

| RX leader interval | Microseconds |
| --- | ---: |
| Wait for a shared stage | 34.47 |
| Read and place payload | 9.73 |
| Completion synchronization | 1.08 |
| Final ACK-drain wait | 3.86 |

The two concurrent control warps each spend about 16.5 microseconds in the
READY-poll interval, 6.6-6.8 in descriptor/allocation work, 4.5-4.8 in metadata,
5.8-6.2 in metadata commit, 4.5-4.8 in ACK publication and 3.7-4.0 in drain.
These control timelines overlap one another and the payload leader: **do not
sum them**. The first stage appears at about 16.2 microseconds, with receive
completion around 40 microseconds. The long stage wait therefore includes the
continuing control/receive pipeline as well as initial remote readiness.

At B1, rank zero is explicitly the token source for this diagnostic capture.
Other ranks average a 43.84-microsecond local window; their last RX CTA spends
39.92 microseconds waiting for a stage and 1.10 reading payload. Empty
notifications and rank arrival skew matter here. The profile does not identify
all that waiting as physical PCIe latency or as an avoidable barrier.

Instrumentation increases TPOT by 3.1% at B1 and 6.3% at B32. These are diagnostic
instruction intervals, not unperturbed PCIe transfer time, and the clocks are
compared only within a rank. Together with the full-model polling ablation,
they prioritize control task supply over simply increasing payload CTA count.

### Ready-first scheduling

Code inspection identified an additional dependency in the first cache version:
a lane with a confirmed READY still waits for unknown lanes' host acquisitions
before the warp can select it. The final cache implementation first ballots its
confirmed pending set. If nonempty it consumes that set without new host reads;
only an empty set triggers polling. Each task is consumed once, so the finite
cache cannot indefinitely postpone unknown peers. This applies to packet READY
and previous-generation TX credit as well as dispatch notifications. See the
[lifetime and synchronization argument](control-progress.md).


The synchronization review also found an existing missing memory-ordering bridge
between the packet lane that acquires READY/credit and lane zero that publishes
the shared stage. The final source adds a control-warp join before publication,
including when the performance options are disabled. Non-direct indexed dispatch
likewise joins before other lanes read host descriptors; direct slots already
have the needed join. Final candidate and control measurements use this same
corrected source. The earlier cache-first prototype without this correction is
not used as the final performance or correctness result.

The final wide-vector eight-rank check exercises changing, skewed and empty
routes, CUDA Graph bucket/channel changes, multiple outstanding handles and
more than 32 packet tasks per CTA. It passes with ready-first caching enabled.
Earlier independent checks cover the observation cache and polling options with
H=2048/int32 and H=258/int64, including multi-batch task reuse. Optional
non-direct dispatch, independent metadata retirement and two-ready-payload
batching are code-reviewed combinations, not claims of final GPU coverage.

The publication runtime is checked against all 45 inputs of its measured
snapshot; unrelated local latency-instrumentation development is excluded.
The opt-in [control-progress preset](../adapters/sglang/presets/host_sm_bf16_ep8_graph_control_progress.sh)
sets dispatch READY caching on, packet caching off and receive-warp polling
off. Its exported EP/NCCL settings are checked against the isolated candidate
below. The existing read-progress preset explicitly leaves all new options off. Neither preset caches a previous
invocation's routes or substitutes workload-specific outputs.


### Coupled cache results and repeat

Only the two identical ready-first candidate sessions are pooled. The final
control includes the required warp handoff fix and disables both new options.

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL, ten samples | 101.801 | 155.051 | 431.710 | 3557.948 |
| Final-source EP, new options off, five samples | 97.356 | 164.839 | 307.801 | 3667.622 |
| Ready-first cache, first session | 97.367 | 163.375 | 291.768 | 3702.625 |
| Ready-first cache, repeat | 96.157 | 163.527 | 264.457 | 3689.694 |
| Ready-first cache, ten-sample mean | 96.762 | 163.451 | 278.113 | 3696.160 |

The candidate B32 TPS mean is **0.8% higher** than the same-source EP control,
while B1 is **0.8% lower**. The B32 sample ranges overlap: this is a small observed
workload tradeoff, not evidence of a broadly superior default. Both new options
therefore remain **off in the existing read-progress preset**; the separate
control-progress preset remains explicitly opt-in. The final isolated preset
uses dispatch caching only, as detailed below. Receive-warp polling does not
establish a model gain and remains off in both presets.

Relative to archived NCCL, the candidate B1/B32 decode means are 5.4%/3.9%
higher, with TTFT means 5.0%/35.6% lower. Most of that advantage predates this
change. The final control's B32 TTFT is unusually high compared with the fresh
previous-publication control (271.325 ms), and candidate TTFT changes from
291.768 to 264.457 ms between identical sessions. This experiment does **not**
establish an additional prefill improvement.

Across the ten candidate samples, B1 TPS spans 162.94-164.17 (sample standard
deviation 0.46) and B32 spans 3647.14-3748.31 (standard deviation 28.07).
B1/B32 TTFT spans 95.32-98.30 / 250.30-318.84 ms. Final-source control B32 TPS
spans 3632.91-3701.44. These are observed ranges, not confidence intervals.

The separate traces with native role counters disabled show dispatch transport
falling from 1.852 ms to 1.734/1.740 ms per 48-layer B32 forward. Packet transport
rises from 1.237 to 1.269/1.279 ms, while GPU forward falls from 8.822 to
8.718/8.725 ms. Dispatch improves modestly, part of the saving is offset in
packet combine, and the overall result agrees with the small end-to-end gain.
Packet preparation and reduction remain about 0.101 and 0.137 ms.

The final fine capture passes the same complete eight-rank, 49,152-sample
validation as the baseline. Its B32 last RX leader reports stage wait 23.60,
payload 9.66, completion 1.06 and ACK-drain wait 3.35 microseconds. The rank-local
window falls from 49.70 to 38.24 microseconds, but this **is not a production
23% speedup**: baseline native instrumentation adds 0.592 ms to TPOT, while the
final capture changes it by -0.037 ms (consistent with noise). The change in
instrumentation perturbation is comparable to the apparent phase-window saving
across 48 layers. Individual phases must not be assigned an uninstrumented
causal saving from these counters alone.

All final model sessions pass the original four exact greedy output probes,
with identical common-prefix logprobs. They verify the expected prompt hashes,
32 output tokens, source manifests, model/framework/image/topology, NUMA page
placement and sampled GPU exclusivity. SHM transport is observed with no
P2P/NET/NVLS and zero NVLink counter deltas. No new NCCL measurement is included.


### Isolate dispatch scheduling from packet scheduling

The coupled cache saves about 0.115 ms in dispatch but adds about 0.037 ms in
packet transport, offsetting roughly one third of the dispatch saving in the
separate B32 traces. That comparison alone does not identify whether the packet
increase comes from packet scheduling or changed arrival times upstream.
The publication therefore provides independent build options:
`READY_CACHE` controls dispatch, and `PACKET_READY_CACHE` controls packet READY
and previous-bank TX credit. Both default to zero; reproducing the coupled
candidate now requires setting both to one.

This split changes only compile-time selection. Preprocessed dispatch/packet
bodies for new 0/0 and 1/1 configurations match the preceding uncached/coupled
paths; 1/0 uses the already validated dispatch cache with the unchanged uncached
packet path. All required warp handoffs and ACKs remain. No additional broad
functional suite is run for this configuration-only isolation.


| Publication source configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| All new scheduling options off | 96.778 | 165.035 | 276.602 | 3660.431 |
| Dispatch cache only | 97.057 | 163.658 | 299.753 | 3697.432 |

Each row has five measured batches per load. Dispatch-only raises B32 TPS by
1.0% and lowers B1 TPS by 0.8% against its exact-source control. Its B32 TPS
range is 3647.03-3740.78 (sample standard deviation 38.85), overlapping the
control's 3631.09-3697.50 (standard deviation 26.40). Candidate B1 TPS spans
162.91-164.06 (standard deviation 0.44). B32 TTFT spans 252.60-340.89 ms for
the candidate and 250.67-304.98 ms for the control. No prefill gain is established.
The dispatch-only session is not pooled with the differently configured
coupled-cache sessions above, and no new repeat is presented as stronger
statistical evidence for this last isolation.

Dispatch-only does not materially outperform the coupled candidate or resolve
its B1 tradeoff. Its separate B32 trace records dispatch 1.745 ms, packet
transport 1.258 ms and forward 8.721 ms. Restoring packet's uncached scheduling
does not fully restore the earlier packet interval; the increase cannot be
assigned entirely to packet READY caching. Arrival timing, compiler layout and
ordinary measurement variation remain possible contributors.

The final opt-in preset enables **dispatch caching only** and matches all 59
exported EP/NCCL settings of this measured configuration. The existing
read-progress preset and all native defaults leave **dispatch cache, packet
cache and receive-warp polling off**. This preserves the previous selection
while keeping the measured experiments independently reproducible. The
necessary acquire-to-publisher synchronization fix is always retained.
Both publication-source model sessions pass the same exact output, source,
placement, transport and exclusivity checks described above.

The remaining evidence favors reducing descriptor/metadata preparation and
host-visible control transactions on the receive critical path. More payload
CTAs or deleting ACKs is not supported by these measurements. Confirmed READY
caching removes one real dependency but cannot eliminate initial remote arrival
or the work required to publish a correct compact layout. The next design
should reduce that work per ready group while preserving dynamic routes and
full payload receive width; simply assigning another warp to polling has not
shown an end-to-end benefit.

## Dispatch descriptor batching experiment, reverted (2026-10-28)

Three temporary schedules were evaluated to reduce the dependency from one
peer's metadata processing to publication of the next receive descriptor.
All retained dynamic token routing, rank deduplication, two control warps with
two stages each, the full 256-thread receive group, and required READY/ACK
ordering. Host payload capacity and active channel counts were unchanged.

The first schedule selected up to two already-ready peers that fit free stages,
reserved one compact output range plus independent per-peer owner ranges,
published both descriptors before metadata processing, and serviced empty
notifications and completed-stage ACKs across control lanes. It never waited to
fill a batch. Metadata was processed as one flattened row range. The second
schedule restored the original single-peer path when only one peer or one free
stage was available. The final schedule also restored warp-uniform per-peer
metadata addressing and acknowledged each completed peer before processing the
next peer's metadata. All three schedules were subsequently removed.

### Full-model comparison

The workload remains Qwen3-30B-A3B BF16, SGLang v0.5.21, the same eight GPUs and
TP/DP/EP=8, 512 input tokens and 32 output tokens, decode Graph buckets
1/2/4/8, eight warmup batches, and five measured batches per concurrency in
each EP session. The archived default-NCCL baseline has ten observations per
concurrency and was not rerun. NVLink and GPU P2P payload paths are disabled.
Values below are arithmetic means; Decode TPS uses the same steady-state
metric as the preceding comparisons.

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL | 101.801 | 155.051 | 431.710 | 3557.948 |
| Previous EP path, fresh control | 96.555 | 164.438 | 288.264 | 3664.559 |
| Initial descriptor batch | 96.747 | 163.597 | 256.268 | 3645.201 |
| Batch with single-peer fast path | 97.490 | 162.817 | 273.205 | 3645.622 |
| Final per-peer metadata and immediate ACK | 97.124 | 163.104 | 298.720 | 3663.673 |

The final candidate decreases B1 TPS by 0.81%; B32 is effectively unchanged
(-0.02%). Its B1 TPS range is 162.60-163.59 (standard deviation 0.42), versus
163.60-165.38 (0.71) for the fresh control. At B32 the ranges are
3641.41-3688.48 (18.93) and 3635.70-3686.31 (22.21), respectively. No throughput
gain is established. Final B1/B32 TTFT ranges are 95.37-98.50 / 278.29-325.55 ms;
control ranges are 95.39-97.53 / 250.88-325.74 ms. The prefill path is unchanged,
and these observations do not establish a TTFT benefit either.

Separate B32 model traces report median per-rank times across the 48-layer
forward step. These are diagnostic observations, not additional throughput
samples or a cross-device clock comparison:

| Schedule | Dispatch transport (ms) | Packet combine transport (ms) | Forward window (ms) |
| --- | ---: | ---: | ---: |
| Previous EP control | 1.823 | 1.248 | 8.807 |
| Initial descriptor batch | 1.825 | 1.293 | 8.819 |
| Single-peer fast path | 1.836 | 1.283 | 8.829 |
| Final per-peer metadata | 1.836 | 1.277 | 8.820 |

Early publication removes a real source-code ordering edge, but none of these
schedules reduced measured dispatch time on this workload. The experiment does
not isolate the contribution of additional control instructions, the frequency
of ready pairs, or upstream rank arrival. The earlier shared-stage wait counter
must not be treated as entirely removable scheduling overhead. Before adding
more batching, measure how often observed-ready work is actually delayed by
local preparation and how often two free stages have two ready peers.

### Validation and rollback

Each candidate passed two focused eight-rank dynamic-route checks: int32 IDs
with hidden width 2048 and int64 IDs with width 258. They exercise changing
routes and weights, empty/skewed ranks, same-rank expert deduplication,
multiple peers, stage reuse, Graph replay and eager fallback. Every completed
model session passed four exact greedy output comparisons with zero
common-prefix logprob difference, matching prompt hashes and output lengths,
source/image/model/topology verification, producer-NUMA placement, and sampled
GPU exclusivity. Transport records show SHM without P2P/NET/NVLS and zero
available NVLink counter deltas.

An earlier control attempt was stopped by the exclusivity monitor when another
GPU workload appeared. A subsequent attempt waited on that interrupted build's
stale extension lock. Neither produced a valid performance sample; both are
excluded. The completed fresh control rebuilt the extension successfully.

All experimental implementation changes, build options and presets were
reverted, including the default-off switch. The retained runtime remains the
preceding implementation; only this result record is published. Its existing
advantage over archived NCCL must not be attributed to this unsuccessful batch
experiment.

## Remote publication before local materialization (2026-10-29)

The bounded dispatch pipeline previously copied its local rows, local expert
metadata and inverse/return indices before writing and publishing remote data.
Local materialization is now performed after remote READY publication. Local
output ranges are still reserved first, so simultaneous receive CTAs cannot
overwrite them. All remote payload and metadata writers still execute their
system fence and CTA join before the existing release publications. Local
expert computation remains ordered after completion of the entire dispatch
kernel. Rank deduplication, dynamic routing, full-width receive groups, return
indices, ACK ownership and deterministic combine reduction are unchanged.

This removes a dependency that remote consumers did not need. It also moves
local-only GPU writes outside the pre-publication system-fence interval. The
change applies to the existing bounded decode pipeline; eager prefill and the
general ring fallback are unchanged. It does not specialize on observed token
values, expert choices, or benchmark batch labels.

### Matched model measurements

The configuration remains eight H20D GPUs, Qwen3-30B-A3B BF16, SGLang v0.5.21,
TP/DP/EP=8, 512 input and 32 output tokens, decode Graph buckets 1/2/4/8,
Triton experts and eager prefill. Both paths disable NVLink and GPU P2P.
Dispatch uses two active decode channels and packet combine uses eight channels
with four-row packets. The archived default-NCCL baseline is reused.

Two independent model sessions were run for both the unchanged EP control and
early publication. Each session uses eight warmup batches followed by five
measured batches at each concurrency. The following values are arithmetic
means of the resulting ten observations per concurrency; Decode TPS is the
aggregate steady-state output throughput used in earlier comparisons.

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Archived default NCCL | 101.801 | 155.051 | 431.710 | 3557.948 |
| Unchanged EP, fresh control | 97.545 | 164.970 | 281.399 | 3656.999 |
| Earlier remote publication | 97.813 | 166.533 | 280.221 | 3697.343 |

Earlier publication improves B1/B32 Decode TPS by **0.95%/1.10%** over the fresh
EP control. The two individual candidate sessions give B1 TPS 166.573/166.492
and B32 TPS 3694.378/3700.308; control sessions give 165.008/164.933 and
3659.725/3654.272. The improvement is small and reproduced across these runs.

Across ten observations, control B1 TPS spans 164.571-165.229 with sample
standard deviation 0.234; the candidate spans 166.356-166.952 with standard
deviation 0.168. Control B32 TPS spans 3609.010-3704.731 (29.717), and the
candidate spans 3673.709-3741.094 (22.680). These are observed variations, not
claims about all serving loads. B32 TTFT spans 252.070-327.241 ms for control
and 251.339-341.765 ms for the candidate. No TTFT improvement is established.

Compared with archived NCCL, the earlier-publication path has 7.41%/3.92%
higher B1/B32 Decode TPS and 3.92%/35.09% lower TTFT. Most of that advantage
predates this change. It must not be attributed entirely to publication order,
nor generalized to consumer GPUs that were not measured here.

Separate model traces report median per-rank times over complete 48-layer
decode forward steps. They include waiting and are not pure payload-transfer
times or additional throughput observations:

| Configuration | Dispatch (ms) | Packet combine (ms) | Forward window (ms) |
| --- | ---: | ---: | ---: |
| Control, session 1 | 1.848 | 1.240 | 8.815 |
| Control, session 2 | 1.843 | 1.228 | 8.803 |
| Earlier publication, session 1 | 1.763 | 1.212 | 8.717 |
| Earlier publication, session 2 | 1.746 | 1.210 | 8.708 |

### Independent changes without a demonstrated gain

The owner-list candidate started from the unchanged control. It explicitly
omitted owner-row/count maintenance only when bounded dispatch produced the
matching per-handle packet return index. Public inverse rows and generic
fallbacks remained valid. It produced B1/B32 TPS 165.134/3655.064 in five
observations per concurrency, providing no end-to-end throughput gain. The
implementation and its extra API argument are not retained.

Two channel ablations started from earlier publication, changing one setting
each. Four dispatch channels produced B1/B32 TPS 166.570/3683.917. Four combine
channels produced 166.296/3690.828. Neither improves on the repeated
two-dispatch/eight-combine result, so the selected preset is unchanged. In the
four-combine-channel trace, dispatch is 1.701 ms but combine increases to
1.334 ms and the full forward is 8.760 ms. A shorter isolated dispatch interval
does not establish an end-to-end benefit.

Both code candidates passed focused eight-rank dynamic-route checks with
int32 IDs at hidden width 2048 and int64 IDs at width 258. Those checks cover
changing weights and routes, empty/skewed ranks, same-rank expert deduplication,
multiple outstanding handles, Graph bucket/channel changes and eager fallback.
All seven completed model sessions have four exact greedy output matches and
zero common-prefix logprob difference. Prompt hashes, output lengths, model,
image, framework source, topology, producer-NUMA placement and sampled GPU
exclusivity were verified. Transport records show SHM without P2P/NET/NVLS and
zero available NVLink counter deltas.

### Inline data/flag LL experiment, not retained

A separate combine prototype tested the data-plus-generation publication idea
used by [NCCL LL](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/prims_ll.h).
It encoded two BF16 values and a 32-bit generation in one naturally aligned
64-bit host word, using the documented mapped-memory access guarantee in the
[CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
Each receiver validated the generation of every word before consuming its data.
The control warp selected peers whose first live word had arrived, while the
complete payload CTA retained its receive width. Ordinary packet-wide READY
publication and the preceding TX system fence were removed from this prototype;
receive completion, ACK credit and deterministic reduction were retained.

Each logical producer/destination/tile queue advanced its generation only for
actual packets. Two alternating banks retained per-queue credit. Periodic
refreshes covered inactive words in both banks to prevent stale data flags from
matching after wrap. A test build shortened flags to eight bits: 520 changing
route steps crossed wrap on 194 logical queues while also exercising delayed
readers and more than 32 live tasks per CTA. That check passed. Four production-
flag checks also passed, covering dynamic Graph routes at hidden widths 2048
and 258 plus delayed-reader bank reuse at 7 and 129 tokens. No dispatch or host
rendezvous was inserted between the lifetime-test combine calls.

The prototype's completed full-model session passed the same output,
transport, NUMA and exclusivity checks. Its five-observation means were:

| Configuration | B1 TTFT (ms) | B1 Decode TPS | B32 TTFT (ms) | B32 Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| Earlier publication, repeated reference | 97.813 | 166.533 | 280.221 | 3697.343 |
| Add inline data/flag combine | 96.699 | 164.444 | 285.357 | 3582.210 |

Decode TPS regresses by 1.25%/3.11%. B1 TPS spans 164.222-164.622 with standard
deviation 0.187, and B32 spans 3549.412-3606.154 with standard deviation 28.655.
The prototype doubles payload wire bytes, adds per-word flag processing and
permits data reads to encounter a partially written packet. Removing the
packet-wide publication dependency did not offset these costs in this
implementation. Its separate B32 trace has dispatch 1.751 ms, packet combine
1.459 ms and a complete forward window of 8.962 ms. Dispatch stays near the
earlier-publication reference, while combine grows from about 1.211 ms. This
comparison does not isolate the individual contributions or rule out other
LL layouts.

The initial prototype was stopped at bootstrap by the existing allocation-ABI
check; its larger wire layout was then described consistently on both sides.
Extending a lifetime test to 520 steps also required bounded, exactly
representable BF16 inputs: its original unbounded integer reference introduced
different rounding before and after reduction. Those failed attempts supplied
no model performance observations. Both issues were resolved before the
successful checks and model comparison above.

The LL implementation, allocation/API extensions and test-only changes are not
retained in the repository. The final runtime change is solely earlier remote
dispatch publication, with the existing two/eight-channel preset unchanged.
