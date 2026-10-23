# Full-model inference comparison

These experiments run the complete Qwen3-30B-A3B-Instruct-2507 model through
SGLang v0.5.21. They are distinct from the synthetic expert function in the
standalone transport replay.

The latest results are in
[Expert route alignment and NUMA inspection](#expert-route-alignment-and-numa-inspection).
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
