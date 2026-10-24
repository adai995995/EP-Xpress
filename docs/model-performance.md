# Full-model inference comparison

These experiments run the complete Qwen3-30B-A3B-Instruct-2507 model through
SGLang v0.5.21. They are distinct from the synthetic expert function in the
standalone transport replay.

The latest investigation is in
[NUMA regression investigation](#numa-regression-investigation).
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
