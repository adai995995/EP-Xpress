// Included inside host_sm.cu's private namespace. Bounded decode only.
// Two control warps each double-buffer peer descriptors. Optional metadata
// and ACK roles progress independently; eight payload warps are always retained.
constexpr int kDispatchRoles = EP_XPRESS_DISPATCH_SPLIT ? 2 : 1;
constexpr int kDispatchControlWarps = 2;
constexpr int kDispatchStagesPerControl = 2;
constexpr int kDispatchStages = kDispatchControlWarps * kDispatchStagesPerControl;
constexpr int kDispatchRxStart = 32 * (kDispatchControlWarps + (EP_XPRESS_DISPATCH_RETIRE ? 3 : 0));
constexpr int kDispatchPipelineThreads = kThreads + kDispatchRxStart;
constexpr uint32_t kDispatchFinished = EP_XPRESS_DISPATCH_RETIRE ? 8 : 3;
constexpr int kDispatchPipelineRows = 32;

__device__ __forceinline__ uint32_t dispatch_stage_load(const uint32_t* p) {
  uint32_t value;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
  return value;
}
__device__ __forceinline__ void dispatch_stage_store(uint32_t* p, uint32_t value) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "r"(address), "r"(value) : "memory");
}
__device__ __forceinline__ void dispatch_stage_arrive(uint32_t* p, uint32_t bit) {
  uint32_t old;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("atom.acq_rel.cta.shared.or.b32 %0, [%1], %2;"
               : "=r"(old) : "r"(address), "r"(bit) : "memory");
}
__device__ __forceinline__ void dispatch_rx_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

struct DispatchPrefetchStage {
  // 0: free; 1: descriptor ready for RX; 2: payload reads complete; 3: finished.
  // Control may still be reading metadata in state 1 or 2. Only its owner
  // reclaims the stage, after its metadata fence/join and RX's release of 2.
  uint32_t state;
  // Independent-role mode: 1=published, bit 2=RX done, bit 4=metadata
  // done, 7=ACK eligible, 8=finished. Only the ACK warp retires 7 -> 0.
  int peer, count, output;
#if EP_XPRESS_DISPATCH_RETIRE
  int owner;
#endif
  int wires[kDispatchPipelineRows];
};

// Pipeline diagnostics record three concurrent leader timelines, not a single
// sequential phase partition. No new CTA/warp barriers are added. TX subphases
// are thread-zero instruction intervals; publication includes writer stragglers.
enum DispatchProfileField {
  DBegin, DEnd, DElapsed, DTxRouteLocal, DTxCredit, DTxMetadata, DTxPayload, DTxPublish, DTxEnd,
  DControl0, DControl1 = DControl0 + 17, DRx = DControl1 + 17,
  DStatus = DRx + 9, DFieldCount
};
enum DispatchControlField {
  DCReady, DCDescriptor, DCMetadata, DCMetadataCommit, DCStageWait, DCAck,
  DCDrain, DCOther, DCBegin, DCEnd, DCReceiveDone, DCPolls, DCMissing,
  DCEmpty, DCRows, DCFirstReady, DCFirstStage
};
enum DispatchRxField {
  DRStageWait, DRPayload, DRCommit, DRDrainWait, DROther, DRBegin, DREnd,
  DRFrames, DRBytes
};
const std::vector<std::string> kDispatchProfileFields = [] {
  std::vector<std::string> fields = {"begin_ns", "end_ns", "elapsed_ns",
      "tx_route_local_ns", "tx_credit_wait_ns", "tx_metadata_ns", "tx_payload_ns", "tx_publish_ns", "tx_end_offset_ns"};
  for (int group = 0; group < 2; ++group) {
    const std::string prefix = "ctrl" + std::to_string(group) + "_";
    for (const auto* suffix : {"ready_poll_ns", "descriptor_alloc_ns", "metadata_ns",
         "metadata_commit_ns", "stage_wait_ns", "ack_publish_ns", "ack_drain_ns", "other_ns",
         "begin_offset_ns", "end_offset_ns", "receive_done_offset_ns", "poll_rounds",
         "missing_rounds", "empty_notifications", "payload_rows", "first_ready_offset_ns",
         "first_stage_offset_ns"}) fields.push_back(prefix + suffix);
  }
  for (const auto* suffix : {"stage_wait_ns", "payload_ns", "completion_ns", "ack_drain_wait_ns",
       "other_ns", "begin_offset_ns", "end_offset_ns", "frames", "payload_bytes"})
    fields.push_back(std::string("rx_") + suffix);
  fields.push_back("status");
  return fields;
}();
template <bool Enabled> struct DispatchProfileStorage {};
template <> struct DispatchProfileStorage<true> {
  int sample;
  uint32_t receive_done[2];
  int64_t values[DFieldCount];
};

__device__ __forceinline__ char* dispatch_pipeline_slot(TransportArgs a, int producer, int channel) {
  char* base = reinterpret_cast<char*>(a.pointers[producer]) + channel * a.channel_stride;
#if EP_XPRESS_DISPATCH_DEFER_RECLAIM
  base += (1 + (a.epoch & 1)) * (a.channel_stride / 3);
#endif
  return base + kArenaHeader;
}

#if EP_XPRESS_DISPATCH_WRITE_UNROLL > 0
template <bool Vector>
__device__ __forceinline__ void dispatch_pipeline_send(TransportArgs a, char* payload,
                                                     const int* rows, int first, int count) {
  constexpr int unroll = EP_XPRESS_DISPATCH_WRITE_UNROLL;
  const int units = a.hidden / (Vector ? 8 : 2);
  const int64_t total = int64_t(count) * units;
  using Value = typename std::conditional<Vector, uint4, uint32_t>::type;
  for (int64_t base = threadIdx.x; base < total; base += blockDim.x * unroll) {
    Value values[unroll];
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      const int64_t i = base + j * blockDim.x;
      if (i < total) {
        const int row = int(i / units), h = int(i % units);
        const auto* source = a.input + int64_t(first + rows[row] * (EP_XPRESS_DISPATCH_DIRECT ? a.channels : 1)) * a.hidden;
        values[j] = reinterpret_cast<const Value*>(source)[h];
      }
    }
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      const int64_t i = base + j * blockDim.x;
      if (i < total) {
        const int64_t target = EP_XPRESS_DISPATCH_DIRECT
            ? int64_t(rows[i / units]) * units + i % units : i;
        if constexpr (Vector) host_store128(payload + target * 16, values[j]);
        else host_store32(payload + target * 4, values[j]);
      }
    }
  }
}
#endif

#if EP_XPRESS_READY_BATCH
template <bool Vector>
__device__ __forceinline__ void dispatch_pipeline_receive_pair(
    TransportArgs a, const DispatchPrefetchStage& first,
    const DispatchPrefetchStage& second, int channel, int64_t x_offset, int rx) {
  // Both stages have already been acquired by the RX leader. Neither can be
  // reclaimed until this full worker group releases its completion below.
  const char* sources[2] = {
      dispatch_pipeline_slot(a, first.peer, channel) + x_offset,
      dispatch_pipeline_slot(a, second.peer, channel) + x_offset};
  const int width = a.hidden / (Vector ? 8 : 2);
  const int64_t totals[2] = {int64_t(first.count) * width, int64_t(second.count) * width};
  const int64_t total = totals[0] > totals[1] ? totals[0] : totals[1];
  using Value = typename std::conditional<Vector, uint4, uint32_t>::type;
  for (int64_t base = rx; base < total; base += kThreads * 2) {
    Value values[4];
    // Two independent loads per stage, interleaved across the two peers.
    // Issue the whole bounded batch before any loaded value is consumed.
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int64_t i = base + j * kThreads;
      if (i < totals[0]) {
        const int64_t source = int64_t(first.wires[i / width]) * width + i % width;
        if constexpr (Vector) values[j * 2] = host_load128(sources[0] + source * 16);
        else values[j * 2] = host_load32(sources[0] + source * 4);
      }
      if (i < totals[1]) {
        const int64_t source = int64_t(second.wires[i / width]) * width + i % width;
        if constexpr (Vector) values[j * 2 + 1] = host_load128(sources[1] + source * 16);
        else values[j * 2 + 1] = host_load32(sources[1] + source * 4);
      }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int64_t i = base + j * kThreads;
      if (i < totals[0])
        reinterpret_cast<Value*>(a.output + int64_t(first.output) * a.hidden)[i] = values[j * 2];
      if (i < totals[1])
        reinterpret_cast<Value*>(a.output + int64_t(second.output) * a.hidden)[i] = values[j * 2 + 1];
    }
  }
}
#endif

template <typename Id, bool Profile = false>
__global__ void host_sm_dispatch_pipeline_kernel(TransportArgs a) {
  if (a.device_epoch) a.epoch = *a.device_epoch;
  const int tid = threadIdx.x, channel = blockIdx.x % a.channels;
  const bool tx = !EP_XPRESS_DISPATCH_SPLIT || blockIdx.x < a.channels;
  const bool rx_role = !EP_XPRESS_DISPATCH_SPLIT || blockIdx.x >= a.channels;
  const int profile_channel = blockIdx.x;
  const int lane = tid % 32;
  // Fixed source slots let every receiver compute addresses from the current
  // bitmap, even when source ranks have different token counts. Only live
  // remote tokens are written; unused slot capacity is never transferred.
  const int first = EP_XPRESS_DISPATCH_DIRECT ? channel
      : int(int64_t(a.input_rows) * channel / a.channels);
  const int token_stride = EP_XPRESS_DISPATCH_DIRECT ? a.channels : 1;
  const int n = EP_XPRESS_DISPATCH_DIRECT
      ? max(0, (a.input_rows + a.channels - 1 - channel) / a.channels)
      : int(int64_t(a.input_rows) * (channel + 1) / a.channels) - first;
  const auto* ids = static_cast<const Id*>(a.ids);
  char* own = reinterpret_cast<char*>(a.pointers[a.rank]) + channel * a.channel_stride;
  char* slot = dispatch_pipeline_slot(a, a.rank, channel);
  const int64_t x_offset = payload_offset(a.chunk_rows, a.topk);
  const int64_t id_offset = ids_offset(a.chunk_rows);
  const int64_t weight_offset = weights_offset(a.chunk_rows, a.topk);
  const uint64_t sequence = (a.epoch << 32) | uint64_t(1);
  const uint64_t start = a.diagnostics ? ticks_ns() : 0;
  __shared__ int local[kDispatchPipelineRows], remote[kDispatchPipelineRows];
  __shared__ uint64_t masks[kDispatchPipelineRows];
  __shared__ int local_count, remote_count, local_base, local_owner;
  __shared__ uint32_t notifications[kMaxRanks];
#if EP_XPRESS_DECODE_CONTROL_RECORDS
  // Each TX warp owns one row-index scratch list, reused only after warp join.
  __shared__ int record_wires[kDispatchPipelineThreads / 32][kDispatchPipelineRows];
#endif
  __shared__ DispatchPrefetchStage stages[kDispatchStages];
  __shared__ int rx_choice;
#if EP_XPRESS_READY_BATCH
  __shared__ int rx_second;
#endif

  __shared__ DispatchProfileStorage<Profile> diag;
  uint64_t diag_mark = 0;
  int diag_phase = DTxRouteLocal;
  if constexpr (Profile) {
    if (!tid) {
      diag.sample = -1;
      if (a.decode_profile[1]) {
        auto* header = a.decode_profile + int64_t(profile_channel) * (DFieldCount + 4);
        const int64_t index = header[0]++;
        if (index < a.decode_profile_samples) diag.sample = int(index);
        else ++header[2];
      }
      if (diag.sample >= 0) {
        for (int i = 0; i < DFieldCount; ++i) diag.values[i] = 0;
        diag.receive_done[0] = diag.receive_done[1] = 0;
        diag.values[DControl0 + DCFirstReady] = diag.values[DControl1 + DCFirstReady] = -1;
        diag.values[DControl0 + DCFirstStage] = diag.values[DControl1 + DCFirstStage] = -1;
        diag.values[DBegin] = diag_mark = ticks_ns();
      }
    }
  }
  // Only leaders update their own disjoint fields. Markers never synchronize.
  // Timestamp transitions only; repeated polling in one phase adds no clock reads.
  const auto mark = [&](int field, bool leader) {
    if constexpr (Profile) {
      if (leader && diag.sample >= 0 && field != diag_phase) {
        const uint64_t now = ticks_ns();
        diag.values[diag_phase] += now - diag_mark;
        diag_mark = now; diag_phase = field;
      }
    }
  };

  if (tid < kDispatchStages) stages[tid].state = 0;
  __syncthreads();
  // Independent resident RX CTAs can consume peers while this channel sends.
  if (tx) {
#if EP_XPRESS_DISPATCH_DEFER_RECLAIM
    mark(DTxCredit, tid == 0);
    // Credits belong to the last publication of THIS physical channel/bank,
    // not epoch-2: routes, eager fallbacks and active widths may change.
    if (tid < 64) {
      const int peer = tid;
      const uint64_t previous = a.dispatch_sent_epochs[channel * 2 + (a.epoch & 1)];
      bool pending = previous && peer < a.world && peer != a.rank;
      while (__any_sync(0xffffffffu, pending) && !__any_sync(0xffffffffu, read_status(a.status) != 0)) {
        if (pending) pending = acquire(control(slot, 1 + peer)) != previous;
        if (a.diagnostics && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
      }
    }
    __syncthreads();
    mark(DTxRouteLocal, tid == 0);
#endif
    if (tid < 32) {
      uint64_t mask = 0;
      if (lane < n) {
        for (int k = 0; k < a.topk; ++k) {
          const int64_t expert = ids[int64_t(first + lane * token_stride) * a.topk + k];
          if (expert == -1) continue;
          if (expert < 0 || expert >= a.experts ||
              (a.diagnostics && !isfinite(a.weights[int64_t(first + lane * token_stride) * a.topk + k]))) {
            fail(a.status, 1); continue;
          }
          if (a.diagnostics)
            for (int j = 0; j < k; ++j)
              if (ids[int64_t(first + lane * token_stride) * a.topk + j] == expert) fail(a.status, 1);
          mask |= uint64_t(1) << a.expert_rank[expert];
        }
        masks[lane] = mask;
        a.expected_ranks[first + lane * token_stride] = int64_t(mask);
      }
      const uint32_t local_bits = __ballot_sync(0xffffffffu, mask & (uint64_t(1) << a.rank));
      const uint32_t remote_bits = __ballot_sync(0xffffffffu, mask & ~(uint64_t(1) << a.rank));
      const uint32_t before = (uint32_t(1) << lane) - 1;
      if (local_bits & (uint32_t(1) << lane)) local[__popc(local_bits & before)] = lane;
      if (remote_bits & (uint32_t(1) << lane)) remote[__popc(remote_bits & before)] = lane;
      if (!lane) {
        local_count = __popc(local_bits); remote_count = __popc(remote_bits);
        local_base = local_count ? atomicAdd(a.output_count, local_count) : 0;
        local_owner = local_count ? atomicAdd(a.owner_counts + a.rank, local_count) : 0;
        if (int64_t(local_base) + local_count > a.capacity ||
            int64_t(local_owner) + local_count > a.capacity) fail(a.status, 2);
      }
    }
    __syncthreads();
    if (!read_status(a.status)) {
      mark(DTxMetadata, tid == 0);
#if !EP_XPRESS_DECODE_CONTROL_RECORDS
      for (int i = tid; i < remote_count * a.topk; i += blockDim.x) {
        const int64_t source = int64_t(first + remote[i / a.topk] * token_stride) * a.topk + i % a.topk;
        const int64_t expert = ids[source];
        const int64_t target = EP_XPRESS_DISPATCH_DIRECT
            ? int64_t(remote[i / a.topk]) * a.topk + i % a.topk : i;
#if EP_XPRESS_ROUTE_PACKED
        // Prepared transport bounds the expert table by INT32_MAX. The route
        // validation above runs on the original ID type before narrowing, so
        // int64 input IDs and -1 padding retain their existing semantics.
        // Keep one source-owned word per route; all readers share this copy.
        const uint64_t route = uint64_t(uint32_t(expert)) |
            (uint64_t(__float_as_uint(expert >= 0 ? a.weights[source] : 0.0f)) << 32);
        host_store64(slot + id_offset + target * 8, route);
#else
        host_store64(slot + id_offset + target * 8, uint64_t(expert));
        host_store32(slot + weight_offset + target * 4,
                     __float_as_uint(expert >= 0 ? a.weights[source] : 0.0f));
#endif
      }
#endif
      for (int peer = tid / 32; peer < a.world; peer += kDispatchPipelineThreads / 32) {
        if (peer == a.rank) continue;
        const bool selected = EP_XPRESS_DISPATCH_DIRECT
            ? lane < n && (masks[lane] & (uint64_t(1) << peer))
            : lane < remote_count && (masks[remote[lane]] & (uint64_t(1) << peer));
        const uint32_t bits = __ballot_sync(0xffffffffu, selected);
#if !EP_XPRESS_DISPATCH_DIRECT
        if (selected) {
          const int offset = __popc(bits & ((uint32_t(1) << lane) - 1));
          const uint64_t descriptor = (uint64_t(first + remote[lane] * token_stride) << 32) | uint32_t(lane);
#if EP_XPRESS_DECODE_CONTROL_RECORDS
          char* record = dispatch_record(a, slot, peer, offset);
          host_store64(record, descriptor);
          record_wires[tid / 32][offset] = lane;
#else
          host_store64(dispatch_index(a, slot, peer) + (int64_t(offset) + 1) * 8, descriptor);
#endif
        }
#endif
#if EP_XPRESS_DECODE_CONTROL_RECORDS
        __syncwarp();
        // Spread contiguous route words over the whole warp. Assigning all
        // top-k stores to one selected token lane serializes small decode rows.
        // This scratch list is rebuilt from the current ballot for every peer.
        for (int i = lane; i < __popc(bits) * a.topk; i += 32) {
          const int row = i / a.topk, k = i % a.topk;
          const int wire = record_wires[tid / 32][row];
          const int64_t input = int64_t(first + remote[wire] * token_stride) * a.topk + k;
          const int64_t expert = ids[input];
          const int local_id = expert >= 0 && expert < a.experts && a.expert_rank[expert] == peer
              ? a.expert_local[expert] : -1;
          const uint32_t weight = __float_as_uint(expert >= 0 ? a.weights[input] : 0.0f);
          host_store64(dispatch_record(a, slot, peer, row) + (int64_t(k) + 1) * 8,
                       (uint64_t(weight) << 32) | uint32_t(local_id));
        }
        __syncwarp(); // Do not overwrite scratch while another lane consumes it.
#endif
        if (!lane) notifications[peer] = EP_XPRESS_DISPATCH_DIRECT ? bits : uint32_t(__popc(bits) + 1);
      }
      mark(DTxPayload, tid == 0);
#if EP_XPRESS_DISPATCH_WRITE_UNROLL > 0
      if (a.hidden % 8 == 0 && (reinterpret_cast<uintptr_t>(slot + x_offset) & 15) == 0 &&
          (reinterpret_cast<uintptr_t>(a.input) & 15) == 0)
        dispatch_pipeline_send<true>(a, slot + x_offset, remote, first, remote_count);
      else
        dispatch_pipeline_send<false>(a, slot + x_offset, remote, first, remote_count);
#else
      for (int64_t i = tid; i < int64_t(remote_count) * (a.hidden / 2); i += blockDim.x) {
        const int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
        const int64_t target = EP_XPRESS_DISPATCH_DIRECT
            ? int64_t(remote[row]) * (a.hidden / 2) + h : i;
        host_store32(slot + x_offset + target * 4,
            reinterpret_cast<const uint32_t*>(a.input + int64_t(first + remote[row] * token_stride) * a.hidden)[h]);
      }
#endif
      if (!tid && a.diagnostics) host_store64(slot + kIdentityOffset, a.identity);
    }
    mark(DTxPublish, tid == 0);
    // All writers, including descriptor/metadata writers, close publication.
    __threadfence_system();
    __syncthreads();
    if (!tid) {
      release(control(slot, 0), sequence);
#if EP_XPRESS_DISPATCH_DEFER_RECLAIM
      a.dispatch_sent_epochs[channel * 2 + (a.epoch & 1)] = sequence;
#endif
    }
    if (tid < a.world && tid != a.rank && !read_status(a.status))
      release(reinterpret_cast<uint64_t*>(dispatch_index(a, slot, tid)),
              (a.epoch << 32) | uint64_t(notifications[tid]));
    __syncthreads();
    // Remote publication depends only on the completed host writes. Materialize
    // the disjoint local output afterwards, while peer RX CTAs can make progress.
    // Its output range was reserved above; later kernels still wait for this
    // entire dispatch, so no consumer can observe an unfinished local row.
    mark(DTxRouteLocal, tid == 0);
    if (!read_status(a.status)) {
      for (int64_t i = tid; i < int64_t(local_count) * (a.hidden / 2); i += blockDim.x) {
        const int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
        reinterpret_cast<uint32_t*>(a.output + int64_t(local_base + row) * a.hidden)[h] =
            reinterpret_cast<const uint32_t*>(a.input + int64_t(first + local[row] * token_stride) * a.hidden)[h];
      }
      for (int i = tid; i < local_count * a.topk; i += blockDim.x) {
        const int row = i / a.topk, k = i % a.topk;
        const int64_t source = int64_t(first + local[row] * token_stride) * a.topk + k;
        const int64_t expert = ids[source];
        const int64_t out = int64_t(local_base + row) * a.topk + k;
        a.output_ids[out] = expert >= 0 && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
        a.output_weights[out] = expert >= 0 ? a.weights[source] : 0.0f;
      }
      if (tid < local_count) {
        a.output_inverse[int64_t(local_base + tid) * 2] = a.rank;
        a.output_inverse[int64_t(local_base + tid) * 2 + 1] = first + local[tid] * token_stride;
        a.local_rows[first + local[tid] * token_stride] = local_base + tid;
        a.owner_rows[int64_t(a.rank) * a.capacity + local_owner + tid] = local_base + tid;
        dispatch_return_row(a, a.rank, first + local[tid] * token_stride, local_base + tid);
      }
    }
    if constexpr (Profile) {
      if (!tid && diag.sample >= 0) {
        const uint64_t now = ticks_ns();
        diag.values[diag_phase] += now - diag_mark;
        diag.values[DTxEnd] = now - diag.values[DBegin];
      }
    }

  } // TX role. RX does not depend on local publication.
  if (rx_role) {
    if (tid < 64) {
      const int group = tid / 32;
      const int stage_base = group * kDispatchStagesPerControl;
      int next_stage = 0;
      const int dc = group == 0 ? DControl0 : DControl1;
      if constexpr (Profile) {
        if (!lane && diag.sample >= 0) {
          diag_mark = ticks_ns(); diag_phase = dc + DCOther;
          diag.values[dc + DCBegin] = diag_mark - diag.values[DBegin];
        }
      }
      // Each lane owns one incoming peer; no cross-group claims or host atomics.
      const int peer = group + lane * kDispatchControlWarps;
      bool pending = peer < a.world && peer != a.rank;
#if EP_XPRESS_READY_CACHE
      // The acquired generation stays immutable until this receiver ACKs it.
      // Keep observations for unselected peers only within this invocation;
      // neither a graph replay nor a later route reuses these registers.
      uint64_t cached_ready = 0;
      bool known_ready = false;
#endif
      uint32_t iteration = 0;
      while (true) {
        if (__any_sync(0xffffffffu, read_status(a.status) != 0)) break;
        int free_stage = -1;
        bool occupied = false;
        // Legacy control joins its own metadata work with RX completion.
        // Independent-role mode observes FREE only after the ACK warp joins
        // both completion bits and releases the host payload.
        for (int i = 0; i < kDispatchStagesPerControl; ++i) {
          const int index = stage_base + (next_stage + i) % kDispatchStagesPerControl;
          auto& candidate = stages[index];
          uint32_t state = __shfl_sync(0xffffffffu,
              lane == 0 ? dispatch_stage_load(&candidate.state) : 0, 0);
          if (!EP_XPRESS_DISPATCH_RETIRE && state == 2) {
            mark(dc + DCAck, lane == 0);
            if (!lane) {
              char* source = dispatch_pipeline_slot(a, candidate.peer, channel);
              release(control(source, 1 + a.rank), sequence);
              dispatch_stage_store(&candidate.state, 0);
            }
            __syncwarp();
            state = 0;
          }
          if (state == 0 && free_stage < 0) free_stage = index;
          occupied |= state != 0;
        }
        const bool incoming = __any_sync(0xffffffffu, pending);
        if (!incoming && !occupied) break;
        if (free_stage < 0 || !incoming) mark(dc + DCStageWait, lane == 0);
        if (free_stage >= 0 && incoming) {
          auto& stage = stages[free_stage];
          mark(dc + DCReady, lane == 0);
          uint64_t ready = 0;
#if EP_XPRESS_READY_CACHE
          // Do not make cached work wait for an unrelated host read. This
          // ballot gives the whole warp the same decision: drain an already
          // acquired task first, or poll the still-unknown peers together.
          uint32_t available = __ballot_sync(0xffffffffu, pending && known_ready);
          if (!available) {
            if (pending) {
              char* source = dispatch_pipeline_slot(a, peer, channel);
              cached_ready = acquire(reinterpret_cast<uint64_t*>(dispatch_index(a, source, a.rank)));
              known_ready = (cached_ready >> 32) == a.epoch &&
                  (EP_XPRESS_DISPATCH_DIRECT || uint32_t(cached_ready) != 0);
            }
            available = __ballot_sync(0xffffffffu, pending && known_ready);
          }
          ready = cached_ready;
#else
          if (pending) {
            char* source = dispatch_pipeline_slot(a, peer, channel);
            ready = acquire(reinterpret_cast<uint64_t*>(dispatch_index(a, source, a.rank)));
          }
          const uint32_t available = __ballot_sync(0xffffffffu,
              pending && (ready >> 32) == a.epoch && (EP_XPRESS_DISPATCH_DIRECT || uint32_t(ready) != 0));
#endif
          if constexpr (Profile) { if (!lane && diag.sample >= 0) {
            ++diag.values[dc + DCPolls];
            diag.values[dc + DCMissing] += available == 0;
            if (available && diag.values[dc + DCFirstReady] < 0)
              diag.values[dc + DCFirstReady] = ticks_ns() - diag.values[DBegin];
          }}
          if (available) {
            mark(dc + DCDescriptor, lane == 0);
            const int leader = __ffs(available) - 1;
            const int selected_peer = __shfl_sync(0xffffffffu, peer, leader);
            const uint32_t notification = __shfl_sync(0xffffffffu, uint32_t(ready), leader);
            const int count = EP_XPRESS_DISPATCH_DIRECT ? __popc(notification) : int(notification) - 1;
            if (lane == leader) pending = false;
            char* source = dispatch_pipeline_slot(a, selected_peer, channel);
            if (count < 0 || count > decode_index_rows(a)) { fail(a.status, 1); break; }
            if (a.diagnostics && !lane && host_load64(source + kIdentityOffset) != a.identity)
              fail(a.status, 5);
            if (!count) {
              mark(dc + DCAck, lane == 0);
              if constexpr (Profile) { if (!lane && diag.sample >= 0) ++diag.values[dc + DCEmpty]; }
              __syncwarp();
              // A count-only reader never touches payload or descriptors.
              if (!lane) release(control(source, 1 + a.rank), sequence);
              mark(dc + DCReady, lane == 0);
            } else {
              int token = 0;
#if EP_XPRESS_DISPATCH_DIRECT
              if (notification & (uint32_t(1) << lane))
                stage.wires[__popc(notification & ((uint32_t(1) << lane) - 1))] = lane;
              __syncwarp();
              if (lane < count) token = channel + stage.wires[lane] * a.channels;
#else
              // Bridge the selected lane's READY acquire to all descriptor
              // readers. The DIRECT path already joins while building wires.
              __syncwarp();
              if (lane < count) {
#if EP_XPRESS_DECODE_CONTROL_RECORDS
                const uint64_t descriptor = host_load64(dispatch_record(a, source, a.rank, lane));
#else
                const uint64_t descriptor = host_load64(dispatch_index(a, source, a.rank) + (int64_t(lane) + 1) * 8);
#endif
                stage.wires[lane] = int(uint32_t(descriptor));
                token = int(descriptor >> 32);
                if (stage.wires[lane] < 0 || stage.wires[lane] >= decode_index_rows(a) ||
                    token < 0 || token >= a.mailbox_tokens) fail(a.status, 2);
              }
#endif
              int output = 0, owner = 0;
              if (!lane) {
                output = atomicAdd(a.output_count, count);
                owner = atomicAdd(a.owner_counts + selected_peer, count);
                if (int64_t(output) + count > a.capacity || int64_t(owner) + count > a.capacity)
                  fail(a.status, 2);
              }
              output = __shfl_sync(0xffffffffu, output, 0);
              owner = __shfl_sync(0xffffffffu, owner, 0);
              __syncwarp();
              if (__any_sync(0xffffffffu, read_status(a.status) != 0)) break;
              // Descriptors and destination are sufficient for payload reads.
              // Publish before route metadata loads; RX never accesses IDs or
              // weights, and this stage remains immutable until both sides join.
              if (!lane) {
                stage.peer = selected_peer; stage.count = count; stage.output = output;
#if EP_XPRESS_DISPATCH_RETIRE
                stage.owner = owner;
#endif
                dispatch_stage_store(&stage.state, 1);
                if constexpr (Profile) { if (diag.sample >= 0 && diag.values[dc + DCFirstStage] < 0)
                  diag.values[dc + DCFirstStage] = ticks_ns() - diag.values[DBegin]; }
              }
              __syncwarp();
              next_stage = (free_stage - stage_base + 1) % kDispatchStagesPerControl;
#if !EP_XPRESS_DISPATCH_RETIRE
              mark(dc + DCMetadata, lane == 0);
              if constexpr (Profile) { if (!lane && diag.sample >= 0) diag.values[dc + DCRows] += count; }
              // Warp lanes issue independent route loads. The other control warp
              // can prefetch another peer while the 256-thread RX group copies.
              for (int i = lane; i < count * a.topk; i += 32) {
                const int64_t out = int64_t(output) * a.topk + i;
#if EP_XPRESS_DECODE_CONTROL_RECORDS
                const uint64_t route = host_load64(dispatch_record(a, source, a.rank, i / a.topk) +
                                                  (int64_t(i % a.topk) + 1) * 8);
                a.output_ids[out] = int32_t(uint32_t(route));
                a.output_weights[out] = __uint_as_float(uint32_t(route >> 32));
#else
                const int64_t wire = int64_t(stage.wires[i / a.topk]) * a.topk + i % a.topk;
#if EP_XPRESS_ROUTE_PACKED
                const uint64_t route = host_load64(source + id_offset + wire * 8);
                const int64_t expert = int32_t(uint32_t(route));
                const float weight = __uint_as_float(uint32_t(route >> 32));
#else
                const int64_t expert = int64_t(host_load64(source + id_offset + wire * 8));
                const float weight = __uint_as_float(host_load32(source + weight_offset + wire * 4));
#endif
                a.output_ids[out] = expert >= 0 && expert < a.experts && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
                a.output_weights[out] = weight;
#endif
              }
              if (lane < count) {
                a.output_inverse[int64_t(output + lane) * 2] = selected_peer;
                a.output_inverse[int64_t(output + lane) * 2 + 1] = token;
                a.owner_rows[int64_t(selected_peer) * a.capacity + owner + lane] = output + lane;
                dispatch_return_row(a, selected_peer, token, output + lane);
              }
              mark(dc + DCMetadataCommit, lane == 0);
              // Every metadata lane completes its host reads and GPU writes
              // before the owner may reap RX completion and publish a host ACK.
              // RX can finish first; state 2 alone never causes an early ACK.
              host_rx_fence();
              __syncwarp();
              mark(dc + DCStageWait, lane == 0);
#endif
            }
          }
        }
        if (a.diagnostics && !(iteration++ & 255u)) {
          if (ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
          if (pending && acquire(reinterpret_cast<uint64_t*>(a.pointers[peer] + channel * a.channel_stride)))
            fail(a.status, 4);
        }
        __syncwarp();
      }
      mark(dc + DCDrain, lane == 0);
      if constexpr (Profile) { if (!lane && diag.sample >= 0) {
        diag.values[dc + DCReceiveDone] = diag_mark - diag.values[DBegin];
        dispatch_stage_store(&diag.receive_done[group], 1);
      }}
#if !EP_XPRESS_DISPATCH_DEFER_RECLAIM
      // Drain only this group's outgoing ACK ownership before the next operation.
      bool unacked = peer < a.world && peer != a.rank;
      while (__any_sync(0xffffffffu, unacked) && !__any_sync(0xffffffffu, read_status(a.status) != 0)) {
        if (unacked) unacked = acquire(control(slot, 1 + peer)) != sequence;
        if (a.diagnostics && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
      }
#endif
      __syncwarp();
      if (!lane)
        for (int i = 0; i < kDispatchStagesPerControl; ++i)
          dispatch_stage_store(&stages[stage_base + i].state, kDispatchFinished);
      if constexpr (Profile) { if (!lane && diag.sample >= 0) {
        const uint64_t now = ticks_ns();
        diag.values[diag_phase] += now - diag_mark;
        diag.values[dc + DCEnd] = now - diag.values[DBegin];
      }}
    }
#if EP_XPRESS_DISPATCH_RETIRE
    else if (tid < 128) {
      // Each metadata warp consumes its control group's immutable stages.
      // Descriptor preparation and ACK service continue while host reads stall.
      const int base = (tid / 32 - 2) * kDispatchStagesPerControl;
      int next = 0;
      while (true) {
        int choice = -1;
        bool done = true;
        if (!lane) {
          for (int i = 0; i < kDispatchStagesPerControl; ++i) {
            const int index = base + (next + i) % kDispatchStagesPerControl;
            const uint32_t state = dispatch_stage_load(&stages[index].state);
            done &= state == kDispatchFinished;
            if ((state & 1) && !(state & 4) && choice < 0) choice = index;
          }
          if (done || read_status(a.status)) choice = -2;
        }
        choice = __shfl_sync(0xffffffffu, choice, 0);
        if (choice == -2) break;
        if (choice < 0) { __nanosleep(64); continue; }
        __syncwarp(); // Bridge the stage leader's acquire to all metadata lanes.
        auto& stage = stages[choice];
        char* source = dispatch_pipeline_slot(a, stage.peer, channel);
        for (int i = lane; i < stage.count * a.topk; i += 32) {
          const int64_t wire = int64_t(stage.wires[i / a.topk]) * a.topk + i % a.topk;
#if EP_XPRESS_ROUTE_PACKED
          const uint64_t route = host_load64(source + id_offset + wire * 8);
          const int64_t expert = int32_t(uint32_t(route));
          const float weight = __uint_as_float(uint32_t(route >> 32));
#else
          const int64_t expert = int64_t(host_load64(source + id_offset + wire * 8));
          const float weight = __uint_as_float(host_load32(source + weight_offset + wire * 4));
#endif
          const int64_t out = int64_t(stage.output) * a.topk + i;
          a.output_ids[out] = expert >= 0 && expert < a.experts && a.expert_rank[expert] == a.rank
              ? a.expert_local[expert] : -1;
          a.output_weights[out] = weight;
        }
        if (lane < stage.count) {
          const int token = channel + stage.wires[lane] * a.channels;
          a.output_inverse[int64_t(stage.output + lane) * 2] = stage.peer;
          a.output_inverse[int64_t(stage.output + lane) * 2 + 1] = token;
          a.owner_rows[int64_t(stage.peer) * a.capacity + stage.owner + lane] = stage.output + lane;
          dispatch_return_row(a, stage.peer, token, stage.output + lane);
        }
        host_rx_fence();
        __syncwarp();
        // No lane touches this generation after arrival: it may be retired.
        if (!lane) dispatch_stage_arrive(&stage.state, 4);
        __syncwarp();
        next = (choice - base + 1) % kDispatchStagesPerControl;
      }
    } else if (tid < kDispatchRxStart) {
      // One owner per ACK, with no remote read-modify-write or cross-rank join.
      while (true) {
        uint32_t state = lane < kDispatchStages ? dispatch_stage_load(&stages[lane].state) : kDispatchFinished;
        if (__all_sync(0xffffffffu, state == kDispatchFinished) || __any_sync(0xffffffffu, read_status(a.status) != 0)) break;
        if (lane < kDispatchStages && state == 7) {
          auto& stage = stages[lane];
          char* source = dispatch_pipeline_slot(a, stage.peer, channel);
          release(control(source, 1 + a.rank), sequence);
          dispatch_stage_store(&stage.state, 0);
        }
        __syncwarp();
      }
    }
#endif
    else {
      const int rx = tid - kDispatchRxStart;
      int next = 0;
      if constexpr (Profile) { if (!rx && diag.sample >= 0) {
        diag_mark = ticks_ns(); diag_phase = DRx + DRStageWait;
        diag.values[DRx + DRBegin] = diag_mark - diag.values[DBegin];
      }}
      while (true) {
#if EP_XPRESS_RX_POLL_WARP
        // Park the other payload warps at the handoff below. The complete
        // first warp polls together, so no lanes wait at a group barrier while
        // other lanes of that same warp must make progress in this loop.
        if (rx < 32) {
          int observed;
          do {
#endif
        if (!rx) {
          rx_choice = -1;
#if EP_XPRESS_READY_BATCH
          rx_second = -1;
#endif
          bool done = true;
          for (int i = 0; i < kDispatchStages; ++i) {
            const int choice = (next + i) % kDispatchStages;
            const uint32_t state = dispatch_stage_load(&stages[choice].state);
            if (state != kDispatchFinished) done = false;
            if (EP_XPRESS_DISPATCH_RETIRE ? (state & 1) && !(state & 2) : state == 1) {
              if (rx_choice < 0) rx_choice = choice;
#if EP_XPRESS_READY_BATCH
              else if (rx_second < 0) rx_second = choice;
#endif
            }
          }
          if (done || read_status(a.status)) rx_choice = -2;
        }
#if EP_XPRESS_RX_POLL_WARP
            observed = __shfl_sync(0xffffffffu, !rx ? rx_choice : -1, 0);
            if (observed == -1) {
              if constexpr (Profile) { if (!rx && diag.sample >= 0) {
                const bool draining = dispatch_stage_load(&diag.receive_done[0]) && dispatch_stage_load(&diag.receive_done[1]);
                mark(DRx + (draining ? DRDrainWait : DRStageWait), true);
              }}
              __nanosleep(64);
            }
          } while (observed == -1);
        }
#endif
        dispatch_rx_sync();
        const int choice = rx_choice;
#if EP_XPRESS_READY_BATCH
        const int second = rx_second;
#endif
        if (choice == -2) break;
#if !EP_XPRESS_RX_POLL_WARP
        if (choice < 0) {
          if constexpr (Profile) { if (!rx && diag.sample >= 0) {
            const bool draining = dispatch_stage_load(&diag.receive_done[0]) && dispatch_stage_load(&diag.receive_done[1]);
            mark(DRx + (draining ? DRDrainWait : DRStageWait), true);
          }}
          dispatch_rx_sync(); __nanosleep(64); continue;
        }
#endif
        mark(DRx + DRPayload, rx == 0);
        const auto& stage = stages[choice];
        char* source = dispatch_pipeline_slot(a, stage.peer, channel);
#if EP_XPRESS_READY_BATCH
        if (second >= 0) {
          const char* other = dispatch_pipeline_slot(a, stages[second].peer, channel);
          if (a.hidden % 8 == 0 &&
              ((reinterpret_cast<uintptr_t>(source + x_offset) |
                reinterpret_cast<uintptr_t>(other + x_offset) |
                reinterpret_cast<uintptr_t>(a.output)) & 15) == 0)
            dispatch_pipeline_receive_pair<true>(a, stage, stages[second], channel, x_offset, rx);
          else dispatch_pipeline_receive_pair<false>(a, stage, stages[second], channel, x_offset, rx);
        } else {
#endif
#if EP_XPRESS_DISPATCH_READ_UNROLL > 0
        if (a.hidden % 8 == 0 &&
            ((reinterpret_cast<uintptr_t>(source + x_offset) | reinterpret_cast<uintptr_t>(a.output)) & 15) == 0) {
          constexpr int unroll = EP_XPRESS_DISPATCH_READ_UNROLL;
          const int width = a.hidden / 8;
          const int64_t vectors = int64_t(stage.count) * width;
          for (int64_t base = rx; base < vectors; base += kThreads * unroll) {
            uint4 values[unroll];
#pragma unroll
            for (int j = 0; j < unroll; ++j) {
              const int64_t i = base + j * kThreads;
              if (i < vectors) values[j] = host_load128(source + x_offset +
                  (int64_t(stage.wires[i / width]) * width + i % width) * 16);
            }
#pragma unroll
            for (int j = 0; j < unroll; ++j) {
              const int64_t i = base + j * kThreads;
              if (i < vectors) reinterpret_cast<uint4*>(a.output + int64_t(stage.output) * a.hidden)[i] = values[j];
            }
          }
        } else
#endif
        for (int64_t i = rx; i < int64_t(stage.count) * (a.hidden / 2); i += kThreads) {
          const int width = a.hidden / 2;
          reinterpret_cast<uint32_t*>(a.output + int64_t(stage.output) * a.hidden)[i] =
              host_load32(source + x_offset + (int64_t(stage.wires[i / width]) * width + i % width) * 4);
        }
#if EP_XPRESS_READY_BATCH
        }
#endif
        mark(DRx + DRCommit, rx == 0);
        if constexpr (Profile) { if (!rx && diag.sample >= 0) {
          ++diag.values[DRx + DRFrames];
          diag.values[DRx + DRBytes] += int64_t(stage.count) * a.hidden * 2;
#if EP_XPRESS_READY_BATCH
          if (second >= 0) {
            ++diag.values[DRx + DRFrames];
            diag.values[DRx + DRBytes] += int64_t(stages[second].count) * a.hidden * 2;
          }
#endif
        }}
        host_rx_fence();
        dispatch_rx_sync();
        // All readers have completed before their leader releases the stage.
        // Peers may reclaim it immediately after the metadata join and ACK.
        if (!rx) {
          if constexpr (EP_XPRESS_DISPATCH_RETIRE) dispatch_stage_arrive(&stages[choice].state, 2);
          else dispatch_stage_store(&stages[choice].state, 2);
#if EP_XPRESS_READY_BATCH
          if (second >= 0) {
            if constexpr (EP_XPRESS_DISPATCH_RETIRE) dispatch_stage_arrive(&stages[second].state, 2);
            else dispatch_stage_store(&stages[second].state, 2);
          }
#endif
        }
        dispatch_rx_sync();
#if EP_XPRESS_READY_BATCH
        next = ((second >= 0 ? second : choice) + 1) % kDispatchStages;
#else
        next = (choice + 1) % kDispatchStages;
#endif
        mark(DRx + DRStageWait, rx == 0);
      }
      if constexpr (Profile) { if (!rx && diag.sample >= 0) {
        const uint64_t now = ticks_ns();
        diag.values[diag_phase] += now - diag_mark;
        diag.values[DRx + DREnd] = now - diag.values[DBegin];
      }}
    }
  } // RX role.
  __syncthreads();
  if constexpr (Profile) { if (!tid && diag.sample >= 0) {
    const uint64_t now = ticks_ns();
    diag.values[DEnd] = now;
    diag.values[DElapsed] = now - diag.values[DBegin];
    diag.values[DStatus] = read_status(a.status);
    auto* out = a.decode_profile + (int64_t(diag.sample + 1) * a.decode_profile_channels + profile_channel) * (DFieldCount + 4);
    for (int i = 0; i < DFieldCount; ++i) out[i] = diag.values[i];
    out[DFieldCount] = a.epoch; out[DFieldCount + 1] = a.input_rows;
    out[DFieldCount + 2] = a.capacity; out[DFieldCount + 3] = a.channels;
  }}
  if (!tid && read_status(a.status)) {
    release(reinterpret_cast<uint64_t*>(own), uint64_t(read_status(a.status)));
    if (a.fatal_errors || acquire(reinterpret_cast<uint64_t*>(own) + 2)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}
