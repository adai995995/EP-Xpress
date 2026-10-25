// Included inside host_sm.cu's private namespace. Bounded decode only.
// Two control warps prefetch independent peers and publish completed ACKs;
// the remaining eight warps retain the full legacy payload-read parallelism.
constexpr int kDispatchControlWarps = 2;
constexpr int kDispatchPipelineThreads = kThreads + 32 * kDispatchControlWarps;
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
__device__ __forceinline__ void dispatch_rx_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

struct DispatchPrefetchStage {
  // 0: control-owned; 1: RX-owned; 2: reads complete; 3: control finished.
  uint32_t state;
  int peer, count, output;
  int wires[kDispatchPipelineRows];
};

template <typename Id>
__global__ void host_sm_dispatch_pipeline_kernel(TransportArgs a) {
  if (a.device_epoch) a.epoch = *a.device_epoch;
  const int tid = threadIdx.x, channel = blockIdx.x;
  const int lane = tid % 32;
  const int first = int(int64_t(a.input_rows) * channel / a.channels);
  const int last = int(int64_t(a.input_rows) * (channel + 1) / a.channels);
  const int n = last - first;
  const auto* ids = static_cast<const Id*>(a.ids);
  char* own = reinterpret_cast<char*>(a.pointers[a.rank]) + channel * a.channel_stride;
  char* slot = own + kArenaHeader;
  const int64_t x_offset = payload_offset(a.chunk_rows, a.topk);
  const int64_t id_offset = ids_offset(a.chunk_rows);
  const int64_t weight_offset = weights_offset(a.chunk_rows, a.topk);
  const uint64_t sequence = (a.epoch << 32) | uint64_t(1);
  const uint64_t start = a.diagnostics ? ticks_ns() : 0;
  __shared__ int local[kDispatchPipelineRows], remote[kDispatchPipelineRows];
  __shared__ uint64_t masks[kDispatchPipelineRows];
  __shared__ int local_count, remote_count, local_base, local_owner;
  __shared__ int counts[kMaxRanks];
  __shared__ DispatchPrefetchStage stages[kDispatchControlWarps];
  __shared__ int rx_choice;

  if (tid < 32) {
    uint64_t mask = 0;
    if (lane < n) {
      for (int k = 0; k < a.topk; ++k) {
        const int64_t expert = ids[int64_t(first + lane) * a.topk + k];
        if (expert == -1) continue;
        if (expert < 0 || expert >= a.experts ||
            (a.diagnostics && !isfinite(a.weights[int64_t(first + lane) * a.topk + k]))) {
          fail(a.status, 1); continue;
        }
        if (a.diagnostics)
          for (int j = 0; j < k; ++j)
            if (ids[int64_t(first + lane) * a.topk + j] == expert) fail(a.status, 1);
        mask |= uint64_t(1) << a.expert_rank[expert];
      }
      masks[lane] = mask;
      a.expected_ranks[first + lane] = int64_t(mask);
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
  if (tid < kDispatchControlWarps) stages[tid].state = 0;
  __syncthreads();
  if (!read_status(a.status)) {
    for (int64_t i = tid; i < int64_t(local_count) * (a.hidden / 2); i += blockDim.x) {
      const int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
      reinterpret_cast<uint32_t*>(a.output + int64_t(local_base + row) * a.hidden)[h] =
          reinterpret_cast<const uint32_t*>(a.input + int64_t(first + local[row]) * a.hidden)[h];
    }
    for (int i = tid; i < local_count * a.topk; i += blockDim.x) {
      const int row = i / a.topk, k = i % a.topk;
      const int64_t source = int64_t(first + local[row]) * a.topk + k;
      const int64_t expert = ids[source];
      const int64_t out = int64_t(local_base + row) * a.topk + k;
      a.output_ids[out] = expert >= 0 && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
      a.output_weights[out] = expert >= 0 ? a.weights[source] : 0.0f;
    }
    if (tid < local_count) {
      a.output_inverse[int64_t(local_base + tid) * 2] = a.rank;
      a.output_inverse[int64_t(local_base + tid) * 2 + 1] = first + local[tid];
      a.local_rows[first + local[tid]] = local_base + tid;
      a.owner_rows[int64_t(a.rank) * a.capacity + local_owner + tid] = local_base + tid;
    }
    for (int i = tid; i < remote_count * a.topk; i += blockDim.x) {
      const int64_t source = int64_t(first + remote[i / a.topk]) * a.topk + i % a.topk;
      const int64_t expert = ids[source];
      host_store64(slot + id_offset + int64_t(i) * 8, uint64_t(expert));
      host_store32(slot + weight_offset + int64_t(i) * 4,
                   __float_as_uint(expert >= 0 ? a.weights[source] : 0.0f));
    }
    for (int peer = tid / 32; peer < a.world; peer += kDispatchPipelineThreads / 32) {
      if (peer == a.rank) continue;
      const bool selected = lane < remote_count && (masks[remote[lane]] & (uint64_t(1) << peer));
      const uint32_t bits = __ballot_sync(0xffffffffu, selected);
      if (selected) {
        const int offset = __popc(bits & ((uint32_t(1) << lane) - 1));
        host_store64(dispatch_index(a, slot, peer) + (int64_t(offset) + 1) * 8,
                     (uint64_t(first + remote[lane]) << 32) | uint32_t(lane));
      }
      if (!lane) counts[peer] = __popc(bits);
    }
    for (int64_t i = tid; i < int64_t(remote_count) * (a.hidden / 2); i += blockDim.x) {
      const int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
      host_store32(slot + x_offset + i * 4,
          reinterpret_cast<const uint32_t*>(a.input + int64_t(first + remote[row]) * a.hidden)[h]);
    }
    if (!tid && a.diagnostics) host_store64(slot + kIdentityOffset, a.identity);
  }
  // All writers, including descriptor/metadata writers, close publication.
  __threadfence_system();
  __syncthreads();
  if (!tid) release(control(slot, 0), sequence); // Protect subsequent ring reuse.
  if (tid < a.world && tid != a.rank && !read_status(a.status))
    release(reinterpret_cast<uint64_t*>(dispatch_index(a, slot, tid)),
            (a.epoch << 32) | uint64_t(counts[tid] + 1));
  __syncthreads();

  if (tid < 64) {
    const int group = tid / 32;
    auto& stage = stages[group];
    // Each lane owns one incoming peer; no cross-group claims or host atomics.
    const int peer = group + lane * kDispatchControlWarps;
    bool pending = peer < a.world && peer != a.rank;
    uint32_t iteration = 0;
    while (__any_sync(0xffffffffu, pending)) {
      if (__any_sync(0xffffffffu, read_status(a.status) != 0)) break;
      uint32_t state = __shfl_sync(0xffffffffu, lane == 0 ? dispatch_stage_load(&stage.state) : 0, 0);
      if (state == 2) {
        if (!lane) {
          char* source = reinterpret_cast<char*>(a.pointers[stage.peer]) +
                         channel * a.channel_stride + kArenaHeader;
          // RX writers fence and join before releasing state=2. This acquire
          // transfers completion to the ACK owner, independently of the other stage.
          release(control(source, 1 + a.rank), sequence);
          dispatch_stage_store(&stage.state, 0);
        }
        __syncwarp();
        state = 0;
      }
      if (state == 0) {
        uint64_t ready = 0;
        if (pending) {
          char* source = reinterpret_cast<char*>(a.pointers[peer]) +
                         channel * a.channel_stride + kArenaHeader;
          ready = acquire(reinterpret_cast<uint64_t*>(dispatch_index(a, source, a.rank)));
        }
        const uint32_t available = __ballot_sync(0xffffffffu,
            pending && (ready >> 32) == a.epoch && uint32_t(ready) != 0);
        if (available) {
          const int leader = __ffs(available) - 1;
          const int selected_peer = __shfl_sync(0xffffffffu, peer, leader);
          const int count = int(__shfl_sync(0xffffffffu, uint32_t(ready), leader)) - 1;
          if (lane == leader) pending = false;
          char* source = reinterpret_cast<char*>(a.pointers[selected_peer]) +
                         channel * a.channel_stride + kArenaHeader;
          if (count < 0 || count > decode_index_rows(a)) { fail(a.status, 1); break; }
          if (a.diagnostics && !lane && host_load64(source + kIdentityOffset) != a.identity)
            fail(a.status, 5);
          if (!count) {
            __syncwarp();
            // A count-only reader never touches payload or descriptors.
            if (!lane) release(control(source, 1 + a.rank), sequence);
          } else {
            int token = 0;
            if (lane < count) {
              const uint64_t descriptor = host_load64(dispatch_index(a, source, a.rank) + (int64_t(lane) + 1) * 8);
              stage.wires[lane] = int(uint32_t(descriptor));
              token = int(descriptor >> 32);
              if (stage.wires[lane] < 0 || stage.wires[lane] >= decode_index_rows(a) ||
                  token < 0 || token >= a.mailbox_tokens) fail(a.status, 2);
            }
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
            // Warp lanes issue independent route loads. The other control warp
            // can prefetch another peer while the 256-thread RX group copies.
            for (int i = lane; i < count * a.topk; i += 32) {
              const int64_t wire = int64_t(stage.wires[i / a.topk]) * a.topk + i % a.topk;
              const int64_t expert = int64_t(host_load64(source + id_offset + wire * 8));
              const float weight = __uint_as_float(host_load32(source + weight_offset + wire * 4));
              const int64_t out = int64_t(output) * a.topk + i;
              a.output_ids[out] = expert >= 0 && expert < a.experts && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
              a.output_weights[out] = weight;
            }
            if (lane < count) {
              a.output_inverse[int64_t(output + lane) * 2] = selected_peer;
              a.output_inverse[int64_t(output + lane) * 2 + 1] = token;
              a.owner_rows[int64_t(selected_peer) * a.capacity + owner + lane] = output + lane;
            }
            // Every metadata reader closes its host accesses before handing the
            // stage to RX, whose later completion permits host-buffer reuse.
            __threadfence_system();
            __syncwarp();
            if (!lane) {
              stage.peer = selected_peer; stage.count = count; stage.output = output;
              dispatch_stage_store(&stage.state, 1);
            }
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
    // The final staged peer may still be in flight after pending becomes empty.
    while (!__any_sync(0xffffffffu, read_status(a.status) != 0)) {
      const uint32_t state = __shfl_sync(0xffffffffu, lane == 0 ? dispatch_stage_load(&stage.state) : 0, 0);
      if (state == 0) break;
      if (state == 2) {
        if (!lane) {
          char* source = reinterpret_cast<char*>(a.pointers[stage.peer]) + channel * a.channel_stride + kArenaHeader;
          release(control(source, 1 + a.rank), sequence);
          dispatch_stage_store(&stage.state, 0);
        }
        __syncwarp();
        break;
      }
      if (a.diagnostics && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
    }
    // Drain only this group's outgoing ACK ownership before the next operation.
    bool unacked = peer < a.world && peer != a.rank;
    while (__any_sync(0xffffffffu, unacked) && !__any_sync(0xffffffffu, read_status(a.status) != 0)) {
      if (unacked) unacked = acquire(control(slot, 1 + peer)) != sequence;
      if (a.diagnostics && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
    }
    __syncwarp();
    if (!lane) dispatch_stage_store(&stage.state, 3);
  } else {
    const int rx = tid - 64;
    int next = 0;
    while (true) {
      if (!rx) {
        rx_choice = -1;
        bool done = true;
        for (int i = 0; i < kDispatchControlWarps; ++i) {
          const int choice = (next + i) % kDispatchControlWarps;
          const uint32_t state = dispatch_stage_load(&stages[choice].state);
          if (state != 3) done = false;
          if (state == 1 && rx_choice < 0) rx_choice = choice;
        }
        if (done || read_status(a.status)) rx_choice = -2;
      }
      dispatch_rx_sync();
      const int choice = rx_choice;
      if (choice == -2) break;
      if (choice < 0) { dispatch_rx_sync(); __nanosleep(64); continue; }
      const auto& stage = stages[choice];
      char* source = reinterpret_cast<char*>(a.pointers[stage.peer]) + channel * a.channel_stride + kArenaHeader;
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
      __threadfence_system();
      dispatch_rx_sync();
      if (!rx) dispatch_stage_store(&stages[choice].state, 2);
      // All RX lanes stop referencing this stage before control can refill it.
      dispatch_rx_sync();
      next = (choice + 1) % kDispatchControlWarps;
    }
  }
  __syncthreads();
  if (!tid && read_status(a.status)) {
    release(reinterpret_cast<uint64_t*>(own), uint64_t(read_status(a.status)));
    if (a.fatal_errors || acquire(reinterpret_cast<uint64_t*>(own) + 2)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}
