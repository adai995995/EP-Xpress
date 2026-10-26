// Optional bounded decode packets. Routing remains per token; publication and
// credit belong to (producer, destination, token tile, bank). Only live rows
// cross PCIe. RX stages independently ready peers into persistent HBM, allowing
// deterministic rank-order reduction without holding host slots until reduce.
#if EP_XPRESS_DECODE_PACKET_ROWS > 0
constexpr int kPacketRows = EP_XPRESS_DECODE_PACKET_ROWS;
constexpr int kPacketBanks = 2;

__host__ __device__ inline int packet_tiles(int tokens) {
  return (tokens + kPacketRows - 1) / kPacketRows;
}
__host__ __device__ inline int64_t packet_stride(int hidden) {
  return (128 + int64_t(kPacketRows) * hidden * 2 + 127) / 128 * 128;
}
__device__ __forceinline__ char* packet_box(TransportArgs a, int producer, int task) {
  return reinterpret_cast<char*>(a.mailbox_pointers[producer]) + kArenaHeader +
      (int64_t(task) * kPacketBanks + a.epoch % kPacketBanks) * packet_stride(a.hidden);
}

// Issue independent host loads before consuming their values in HBM stores.
// A load-then-store loop would serialize the PCIe round trip for every row.
template <bool Vector>
__device__ __forceinline__ void packet_receive(TransportArgs a, char* payload,
                                              int peer, int tile, unsigned mask) {
  constexpr int unroll = 4;
  const int units = a.hidden / (Vector ? 8 : 2);
  const int64_t total = int64_t(kPacketRows) * units;
  using Value = typename std::conditional<Vector, uint4, uint32_t>::type;
  for (int64_t base = threadIdx.x; base < total; base += blockDim.x * unroll) {
    Value values[unroll];
    bool live[unroll];
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      const int64_t i = base + j * blockDim.x;
      const int offset = int(i / units), h = int(i % units);
      live[j] = i < total && (mask & (1u << offset));
      if (live[j]) {
        const char* source = payload + int64_t(offset) * a.hidden * 2;
        if constexpr (Vector) values[j] = host_load128(source + int64_t(h) * 16);
        else values[j] = host_load32(source + int64_t(h) * 4);
      }
    }
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      if (live[j]) {
        const int64_t i = base + j * blockDim.x;
        const int token = tile * kPacketRows + int(i / units), h = int(i % units);
        auto* target = a.packet_scratch + (int64_t(peer) * a.mailbox_tokens + token) * a.hidden;
        if constexpr (Vector) reinterpret_cast<uint4*>(target)[h] = values[j];
        else reinterpret_cast<uint32_t*>(target)[h] = values[j];
      }
    }
  }
}

// Exactly one CTA owns each destination's row index. Rebuild from the current
// owner lists on every invocation, including empty and skipped routes. There
// is no CPU inspection, route cache, or assumption that graph shapes are live.
__global__ void host_sm_packet_prepare_kernel(TransportArgs a) {
  const int peer = blockIdx.x, tid = threadIdx.x;
  __shared__ int send_count, receive_count;
  if (!tid) {
    send_count = receive_count = 0;
    // Even invalid dynamic counts leave a bounded, empty task list.
    a.packet_counts[peer] = a.packet_counts[a.world + peer] = 0;
  }
  int* rows = a.packet_rows + int64_t(peer) * a.mailbox_tokens;
  const int tiles = packet_tiles(a.mailbox_tokens);
  if (!a.return_rows) {
    for (int t = tid; t < a.mailbox_tokens; t += blockDim.x) rows[t] = -1;
    __syncthreads();
    const int count = a.owner_counts[peer];
    if (count < 0 || count > a.input_rows) { if (!tid) fail(a.status, 1); return; }
    if (peer != a.rank) {
      for (int i = tid; i < count; i += blockDim.x) {
        const int row = a.owner_rows[int64_t(peer) * a.input_rows + i];
        if (row < 0 || row >= a.input_rows) { fail(a.status, 1); continue; }
        const int64_t token = a.input_inverse[int64_t(row) * 2 + 1];
        if (token < 0 || token >= a.mailbox_tokens) { fail(a.status, 2); continue; }
        if (a.diagnostics && a.input_inverse[int64_t(row) * 2] != peer) fail(a.status, 1);
        rows[token] = row; // Dispatch guarantees one combined row per owner/token.
      }
    }
    __syncthreads();
  } else {
    // Publish the shared task counters before other warps append live tiles.
    __syncthreads();
  }
  for (int tile = tid; tile < tiles; tile += blockDim.x) {
    unsigned send = 0, receive = 0;
    for (int i = 0; i < kPacketRows; ++i) {
      const int token = tile * kPacketRows + i;
      if (peer != a.rank && token < a.mailbox_tokens && rows[token] >= 0) send |= 1u << i;
      if (token < a.capacity) {
        const uint64_t mask = a.expected_ranks[token];
        if (peer != a.rank && (mask & (uint64_t(1) << peer))) receive |= 1u << i;
        if (peer == a.rank) {
          const int row = a.local_rows[token];
          if ((mask >> a.world) || row < -1 || row >= a.input_rows ||
              bool(mask & (uint64_t(1) << peer)) != (row >= 0)) fail(a.status, 1);
        }
      }
    }
    a.packet_send_masks[peer * tiles + tile] = send;
    a.packet_recv_masks[peer * tiles + tile] = receive;
    if (send) a.packet_tasks[peer * tiles + atomicAdd(&send_count, 1)] = peer * tiles + tile;
    if (receive) a.packet_tasks[a.world * tiles + peer * tiles + atomicAdd(&receive_count, 1)] = peer * tiles + tile;
  }
  __syncthreads();
  if (!tid) {
    a.packet_counts[peer] = send_count;
    a.packet_counts[a.world + peer] = receive_count;
  }
}

// Independent, cooperatively resident full-width TX/RX CTAs. Each control warp
// polls a bounded window of tasks and consumes any ready member, rather than
// waiting for all contributors of the first token. No cross-CTA barrier or
// host atomic is needed. A window drains before moving on; this is bounded
// lookahead, not a claim of globally work-conserving scheduling.
template <bool Profile = false>
__global__ void host_sm_packet_transport_kernel(TransportArgs a) {
  if (a.device_epoch) a.epoch = *a.device_epoch;
  const int tid = threadIdx.x, channel = blockIdx.x % a.channels;
  const bool send = blockIdx.x < a.channels;
  const int tiles = packet_tiles(a.mailbox_tokens);
  constexpr int stride = PFieldCount + 4;
  __shared__ int error, selected, sample;
  __shared__ unsigned chosen_mask;
  __shared__ int offsets[kMaxRanks + 1];
  ProfileState<Profile> profile;
  if constexpr (Profile) {
    if (!tid) {
      sample = -1;
      if (a.decode_profile[1]) {
        auto* header = a.decode_profile + int64_t(blockIdx.x) * stride;
        const int64_t index = header[0]++;
        if (index < a.decode_profile_samples) sample = int(index);
        else ++header[2];
      }
    }
    __syncthreads();
    if (sample >= 0) profile_init(profile);
  }
  const auto phase = [&](int field) {
    if constexpr (Profile) { if (sample >= 0) profile_phase(profile, field); }
  };
  if (tid < a.world) offsets[tid + 1] = a.packet_counts[(send ? 0 : a.world) + tid];
  __syncthreads();
  if (!tid) {
    error = read_status(a.status);
    offsets[0] = 0;
    for (int p = 0; p < a.world; ++p) offsets[p + 1] += offsets[p];
  }
  __syncthreads();
  const int tasks = offsets[a.world];
  const uint64_t start = a.diagnostics ? ticks_ns() : 0;
  uint32_t iteration = 0;
  for (int base = 0; channel + base * a.channels < tasks; base += 32) {
    const int ordinal = channel + (base + tid) * a.channels;
    int task = -1;
    unsigned mask = 0, pending = 0;
    char* box = nullptr;
    uint64_t previous = 0;
    if (tid < 32) {
      if (ordinal < tasks) {
        int peer = 0;
        while (peer + 1 < a.world && ordinal >= offsets[peer + 1]) ++peer;
        task = a.packet_tasks[(send ? 0 : a.world * tiles) + peer * tiles + ordinal - offsets[peer]];
        mask = send ? a.packet_send_masks[task] : a.packet_recv_masks[task];
        if (mask) {
          const int peer = task / tiles;
          box = packet_box(a, send ? a.rank : peer,
                           send ? task : a.rank * tiles + task % tiles);
          if (send) previous = a.packet_sent_epochs[int64_t(task) * kPacketBanks + a.epoch % kPacketBanks];
        }
      }
      pending = __ballot_sync(0xffffffff, mask != 0);
    }
    while (true) {
      phase(send ? PSendMetadata : PPeerPoll);
      if (tid < 32) {
        unsigned ready = 0;
        // Let the control warp poll without waking the payload workers for
        // every miss. Exit only for progress, exhaustion, or sticky failure.
        do {
          bool available = false;
          if (pending & (1u << tid))
            available = send ? (!previous || acquire(control(box, 1)) == previous)
                             : (acquire(control(box, 0)) == a.epoch);
          ready = __ballot_sync(0xffffffff, available);
          if (!tid) error = mailbox_health(a, start, iteration++);
          const int failed = __shfl_sync(0xffffffff, tid == 0 ? error : 0, 0);
          if constexpr (Profile) { if (!tid && sample >= 0 && pending) {
            ++profile.values[send ? PCreditPolls : PReadyPolls];
            profile.values[send ? PCreditBlockedObservations : PUnreadyObservations] += !ready;
          }}
          if (ready || !pending || failed) break;
          __nanosleep(64);
        } while (true);
        const int lane = __ffs(ready) - 1;
        const unsigned next_mask = __shfl_sync(0xffffffff, mask, lane >= 0 ? lane : 0);
        const int next_task = __shfl_sync(0xffffffff, task, lane >= 0 ? lane : 0);
        if (!tid) {
          selected = lane < 0 ? -1 : next_task;
          chosen_mask = next_mask;
        }
        if (lane >= 0) pending &= ~(1u << lane);
      }
      // Acquired READY and selected descriptor become visible to all readers.
      __syncthreads();
      if (selected < 0 || error) break;
      const int peer = selected / tiles, tile = selected % tiles;
      char* payload = packet_box(a, send ? a.rank : peer,
                                send ? selected : a.rank * tiles + tile) + 128;
      if (a.diagnostics) {
        if (!tid && !send && (host_load64(payload - 112) != a.identity ||
                            host_load32(payload - 104) != chosen_mask)) fail(a.status, 5);
        __syncthreads();
        if (!tid) error = read_status(a.status);
        __syncthreads();
        if (error) break;
      }
      phase(send ? PSendPayload : PReceivePayload);
      // Flatten all live row/vector units over the full CTA. Sparse holes are
      // skipped, not transmitted or read. Partial final tiles commit promptly.
      const bool vector = a.hidden % 8 == 0 &&
          (!send || (reinterpret_cast<uintptr_t>(a.input) & 15) == 0);
      const int units = a.hidden / (vector ? 8 : 2);
      if (!send) {
        if (vector) packet_receive<true>(a, payload, peer, tile, chosen_mask);
        else packet_receive<false>(a, payload, peer, tile, chosen_mask);
      }
      for (int64_t i = tid; send && i < int64_t(kPacketRows) * units; i += blockDim.x) {
        const int offset = int(i / units), h = int(i % units);
        if (!(chosen_mask & (1u << offset))) continue;
        const int token = tile * kPacketRows + offset;
        char* host = payload + int64_t(offset) * a.hidden * 2;
        const int row = a.packet_rows[int64_t(peer) * a.mailbox_tokens + token];
        const auto* source = a.input + int64_t(row) * a.hidden;
        if (vector) {
          // One vector store preserves coalescing across the whole warp;
          // four strided scalar stores would fragment the same cache lines.
          host_store128(host + int64_t(h) * 16, reinterpret_cast<const uint4*>(source)[h]);
        } else host_store32(host + int64_t(h) * 4, reinterpret_cast<const uint32_t*>(source)[h]);
      }
      phase(send ? PPublish : PAck);
      // Every writer/reader completes before the one owner publishes READY or
      // ACK. RX no longer needs the host packet after staging; reduction only
      // touches its private HBM copy. No other rank participates in this credit.
      __threadfence_system();
      __syncthreads();
      if (!tid) {
        char* header = payload - 128;
        if (send && a.diagnostics) {
          host_store64(header + 16, a.identity);
          host_store32(header + 24, chosen_mask);
        }
        release(control(header, send ? 0 : 1), a.epoch);
        if (send) a.packet_sent_epochs[int64_t(selected) * kPacketBanks + a.epoch % kPacketBanks] = a.epoch;
        if constexpr (Profile) { if (sample >= 0) {
          const int rows = __popc(chosen_mask);
          ++profile.values[send ? PSendFrames : PReceiveFrames];
          profile.values[send ? PSendRows : PReceiveRows] += rows;
          profile.values[send ? PWrittenBytes : PReadBytes] += int64_t(rows) * a.hidden * 2;
        }}
      }
      __syncthreads(); // Protect selected/mask while workers finish this packet.
    }
    // The exhausted/error branch skips the payload's trailing barrier. Retire
    // its shared decision before the control warp can overwrite it for the
    // next window; otherwise a lagging warp can observe a future selection.
    __syncthreads();
  }
  if constexpr (Profile) { if (sample >= 0) {
    __syncthreads();
    if (!tid) {
      const uint64_t now = ticks_ns();
      profile.values[profile.phase] += now - profile.mark;
      profile.values[PEnd] = now;
      profile.values[PElapsed] = now - profile.values[PBegin];
      profile.values[PStatus] = error;
      auto* out = a.decode_profile +
          (int64_t(sample + 1) * a.decode_profile_channels + blockIdx.x) * stride;
      for (int i = 0; i < PFieldCount; ++i) out[i] = profile.values[i];
      out[PFieldCount] = a.epoch; out[PFieldCount + 1] = a.capacity;
      out[PFieldCount + 2] = a.input_rows; out[PFieldCount + 3] = a.channels;
    }
  }}
  if (!tid && error) {
    auto* own = reinterpret_cast<uint64_t*>(a.mailbox_pointers[a.rank]);
    release(own, uint64_t(error));
    if (a.fatal_errors || acquire(own + 2)) { __threadfence_system(); asm volatile("trap;"); }
  }
}

// Separate stream-ordered finalizer preserves ascending-rank FP32 addition,
// intermediate BF16 rounding, then local BF16 addition. Unrouted scratch rows
// are never read, so neither capacity clears nor cached routes are needed.
__global__ void host_sm_packet_reduce_kernel(TransportArgs a) {
  if (read_status(a.status)) return;
  for (int token = blockIdx.x; token < a.capacity; token += gridDim.x) {
    const uint64_t readers = a.expected_ranks[token];
    const int local = a.local_rows[token];
    if (a.hidden % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(a.input) | reinterpret_cast<uintptr_t>(a.output)) & 15) == 0) {
      mailbox_reduce_vectors<2, true>(a, token, local, readers,
          reinterpret_cast<const uint32_t*>(a.packet_scratch + int64_t(token) * a.hidden),
          int64_t(a.mailbox_tokens) * (a.hidden / 2));
      continue;
    }
    for (int h = threadIdx.x; h < a.hidden / 2; h += blockDim.x) {
      float sum0 = 0, sum1 = 0;
      for (int peer = 0; peer < a.world; ++peer) {
        if (peer == a.rank) continue;
        uint32_t bits = 0;
        if (readers & (uint64_t(1) << peer))
          bits = reinterpret_cast<const uint32_t*>(a.packet_scratch +
              (int64_t(peer) * a.mailbox_tokens + token) * a.hidden)[h];
        const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
        sum0 += value.x; sum1 += value.y;
      }
      const at::BFloat16 rounded0(sum0), rounded1(sum1);
      const int64_t offset = int64_t(local) * a.hidden + h * 2;
      a.output[int64_t(token) * a.hidden + h * 2] =
          at::BFloat16(float(rounded0) + (local >= 0 ? float(a.input[offset]) : 0.0f));
      a.output[int64_t(token) * a.hidden + h * 2 + 1] =
          at::BFloat16(float(rounded1) + (local >= 0 ? float(a.input[offset + 1]) : 0.0f));
    }
  }
}
#endif
