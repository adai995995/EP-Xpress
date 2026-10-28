// Full-width packet workers with an independent control/commit warp. The two
// shared descriptors are generation-protected by FREE -> READY -> DONE -> FREE.
// Neither role reuses a descriptor while the other role still references it.
#if EP_XPRESS_PACKET_PIPELINE
struct PacketPipelineStage {
  uint32_t state;
  int task;
  unsigned mask;
  char* box;
};

#if EP_XPRESS_READY_BATCH
// Both descriptors were already READY when the worker leader selected them.
// Keep the complete payload group and issue loads from both peers before any
// of their values are consumed. Two loads per descriptor per iteration bound
// the live value set to the ordinary single-packet unroll's four vectors.
template <bool Vector>
__device__ __forceinline__ void packet_receive_ready_pair(
    TransportArgs a, const PacketPipelineStage& first,
    const PacketPipelineStage& second, int worker, int tiles) {
  constexpr int unroll = 2;
  using Value = typename std::conditional<Vector, uint4, uint32_t>::type;
  const char* payloads[2] = {first.box + 128, second.box + 128};
  const unsigned masks[2] = {first.mask, second.mask};
  const int peers[2] = {first.task / tiles, second.task / tiles};
  const int token_bases[2] = {(first.task % tiles) * kPacketRows,
                             (second.task % tiles) * kPacketRows};
  const int units = a.hidden / (Vector ? 8 : 2);
  const int64_t total = int64_t(kPacketRows) * units;
  for (int64_t base = worker; base < total; base += kThreads * unroll) {
    Value values[2][unroll];
    bool live[2][unroll];
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      const int64_t i = base + j * kThreads;
      const int offset = int(i / units), h = int(i % units);
#pragma unroll
      for (int packet = 0; packet < 2; ++packet) {
        live[packet][j] = i < total && (masks[packet] & (1u << offset));
        if (live[packet][j]) {
          const char* source = payloads[packet] + int64_t(offset) * a.hidden * 2;
          if constexpr (Vector) values[packet][j] = host_load128(source + int64_t(h) * 16);
          else values[packet][j] = host_load32(source + int64_t(h) * 4);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < unroll; ++j) {
      const int64_t i = base + j * kThreads;
      const int offset = int(i / units), h = int(i % units);
#pragma unroll
      for (int packet = 0; packet < 2; ++packet) {
        if (live[packet][j]) {
          const int token = token_bases[packet] + offset;
          auto* target = a.packet_scratch +
              (int64_t(peers[packet]) * a.mailbox_tokens + token) * a.hidden;
          if constexpr (Vector) reinterpret_cast<uint4*>(target)[h] = values[packet][j];
          else reinterpret_cast<uint32_t*>(target)[h] = values[packet][j];
        }
      }
    }
  }
}
#endif

__device__ void packet_pipeline_transport(TransportArgs a) {
  if (a.device_epoch) a.epoch = *a.device_epoch;
  const int tid = threadIdx.x, lane = tid % 32;
  const int channel = blockIdx.x % a.channels;
  const bool send = blockIdx.x < a.channels;
  const int tiles = packet_tiles(a.mailbox_tokens);
  __shared__ PacketPipelineStage stages[2];
  __shared__ int offsets[kMaxRanks + 1], choice;
#if EP_XPRESS_READY_BATCH
  __shared__ int second_choice;
#endif
  if (tid < 2) stages[tid].state = 0;
  if (tid < a.world) offsets[tid + 1] = a.packet_counts[(send ? 0 : a.world) + tid];
  __syncthreads();
  if (!tid) {
    offsets[0] = 0;
    for (int p = 0; p < a.world; ++p) offsets[p + 1] += offsets[p];
  }
  __syncthreads();
  const int tasks = offsets[a.world];
  const uint64_t start = a.diagnostics ? ticks_ns() : 0;
  if (tid < 32) {
    int base = 0, task = -1;
    unsigned mask = 0, pending = 0;
    char* box = nullptr;
    uint64_t previous = 0;
    uint32_t iteration = 0;
#if EP_XPRESS_PACKET_READY_CACHE
    // This lane owns one task in the current batch. A successful observation
    // remains valid until that task is consumed: RX has not ACKed the slot,
    // and TX has not yet reused its credited generation.
    bool known_ready = false;
#endif
    while (true) {
      int free_stage = -1;
      bool occupied = false;
      // Reap before polling: an unready peer must not delay a completed ACK.
      if (!lane) {
        for (int i = 0; i < 2; ++i) {
          auto& stage = stages[i];
          uint32_t state = dispatch_stage_load(&stage.state);
          if (state == 2) {
            release(control(stage.box, send ? 0 : 1), a.epoch);
            if (send) a.packet_sent_epochs[int64_t(stage.task) * kPacketBanks + a.epoch % kPacketBanks] = a.epoch;
            dispatch_stage_store(&stage.state, 0);
            state = 0;
          }
          if (!state && free_stage < 0) free_stage = i;
          occupied |= state != 0;
        }
      }
      free_stage = __shfl_sync(0xffffffffu, free_stage, 0);
      occupied = __shfl_sync(0xffffffffu, int(occupied), 0);
      if (!pending && channel + base * a.channels < tasks) {
        const int ordinal = channel + (base + lane) * a.channels;
        mask = 0;
#if EP_XPRESS_PACKET_READY_CACHE
        // Never carry a READY/credit observation into a different task.
        known_ready = false;
#endif
        if (ordinal < tasks) {
          int peer = 0;
          while (peer + 1 < a.world && ordinal >= offsets[peer + 1]) ++peer;
          task = a.packet_tasks[(send ? 0 : a.world * tiles) + peer * tiles + ordinal - offsets[peer]];
          mask = send ? a.packet_send_masks[task] : a.packet_recv_masks[task];
          box = packet_box(a, send ? a.rank : peer, send ? task : a.rank * tiles + task % tiles);
          previous = send ? a.packet_sent_epochs[int64_t(task) * kPacketBanks + a.epoch % kPacketBanks] : 0;
        }
        pending = __ballot_sync(0xffffffffu, mask != 0);
        base += 32;
      }
      int error = !lane ? mailbox_health(a, start, iteration++) : 0;
      error = __shfl_sync(0xffffffffu, error, 0);
      if (error || (!pending && !occupied && channel + base * a.channels >= tasks)) break;
      if (pending && free_stage >= 0) {
#if EP_XPRESS_PACKET_READY_CACHE
        // Consume known-ready work before issuing another lane's host load.
        // Otherwise the following ballot would still wait for unknown peers'
        // PCIe reads even though a cached task can already fill this stage.
        unsigned ready = __ballot_sync(0xffffffffu,
            (pending & (1u << lane)) && known_ready);
        if (!ready) {
          if (pending & (1u << lane))
            known_ready = send ? (!previous || acquire(control(box, 1)) == previous)
                               : acquire(control(box, 0)) == a.epoch;
          ready = __ballot_sync(0xffffffffu,
              (pending & (1u << lane)) && known_ready);
        }
#else
        bool available = false;
        if (pending & (1u << lane)) {
          available = send ? (!previous || acquire(control(box, 1)) == previous)
                           : acquire(control(box, 0)) == a.epoch;
        }
        const unsigned ready = __ballot_sync(0xffffffffu, available);
#endif
        if (ready) {
          const int selected = __ffs(ready) - 1;
          const int selected_task = __shfl_sync(0xffffffffu, task, selected);
          const unsigned selected_mask = __shfl_sync(0xffffffffu, mask, selected);
          const uint64_t selected_box = __shfl_sync(0xffffffffu, uint64_t(box), selected);
          // READY/credit may have been acquired by another lane. Register
          // shuffles do not order memory: bridge it to the stage publisher.
          __syncwarp();
          if (!lane) {
            auto& stage = stages[free_stage];
            stage.task = selected_task;
            stage.mask = selected_mask;
            stage.box = reinterpret_cast<char*>(selected_box);
            dispatch_stage_store(&stage.state, 1);
          }
          pending &= ~(1u << selected);
        }
      }
      __syncwarp();
    }
    if (!lane) for (auto& stage : stages) dispatch_stage_store(&stage.state, 3);
  } else {
    const int worker = tid - 32;
    int next = 0;
    while (true) {
#if EP_XPRESS_RX_POLL_WARP
      // Only the first complete RX worker warp retries an empty shared-stage
      // scan. The other seven warps wait at the one handoff barrier below.
      // TX retains its one-scan-per-worker-iteration behavior.
      if (send || worker < 32) {
        int polled_choice = -1;
        do {
#endif
      if (!worker) {
        choice = -1;
#if EP_XPRESS_READY_BATCH
        second_choice = -1;
#endif
        bool done = true;
        for (int i = 0; i < 2; ++i) {
          const int index = (next + i) % 2;
          const uint32_t state = dispatch_stage_load(&stages[index].state);
          done &= state == 3;
          if (state == 1) {
            if (choice < 0) choice = index;
#if EP_XPRESS_READY_BATCH
            else if (!send) second_choice = index;
#endif
          }
        }
        if (done || read_status(a.status)) choice = -2;
      }
#if EP_XPRESS_RX_POLL_WARP
          if (send) break;
          polled_choice = __shfl_sync(0xffffffffu, !worker ? choice : -1, 0);
          if (polled_choice == -1) __nanosleep(64);
        } while (polled_choice == -1);
      }
#endif
      dispatch_rx_sync();
      const int selected = choice;
      if (selected == -2) break;
      if (selected < 0) { dispatch_rx_sync(); __nanosleep(64); continue; }
#if EP_XPRESS_READY_BATCH
      const int selected_second = second_choice;
#endif
      const auto& stage = stages[selected];
      const int peer = stage.task / tiles, tile = stage.task % tiles;
      char* payload = stage.box + 128;
      const bool vector = a.hidden % 8 == 0 &&
          (!send || (reinterpret_cast<uintptr_t>(a.input) & 15) == 0);
      if (a.diagnostics && !worker) {
        if (send) {
          host_store64(stage.box + 16, a.identity);
          host_store32(stage.box + 24, stage.mask);
        } else if (host_load64(stage.box + 16) != a.identity || host_load32(stage.box + 24) != stage.mask) {
          fail(a.status, 5);
        }
#if EP_XPRESS_READY_BATCH
        if (!send && selected_second >= 0) {
          const auto& second = stages[selected_second];
          if (host_load64(second.box + 16) != a.identity || host_load32(second.box + 24) != second.mask)
            fail(a.status, 5);
        }
#endif
      }
      if (!send) {
#if EP_XPRESS_READY_BATCH
        if (selected_second >= 0) {
          if (vector) packet_receive_ready_pair<true>(a, stage, stages[selected_second], worker, tiles);
          else packet_receive_ready_pair<false>(a, stage, stages[selected_second], worker, tiles);
        } else
#endif
        {
          if (vector) packet_receive<true>(a, payload, peer, tile, stage.mask, worker, kThreads);
          else packet_receive<false>(a, payload, peer, tile, stage.mask, worker, kThreads);
        }
      } else {
        const int units = a.hidden / (vector ? 8 : 2);
        for (int64_t i = worker; i < int64_t(kPacketRows) * units; i += kThreads) {
          const int offset = int(i / units), h = int(i % units);
          if (!(stage.mask & (1u << offset))) continue;
          const int token = tile * kPacketRows + offset;
          const int row = a.packet_rows[int64_t(peer) * a.mailbox_tokens + token];
          const auto* source = a.input + int64_t(row) * a.hidden;
          char* host = payload + int64_t(offset) * a.hidden * 2;
          if (vector) host_store128(host + int64_t(h) * 16, reinterpret_cast<const uint4*>(source)[h]);
          else host_store32(host + int64_t(h) * 4, reinterpret_cast<const uint32_t*>(source)[h]);
        }
      }
      if (send) __threadfence_system();
      else host_rx_fence();
      dispatch_rx_sync();
      if (!worker) {
        // Both reads and HBM writes have joined before either descriptor can
        // be ACKed/reused. No worker refers to a descriptor after this release.
        dispatch_stage_store(&stages[selected].state, 2);
#if EP_XPRESS_READY_BATCH
        if (selected_second >= 0) dispatch_stage_store(&stages[selected_second].state, 2);
#endif
      }
      dispatch_rx_sync();
#if EP_XPRESS_READY_BATCH
      next = ((selected_second >= 0 ? selected_second : selected) + 1) % 2;
#else
      next = (selected + 1) % 2;
#endif
    }
  }
  __syncthreads();
  if (!tid && read_status(a.status)) {
    auto* own = reinterpret_cast<uint64_t*>(a.mailbox_pointers[a.rank]);
    release(own, uint64_t(read_status(a.status)));
    if (a.fatal_errors || acquire(own + 2)) { __threadfence_system(); asm volatile("trap;"); }
  }
}
#endif
