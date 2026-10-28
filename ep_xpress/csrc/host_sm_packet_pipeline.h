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

__device__ void packet_pipeline_transport(TransportArgs a) {
  if (a.device_epoch) a.epoch = *a.device_epoch;
  const int tid = threadIdx.x, lane = tid % 32;
  const int channel = blockIdx.x % a.channels;
  const bool send = blockIdx.x < a.channels;
  const int tiles = packet_tiles(a.mailbox_tokens);
  __shared__ PacketPipelineStage stages[2];
  __shared__ int offsets[kMaxRanks + 1], choice;
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
        bool available = false;
        if (pending & (1u << lane))
          available = send ? (!previous || acquire(control(box, 1)) == previous)
                           : acquire(control(box, 0)) == a.epoch;
        const unsigned ready = __ballot_sync(0xffffffffu, available);
        if (ready) {
          const int selected = __ffs(ready) - 1;
          const int selected_task = __shfl_sync(0xffffffffu, task, selected);
          const unsigned selected_mask = __shfl_sync(0xffffffffu, mask, selected);
          const uint64_t selected_box = __shfl_sync(0xffffffffu, uint64_t(box), selected);
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
      if (!worker) {
        choice = -1;
        bool done = true;
        for (int i = 0; i < 2; ++i) {
          const int index = (next + i) % 2;
          const uint32_t state = dispatch_stage_load(&stages[index].state);
          done &= state == 3;
          if (state == 1 && choice < 0) choice = index;
        }
        if (done || read_status(a.status)) choice = -2;
      }
      dispatch_rx_sync();
      const int selected = choice;
      if (selected == -2) break;
      if (selected < 0) { dispatch_rx_sync(); __nanosleep(64); continue; }
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
      }
      if (!send) {
        if (vector) packet_receive<true>(a, payload, peer, tile, stage.mask, worker, kThreads);
        else packet_receive<false>(a, payload, peer, tile, stage.mask, worker, kThreads);
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
      __threadfence_system();
      dispatch_rx_sync();
      if (!worker) dispatch_stage_store(&stages[selected].state, 2);
      dispatch_rx_sync();
      next = (selected + 1) % 2;
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
