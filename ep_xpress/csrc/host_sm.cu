// GPU-initiated mapped-host transport. No host atomics, peer pointers or CE
// payload copies. Each all-resident channel CTA progresses both directions.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <cub/block/block_scan.cuh>
#include "host_sm.h"

namespace {
constexpr int kThreads = 256;
constexpr int kMaxRanks = 63;
constexpr int64_t kArenaHeader = 128;
constexpr int64_t kSlotHeader = 640;
constexpr int64_t kRowsOffset = 512;
constexpr int64_t kIdentityOffset = 520;

// Diagnostic schema v1. Times are GPU-local nanoseconds; begin/end must not be
// compared across GPUs. The nine phase durations partition elapsed_ns exactly.
// Idle durations are a SECOND, overlapping view of completed loop iterations:
// only iterations with no cursor/frame progress are classified, exclusively.
// Do not add idle durations to phase durations or sum channel times as latency.
// Payload phases include GPU instructions and diagnostic CTA synchronization;
// publication/ACK phases include system fences. None is pure PCIe wire time.
// Initial setup is included in other_control. Final profile-record stores,
// abort publication, and the separate init/finalize kernels are excluded.
// A progress iteration advances the local cursor or publishes/consumes any
// frame, including terminal/nonrecipient frames. Ready/credit flags are the
// snapshots observed during that iteration, not a continuously sampled state.
// Credit poll counts cover the next producer slot; final-drain slot checks are
// in other_control and idle_final_drain, not in the credit poll counters.
// Frames include terminals; rows/bytes count payload actually written/read,
// exclude local bypass, and receive rows count only this rank's selected rows.
// first_receive is first observed ready frame; first_payload_receive is the
// start of the first selected payload read. Publish milestones follow release.
// Missing milestones are -1. Error exits may leave an iteration unclassified;
// phase partition and observed status are still written. completion_phase is
// 0..8 in the nine-phase order below (the phase where completion is observed).
enum ProfileField {
  PBegin, PEnd, PElapsed,
  PRouteLocal, PSendMetadata, PSendPayload, PPublish, PPeerPoll,
  PReceiveMetadata, PReceivePayload, PAck, POther,
  PIdleReady, PIdleCredit, PIdleBoth, PIdleDrain, PIdleOther,
  PFirstPublish, PFirstReceive, PWrittenBytes, PReadBytes,
  PSendFrames, PReceiveFrames, PSendRows, PReceiveRows, PLocalRows,
  PIterations, PReadyPolls, PUnreadyObservations, PCreditPolls,
  PCreditBlockedObservations, PIdleIterations, PCreditProgressIterations,
  PUnreadyProgressIterations, PStatus, PCompletionPhase,
  PFirstPayloadPublish, PFirstPayloadReceive, PTerminalSent,
  PTerminalReceived, PProgressIterations, PLastProgress, PFieldCount
};
const std::vector<std::string> kProfileFields = {
  "begin_ns", "end_ns", "elapsed_ns", "route_local_ns", "send_metadata_ns",
  "send_payload_ns", "publish_fence_ready_ns", "peer_poll_ns",
  "receive_metadata_select_ns", "receive_payload_ns", "ack_publish_ns",
  "other_control_ns", "idle_wait_ready_ns", "idle_wait_credit_ns",
  "idle_wait_both_ns", "idle_final_drain_ns", "idle_other_ns",
  "first_publish_elapsed_ns", "first_receive_elapsed_ns", "payload_written_bytes",
  "payload_read_bytes", "send_frames", "receive_frames", "send_payload_rows",
  "receive_payload_rows", "local_rows", "loop_iterations", "peer_ready_polls",
  "peer_unready_observations", "credit_polls", "credit_blocked_observations",
  "idle_iterations", "credit_blocked_progress_iterations",
  "ready_missing_progress_iterations", "status", "completion_phase",
  "first_payload_publish_elapsed_ns", "first_payload_receive_elapsed_ns",
  "terminal_sent", "terminal_received", "progress_iterations",
  "last_progress_elapsed_ns"
};

template <bool Enabled> struct ProfileState {};
template <> struct ProfileState<true> {
  int64_t values[PFieldCount];
  uint64_t mark, iteration_begin;
  int phase;
  bool progress, credit_missing, ready_missing;
  int credit_blocked[kMaxRanks];
};

int64_t align_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Only single-writer, naturally aligned 64-bit loads/stores are used for
// synchronization. Mapped-host atomic RMW is intentionally never used.
__device__ __forceinline__ uint64_t acquire(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ void release(uint64_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(address), "l"(value) : "memory");
}
__device__ __forceinline__ uint32_t host_load32(const void* address) {
  uint32_t value;
  // Discard cached system-memory data on slot reuse. A ready acquire alone
  // must not be confused with invalidation of unrelated cached payload lines.
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ uint64_t host_load64(const void* address) {
  const auto* p = static_cast<const uint32_t*>(address);
  return uint64_t(host_load32(p)) | (uint64_t(host_load32(p + 1)) << 32);
}
__device__ __forceinline__ void host_store32(void* address, uint32_t value) {
  *static_cast<volatile uint32_t*>(address) = value;
}
__device__ __forceinline__ void host_store64(void* address, uint64_t value) {
  *static_cast<volatile uint64_t*>(address) = value;
}
__device__ __forceinline__ uint64_t ticks_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

template <bool Enabled>
__device__ __forceinline__ void profile_init(ProfileState<Enabled>& p) {
  if constexpr (Enabled) {
    if (threadIdx.x == 0) {
      const uint64_t now = ticks_ns();
      for (int i = 0; i < PFieldCount; ++i) p.values[i] = 0;
      p.values[PBegin] = now;
      p.values[PFirstPublish] = p.values[PFirstReceive] = -1;
      p.values[PFirstPayloadPublish] = p.values[PFirstPayloadReceive] = -1;
      p.phase = POther;
      p.mark = now;
    }
  }
}

// Both barriers disappear entirely in the normal specialization. The first
// closes the preceding CTA work; the second starts the next phase together.
// Boundary overhead is intentionally present in diagnostic phase durations.
template <bool Enabled>
__device__ __forceinline__ void profile_phase(ProfileState<Enabled>& p, int phase) {
  if constexpr (Enabled) {
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint64_t now = ticks_ns();
      p.values[p.phase] += now - p.mark;
      p.phase = phase;
      p.mark = now;
    }
    __syncthreads();
  }
}

template <bool Enabled>
__device__ __forceinline__ void profile_iteration_begin(ProfileState<Enabled>& p) {
  if constexpr (Enabled) {
    if (threadIdx.x == 0) {
      p.iteration_begin = ticks_ns();
      p.progress = p.credit_missing = p.ready_missing = false;
      ++p.values[PIterations];
    }
  }
}

template <bool Enabled>
__device__ __forceinline__ void profile_iteration_end(ProfileState<Enabled>& p,
                                                      bool final_drain) {
  if constexpr (Enabled) {
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint64_t now = ticks_ns();
      if (p.progress) {
        ++p.values[PProgressIterations];
        p.values[PCreditProgressIterations] += p.credit_missing;
        p.values[PUnreadyProgressIterations] += p.ready_missing;
        p.values[PLastProgress] = now - p.values[PBegin];
      } else {
        ++p.values[PIdleIterations];
        const int field = final_drain ? PIdleDrain :
            p.credit_missing && p.ready_missing ? PIdleBoth :
            p.credit_missing ? PIdleCredit : p.ready_missing ? PIdleReady : PIdleOther;
        p.values[field] += now - p.iteration_begin;
      }
    }
    __syncthreads();
  }
}

template <bool Enabled>
__device__ __forceinline__ void profile_finish(ProfileState<Enabled>& p,
                                               int64_t* output, int status) {
  if constexpr (Enabled) {
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint64_t now = ticks_ns();
      p.values[p.phase] += now - p.mark;
      p.values[PEnd] = now;
      p.values[PElapsed] = now - p.values[PBegin];
      p.values[PStatus] = status;
      p.values[PCompletionPhase] = p.phase - PRouteLocal;
      for (int i = 0; i < PFieldCount; ++i)
        output[int64_t(blockIdx.x) * PFieldCount + i] = p.values[i];
      __threadfence();
    }
  }
}
__device__ __forceinline__ uint64_t* control(char* p, int word) {
  return reinterpret_cast<uint64_t*>(p) + word;
}

// Queue payload uses separate arrays for masks, tokens, IDs, weights and X.
// The existing row-stride allocation bounds these arrays, including alignment.
__host__ __device__ int64_t ids_offset(int rows) {
  return (kSlotHeader + int64_t(rows) * 12 + 7) / 8 * 8;
}
__host__ __device__ int64_t weights_offset(int rows, int topk) {
  return ids_offset(rows) + int64_t(rows) * topk * 8;
}
__host__ __device__ int64_t payload_offset(int rows, int topk) {
  return weights_offset(rows, topk) + int64_t(rows) * topk * 4;
}

struct TransportArgs {
  const at::BFloat16* input;
  const void* ids;
  const float* weights;
  const int32_t* expert_rank;
  const int32_t* expert_local;
  const int64_t* pointers;
  const int64_t* mailbox_pointers;
  int64_t* expected_ranks;
  int32_t* local_rows;
  int mailbox_tokens;
  int64_t mailbox_stride;
  const int64_t* input_inverse;
  const int32_t* input_count;
  at::BFloat16* output;
  int64_t* output_ids;
  float* output_weights;
  int64_t* output_inverse;
  int32_t* output_count;
  int32_t* owner_rows;
  int32_t* owner_counts;
  at::BFloat16* remote;
  int32_t* status;
  int64_t* profile;
  int experts, world, rank, slots, chunk_rows, channels;
  int input_rows, hidden, topk, capacity;
  int64_t slot_stride, channel_stride;
  uint64_t epoch, timeout_ns, identity;
  bool diagnostics = true;
  bool trusted_placement = false;
  bool fatal_errors = false;
};

// Notifications are independent of payload READY: only selected readers see a
// new sequence. A reader never inspects another reader's mutable payload slot.
__device__ __forceinline__ uint64_t* dispatch_notify(TransportArgs a, char* base,
                                                    int slot, int reader) {
  return reinterpret_cast<uint64_t*>(base + kArenaHeader + a.slots * a.slot_stride) +
      slot * kMaxRanks + reader;
}
__device__ __forceinline__ uint64_t* dispatch_terminal(TransportArgs a, char* base) {
  return dispatch_notify(a, base, a.slots, 0);
}

// These atomics address this GPU's VRAM only. Host synchronization words retain
// exactly one GPU writer per producer/channel or consumer/channel.
__device__ __forceinline__ void fail(int32_t* status, int code) {
  atomicCAS(status, 0, code);
}
__device__ __forceinline__ int32_t read_status(const int32_t* status) {
  int32_t value;
  // Observe the sticky device-local atomic word without issuing a contended
  // read-modify-write from every persistent CTA. No payload depends on it.
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(status) : "memory");
  return value;
}

// Separate ordinary grids keep initialization and final reduction from being
// serialized through the small persistent communication grid.
template <bool Combine>
__global__ void host_sm_init_kernel(TransportArgs a) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  if (!Combine && i == 0) *a.output_count = 0;
  if (!Combine && a.local_rows)
    for (int64_t j = i; j < a.input_rows; j += stride) a.local_rows[j] = -1;
  if (!Combine && a.owner_counts)
    for (int64_t j = i; j < a.world; j += stride) a.owner_counts[j] = 0;
  for (int64_t j = i; j < int64_t(a.capacity) * a.hidden; j += stride)
    a.output[j] = at::BFloat16(0.0f);
  if (Combine) {
    for (int64_t j = i; j < int64_t(a.world) * a.capacity * a.hidden; j += stride)
      a.remote[j] = at::BFloat16(0.0f);
  } else {
    for (int64_t j = i; j < int64_t(a.capacity) * a.topk; j += stride) {
      a.output_ids[j] = -1;
      a.output_weights[j] = 0.0f;
    }
    for (int64_t j = i; j < int64_t(a.capacity) * 2; j += stride)
      a.output_inverse[j] = -1;
  }
}

template <bool Combine>
__global__ void host_sm_finalize_kernel(TransportArgs a) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  if (!Combine) {
    if (i == 0 && *a.status) *a.output_count = 0;
  } else if (!*a.status) {
    for (int64_t j = i; j < int64_t(a.capacity) * a.hidden; j += stride) {
      float sum = 0.0f;
      for (int peer = 0; peer < a.world; ++peer)
        if (peer != a.rank) sum += float(a.remote[int64_t(peer) * a.capacity * a.hidden + j]);
      const at::BFloat16 rounded(sum);
      a.output[j] = at::BFloat16(float(rounded) + float(a.output[j]));
    }
  }
}

// status: 0 success, 1 invalid route/inverse, 2 receive capacity exceeded,
// 3 timeout, 4 peer abort, 5 operation identity mismatch. An error is sticky.
// More than one channel requires a cooperative, all-resident launch. Separate
// producer-only and receiver-only blocks would not provide this liveness.
template <bool Combine, typename Id, bool Profile = false, bool Selective = false>
__global__ void host_sm_transport_kernel(TransportArgs a) {
  using Scan = cub::BlockScan<uint64_t, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int cursor, end, error, producer_done, sent_chunks;
  __shared__ int peer_done[kMaxRanks], peer_chunk[kMaxRanks], received_rows[kMaxRanks];
  __shared__ int local_rows[kThreads], remote_rows[kThreads], selected_rows[kThreads];
  __shared__ int tokens[kThreads], selected_tokens[kThreads];
  __shared__ uint64_t masks[kThreads];
  __shared__ int num_local, num_remote, num_selected, output_base, owner_base, complete;
  __shared__ uint64_t slot_readers[64], frame_readers;
  __shared__ uint64_t received_sequence[kMaxRanks];
  __shared__ int received_slot[kMaxRanks], terminal_sent;
  __shared__ uint64_t previous, start;
  __shared__ ProfileState<Profile> profile;
  profile_init(profile);
  const int tid = threadIdx.x, channel = blockIdx.x;
  char* own = reinterpret_cast<char*>(a.pointers[a.rank]) + channel * a.channel_stride;
  const Id* ids = static_cast<const Id*>(a.ids);
  const int wire_topk = Combine ? 0 : a.topk;
  const int64_t id_offset = ids_offset(a.chunk_rows);
  const int64_t weight_offset = weights_offset(a.chunk_rows, wire_topk);
  const int64_t x_offset = payload_offset(a.chunk_rows, wire_topk);
  if (tid == 0) {
    int n = Combine ? *a.input_count : a.input_rows;
    if (n < 0 || n > a.input_rows) { fail(a.status, 1); n = 0; }
    cursor = int(int64_t(n) * channel / a.channels);
    end = int(int64_t(n) * (channel + 1) / a.channels);
    producer_done = sent_chunks = terminal_sent = 0;
    start = a.diagnostics ? ticks_ns() : 0;
  }
  for (int peer = tid; peer < a.world; peer += blockDim.x) {
    peer_done[peer] = peer == a.rank;
    peer_chunk[peer] = 0;
  }
  if constexpr (Selective)
    for (int slot = tid; slot < a.slots; slot += blockDim.x) slot_readers[slot] = 0;
  __syncthreads();
  uint32_t iteration = 0;
  while (true) {
    const bool check_health = a.diagnostics && (iteration++ & 255u) == 0;
    profile_iteration_begin(profile);
    profile_phase(profile, PPeerPoll);
    // Check different peers concurrently; never spin on a blocked producer.
    if (check_health && tid < a.world) {
      auto* p = reinterpret_cast<uint64_t*>(a.pointers[tid] + channel * a.channel_stride);
      if (acquire(p) || acquire(p + 1)) fail(a.status, 4);
    }
    __syncthreads();
    if (tid == 0) error = read_status(a.status);
    __syncthreads();
    if (error) break;
    char* send_slot = own + kArenaHeader + (sent_chunks % a.slots) * a.slot_stride;
    if (tid == 0) previous = producer_done ? 0 : acquire(control(send_slot, 0));
    __syncthreads();
    bool blocked = !producer_done && tid < a.world && tid != a.rank &&
                   (!Selective || (slot_readers[sent_chunks % a.slots] & (uint64_t(1) << tid))) &&
                   acquire(control(send_slot, 1 + tid)) != previous;
    if constexpr (Profile) {
      if (tid < a.world) profile.credit_blocked[tid] = blocked;
    }
    const int any_blocked = __syncthreads_or(blocked);
    if constexpr (Profile) {
      if (tid == 0) {
        profile.credit_missing = any_blocked;
        if (!producer_done) profile.values[PCreditPolls] += a.world - 1;
        for (int peer = 0; peer < a.world; ++peer)
          profile.values[PCreditBlockedObservations] += profile.credit_blocked[peer];
      }
    }
    if (!producer_done && !any_blocked) {
      profile_phase(profile, PRouteLocal);
      const int batch = min(kThreads, min(a.chunk_rows, end - cursor));
      uint64_t mask = 0;
      int token = cursor + tid;
      if (tid < batch) {
        const int row = cursor + tid;
        if (Combine) {
          const int64_t destination = a.input_inverse[int64_t(row) * 2];
          const int64_t index = a.input_inverse[int64_t(row) * 2 + 1];
          if (destination < 0 || destination >= a.world || index < 0 || index > INT32_MAX)
            fail(a.status, 1);
          else {
            mask = uint64_t(1) << destination;
            token = int(index);
            if (destination == a.rank && token >= a.capacity) fail(a.status, 2);
          }
        } else {
          for (int k = 0; k < a.topk; ++k) {
            const int64_t expert = ids[int64_t(row) * a.topk + k];
            if (expert == -1) continue;
            if (expert < 0 || expert >= a.experts ||
                (a.diagnostics && !isfinite(a.weights[int64_t(row) * a.topk + k]))) {
              fail(a.status, 1); continue;
            }
            const int destination = a.expert_rank[expert];
            if (!a.trusted_placement &&
                (destination < 0 || destination >= a.world || a.expert_local[expert] < 0)) {
              fail(a.status, 1); continue;
            }
            if (a.diagnostics)
              for (int j = 0; j < k; ++j)
                if (ids[int64_t(row) * a.topk + j] == expert) fail(a.status, 1);
            mask |= uint64_t(1) << destination;
          }
        }
      }
      if (!Combine && a.expected_ranks && tid < batch) a.expected_ranks[token] = int64_t(mask);
      masks[tid] = mask;
      tokens[tid] = token;
      const bool local = (mask & (uint64_t(1) << a.rank)) != 0;
      const bool remote = (mask & ~(uint64_t(1) << a.rank)) != 0;
      const uint64_t flags = uint64_t(local) | (uint64_t(remote) << 32);
      uint64_t prefix, aggregate;
      Scan(scan_storage).ExclusiveSum(flags, prefix, aggregate);
      if (local) local_rows[uint32_t(prefix)] = tid;
      if (remote) remote_rows[uint32_t(prefix >> 32)] = tid;
      if (tid == 0) {
        num_local = uint32_t(aggregate);
        num_remote = uint32_t(aggregate >> 32);
        output_base = Combine || !num_local ? 0 : atomicAdd(a.output_count, num_local);
        owner_base = !a.owner_counts || !num_local ? 0 : atomicAdd(a.owner_counts + a.rank, num_local);
        if (!Combine && (int64_t(output_base) + num_local > a.capacity ||
            (a.owner_counts && int64_t(owner_base) + num_local > a.capacity))) fail(a.status, 2);
        error = read_status(a.status);
      }
      __syncthreads();
      if (!error) {
        // Flatten rows and hidden words: no per-token CTA barrier or scalar
        // host-metadata dependency stands between successive payload rows.
        for (int64_t i = tid; i < int64_t(num_local) * (a.hidden / 2); i += blockDim.x) {
          int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          const int source = local_rows[row];
          const int target = Combine ? tokens[source] : output_base + row;
          reinterpret_cast<uint32_t*>(a.output + int64_t(target) * a.hidden)[h] =
              reinterpret_cast<const uint32_t*>(a.input + int64_t(cursor + source) * a.hidden)[h];
        }
        if (!Combine) {
          for (int i = tid; i < num_local * a.topk; i += blockDim.x) {
            const int row = i / a.topk, k = i % a.topk;
            const int source = cursor + local_rows[row];
            const int64_t expert = ids[int64_t(source) * a.topk + k];
            const int64_t out = int64_t(output_base + row) * a.topk + k;
            a.output_ids[out] = expert >= 0 && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
            a.output_weights[out] = expert >= 0 ? a.weights[int64_t(source) * a.topk + k] : 0.0f;
          }
          if (tid < num_local) {
            a.output_inverse[int64_t(output_base + tid) * 2] = a.rank;
            a.output_inverse[int64_t(output_base + tid) * 2 + 1] = cursor + local_rows[tid];
            if (a.local_rows) a.local_rows[cursor + local_rows[tid]] = output_base + tid;
            if (a.owner_rows)
              a.owner_rows[int64_t(a.rank) * a.capacity + owner_base + tid] = output_base + tid;
          }
        }
        if constexpr (Profile) {
          if (tid == 0) profile.values[PLocalRows] += num_local;
        }
        profile_phase(profile, PSendMetadata);
        if (tid < num_remote) {
          int source = remote_rows[tid];
          host_store64(send_slot + kSlotHeader + tid * 8, masks[source] & ~(uint64_t(1) << a.rank));
          host_store32(send_slot + kSlotHeader + int64_t(a.chunk_rows) * 8 + tid * 4, tokens[source]);
        }
        if (!Combine) {
          for (int i = tid; i < num_remote * a.topk; i += blockDim.x) {
            int row = i / a.topk, k = i % a.topk;
            int source = cursor + remote_rows[row];
            int64_t expert = ids[int64_t(source) * a.topk + k];
            host_store64(send_slot + id_offset + int64_t(i) * 8, uint64_t(expert));
            host_store32(send_slot + weight_offset + int64_t(i) * 4,
                         __float_as_uint(expert >= 0 ? a.weights[int64_t(source) * a.topk + k] : 0.0f));
          }
        }
        profile_phase(profile, PSendPayload);
        for (int64_t i = tid; i < int64_t(num_remote) * (a.hidden / 2); i += blockDim.x) {
          int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          const uint32_t word = reinterpret_cast<const uint32_t*>(
              a.input + int64_t(cursor + remote_rows[row]) * a.hidden)[h];
          host_store32(send_slot + x_offset + i * 4, word);
        }
        __syncthreads();
        if constexpr (Profile) {
          if (tid == 0) {
            profile.values[PWrittenBytes] += int64_t(num_remote) * a.hidden * 2;
            profile.values[PSendRows] += num_remote;
            profile.progress |= batch != 0;
          }
        }
        profile_phase(profile, PPublish);
        if (tid == 0) cursor += batch;
        __syncthreads();
        if (num_remote || (!Selective && cursor == end)) {
          if (tid == 0) {
            host_store32(send_slot + kRowsOffset, num_remote ? num_remote : uint32_t(-1));
            if (a.diagnostics) host_store64(send_slot + kIdentityOffset, a.identity);
            if constexpr (Selective) {
              frame_readers = 0;
              for (int r = 0; r < num_remote; ++r) frame_readers |= masks[remote_rows[r]];
              frame_readers &= ~(uint64_t(1) << a.rank);
              slot_readers[sent_chunks % a.slots] = frame_readers;
            }
          }
          __threadfence_system();
          __syncthreads();
          if (tid == 0) {
            release(control(send_slot, 0), (a.epoch << 32) | uint64_t(sent_chunks + 1));
            ++sent_chunks;
            if (!num_remote) producer_done = 1;
            if constexpr (Profile) {
              const int64_t elapsed = ticks_ns() - profile.values[PBegin];
              if (profile.values[PFirstPublish] < 0) profile.values[PFirstPublish] = elapsed;
              if (num_remote && profile.values[PFirstPayloadPublish] < 0)
                profile.values[PFirstPayloadPublish] = elapsed;
              ++profile.values[PSendFrames];
              profile.values[PTerminalSent] += !num_remote;
              profile.progress = true;
            }
          }
          if constexpr (Selective) {
            __syncthreads();
            if (tid < a.world && (frame_readers & (uint64_t(1) << tid)))
              release(dispatch_notify(a, own, (sent_chunks - 1) % a.slots, tid),
                      (a.epoch << 32) | uint64_t(sent_chunks));
          }
        }
        if constexpr (Selective) {
          if (tid == 0 && cursor == end) producer_done = 1;
        }
      }
    }
    __syncthreads();
    profile_phase(profile, PPeerPoll);
    // Poll peers concurrently, then process each available chunk in parallel
    // tiles. ACK is published only after every thread has finished all reads.
    if (tid < a.world) {
      int rows = 0;
      if (!peer_done[tid]) {
        char* peer = reinterpret_cast<char*>(a.pointers[tid]) + channel * a.channel_stride;
        char* slot = nullptr;
        if constexpr (Selective) {
          // Every live notification protects its slot until this reader ACKs.
          // Nonreaders skip frame sequences without inspecting slot metadata.
          for (int index = 0; index < a.slots; ++index) {
            const uint64_t sequence = acquire(dispatch_notify(a, peer, index, a.rank));
            char* candidate = peer + kArenaHeader + index * a.slot_stride;
            if ((sequence >> 32) == a.epoch &&
                sequence != acquire(control(candidate, 1 + a.rank))) {
              slot = candidate;
              received_slot[tid] = index;
              received_sequence[tid] = sequence;
              break;
            }
          }
          if (!slot && acquire(dispatch_terminal(a, peer)) == (a.epoch << 32)) {
            if (a.diagnostics && host_load64(peer + 24) != a.identity) fail(a.status, 5);
            received_slot[tid] = -1;
            rows = -1;
          }
        } else {
          char* candidate = peer + kArenaHeader + (peer_chunk[tid] % a.slots) * a.slot_stride;
          uint64_t expected = (a.epoch << 32) | uint64_t(peer_chunk[tid] + 1);
          if (acquire(control(candidate, 0)) == expected) slot = candidate;
        }
        if (slot) {
          rows = int32_t(host_load32(slot + kRowsOffset));
          if (a.diagnostics && peer_chunk[tid] == 0 &&
              host_load64(slot + kIdentityOffset) != a.identity) fail(a.status, 5);
          if (rows < (Selective ? 1 : -1) || rows == 0 || rows > a.chunk_rows) fail(a.status, 1);
        }
      }
      received_rows[tid] = rows;
    }
    __syncthreads();
    if (tid == 0) error = read_status(a.status);
    __syncthreads();
    if constexpr (Profile) {
      if (tid == 0) {
        for (int peer = 0; peer < a.world; ++peer) {
          if (peer_done[peer]) continue;
          ++profile.values[PReadyPolls];
          if (!received_rows[peer]) {
            ++profile.values[PUnreadyObservations];
            profile.ready_missing = true;
          } else if (profile.values[PFirstReceive] < 0) {
            profile.values[PFirstReceive] = ticks_ns() - profile.values[PBegin];
          }
        }
      }
    }
    if (error) break;
    profile_phase(profile, POther);
    for (int peer = 0; peer < a.world; ++peer) {
      const int rows = received_rows[peer];
      if (!rows) continue;
      char* peer_base = reinterpret_cast<char*>(a.pointers[peer]) + channel * a.channel_stride;
      const int slot_index = Selective ? received_slot[peer] : peer_chunk[peer] % a.slots;
      char* slot = slot_index < 0 ? nullptr :
          peer_base + kArenaHeader + int64_t(slot_index) * a.slot_stride;
      for (int tile = 0; tile < rows; tile += kThreads) {
        profile_phase(profile, PReceiveMetadata);
        const int wire_row = tile + tid;
        bool selected = false;
        int token = 0;
        if (wire_row < rows) {
          uint64_t mask = host_load64(slot + kSlotHeader + int64_t(wire_row) * 8);
          token = int(host_load32(slot + kSlotHeader + int64_t(a.chunk_rows) * 8 + wire_row * 4));
          selected = (mask & (uint64_t(1) << a.rank)) != 0;
          if (selected && (token < 0 || (Combine && token >= a.capacity))) fail(a.status, 2);
        }
        uint64_t prefix, aggregate;
        Scan(scan_storage).ExclusiveSum(uint64_t(selected), prefix, aggregate);
        if (selected) {
          selected_rows[prefix] = wire_row;
          selected_tokens[prefix] = token;
        }
        if (tid == 0) {
          num_selected = int(aggregate);
          output_base = Combine || !num_selected ? 0 : atomicAdd(a.output_count, num_selected);
          owner_base = !a.owner_counts || !num_selected ? 0 : atomicAdd(a.owner_counts + peer, num_selected);
          if (!Combine && (int64_t(output_base) + num_selected > a.capacity ||
              (a.owner_counts && int64_t(owner_base) + num_selected > a.capacity))) fail(a.status, 2);
          error = read_status(a.status);
        }
        __syncthreads();
        if (error) break;
        profile_phase(profile, PReceivePayload);
        if constexpr (Profile) {
          if (tid == 0 && num_selected && profile.values[PFirstPayloadReceive] < 0)
            profile.values[PFirstPayloadReceive] = profile.mark - profile.values[PBegin];
        }
        for (int64_t i = tid; i < int64_t(num_selected) * (a.hidden / 2); i += blockDim.x) {
          int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          const int source = selected_rows[row];
          const uint32_t packed = host_load32(slot + x_offset +
              (int64_t(source) * (a.hidden / 2) + h) * 4);
          if (Combine) {
            auto* dst = reinterpret_cast<uint32_t*>(
                a.remote + (int64_t(peer) * a.capacity + selected_tokens[row]) * a.hidden);
            dst[h] = packed;
          } else reinterpret_cast<uint32_t*>(a.output + int64_t(output_base + row) * a.hidden)[h] = packed;
        }
        if (!Combine) {
          for (int i = tid; i < num_selected * a.topk; i += blockDim.x) {
            const int row = i / a.topk, k = i % a.topk;
            const int64_t wire = int64_t(selected_rows[row]) * a.topk + k;
            const int64_t expert = int64_t(host_load64(slot + id_offset + wire * 8));
            const int64_t out = int64_t(output_base + row) * a.topk + k;
            a.output_ids[out] = expert >= 0 && expert < a.experts && a.expert_rank[expert] == a.rank ? a.expert_local[expert] : -1;
            a.output_weights[out] = __uint_as_float(host_load32(slot + weight_offset + wire * 4));
          }
          if (tid < num_selected) {
            a.output_inverse[int64_t(output_base + tid) * 2] = peer;
            a.output_inverse[int64_t(output_base + tid) * 2 + 1] = selected_tokens[tid];
            if (a.owner_rows)
              a.owner_rows[int64_t(peer) * a.capacity + owner_base + tid] = output_base + tid;
          }
        }
        __syncthreads();
        if constexpr (Profile) {
          if (tid == 0) {
            profile.values[PReadBytes] += int64_t(num_selected) * a.hidden * 2;
            profile.values[PReceiveRows] += num_selected;
          }
        }
      }
      profile_phase(profile, PAck);
      __threadfence_system();
      __syncthreads();
      if (tid == 0) {
        if constexpr (Selective) {
          if (rows == -1) release(dispatch_terminal(a, peer_base) + 1 + a.rank, a.epoch << 32);
          else release(control(slot, 1 + a.rank), received_sequence[peer]);
        } else release(control(slot, 1 + a.rank), (a.epoch << 32) | uint64_t(peer_chunk[peer] + 1));
        ++peer_chunk[peer];
        if (rows == -1) peer_done[peer] = 1;
        if constexpr (Profile) {
          ++profile.values[PReceiveFrames];
          profile.values[PTerminalReceived] += rows == -1;
          profile.progress = true;
        }
      }
      __syncthreads();
      if (error) break;
      profile_phase(profile, POther);
    }
    const int unfinished_peers = __syncthreads_or(tid < a.world && !peer_done[tid]);
    bool unacked = false;
    if (producer_done && (Selective || !unfinished_peers)) {
      for (int i = tid; i < a.slots * a.world; i += blockDim.x) {
        int peer = i % a.world;
        if (peer == a.rank) continue;
        if constexpr (Selective)
          if (!(slot_readers[i / a.world] & (uint64_t(1) << peer))) continue;
        char* slot = own + kArenaHeader + (i / a.world) * a.slot_stride;
        if (acquire(control(slot, 1 + peer)) != acquire(control(slot, 0))) unacked = true;
      }
    }
    int outstanding = __syncthreads_or(unacked);
    if constexpr (Selective) {
      // Publish end-of-operation only after every selected reader has released
      // all payload slots. A separate terminal ACK protects epoch reuse, without
      // making nonreaders part of payload-slot credits.
      if (tid == 0 && producer_done && !outstanding && !terminal_sent) {
        if (a.diagnostics) host_store64(own + 24, a.identity);
        release(dispatch_terminal(a, own), a.epoch << 32);
        terminal_sent = 1;
      }
      __syncthreads();
      const bool terminal_unacked = terminal_sent && tid < a.world && tid != a.rank &&
          acquire(dispatch_terminal(a, own) + 1 + tid) != (a.epoch << 32);
      outstanding |= __syncthreads_or(terminal_unacked);
    }
    if (tid == 0) {
      complete = producer_done && !unfinished_peers && !outstanding;
      if (!complete && check_health && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
      error = read_status(a.status);
    }
    __syncthreads();
    if (complete || error) {
      profile_iteration_end(profile, producer_done && !unfinished_peers && outstanding);
      break;
    }
    __nanosleep(64);
    profile_iteration_end(profile, producer_done && !unfinished_peers && outstanding);
  }
  profile_finish(profile, a.profile, error);
  if (tid == 0 && error) {
    release(reinterpret_cast<uint64_t*>(own), uint64_t(error));
    if (a.fatal_errors || host_load64(own + 16)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}

int64_t channel_bytes(int64_t hidden, int64_t topk, int64_t slots, int64_t rows) {
  const int64_t row = align_up(16 + topk * 12 + hidden * 2, 16);
  // Selective dispatch notification words follow payload slots. The legacy
  // dispatch path ignores the suffix; combine keeps its existing arena layout.
  const int64_t notifications = topk ? (slots * kMaxRanks + kMaxRanks + 1) * 8 : 0;
  return align_up(kArenaHeader + slots * align_up(kSlotHeader + rows * row, 128) + notifications, 4096);
}

// A combine CTA owns both directions of exactly one rank pair. Neither its
// producer cursor nor its credit depends on any other destination. Keeping
// send and receive together, with the entire grid resident, ensures a full
// outbound ring never prevents the inbound ring from being drained.
template <bool Profile = false>
__global__ void host_sm_pair_combine_kernel(TransportArgs a) {
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int cursor, end, error, sent, received, send_done, recv_done;
  __shared__ int selected[kThreads], tokens[kThreads], count, next, rows, complete;
  __shared__ uint64_t start, previous;
  __shared__ ProfileState<Profile> profile;
  profile_init(profile);
  const int tid = threadIdx.x;
  const int channel = blockIdx.x / a.world, peer = blockIdx.x % a.world;
  const bool local = peer == a.rank;
  char* own = reinterpret_cast<char*>(a.pointers[a.rank]) +
      (int64_t(channel) * a.world + peer) * a.channel_stride;
  char* other = reinterpret_cast<char*>(a.pointers[peer]) +
      (int64_t(channel) * a.world + a.rank) * a.channel_stride;
  const int64_t x_offset = payload_offset(a.chunk_rows, 0);
  if (tid == 0) {
    int n = a.owner_counts ? a.owner_counts[peer] : *a.input_count;
    if (n < 0 || n > a.input_rows) { fail(a.status, 1); n = 0; }
    cursor = int(int64_t(n) * channel / a.channels);
    end = int(int64_t(n) * (channel + 1) / a.channels);
    sent = received = send_done = 0;
    recv_done = local;
    start = a.diagnostics ? ticks_ns() : 0;
  }
  __syncthreads();
  uint32_t iteration = 0;
  while (true) {
    const bool check_health = a.diagnostics && (iteration++ & 255u) == 0;
    profile_iteration_begin(profile);
    profile_phase(profile, PPeerPoll);
    if (tid == 0) {
      if (check_health && (acquire(control(own, 0)) || acquire(control(own, 1)) ||
          (!local && (acquire(control(other, 0)) || acquire(control(other, 1)))))) fail(a.status, 4);
      error = read_status(a.status);
    }
    __syncthreads();
    if (error) break;
    char* send_slot = own + kArenaHeader + (sent % a.slots) * a.slot_stride;
    if (tid == 0) {
      previous = local || send_done ? 0 : acquire(control(send_slot, 0));
      // Exactly one consumer owns ACK word 1 in this directed queue.
      complete = !local && !send_done && acquire(control(send_slot, 1)) != previous;
      if constexpr (Profile) {
        profile.credit_missing = complete;
        profile.values[PCreditPolls] += !local && !send_done;
        profile.values[PCreditBlockedObservations] += complete;
      }
    }
    __syncthreads();
    if (!send_done && !complete) {
      profile_phase(profile, PRouteLocal);
      const int limit = local ? kThreads : min(kThreads, a.chunk_rows);
      const int batch = min(a.owner_rows ? limit : kThreads, end - cursor);
      if (a.owner_rows) {
        // Dispatch has already grouped compact row IDs by return owner. Each
        // pair reads its exact contiguous subrange once, with no inverse scan.
        if (tid < batch) {
          const int row = a.owner_rows[int64_t(peer) * a.input_rows + cursor + tid];
          if (row < 0 || row >= a.input_rows) fail(a.status, 1);
          else {
            const int64_t index = a.input_inverse[int64_t(row) * 2 + 1];
            if (index < 0 || index > INT32_MAX ||
                (a.diagnostics && a.input_inverse[int64_t(row) * 2] != peer)) fail(a.status, 1);
            else {
              selected[tid] = row;
              tokens[tid] = int(index);
              if (local && index >= a.capacity) fail(a.status, 2);
            }
          }
        }
        __syncthreads();
        if (tid == 0) {
          count = batch;
          next = cursor + batch;
          error = read_status(a.status);
        }
      } else {
        // Preserve the direct raw API for callers that supply only inverse.
        bool selected_here = false;
        int token = 0;
        if (tid < batch) {
          const int64_t destination = a.input_inverse[int64_t(cursor + tid) * 2];
          if (destination < 0 || destination >= a.world) fail(a.status, 1);
          else if (destination == peer) {
            const int64_t index = a.input_inverse[int64_t(cursor + tid) * 2 + 1];
            if (index < 0 || index > INT32_MAX) fail(a.status, 1);
            else {
              selected_here = true;
              token = int(index);
              if (local && token >= a.capacity) fail(a.status, 2);
            }
          }
        }
        int prefix, total;
        Scan(scan_storage).ExclusiveSum(int(selected_here), prefix, total);
        if (selected_here && prefix < limit) {
          selected[prefix] = cursor + tid;
          tokens[prefix] = token;
        }
        __syncthreads();
        if (tid == 0) {
          count = min(total, limit);
          next = total > limit ? selected[limit - 1] + 1 : cursor + batch;
          error = read_status(a.status);
        }
      }
      __syncthreads();
      if (error) break;
      if (local) {
        for (int64_t i = tid; i < int64_t(count) * (a.hidden / 2); i += blockDim.x) {
          const int r = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          reinterpret_cast<uint32_t*>(a.output + int64_t(tokens[r]) * a.hidden)[h] =
              reinterpret_cast<const uint32_t*>(a.input + int64_t(selected[r]) * a.hidden)[h];
        }
      } else {
        profile_phase(profile, PSendMetadata);
        if (tid < count) host_store32(send_slot + kSlotHeader + tid * 4, tokens[tid]);
        profile_phase(profile, PSendPayload);
        for (int64_t i = tid; i < int64_t(count) * (a.hidden / 2); i += blockDim.x) {
          const int r = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          host_store32(send_slot + x_offset + i * 4,
              reinterpret_cast<const uint32_t*>(a.input + int64_t(selected[r]) * a.hidden)[h]);
        }
      }
      __syncthreads();
      if (tid == 0) {
        cursor = next;
        if constexpr (Profile) {
          profile.progress |= batch != 0;
          profile.values[PLocalRows] += local ? count : 0;
          profile.values[PSendRows] += local ? 0 : count;
          profile.values[PWrittenBytes] += local ? 0 : int64_t(count) * a.hidden * 2;
        }
      }
      __syncthreads();
      profile_phase(profile, PPublish);
      if (local) {
        if (tid == 0) send_done = cursor == end;
      } else if (count || cursor == end) {
        if (tid == 0) {
          host_store32(send_slot + kRowsOffset, count ? count : uint32_t(-1));
          if (a.diagnostics) host_store64(send_slot + kIdentityOffset, a.identity);
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
          release(control(send_slot, 0), (a.epoch << 32) | uint64_t(++sent));
          send_done = !count;
          if constexpr (Profile) {
            const int64_t elapsed = ticks_ns() - profile.values[PBegin];
            if (profile.values[PFirstPublish] < 0) profile.values[PFirstPublish] = elapsed;
            if (count && profile.values[PFirstPayloadPublish] < 0)
              profile.values[PFirstPayloadPublish] = elapsed;
            ++profile.values[PSendFrames];
            profile.values[PTerminalSent] += !count;
            profile.progress = true;
          }
        }
      }
    }
    __syncthreads();
    profile_phase(profile, PPeerPoll);
    char* recv_slot = other + kArenaHeader + (received % a.slots) * a.slot_stride;
    if (tid == 0) {
      rows = 0;
      if (!recv_done) {
        if (acquire(control(recv_slot, 0)) == ((a.epoch << 32) | uint64_t(received + 1))) {
          rows = int32_t(host_load32(recv_slot + kRowsOffset));
          if (a.diagnostics && received == 0 &&
              host_load64(recv_slot + kIdentityOffset) != a.identity) fail(a.status, 5);
          if (rows < -1 || rows == 0 || rows > min(kThreads, a.chunk_rows)) fail(a.status, 1);
        }
        if constexpr (Profile) {
          ++profile.values[PReadyPolls];
          profile.ready_missing = !rows;
          profile.values[PUnreadyObservations] += !rows;
          if (rows && profile.values[PFirstReceive] < 0)
            profile.values[PFirstReceive] = ticks_ns() - profile.values[PBegin];
        }
      }
      error = read_status(a.status);
    }
    __syncthreads();
    if (error) break;
    if (rows) {
      profile_phase(profile, PReceiveMetadata);
      if (tid < rows) {
        const int token = int(host_load32(recv_slot + kSlotHeader + tid * 4));
        tokens[tid] = token;
        if (token < 0 || token >= a.capacity) fail(a.status, 2);
      }
      __syncthreads();
      if (tid == 0) error = read_status(a.status);
      __syncthreads();
      if (error) break;
      profile_phase(profile, PReceivePayload);
      if constexpr (Profile) {
        if (tid == 0 && rows > 0 && profile.values[PFirstPayloadReceive] < 0)
          profile.values[PFirstPayloadReceive] = profile.mark - profile.values[PBegin];
      }
      for (int64_t i = tid; i < int64_t(rows) * (a.hidden / 2); i += blockDim.x) {
        const int r = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
        auto* dst = reinterpret_cast<uint32_t*>(
            a.remote + (int64_t(peer) * a.capacity + tokens[r]) * a.hidden);
        dst[h] = host_load32(recv_slot + x_offset + i * 4);
      }
      profile_phase(profile, PAck);
      // Every reader must finish before publishing the sole consumer's ACK.
      __threadfence_system();
      __syncthreads();
      if (tid == 0) {
        release(control(recv_slot, 1), (a.epoch << 32) | uint64_t(++received));
        recv_done = rows == -1;
        if constexpr (Profile) {
          ++profile.values[PReceiveFrames];
          profile.values[PTerminalReceived] += rows == -1;
          profile.values[PReceiveRows] += max(0, rows);
          profile.values[PReadBytes] += int64_t(max(0, rows)) * a.hidden * 2;
          profile.progress = true;
        }
      }
    }
    __syncthreads();
    profile_phase(profile, POther);
    bool unacked = false;
    if (!local && send_done && recv_done) {
      for (int i = tid; i < a.slots; i += blockDim.x) {
        char* slot = own + kArenaHeader + i * a.slot_stride;
        if (acquire(control(slot, 1)) != acquire(control(slot, 0))) unacked = true;
      }
    }
    const int outstanding = __syncthreads_or(unacked);
    if (tid == 0) {
      complete = send_done && recv_done && !outstanding;
      if (!complete && check_health && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
      error = read_status(a.status);
    }
    __syncthreads();
    if (complete || error) {
      profile_iteration_end(profile, send_done && recv_done && outstanding);
      break;
    }
    __nanosleep(64);
    profile_iteration_end(profile, send_done && recv_done && outstanding);
  }
  profile_finish(profile, a.profile, error);
  if (tid == 0 && error) {
    release(control(own, 0), uint64_t(error));
    if (a.fatal_errors || host_load64(own + 16)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}

// Decode uses bounded, direct-address return mailboxes. A producer owns one
// payload per (destination rank, original token); only that destination writes
// its ACK. The preceding call drains every outgoing ACK before returning on
// the serialized transport stream, so this call never waits for old credit
// before sending. Every resident CTA publishes all its sends before waiting
// for incoming data. No receive can be blocked behind a credit-waiting sender.
__device__ __forceinline__ char* mailbox(TransportArgs a, int producer,
                                       int destination, int token) {
  return reinterpret_cast<char*>(a.mailbox_pointers[producer]) + kArenaHeader +
      (int64_t(destination) * a.mailbox_tokens + token) * a.mailbox_stride;
}

__device__ __forceinline__ int mailbox_health(TransportArgs a, uint64_t start,
                                              uint32_t iteration) {
  if (a.diagnostics && (iteration & 255u) == 0) {
    for (int peer = 0; peer < a.world; ++peer) {
      auto* header = reinterpret_cast<uint64_t*>(a.mailbox_pointers[peer]);
      if (acquire(header) || acquire(header + 1)) fail(a.status, 4);
    }
    if (ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
  }
  return read_status(a.status);
}

__global__ void host_sm_mailbox_combine_kernel(TransportArgs a) {
  __shared__ int begin, end, row, token, error, local_row, blocked;
  __shared__ uint64_t readers;
  const int tid = threadIdx.x, channel = blockIdx.x;
  const uint64_t start = a.diagnostics ? ticks_ns() : 0;
  if (tid == 0) error = read_status(a.status);
  __syncthreads();

  // No intermediate HBM copy or capacity-sized clear. All sends are bounded
  // by dispatch's owner list; each payload is written exactly once.
  for (int peer = 0; peer < a.world && !error; ++peer) {
    if (peer == a.rank) continue;
    if (tid == 0) {
      int n = a.owner_counts[peer];
      if (n < 0 || n > a.input_rows) { fail(a.status, 1); n = 0; }
      begin = int(int64_t(n) * channel / a.channels);
      end = int(int64_t(n) * (channel + 1) / a.channels);
      error = read_status(a.status);
    }
    __syncthreads();
    for (int index = begin; index < end && !error; ++index) {
      if (tid == 0) {
        row = a.owner_rows[int64_t(peer) * a.input_rows + index];
        if (row < 0 || row >= a.input_rows) fail(a.status, 1);
        else {
          const int64_t original = a.input_inverse[int64_t(row) * 2 + 1];
          if (original < 0 || original >= a.mailbox_tokens) fail(a.status, 2);
          else token = int(original);
          if (a.diagnostics && a.input_inverse[int64_t(row) * 2] != peer) fail(a.status, 1);
        }
        error = read_status(a.status);
      }
      __syncthreads();
      if (error) break;
      char* box = mailbox(a, a.rank, peer, token);
      const auto* source = reinterpret_cast<const uint32_t*>(a.input + int64_t(row) * a.hidden);
      for (int h = tid; h < a.hidden / 2; h += blockDim.x)
        host_store32(box + 32 + int64_t(h) * 4, source[h]);
      // All payload writers publish their system writes before READY's owner.
      __threadfence_system();
      __syncthreads();
      if (tid == 0) {
        if (a.diagnostics) host_store64(box + 16, a.identity);
        release(control(box, 0), a.epoch);
      }
      __syncthreads();
    }
  }

  // A token CTA waits only for its actual remote contributors, then performs
  // the exact legacy arithmetic: ascending-rank FP32 sum -> BF16 -> add local
  // BF16 -> BF16. The local compact row is provided by dispatch.
  const int first = int(int64_t(a.capacity) * channel / a.channels);
  const int last = int(int64_t(a.capacity) * (channel + 1) / a.channels);
  for (int original = first; original < last && !error; ++original) {
    if (tid == 0) {
      readers = uint64_t(a.expected_ranks[original]);
      local_row = a.local_rows[original];
      if ((readers >> a.world) || local_row < -1 || local_row >= a.input_rows ||
          bool(readers & (uint64_t(1) << a.rank)) != (local_row >= 0)) fail(a.status, 1);
      error = read_status(a.status);
    }
    __syncthreads();
    uint32_t iteration = 0;
    do {
      bool missing = false;
      if (tid < a.world && tid != a.rank && (readers & (uint64_t(1) << tid)))
        missing = acquire(control(mailbox(a, tid, a.rank, original), 0)) != a.epoch;
      const int waiting = __syncthreads_or(missing);
      if (tid == 0) {
        blocked = waiting;
        error = mailbox_health(a, start, iteration++);
      }
      __syncthreads();
      if (blocked && !error) __nanosleep(64);
    } while (blocked && !error);
    if (error) break;
    if (a.diagnostics && tid < a.world && tid != a.rank &&
        (readers & (uint64_t(1) << tid)) &&
        host_load64(mailbox(a, tid, a.rank, original) + 16) != a.identity) fail(a.status, 5);
    __syncthreads();
    if (tid == 0) error = read_status(a.status);
    __syncthreads();
    if (error) break;
    for (int h = tid; h < a.hidden / 2; h += blockDim.x) {
      float sum0 = 0.0f, sum1 = 0.0f;
      for (int peer = 0; peer < a.world; ++peer) {
        if (peer == a.rank) continue;
        uint32_t bits = 0;
        if (readers & (uint64_t(1) << peer))
          bits = host_load32(mailbox(a, peer, a.rank, original) + 32 + int64_t(h) * 4);
        const auto packed = *reinterpret_cast<const __nv_bfloat162*>(&bits);
        const float2 value = __bfloat1622float2(packed);
        sum0 += value.x;
        sum1 += value.y;
      }
      const at::BFloat16 rounded0(sum0), rounded1(sum1);
      const int64_t offset = int64_t(local_row) * a.hidden + h * 2;
      const float local0 = local_row >= 0 ? float(a.input[offset]) : 0.0f;
      const float local1 = local_row >= 0 ? float(a.input[offset + 1]) : 0.0f;
      a.output[int64_t(original) * a.hidden + h * 2] = at::BFloat16(float(rounded0) + local0);
      a.output[int64_t(original) * a.hidden + h * 2 + 1] = at::BFloat16(float(rounded1) + local1);
    }
    // ACK follows every reader's final host load, allowing safe next-call reuse.
    __threadfence_system();
    __syncthreads();
    if (tid < a.world && tid != a.rank && (readers & (uint64_t(1) << tid)))
      release(control(mailbox(a, tid, a.rank, original), 1), a.epoch);
    __syncthreads();
  }

  // Each CTA drains only the rows it published. There is no terminal frame or
  // all-rank ACK: a rank with no incoming/outgoing rows can finish immediately.
  for (int peer = 0; peer < a.world && !error; ++peer) {
    if (peer == a.rank) continue;
    const int n = a.owner_counts[peer];  // Bounds were checked in the send phase.
    const int first_send = int(int64_t(n) * channel / a.channels);
    const int last_send = int(int64_t(n) * (channel + 1) / a.channels);
    for (int index = first_send; index < last_send && !error; ++index) {
      if (tid == 0) {
        const int sent_row = a.owner_rows[int64_t(peer) * a.input_rows + index];
        token = int(a.input_inverse[int64_t(sent_row) * 2 + 1]);
      }
      __syncthreads();
      uint32_t iteration = 0;
      do {
        if (tid == 0) {
          blocked = acquire(control(mailbox(a, a.rank, peer, token), 1)) != a.epoch;
          error = mailbox_health(a, start, iteration++);
        }
        __syncthreads();
        if (blocked && !error) __nanosleep(64);
      } while (blocked && !error);
    }
  }
  if (tid == 0 && error) {
    auto* own = reinterpret_cast<uint64_t*>(a.mailbox_pointers[a.rank]);
    release(own, uint64_t(error));
    if (a.fatal_errors || acquire(own + 2)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}

int max_channels_current_device(int world = 1) {
  // Capability is fixed for this extension/device. Validate direct native
  // calls without repeating attribute and occupancy queries on every layer.
  static std::mutex mutex;
  static std::map<std::pair<int, int>, int> capacities;
  int device, cooperative, sms;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_pair(device, world);
  const auto cached = capacities.find(key);
  if (cached != capacities.end()) return cached->second;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  if (!cooperative) { capacities.emplace(key, 0); return 0; }
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int a, b, c, diagnostic, selective_a, selective_b, mailbox_capacity;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a, host_sm_transport_kernel<false, int32_t>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, host_sm_transport_kernel<false, int64_t>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c, host_sm_pair_combine_kernel<false>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&diagnostic, host_sm_pair_combine_kernel<true>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&selective_a, host_sm_transport_kernel<false, int32_t, false, true>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&selective_b, host_sm_transport_kernel<false, int64_t, false, true>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mailbox_capacity, host_sm_mailbox_combine_kernel, kThreads, 0));
  const int capacity = std::min({32, sms * std::min({a, b, selective_a, selective_b, mailbox_capacity}),
                                 sms * std::min(c, diagnostic) / world});
  capacities.emplace(key, capacity);
  return capacity;
}

template <bool Combine, typename Id, bool Profile = false, bool Selective = false>
void launch_transport(TransportArgs args, cudaStream_t stream) {
  const int64_t elements = int64_t(args.capacity) * args.hidden * (Combine ? args.world : 1);
  const int blocks = int(std::max<int64_t>(1, std::min<int64_t>(4096, (elements + kThreads - 1) / kThreads)));
  host_sm_init_kernel<Combine><<<blocks, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int lanes = args.channels * (Combine ? args.world : 1);
  if (lanes == 1) {
    if constexpr (Combine) host_sm_pair_combine_kernel<Profile><<<1, kThreads, 0, stream>>>(args);
    else host_sm_transport_kernel<false, Id, Profile, Selective><<<1, kThreads, 0, stream>>>(args);
  } else {
    void* parameters[] = {&args};
    const void* kernel;
    if constexpr (Combine) kernel = reinterpret_cast<const void*>(host_sm_pair_combine_kernel<Profile>);
    else kernel = reinterpret_cast<const void*>(host_sm_transport_kernel<false, Id, Profile, Selective>);
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        kernel, dim3(lanes), dim3(kThreads), parameters, 0, stream));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Fatal inference contexts terminate on a device error; zeroing count after
  // an error cannot be observed safely and needs no normal-path kernel launch.
  if (Combine || !args.fatal_errors) {
    host_sm_finalize_kernel<Combine><<<Combine ? blocks : 1, kThreads, 0, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void tensor_check(const torch::Tensor& value, at::ScalarType dtype, int dimensions,
                  const torch::Device& device, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.device() == device && value.scalar_type() == dtype &&
              value.dim() == dimensions && value.is_contiguous(), name, " has an invalid device, dtype or layout");
}
void common_check(torch::Tensor input, torch::Tensor pointers, int64_t rank,
                  int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
                  torch::Tensor output, torch::Tensor status, int64_t channels, bool combine = false) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous CUDA BF16 [rows, hidden]");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % 2 == 0 && input.size(1) <= INT32_MAX && input.size(0) <= INT32_MAX,
              "hidden must be positive/even and dimensions must fit int32");
  tensor_check(pointers, at::kLong, 1, input.device(), "arena pointers");
  TORCH_CHECK(pointers.numel() > 0 && pointers.numel() <= kMaxRanks && rank >= 0 && rank < pointers.numel(), "invalid rank/world");
  TORCH_CHECK(epoch > 0 && epoch < (int64_t(1) << 31) && slots > 0 && slots <= 64 && chunk_rows > 0 && chunk_rows <= 65536,
              "invalid epoch, slots or chunk_rows");
  TORCH_CHECK(timeout_ms > 0 && timeout_ms <= 3600000, "invalid timeout");
  tensor_check(output, at::kBFloat16, 2, input.device(), "output");
  TORCH_CHECK(output.size(1) == input.size(1) && output.size(0) <= INT32_MAX, "invalid output shape");
  tensor_check(status, at::kInt, 1, input.device(), "status");
  TORCH_CHECK(status.numel() == 1, "status must contain one element");
  const c10::cuda::CUDAGuard guard(input.device());
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream(), &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
              "host_sm does not support CUDA Graph capture or replay");
  TORCH_CHECK(channels >= 1 && channels <= 32, "invalid channel count");
  const int world = combine ? int(pointers.numel()) : 1;
  TORCH_CHECK((channels == 1 && world == 1) || channels <= max_channels_current_device(world),
              "requested channels require cooperative all-resident launch support and capacity");
}

void operation_check(const torch::Tensor& input, const torch::Tensor& output,
                     const torch::Device& device, int hidden, int64_t epoch) {
  tensor_check(input, at::kBFloat16, 2, device, "input");
  tensor_check(output, at::kBFloat16, 2, device, "output");
  TORCH_CHECK(input.size(1) == hidden && output.size(1) == hidden &&
                  input.size(0) <= INT32_MAX && output.size(0) <= INT32_MAX,
              "invalid input or output shape for prepared transport");
  TORCH_CHECK(epoch > 0 && epoch < (int64_t(1) << 31), "invalid epoch");
}

void owner_check(const torch::Tensor& rows, const torch::Tensor& counts,
                 const torch::Device& device, int world, int64_t capacity) {
  tensor_check(rows, at::kInt, 2, device, "owner row indices");
  tensor_check(counts, at::kInt, 1, device, "owner row counts");
  TORCH_CHECK(rows.size(0) == world && rows.size(1) == capacity && counts.numel() == world,
              "invalid owner row-index shape");
}

void active_channel_check(TransportArgs& a, int64_t requested) {
  if (!requested) return;
  TORCH_CHECK(requested >= 1 && requested <= a.channels,
              "active_channels must not exceed the prepared channel allocation");
  a.channels = int(requested);
}

cudaStream_t eager_stream(const torch::Device& device) {
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
              "host_sm does not support CUDA Graph capture or replay");
  return stream;
}
}  // namespace

struct PreparedHostSM::Impl {
  // Keep all static device addresses alive. Bootstrap tensors are immutable for
  // this object's lifetime; contexts own the underlying registered host arenas.
  std::vector<torch::Tensor> retained;
  torch::Device device = torch::Device(torch::kCPU);
  TransportArgs dispatch{};
  TransportArgs combine{};
  uint64_t last_mailbox_epoch = 0;
  cudaStream_t mailbox_stream = nullptr;
};

PreparedHostSM::PreparedHostSM(torch::Tensor expert_rank, torch::Tensor expert_local,
    torch::Tensor dispatch_pointers, torch::Tensor combine_pointers,
    torch::Tensor status, int64_t rank, int64_t hidden, int64_t topk,
    int64_t slots, int64_t dispatch_rows, int64_t combine_rows,
    int64_t timeout_ms, int64_t channels, bool diagnostics, bool fatal_errors,
    c10::optional<torch::Tensor> mailbox_pointers, int64_t mailbox_tokens) {
  TORCH_CHECK(status.is_cuda(), "status must be a CUDA tensor");
  const auto device = status.device();
  tensor_check(status, at::kInt, 1, device, "status");
  TORCH_CHECK(status.numel() == 1, "status must contain one element");
  tensor_check(dispatch_pointers, at::kLong, 1, device, "dispatch arena pointers");
  tensor_check(combine_pointers, at::kLong, 1, device, "combine arena pointers");
  const int64_t world = dispatch_pointers.numel();
  TORCH_CHECK(world > 0 && world <= kMaxRanks && rank >= 0 && rank < world &&
                  combine_pointers.numel() == world, "invalid rank/world");
  TORCH_CHECK(hidden > 0 && hidden <= INT32_MAX && hidden % 2 == 0 &&
                  topk > 0 && topk <= 256, "invalid hidden/topk");
  TORCH_CHECK(slots > 0 && slots <= 64 && dispatch_rows > 0 && dispatch_rows <= 65536 &&
                  combine_rows > 0 && combine_rows <= 65536, "invalid queue dimensions");
  TORCH_CHECK(timeout_ms > 0 && timeout_ms <= 3600000, "invalid timeout");
  TORCH_CHECK(channels >= 1 && channels <= 32, "invalid channel count");
  tensor_check(expert_rank, at::kInt, 1, device, "expert rank");
  tensor_check(expert_local, at::kInt, 1, device, "expert local");
  TORCH_CHECK(expert_rank.numel() > 0 && expert_rank.numel() <= INT32_MAX &&
                  expert_rank.sizes() == expert_local.sizes(), "invalid expert placement shape");
  const c10::cuda::CUDAGuard guard(device);
  eager_stream(device);
  TORCH_CHECK((channels == 1 && world == 1) || channels <= max_channels_current_device(int(world)),
              "requested channels require cooperative all-resident launch support and capacity");
  // This is the only host read of placement and arena tables. Dynamic route
  // expert indices remain bounds checked on the GPU before table access.
  const auto ranks_cpu = expert_rank.cpu();
  const auto locals_cpu = expert_local.cpu();
  for (int64_t i = 0; i < expert_rank.numel(); ++i) {
    TORCH_CHECK(ranks_cpu.data_ptr<int32_t>()[i] >= 0 &&
                    ranks_cpu.data_ptr<int32_t>()[i] < world &&
                    locals_cpu.data_ptr<int32_t>()[i] >= 0, "invalid expert placement entry");
  }
  for (const auto& pointers : {dispatch_pointers, combine_pointers}) {
    const auto cpu = pointers.cpu();
    for (int64_t i = 0; i < world; ++i) {
      const auto address = uint64_t(cpu.data_ptr<int64_t>()[i]);
      TORCH_CHECK(address != 0 && address % 8 == 0, "invalid mapped arena address");
    }
  }
  TORCH_CHECK(mailbox_pointers.has_value() == (mailbox_tokens > 0) &&
                  mailbox_tokens >= 0 && mailbox_tokens <= 65536, "invalid mailbox configuration");
  if (mailbox_pointers.has_value()) {
    tensor_check(*mailbox_pointers, at::kLong, 1, device, "mailbox arena pointers");
    TORCH_CHECK(mailbox_pointers->numel() == world, "invalid mailbox pointer count");
    const auto cpu = mailbox_pointers->cpu();
    for (int64_t i = 0; i < world; ++i) {
      const auto address = uint64_t(cpu.data_ptr<int64_t>()[i]);
      TORCH_CHECK(address != 0 && address % 8 == 0, "invalid mapped mailbox address");
    }
  }
  impl_ = std::make_shared<Impl>();
  impl_->retained = {expert_rank, expert_local, dispatch_pointers, combine_pointers, status};
  if (mailbox_pointers.has_value()) impl_->retained.push_back(*mailbox_pointers);
  impl_->device = device;
  for (bool combine : {false, true}) {
    auto& a = combine ? impl_->combine : impl_->dispatch;
    a.pointers = (combine ? combine_pointers : dispatch_pointers).data_ptr<int64_t>();
    a.mailbox_pointers = mailbox_pointers.has_value() ? mailbox_pointers->data_ptr<int64_t>() : nullptr;
    a.mailbox_tokens = int(mailbox_tokens);
    a.mailbox_stride = align_up(32 + hidden * 2, 128);
    a.expert_rank = expert_rank.data_ptr<int32_t>();
    a.expert_local = expert_local.data_ptr<int32_t>();
    a.status = status.data_ptr<int32_t>();
    a.experts = int(expert_rank.numel());
    a.world = int(world); a.rank = int(rank); a.hidden = int(hidden);
    a.topk = combine ? 0 : int(topk); a.slots = int(slots);
    a.chunk_rows = int(combine ? combine_rows : dispatch_rows); a.channels = int(channels);
    const int64_t row = align_up(16 + int64_t(a.topk) * 12 + hidden * 2, 16);
    a.slot_stride = align_up(kSlotHeader + int64_t(a.chunk_rows) * row, 128);
    a.channel_stride = channel_bytes(hidden, a.topk, slots, a.chunk_rows);
    a.timeout_ns = uint64_t(timeout_ms) * 1000000;
    a.diagnostics = diagnostics; a.trusted_placement = true; a.fatal_errors = fatal_errors;
  }
}

void PreparedHostSM::dispatch(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
    int64_t epoch, torch::Tensor output, torch::Tensor output_ids,
    torch::Tensor output_weights, torch::Tensor inverse, torch::Tensor count,
    torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t active_channels,
    int64_t identity, c10::optional<torch::Tensor> expected_ranks,
    c10::optional<torch::Tensor> local_rows) {
  auto a = impl_->dispatch;
  active_channel_check(a, active_channels);
  const auto device = impl_->device;
  operation_check(input, output, device, a.hidden, epoch);
  TORCH_CHECK(identity >= 0, "invalid operation identity");
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong,
              "ids must have int32 or int64 dtype");
  tensor_check(ids, ids.scalar_type(), 2, device, "ids");
  tensor_check(weights, at::kFloat, 2, device, "weights");
  TORCH_CHECK(ids.size(0) == input.size(0) && ids.size(1) == a.topk &&
                  ids.sizes() == weights.sizes(), "invalid routing shape");
  tensor_check(output_ids, at::kLong, 2, device, "output ids");
  tensor_check(output_weights, at::kFloat, 2, device, "output weights");
  TORCH_CHECK(output_ids.size(0) == output.size(0) && output_ids.size(1) == a.topk &&
                  output_ids.sizes() == output_weights.sizes(), "invalid output routing shape");
  tensor_check(inverse, at::kLong, 2, device, "inverse");
  TORCH_CHECK(inverse.size(0) == output.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, device, "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  owner_check(owner_rows, owner_counts, device, a.world, output.size(0));
  TORCH_CHECK(expected_ranks.has_value() == local_rows.has_value(), "both fused routing tensors are required");
  if (expected_ranks.has_value()) {
    TORCH_CHECK(a.mailbox_pointers && input.size(0) <= a.mailbox_tokens, "input exceeds prepared mailbox capacity");
    tensor_check(*expected_ranks, at::kLong, 1, device, "expected rank masks");
    tensor_check(*local_rows, at::kInt, 1, device, "local compact rows");
    TORCH_CHECK(expected_ranks->numel() == input.size(0) && local_rows->numel() == input.size(0),
                "invalid fused routing shape");
    a.expected_ranks = expected_ranks->data_ptr<int64_t>();
    a.local_rows = local_rows->data_ptr<int32_t>();
  }
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = eager_stream(device);
  a.input = input.data_ptr<at::BFloat16>(); a.ids = ids.data_ptr();
  a.weights = weights.data_ptr<float>(); a.output = output.data_ptr<at::BFloat16>();
  a.output_ids = output_ids.data_ptr<int64_t>(); a.output_weights = output_weights.data_ptr<float>();
  a.output_inverse = inverse.data_ptr<int64_t>(); a.output_count = count.data_ptr<int32_t>();
  a.owner_rows = owner_rows.data_ptr<int32_t>(); a.owner_counts = owner_counts.data_ptr<int32_t>();
  a.input_rows = int(input.size(0)); a.capacity = int(output.size(0));
  a.epoch = uint64_t(epoch); a.identity = uint64_t(identity);
  if (ids.scalar_type() == at::kInt) launch_transport<false, int32_t, false, true>(a, stream);
  else launch_transport<false, int64_t, false, true>(a, stream);
}

void PreparedHostSM::combine(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
    torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t epoch,
    torch::Tensor output, c10::optional<torch::Tensor> remote, int64_t active_channels, int64_t identity,
    c10::optional<torch::Tensor> profile, c10::optional<torch::Tensor> expected_ranks,
    c10::optional<torch::Tensor> local_rows) {
  auto a = impl_->combine;
  active_channel_check(a, active_channels);
  const auto device = impl_->device;
  operation_check(partial, output, device, a.hidden, epoch);
  TORCH_CHECK(identity >= 0, "invalid operation identity");
  tensor_check(inverse, at::kLong, 2, device, "inverse");
  TORCH_CHECK(inverse.size(0) == partial.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, device, "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  TORCH_CHECK(expected_ranks.has_value() == local_rows.has_value(), "both fused routing tensors are required");
  const bool fused = expected_ranks.has_value();
  if (fused) {
    TORCH_CHECK(a.mailbox_pointers && output.size(0) <= a.mailbox_tokens,
                "output exceeds prepared mailbox capacity");
    TORCH_CHECK(!profile.has_value(), "mailbox combine does not support phase profiling");
    TORCH_CHECK(uint64_t(epoch) > impl_->last_mailbox_epoch,
                "mailbox combine epochs must increase on the serialized transport stream");
    tensor_check(*expected_ranks, at::kLong, 1, device, "expected rank masks");
    tensor_check(*local_rows, at::kInt, 1, device, "local compact rows");
    TORCH_CHECK(expected_ranks->numel() == output.size(0) && local_rows->numel() == output.size(0),
                "invalid fused routing shape");
    a.expected_ranks = expected_ranks->data_ptr<int64_t>();
    a.local_rows = local_rows->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(remote.has_value(), "ring combine requires remote accumulation storage");
    tensor_check(*remote, at::kBFloat16, 3, device, "remote accumulation");
    TORCH_CHECK(remote->size(0) == a.world && remote->size(1) == output.size(0) &&
                    remote->size(2) == a.hidden, "invalid accumulation shape");
    a.remote = remote->data_ptr<at::BFloat16>();
  }
  if (profile.has_value()) {
    tensor_check(*profile, at::kLong, 2, device, "profile");
    TORCH_CHECK(profile->size(0) == a.channels * a.world && profile->size(1) == PFieldCount,
                "profile must have shape [channels * world, ", PFieldCount, "]");
  }
  owner_check(owner_rows, owner_counts, device, a.world, partial.size(0));
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = eager_stream(device);
  a.input = partial.data_ptr<at::BFloat16>(); a.input_inverse = inverse.data_ptr<int64_t>();
  a.input_count = count.data_ptr<int32_t>(); a.output = output.data_ptr<at::BFloat16>();
  a.profile = profile.has_value() ? profile->data_ptr<int64_t>() : nullptr;
  a.owner_rows = owner_rows.data_ptr<int32_t>(); a.owner_counts = owner_counts.data_ptr<int32_t>();
  a.input_rows = int(partial.size(0)); a.capacity = int(output.size(0));
  a.epoch = uint64_t(epoch); a.identity = uint64_t(identity);
  if (fused) {
    TORCH_CHECK(!impl_->last_mailbox_epoch || impl_->mailbox_stream == stream,
                "mailbox combine must use one serialized CUDA stream");
    // No init/finalizer and no world-by-token device scratch for this path.
    if (a.channels == 1) host_sm_mailbox_combine_kernel<<<1, kThreads, 0, stream>>>(a);
    else {
      void* parameters[] = {&a};
      C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
          reinterpret_cast<const void*>(host_sm_mailbox_combine_kernel),
          dim3(a.channels), dim3(kThreads), parameters, 0, stream));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    impl_->last_mailbox_epoch = uint64_t(epoch);
    impl_->mailbox_stream = stream;
  } else if (profile.has_value()) launch_transport<true, int64_t, true>(a, stream);
  else launch_transport<true, int64_t>(a, stream);
}

int64_t host_sm_mailbox_bytes(int64_t hidden, int64_t tokens, int64_t world) {
  TORCH_CHECK(hidden > 0 && hidden <= INT32_MAX && hidden % 2 == 0 &&
                  tokens > 0 && tokens <= 65536 && world > 0 && world <= kMaxRanks,
              "invalid mailbox dimensions");
  return align_up(kArenaHeader + world * tokens * align_up(32 + hidden * 2, 128), 4096);
}

int64_t host_sm_max_channels(int64_t world) {
  TORCH_CHECK(world >= 1 && world <= kMaxRanks, "invalid queue peer count");
  return max_channels_current_device(int(world));
}

std::vector<std::string> host_sm_profile_fields() {
  TORCH_INTERNAL_ASSERT(kProfileFields.size() == PFieldCount);
  return kProfileFields;
}

int64_t host_sm_arena_bytes(int64_t hidden, int64_t topk, int64_t slots,
                            int64_t chunk_rows, int64_t channels, int64_t queue_peers) {
  TORCH_CHECK(hidden > 0 && hidden <= INT32_MAX && hidden % 2 == 0 && topk >= 0 && topk <= 256 &&
              slots > 0 && slots <= 64 && chunk_rows > 0 && chunk_rows <= 65536 &&
              channels > 0 && channels <= 32 && queue_peers > 0 && queue_peers <= kMaxRanks,
              "invalid arena dimensions or channels");
  return channels * queue_peers * channel_bytes(hidden, topk, slots, chunk_rows);
}

void host_sm_dispatch_cuda(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
    torch::Tensor expert_rank, torch::Tensor expert_local, torch::Tensor pointers,
    int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
    torch::Tensor output, torch::Tensor output_ids, torch::Tensor output_weights,
    torch::Tensor inverse, torch::Tensor count, torch::Tensor status, int64_t identity,
    int64_t channels) {
  common_check(input, pointers, rank, epoch, slots, chunk_rows, timeout_ms, output, status, channels);
  const auto device = input.device();
  TORCH_CHECK(ids.scalar_type() == at::kLong || ids.scalar_type() == at::kInt,
              "ids must have int32 or int64 dtype");
  tensor_check(ids, ids.scalar_type(), 2, device, "ids");
  tensor_check(weights, at::kFloat, 2, device, "weights");
  TORCH_CHECK(ids.sizes() == weights.sizes() && ids.size(0) == input.size(0) && ids.size(1) > 0 && ids.size(1) <= 256, "invalid routing shape");
  tensor_check(expert_rank, at::kInt, 1, device, "expert rank");
  tensor_check(expert_local, at::kInt, 1, device, "expert local");
  TORCH_CHECK(expert_rank.numel() > 0 && expert_rank.numel() <= INT32_MAX && expert_rank.sizes() == expert_local.sizes(), "invalid expert placement shape");
  tensor_check(output_ids, at::kLong, 2, device, "output ids");
  tensor_check(output_weights, at::kFloat, 2, device, "output weights");
  TORCH_CHECK(output_ids.size(0) == output.size(0) && output_ids.size(1) == ids.size(1) && output_ids.sizes() == output_weights.sizes(), "invalid output routing shape");
  tensor_check(inverse, at::kLong, 2, device, "inverse");
  TORCH_CHECK(inverse.size(0) == output.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, device, "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  const c10::cuda::CUDAGuard guard(device);
  TransportArgs a{};
  a.input = input.data_ptr<at::BFloat16>();
  a.ids = ids.data_ptr();
  a.weights = weights.data_ptr<float>();
  a.expert_rank = expert_rank.data_ptr<int32_t>();
  a.expert_local = expert_local.data_ptr<int32_t>();
  a.pointers = pointers.data_ptr<int64_t>();
  a.output = output.data_ptr<at::BFloat16>();
  a.output_ids = output_ids.data_ptr<int64_t>();
  a.output_weights = output_weights.data_ptr<float>();
  a.output_inverse = inverse.data_ptr<int64_t>();
  a.output_count = count.data_ptr<int32_t>();
  a.status = status.data_ptr<int32_t>();
  a.experts = expert_rank.numel(); a.world = pointers.numel(); a.rank = rank;
  a.slots = slots; a.chunk_rows = chunk_rows; a.channels = channels;
  a.input_rows = input.size(0); a.hidden = input.size(1); a.topk = ids.size(1); a.capacity = output.size(0);
  const int64_t row = align_up(16 + a.topk * 12 + int64_t(a.hidden) * 2, 16);
  a.slot_stride = align_up(kSlotHeader + chunk_rows * row, 128);
  a.channel_stride = channel_bytes(a.hidden, a.topk, slots, chunk_rows);
  a.epoch = epoch; a.timeout_ns = uint64_t(timeout_ms) * 1000000; a.identity = identity;
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  if (ids.scalar_type() == at::kInt) launch_transport<false, int32_t>(a, stream.stream());
  else launch_transport<false, int64_t>(a, stream.stream());
}

void host_sm_combine_cuda(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
    torch::Tensor pointers, int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows,
    int64_t timeout_ms, torch::Tensor output, torch::Tensor remote, torch::Tensor status,
    int64_t identity, int64_t channels, c10::optional<torch::Tensor> profile) {
  common_check(partial, pointers, rank, epoch, slots, chunk_rows, timeout_ms, output, status, channels, true);
  const c10::cuda::CUDAGuard guard(partial.device());
  tensor_check(inverse, at::kLong, 2, partial.device(), "inverse");
  TORCH_CHECK(inverse.size(0) == partial.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, partial.device(), "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  tensor_check(remote, at::kBFloat16, 3, partial.device(), "remote accumulation");
  TORCH_CHECK(remote.size(0) == pointers.numel() && remote.size(1) == output.size(0) &&
              remote.size(2) == output.size(1), "invalid accumulation shape");
  if (profile.has_value()) {
    tensor_check(*profile, at::kLong, 2, partial.device(), "profile");
    TORCH_CHECK(profile->size(0) == channels * pointers.numel() && profile->size(1) == PFieldCount,
                "profile must have shape [channels * world, ", PFieldCount, "]");
  }
  TransportArgs a{};
  a.input = partial.data_ptr<at::BFloat16>();
  a.input_inverse = inverse.data_ptr<int64_t>();
  a.input_count = count.data_ptr<int32_t>();
  a.pointers = pointers.data_ptr<int64_t>();
  a.output = output.data_ptr<at::BFloat16>();
  a.remote = remote.data_ptr<at::BFloat16>();
  a.status = status.data_ptr<int32_t>();
  a.profile = profile.has_value() ? profile->data_ptr<int64_t>() : nullptr;
  a.world = pointers.numel(); a.rank = rank;
  a.slots = slots; a.chunk_rows = chunk_rows; a.channels = channels;
  a.input_rows = partial.size(0); a.hidden = partial.size(1); a.topk = 0; a.capacity = output.size(0);
  const int64_t row = align_up(16 + int64_t(a.hidden) * 2, 16);
  a.slot_stride = align_up(kSlotHeader + chunk_rows * row, 128);
  a.channel_stride = channel_bytes(a.hidden, 0, slots, chunk_rows);
  a.epoch = epoch; a.timeout_ns = uint64_t(timeout_ms) * 1000000; a.identity = identity;
  const auto stream = c10::cuda::getCurrentCUDAStream(partial.get_device());
  if (profile.has_value()) launch_transport<true, int64_t, true>(a, stream.stream());
  else launch_transport<true, int64_t>(a, stream.stream());
}
