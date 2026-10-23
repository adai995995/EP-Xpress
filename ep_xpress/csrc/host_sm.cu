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
#include <cub/block/block_scan.cuh>

namespace {
constexpr int kThreads = 256;
constexpr int kMaxRanks = 63;
constexpr int64_t kArenaHeader = 128;
constexpr int64_t kSlotHeader = 640;
constexpr int64_t kRowsOffset = 512;
constexpr int64_t kIdentityOffset = 520;

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
  const int64_t* input_inverse;
  const int32_t* input_count;
  at::BFloat16* output;
  int64_t* output_ids;
  float* output_weights;
  int64_t* output_inverse;
  int32_t* output_count;
  at::BFloat16* remote;
  int32_t* status;
  int experts, world, rank, slots, chunk_rows, channels;
  int input_rows, hidden, topk, capacity;
  int64_t slot_stride, channel_stride;
  uint64_t epoch, timeout_ns, identity;
};

// These atomics address this GPU's VRAM only. Host synchronization words retain
// exactly one GPU writer per producer/channel or consumer/channel.
__device__ __forceinline__ void fail(int32_t* status, int code) {
  atomicCAS(status, 0, code);
}

// Separate ordinary grids keep initialization and final reduction from being
// serialized through the small persistent communication grid.
template <bool Combine>
__global__ void host_sm_init_kernel(TransportArgs a) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  if (!Combine && i == 0) *a.output_count = 0;
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
template <bool Combine, typename Id>
__global__ void host_sm_transport_kernel(TransportArgs a) {
  using Scan = cub::BlockScan<uint64_t, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int cursor, end, error, producer_done, sent_chunks;
  __shared__ int peer_done[kMaxRanks], peer_chunk[kMaxRanks], received_rows[kMaxRanks];
  __shared__ int local_rows[kThreads], remote_rows[kThreads], selected_rows[kThreads];
  __shared__ int tokens[kThreads], selected_tokens[kThreads];
  __shared__ uint64_t masks[kThreads];
  __shared__ int num_local, num_remote, num_selected, output_base, complete;
  __shared__ uint64_t previous, start;
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
    producer_done = sent_chunks = 0;
    start = ticks_ns();
  }
  for (int peer = tid; peer < a.world; peer += blockDim.x) {
    peer_done[peer] = peer == a.rank;
    peer_chunk[peer] = 0;
  }
  __syncthreads();
  while (true) {
    // Check different peers concurrently; never spin on a blocked producer.
    if (tid < a.world) {
      auto* p = reinterpret_cast<uint64_t*>(a.pointers[tid] + channel * a.channel_stride);
      if (acquire(p) || acquire(p + 1)) fail(a.status, 4);
    }
    __syncthreads();
    if (tid == 0) error = atomicAdd(a.status, 0);
    __syncthreads();
    if (error) break;
    char* send_slot = own + kArenaHeader + (sent_chunks % a.slots) * a.slot_stride;
    if (tid == 0) previous = producer_done ? 0 : acquire(control(send_slot, 0));
    __syncthreads();
    bool blocked = !producer_done && tid < a.world && tid != a.rank &&
                   acquire(control(send_slot, 1 + tid)) != previous;
    const int any_blocked = __syncthreads_or(blocked);
    if (!producer_done && !any_blocked) {
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
            if (expert < 0 || expert >= a.experts || !isfinite(a.weights[int64_t(row) * a.topk + k])) {
              fail(a.status, 1); continue;
            }
            const int destination = a.expert_rank[expert];
            if (destination < 0 || destination >= a.world || a.expert_local[expert] < 0) {
              fail(a.status, 1); continue;
            }
            for (int j = 0; j < k; ++j)
              if (ids[int64_t(row) * a.topk + j] == expert) fail(a.status, 1);
            mask |= uint64_t(1) << destination;
          }
        }
      }
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
        if (!Combine && int64_t(output_base) + num_local > a.capacity) fail(a.status, 2);
        error = atomicAdd(a.status, 0);
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
          }
        }
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
        for (int64_t i = tid; i < int64_t(num_remote) * (a.hidden / 2); i += blockDim.x) {
          int row = int(i / (a.hidden / 2)), h = int(i % (a.hidden / 2));
          const uint32_t word = reinterpret_cast<const uint32_t*>(
              a.input + int64_t(cursor + remote_rows[row]) * a.hidden)[h];
          host_store32(send_slot + x_offset + i * 4, word);
        }
        __syncthreads();
        if (tid == 0) cursor += batch;
        __syncthreads();
        if (num_remote || cursor == end) {
          if (tid == 0) {
            host_store32(send_slot + kRowsOffset, num_remote ? num_remote : uint32_t(-1));
            host_store64(send_slot + kIdentityOffset, a.identity);
          }
          __threadfence_system();
          __syncthreads();
          if (tid == 0) {
            release(control(send_slot, 0), (a.epoch << 32) | uint64_t(sent_chunks + 1));
            ++sent_chunks;
            if (!num_remote) producer_done = 1;
          }
        }
      }
    }
    __syncthreads();
    // Poll peers concurrently, then process each available chunk in parallel
    // tiles. ACK is published only after every thread has finished all reads.
    if (tid < a.world) {
      int rows = 0;
      if (!peer_done[tid]) {
        char* peer = reinterpret_cast<char*>(a.pointers[tid]) + channel * a.channel_stride;
        char* slot = peer + kArenaHeader + (peer_chunk[tid] % a.slots) * a.slot_stride;
        uint64_t expected = (a.epoch << 32) | uint64_t(peer_chunk[tid] + 1);
        if (acquire(control(slot, 0)) == expected) {
          rows = int32_t(host_load32(slot + kRowsOffset));
          if (host_load64(slot + kIdentityOffset) != a.identity) fail(a.status, 5);
          if (rows < -1 || rows == 0 || rows > a.chunk_rows) fail(a.status, 1);
        }
      }
      received_rows[tid] = rows;
    }
    __syncthreads();
    if (tid == 0) error = atomicAdd(a.status, 0);
    __syncthreads();
    if (error) break;
    for (int peer = 0; peer < a.world; ++peer) {
      const int rows = received_rows[peer];
      if (!rows) continue;
      char* peer_base = reinterpret_cast<char*>(a.pointers[peer]) + channel * a.channel_stride;
      char* slot = peer_base + kArenaHeader + (peer_chunk[peer] % a.slots) * a.slot_stride;
      for (int tile = 0; tile < rows; tile += kThreads) {
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
          if (!Combine && int64_t(output_base) + num_selected > a.capacity) fail(a.status, 2);
          error = atomicAdd(a.status, 0);
        }
        __syncthreads();
        if (error) break;
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
          }
        }
        __syncthreads();
      }
      __threadfence_system();
      __syncthreads();
      if (tid == 0) {
        release(control(slot, 1 + a.rank), (a.epoch << 32) | uint64_t(peer_chunk[peer] + 1));
        ++peer_chunk[peer];
        if (rows == -1) peer_done[peer] = 1;
      }
      __syncthreads();
      if (error) break;
    }
    const int unfinished_peers = __syncthreads_or(tid < a.world && !peer_done[tid]);
    bool unacked = false;
    if (producer_done && !unfinished_peers) {
      for (int i = tid; i < a.slots * a.world; i += blockDim.x) {
        int peer = i % a.world;
        if (peer == a.rank) continue;
        char* slot = own + kArenaHeader + (i / a.world) * a.slot_stride;
        if (acquire(control(slot, 1 + peer)) != acquire(control(slot, 0))) unacked = true;
      }
    }
    const int outstanding = __syncthreads_or(unacked);
    if (tid == 0) {
      complete = producer_done && !unfinished_peers && !outstanding;
      if (!complete && ticks_ns() - start > a.timeout_ns) fail(a.status, 3);
      error = atomicAdd(a.status, 0);
    }
    __syncthreads();
    if (complete || error) break;
    __nanosleep(64);
  }
  if (tid == 0 && error) {
    release(reinterpret_cast<uint64_t*>(own), uint64_t(error));
    if (host_load64(own + 16)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}

int64_t channel_bytes(int64_t hidden, int64_t topk, int64_t slots, int64_t rows) {
  const int64_t row = align_up(16 + topk * 12 + hidden * 2, 16);
  return align_up(kArenaHeader + slots * align_up(kSlotHeader + rows * row, 128), 4096);
}

int max_channels_current_device() {
  // Capability is fixed for this extension/device. Validate direct native
  // calls without repeating attribute and occupancy queries on every layer.
  static std::mutex mutex;
  static std::map<int, int> capacities;
  int device, cooperative, sms;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = capacities.find(device);
  if (cached != capacities.end()) return cached->second;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  if (!cooperative) { capacities.emplace(device, 0); return 0; }
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int a, b, c;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a, host_sm_transport_kernel<false, int32_t>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, host_sm_transport_kernel<false, int64_t>, kThreads, 0));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c, host_sm_transport_kernel<true, int64_t>, kThreads, 0));
  const int capacity = std::min(32, sms * std::min({a, b, c}));
  capacities.emplace(device, capacity);
  return capacity;
}

template <bool Combine, typename Id>
void launch_transport(TransportArgs args, cudaStream_t stream) {
  const int64_t elements = int64_t(args.capacity) * args.hidden * (Combine ? args.world : 1);
  const int blocks = int(std::max<int64_t>(1, std::min<int64_t>(4096, (elements + kThreads - 1) / kThreads)));
  host_sm_init_kernel<Combine><<<blocks, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (args.channels == 1) {
    host_sm_transport_kernel<Combine, Id><<<1, kThreads, 0, stream>>>(args);
  } else {
    void* parameters[] = {&args};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(host_sm_transport_kernel<Combine, Id>),
        dim3(args.channels), dim3(kThreads), parameters, 0, stream));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  host_sm_finalize_kernel<Combine><<<Combine ? blocks : 1, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_check(const torch::Tensor& value, at::ScalarType dtype, int dimensions,
                  const torch::Device& device, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.device() == device && value.scalar_type() == dtype &&
              value.dim() == dimensions && value.is_contiguous(), name, " has an invalid device, dtype or layout");
}
void common_check(torch::Tensor input, torch::Tensor pointers, int64_t rank,
                  int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
                  torch::Tensor output, torch::Tensor status, int64_t channels) {
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
  TORCH_CHECK(channels == 1 || channels <= max_channels_current_device(),
              "requested channels require cooperative all-resident launch support and capacity");
}
}  // namespace

int64_t host_sm_max_channels() {
  return max_channels_current_device();
}

int64_t host_sm_arena_bytes(int64_t hidden, int64_t topk, int64_t slots,
                            int64_t chunk_rows, int64_t channels) {
  TORCH_CHECK(hidden > 0 && hidden <= INT32_MAX && hidden % 2 == 0 && topk >= 0 && topk <= 256 &&
              slots > 0 && slots <= 64 && chunk_rows > 0 && chunk_rows <= 65536 &&
              channels > 0 && channels <= 32, "invalid arena dimensions or channels");
  return channels * channel_bytes(hidden, topk, slots, chunk_rows);
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
    int64_t identity, int64_t channels) {
  common_check(partial, pointers, rank, epoch, slots, chunk_rows, timeout_ms, output, status, channels);
  tensor_check(inverse, at::kLong, 2, partial.device(), "inverse");
  TORCH_CHECK(inverse.size(0) == partial.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, partial.device(), "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  tensor_check(remote, at::kBFloat16, 3, partial.device(), "remote accumulation");
  TORCH_CHECK(remote.size(0) == pointers.numel() && remote.size(1) == output.size(0) &&
              remote.size(2) == output.size(1), "invalid accumulation shape");
  const c10::cuda::CUDAGuard guard(partial.device());
  TransportArgs a{};
  a.input = partial.data_ptr<at::BFloat16>();
  a.input_inverse = inverse.data_ptr<int64_t>();
  a.input_count = count.data_ptr<int32_t>();
  a.pointers = pointers.data_ptr<int64_t>();
  a.output = output.data_ptr<at::BFloat16>();
  a.remote = remote.data_ptr<at::BFloat16>();
  a.status = status.data_ptr<int32_t>();
  a.world = pointers.numel(); a.rank = rank;
  a.slots = slots; a.chunk_rows = chunk_rows; a.channels = channels;
  a.input_rows = partial.size(0); a.hidden = partial.size(1); a.topk = 0; a.capacity = output.size(0);
  const int64_t row = align_up(16 + int64_t(a.hidden) * 2, 16);
  a.slot_stride = align_up(kSlotHeader + chunk_rows * row, 128);
  a.channel_stride = channel_bytes(a.hidden, 0, slots, chunk_rows);
  a.epoch = epoch; a.timeout_ns = uint64_t(timeout_ms) * 1000000; a.identity = identity;
  const auto stream = c10::cuda::getCurrentCUDAStream(partial.get_device());
  launch_transport<true, int64_t>(a, stream.stream());
}
