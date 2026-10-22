// GPU-initiated mapped-host transport. No host atomics, peer pointers or CE
// payload copies. One CTA makes send and receive progress on each device.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <limits>

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

// status: 0 success, 1 invalid route/inverse, 2 receive capacity exceeded,
// 3 timeout, 4 a peer aborted, 5 operation identity mismatch. A failed arena is poisoned until destruction.
template <bool Combine>
__global__ void transport_kernel(
    const at::BFloat16* input, const int64_t* ids, const float* weights,
    const int32_t* expert_rank, const int32_t* expert_local, int experts,
    const int64_t* pointers, int world, int rank, uint64_t epoch, int slots,
    int chunk_rows, int64_t row_stride, int64_t slot_stride, uint64_t timeout_ns,
    int input_rows, int hidden, int topk, int capacity,
    const int64_t* input_inverse, const int32_t* input_count,
    at::BFloat16* output, int64_t* output_ids, float* output_weights,
    int64_t* output_inverse, int32_t* output_count, at::BFloat16* remote,
    int32_t* status, uint64_t identity) {
  __shared__ int cursor, count, error, send_rows, action, selected, token;
  __shared__ int producer_done, peer_done[kMaxRanks], peer_chunk[kMaxRanks];
  __shared__ int sent_chunks, source_rows, row_index;
  __shared__ uint64_t mask, start;
  const int tid = threadIdx.x;
  char* own = reinterpret_cast<char*>(pointers[rank]);
  const int payload_offset = 16 + (Combine ? 0 : topk * 12);
  if (tid == 0) {
    cursor = count = send_rows = producer_done = sent_chunks = 0;
    error = *status;
    source_rows = Combine ? *input_count : input_rows;
    if (source_rows < 0 || source_rows > input_rows) error = 1;
    start = ticks_ns();
  }
  for (int r = tid; r < world; r += blockDim.x) {
    peer_done[r] = r == rank;
    peer_chunk[r] = 0;
  }
  // Capacity padding is intentionally visible to framework expert kernels.
  for (int64_t i = tid; i < int64_t(capacity) * hidden; i += blockDim.x) {
    output[i] = at::BFloat16(0.0f);
    if (Combine)
      for (int peer = 0; peer < world; ++peer)
        remote[int64_t(peer) * capacity * hidden + i] = at::BFloat16(0.0f);
  }
  if (!Combine) {
    for (int64_t i = tid; i < int64_t(capacity) * topk; i += blockDim.x) {
      output_ids[i] = -1;
      output_weights[i] = 0.0f;
    }
    for (int64_t i = tid; i < int64_t(capacity) * 2; i += blockDim.x)
      output_inverse[i] = -1;
  }
  __syncthreads();
  while (true) {
    // Never wait inside the producer: if a slot is occupied, drain receivers.
    char* send_slot = own + kArenaHeader + (sent_chunks % slots) * slot_stride;
    if (tid == 0) {
      action = !producer_done && !error;
      if (action) {
        uint64_t previous = acquire(control(send_slot, 0));
        for (int r = 0; r < world; ++r)
          if (r != rank && acquire(control(send_slot, 1 + r)) != previous)
            action = 0;
      }
      send_rows = 0;
    }
    __syncthreads();
    if (action) {
      while (cursor < source_rows && send_rows < chunk_rows && !error) {
        if (tid == 0) {
          row_index = cursor++;
          mask = 0;
          token = row_index;
          if (Combine) {
            const int64_t destination = input_inverse[row_index * 2];
            const int64_t index = input_inverse[row_index * 2 + 1];
            if (destination < 0 || destination >= world || index < 0 || index > INT32_MAX)
              error = 1;
            else { mask = uint64_t(1) << destination; token = int(index); }
          } else {
            for (int k = 0; k < topk; ++k) {
              const int64_t expert = ids[int64_t(row_index) * topk + k];
              const float weight = weights[int64_t(row_index) * topk + k];
              if (expert == -1) continue;
              if (expert < 0 || expert >= experts || !isfinite(weight)) { error = 1; break; }
              const int destination = expert_rank[expert];
              if (destination < 0 || destination >= world || expert_local[expert] < 0) {
                error = 1; break;
              }
              for (int j = 0; j < k; ++j)
                if (ids[int64_t(row_index) * topk + j] == expert) error = 1;
              mask |= uint64_t(1) << destination;
            }
            // An all--1 scheduler padding row has no destinations.
          }
          selected = -1;
          if (!error && (mask & (uint64_t(1) << rank))) {
            selected = Combine ? token : count++;
            if (selected >= capacity) error = 2;
          }
        }
        __syncthreads();
        if (error) break;
        if (selected >= 0) {
          for (int h = tid; h < hidden; h += blockDim.x)
            output[int64_t(selected) * hidden + h] = input[int64_t(row_index) * hidden + h];
          if (!Combine) {
            for (int k = tid; k < topk; k += blockDim.x) {
              const int64_t expert = ids[int64_t(row_index) * topk + k];
              output_ids[int64_t(selected) * topk + k] =
                  expert >= 0 && expert_rank[expert] == rank ? expert_local[expert] : -1;
              output_weights[int64_t(selected) * topk + k] =
                  expert >= 0 ? weights[int64_t(row_index) * topk + k] : 0.0f;
            }
            if (tid == 0) {
              output_inverse[selected * 2] = rank;
              output_inverse[selected * 2 + 1] = token;
            }
          }
        }
        if (mask & ~(uint64_t(1) << rank)) {
          char* row = send_slot + kSlotHeader + send_rows * row_stride;
          if (tid == 0) {
            host_store64(row, mask & ~(uint64_t(1) << rank));
            host_store32(row + 8, uint32_t(token));
          }
          if (!Combine) {
            for (int k = tid; k < topk; k += blockDim.x) {
              host_store64(row + 16 + k * 8, uint64_t(ids[int64_t(row_index) * topk + k]));
              host_store32(row + 16 + topk * 8 + k * 4,
                           __float_as_uint(ids[int64_t(row_index) * topk + k] >= 0 ?
                               weights[int64_t(row_index) * topk + k] : 0.0f));
            }
          }
          const auto* src = reinterpret_cast<const uint32_t*>(input + int64_t(row_index) * hidden);
          for (int h = tid; h < hidden / 2; h += blockDim.x)
            host_store32(row + payload_offset + h * 4, src[h]);
          __syncthreads();
          if (tid == 0) ++send_rows;
        }
        __syncthreads();
      }
      if (!error) {
        if (tid == 0) {
          host_store32(send_slot + kRowsOffset, send_rows ? send_rows : uint32_t(-1));
          host_store64(send_slot + kIdentityOffset, identity);
        }
        // Each writing thread fences its own stores, then publication by thread
        // zero releases the entire CTA's payload to every system observer.
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
          release(control(send_slot, 0), (epoch << 32) | uint64_t(sent_chunks + 1));
          ++sent_chunks;
          if (!send_rows) producer_done = 1;
        }
      }
    }
    __syncthreads();
    for (int peer = 0; peer < world; ++peer) {
      char* peer_base = reinterpret_cast<char*>(pointers[peer]);
      char* recv_slot = peer_base + kArenaHeader + (peer_chunk[peer] % slots) * slot_stride;
      if (tid == 0) {
        action = 0;
        if (!peer_done[peer] && !error) {
          uint64_t expected = (epoch << 32) | uint64_t(peer_chunk[peer] + 1);
          if (acquire(control(recv_slot, 0)) == expected) {
            action = int32_t(host_load32(recv_slot + kRowsOffset));
            if (host_load64(recv_slot + kIdentityOffset) != identity) error = 5;
            if (action < -1 || action > chunk_rows || action == 0) error = 1;
          }
        }
      }
      __syncthreads();
      if (action && !error) {
        for (int r = 0; r < action; ++r) {
          char* row = recv_slot + kSlotHeader + r * row_stride;
          if (tid == 0) {
            mask = host_load64(row);
            token = int(host_load32(row + 8));
            selected = -1;
            if (mask & (uint64_t(1) << rank)) {
              selected = Combine ? token : count++;
              if (token < 0 || selected >= capacity) error = 2;
            }
          }
          __syncthreads();
          if (error) break;
          if (selected >= 0) {
            auto* dst = reinterpret_cast<uint32_t*>(output + int64_t(selected) * hidden);
            for (int h = tid; h < hidden / 2; h += blockDim.x) {
              const uint32_t packed = host_load32(row + payload_offset + h * 4);
              if (Combine) {
                auto* plane = reinterpret_cast<uint32_t*>(
                    remote + (int64_t(peer) * capacity + selected) * hidden);
                plane[h] = packed;
              } else dst[h] = packed;
            }
            if (!Combine) {
              for (int k = tid; k < topk; k += blockDim.x) {
                const int64_t expert = int64_t(host_load64(row + 16 + k * 8));
                output_ids[int64_t(selected) * topk + k] =
                    expert >= 0 && expert < experts && expert_rank[expert] == rank ? expert_local[expert] : -1;
                output_weights[int64_t(selected) * topk + k] =
                    __uint_as_float(host_load32(row + 16 + topk * 8 + k * 4));
              }
              if (tid == 0) {
                output_inverse[selected * 2] = peer;
                output_inverse[selected * 2 + 1] = token;
              }
            }
          }
          __syncthreads();
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
          release(control(recv_slot, 1 + rank), (epoch << 32) | uint64_t(peer_chunk[peer] + 1));
          ++peer_chunk[peer];
          if (action == -1) peer_done[peer] = 1;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      bool complete = producer_done;
      for (int peer = 0; peer < world; ++peer) {
        if (!peer_done[peer]) complete = false;
        if ((acquire(reinterpret_cast<uint64_t*>(pointers[peer])) != 0 ||
             acquire(reinterpret_cast<uint64_t*>(pointers[peer]) + 1) != 0) && !error) error = 4;
      }
      // Drain ACKs even for our final frame before returning or the next epoch
      // could overwrite a frame that a delayed peer has not consumed yet.
      if (complete) {
        for (int s = 0; s < slots; ++s) {
          char* slot = own + kArenaHeader + s * slot_stride;
          uint64_t ready = acquire(control(slot, 0));
          for (int peer = 0; peer < world; ++peer)
            if (peer != rank && acquire(control(slot, 1 + peer)) != ready) complete = false;
        }
      }
      if (ticks_ns() - start > timeout_ns && !complete && !error) error = 3;
      if (error) release(reinterpret_cast<uint64_t*>(own), uint64_t(error));
      action = complete || error;
    }
    __syncthreads();
    if (action) break;
    __nanosleep(64);
  }
  if (Combine && !error) {
    for (int64_t i = tid; i < int64_t(capacity) * hidden; i += blockDim.x) {
      float sum = 0.0f;
      for (int peer = 0; peer < world; ++peer)
        if (peer != rank) sum += float(remote[int64_t(peer) * capacity * hidden + i]);
      const at::BFloat16 rounded = at::BFloat16(sum);
      output[i] = at::BFloat16(float(rounded) + float(output[i]));
    }
  }
  // A failed operation returns no valid rows. It must be checked by the caller
  // before accepting a result; the sticky status and arena abort forbid reuse.
  if (tid == 0) {
    if (!Combine) *output_count = error ? 0 : count;
    *status = error;
    // The framework may accept a final layer result without another dispatch.
    // An opt-in, bootstrap-only header flag makes transport errors fatal so an
    // existing output synchronization reports them, without a healthy-path
    // host fence or another kernel launch. Such failures require worker restart.
    if (error && host_load64(own + 16)) {
      __threadfence_system();
      asm volatile("trap;");
    }
  }
}

void tensor_check(const torch::Tensor& value, at::ScalarType dtype, int dimensions,
                  const torch::Device& device, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.device() == device && value.scalar_type() == dtype &&
              value.dim() == dimensions && value.is_contiguous(), name, " has an invalid device, dtype or layout");
}
void common_check(torch::Tensor input, torch::Tensor pointers, int64_t rank,
                  int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
                  torch::Tensor output, torch::Tensor status) {
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
}
}  // namespace

int64_t host_sm_arena_bytes(int64_t hidden, int64_t topk, int64_t slots, int64_t chunk_rows) {
  TORCH_CHECK(hidden > 0 && hidden <= INT32_MAX && hidden % 2 == 0 && topk >= 0 && topk <= 256 &&
              slots > 0 && slots <= 64 && chunk_rows > 0 && chunk_rows <= 65536, "invalid arena dimensions");
  const int64_t row = align_up(16 + topk * 12 + hidden * 2, 16);
  return align_up(kArenaHeader + slots * align_up(kSlotHeader + chunk_rows * row, 128), 4096);
}

void host_sm_dispatch_cuda(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
    torch::Tensor expert_rank, torch::Tensor expert_local, torch::Tensor pointers,
    int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
    torch::Tensor output, torch::Tensor output_ids, torch::Tensor output_weights,
    torch::Tensor inverse, torch::Tensor count, torch::Tensor status, int64_t identity) {
  common_check(input, pointers, rank, epoch, slots, chunk_rows, timeout_ms, output, status);
  const auto device = input.device();
  tensor_check(ids, at::kLong, 2, device, "ids");
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
  const int64_t row = align_up(16 + ids.size(1) * 12 + input.size(1) * 2, 16);
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  transport_kernel<false><<<1, kThreads, 0, stream.stream()>>>(
      input.data_ptr<at::BFloat16>(), ids.data_ptr<int64_t>(), weights.data_ptr<float>(),
      expert_rank.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(), expert_rank.numel(),
      pointers.data_ptr<int64_t>(), pointers.numel(), rank, epoch, slots, chunk_rows, row,
      align_up(kSlotHeader + chunk_rows * row, 128), uint64_t(timeout_ms) * 1000000,
      input.size(0), input.size(1), ids.size(1), output.size(0), nullptr, nullptr,
      output.data_ptr<at::BFloat16>(), output_ids.data_ptr<int64_t>(), output_weights.data_ptr<float>(),
      inverse.data_ptr<int64_t>(), count.data_ptr<int32_t>(), nullptr, status.data_ptr<int32_t>(), uint64_t(identity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void host_sm_combine_cuda(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
    torch::Tensor pointers, int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows,
    int64_t timeout_ms, torch::Tensor output, torch::Tensor remote, torch::Tensor status, int64_t identity) {
  common_check(partial, pointers, rank, epoch, slots, chunk_rows, timeout_ms, output, status);
  tensor_check(inverse, at::kLong, 2, partial.device(), "inverse");
  TORCH_CHECK(inverse.size(0) == partial.size(0) && inverse.size(1) == 2, "invalid inverse shape");
  tensor_check(count, at::kInt, 1, partial.device(), "count");
  TORCH_CHECK(count.numel() == 1, "count must contain one element");
  tensor_check(remote, at::kBFloat16, 3, partial.device(), "remote accumulation");
  TORCH_CHECK(remote.size(0) == pointers.numel() && remote.size(1) == output.size(0) &&
              remote.size(2) == output.size(1), "invalid accumulation shape");
  const c10::cuda::CUDAGuard guard(partial.device());
  const int64_t row = align_up(16 + partial.size(1) * 2, 16);
  const auto stream = c10::cuda::getCurrentCUDAStream(partial.get_device());
  transport_kernel<true><<<1, kThreads, 0, stream.stream()>>>(
      partial.data_ptr<at::BFloat16>(), nullptr, nullptr, nullptr, nullptr, 0,
      pointers.data_ptr<int64_t>(), pointers.numel(), rank, epoch, slots, chunk_rows, row,
      align_up(kSlotHeader + chunk_rows * row, 128), uint64_t(timeout_ms) * 1000000,
      partial.size(0), partial.size(1), 0, output.size(0), inverse.data_ptr<int64_t>(), count.data_ptr<int32_t>(),
      output.data_ptr<at::BFloat16>(), nullptr, nullptr, nullptr, nullptr,
      remote.data_ptr<at::BFloat16>(), status.data_ptr<int32_t>(), uint64_t(identity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
