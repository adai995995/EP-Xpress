#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__global__ void add_partial_kernel(float* accum, const at::BFloat16* partial,
                                    const int64_t* indices, int64_t elements,
                                    int64_t hidden, int64_t tokens) {
  for (int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < elements; element += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = element / hidden;
    const int64_t column = element % hidden;
    const int64_t token = indices[row];
    CUDA_KERNEL_ASSERT(token >= 0 && token < tokens);
    // The route contract guarantees one writer per output element in this call.
    accum[token * hidden + column] += static_cast<float>(partial[element]);
  }
}

__global__ void finish_combine_kernel(const float* remote,
                                       const at::BFloat16* local,
                                       at::BFloat16* result, int64_t elements) {
  for (int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < elements; element += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const at::BFloat16 rounded_remote = static_cast<at::BFloat16>(remote[element]);
    result[element] = static_cast<at::BFloat16>(
        static_cast<float>(rounded_remote) + static_cast<float>(local[element]));
  }
}

int blocks_for(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements - 1) / kThreads + 1, 4096));
}

}  // namespace

void add_partial_cuda(torch::Tensor accum, torch::Tensor partial,
                      torch::Tensor token_indices) {
  if (partial.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(accum.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(accum.get_device());
  add_partial_kernel<<<blocks_for(partial.numel()), kThreads, 0, stream.stream()>>>(
      accum.data_ptr<float>(), partial.data_ptr<at::BFloat16>(),
      token_indices.data_ptr<int64_t>(), partial.numel(), partial.size(1), accum.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor finish_combine_cuda(torch::Tensor remote, torch::Tensor local) {
  const c10::cuda::CUDAGuard guard(local.device());
  auto result = torch::empty_like(local);
  if (result.numel() == 0) return result;
  const auto stream = c10::cuda::getCurrentCUDAStream(local.get_device());
  finish_combine_kernel<<<blocks_for(result.numel()), kThreads, 0, stream.stream()>>>(
      remote.data_ptr<float>(), local.data_ptr<at::BFloat16>(),
      result.data_ptr<at::BFloat16>(), result.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
