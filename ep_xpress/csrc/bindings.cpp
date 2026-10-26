#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "host_sm.h"

void add_partial_cuda(torch::Tensor accum, torch::Tensor partial,
                      torch::Tensor token_indices);
torch::Tensor finish_combine_cuda(torch::Tensor remote, torch::Tensor local);

int64_t host_sm_max_channels(int64_t world);
int64_t host_sm_mailbox_bytes(int64_t hidden, int64_t tokens, int64_t world);
int64_t host_sm_packet_rows();
int64_t host_sm_dispatch_banks();
int64_t host_sm_dispatch_roles();
bool host_sm_reuse_return_index();
std::vector<std::string> host_sm_profile_fields();
std::vector<std::string> host_sm_decode_profile_fields(bool dispatch);
int64_t host_sm_arena_bytes(int64_t hidden, int64_t topk, int64_t slots, int64_t chunk_rows, int64_t channels, int64_t queue_peers);
void host_sm_dispatch_cuda(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
    torch::Tensor expert_rank, torch::Tensor expert_local, torch::Tensor pointers,
    int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows, int64_t timeout_ms,
    torch::Tensor output, torch::Tensor output_ids, torch::Tensor output_weights,
    torch::Tensor inverse, torch::Tensor count, torch::Tensor status, int64_t identity, int64_t channels);
void host_sm_combine_cuda(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
    torch::Tensor pointers, int64_t rank, int64_t epoch, int64_t slots, int64_t chunk_rows,
    int64_t timeout_ms, torch::Tensor output, torch::Tensor remote, torch::Tensor status,
    int64_t identity, int64_t channels, c10::optional<torch::Tensor> profile);

namespace {

std::mutex registration_mutex;
std::map<uintptr_t, size_t> registrations;

void check_matrix(const torch::Tensor& tensor, const char* name,
                  at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(), name,
              " must be a contiguous two-dimensional tensor");
}

void register_host_flags(uintptr_t address, int64_t nbytes, unsigned int flags) {
  TORCH_CHECK(address != 0 && nbytes > 0,
              "host address and size must be positive");
  const auto size = static_cast<size_t>(nbytes);
  TORCH_CHECK(size <= std::numeric_limits<uintptr_t>::max() - address,
              "host address range overflows");
  std::lock_guard<std::mutex> lock(registration_mutex);
  auto after = registrations.lower_bound(address);
  if (after != registrations.end()) {
    TORCH_CHECK(address + size <= after->first,
                "host range overlaps an existing registration");
  }
  if (after != registrations.begin()) {
    auto before = std::prev(after);
    TORCH_CHECK(before->first + before->second <= address,
                "host range overlaps an existing registration");
  }
  C10_CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(address), size,
                                  flags));
  registrations.emplace(address, size);
}

void register_host(uintptr_t address, int64_t nbytes) {
  register_host_flags(address, nbytes, cudaHostRegisterPortable);
}

uintptr_t register_mapped_host(uintptr_t address, int64_t nbytes) {
  int device, can_map;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device));
  TORCH_CHECK(can_map, "CUDA device does not support mapped host memory");
  register_host_flags(address, nbytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
  void* mapped = nullptr;
  const auto result = cudaHostGetDevicePointer(&mapped, reinterpret_cast<void*>(address), 0);
  if (result != cudaSuccess) {
    std::lock_guard<std::mutex> lock(registration_mutex);
    cudaHostUnregister(reinterpret_cast<void*>(address));
    registrations.erase(address);
    C10_CUDA_CHECK(result);
  }
  return reinterpret_cast<uintptr_t>(mapped);
}

void unregister_host(uintptr_t address) {
  std::lock_guard<std::mutex> lock(registration_mutex);
  auto entry = registrations.find(address);
  TORCH_CHECK(entry != registrations.end(), "host address is not registered");
  C10_CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void*>(address)));
  registrations.erase(entry);
}

void copy_d2h_async(uintptr_t destination, torch::Tensor source,
                    int64_t row_start, int64_t row_count, uintptr_t stream) {
  check_matrix(source, "src", at::kBFloat16);
  TORCH_CHECK(row_start >= 0 && row_start <= source.size(0) && row_count >= 0 &&
                  row_count <= source.size(0) - row_start,
              "row slice is outside src");
  const auto row_bytes = static_cast<size_t>(source.size(1)) * sizeof(at::BFloat16);
  const auto bytes = static_cast<size_t>(row_count) * row_bytes;
  std::lock_guard<std::mutex> lock(registration_mutex);
  auto after = registrations.upper_bound(destination);
  TORCH_CHECK(after != registrations.begin(), "destination is not registered");
  auto entry = std::prev(after);
  const auto offset = destination - entry->first;
  TORCH_CHECK(offset <= entry->second && bytes <= entry->second - offset,
              "destination copy range is outside its host registration");
  if (bytes == 0) return;
  const c10::cuda::CUDAGuard guard(source.device());
  const auto* src = reinterpret_cast<const char*>(source.data_ptr()) +
                    static_cast<size_t>(row_start) * row_bytes;
  C10_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(destination), src, bytes,
                                 cudaMemcpyDeviceToHost,
                                 reinterpret_cast<cudaStream_t>(stream)));
}

void add_partial(torch::Tensor accum, torch::Tensor partial,
                  torch::Tensor token_indices) {
  check_matrix(accum, "accum", at::kFloat);
  check_matrix(partial, "partial", at::kBFloat16);
  TORCH_CHECK(token_indices.is_cuda() && token_indices.scalar_type() == at::kLong,
              "token_indices must be a CUDA int64 tensor");
  TORCH_CHECK(token_indices.dim() == 1 && token_indices.is_contiguous(),
              "token_indices must be a contiguous vector");
  TORCH_CHECK(accum.device() == partial.device() &&
                  accum.device() == token_indices.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(accum.size(1) == partial.size(1) &&
                  partial.size(0) == token_indices.numel(),
              "partial shape or index count does not match");
  // Bounds and uniqueness are validated by the public Python wrapper or by
  // route construction before using its explicitly trusted asynchronous path.
  add_partial_cuda(accum, partial, token_indices);
}

torch::Tensor finish_combine(torch::Tensor remote, torch::Tensor local) {
  check_matrix(remote, "remote", at::kFloat);
  check_matrix(local, "local", at::kBFloat16);
  TORCH_CHECK(remote.device() == local.device() && remote.sizes() == local.sizes(),
              "remote and local must have the same shape and device");
  return finish_combine_cuda(remote, local);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<PreparedHostSM, std::shared_ptr<PreparedHostSM>>(module, "PreparedHostSM")
      .def("enable_device_epochs", &PreparedHostSM::enable_device_epochs)
      .def("prepare_decode_profile", &PreparedHostSM::prepare_decode_profile,
          pybind11::arg("storage"), pybind11::arg("dispatch") = false)
      .def("dispatch", &PreparedHostSM::dispatch,
          pybind11::arg("x"), pybind11::arg("ids"), pybind11::arg("weights"),
          pybind11::arg("epoch"), pybind11::arg("output"), pybind11::arg("output_ids"),
          pybind11::arg("output_weights"), pybind11::arg("inverse"), pybind11::arg("count"),
          pybind11::arg("owner_rows"), pybind11::arg("owner_counts"),
          pybind11::arg("active_channels") = 0, pybind11::arg("identity") = 0,
          pybind11::arg("expected_ranks") = pybind11::none(),
          pybind11::arg("local_rows") = pybind11::none(),
          pybind11::arg("profile") = pybind11::none(),
          pybind11::arg("return_rows") = pybind11::none())
      .def("combine", &PreparedHostSM::combine,
          pybind11::arg("partial"), pybind11::arg("inverse"), pybind11::arg("count"),
          pybind11::arg("owner_rows"), pybind11::arg("owner_counts"),
          pybind11::arg("epoch"), pybind11::arg("output"), pybind11::arg("remote"),
          pybind11::arg("active_channels") = 0, pybind11::arg("identity") = 0, pybind11::arg("profile") = pybind11::none(),
          pybind11::arg("expected_ranks") = pybind11::none(),
          pybind11::arg("local_rows") = pybind11::none(),
          pybind11::arg("return_rows") = pybind11::none());
  module.def("prepare_host_sm", [](torch::Tensor expert_rank, torch::Tensor expert_local,
      torch::Tensor dispatch_pointers, torch::Tensor combine_pointers, torch::Tensor status,
      int64_t rank, int64_t hidden, int64_t topk, int64_t slots, int64_t dispatch_rows,
      int64_t combine_rows, int64_t timeout_ms, int64_t channels,
      bool diagnostics, bool fatal_errors, c10::optional<torch::Tensor> mailbox_pointers,
      int64_t mailbox_tokens, bool combine_warp_specialization, bool decode_single_frame) {
    return std::make_shared<PreparedHostSM>(expert_rank, expert_local, dispatch_pointers,
        combine_pointers, status, rank, hidden, topk, slots, dispatch_rows, combine_rows,
        timeout_ms, channels, diagnostics, fatal_errors, mailbox_pointers, mailbox_tokens,
        combine_warp_specialization, decode_single_frame);
  }, pybind11::arg("expert_rank"), pybind11::arg("expert_local"),
      pybind11::arg("dispatch_pointers"), pybind11::arg("combine_pointers"),
      pybind11::arg("status"), pybind11::arg("rank"), pybind11::arg("hidden"),
      pybind11::arg("topk"), pybind11::arg("slots"), pybind11::arg("dispatch_rows"),
      pybind11::arg("combine_rows"), pybind11::arg("timeout_ms"), pybind11::arg("channels"),
      pybind11::kw_only(), pybind11::arg("diagnostics") = false,
      pybind11::arg("fatal_errors") = false,
      pybind11::arg("mailbox_pointers") = pybind11::none(),
      pybind11::arg("mailbox_tokens") = 0,
      pybind11::arg("combine_warp_specialization") = false,
      pybind11::arg("decode_single_frame") = false);
  module.def("register_host", &register_host);
  module.def("register_mapped_host", &register_mapped_host);
  module.def("host_sm_max_channels", &host_sm_max_channels, pybind11::arg("world") = 1);
  module.def("host_sm_profile_fields", &host_sm_profile_fields);
  module.def("host_sm_decode_profile_fields", &host_sm_decode_profile_fields, pybind11::arg("dispatch") = false);
  module.def("host_sm_packet_rows", &host_sm_packet_rows);
  module.def("host_sm_reuse_return_index", &host_sm_reuse_return_index);
  module.def("host_sm_dispatch_roles", &host_sm_dispatch_roles);
  module.def("host_sm_dispatch_banks", &host_sm_dispatch_banks);
  module.def("host_sm_mailbox_bytes", &host_sm_mailbox_bytes,
      pybind11::arg("hidden"), pybind11::arg("tokens"), pybind11::arg("world"));
  module.def("host_sm_arena_bytes", &host_sm_arena_bytes,
      pybind11::arg("hidden"), pybind11::arg("topk"), pybind11::arg("slots"),
      pybind11::arg("chunk_rows"), pybind11::arg("channels"),
      pybind11::arg("queue_peers") = 1);
  module.def("host_sm_dispatch", &host_sm_dispatch_cuda);
  module.def("host_sm_combine", &host_sm_combine_cuda,
      pybind11::arg("partial"), pybind11::arg("inverse"), pybind11::arg("count"),
      pybind11::arg("pointers"), pybind11::arg("rank"), pybind11::arg("epoch"),
      pybind11::arg("slots"), pybind11::arg("chunk_rows"), pybind11::arg("timeout_ms"),
      pybind11::arg("output"), pybind11::arg("remote"), pybind11::arg("status"),
      pybind11::arg("identity"), pybind11::arg("channels"),
      pybind11::arg("profile") = pybind11::none());
  module.def("unregister_host", &unregister_host);
  module.def("copy_d2h_async", &copy_d2h_async);
  module.def("add_partial", &add_partial);
  module.def("finish_combine", &finish_combine);
}
