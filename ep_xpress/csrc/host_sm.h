#pragma once

#include <torch/extension.h>
#include <memory>

// Bootstrap owns the immutable placement, arena pointer tables and status
// storage. A prepared operator retains these tensors and their validated layout;
// callers still retain every per-operation tensor until its stream completes.
class PreparedHostSM {
 public:
  PreparedHostSM(torch::Tensor expert_rank, torch::Tensor expert_local,
      torch::Tensor dispatch_pointers, torch::Tensor combine_pointers,
      torch::Tensor status, int64_t rank, int64_t hidden, int64_t topk,
      int64_t slots, int64_t dispatch_rows, int64_t combine_rows,
      int64_t timeout_ms, int64_t channels, bool diagnostics, bool fatal_errors);
  void dispatch(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
      int64_t epoch, torch::Tensor output, torch::Tensor output_ids,
      torch::Tensor output_weights, torch::Tensor inverse, torch::Tensor count,
      torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t active_channels,
      int64_t identity);
  void combine(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
      torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t epoch,
      torch::Tensor output, torch::Tensor remote, int64_t active_channels, int64_t identity,
      c10::optional<torch::Tensor> profile);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
