#pragma once

#include <torch/extension.h>
#include <memory>

// Bootstrap owns the immutable placement, arena pointer tables and status
// storage. A prepared operator retains these tensors and their validated layout;
// callers still retain every per-operation tensor until its stream completes.
// Fused mailbox combines use one serialized stream and increasing epochs.
class PreparedHostSM {
 public:
  PreparedHostSM(torch::Tensor expert_rank, torch::Tensor expert_local,
      torch::Tensor dispatch_pointers, torch::Tensor combine_pointers,
      torch::Tensor status, int64_t rank, int64_t hidden, int64_t topk,
      int64_t slots, int64_t dispatch_rows, int64_t combine_rows,
      int64_t timeout_ms, int64_t channels, bool diagnostics, bool fatal_errors,
      c10::optional<torch::Tensor> mailbox_pointers, int64_t mailbox_tokens,
      bool combine_warp_specialization, bool decode_single_frame);
  // Before any operation. Caller serializes all eager/replay work and keeps
  // this object and its arenas alive until every captured graph is destroyed.
  void enable_device_epochs();
  // Fixed diagnostic storage; install once before submissions/Graph capture.
  void prepare_decode_profile(torch::Tensor storage);
  void dispatch(torch::Tensor input, torch::Tensor ids, torch::Tensor weights,
      int64_t epoch, torch::Tensor output, torch::Tensor output_ids,
      torch::Tensor output_weights, torch::Tensor inverse, torch::Tensor count,
      torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t active_channels,
      int64_t identity, c10::optional<torch::Tensor> expected_ranks,
      c10::optional<torch::Tensor> local_rows, c10::optional<torch::Tensor> profile);
  void combine(torch::Tensor partial, torch::Tensor inverse, torch::Tensor count,
      torch::Tensor owner_rows, torch::Tensor owner_counts, int64_t epoch,
      torch::Tensor output, c10::optional<torch::Tensor> remote, int64_t active_channels, int64_t identity,
      c10::optional<torch::Tensor> profile, c10::optional<torch::Tensor> expected_ranks,
      c10::optional<torch::Tensor> local_rows);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
