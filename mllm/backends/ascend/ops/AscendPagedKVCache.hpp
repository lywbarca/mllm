// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <memory>
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/PagedAttnOp.hpp"
#include "mllm/engine/paged_attention/PagedBlockManager.hpp"

namespace mllm::ascend {

struct AscendPagedKVCacheOptions {
  int32_t block_size = 128;
  int32_t num_blocks = 0;
  int32_t max_sequence_length = 2048;
  int32_t layer_nums = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  bool radix_cache = false;
  aops::PagedAttnCacheLayout layout = aops::PagedAttnCacheLayout::kNZ;
};

// Fixed physical pool shared by all resident sequences, with per-layer K/V tensors.
// Methods are synchronous on the global ATB stream; callers serialize access.
class AscendPagedKVCache {
 public:
  using SequenceId = paged_attention::PagedBlockManager::SequenceId;
  explicit AscendPagedKVCache(const AscendPagedKVCacheOptions& options);
  AscendPagedKVCache(const AscendPagedKVCache&) = delete;
  AscendPagedKVCache& operator=(const AscendPagedKVCache&) = delete;

  const AscendPagedKVCacheOptions& options() const { return options_; }
  const paged_attention::PagedBlockManager& blocks() const { return blocks_; }
  void createSequence(SequenceId id) { blocks_.create(id); }
  void forkSequence(SequenceId source, SequenceId destination) { blocks_.fork(source, destination); }
  void releaseSequence(SequenceId id) { blocks_.release(id); }
  void clear();  // Reset requests; retain radix prefixes if enabled.
  bool radixEnabled() const { return blocks_.radixEnabled(); }
  void clearPrefixCache() { blocks_.clearRadixCache(); }
  int32_t reusePrefix(SequenceId id, const std::vector<int64_t>& tokens) { return blocks_.reusePrefix(id, tokens); }

  // Restores an initially empty request if slicing, preparation or forward fails.
  // Declare before StepGuard so any pending writes are aborted first.
  class PrefixGuard {
   public:
    PrefixGuard(AscendPagedKVCache* cache, SequenceId id) : cache_(cache), id_(id) {}
    ~PrefixGuard() {
      if (attached_) { cache_->blocks_.resetSequence(id_); }
    }
    void attached(bool value) { attached_ = value; }
    void commit() { attached_ = false; }
    PrefixGuard(const PrefixGuard&) = delete;
    PrefixGuard& operator=(const PrefixGuard&) = delete;

   private:
    AscendPagedKVCache* cache_;
    SequenceId id_;
    bool attached_ = false;
  };

  // All sequences in this tensor batch append the same number of tokens; their
  // historical lengths can differ. False is recoverable physical-pool exhaustion.
  bool beginStep(const std::vector<SequenceId>& ids, int32_t token_count, const std::vector<std::vector<int64_t>>& tokens = {});
  void writeLayer(int32_t layer, const Tensor& k, const Tensor& v);  // [B,S,Hkv,D], post-RoPE K
  void commitStep();
  void abortStep();
  bool pending() const { return blocks_.pending(); }
  // Prepare causal per-query metadata once per step, shared by all decoder layers.
  void prepareGraphMetadata();
  void markGraphLayerWritten(int32_t layer);
  Tensor graphBlockTable() const { return graph_table_; }
  Tensor graphLengthsHost() const { return graph_lengths_host_; }
  Tensor graphLengthsDevice() const { return graph_lengths_device_; }
  Tensor slotMapping() const { return slot_mapping_; }
  Tensor blockTable() const { return block_table_; }
  Tensor contextLengths() const { return context_lengths_; }
  Tensor keyPool(int32_t layer) const { return k_pool_.at(layer); }
  Tensor valuePool(int32_t layer) const { return v_pool_.at(layer); }

  // Diagnostic gather for storage tests. Model prefill/decode read pools directly.
  std::array<Tensor, 2> gatherLayer(int32_t layer, size_t batch_index) const;
  uint64_t poolBytes() const;

  // Roll back a prepared step if a later model operation throws.
  class StepGuard {
   public:
    explicit StepGuard(AscendPagedKVCache* cache) : cache_(cache) {}
    ~StepGuard() {
      if (cache_ && cache_->pending()) { cache_->abortStep(); }
    }
    StepGuard(const StepGuard&) = delete;
    StepGuard& operator=(const StepGuard&) = delete;

   private:
    AscendPagedKVCache* cache_;
  };

 private:
  static AscendPagedKVCacheOptions validate(AscendPagedKVCacheOptions options);
  void resetStepTensors();
  AscendPagedKVCacheOptions options_;
  paged_attention::PagedBlockManager blocks_;
  std::vector<Tensor> k_pool_, v_pool_;
  std::vector<bool> written_;
  Tensor block_table_, context_lengths_, slot_mapping_;
  Tensor graph_table_, graph_lengths_host_, graph_lengths_device_;
  int32_t step_tokens_ = 0;
};

}  // namespace mllm::ascend
