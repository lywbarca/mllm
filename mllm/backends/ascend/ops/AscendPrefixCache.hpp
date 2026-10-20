// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Ascend prefix (radix) cache, phase 1.
//
// Design notes:
//
// 1. The radix tree itself (mllm/engine/prefix_cache/RadixTree) is device agnostic: it only ever stores
//    vp_addr_t (uint32_t) integers. We reuse it as-is and never touch device memory from it.
//
// 2. RadixTree is hard-wired to "one address per token": both insert() and search() index the value's
//    address vector in parallel with token_ids. To get block granularity without forking the tree we
//    store, for every token, the physical block id that token lives in. All block_size tokens of one
//    block therefore carry the same id.
//
//    Recovering the block table is NOT simply sampling every block_size-th entry. Radix node
//    boundaries fall wherever two sequences diverge, which has nothing to do with block boundaries,
//    so one block window can legitimately contain two different ids. See matchBlocks().
//
// 3. The block table is shared across layers (vLLM style): physical block p means slot p in *every*
//    layer's pool buffer. So we configure the tree with transformer_blocks_num = 1 and only use
//    index 0 of RadixTreeNodeValue.
//
// 4. Only *full* blocks are ever promoted. A partially filled block would otherwise be handed to a
//    different continuation while still holding the previous sequence's tokens.
//
// 5. Phase 1 does NOT change the attention kernel. Matched blocks are gathered (device->device) into
//    the caller's existing contiguous working buffer, so both the eager path and the ATB graph path
//    see an ordinary dense KV cache. The working buffer is a private copy, which is what makes the
//    whole scheme safe without refcounting.
//
// Not implemented yet (phase 1 scope):
// - LRU / refcount eviction. RadixTree exposes no erase and its ref_count is never consumed, so the
//   pool is reclaimed all at once via reset() when it cannot serve a promotion.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/engine/prefix_cache/RadixTree.hpp"

namespace mllm::ascend {

struct AscendPrefixCacheOptions {
  // Tokens per block. Keeps the per-head copy at block_size * head_dim contiguous elements.
  int32_t block_size = 128;

  // Number of physical blocks in the pool, shared by all layers.
  int32_t num_blocks = 0;

  int32_t layer_nums = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  DataTypes dtype = kFloat16;
};

struct AscendPrefixMatch {
  // Always a multiple of block_size (a partial trailing block is never usable).
  int32_t matched_tokens = 0;

  // matched_tokens / block_size entries, in sequence order.
  std::vector<int32_t> block_ids;

  [[nodiscard]] inline bool empty() const { return matched_tokens == 0; }
};

class AscendPrefixCache {
 public:
  using ptr_t = std::shared_ptr<AscendPrefixCache>;

  AscendPrefixCache() = default;

  explicit AscendPrefixCache(const AscendPrefixCacheOptions& options);

  /// Longest reusable cached prefix of token_ids. Always leaves at least one token for the caller to
  /// actually run a forward pass on.
  AscendPrefixMatch match(const std::vector<int64_t>& token_ids);

  /// Copy the matched blocks out of the pool into caller-owned contiguous working buffers, one K and
  /// one V tensor per layer. Both pool and working buffers are head-major
  /// ([1, kv_heads, <seq capacity>, head_dim]); the two sequence capacities differ, hence
  /// working_max_len. Tokens land at positions [0, match.matched_tokens).
  void gatherToWorkingBuffer(const AscendPrefixMatch& match, const std::vector<Tensor>& k_working,
                             const std::vector<Tensor>& v_working, int32_t working_max_len);

  /// Copy the full blocks of a finished sequence out of the working buffers into freshly allocated
  /// pool blocks and insert them into the radix tree.
  ///
  /// Works out for itself how much of token_ids the tree already covers, so it is safe to call
  /// repeatedly on a growing sequence (prompt, then prompt + generated tokens) and it never
  /// allocates a block for a prefix it already owns.
  ///
  /// PRECONDITION: the working buffers hold exactly this sequence at positions
  /// [0, token_ids.size()) -- both any gathered prefix and everything computed after it.
  ///
  /// Promotes as many whole blocks as the pool can still serve; a short pool just means less reuse.
  void promote(const std::vector<int64_t>& token_ids, const std::vector<Tensor>& k_working,
               const std::vector<Tensor>& v_working, int32_t working_max_len);

  /// Drop every cached prefix and return all blocks to the free list.
  void reset();

  [[nodiscard]] inline int32_t blockSize() const { return options_.block_size; }

  [[nodiscard]] inline int32_t freeBlocks() const { return static_cast<int32_t>(free_blocks_.size()); }

  [[nodiscard]] inline const AscendPrefixCacheOptions& options() const { return options_; }

  /// "dot -Tpng tree.dot -o tree.png"
  void dot(const std::string& fp) const;

 private:
  /// Whole blocks of token_ids the tree currently holds, soundly.
  ///
  /// `leave_one_token` caps the result so the caller still has something to compute; promote() wants
  /// it off, since it is asking what it already owns rather than what it may reuse.
  AscendPrefixMatch matchBlocks(const std::vector<int64_t>& token_ids, bool leave_one_token);

  /// -1 when the pool is exhausted.
  int32_t allocBlock();

  /// Move one block (all heads) between a pool buffer and a working buffer.
  void copyBlockBetween(char* pool_base, char* working_base, int32_t block_id, int32_t token_pos,
                        int32_t working_max_len, bool pool_to_working) const;

  AscendPrefixCacheOptions options_;

  size_t element_size_ = 0;
  size_t block_head_bytes_ = 0;    //< block_size * head_dim * element_size
  int32_t pool_seq_capacity_ = 0;  //< num_blocks * block_size

  // Pool buffers, one K and one V per layer: [1, kv_heads, num_blocks * block_size, head_dim].
  std::vector<Tensor> k_pool_;
  std::vector<Tensor> v_pool_;

  std::vector<int32_t> free_blocks_;

  std::shared_ptr<prefix_cache::RadixTree> tree_ = nullptr;
};

}  // namespace mllm::ascend
