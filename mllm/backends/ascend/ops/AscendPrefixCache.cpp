// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendPrefixCache.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <fstream>
#include <numeric>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace {
// The block table is shared by every layer, so the tree only needs a single address lane.
constexpr int32_t kSharedBlockTableLanes = 1;
}  // namespace

AscendPrefixCache::AscendPrefixCache(const AscendPrefixCacheOptions& options) : options_(options) {
  MLLM_RT_ASSERT(options_.block_size > 0);
  MLLM_RT_ASSERT(options_.num_blocks > 0);
  MLLM_RT_ASSERT(options_.layer_nums > 0);
  MLLM_RT_ASSERT(options_.kv_heads > 0);
  MLLM_RT_ASSERT(options_.head_dim > 0);

  element_size_ = bytesOfType(options_.dtype) / lanesOfType(options_.dtype);
  block_head_bytes_ = static_cast<size_t>(options_.block_size) * options_.head_dim * element_size_;
  pool_seq_capacity_ = options_.num_blocks * options_.block_size;

  k_pool_.reserve(options_.layer_nums);
  v_pool_.reserve(options_.layer_nums);
  for (int32_t layer_idx = 0; layer_idx < options_.layer_nums; ++layer_idx) {
    k_pool_.emplace_back(
        Tensor::zeros({1, options_.kv_heads, pool_seq_capacity_, options_.head_dim}, options_.dtype, kAscend));
    v_pool_.emplace_back(
        Tensor::zeros({1, options_.kv_heads, pool_seq_capacity_, options_.head_dim}, options_.dtype, kAscend));
  }

  free_blocks_.resize(options_.num_blocks);
  // Reverse fill so that back() hands out block 0 first.
  std::iota(free_blocks_.rbegin(), free_blocks_.rend(), 0);

  prefix_cache::RadixTreeOptions tree_options = {
      .enable_lru_eviction = false,      // Not supported by RadixTree yet.
      .eviction_threshold = 0.9f,        // Not supported by RadixTree yet.
      .enable_path_compression = false,  // Not supported by RadixTree yet.
      .min_compression_length = 2,       // Not supported by RadixTree yet.
      .transformer_blocks_num = kSharedBlockTableLanes,
  };
  tree_ = std::make_shared<prefix_cache::RadixTree>(tree_options);
}

int32_t AscendPrefixCache::allocBlock() {
  if (free_blocks_.empty()) { return -1; }
  int32_t block_id = free_blocks_.back();
  free_blocks_.pop_back();
  return block_id;
}

void AscendPrefixCache::reset() {
  tree_->clear();
  free_blocks_.resize(options_.num_blocks);
  std::iota(free_blocks_.rbegin(), free_blocks_.rend(), 0);
}

void AscendPrefixCache::dot(const std::string& fp) const { std::ofstream(fp) << tree_->dot(); }

void AscendPrefixCache::copyBlockBetween(char* pool_base, char* working_base, int32_t block_id, int32_t token_pos,
                                         int32_t working_max_len, bool pool_to_working) const {
  const size_t pool_head_stride = static_cast<size_t>(pool_seq_capacity_) * options_.head_dim * element_size_;
  const size_t working_head_stride = static_cast<size_t>(working_max_len) * options_.head_dim * element_size_;
  const size_t pool_block_offset = static_cast<size_t>(block_id) * block_head_bytes_;
  const size_t working_token_offset = static_cast<size_t>(token_pos) * options_.head_dim * element_size_;

  for (int32_t h = 0; h < options_.kv_heads; ++h) {
    char* pool_ptr = pool_base + h * pool_head_stride + pool_block_offset;
    char* working_ptr = working_base + h * working_head_stride + working_token_offset;

    char* dst = pool_to_working ? working_ptr : pool_ptr;
    const char* src = pool_to_working ? pool_ptr : working_ptr;

    auto ret = aclrtMemcpy(dst, block_head_bytes_, src, block_head_bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) { MLLM_ACL_CHECK(ret); }
  }
}

AscendPrefixMatch AscendPrefixCache::matchBlocks(const std::vector<int64_t>& token_ids, bool leave_one_token) {
  AscendPrefixMatch result;
  if (token_ids.empty()) { return result; }

  auto search = tree_->search(prefix_cache::RadixTreeNodeKey(prefix_cache::VectorView<int64_t>(token_ids)));
  if (!search.success || search.matched_length <= 0) { return result; }

  // RadixTree::search descends into a child even when that child only matched partially, and then
  // goes on matching against that child's children -- which sit after the child's *full* key. So a
  // partially matched intermediate node means everything reported after it is stitched together
  // from non-adjacent positions of the original sequence: the token ids line up but their RoPE
  // positions do not. Trust the path only up to and including the first partial node.
  int32_t trustworthy = 0;
  for (const auto& [node, len] : search.path) {
    if (len == 0) { continue; }  // Root sentinel.
    trustworthy += len;
    if (len < static_cast<int32_t>(node->key.token_ids.size())) { break; }
  }
  const int32_t matched_length = std::min<int32_t>(search.matched_length, trustworthy);
  if (matched_length <= 0) { return result; }

  int32_t max_blocks = matched_length / options_.block_size;
  if (leave_one_token) {
    max_blocks = std::min(max_blocks, (static_cast<int32_t>(token_ids.size()) - 1) / options_.block_size);
  }
  if (max_blocks <= 0) { return result; }

  const auto& per_token_blocks = search.k_cache_addresses[0];
  MLLM_RT_ASSERT(static_cast<int32_t>(per_token_blocks.size()) >= max_blocks * options_.block_size);

  // Matching `block_size` tokens is NOT enough to reuse a block. Radix node boundaries are
  // independent of block boundaries, so a path can match the first tokens of a block and then
  // branch away -- the remaining slots of that block then belong to a different continuation and
  // holding on to it would feed the wrong KV. A block is only sound when every one of its tokens
  // matched and they all report the same id; the first block that fails ends the usable prefix.
  result.block_ids.reserve(max_blocks);
  for (int32_t b = 0; b < max_blocks; ++b) {
    const int32_t base = b * options_.block_size;
    const auto addr = per_token_blocks[base];
    if (addr == INVALID_VP_ADDR) { break; }

    bool uniform = true;
    for (int32_t i = 1; i < options_.block_size; ++i) {
      if (per_token_blocks[base + i] != addr) {
        uniform = false;
        break;
      }
    }
    if (!uniform) { break; }

    result.block_ids.push_back(static_cast<int32_t>(addr));
  }

  result.matched_tokens = static_cast<int32_t>(result.block_ids.size()) * options_.block_size;
  if (result.matched_tokens == 0) { result.block_ids.clear(); }
  return result;
}

AscendPrefixMatch AscendPrefixCache::match(const std::vector<int64_t>& token_ids) {
  return matchBlocks(token_ids, /*leave_one_token=*/true);
}

void AscendPrefixCache::gatherToWorkingBuffer(const AscendPrefixMatch& match, const std::vector<Tensor>& k_working,
                                              const std::vector<Tensor>& v_working, int32_t working_max_len) {
  if (match.empty()) { return; }

  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(k_working.size()), options_.layer_nums);
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(v_working.size()), options_.layer_nums);
  if (match.matched_tokens > working_max_len) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "AscendPrefixCache: matched prefix {} exceeds working buffer capacity {}",
                    match.matched_tokens, working_max_len);
  }

  for (int32_t layer_idx = 0; layer_idx < options_.layer_nums; ++layer_idx) {
    char* k_pool_base = static_cast<char*>(k_pool_[layer_idx].ptr<void>());
    char* v_pool_base = static_cast<char*>(v_pool_[layer_idx].ptr<void>());
    char* k_work_base = static_cast<char*>(k_working[layer_idx].ptr<void>());
    char* v_work_base = static_cast<char*>(v_working[layer_idx].ptr<void>());

    for (size_t b = 0; b < match.block_ids.size(); ++b) {
      const int32_t token_pos = static_cast<int32_t>(b) * options_.block_size;
      copyBlockBetween(k_pool_base, k_work_base, match.block_ids[b], token_pos, working_max_len, true);
      copyBlockBetween(v_pool_base, v_work_base, match.block_ids[b], token_pos, working_max_len, true);
    }
  }

  // Mirrors AscendKVCache: the cache must be visible before the next layer reads it.
  syncGlobalAtbStream();
}

void AscendPrefixCache::promote(const std::vector<int64_t>& token_ids, const std::vector<Tensor>& k_working,
                                const std::vector<Tensor>& v_working, int32_t working_max_len) {
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(k_working.size()), options_.layer_nums);
  MLLM_RT_ASSERT_EQ(static_cast<int32_t>(v_working.size()), options_.layer_nums);

  const int32_t total_full_blocks = static_cast<int32_t>(token_ids.size()) / options_.block_size;
  if (total_full_blocks == 0) { return; }
  if (total_full_blocks * options_.block_size > working_max_len) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "AscendPrefixCache: sequence {} exceeds working buffer capacity {}",
                    token_ids.size(), working_max_len);
  }

  // Ask the tree what it already holds rather than trusting a caller-held match. Two reasons:
  // RadixTree::insert silently discards the value when the key is already fully present, so
  // allocating first and asking later leaks blocks on every repeat of a cached prompt; and a match
  // handed across calls goes stale the moment anyone flushes the pool.
  auto owned = matchBlocks(token_ids, /*leave_one_token=*/false);
  int32_t cached_blocks = static_cast<int32_t>(owned.block_ids.size());
  std::vector<int32_t> block_table = std::move(owned.block_ids);

  if (total_full_blocks <= cached_blocks) { return; }
  int32_t wanted = total_full_blocks - cached_blocks;

  // Flush-on-full. Safe at any point because the working buffers hold a complete private copy of the
  // sequence, so dropping the tree cannot invalidate anything we still need to read.
  if (static_cast<int32_t>(free_blocks_.size()) < wanted) {
    reset();
    cached_blocks = 0;
    block_table.clear();
    wanted = total_full_blocks;
  }

  const int32_t new_blocks = std::min<int32_t>(wanted, static_cast<int32_t>(free_blocks_.size()));
  if (new_blocks == 0) { return; }

  block_table.reserve(cached_blocks + new_blocks);
  for (int32_t i = 0; i < new_blocks; ++i) {
    const int32_t block_id = allocBlock();
    MLLM_RT_ASSERT(block_id >= 0);
    block_table.push_back(block_id);
  }

  // Copy only the blocks that are new to the tree; the reused ones are already in the pool.
  for (int32_t layer_idx = 0; layer_idx < options_.layer_nums; ++layer_idx) {
    char* k_pool_base = static_cast<char*>(k_pool_[layer_idx].ptr<void>());
    char* v_pool_base = static_cast<char*>(v_pool_[layer_idx].ptr<void>());
    char* k_work_base = static_cast<char*>(k_working[layer_idx].ptr<void>());
    char* v_work_base = static_cast<char*>(v_working[layer_idx].ptr<void>());

    for (int32_t b = cached_blocks; b < cached_blocks + new_blocks; ++b) {
      const int32_t token_pos = b * options_.block_size;
      copyBlockBetween(k_pool_base, k_work_base, block_table[b], token_pos, working_max_len, false);
      copyBlockBetween(v_pool_base, v_work_base, block_table[b], token_pos, working_max_len, false);
    }
  }
  syncGlobalAtbStream();

  // RadixTree wants one address per token, parallel to token_ids.
  const int32_t promoted_tokens = (cached_blocks + new_blocks) * options_.block_size;
  std::vector<prefix_cache::vp_addr_t> per_token_blocks(promoted_tokens);
  for (int32_t t = 0; t < promoted_tokens; ++t) {
    per_token_blocks[t] = static_cast<prefix_cache::vp_addr_t>(block_table[t / options_.block_size]);
  }
  std::vector<int64_t> promoted_ids(token_ids.begin(), token_ids.begin() + promoted_tokens);

  prefix_cache::RadixTreeNodeValue value;
  value.k_cache_addresses.resize(kSharedBlockTableLanes);
  value.v_cache_addresses.resize(kSharedBlockTableLanes);
  value.k_cache_addresses[0] = prefix_cache::VectorView<prefix_cache::vp_addr_t>(per_token_blocks);
  value.v_cache_addresses[0] = value.k_cache_addresses[0];

  tree_->insert(prefix_cache::RadixTreeNodeKey(prefix_cache::VectorView<int64_t>(promoted_ids)), value);
}

}  // namespace mllm::ascend
