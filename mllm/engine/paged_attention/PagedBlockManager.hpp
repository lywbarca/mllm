// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mllm::paged_attention {

// Host metadata only. One physical block ID denotes the same slot in every layer.
// Single-owner/single-threaded: one batch append may be in flight at a time.
class PagedBlockManager {
 public:
  using SequenceId = int64_t;
  struct Sequence {
    int32_t length = 0;
    std::vector<int32_t> blocks;
    std::vector<int64_t> tokens;  // Populated only when radix caching is enabled.
  };
  struct Append {
    SequenceId id;
    int32_t old_length;
    Sequence next;
    int32_t replaced_block = -1;
  };

  PagedBlockManager(int32_t num_blocks, int32_t block_size, int32_t max_length)
      : block_size_(block_size), max_length_(max_length) {
    if (num_blocks <= 0 || block_size <= 0 || max_length <= 0
        || static_cast<int64_t>(num_blocks) * block_size > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("Invalid paged KV capacity or int32 slot range");
    }
    refs_.resize(num_blocks, 0);
    cache_refs_.resize(num_blocks, 0);
    free_.reserve(num_blocks);
    for (int32_t p = num_blocks; p > 0; --p) { free_.push_back(p - 1); }
  }
  PagedBlockManager(const PagedBlockManager&) = delete;
  PagedBlockManager& operator=(const PagedBlockManager&) = delete;

  void create(SequenceId id) {
    requireIdle();
    if (!sequences_.emplace(id, Sequence{}).second) { throw std::invalid_argument("Duplicate sequence ID"); }
  }
  void fork(SequenceId source, SequenceId destination) {
    requireIdle();
    auto snapshot = sequence(source);
    if (!sequences_.emplace(destination, std::move(snapshot)).second) {
      throw std::invalid_argument("Duplicate fork destination");
    }
    for (auto block : sequence(destination).blocks) { ++refs_[block]; }
  }
  void release(SequenceId id) {
    requireIdle();
    const auto& seq = sequence(id);
    for (auto block : seq.blocks) { decref(block); }
    sequences_.erase(id);
  }
  void clear() {
    requireIdle();
    radix_root_.children.clear();
    cached_blocks_ = cached_nodes_ = 0;
    std::fill(cache_refs_.begin(), cache_refs_.end(), 0);
    sequences_.clear();
    free_.clear();
    std::fill(refs_.begin(), refs_.end(), 0);
    for (int32_t p = static_cast<int32_t>(refs_.size()); p > 0; --p) { free_.push_back(p - 1); }
  }
  // Enable before any request has KV. A cache belongs to one model/weight set.
  void enableRadixCache() {
    requireIdle();
    for (const auto& item : sequences_) {
      if (item.second.length != 0) { throw std::logic_error("Enable radix before prefill"); }
    }
    radix_enabled_ = true;
  }
  bool radixEnabled() const { return radix_enabled_; }
  size_t cachedBlocks() const { return cached_blocks_; }
  size_t cachedNodes() const { return cached_nodes_; }
  size_t cacheRefCount(int32_t block) const { return cache_refs_.at(block); }

  // Request reset preserves reusable prefixes; clear() explicitly drops both.
  void clearSequences() {
    requireIdle();
    for (const auto& item : sequences_) {
      for (auto block : item.second.blocks) { decref(block); }
    }
    sequences_.clear();
  }
  void resetSequence(SequenceId id) {
    requireIdle();
    auto& seq = sequences_.at(id);
    for (auto block : seq.blocks) { decref(block); }
    seq = Sequence{};
  }
  void clearRadixCache() {
    requireIdle();
    dropRadix(radix_root_);
  }

  // Match token prefixes, including inside an edge. Never skip unmatched tokens.
  // Leave at least one query token to compute logits, even for an exact hit.
  int32_t reusePrefix(SequenceId id, const std::vector<int64_t>& tokens) {
    requireIdle();
    auto& seq = sequences_.at(id);
    if (!radix_enabled_ || seq.length != 0) { throw std::logic_error("Prefix reuse requires an empty radix request"); }
    if (tokens.empty() || tokens.size() > static_cast<size_t>(max_length_)) {
      throw std::invalid_argument("Invalid prefix prompt length");
    }
    Sequence matched;
    RadixNode* node = &radix_root_;
    const size_t limit = tokens.size() - 1;
    size_t position = 0;
    while (position < limit) {
      auto it = node->children.find(tokens[position]);
      if (it == node->children.end()) { break; }
      node = it->second.get();
      size_t common = 0;
      while (common < node->edge.size() && position + common < limit && node->edge[common] == tokens[position + common]) {
        ++common;
      }
      if (common == 0) { break; }
      const size_t logical_block = position / block_size_;
      if (matched.blocks.size() == logical_block) {
        matched.blocks.push_back(node->block);
      } else {
        // A split node and its child may describe different physical copies of
        // the same logical block. The deepest matched copy contains the entire
        // block prefix; do not sample node boundaries to reconstruct block IDs.
        matched.blocks[logical_block] = node->block;
      }
      node->last_use = ++radix_clock_;
      position += common;
      if (common != node->edge.size()) { break; }
    }
    matched.length = static_cast<int32_t>(position);
    matched.tokens.assign(tokens.begin(), tokens.begin() + matched.length);
    for (auto block : matched.blocks) { ++refs_[block]; }
    seq = std::move(matched);
    return seq.length;
  }

  // LRU leaf eviction never reclaims blocks still referenced by active requests.
  // One call removes one leaf; several split nodes can retain the same block.
  // Returns false when all remaining leaves are protected.
  bool evictRadixBlock() {
    requireIdle();
    RadixNode* victim = nullptr;
    findVictim(radix_root_, victim);
    if (!victim) { return false; }
    auto* parent = victim->parent;
    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
      if (it->second.get() == victim) {
        releaseCacheRef(victim->block);
        parent->children.erase(it);
        return true;
      }
    }
    return false;
  }

  const Sequence& sequence(SequenceId id) const { return sequences_.at(id); }
  bool contains(SequenceId id) const { return sequences_.count(id) != 0; }
  size_t freeBlocks() const { return free_.size(); }
  size_t numBlocks() const { return refs_.size(); }
  int32_t blockSize() const { return block_size_; }
  int32_t maxLength() const { return max_length_; }
  size_t refCount(int32_t block) const { return refs_.at(block); }
  bool pending() const { return !pending_.empty(); }
  const std::vector<Append>& appends() const { return pending_; }
  const std::vector<std::pair<int32_t, int32_t>>& copies() const { return copies_; }

  // Reserve the entire batch before changing state. false means capacity pressure:
  // no sequence changed. Radix-only leaves may have been evicted. Never evict live KV.
  // Each shared partial tail is copied even if its other owners also append this step.
  bool beginAppend(const std::vector<SequenceId>& ids, const std::vector<int32_t>& counts,
                   const std::vector<std::vector<int64_t>>& tokens = {}) {
    requireIdle();
    if (ids.empty() || ids.size() != counts.size()) { throw std::invalid_argument("Invalid append batch"); }
    if (radix_enabled_ && tokens.size() != ids.size()) { throw std::invalid_argument("Radix append needs token IDs"); }
    std::unordered_set<SequenceId> seen;
    size_t needed = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (!seen.insert(ids[i]).second) { throw std::invalid_argument("Duplicate sequence in append batch"); }
      const auto& seq = sequence(ids[i]);
      if (counts[i] <= 0 || counts[i] > max_length_ - seq.length) {
        throw std::invalid_argument("Append exceeds sequence capacity or has no tokens");
      }
      if (radix_enabled_ && tokens[i].size() != static_cast<size_t>(counts[i])) {
        throw std::invalid_argument("Radix token count mismatch");
      }
      needed += blockCount(seq.length + counts[i]) - seq.blocks.size();
      if (seq.length % block_size_ != 0 && refs_[seq.blocks.back()] > 1) { ++needed; }
    }
    while (needed > free_.size() && evictRadixBlock()) {}
    if (needed > free_.size()) { return false; }

    // All potentially throwing allocations happen before the free list is mutated.
    std::vector<Append> plan;
    std::vector<int32_t> reserved;
    std::vector<std::pair<int32_t, int32_t>> copies;
    plan.reserve(ids.size());
    reserved.reserve(needed);
    copies.reserve(ids.size());
    auto reserve = [&]() {
      const auto p = free_[free_.size() - reserved.size() - 1];
      reserved.push_back(p);
      return p;
    };
    for (size_t i = 0; i < ids.size(); ++i) {
      const auto& seq = sequence(ids[i]);
      Append entry{ids[i], seq.length, seq, -1};
      entry.next.length += counts[i];
      if (radix_enabled_) { entry.next.tokens.insert(entry.next.tokens.end(), tokens[i].begin(), tokens[i].end()); }
      entry.next.blocks.reserve(blockCount(entry.next.length));
      if (seq.length % block_size_ != 0 && refs_[seq.blocks.back()] > 1) {
        entry.replaced_block = seq.blocks.back();
        const auto replacement = reserve();
        entry.next.blocks.back() = replacement;
        copies.emplace_back(entry.replaced_block, replacement);
      }
      while (entry.next.blocks.size() < blockCount(entry.next.length)) { entry.next.blocks.push_back(reserve()); }
      plan.push_back(std::move(entry));
    }
    for (auto block : reserved) { refs_[block] = 1; }
    free_.resize(free_.size() - needed);
    pending_ = std::move(plan);
    reserved_ = std::move(reserved);
    copies_ = std::move(copies);
    return true;
  }

  // Call only after all layers have written their new KV and device work has completed.
  void commit() {
    if (!pending()) { throw std::logic_error("No append to commit"); }
    for (auto& entry : pending_) {
      if (entry.replaced_block >= 0) { decref(entry.replaced_block); }
      sequences_.at(entry.id) = std::move(entry.next);
    }
    // Cache publication is best-effort after commit. Allocation failure must not
    // report a failed forward after sequence KV has already been committed.
    if (radix_enabled_) {
      try {
        for (const auto& entry : pending_) { publishPrefix(sequences_.at(entry.id)); }
      } catch (const std::bad_alloc&) {
        // Any nodes published before the failure are valid, fully written blocks.
      }
    }
    pending_.clear();
    reserved_.clear();
    copies_.clear();
  }
  // Writes beyond an old private tail's length become invisible. Shared tails were
  // copied, so abort cannot modify another request's committed tokens.
  void abort() noexcept {
    for (auto block : reserved_) { decref(block); }
    pending_.clear();
    reserved_.clear();
    copies_.clear();
  }

 private:
  // Token-compressed edges never cross a physical block boundary. Split nodes
  // record the block and offset of their token interval. Children may use a COW
  // copy containing the same block prefix; each node retains its own block ref.
  struct RadixNode {
    int32_t block = -1;
    int32_t offset = 0;
    std::vector<int64_t> edge;
    uint64_t last_use = 0;
    RadixNode* parent = nullptr;
    std::map<int64_t, std::unique_ptr<RadixNode>> children;
  };
  void retainCacheRef(int32_t block) noexcept {
    if (cache_refs_[block]++ == 0) { ++cached_blocks_; }
    ++refs_[block];
    ++cached_nodes_;
  }
  void releaseCacheRef(int32_t block) noexcept {
    if (--cache_refs_[block] == 0) { --cached_blocks_; }
    --cached_nodes_;
    decref(block);
  }
  // Allocate everything before changing the old edge, so an allocation failure
  // leaves the original subtree and all ownership counts intact.
  RadixNode* splitEdge(std::unique_ptr<RadixNode>& slot, size_t common) {
    auto middle = std::make_unique<RadixNode>();
    middle->block = slot->block;
    middle->offset = slot->offset;
    middle->edge.assign(slot->edge.begin(), slot->edge.begin() + common);
    middle->parent = slot->parent;
    middle->last_use = slot->last_use;
    std::vector<int64_t> suffix(slot->edge.begin() + common, slot->edge.end());
    auto child_slot = middle->children.emplace(suffix[0], nullptr).first;
    slot->edge.swap(suffix);
    slot->offset += static_cast<int32_t>(common);
    slot->parent = middle.get();
    child_slot->second = std::move(slot);
    retainCacheRef(middle->block);
    slot = std::move(middle);
    return slot.get();
  }
  // New COW storage contains the whole logical block prefix. Redirect same-block
  // ancestors to it so repeated appends do not pin one obsolete block per token.
  // Sibling leaves retain their own versions, and live requests keep their refs.
  void rebindBlockPrefix(RadixNode* child) noexcept {
    const int32_t block = child->block;
    while (child->offset > 0) {
      auto* parent = child->parent;
      if (parent->block != block) {
        retainCacheRef(block);
        releaseCacheRef(parent->block);
        parent->block = block;
      }
      child = parent;
    }
  }
  // Keep unbranched same-block paths compressed after incremental publication.
  // Prefix endpoints do not need terminal markers: every committed token prefix
  // is reusable, including one ending inside an edge.
  RadixNode* compactBlockPath(RadixNode* node) {
    while (node->offset > 0 && node->parent->children.size() == 1) {
      auto* parent = node->parent;
      std::vector<int64_t> combined = parent->edge;
      combined.insert(combined.end(), node->edge.begin(), node->edge.end());
      auto grandchildren = std::move(node->children);
      parent->edge.swap(combined);
      parent->last_use = std::max(parent->last_use, node->last_use);
      releaseCacheRef(node->block);
      parent->children = std::move(grandchildren);  // Destroys node, not its moved children.
      for (auto& item : parent->children) { item.second->parent = parent; }
      node = parent;
    }
    return node;
  }
  void publishPrefix(const Sequence& seq) {
    RadixNode* node = &radix_root_;
    size_t position = 0;
    while (position < seq.tokens.size()) {
      auto it = node->children.find(seq.tokens[position]);
      if (it == node->children.end()) {
        auto child = std::make_unique<RadixNode>();
        child->block = seq.blocks[position / block_size_];
        child->offset = static_cast<int32_t>(position % block_size_);
        const size_t count = std::min(seq.tokens.size() - position, static_cast<size_t>(block_size_ - child->offset));
        child->edge.assign(seq.tokens.begin() + position, seq.tokens.begin() + position + count);
        child->parent = node;
        const int64_t first_token = child->edge[0];
        it = node->children.emplace(first_token, std::move(child)).first;
        retainCacheRef(it->second->block);
      }
      auto* child = it->second.get();
      size_t common = 0;
      while (common < child->edge.size() && position + common < seq.tokens.size()
             && child->edge[common] == seq.tokens[position + common]) {
        ++common;
      }
      // The map key guarantees common > 0. Split at divergence or a shorter key.
      if (common < child->edge.size()) { child = splitEdge(it->second, common); }
      position += common;
      node = child;
      node->last_use = ++radix_clock_;
      rebindBlockPrefix(node);
      node = compactBlockPath(node);
    }
  }
  void findVictim(RadixNode& node, RadixNode*& victim) {
    for (auto& item : node.children) {
      auto& child = *item.second;
      if (child.children.empty() && refs_[child.block] == cache_refs_[child.block]
          && (!victim || child.last_use < victim->last_use)) {
        victim = &child;
      }
      findVictim(child, victim);
    }
  }
  void dropRadix(RadixNode& node) {
    for (auto& item : node.children) {
      dropRadix(*item.second);
      releaseCacheRef(item.second->block);
    }
    node.children.clear();
  }
  bool radix_enabled_ = false;
  size_t cached_blocks_ = 0, cached_nodes_ = 0;
  uint64_t radix_clock_ = 0;
  RadixNode radix_root_;
  size_t blockCount(int32_t length) const { return (static_cast<int64_t>(length) + block_size_ - 1) / block_size_; }
  void requireIdle() const {
    if (pending()) { throw std::logic_error("Paged KV append is still in flight"); }
  }
  void decref(int32_t block) noexcept {
    if (--refs_[block] == 0) { free_.push_back(block); }
  }
  int32_t block_size_, max_length_;
  std::vector<size_t> refs_, cache_refs_;
  std::vector<int32_t> free_;
  std::unordered_map<SequenceId, Sequence> sequences_;
  std::vector<Append> pending_;
  std::vector<int32_t> reserved_;
  std::vector<std::pair<int32_t, int32_t>> copies_;
};

}  // namespace mllm::paged_attention
