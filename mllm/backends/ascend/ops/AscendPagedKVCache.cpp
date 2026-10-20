// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendPagedKVCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <atb/infer_op_params.h>
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

namespace mllm::ascend {
namespace {

void copyDevice(void* dst, const void* src, size_t bytes) {
  const auto status = aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
  MLLM_ACL_CHECK(status);
}

Tensor upload(const std::vector<int32_t>& values, const Tensor::shape_t& shape) {
  auto tensor = Tensor::empty(shape, kInt32, kAscend).alloc();
  const auto status = aclrtMemcpy(tensor.ptr<void>(), tensor.bytes(), values.data(), values.size() * sizeof(int32_t),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
  MLLM_ACL_CHECK(status);
  return tensor;
}

void reshapeAndCache(atb::VariantPack& pack) {
  // Default K_CACHE_V_CACHE, COMPRESS_TYPE_UNDEFINED. On 310P the cache
  // descriptors carry the physical NZ shape, as in ATB's inference demo.
  atb::infer::ReshapeAndCacheParam param;
  atb::Operation* op = nullptr;
  auto status = atb::CreateOperation(param, &op);
  MLLM_ATB_CHECK(status);
  MLLM_RT_ASSERT(op != nullptr);
  auto* context = getGlobalAtbContext();
  uint64_t workspace_size = 0;
  status = op->Setup(pack, workspace_size, context);
  MLLM_ATB_CHECK(status);
  MLLM_RT_ASSERT(workspace_size <= std::numeric_limits<uint32_t>::max() - 64ULL);
  void* workspace = nullptr;
  int block = -1;
  auto& manager = getAscendMemoryManager();
  if (workspace_size) {
    manager.allocateBlock(static_cast<uint32_t>(workspace_size), block);
    MLLM_RT_ASSERT(block >= 0);
    manager.getBlockPtr(block, workspace);
    MLLM_RT_ASSERT(workspace != nullptr);
  }
  status = op->Execute(pack, reinterpret_cast<uint8_t*>(workspace), workspace_size, context);
  MLLM_ATB_CHECK(status);
  syncGlobalAtbStream();
  if (block >= 0) { manager.freeBlock(block); }
  status = atb::DestroyOperation(op);
  MLLM_ATB_CHECK(status);
}

}  // namespace

AscendPagedKVCacheOptions AscendPagedKVCache::validate(AscendPagedKVCacheOptions options) {
  if (options.block_size <= 0 || options.block_size % 16 != 0 || options.num_blocks <= 0 || options.max_sequence_length <= 0
      || options.layer_nums <= 0 || options.kv_heads <= 0 || options.head_dim <= 0 || options.head_dim > 256) {
    throw std::invalid_argument("Invalid Ascend paged KV dimensions");
  }
  if (options.layout != aops::PagedAttnCacheLayout::kND && options.layout != aops::PagedAttnCacheLayout::kNZ) {
    throw std::invalid_argument("Invalid Ascend paged KV layout");
  }
  if (options.layout == aops::PagedAttnCacheLayout::kNZ
      && (options.head_dim % 16 || static_cast<int64_t>(options.block_size) * options.head_dim > 16384)) {
    throw std::invalid_argument("NZ requires D%16=0 and block_size*D<=16384");
  }
  // The allocator/memory pool accepts uint32 byte counts. Check each tensor
  // before multiplication or allocation, including its alignment overhead.
  uint64_t bytes = sizeof(mllm_fp16_t);
  for (auto dim : {options.num_blocks, options.block_size, options.kv_heads, options.head_dim}) {
    if (bytes > (std::numeric_limits<uint32_t>::max() - 64ULL) / dim) {
      throw std::invalid_argument("A paged KV layer tensor exceeds the Ascend allocator's uint32 capacity");
    }
    bytes *= dim;
  }
  return options;
}

AscendPagedKVCache::AscendPagedKVCache(const AscendPagedKVCacheOptions& options)
    : options_(validate(options)),
      blocks_(options_.num_blocks, options_.block_size, options_.max_sequence_length),
      written_(options_.layer_nums, false) {
  if (options_.radix_cache) { blocks_.enableRadixCache(); }
  const auto& o = options_;
  const auto shape = o.layout == aops::PagedAttnCacheLayout::kNZ
                         ? Tensor::shape_t{o.num_blocks, o.kv_heads * o.head_dim / 16, o.block_size, 16}
                         : Tensor::shape_t{o.num_blocks, o.block_size, o.kv_heads, o.head_dim};
  k_pool_.reserve(o.layer_nums);
  v_pool_.reserve(o.layer_nums);
  for (int layer = 0; layer < o.layer_nums; ++layer) {
    // Only committed/written slots are readable, so zeroing the full pool is unnecessary.
    k_pool_.push_back(Tensor::empty(shape, kFloat16, kAscend).alloc());
    v_pool_.push_back(Tensor::empty(shape, kFloat16, kAscend).alloc());
  }
}

uint64_t AscendPagedKVCache::poolBytes() const {
  return static_cast<uint64_t>(options_.num_blocks) * options_.block_size * options_.kv_heads * options_.head_dim
         * sizeof(mllm_fp16_t) * 2 * options_.layer_nums;
}

bool AscendPagedKVCache::beginStep(const std::vector<SequenceId>& ids, int32_t token_count,
                                   const std::vector<std::vector<int64_t>>& tokens) {
  if (ids.empty() || ids.size() > 2048 || token_count <= 0
      || static_cast<uint64_t>(ids.size()) * token_count > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("Invalid paged KV tensor batch");
  }
  if (!blocks_.beginAppend(ids, std::vector<int32_t>(ids.size(), token_count), tokens)) { return false; }
  try {
    step_tokens_ = token_count;
    const auto& appends = blocks_.appends();
    size_t width = 1;
    for (const auto& entry : appends) { width = std::max(width, entry.next.blocks.size()); }
    if (width * ids.size() > (std::numeric_limits<uint32_t>::max() - 64ULL) / sizeof(int32_t)
        || ids.size() * static_cast<uint64_t>(token_count) > (std::numeric_limits<uint32_t>::max() - 64ULL) / sizeof(int32_t)) {
      throw std::invalid_argument("Paged metadata exceeds the Ascend allocator capacity");
    }
    std::vector<int32_t> table(ids.size() * width, 0), lengths, slots;
    lengths.reserve(ids.size());
    slots.reserve(ids.size() * token_count);
    for (size_t b = 0; b < appends.size(); ++b) {
      const auto& entry = appends[b];
      std::copy(entry.next.blocks.begin(), entry.next.blocks.end(), table.begin() + b * width);
      lengths.push_back(entry.next.length);
      for (int s = entry.old_length; s < entry.next.length; ++s) {
        slots.push_back(entry.next.blocks[s / options_.block_size] * options_.block_size + s % options_.block_size);
      }
    }
    block_table_ = upload(table, {static_cast<int32_t>(ids.size()), static_cast<int32_t>(width)});
    slot_mapping_ = upload(slots, {static_cast<int32_t>(slots.size())});
    context_lengths_ = Tensor::fromVector(lengths, {static_cast<int32_t>(ids.size())}, kInt32, kCPU);

    // COW copies include every layer. The source remains referenced until commit.
    syncGlobalAtbStream();
    const size_t block_bytes =
        static_cast<size_t>(options_.block_size) * options_.kv_heads * options_.head_dim * sizeof(mllm_fp16_t);
    for (const auto& [source, destination] : blocks_.copies()) {
      for (int layer = 0; layer < options_.layer_nums; ++layer) {
        for (const auto& pool : {k_pool_[layer], v_pool_[layer]}) {
          auto* data = static_cast<char*>(pool.ptr<void>());
          copyDevice(data + destination * block_bytes, data + source * block_bytes, block_bytes);
        }
      }
    }
    return true;
  } catch (...) {
    abortStep();
    throw;
  }
}

void AscendPagedKVCache::writeLayer(int32_t layer, const Tensor& k, const Tensor& v) {
  if (!pending() || layer < 0 || layer >= options_.layer_nums || written_[layer]) {
    throw std::logic_error("Paged KV layer write requires one prepared, unwritten layer");
  }
  const Tensor::shape_t expected{static_cast<int32_t>(blocks_.appends().size()), step_tokens_, options_.kv_heads,
                                 options_.head_dim};
  for (const auto& tensor : {k, v}) {
    MLLM_RT_ASSERT(!tensor.isNil() && tensor.shape() == expected);
    MLLM_RT_ASSERT(tensor.dtype() == kFloat16 && tensor.device() == kAscend && tensor.isContiguous());
  }
  atb::VariantPack pack;
  const std::array<Tensor, 5> inputs{k, v, k_pool_[layer], v_pool_[layer], slot_mapping_};
  for (size_t i = 0; i < inputs.size(); ++i) {
    atb::Tensor t{};
    fillAtbTensor(inputs[i], t);
    if (i < 2) {
      t.desc.shape.dimNum = 3;
      t.desc.shape.dims[0] = static_cast<int64_t>(expected[0]) * step_tokens_;
      t.desc.shape.dims[1] = options_.kv_heads;
      t.desc.shape.dims[2] = options_.head_dim;
    } else if (i < 4 && options_.layout == aops::PagedAttnCacheLayout::kNZ) {
      t.desc.format = ACL_FORMAT_FRACTAL_NZ;
    }
    pack.inTensors.push_back(t);
  }
  pack.outTensors.push_back(pack.inTensors[2]);
  pack.outTensors.push_back(pack.inTensors[3]);
  reshapeAndCache(pack);
  written_[layer] = true;
}

std::array<Tensor, 2> AscendPagedKVCache::gatherLayer(int32_t layer, size_t batch_index) const {
  if (!pending() || layer < 0 || layer >= options_.layer_nums || !written_[layer]) {
    throw std::logic_error("Cannot read unwritten paged KV");
  }
  const auto& seq = blocks_.appends().at(batch_index).next;
  const auto& o = options_;
  const bool nz = o.layout == aops::PagedAttnCacheLayout::kNZ;
  const int chunk = nz ? 16 : o.head_dim;
  const Tensor::shape_t shape{1, o.kv_heads, seq.length, o.head_dim};
  std::array<Tensor, 2> result{Tensor::empty(shape, kFloat16, kAscend).alloc(),
                               Tensor::empty(shape, kFloat16, kAscend).alloc()};
  syncGlobalAtbStream();
  for (int s = 0; s < seq.length; ++s) {
    const size_t block = seq.blocks[s / o.block_size];
    const int slot = s % o.block_size;
    for (int h = 0; h < o.kv_heads; ++h) {
      for (int d = 0; d < o.head_dim; d += chunk) {
        const size_t src =
            nz ? ((block * (o.kv_heads * o.head_dim / 16) + (h * o.head_dim + d) / 16) * o.block_size + slot) * 16
               : ((block * o.block_size + slot) * o.kv_heads + h) * o.head_dim + d;
        const size_t dst = (static_cast<size_t>(h) * seq.length + s) * o.head_dim + d;
        for (int kv = 0; kv < 2; ++kv) {
          const auto& pool = kv == 0 ? k_pool_[layer] : v_pool_[layer];
          copyDevice(result[kv].ptr<mllm_fp16_t>() + dst, pool.ptr<mllm_fp16_t>() + src, chunk * sizeof(mllm_fp16_t));
        }
      }
    }
  }
  return result;
}

void AscendPagedKVCache::prepareGraphMetadata() {
  if (!pending()) { throw std::logic_error("Graph metadata requires a pending append"); }
  if (!graph_table_.isNil()) { return; }
  const auto& appends = blocks_.appends();
  int width = block_table_.size(1);
  std::vector<int32_t> table, lengths;
  for (const auto& entry : appends) {
    for (int s = 0; s < step_tokens_; ++s) {
      size_t row = table.size();
      table.resize(row + width, 0);
      std::copy(entry.next.blocks.begin(), entry.next.blocks.end(), table.begin() + row);
      lengths.push_back(entry.old_length + s + 1);
    }
  }
  graph_table_ = upload(table, {static_cast<int32_t>(lengths.size()), width});
  graph_lengths_host_ = Tensor::fromVector(lengths, {static_cast<int32_t>(lengths.size())}, kInt32, kCPU);
  graph_lengths_device_ = upload(lengths, {static_cast<int32_t>(lengths.size())});
}

void AscendPagedKVCache::markGraphLayerWritten(int32_t layer) {
  if (!pending() || layer < 0 || layer >= options_.layer_nums || written_[layer]) {
    throw std::logic_error("Graph KV write requires one pending unwritten layer");
  }
  written_[layer] = true;
}

void AscendPagedKVCache::resetStepTensors() {
  graph_table_ = graph_lengths_host_ = graph_lengths_device_ = Tensor::nil();
  block_table_ = context_lengths_ = slot_mapping_ = Tensor::nil();
  step_tokens_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

void AscendPagedKVCache::commitStep() {
  if (!pending() || std::find(written_.begin(), written_.end(), false) != written_.end()) {
    throw std::logic_error("Cannot commit paged KV before every layer is written");
  }
  syncGlobalAtbStream();
  blocks_.commit();
  resetStepTensors();
}

void AscendPagedKVCache::abortStep() {
  syncGlobalAtbStream();
  blocks_.abort();
  resetStepTensors();
}

void AscendPagedKVCache::clear() {
  if (pending()) { throw std::logic_error("Cannot clear an in-flight paged KV step"); }
  syncGlobalAtbStream();
  blocks_.clearSequences();
  resetStepTensors();
}

}  // namespace mllm::ascend
