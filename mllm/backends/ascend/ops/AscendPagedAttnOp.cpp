// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendPagedAttnOp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <atb/infer_op_params.h>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

namespace mllm::ascend {

AscendPagedAttnOp::AscendPagedAttnOp(const aops::PagedAttnOpOptions& options) : aops::PagedAttnOp(options) {
  if (options_.impl_type != aops::PagedAttnImplType::kDefault || options_.need_attn_weights || options_.fuse_rope
      || options_.high_precision_exp || options_.prefix_cache_ctx != nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendPagedAttnOp requires kDefault, no attention weights, no fused RoPE, "
                                            "no high_precision_exp and no prefix_cache_ctx");
  }
  MLLM_RT_ASSERT(options_.head_repeat_times > 0);
  MLLM_RT_ASSERT(options_.kv_cache_layout == aops::PagedAttnCacheLayout::kND
                 || options_.kv_cache_layout == aops::PagedAttnCacheLayout::kNZ);
}

void AscendPagedAttnOp::validateInputs(const std::vector<Tensor>& inputs) const {
  MLLM_RT_ASSERT_EQ(inputs.size(), 5);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& t = inputs[i];
    MLLM_RT_ASSERT(!t.isNil());
    MLLM_RT_ASSERT(t.isContiguous());
    MLLM_RT_ASSERT(t.device() == (i == 4 ? kCPU : kAscend));
    MLLM_RT_ASSERT(t.dtype() == (i < 3 ? kFloat16 : kInt32));
    for (const auto dim : t.shape()) { MLLM_RT_ASSERT(dim > 0); }
  }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& table = inputs[3];
  const auto& lengths = inputs[4];
  MLLM_RT_ASSERT_EQ(q.rank(), 4);
  // Multi-token chunks currently belong to one request; decode may batch requests.
  MLLM_RT_ASSERT(q.size(1) == 1 || q.size(0) == 1);
  MLLM_RT_ASSERT_EQ(k.rank(), 4);
  MLLM_RT_ASSERT(k.shape() == v.shape());
  MLLM_RT_ASSERT_EQ(table.rank(), 2);
  MLLM_RT_ASSERT_EQ(lengths.rank(), 1);
  MLLM_RT_ASSERT_EQ(table.size(0), q.size(0));
  MLLM_RT_ASSERT_EQ(lengths.size(0), q.size(0));

  const int heads = q.size(2);
  const int dim = q.size(3);
  MLLM_RT_ASSERT_EQ(heads % options_.head_repeat_times, 0);
  const int kv_heads = heads / options_.head_repeat_times;
  MLLM_RT_ASSERT(dim <= 256);
  const bool nz = options_.kv_cache_layout == aops::PagedAttnCacheLayout::kNZ;
  const int block_size = k.size(nz ? 2 : 1);
  MLLM_RT_ASSERT_EQ(block_size % 16, 0);
  if (nz) {
    MLLM_RT_ASSERT_EQ(dim % 16, 0);
    MLLM_RT_ASSERT_EQ(k.size(1), static_cast<int64_t>(kv_heads) * dim / 16);
    MLLM_RT_ASSERT_EQ(k.size(3), 16);
    MLLM_RT_ASSERT(q.size(0) <= 2048);
    MLLM_RT_ASSERT(static_cast<int64_t>(block_size) * dim <= 128 * 128);
  } else {
    MLLM_RT_ASSERT_EQ(k.size(2), kv_heads);
    MLLM_RT_ASSERT_EQ(k.size(3), dim);
  }
}

void AscendPagedAttnOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  validateInputs(inputs);
  outputs.emplace_back(Tensor::empty(inputs[0].shape(), kFloat16, kAscend));
  outputs.emplace_back(Tensor::nil());
}

void AscendPagedAttnOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendPagedAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  validateInputs(inputs);
  MLLM_RT_ASSERT_EQ(outputs.size(), 2);
  MLLM_RT_ASSERT(!outputs[0].isNil());
  MLLM_RT_ASSERT(outputs[1].isNil());
  MLLM_RT_ASSERT(outputs[0].shape() == inputs[0].shape());
  MLLM_RT_ASSERT(outputs[0].dtype() == kFloat16 && outputs[0].device() == kAscend);
  MLLM_RT_ASSERT(outputs[0].isContiguous());
  for (const auto& t : inputs) { MLLM_RT_ASSERT(t.ptr<void>() != nullptr); }
  MLLM_RT_ASSERT(outputs[0].ptr<void>() != nullptr);

  const auto& q = inputs[0];
  const auto& lengths = inputs[4];
  const bool nz = options_.kv_cache_layout == aops::PagedAttnCacheLayout::kNZ;
  const int block_size = inputs[1].size(nz ? 2 : 1);
  const auto* lens = lengths.ptr<int32_t>();
  for (int b = 0; b < q.size(0); ++b) {
    MLLM_RT_ASSERT(lens[b] >= q.size(1));
    const int64_t needed = (static_cast<int64_t>(lens[b]) + block_size - 1) / block_size;
    MLLM_RT_ASSERT(needed <= inputs[3].size(1));
  }

  atb::infer::PagedAttentionParam param;
  param.headNum = q.size(2);
  param.kvHeadNum = q.size(2) / options_.head_repeat_times;
  param.qkScale = 1.f / std::sqrt(static_cast<float>(q.size(3)));
  // Default parameters select unquantized, unmasked, single-token decode.
  atb::Context* context = getGlobalAtbContext();
  atb::Operation* op = nullptr;
  auto status = atb::CreateOperation(param, &op);
  MLLM_ATB_CHECK(status);
  MLLM_RT_ASSERT(op != nullptr);

  // Treat each chunk query as a one-token logical request sharing the same
  // physical pages. Query i sees exactly history+i+1 KV tokens. No KV gather
  // or dense mask is needed, and no speculative-decoding API is required.
  const bool prefill = q.size(1) > 1;
  const int total_queries = prefill ? q.size(1) : q.size(0);
  const int tile_capacity = prefill ? std::min(total_queries, 128) : total_queries;
  const uint64_t query_bytes = static_cast<uint64_t>(q.size(2)) * q.size(3) * sizeof(uint16_t);
  Tensor repeated_table;
  if (prefill) {
    const uint64_t table_bytes = static_cast<uint64_t>(tile_capacity) * inputs[3].bytes();
    MLLM_RT_ASSERT(table_bytes <= std::numeric_limits<uint32_t>::max() - 64ULL);
    repeated_table = Tensor::empty({tile_capacity, inputs[3].size(1)}, kInt32, kAscend).alloc();
    syncGlobalAtbStream();
    auto copy_status = aclrtMemcpy(repeated_table.ptr<void>(), repeated_table.bytes(), inputs[3].ptr<void>(), inputs[3].bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE);
    MLLM_ACL_CHECK(copy_status);
    // Double the initialized rows on device; copies never overlap.
    for (int filled = 1; filled < tile_capacity;) {
      const int rows = std::min(filled, tile_capacity - filled);
      const uint64_t offset = static_cast<uint64_t>(filled) * inputs[3].bytes();
      copy_status =
          aclrtMemcpy(static_cast<uint8_t*>(repeated_table.ptr<void>()) + offset, repeated_table.bytes() - offset,
                      repeated_table.ptr<void>(), static_cast<uint64_t>(rows) * inputs[3].bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
      MLLM_ACL_CHECK(copy_status);
      filled += rows;
    }
  }

  for (int start = 0; start < total_queries;) {
    const int count = std::min(tile_capacity, total_queries - start);
    Tensor tile_lengths = lengths;
    if (prefill) {
      tile_lengths = Tensor::empty({count}, kInt32, kCPU).alloc();
      const int history = lens[0] - q.size(1);
      for (int i = 0; i < count; ++i) { tile_lengths.ptr<int32_t>()[i] = history + start + i + 1; }
    }
    atb::VariantPack pack;
    for (size_t i = 0; i < 4; ++i) {
      atb::Tensor t{};
      fillAtbTensor(prefill && i == 3 ? repeated_table : inputs[i], t);
      if (i == 0) {
        t.desc.shape.dimNum = 3;
        t.desc.shape.dims[0] = count;
        t.desc.shape.dims[1] = q.size(2);
        t.desc.shape.dims[2] = q.size(3);
        t.deviceData = static_cast<uint8_t*>(q.ptr<void>()) + static_cast<uint64_t>(start) * query_bytes;
        t.dataSize = static_cast<uint64_t>(count) * query_bytes;
      } else if (nz && (i == 1 || i == 2)) {
        t.desc.format = ACL_FORMAT_FRACTAL_NZ;
      } else if (prefill && i == 3) {
        t.desc.shape.dims[0] = count;
        t.dataSize = static_cast<uint64_t>(count) * inputs[3].bytes();
      }
      pack.inTensors.push_back(t);
    }
    atb::Tensor atb_lengths{};
    fillAtbTensorDesc(tile_lengths, atb_lengths.desc);
    atb_lengths.hostData = tile_lengths.ptr<void>();
    atb_lengths.dataSize = tile_lengths.bytes();
    Tensor device_lengths;
    if (nz) {
      device_lengths = Tensor::empty(tile_lengths.shape(), kInt32, kAscend).alloc();
      const auto copy_status = aclrtMemcpy(device_lengths.ptr<void>(), device_lengths.bytes(), tile_lengths.ptr<void>(),
                                           tile_lengths.bytes(), ACL_MEMCPY_HOST_TO_DEVICE);
      MLLM_ACL_CHECK(copy_status);
      atb_lengths.deviceData = device_lengths.ptr<void>();
    }
    pack.inTensors.push_back(atb_lengths);
    atb::Tensor out{};
    fillAtbTensor(outputs[0], out);
    out.desc.shape = pack.inTensors[0].desc.shape;
    out.deviceData = static_cast<uint8_t*>(outputs[0].ptr<void>()) + static_cast<uint64_t>(start) * query_bytes;
    out.dataSize = static_cast<uint64_t>(count) * query_bytes;
    pack.outTensors.push_back(out);
    uint64_t workspace_size = 0;
    status = op->Setup(pack, workspace_size, context);
    MLLM_ATB_CHECK(status);
    // The memory pool takes uint32_t and adds alignment padding internally.
    MLLM_RT_ASSERT(workspace_size <= std::numeric_limits<uint32_t>::max() - 64ULL);
    void* workspace = nullptr;
    int workspace_block_id = -1;
    auto& mem_mgr = getAscendMemoryManager();
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), workspace_block_id);
      MLLM_RT_ASSERT(workspace_block_id >= 0);
      mem_mgr.getBlockPtr(workspace_block_id, workspace);
      MLLM_RT_ASSERT(workspace != nullptr);
    }
    status = op->Execute(pack, reinterpret_cast<uint8_t*>(workspace), workspace_size, context);
    MLLM_ATB_CHECK(status);
    syncGlobalAtbStream();
    if (workspace_block_id >= 0) { mem_mgr.freeBlock(workspace_block_id); }
    start += count;
  }

  status = atb::DestroyOperation(op);
  MLLM_ATB_CHECK(status);
}

}  // namespace mllm::ascend
