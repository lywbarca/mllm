// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <cstdlib>
#include <cstring>
#include <atb/infer_op_params.h>
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {
inline bool useNZWeights() {
  const char* flag = std::getenv("MLLM_ASCEND_NZ_WEIGHTS");
  const char* soc = aclrtGetSocName();
  return (!flag || flag[0] != '0') && soc && std::strstr(soc, "310P");
}
// Operation and temporary workspace are released only after the stream completes.
inline void executePreparedOp(atb::Operation* op, atb::VariantPack& pack) {
  uint64_t bytes = 0;
  MLLM_ATB_CHECK(op->Setup(pack, bytes, getGlobalAtbContext()));
  void* workspace = nullptr;
  if (bytes) MLLM_ACL_CHECK(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  MLLM_ATB_CHECK(op->Execute(pack, static_cast<uint8_t*>(workspace), bytes, getGlobalAtbContext()));
  syncGlobalAtbStream();
  if (workspace) MLLM_ACL_CHECK(aclrtFree(workspace));
}
inline Tensor toNZ(const Tensor& source) {
  MLLM_RT_ASSERT(source.device() == kAscend && (source.dtype() == kFloat16 || source.dtype() == kInt8));
  atb::infer::TransdataParam param;
  param.transdataType = atb::infer::TransdataParam::ND_TO_FRACTAL_NZ;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
  atb::Tensor input;
  fillAtbTensor(source, input);
  atb::SVector<atb::TensorDesc> in{input.desc}, out{atb::TensorDesc{}};
  MLLM_ATB_CHECK(op->InferShape(in, out));
  std::vector<int32_t> shape;
  for (uint64_t i=0;i<out[0].shape.dimNum;++i) shape.push_back(out[0].shape.dims[i]);
  auto result = Tensor::empty(shape, source.dtype(), kAscend).alloc();
  atb::Tensor output;
  fillAtbTensor(result, output);
  output.desc = out[0];
  atb::VariantPack pack;
  pack.inTensors = {input}; pack.outTensors = {output};
  executePreparedOp(op, pack);
  atb::DestroyOperation(op);
  return result;
}
// Keep the canonical ND parameter for serialization/device migration.
// Only the backend's execution artifact has a physical NZ shape.
inline Tensor packLinearWeightNZ(const Tensor& source) {
  if (!useNZWeights() || source.isNil() || source.device()!=kAscend || (source.dtype()!=kFloat16 && source.dtype()!=kInt8) ||
      source.rank()!=2 || source.size(0)%16 || source.size(1)%(source.dtype()==kInt8 ? 32 : 16)) return Tensor::nil();
  auto view_source=source; return toNZ(view_source.view({1,source.size(0),source.size(1)}));
}
// Concatenate output channels once, then keep only the execution artifact.
class AscendPackedProjection {
 public:
  Tensor prepare(const std::vector<Tensor>& weights) {
    bool same = sources_.size()==weights.size() && !packed_.isNil();
    for (size_t i=0;same && i<weights.size();++i)
      same = sources_[i].ptr<void>()==weights[i].ptr<void>();
    if (same) return packed_;
    int rows=0, cols=weights.at(0).size(1);
    for (const auto& w:weights) {
      MLLM_RT_ASSERT(w.rank()==2 && w.size(1)==cols && w.dtype()==kFloat16 && w.device()==kAscend);
      rows+=w.size(0);
    }
    auto joined=Tensor::empty({rows,cols},kFloat16,kAscend).alloc();
    size_t offset=0;
    for (const auto& w:weights) {
      size_t bytes=static_cast<size_t>(w.size(0))*cols*sizeof(mllm_fp16_t);
      MLLM_ACL_CHECK(aclrtMemcpy(static_cast<uint8_t*>(joined.ptr<void>())+offset,bytes,
                               w.ptr<void>(),bytes,ACL_MEMCPY_DEVICE_TO_DEVICE));
      offset+=bytes;
    }
    packed_=packLinearWeightNZ(joined);
    sources_=weights; // Retain source identity so allocator reuse cannot hide a reload.
    return packed_;
  }
 private:
  std::vector<Tensor> sources_;
  Tensor packed_;
};
inline Tensor linearWithNZWeight(const Tensor& x, const Tensor& packed, int out_features) {
  auto shape=x.shape(); shape.back()=out_features;
  auto y=Tensor::empty(shape,x.dtype(),x.device()).alloc();
  atb::infer::LinearParam param;
  param.transposeA=false; param.transposeB=true; param.hasBias=false;
  atb::Operation* op=nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param,&op));
  atb::Tensor a,b,c;
  fillAtbTensor(x,a); fillAtbTensor(packed,b); fillAtbTensor(y,c);
  b.desc.format=ACL_FORMAT_FRACTAL_NZ;
  atb::VariantPack pack; pack.inTensors={a,b}; pack.outTensors={c};
  executePreparedOp(op,pack);
  atb::DestroyOperation(op);
  return y;
}
// Persistent output projection resources. Execute remains synchronous because
// callers consume logits on the CPU; Setup refreshes all shapes/pointers.
class AscendNZLinearRunner {
 public:
  AscendNZLinearRunner() = default;
  ~AscendNZLinearRunner() {
    if (workspace_) aclrtFree(workspace_);
    if (op_) atb::DestroyOperation(op_);
  }
  AscendNZLinearRunner(const AscendNZLinearRunner&) = delete;
  AscendNZLinearRunner& operator=(const AscendNZLinearRunner&) = delete;
  uint64_t operationCreations() const { return operation_creations_; }
  uint64_t workspaceAllocations() const { return workspace_allocations_; }
  Tensor execute(const Tensor& x, const Tensor& packed, int out_features) {
    const char* flag=std::getenv("MLLM_ASCEND_REUSE_OUTPUT_LINEAR");
    if (flag && flag[0]=='0') return linearWithNZWeight(x,packed,out_features);
    if (!op_) {
      atb::infer::LinearParam param;
      param.transposeA=false; param.transposeB=true; param.hasBias=false;
      MLLM_ATB_CHECK(atb::CreateOperation(param,&op_));
      ++operation_creations_;
    }
    auto shape=x.shape();shape.back()=out_features;
    auto y=Tensor::empty(shape,x.dtype(),x.device()).alloc();
    atb::Tensor a,b,c;
    fillAtbTensor(x,a);fillAtbTensor(packed,b);fillAtbTensor(y,c);
    b.desc.format=ACL_FORMAT_FRACTAL_NZ;
    atb::VariantPack pack;pack.inTensors={a,b};pack.outTensors={c};
    uint64_t required=0;
    MLLM_ATB_CHECK(op_->Setup(pack,required,getGlobalAtbContext()));
    if (required>workspace_size_) {
      if (workspace_) { MLLM_ACL_CHECK(aclrtFree(workspace_));workspace_=nullptr; }
      MLLM_ACL_CHECK(aclrtMalloc(&workspace_,required,ACL_MEM_MALLOC_HUGE_FIRST));
      workspace_size_=required;++workspace_allocations_;
    }
    MLLM_ATB_CHECK(op_->Execute(pack,static_cast<uint8_t*>(workspace_),workspace_size_,getGlobalAtbContext()));
    syncGlobalAtbStream();
    return y;
  }
 private:
  atb::Operation* op_=nullptr;
  void* workspace_=nullptr;
  uint64_t workspace_size_=0,operation_creations_=0,workspace_allocations_=0;
};

}
