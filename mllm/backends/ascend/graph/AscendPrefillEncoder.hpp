// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <cmath>
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"

namespace mllm::ascend {
// Single-request, zero-history causal prefill. KV ownership/writes remain
// with AscendPagedAttentionPlugin; this operation computes only attention.
class AscendPrefillEncoder {
 public:
  AscendPrefillEncoder() = default;
  AscendPrefillEncoder(const AscendPrefillEncoder&) = delete;
  AscendPrefillEncoder& operator=(const AscendPrefillEncoder&) = delete;
  ~AscendPrefillEncoder() { if (op_) atb::DestroyOperation(op_); }
  void prepare(int tokens, int heads, int kv_heads, int dim) {
    if (op_ && tokens_==tokens) return;
    if (op_) { atb::DestroyOperation(op_); op_=nullptr; }
    tokens_=tokens; heads_=heads; kv_heads_=kv_heads; dim_=dim;
    auto mask=Tensor::empty({1,tokens,tokens},kFloat16,kCPU).alloc();
    auto* values=mask.ptr<mllm_fp16_t>();
    for(int i=0;i<tokens;++i) for(int j=0;j<tokens;++j)
      values[i*tokens+j]=static_cast<mllm_fp16_t>(j>i ? -10000.f : 0.f);
    mask_=toNZ(mask.to(kAscend));
    lengths_=Tensor::fromVector(std::vector<int32_t>{tokens},{1},kInt32,kCPU).to(kAscend);
    atb::infer::SelfAttentionParam p;
    p.calcType=atb::infer::SelfAttentionParam::PA_ENCODER;
    p.headNum=heads; p.kvHeadNum=kv_heads; p.qkScale=1.f/std::sqrt(float(dim));
    p.maskType=atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    p.isTriuMask=1;
    MLLM_ATB_CHECK(atb::CreateOperation(p,&op_));
  }
  atb::VariantPack pack(const atb::VariantPack& parent) {
    atb::VariantPack result;
    for(int i=0;i<3;++i) {
      auto t=parent.inTensors[i];t.desc.shape.dimNum=3;
      t.desc.shape.dims[0]=tokens_;t.desc.shape.dims[1]=(i==0?heads_:kv_heads_);
      t.desc.shape.dims[2]=dim_;
      result.inTensors.push_back(t);
    }
    atb::Tensor mask,lengths;fillAtbTensor(mask_,mask);mask.desc.format=ACL_FORMAT_FRACTAL_NZ;
    fillAtbTensor(lengths_,lengths);lengths.hostData=&tokens_;
    result.inTensors.push_back(mask);result.inTensors.push_back(lengths);
    auto out=parent.outTensors[0];out.desc=result.inTensors[0].desc;
    result.outTensors.push_back(out);return result;
  }
  atb::Status setup(const atb::VariantPack& parent,uint64_t& bytes,atb::Context* ctx) {
    auto p=pack(parent);return op_->Setup(p,bytes,ctx);
  }
  atb::Status execute(const atb::VariantPack& parent,uint8_t* workspace,uint64_t bytes,atb::Context* ctx) {
    auto p=pack(parent);return op_->Execute(p,workspace,bytes,ctx);
  }
 private:
  int32_t tokens_=0,heads_=0,kv_heads_=0,dim_=0;
  Tensor mask_,lengths_;
  atb::Operation* op_=nullptr;
};
}
