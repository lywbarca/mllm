// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <atb/operation_infra.h>
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendPrefillEncoder.hpp"

namespace mllm::ascend {
// Inputs: Q/K/V [B,S,H,D], K/V pools, slots, per-query block table,
// causal lengths (host), causal lengths (device).
// The enclosing decoder graph owns workspace. Deferred submission retains
// graph tensors until the model-step scope synchronizes the stream.
class AscendPagedAttentionPlugin final : public atb::OperationInfra {
 public:
  AscendPagedAttentionPlugin(int heads, int kv_heads, int dim, bool nz)
      : heads_(heads), kv_heads_(kv_heads), dim_(dim), nz_(nz) {
    atb::infer::ReshapeAndCacheParam param;
    MLLM_ATB_CHECK(atb::CreateOperation(param, &writer_));
  }
  ~AscendPagedAttentionPlugin() override {
    for (auto* op : attention_) { atb::DestroyOperation(op); }
    atb::DestroyOperation(writer_);
  }
  std::string GetName() const override { return "PagedKVWriteAndAttention"; }
  uint32_t GetInputNum() const override { return 9; }
  uint32_t GetOutputNum() const override { return 1; }
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& in,
                         atb::SVector<atb::TensorDesc>& out) const override {
    if (in.size() != GetInputNum() || out.size() != GetOutputNum()) return atb::ERROR_INVALID_PARAM;
    out[0] = in[0];
    return atb::NO_ERROR;
  }
  atb::VariantPack writePack(const atb::VariantPack& pack) const {
    atb::VariantPack result;
    for (int i : {1,2,3,4,5}) result.inTensors.push_back(pack.inTensors[i]);
    for (int i=0;i<2;++i) {
      auto& shape=result.inTensors[i].desc.shape;
      shape.dimNum=3; shape.dims[0]=queries_; shape.dims[1]=kv_heads_; shape.dims[2]=dim_;
    }
    if (nz_) {
      result.inTensors[2].desc.format=ACL_FORMAT_FRACTAL_NZ;
      result.inTensors[3].desc.format=ACL_FORMAT_FRACTAL_NZ;
    }
    result.outTensors.push_back(result.inTensors[2]);
    result.outTensors.push_back(result.inTensors[3]);
    return result;
  }
  atb::VariantPack attentionPack(const atb::VariantPack& pack, int start) const {
    const int count=std::min(128,queries_-start);
    const uint64_t row_bytes=static_cast<uint64_t>(heads_)*dim_*2;
    atb::VariantPack result;
    auto q=pack.inTensors[0];
    q.desc.shape.dimNum=3;
    q.desc.shape.dims[0]=count; q.desc.shape.dims[1]=heads_; q.desc.shape.dims[2]=dim_;
    if (q.deviceData) q.deviceData=static_cast<uint8_t*>(q.deviceData)+start*row_bytes;
    q.dataSize=count*row_bytes;
    result.inTensors.push_back(q);
    for (int i : {3,4}) {
      auto t=pack.inTensors[i];
      if(nz_) t.desc.format=ACL_FORMAT_FRACTAL_NZ;
      result.inTensors.push_back(t);
    }
    auto table=pack.inTensors[6];
    uint64_t table_row=table.desc.shape.dims[1]*sizeof(int32_t);
    table.desc.shape.dims[0]=count;
    if(table.deviceData) table.deviceData=static_cast<uint8_t*>(table.deviceData)+start*table_row;
    table.dataSize=count*table_row;
    result.inTensors.push_back(table);
    auto lengths=pack.inTensors[7];
    lengths.desc.shape.dims[0]=count;
    lengths.hostData=static_cast<int32_t*>(lengths.hostData)+start;
    lengths.deviceData=nz_ ? static_cast<int32_t*>(pack.inTensors[8].deviceData)+start : nullptr;
    lengths.dataSize=count*sizeof(int32_t);
    result.inTensors.push_back(lengths);
    auto out=pack.outTensors[0];
    out.desc=q.desc;
    if(out.deviceData) out.deviceData=static_cast<uint8_t*>(out.deviceData)+start*row_bytes;
    out.dataSize=count*row_bytes;
    result.outTensors.push_back(out);
    return result;
  }
  atb::Status Setup(const atb::VariantPack& pack, uint64_t& workspace, atb::Context* ctx) override {
    if (pack.inTensors.size() != GetInputNum() || pack.outTensors.size() != GetOutputNum())
      return atb::ERROR_INVALID_PARAM;
    const auto& q = pack.inTensors[0].desc.shape;
    if (q.dimNum != 4 || q.dims[2] != heads_ || q.dims[3] != dim_)
      return atb::ERROR_INVALID_PARAM;
    queries_=q.dims[0]*q.dims[1];
    for (int i : {1,2}) {
      const auto& kv=pack.inTensors[i].desc.shape;
      if(kv.dimNum!=4 || kv.dims[0]!=q.dims[0] || kv.dims[1]!=q.dims[1] ||
         kv.dims[2]!=kv_heads_ || kv.dims[3]!=dim_) return atb::ERROR_INVALID_PARAM;
    }
    if(pack.inTensors[5].desc.shape.dims[0]!=queries_ ||
       pack.inTensors[6].desc.shape.dimNum!=2 || pack.inTensors[6].desc.shape.dims[0]!=queries_ ||
       pack.inTensors[7].desc.shape.dims[0]!=queries_ ||
       pack.inTensors[8].desc.shape.dims[0]!=queries_) return atb::ERROR_INVALID_PARAM;
    if (queries_<=0 || !pack.inTensors[7].hostData) return atb::ERROR_INVALID_PARAM;
    const char* encoder_flag=std::getenv("MLLM_ASCEND_PREFILL_ENCODER");
    const char* soc=aclrtGetSocName();
    use_encoder_=soc && std::strstr(soc,"310P") && nz_ && q.dims[0]==1 && queries_>1 && (!encoder_flag || encoder_flag[0]!='0');
    auto* lengths=static_cast<const int32_t*>(pack.inTensors[7].hostData);
    for(int i=0;use_encoder_ && i<queries_;++i) use_encoder_=lengths[i]==i+1;
    if (use_encoder_) {
      encoder_.prepare(queries_,heads_,kv_heads_,dim_);
      auto writer_pack=writePack(pack);
      auto status=writer_->Setup(writer_pack,workspace,ctx);
      if(status!=atb::NO_ERROR) return status;
      uint64_t bytes=0;status=encoder_.setup(pack,bytes,ctx);
      workspace=std::max(workspace,bytes);return status;
    }
    while (attention_.size()<static_cast<size_t>((queries_+127)/128)) {
      atb::infer::PagedAttentionParam param;
      param.headNum=heads_; param.kvHeadNum=kv_heads_; param.qkScale=1.f/std::sqrt(float(dim_));
      atb::Operation* op=nullptr;
      auto status=atb::CreateOperation(param,&op);
      if(status!=atb::NO_ERROR) return status;
      attention_.push_back(op);
    }
    auto writer_pack=writePack(pack);
    auto status=writer_->Setup(writer_pack,workspace,ctx);
    if(status!=atb::NO_ERROR) return status;
    for(int start=0;start<queries_;start+=128) {
      auto pa_pack=attentionPack(pack,start);
      uint64_t bytes=0;
      status=attention_[start/128]->Setup(pa_pack,bytes,ctx);
      if(status!=atb::NO_ERROR) return status;
      workspace=std::max(workspace,bytes);
    }
    return atb::NO_ERROR;
  }
  atb::Status Execute(const atb::VariantPack& pack,uint8_t* workspace,uint64_t bytes,atb::Context* ctx) override {
    auto writer_pack=writePack(pack);
    auto status=writer_->Execute(writer_pack,workspace,bytes,ctx);
    if(status!=atb::NO_ERROR) return status;
    if(use_encoder_) return encoder_.execute(pack,workspace,bytes,ctx);
    for(int start=0;start<queries_;start+=128) {
      auto pa_pack=attentionPack(pack,start);
      status=attention_[start/128]->Execute(pa_pack,workspace,bytes,ctx);
      if(status!=atb::NO_ERROR) return status;
    }
    return atb::NO_ERROR;
  }
 private:
  AscendPrefillEncoder encoder_;
  bool use_encoder_=false;
  int heads_,kv_heads_,dim_,queries_=0;
  bool nz_;
  atb::Operation* writer_=nullptr;
  std::vector<atb::Operation*> attention_;
};
}
