// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <limits>
#include <memory>
#include <aclnnop/aclnn_argmax.h>
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {
// Owns reusable device output/workspace. Logits belong to one generation step.
class AscendGreedySelector {
 public:
  int64_t select(Tensor logits, const std::vector<int64_t>& suppressed) {
    MLLM_RT_ASSERT(logits.device()==kAscend && logits.dtype()==kFloat16);
    const int64_t vocab=logits.shape().back();
    auto* row=logits.ptr<mllm_fp16_t>() + logits.numel()-vocab;
    // The projection is complete before selection; only EOS scalars are changed.
    const mllm_fp16_t lowest=static_cast<mllm_fp16_t>(-std::numeric_limits<float>::infinity());
    for (auto id:suppressed) if(id>=0 && id<vocab)
      MLLM_ACL_CHECK(aclrtMemcpy(row+id,sizeof(lowest),&lowest,sizeof(lowest),ACL_MEMCPY_HOST_TO_DEVICE));
    if (index_.isNil()) index_=Tensor::empty({1},kInt64,kAscend).alloc();
    int64_t one=1;
    using Handle=std::unique_ptr<aclTensor,decltype(&aclDestroyTensor)>;
    Handle input(aclCreateTensor(&vocab,1,ACL_FLOAT16,&one,0,ACL_FORMAT_ND,&vocab,1,row),aclDestroyTensor);
    Handle output(aclCreateTensor(&one,1,ACL_INT64,&one,0,ACL_FORMAT_ND,&one,1,index_.ptr<void>()),aclDestroyTensor);
    MLLM_RT_ASSERT(input && output);
    uint64_t bytes=0;
    aclOpExecutor* executor=nullptr;
    MLLM_ACL_CHECK(aclnnArgMaxGetWorkspaceSize(input.get(),0,true,output.get(),&bytes,&executor));
    if (bytes>workspace_bytes_) {
      MLLM_RT_ASSERT(bytes<=static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
      workspace_=Tensor::empty({static_cast<int32_t>(bytes)},kInt8,kAscend).alloc();
      workspace_bytes_=bytes;
    }
    MLLM_ACL_CHECK(aclnnArgMax(bytes ? workspace_.ptr<void>() : nullptr,bytes,executor,getGlobalAtbStream()));
    syncGlobalAtbStream();
    int64_t result=-1;
    MLLM_ACL_CHECK(aclrtMemcpy(&result,sizeof(result),index_.ptr<void>(),sizeof(result),ACL_MEMCPY_DEVICE_TO_HOST));
    return result;
  }
 private:
  Tensor workspace_,index_;
  uint64_t workspace_bytes_=0;
};
}
