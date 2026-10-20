#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "mllm/mllm.hpp"
#include "mllm/nn/layers/PagedAttn.hpp"
#include "mllm/backends/ascend/ops/AscendPagedKVCache.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

#include "mllm/backends/ascend/graph/AscendPagedAttentionPlugin.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
namespace {
void checkRealPagedAttention(bool graph) {
  using namespace mllm;
  const char* soc = aclrtGetSocName();
  if (!soc || std::string(soc).find("310P") == std::string::npos) GTEST_SKIP();
  constexpr int H=4, K=2, D=64, N=8;
  ascend::AscendPagedKVCache cache({.block_size=128, .num_blocks=N,
      .max_sequence_length=512, .layer_nums=1, .kv_heads=K, .head_dim=D,
      .layout=aops::PagedAttnCacheLayout::kNZ});
  aops::PagedAttnOpOptions options;
  options.head_repeat_times=H/K;
  options.impl_type=aops::PagedAttnImplType::kDefault;
  options.kv_cache_layout=aops::PagedAttnCacheLayout::kNZ;
  nn::PagedAttn attn(options);
  attn.to(kAscend);
  std::unique_ptr<ascend::AscendGraphExecutor> executor;
  if (graph) {
    ascend::AscendGraphBuilder builder;
    std::vector<std::string> names{"q","k","v","kp","vp","slots","table","lengths_host","lengths_device"};
    builder.beginGraph("PagedAttentionOwnershipGraph",names,{"out"},
        [](const atb::SVector<atb::TensorDesc>& in,atb::SVector<atb::TensorDesc>& out) {
          out[0]=in[0]; return atb::NO_ERROR;
        });
    builder.addOperation(new ascend::AscendPagedAttentionPlugin(H,K,D,true),names,{"out"});
    executor=std::make_unique<ascend::AscendGraphExecutor>(builder.build(),ascend::getGlobalAtbContext());
  }
  struct History { std::vector<float> k,v; };
  std::map<int64_t,History> histories;
  int step=0;
  auto append = [&](std::vector<int64_t> ids, int count, bool abort=false) {
    const int batch=ids.size();
    auto next=histories;
    auto q=Tensor::empty({batch,count,H,D},kFloat16,kCPU).alloc();
    auto k=Tensor::empty({batch,count,K,D},kFloat16,kCPU).alloc();
    auto v=Tensor::empty({batch,count,K,D},kFloat16,kCPU).alloc();
    for (size_t i=0;i<q.numel();++i)
      q.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.4f*std::sin((i+step*31)*0.071f));
    for (int b=0;b<batch;++b) {
      for (int i=0;i<count*K*D;++i) {
        int index=b*count*K*D+i;
        k.ptr<mllm_fp16_t>()[index]=static_cast<mllm_fp16_t>(0.4f*std::cos((i+ids[b]*17+step*43)*0.053f));
        v.ptr<mllm_fp16_t>()[index]=static_cast<mllm_fp16_t>(0.4f*std::sin((i+ids[b]*29+step*19)*0.097f));
        next[ids[b]].k.push_back(static_cast<float>(k.ptr<mllm_fp16_t>()[index]));
        next[ids[b]].v.push_back(static_cast<float>(v.ptr<mllm_fp16_t>()[index]));
      }
    }
    ASSERT_TRUE(cache.beginStep(ids,count));
    ascend::AscendPagedKVCache::StepGuard guard(&cache);
    auto copies=cache.blocks().copies().size();
    Tensor result;
    if (graph) {
      cache.prepareGraphMetadata();
      std::vector<Tensor> inputs{q.to(kAscend),k.to(kAscend),v.to(kAscend),cache.keyPool(0),cache.valuePool(0),
          cache.slotMapping(),cache.graphBlockTable(),cache.graphLengthsHost(),cache.graphLengthsDevice()};
      std::vector<Tensor> outputs{Tensor::empty(q.shape(),kFloat16,kAscend).alloc()};
      executor->execute(inputs,outputs);
      cache.markGraphLayerWritten(0);
      result=outputs[0];
    } else {
      cache.writeLayer(0,k.to(kAscend),v.to(kAscend));
      result=attn(q.to(kAscend),cache.keyPool(0),cache.valuePool(0),
                  cache.blockTable(),cache.contextLengths())[0];
    }
    auto actual=result.to(kCPU);
    ASSERT_EQ(actual.dtype(),kFloat16);
    ASSERT_EQ(actual.shape(),q.shape());
    float max_error=0;
    for(int b=0;b<batch;++b) {
      const auto& history=next.at(ids[b]);
      int total=history.k.size()/(K*D);
      for(int s=0;s<count;++s) for(int h=0;h<H;++h) {
        int visible=total-count+s+1, kh=h/(H/K);
        std::vector<float> scores(visible);
        for(int j=0;j<visible;++j) {
          float dot=0;
          for(int d=0;d<D;++d)
            dot+=static_cast<float>(q.ptr<mllm_fp16_t>()[((b*count+s)*H+h)*D+d])*
                 history.k[(j*K+kh)*D+d];
          scores[j]=dot/std::sqrt(float(D));
        }
        float maximum=*std::max_element(scores.begin(),scores.end()), sum=0;
        for(auto& x:scores) { x=std::exp(x-maximum); sum+=x; }
        for(int d=0;d<D;++d) {
          float expected=0;
          for(int j=0;j<visible;++j) expected+=scores[j]/sum*history.v[(j*K+kh)*D+d];
          float got=static_cast<float>(actual.ptr<mllm_fp16_t>()[((b*count+s)*H+h)*D+d]);
          ASSERT_TRUE(std::isfinite(got));
          max_error=std::max(max_error,std::abs(got-expected));
          ASSERT_NEAR(got,expected,0.002f+0.01f*std::abs(expected))
              <<"step="<<step<<" seq="<<ids[b]<<" token="<<s<<" head="<<h<<" dim="<<d;
        }
      }
    }
    if(abort) cache.abortStep();
    else { cache.commitStep(); histories=std::move(next); }
    for(auto id:ids)
      ASSERT_EQ(cache.blocks().sequence(id).length,histories[id].k.size()/(K*D));
    std::cout<<"[RealPagedAttention] step="<<step++<<" batch="<<batch
             <<" append="<<count<<" abort="<<abort<<" cow_copies="<<copies
             <<" free_blocks="<<cache.blocks().freeBlocks()
             <<" max_abs_error="<<max_error<<std::endl;
  };
  cache.createSequence(10);
  append({10},127); ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_EQ(cache.blocks().sequence(10).blocks.size(),1);
  append({10},1); ASSERT_FALSE(testing::Test::HasFatalFailure());
  append({10},1); ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_EQ(cache.blocks().sequence(10).blocks.size(),2);
  cache.forkSequence(10,20); histories[20]=histories[10];
  auto shared=cache.blocks().sequence(10).blocks;
  ASSERT_EQ(cache.blocks().refCount(shared.back()),2);
  append({20,10},1,true); ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_EQ(cache.blocks().sequence(10).blocks,shared);
  ASSERT_EQ(cache.blocks().sequence(20).blocks,shared);
  ASSERT_EQ(cache.blocks().freeBlocks(),N-2);
  append({20,10},1); ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_EQ(cache.blocks().sequence(10).blocks.front(),cache.blocks().sequence(20).blocks.front());
  ASSERT_NE(cache.blocks().sequence(10).blocks.back(),cache.blocks().sequence(20).blocks.back());
  append({20},2); ASSERT_FALSE(testing::Test::HasFatalFailure());
  append({10,20},1); ASSERT_FALSE(testing::Test::HasFatalFailure());
  cache.releaseSequence(10); cache.releaseSequence(20);
  ASSERT_EQ(cache.blocks().freeBlocks(),N);
  cache.createSequence(30);
  append({30},129); ASSERT_FALSE(testing::Test::HasFatalFailure());
  append({30},1); ASSERT_FALSE(testing::Test::HasFatalFailure());
  cache.releaseSequence(30);
  ASSERT_EQ(cache.blocks().freeBlocks(),N);
  for(int count : {256,512}) {
    cache.createSequence(count);
    append({count},count); ASSERT_FALSE(testing::Test::HasFatalFailure());
    cache.releaseSequence(count);
    ASSERT_EQ(cache.blocks().freeBlocks(),N);
  }
}

}  // namespace
TEST(AscendRealPagedAttentionTest, AppendForkRollbackAndReuseFP16) { checkRealPagedAttention(false); }
TEST(AscendRealPagedAttentionTest, GraphAppendForkRollbackAndReuseFP16) { checkRealPagedAttention(true); }
