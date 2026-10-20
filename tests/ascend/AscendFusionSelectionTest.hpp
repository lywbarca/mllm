#pragma once
#include <gtest/gtest.h>
#include "mllm/models/qwen_ascend/qwen_ascend_graph_ops.hpp"
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"
#include "mllm/backends/ascend/ops/AscendGreedySelector.hpp"

TEST(AscendFusionSelectionTest, PackedProjectionChannelsAndReload) {
  using namespace mllm;
  if(!ascend::useNZWeights())GTEST_SKIP();
  ascend::AscendPackedProjection cache;
  ascend::AscendNZLinearRunner runner;
  for(int version=0;version<2;++version) {
    std::vector<Tensor> host,device;
    for(int rows:{64,32,32}) {
      auto w=Tensor::empty({rows,32},kFloat16,kCPU).alloc();
      for(size_t i=0;i<w.numel();++i)
        w.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.2f*std::sin(float(i+version*11+host.size()*7)));
      host.push_back(w);device.push_back(w.to(kAscend));
    }
    auto packed=cache.prepare(device);
    EXPECT_EQ(packed.ptr<void>(),cache.prepare(device).ptr<void>());
    for(int count:{1,17,256}) {
      auto x=Tensor::empty({1,count,32},kFloat16,kCPU).alloc();
      for(size_t i=0;i<x.numel();++i)x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.3f*std::cos(float(i)));
      auto got=runner.execute(x.to(kAscend),packed,128).to(kCPU);
      for(int m=0;m<count;++m) {
        int offset=0;
        for(const auto& w:host) {
          for(int n=0;n<w.size(0);++n) {
            float ref=0;
            for(int k=0;k<32;++k)ref+=float(x.ptr<mllm_fp16_t>()[m*32+k])*float(w.ptr<mllm_fp16_t>()[n*32+k]);
            ASSERT_NEAR(float(got.ptr<mllm_fp16_t>()[m*128+offset+n]),ref,0.002f);
          }
          offset+=w.size(0);
        }
      }
    }
  }
}

TEST(AscendFusionSelectionTest, DeviceGreedyTieTailAndMultipleEos) {
  using namespace mllm;
  ascend::AscendGreedySelector selector;
  for(int vocab:{32,151936}) {
    auto x=Tensor::empty({1,1,vocab},kFloat16,kCPU).alloc();
    std::fill_n(x.ptr<mllm_fp16_t>(),vocab,static_cast<mllm_fp16_t>(-3.f));
    x.ptr<mllm_fp16_t>()[7]=static_cast<mllm_fp16_t>(9.f);
    x.ptr<mllm_fp16_t>()[vocab-1]=static_cast<mllm_fp16_t>(9.f);
    x.ptr<mllm_fp16_t>()[11]=static_cast<mllm_fp16_t>(8.f);
    EXPECT_EQ(selector.select(x.to(kAscend),{}),7); // First index wins ties.
    EXPECT_EQ(selector.select(x.to(kAscend),{7,-1,vocab}),vocab-1);
    EXPECT_EQ(selector.select(x.to(kAscend),{7,vocab-1}),11);
    EXPECT_EQ(selector.select(x.to(kAscend),{7,vocab-1,11}),0);
  }
}

TEST(AscendFusionSelectionTest, PackedSwiGluMatchesReference) {
  using namespace mllm;
  for(int count:{1,17,256}) {
    auto x=Tensor::empty({1,count,64},kFloat16,kCPU).alloc();
    for(size_t i=0;i<x.numel();++i)x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(3.f*std::sin(float(i)));
    auto dev=x.to(kAscend);
    auto out=Tensor::empty({1,count,32},kFloat16,kAscend).alloc();
    atb::infer::ActivationParam param;param.activationType=atb::infer::ACTIVATION_SWIGLU_FORWARD;param.dim=-1;
    atb::Operation* op=nullptr;MLLM_ATB_CHECK(atb::CreateOperation(param,&op));
    atb::Tensor a,b;ascend::fillAtbTensor(dev,a);ascend::fillAtbTensor(out,b);
    atb::VariantPack pack;pack.inTensors={a};pack.outTensors={b};
    ascend::executePreparedOp(op,pack);atb::DestroyOperation(op);
    auto got=out.to(kCPU);
    for(int m=0;m<count;++m)for(int j=0;j<32;++j) {
      float gate=float(x.ptr<mllm_fp16_t>()[m*64+j]),up=float(x.ptr<mllm_fp16_t>()[m*64+32+j]);
      ASSERT_NEAR(float(got.ptr<mllm_fp16_t>()[m*32+j]),gate/(1.f+std::exp(-gate))*up,0.01f);
    }
  }
}


TEST(AscendFusionSelectionTest, AttentionScaleFp16OnX86) {
  using namespace mllm;
  models::qwen_ascend::QwenAscendDecoderGraphRunner runner;
  auto scale=runner.attentionScaleTensor(128,kFloat16,kAscend);
  auto again=runner.attentionScaleTensor(128,kFloat16,kAscend);
  EXPECT_EQ(scale.ptr<void>(),again.ptr<void>());
  auto host=scale.to(kCPU);
  EXPECT_NEAR(float(host.ptr<mllm_fp16_t>()[0]),1.f/std::sqrt(128.f),0.00005f);
}
