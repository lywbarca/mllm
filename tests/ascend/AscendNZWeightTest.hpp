#pragma once
#include <gtest/gtest.h>
#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"
TEST(AscendNZWeightTest, LinearCacheReuseReloadAndCanonicalParams) {
  using namespace mllm;
  if (!ascend::useNZWeights()) GTEST_SKIP();
  aops::LinearOpOptions options; options.in_channels=32;options.out_channels=48;options.bias=false;
  ascend::AscendLinearOp op(options);op.setName("test");
  auto x=Tensor::empty({1,3,32},kFloat16,kCPU).alloc();
  for(size_t i=0;i<x.numel();++i) x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.3f*std::sin(float(i)));
  auto device_x=x.to(kAscend);
  for(int reload=0;reload<2;++reload) {
    auto w=Tensor::empty({48,32},kFloat16,kCPU).alloc();
    for(size_t i=0;i<w.numel();++i)
      w.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.2f*std::cos(float(i+reload*7)));
    auto params=ParameterFile::create();params->push("test.weight",w);op.load(params);
    ASSERT_FALSE(op.nzWeight().isNil());
    auto* packed=op.nzWeight().ptr<void>();
    auto canonical=op.getParams()->pull("test.weight").to(kCPU);
    ASSERT_EQ(canonical.shape(),w.shape());
    EXPECT_EQ(std::memcmp(canonical.ptr<void>(),w.ptr<void>(),w.bytes()),0);
    for(int repeat=0;repeat<2;++repeat) {
      std::vector<Tensor> in{device_x},out;
      op.reshape(in,out);op.setup(in,out);op.forward(in,out);
      auto got=out[0].to(kCPU);
      EXPECT_EQ(packed,op.nzWeight().ptr<void>());
      for(int m=0;m<3;++m) for(int n=0;n<48;++n) {
        float ref=0;
        for(int k=0;k<32;++k) ref+=float(x.ptr<mllm_fp16_t>()[m*32+k])*float(w.ptr<mllm_fp16_t>()[n*32+k]);
        EXPECT_NEAR(float(got.ptr<mllm_fp16_t>()[m*48+n]),ref,0.002f);
      }
    }
  }
  EXPECT_TRUE(ascend::packLinearWeightNZ(Tensor::zeros({17,32},kFloat16,kCPU).to(kAscend)).isNil());
}
