#pragma once
#include <gtest/gtest.h>
#include "mllm/mllm.hpp"
#include "mllm/core/aops/FillOp.hpp"

TEST(AscendFillOwnershipTest, RepeatedFillPreservesStorageAndAliases) {
  using namespace mllm;
  auto tensor=Tensor::empty({2,16},kFloat16,kAscend).alloc();
  auto alias=tensor;
  void* original=tensor.ptr<void>();
  for(float value : {0.f,1.f,-0.25f}) {
    auto output=Context::instance().buildOpAndSubmitTask(
        OpTypes::kFill,aops::FillOpOptions{.type=aops::FillOpTypes::kSpecific,.value=value},{tensor})[0];
    EXPECT_EQ(tensor.ptr<void>(),original);
    EXPECT_EQ(output.ptr<void>(),original);
    EXPECT_EQ(alias.ptr<void>(),original);
    auto cpu=alias.to(kCPU);
    for(size_t i=0;i<cpu.numel();++i)
      EXPECT_EQ(static_cast<float>(cpu.ptr<mllm_fp16_t>()[i]),value);
  }
}
TEST(AscendFillOwnershipTest, ZerosOnesAndScalarMultiply) {
  using namespace mllm;
  auto zeros=Tensor::zeros({2,16},kFloat16,kAscend);
  auto ones=Tensor::ones({2,16},kFloat16,kAscend);
  auto result=(ones*0.5f).to(kCPU);
  auto z=zeros.to(kCPU);
  for(size_t i=0;i<result.numel();++i) {
    EXPECT_EQ(static_cast<float>(result.ptr<mllm_fp16_t>()[i]),0.5f);
    EXPECT_EQ(static_cast<float>(z.ptr<mllm_fp16_t>()[i]),0.f);
  }
}
