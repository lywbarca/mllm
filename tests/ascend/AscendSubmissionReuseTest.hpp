#pragma once
#include <gtest/gtest.h>
#include "mllm/backends/ascend/graph/AscendGraphSubmissionScope.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"

TEST(AscendSubmissionReuseTest, DeferredChainRetainsStorageAndDrainsOnException) {
  using namespace mllm;
  auto weight=Tensor::zeros({16,16},kFloat16,kCPU);
  for(int i=0;i<16;++i)weight.ptr<mllm_fp16_t>()[i*16+i]=static_cast<mllm_fp16_t>(1.f);
  weight=weight.to(kAscend);
  std::vector<std::unique_ptr<ascend::AscendGraphExecutor>> executors;
  for(int i=0;i<4;++i) {
    atb::infer::LinearParam p;p.transposeA=false;p.transposeB=true;p.hasBias=false;
    atb::Operation* op=nullptr;MLLM_ATB_CHECK(atb::CreateOperation(p,&op));
    ascend::AscendGraphBuilder b;
    b.beginGraph("DeferredIdentity"+std::to_string(i),{"x","w"},{"y"},
      [](const atb::SVector<atb::TensorDesc>& in,atb::SVector<atb::TensorDesc>& out) {
        out[0]=in[0];return atb::NO_ERROR;
      });
    b.addOperation(op,{"x","w"},{"y"});
    executors.push_back(std::make_unique<ascend::AscendGraphExecutor>(b.build(),ascend::getGlobalAtbContext()));
  }
  for(bool fail : {false,true}) {
    auto host=Tensor::empty({1,8,16},kFloat16,kCPU).alloc();
    for(size_t i=0;i<host.numel();++i)host.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(float(i)/128);
    Tensor result;
    try {
      ascend::AscendGraphSubmissionScope scope(true);
      auto x=host.to(kAscend);
      std::vector<void*> in_flight;
      for(auto& executor:executors) {
        auto y=Tensor::empty(x.shape(),kFloat16,kAscend).alloc();
        for(auto* pointer:in_flight)EXPECT_NE(y.ptr<void>(),pointer);
        std::vector<Tensor> in{x,weight},out{y};
        scope.retain(in,out);
        executor->execute(in,out,{},false);
        in_flight.push_back(x.ptr<void>());
        x=y;
      }
      result=x;
      if(fail)throw std::runtime_error("exercise unwind after submissions");
      scope.finish();
    } catch(const std::runtime_error&) {
      ASSERT_TRUE(fail);
    }
    auto got=result.to(kCPU);
    EXPECT_EQ(std::memcmp(got.ptr<void>(),host.ptr<void>(),host.bytes()),0);
  }
}

TEST(AscendSubmissionReuseTest, OutputProjectionReusesResourcesAcrossShapesAndWeights) {
  using namespace mllm;
  if(!ascend::useNZWeights())GTEST_SKIP();
  ascend::AscendNZLinearRunner runner;
  for(int version=0;version<2;++version) {
    auto w=Tensor::empty({64,32},kFloat16,kCPU).alloc();
    for(size_t i=0;i<w.numel();++i)w.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.2f*std::cos(float(i+version*13)));
    auto nz=ascend::packLinearWeightNZ(w.to(kAscend));
    for(int count:{1,17,1,2,2}) {
      auto x=Tensor::empty({1,count,32},kFloat16,kCPU).alloc();
      for(size_t i=0;i<x.numel();++i)x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(0.3f*std::sin(float(i)));
      auto dev=x.to(kAscend);
      auto first=runner.execute(dev,nz,64);
      auto allocations=runner.workspaceAllocations();
      auto second=runner.execute(dev,nz,64).to(kCPU);
      EXPECT_EQ(runner.operationCreations(),1);
      EXPECT_EQ(runner.workspaceAllocations(),allocations);
      auto got=first.to(kCPU);
      EXPECT_EQ(std::memcmp(got.ptr<void>(),second.ptr<void>(),got.bytes()),0);
      for(int m=0;m<count;++m)for(int n=0;n<64;++n) {
        float ref=0;for(int k=0;k<32;++k)ref+=float(x.ptr<mllm_fp16_t>()[m*32+k])*float(w.ptr<mllm_fp16_t>()[n*32+k]);
        EXPECT_NEAR(float(got.ptr<mllm_fp16_t>()[m*64+n]),ref,0.002f);
      }
    }
  }
}
