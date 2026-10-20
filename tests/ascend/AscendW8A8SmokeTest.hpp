#pragma once
#include <gtest/gtest.h>
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"
#include "mllm/backends/ascend/ops/AscendLinearQuant.hpp"
#include "mllm/backends/ascend/graph/AscendLinearW8A8PluginOperation.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"

TEST(AscendW8A8SmokeTest, StaticLinearMatchesQuantizedCpuReference) {
  using namespace mllm;
  constexpr int K=128,N=64;
  auto weight=Tensor::empty({N,K},kInt8,kCPU).alloc();
  auto sw=Tensor::empty({N},kFloat32,kCPU).alloc();
  for(float step : {1.f/32, 0.0317f}) {
  auto sx=Tensor::empty({1},kFloat32,kCPU).alloc();sx.ptr<float>()[0]=step;
  for(int n=0;n<N;++n) {
    sw.ptr<float>()[n]=float(n%4+1)/256;
    for(int k=0;k<K;++k)weight.ptr<int8_t>()[n*K+k]=int8_t((n*7+k*3)%17-8);
  }
  auto artifacts=ascend::prepareLinearW8A8Artifacts("smoke",N,sw,sx);
  auto device_weight=weight.to(kAscend);
  for(bool nz:{false,true}) {
  if(nz && !ascend::useNZWeights())continue;
  auto execution_weight=nz ? ascend::packLinearWeightNZ(device_weight) : device_weight;
  ASSERT_FALSE(execution_weight.isNil());
  ascend::AscendGraphBuilder b;
  b.beginGraph("W8A8Smoke",{"x","w","bias","scale"},{"y"},
      [](const atb::SVector<atb::TensorDesc>& in,atb::SVector<atb::TensorDesc>& out) {
        out[0]=in[0];out[0].shape.dims[out[0].shape.dimNum-1]=N;return atb::NO_ERROR;
      });
  b.addOperation(new ascend::AscendLinearW8A8PluginOperation(1.f/artifacts.scale_x,"_smoke"),
                 {"x","w","bias","scale"},{"y"});
  ascend::AscendGraphExecutor executor(b.build(),ascend::getGlobalAtbContext());
  // Independently obtain native INT8 activations, then validate fused matmul
  // and dequant against an integer CPU accumulation. Quant accuracy itself is
  // covered against an independent mathematical oracle in AscendNativeQuantTest.
  ascend::AscendGraphBuilder qb;
  qb.beginGraph("W8A8QuantReference",{"x","s","z"},{"q"},
    [](const atb::SVector<atb::TensorDesc>& in,atb::SVector<atb::TensorDesc>& out) {
      out[0]=in[0];out[0].dtype=ACL_INT8;return atb::NO_ERROR;
    });
  atb::infer::ElewiseParam qp;qp.elewiseType=atb::infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL;
  atb::Operation* qop=nullptr;ASSERT_EQ(atb::CreateOperation(qp,&qop),atb::NO_ERROR);
  qb.addOperation(qop,{"x","s","z"},{"q"});
  ascend::AscendGraphExecutor qe(qb.build(),ascend::getGlobalAtbContext());
  auto qs=Tensor::empty({1},kFloat16,kCPU).alloc();
  qs.ptr<mllm_fp16_t>()[0]=static_cast<mllm_fp16_t>(artifacts.scale_x);
  auto qsd=qs.to(kAscend),qz=Tensor::zeros({1},kInt8,kCPU).to(kAscend);
  for(int rows:{1,17,256,512,1}) {
    auto x=Tensor::empty({1,rows,K},kFloat16,kCPU).alloc();
    for(size_t i=0;i<x.numel();++i)
      x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(float(int(i%543)-271)/64);
    auto y=Tensor::empty({1,rows,N},kFloat16,kAscend).alloc();
    std::vector<Tensor> inputs{x.to(kAscend),execution_weight,artifacts.bias_int32_npu,artifacts.deq_scale_npu},outputs{y};
    executor.execute(inputs,outputs,nz ? std::vector<size_t>{1} : std::vector<size_t>{});
    auto got=y.to(kCPU);
    auto q=Tensor::empty({1,rows,K},kInt8,kAscend).alloc();
    std::vector<Tensor> qi{inputs[0],qsd,qz},qo{q};qe.execute(qi,qo);
    auto qc=q.to(kCPU);
    for(int m=0;m<rows;++m)for(int n=0;n<N;++n) {
      int sum=0;
      for(int k=0;k<K;++k)
        sum+=int(qc.ptr<int8_t>()[m*K+k])*int(weight.ptr<int8_t>()[n*K+k]);
      float expected=sum*sw.ptr<float>()[n]*artifacts.scale_x;
      ASSERT_NEAR(float(got.ptr<mllm_fp16_t>()[m*N+n]),expected,0.005f);
    }
  }
}
}

}
