#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"

TEST(AscendNativeQuantTest, StaticQuantCpuReferenceAndShapeReuse) {
  using namespace mllm;
  // Exercise both broadcast scales (W8A8) and true per-channel scales,
  // including ties, saturation, non-power-of-two scales and 3D activations.
  for (int count : {1, 128}) {
    ascend::AscendGraphBuilder b;
    b.beginGraph("NativeQuantTest", {"x","s","z"}, {"y"},
      [](const atb::SVector<atb::TensorDesc>& in, atb::SVector<atb::TensorDesc>& out) {
        out[0]=in[0];out[0].dtype=ACL_INT8;return atb::NO_ERROR;
      });
    atb::infer::ElewiseParam p;
    p.elewiseType=atb::infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL;
    atb::Operation* op=nullptr;
    ASSERT_EQ(atb::CreateOperation(p,&op),atb::NO_ERROR);
    b.addOperation(op,{"x","s","z"},{"y"});
    ascend::AscendGraphExecutor exec(b.build(),ascend::getGlobalAtbContext());
    for (float step : {1.f/32,0.0317f}) {
      auto s=Tensor::empty({count},kFloat16,kCPU).alloc();
      auto z=Tensor::zeros({count},kInt8,kCPU);
      for(int k=0;k<count;++k)s.ptr<mllm_fp16_t>()[k]=static_cast<mllm_fp16_t>(step*(1+k%3));
      auto ds=s.to(kAscend),dz=z.to(kAscend);
      for(int rows : {1,17,256,512,1}) {
        auto x=Tensor::empty({1,rows,128},kFloat16,kCPU).alloc();
        for(size_t i=0;i<x.numel();++i)x.ptr<mllm_fp16_t>()[i]=static_cast<mllm_fp16_t>(float(int(i%543)-271)/64);
        auto y=Tensor::empty({1,rows,128},kInt8,kAscend).alloc();
        std::vector<Tensor> inputs{x.to(kAscend),ds,dz},outputs{y};
        exec.execute(inputs,outputs);
        auto got=y.to(kCPU);
        for(size_t i=0;i<x.numel();++i) {
          float value=float(x.ptr<mllm_fp16_t>()[i])/float(s.ptr<mllm_fp16_t>()[count==1?0:i%128]);
          int expected=std::clamp(int(std::nearbyint(value)),-128,127);
          int actual=int(got.ptr<int8_t>()[i]);
          // FP16 vector division can move values across a rounding boundary.
          // Require <= 1 code difference AND <= 0.625 quantization step error.
          // Exact power-of-two scalar scales also verify ties-to-even exactly.
          ASSERT_LE(std::abs(actual-expected),1);
          ASSERT_LE(std::abs(float(actual)-std::clamp(value,-128.f,127.f)),0.625f)
              << "step="<<step<<" rows="<<rows<<" i="<<i;
          if(count==1 && step==1.f/32) ASSERT_EQ(actual,expected);
        }
      }
    }
  }
}
