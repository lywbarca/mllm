#pragma once
#include <gtest/gtest.h>
#include "mllm/mllm.hpp"
TEST(AscendHostCastTest, FP16LogitsRoundTrip) {
  using namespace mllm;
  const std::vector<float> values{-65504.f, -3.5f, -0.125f, 0.f, 0.00006103515625f, 0.5f, 7.f, 65504.f};
  auto source = Tensor::fromVector(values, {8}, kFloat32);
  auto half = source.to(kFloat16);
  auto restored = half.to(kFloat32);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(static_cast<float>(half.ptr<mllm_fp16_t>()[i]), values[i]);
    EXPECT_EQ(restored.ptr<float>()[i], values[i]);
  }
}
