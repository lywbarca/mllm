#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/CausalMask.hpp"
#include "mllm/backends/ascend/ops/AscendKVCacheOp.hpp"

namespace {
void runContiguousAttention(bool concat) {
  using namespace mllm;
  constexpr int Hq = 4, Hkv = 2, D = 32, capacity = 32;
  ascend::AscendKVCache cache(capacity, 1, Hkv, D, kFloat16, Hq / Hkv);
  ASSERT_EQ(cache.pagedCache(), nullptr);
  nn::CausalMask mask;
  mask.to(kAscend);
  std::vector<std::vector<float>> hostK(Hkv), hostV(Hkv);
  Tensor keys, values;
  int past = 0;
  for (int length : {8, 1, 1, 2}) {
    auto q = Tensor::empty({1, Hq, length, D}, kFloat16, kCPU).alloc();
    auto k = Tensor::empty({1, Hkv, length, D}, kFloat16, kCPU).alloc();
    auto v = Tensor::empty({1, Hkv, length, D}, kFloat16, kCPU).alloc();
    for (int h = 0; h < Hq; ++h)
      for (int s = 0; s < length; ++s)
        for (int d = 0; d < D; ++d)
          q.ptr<mllm_fp16_t>()[(h * length + s) * D + d] =
              static_cast<mllm_fp16_t>(0.5f * std::sin((h * 13 + (past+s)*7 + d)*0.17f));
    for (int h = 0; h < Hkv; ++h)
      for (int s = 0; s < length; ++s)
        for (int d = 0; d < D; ++d) {
          const int i = (h * length + s) * D + d;
          k.ptr<mllm_fp16_t>()[i] = static_cast<mllm_fp16_t>(
              0.5f * std::cos((h*19 + (past+s)*11 + d)*0.13f));
          v.ptr<mllm_fp16_t>()[i] = static_cast<mllm_fp16_t>(
              0.5f * std::sin((h*23 + (past+s)*5 + d)*0.23f));
          hostK[h].push_back(static_cast<float>(k.ptr<mllm_fp16_t>()[i]));
          hostV[h].push_back(static_cast<float>(v.ptr<mllm_fp16_t>()[i]));
        }
    auto kd = k.to(kAscend), vd = v.to(kAscend);
    const int total = past + length;
    if (concat) {
      keys = past == 0 ? kd : nn::functional::concat({keys, kd}, 2);
      values = past == 0 ? vd : nn::functional::concat({values, vd}, 2);
    } else {
      auto kv = cache.updateKVCache(0, kd, vd);
      keys = kv[0]; values = kv[1];
      ASSERT_EQ(cache.getCurrentSeqCnt(0), total);
    }
    auto kh = keys.to(kCPU), vh = values.to(kCPU);
    ASSERT_EQ(kh.numel(), Hkv * total * D);
    for (int h = 0; h < Hkv; ++h)
      for (int j = 0; j < total * D; ++j) {
        ASSERT_EQ(static_cast<float>(kh.ptr<mllm_fp16_t>()[h*total*D+j]), hostK[h][j]);
        ASSERT_EQ(static_cast<float>(vh.ptr<mllm_fp16_t>()[h*total*D+j]), hostV[h][j]);
      }
    auto kr = ascend::repeatInterleaveForGQA(keys, Hq / Hkv);
    auto vr = ascend::repeatInterleaveForGQA(values, Hq / Hkv);
    auto scores = nn::functional::matmul(q.to(kAscend), kr, false, true) *
                  (1.f / std::sqrt(static_cast<float>(D)));
    auto probs = nn::functional::softmax(mask(scores), -1);
    auto result = nn::functional::matmul(probs, vr, false, false).to(kCPU);
    ASSERT_EQ(result.dtype(), kFloat16);
    float max_error = 0;
    for (int h = 0; h < Hq; ++h)
      for (int s = 0; s < length; ++s) {
        const int kvh = h / (Hq / Hkv);
        const int visible = past + s + 1;
        std::vector<float> logits(visible);
        for (int j = 0; j < visible; ++j) {
          float dot = 0;
          for (int d = 0; d < D; ++d)
            dot += static_cast<float>(q.ptr<mllm_fp16_t>()[(h*length+s)*D+d]) *
                   hostK[kvh][j*D+d];
          logits[j] = dot / std::sqrt(static_cast<float>(D));
        }
        const float maximum = *std::max_element(logits.begin(), logits.end());
        float denom = 0;
        for (auto& x : logits) { x = std::exp(x - maximum); denom += x; }
        for (int d = 0; d < D; ++d) {
          float expected = 0;
          for (int j = 0; j < visible; ++j)
            expected += logits[j] / denom * hostV[kvh][j*D+d];
          const float actual = static_cast<float>(result.ptr<mllm_fp16_t>()[(h*length+s)*D+d]);
          ASSERT_TRUE(std::isfinite(actual));
          max_error = std::max(max_error, std::abs(actual-expected));
          ASSERT_NEAR(actual, expected, 0.005f + 0.01f * std::abs(expected))
              << "past=" << past << " head=" << h << " token=" << s << " dim=" << d;
        }
      }
    std::cout << "[ContiguousAttention] mode=" << (concat ? "concat" : "cache-append")
              << " append=" << length << " total=" << total << " max_abs_error=" << max_error << std::endl;
    past = total;
  }
}
}
TEST(AscendContiguousAttentionTest, ConcatPrefillAndDecodeFP16) { runContiguousAttention(true); }
TEST(AscendContiguousAttentionTest, LegacyCachePrefillAndDecodeFP16) { runContiguousAttention(false); }
