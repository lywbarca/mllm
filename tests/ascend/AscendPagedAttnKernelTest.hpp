// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "KernelTestHelper.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/layers/PagedAttn.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"

class AscendPagedAttnKernelTest : public KernelTest {
 public:
  void checkAttention(bool nz, int repeat, int query_len = 1) {
    using namespace mllm;  // NOLINT
    constexpr int kv_heads = 2, dim = 64, block_size = 128, blocks = 4;
    const int batch = query_len == 1 ? 2 : 1;
    const int heads = kv_heads * repeat;
    auto q = Tensor::random({batch, query_len, heads, dim}, -0.5f, 0.5f, kFloat16, kCPU);
    auto k = Tensor::random({blocks, block_size, kv_heads, dim}, -0.5f, 0.5f, kFloat16, kCPU);
    auto v = Tensor::random({blocks, block_size, kv_heads, dim}, -0.5f, 0.5f, kFloat16, kCPU);
    // Shared first block, non-monotonic physical pages, distinct second blocks.
    const std::vector<int32_t> ids = batch == 2 ? std::vector<int32_t>{2, 0, 2, 3} : std::vector<int32_t>{2, 0};
    auto table = Tensor::fromVector(ids, {batch, 2}, kInt32).to(kAscend);
    auto uploadCache = [&](const Tensor& src) {
      if (!nz) {
        auto t = src;
        return t.to(kAscend);
      }
      auto packed = Tensor::empty({blocks, kv_heads * dim / 16, block_size, 16}, kFloat16, kCPU).alloc();
      for (int p = 0; p < blocks; ++p) {
        for (int s = 0; s < block_size; ++s) {
          for (int h = 0; h < kv_heads; ++h) {
            for (int d = 0; d < dim; ++d) {
              const int c = h * dim + d;
              const int dst = ((p * (kv_heads * dim / 16) + c / 16) * block_size + s) * 16 + c % 16;
              packed.ptr<mllm_fp16_t>()[dst] = src.ptr<mllm_fp16_t>()[((p * block_size + s) * kv_heads + h) * dim + d];
            }
          }
        }
      }
      return packed.to(kAscend);
    };
    auto kd = uploadCache(k);
    auto vd = uploadCache(v);
    auto qd = q.to(kAscend);
    aops::PagedAttnOpOptions options;
    options.head_repeat_times = repeat;
    options.impl_type = aops::PagedAttnImplType::kDefault;
    options.kv_cache_layout = nz ? aops::PagedAttnCacheLayout::kNZ : aops::PagedAttnCacheLayout::kND;
    nn::PagedAttn attention(options);
    attention.to(kAscend);

    // Reuse one layer and change host lengths, including a partial final block.
    const auto cases = query_len == 1 ? std::vector<std::vector<int32_t>>{{1, 127}, {128, 129}, {133, 255}}
                                      : std::vector<std::vector<int32_t>>{{query_len}, {std::max(129, query_len + 13)}, {255}};
    for (const auto& lens : cases) {
      auto lengths = Tensor::fromVector(lens, {batch}, kInt32);
      auto result = attention(qd, kd, vd, table, lengths);
      ASSERT_TRUE(result[1].isNil());
      ASSERT_EQ(result[0].shape(), q.shape());
      auto actual = result[0].to(kCPU);
      for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < query_len; ++token) {
          // Reference uses only the causal prefix, including non-block-aligned history.
          const int visible = lens[b] - query_len + token + 1;
          for (int h = 0; h < heads; ++h) {
            const int kh = h / repeat;
            std::vector<float> scores(visible);
            auto offset = [&](int s, int d) {
              const int p = ids[b * 2 + s / block_size];
              return ((p * block_size + s % block_size) * kv_heads + kh) * dim + d;
            };
            for (int s = 0; s < visible; ++s) {
              float dot = 0.f;
              for (int d = 0; d < dim; ++d) {
                dot += static_cast<float>(q.ptr<mllm_fp16_t>()[((b * query_len + token) * heads + h) * dim + d])
                       * static_cast<float>(k.ptr<mllm_fp16_t>()[offset(s, d)]);
              }
              scores[s] = dot / std::sqrt(static_cast<float>(dim));
            }
            const float max_score = *std::max_element(scores.begin(), scores.end());
            float sum = 0.f;
            for (auto& score : scores) {
              score = std::exp(score - max_score);
              sum += score;
            }
            for (int d = 0; d < dim; ++d) {
              float expected = 0.f;
              for (int s = 0; s < visible; ++s) {
                expected += scores[s] / sum * static_cast<float>(v.ptr<mllm_fp16_t>()[offset(s, d)]);
              }
              const int i = ((b * query_len + token) * heads + h) * dim + d;
              EXPECT_NEAR(static_cast<float>(actual.ptr<mllm_fp16_t>()[i]), expected, 2e-3f + 1e-2f * std::abs(expected))
                  << "batch=" << b << " token=" << token << " head=" << h << " dim=" << d;
            }
          }
        }
      }
    }
  }
};

TEST_F(AscendPagedAttnKernelTest, DecodeNZ310P) {
  const char* name = aclrtGetSocName();
  if (name == nullptr || std::string(name).find("310P") == std::string::npos) {
    GTEST_SKIP() << "NZ paged attention requires a supported 310P ATB runtime";
  }
  checkAttention(true, 1);
  checkAttention(true, 4);
}

TEST_F(AscendPagedAttnKernelTest, DecodeND910B) {
  const char* name = aclrtGetSocName();
  if (name == nullptr || std::string(name).find("910B") == std::string::npos) { GTEST_SKIP() << "ND test targets Ascend 910B"; }
  checkAttention(false, 1);
  checkAttention(false, 4);
}

TEST_F(AscendPagedAttnKernelTest, CausalPrefillNZ310P) {
  const char* name = aclrtGetSocName();
  if (name == nullptr || std::string(name).find("310P") == std::string::npos) { GTEST_SKIP() << "Requires 310P"; }
  checkAttention(true, 1, 17);
  checkAttention(true, 4, 129);  // Cross the 128-query submission boundary.
}

TEST_F(AscendPagedAttnKernelTest, CausalPrefillND910B) {
  const char* name = aclrtGetSocName();
  if (name == nullptr || std::string(name).find("910B") == std::string::npos) { GTEST_SKIP() << "Requires 910B"; }
  checkAttention(false, 1, 17);
  checkAttention(false, 4, 129);
}
