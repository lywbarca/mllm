// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <optional>
#include <numeric>
#include <random>
#include "KernelTestHelper.hpp"
#include "mllm/backends/ascend/ops/AscendPagedKVCache.hpp"
#include "mllm/models/qwen_ascend/modeling_qwen_ascend.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"

namespace {
std::optional<mllm::aops::PagedAttnCacheLayout> pagedTestLayout() {
  const char* soc = aclrtGetSocName();
  const std::string name = soc ? soc : "";
  if (name.find("310P") != std::string::npos) { return mllm::aops::PagedAttnCacheLayout::kNZ; }
  if (name.find("910B") != std::string::npos) { return mllm::aops::PagedAttnCacheLayout::kND; }
  return std::nullopt;
}
}  // namespace

class AscendPagedKVCacheTest : public KernelTest {};

TEST_F(AscendPagedKVCacheTest, RadixOwnsPhysicalStorageAcrossLayersAndEviction) {
  using namespace mllm;
  const auto layout = pagedTestLayout();
  if (!layout) { GTEST_SKIP() << "Requires 310P or 910B"; }
  ascend::AscendPagedKVCache cache({.block_size = 16, .num_blocks = 6,
      .max_sequence_length = 64, .layer_nums = 2, .kv_heads = 2,
      .head_dim = 64, .radix_cache = true, .layout = *layout});
  // Values depend on the complete causal history, layer, head, dimension and K/V.
  // Looking up the right token at the wrong position/branch must fail too.
  auto value = [](const std::vector<int64_t>& history, int s, int layer, int h, int d, int kv) {
    uint32_t hash = 17;
    for (int i = 0; i <= s; ++i) hash = hash * 31 + static_cast<uint32_t>(history[i]);
    return static_cast<mllm_fp16_t>((int(hash % 127) - 63 + layer*131 + h*17 + d + kv*257) / 1024.f);
  };
  auto append = [&](int64_t id, const std::vector<int64_t>& suffix, bool abort = false) {
    auto history = cache.blocks().sequence(id).tokens;
    const int old = history.size(), count = suffix.size();
    history.insert(history.end(), suffix.begin(), suffix.end());
    ASSERT_TRUE(cache.beginStep({id}, count, {suffix}));
    ascend::AscendPagedKVCache::StepGuard guard(&cache);
    for (int layer = 0; layer < 2; ++layer) {
      std::array<Tensor, 2> input;
      for (int kv = 0; kv < 2; ++kv) {
        input[kv] = Tensor::empty({1, count, 2, 64}, kFloat16, kCPU).alloc();
        for (int s = 0; s < count; ++s) for (int h = 0; h < 2; ++h) for (int d = 0; d < 64; ++d)
          input[kv].ptr<mllm_fp16_t>()[(s*2+h)*64+d] = value(history, old+s, layer, h, d, kv);
      }
      cache.writeLayer(layer, input[0].to(kAscend), input[1].to(kAscend));
      auto actual = cache.gatherLayer(layer, 0);
      for (int kv = 0; kv < 2; ++kv) {
        auto host = actual[kv].to(kCPU);
        for (int s = 0; s < int(history.size()); ++s) for (int h = 0; h < 2; ++h) for (int d = 0; d < 64; ++d)
          ASSERT_EQ(float(host.ptr<mllm_fp16_t>()[(h*history.size()+s)*64+d]),
                    float(value(history, s, layer, h, d, kv))) << "layer=" << layer << " token=" << s;
      }
    }
    if (abort) cache.abortStep(); else cache.commitStep();
  };
  std::vector<int64_t> prompt(23);
  std::iota(prompt.begin(), prompt.end(), 1);
  cache.createSequence(0);
  append(0, prompt); ASSERT_FALSE(HasFatalFailure());
  const auto original = cache.blocks().sequence(0).blocks;
  cache.releaseSequence(0);
  ASSERT_EQ(cache.blocks().freeBlocks(), 4);
  cache.createSequence(1);
  ASSERT_EQ(cache.reusePrefix(1, prompt), 22);
  ASSERT_EQ(cache.blocks().sequence(1).blocks, original);
  append(1, {prompt.back()}); ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(cache.blocks().sequence(1).blocks.front(), original.front());
  EXPECT_NE(cache.blocks().sequence(1).blocks.back(), original.back());
  // A branch within a block gets its own COW copy; original request stays pinned.
  cache.createSequence(2);
  auto branch = prompt; branch[20] = 100;
  ASSERT_EQ(cache.reusePrefix(2, branch), 20);
  const auto before_abort = cache.blocks().sequence(2).blocks;
  append(2, {100, 22, 23}, true); ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(cache.blocks().sequence(2).blocks, before_abort);
  EXPECT_EQ(cache.blocks().sequence(2).length, 20);
  append(2, {100, 22, 23}); ASSERT_FALSE(HasFatalFailure());
  cache.releaseSequence(2);
  const auto pinned = cache.blocks().sequence(1).blocks;
  // More unrelated KV than fits in six physical blocks forces actual eviction.
  for (int i = 0; i < 10; ++i) {
    cache.createSequence(3);
    auto unrelated = prompt; unrelated[0] = 1000+i;
    ASSERT_EQ(cache.reusePrefix(3, unrelated), 0);
    append(3, unrelated); ASSERT_FALSE(HasFatalFailure());
    cache.releaseSequence(3);
    ASSERT_EQ(cache.blocks().sequence(1).blocks, pinned);
  }
  append(1, {24}); ASSERT_FALSE(HasFatalFailure());
  cache.clearPrefixCache();
  EXPECT_EQ(cache.blocks().cachedBlocks(), 0);
  // Clearing the tree must not free the live request's actual K/V.
  append(1, {25}); ASSERT_FALSE(HasFatalFailure());
  cache.releaseSequence(1);
  cache.clearPrefixCache();
  EXPECT_EQ(cache.blocks().freeBlocks(), 6);
}

TEST_F(AscendPagedKVCacheTest, BatchedRoPEUsesEachRequestsEmbeddingRow) {
  using namespace mllm;  // NOLINT
  if (!pagedTestLayout()) { GTEST_SKIP() << "Requires 310P or 910B"; }
  auto x = Tensor::empty({2, 1, 2, 64}, kFloat16, kCPU).alloc();
  auto sin = Tensor::empty({2, 1, 64}, kFloat16, kCPU).alloc();
  auto cos = Tensor::empty({2, 1, 64}, kFloat16, kCPU).alloc();
  for (int b = 0; b < 2; ++b) {
    for (int d = 0; d < 64; ++d) {
      sin.ptr<mllm_fp16_t>()[b * 64 + d] = static_cast<mllm_fp16_t>(b == 0 ? 0.f : 1.f);
      cos.ptr<mllm_fp16_t>()[b * 64 + d] = static_cast<mllm_fp16_t>(b == 0 ? 1.f : 0.f);
      for (int h = 0; h < 2; ++h) { x.ptr<mllm_fp16_t>()[(b * 2 + h) * 64 + d] = static_cast<mllm_fp16_t>(d < 32 ? 1.f : 2.f); }
    }
  }
  nn::RoPE rope(10000.f, 64, aops::RoPEOpOptionsInputType::kBSHD);
  rope.to(kAscend);
  auto result = rope(x.to(kAscend), sin.to(kAscend), cos.to(kAscend)).to(kCPU);
  for (int b = 0; b < 2; ++b) {
    for (int h = 0; h < 2; ++h) {
      for (int d = 0; d < 64; ++d) {
        const float expected = b == 0 ? (d < 32 ? 1.f : 2.f) : (d < 32 ? -2.f : 1.f);
        EXPECT_NEAR(static_cast<float>(result.ptr<mllm_fp16_t>()[(b * 2 + h) * 64 + d]), expected, 1e-3f);
      }
    }
  }
}

TEST_F(AscendPagedKVCacheTest, WriterForkRollbackAndReclaim) {
  using namespace mllm;  // NOLINT
  const auto layout = pagedTestLayout();
  if (!layout) { GTEST_SKIP() << "Requires 310P or 910B"; }
  ascend::AscendPagedKVCache cache({.block_size = 16,
                                    .num_blocks = 6,
                                    .max_sequence_length = 64,
                                    .layer_nums = 2,
                                    .kv_heads = 2,
                                    .head_dim = 64,
                                    .layout = *layout});
  cache.createSequence(10);
  auto key = Tensor::random({1, 17, 2, 64}, -0.5f, 0.5f, kFloat16, kCPU);
  auto value = Tensor::random({1, 17, 2, 64}, -0.5f, 0.5f, kFloat16, kCPU);
  ASSERT_TRUE(cache.beginStep({10}, 17));
  auto kd = key.to(kAscend), vd = value.to(kAscend);
  cache.writeLayer(0, kd, vd);
  EXPECT_THROW(cache.commitStep(), std::logic_error);  // Cross-layer consistency gate.
  cache.writeLayer(1, vd, kd);
  auto checkGather = [&](int layer, size_t batch, const Tensor& expected_key, const Tensor& expected_value, int prefix_len) {
    auto gathered = cache.gatherLayer(layer, batch);
    for (int kv = 0; kv < 2; ++kv) {
      auto actual = gathered[kv].to(kCPU);
      const auto& expected = kv == 0 ? expected_key : expected_value;
      const int length = actual.size(2);
      for (int s = 0; s < prefix_len; ++s) {
        for (int h = 0; h < 2; ++h) {
          for (int d = 0; d < 64; ++d) {
            EXPECT_EQ(static_cast<float>(actual.ptr<mllm_fp16_t>()[(h * length + s) * 64 + d]),
                      static_cast<float>(expected.ptr<mllm_fp16_t>()[(s * 2 + h) * 64 + d]));
          }
        }
      }
    }
  };
  checkGather(0, 0, key, value, 17);
  checkGather(1, 0, value, key, 17);
  cache.commitStep();
  cache.forkSequence(10, 20);
  EXPECT_EQ(cache.blocks().freeBlocks(), 4);
  const auto shared = cache.blocks().sequence(10).blocks;

  auto new_key = Tensor::random({2, 1, 2, 64}, -0.5f, 0.5f, kFloat16, kCPU).to(kAscend);
  auto new_value = Tensor::random({2, 1, 2, 64}, -0.5f, 0.5f, kFloat16, kCPU).to(kAscend);
  ASSERT_TRUE(cache.beginStep({20, 10}, 1));  // Reorder batch and split both shared tails.
  EXPECT_EQ(cache.blocks().copies().size(), 2);
  cache.writeLayer(0, new_key, new_value);
  cache.abortStep();
  EXPECT_EQ(cache.blocks().sequence(10).blocks, shared);
  EXPECT_EQ(cache.blocks().sequence(20).length, 17);
  EXPECT_EQ(cache.blocks().freeBlocks(), 4);

  ASSERT_TRUE(cache.beginStep({20, 10}, 1));
  for (int layer = 0; layer < 2; ++layer) { cache.writeLayer(layer, new_key, new_value); }
  for (size_t b = 0; b < 2; ++b) {
    checkGather(0, b, key, value, 17);
    checkGather(1, b, value, key, 17);
    auto gathered = cache.gatherLayer(0, b);
    auto actual_k = gathered[0].to(kCPU), actual_v = gathered[1].to(kCPU);
    auto expected_k = new_key.to(kCPU), expected_v = new_value.to(kCPU);
    for (int h = 0; h < 2; ++h) {
      for (int d = 0; d < 64; ++d) {
        const int dst = (h * 18 + 17) * 64 + d;
        const int src = (static_cast<int>(b) * 2 + h) * 64 + d;
        EXPECT_EQ(static_cast<float>(actual_k.ptr<mllm_fp16_t>()[dst]), static_cast<float>(expected_k.ptr<mllm_fp16_t>()[src]));
        EXPECT_EQ(static_cast<float>(actual_v.ptr<mllm_fp16_t>()[dst]), static_cast<float>(expected_v.ptr<mllm_fp16_t>()[src]));
      }
    }
  }
  cache.commitStep();
  EXPECT_EQ(cache.blocks().freeBlocks(), 3);  // One shared full block, two private tails.
  cache.releaseSequence(10);
  cache.releaseSequence(20);
  EXPECT_EQ(cache.blocks().freeBlocks(), 6);
}

TEST_F(AscendPagedKVCacheTest, QwenRadixPrefillDecodeForkAndBatchLogits) {
  struct GraphEnvGuard {
    std::string previous;
    bool present;
    GraphEnvGuard() : present(std::getenv("MLLM_ASCEND_QWEN_DECODER_GRAPH") != nullptr) {
      if (present) previous=std::getenv("MLLM_ASCEND_QWEN_DECODER_GRAPH");
      setenv("MLLM_ASCEND_QWEN_DECODER_GRAPH","1",1);
    }
    ~GraphEnvGuard() {
      if (present) setenv("MLLM_ASCEND_QWEN_DECODER_GRAPH",previous.c_str(),1);
      else unsetenv("MLLM_ASCEND_QWEN_DECODER_GRAPH");
    }
  } graph_env;
  using namespace mllm;                       // NOLINT
  using namespace mllm::models::qwen_ascend;  // NOLINT
  const auto layout = pagedTestLayout();
  if (!layout) { GTEST_SKIP() << "Requires 310P or 910B"; }
  QwenAscendConfig cfg;
  cfg.hidden_size = 256;
  cfg.head_dim = 64;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.num_hidden_layers = 2;
  cfg.intermediate_size = 512;
  cfg.vocab_size = 64;
  cfg.max_cache_length = cfg.max_position_embeddings = 512;
  cfg.tie_word_embeddings = true;
  cfg.paged_attention_enabled = true;
  cfg.paged_radix_cache = true;
  cfg.paged_block_size = 16;
  cfg.paged_num_blocks = 16;
  cfg.paged_layout = *layout;
  auto weights = ParameterFile::create();
  std::mt19937 rng(73);
  std::uniform_real_distribution<float> random(-0.03f, 0.03f);
  auto weight = [&](const std::string& name, const Tensor::shape_t& shape, bool norm = false) {
    auto t = Tensor::empty(shape, kFloat16, kCPU).alloc();
    for (size_t i = 0; i < t.numel(); ++i) { t.ptr<mllm_fp16_t>()[i] = static_cast<mllm_fp16_t>(norm ? 1.f : random(rng)); }
    weights->push(name + ".weight", t);
  };
  weight("model.embed_tokens", {64, 256});
  weight("model.norm", {256}, true);
  for (int i = 0; i < 2; ++i) {
    const auto layer = "model.layers." + std::to_string(i);
    weight(layer + ".input_layernorm", {256}, true);
    weight(layer + ".post_attention_layernorm", {256}, true);
    weight(layer + ".self_attn.q_norm", {64}, true);
    weight(layer + ".self_attn.k_norm", {64}, true);
    weight(layer + ".self_attn.q_proj", {256, 256});
    weight(layer + ".self_attn.k_proj", {128, 256});
    weight(layer + ".self_attn.v_proj", {128, 256});
    weight(layer + ".self_attn.o_proj", {256, 256});
    weight(layer + ".mlp.gate_proj", {512, 256});
    weight(layer + ".mlp.up_proj", {512, 256});
    weight(layer + ".mlp.down_proj", {256, 512});
  }
  QwenAscendForCausalLM model(cfg);
  model.to(kAscend);
  model.load(weights);
  auto reference_cfg = cfg;
  reference_cfg.paged_attention_enabled = false;
  reference_cfg.paged_radix_cache = false;
  QwenAscendForCausalLM reference(reference_cfg);
  reference.to(kAscend);
  reference.load(weights);
  auto tokens = [](const std::vector<int64_t>& ids) {
    return Tensor::fromVector(ids, {1, static_cast<int32_t>(ids.size())}, kInt64);
  };
  uint64_t last_graph_calls = 0;
  auto compare = [&](const Tensor& logits, int row, const std::vector<int64_t>& history) {
    setenv("MLLM_ASCEND_QWEN_DECODER_GRAPH", "0", 1);
    auto expected = reference.forward(tokens(history));
    setenv("MLLM_ASCEND_QWEN_DECODER_GRAPH", "1", 1);  // Recompute the whole causal prefix independently.
    ASSERT_EQ(reference.graphExecutionCount(), 0);
    if (row == 0) {
      ASSERT_EQ(model.graphExecutionCount(), last_graph_calls + cfg.num_hidden_layers);
      last_graph_calls = model.graphExecutionCount();
    }
    ASSERT_EQ(logits.dtype(), kFloat32);
    for (int v = 0; v < cfg.vocab_size; ++v) {
      const float target = expected.ptr<float>()[v];
      EXPECT_NEAR(logits.ptr<float>()[row * cfg.vocab_size + v], target, 0.01f + 0.02f * std::abs(target));
    }
  };
  std::vector<int64_t> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto output = model.forward({{"sequence", tokens(prompt)}}, {});
  compare(output.at("sequence"), 0, prompt);
  // A second chunk appends to a partial block and spans a block boundary.
  auto chunk = tokens({16, 17});
  output = model.forward({{"sequence", chunk}}, {});
  prompt.insert(prompt.end(), {16, 17});
  compare(output.at("sequence"), 0, prompt);
  model.forkPagedSequence(0, 1);
  model.selectPagedSequences({1});
  auto child = prompt;
  // Multi-token suffix reads a forked, partial-block prefix directly from the pool.
  child.insert(child.end(), {18, 23, 24});
  output = model.forward({{"sequence", tokens({18, 23, 24})}}, {});
  compare(output.at("sequence"), 0, child);
  model.selectPagedSequences({1, 0});
  auto batch = Tensor::fromVector(std::vector<int64_t>{19, 20}, {2, 1}, kInt64);
  output = model.forward({{"sequence", batch}}, {});
  child.push_back(19);
  prompt.push_back(20);
  compare(output.at("sequence"), 0, child);
  compare(output.at("sequence"), 1, prompt);
  EXPECT_EQ(output.at("position_ids").ptr<int64_t>()[0], 20);
  EXPECT_EQ(output.at("position_ids").ptr<int64_t>()[1], 17);
  // Exercise ARGeneration's previous-position metadata convention.
  output["sequence"] = Tensor::fromVector(std::vector<int64_t>{21, 22}, {2, 1}, kInt64);
  output = model.forward(output, {});
  child.push_back(21);
  prompt.push_back(22);
  compare(output.at("sequence"), 0, child);
  compare(output.at("sequence"), 1, prompt);
  const auto first_block = model.kvCache().pagedCache()->blocks().sequence(0).blocks[0];
  model.releasePagedSequence(0);
  model.releasePagedSequence(1);
  EXPECT_GT(model.kvCache().pagedCache()->blocks().cachedBlocks(), 0);
  // Diverge inside the first physical block: reuse 14 tokens, then COW its tail.
  auto divergent = prompt;
  divergent[14] = 33;
  auto divergent_output = model.forward(tokens(divergent));
  compare(divergent_output, 0, divergent);
  EXPECT_EQ(model.lastPagedReusedTokens(), 14);
  // Convenience forward resets requests but keeps published prefixes.
  auto replay = model.forward(tokens(prompt));
  compare(replay, 0, prompt);
  EXPECT_EQ(model.lastPagedReusedTokens(), static_cast<int32_t>(prompt.size()) - 1);
  EXPECT_EQ(model.kvCache().pagedCache()->blocks().sequence(0).blocks[0], first_block);
  auto* cache = model.kvCache().pagedCache();
  cache->createSequence(42);
  const auto refs_before_failure = cache->blocks().refCount(first_block);
  {
    mllm::ascend::AscendPagedKVCache::PrefixGuard prefix_guard(cache, 42);
    prefix_guard.attached(cache->reusePrefix(42, prompt) > 0);
    EXPECT_EQ(cache->blocks().sequence(42).length, static_cast<int32_t>(prompt.size()) - 1);
    ASSERT_TRUE(cache->beginStep({42}, 1, {{prompt.back()}}));
    mllm::ascend::AscendPagedKVCache::StepGuard step_guard(cache);
    // No layer wrote KV: failure aborts the append, then detaches the reused prefix.
    EXPECT_THROW(cache->commitStep(), std::logic_error);
  }
  EXPECT_FALSE(cache->pending());
  EXPECT_EQ(cache->blocks().sequence(42).length, 0);
  EXPECT_EQ(cache->blocks().refCount(first_block), refs_before_failure);
  cache->releaseSequence(42);
  const auto free_before_flush = model.kvCache().pagedCache()->blocks().freeBlocks();
  const auto active_blocks_before_flush = cache->blocks().sequence(0).blocks;
  model.clearPagedPrefixCache();
  EXPECT_EQ(model.kvCache().pagedCache()->blocks().cachedBlocks(), 0);
  EXPECT_GE(model.kvCache().pagedCache()->blocks().freeBlocks(), free_before_flush);
  EXPECT_EQ(cache->blocks().sequence(0).blocks, active_blocks_before_flush);
  model.clearCache();
  EXPECT_EQ(model.kvCache().pagedCache()->blocks().freeBlocks(), 16);
  EXPECT_TRUE(model.kvCache().pagedCache()->blocks().contains(0));
  // Exercise multiple PA submissions in one decoder graph, then reuse that graph
  // for single-token decode with a different block table and query shape.
  std::vector<int64_t> long_prompt(129);
  for (size_t i=0;i<long_prompt.size();++i) long_prompt[i]=1+i%60;
  auto long_output=model.forward(tokens(long_prompt));
  compare(long_output,0,long_prompt);
  long_prompt.push_back(7);
  auto decoded=model.forward({{"sequence",tokens({7})}},{});
  compare(decoded.at("sequence"),0,long_prompt);
  model.clearCache();
  model.clearPagedPrefixCache();
  EXPECT_EQ(cache->blocks().freeBlocks(),16);

}
