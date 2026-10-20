// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once
#include "mllm/backends/ascend/ops/AscendGreedySelector.hpp"

#ifndef MLLM_BUILD_ASCEND_BACKEND
#error "QwenAscend model requires MLLM_BUILD_ASCEND_BACKEND to be enabled"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/ascend/ops/AscendKVCacheOp.hpp"
#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"
#include "mllm/backends/ascend/ops/AscendNZWeight.hpp"
#include "mllm/backends/ascend/graph/AscendGraphSubmissionScope.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen_ascend/configuration_qwen_ascend.hpp"
#include "mllm/models/qwen_ascend/qwen_ascend_graph_ops.hpp"
#include "mllm/models/qwen_ascend/qwen_ascend_rope.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/layers/PagedAttn.hpp"
#include <unordered_set>

namespace mllm::models::qwen_ascend {

class QwenAscendDecoder;

inline mllm::ascend::AscendLinearOp* getAscendLinearOpPtr(const nn::Linear& linear) {
  return dynamic_cast<mllm::ascend::AscendLinearOp*>(linear.impl()->getInstancedOp().get());
}

inline mllm::ascend::AscendLinearOp& checkedAscendW8A8LinearOp(const nn::Linear& linear) {
  auto* op = getAscendLinearOpPtr(linear);
  MLLM_RT_ASSERT(op != nullptr);
  MLLM_RT_ASSERT(op->isW8A8());
  return *op;
}

class QwenAscendMLP final : public nn::Module {
  friend class QwenAscendDecoder;

  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  QwenAscendMLP() = default;
  QwenAscendMLP(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class QwenAscendAttention final : public nn::Module {
  friend class QwenAscendDecoder;

  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;
  nn::PagedAttn paged_attention_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  bool attention_bias_ = false;
  bool paged_graph_ = false;
  bool paged_graph_nz_ = true;
  float rms_norm_epsilon_ = 1e-5f;

 public:
  QwenAscendAttention() = default;

  QwenAscendAttention(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    attention_bias_ = cfg.attention_bias;
    paged_graph_ = cfg.paged_attention_enabled;
    paged_graph_nz_ = cfg.paged_layout == aops::PagedAttnCacheLayout::kNZ;
    rms_norm_epsilon_ = cfg.rms_norm_eps;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias);

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
    if (cfg.paged_attention_enabled) {
      aops::PagedAttnOpOptions options;
      options.head_repeat_times = num_key_value_groups_;
      options.impl_type = aops::PagedAttnImplType::kDefault;
      options.kv_cache_layout = cfg.paged_layout;
      paged_attention_ = reg<nn::PagedAttn>("paged_attention", options);
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    (void)inputs[3];
    auto past_kv_cache = args[0].get<mllm::ascend::AscendKVCache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int batch_size = inputs[0].shape()[0];
    int seq_len = inputs[0].shape()[1];

    query_states = query_states.view({batch_size, seq_len, num_attention_heads_, head_dim_});
    key_states = key_states.view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    value_states = value_states.view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // Ascend RoPE currently reads [B, S, H, D], so apply RoPE before transpose.
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto* paged = past_kv_cache->pagedCache();
    if (paged) {
      paged->writeLayer(layer_idx_, key_states, value_states);
      auto result = paged_attention_(query_states, paged->keyPool(layer_idx_), paged->valuePool(layer_idx_),
                                     paged->blockTable(), paged->contextLengths());
      return {o_proj_(result[0].view({batch_size, seq_len, num_attention_heads_ * head_dim_}))};
    }

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    auto [key_cached, value_cached] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    if (num_key_value_groups_ > 1) {
      key_cached = mllm::ascend::repeatInterleaveForGQA(key_cached, num_key_value_groups_);
      value_cached = mllm::ascend::repeatInterleaveForGQA(value_cached, num_key_value_groups_);
    }

    Tensor attn = nn::functional::matmul(query_states, key_cached, false, true) * (1.f / sqrtf(head_dim_));
    attn = mask_(attn);
    attn = softmax_(attn);

    auto output = nn::functional::matmul(attn, value_cached);
    output = output.transpose(1, 2).view({batch_size, seq_len, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }

  int layer_idx_;
};

class QwenAscendDecoder final : public nn::Module {
 public:
  QwenAscendAttention self_attn_;
  QwenAscendMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  QwenAscendDecoder() = default;

  QwenAscendDecoder(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<QwenAscendAttention>("self_attn", cfg);
    mlp_ = reg<QwenAscendMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
    attention_bias_ = cfg.attention_bias;
    paged_graph_ = cfg.paged_attention_enabled;
    paged_graph_nz_ = cfg.paged_layout == aops::PagedAttnCacheLayout::kNZ;
    rms_norm_epsilon_ = cfg.rms_norm_eps;
    max_cache_length_ = cfg.max_cache_length;
    decoder_graph_runner_.configure(cfg.max_cache_length);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

 private:
  bool attention_bias_ = false;
  bool paged_graph_ = false;
  bool paged_graph_nz_ = true;
  float rms_norm_epsilon_ = 1e-5f;
  int32_t max_cache_length_ = 0;
  bool packed_projections_ = false;
  mllm::ascend::AscendPackedProjection packed_qkv_, packed_gate_up_;
  QwenAscendDecoderGraphRunner decoder_graph_runner_;

  bool isW8A8Mode() const;
  bool canUseGraph(const Tensor& hidden_states) const;
  void ensureGraphExecutor();
  void buildDecoderGraph();
  void buildDecoderGraphFP16();
  void buildDecoderGraphW8A8();
 public:
  uint64_t graphExecutionCount() const { return decoder_graph_runner_.executionCount(); }
};

}  // namespace mllm::models::qwen_ascend

#include "mllm/models/qwen_ascend/qwen_ascend_decoder_graph.hpp"

namespace mllm::models::qwen_ascend {

class QwenAscendText final : public nn::Module {
  nn::ModuleList<QwenAscendDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;

 public:
  QwenAscendText() = default;

  QwenAscendText(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<QwenAscendDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (int idx = 0; idx < decode_blocks_.list().size(); ++idx) { decode_blocks_.list()[idx].self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = embedding_(inputs[0]);

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto local_rope_pos_ids = inputs[3];
    auto& kv_cache = args[0];

    const char* async_flag=std::getenv("MLLM_ASCEND_ASYNC_DECODER");
    auto* cache=kv_cache.get<mllm::ascend::AscendKVCache*>();
    const char* soc=aclrtGetSocName();
    mllm::ascend::AscendGraphSubmissionScope submission(
        soc && std::strstr(soc,"310P") && cache->pagedCache() && isQwenAscendDecoderGraphEnabled()
        && (!async_flag || async_flag[0]!='0'));
    for (auto& block : blocks) {
      x = block(x, llm_embedding_sin, llm_embedding_cos, local_rope_pos_ids,
                kv_cache, AnyValue(&submission))[0];
    }
    submission.finish();

    x = norm_(x);
    return {x};
  }

  uint64_t graphExecutionCount() {
    uint64_t count=0;
    for (auto& block : decode_blocks_.list()) count += block.graphExecutionCount();
    return count;
  }
  [[nodiscard]] Tensor embeddingWeight() const { return embedding_.weight(); }
};

class QwenAscendForCausalLM : public models::ARGeneration, public nn::Module {
 public:
  explicit QwenAscendForCausalLM(const QwenAscendConfig& cfg) : cfg_(cfg) {
    if (cfg.num_key_value_heads <= 0 || cfg.num_attention_heads <= 0 || cfg.num_attention_heads % cfg.num_key_value_heads != 0
        || cfg.num_hidden_layers <= 0 || cfg.head_dim <= 0 || cfg.max_cache_length <= 0) {
      throw std::invalid_argument("Invalid Qwen attention/cache configuration");
    }
    if (cfg.paged_radix_cache && !cfg.paged_attention_enabled) {
      throw std::invalid_argument("paged_attention.radix_cache requires paging enabled");
    }
    int num_key_value_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
    kv_cache_ = mllm::ascend::AscendKVCache(cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
                                            kFloat16, num_key_value_groups, !cfg.paged_attention_enabled);
    if (cfg.paged_attention_enabled) {
      if (cfg.paged_block_size <= 0 || cfg.paged_num_blocks < 0) {
        throw std::invalid_argument("Invalid paged block configuration");
      }
      const int32_t num_blocks =
          cfg.paged_num_blocks == 0 ? static_cast<int32_t>(
              (static_cast<int64_t>(cfg.max_cache_length) + cfg.paged_block_size - 1) / cfg.paged_block_size
              + (cfg.paged_radix_cache ? 1 : 0))
                                    : cfg.paged_num_blocks;
      kv_cache_.attachPagedCache(std::make_shared<mllm::ascend::AscendPagedKVCache>(mllm::ascend::AscendPagedKVCacheOptions{
          .block_size = cfg.paged_block_size,
          .num_blocks = num_blocks,
          .max_sequence_length = cfg.max_cache_length,
          .layer_nums = cfg.num_hidden_layers,
          .kv_heads = cfg.num_key_value_heads,
          .head_dim = cfg.head_dim,
          .radix_cache = cfg.paged_radix_cache,
          .layout = cfg.paged_layout,
      }));
      kv_cache_.pagedCache()->createSequence(0);
    }

    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<QwenAscendText>("model", cfg);
    if (!cfg.tie_word_embeddings) { lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false); }

    inv_freq_ = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
  }

  void load(const ParameterFile::ptr_t& parameters) {
    tied_output_nz_ = Tensor::nil();
    tied_output_source_ = Tensor::nil();
    nn::Module::load(parameters);
    prepareTiedOutputWeight();
  }

  void prepareTiedOutputWeight() {
    if (!tie_word_embeddings_) return;
    auto source=llm_.embeddingWeight();
    if (source.isNil() || source.device()!=kAscend) return;
    if (tied_output_source_.isNil() || tied_output_source_.ptr<void>()!=source.ptr<void>()) {
      tied_output_nz_=mllm::ascend::packLinearWeightNZ(source);
      tied_output_source_=source;
    }
  }

  models::ARGenerationOutputPast forwardForGeneration(const models::ARGenerationOutputPast& input,
      const models::ARGenerationArgs& args, bool greedy) override {
    const char* flag=std::getenv("MLLM_ASCEND_DEVICE_GREEDY");
    const char* soc=aclrtGetSocName();
    bool device_logits=greedy && input.at("sequence").size(0)==1 && (!flag || flag[0]!='0') && soc && std::strstr(soc,"310P");
    return forwardImpl(input,args,device_logits);
  }
  DeviceTypes generationInputDevice(const Tensor&) const override { return kCPU; }
  int64_t selectGreedy(Tensor logits, bool suppress_eos, int64_t eos_token_id) override {
    if(logits.device()!=kAscend || logits.dtype()!=kFloat16)
      return ARGeneration::selectGreedy(logits,suppress_eos,eos_token_id);
    std::vector<int64_t> suppressed;
    if(suppress_eos) {
      suppressed.assign(additional_eos_token_ids_.begin(),additional_eos_token_ids_.end());
      suppressed.push_back(eos_token_id);
    }
    return greedy_selector_.select(logits,suppressed);
  }
  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    return forwardImpl(input,args,false);
  }
  models::ARGenerationOutputPast forwardImpl(const models::ARGenerationOutputPast& input,
      const models::ARGenerationArgs& args, bool device_logits) {
    auto sequence = input.at("sequence");
    MLLM_RT_ASSERT(!sequence.isNil() && sequence.rank() == 2);

    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];
    const bool is_prefill = seq_len > 1;
    auto* paged = kv_cache_.pagedCache();
    if (paged) {
      if (batch_size <= 0 || batch_size != static_cast<int32_t>(paged_sequence_ids_.size()) || seq_len <= 0
          || (batch_size > 1 && seq_len > 1)) {
        throw std::invalid_argument("Paged forward needs selected sequence IDs; prefill one request at a time");
      }
    }

    mllm::ascend::AscendPagedKVCache::PrefixGuard prefix_guard(paged, paged && batch_size == 1 ? paged_sequence_ids_[0] : 0);
    std::vector<std::vector<int64_t>> paged_tokens;
    int32_t paged_reused_tokens = 0;
    if (paged && paged->radixEnabled()) {
      // Token identity must describe exactly the KV written by this forward.
      if (sequence.device() != kCPU || sequence.dtype() != kInt64 || !sequence.isContiguous()) {
        throw std::invalid_argument("Paged radix requires contiguous CPU int64 token IDs");
      }
      if (batch_size == 1 && !input.count("position_ids") && paged->blocks().sequence(paged_sequence_ids_[0]).length == 0) {
        const auto* data = sequence.ptr<int64_t>();
        const std::vector<int64_t> prompt(data, data + seq_len);
        paged_reused_tokens = paged->reusePrefix(paged_sequence_ids_[0], prompt);
        prefix_guard.attached(paged_reused_tokens > 0);
        if (paged_reused_tokens > 0) {
          const int32_t remaining = seq_len - paged_reused_tokens;
          auto suffix = Tensor::empty({1, remaining}, kInt64, kCPU).alloc();
          std::copy(prompt.begin() + paged_reused_tokens, prompt.end(), suffix.ptr<int64_t>());
          sequence = suffix;
          seq_len = remaining;
        }
      }
      paged_tokens.resize(batch_size);
      for (int b = 0; b < batch_size; ++b) {
        const auto* data = sequence.ptr<int64_t>() + b * seq_len;
        paged_tokens[b].assign(data, data + seq_len);
      }
    }

    // Prefix cache. Tracking token ids means reading `sequence` on the host, and reuse needs to own
    // position_ids, so bail out on anything that does not hold.
    const bool prefix_cache_usable =
        kv_cache_.prefixCache() != nullptr && batch_size == 1 && sequence.device() == kCPU && sequence.dtype() == kInt64;

    // On a prefill, skip the leading tokens whose KV a previous sequence already computed.
    // reusePrefix() copies them straight into the KV buffers, so all that is left is to drop them
    // from the input and shift position_ids. A caller-supplied position_ids would have to be sliced
    // and offset to stay consistent, so we decline to reuse in that case rather than guess.
    int32_t reused_tokens = 0;
    const bool track_prefix = prefix_cache_usable && is_prefill;
    if (track_prefix) {
      const auto* ids_ptr = sequence.ptr<int64_t>();
      resident_token_ids_.assign(ids_ptr, ids_ptr + seq_len);

      if (!input.count("position_ids")) {
        reused_tokens = kv_cache_.reusePrefix(resident_token_ids_);
        if (reused_tokens > 0) {
          const auto remaining = seq_len - reused_tokens;
          auto suffix = Tensor::empty({batch_size, remaining}, kInt64, kCPU).alloc();
          std::copy(resident_token_ids_.begin() + reused_tokens, resident_token_ids_.end(), suffix.ptr<int64_t>());
          sequence = suffix;
          seq_len = remaining;
        }
      }
    } else if (prefix_cache_usable && !resident_token_ids_.empty()) {
      resident_token_ids_.push_back(*sequence.ptr<int64_t>());
    } else if (is_prefill) {
      // Not tracking this sequence, so anything we remember from an earlier one is now stale and
      // must not be committed against these buffers.
      resident_token_ids_.clear();
    }

    Tensor position_ids = Tensor::nil();
    if (paged) {
      // Logical cache lengths are authoritative, including after request reorder/fork.
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      Tensor previous;
      if (input.count("position_ids")) {
        previous = input.at("position_ids");
        MLLM_RT_ASSERT(!previous.isNil() && previous.device() == kCPU && previous.dtype() == kInt64);
        MLLM_RT_ASSERT(previous.rank() == 2 && previous.size(0) == batch_size && previous.size(1) > 0);
        MLLM_RT_ASSERT(previous.isContiguous());
      }
      for (int b = 0; b < batch_size; ++b) {
        const int32_t start = paged->blocks().sequence(paged_sequence_ids_[b]).length;
        if (seq_len > cfg_.max_cache_length - start) { throw std::invalid_argument("Paged sequence is too long"); }
        if (!previous.isNil()) {
          if (seq_len == 1 && start > 0) {
            MLLM_RT_ASSERT_EQ(previous.ptr<int64_t>()[b * previous.size(1) + previous.size(1) - 1], start - 1);
          } else {
            MLLM_RT_ASSERT_EQ(previous.size(1), seq_len);
            for (int s = 0; s < seq_len; ++s) { MLLM_RT_ASSERT_EQ(previous.ptr<int64_t>()[b * seq_len + s], start + s); }
          }
        }
        for (int s = 0; s < seq_len; ++s) { position_ids.ptr<int64_t>()[b * seq_len + s] = start + s; }
      }
    } else if (input.count("position_ids")) {
      position_ids = input.at("position_ids");

      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
          // Reused tokens already occupy positions [0, reused_tokens).
          position_ids_ptr[b * seq_len + s] = reused_tokens + s;
        }
      }
    }

    auto [sin_emb, cos_emb] = rope_cache_.getEmbeddings(position_ids, inv_freq_, cfg_.max_cache_length);
    auto local_rope_pos_ids = makeLocalRoPEPositionIds(batch_size, seq_len);
    Tensor input_ids_device = sequence.to(kAscend);
    if (paged && !paged->beginStep(paged_sequence_ids_, seq_len, paged_tokens)) {
      throw std::runtime_error(
          "Paged KV pool exhausted: release a finished sequence, reduce the batch, or increase num_blocks");
    }
    mllm::ascend::AscendPagedKVCache::StepGuard paged_step(paged);
    auto hidden_states = llm_(input_ids_device, sin_emb, cos_emb, local_rope_pos_ids, AnyValue(&kv_cache_))[0];

    // The prompt's KV is complete now, so publish it. Generated tokens are published later, by
    // clearCache(), once the sequence is actually finished.
    if (track_prefix && !resident_token_ids_.empty()) { kv_cache_.commitPrefix(resident_token_ids_); }

    auto hidden_seq_len = hidden_states.shape()[1];
    hidden_states = hidden_states[{kAll, {hidden_seq_len - 1}, kAll}];

    if (tie_word_embeddings_) {
      auto embed_weight = llm_.embeddingWeight();
      prepareTiedOutputWeight();
      hidden_states = tied_output_nz_.isNil()
          ? nn::functional::matmul(hidden_states, embed_weight, false, true)
          : tied_output_runner_.execute(hidden_states, tied_output_nz_, cfg_.vocab_size);
    } else {
      hidden_states = lm_head_(hidden_states);
    }

    if (!device_logits) hidden_states = hidden_states.to(kCPU).to(kFloat32);
    models::ARGenerationOutputPast result{
        {"sequence", hidden_states},
        {"position_ids", position_ids},
    };
    if (paged) { paged->commitStep(); }
    prefix_guard.commit();
    last_paged_reused_tokens_ = paged_reused_tokens;
    return result;
  }

  Tensor forward(const Tensor& input_ids) {
    clearCache();
    auto result = forward({{"sequence", input_ids}}, {});
    return result.at("sequence");
  }

  uint64_t graphExecutionCount() { return llm_.graphExecutionCount(); }

  inline mllm::ascend::AscendKVCache& kvCache() { return kv_cache_; }

  using PagedSequenceId = mllm::ascend::AscendPagedKVCache::SequenceId;
  int32_t lastPagedReusedTokens() const { return last_paged_reused_tokens_; }
  void clearPagedPrefixCache() { checkedPagedCache().clearPrefixCache(); }
  void createPagedSequence(PagedSequenceId id) { checkedPagedCache().createSequence(id); }
  void forkPagedSequence(PagedSequenceId source, PagedSequenceId destination) {
    checkedPagedCache().forkSequence(source, destination);
  }
  void releasePagedSequence(PagedSequenceId id) {
    checkedPagedCache().releaseSequence(id);
    paged_sequence_ids_.erase(std::remove(paged_sequence_ids_.begin(), paged_sequence_ids_.end(), id),
                              paged_sequence_ids_.end());
  }
  void selectPagedSequences(const std::vector<PagedSequenceId>& ids) {
    auto& cache = checkedPagedCache();
    if (cache.pending() || ids.empty() || ids.size() > 2048) {
      throw std::invalid_argument("Invalid paged sequence selection");
    }
    std::unordered_set<PagedSequenceId> seen;
    for (auto id : ids) {
      if (!cache.blocks().contains(id) || !seen.insert(id).second) {
        throw std::invalid_argument("Unknown or duplicate paged sequence ID");
      }
    }
    paged_sequence_ids_ = ids;
  }

  /// Turn on cross-sequence prefix reuse. Sizing the pool is the caller's call: num_blocks *
  /// block_size tokens of KV are allocated per layer, on top of the existing max_cache_length
  /// working buffers.
  void enablePrefixCache(int32_t num_blocks, int32_t block_size = 128) {
    if (kv_cache_.pagedCache()) { throw std::logic_error("Archive prefix cache cannot be combined with paged KV"); }
    kv_cache_.attachPrefixCache(std::make_shared<mllm::ascend::AscendPrefixCache>(mllm::ascend::AscendPrefixCacheOptions{
        .block_size = block_size,
        .num_blocks = num_blocks,
        .layer_nums = cfg_.num_hidden_layers,
        .kv_heads = cfg_.num_key_value_heads,
        .head_dim = cfg_.head_dim,
        .dtype = kFloat16,
    }));
  }

  void clearCache() {
    // Last chance to publish the generated tokens: clearCache() resets the sequence counters that
    // commitPrefix() checks against.
    if (kv_cache_.prefixCache() && !resident_token_ids_.empty()) { kv_cache_.commitPrefix(resident_token_ids_); }
    resident_token_ids_.clear();
    last_paged_reused_tokens_ = 0;
    kv_cache_.clearCache();
    if (auto* paged = kv_cache_.pagedCache()) {
      paged->createSequence(0);
      paged_sequence_ids_ = {0};
    }
    rope_cache_.clear();
  }

  void perfSummary() override {
    if (llm_decode_end_time_.time_since_epoch().count() == 0 && ar_steps_ > 1) { decodeEventEndTimePoint(); }
    models::ARGeneration::perfSummary();
  }

 private:
  mllm::ascend::AscendPagedKVCache& checkedPagedCache() {
    auto* cache = kv_cache_.pagedCache();
    if (!cache) { throw std::logic_error("Enable paged_attention in QwenAscendConfig before constructing the model"); }
    return *cache;
  }
  int32_t last_paged_reused_tokens_ = 0;
  std::vector<PagedSequenceId> paged_sequence_ids_{0};
  const QwenAscendConfig& cfg_;
  QwenAscendText llm_;
  nn::Linear lm_head_;
  Tensor inv_freq_;
  Tensor tied_output_nz_;
  Tensor tied_output_source_;
  mllm::ascend::AscendNZLinearRunner tied_output_runner_;
  mllm::ascend::AscendGreedySelector greedy_selector_;
  bool tie_word_embeddings_ = false;
  mllm::ascend::AscendKVCache kv_cache_;
  QwenAscendRoPECache rope_cache_;
  // Every token currently resident in the KV cache: the prompt, plus each token generated since.
  // Only tracked when a prefix cache is attached.
  std::vector<int64_t> resident_token_ids_;
};

}  // namespace mllm::models::qwen_ascend
