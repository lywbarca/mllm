// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/ascend/graph/AscendPagedAttentionPlugin.hpp"
#include <string>
#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendAttentionWithKVCachePluginOperation.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/graph/AscendLinearW8A8PluginOperation.hpp"
#include "mllm/models/qwen_ascend/qwen_ascend_graph_ops.hpp"

namespace mllm::models::qwen_ascend {

inline std::vector<Tensor> QwenAscendDecoder::forward(const std::vector<Tensor>& inputs,
                                                      const std::vector<AnyValue>& args) {
  auto llm_embedding_sin = inputs[1];
  auto llm_embedding_cos = inputs[2];
  auto local_rope_pos_ids = inputs[3];
  auto past_kv_cache = args[0].get<mllm::ascend::AscendKVCache*>();

  auto* paged = past_kv_cache->pagedCache();
  if (canUseGraph(inputs[0])) {
    if (paged) paged->prepareGraphMetadata();
    ensureGraphExecutor();

    const int batch_size = inputs[0].shape()[0];
    const int seq_len = inputs[0].shape()[1];
    const auto dtype = inputs[0].dtype();
    const auto device = inputs[0].device();
    const int old_seq_len = paged ? 0 : past_kv_cache->getCurrentSeqCnt(self_attn_.layer_idx_);
    if (old_seq_len + seq_len > past_kv_cache->getMaxCacheLength()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError,
                      "AscendKVCache: sequence length {} + {} exceeds max_cache_length {}",
                      old_seq_len,
                      seq_len,
                      past_kv_cache->getMaxCacheLength());
    }

    auto attn_scale = decoder_graph_runner_.attentionScaleTensor(self_attn_.head_dim_, dtype, device);
    auto current_seq_len = decoder_graph_runner_.currentSeqLenTensor(old_seq_len, paged != nullptr);
    auto output = Tensor::empty({batch_size, seq_len, self_attn_.hidden_size_}, dtype, device).alloc();

    std::vector<Tensor> graph_inputs;
    if (isW8A8Mode()) {
      auto& q_op    = checkedAscendW8A8LinearOp(self_attn_.q_proj_);
      auto& k_op    = checkedAscendW8A8LinearOp(self_attn_.k_proj_);
      auto& v_op    = checkedAscendW8A8LinearOp(self_attn_.v_proj_);
      auto& o_op    = checkedAscendW8A8LinearOp(self_attn_.o_proj_);
      auto& gate_op = checkedAscendW8A8LinearOp(mlp_.gate_proj_);
      auto& up_op   = checkedAscendW8A8LinearOp(mlp_.up_proj_);
      auto& down_op = checkedAscendW8A8LinearOp(mlp_.down_proj_);
      graph_inputs = {
          inputs[0],
          input_layer_norm_.weight(),
          llm_embedding_sin,
          llm_embedding_cos,
          local_rope_pos_ids,
          self_attn_.q_proj_.weight(), q_op.biasInt32Npu(), q_op.deqScaleNpu(),
          self_attn_.k_proj_.weight(), k_op.biasInt32Npu(), k_op.deqScaleNpu(),
          self_attn_.v_proj_.weight(), v_op.biasInt32Npu(), v_op.deqScaleNpu(),
          self_attn_.rms_norm_q_.weight(),
          self_attn_.rms_norm_k_.weight(),
          paged ? paged->keyPool(self_attn_.layer_idx_) : past_kv_cache->getKCacheBuffer(self_attn_.layer_idx_),
          paged ? paged->valuePool(self_attn_.layer_idx_) : past_kv_cache->getVCacheBuffer(self_attn_.layer_idx_),
          current_seq_len,
          attn_scale,
          self_attn_.o_proj_.weight(), o_op.biasInt32Npu(), o_op.deqScaleNpu(),
          post_attention_layer_norm_.weight(),
          mlp_.gate_proj_.weight(), gate_op.biasInt32Npu(), gate_op.deqScaleNpu(),
          mlp_.up_proj_.weight(),   up_op.biasInt32Npu(),   up_op.deqScaleNpu(),
          mlp_.down_proj_.weight(), down_op.biasInt32Npu(), down_op.deqScaleNpu(),
      };
    } else {
      graph_inputs = {
          inputs[0],
          input_layer_norm_.weight(),
          llm_embedding_sin,
          llm_embedding_cos,
          local_rope_pos_ids,
          self_attn_.q_proj_.weight(),
          self_attn_.k_proj_.weight(),
          self_attn_.v_proj_.weight(),
          self_attn_.rms_norm_q_.weight(),
          self_attn_.rms_norm_k_.weight(),
          paged ? paged->keyPool(self_attn_.layer_idx_) : past_kv_cache->getKCacheBuffer(self_attn_.layer_idx_),
          paged ? paged->valuePool(self_attn_.layer_idx_) : past_kv_cache->getVCacheBuffer(self_attn_.layer_idx_),
          current_seq_len,
          attn_scale,
          self_attn_.o_proj_.weight(),
          post_attention_layer_norm_.weight(),
          mlp_.gate_proj_.weight(),
          mlp_.up_proj_.weight(),
          mlp_.down_proj_.weight(),
      };
      if (attention_bias_) {
        graph_inputs.push_back(self_attn_.q_proj_.bias());
        graph_inputs.push_back(self_attn_.k_proj_.bias());
        graph_inputs.push_back(self_attn_.v_proj_.bias());
        graph_inputs.push_back(self_attn_.o_proj_.bias());
      }
    }
    if (paged) {
      graph_inputs.insert(graph_inputs.end(), {paged->slotMapping(), paged->graphBlockTable(),
          paged->graphLengthsHost(), paged->graphLengthsDevice()});
    }
    std::vector<size_t> nz_inputs;
    {
      const std::vector<std::pair<size_t, const nn::Linear*>> weights = isW8A8Mode()
          ? std::vector<std::pair<size_t,const nn::Linear*>>{
              {5,&self_attn_.q_proj_},{8,&self_attn_.k_proj_},{11,&self_attn_.v_proj_},
              {20,&self_attn_.o_proj_},{24,&mlp_.gate_proj_},{27,&mlp_.up_proj_},{30,&mlp_.down_proj_}}
          : std::vector<std::pair<size_t,const nn::Linear*>>{
              {5,&self_attn_.q_proj_},{6,&self_attn_.k_proj_},{7,&self_attn_.v_proj_},
              {14,&self_attn_.o_proj_},{16,&mlp_.gate_proj_},{17,&mlp_.up_proj_},{18,&mlp_.down_proj_}};
      for (auto [index, linear] : weights) {
        auto* op=getAscendLinearOpPtr(*linear);
        if (op && !op->nzWeight().isNil()) {
          MLLM_RT_ASSERT(graph_inputs[index].ptr<void>()==linear->weight().ptr<void>());
          graph_inputs[index]=op->nzWeight(); nz_inputs.push_back(index);
        }
      }
    }
    if (packed_projections_) {
      graph_inputs[5]=packed_qkv_.prepare({self_attn_.q_proj_.weight(),self_attn_.k_proj_.weight(),self_attn_.v_proj_.weight()});
      graph_inputs[16]=packed_gate_up_.prepare({mlp_.gate_proj_.weight(),mlp_.up_proj_.weight()});
    }
    std::vector<Tensor> graph_outputs = {output};
    auto* submission = args.size()>1 ? args[1].get<mllm::ascend::AscendGraphSubmissionScope*>() : nullptr;
    const bool deferred = submission && submission->enabled();
    if (deferred) submission->retain(graph_inputs,graph_outputs);
    decoder_graph_runner_.execute(graph_inputs, graph_outputs, nz_inputs, !deferred);
    if (paged) paged->markGraphLayerWritten(self_attn_.layer_idx_);
    else past_kv_cache->advanceSeqCnt(self_attn_.layer_idx_, seq_len);
    return {output};
  }

  auto x = input_layer_norm_(inputs[0]);
  x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, local_rope_pos_ids, args[0])[0];
  auto tmp = x + inputs[0];
  x = post_attention_layer_norm_(tmp);
  x = mlp_(x)[0];
  x = x + tmp;
  return {x};
}

inline bool QwenAscendDecoder::isW8A8Mode() const {
  return !self_attn_.q_proj_.weight().isNil()
      && self_attn_.q_proj_.weight().dtype() == kInt8;
}

inline bool QwenAscendDecoder::canUseGraph(const Tensor& hidden_states) const {
  if (!isQwenAscendDecoderGraphEnabled()) return false;
  if (!paged_graph_ && hidden_states.shape()[0] != 1) return false;
  if (hidden_states.device() != kAscend) return false;
  if (hidden_states.dtype() != kFloat16) return false;

  if (input_layer_norm_.weight().isNil()
      || input_layer_norm_.weight().device() != kAscend) return false;
  if (post_attention_layer_norm_.weight().isNil()
      || post_attention_layer_norm_.weight().device() != kAscend) return false;

  auto weight_on_npu = [](const nn::Linear& l) {
    return !l.weight().isNil() && l.weight().device() == kAscend;
  };
  if (!weight_on_npu(self_attn_.q_proj_) || !weight_on_npu(self_attn_.k_proj_)
      || !weight_on_npu(self_attn_.v_proj_) || !weight_on_npu(self_attn_.o_proj_)
      || !weight_on_npu(mlp_.gate_proj_) || !weight_on_npu(mlp_.up_proj_)
      || !weight_on_npu(mlp_.down_proj_)) return false;

  const DataTypes w_dtype = self_attn_.q_proj_.weight().dtype();
  if (self_attn_.k_proj_.weight().dtype() != w_dtype
      || self_attn_.v_proj_.weight().dtype() != w_dtype
      || self_attn_.o_proj_.weight().dtype() != w_dtype
      || mlp_.gate_proj_.weight().dtype()    != w_dtype
      || mlp_.up_proj_.weight().dtype()      != w_dtype
      || mlp_.down_proj_.weight().dtype()    != w_dtype) return false;

  if (w_dtype == kFloat16) {
    if (attention_bias_) {
      auto bias_on_npu = [](const nn::Linear& l) {
        return !l.bias().isNil() && l.bias().device() == kAscend;
      };
      if (!bias_on_npu(self_attn_.q_proj_) || !bias_on_npu(self_attn_.k_proj_)
          || !bias_on_npu(self_attn_.v_proj_) || !bias_on_npu(self_attn_.o_proj_)) return false;
    }
    return true;
  }

  if (w_dtype == kInt8) {
    auto w8a8_ready = [](const nn::Linear& l) {
      const auto* op = getAscendLinearOpPtr(l);
      return op != nullptr && op->isW8A8();
    };
    if (!w8a8_ready(self_attn_.q_proj_) || !w8a8_ready(self_attn_.k_proj_)
        || !w8a8_ready(self_attn_.v_proj_) || !w8a8_ready(self_attn_.o_proj_)
        || !w8a8_ready(mlp_.gate_proj_)    || !w8a8_ready(mlp_.up_proj_)
        || !w8a8_ready(mlp_.down_proj_)) return false;
    return true;
  }

  return false;
}

inline void QwenAscendDecoder::ensureGraphExecutor() {
  if (!decoder_graph_runner_.hasExecutor()) {
    buildDecoderGraph();
  }
}

inline void QwenAscendDecoder::buildDecoderGraph() {
  if (isW8A8Mode()) {
    buildDecoderGraphW8A8();
    return;
  }
  const char* packed_flag=std::getenv("MLLM_ASCEND_PACKED_PROJECTIONS");
  packed_projections_=!attention_bias_ && mllm::ascend::useNZWeights()
      && (!packed_flag || packed_flag[0]!='0');
  if (packed_projections_) {
    for (const auto& w : {self_attn_.q_proj_.weight(),self_attn_.k_proj_.weight(),
                         self_attn_.v_proj_.weight(),mlp_.gate_proj_.weight(),mlp_.up_proj_.weight()})
      if(w.rank()!=2 || w.size(0)%16 || w.size(1)%16) packed_projections_=false;
  }
  buildDecoderGraphFP16();
}

inline void QwenAscendDecoder::buildDecoderGraphFP16() {
  using namespace mllm::ascend;

  std::vector<std::string> input_names = {
      "hidden_states",
      "input_norm_weight",
      "sin_emb",
      "cos_emb",
      "rope_pos_ids",
      "q_weight",
      "k_weight",
      "v_weight",
      "q_norm_weight",
      "k_norm_weight",
      "k_cache_storage",
      "v_cache_storage",
      "current_seq_len",
      "attn_scale",
      "o_weight",
      "post_norm_weight",
      "gate_weight",
      "up_weight",
      "down_weight",
  };
  if (attention_bias_) {
    input_names.push_back("q_bias");
    input_names.push_back("k_bias");
    input_names.push_back("v_bias");
    input_names.push_back("o_bias");
  }

  if (paged_graph_) {
    input_names.insert(input_names.end(), {"paged_slots","paged_table","paged_lengths_host","paged_lengths_device"});
  }
  AscendGraphBuilder builder;
  builder.beginGraph(
      (paged_graph_ ? "QwenAscendPagedDecoderGraph_" : "QwenAscendDecoderGraph_") + std::to_string(self_attn_.layer_idx_),
      input_names,
      {"decoder_output"},
      [this](const atb::SVector<atb::TensorDesc>& inTensorDescs,
             atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
        if (inTensorDescs.empty() || outTensorDescs.empty()) {
          return atb::NO_ERROR;
        }
        const auto& hidden_desc = inTensorDescs.at(0);
        outTensorDescs.at(0) = hidden_desc;
        return atb::NO_ERROR;
      });

  builder.addOperation(
      createRmsNormGraphOp(rms_norm_epsilon_),
      {"hidden_states", "input_norm_weight"},
      {"normed_hidden_states"});

  if (packed_projections_) {
    builder.addOperation(createLinearGraphOp(false), {"normed_hidden_states","q_weight"}, {"qkv_linear"});
    atb::infer::SplitParam split;
    split.splitDim=2; split.splitNum=3;
    split.splitSizes={self_attn_.num_attention_heads_*self_attn_.head_dim_,
                     self_attn_.num_key_value_heads_*self_attn_.head_dim_,
                     self_attn_.num_key_value_heads_*self_attn_.head_dim_};
    atb::Operation* op=nullptr; MLLM_ATB_CHECK(atb::CreateOperation(split,&op));
    builder.addOperation(op,{"qkv_linear"},{"q_linear","k_linear","v_linear"});
  } else if (attention_bias_) {
    builder.addOperation(createLinearGraphOp(true), {"normed_hidden_states", "q_weight", "q_bias"}, {"q_linear"});
    builder.addOperation(createLinearGraphOp(true), {"normed_hidden_states", "k_weight", "k_bias"}, {"k_linear"});
    builder.addOperation(createLinearGraphOp(true), {"normed_hidden_states", "v_weight", "v_bias"}, {"v_linear"});
  } else {
    builder.addOperation(createLinearGraphOp(false), {"normed_hidden_states", "q_weight"}, {"q_linear"});
    builder.addOperation(createLinearGraphOp(false), {"normed_hidden_states", "k_weight"}, {"k_linear"});
    builder.addOperation(createLinearGraphOp(false), {"normed_hidden_states", "v_weight"}, {"v_linear"});
  }

  builder.reshape("q_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_attention_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "q_linear_4d");
  builder.reshape("k_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_key_value_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "k_linear_4d");
  builder.reshape("v_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_key_value_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "v_linear_4d");

  builder.addOperation(createRmsNormGraphOp(self_attn_.rms_norm_epsilon_),
                       {"q_linear_4d", "q_norm_weight"},
                       {"q_normed"});
  builder.addOperation(createRmsNormGraphOp(self_attn_.rms_norm_epsilon_),
                       {"k_linear_4d", "k_norm_weight"},
                       {"k_normed"});

  builder.reshape("rope_pos_ids",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 1;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                  },
                  "rope_pos_ids_1d");
  builder.reshape("cos_emb",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 2;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                    newShape.dims[1] = oldShape.dims[2];
                  },
                  "cos_emb_2d");
  builder.reshape("sin_emb",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 2;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                    newShape.dims[1] = oldShape.dims[2];
                  },
                  "sin_emb_2d");

  builder.addOperation(createRoPEGraphOp(),
                       {"q_normed", "q_normed", "cos_emb_2d", "sin_emb_2d", "rope_pos_ids_1d"},
                       {"q_rope_4d", "q_rope_unused"});
  builder.addOperation(createRoPEGraphOp(),
                       {"k_normed", "k_normed", "cos_emb_2d", "sin_emb_2d", "rope_pos_ids_1d"},
                       {"k_rope_4d", "k_rope_unused"});

  if (paged_graph_) {
    builder.addOperation(new AscendPagedAttentionPlugin(self_attn_.num_attention_heads_,
        self_attn_.num_key_value_heads_, self_attn_.head_dim_, paged_graph_nz_),
        {"q_rope_4d","k_rope_4d","v_linear_4d","k_cache_storage","v_cache_storage",
         "paged_slots","paged_table","paged_lengths_host","paged_lengths_device"},
        {"attn_context_transposed"});
  } else {
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"q_rope_4d"}, {"query_states"});
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"k_rope_4d"}, {"key_states"});
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"v_linear_4d"}, {"value_states"});

    std::vector<std::string> attention_inputs = {
        "query_states",
        "key_states",
        "value_states",
        "k_cache_storage",
        "v_cache_storage",
        "current_seq_len",
        "attn_scale",
    };
    const std::string layer_sfx = "_L" + std::to_string(self_attn_.layer_idx_);
    builder.addOperation(createAttentionWithKVCachePluginGraphOp(self_attn_.num_attention_heads_,
                                                                 self_attn_.num_key_value_heads_,
                                                                 self_attn_.head_dim_,
                                                                 max_cache_length_,
                                                                 false,
                                                                 0,
                                                                 layer_sfx,
                                                                 decoder_graph_runner_.setupBucketSize()),
                         attention_inputs,
                         {"attn_context"});
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"attn_context"}, {"attn_context_transposed"});
  }
  builder.reshape("attn_context_transposed",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 3;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = oldShape.dims[2] * oldShape.dims[3];
                  },
                  "attn_context_flat");

  if (attention_bias_) {
    builder.addOperation(createLinearGraphOp(true), {"attn_context_flat", "o_weight", "o_bias"}, {"attn_output"});
  } else {
    builder.addOperation(createLinearGraphOp(false), {"attn_context_flat", "o_weight"}, {"attn_output"});
  }
  builder.addOperation(createAddGraphOp(), {"attn_output", "hidden_states"}, {"residual_after_attn"});
  builder.addOperation(createRmsNormGraphOp(rms_norm_epsilon_),
                       {"residual_after_attn", "post_norm_weight"},
                       {"mlp_input"});
  const char* fused_flag=std::getenv("MLLM_ASCEND_FUSED_MLP_ACTIVATION");
  const bool fused_activation=packed_projections_ && (!fused_flag || fused_flag[0]!='0');
  if (packed_projections_) {
    builder.addOperation(createLinearGraphOp(false), {"mlp_input","gate_weight"}, {"gate_up"});
    if (fused_activation) {
      atb::infer::ActivationParam activation;
      activation.activationType=atb::infer::ACTIVATION_SWIGLU_FORWARD;
      activation.dim=-1;
      atb::Operation* op=nullptr; MLLM_ATB_CHECK(atb::CreateOperation(activation,&op));
      builder.addOperation(op,{"gate_up"},{"mlp_gated"});
    } else {
      atb::infer::SplitParam split; split.splitDim=2; split.splitNum=2;
      atb::Operation* op=nullptr; MLLM_ATB_CHECK(atb::CreateOperation(split,&op));
      builder.addOperation(op,{"gate_up"},{"gate_proj_out","up_proj_out"});
    }
  } else {
    builder.addOperation(createLinearGraphOp(false), {"mlp_input", "gate_weight"}, {"gate_proj_out"});
    builder.addOperation(createLinearGraphOp(false), {"mlp_input", "up_weight"}, {"up_proj_out"});
  }
  if (!fused_activation) {
    builder.addOperation(createSiLUGraphOp(), {"gate_proj_out"}, {"gate_activated"});
    builder.addOperation(createMulGraphOp(), {"gate_activated", "up_proj_out"}, {"mlp_gated"});
  }
  builder.addOperation(createLinearGraphOp(false), {"mlp_gated", "down_weight"}, {"mlp_output"});
  builder.addOperation(createAddGraphOp(), {"mlp_output", "residual_after_attn"}, {"decoder_output"});

  decoder_graph_runner_.setExecutor(
      std::make_unique<AscendGraphExecutor>(builder.build(), mllm::ascend::getGlobalAtbContext()));
}

inline void QwenAscendDecoder::buildDecoderGraphW8A8() {
  using namespace mllm::ascend;

  auto& q_op    = checkedAscendW8A8LinearOp(self_attn_.q_proj_);
  auto& k_op    = checkedAscendW8A8LinearOp(self_attn_.k_proj_);
  auto& v_op    = checkedAscendW8A8LinearOp(self_attn_.v_proj_);
  auto& o_op    = checkedAscendW8A8LinearOp(self_attn_.o_proj_);
  auto& gate_op = checkedAscendW8A8LinearOp(mlp_.gate_proj_);
  auto& up_op   = checkedAscendW8A8LinearOp(mlp_.up_proj_);
  auto& down_op = checkedAscendW8A8LinearOp(mlp_.down_proj_);

  std::vector<std::string> input_names = {
      "hidden_states",
      "input_norm_weight",
      "sin_emb",
      "cos_emb",
      "rope_pos_ids",
      "q_weight", "q_bias_i32", "q_deq_scale",
      "k_weight", "k_bias_i32", "k_deq_scale",
      "v_weight", "v_bias_i32", "v_deq_scale",
      "q_norm_weight",
      "k_norm_weight",
      "k_cache_storage",
      "v_cache_storage",
      "current_seq_len",
      "attn_scale",
      "o_weight", "o_bias_i32", "o_deq_scale",
      "post_norm_weight",
      "gate_weight", "gate_bias_i32", "gate_deq_scale",
      "up_weight",   "up_bias_i32",   "up_deq_scale",
      "down_weight", "down_bias_i32", "down_deq_scale",
  };

  if (paged_graph_) {
    input_names.insert(input_names.end(), {"paged_slots","paged_table","paged_lengths_host","paged_lengths_device"});
  }
  AscendGraphBuilder builder;
  builder.beginGraph(
      (paged_graph_ ? "QwenAscendPagedDecoderGraphW8A8_" : "QwenAscendDecoderGraphW8A8_") + std::to_string(self_attn_.layer_idx_),
      input_names,
      {"decoder_output"},
      [this](const atb::SVector<atb::TensorDesc>& inTensorDescs,
             atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
        if (inTensorDescs.empty() || outTensorDescs.empty()) return atb::NO_ERROR;
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
      });

  builder.addOperation(
      createRmsNormGraphOp(rms_norm_epsilon_),
      {"hidden_states", "input_norm_weight"},
      {"normed_hidden_states"});

  const std::string layer_sfx = "_L" + std::to_string(self_attn_.layer_idx_);
  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / q_op.scaleX(), "_q" + layer_sfx),
      {"normed_hidden_states", "q_weight", "q_bias_i32", "q_deq_scale"},
      {"q_linear"});
  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / k_op.scaleX(), "_k" + layer_sfx),
      {"normed_hidden_states", "k_weight", "k_bias_i32", "k_deq_scale"},
      {"k_linear"});
  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / v_op.scaleX(), "_v" + layer_sfx),
      {"normed_hidden_states", "v_weight", "v_bias_i32", "v_deq_scale"},
      {"v_linear"});

  builder.reshape("q_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_attention_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "q_linear_4d");
  builder.reshape("k_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_key_value_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "k_linear_4d");
  builder.reshape("v_linear",
                  [this](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 4;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = self_attn_.num_key_value_heads_;
                    newShape.dims[3] = self_attn_.head_dim_;
                  },
                  "v_linear_4d");

  builder.addOperation(createRmsNormGraphOp(self_attn_.rms_norm_epsilon_),
                       {"q_linear_4d", "q_norm_weight"},
                       {"q_normed"});
  builder.addOperation(createRmsNormGraphOp(self_attn_.rms_norm_epsilon_),
                       {"k_linear_4d", "k_norm_weight"},
                       {"k_normed"});

  builder.reshape("rope_pos_ids",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 1;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                  },
                  "rope_pos_ids_1d");
  builder.reshape("cos_emb",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 2;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                    newShape.dims[1] = oldShape.dims[2];
                  },
                  "cos_emb_2d");
  builder.reshape("sin_emb",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 2;
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                    newShape.dims[1] = oldShape.dims[2];
                  },
                  "sin_emb_2d");

  builder.addOperation(createRoPEGraphOp(),
                       {"q_normed", "q_normed", "cos_emb_2d", "sin_emb_2d", "rope_pos_ids_1d"},
                       {"q_rope_4d", "q_rope_unused"});
  builder.addOperation(createRoPEGraphOp(),
                       {"k_normed", "k_normed", "cos_emb_2d", "sin_emb_2d", "rope_pos_ids_1d"},
                       {"k_rope_4d", "k_rope_unused"});

  if (paged_graph_) {
    builder.addOperation(new AscendPagedAttentionPlugin(self_attn_.num_attention_heads_,
        self_attn_.num_key_value_heads_, self_attn_.head_dim_, paged_graph_nz_),
        {"q_rope_4d","k_rope_4d","v_linear_4d","k_cache_storage","v_cache_storage",
         "paged_slots","paged_table","paged_lengths_host","paged_lengths_device"},
        {"attn_context_transposed"});
  } else {
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"q_rope_4d"}, {"query_states"});
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"k_rope_4d"}, {"key_states"});
    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"v_linear_4d"}, {"value_states"});

    builder.addOperation(createAttentionWithKVCachePluginGraphOp(self_attn_.num_attention_heads_,
                                                                 self_attn_.num_key_value_heads_,
                                                                 self_attn_.head_dim_,
                                                                 max_cache_length_,
                                                                 false,
                                                                 0,
                                                                 layer_sfx,
                                                                 decoder_graph_runner_.setupBucketSize()),
                         {"query_states", "key_states", "value_states",
                          "k_cache_storage", "v_cache_storage",
                          "current_seq_len", "attn_scale"},
                         {"attn_context"});

    builder.addOperation(createTransposeGraphOp(4, 1, 2), {"attn_context"}, {"attn_context_transposed"});
  }
  builder.reshape("attn_context_transposed",
                  [](const atb::Dims& oldShape, atb::Dims& newShape) {
                    newShape.dimNum = 3;
                    newShape.dims[0] = oldShape.dims[0];
                    newShape.dims[1] = oldShape.dims[1];
                    newShape.dims[2] = oldShape.dims[2] * oldShape.dims[3];
                  },
                  "attn_context_flat");

  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / o_op.scaleX(), "_o" + layer_sfx),
      {"attn_context_flat", "o_weight", "o_bias_i32", "o_deq_scale"},
      {"attn_output"});

  builder.addOperation(createAddGraphOp(), {"attn_output", "hidden_states"}, {"residual_after_attn"});
  builder.addOperation(createRmsNormGraphOp(rms_norm_epsilon_),
                       {"residual_after_attn", "post_norm_weight"},
                       {"mlp_input"});

  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / gate_op.scaleX(), "_gate" + layer_sfx),
      {"mlp_input", "gate_weight", "gate_bias_i32", "gate_deq_scale"},
      {"gate_proj_out"});
  builder.addOperation(createSiLUGraphOp(), {"gate_proj_out"}, {"gate_activated"});
  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / up_op.scaleX(), "_up" + layer_sfx),
      {"mlp_input", "up_weight", "up_bias_i32", "up_deq_scale"},
      {"up_proj_out"});
  builder.addOperation(createMulGraphOp(), {"gate_activated", "up_proj_out"}, {"mlp_gated"});
  builder.addOperation(
      createLinearW8A8PluginGraphOp(1.0f / down_op.scaleX(), "_down" + layer_sfx),
      {"mlp_gated", "down_weight", "down_bias_i32", "down_deq_scale"},
      {"mlp_output"});

  builder.addOperation(createAddGraphOp(), {"mlp_output", "residual_after_attn"}, {"decoder_output"});

  decoder_graph_runner_.setExecutor(
      std::make_unique<AscendGraphExecutor>(builder.build(), mllm::ascend::getGlobalAtbContext()));
}

}  // namespace mllm::models::qwen_ascend
