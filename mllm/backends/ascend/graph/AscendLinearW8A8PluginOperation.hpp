// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <string>
#include <atb/operation_infra.h>
#include "mllm/core/Tensor.hpp"

namespace mllm::ascend {

// FP16 -> ATB QUANT_PER_CHANNEL -> INT8 Linear with fused dequant -> FP16.
// The static activation scale is FP16 (the native quant operator's contract).
// Inputs: x_fp16, weight_int8 (ND or NZ), bias_int32, dequant_scale_uint64.
// KV cache is not involved in this operation.
class AscendLinearW8A8PluginOperation final : public atb::OperationInfra {
 public:
  explicit AscendLinearW8A8PluginOperation(float inv_scale_x, std::string name_suffix = "");
  ~AscendLinearW8A8PluginOperation() override;
  std::string GetName() const override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& in,
                         atb::SVector<atb::TensorDesc>& out) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack&, uint64_t&, atb::Context*) override;
  atb::Status Execute(const atb::VariantPack&, uint8_t*, uint64_t, atb::Context*) override;

 private:
  std::string name_suffix_;
  Tensor scale_, offset_;
  atb::Operation* quant_op_{nullptr};
  atb::Operation* linear_op_{nullptr};
  uint64_t quant_bytes_{0}, quant_ws_bytes_{0}, linear_ws_bytes_{0}, total_bytes_{0};
  void makePacks(const atb::VariantPack&, uint8_t*, atb::VariantPack&, atb::VariantPack&) const;
};
atb::Operation* createLinearW8A8PluginGraphOp(float inv_scale_x, std::string name_suffix = "");
}  // namespace mllm::ascend
