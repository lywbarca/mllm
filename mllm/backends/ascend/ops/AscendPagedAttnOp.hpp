// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/PagedAttnOp.hpp"

namespace mllm::ascend {

// Causal ATB attention: q[B,S,H,D] (S=1 or B=1), K/V blocks, blockTables[B,M],
// contextLens[B]. K/V and blockTables reside on Ascend; contextLens is on CPU.
// Lengths include all S appended tokens. Chunk queries read history+i+1 tokens.
// Returns {output[B,S,H,D], nil}, preserving nn::PagedAttn's two-output ABI.
class AscendPagedAttnOp final : public aops::PagedAttnOp {
 public:
  explicit AscendPagedAttnOp(const aops::PagedAttnOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  void validateInputs(const std::vector<Tensor>& inputs) const;
};

class AscendPagedAttnOpFactory final : public TypedOpFactory<OpTypes::kPagedAttn, aops::PagedAttnOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::PagedAttnOpOptions& options) override {
    return std::make_shared<AscendPagedAttnOp>(options);
  }
};

}  // namespace mllm::ascend
