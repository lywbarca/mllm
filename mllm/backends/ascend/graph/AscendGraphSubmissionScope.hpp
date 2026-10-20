// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {
// One model step, one stream, distinct persistent graph executors per layer.
// Hold every input/output (including host metadata) until the stream completes.
// The scope must end before reusing graph state/workspace or changing KV metadata.
class AscendGraphSubmissionScope {
 public:
  explicit AscendGraphSubmissionScope(bool enabled) : enabled_(enabled) {}
  ~AscendGraphSubmissionScope() { finish(); }
  AscendGraphSubmissionScope(const AscendGraphSubmissionScope&) = delete;
  AscendGraphSubmissionScope& operator=(const AscendGraphSubmissionScope&) = delete;
  bool enabled() const { return enabled_; }
  void retain(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs) {
    retained_.insert(retained_.end(),inputs.begin(),inputs.end());
    retained_.insert(retained_.end(),outputs.begin(),outputs.end());
    pending_=true;
  }
  void finish() {
    if (pending_) {
      syncGlobalAtbStream();
      pending_=false;
      retained_.clear();
    }
  }
 private:
  bool enabled_=false;
  bool pending_=false;
  std::vector<Tensor> retained_;
};
}
