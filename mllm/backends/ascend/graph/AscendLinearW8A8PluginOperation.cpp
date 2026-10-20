// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/ascend/graph/AscendLinearW8A8PluginOperation.hpp"
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include <atb/utils.h>
#include <cmath>
#include <utility>
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {
namespace {
uint64_t alignUp(uint64_t n) { return (n + 511) / 512 * 512; }
}

AscendLinearW8A8PluginOperation::AscendLinearW8A8PluginOperation(float inv_scale_x, std::string suffix)
    : name_suffix_(std::move(suffix)) {
  // ATB quant-per-channel divides by scale; it accepts a scalar broadcast too.
  const auto sx = static_cast<mllm_fp16_t>(1.f / inv_scale_x);
  if (!std::isfinite(float(sx)) || float(sx) <= 0.f) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "W8A8 native activation scale must be positive finite FP16");
  }
  auto scale_cpu = Tensor::empty({1}, kFloat16, kCPU).alloc();
  scale_cpu.ptr<mllm_fp16_t>()[0] = sx;
  auto offset_cpu = Tensor::empty({1}, kInt8, kCPU).alloc();
  offset_cpu.ptr<int8_t>()[0] = 0;
  scale_ = scale_cpu.to(kAscend);
  offset_ = offset_cpu.to(kAscend);
  atb::infer::ElewiseParam qp;
  qp.elewiseType = atb::infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL;
  MLLM_ATB_CHECK(atb::CreateOperation(qp, &quant_op_));
  atb::infer::LinearParam lp;
  lp.transposeA = false;
  lp.transposeB = true;
  lp.hasBias = true;
  lp.outDataType = ACL_FLOAT16;
  lp.enAccum = false;
  lp.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  lp.quantMode = atb::infer::LinearParam::PER_CHANNEL;
  MLLM_ATB_CHECK(atb::CreateOperation(lp, &linear_op_));
}
AscendLinearW8A8PluginOperation::~AscendLinearW8A8PluginOperation() {
  if (quant_op_) atb::DestroyOperation(quant_op_);
  if (linear_op_) atb::DestroyOperation(linear_op_);
}
std::string AscendLinearW8A8PluginOperation::GetName() const {
  return "AscendLinearW8A8PluginOperation" + name_suffix_;
}
uint32_t AscendLinearW8A8PluginOperation::GetInputNum() const { return 4; }
uint32_t AscendLinearW8A8PluginOperation::GetOutputNum() const { return 1; }
atb::Status AscendLinearW8A8PluginOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& in, atb::SVector<atb::TensorDesc>& out) const {
  if (in.size() != 4 || out.size() != 1) return atb::ERROR_INVALID_TENSOR_NUM;
  if (in[0].shape.dimNum < 2 || in[1].shape.dimNum < 2) return atb::ERROR_INVALID_TENSOR_DIM;
  out[0] = in[0];
  out[0].dtype = ACL_FLOAT16;
  out[0].format = ACL_FORMAT_ND;
  const auto& w = in[1];
  out[0].shape.dims[out[0].shape.dimNum - 1] =
      w.shape.dims[w.format == ACL_FORMAT_FRACTAL_NZ ? w.shape.dimNum - 2 : 0];
  return atb::NO_ERROR;
}
void AscendLinearW8A8PluginOperation::makePacks(
    const atb::VariantPack& vp, uint8_t* xq, atb::VariantPack& qvp, atb::VariantPack& lvp) const {
  atb::Tensor quantized{};
  quantized.desc = vp.inTensors[0].desc;
  quantized.desc.dtype = ACL_INT8;
  quantized.deviceData = xq;
  quantized.dataSize = atb::Utils::GetTensorSize(quantized);
  atb::Tensor scale{}, offset{};
  fillAtbTensor(scale_, scale);
  fillAtbTensor(offset_, offset);
  qvp.inTensors = {vp.inTensors[0], scale, offset};
  qvp.outTensors = {quantized};
  lvp.inTensors = {quantized, vp.inTensors[1], vp.inTensors[2], vp.inTensors[3]};
  lvp.outTensors = vp.outTensors;
}
atb::Status AscendLinearW8A8PluginOperation::Setup(
    const atb::VariantPack& vp, uint64_t& workspace_size, atb::Context* context) {
  if (vp.inTensors.size() != 4 || vp.outTensors.size() != 1) return atb::ERROR_INVALID_TENSOR_NUM;
  atb::VariantPack qvp, lvp;
  makePacks(vp, nullptr, qvp, lvp);
  uint64_t qw = 0, lw = 0;
  auto st = quant_op_->Setup(qvp, qw, context);
  if (st != atb::NO_ERROR) return st;
  st = linear_op_->Setup(lvp, lw, context);
  if (st != atb::NO_ERROR) return st;
  quant_bytes_ = alignUp(qvp.outTensors[0].dataSize);
  quant_ws_bytes_ = alignUp(qw);
  linear_ws_bytes_ = alignUp(lw);
  total_bytes_ = quant_bytes_ + quant_ws_bytes_ + linear_ws_bytes_;
  workspace_size = total_bytes_;
  return atb::NO_ERROR;
}
atb::Status AscendLinearW8A8PluginOperation::Execute(
    const atb::VariantPack& vp, uint8_t* workspace, uint64_t workspace_size, atb::Context* context) {
  if (vp.inTensors.size() != 4 || vp.outTensors.size() != 1) return atb::ERROR_INVALID_TENSOR_NUM;
  if (!workspace || workspace_size < total_bytes_) return atb::ERROR_INTERNAL_ERROR;
  atb::VariantPack qvp, lvp;
  makePacks(vp, workspace, qvp, lvp);
  // Rebind actual graph workspace pointers. Both operations use the enclosing
  // decoder's stream; no per-linear synchronization or runtime allocation.
  uint64_t qw = 0, lw = 0;
  auto st = quant_op_->Setup(qvp, qw, context);
  if (st != atb::NO_ERROR) return st;
  if (qw > quant_ws_bytes_ || alignUp(qvp.outTensors[0].dataSize) != quant_bytes_)
    return atb::ERROR_INTERNAL_ERROR;
  st = linear_op_->Setup(lvp, lw, context);
  if (st != atb::NO_ERROR) return st;
  if (lw > linear_ws_bytes_) return atb::ERROR_INTERNAL_ERROR;
  st = quant_op_->Execute(qvp, workspace + quant_bytes_, quant_ws_bytes_, context);
  if (st != atb::NO_ERROR) return st;
  return linear_op_->Execute(lvp, workspace + quant_bytes_ + quant_ws_bytes_, linear_ws_bytes_, context);
}
atb::Operation* createLinearW8A8PluginGraphOp(float inv_scale_x, std::string suffix) {
  return new AscendLinearW8A8PluginOperation(inv_scale_x, std::move(suffix));
}
}  // namespace mllm::ascend
