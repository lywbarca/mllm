# Qwen3-0.6B W8A8 / FP16 KV trial on 310P

The quantized artifact is separate from the FP16 model:
 /home/ubuntu/models/Qwen3-0.6B-W8A8/model-w8a8.mllm
It links to the generated file under /home/ubuntu/mindie-workspace/quant-w8a8.

## Quantization scope

- 196 decoder Linear weights: symmetric INT8, per output-channel FP32 scale.
- Inputs to these Linear operations: static symmetric INT8, per tensor scale.
- Embedding, tied output head, norm/RoPE/attention computations and KV storage: FP16.
- Parameter names match the existing AscendLinearOp loader contract:
  .weight, .weight.scale, .weight.scale_x.
- Zero activation offset means the existing zero INT32 bias is appropriate.

The reproducible calibration script and exact calibration token IDs are under
/home/ubuntu/w8a8-results. Calibration uses 32 bilingual samples (3389 tokens):
16 entries from the installed boolq calibration file and 16 additional prompts.
It does not use benchmark prompts or validation questions.

SmoothQuant alpha=0.5 rescales:
1. RMSNorm output and shared Q/K/V or gate/up weight inputs;
2. shared V features and corresponding GQA output projection channels;
3. the linear up branch of SwiGLU and down projection inputs.

The FP32 pre/post transformation check observed maximum last-logit difference
2.12e-5 on one calibration sample. This checks the transformation wiring; it
does not establish quantized model quality. Subsequent minmax calibration
collects new activation scales. Quantized weights are written in ModelFileV2
format using its documented descriptor layout. Original FP16 weights are intact.

## Runtime

The static W8A8 decoder graph uses MULS, Round, Clamp, Cast and ATB INT8 Linear.
FP16 output is passed to the existing norm/attention/KV operations.
This trial adds INT8 NZ weight caching, so the same large weight conversion need
not be repeated for every token. It preserves canonical INT8 weights and scales.
Graph input mappings and output channel inference handle physical NZ shapes.

NZ layout is fixed when an ATB graph is created. The test constructs separate
ND/NZ graphs because changing the weight format on one live ATB Linear operation
causes a vendor graph Setup error in this version. Sequence-length changes
within a fixed layout are tested (1,17,256,512,1).

The FP16 QKV/gate-up fusion is not enabled in the current W8A8 branch. Therefore
FP16 and W8A8 timings compare their current implementations, not equivalent
fully optimized kernels. Static W8A8 is selected by decoder graph execution;
the older eager INT8 implementation uses its separate dynamic-quantization path.

## Running

Source /home/ubuntu/Ascend-mllm-8.2.RC1/set_env.sh and set:
 MLLM_ASCEND_QWEN_DECODER_GRAPH=1
 MLLM_ASCEND_BENCH_MODEL_FILE=model-w8a8.mllm

Then run:
 /home/ubuntu/mllm/build-x86-ascend/bin/mllm-qwen-ascend-benchmark /home/ubuntu/models/Qwen3-0.6B-W8A8 paged 256 1

For the FP16 reference, use its original directory and model-fp16.mllm.
The benchmark permits variable short-prompt output length for quantization,
while the 256/512 tests still enforce 64 outputs and 63 decode steps.
One full prefill is warmed up; first decode shape setup remains in decode time,
matching the previous FP16 benchmark method.

## Validation and limitations

The operator regression compares both ND and cached NZ weights against a CPU
INT8 reference, including rounding ties, saturation, shape growth and reuse.
FP16 NZ and fusion/selection regressions are also run.

Six held-out prompts and a 64-step same-history comparison are available in
validation.log. The latter has 53/64 matching top-1 IDs and mean KL 0.217894.
This is top-1 agreement with the FP16 model, not task accuracy. Arithmetic,
capital-city and translation examples remain usable, but explanations differ
and factual mistakes occur. Small-model FP16 answers also have imperfections.
The programming example is capped at 64 tokens and may be incomplete.

This limited calibration and spot check is not a production-quality accuracy
evaluation. The trial must not be described as lossless or assumed faster merely
because it uses INT8. See /home/ubuntu/w8a8-results/RESULTS.md for measurements.
