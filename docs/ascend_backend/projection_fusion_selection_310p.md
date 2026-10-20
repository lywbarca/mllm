# 310P FP16 projection fusion and device greedy selection

This extends the NZ-weight / dedicated-prefill / asynchronous-decoder path.

## Projection and activation traffic

On aligned FP16 310P decoder graphs without attention bias, concatenate Q/K/V
weights along output channels and cache their NZ representation. Do the same
for gate/up. Canonical ND parameters remain available for serialization and
other execution paths. The packed cache retains source tensors and rebuilds
when source storage changes on reload. Packing occurs on the first graph forward
and is excluded from warmed inference timings.

QKV is one Linear followed by an unequal Split (Q, K, V channel sizes). Gate/up
is one Linear followed by SwiGLU. For Qwen3-0.6B this reduces decoder Linear
calls from seven to four per layer, or 197 to 113 per token including the output
head. It also eliminates repeated conversions of the same activation into NZ
and reduces result conversions associated with the removed Linear operations.
SwiGLU avoids materializing separate split/SiLU/multiply outputs.

This is **not** an all-NZ decoder. ATB still receives ND activations at its public
Linear, norm, RoPE, and attention boundaries. In particular, SelfAttention's
310P input contract requires ND Q/K/V. The conversion savings come from
sharing projection work, not from relabeling ND buffers as NZ or retaining
unsupported NZ tensors across attention/norm.

Extra packed weights consume approximately 560 MiB for this 28-layer model,
in addition to existing canonical ND and individual NZ weight caches.
INT8, biased attention, non-310P, and unaligned projections keep the old graph.

## Greedy generation

The generation API now has backend hooks for generation forward, greedy
selection, and the next token's input device. Default hooks keep the CPU
selection behavior. The Qwen Ascend override retains FP16 logits on device only
for batch-one greedy generation on 310P; normal forward and stochastic sampling
still return CPU float32 logits.

aclnnArgMax selects the token on the NPU and copies one int64 result (8 bytes)
to the host instead of the 151936-element FP16 logits (303872 bytes). Workspace
and result storage are reused. EOS suppression honors the primary override and
additional EOS IDs with the existing min_new_tokens step boundary. Tests cover
first-index tie selection, the vocabulary tail, multiple EOS IDs, and invalid
EOS IDs. Next-token IDs stay on CPU until normal model input preparation,
avoiding an unnecessary token round trip.

Selection assumes finite model FP16 logits; equivalence for NaN or all-infinite
logits is not claimed. As with the existing CPU sampler, suppression modifies
the current step's logits.

The random-sampling regression test also exposed existing explicit deletion of
aliased logits handles in temperature/top-k/top-p sampling. Those paths now let
Tensor ownership release the storage.

## Switches

All are enabled by default where eligible. Set before building/loading a model:

- MLLM_ASCEND_PACKED_PROJECTIONS=0: separate Q/K/V and gate/up Linear ops.
- MLLM_ASCEND_FUSED_MLP_ACTIVATION=0: packed gate/up followed by Split, SiLU, Mul.
- MLLM_ASCEND_DEVICE_GREEDY=0: CPU logits transfer and greedy selection.

Existing NZ-weight, prefill encoder, asynchronous decoder, graph constant, and
output resource switches remain available. Do not change graph-layout switches
during a live model's execution.

## Verification

AscendFusionSelectionTest checks packed channel ordering against a CPU
reference, shapes 1/17/256, packed cache reuse/reload, large-vocabulary greedy
selection, and SwiGLU numerical accuracy. ARGenerationPerformanceTest covers
greedy/sampling dispatch through chat, streamGenerate, and generate, alongside
existing min_new_tokens and timing tests.

mllm-qwen-ascend-validate-fusion MODEL_DIR BASELINE_LOG compares 64 model steps
using identical input histories. Direct forward remains the CPU FP32 logits
inspection API. FP16 fusion may change rounding and therefore greedy output
when competing logits are nearly tied; exact token equality is not a universal
acceptance criterion.

Benchmarks, profile exports, ablations, and incremental patches for this change
are under /home/ubuntu/fusion-select-results on 310p.


## Attention scale initialization

A supplementary contiguous-graph smoke test exposed the pre-existing use of CPU
FP16 elementwise multiplication to create the attention scale, unsupported on
x86. GraphRunner now initializes an FP32 scalar, casts it to the requested dtype,
and uploads it once. The paged attention plugin computes its own scale and does
not consume this legacy scalar, so the paged benchmark comparison is unaffected.
A regression test checks the FP16 value and cached storage identity.
