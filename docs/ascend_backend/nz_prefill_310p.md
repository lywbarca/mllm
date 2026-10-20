# 310P FP16 weight packing and paged prefill

The 310P backend prepares persistent NZ copies of aligned FP16 Linear weights during load(). Decoder graphs pass explicit NZ descriptors for these execution artifacts; the eager Linear path uses the same cache. Unaligned weights and other devices retain their previous path. INT8 W8A8 execution is unchanged.

Canonical ND parameters remain available to getParams() and device migration. This intentionally uses additional memory rather than changing generic Tensor layout/serialization semantics. Reloading a Linear invalidates/rebuilds its packed weight. Parameters must not be mutated in place during inference.

Qwen's tied embedding retains its ND table for ACLNN embedding lookup. The model prepares a separate NZ output-projection cache after loading. The forward fallback prepares it if the model was loaded through a base-class interface or the embedding storage changed.

For a single request with zero prior KV and more than one input token, the NZ paged decoder graph uses ATB SelfAttentionParam::PA_ENCODER, a causal mask and actual sequence length. ReshapeAndCache still writes the real paged KV pools. Decode, batched requests, prefix reuse and multi-token appends with history retain the paged decoder attention path. Mask and sequence metadata are cached per layer and rebuilt when the prefill length changes. The current mask representation is quadratic in prompt length.

Debug/baseline switches (set before loading/running):

- MLLM_ASCEND_NZ_WEIGHTS=0: disable weight prepacking.
- MLLM_ASCEND_PREFILL_ENCODER=0: retain the previous tiled paged attention prefill.
- MLLM_ASCEND_QWEN_DECODER_GRAPH=0: existing eager path. The specialized prefill is implemented in the paged decoder graph.

Validation:

- AscendNZWeightTest: CPU-reference Linear values, repeated cache reuse, reloading changed weights, canonical parameter preservation, unaligned fallback.
- AscendRealPagedAttentionTest: CPU-reference causal attention, 127/129/256/512-token prefill, subsequent decode, cross-block writes, forks, rollback, batch reordering.
- AscendPagedKVCacheTest: full-model logits versus independent recomputation, prefix reuse and cache reclamation.
- mllm-qwen-ascend-validate-nz-prefill MODEL_DIR BASELINE_LOG: teacher-forced comparison across 64 steps using a benchmark log's INPUT_IDS and OUTPUT_IDS. Reports finite-logit checks, maximum absolute error, RMS error and argmax changes.

The optimized prefill and old paged decoder attention use different FP16 arithmetic paths. Bit-identical logits or generated text are not guaranteed. On the 512-token Qwen3-0.6B prompt, the first token disagreement was at output index 12: the baseline's competing logits tied, while the new path separated them by 0.015625. Across 64 teacher-forced steps, maximum absolute error was 0.0859375 and maximum RMS error was 0.0144685; only that step changed argmax.
