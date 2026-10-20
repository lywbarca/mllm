# 310P decoder submission and output projection reuse

The paged Qwen decoder graph can submit consecutive layers to the global ATB stream without synchronizing after each layer. A model-step AscendGraphSubmissionScope retains every graph input and output, including CPU metadata. It synchronizes once after the decoder stack, before final RMSNorm. Its destructor also synchronizes before releasing retained tensors during exception unwinding, before the outer paged KV StepGuard can roll back/reuse blocks.

Each decoder layer has a distinct persistent graph executor and workspace. Executors, input metadata, and workspaces must not be reconfigured, reused for another step, or destroyed until the submission scope finishes. This is a single-stream, single-model-step optimization, not support for concurrent calls on the same model. Ordinary AscendGraphExecutor callers remain synchronous by default. The scope temporarily retains all layer activations, increasing peak activation memory.

The paged graph's unused current-sequence-length placeholder is initialized once and reused. Non-paged graphs still update their actual length each call. Actual paged context lengths and block tables are still updated per model step.

The tied NZ output projection owns one persistent ATB Linear operation and a workspace that only grows when required. Setup still runs per call to refresh shapes and data pointers, including weight reloads. Output tensors are separately allocated, preserving normal output ownership. Execution remains synchronous at this CPU-logit consumption boundary.

Defaults enable these optimizations for the supported paths. For an A/B baseline, set these before running:

- MLLM_ASCEND_ASYNC_DECODER=0: keep per-layer decoder synchronization.
- MLLM_ASCEND_CACHE_GRAPH_CONSTANTS=0: keep uploading the paged length placeholder.
- MLLM_ASCEND_REUSE_OUTPUT_LINEAR=0: recreate the tied output projection operation/workspace.

Keep NZ weights and specialized prefill enabled for comparisons against the preceding implementation.

Tests additionally cover deferred graph chains retaining storage, exception unwinding after submission, and output projection reuse over shape and weight changes against a CPU reference. Existing full-model paged tests cover prefix reuse, chunked appends, forks, batch reordering, cache clearing, and changing prefill lengths.

When graph profiling is enabled, an asynchronous layer's execute/total time measures host submission, not completed device computation. The single stack-level wait is outside those per-layer statistics. Use device profiling or complete model-step timings for latency comparisons.
