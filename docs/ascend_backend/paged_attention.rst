ATB Paged Attention
===================

``AscendPagedAttnOp`` implements ``OpTypes::kPagedAttn`` using
``atb::infer::PagedAttentionParam``. It follows the existing Ascend Softmax/SiLU
operation lifecycle and is registered in ``AscendBackend``. The existing
``ops/*.cpp`` CMake glob includes it after CMake is configured again.

Input contract
--------------

This implementation supports FP16 MHA/GQA decode and causal prefill, including
chunks appended to existing or forked histories. Inputs are contiguous:

* Query: ``[B, S, Hq, D]`` (``S=1`` or ``B=1``), FP16, Ascend.
* Key and value cache: FP16, Ascend, identical shapes. ND storage is
  ``[num_blocks, block_size, Hkv, D]``. NZ storage for 310P is
  ``[num_blocks, Hkv * D / 16, block_size, 16]``.
* Block table: ``[B, max_blocks]``, int32, Ascend. Each row maps logical blocks
  to physical block IDs. All entries, including padding entries, must be valid
  IDs in ``[0, num_blocks)``. The caller owns this invariant; the operator does
  not copy the device table back to the host on each call.
* Context lengths: ``[B]``, int32, CPU. Each length includes all ``S`` new tokens,
  whose K/V must already be written to the cache. Lengths must be at least ``S`` and
  fit in the corresponding block-table row.

The result is ``{output, nil}``, where output has the query's shape and FP16
dtype. The nil slot preserves the existing ``nn::PagedAttn`` interface.
``head_repeat_times`` is ``Hq / Hkv``; the attention scale is ``1 / sqrt(D)``.

``block_size`` must be a multiple of 16 and ``D <= 256``. For NZ, ``D`` must
also be a multiple of 16, ``B <= 2048``, and ``block_size * D <= 16384``.
Use the layout supported by the installed ATB runtime and device. Selecting
NZ only sets the ATB descriptor: it does **not** convert an ND buffer to NZ.
For NZ, element ``[block, token, head, d]`` is physically stored at
``[block, (head*D+d)/16, token, (head*D+d)%16]``.

Usage
-----

After initializing mllm and its Ascend backend, and preparing the five inputs:

.. code-block:: cpp

   #include "mllm/nn/layers/PagedAttn.hpp"

   mllm::aops::PagedAttnOpOptions options;
   options.head_repeat_times = 4;  // e.g. 8 query heads and 2 KV heads
   options.impl_type = mllm::aops::PagedAttnImplType::kDefault;
   options.kv_cache_layout = mllm::aops::PagedAttnCacheLayout::kNZ;  // 310P
   mllm::nn::PagedAttn attention(options);
   attention.to(mllm::kAscend);
   auto [output, unused_weights] = attention(q, key_cache, value_cache,
                                            block_table, context_lengths_cpu);

The CPU default ``kAllFp32`` is explicitly rejected. Attention weights, fused
RoPE, ``high_precision_exp``, and ``prefix_cache_ctx`` are not supported.
RoPE is applied before this operation. For multi-token inputs, query ``i``
attends only to KV positions ``0 .. context_length-S+i`` (inclusive).

Paged KV infrastructure and Qwen integration
-------------------------------------------

``PagedBlockManager`` owns CPU sequence metadata, a free-block list and physical
block reference counts. ``AscendPagedKVCache`` owns one K/V tensor pair per layer
and shares the same physical block IDs across layers. ATB ``ReshapeAndCache``
writes post-RoPE keys and values using an int32 slot mapping. It writes the actual
ND/NZ pool consumed by attention; no separate archive pool is allocated.

Configure Qwen before constructing the model. Add this section to the existing
model JSON (all ordinary model fields are still required):

.. code-block:: json

   "paged_attention": {
     "enabled": true,
     "block_size": 128,
     "num_blocks": 64,
     "radix_cache": true,
     "layout": "NZ"
   }

The equivalent C++ fields are ``paged_attention_enabled``, ``paged_block_size``,
``paged_num_blocks``, ``paged_radix_cache`` and ``paged_layout`` in ``QwenAscendConfig``. Paging is off
by default. ``num_blocks=0`` selects ``ceil(max_cache_length/block_size)`` blocks
for one maximum-length request, plus one COW spare when radix caching is enabled;
explicitly increase the capacity for concurrent
requests. The pool footprint is
``2 * layer_nums * num_blocks * block_size * kv_heads * head_dim * sizeof(fp16)``.
This excludes model weights, workspace and transient prefill tensors. Each layer
tensor must fit the existing Ascend allocator's uint32 byte limit; oversized
configurations are rejected before allocation. ``poolBytes()`` reports the KV
pool's byte count and ``blocks().freeBlocks()`` reports logical free capacity.

When paging is enabled, Qwen does not allocate the old permanent contiguous KV
buffers. Each forward reserves the complete batch, constructs block tables and
slot mappings once, and writes each layer exactly once. Decode reads the pools
directly via ``nn::PagedAttn``. Single-request prefill (including chunked prefill
and suffixes of forked requests) now reads the same pools directly as well.
The model no longer calls ``gatherLayer``; that helper remains for storage tests.
Paged decoder graph
~~~~~~~~~~~~~~~~~~~

With MLLM_ASCEND_QWEN_DECODER_GRAPH=1 (the default), eligible Qwen decoder
layers execute as ATB graphs, including paged prefill and batched decode.
Set the variable to 0 to retain the eager reference path.

The graph includes normalization, projections, RoPE, paged KV writing,
attention, residuals and MLP. AscendPagedAttentionPlugin submits ATB
ReshapeAndCache followed by PagedAttention on the same context. It does
not call mllm eager operators, allocate device workspace, or synchronize between
its operations. The enclosing executor manages workspace and synchronizes once
per decoder layer.

prepareGraphMetadata() expands each request's block-table row and causal
lengths once per step and shares them across layers. Storage is
B*S*max_blocks*sizeof(int32) for the table, plus host/device lengths.
Attention submissions remain tiled at 128 queries; each tile owns an ATB
operation so Setup state is not overwritten before Execute. Setup runs each
forward to refresh shapes and addresses.

Block reservation and COW remain outside the graph. A layer is marked written
only after its graph completes; the model commits after all layers and logits
succeed. Abort hides private-tail writes and releases COW reservations.
Metadata stays alive through graph synchronization.

model.graphExecutionCount() exposes completed decoder-layer graph calls.
MLLM_PROFILE_ASCEND_GRAPH=1 also logs the graph names and timings.
The 310P validation covers FP16 NZ, prefill above 128 queries, decode, batch
reordering, shared tails, abort/retry, radix reuse and complete Qwen3-0.6B
generation. W8A8 graph wiring shares the paged node but has not been validated
with quantized weights; ND hardware has not been tested in this environment.

Causal prefill without KV gather
-------------------------------

The basic ATB operation accepts a block-table row and a visible KV length for
each query. The wrapper interprets the ``S`` queries as logical one-token
requests sharing the physical blocks. With history length ``H``, their lengths
are ``H+1, H+2, ..., H+S``. This is mathematically the same causal attention:
future K/V may already be written, but they are outside each query's visible
length. No K/V content is copied or expanded for attention.

In the eager operator, queries are submitted in tiles of at most 128. The block-table row is replicated
on device once per call (using non-overlapping doubling copies); the same table
is reused for subsequent tiles. Query and output descriptors point directly into
the original tensors. Only table metadata and per-query lengths are expanded.
Temporary table memory is ``min(S,128) * max_blocks * sizeof(int32)``. NZ lengths
have both host and device storage, alive through execution and synchronization.

This uses the basic unquantized, unmasked ATB API, not ``CALC_TYPE_SPEC``.
The 8.2.RC1 documentation also lists parallel decoding on Atlas inference
products with ``qSeqLens`` and ``MASK_TYPE_SPEC`` (or ``UNDEFINED``). That is
another possible implementation; it requires its own mask layout and parameter
plumbing. The current per-query-length implementation avoids a dense causal
mask and works with the existing decode contract. It is not a specialized
prefill throughput optimization: repeated KV reads, per-layer metadata copies,
and Setup/Execute/synchronization per tile may be expensive. Benchmark before
making performance claims or choosing a native multi-query implementation.

Request lifecycle and sharing
-----------------------------

Sequence ID 0 exists and is selected initially, so the existing single-request
generation interface works. For manual batching, use the dictionary ``forward``
overload and explicit sequence selection:

.. code-block:: cpp

   // model was constructed with paging enabled, moved to Ascend and loaded.
   model.selectPagedSequences({0});
   auto prefill = model.forward({{"sequence", prompt_cpu}}, {}); // prompt_cpu: [1,S], int64
   model.forkPagedSequence(0, 1);  // Share the already computed prompt's KV.
   model.selectPagedSequences({1, 0});
   auto tokens = mllm::Tensor::fromVector(std::vector<int64_t>{42, 43},
                                         {2, 1}, mllm::kInt64);
   auto result = model.forward({{"sequence", tokens}}, {});
   model.releasePagedSequence(1);
   model.releasePagedSequence(0);

``createPagedSequence(id)`` creates an independent empty request. Prefill requests
one at a time; batched decode accepts one token per selected sequence and supports
different historical lengths. Selection order defines input/output batch order.
``forkPagedSequence`` shares all committed blocks; appending to a shared partial
tail allocates and copies that block in **every layer** before writing. Full shared
blocks stay immutable. Releasing a request recycles only blocks with zero references.

The pool is fixed-size and never silently evicts live requests. A failed batch
reservation leaves active sequences unchanged; idle radix leaves may have been
evicted to try to satisfy the reservation. The model throws a capacity error.
Release completed requests or reduce the selected batch, then retry. A step commits
only after all layers have written and the model has produced logits. A guard
rolls back reserved blocks when a C++ exception interrupts forward. ACL/ATB fatal
errors retain the backend's existing process-failure behavior, not retry semantics.

RoPE positions come from each selected request's committed length. Omit
``position_ids`` in manual batching. When chaining ARGeneration output, decode
accepts the previous step's positions and checks that the last position matches
``cached_length - 1``. Custom nonzero position origins are not supported. Request
reordering must also reorder any supplied previous-position metadata.

``clearCache()`` discards all requests and recreates empty sequence 0. With radix
caching enabled it preserves cached prefixes; otherwise all blocks are recycled. The convenience ``forward(Tensor)`` calls ``clearCache()`` first;
do not use that overload for incremental/multi-request serving. ARGeneration's
built-in sampler remains single-request; callers of batched ``forward`` consume
and sample each row of logits themselves. Access to a model/cache is serialized.

This infrastructure supplies request lifecycle, shared storage and optional token-key
radix/LRU caching. It does not include an admission/preemption scheduler. The earlier
archive ``enablePrefixCache`` mode is explicitly rejected when paging is enabled.

Radix prefix reuse on the physical pool
--------------------------------------

Set ``paged_attention.radix_cache`` to true (default false) before constructing the
model. ``enabled`` must also be true. The block manager keeps a token-level radix
tree with compressed edges, splitting an edge at divergence or when inserting a
shorter prefix. Edges are also segmented at physical block boundaries. Each node
records a physical block, its starting token offset within that block, and its
edge tokens. Lookup may stop inside an edge; it never skips mismatched tokens to
visit descendants. Duplicate publication does not add references. This dedicated
tree does not use the old archive RadixTree or its per-token pointer representation.

After all layers finish and the step commits, the entire committed token history
is eligible for caching, including a partial tail. For example, with block size 4,
cached ``abce`` matches 3 tokens of ``abcd``. After ``abcd`` commits, the tree has
an ``abc`` prefix and ``d``/``e`` branches. Tree splitting changes metadata only.
Reuse attaches the original physical blocks; writing ``d`` copies the shared tail
block in every layer before modifying it, preserving cached ``abce``. Complete
prefix blocks remain shared. Exact prompt hits reuse all but the final token so
logits can still be computed.

Split nodes and their children may reference different COW versions of the same
logical block. Lookup selects the deepest matched node's physical copy for that
block: it contains all earlier tokens in the block prefix. Node boundaries are
not sampled to reconstruct block tables. New COW copies can also back same-block
ancestor prefixes, allowing obsolete copies to be reclaimed; siblings and live
requests retain their own references. Unbranched same-block edges are merged after
incremental publication to avoid a node/version per appended token.

Only a shared partial tail requires a KV copy. Full prefix blocks are attached
without copying, and attention still reads the physical pools without gather or
a permanent contiguous working buffer. COW requires free capacity; the automatic
pool size includes one spare block for a single request with radix enabled. Explicit
pool sizing must account for active requests, retained branches and COW capacity.

Automatic lookup requires one empty request, CPU contiguous int64 token IDs, and
no caller-provided ``position_ids``. Explicit-position inputs skip lookup but must
satisfy the model's normal zero-origin position checks. All radix-enabled forward
calls require CPU contiguous int64 tokens so committed token identity cannot drift
from the written KV. Batched single-token decode tracks each selected request's
history in input order. Fork copies logical token history and shares block IDs.
``lastPagedReusedTokens()`` reports the last successful forward's reused token count.

Releasing/resetting requests keeps cache references alive. On allocation pressure,
the least recently used leaf whose physical block has no active request reference
is evicted; this
continues until enough blocks are available or all remaining leaves are protected.
Internal nodes become eligible after their children are removed. Several nodes
may retain the same physical block, so evicting one node does not necessarily free
a block. Total references and cache-node references are counted separately. Active sequences
are never evicted. Eviction scans the tree, so this is a simple host LRU policy,
not a throughput-optimized scheduler. Retained physical KV is bounded by pool capacity; split-node metadata can
include multiple token intervals per physical block. ``cachedBlocks()`` counts
unique cached physical blocks and ``cachedNodes()`` counts tree nodes. Token-history metadata additionally scales with the active requests.

The matched prefix attachment is guarded: preparation failure, pool exhaustion or
a C++ exception aborts pending writes and restores the initially empty request.
Cache publication only happens after commit. Host allocation failure during
publication leaves a valid partial cache; it does not turn a committed forward into
a reported failure. ACL/ATB fatal errors retain their existing process-exit behavior.

``clearCache()`` resets requests while retaining reusable radix prefixes, including
when called by ``forward(Tensor)``. ``clearPagedPrefixCache()`` explicitly drops all
tree references and preserves active request KV. To discard everything, call both.
A cache is private to a model instance and assumes fixed weights, adapters, RoPE
configuration and tokenizer interpretation. Clear both requests and prefixes before
changing any of these; token equality alone cannot validate KV from different models.


The CPU lengths are bound through ATB ``hostData``. The NZ path additionally
uploads a device copy for the 310P runner. Workspace and the length copy stay
alive until the global ATB stream has synchronized.

Validation status
-----------------

The Ascend code has not been compiled or executed: the current phoenix workspace
has no CANN/ATB installation. API/header compatibility with the eventual CANN
8.2.RC1 + ATB installation and hardware correctness still need verification.

``AscendPagedAttnKernelTest.hpp`` is wired into ``Mllm-Test-AscendKernel``.
It contains device-gated 310P/NZ and 910B/ND tests comparing against independent
FP32 CPU attention. Cases cover MHA/GQA, shared/permuted physical blocks,
partial blocks, block-boundary lengths, and changing lengths on the same layer.
New causal cases check every output query for 17- and 129-token chunks, with
zero/nonzero histories, MHA/GQA and the 128-query tile boundary.
These tests have been authored but **not run**.

``PagedBlockManagerTest.cpp`` is independent of mllm/CANN and checks exact block
boundaries, atomic capacity failure, fork/COW, abort, reclamation and 1,000
deterministic randomized operations against independent per-sequence token histories
stored across two simulated physical layer pools. It is registered as the
``Mllm-Test-Engine-PagedBlockManager`` CMake target and ``PagedBlockManager`` CTest.
It can also be built without configuring mllm or initializing submodules:

.. code-block:: bash

   c++ -std=c++17 -Wall -Wextra -Werror -I. tests/engine/PagedBlockManagerTest.cpp -o /tmp/mllm-paged-block-test
   /tmp/mllm-paged-block-test

On 2026-10-14, the user authorized this standalone CPU test. The command above
compiled with ``-Wall -Wextra -Werror`` and ran successfully on phoenix; all fixed
cases and 1,000 randomized operations passed. This validates host block ownership
and rollback, not the Ascend writer or attention implementation.
The extended standalone CPU test passed on phoenix with GCC7.5 and
``-Wall -Wextra -Werror``. Optional sanitizer verification was unavailable: the
combined ASan/UBSan build failed to link the UBSan runtime, and an ASan-only
binary could not load ``libasan.so.5``. Neither sanitizer check is counted as a pass.

The extended test also covers radix exact/partial/divergent matches, duplicate
publication, LRU order, live-reference protection, capacity failure, flush and
1,000 additional randomized prefix-reuse/fork/COW/abort histories against two
simulated layer pools with prefix-dependent KV values. Another 500 cases compare
the exact token match length against an independent longest-prefix corpus oracle
without eviction (2,500 randomized cases total). Fixed regressions include
``abce``/``abcd``, shorter-key splits, cross-block divergence, aborted suffixes,
branch preservation and repeated appends with one COW spare.

``AscendPagedKVCacheTest`` adds actual ND/NZ writer/gather tests with two layers,
forked partial blocks, rollback and reclamation. Its synthetic two-layer Qwen test
compares prefill, chunked prefill, a multi-token suffix after fork and a reordered batch with unequal
lengths against full-prefix recomputation through the original contiguous model.
It also checks ARGeneration position metadata, automatic radix reuse after
request release/reset, physical block identity, in-block divergence/COW, cache flush and final reclamation. Device
tests select 310P or 910B explicitly and skip other hardware. After configuring and
building on the target installation, run:

.. code-block:: bash

   Mllm-Test-AscendKernel --gtest_filter='AscendPagedAttnKernelTest.*:AscendPagedKVCacheTest.*'

Both the device tests and end-to-end logits comparison remain **unexecuted**.
No accuracy, latency or throughput claim follows from the source checks.

References
----------

* `ATB 8.2.RC1 input/output contract <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendtbapi/ascendtb_01_0206.html>`_
* `ATB 8.2.RC1 constraints <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendtbapi/ascendtb_01_0207.html>`_
* `ATB 8.2.RC1 parallel decoding <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendtbapi/ascendtb_01_0215.html>`_
* `ATB 8.2 ReshapeAndCache inference example <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendtbapi/ascendtb_01_0246.html>`_

These accessible references describe the basic ATB API; newer optional modes
are intentionally not used. They do not replace validation against the target
installation's headers and hardware.
