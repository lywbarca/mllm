#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen_ascend/modeling_qwen_ascend.hpp>

MLLM_MAIN({
  using namespace mllm;
  using namespace mllm::models::qwen_ascend;
  if (argc != 3) {
    std::cerr << "usage: validate-radix MODEL_DIR PROMPT_LOG\n";
    return 2;
  }
  Logger::level() = LogLevel::kWarn;
  initAscendBackend();
  int failures = 0;
  {
    std::vector<int64_t> prompt;
    std::ifstream file(argv[2]);
    std::string line;
    while (std::getline(file, line)) {
      if (line.starts_with("INPUT_IDS ")) {
        std::istringstream stream(line.substr(10));
        int64_t token;
        while (stream >> token) prompt.push_back(token);
        break;
      }
    }
    if (prompt.size() < 256 || prompt.size() > 512) throw std::runtime_error("Need a 256-512 token prompt");
    const std::string root = argv[1];
    QwenAscendConfig cfg(root + "/config-bench-paged.json");
    cfg.paged_attention_enabled = true;
    cfg.paged_radix_cache = false;
    cfg.paged_num_blocks = 8;
    QwenAscendForCausalLM reference(cfg);
    reference.to(kAscend);
    auto weights = mllm::load(root + "/model-fp16.mllm", ModelFileVersion::kV2);
    reference.load(weights);
    cfg.paged_radix_cache = true;
    QwenAscendForCausalLM cached(cfg);
    cached.to(kAscend);
    cached.load(weights);
    auto* pool = cached.kvCache().pagedCache();
    auto check = [&](bool ok, const std::string& message) {
      if (!ok) { ++failures; std::cerr << "FAIL " << message << std::endl; }
    };
    auto tensor = [](const std::vector<int64_t>& ids) {
      return Tensor::fromVector(ids, {1, static_cast<int32_t>(ids.size())}, kInt64, kCPU);
    };
    auto top = [&](const Tensor& output) {
      const auto* data = output.ptr<float>();
      return static_cast<int64_t>(std::max_element(data, data + cfg.vocab_size) - data);
    };
    int comparisons = 0, top_mismatches = 0;
    auto compare = [&](const std::string& name, const Tensor& expected, const Tensor& actual) {
      double err2 = 0, norm_a = 0, norm_b = 0, dot = 0, maximum = 0;
      bool finite = true;
      for (int i = 0; i < cfg.vocab_size; ++i) {
        const double a = expected.ptr<float>()[i], b = actual.ptr<float>()[i];
        finite = finite && std::isfinite(a) && std::isfinite(b);
        maximum = std::max(maximum, std::abs(a - b));
        err2 += (a-b)*(a-b); norm_a += a*a; norm_b += b*b; dot += a*b;
      }
      const double rms = std::sqrt(err2/cfg.vocab_size);
      const double rel = std::sqrt(err2/std::max(norm_a, 1e-30));
      const double cosine = dot/std::sqrt(std::max(norm_a*norm_b, 1e-30));
      const bool same_top = top(expected) == top(actual);
      ++comparisons; top_mismatches += !same_top;
      // Declared before measurement: allow FP16 prefill/decode kernel rounding,
      // while checking every vocabulary logit, not only the selected token.
      check(finite && maximum <= 0.15 && rms <= 0.03 && rel <= 0.01 && cosine >= 0.9999, name + " logits");
      std::cout << "LOGITS case=" << name << " max_abs=" << maximum << " rms=" << rms
                << " relative_l2=" << rel << " cosine=" << cosine << " same_top=" << same_top << std::endl;
    };
    auto run = [&](const std::string& name, const std::vector<int64_t>& ids, int reused) {
      auto expected = reference.forward(tensor(ids));
      auto actual = cached.forward(tensor(ids));
      check(cached.lastPagedReusedTokens() == reused, name + " reuse length");
      check(pool->blocks().sequence(0).tokens == ids, name + " token history");
      compare(name, expected, actual);
      std::cout << "CACHE case=" << name << " reused=" << cached.lastPagedReusedTokens()
                << " blocks=" << pool->blocks().cachedBlocks() << " free=" << pool->blocks().freeBlocks() << std::endl;
      return std::make_pair(expected, actual);
    };
    run("cold", prompt, 0);
    const auto cold_blocks = pool->blocks().sequence(0).blocks;
    run("exact", prompt, static_cast<int>(prompt.size())-1);
    check(pool->blocks().sequence(0).blocks.front() == cold_blocks.front(), "full block physically shared");
    // Branch on both sides of a block boundary, then replay the original branch.
    for (int split : {127, 129}) {
      auto branch = prompt;
      branch[split] = (branch[split] + 100) % cfg.vocab_size;
      cached.clearCache(); cached.clearPagedPrefixCache();
      run("seed-" + std::to_string(split), prompt, 0);
      run("branch-" + std::to_string(split), branch, split);
      run("replay-" + std::to_string(split), prompt, static_cast<int>(prompt.size())-1);
    }
    cached.clearCache(); cached.clearPagedPrefixCache();
    run("short-seed", prompt, 0);
    std::vector<int64_t> shorter(prompt.begin(), prompt.begin()+130);
    run("short-key", shorter, 129);
    auto pair = run("original-after-short", prompt, static_cast<int>(prompt.size())-1);
    // Teacher forcing isolates cache correctness from autoregressive divergence.
    for (int step = 0; step < 8; ++step) {
      auto input = tensor({top(pair.first)});
      pair.first = reference.forward({{"sequence", input}}, {}).at("sequence");
      pair.second = cached.forward({{"sequence", input}}, {}).at("sequence");
      compare("decode-" + std::to_string(step), pair.first, pair.second);
    }
    // Keep an active request pinned while other branches exceed pool capacity.
    cached.forkPagedSequence(0, 99);
    const auto protected_blocks = pool->blocks().sequence(99).blocks;
    auto protected_tokens = pool->blocks().sequence(99).tokens;
    cached.releasePagedSequence(0);
    for (int i = 0; i < 10; ++i) {
      cached.createPagedSequence(0); cached.selectPagedSequences({0});
      auto unrelated = shorter;
      unrelated[0] = 1000+i;
      auto expected = reference.forward(tensor(unrelated));
      auto actual = cached.forward({{"sequence", tensor(unrelated)}}, {}).at("sequence");
      check(cached.lastPagedReusedTokens() == 0, "unrelated prefix misses");
      compare("pressure-" + std::to_string(i), expected, actual);
      check(pool->blocks().sequence(99).blocks == protected_blocks, "active blocks protected");
      cached.releasePagedSequence(0);
    }
    cached.selectPagedSequences({99});
    protected_tokens.push_back(7);
    auto expected = reference.forward(tensor(protected_tokens));
    auto actual = cached.forward({{"sequence", tensor({7})}}, {}).at("sequence");
    compare("protected-after-eviction", expected, actual);
    cached.clearCache(); cached.clearPagedPrefixCache();
    check(pool->blocks().freeBlocks() == 8, "all storage reclaimed");
    // Same model, same input, hot kernels; cold means explicitly empty radix.
    for (int round = 0; round < 3; ++round) {
      cached.clearCache(); cached.clearPagedPrefixCache();
      for (const auto& mode : {"cold", "hot"}) {
        cached.clearCache();
        const auto start = std::chrono::steady_clock::now();
        auto logits = cached.forward({{"sequence", tensor(prompt)}}, {}).at("sequence");
        const auto stop = std::chrono::steady_clock::now();
        check(cached.lastPagedReusedTokens() == (std::string(mode)=="cold" ? 0 : static_cast<int>(prompt.size())-1), "timing reuse");
        std::cout << "TIMING mode=" << mode << " round=" << round << " input=" << prompt.size()
                  << " reused=" << cached.lastPagedReusedTokens()
                  << " first_logits_us=" << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()
                  << " top=" << top(logits) << std::endl;
      }
    }
    cached.clearCache(); cached.clearPagedPrefixCache();
    check(pool->blocks().freeBlocks() == 8, "final reclamation");
    std::cout << "SUMMARY comparisons=" << comparisons << " top_mismatches=" << top_mismatches
              << " failures=" << failures << " pool_bytes=" << pool->poolBytes() << std::endl;
  }
  shutdownContext();
  return failures ? 1 : 0;
})
