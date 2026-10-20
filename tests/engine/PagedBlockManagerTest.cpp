// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Standalone host test: no mllm runtime, CANN, device or submodule dependencies.
#include "mllm/engine/paged_attention/PagedBlockManager.hpp"
#include <array>
#include <iostream>
#include <map>
#include <random>
#include <string>

using mllm::paged_attention::PagedBlockManager;

static void check(bool ok, const char* message) {
  if (!ok) { throw std::runtime_error(message); }
}

template<class F>
static void mustThrow(F fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception&) { threw = true; }
  check(threw, "Expected a rejected operation");
}

static void boundaries() {
  PagedBlockManager manager(2, 4, 12);
  manager.create(1);
  check(manager.beginAppend({1}, {3}), "Initial reservation");
  mustThrow([&] { manager.release(1); });
  manager.commit();
  manager.fork(1, 2);
  check(manager.freeBlocks() == 1, "Fork must share storage");
  const auto original = manager.sequence(1).blocks;
  check(manager.refCount(original[0]) == 2, "Fork reference count");
  // Both shared tails need private copies, but only one block is free.
  check(!manager.beginAppend({1, 2}, {1, 1}), "Atomic batch OOM");
  check(!manager.pending() && manager.freeBlocks() == 1 && manager.sequence(1).blocks == original, "OOM must not change state");
  check(manager.beginAppend({2}, {1}), "Partial-block COW");
  check(manager.copies().size() == 1 && manager.appends()[0].next.blocks != original, "COW mapping");
  manager.abort();
  check(manager.freeBlocks() == 1 && manager.refCount(original[0]) == 2, "Abort releases reservation");
  manager.release(1);
  check(manager.beginAppend({2}, {1}), "Private tail fill");
  check(manager.copies().empty(), "Private tail must not copy");
  manager.commit();
  manager.fork(2, 3);
  check(manager.beginAppend({3}, {1}), "Append after full shared block");
  check(manager.copies().empty(), "Full shared blocks remain immutable and shared");
  manager.commit();
  check(manager.freeBlocks() == 0, "Pool fully used");
  manager.release(2);
  manager.release(3);
  check(manager.freeBlocks() == 2, "All blocks reclaimed");
  manager.create(4);
  mustThrow([&] { manager.create(4); });
  mustThrow([&] { manager.beginAppend({4, 4}, {1, 1}); });
  mustThrow([&] { manager.beginAppend({4}, {13}); });
  mustThrow([&] { manager.beginAppend({4}, {0}); });
  mustThrow([&] { manager.fork(4, 4); });
  check(manager.freeBlocks() == 2, "Rejected operations must not allocate");
  manager.clear();
  check(!manager.contains(4), "Clear removes sequence state");
}

static void randomizedHistories() {
  constexpr int count = 16, size = 4, max_length = 24, layers = 2;
  PagedBlockManager manager(count, size, max_length);
  std::array<std::array<int, count * size>, layers> memory{};
  std::map<int64_t, std::vector<int>> histories;
  int64_t next_id = 0;
  int next_token = 1;
  std::mt19937 rng(142857);
  auto create = [&] {
    manager.create(next_id);
    histories.emplace(next_id++, std::vector<int>{});
  };
  create();
  auto verify = [&] {
    std::vector<size_t> refs(count, 0);
    for (const auto& [id, history] : histories) {
      const auto& seq = manager.sequence(id);
      check(seq.length == static_cast<int32_t>(history.size()), "Logical lengths disagree");
      for (auto p : seq.blocks) { ++refs.at(p); }
      for (size_t s = 0; s < history.size(); ++s) {
        const int slot = seq.blocks.at(s / size) * size + s % size;
        for (int layer = 0; layer < layers; ++layer) {
          check(memory[layer][slot] == history[s] * layers + layer, "Committed KV was corrupted");
        }
      }
    }
    size_t free = 0;
    for (int p = 0; p < count; ++p) {
      check(manager.refCount(p) == refs[p], "Reference count disagrees with live ownership");
      if (refs[p] == 0) { ++free; }
    }
    check(free == manager.freeBlocks(), "Leaked or doubly freed physical blocks");
  };
  for (int iteration = 0; iteration < 1000; ++iteration) {
    if (histories.empty()) { create(); }
    auto selected = histories.begin();
    std::advance(selected, rng() % histories.size());
    const auto id = selected->first;
    const auto action = rng() % 5;
    if (action == 0 && histories.size() < 8) {
      manager.fork(id, next_id);
      histories[next_id++] = selected->second;
    } else if (action == 1) {
      manager.release(id);
      histories.erase(id);
    } else if (action == 2 && histories.size() < 8) {
      create();
    } else {
      std::vector<int64_t> ids;
      std::vector<int32_t> counts;
      for (auto it = selected; it != histories.end() && ids.size() < 2; ++it) {
        const int remaining = max_length - static_cast<int>(it->second.size());
        if (remaining) {
          ids.push_back(it->first);
          counts.push_back(1 + rng() % std::min(remaining, 6));
        }
      }
      if (!ids.empty() && manager.beginAppend(ids, counts)) {
        auto expected = histories;
        for (auto [src, dst] : manager.copies()) {
          for (auto& layer : memory) { std::copy_n(layer.begin() + src * size, size, layer.begin() + dst * size); }
        }
        for (const auto& entry : manager.appends()) {
          for (int s = entry.old_length; s < entry.next.length; ++s) {
            const int token = next_token++;
            expected[entry.id].push_back(token);
            const int slot = entry.next.blocks.at(s / size) * size + s % size;
            for (int layer = 0; layer < layers; ++layer) { memory[layer][slot] = token * layers + layer; }
          }
        }
        if (rng() % 4 == 0) {
          manager.abort();
        } else {
          manager.commit();
          histories = std::move(expected);
        }
      }
    }
    verify();
  }
  manager.clear();
  check(manager.freeBlocks() == count, "Final reclamation");
}

static void radixBoundaries() {
  PagedBlockManager m(8, 2, 32);
  m.enableRadixCache();
  auto publish = [&](int id, const std::vector<int64_t>& tokens) {
    m.create(id);
    check(m.beginAppend({id}, {static_cast<int32_t>(tokens.size())}, {tokens}), "Radix publication reserve");
    m.commit();
  };
  publish(1, {1, 2, 3, 4, 5, 6, 7, 8});
  const auto original = m.sequence(1).blocks;
  check(m.cachedBlocks() == 4 && m.refCount(original[0]) == 2, "Tree owns physical block references");
  m.create(2);
  check(m.reusePrefix(2, {1, 2, 3, 4, 5, 6, 7, 8}) == 7, "Exact hit must leave one token for logits");
  check(m.sequence(2).blocks == original, "Reuse attaches same blocks");
  m.resetSequence(2);
  check(m.reusePrefix(2, {1, 9, 3, 4, 5, 6}) == 1, "Partial edge must stop before unmatched descendants");
  m.resetSequence(2);
  check(m.reusePrefix(2, {1, 2, 2000, 2001, 2002, 2003}) == 2, "Divergent suffix stops at first missing edge");
  m.resetSequence(2);
  // A second independent computation of identical tokens must not retain duplicate blocks.
  publish(3, {1, 2, 3, 4, 5, 6, 7, 8});
  check(m.cachedBlocks() == 4, "Duplicate publication creates no cache references");
  m.release(3);
  check(m.freeBlocks() == 4, "Duplicate physical blocks reclaimed");
  m.release(1);
  check(m.freeBlocks() == 4, "Released requests leave cached KV resident");
  check(m.reusePrefix(2, {1, 2, 3, 4, 5, 6, 7, 8, 9}) == 8, "Cross-request reuse after release");
  check(!m.evictRadixBlock(), "Active leaves are protected");
  m.clearRadixCache();
  check(m.cachedBlocks() == 0 && m.freeBlocks() == 4, "Flush cannot free active KV");
  m.release(2);
  check(m.freeBlocks() == 8, "Final references reclaimed after flush");

  // Leaf LRU ordering and eviction on capacity pressure.
  PagedBlockManager lru(2, 2, 16);
  lru.enableRadixCache();
  auto insert = [&](int id, std::vector<int64_t> tokens) {
    lru.create(id);
    check(lru.beginAppend({id}, {2}, {tokens}), "LRU reserve");
    lru.commit();
    lru.release(id);
  };
  insert(1, {1, 2});
  insert(2, {3, 4});
  lru.create(3);
  check(lru.reusePrefix(3, {1, 2, 9}) == 2, "Touch and protect older branch");
  check(lru.beginAppend({3}, {1}, {{9}}), "Evict idle branch to append active request");
  check(lru.cachedBlocks() == 1, "Only idle leaf evicted");
  lru.abort();
  check(lru.sequence(3).tokens == std::vector<int64_t>({1, 2}), "Abort keeps matched token history");
  lru.resetSequence(3);
  check(lru.reusePrefix(3, {3, 4, 9}) == 0, "Evicted branch must miss");
  check(lru.reusePrefix(3, {1, 2, 9}) == 2, "Recently used branch survives");
  lru.clear();
  check(lru.cachedBlocks() == 0 && lru.freeBlocks() == 2, "Full clear resets tree and pool");

  // Unprotected LRU order (not merely active-reference protection).
  PagedBlockManager order(3, 2, 16);
  order.enableRadixCache();
  for (int id = 0; id < 3; ++id) {
    order.create(id);
    check(order.beginAppend({id}, {2}, {{id * 2, id * 2 + 1}}), "LRU branch insert");
    order.commit();
    order.release(id);
  }
  order.create(3);
  check(order.reusePrefix(3, {0, 1, 9}) == 2, "Touch oldest branch");
  order.resetSequence(3);
  check(order.evictRadixBlock(), "Evict least recently used idle leaf");
  check(order.reusePrefix(3, {2, 3, 9}) == 0, "Untouched middle branch is now oldest");
  check(order.reusePrefix(3, {0, 1, 9}) == 2, "Touched oldest branch remains");
  order.resetSequence(3);
  // No room for an oversized append: cache eviction may occur, active state cannot change.
  order.reusePrefix(3, {4, 5, 9});
  const auto protected_blocks = order.sequence(3).blocks;
  check(!order.beginAppend({3}, {6}, {{6, 7, 8, 9, 10, 11}}), "OOM after reclaiming idle cache");
  check(order.sequence(3).blocks == protected_blocks && !order.pending(), "OOM preserves live sequence");
  order.clear();

  PagedBlockManager tail(3, 4, 16);
  tail.enableRadixCache();
  tail.create(0);
  mustThrow([&] { tail.beginAppend({0}, {3}); });
  check(tail.beginAppend({0}, {3}, {{1, 2, 3}}), "Partial block reserve");
  tail.commit();
  check(tail.cachedBlocks() == 1, "Committed partial blocks are cached");
  tail.fork(0, 1);
  check(tail.beginAppend({1}, {1}, {{4}}), "COW with radix history");
  check(tail.copies().size() == 1, "Shared partial block copied");
  tail.abort();
  check(tail.cachedBlocks() == 1 && tail.sequence(1).tokens.size() == 3, "Abort does not publish tokens");
  check(tail.beginAppend({1}, {1}, {{5}}), "Retry COW");
  tail.commit();
  tail.clearSequences();
  tail.create(2);
  check(tail.reusePrefix(2, {1, 2, 3, 4, 9}) == 3, "Aborted suffix cannot hit beyond committed prefix");
  tail.resetSequence(2);
  check(tail.reusePrefix(2, {1, 2, 3, 5, 9}) == 4, "Committed COW tokens hit");
  tail.clearRadixCache();
  tail.release(2);
  check(tail.freeBlocks() == 3, "Radix COW final reclamation");
}

static void tokenRadixSplits() {
  PagedBlockManager m(16, 4, 32);
  m.enableRadixCache();
  std::array<std::vector<int64_t>, 2> memory;
  for (auto& layer : memory) { layer.resize(64, -1); }
  auto append = [&](int id, const std::vector<int64_t>& tokens, bool commit = true) {
    check(m.beginAppend({id}, {static_cast<int32_t>(tokens.size())}, {tokens}), "Token split reservation");
    for (auto copy : m.copies()) {
      for (auto& layer : memory) { std::copy_n(layer.begin() + copy.first * 4, 4, layer.begin() + copy.second * 4); }
    }
    const auto& entry = m.appends()[0];
    for (int i = entry.old_length; i < entry.next.length; ++i) {
      for (int layer = 0; layer < 2; ++layer) {
        memory[layer][entry.next.blocks[i / 4] * 4 + i % 4] = tokens[i - entry.old_length] * 2 + layer;
      }
    }
    if (commit) {
      m.commit();
    } else {
      m.abort();
    }
  };
  auto verify = [&](int id, const std::vector<int64_t>& expected) {
    const auto& seq = m.sequence(id);
    check(seq.tokens == expected, "Token split history");
    for (size_t i = 0; i < expected.size(); ++i) {
      for (int layer = 0; layer < 2; ++layer) {
        check(memory[layer][seq.blocks[i / 4] * 4 + i % 4] == expected[i] * 2 + layer, "Split node/block mapping corrupted KV");
      }
    }
  };
  m.create(0);
  append(0, {'a', 'b', 'c', 'e'});
  const auto e_block = m.sequence(0).blocks[0];
  m.release(0);
  m.create(1);
  check(m.reusePrefix(1, {'a', 'b', 'c', 'd'}) == 3, "abce matches abc of abcd");
  check(m.sequence(1).blocks[0] == e_block, "Partial hit attaches original block without copying");
  append(1, {'d'}, false);
  verify(1, {'a', 'b', 'c'});
  check(m.cachedNodes() == 1, "Aborted suffix must not split or publish");
  append(1, {'d'});
  const auto d_block = m.sequence(1).blocks[0];
  check(d_block != e_block, "Shared partial tail uses COW");
  check(m.cachedNodes() == 3 && m.cachedBlocks() == 2, "abc parent plus d/e leaves, two physical blocks");
  check(memory[0][e_block * 4 + 3] == 'e' * 2, "COW must preserve old e");
  verify(1, {'a', 'b', 'c', 'd'});
  m.create(2);
  check(m.reusePrefix(2, {'a', 'b', 'c', 'e', '!'}) == 4, "Original e branch survives split");
  check(m.sequence(2).blocks[0] == e_block, "Child e chooses its physical block, not parent copy");
  verify(2, {'a', 'b', 'c', 'e'});
  m.resetSequence(2);
  check(m.reusePrefix(2, {'a', 'b', 'c', 'd', '!'}) == 4, "New d branch hits");
  check(m.sequence(2).blocks[0] == d_block, "Child d mapping");
  m.resetSequence(2);
  m.create(3);
  append(3, {'a', 'b'});
  m.release(3);
  check(m.cachedNodes() == 4, "Shorter insertion splits abc into ab/c");
  check(m.reusePrefix(2, {'a', 'b', 'x', 'd'}) == 2, "Mismatch inside prefix stops before d child");
  append(2, {'x', 'd'});
  verify(2, {'a', 'b', 'x', 'd'});
  m.release(2);
  m.create(2);
  check(m.reusePrefix(2, {'a', 'b', 'c', 'e', 1, 2, 3, 4, 5}) == 4, "Cross-block extension reuses e branch");
  append(2, {1, 2, 3, 4, 5});
  verify(2, {'a', 'b', 'c', 'e', 1, 2, 3, 4, 5});
  m.release(2);
  m.create(2);
  check(m.reusePrefix(2, {'a', 'b', 'c', 'e', 1, 2, 9}) == 6, "Match inside second physical block");
  append(2, {9});
  verify(2, {'a', 'b', 'c', 'e', 1, 2, 9});
  m.release(2);
  m.create(2);
  check(m.reusePrefix(2, {'a', 'b', 'c', 'e', 1, 2, 3, 4, 5, 6}) == 9, "Split must preserve longer sibling");
  verify(2, {'a', 'b', 'c', 'e', 1, 2, 3, 4, 5});
  m.clearSequences();
  m.clearRadixCache();
  check(m.freeBlocks() == 16 && m.cachedNodes() == 0, "Split tree flush releases every node reference");

  // One live partial block plus one COW spare suffices for repeated single-token
  // appends. Ancestor nodes must not pin a separate obsolete version per token.
  PagedBlockManager rolling(2, 8, 16);
  rolling.enableRadixCache();
  rolling.create(0);
  for (int token = 0; token < 8; ++token) {
    check(rolling.beginAppend({0}, {1}, {{token}}), "Repeated token append with one COW spare");
    if (token > 0) { check(rolling.copies().size() == 1, "Cached tail requires COW"); }
    rolling.commit();
    check(rolling.cachedBlocks() == 1 && rolling.freeBlocks() == 1, "Obsolete same-block copies reclaimed");
  }
  rolling.release(0);
  size_t evictions = 0;
  while (rolling.evictRadixBlock()) { ++evictions; }
  check(evictions == 1 && rolling.freeBlocks() == 2 && rolling.cachedNodes() == 0,
        "Compressed incremental path frees its block");
}

// Actual simulated per-layer KV is checked after prefix reuse, pressure eviction,
// fork, COW and abort. Expected values are independent full logical histories.
static void randomizedRadixHistories() {
  constexpr int count = 12, size = 4, layers = 2;
  PagedBlockManager m(count, size, 32);
  m.enableRadixCache();
  std::array<std::vector<int64_t>, layers> memory;
  for (auto& layer : memory) { layer.resize(count * size, -1); }
  auto kvValue = [](const std::vector<int64_t>& history, int position, int layer) {
    uint64_t value = 17;
    for (int i = 0; i <= position; ++i) { value = (value * 131 + history[i]) % 1000000007; }
    return static_cast<int64_t>(value * layers + layer);
  };
  std::mt19937 rng(919);
  for (int iteration = 0; iteration < 1000; ++iteration) {
    m.clearSequences();
    m.create(0);
    const int length = 1 + rng() % 31;
    std::vector<int64_t> prompt(length);
    // Frequently share two full blocks, then branch at arbitrary offsets.
    const int branch = rng() % 5, divergence = rng() % 17;
    for (int i = 0; i < length; ++i) { prompt[i] = i < divergence ? i + 1 : 100 * branch + i + 1; }
    const int hit = m.reusePrefix(0, prompt);
    check(hit >= 0 && hit < length, "Token prefix and logits remainder");
    auto verify = [&](int id, const std::vector<int64_t>& expected) {
      const auto& seq = m.sequence(id);
      check(seq.tokens == expected && seq.length == static_cast<int>(expected.size()), "Committed token history");
      for (int i = 0; i < seq.length; ++i) {
        const int slot = seq.blocks.at(i / size) * size + i % size;
        for (int layer = 0; layer < layers; ++layer) {
          check(memory[layer][slot] == kvValue(expected, i, layer), "Radix hit returned wrong layer KV");
        }
      }
    };
    verify(0, {prompt.begin(), prompt.begin() + hit});
    auto write = [&]() {
      for (auto pair : m.copies()) {
        for (auto& layer : memory) { std::copy_n(layer.begin() + pair.first * size, size, layer.begin() + pair.second * size); }
      }
      for (const auto& entry : m.appends()) {
        for (int i = entry.old_length; i < entry.next.length; ++i) {
          const int slot = entry.next.blocks.at(i / size) * size + i % size;
          for (int layer = 0; layer < layers; ++layer) { memory[layer][slot] = kvValue(entry.next.tokens, i, layer); }
        }
      }
    };
    std::vector<int64_t> suffix(prompt.begin() + hit, prompt.end());
    check(m.beginAppend({0}, {length - hit}, {suffix}), "Random radix admission must evict idle leaves");
    write();
    if (rng() % 4 == 0) {
      m.abort();
      verify(0, {prompt.begin(), prompt.begin() + hit});
    } else {
      m.commit();
      verify(0, prompt);
      m.fork(0, 1);
      const std::vector<int64_t> next{1000 + iteration};
      if (m.beginAppend({1}, {1}, {next})) {
        write();
        if (rng() % 2) {
          m.abort();
          verify(1, prompt);
        } else {
          m.commit();
          auto child = prompt;
          child.push_back(next[0]);
          verify(1, child);
        }
      }
      verify(0, prompt);
      m.release(1);
    }
    size_t refs = 0, free = 0;
    for (int block = 0; block < count; ++block) {
      refs += m.refCount(block);
      free += m.refCount(block) == 0;
    }
    check(refs == m.cachedNodes() + m.sequence(0).blocks.size(), "No leaked radix references");
    check(free == m.freeBlocks(), "Radix free-list accounting");
  }
  m.clearSequences();
  m.clearRadixCache();
  check(m.freeBlocks() == count, "All random radix blocks reclaimed");
}

// With no eviction, a separate corpus supplies an exact longest-common-prefix
// oracle. This catches implementations that return valid but unnecessarily short hits.
static void randomizedLongestPrefixes() {
  PagedBlockManager m(8192, 4, 48);
  m.enableRadixCache();
  std::vector<std::vector<int64_t>> corpus;
  std::mt19937 rng(113);
  for (int iteration = 0; iteration < 500; ++iteration) {
    std::vector<int64_t> prompt;
    if (!corpus.empty()) { prompt = corpus[rng() % corpus.size()]; }
    if (iteration % 3 != 0 || prompt.empty()) {
      prompt.resize(1 + rng() % 40);
      const size_t from = rng() % prompt.size();
      for (size_t i = from; i < prompt.size(); ++i) { prompt[i] = rng() % 8; }
    }
    size_t expected = 0;
    for (const auto& cached : corpus) {
      size_t common = 0;
      while (common < std::min(cached.size(), prompt.size() - 1) && cached[common] == prompt[common]) { ++common; }
      expected = std::max(expected, common);
    }
    m.create(0);
    const int hit = m.reusePrefix(0, prompt);
    check(static_cast<size_t>(hit) == expected, "Exact token LCP differs from independent corpus oracle");
    const std::vector<int64_t> suffix(prompt.begin() + hit, prompt.end());
    check(m.beginAppend({0}, {static_cast<int32_t>(suffix.size())}, {suffix}), "No-pressure radix append");
    m.commit();
    m.release(0);
    corpus.push_back(prompt);
  }
  m.clearRadixCache();
  check(m.freeBlocks() == m.numBlocks(), "Corpus split/merge references fully reclaimed");
}

int main() {
  try {
    boundaries();
    randomizedHistories();
    radixBoundaries();
    tokenRadixSplits();
    randomizedRadixHistories();
    randomizedLongestPrefixes();
    std::cout << "PagedBlockManager: ownership, COW, rollback, token-radix splits/LRU and 2500 randomized cases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
