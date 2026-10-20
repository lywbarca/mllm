// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <random>
#include <algorithm>
#include <limits>

#include <fmt/core.h>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/UnsafeMacros.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models {

ARGenerationChatContext::ARGenerationChatContext(ARGeneration& gen, const ARGenerationOutputPast& input,
                                                 const ARGenerationArgs& args)
    : gen_(gen), input_(input), args_(args) {
  MLLM_EMPTY_SCOPE;
}

ARGenerationChatIterator ARGenerationChatContext::begin() { return {gen_, input_, args_}; }

ARGenerationChatIterator ARGenerationChatContext::end() { return {}; }

ARGenerationChatIterator::ARGenerationChatIterator(ARGeneration& gen, const ARGenerationOutputPast& initial_input,
                                                   const ARGenerationArgs& args)
    : gen_(&gen), current_input_(initial_input), args_(args), finished_(false) {
  temperature_ = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  top_k_ = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  top_p_ = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  max_length_ = args.count("max_length") ? args.at("max_length").get<int>() : gen.max_length_;
  min_new_tokens_ = args.count("min_new_tokens") ? args.at("min_new_tokens").get<int>() : 0;
  eos_token_id_ = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : gen.eos_token_id_;
  do_sample_ = args.count("do_sample") ? args.at("do_sample").get<bool>() : gen.do_sample_;
  if (max_length_ <= 0) { throw std::invalid_argument("max_length must be positive"); }
  if (min_new_tokens_ < 0 || min_new_tokens_ > max_length_) {
    throw std::invalid_argument("min_new_tokens must be between 0 and max_length");
  }

  step();
}

ARGenerationChatIterator::ARGenerationChatIterator() : gen_(nullptr), finished_(true) { MLLM_EMPTY_SCOPE; }  // NOLINT

ARGenerationChatIterator::reference ARGenerationChatIterator::operator*() const { return current_step_; }

ARGenerationChatIterator::pointer ARGenerationChatIterator::operator->() const { return &current_step_; }

ARGenerationChatIterator& ARGenerationChatIterator::operator++() {
  if (finished_ || gen_ == nullptr) {
    MLLM_WARN("Invalid iterator: cannot increment an end() iterator.");
    return *this;
  }
  step();
  return *this;
}

bool ARGenerationChatIterator::operator==(const ARGenerationChatIterator& other) const {
  if (this->finished_ && other.finished_) return true;
  if (this->finished_ || other.finished_) return false;
  if (this->gen_ != other.gen_) return false;
  return this->finished_ == other.finished_;
}

bool ARGenerationChatIterator::operator!=(const ARGenerationChatIterator& other) const { return !(*this == other); }

__MLLM_UNSAFE_OPT_BEGIN_O3
void ARGenerationChatIterator::step() {
  if (finished_ || gen_ == nullptr) {
    MLLM_WARN("Invalid iterator: cannot step an end() iterator.");
    return;
  }

  if (step_count_ >= max_length_) {
    gen_->generationEventEndTimePoint();
    finished_ = true;
    return;
  }

  bool use_sampling = do_sample_ || (temperature_ != 1.0f) || (top_k_ > 0) || (top_p_ > 0.0f);

  // Timing
  if (step_count_ == 0) {
    gen_->prefillEventStartTimePoint();
  } else {
    gen_->decodeEventStartTimePoint();
  }

  ARGenerationOutputPast output = gen_->forwardForGeneration(current_input_, args_, !use_sampling);

  // Timing
  if (step_count_ == 0) {
    if (current_input_.count("sequence")) { gen_->ar_prefill_tokens_ = current_input_["sequence"].shape()[1]; }
    gen_->prefillEventEndTimePoint();
  }

  Tensor logits = output["sequence"];
  auto device = gen_->generationInputDevice(logits);

  int64_t next_token_id;
  if (use_sampling) {
    logits = logits.to(kCPU);
    if (step_count_ + 1 < min_new_tokens_) { gen_->suppressEosLogits(logits, eos_token_id_); }
    if (top_k_ > 0) {
      next_token_id = gen_->sampleTopK(logits, top_k_, temperature_);
    } else if (top_p_ > 0.0f) {
      next_token_id = gen_->sampleTopP(logits, top_p_, temperature_);
    } else {
      next_token_id = gen_->sampleTemperature(logits, temperature_);
    }
  } else {
    next_token_id = gen_->selectGreedy(logits, step_count_ + 1 < min_new_tokens_, eos_token_id_);
  }

  current_step_.current_step = step_count_;
  current_step_.cur_token_id = next_token_id;

  if (step_count_ == 0) {
    gen_->firstTokenEventTimePoint();
  } else {
    gen_->decodeEventEndTimePoint();
  }

  step_count_++;
  gen_->ar_steps_++;

  if (step_count_ >= min_new_tokens_ && gen_->isEosToken(next_token_id, eos_token_id_)) {
    gen_->generationEventEndTimePoint();
    finished_ = true;
    return;
  }

  // [B, S]
  current_input_ = std::move(output);

  current_input_["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
  current_input_["sequence"].at<mllm_int64_t>({0, 0}) = next_token_id;
  current_input_["sequence"] = current_input_["sequence"].to(device);
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3
ARGenerationOutputPast ARGeneration::generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
  ARGenerationOutputPast past = input;
  std::vector<int64_t> generated_tokens;

  // Init values
  float temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  int top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  float top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  int min_new_tokens = args.count("min_new_tokens") ? args.at("min_new_tokens").get<int>() : 0;
  int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  bool do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_;
  if (max_length <= 0) { throw std::invalid_argument("max_length must be positive"); }
  if (min_new_tokens < 0 || min_new_tokens > max_length) {
    throw std::invalid_argument("min_new_tokens must be between 0 and max_length");
  }

  bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);

  for (int i = 0; i < max_length; ++i) {
    // Timing
    if (i == 0) {
      prefillEventStartTimePoint();
      if (past.count("sequence") > 0) { ar_prefill_tokens_ = past["sequence"].shape()[1]; }
    } else {
      decodeEventStartTimePoint();
    }

    ARGenerationOutputPast output = forwardForGeneration(past, args, !use_sampling);

    // Timing
    if (i == 0) { prefillEventEndTimePoint(); }

    Tensor logits = output["sequence"];

    int64_t next_token_id;
    if (use_sampling) {
      logits = logits.to(kCPU);
      if (i + 1 < min_new_tokens) { suppressEosLogits(logits, eos_token_id); }
      if (top_k > 0) {
        next_token_id = sampleTopK(logits, top_k, temperature);
      } else if (top_p > 0.0f) {
        next_token_id = sampleTopP(logits, top_p, temperature);
      } else {
        next_token_id = sampleTemperature(logits, temperature);
      }
    } else {
      next_token_id = selectGreedy(logits, i + 1 < min_new_tokens, eos_token_id);
    }

    generated_tokens.push_back(next_token_id);
    ar_steps_++;

    if (i == 0) {
      firstTokenEventTimePoint();
    } else {
      decodeEventEndTimePoint();
    }

    if (i + 1 >= min_new_tokens && isEosToken(next_token_id, eos_token_id)) { break; }

    // [B, S]
    past = output;
    past["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    past["sequence"].at<mllm_int64_t>({0, 0}) = next_token_id;
    past["sequence"] = past["sequence"].to(generationInputDevice(logits));
  }

  generationEventEndTimePoint();

  // From blob
  Tensor generated_tensor = Tensor::empty({(int32_t)generated_tokens.size()}, kInt64, kCPU).alloc();

  std::copy(generated_tokens.begin(), generated_tokens.end(), generated_tensor.ptr<mllm_int64_t>());

  // Clear things in output.
  past["logits"] = Tensor::nil();
  past["input_ids"] = Tensor::nil();
  past["generated_sequence"] = generated_tensor;
  return past;
}
__MLLM_UNSAFE_OPT_END

void ARGeneration::streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                                  const std::function<void(int64_t)>& callback) {
  ARGenerationOutputPast past = input;

  // Init values
  float temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  int top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  float top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  int min_new_tokens = args.count("min_new_tokens") ? args.at("min_new_tokens").get<int>() : 0;
  int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  bool do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_;
  if (max_length <= 0) { throw std::invalid_argument("max_length must be positive"); }
  if (min_new_tokens < 0 || min_new_tokens > max_length) {
    throw std::invalid_argument("min_new_tokens must be between 0 and max_length");
  }

  bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);

  for (int i = 0; i < max_length; ++i) {
    // Timing
    if (i == 0) {
      prefillEventStartTimePoint();
    } else {
      decodeEventStartTimePoint();
    }

    ARGenerationOutputPast output = forwardForGeneration(past, args, !use_sampling);

    // Timing
    if (i == 0) {
      if (past.count("sequence") > 0) { ar_prefill_tokens_ = past["sequence"].shape()[1]; }
      prefillEventEndTimePoint();
    }

    Tensor logits = output["sequence"];
    auto device = generationInputDevice(logits);

    int64_t next_token_id;
    if (use_sampling) {
      logits = logits.to(kCPU);
      if (i + 1 < min_new_tokens) { suppressEosLogits(logits, eos_token_id); }
      if (top_k > 0) {
        next_token_id = sampleTopK(logits, top_k, temperature);
      } else if (top_p > 0.0f) {
        next_token_id = sampleTopP(logits, top_p, temperature);
      } else {
        next_token_id = sampleTemperature(logits, temperature);
      }
    } else {
      next_token_id = selectGreedy(logits, i + 1 < min_new_tokens, eos_token_id);
    }

    ar_steps_++;
    if (i == 0) {
      firstTokenEventTimePoint();
    } else {
      decodeEventEndTimePoint();
    }

    callback(next_token_id);

    if (i + 1 >= min_new_tokens && isEosToken(next_token_id, eos_token_id)) { break; }

    // [B, S]
    past = output;
    past["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    past["sequence"].at<mllm_int64_t>({0, 0}) = next_token_id;
    past["sequence"] = past["sequence"].to(device);
  }

  generationEventEndTimePoint();
}

IROutput ARGeneration::trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) { return {}; }

ARGenerationPerformanceStats ARGeneration::perfStats() const {
  ARGenerationPerformanceStats stats;
  stats.completed = generation_completed_;
  stats.prefill_tokens = ar_prefill_tokens_;
  stats.generated_tokens = ar_steps_;
  stats.decode_steps = ar_steps_ > 0 ? ar_steps_ - 1 : 0;

  if (!prefill_started_ || !prefill_finished_ || !first_token_recorded_) { return stats; }

  stats.prefill_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(llm_prefill_end_time_ - llm_prefill_start_time_).count();
  stats.decode_duration_us = llm_decode_duration_.count();
  stats.ttft_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(llm_first_token_time_ - llm_prefill_start_time_).count();
  stats.total_duration_us = stats.ttft_duration_us + stats.decode_duration_us;
  stats.valid = stats.prefill_duration_us >= 0 && stats.decode_duration_us >= 0 && stats.ttft_duration_us >= 0;
  return stats;
}

void ARGeneration::perfSummary() {
  const auto stats = perfStats();

  fmt::print(fg(fmt::color::cyan), "\n{:=^50}\n", " Performance Summary ");
  if (!stats.valid) {
    fmt::print(fg(fmt::color::yellow), "No valid generation timing is available.\n");
    fmt::print(fg(fmt::color::cyan), "{:=^50}\n", "");
    return;
  }

  double avg_decode_duration = 0;
  if (stats.decode_steps > 0) {
    avg_decode_duration = static_cast<double>(stats.decode_duration_us) / static_cast<double>(stats.decode_steps);
  }

  double prefill_tokens_per_sec = 0;
  double decode_tokens_per_sec = 0;

  if (stats.prefill_duration_us > 0) {
    prefill_tokens_per_sec = static_cast<double>(stats.prefill_tokens) / (stats.prefill_duration_us / 1000000.0);
  }

  if (stats.decode_duration_us > 0 && stats.decode_steps > 0) {
    decode_tokens_per_sec = static_cast<double>(stats.decode_steps) / (stats.decode_duration_us / 1000000.0);
  }

  fmt::print(fg(fmt::color::white), "{:<20}: ", "Total time");
  fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs\n", static_cast<double>(stats.total_duration_us));

  fmt::print(fg(fmt::color::white), "{:<20}: ", "Prefill time");
  fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs", static_cast<double>(stats.prefill_duration_us));
  if (prefill_tokens_per_sec > 0) { fmt::print(fg(fmt::color::white), " ({:>6.2f} tokens/s)", prefill_tokens_per_sec); }
  fmt::print("\n");

  fmt::print(fg(fmt::color::white), "{:<20}: ", "Decode time");
  fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs", static_cast<double>(stats.decode_duration_us));
  if (decode_tokens_per_sec > 0) { fmt::print(fg(fmt::color::white), " ({:>6.2f} tokens/s)", decode_tokens_per_sec); }
  fmt::print("\n");

  fmt::print(fg(fmt::color::white), "{:<20}: ", "TTFT");
  fmt::print(fg(fmt::color::magenta), "{:>10.2f} μs\n", static_cast<double>(stats.ttft_duration_us));

  fmt::print(fg(fmt::color::white), "{:<20}: ", "Prefill tokens");
  fmt::print(fg(fmt::color::green), "{:>10}\n", stats.prefill_tokens);

  fmt::print(fg(fmt::color::white), "{:<20}: ", "Decode steps");
  fmt::print(fg(fmt::color::green), "{:>10}\n", stats.decode_steps);

  if (stats.decode_steps > 0) {
    fmt::print(fg(fmt::color::white), "{:<20}: ", "Avg decode time");
    fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs/token\n", avg_decode_duration);
  }

  fmt::print(fg(fmt::color::cyan), "{:=^50}\n", "");

  if (!custom_event_time_.empty()) {
    fmt::print(fg(fmt::color::magenta), "\n{:=^50}\n", " Custom Events ");
    for (const auto& pair : custom_event_time_) {
      const auto& name = pair.first;
      if (!completed_custom_events_.count(name)) { continue; }
      const auto& time_points = pair.second;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_points.second - time_points.first).count();

      fmt::print(fg(fmt::color::white), "{:<20}: ", name);
      fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs\n", (double)duration);
    }
    fmt::print(fg(fmt::color::magenta), "{:=^50}\n", "");
  }
}

int64_t ARGeneration::selectGreedy(Tensor logits, bool suppress_eos, int64_t eos_token_id) {
  logits=logits.to(kCPU);
  if (suppress_eos) suppressEosLogits(logits,eos_token_id);
  return sampleGreedy(logits);
}

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
int64_t ARGeneration::sampleGreedy(Tensor& logits) {
  Tensor last_logits = getLastLogits(logits);

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(last_logits.dtype(), kFloat32);

  auto logits_data = last_logits.ptr<float>();
  int vocab_size = last_logits.shape().back();

  auto max_it = std::max_element(logits_data, logits_data + vocab_size);

  return std::distance(logits_data, max_it);
}
__MLLM_UNSAFE_OPT_END

int64_t ARGeneration::sampleTemperature(Tensor& logits, float temperature) {
  Tensor last_logits = getLastLogits(logits);

  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  Tensor probs = nn::functional::softmax(last_logits, -1);

  // last_logits may alias logits (including the caller's output).
  // Let Tensor ownership release storage after sampling.

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);

  return categoricalSample(probs);
}

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
int64_t ARGeneration::sampleTopK(Tensor& logits, int k, float temperature) {
  // We need the last logits
  Tensor last_logits = getLastLogits(logits);

  // Apply temperature.
  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  // Get Softmax probabilities
  Tensor probs = nn::functional::softmax(last_logits, -1);

  // last_logits may alias logits (including the caller's output).
  // Let Tensor ownership release storage after sampling.

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  // Indexing
  std::vector<int> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });
  // Get top k
  std::vector<float> top_k_probs(k);
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    top_k_probs[i] = prob_data[indices[i]];
    sum += top_k_probs[i];
  }
  // Norm probs
  for (int i = 0; i < k; ++i) { top_k_probs[i] *= (1.f / sum); }

  return indices[sampleFromDistribution(top_k_probs)];
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
int64_t ARGeneration::sampleTopP(Tensor& logits, float p, float temperature) {
  Tensor last_logits = getLastLogits(logits);

  if (temperature != 1.0f) { last_logits = last_logits * (1.f / temperature); }

  Tensor probs = nn::functional::softmax(last_logits, -1);

  // last_logits may alias logits (including the caller's output).
  // Let Tensor ownership release storage after sampling.

  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  std::vector<int> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });
  std::vector<float> top_probs;
  float cumulative_prob = 0.0f;
  int i = 0;
  for (; i < vocab_size && cumulative_prob < p; ++i) {
    top_probs.push_back(prob_data[indices[i]]);
    cumulative_prob += prob_data[indices[i]];
  }
  float sum = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
  for (float& prob : top_probs) { prob *= (1.f / sum); }

  return indices[sampleFromDistribution(top_probs)];
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
ARGenerationChatContext ARGeneration::chat(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
  return {*this, input, args};
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
int64_t ARGeneration::categoricalSample(const Tensor& probs) {
  // FIXME:
  // Prob may be in fp16 dtype. We need to handle it.
  MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
  auto prob_data = probs.ptr<float>();
  int vocab_size = probs.shape().back();

  std::vector<float> cumulative_probs(vocab_size);
  std::partial_sum(prob_data, prob_data + vocab_size, cumulative_probs.begin());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  float r = dis(gen);

  auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
  return std::distance(cumulative_probs.begin(), it);
}
__MLLM_UNSAFE_OPT_END

Tensor ARGeneration::getLastLogits(Tensor& logits) {
  // [B, S, D] for almost all TextLLM, VLM.
  if (logits.shape().size() == 3) {
    if (logits.shape()[1] == 1) {
      return logits;
    } else {
      // Get the last logits
      return logits[{
          kAll,
          logits.shape()[1] - 1,
          kAll,
      }];
    }
  } else {
    throw std::runtime_error("getLastLogits suppose inputs logits is [B, S, D] layout.");
  }
}

int ARGeneration::sampleFromDistribution(const std::vector<float>& probs) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(gen);
}

void ARGeneration::prefillEventStartTimePoint() {
  ar_steps_ = 0;
  ar_prefill_tokens_ = 0;
  llm_decode_duration_ = std::chrono::microseconds::zero();
  prefill_started_ = true;
  prefill_finished_ = false;
  first_token_recorded_ = false;
  decode_event_active_ = false;
  generation_completed_ = false;
  custom_event_time_.clear();
  completed_custom_events_.clear();
  llm_prefill_start_time_ = PerformanceClock::now();
}

void ARGeneration::prefillEventEndTimePoint() {
  llm_prefill_end_time_ = PerformanceClock::now();
  prefill_finished_ = prefill_started_;
}

void ARGeneration::firstTokenEventTimePoint() {
  llm_first_token_time_ = PerformanceClock::now();
  first_token_recorded_ = prefill_finished_;
}

void ARGeneration::decodeEventStartTimePoint() {
  if (decode_event_active_) { return; }
  llm_decode_start_time_ = PerformanceClock::now();
  decode_event_active_ = true;
}

void ARGeneration::decodeEventEndTimePoint() {
  if (!decode_event_active_) { return; }
  llm_decode_end_time_ = PerformanceClock::now();
  llm_decode_duration_ += std::chrono::duration_cast<std::chrono::microseconds>(llm_decode_end_time_ - llm_decode_start_time_);
  decode_event_active_ = false;
}

void ARGeneration::generationEventEndTimePoint() {
  decodeEventEndTimePoint();
  generation_completed_ = true;
}

bool ARGeneration::isEosToken(int64_t token_id, int64_t primary_eos_token_id) const {
  return token_id == primary_eos_token_id || additional_eos_token_ids_.contains(token_id);
}

void ARGeneration::suppressEosLogits(Tensor& logits, int64_t primary_eos_token_id) {
  auto last_logits = getLastLogits(logits);
  if (last_logits.dtype() != kFloat32) { throw std::runtime_error("min_new_tokens currently requires float32 logits"); }

  const auto vocab_size = static_cast<int64_t>(last_logits.shape().back());
  auto* logits_data = last_logits.ptr<float>();
  const auto suppress = [&](int64_t token_id) {
    if (token_id >= 0 && token_id < vocab_size) { logits_data[token_id] = std::numeric_limits<float>::lowest(); }
  };
  suppress(primary_eos_token_id);
  for (const auto token_id : additional_eos_token_ids_) { suppress(token_id); }
}

void ARGeneration::customEventStartTimePoint(const std::string& name) {
  completed_custom_events_.erase(name);
  custom_event_time_[name].first = PerformanceClock::now();
}

void ARGeneration::customEventEndTimePoint(const std::string& name) {
  auto event = custom_event_time_.find(name);
  if (event == custom_event_time_.end()) { return; }
  event->second.second = PerformanceClock::now();
  completed_custom_events_.insert(name);
}

}  // namespace mllm::models
