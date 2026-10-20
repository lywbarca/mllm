#include <algorithm>
#include <iostream>
#include <vector>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen_ascend/modeling_qwen_ascend.hpp>
#include <mllm/models/qwen_ascend/tokenization_qwen_ascend.hpp>
#include <mllm/backends/ascend/AscendCommon.hpp>
MLLM_MAIN({
  if (argc != 5) { std::cerr << "usage: benchmark MODEL_DIR MODE INPUT_TOKENS RUN_ID"<<std::endl; return 2; }
  const std::string root=argv[1], mode=argv[2];
  const int length=std::stoi(argv[3]), run=std::stoi(argv[4]);
  mllm::Logger::level()=mllm::LogLevel::kWarn;
  mllm::initAscendBackend();
  {
    using namespace mllm;
    using namespace mllm::models::qwen_ascend;
    QwenAscendConfig cfg(root+"/config-bench-"+mode+".json");
    QwenAscendForCausalLM model(cfg);
    model.to(kAscend);
    const char* weight_env=std::getenv("MLLM_ASCEND_BENCH_MODEL_FILE");
    const std::string weight_file=weight_env ? weight_env : "model-fp16.mllm";
    model.load(mllm::load(root+"/"+weight_file,ModelFileVersion::kV2));
    std::cout<<"WEIGHT_FILE "<<weight_file<<" KV_CACHE FP16"<<std::endl;
    QwenAscendTokenizer tokenizer(root+"/tokenizer.json");
    QwenAscendMessage msg;
    msg.prompt="请阅读以下资料。";
    for(int i=0;i<100;++i)
      msg.prompt+="人工智能可以帮助人们整理信息、学习知识和解决问题。使用时需要核对事实，保护隐私，并结合实际需求判断结果。";
    msg.prompt+="请根据这些资料，详细解释人工智能的用途和使用注意事项。";
    if (length == 18) msg.prompt="请用一句话介绍你自己。";
    auto full=tokenizer.convertMessage(msg).at("sequence");
    if (full.numel()<static_cast<size_t>(length)) throw std::runtime_error("Prompt too short");
    auto prompt=Tensor::empty({1,length},kInt64,kCPU).alloc();
    // Preserve the chat prefix and the final question/assistant suffix; trim context only.
    const int suffix=std::min(48,length);
    std::copy_n(full.ptr<int64_t>(),length-suffix,prompt.ptr<int64_t>());
    std::copy_n(full.ptr<int64_t>()+full.numel()-suffix,suffix,prompt.ptr<int64_t>()+length-suffix);
    std::cout<<"INPUT_IDS";
    for(int i=0;i<length;++i) std::cout<<" "<<prompt.ptr<int64_t>()[i];
    std::cout<<std::endl;
    models::ARGenerationOutputPast inputs{{"sequence",prompt}};
    model.clearCache();
    std::cout<<"WARMUP_BEGIN"<<std::endl;
    (void)model.forward(inputs,{});
    ascend::syncGlobalAtbStream();
    model.clearCache();
    std::cout<<"MEASURE_BEGIN"<<std::endl;
    models::ARGenerationArgs args{{"max_length",AnyValue(64)},{"min_new_tokens",AnyValue(length == 18 ? 0 : 64)},{"do_sample",AnyValue(false)}};
    std::vector<int64_t> generated;
    for(auto& item:model.chat(inputs,args)) generated.push_back(item.cur_token_id);
    ascend::syncGlobalAtbStream();
    auto stats=model.perfStats();
    const int expected_tokens = length == 18 ? stats.generated_tokens : 64;
    if(!stats.valid || !stats.completed || expected_tokens<1 || expected_tokens>64 || stats.prefill_tokens!=length ||
       stats.generated_tokens!=expected_tokens || stats.decode_steps!=expected_tokens-1)
      throw std::runtime_error("Benchmark token counts or timing invalid");
    std::cout<<"RESULT mode="<<mode<<" input="<<length<<" output="<<stats.generated_tokens
             <<" run="<<run<<" prefill_us="<<stats.prefill_duration_us
             <<" decode_us="<<stats.decode_duration_us<<" decode_steps="<<stats.decode_steps
             <<" prefill_tps="<<double(length)*1e6/stats.prefill_duration_us
             <<" decode_tps="<<double(stats.decode_steps)*1e6/stats.decode_duration_us<<std::endl;
    std::cout<<"OUTPUT_IDS";
    for(auto id:generated) std::cout<<" "<<id;
    std::cout<<std::endl<<"OUTPUT_TEXT "<<tokenizer.decode(generated)<<std::endl;
  }
  mllm::shutdownContext();
  return 0;
})
