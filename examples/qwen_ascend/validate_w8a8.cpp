#include <fstream>
#include <sstream>
#include <cmath>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen_ascend/modeling_qwen_ascend.hpp>
#include <mllm/models/qwen_ascend/tokenization_qwen_ascend.hpp>
MLLM_MAIN({
  using namespace mllm;using namespace mllm::models::qwen_ascend;
  if(argc!=4)return 2;
  Logger::level()=LogLevel::kWarn;initAscendBackend();
  {
    std::string fpdir=argv[1],qdir=argv[2];
    QwenAscendConfig cfg(fpdir+"/config-bench-paged.json");
    QwenAscendForCausalLM baseline(cfg),quantized(cfg);
    baseline.to(kAscend);quantized.to(kAscend);
    baseline.load(mllm::load(fpdir+"/model-fp16.mllm",ModelFileVersion::kV2));
    quantized.load(mllm::load(qdir+"/model-w8a8.mllm",ModelFileVersion::kV2));
    QwenAscendTokenizer tokenizer(fpdir+"/tokenizer.json");
    const std::vector<std::string> prompts={
      "请用一句话介绍你自己。",
      "只回答数字：17乘以23等于多少？",
      "法国的首都是哪个城市？请简短回答。",
      "把“今天的天气很好”翻译成英文，只输出译文。",
      "用Python写一个函数，判断一个整数是否为偶数。",
      "请用两句话解释为什么天空通常是蓝色的。"};
    for(size_t i=0;i<prompts.size();++i)for(int variant=0;variant<2;++variant) {
      auto& model=variant ? quantized : baseline;
      model.clearCache();QwenAscendMessage message;message.prompt=prompts[i];
      auto input=tokenizer.convertMessage(message);
      models::ARGenerationArgs args{{"max_length",AnyValue(64)},{"do_sample",AnyValue(false)}};
      std::vector<int64_t> tokens;
      for(const auto& item:model.chat(input,args))tokens.push_back(item.cur_token_id);
      std::cout<<"QUALITY case="<<i<<" model="<<(variant ? "w8a8" : "fp16")
               <<" generated="<<model.perfStats().generated_tokens
               <<" text="<<tokenizer.decode(tokens)<<std::endl;
    }
    baseline.clearCache();quantized.clearCache();
    std::ifstream file(argv[3]);std::string line;
    std::vector<int64_t> prompt,generated;
    while(std::getline(file,line)) {
      auto* dst=line.starts_with("INPUT_IDS ") ? &prompt : line.starts_with("OUTPUT_IDS ") ? &generated : nullptr;
      if(dst) {std::istringstream stream(line.substr(line.find(' ')+1));int64_t token;while(stream>>token)dst->push_back(token);}
    }
    if(prompt.empty()||generated.size()<63)throw std::runtime_error("Invalid reference log");
    int agreed=0;double total_kl=0;
    for(int step=0;step<64;++step) {
      auto ids=step==0 ? prompt : std::vector<int64_t>{generated[step-1]};
      auto input=Tensor::fromVector(ids,{1,int(ids.size())},kInt64,kCPU);
      auto a=baseline.forward({{"sequence",input}},{}).at("sequence");
      auto b=quantized.forward({{"sequence",input}},{}).at("sequence");
      float max_error=0;double squares=0;int best_a=0,best_b=0;
      for(int i=0;i<cfg.vocab_size;++i) {
        float x=a.ptr<float>()[i],y=b.ptr<float>()[i],e=std::abs(x-y);
        if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("Nonfinite logits");
        max_error=std::max(max_error,e);squares+=e*e;
        if(x>a.ptr<float>()[best_a])best_a=i;if(y>b.ptr<float>()[best_b])best_b=i;
      }
      double za=0,zb=0,kl=0;
      double ma=a.ptr<float>()[best_a],mb=b.ptr<float>()[best_b];
      for(int i=0;i<cfg.vocab_size;++i) {za+=std::exp(a.ptr<float>()[i]-ma);zb+=std::exp(b.ptr<float>()[i]-mb);}
      double la=ma+std::log(za),lb=mb+std::log(zb);
      for(int i=0;i<cfg.vocab_size;++i) {
        double logp=a.ptr<float>()[i]-la,logq=b.ptr<float>()[i]-lb;
        kl+=std::exp(logp)*(logp-logq);
      }
      agreed+=best_a==best_b;total_kl+=kl;
      std::cout<<"LOGITS step="<<step<<" max_abs="<<max_error<<" rms="<<std::sqrt(squares/cfg.vocab_size)
        <<" fp16_top="<<best_a<<" w8a8_top="<<best_b<<" kl="<<kl<<std::endl;
    }
    std::cout<<"VALIDATION top1_agreed="<<agreed<<" steps=64 mean_kl="<<total_kl/64<<std::endl;
  }
  shutdownContext();return 0;
})
