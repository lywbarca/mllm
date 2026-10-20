#include <fstream>
#include <sstream>
#include <numeric>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen_ascend/modeling_qwen_ascend.hpp>
MLLM_MAIN({
  using namespace mllm; using namespace mllm::models::qwen_ascend;
  if(argc!=3) return 2;
  Logger::level()=LogLevel::kWarn;initAscendBackend();
  {
    std::ifstream file(argv[2]);std::string line;
    std::vector<int64_t> prompt,generated;
    while(std::getline(file,line)) {
      auto* dst=line.starts_with("INPUT_IDS ") ? &prompt : line.starts_with("OUTPUT_IDS ") ? &generated : nullptr;
      if(dst) {std::istringstream stream(line.substr(line.find(' ')+1));int64_t token;while(stream>>token)dst->push_back(token);}
    }
    if(prompt.empty() || generated.size()<63) return 3;
    std::string root=argv[1];
    QwenAscendConfig cfg(root+"/config-bench-paged.json");
    auto weights=mllm::load(root+"/model-fp16.mllm",ModelFileVersion::kV2);
    setenv("MLLM_ASCEND_NZ_WEIGHTS","0",1);
    QwenAscendForCausalLM baseline(cfg);baseline.to(kAscend);baseline.load(weights);
    setenv("MLLM_ASCEND_NZ_WEIGHTS","1",1);
    QwenAscendForCausalLM optimized(cfg);optimized.to(kAscend);optimized.load(weights);
    for(int step=0;step<64;++step) {
      auto ids=step==0 ? prompt : std::vector<int64_t>{generated[step-1]};
      auto input=Tensor::fromVector(ids,{1,int(ids.size())},kInt64,kCPU);
      setenv("MLLM_ASCEND_NZ_WEIGHTS","0",1);setenv("MLLM_ASCEND_PREFILL_ENCODER","0",1);
      auto a=baseline.forward({{"sequence",input}},{}).at("sequence");
      setenv("MLLM_ASCEND_NZ_WEIGHTS","1",1);setenv("MLLM_ASCEND_PREFILL_ENCODER","1",1);
      auto b=optimized.forward({{"sequence",input}},{}).at("sequence");
      float max_error=0;double squares=0;int best_a=0,best_b=0;
      for(int i=0;i<cfg.vocab_size;++i) {
        float x=a.ptr<float>()[i],y=b.ptr<float>()[i],error=std::abs(x-y);
        if(!std::isfinite(x)||!std::isfinite(y)) throw std::runtime_error("Nonfinite logits");
        max_error=std::max(max_error,error);squares+=error*error;
        if(x>a.ptr<float>()[best_a])best_a=i;if(y>b.ptr<float>()[best_b])best_b=i;
      }
      std::cout<<"LOGITS step="<<step<<" max_abs="<<max_error<<" rms="<<std::sqrt(squares/cfg.vocab_size)
        <<" baseline_top="<<best_a<<" new_top="<<best_b
        <<" baseline_margin="<<a.ptr<float>()[best_a]-a.ptr<float>()[best_b]
        <<" new_margin="<<b.ptr<float>()[best_b]-b.ptr<float>()[best_a]<<std::endl;
    }
  }
  shutdownContext();return 0;
})
