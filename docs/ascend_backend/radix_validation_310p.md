# 分页 KV + Radix 存储接入验收（2026-10-17）

## 结论

已在 `310p:/home/ubuntu/mllm` 启用并实际验收分页 KV 的 radix 前缀共享。现有实现已经把 token radix 节点绑定到真实分页池的 block ID；本轮沿用这条实现，增加可运行配置、真实权重验收程序和跨层物理存储回归测试，没有另建归档 KV 副本或重写树。

新请求最长前缀匹配后直接挂接物理块，仅计算剩余后缀。整块共享；半块续写通过跨所有层的 COW 保护旧分支。所有层写完并同步后才发布前缀。请求释放保留缓存，LRU 在压力下回收无活跃请求引用的叶节点；清空缓存不破坏在役 KV。

## 本轮产物

- `examples/qwen_ascend/config_qwen3_06b_radix.json`：Qwen3-0.6B 可运行配置，NZ、128-token block、8 blocks、最大请求 768 token、`radix_cache=true`。
- 模型目录的 `config-bench-radix.json`：同一配置，供现有 benchmark 的 `radix` 模式使用。原 `config-bench-paged.json` 保留为无 radix 对照。
- `examples/qwen_ascend/validate_radix.cpp` 及对应 CMake 目标：真实权重、全词表 logits 对照、容量压力、活跃请求保护、冷/热延迟测量。
- `tests/ascend/AscendPagedKVCacheTest.hpp` 新增 `RadixOwnsPhysicalStorageAcrossLayersAndEviction`：实际设备逐元素核对两层 K/V，包含前缀共享、尾块 COW、回滚、淘汰、清树及完全回收。
- 本目录 `source/` 保存相关源码快照，`results/` 保存日志、测试 XML、源文件 SHA256；远端原始结果在 `/home/ubuntu/radix-integration-results/`。

## 环境与边界

- 设备：Ascend 310P1，设备 0；独立 CANN 8.2.RC1 + ATB 环境 `/home/ubuntu/Ascend-mllm-8.2.RC1/set_env.sh`。
- 源码基线提交 `1d3c3d8e506e3d12e902af82866972623b240ce0` 加现有本地改动；它不是干净上游版本。此次保留其他改动。
- 真实模型：`/home/ubuntu/models/Qwen3-0.6B/model-fp16.mllm`，28 层、FP16 权重及 KV。每模型分页池 117,440,512 bytes（112 MiB）。
- 启用现有 NZ 权重、Decoder graph、prefill encoder、异步提交、常量缓存、输出投影复用、packed projections 与 fused MLP；完整环境变量见 `run.sh`。
- 本轮结论限定 FP16，未测试 W8A8、310B/910B、生产调度器或任意并发负载。

## 测试结果

1. 重新编译独立 CPU 块管理测试：固定回归和 2500 随机场景通过。
2. 重新编译 Ascend 测试程序：`AscendPagedKVCacheTest.*:AscendRealPagedAttentionTest.*` 共 **6/6 通过，0 skip、0 failure**。
3. 真实 Qwen3-0.6B：256、512 token 各 30 次全词表比较，共 **60/60 通过，下一 token argmax 60/60 一致**。覆盖冷启动、精确命中、127/129 位置分叉、原分支重放、短 key、8 步 teacher-forced decode、10 个不相关请求造成淘汰、受保护请求继续运行。
4. 两层存储测试按完整历史计算独立期望值，逐元素检查实际 K/V；全部 exact。结束时物理块全部回收。
5. 完整生成入口：`benchmark ... radix 256 1` 成功生成 64 token / 63 decode steps；prefill 24.029 ms，decode 55.8446 token/s。该 benchmark 先做同 prompt warmup，故这里是**热前缀**，不能与普通冷 prefill 吞吐直接比较。

真实权重精度门槛在测量前固定为 finite、max_abs ≤ 0.15、RMS ≤ 0.03、relative L2 ≤ 0.01、cosine ≥ 0.9999。实际最坏 max_abs 0.078125、RMS 0.0116661、relative L2 0.00296072，最低 cosine 0.999996。非零误差发生在部分前缀分叉的后缀计算，不能称所有 logits bit-exact。

## 首 logits 延迟

同一模型、同一输入、已预热执行环境；冷前缀测量前显式清树，热前缀重置请求但保留树。每组 3 次，表中为中位数。时间含输入 Tensor 构建、forward 和 CPU logits 返回，不含模型加载、tokenization 或网络，不是完整服务 TTFT。

| 输入 token | 热命中 token | 冷前缀 | 热前缀 | 延迟降低 |
|---|---|---|---|---|
| 256 | 255 | 34.124 ms | 19.955 ms | 41.5% |
| 512 | 511 | 57.110 ms | 21.605 ms | 62.2% |

精确命中仍留最后一个 token 计算 logits。这里只测了两个固定 prompt、每组三次，不外推到所有请求分布。

## 复现

在 310P 主机 source 上述环境后，构建目标 `mllm-qwen-ascend-validate-radix`、`Mllm-Test-AscendKernel` 和 `Mllm-Test-Engine-PagedBlockManager`。新增 CMake 目标前需先使用现有 CMake 重新 configure（保留 build cache）。

运行 `bash /home/ubuntu/radix-integration-results/run.sh` 复现真实模型对照及计时；`generation.sh` 复现生成入口。

测试过滤器：`--gtest_filter=AscendPagedKVCacheTest.*:AscendRealPagedAttentionTest.*`。必须核对 XML 的失败和跳过计数。

模型调用：新请求使用 `clearCache()` 重置请求并保留前缀，再调用 `forward`；冷缓存基线还需 `clearPagedPrefixCache()`。更换模型权重、adapter 或位置编码语义时必须同时清除请求和前缀。固定模型实例内使用缓存，不跨模型混用。
