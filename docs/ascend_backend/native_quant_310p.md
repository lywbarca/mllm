# 310P 原生量化接入 W8A8
日期：2026-10-17。设备：Ascend 310P1；CANN 8.2.RC1，ATB 8.2.RC1.B150。

## 接口实测
- ATB ELEWISE_QUANT：CreateOperation 返回 1；日志明确仅支持 Atlas 800I A2。
- ATB ELEWISE_QUANT_PER_CHANNEL：支持。FP16 输入、FP16 scale、INT8 zero point，输出 INT8。
  初次误用 FP32 scale/offset 时 InferShape 返回 13；按实际支持类型修正后通过。
  scale 是除数而不是乘数，支持 [1] 广播和逐通道 scale、三维激活。
- ATB ELEWISE_DEQUANT_PER_CHANNEL：CreateOperation 返回 1；日志明确仅支持 Atlas 800I A2。
- ATB INT8 Linear 融合反量化：支持，INT8 x/weight + INT32 bias + UINT64 dequant scale -> FP16。
  已用 CPU INT32 累加与 scale 乘法核验，而不是仅检查返回码。
- ELEWISE_DYNAMIC_QUANT：另做创建/执行冒烟，成功；本次没有做其数值验收或切换动态量化。

## 已修改
静态 W8A8 decoder graph 的 Linear 插件现在执行：
FP16 -> ATB QUANT_PER_CHANNEL -> ATB INT8 Linear（融合反量化）-> FP16。
删除旧 MULS/Round/Clamp/Cast 组合、其执行器缓存及专属计时实现。
scale/zero point 在插件构造时分配并持有；计算继续处于 decoder graph 的同一 stream。
原始权重文件不变；加载时 activation scale 转为正有限 FP16，融合反量化使用同一有效 scale。
KV cache、attention 和输出 head 继续 FP16。既有 eager 动态量化路径不属于此次静态 graph 修改范围。
独立反量化算子不支持，因此没有插入额外反量化节点；Linear 原生融合反量化已经满足 W8A8 输出需求。

## 数值与回归
原生量化覆盖 1/17/256/512/1 行、标量及 128 通道 scale、三维输入、正负饱和和舍入边界。
精确的标量 1/32 scale 要求逐元素完全一致（包括 ties-to-even）。
非精确比例使用独立理想量化参考，要求 INT8 差值 <= 1 且相对截断后的理想缩放值误差 <= 0.625 个量化步长：
0.5 的整数舍入误差加 FP16 运算误差余量，不能解释为所有输入逐位相同。
融合 Linear 用独立原生量化结果作 INT8 输入，CPU INT32 累加再乘量化比例，
覆盖 ND/NZ 权重、两种 scale 和长度切换，FP16 输出误差阈值 0.005。
共 16 项测试通过、2 项仅适用 910B ND 的测试跳过；git diff --check 通过。
模型 18 token 短输入及 256/512 输入、64 输出均运行完成，日志内存池 Used 为 0，NPU 健康正常。

## 性能
同一 W8A8 权重、真实分页 FP16 KV、batch=1，每次一次 prefill warmup；
256/512 输入、64 输出（计时 63 个 decode step），每组独立运行三次，包含首次 decode 形状切换。
本次仅替换原生量化，没有移植 FP16 的 QKV/gate-up/SwiGLU 融合。

| 输入 | 手动量化 prefill ms（此前） | 原生量化 prefill ms | 手动量化 decode token/s（此前） | 原生量化 decode token/s |
|---|---:|---:|---:|---:|
| 256 | 34.43 | 28.60 | 47.37 | 60.46 |
| 512 | 61.04 | 50.04 | 49.16 | 64.04 |

256 decode 三次为 53.50/62.35/65.53；512 为 62.81/64.01/65.31 token/s，存在运行波动。
此前 FP16 为 77.58/75.53 token/s；本次原生 W8A8 仍未超过该结果。
历史手动量化和本次原生量化不在同一时刻交替运行，且输出 token 有差异；速度表是实测对照，不是严格同轨迹微基准。

## Profiling
对比相同位置的 50 个稳定 decode 间隔（跳过预填充和首次 decode）。
原生采集后段缺失一个 Gather 标记，选取的前 50 个区间不包含该区间；少量 kernel 记录仍有掉点，计数为约数。
每 token：1801.14 -> 1212.96，减少约 588（32.7%）。
- QuantPerChannelKernel：0 -> 195.82。
- Round / ClipByValueV2：各约 195.88 -> 0。
- MulsF16Kernel：223.94 -> 28（剩余为 attention 缩放）。
- Cast：196.90 -> 0.98（剩余非激活量化 Cast）。
- 同步约 11 次/token，稳定段 aclrtMalloc/Free 仍为 0。
INT8 Linear 和输出格式转换仍约 196/225 次/token，后续融合空间仍在。
op_summary 实测量化输出 INT8；INT8 MatMul 输出 FP16；PagedAttention 与 KV 写入保持 FP16。

## 模型质量限制
六条独立提示词仍有质量问题：例如自我介绍输出“我是小明，一个热爱学习和探索的普通学生”，
不能将本次接入成功称为量化质量合格。算术 17*23=391、巴黎、翻译等样例能回答。
与 FP16 使用同一 512-token 输入和相同后续历史，64 步 logits 有限；
top-1 一致 53/64，平均 KL 0.160661（此前手动量化 53/64、KL 0.217894）。
这不是准确率评测，也不能据少量样例断言整体质量改善；仍需校准和更全面质量验收。

## 文件
- before/：本次修改前源文件快照。
- source/：本次源文件。
- native-quant.patch：相对本次修改前快照的增量补丁。
- native-tests.log、regression.log、quality.log、w8a8-*.log：测试与运行记录。
- profile-comparison.json、benchmark-summary.json：结构化统计。
- probe.cpp、probe-*.log、channel-*.log：独立接口探测。
