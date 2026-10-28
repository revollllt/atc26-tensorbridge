# TensorBridge 性能优化架构设计

本文是 `perf` 分支上性能工作的总体设计：要优化哪几层、每层的关键决策、层与层之间的接口，以及验收方式。逐项的现状与证据见 [`OPTIMIZATION_CHECKLIST.md`](OPTIMIZATION_CHECKLIST.md)，约束见 `.specify/memory/constitution.md`，需求以 `specs/` 下的各个 spec 为准。

## 1. 目标与边界

- **两个设备目标。** H100 上在现有基础上继续提速；H20 上做一版调优，不靠 shape 表。
- **只做通用优化。** 配置只由 `(M, N, K)`、设备属性和按 GPU 拟合的代价模型常数决定（constitution 原则 I）。
- **基线。** 重点对比 humming（`humming-kernels` v0.1.18）的 MXFP4 权重 × FP8 激活模式，同时对比 cuBLAS FP8。humming 不支持 group 16 的权重配 8 位激活（`humming/schema/humming.py:382`），所以两者格式不同：TensorBridge 是 NVFP4（E4M3 scale、group 16），humming 是 MXFP4（UE8M0 scale、group 32），只比较相同 shape 上的延迟，精度单独比较。
- **不在本轮范围内。** Blackwell（sm_100）；非 Hopper 设备；新的量化格式（MXFP4 输入作为后续选项，见 W1）；`ae/` 和 `eval/` 的复现流程。

## 2. 现状画像

| 层 | 现状 | 主要问题 | 证据 |
| --- | --- | --- | --- |
| 反量化 | 寄存器内 add+mask FPMA bridge | 稳态 mainloop 每 K block 每线程约 213 条指令，约 94% 与 BM 无关；H20 每 SM 的权重到达速率是 H100 的 2 倍，decode 可能受限于发射 | checklist「当前 mainloop 的指令构成」 |
| 配置选择 | `sm90.hpp` 经验规则 + `sm90_tuned.hpp` + 26 个 shape 的 tuned JSON | 违反原则 I；worktree 里的代价模型约 700 个候选，靠按精确 shape 记忆化摊开销 | checklist E2、E4 |
| mainloop / 搬运 | lagged 调度只覆盖 5 种 BM；Stream-K slice 走保守调度；每个 tile 的环从 stage 0 重开；输出 tile 与环共用 smem | 中等 M 和 Stream-K 尾波不在最优调度上 | checklist B4、B5、C4、C8 |
| launch / host | 每次调用编码 3 个 TMA 描述符、设置一次 smem 属性；无 PDL；首次遇到配置时才 JIT；每个线性层 2 个 kernel（量化 + GEMM） | decode 的固定开销占比高 | checklist F2、F3、F4、D5 |
| 功能 | 只支持 dense | MoE 模型无法走本 kernel | checklist G1 |

## 3. 总体结构

```text
                     ┌──────────── W0 测量基座（贯穿各层） ────────────┐
Host   vLLM 插件 ──► fp8_nvfp4_gemm ──► 配置选择（W2）──► kernel 注册与预编译（W4）──► launcher（W4）
                                          │ 区间表：(N, K, 设备) → [M 区间 → config, kernel]
Device persistent scheduler（W2 / W5）
         ├─ producer：TMA / cp.async 加载 A、B、scale（W3）
         ├─ consumer：S2R ──► 反量化（W1）──► WGMMA（W3）
         └─ epilogue：scale、BF16、STSM、TMA store（W3）
MoE    分组调度与分组布局（W5），复用上面所有层
```

工作流与 spec 的对应：

| 工作流 | 内容 | spec | checklist |
| --- | --- | --- | --- |
| W0 | 测量基座与基线（含 humming MXFP4 × FP8） | [001](../specs/001-bench-baselines/spec.md) | H1–H4 |
| W1 | LUT 反量化 | [002](../specs/002-lut-dequant/spec.md) | A5、A6、A7 |
| W2 | 基于代价模型的 tile 与 Stream-K 选择 | [003](../specs/003-cost-model-selector/spec.md) | E2–E4 |
| W3 | mainloop 与数据搬运 | [004](../specs/004-mainloop-data-movement/spec.md) | B4、B5、B7–B9、C4、C6–C8 |
| W4 | launch 与 host 开销 | [005](../specs/005-launch-host-overhead/spec.md) | F2–F4、D5 |
| W5 | MoE grouped GEMM | [006](../specs/006-moe-grouped-gemm/spec.md) | G1 |

## 4. 各工作流设计

### W0 测量基座

所有后续结论都依赖它，必须最先完成。

- **单进程交替测量。** 一个入口在同一进程内构造 TensorBridge 与各基线，每轮交替顺序，每个 kernel 以 CUDA graph 重放若干次计时，取多轮中位数。现有 `ae/figure11/measure.py` 的流程可作参考，但新工具放在 `tools/bench/`，不改 `ae/`。
- **候选与基线同进程。** TensorBridge 的两个版本之间做 A/B 时，开发期的改动以编译期开关或配置字段的形式存在，两版在同一进程内各自 JIT；方案被接受后删除开关，不留死代码（原则 VI）。
- **基线。** humming v0.1.18 MXFP4 g32 × FP8（dense 与 MoE），cuBLAS FP8（per-token × per-channel）。可选：cuBLAS BF16、vLLM 的 CUTLASS W4A8 与 Marlin NVFP4A16。所有 kernel 都输入已量化的 FP8 激活，激活量化不计入时间。
- **L2 状态。** 默认让权重在总量超过 2 倍 L2 的多份副本间轮换，模拟逐层流式读取；另提供热 L2 模式。报告中注明所用模式。humming 自带的 `do_bench` 是冷 L2、无 CUDA graph，不能直接拿来比。
- **shape 集。** 论文 Figure 11 的 40 个、代价模型拟合集 410 个、held-out 695 个、decode 扫描（M = 1 到 128）、MoE 分组 shape。全部以 CSV 入库。
- **输出。** 每个 shape 的延迟、达到的 TFLOPS 与 HBM 带宽、占实测峰值的比例（峰值由基座自己测：大方阵 cuBLAS FP8 和设备内拷贝），每个 shape 集的几何平均与最差 shape，以及元数据：GPU、SM 数、驱动、torch、CUDA、TensorBridge 与 humming 的提交号。
- **剖析脚本。** 从 JIT 产生的 cubin 统计稳态循环的每 K block 指令数；ncu 指标集（issue active、tensor pipe 利用率、DRAM 吞吐）。
- **环境。** lab-H100 用 sbatch；H20-shared 的代码与环境放在 `~/atc26-ae`，计算节点离线，需在登录节点预取 humming wheel，并预填 TensorBridge 与 humming 的 JIT 缓存。

### W1 LUT 反量化

把 DeepGEMM 集成里的 LUT bridge（`integrations/deepgemm/deep_gemm/include/deep_gemm/common/fpma.cuh`）移植到 `include/tensorbridge/dequant/fpma.cuh`，替换现有的 add+mask 实现。

- **机制。** SNC 重编码后，一个预折叠 scale 字节 `s` 对应的 8 个正值字节是 `[s, 0, s+8, s+12, s+16, s+20, s+24, s+28]`，零值项直接在表里。每 4 个值：2 条 `IMAD` 建表（`s * 0x01010001 + 0x0c080000`、`s * 0x01010101 + 0x1c181410`），1 条 `PRMT` 查表，1 条 `LOP3` 插入符号位。不再需要零值掩码。
- **指令数（SASS，每个 k32 call、每线程 16 个权重）。**

  | bridge | 指令数 | 每权重 |
  | --- | --- | --- |
  | 现有 add+mask，scale 打包 | 40 | 2.50 |
  | 现有 add+mask，scale 预广播 | 35.25 | 2.20 |
  | DeepGEMM 集成的 LUT | 30 | 1.875 |
  | LUT，每字一次 `& 0x77777777` 加裸 `prmt` | 28 | 1.75 |

  CUDA 的 `__byte_perm` 只用选择子的低 3 位，编译器因此在每条 `PRMT` 前插一条 `LOP3 &0x7777`；选择子的第 3 位存的是符号，不能省掉掩码。改成每个字掩码一次、再用内联 PTX 的 `prmt.b32`，每个 call 少 2 条。移植时采用后者。
- **已有的整 kernel 证据。** DeepGEMM 集成的消融（`Tensorbridge-DeepGEMM/sbatch/ablation/README.md` Round 16）：离线分组的 LUT 比 add+mask 快 1.027 到 1.087 倍，整个 kernel 的 SASS 指令从 1176 条降到 1040 条；在线分组的 LUT 没有收益（0.977 到 1.009 倍）。
- **数值。** 对 SNC 重编码后的输入，输出字节与现有 bridge 逐比特相同：两者都是 `s + (m << 2)` 加符号；零值的符号在 SNC 中已被丢弃。现有的逐比特测试应在不改参考实现的前提下通过，另加 99 个合法 scale 字节 × 16 个编码的穷举对比。
- **离线布局。** 以 DeepGEMM 集成的 `reorder_nvfp4_dual_mma_preinterleaved` 为准：32 位字中第 i 个 nibble 的低 3 位，i = 0 到 3 时是 K-low 第 i 个值的量级，i = 4 到 7 时是 K-high 第 i − 4 个值的量级；第 3 位在 i 为偶数时是 K-low 第 i / 2 个值的符号，为奇数时是 K-high 第 (i − 1) / 2 个值的符号。符号位的位置与现有布局相同，只是量级位重新分组。双 MMA 配对（每线程 16 B 供两个 k32 call）、64 B swizzle 与 128×128 tile 的线程序都不变，改动集中在 `quant/nvfp4.py` 的 `to_thread_order`。
- **简化配置空间。** add+mask 特有的两个变体（`prebroadcast_scale` 与两种掩码）随之删除，W2 的候选集变小。
- **与 fork 负结果的关系。** fork 中的 LUT 方案（运行时建表 −33%、常量表 −10%、临时表 −12.3%）建表代价更高，或把表放在内存里；本方案每个 scale 只用 2 条 `IMAD` 在寄存器里建表，把零值折进表，并且离线完成 nibble 分组。checklist A6 据上面的 SASS 与 DeepGEMM Round 16 证据重开；AE kernel 上仍须在 H100 prefill、decode 与 H20 上重新做 A/B。
- **后续机会（不在本 spec 内）。**
  - **精确模式：** 目前只有偶数项（0.5、1、2、4）带 FPMA 的 `/6` 近似误差，奇数项（1.5、3、6）本来就精确。若离线在 scale 字节的第 7 位记一个修正位，建表时多扣一次，即可让全部表项等于 RNE E4M3，代价是每个 scale 多约 2 到 3 条整数指令。
  - **MXFP4 输入（A8）：** UE8M0 scale 的表项可以精确构造。

### W2 基于代价模型的 tile 与 Stream-K 选择

参考 DeepGEMM 的选择器结构，并把模型求值移出热路径。

- **三层结构。**
  1. **模型。** `candidates(desc)` 枚举合法配置，`estimate_us(desc, config, constants)` 给出闭式延迟估计，取最小者。以 worktree 中的统一调度器为起点：加性的反量化与 WGMMA 项、Little 定律的环深度、共享 L2 带宽、与设备调度器完全一致的 wave 与 Stream-K 切分。
  2. **区间表。** 某个 `(N, K)` 第一次注册时（权重准备阶段或 vLLM 加载权重时），对 M 从 1 到 `M_max` 求值一遍，压缩成「M 区间 → config」表，按 `(N, K, SM 数)` 存在内存里。表由模型现场推导，仓库中不存任何按 shape 的表。
  3. **在线查找。** 热路径只做一次对 M 的二分查找，返回 config 和预编译好的 kernel id（与 W4 的预编译共用一张表）；不在热路径上求值模型。
- **开销预算与现状。** worktree 中的统一调度器（登录节点实测）：一次冷求值 187 到 191 µs（132 SM）、139 到 145 µs（78 SM），每次平均约 314 个候选，Stream-K 候选约占 2/3 的耗时；记忆化后的热查找 0.08 到 0.11 µs。对一个 `(N, K)` 预计算 M = 1..8192 需要约 1.5 s，结果只有 50 到 71 个连续 M 区间（M ≤ 216 时区间宽 8，之后逐渐变宽）。目标：
  - 在线选择不超过 1 µs，热路径上不求值模型；
  - 每个 `(N, K)` 建表不超过 50 ms：只在 M 网格上求值（M ≤ 256 时步长 8，之后按几何增长的步长），并像 DeepGEMM 那样先用闭式下界剪掉明显劣势的候选，再精算；
  - 建表结果与逐 M 精确求值所选配置的预测延迟差在 1% 以内。
- **候选维度。** BM（8 到 256，取 8 的倍数，范围受 W3 调度泛化影响）、BN（W3 引入 64 之后为 {64, 128}）、级数（由 smem 容量决定上限）、Stream-K 开关、tile 光栅顺序。H20 相关的维度（每 SM 2 个 CTA、CTA 内 K 切分、参与的 SM 数）在 W3 证明有效后加入。
- **按设备拟合常数。** 常数集按 GPU 型号索引；SM 数、L2、smem 取自设备属性；其余常数用 `tools/sm90_cost_model`（plan、sweep、fit，从 worktree 移入）在 H100 和 H20 上各拟合一次。未拟合的型号退回到「属性推导项 + 最接近型号的常数」，并打印一次警告。
- **移除 shape 特化。** 删除 `sm90_tuned.hpp`、`tuned/*.json`、`sm90.hpp` 中的经验阈值；`autotune.py` 改为拟合流程的测量工具。
- **论文一致性。** 论文 Sec. 4.6 描述的是旧的 wave-fill 预测器，Figure 11 也是用旧选择器测的。AE tag `atc26-ae` 保留旧实现；本工作流的 spec 需注明这一差异。

### W3 mainloop 与数据搬运

每一项都以配置字段或编译期参数的形式出现，由 W2 决定何时启用；每一项都要过 W0 的 A/B。

- **P1：调度泛化。**
  - lagged 调度（4 个寄存器 buffer、`wait<3>`）推广到所有 BM，按 BM 校核寄存器预算，ptxas 报告不得有溢出（B4）。
  - Stream-K slice 改用同一套 lagged 调度（B5）。
- **P2：流水线状态与 tile 间重叠。**
  - 用「index + phase」的流水线状态替代逐级 phase 数组，环跨 tile 连续；级数按 smem 容量取上限，不再固定在 4 到 6（C4）。
  - 先测量下一个 tile 首级加载的暴露时间；若显著，在 epilogue 期间把下一个 tile 的首级预取到 L2，或在 smem 允许时给输出 tile 单独分配空间（C8）。
- **P3：H20 驱动的候选，以 W0 的 H20 剖析为前提。** BN = 64（DeepGEMM 集成已支持）、CTA 内 K 切分、每 SM 2 个 CTA 的小 tile 变体、decode 时权重加载的 L2 EVICT_FIRST 提示（B7–B9、C6）。
- **不重做的 ⊘ 项。** scale pair-pack、producer warp 反量化、ping-pong / cooperative 重排、B multicast、scale 改用 TMA。只有新的剖析证据才能重开。

### W4 launch 与 host 开销

- **先测量。** 分解 eager 模式下每次调用的 host 时间（Python 分派、配置选择、kernel 查找、描述符编码、属性设置、launch），以及 decode CUDA graph 中相邻 kernel 之间的 GPU 空隙。
- **描述符。** 权重的 TMA 描述符在权重准备时编码一次，随权重保存；激活与输出的描述符按指针与 M 缓存，或用替换地址的方式更新，不再每次完整编码。
- **一次性设置。** smem 属性在注册 kernel 时设置一次；SM 数在进程内缓存；配置查找与 launch 合并为一次 Python → C++ 调用。
- **PDL。** launch 时设置 programmatic stream serialization 属性；producer 在第一次读取依赖数据前执行 `griddepcontrol.wait`，mainloop 结束后执行 `launch_dependents`。
- **预编译。** W2 的区间表建好后，在加载阶段编译表中出现的全部 kernel；提供显式的 warmup 入口，供 vLLM 在捕获 CUDA graph 前调用。
- **减少每个线性层的 kernel 数。** 把 per-token FP8 量化融进前一个算子（vLLM 的 RMSNorm + 量化、SiLU-mul + 量化融合），使每个线性层只剩 GEMM 一个 kernel。

### W5 MoE grouped GEMM

在 AE kernel 上扩展分组能力，不直接搬 DeepGEMM 集成的 kernel：两者的权重重排不兼容（DeepGEMM 集成用 k-block 优先的 64 通道原子、无 swizzle 的 256 B 行；AE 用 128×128 tile 的线程序、64 B swizzle），而 W1–W4 的优化都落在 AE kernel 上。DeepGEMM 集成的测试用例（与非分组 kernel 逐比特一致、CUDA graph 重放、极小 expected M、每专家 global scale）直接作为验收场景。

- **目标模型。** Qwen3.6-35B-A3B：hidden 2048，专家中间维 512，256 个专家，top-8，40 层。每个专家的两个 GEMM 为 gate_up（N = 1024，K = 2048）与 down（N = 2048，K = 512）。TP = 2、4 时 N、K 仍是 128 的倍数；TP = 8 时 down 的 K = 64，超出当前对齐要求，不在本轮范围内（除非 checklist G2 先完成）。
- **两种布局。**
  - contiguous（prefill）：各专家的 token 按专家拼接，每段补齐到 BM 的倍数；调度器读每个 M 块的专家 id，跳过补齐块（DeepGEMM 集成对补齐块仍加载全部 K block 并写零，这里不再这样做）。
  - masked（decode，可被 CUDA graph 捕获）：每个专家固定容量 `Mcap`，按 `masked_m[e]` 生成 tile，空专家不产生 tile；要求 `Mcap` 是 BM 的倍数。
- **kernel 改动。**
  - 调度器按分组类型模板化，对外给出当前 tile 的专家 id。
  - producer 按专家偏移权重行（`e * N + n`）、scale（`e * K / 16 * N`）和 masked 布局下的激活行（`e * Mcap + m`）。
  - epilogue 按专家读 global scale，并按该专家的行数约束写回。
  - 第一版分组模式不开 Stream-K 与 multicast：tile 数依赖数据，现有的 Stream-K 切分与锁编号都假设 tile 数在编译期可知。
- **权重准备。** `prepare` 接受 `[E, N, K/2]` 的打包权重与 `[E, N, K/16]` 的 scale，输出 `[E·N, K/8]` 的线程序权重、`[E, K/16, N]` 的预折叠 scale 和 `[E]` 的 global scale（乘 6）；scale 字节的合法性按专家检查。gate 与 up 的 global scale 在该 checkpoint 中逐专家相等，可合并为每专家一个值；若不相等，则依赖 checklist D4 的每通道 scale。
- **配置选择。** W2 的模型以「每个专家的期望行数」代替 M 选择 BM，并考虑补齐浪费；区间表按 `(N, K, E, 分组类型)` 建立。
- **vLLM 接入（0.20.2 fork）。**
  - `TensorBridgeConfig` 增加对 `FusedMoE` 层的 `get_quant_method`：当前 compressed-tensors 的 MoE 分派遇到「NVFP4 权重 + FP8 激活」会抛出 ValueError。
  - 新增 MoE method（create_weights、process_weights_after_loading、量化配置、apply），以及一个 contiguous 格式的 experts 类，复用单卡的 NoDPEP prepare/finalize 与 `FusedMoEKernel`；decode 的 masked experts 类作为第二步。
  - GEMM1 的输入用 vLLM 已有的 per-token FP8 量化；GEMM2 的输入需要一个「SiLU-mul + per-token FP8 量化」融合 kernel（vLLM 现有的是 per-128 分组版本）。
  - 参考实现：服务器上的 `tensorbridge-vllm-plugin`（面向 vLLM 0.26，基于 DeepGEMM 集成）已有 MoE method 与 contiguous / masked 两个 experts 类。
  - 该 checkpoint 的架构名 `Qwen3_5MoeForCausalLM` 未在 fork 中注册，需要 `hf_overrides` 或补注册。
- **基线。** humming v0.1.18 的 MoE MXFP4 × FP8（GROUPED_CONTIGUOUS 与 INDEXED）、vLLM 默认的 Marlin NVFP4A16 MoE、DeepGEMM FP8 分组 GEMM；DeepGEMM 集成里的 TensorBridge 分组 kernel 作为内部参照。

## 5. 跨工作流接口

| 接口 | 现状 | 演进 |
| --- | --- | --- |
| `GemmConfig`（C++ `config.hpp` 与 Python `jit/codegen.py` 字段一一对应） | 9 个字段 | W1 删除 `prebroadcast_scale`；W3 增加 BN、每 SM CTA 数、K 切分等字段；W5 增加分组类型 |
| 权重准备产物 `NVFP4Weight` | 线程序权重、预折叠 scale、global scale、locks | W1 改为 LUT 布局；W4 增加缓存的权重描述符；W5 增加专家维度与每专家 global scale |
| 配置选择 | 每次调用经 `runtime.select_config` 进入 C++ | W2 改为按 `(N, K, SM 数)` 的区间表；W4 让查找与 launch 合并为一次调用 |
| kernel 注册 | `(N, K, config)` → kernel id，首次使用时编译 | W4 在建表后预编译，并提供 warmup 入口 |

## 6. 验收

- 每项改动都按 constitution 的门禁验收：`pytest tests`；H100 与 H20 的同进程 A/B，含 humming MXFP4 × FP8 与 cuBLAS FP8 基线；改变数值时的 PPL 抽查；checklist 与 `docs/DESIGN.md` 同步；按原则 VI 自查。
- 各 spec 的成功标准给出具体阈值；未达标的方案按原则 III 把负结果写回 checklist。

## 7. 顺序与里程碑

| 里程碑 | 内容 | 依赖 |
| --- | --- | --- |
| M0 | W0：测量基座、humming 与 cuBLAS 基线、H100 与 H20 的现状报告与剖析 | 无 |
| M1 | W1 LUT 反量化；W4 中不依赖 W2 的部分（描述符缓存、一次性设置、PDL） | M0 |
| M2 | W3 P1 调度泛化；W2 选择器合入、区间表、H100 与 H20 常数拟合 | M1（W1 改变了反量化成本，拟合要在它之后） |
| M3 | W3 P2、P3 与重新拟合；W4 预编译与量化融合 | M2 |
| M4 | W5 MoE grouped GEMM | M2；复用 W1–W4 的成果 |

## 8. 风险

- **LUT bridge 的收益未经 A/B 验证。** fork 中的 LUT 方案都是负收益。若 W1 在 H100 prefill 上回退，需要按设备与 M 区间由 W2 选择 bridge，或放弃。
- **humming 的 MXFP4 g32 × per-token FP8 组合在其上游未见测试。** W0 需要先做正确性检查。
- **H20 计算节点离线。** 依赖、wheel 和 JIT 缓存都要提前准备。
- **Stream-K 的 BF16 归约不确定。** 一旦改动归约顺序，需要同步 `docs/DESIGN.md` 的「Scope」。
- **论文一致性。** 选择器和 mainloop 的改动会使论文 Sec. 4.6 与 Figure 11 不再对应 `perf` 分支的状态；复现以 `atc26-ae` 为准。
