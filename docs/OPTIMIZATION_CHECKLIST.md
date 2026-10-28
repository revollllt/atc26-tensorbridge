# TensorBridge 优化 checklist（H100 / H20）

本文列出 TensorBridge kernel 可做的 GEMM 优化，标出哪些已经实现、哪些还没有、哪些已经试过并证伪，作为后续用 spec-kit 拆分 feature 的输入。范围是 `tensorbridge/` 下的 NVFP4 权重 × FP8 激活 GEMM 及其 vLLM 接入，不含 `eval/` 和 `ae/` 的复现流程。

依据：

- 代码：GitHub `main`（`dd45d51`；kernel 部分与 `8c1d9e9` 相同），以及 lab-H100 上未合入的 `unified-scheduler` worktree（`tensorbridge-ae-sched`）。
- 论文：`Tensorbridge-AE-paper/sec/04-Design.tex`（Sec. 4 的五个机制）。
- humming 上游：`vllm-project/humming`（原 `inclusionAI/humming`，`de505d2`，2026-10-28）的 sm90 FP8 激活 × MXFP4 权重路径，主要是 `kernel/humming_ws.cuh`、`datatype/dequant_fused.cuh`、`tune/sm90*.py`。
- humming fork：`revollllt/humming` 中 2026 年 5 月到 6 月 NVFP4 W4A8 开发期的实验记录（`docs/changes/`）。TensorBridge 由这一版重构而来。
- SASS：在 lab-H100 上用 CUDA 12.8 `nvcc` 和 JIT 相同的 ptxas 选项编译 4 个配置（BM = 16、128、176、256，data parallel），统计稳态 mainloop 循环体的指令构成。JIT 实际走 NVRTC，指令数可能略有出入。
- 其他参考：CUTLASS、DeepGEMM、LiquidGEMM、Machete。

## 准入原则

只收录对任意 `(M, N, K)` 和任意 Hopper 设备都成立的通用优化。以下做法不再新增，已有的逐步移除：

- 按精确 shape 匹配的规则或表，例如 `csrc/heuristics/sm90_tuned.hpp`、`tuned/NVIDIA_H100_80GB_HBM3.json`。
- 写死 H100 数值的经验阈值，例如 `sm90.hpp` 里的 `desc.m == 256 && desc.n > 8192` 和 `kMaxReducedElements = 1500000`。
- 按设备名分支的手写规则，例如 humming 的 `Sm90H20Heuristics`。

设备差异只允许通过两种方式进入配置选择：设备属性（SM 数、L2、smem），以及对每种 GPU 拟合一次的代价模型常数。

## 状态图例

| 标记 | 含义 |
| --- | --- |
| ✅ | 已在 `main` 实现 |
| ◐ | 部分实现，或在分支 / worktree 中未合入 |
| ☐ | 未实现，未见实测 |
| ⊘ | 在 fork 中实测为负收益或无收益，不再投入；除非新场景下的 profile 证明它在关键路径上 |
| ✗ | 违反准入原则，应移除 |

⊘ 的适用范围：fork 的实验几乎都在 H100 上、以 M = 2048、N = 18944、K = 3584 和 N = K = 8192 这类计算瓶颈的 prefill shape 做同节点 ABBA。decode 和 H20 基本没有测过，所以 ⊘ 项在 H20 decode 上可以重新评估，但必须先有 ncu 证据。

fork 的最终状态（2026-05-30）：在 M = 2048、N = 18944、K = 3584 上为 219.9 µs，CUTLASS W4A8 为 213.3 µs，差距 3.1%；两者都只达到 FP8 峰值的约 65% 到 70%，瓶颈被归为 Tensor Core 供数与控制开销，而不是 HBM。M ≤ 128 的 decode 上比 CUTLASS 快 3% 到 22%。

## H100 与 H20 的差异

| 指标 | H100 SXM | H20 | H20 / H100 |
| --- | --- | --- | --- |
| SM 数 | 132 | 78（实测） | 0.59 |
| L2 | 50 MB | 60 MB（实测） | 1.2 |
| 每 block 可用 smem | 227 KB | 227 KB（实测，232448 B） | 1 |
| FP8 dense 峰值 | 1979 TFLOPS | 296 TFLOPS〔规格值，待实测〕 | 0.15 |
| HBM 带宽 | 3.35 TB/s | 4.0 TB/s〔规格值，待实测〕 | 1.19 |
| 每 SM FP8 算力 | 15.0 TFLOPS | 3.8 TFLOPS | 0.25 |
| 每 SM HBM 带宽 | 25.4 GB/s | 51.3 GB/s | 2.0 |
| FP8 ridge point | 591 FLOP/B | 74 FLOP/B | 0.13 |

由此推出三点，均需在 H20 上实测确认：

1. **计算瓶颈区间大幅提前。** NVFP4 权重连同 E4M3 scale 约 0.5625 B/权重，GEMM 的算术强度约为 3.56·M FLOP/B（忽略激活和输出字节）。H100 上约 M ≳ 166 进入计算瓶颈，H20 上约 M ≳ 21。FP8 权重（1 B/权重）的对应阈值是 H100 约 296、H20 约 37。因此在 H20 上，W4 相对 FP8 的理论收益基本只存在于 M ≲ 37 的 decode 区间；更大的 M 上，TensorBridge 只能靠 Tensor Core 利用率追平 cuBLAS FP8。
2. **decode 在 H20 上可能受限于指令发射，而不是 HBM。** 每个 SM 每秒到达的权重是 H100 的 2 倍，但 CUDA core 的发射能力不变。按 1.98 GHz〔H20 实际 SM 时钟待确认〕估算，跑满 HBM 时每个权重只能分到约 2.8 条线程指令（H100 约 5.6 条）。SASS 显示当前稳态 mainloop 约 3.3 条/权重（见下节），即使每周期发满 4 条 warp 指令，H20 decode 也只能达到 HBM 带宽的约 85%。fork 测得 SNC（零标记掩码）的运行时开销在 H100 上 M ≤ 128 时约为 0、M = 2048 时约 +12 µs，与「H100 decode 有发射余量」一致；H20 的余量只有 H100 的一半。
3. **大 M 时反量化更容易藏住，瓶颈转向 WGMMA 利用率和 tile 间空泡。** BM = 256 时，一个 K block 的 WGMMA 在 H20 上约 2.2 µs，而 H100 拟合出的反量化固定开销约 0.34 µs/K block；即使两者完全串行，反量化也只占总时间的约 13%。此时 tile 间的空泡（C8）和 78 SM 下的 wave 量化（E 组）更重要。

## 当前 mainloop 的指令构成

BM = 128、4 级、data parallel、N = K = 8192 的稳态循环体（4 个 K block、16 条 `QGMMA`）共 851 条指令，折合每个 K block 每个 consumer 线程约 213 条，每线程 64 个权重，约 3.3 条/权重。BM = 16（6 级）为 205 条，BM = 256（预广播 scale）为 215 条，BM = 176（2-buffer）为 194 条。

| 每 K block 每线程 | 条数 | 说明 |
| --- | --- | --- |
| `LOP3.LUT` | 67 | FPMA 的拆半、符号合并、掩码与运算，以及 scale 拼字 |
| `IMAD`（含 `IMAD.SHL`） | 约 36 | 左移、把 scale 字节复制到 4 个 lane |
| `PRMT` | 19 | 主要是零标记掩码 |
| `VIADD`、`SHF` | 28 | scale 加法、掩码中的加法、右移 |
| `LDS.U8` | 16 | weight scale，每个 scale 1 次 |
| `LDS.128` | 2 | 权重，每次读两个 k32 call |
| `QGMMA` + `WARPGROUP.ARRIVE` + `WARPGROUP.DEPBAR` | 12 | 4 个 k32 call |
| 其余（`UMOV`、`ISETP`、`BRA`、barrier 等） | 约 33 | — |

每个 K block 的指令数几乎不随 BM 变化，其中只有 12 条用于发射 WGMMA，其余约 94% 是反量化、scale、S2R 与控制开销。BM 越小，这部分固定开销摊到每个 token 上越多。在 H100 prefill 上这部分大多藏在 WGMMA 的空隙里（fork 的 SASS 对比：指令总数比 CUTLASS W4A8 多约 32%，记录的结论是多出的指令藏在空隙里），在 H20 decode 上则可能直接决定性能。

## A. 权重反量化（FPMA bridge）与 S2R

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| A1 | 寄存器内 FPMA bridge，只用整数指令 | ✅ | `dequant/fpma.cuh`：编码域加法 + 零标记掩码 | 全部 | 论文 Sec. 4.2 |
| A2 | 离线 scale prefold 与 SNC 重编码 | ✅ | `quant/nvfp4.py` `prepare`、`recode_snc` | 全部 | 论文 Sec. 4.2 |
| A3 | 双 MMA 预交织权重布局，一次 16 B LDS 供两个 k32 call | ✅ | `memory/s2r.cuh` `load_weights_pair`；`to_thread_order` | 全部 | 论文 Sec. 4.5；LiquidGEMM Dual-MMA |
| A4 | 掩码两种实现（算术 / PRMT LUT）与 scale 预广播 | ✅ | `kPrebroadcastScale` 开关，由启发式选择 | BM = 256 大 K | — |
| A5 | weight scale 按线程序离线重排 | ⊘ | AE 中每个 K block 16 次 `LDS.U8`，而权重只要 2 次 `LDS.128`；每线程每个 K block 恰好用到 16 个 scale 字节，可降到 1 次 `LDS.128`。fork 的离线 pair-pack（O1 到 O5）去掉了全部 scale 窄读，ABBA 反而慢 1.1% 和 1.5%，结论是 scale S2R 不在 prefill 的关键路径上。仅当 H20 decode 被证实受限于发射时重测 | H20 decode | humming 上游：1 次 32 位 LDS 取齐一个线程的 4 个 scale（`loader_bs.cuh` `load_layout1`） |
| A6 | bridge 指令数压缩 | ⊘ | `dequant_word` 约 17 条 / 8 个权重，零标记掩码约占 8 条。fork 试过的路线都失败：运行时建 LUT −33%，常量表 −10%，临时表 −12.3%，LUT 方案指令数反而从 21 增到 28；砍掉 32 条 ALU 的 SNC 变体收益为 0 µs。humming 上游的 per-scale 寄存器 LUT（2 条 `IMAD` 建表，零值项不加 scale，因而无需掩码）适用于 MXFP4 group 32；NVFP4 group 16 下一个 k32 call 内每个 scale 只覆盖本线程 4 个权重，建表开销摊薄一半，这与 fork 的负结果一致。同 A5，只在 H20 decode 上有条件地重开 | H20 decode | humming `datatype/dequant_fused.cuh` |
| A7 | 可选 ULP 精确模式的开销 | ◐ | `tb-ulp-work` 原型在 scale 字节 0x23 到 0x7E 内逐比特等于 RNE E4M3，但 40 个 Figure 11 shape 几何平均慢约 18% | 需要精确数值的部署 | — |
| A8 | MXFP4（E2M1 + UE8M0，group 32）输入 | ☐ | 只支持 NVFP4（E4M3 scale，group 16）。UE8M0 是纯指数，加到 FP8 指数域上是精确的，不存在 FPMA 的尾数误差；代价是 E4M3 只容纳约 11 个 binade，过小 group 的值需离线重新取整 | 兼容 MXFP4 checkpoint（如 gpt-oss） | humming 上游：scale 钳到 11 个 binade，过小的 group 离线重新取整（`process_mxfp4.cuh`），2^6 的偏置差放到 epilogue |

## B. MMA 与 mainloop 调度

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| B1 | swap-AB：Y^T = W X^T，权重作 RS WGMMA 的寄存器操作数，token 维映射到 n | ✅ | `common/tile.cuh`；BM 取 8 到 256 中 8 的倍数 | 全部 | 论文 Sec. 4.3；humming 上游同样做法 |
| B2 | producer / consumer warp specialization 与 `setmaxnreg` | ✅ | 1 个 producer WG（40 寄存器）+ 2 个 consumer WG（232 寄存器） | 全部 | CUTLASS、DeepGEMM |
| B3 | issue-first lagged 调度，4 个寄存器 buffer，`wait<3>` | ✅ | 只覆盖 BM ∈ {16, 32, 64, 128, 256}。fork 中与 CUTLASS 对齐后定型（issue → `wait<3>` → S2R(k+2) → dequant(k+1)）。humming 上游没有这一层：反量化在 `wait_group 0` 之后执行 | 这 5 种 BM | 论文 Sec. 4.4 |
| B4 | lagged 调度推广到所有 BM | ☐ | 其余 BM（包括默认上限 176）走 2-buffer、每个 k32 call 后 `wait<0>` 的串行路径；`kH100` 拟合出的每 token WGMMA 开销 2-buffer 为 0.001563 µs/K block，lagged 为 0.001105（多约 41%）。BM = 176 只有 88 个累加器，少于 BM = 256 的 128 个，寄存器上应放得下；限制原因未记录，需先验证 | H100 中等 M；H20 全部 | — |
| B5 | Stream-K slice 也用 lagged 调度 | ☐ | slice 走保守调度（逐个 `wait` 后一起发 4 个 WGMMA）。tile 数少于 SM 数时全部工作都走这条路径，H20 只有 78 SM，触发更频繁 | decode、小 N | — |
| B6 | 用重排 consumer 结构来重叠反量化与 WGMMA | ⊘ | 代价模型以 `dequant + wgmma × BM` 相加的形式拟合，BM = 128 时固定项（0.34 µs）大于 WGMMA 项（0.14 µs）；但 fork 在新内核上把反量化整体打桩只快约 2%（旧内核上的同类探针快 35%，两组结果未调和），说明大 BM 下它多半已被藏住。fork 试过的结构全部失败：累加器 ping-pong 超出寄存器预算约 24 个；按 M 切分的 cooperative 调度 −42%；time stagger 因栈增长被否决；k3 ping-pong 源 buffer −13% 到 −20%（ptxas 把 WGMMA 串行化） | — | fork `26-05-10_00-46`、`26-05-12_15-30` |
| B7 | 更多 CTA tile 形状：BN ∈ {64, 256}，BK ∈ {64, 256} | ☐ | BN = BK = 128 写死在 `Tile` 中。BN = 64 不靠 Stream-K 也能增加 tile 数；BK = 256 让每次 barrier 握手搬更多字节 | decode、小 N、78 SM | humming 上游枚举 block_n ∈ {128, 256} |
| B8 | CTA 内 K 切分（多组 consumer 各算一段 K，smem 归约） | ☐ | decode 时只有 2 个 consumer WG，且按通道划分；K 切分可在不做全局归约的前提下增加在途 warp | decode、小 N | humming 上游：H100 上 N ≤ 4096 的 decode 用 4 路 K-warp（`epilogue/smem_reducer.cuh`） |
| B9 | 每 SM 2 个 CTA | ☐ | `__launch_bounds__(384, 1)`，grid 固定为 SM 数。2 个 CTA 可让一个 CTA 的反量化、epilogue 与另一个的 WGMMA 自然重叠，不必改 consumer 结构（与 B6 的失败路线不同）；代价是每线程寄存器减半，只适合小 tile | H20（Tensor Core 慢，重叠更值钱） | humming 上游 H20 默认配置：(128, 128, 128) tile、每 SM 2 个 CTA |
| B10 | 极小 M 的 CUDA core GEMV 路径 | ☐ | M ≤ 8 时 Tensor Core 不是瓶颈，瓶颈是每权重的转换指令；CUDA core 直接对 BF16 激活做 FMA 可省去激活量化和 FPMA 掩码。作为一种 backend 由模型选择，属于探索项 | H20 decode M = 1 到 4 | Marlin 类 W4A16 GEMV |
| B11 | FP32 两级累加（每若干 K block 提升一次） | ☐ | 全 K 直接累加在 WGMMA 累加器中；Hopper FP8 Tensor Core 累加精度有限，K = 28672 等大 K 存在误差风险 | 数值，大 K | DeepGEMM promotion |
| B12 | producer warp 做反量化，FP8 权重写回 smem，consumer 用 SS WGMMA | ⊘ | fork f0.x 系列：功能与逐字节校验全部通过，但同时运行时 producer 的服务时间是 consumer 每级预算的 1.64 倍；无端到端数据，已回退到 consumer 内反量化。论文 Sec. 4.4 也论证了不采用这种结构 | — | fork `26-05-14_20-55` |

## C. 数据搬运（G2S、smem、L2）

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| C1 | A、B 用 TMA 加载，TMA 描述符预取，L2 128B promotion | ✅ | `memory/g2s.cuh`；A 128B swizzle，B 64B swizzle；`launcher/tma.h` 设 `L2_PROMOTION_L2_128B`（fork 实测约 +0.5 µs） | 全部 | — |
| C2 | mbarrier 环形缓冲，`try_wait` 提前探测下一级 | ✅ | `Consumer::probe_stage`（fork 实测约 +1%） | 全部 | — |
| C3 | weight scale 或权重改用其他 TMA 形式加载 | ⊘ | 现为 64 个 producer 线程各发一次 16 B `cp.async`。fork：scale 改 TMA −2.0% / −0.2%；权重改无 swizzle 的 3D TMA −2.3% / −0.1% | — | fork `26-05-26_03-45`、`03-33` |
| C4 | 用「index + phase」流水线状态替代逐级 phase 数组，环跨 tile 连续，按 smem 容量与 Little 定律定级数 | ☐ | unified scheduler 把级数限制在 4 到 6，原因是更深的环会让逐级状态溢出到 local memory；每个 tile 都从 stage 0 重新开始。BM = 16 时每级约 11 KB，6 级只用约 66 KB smem。fork 在 prefill 上 5 级与 4 级持平或慢约 5%，所以此项只针对 decode / H20 | decode，H20 尤甚 | CUTLASS `PipelineState`；humming 上游 continuous ring |
| C5 | 线程块集群 multicast | ⊘ | 只支持 B multicast、集群大小 2、仅 data parallel；Stream-K 与 multicast 组合有已知的 stale smem 问题（编译期禁止）。fork：B multicast −6.1% / −4.3%，改成 CUTLASS 的双 barrier 后仍是回退，CUTLASS 自己的 2×1×1 集群也略慢于 1×1×1；unified scheduler 消融后不再选。H20 未测 | — | fork `26-05-26_04-00`、`26-05-27` |
| C6 | TMA 加载的 L2 cache hint | ☐ | `tma_load_2d` 未带 `.L2::cache_hint`。fork 没有做过实验，prefill 上 L2 读命中率 92%，收益可能有限；decode 时权重只读一次，可试 EVICT_FIRST | decode | CUTLASS `CacheHintSm90`；humming 上游有 `evict_first` 开关但默认关闭 |
| C7 | 面向 L2 的光栅化 | ◐ | 只有 N-fast 和 M-fast 两种顺序。fork 中 M-fast 把 DRAM 读从 297 MB 降到 47 MB 并采纳；CUTLASS 式 swizzle 估计收益 0.5 到 1 µs，未实现。unified scheduler 在 (8192, 32768, 512) 上因 tile 顺序选择落后旧规则约 14%（已知 miss） | 大 shape prefill | humming 上游：权重超过 0.7 倍 L2 时启用 group-M 光栅化（`tune/raster.py`） |
| C8 | epilogue 与下一个 tile 的 prologue 重叠 | ☐ | AE 中输出 tile `d` 与 stage 环共用 smem（`storage.cuh` 的 union），producer 必须 `wait_tile_stored()` 后才能加载下一个 tile。fork 用 clock64 测得 epilogue 约 1.0 µs/tile，估计重叠只能省约 0.8 µs/kernel 而未做；记录没有说明当时 producer 是否也要等 epilogue 写完，AE 中下一 tile 首级加载的暴露时间需重新测量。低成本做法是在 epilogue 期间用 `cp.async.bulk.prefetch.tensor` 把下一 tile 的首级预取到 L2 | 多 tile/CTA、小 K 的 prefill | humming 上游的 epilogue 期间 L2 预取 |

## D. Epilogue 与算子融合

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| D1 | `stmatrix.x2.trans` 转置写回 + TMA store | ✅ | BM 为 32 的倍数时使用，其余 BM 走标量写回。fork 实测 −12 µs；关键是 128B swizzle，无 swizzle 的 STSM 反而 +4.6 µs；`.x4` 触发 ptxas 寄存器分配缺陷 | 全部 | — |
| D2 | per-token scale 与 global scale 在 epilogue 合并 | ✅ | `r2s.cuh` | 全部 | — |
| D3 | bias 融合 | ☐ | vLLM 插件中单独计算 `output + bias` | 带 bias 的模型 | humming 上游在 epilogue 中加 bias |
| D4 | 每输出通道的 scale 向量 | ☐ | 插件要求融合线性层（QKV、gate/up）共享一个 global scale，否则直接报错；改为每通道 scale 后可兼容更多 checkpoint | checkpoint 兼容性 | — |
| D5 | 激活量化融合 | ☐ | 每次 GEMM 前单独启动一个 Triton 量化 kernel（`quant/fp8.py`）；可融入前一个 RMSNorm / SiLU-mul（vLLM fusion pass），或让 kernel 直接读 BF16 激活、在 producer 中量化 | decode 端到端 | humming 上游把激活函数与量化融合在单独的 `process_input` kernel 中 |
| D6 | 输出侧融合：SiLU-mul、FP8 输出 | ☐ | 只输出 BF16 | gate/up 投影 | CUTLASS EVT |
| D7 | Stream-K 部分和用 FP32 workspace | ☐ | 现在每个 slice 先舍入到 BF16 再用 TMA reduce-add 累加；切成 3 片以上时结果不可逐比特复现（相差 1 到 2 个 BF16 ULP，见 `docs/DESIGN.md`） | 数值与可复现性 | CUTLASS、DeepGEMM；humming 上游同样以 16 位精度归约 |

## E. 任务调度与配置选择

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| E1 | persistent kernel（grid = SM 数），data parallel + Stream-K 尾波 | ✅ | `scheduler/gemm.cuh` | 全部 | 论文 Sec. 4.6 |
| E2 | 基于代价模型的统一配置选择 | ◐ | `unified-scheduler` worktree，未提交：对全部合法配置用延迟模型打分并取最小；1105 个 shape 上几何平均比旧启发式快 7.0%（held-out 695 个快 9.7%），论文 Figure 11 脚本上持平（0.9985）；已删除 `sm90_tuned.hpp`。待决定是否合入，以及论文 Sec. 4.6 的描述是否同步 | 全部 | DeepGEMM、FlashInfer 启发式 |
| E3 | 代价模型按设备参数化，并扩展候选维度 | ☐ | 常数 `kH100` 写死。应按设备选择常数，SM 数、L2、smem 取自设备属性，其余常数用 `tools/sm90_cost_model`（sweep + fit）对每种 GPU 拟合一次。候选集需随 B7、B8、B9 扩展，并允许参与的 SM 数小于总数 | H20 调优的主路径 | humming 上游可缩小参与的 SM 数 |
| E4 | 移除按 shape 的规则与表 | ✗ | `main` 上仍有 `sm90_tuned.hpp`、`sm90.hpp` 中的经验阈值，以及 26 个精确 shape 的 `tuned/NVIDIA_H100_80GB_HBM3.json`（`runtime.select_config` 优先读取）。`autotune.py` 只保留为拟合代价模型用的测量工具 | 可维护性 | — |

## F. Launch 与 host 开销

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| F1 | driver API 在当前 stream 上启动，可被 CUDA graph 捕获；按 `(N, K, config)` JIT 并缓存 cubin | ✅ | `csrc/launcher`、`jit/` | 全部 | — |
| F2 | PDL（programmatic dependent launch） | ☐ | 未设置 launch 属性，也没有 `griddepcontrol`。代价模型拟合出每个 kernel 约 2 µs 的固定启动与准备开销（`launch_us`），decode 下单个 GEMM 只有几 µs。fork 加过头文件和测试脚本，但从未做过 A/B | decode | humming 上游：producer 在首次加载前 `griddepcontrol.wait`，launcher 设置 PDL 属性，但默认关闭 |
| F3 | 每次 launch 的 host 开销 | ☐ | 每次调用都编码 3 个 TMA 描述符并调用 `cuFuncSetAttribute`；B 的描述符在权重准备时即可确定，smem 属性可在注册 kernel 时设置一次 | eager 模式的 prefill | — |
| F4 | 加载时按 M 区间预编译 | ☐ | 首次遇到某个 `(N, K, config)` 时才 JIT 编译，服务中首个请求会撞上编译延迟 | 服务冷启动 | humming 上游在 `layer.transform()` 时编译整张 M 区间表 |

## G. 功能覆盖

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| G1 | MoE grouped GEMM（contiguous / masked 布局） | ☐ | 只支持 dense；论文中 MoE shape 按 dense GEMM 测量，MoE 模型的实际量化推理不走本 kernel | MoE 服务 | humming 上游在同一套 kernel 中支持 INDEXED（`sorted_ids`）、GROUPED_CONTIGUOUS、GROUPED_MASKED；DeepGEMM |
| G2 | N、K 不是 128 倍数时的尾部处理 | ☐ | 静态断言要求 128 对齐 | 少数 TP 切分 | — |
| G3 | FP16 输出 | ☐ | 只输出 BF16 | FP16 模型 | — |

## H. 测量与回归基础设施

| ID | 优化项 | 状态 | 现状与证据 | 主要收益场景 | 参考 |
| --- | --- | --- | --- | --- | --- |
| H1 | 同进程交替 A/B 测量与固定 shape 集 | ◐ | 在 worktree 的 `tools/sm90_cost_model`：410 个拟合 shape、695 个 held-out shape、论文 40 个 shape；大 shape 跨节点波动 6% 到 15%，只认同进程交替结果 | 所有性能结论 | fork 的同节点 ABBA 流程 |
| H2 | 每个 shape 的 roofline 效率报告（占 FP8 峰值、占 HBM 带宽的比例） | ☐ | 无 | 判断剩余空间 | — |
| H3 | SASS / ncu 审计脚本（每权重指令数、溢出、tensor pipe 利用率、issue 利用率） | ◐ | JIT 已用 `--ptxas-options=-v` 记录寄存器与溢出；本文的 SASS 统计是一次性脚本，未入库 | 判断 ⊘ 项能否重开 | — |
| H4 | 性能回归门禁 | ☐ | `tests/` 只检查正确性 | 长期维护 | — |

## H20 调优步骤

H20 上不建立 `tuned/NVIDIA_H20.json` 一类的 shape 表。runtime 按设备名查表，H20 没有表时直接使用模型选择。

1. **基线。** 在 H20 上运行 `pytest tests` 和 Figure 11 的 40 个 shape，同时测 cuBLAS BF16 / FP8、vLLM CUTLASS W4A8、NVFP4A16 Marlin。
2. **定位瓶颈。** 对 M = 16、128、512、4096 各取代表 shape，用 ncu 确认上文三点推论：decode 是否受限于发射（issue 利用率、达到的 HBM 带宽），大 M 的 tensor pipe 利用率是多少。
3. **拟合常数。** 合入 E2，完成 E3，用 `tools/sm90_cost_model` 在 H20 上 sweep 并拟合 `kH20`。
4. **按瓶颈做通用优化。**
   - decode：F2（从未测过）、C4、B9、B8；若第 2 步证实受限于发射，再重开 A5、A6，并评估 B10。
   - prefill：B4、B5、C8（先测量暴露时间）。
   - 每项都在 H100 和 H20 上同时回归。

humming 上游为 H20 单独写了一套启发式（设备名含 H20 且不含 H200 时启用），与 H100 的主要差别：

- dense 的 overlap 配置要求 M > 64，block_m 上限从 176 降到 96。
- 默认配置用 (128, 128, 128) tile、每 SM 2 个 CTA，多数情况下选非 warp specialization 的 `cp.async` kernel；只有 bm ≥ 48、K/bk ≥ 24 等条件同时满足时才用 warp specialization（3 级）。
- bm ≤ 32 时允许 5 级；K > 1024 时开启 Stream-K 并缩小参与的 SM 数。

这类按设备写死的规则不符合准入原则，但它说明在 H20 上值得进入候选空间的维度：每 SM 2 个 CTA（B9）、更小的 tile、CTA 内 K 切分（B8）、可变的参与 SM 数（E3）。我们的做法是把这些维度加入代价模型的候选集，由 `kH20` 常数决定取舍。

## 建议的 spec-kit 拆分

先用 `/speckit-constitution` 写入以下原则，再按顺序为每组建 feature：

- 通用性：即上文的准入原则。
- 可测量：性能结论只认同进程交替 A/B、CUDA graph 计时、固定 shape 集，并同时报告 H100 和 H20。
- 先证据后实现：⊘ 项只有在 profile 证明其在关键路径上时才能重开，并在 spec 中引用对应的 fork 记录。
- 数值不回退：bridge 的逐比特测试必须通过；数值相关改动需附 PPL 抽查。
- AE 可复现：在开始长期优化前给 AE 提交打 tag，论文结果始终可从该 tag 复现；性能工作走分支。

| 顺序 | feature | 包含的项 |
| --- | --- | --- |
| 1 | 配置选择合入与设备参数化 | E2、E3、E4、H1 |
| 2 | H20 基线、profile 与常数拟合 | H20 步骤 1 到 3、H2、H3 |
| 3 | mainloop 调度泛化 | B4、B5、C4 |
| 4 | launch 开销 | F2、F3、F4 |
| 5 | decode 候选空间扩展 | B7、B8、B9 |
| 6 | tile 间重叠 | C8 |
| 7 | 服务侧融合 | D3、D4、D5 |
| 8 | 数值选项 | D7、A7、B11 |
| 9 | MoE grouped GEMM | G1 |
| 条件 | H20 decode 发射瓶颈 | A5、A6、B10，仅在 feature 2 证实后开启 |
