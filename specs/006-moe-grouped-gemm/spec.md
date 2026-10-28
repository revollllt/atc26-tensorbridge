# Feature Specification: MoE grouped GEMM

**Feature Branch**: `perf`（spec 目录 `specs/006-moe-grouped-gemm`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "为 TensorBridge 增加 MoE grouped GEMM 支持，并接入 vLLM（架构文档 W5，checklist G1）。在 AE kernel 上扩展，以 DeepGEMM 集成的测试场景为验收。contiguous 与 masked 两种布局；第一版不开 Stream-K 与 multicast；每专家 global scale；配置选择以每专家期望行数代替 M。vLLM 0.20.2 fork 中接管 FusedMoE 的 quant method。目标模型 Qwen3.6-35B-A3B。基线：humming MoE MXFP4×FP8、Marlin NVFP4A16 MoE、DeepGEMM FP8 分组 GEMM。验收：正确性、kernel 级对比、vLLM 端到端、精度。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 正确的分组 GEMM（Priority: P1）

一次调用计算多个专家各自的 GEMM：每个专家有自己的 NVFP4 权重和 global scale，每个专家分到的 token 数各不相同。支持两种输入布局：

- **contiguous**：各专家的 token 按专家拼接，每段补齐到 tile 高度的倍数，用于 prefill；补齐部分不产生计算和访存。
- **masked**：每个专家有固定容量，实际行数由一个按专家的计数给出，用于 decode；形状固定，可被 CUDA graph 捕获。

**Why this priority**: 没有分组 GEMM，MoE 模型无法使用 TensorBridge；正确性是后续性能与集成的前提。

**Independent Test**: 用 DeepGEMM 集成中已有的测试场景验证：与逐专家调用非分组 kernel 的结果逐比特一致、CUDA graph 重放、极小的期望行数、每专家不同的 global scale、空专家与极端不均衡的负载。

**Acceptance Scenarios**:

1. **Given** 若干专家及其 token，**When** 用分组 GEMM 计算，**Then** 每个专家的输出行与对该专家单独调用非分组 kernel（同一 data-parallel 配置）的结果逐比特相同。
2. **Given** 某些专家没有分到 token，**When** 运行 masked 布局，**Then** 这些专家不产生任何 tile，其余专家结果正确。
3. **Given** contiguous 布局中补齐的行，**When** 运行，**Then** 补齐块不加载权重、不做计算；补齐行的输出内容不作保证，调用方不读取它们。
4. **Given** masked 布局被 CUDA graph 捕获，**When** 以不同的每专家计数多次重放，**Then** 每次结果都正确。

---

### User Story 2 - kernel 级性能对比（Priority: P1）

在目标模型的专家 shape 上，对多种 token 数与负载分布，比较 TensorBridge 分组 GEMM 与 humming 的 MoE MXFP4 × FP8、vLLM 默认的 Marlin NVFP4A16 MoE、DeepGEMM 的 FP8 分组 GEMM；DeepGEMM 集成中的 TensorBridge 分组 kernel 作为内部参照。

**Why this priority**: MoE 是 serving 的主流形态，也是 humming 在 H20 上重点优化的场景；需要明确 TensorBridge 在这里的位置。

**Independent Test**: 使用 spec 001 准备好的 MoE shape 集与负载分布，在 H100 与 H20 上生成对比报告。

**Acceptance Scenarios**:

1. **Given** Qwen3.6-35B-A3B 的两个专家 shape（gate_up：N = 1024、K = 2048；down：N = 2048、K = 512）以及 decode 与 prefill 的 token 数，**When** 运行对比，**Then** 报告给出各实现的延迟与 TensorBridge 的相对比值。
2. **Given** 均衡与不均衡两种负载分布，**When** 运行对比，**Then** 两种分布都在报告中，且 TensorBridge 的结果都通过 User Story 1 的正确性检查。

---

### User Story 3 - 在 vLLM 中服务 MoE 模型（Priority: P1）

维护者用 AE 仓库的 vLLM fork 加载 Qwen3.6-35B-A3B 的 NVFP4 checkpoint（权重 NVFP4、激活动态 per-token FP8），MoE 层走 TensorBridge 分组 GEMM，完成离线与在线 serving。

**Why this priority**: 当前 fork 遇到「NVFP4 权重 + FP8 激活」的 MoE 层会直接报错；论文的 serving 实验因此只覆盖 dense 模型。

**Independent Test**: 加载模型并运行一组离线与在线 serving 请求，与 Marlin NVFP4A16 MoE 路径比较吞吐与 TPOT。

**Acceptance Scenarios**:

1. **Given** Qwen3.6-35B-A3B 的 checkpoint，**When** 用 TensorBridge 量化方法加载，**Then** 全部 MoE 层由 TensorBridge 处理，加载不报错。
2. **Given** 加载完成的模型，**When** 运行离线 serving（批大小 1 到 256），**Then** 输出吞吐与 TPOT 写入报告，并与 Marlin NVFP4A16 MoE 路径对比。
3. **Given** TP 为 8（down 投影每卡 K = 64，不满足对齐要求），**When** 尝试加载，**Then** 在加载阶段给出明确的不支持提示。

---

### User Story 4 - decode 用 masked 布局进 CUDA graph（Priority: P2）

第一步 serving 统一用 contiguous 布局；第二步在 decode 阶段改用 masked 布局，使整个 decode 步可以被 CUDA graph 捕获。

**Why this priority**: decode 的批小、kernel 短，CUDA graph 对 TPOT 影响大；但它依赖 User Story 3 先跑通。

**Independent Test**: 在 decode 中开启 CUDA graph，比较 TPOT 与 User Story 3 的结果。

**Acceptance Scenarios**:

1. **Given** decode 使用 masked 布局，**When** 开启 CUDA graph，**Then** 输出与不开时一致，TPOT 写入报告并与 User Story 3 对比。

---

### User Story 5 - 精度与仿真一致（Priority: P2）

真实 kernel 路径下模型的 WikiText2 PPL 与 zero-shot 平均准确率，与 `eval/experts.py` 对 bridge 权重的逐比特仿真结果一致。

**Why this priority**: 论文 Table 3 与 4 中 MoE 模型的 Ours 行来自仿真；真实 kernel 路径需要证明两者一致。

**Independent Test**: 用 `eval/` 的流程分别跑仿真与真实 kernel 路径，比较 PPL 与 zero-shot。

**Acceptance Scenarios**:

1. **Given** 同一个 checkpoint，**When** 分别以仿真与真实 kernel 路径评估，**Then** PPL 相差不超过 0.03，zero-shot 平均准确率相差不超过 0.3 个百分点。

---

### User Story 6 - 分组配置的模型选择（Priority: P3）

分组 GEMM 的 tile 等配置由 spec 003 的代价模型选择：以每个专家的期望行数代替 M，并计入补齐带来的浪费；区间表按 `(N, K, 专家数, 布局)` 建立。

**Why this priority**: 先用固定的合理配置跑通正确性与集成，再接入选择器。

**Independent Test**: 在 MoE shape 集上比较模型所选配置与实测最优候选。

**Acceptance Scenarios**:

1. **Given** MoE shape 集与负载分布，**When** 实测模型所选配置与全部候选，**Then** 几何平均差距不超过 5%。

---

### Edge Cases

- 全部 token 都路由到同一个专家：结果正确，性能退化到单个 dense GEMM 的水平以内。
- masked 布局中某专家的实际行数超过容量：在调用边界报错，不静默截断。
- gate 与 up 两个投影的 global scale 不相等：在加载阶段报错，提示需要每通道 scale（checklist D4）；本 checkpoint 中两者逐专家相等。
- 某个专家的 scale 字节超出 bridge 支持的范围：按专家检查，报告是哪一个专家、哪一层。
- 专家并行（EP）与数据并行：不在本 spec 范围内。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统必须提供分组 GEMM，支持 contiguous 与 masked 两种布局，语义如 User Story 1 所述。
- **FR-002**: 每个专家必须可以有自己的 global scale。
- **FR-003**: 对任意专家，分组 GEMM 的输出行必须与对该专家单独调用非分组 kernel（同一 data-parallel 配置）的结果逐比特相同。
- **FR-004**: contiguous 布局中的补齐块不得加载权重或做计算；masked 布局中没有 token 的专家不得产生 tile。
- **FR-005**: masked 布局的调用形状必须与每专家的实际行数无关，以便被 CUDA graph 捕获；每专家计数在设备侧给出。
- **FR-006**: 权重准备必须接受堆叠的专家权重与 scale，并按专家检查 scale 字节的合法性；gate 与 up 的 global scale 不相等时必须在加载阶段报错。
- **FR-007**: 第一版分组模式不得启用 Stream-K 与 multicast。
- **FR-008**: vLLM 插件必须接管 MoE 层：在当前会报错的「NVFP4 权重 + 动态 per-token FP8 激活」组合上，改由 TensorBridge 处理；第二个 GEMM 的输入必须由融合的 SiLU-mul 加 per-token FP8 量化生成。
- **FR-009**: 目标模型 Qwen3.6-35B-A3B 必须能在 fork 中加载（其架构名需通过覆盖或注册识别）；TP 为 1、2、4 时必须可用，TP 为 8 时必须在加载阶段明确拒绝。
- **FR-010**: 精度验收必须比较真实 kernel 路径与 `eval/experts.py` 仿真的 PPL 与 zero-shot。
- **FR-011**: 性能验收必须按 constitution 原则 II，在 H100 与 H20 上进行，基线如 User Story 2 所列。
- **FR-012**: 分组配置的选择必须接入 spec 003 的选择器（User Story 6），不得引入按 shape 的规则。

### Key Entities

- **专家组**：同一 MoE 层中全部专家的权重、scale 与 global scale。
- **分组布局描述**：contiguous 布局下每个 tile 所属的专家，或 masked 布局下每个专家的实际行数与容量。
- **负载分布**：一次调用中 token 在专家间的分布（均衡、不均衡、单专家），用于测试与基准。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: User Story 1 的全部场景通过，逐比特一致率 100%。
- **SC-002**: 在目标模型的专家 shape 上，TensorBridge 分组 GEMM 相对 Marlin NVFP4A16 MoE 的几何平均提速不低于 1.3 倍（H100 与 H20 各自统计）；相对 humming MoE MXFP4 × FP8 的比值写入报告，目标为不慢于 humming。
- **SC-003**: vLLM 中 Qwen3.6-35B-A3B 的离线 serving，批大小 16 及以上时，输出吞吐比 Marlin NVFP4A16 MoE 路径至少高 10%。
- **SC-004**: 真实 kernel 路径与仿真的 PPL 相差不超过 0.03，zero-shot 平均准确率相差不超过 0.3 个百分点。
- **SC-005**: contiguous 布局中补齐块占用的时间不超过 kernel 总时间的 2%。

## Assumptions

- spec 001 的测量基座与 MoE shape 集已可用。本 spec 在 spec 002 到 004 之后实施，以便直接受益于这些优化；正确性部分可以提前开始。
- 扩展的是 AE kernel，而不是直接移植 DeepGEMM 集成的分组 kernel：两者的权重重排不兼容。DeepGEMM 集成的测试用例作为验收场景复用。
- 服务器上的 `tensorbridge-vllm-plugin`（面向 vLLM 0.26、基于 DeepGEMM 集成）可作为 MoE method 与 experts 类的参考实现。
- humming 的 MoE 只在 kernel 级对比：它在上游 vLLM 中可用，但与 AE 的 vLLM 0.20.2 fork 不是同一 serving 栈，端到端对比不在本 spec 范围内。
- Qwen3.6-35B-A3B 的 checkpoint 中，gate 与 up 的 global scale 逐专家相等（已核对）；每层有一个共享专家，按 dense 线性层处理。
- 专家并行（EP）、数据并行与 TP = 8 不在本 spec 范围内。
- SC-002 与 SC-003 的倍数是初始目标；spec 001 的 MoE 基线报告出来后，plan 阶段再确认。
