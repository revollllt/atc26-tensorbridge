# Feature Specification: 统一测量基座与基线

**Feature Branch**: `perf`（spec 目录 `specs/001-bench-baselines`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "为 TensorBridge 的长期性能优化建立统一的测量基座（架构文档 W0）：同一进程内交替测量 TensorBridge 与基线、CUDA graph 计时；基线必须包含 humming v0.1.18 的 MXFP4 权重 × FP8 激活模式（重点，dense 与 MoE）与 cuBLAS FP8；固定 shape 集入库；输出延迟、吞吐、占实测峰值比例与环境元数据；附带剖析脚本；H100 与 H20 都能运行，并产出两台设备的现状报告。对应 checklist H1–H4。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 候选改动的同进程 A/B（Priority: P1）

维护者完成一项 kernel 改动后，在一台 GPU 上把改动版本与当前版本放在同一进程内交替测量，得到每个 shape 集的几何平均和最差 shape，用来决定改动是否合入。

**Why this priority**: constitution 原则 II 规定，性能结论只认同进程交替 A/B。没有这项能力，后续所有优化 spec 都无法验收。

**Independent Test**: 用同一版本分别充当「候选」和「当前」做一次 A/B，结果应接近 1.0，并给出每个 shape 集的几何平均、最差 shape 和各轮离散度。

**Acceptance Scenarios**:

1. **Given** 一个以配置字段或编译期开关区分的候选版本，**When** 维护者在 H100 上对论文 40 个 shape 运行 A/B，**Then** 报告列出每个 shape 的两版延迟与比值，以及该集合的几何平均和最差 shape。
2. **Given** 同一节点上连续两次运行同一个 A/B，**When** 比较两次报告，**Then** 每个 shape 集的几何平均相差不超过 1%。

---

### User Story 2 - 与 humming 和 cuBLAS FP8 的同 shape 对比（Priority: P1）

维护者需要知道 TensorBridge 相对最接近的公开实现处于什么位置：在相同 shape 上，与 humming 的 MXFP4 权重 × FP8 激活模式和 cuBLAS FP8 同进程交替测量。

**Why this priority**: constitution 原则 II 把这两个基线列为必测项；只和自己比，无法判断优化是否到位。

**Independent Test**: 对论文 40 个 shape 生成一份报告，其中每个 shape 都有 TensorBridge、humming MXFP4 × FP8、cuBLAS FP8 三列延迟，以及 TensorBridge 相对两者的比值。

**Acceptance Scenarios**:

1. **Given** humming 固定在 v0.1.18，**When** 运行对比，**Then** 报告记录 humming 的版本号，并对它支持的每个 shape 给出延迟。
2. **Given** humming 不支持的 shape，**When** 运行对比，**Then** 该 shape 在 humming 列标为「不支持」并注明原因，其余 kernel 照常测量，整个运行不中断。
3. **Given** 某个基线在正确性预检中失败，**When** 运行对比，**Then** 报告把该基线在受影响 shape 上标为「无效」，不参与比值计算。

---

### User Story 3 - H100 与 H20 的现状报告（Priority: P2）

在开始任何优化之前，维护者在 H100 和 H20 上各产出一份完整的现状报告：覆盖全部 shape 集、全部必测基线、占实测峰值的比例。后续每个 spec 都以它为对照。

**Why this priority**: H20 的硬件比例与 H100 差异很大（每 SM 的 Tensor Core 算力约为 1/4，每 SM 的 HBM 带宽约为 2 倍），后续的优化排序取决于两台设备上的实测位置。

**Independent Test**: 两台设备各有一份入库的报告，包含全部 shape 集的结果文件、汇总表和环境元数据。

**Acceptance Scenarios**:

1. **Given** H20 计算节点无法联网，**When** 维护者按文档在登录节点准备好依赖和编译缓存后提交作业，**Then** 作业在计算节点上完成测量，过程中不访问网络。
2. **Given** 依赖或编译缓存未准备好，**When** 作业在计算节点上启动，**Then** 它在开始计时前就失败，并指出缺少的条目。

---

### User Story 4 - MoE 分组 GEMM 的基线（Priority: P2）

维护者需要在 MoE 专家 shape 上测量 humming 的分组 MXFP4 × FP8 GEMM，为 spec 006（MoE grouped GEMM）预先建立对照。

**Why this priority**: MoE 支持是后续工作流之一；先有基线，才能为它设定目标。

**Independent Test**: 对 Qwen3.6-35B-A3B 的专家 shape，在若干组 token 数与专家负载分布下，给出 humming 分组 GEMM 的延迟。

**Acceptance Scenarios**:

1. **Given** 一组 token 数和专家负载分布（均衡、不均衡），**When** 运行 MoE 基线，**Then** 报告给出 humming 在 contiguous 与 indexed 两种分组方式下的延迟。
2. **Given** TensorBridge 的分组 GEMM 尚未实现，**When** 运行 MoE 基线，**Then** 报告中 TensorBridge 一列为空，运行不受影响；spec 006 完成后同一套 shape 与负载分布可直接加入 TensorBridge。

---

### User Story 5 - 瓶颈剖析（Priority: P3）

维护者在提出优化前，需要对指定配置拿到瓶颈证据：编译产物中稳态循环每 K block 的指令数与构成，以及关键硬件计数器。

**Why this priority**: constitution 原则 III 要求每个优化 spec 引用瓶颈证据；这项能力把证据收集变成可复现的步骤。

**Independent Test**: 对当前 kernel 的 BM = 128、4 级配置运行指令统计，结果与 checklist 中记录的值（每 K block 每线程 205 到 215 条）一致。

**Acceptance Scenarios**:

1. **Given** 一个 kernel 配置，**When** 运行指令统计，**Then** 输出每 K block 每线程的指令总数和按操作码分类的条数。
2. **Given** 一个 shape 和配置，**When** 运行硬件计数器剖析，**Then** 输出发射利用率、Tensor Core 利用率和 DRAM 吞吐。

---

### Edge Cases

- 某个 shape 的权重轮换所需显存超过设备容量：减少副本数，在报告中把该 shape 的 L2 模式标为「受限」。
- 基线首次调用需要较长的即时编译：编译在计时开始前完成，不计入延迟；编译失败按「不支持」处理。
- 两版 TensorBridge 的 A/B 中，某个 shape 只有一版能运行：该 shape 标为「仅一侧可运行」，不进入几何平均，单独列出。
- GPU 时钟或温度波动：每个 shape 记录各轮的最小值与最大值；离散度超过阈值的 shape 在报告中标出。
- 输出中出现非有限值：该 kernel 在该 shape 上标为「无效」，并单独列出。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统必须在同一进程内交替测量所有参与比较的 kernel，每轮改变顺序；每个 kernel 以 CUDA graph 重放计时，报告多轮的中位数，并记录各轮的最小值与最大值。
- **FR-002**: 所有参与比较的 kernel 必须接收已量化好的 FP8 激活；激活量化不计入延迟。
- **FR-003**: 必测基线必须包含：humming v0.1.18 的 MXFP4（UE8M0 scale、group 32）权重 × per-token FP8 激活模式，dense 与 MoE 分组两种；cuBLAS FP8（per-token × per-channel scale）。可选基线：cuBLAS BF16、vLLM 的 CUTLASS W4A8、Marlin NVFP4A16，在环境提供时加入。
- **FR-004**: 每个基线在计时前必须做一次正确性预检：与该基线自身格式的参考反量化结果比较；预检失败的基线在受影响 shape 上标为「无效」。
- **FR-005**: 系统必须支持两种 L2 状态：默认的流式模式（权重在总量超过 2 倍 L2 容量的多份副本间轮换）和热 L2 模式；报告必须注明所用模式。
- **FR-006**: 以下 shape 集必须作为版本化文件入库：论文 Figure 11 的 40 个 shape、代价模型拟合集 410 个、held-out 集 695 个、decode 扫描（M = 1 到 128）、Qwen3.6-35B-A3B 的 MoE 专家 shape（含 token 数与负载分布）。
- **FR-007**: 每个 shape 的结果必须包含：延迟、达到的 TFLOPS、达到的 HBM 带宽、占设备实测算力峰值与带宽峰值的比例。每个 shape 集必须给出几何平均和最差 shape。
- **FR-008**: 设备的算力峰值与带宽峰值必须由系统在同一设备上实测并记录，不使用规格值。
- **FR-009**: 每份报告必须记录环境元数据：GPU 型号、SM 数、L2 容量、驱动版本、torch 与 CUDA 版本、TensorBridge 与 humming 的提交号、L2 模式、计时参数。
- **FR-010**: 系统必须支持两个 TensorBridge 变体在同一进程内的 A/B，变体以配置字段或编译期开关区分。
- **FR-011**: 系统必须能在 H100 集群（作业调度提交）和 H20 集群（计算节点离线）上运行；离线节点上缺少依赖或编译缓存时，必须在计时开始前失败并列出缺失项。
- **FR-012**: 系统必须提供剖析能力：从编译产物统计稳态循环每 K block 每线程的指令数与按操作码分类的条数；对指定 shape 采集发射利用率、Tensor Core 利用率与 DRAM 吞吐。
- **FR-013**: 结果必须以机器可读的文件保存，并生成一份人可读的汇总表。
- **FR-014**: H100 与 H20 的现状报告必须作为后续 spec 的对照基准入库。

### Key Entities

- **Shape 集**：一组 `(M, N, K)`（MoE 另含专家数、top-k、token 数与负载分布），有名称和版本。
- **Kernel 条目**：一个参与测量的实现，包括 TensorBridge 的某个变体或某个基线，带有名称、版本和适用范围。
- **测量记录**：某个 kernel 在某个 shape 上的一次测量结果，包括各轮延迟、派生指标和有效性标记。
- **设备画像**：一台设备的实测算力峰值、带宽峰值与环境元数据。
- **报告**：一组测量记录按 shape 集汇总后的结果，附设备画像与元数据。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 同一节点上重复运行同一 A/B，每个 shape 集的几何平均相差不超过 1%。
- **SC-002**: 在一块 H100 上，论文 40 个 shape 加全部必测基线的一次完整对比不超过 15 分钟（不含首次编译）；全部 shape 集（含 decode 扫描）不超过 4 小时。
- **SC-003**: H100 与 H20 的现状报告中，TensorBridge 与 cuBLAS FP8 覆盖 100% 的 dense shape；humming 覆盖其支持的全部 shape，不支持的 shape 全部逐个列出原因。
- **SC-004**: 维护者在每台设备上用一条文档化的命令即可完成一次 A/B，并得到汇总表。
- **SC-005**: 对当前 kernel 的指令统计结果与 checklist 中记录的值相差不超过 2%。

## Assumptions

- humming 固定为 v0.1.18（vLLM main 当前固定的版本）。它的 wheel 基于 torch 2.11 构建，可以和 TensorBridge 安装在同一环境里，从而满足同进程测量。
- humming 不支持 group 16 的权重配 8 位激活，因此 TensorBridge（NVFP4）与 humming（MXFP4）的格式不同；本 spec 只比较延迟，精度对比不在范围内。
- humming 上游的测试未覆盖「MXFP4 g32 权重 × per-token FP8 激活」这一组合，FR-004 的正确性预检就是为此准备的。
- 两台集群都没有锁定 GPU 时钟的权限；以交替测量和多轮统计来抵消波动。
- 代价模型拟合集与 held-out 集来自此前 unified scheduler 工作中使用的 shape 清单，在本 spec 中转为入库文件。
- 可选基线依赖 vLLM fork 的算子，只在装有 fork 的环境中提供；H20 环境不保证提供。
- TensorBridge 的 MoE 分组 GEMM 由 spec 006 实现；本 spec 只为它准备 shape、负载分布和 humming 基线。
