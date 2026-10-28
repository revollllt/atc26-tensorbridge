# Feature Specification: launch 与 host 开销

**Feature Branch**: `perf`（spec 目录 `specs/005-launch-host-overhead`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "降低 TensorBridge 每次 GEMM 调用的 launch 与 host 开销（架构文档 W4，checklist F2–F4、D5）。先测量 eager 模式每次调用的 host 时间分解与 decode CUDA graph 中相邻 kernel 的 GPU 空隙。内容：描述符只编码一次或缓存、kernel 属性与设备属性只设置或查询一次、配置查找与 launch 合并为一次调用、PDL、按区间表在加载时预编译并提供 warmup 入口、把 per-token FP8 量化融进前一个算子。验收：eager host 时间、decode graph 内 kernel 间空隙、vLLM decode 每步时间与 TPOT（dense 模型，H100 与 H20），冷启动无首请求编译。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 先把开销量清楚（Priority: P1）

在动手之前，维护者需要知道一次 GEMM 调用的固定开销花在哪里：eager 模式下 host 侧每一步（Python 分派、配置选择、kernel 查找、描述符编码、属性设置、launch）各占多少；decode 的 CUDA graph 中，GEMM 与前后 kernel 之间的 GPU 空隙有多大。

**Why this priority**: 代价模型拟合出每个 kernel 约 2 µs 的固定开销，而 decode 时单个 GEMM 只有几 µs。不先测量，就无法判断各项改动的优先级，也无法验收。

**Independent Test**: 在 H100 与 H20 上各产出一份开销分解报告。

**Acceptance Scenarios**:

1. **Given** 一组 decode 与 prefill 的 shape，**When** 在 eager 模式下运行开销分解，**Then** 报告给出每一步的平均 host 时间及其占比。
2. **Given** 一个 dense 模型的 decode CUDA graph，**When** 采集 GPU 时间线，**Then** 报告给出每个线性层前后的 kernel 间空隙，以及 GEMM 内部第一次读取依赖数据之前的准备时间。

---

### User Story 2 - 每次调用只做必要的 host 工作（Priority: P1）

每次 GEMM 调用的 host 工作只剩真正随调用变化的部分。权重的 TMA 描述符在权重准备时编码一次；激活与输出的描述符按需缓存或只替换地址；kernel 属性在注册时设置一次；设备属性在进程内只查询一次；配置查找与 launch 合并为一次从 Python 到 C++ 的调用。

**Why this priority**: 超出 CUDA graph 捕获范围的批大小（大 prefill）和不用 CUDA graph 的部署都在 eager 模式下运行，每次调用的 host 开销直接计入延迟。

**Independent Test**: 对 User Story 1 的同一组 shape 重新做开销分解，并验证输出与改动前逐比特相同。

**Acceptance Scenarios**:

1. **Given** User Story 1 的现状报告，**When** 改动后重新测量，**Then** eager 模式每次调用的 host 时间至少减少 50%。
2. **Given** 连续调用中激活或输出的地址、M 不断变化，**When** 运行 GEMM，**Then** 每次结果都正确，缓存不会无界增长。
3. **Given** 同一组输入，**When** 分别运行改动前后的版本，**Then** 输出逐比特相同。

---

### User Story 3 - 用 PDL 隐藏 kernel 的准备时间（Priority: P1）

decode 时 GEMM 的 barrier 初始化、描述符预取等准备工作，与前一个 kernel 的收尾重叠：GEMM 启动后先做与输入无关的准备，等到真正读取依赖数据前再等待前一个 kernel；mainloop 结束后允许后继 kernel 提前启动。

**Why this priority**: decode 每步有几十到上百次线性层调用，每次省下的准备时间都会累积到每个 token 的延迟上。humming 实现了同样的机制但默认关闭，fork 中也没有做过 A/B，checklist F2 是未测项。

**Independent Test**: 在 decode CUDA graph 中比较开启前后的 kernel 间空隙与每步时间。

**Acceptance Scenarios**:

1. **Given** 一个写入 GEMM 输入的前驱 kernel，**When** 两者以 PDL 方式连续执行，**Then** GEMM 的结果与串行执行逐比特相同。
2. **Given** 一个 dense 模型的 decode CUDA graph，**When** 开启 PDL，**Then** 线性层前的 kernel 间空隙与准备时间合计至少减少 50%。
3. **Given** 开启 PDL 的 kernel，**When** 被 CUDA graph 捕获并重放，**Then** 行为正确。

---

### User Story 4 - 服务过程中不再触发编译（Priority: P2）

模型加载时，系统按 spec 003 的区间表把该模型会用到的全部 kernel 编译好；另提供一个显式的预热入口，供 vLLM 在捕获 CUDA graph 之前调用。服务开始后，任何请求都不会遇到首次编译的延迟。

**Why this priority**: 现在 kernel 在首次遇到某个配置时才编译，服务中的第一个请求可能被阻塞数秒。

**Independent Test**: 启动服务并跑一轮完整的 benchmark，统计服务期间发生的编译次数与加载时间的变化。

**Acceptance Scenarios**:

1. **Given** 一个已加载并预热的模型，**When** 运行一轮覆盖各种批大小的 benchmark，**Then** 服务期间发生的编译次数为 0。
2. **Given** 磁盘编译缓存已存在，**When** 加载模型，**Then** 加载时间比现在增加不超过 10 s。

---

### User Story 5 - 每个线性层只剩一个 kernel（Priority: P2）

现在每个线性层要先跑一个 per-token FP8 量化 kernel，再跑 GEMM。把量化融进前一个算子（RMSNorm、SiLU-mul），使 QKV、gate_up、down 投影前不再单独启动量化 kernel。

**Why this priority**: decode 每层少 3 次 launch，同时少一次激活的读写。

**Independent Test**: 统计一个 decoder 层的 kernel 数，并在 vLLM 中比较 decode 每步时间与 PPL。

**Acceptance Scenarios**:

1. **Given** 一个 dense 模型的 decoder 层，**When** 统计 decode 时的 kernel 序列，**Then** QKV、gate_up、down 三处投影前没有单独的量化 kernel。
2. **Given** 融合改变了量化前的中间舍入，**When** 按 constitution 原则 IV 做 PPL 抽查，**Then** WikiText2 PPL 与改动前相差不超过 0.01。

---

### Edge Cases

- 调用方传入自己分配的输出张量，或每次调用的激活地址都不同：描述符缓存必须以地址和形状为键，或改为只替换地址；不得返回指向旧地址的描述符。
- 权重被移动或重新分配（如卸载后重新加载）：缓存在权重上的描述符必须随之重建。
- 前驱 kernel 不支持 PDL：GEMM 中的等待退化为等待前驱完全结束，结果仍然正确。
- 某个 `(N, K)` 的区间表包含很多不同的配置：预编译并行进行，并复用磁盘缓存。
- 前一个算子没有可用的融合量化实现（例如 o_proj 前的 attention 输出）：保留单独的量化 kernel，不阻塞其他位置的融合。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 必须提供开销分解能力：eager 模式下每次调用各步骤的 host 时间，以及 decode CUDA graph 中线性层前后的 kernel 间空隙与 GEMM 的准备时间；在 H100 与 H20 上各产出一份现状报告。
- **FR-002**: 权重的 TMA 描述符必须在权重准备时编码一次并随权重保存；激活与输出的描述符必须通过有界缓存或地址替换获得，在地址或形状变化时保持正确。
- **FR-003**: kernel 属性必须在注册时设置一次，设备属性必须在进程内只查询一次。
- **FR-004**: 每次 GEMM 调用的配置查找与 launch 必须在一次从 Python 到 C++ 的调用中完成。
- **FR-005**: kernel 必须支持 PDL：在第一次读取依赖数据之前等待前驱，在 mainloop 结束后允许后继启动；launch 时设置相应属性；被 CUDA graph 捕获后行为正确。
- **FR-006**: 系统必须在加载时按 spec 003 的区间表预编译该模型用到的全部 kernel，并提供显式的预热入口；编译结果写入磁盘缓存并在之后复用。
- **FR-007**: vLLM 接入必须在 RMSNorm 与 SiLU-mul 之后使用融合的 per-token FP8 量化，把量化结果直接交给 GEMM；没有融合实现的位置保留单独的量化。
- **FR-008**: FR-002 到 FR-006 不得改变 kernel 的输出；FR-007 若改变了舍入，必须按 constitution 原则 IV 附 PPL 抽查。
- **FR-009**: 端到端验收必须在 H100 与 H20 上，用至少一个 dense 模型测量 vLLM decode 的每步时间与 TPOT。

### Key Entities

- **开销分解报告**：一次 GEMM 调用在 host 与 GPU 上的固定开销，按步骤拆分。
- **描述符缓存**：按地址与形状索引的激活与输出描述符，容量有上限。
- **预编译集合**：某个模型的全部 `(N, K)` 区间表中出现的 kernel，在加载或预热时编译。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: eager 模式下每次 GEMM 调用的 host 时间比 User Story 1 的现状至少减少 50%，两台设备都达到。
- **SC-002**: decode CUDA graph 中，线性层前的 kernel 间空隙与 GEMM 准备时间合计至少减少 50%。
- **SC-003**: 一个 8B dense 模型在 vLLM 中、批大小 1 与 16 时，decode 的 TPOT 在 H100 与 H20 上都至少改善 3%。
- **SC-004**: 预热后的完整 benchmark 运行期间，编译次数为 0；磁盘缓存存在时，模型加载时间增加不超过 10 s。
- **SC-005**: 引入量化融合后，WikiText2 PPL 与改动前相差不超过 0.01；不涉及融合的改动输出逐比特相同。

## Assumptions

- spec 001 的测量基座已可用；端到端测量使用 AE 仓库中 vLLM 0.20.2 fork 的 serving 流程。
- spec 003 的区间表先于本 spec 的 User Story 4 合入；在那之前，预编译按现有选择逻辑在 vLLM 的 CUDA graph 捕获尺寸上进行。
- vLLM fork 已有可用于 per-token FP8 的 RMSNorm 与 SiLU-mul 融合量化实现，或能以较小改动提供；具体可用性在 plan 阶段核实。若不可用，User Story 5 降级为在插件内提供融合实现。
- PDL 需要 Hopper 与 CUDA 12.x，两台设备都满足。
- SC-003 的 3% 是初始目标；实际收益取决于 decode 中线性层的占比，plan 阶段按 User Story 1 的报告再确认。
