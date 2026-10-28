# Feature Specification: LUT 反量化

**Feature Branch**: `perf`（spec 目录 `specs/002-lut-dequant`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "把 DeepGEMM 集成中的 LUT FPMA bridge 移植到 TensorBridge 的 S2R 反量化路径，替换现有 add+mask bridge（架构文档 W1，checklist A6 重开、A5 条件项）。输出字节与现有 bridge 逐比特一致；离线布局改为量级分组；删除 add+mask 专有的变体；在 H100 与 H20 上同进程 A/B 验收。scale 线程序重排只在 H20 decode 剖析证明发射受限时进行。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 数值不变、指令更少的反量化（Priority: P1）

维护者把 kernel 中把 NVFP4 权重转换成 FP8 操作数的步骤换成查表实现：对每个 scale 先在寄存器中构造一张 8 字节的表，再用一次字节置换取出 4 个值。零值直接作为表中的一项，不再需要单独的零值掩码。转换后的每个字节都必须与现有实现完全相同。

**Why this priority**: 稳态 mainloop 中约 94% 的指令与 BM 无关，其中大部分是这一步转换（见 checklist「当前 mainloop 的指令构成」）。在 H20 上，每个 SM 的权重到达速率是 H100 的 2 倍，decode 很可能受限于这部分指令的发射。这一步是所有反量化相关优化的基础。

**Independent Test**: 对全部合法的 scale 字节与全部 E2M1 编码做穷举，比较新旧两种转换的输出字节；再在若干 shape 上比较新旧 kernel 的整矩阵输出。

**Acceptance Scenarios**:

1. **Given** 任意合法的原始 scale 字节（0x1C 到 0x7E）与任意经过 SNC 重编码的 E2M1 编码，**When** 分别用新旧实现转换，**Then** 两者的 FP8 字节完全相同。
2. **Given** 同一组输入与同一个 data-parallel 配置，**When** 分别运行新旧 kernel，**Then** 输出矩阵逐比特相同。
3. **Given** 新实现编译出的稳态循环，**When** 用 spec 001 的剖析能力统计，**Then** 每 16 个权重（一个 k32 call、每线程）的转换指令不超过 28 条（现有实现：scale 打包时 40 条、预广播时 35.25 条）。

---

### User Story 2 - 两台设备上的性能验收（Priority: P1）

维护者用 spec 001 的测量基座，在 H100 和 H20 上对新旧实现做同进程 A/B，同时记录 humming MXFP4 × FP8 与 cuBLAS FP8 基线，以此决定是否合入。

**Why this priority**: checklist 中的 LUT 路线（A6）曾在 fork 里实测为负收益。这次重开的依据是新设计的 SASS 指令数，以及 DeepGEMM 集成 Round 16 的整 kernel 收益（1.027 到 1.087 倍）；但它在 AE kernel 上必须重新验证。

**Independent Test**: 对全部 shape 集生成 A/B 报告，检查几何平均、最差 shape 与 decode 扫描的收益。

**Acceptance Scenarios**:

1. **Given** H100 上的全部 shape 集，**When** 运行 A/B，**Then** 每个 shape 集的几何平均不低于 1.00，且没有 shape 慢 3% 以上。
2. **Given** H20 上的 decode 扫描（M = 1 到 128），**When** 运行 A/B，**Then** 几何平均提速不低于 5%。
3. **Given** A/B 结果未达标，**When** 维护者结束本 spec，**Then** 负结果与数据写回 checklist A6，现有实现保持不变。

---

### User Story 3 - 配置空间随之简化（Priority: P2）

现有实现有两个只为 add+mask 存在的变体：scale 预广播，以及两种零值掩码写法。换成查表实现后，这些变体不再有意义，应从配置、选择逻辑和已有的调优记录中删除。

**Why this priority**: 配置维度越少，spec 003 的代价模型越小、越准；不再存在的变体留在代码里，违反原则 VI。

**Independent Test**: 检查配置结构、配置选择与 JIT 生成的代码中不再出现这些变体，现有测试全部通过。

**Acceptance Scenarios**:

1. **Given** 新实现已合入，**When** 列出可选的配置维度，**Then** 其中不再有 scale 预广播或掩码写法。
2. **Given** 现有选择逻辑和调优表中引用了这些变体，**When** 本 spec 完成，**Then** 这些引用均已删除，选择结果在其他维度上不变。

---

### User Story 4 - scale 按线程序重排（Priority: P3，条件项）

如果 spec 001 在 H20 decode 上的剖析表明 kernel 受限于指令发射，并且换成查表实现后 scale 的逐字节读取成了显著部分（当前每 K block 每线程 16 次单字节读取，而权重只需 2 次 16 字节读取），则把 scale 也按线程序离线重排，使每 K block 每线程只需 1 次 16 字节读取。

**Why this priority**: 这项（checklist A5）在 fork 的 H100 prefill 上实测为负收益（−1.1% 与 −1.5%）；按原则 III，只有在新场景下证明它处于关键路径上才能做。

**Independent Test**: 先给出 H20 decode 的剖析证据；满足条件后，做与 User Story 2 相同的 A/B。

**Acceptance Scenarios**:

1. **Given** H20 decode 的剖析不显示发射受限，**When** 评估本故事，**Then** 不实施，并在 checklist A5 中记录判断依据。
2. **Given** 剖析显示发射受限且已实施，**When** 运行 A/B，**Then** H20 decode 扫描的几何平均进一步提升，H100 各 shape 集不回退。

---

### Edge Cases

- 原始 scale 字节低于 0x1C 的 group：仍按现有规则整组置零；置零后的编码经 SNC 映射到表中的零值项，输出必须为 0。
- 原始 scale 字节为上限 0x7E：表中最大项恰为 0x7E，不得向相邻字节进位或进入符号位。
- 负零：SNC 已丢弃零值的符号，输出必须是 +0，与现有实现一致。
- 超出范围的外来 checkpoint（scale 字节大于 0x7E）：沿用现有的 `clamp_scale` 行为，本 spec 不改变。
- Stream-K slice、非 4-buffer 的 BM 等所有 mainloop 路径都必须使用新实现，不能遗留旧实现的分支。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 新的转换实现对全部合法输入（原始 scale 字节 0x1C 到 0x7E，SNC 重编码后的全部 16 种编码）的输出字节必须与现有实现完全相同。
- **FR-002**: 新实现中，每个 scale 对应的 8 项查找表必须在寄存器中构造，零值必须作为表项处理，不得再有单独的零值掩码步骤。
- **FR-003**: 离线权重准备必须产出新实现所需的布局：每个 32 位字中，前 4 个 nibble 的量级位存 K-low 的 4 个值、后 4 个 nibble 的量级位存 K-high 的 4 个值，符号位位置保持不变。checkpoint 格式不变，转换只发生在加载时。
- **FR-004**: kernel 的所有 mainloop 路径（各种 BM、data parallel 与 Stream-K）都必须使用新实现。
- **FR-005**: scale 预广播与两种零值掩码变体必须从配置、配置选择、JIT 生成代码与调优记录中删除。
- **FR-006**: 对同一输入与同一 data-parallel 配置，新旧 kernel 的输出矩阵必须逐比特相同。
- **FR-007**: 性能验收必须按 constitution 原则 II 进行：H100 与 H20、全部 shape 集、同进程 A/B，报告含 humming MXFP4 × FP8 与 cuBLAS FP8 基线。
- **FR-008**: 无论结果正负，A/B 结果与 SASS 统计都必须写回 checklist A6（以及 A5，如适用）。
- **FR-009**: scale 线程序重排（User Story 4）只有在 spec 001 的 H20 decode 剖析证明发射受限时才可实施，并须引用 fork 中对应的实验记录（`26-05-26_00-15` 等 pair-pack 记录）说明新场景的差异。

### Key Entities

- **转换表**：由一个预折叠 scale 字节构造的 8 字节表，下标是 SNC 重编码后的 3 位量级，第 1 项为零值。
- **准备后的权重**：加载时由 checkpoint 生成的 kernel 布局，包括按线程序排列且量级分组的打包权重、预折叠 scale 与 global scale。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 穷举测试与整矩阵比较中，新旧输出 100% 逐比特相同。
- **SC-002**: 每 16 个权重的转换指令不超过 28 条；稳态循环每 K block 每线程的指令总数比现有的 205 到 215 条至少少 15%。
- **SC-003**: H100 上每个 shape 集的几何平均不低于 1.00，没有 shape 慢 3% 以上；论文 40 个 shape 的几何平均提速不低于 2%。
- **SC-004**: H20 上 decode 扫描（M = 1 到 128）的几何平均提速不低于 5%，其余 shape 集不回退。
- **SC-005**: 加载时的权重准备耗时增加不超过 10%（论文中 Qwen3-14B 的 280 个线性层为 0.50 s）。

## Assumptions

- spec 001 的测量基座与剖析能力已经可用；本 spec 的 A/B 与 SASS 统计都使用它。
- 由于输出逐比特相同，本 spec 不需要 constitution 原则 IV 要求的 PPL 抽查；若实现过程中改变了任何数值，则需要补做。
- DeepGEMM 集成（`integrations/deepgemm`）保留自己的副本，本 spec 不改动它。
- 成功标准中的提速目标是基于 DeepGEMM 集成 Round 16 与 SASS 统计设定的初始目标；未达标时按 User Story 2 第 3 条处理，不强行合入。
- 精确（ULP）模式与 MXFP4 输入是在查表实现之上的后续机会，见架构文档 W1，不在本 spec 范围内。
