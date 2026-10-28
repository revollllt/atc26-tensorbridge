# Feature Specification: mainloop 调度与数据搬运

**Feature Branch**: `perf`（spec 目录 `specs/004-mainloop-data-movement`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "优化 TensorBridge 的 mainloop 调度与数据搬运（架构文档 W3，checklist B4、B5、B7–B9、C4、C6–C8）。P1 调度泛化：lagged 调度推广到所有 BM，Stream-K slice 也用它。P2：流水线状态改为 index + phase、环跨 tile 连续、级数由 smem 容量决定；先测量下一 tile 首级加载的暴露时间，显著时再处理。P3（以 H20 剖析为前提）：BN = 64、CTA 内 K 切分、每 SM 2 个 CTA、decode 权重加载的 L2 提示。不重做 fork 已证伪项。每项以配置维度出现，由 spec 003 的选择器决定启用；每项过同进程 A/B；数值不变。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 所有 token tile 都走重叠调度（Priority: P1）

当前只有 BM ∈ {16, 32, 64, 128, 256} 时，kernel 才使用「一个 WGMMA 在途、同时加载并转换下一组权重」的 lagged 调度。其余 BM（包括启发式默认上限 176）在每个 k32 call 后都要等 WGMMA 全部完成，转换与矩阵乘完全串行。本故事让所有合法 BM 都使用 lagged 调度。

**Why this priority**: 代价模型的 H100 拟合中，非 lagged 调度每个 token 的 WGMMA 开销比 lagged 高约 41%（0.001563 对 0.001105 µs/K block）。中等 M 的 shape 常被分到这些 BM 上。

**Independent Test**: 对 8 到 256 之间每个 8 的倍数的 BM 编译 kernel，检查无寄存器溢出并通过正确性测试；再对原先落在非 lagged BM 上的 shape 做 A/B。

**Acceptance Scenarios**:

1. **Given** 8 到 256 之间任意 8 的倍数的 BM，**When** 编译 kernel，**Then** 它使用 lagged 调度，且编译报告中没有寄存器溢出；做不到的 BM 从候选集中移除，并记录原因。
2. **Given** 一个 data-parallel 配置，**When** 分别运行改动前后的 kernel，**Then** 输出逐比特相同。
3. **Given** H100 上原先被分到非 lagged BM 的 shape，**When** 做同进程 A/B，**Then** 几何平均提速不低于 5%。

---

### User Story 2 - Stream-K 尾波不再降级（Priority: P1）

Stream-K 的 slice 目前走保守调度：逐个等待 WGMMA 完成后再一次发出 4 个调用。当 tile 数少于 SM 数时，全部工作都在 slice 上完成，decode 和小 N 的 shape 因此吃亏；H20 只有 78 个 SM，这种情况更常见。本故事让 slice 也使用 lagged 调度。

**Why this priority**: decode 是 TensorBridge 相对 FP8 收益最大的区间（H20 上 W4 相对 FP8 的理论收益基本只在 M ≲ 37）。

**Independent Test**: 对 tile 数少于 SM 数的 shape（decode 扫描中的大部分）做 A/B。

**Acceptance Scenarios**:

1. **Given** 一个启用 Stream-K 的配置，**When** 运行改动后的 kernel，**Then** 结果与参考实现的差异仍在现有容差内（切成 3 片以上时至多相差 1 到 2 个 BF16 ULP）。
2. **Given** H100 与 H20 上 tile 数少于 SM 数的 shape，**When** 做 A/B，**Then** 两台设备都不回退，且至少一台的几何平均提速不低于 3%。

---

### User Story 3 - 更深、跨 tile 连续的流水线（Priority: P2）

当前每个 tile 都从第 0 级重新开始填充流水线；级数被限制在 4 到 6，原因是更深的环会让逐级的状态溢出到 local memory。BM = 16 时每级约 11 KB，6 级只用了约 66 KB 共享内存。本故事让流水线状态只用一个下标加一个相位位，环跨 tile 连续，级数上限由共享内存容量决定。

**Why this priority**: H20 每个 SM 的 HBM 带宽是 H100 的 2 倍，需要更多在途字节才能跑满带宽；decode 的小 tile 有大量空闲共享内存可用。

**Independent Test**: 在 decode 扫描上，比较级数可由容量决定的版本与现有版本；检查环跨 tile 连续后多 tile 的 shape 不回退。

**Acceptance Scenarios**:

1. **Given** BM = 16 的配置，**When** 选择器考虑更深的级数，**Then** 编译结果无寄存器溢出，正确性测试通过。
2. **Given** H20 的 decode 扫描，**When** 做 A/B，**Then** 几何平均提速不低于 3%，H100 各 shape 集不回退。

---

### User Story 4 - 消除 tile 之间的空泡（Priority: P2）

输出 tile 与流水线共用共享内存，因此 producer 必须等上一个 tile 的结果写回之后，才能开始加载下一个 tile 的数据；下一个 tile 首级加载的延迟完全暴露。先测量这段暴露时间，确有必要再处理。

**Why this priority**: 每个 CTA 处理多个 tile、K 又较小时，这段空泡的占比最高。但 fork 中对 epilogue 的类似测量认为收益有限（epilogue 约 1.0 µs/tile，估计每个 kernel 只能省约 0.8 µs），所以先测再做。

**Independent Test**: 在每 CTA 多 tile 的 shape 上测量暴露时间占 kernel 时间的比例；超过阈值时实施，并做 A/B。

**Acceptance Scenarios**:

1. **Given** 多 tile、小 K 的 shape，**When** 测量下一 tile 首级加载的暴露时间，**Then** 报告给出它占 kernel 时间的比例。
2. **Given** 暴露时间在任一 shape 集上超过 kernel 时间的 2%，**When** 实施缓解（在 epilogue 期间把下一 tile 的首级预取到 L2，或在共享内存允许时给输出单独分配空间），**Then** 该 shape 集几何平均提速不低于暴露比例的一半，其余不回退。
3. **Given** 暴露时间不超过 2%，**When** 评估本故事，**Then** 不实施，并把测量结果写回 checklist C8。

---

### User Story 5 - H20 驱动的新配置维度（Priority: P3，条件项）

如果 spec 001 的 H20 剖析显示 decode 或小 N 的 shape 受限于并行度或发射，则依次评估以下维度：BN = 64 的 tile、CTA 内沿 K 切分并在共享内存中归约、每 SM 运行 2 个小 tile 的 CTA、decode 时对只读一次的权重加载使用 L2 驱逐优先提示。每个维度都作为新的配置维度交给 spec 003 的选择器。

**Why this priority**: 这些都是 humming 在 H20 上采用、或 DeepGEMM 集成已经支持的做法，但在 AE kernel 上没有测过；是否值得做取决于 H20 剖析。

**Independent Test**: 对每个维度，先给出剖析证据，再做针对目标 shape 集的 A/B。

**Acceptance Scenarios**:

1. **Given** 某个维度已实现并纳入选择器，**When** 在 H20 的目标 shape 集上做 A/B，**Then** 几何平均提速不低于 3% 才保留，否则删除并在 checklist 中记录负结果。
2. **Given** 某个维度被保留，**When** 在 H100 上做 A/B，**Then** 各 shape 集不回退（选择器可以在 H100 上不选它）。

---

### Edge Cases

- 某个 BM 在 lagged 调度下寄存器不够：从候选集中移除并记录，不退回串行调度。
- 级数增加后共享内存不足以放下输出 tile：该组合不合法，不进入候选。
- 分组 GEMM（spec 006）首版不使用 Stream-K 与 multicast，本 spec 的改动不能破坏这一前提。
- multicast 集群仍只用于 data parallel；Stream-K 与 multicast 的已知问题不在本 spec 范围内。
- 每 SM 2 个 CTA 时，每个线程可用的寄存器减半：只对寄存器需求足够小的 tile 开放。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 8 到 256 之间所有 8 的倍数的 BM 都必须使用 lagged 调度，且编译结果没有寄存器溢出；做不到的 BM 必须从候选集中移除，并记录原因。
- **FR-002**: Stream-K 的 slice 必须使用与 data-parallel tile 相同的 lagged 调度。
- **FR-003**: 流水线状态必须只由一个下标与一个相位位表示，环跨 tile 连续；级数上限由共享内存容量决定，并作为配置维度交给 spec 003。
- **FR-004**: 必须测量下一 tile 首级加载的暴露时间；只有当它在某个 shape 集上超过 kernel 时间的 2% 时，才实施缓解。
- **FR-005**: User Story 5 的每个维度都必须先有 spec 001 的 H20 剖析证据，再实施；实施后作为配置维度交给 spec 003，并在 A/B 不达标时删除。
- **FR-006**: checklist 中已证伪（⊘）的方案（scale pair-pack、producer warp 反量化、ping-pong 与 cooperative 重排、B multicast、scale 改用 TMA）不得实施，除非有新的剖析证据，并引用对应的 fork 实验记录说明差异。
- **FR-007**: data-parallel 配置的输出必须与改动前逐比特相同；Stream-K 配置的结果必须在现有容差内。
- **FR-008**: 正确性测试必须覆盖每个合法 BM，以及 data parallel 与 Stream-K 两种模式。
- **FR-009**: 每项改动都必须在 H100 与 H20 上做同进程 A/B（含 humming MXFP4 × FP8 与 cuBLAS FP8 基线），结果写回 checklist 对应条目。
- **FR-010**: 每次配置维度变化后，必须按 spec 003 重新拟合两台设备的常数。

### Key Entities

- **mainloop 调度**：一个 K block 内加载、转换与矩阵乘的交错方式（lagged 或串行）。
- **流水线状态**：producer 与 consumer 共享的环形缓冲位置，由下标与相位表示。
- **配置维度**：本 spec 新增或改变的可选参数（级数上限、BN、每 SM CTA 数、K 切分、L2 提示），由 spec 003 的选择器取值。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有合法 BM 都能编译为 lagged 调度且无寄存器溢出，或被明确记录为不支持；正确性测试 100% 通过。
- **SC-002**: H100 上原先落在非 lagged BM 的 shape，几何平均提速不低于 5%。
- **SC-003**: tile 数少于 SM 数的 shape，两台设备都不回退，至少一台几何平均提速不低于 3%。
- **SC-004**: 每项改动合入后，H100 与 H20 的每个 shape 集几何平均不低于 1.00，没有 shape 慢 3% 以上。
- **SC-005**: User Story 5 中每个被保留的维度，在 H20 的目标 shape 集上几何平均提速不低于 3%；未达标的维度不保留，负结果全部写回 checklist。

## Assumptions

- spec 001 的测量基座与剖析能力已可用。
- spec 002 先于本 spec 合入；它改变了转换步骤的寄存器用量，因此各 BM 的寄存器预算以 spec 002 合入后的 kernel 为准。
- 新的配置维度由 spec 003 的选择器决定何时启用；在选择器合入前，A/B 以显式指定配置的方式进行。
- humming 在 H20 上采用每 SM 2 个 CTA 与较小的 tile，DeepGEMM 集成支持 BN = 64；这些只作为候选来源，是否采用以 A/B 为准。
