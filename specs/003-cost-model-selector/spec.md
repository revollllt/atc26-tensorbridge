# Feature Specification: 基于代价模型的配置选择

**Feature Branch**: `perf`（spec 目录 `specs/003-cost-model-selector`）

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "参考 DeepGEMM 的选择器结构，用代价模型为每个 (M, N, K) 选择 tile 形状、级数、Stream-K 与光栅顺序，替代经验规则与按 shape 的调优表（架构文档 W2，checklist E2–E4）。在线开销必须很小：模型、按 (N, K, SM 数) 建的 M 区间表、热路径二分查找三层。常数按 GPU 型号用仓库内的拟合工具在 H100 与 H20 上各拟合一次。删除按 shape 的规则与表。注明论文 Sec. 4.6 与 Figure 11 使用旧选择器。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 不依赖 shape 表的配置选择（Priority: P1）

对任意 `(M, N, K)`，kernel 的配置（token tile、级数、是否切 Stream-K 尾波、tile 光栅顺序）都由一个代价模型决定。模型只依赖 shape、设备属性和按 GPU 型号拟合的常数。仓库中不再有按精确 shape 匹配的规则或调优表。

**Why this priority**: 这是 constitution 原则 I 的直接要求；现有的经验阈值与 26 个 shape 的调优表只对测过的 shape 有效，H20 上没有任何调优。

**Independent Test**: 删除全部 shape 特化产物后，在 H100 的 held-out 集上，比较模型所选配置与实测最优候选，以及与当前 `main` 的选择结果。

**Acceptance Scenarios**:

1. **Given** 仓库中已删除全部 shape 特化产物，**When** 对任意 shape 请求配置，**Then** 返回的配置只取决于 shape、设备属性和拟合常数。
2. **Given** H100 的 held-out 集，**When** 对每个 shape 实测模型所选配置与全部候选，**Then** 所选配置与最优候选的几何平均差距不超过 3%。
3. **Given** 当前 `main` 的选择逻辑（含调优表）作为对照，**When** 在全部 1105 个 shape 上做同进程 A/B，**Then** 模型选择的几何平均提速不低于 5%。

---

### User Story 2 - 在线开销可忽略（Priority: P1）

serving 时每次 GEMM 调用都要选配置，M 随批大小变化。选择必须几乎不占 host 时间，也不能在首次遇到某个 M 时阻塞请求。

**Why this priority**: 现有的模型实现冷求值一次要 187 到 191 µs（132 SM），对一个 `(N, K)` 预计算全部 M 要约 1.5 s；而 decode 阶段单个 GEMM 只有几 µs。参考 DeepGEMM 的做法，模型本身要轻，且结果要提前算好。

**Independent Test**: 在加载一个模型时测量建表总时间；在 serving 的热路径上测量每次选择的 host 耗时。

**Acceptance Scenarios**:

1. **Given** 一个 `(N, K)` 在加载阶段注册，**When** 为它建立 M 区间表，**Then** 建表不超过 50 ms，表项不超过 100 个。
2. **Given** 区间表已建立，**When** 热路径上按 M 选择配置，**Then** 每次选择不超过 1 µs，且不求值模型。
3. **Given** 区间表的 M 范围内的任意 M，**When** 比较查表结果与逐 M 精确求值的结果，**Then** 两者所选配置的预测延迟相差不超过 1%。
4. **Given** M 超出区间表的范围，**When** 请求配置，**Then** 直接求值模型并记住结果，结果正确且不报错。

---

### User Story 3 - 为新的 GPU 型号拟合一次（Priority: P2）

维护者在一种新的 Hopper GPU（首先是 H20）上，用仓库内的工具测量一批 shape 与配置、拟合模型常数，把常数加入仓库，此后该型号的选择不再需要任何人工规则。

**Why this priority**: H20 的硬件比例与 H100 差别很大，直接套用 H100 的常数或阈值不可靠；这是 H20 调优的主路径。

**Independent Test**: 在 H20 上按文档完成一次拟合，用 held-out 集验证，并与当前 `main` 在 H20 上的选择做 A/B。

**Acceptance Scenarios**:

1. **Given** 一台 H20 与文档化的步骤，**When** 维护者完成测量与拟合，**Then** 得到一组可提交的常数，并附带所用 shape 集、GPU 型号与拟合误差。
2. **Given** H20 的常数已入库，**When** 在 H20 上与当前 `main` 的选择做同进程 A/B，**Then** 全部 1105 个 shape 的几何平均提速不低于 5%，held-out 集上与最优候选的差距不超过 3%。
3. **Given** 一种尚未拟合的 GPU 型号，**When** 请求配置，**Then** 系统用设备属性推导的项加最接近型号的常数做选择，并只打印一次警告。

---

### User Story 4 - 候选维度随 kernel 演进（Priority: P3）

spec 002 删除了 scale 预广播变体，spec 004 会增加新的配置维度（如 BN = 64、每 SM 2 个 CTA、CTA 内 K 切分）。每次 kernel 的配置维度变化后，模型、拟合工具与候选枚举同步更新并重新拟合，新维度只在模型预测更优时被选中。

**Why this priority**: 选择器是各项 kernel 优化发挥作用的入口；但它依赖这些优化先落地。

**Independent Test**: 在某个新维度合入后重新拟合，检查 User Story 1 与 3 的验收在两台设备上仍然成立。

**Acceptance Scenarios**:

1. **Given** 一个新的配置维度已合入 kernel，**When** 更新候选与模型并重新拟合，**Then** 两台设备上 held-out 集与最优候选的差距仍不超过 3%。

---

### Edge Cases

- 多个候选的预测延迟在模型误差以内：采用确定的平局规则（例如级数更少者优先），保证同一输入总是得到同一配置。
- 某个候选在当前设备上不合法（共享内存或寄存器不足）：枚举时排除，不进入求值。
- 进程内可用 SM 数小于物理 SM 数（如资源被分区）：区间表按实际可用 SM 数建立。
- 多个线程同时注册同一个 `(N, K)`：只建一次表，其余调用等待并复用结果。
- 强制指定配置（实验或测试）：仍可显式传入配置，绕过选择器。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 配置选择的结果只能取决于 `(M, N, K)`、设备属性（SM 数、L2 容量、共享内存容量）与按 GPU 型号拟合的常数。
- **FR-002**: 必须删除全部按 shape 的规则与表：`sm90_tuned.hpp`、`tuned/*.json`，以及 `sm90.hpp` 中写死 H100 数值的经验阈值；`autotune.py` 改为拟合流程中的测量工具，不再产出供运行时读取的表。
- **FR-003**: 候选集合必须由 kernel 支持的配置维度生成，只包含在当前设备上合法的配置；求值前允许用闭式下界剪掉明显劣势的候选。
- **FR-004**: 每个 `(N, K, 可用 SM 数)` 第一次注册时（加载权重或显式预热时），必须建立覆盖到配置上限 M（默认 16384）的 M 区间表；超出范围的 M 直接求值并记忆。
- **FR-005**: 热路径上的配置选择不得求值模型，只做区间查找。
- **FR-006**: 区间表在其范围内的选择，与逐 M 精确求值所选配置的预测延迟相差不得超过 1%。
- **FR-007**: 每组拟合常数必须能由仓库中的测量与拟合工具重新生成，并记录所用 GPU 型号、shape 集与拟合误差；H100 与 H20 各一组。
- **FR-008**: 对未拟合的 GPU 型号，必须退回到「设备属性推导项 + 最接近型号的常数」，并在进程中只警告一次。
- **FR-009**: 必须提供评估工具：在指定 shape 集上实测模型所选配置与全部候选，报告与最优候选的差距（几何平均与最差 shape）。
- **FR-010**: 显式传入配置的调用必须绕过选择器，行为与现在相同。
- **FR-011**: `docs/DESIGN.md` 必须更新为新的选择机制；本 spec 必须注明论文 Sec. 4.6 描述的是旧的 wave-fill 预测器、Figure 11 用旧选择器测得，复现以 tag `atc26-ae` 为准。

### Key Entities

- **配置**：kernel 一次启动所需的全部可选参数（token tile、级数、Stream-K、光栅顺序，以及后续 spec 增加的维度）。
- **代价模型**：给定 shape、配置与设备常数，返回预测延迟的函数。
- **设备常数集**：按 GPU 型号拟合得到的一组模型常数，附拟合来源与误差。
- **区间表**：某个 `(N, K, 可用 SM 数)` 下「M 区间 → 配置」的有序列表，在加载时由模型推导。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: H100 与 H20 的 held-out 集上，模型所选配置与实测最优候选的几何平均差距不超过 3%，最差 shape 不超过 10%。
- **SC-002**: 与当前 `main` 的选择（含调优表）相比，全部 1105 个 shape 的几何平均提速在 H100 上不低于 5%、在 H20 上不低于 5%；论文 40 个 shape 的几何平均在两台设备上都不低于 `main`。
- **SC-003**: 热路径上每次选择不超过 1 µs；每个 `(N, K)` 建表不超过 50 ms，表项不超过 100 个。
- **SC-004**: 区间表范围内，查表与逐 M 精确选择的预测延迟差不超过 1% 的 M 占 100%。
- **SC-005**: 为一种新的 GPU 型号完成测量与拟合，不超过 8 个 GPU 小时，且全程按文档执行、无需修改代码。

## Assumptions

- spec 001 的测量基座已可用，用于拟合测量、held-out 评估与 A/B。
- 起点是 lab-H100 上 `unified-scheduler` worktree 中的延迟模型与拟合工具（`tensorbridge-ae-sched`，未提交）；它此前在 1105 个 shape 上比旧启发式快 7.0%，held-out 695 个快 9.7%，论文 Figure 11 脚本上持平。
- 配置维度会随 spec 002 与 spec 004 变化；常数在每次维度变化后重新拟合，本 spec 的验收以最后一次拟合为准。
- 配置上限 M 取 16384，覆盖 vLLM 默认的单批最大 token 数；可在注册时调整。
- MoE 的分组配置（以每个专家的期望行数为输入）由 spec 006 在本 spec 的结构上扩展。
