<!--
Sync Impact Report
- Version change: template (unversioned) -> 1.0.0
- Modified principles: none (initial ratification)
- Added principles:
  - I. 通用优化，不做 shape 特化
  - II. 可测量的性能结论
  - III. 先证据后实现
  - IV. 数值不回退
  - V. AE 可复现
- Added sections: 范围与技术约束；开发流程与质量门禁；Governance
- Removed sections: none
- Templates: plan-template.md reads the Constitution Check gates from this file at run time; no
  template edits needed. spec-template.md and tasks-template.md have no constitution references.
- Deferred TODOs: none
-->

# TensorBridge Constitution

## Core Principles

### I. 通用优化，不做 shape 特化

- 每项优化必须对任意 `(M, N, K)` 和任意 Hopper 设备都成立。
- 不得新增按精确 shape 匹配的规则或表，例如 `csrc/heuristics/sm90_tuned.hpp`、`tensorbridge/tuned/*.json`。
- 不得写死某款 GPU 的经验阈值，不得按设备名分支编写规则。
- 配置选择只能是以下三者的函数：`(M, N, K)`；设备属性（SM 数、L2 容量、smem 容量）；对每种 GPU 拟合一次的代价模型常数。
- 拟合常数必须能由仓库中的测量与拟合脚本重新生成，并注明拟合所用的 GPU 和 shape 集。

理由：按 shape 调优的规则只覆盖测过的 shape，换模型、换 TP 切分或换设备就失效，而且无法长期维护。

### II. 可测量的性能结论

- 性能结论必须来自同一进程内交替执行的 A/B 测量，用 CUDA graph 计时。
- 必须在固定 shape 集上报告：论文 Figure 11 的 40 个 shape、代价模型拟合集、held-out 集。
- 必须同时报告 H100 和 H20 的结果；只在一种 GPU 上得到的结论只能作为中间结果，不能据此合入。
- 每个 shape 集都要报告几何平均和最差 shape。
- 不接受跨节点、跨进程或不同时段测量结果之间的对比。

理由：同一 shape 在不同节点上的耗时可相差 6% 到 15%，超过多数优化本身的收益。

### III. 先证据后实现

- 每个优化 spec 必须写明它对应的 checklist ID（`docs/OPTIMIZATION_CHECKLIST.md`），并引用它所针对的瓶颈证据：ncu 指标、SASS 统计或 roofline 分析。
- checklist 中标为 ⊘（已证伪）的方案，只有在新场景（例如 H20 decode）的 profile 证明它位于关键路径上时才能重开。重开的 spec 必须引用对应的 fork 实验记录（`revollllt/humming` 的 `docs/changes/`），并说明新场景与原实验的差异。
- 实现完成后，无论收益为正还是为负，都必须把测量结果和状态写回 checklist。

理由：fork 中已有多项方案在 H100 prefill 上实测无效；不记录证据和负结果，同样的方案会被反复提出。

### IV. 数值不回退

- `tests/` 中 FPMA bridge 的逐比特测试必须全部通过。
- 改变 kernel 输出数值的改动（bridge、累加方式、Stream-K 归约、epilogue 舍入等）必须附 PPL 抽查：至少一个 dense 模型的 WikiText2 PPL（`eval/`），与改动前对比。
- 引入运行间不确定性的改动，必须在 `docs/DESIGN.md` 的「Scope」一节中注明。

理由：TensorBridge 的价值建立在「NVFP4A8 精度接近精确解码」之上，性能优化不能以不可见的精度损失为代价。

### V. AE 可复现

- 本地 tag `atc26-ae` 指向 `dd45d51`，是论文结果的复现基线，不得移动或删除。该提交中 Table 3/4 的参考值与 camera-ready 一致；更早的公开 tag `v0.1.1`（`8c1d9e9`）的 W4A8 参考值是旧的。
- 长期优化必须在分支上进行。
- 不得改写 `atc26-ae` 之前的历史；该 tag 引用的子模块提交必须在子模块仓库中保持可达。
- 若改动使论文描述失准（例如 Sec. 4.6 的配置选择、Figure 11 的数值），spec 必须注明受影响的章节。
- `atc26-ae` 目前只存在于本地；是否发布由维护者决定。

理由：artifact 已用于 ATC '26 评审，读者必须始终能从固定提交复现论文结果。

## 范围与技术约束

- 范围：`tensorbridge/`（kernel、JIT、配置选择、vLLM 插件）、`tests/`、`docs/`。`ae/` 和 `eval/` 只接受维护性修复。
- 目标硬件：Hopper（sm_90a）。参考设备为 H100 SXM（132 SM）和 H20（78 SM）。
- 优化 backlog：`docs/OPTIMIZATION_CHECKLIST.md`。代码结构：`docs/DESIGN.md`。
- 已有的 shape 特化产物属于待移除项（checklist E4），新代码不得依赖它们：`sm90_tuned.hpp`、`tuned/NVIDIA_H100_80GB_HBM3.json`，以及 `sm90.hpp` 中的经验阈值。

## 开发流程与质量门禁

- 每个 feature 按 spec-kit 流程推进：`/speckit-specify`、`/speckit-plan`、`/speckit-tasks`、`/speckit-implement`。
- 合入前必须满足以下门禁：
  1. `pytest tests` 全部通过。
  2. 按原则 II 给出 H100 和 H20 的 A/B 报告。
  3. 改变数值的改动附原则 IV 要求的 PPL 抽查。
  4. checklist 中对应条目的状态与证据，以及 `docs/DESIGN.md` 的代码地图，已同步更新。

## Governance

- 本 constitution 高于仓库中的其他开发约定。计划、spec 或代码评审与本文冲突时，以本文为准。
- 修改本文件必须单独提交，并按语义化版本升级：删除或重新定义原则为 MAJOR；新增原则或实质扩展为 MINOR；措辞澄清为 PATCH。
- `/speckit-plan` 的 Constitution Check 必须逐条核对原则 I 到 V。确有必要违反时，必须在 plan 的 Complexity Tracking 中说明理由和替代方案。
- 运行时指导文档：`docs/OPTIMIZATION_CHECKLIST.md`、`docs/DESIGN.md`。

**Version**: 1.0.0 | **Ratified**: 2026-10-28 | **Last Amended**: 2026-10-28
