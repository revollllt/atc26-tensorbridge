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
- 性能报告必须包含以下基线，在相同 shape 上与 TensorBridge 同一进程内交替测量：
  - humming（`vllm-project/humming` 上游）的 MXFP4 权重 × FP8 激活模式，为重点对比基线；报告中记录 humming 的提交号。
  - cuBLAS FP8。

理由：同一 shape 在不同节点上的耗时可相差 6% 到 15%，超过多数优化本身的收益。humming 的 MXFP4 × FP8 是与 TensorBridge 定位最接近的公开实现，只和自己比无法说明优化是否到位。

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
- 长期优化统一在远程分支 `perf` 上进行，不得直接改动 `main`。`main` 只接收从 `perf` 合入、且通过全部门禁的改动。
- 不得改写 `atc26-ae` 之前的历史；该 tag 引用的子模块提交必须在子模块仓库中保持可达。
- 若改动使论文描述失准（例如 Sec. 4.6 的配置选择、Figure 11 的数值），spec 必须注明受影响的章节。
- `atc26-ae` 目前只存在于本地；是否发布由维护者决定。

理由：artifact 已用于 ATC '26 评审，读者必须始终能从固定提交复现论文结果。

### VI. 代码规范

以下八条是硬性要求，不是建议。写代码前先读本节，写完后逐条自查。适用于新增或修改的 Python 与 CUDA C++ 代码；既有代码在被修改时一并改到符合。

1. **不拆碎函数。**
   - 只被调用一两次的短 helper，内联到调用处，不单独抽出。
   - 只有两种情况才抽函数：它真正在多处复用；或它代表一个有独立含义的完整概念。
   - 「让主函数看起来短」不是理由。一个线性可读的 60 行函数，好过 6 个需要来回跳转的 10 行函数。
2. **不滥用下划线前缀。**
   - 只有定义在另一个函数内部的函数可以以 `_` 开头。
   - 普通变量、模块级函数、类属性都以普通字母开头，可读性优先。
   - 不要为了标记「内部」给所有东西加 `_`。
3. **不写过度防御的代码。**
   - 不为不可能发生的情况写 try/except。只在真实的失败边界处理异常：I/O、外部输入、第三方调用、数值退化。
   - 不用 try/except 掩盖逻辑错误。不该到达的分支，就让它抛出。
   - 单元测试要简洁有力，覆盖函数真实的失败模式；不为每个防御分支写一个用例，不写只为提升覆盖率的测试。
4. **常量定义在使用处。**
   - 只被一个文件使用的常量，直接定义在该文件中。
   - 只有真正共享的常量才放进公共模块。去掉无意义的跨文件 import。
5. **不用 `Any` 逃避类型检查。**
   - 不得为了通过 pre-commit、mypy 或其他类型检查而使用 `Any`。
   - 类型难写，说明设计需要收紧：使用具体类型、`TypedDict`、`Protocol`、泛型或 `Union`，而不是 `Any`。
   - 每个函数签名的参数和返回值都带真实的类型注解。
6. **名字要有真实含义。**
   - 变量名必须表达其物理或数学含义，不用 `tmp`、`data2`、`res`、`x1`。
   - 张量和数组的名字要让语义可见（如 `attn_logits`、`kv_cache_len`）；必要时在注释中注明 shape 和 dtype，不让读者从代码里反推。
7. **分支穷尽，但不堆叠。**
   - 每个 `if` 都有对应的 `else`，或显式的提前 return / raise，不留隐式贯穿路径。
   - 同时避免长 if/elif/else 链：在函数入口一次性做类型和数据校验，函数体保持扁平。
   - 在边界尽早失败，函数体内不再重复校验。
8. **不用动态类型技巧。**
   - 不用 `getattr` / `hasattr` / `setattr` 探测或嗅探属性。
   - 对象有哪些属性，应能从它的类型定义静态得知。可选行为用显式字段、`Protocol` 或多态表达，而不是运行时检查。

理由：本仓库要长期维护，kernel 与 host 代码需要能被直接读懂和审查；这些约束让代码结构可预期，并让类型检查和代码评审真正起作用。

## 范围与技术约束

- 范围：`tensorbridge/`（kernel、JIT、配置选择、vLLM 插件）、`tests/`、`docs/`。`ae/` 和 `eval/` 只接受维护性修复。
- 目标硬件：Hopper（sm_90a）。参考设备为 H100 SXM（132 SM）和 H20（78 SM）。
- 优化 backlog：`docs/OPTIMIZATION_CHECKLIST.md`。代码结构：`docs/DESIGN.md`。
- 已有的 shape 特化产物属于待移除项（checklist E4），新代码不得依赖它们：`sm90_tuned.hpp`、`tuned/NVIDIA_H100_80GB_HBM3.json`，以及 `sm90.hpp` 中的经验阈值。

## 开发流程与质量门禁

- 每个 feature 按 spec-kit 流程推进：`/speckit-specify`、`/speckit-plan`、`/speckit-tasks`、`/speckit-implement`。
- 所有改动在远程分支 `perf` 上进行（见原则 V）。
- 合入前必须满足以下门禁：
  1. `pytest tests` 全部通过。
  2. 按原则 II 给出 H100 和 H20 的 A/B 报告，含 humming MXFP4 × FP8 与 cuBLAS FP8 基线。
  3. 改变数值的改动附原则 IV 要求的 PPL 抽查。
  4. checklist 中对应条目的状态与证据，以及 `docs/DESIGN.md` 的代码地图，已同步更新。
  5. 代码已按原则 VI 逐条自查。

## Governance

- 本 constitution 高于仓库中的其他开发约定。计划、spec 或代码评审与本文冲突时，以本文为准。
- 修改本文件必须单独提交，并按语义化版本升级：删除或重新定义原则为 MAJOR；新增原则或实质扩展为 MINOR；措辞澄清为 PATCH。
- `/speckit-plan` 的 Constitution Check 必须逐条核对原则 I 到 VI。确有必要违反时，必须在 plan 的 Complexity Tracking 中说明理由和替代方案。
- 运行时指导文档：`docs/OPTIMIZATION_CHECKLIST.md`、`docs/DESIGN.md`。

**Version**: 1.1.0 | **Ratified**: 2026-10-28 | **Last Amended**: 2026-10-28
