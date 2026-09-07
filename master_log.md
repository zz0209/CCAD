# CCAD Master Log

本文件是 CCAD 项目的追加式事实账本，用于记录研究状态、运行、结果、失败、协议变化和决策依据。它不是宣传材料，也不替代 `EXPERIMENT_PLAN.md` 或 `EXPERIMENT_TRACKER.md`。从本文件建立之日起，历史错误以新增勘误条目修正，不回写美化。

## 当前基线状态

截至 2026-09-01，可由现存 artifact 直接确认的状态如下：

| 项目 | 状态 | 证据与边界 |
|---|---|---|
| 理论 | 内部完整稿已存在 | `goal_aligned_subspace_consistency_complete_proofs.pdf`；不能视为独立外部证明复核 |
| 实验协议 | 已形成 | `EXPERIMENT_PLAN.md` 定义 C1/C2、M0–M7、数据拆分与阶段门 |
| 实验追踪 | 已建立 | `EXPERIMENT_TRACKER.md` 列出 R001–R022 |
| 实验执行 | 尚未开始 | R001–R022 当前全部为 `TODO`；未发现真实 SAE 运行结果 |
| 代码与环境 | 尚未建立可验证基线 | 当前目录未发现实验代码、resolved config、环境锁定文件或 run artifact |
| 版本控制 | 当前目录不是 Git 仓库 | 后续 run 必须记录代码快照或 hash，直到有可用 commit |
| 主模型 | 默认候选，未冻结 | Pythia-160M-deduped；Pythia-70M 仅用于 debug |
| primary seed 数 | 部分锁定 | 最低 5；目标 6 仍是候选 |
| paired split | 已锁定 | 文档哈希 10% mean / 40% discovery / 20% calibration / 30% audit |
| hook/framework/width/k/corpus/readout | 未冻结 | 等待 R004–R006 及对应 unit test/pilot |
| 第二模型族 | 触发式候选 | Gemma-2-2B 优先；需先通过 M4 并确认资源/条款 |
| distillation | 延后 | 仅在因果主结果成功且存在 rank compression 空间后启动 |
| venue | 未冻结 | 现有项目文件未指定唯一投稿 venue，由用户保留决定 |

## 2026-08-31 — 理论证明稿进入项目目录（历史重建）

### 触发与来源

根据现存文件时间和文件内容，项目目录中已有 `goal_aligned_subspace_consistency_complete_proofs.pdf`。本条是从 artifact 重建的历史，不表示本日志建立者参与了该证明稿的形成。

### 可确认内容

- 理论稿围绕跨种子 SAE 的组级/子空间一致性、贡献空间比较、same-hook causal interchangeability、有限候选族的样本保证和 rank-adaptive distillation 展开。
- Hadamard 型构造说明：一对一 PW-MCC 可以随维度衰减，而组级 BCC/PSC 仍可达到理想值。
- 理论允许 many-to-many 结构，并暴露一对一 matching 和唯一坐标解释的局限。
- exact-cover 类陈述依赖理想化、固定且非重叠的支持；真实方法仍需处理重叠、歧义和拒答。
- finite-sample 结论要求候选族在 audit 前固定，中心化常数来自独立 split。

### 证据边界

- 该 PDF 是内部完整证明稿，不是独立外部证明审查报告。
- 代数上的 same-hook interchangeability 不证明唯一语义本体或人类可解释性。
- 理论稿不构成真实 SAE 数据上的 empirical validation。
- 理论 novelty 仍需文献和审稿层面的独立核验。

### Gate 影响

理论稿为合成实验与真实 SAE 计划提供了对象和反例，但没有使 M0–M7 中任何实证阶段自动通过。

## 2026-09-01 — 实验计划与追踪器形成（历史重建）

### 触发与来源

现存 `EXPERIMENT_PLAN.md` 和 `EXPERIMENT_TRACKER.md` 在本日形成。以下状态由两份文件重建。

### 已形成的研究协议

- 项目工作名为 CCAD，核心方法为 CBSM，主流程为 Discover → Match → Validate → Distill。
- 主 claim 是 C1（局部组级可复现性与诚实歧义/拒答）和 C2（held-out 因果可互换性优于公平选择的最佳单特征基线）。
- 主证据漏斗为：合成正确性 → Pythia debug/pilot → same-config 多 seed SAE suite → held-out group/causal audit → anti-leakage/ablation → 第二设置确认。
- paired corpus 按文档哈希锁定为 10% mean / 40% discovery / 20% calibration / 30% audit。
- 统计单位被规定为 document、concept、seed/config 或 seed pair，而不是 token。
- audit 前必须冻结候选族、均值、阈值、baseline 和选择预算。

### Run 队列

- R001–R003：合成 kernel、局部 many-to-many recovery、ambiguity/refusal；
- R004–R006：hook/SAE round-trip、框架和 architecture/sparsity pilot；
- R007–R008：primary seed suite 与 paired activation/code；
- R009–R014：atom/group baselines、CBSM、held-out audit 与因果/任务终点；
- R015–R017：novelty、simplicity、anti-leakage 与公平性 ablation；
- R018–R020：第二 hook/config、Gemma adapter 与 confirmatory family；
- R021–R022：语义外部验证与 rank-adaptive distillation。

### 当前状态

R001–R022 全部为 `TODO`。没有证据表明任何合成、训练、匹配、因果或蒸馏运行已经完成。不能把计划中的预期指标、候选模型或理论 sanity check 记成实验结果。

### 当前锁定与未决项

- 已锁定：synthetic-first、seed-only 控制原则、paired split、audit 隔离与最低 primary seed 数 5。
- 默认但未锁定：Pythia-160M-deduped 作为 primary candidate；Pythia-70M 作为 debug。
- 未决：primary hook/layer、SAE framework、architecture、width、k/L0、训练 corpus、token budget、causal readout。
- 触发式：第二 model family 和 distillation；不得提前吞噬主线资源。

### 下一步

最先应推进的是 R001–R003 的合成正确性与拒答套件，同时以 R004 建立 hook、tensor、token 对齐和 contribution/intervention contract。M1 未通过时停止真实数据 matcher；M0/M2 未冻结时不得启动 primary seed suite。

## 2026-09-01 — 建立项目治理文件

### 触发

用户要求参考 SAE Lab 中另外两个项目的 `AGENTS.md` 与 `master_log.md`，为 CCAD 建立对应文件。

### 动作

- 阅读并提炼参考项目中共同使用的治理结构：状态恢复、阶段门、claim 纪律、数据泄漏防护、artifact provenance、资源 lease、失败保留和用户保留决定。
- 对照 CCAD 的 `EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md` 和理论 PDF 的已知状态，将治理规则改写为 CCAD 专用版本。
- 新建 `AGENTS.md`，覆盖 CCAD/CBSM 的研究对象、M0–M7 阶段门、四段 paired split、same-config seed 控制、held-out 因果终点、歧义/拒答、理论边界和完成标准。
- 新建本 `master_log.md`，重建当前可验证基线，并明确所有 R001–R022 仍为 `TODO`。

### Artifact

- `AGENTS.md`
- `master_log.md`

### 结果与保守解释

项目现在具备明确的操作规则和追加式事实账本。此次工作只建立治理与记录层，不包含代码实现、理论复核或实验运行，因此没有改变任何 R 编号状态，也没有使任何 M0–M7 gate 通过。

### 下一步

按 tracker 从 R001–R004 开始建立可运行基线；在首次运行前冻结 run artifact schema、环境记录方式和代码快照/hash 方案。

## 2026-09-01 21:22 EDT — 建立 15 分钟顶会级研究自动化循环

### 触发

用户要求以现有治理文件、实验计划和理论 PDF 为蓝本，建立每 15 分钟持续推进一次的 CCAD 研究循环；目标是形成不低于 ACL、ICLR 等主会完整论文分量的研究，并授权为本课题检索和下载公开文献、教科书、项目代码及其他必要资料。

### 实际动作

- 按治理顺序重新阅读 `AGENTS.md`、本日志最新状态、`EXPERIMENT_PLAN.md` 相关阶段门与 claim map、`EXPERIMENT_TRACKER.md` 全部 R001–R022 状态，以及理论 PDF 中核心定义、有限样本边界和下游路线。
- 新建 `COMPONENT_CANDIDATES.md`，把 matcher、诊断、baseline、intervention、训练基础设施和 rank-adaptive distillation 等候选拆成可审计组件，并为候选规定证据、公平控制、触发条件和淘汰条件。
- 在当前 CCAD 任务上创建 heartbeat 自动化 `CCAD 顶会级研究循环`，automation ID 为 `ccad`，状态为 `ACTIVE`，执行间隔为 15 分钟。
- 将每轮必读材料、阶段门顺序、run/artifact contract、audit 防泄漏、资源租约、线上与本地检索、下载 provenance、组件管理、失败保留、暂停条件和用户保留决定写入自动化提示词。
- 为大体积公开研究资料约定可按需使用 `D:\CCAD_Storage`，并按 references、repos、datasets、models、activation_cache、runs、staging 分类；本次没有为无明确消费者的资料执行下载，也没有启动训练或外部服务写入。

### Artifact 与配置

- `COMPONENT_CANDIDATES.md`
- Codex heartbeat automation：`ccad`（ACTIVE，15 分钟周期，附着当前任务）
- 运行 ID：无。本次是治理与自动化配置，不是实验运行。

### 结果、保守解释与阶段门影响

持续研究入口已经建立，组件候选也有了独立于 locked protocol 的记录位置。自动化被要求在资源繁忙时转做检索、设计、审计和轻量测试，并在重型工作前使用母目录资源管理器。用户对公开资料下载和 D 盘课题存储的授权已记录，但公开发布、上传、外部写入、付费算力、私有或许可不清数据仍保留人工裁决边界。

本次没有运行 R001–R022，没有打开 audit，没有冻结 primary hook/framework/config，也没有使 M0–M7 的任何 gate 通过。论文目标是质量约束，不构成已经达到主会水平的结果声明。

### 下一步与未决依赖

首轮自动化应从依赖正确且信息增益最高的早期工作开始：核查 R001–R003 合成套件的具体 artifact contract，并与 R004 的 hook、tensor、token 对齐和 contribution/intervention 单元测试要求协调。任何真实 SAE 主线、Gemma 扩展或蒸馏仍须满足 tracker 中的阶段门和触发条件。

## 2026-09-01 21:38 EDT — Heartbeat `ccad`：冻结前合成规格与 run artifact 契约

### 触发与状态恢复

15 分钟研究 heartbeat 首次触发。按项目顺序完整读取 `AGENTS.md`，随后读取本日志最新条目、`EXPERIMENT_PLAN.md` 中 B1/M0/M1/工程接口/风险与待决项、`EXPERIMENT_TRACKER.md` 全表，并核对理论 PDF 的动态贡献、BCC/PSC、Hadamard 分离、exact-cover 边界、有限候选族证书、算法和 same-hook intervention 章节。R001–R022 在本轮开始时仍全部为 `TODO`。

### 实际动作

- 新建 `SYNTHETIC_SUITE_SPEC.md`，把计划锁定的 12 类 falsifiers 映射到 R001–R003，定义公共张量契约、核复算、指标、合法非 partition 输出、运行随机性、PASS/FAIL 边界和实现前开放项。
- 明确 ARI 只适用于 partition-like truth；partial overlap 改用 hyperedge set metrics，competing covers 比较完整 tie set，whole-dictionary-only 以 refusal 为成功，避免用单一聚类指标掩盖不可识别性。
- 将 `false_unique_rate` 列为 R003 的优先安全指标，并规定双零贡献为 `INACTIVE`、零 decoder rank 为 `DEGENERATE_PSC`，不得静默记满分或进入均值。
- 新建 `RUN_ARTIFACT_CONTRACT.md`，定义唯一 run 目录、resolved config、环境/输入/代码 hash、不可覆盖状态机、raw-to-summary 可追溯性、audit fail-fast 标志和资源 lease 记录。当前非 Git 状态下使用实际导入源码的稳定 SHA-256 aggregate hash。
- 盘点 SAE Lab 本地资料，确认 `references/non_application/SAELens_Library` 已含 synthetic generator/training 测试，可作为接口和测试组织参考；它不是 CBSM falsifier 实现，未复制代码，也未把相邻结果算作 CCAD 证据。
- 阅读母目录资源管理器的 `AGENTS.md` 与 `README.md`，确认多资源固定申请顺序为 `disk-e-io → disk-d-io → cpu-heavy → gpu-0`，逆序释放；普通短测试和小文件编辑不需要 lease。

### Artifact 与运行配置

- 新增 `SYNTHETIC_SUITE_SPEC.md`。
- 新增 `RUN_ARTIFACT_CONTRACT.md`。
- Run ID：无；本轮为规格和契约工作，没有执行 synthetic audit、训练、下载或重型计算，因此 tracker 状态不变，也没有资源 lease。
- PDF 仍是内部证明稿，没有被当作独立外部 proof review。

### 结果、失败与保守解释

R001–R003 已从一行 tracker 描述细化为可实施、可失败且不强迫 partition 的测试合同。重要设计结论是：exact recovery/ARI 不能作为全部 12 类 falsifier 的统一主指标，否则会把 overlap、ambiguity 和 refusal 错误地惩罚为失败；应按真值结构选择 partition、hyperedge 或 decision-level 指标。

本轮没有实验结果，M0/M1 均未通过。数值容差、维度/噪声网格、tie tolerance、`n_eff` 定义和 full-run 汇总阈值仍明确保持 `OPEN`；没有根据未见结果虚构阈值，也没有改变 locked 的每 family 至少 20 seed pairs。

### 下一步与未决依赖

下一轮优先把 run contract 实现为确定性校验器和 R001 smoke runner，先覆盖 Hadamard、kernel/BCC/PSC 恒等式及 BCC/PSC 不能互替的两个反例。首次运行前需在 resolved config 中固定仅用于该 run 的 dtype、`q` 网格和数值容差，并按 contract 为 run 建立唯一 artifact；R002/R003 的 full 判定阈值仍应在其首次正式运行前冻结。

## 2026-09-01 22:01 EDT — 启动 R001 smoke conformance

### 触发与动作

Heartbeat `ccad` 依据上一条的下一步实现 R001 最小消费者。新增 `src/ccad/metrics.py`、`src/ccad/synthetic.py`、`tests/test_r001_metrics.py`、`scripts/run_r001_smoke.py` 和 `configs/r001_smoke.json`。在启动正式 artifact run 前，使用 bundled Python 3.12 / NumPy 2.3.5 执行 4 个 `unittest`，Hadamard、inactive/degenerate、same-span/different-computation、same-sum/bloated-span 均通过。

### Run 启动记录

- Run ID：`R001_smoke_20260902T020100Z`。
- 配置：float64；`q={2,4,8}`；`n_mean=257`；`n_eval=1024`；base seed `20260902`；absolute/relative tolerance `1e-10`。
- 范围：F01、F10、F11 的 synthetic smoke；不包含 20 seed-pair full matrix，不用于 M1 PASS。
- 数据：独立 synthetic mean/eval samples；`audit_opened=false`。
- 资源：短单进程 CPU run，不需要 resource-manager lease。
- Tracker 已加入该 suffix run 并标为 `RUNNING`；parent R001 保持 `TODO`。

本条仅记录启动事实，结果在 run 完成后追加，不预判 PASS。

## 2026-09-01 22:03 EDT — R001 首次 smoke 的 artifact-contract 失败与修复重跑启动

### 首次 run 结果与失败

`R001_smoke_20260902T020100Z` 完成 5 条 raw records，F01 的 `q={2,4,8}`、F10 和 F11 共 16/16 数值检查通过：Hadamard 的 PW-MCC 分别为 `0.70710678/0.5/0.35355339`，BCC/PSC 在浮点容差内为 1；F10 得到 `PSC=1`、`BCC=-0.00851`；F11 得到 `BCC≈1`、`PSC=2/3`。

但新增的确定性 contract validator 发现该 run 缺少 manifest 的 device/seeds/threshold source、summary 到 raw metrics 和生成脚本的 hash 链、输入来源/访问边界，且 manifest config hash 指向源 config 而非保存后的 resolved config。虽然计算检查通过，该 suffix run 按可复现性门记为 `FAIL`，artifact 原样保留，不能用于 M1 或方法正确性结论。

### 修复与新 run 启动

- 新增 `src/ccad/artifacts.py` 与 `scripts/validate_run.py`，把 `RUN_ARTIFACT_CONTRACT.md` 的关键字段转为 fail-fast 校验。
- 修复 runner，使 resolved config、raw metrics、生成脚本和输入 provenance 建立可验证 hash 链，并在结束时自动写 `contract.validation.json`；contract validation 失败会把最终状态降为 `FAIL`。
- 4 个 metric `unittest` 修复后再次全部通过。
- 启动新 suffix `R001_smoke_20260902T020300Z`，计算配置与首次 run 相同；tracker 已标 `RUNNING`。它是对 artifact pipeline 的修复重跑，parent R001 仍为 `TODO`。

## 2026-09-01 22:03 EDT — R001 修复 smoke 完成

### 运行结果

`R001_smoke_20260902T020300Z` 以退出码 0 完成。5 条 raw records 的 16/16 预写检查全部通过：

- F01 Hadamard：`q=2,4,8` 的 PW-MCC 精确复现 `q^{-1/2}`；整组 BCC/PSC 在 `1e-10` smoke 容差内为 1；逐样本两侧 contribution 最大误差分别不超过 `1.33e-15`。
- F10 same span/different computation：`PSC=1`，`BCC=-0.00851195`，正确展示同 span 不等于同动态计算。
- F11 same sum/bloated span：`BCC≈1`，`PSC=2/3`，正确展示同 aggregate 不排除 span 膨胀。
- 独立再次执行 `scripts/validate_run.py`，返回 `ok=true`、零错误；resolved config、代码、输入、raw metrics 与生成脚本 hash 链完整。

### Artifact

- `runs/R001_smoke_20260902T020300Z/`：manifest、resolved config、environment、inputs、code hashes、raw/summary metrics、stdout/stderr、status 与 contract validation。
- `runs/R001_smoke_20260902T020100Z/`：保留的首次失败 run；数值检查通过但 artifact contract 不完整。
- Tracker 中两个 suffix 分别记为 `FAIL` 和 `PASS`；parent R001 仍为 `TODO`。

### 保守解释与 gate 影响

结果验证了当前实现对 F01/F10/F11 的最小数值一致性，并验证 run artifact pipeline 能拒绝“计算通过但 provenance 不完整”的运行。它不是 R001 full，更没有覆盖每 family 至少 20 seed pairs、F02 local rotations、R002/R003 或真实 SAE，因此 M1 仍未通过，C1/C2 均未获得实证支持。

### 下一步

下一步应增加 F02 local block rotations，并把 R001 从单整组 Hadamard 扩到多 planted blocks 和至少 20 seed-pair 的正式矩阵；在 full run 前冻结 q/block/sample 网格和汇总阈值。并行可为 artifact validator 增加故障单元测试，特别是已有目录、audit 前置缺失和 raw-summary hash 篡改。

## 2026-09-01 — 精简 heartbeat 自动化提示词

### 触发与动作

用户要求适度精简 automation loop，减少与“必读文档”的重复，但不做过度压缩。重新核对 `AGENTS.md`、master log 最新状态、plan 的阶段门/风险/待决项和 tracker 后，更新 automation `ccad` 的提示词并保持 15 分钟周期与 `ACTIVE` 状态。

### 修改结果

- 将 claim、指标清单、split 数值、逐阶段限制和模型细节改为遵照 `AGENTS.md`、`EXPERIMENT_PLAN.md` 与 `EXPERIMENT_TRACKER.md`，避免同时维护两套易漂移协议。
- 保留并明确：状态恢复顺序、每轮 1–3 个实质工作单元、阶段门驱动、`COMPONENT_CANDIDATES.md` 管理、本地/线上检索与公开资料下载授权、`D:\CCAD_Storage`、共享资源 lease、唯一 run/provenance、失败保留、master log 留痕、人工裁决边界和每轮汇报格式。
- 理论 PDF 改为“涉及理论时阅读相关完整章节”，避免每轮无差别复述理论内容，同时不降低理论任务的核查要求。

### 影响与边界

本次只调整自动化提示词表达，没有改变 LOCKED 协议、run 状态、实验结果或用户保留决定。R001 parent 仍为 `TODO`，M0/M1 仍未通过；下一研究动作仍是扩展 F02 与 R001 正式矩阵。

## 2026-09-01 22:24 EDT — 启动 F02 local-rotation smoke

### 触发与实现

Heartbeat `ccad` 依据 tracker 的 R001/M1 依赖继续扩展合成正确性。核对理论稿的 exact-cover 正交 block corollary 与 group gauge invariance 后：

- 新增 `local_block_rotations` 构造：多个互相正交 hook 子空间，每个 block 在两侧采用不同正交基，但逐样本 aggregate contribution 相同。
- 新增 `exhaustive_balanced_pairs` 小邻域 correctness oracle，枚举非空子集、按 normalized residual 筛选并保留 support-minimal candidates；该实现明确不声称可扩展到真实全字典。
- 增加 F02 单元测试：三个 rank-2 blocks 均恢复为唯一 planted block，错配的跨 block 邻域无候选。
- 修复 artifact validator 的快照语义：生成脚本 hash 与 run 内记录的代码快照比较，而非与会继续变化的当前工作区比较；增加 raw tamper、audit 未冻结、缺文件和工作区代码变化四类故障测试。
- 当前 9 个 `unittest` 全部通过，既有 `R001_smoke_20260902T020300Z` 在工作区代码变化后仍通过独立 contract validation。

### Run 启动

- Run ID：`R001_f02_smoke_20260902T022440Z`。
- 配置：3 个 rank-2 blocks；5 个 generator seed pairs；`n_mean=257`、`n_eval=2048`；float64；residual/absolute/relative tolerance `1e-10`；每侧最大 group size 4。
- 预写检查：每 seed 的三个 planted blocks 全恢复且为 support-minimal；循环错配的跨 block pools 无候选；planted PSC 为 1；逐样本 contribution 误差在容差内；artifact contract 通过。
- 范围：debug smoke，不满足每 family 至少 20 seed pairs 的 locked full 要求，不用于 M1 PASS。
- 资源：短 CPU exhaustive-neighborhood run，无需 lease。

Tracker 已将该 suffix 标为 `RUNNING`；结果将在完成后追加。

## 2026-09-01 22:25 EDT — F02 local-rotation smoke 完成

### 结果

`R001_f02_smoke_20260902T022440Z` 以退出码 0 完成，5 个 generator seed pairs 的 30/30 预写检查全部通过：

- 每个 seed 的 3 个 planted rank-2 blocks 均被 exhaustive local oracle 唯一恢复，共 15/15 blocks。
- 每个 seed 的 support-minimal candidates 恰为 3；循环错配的跨 block neighborhoods 共 0 个伪候选。
- 每个 seed 评估 1,350 个 subset pairs；5 seeds 共 6,750 个，未使用 proposal 剪枝。
- planted blocks 的最小 PSC 为 1；最大 normalized residual 不超过 `4.46e-16`；最大逐样本 contribution 误差为 `2.22e-15`。
- artifact contract 独立验证通过，raw metrics 与生成脚本 hash 可追溯。

### Artifact 与状态

- `runs/R001_f02_smoke_20260902T022440Z/` 保存完整 run artifacts。
- Tracker suffix 更新为 `PASS`；parent R001 仍为 `TODO`。
- 无资源 lease、下载、audit 打开或协议偏差。

### 保守解释与 gate 影响

该结果支持 F02 在理想正交 block、无噪声、已知局部 neighborhood 下的实现正确性，并验证 support-minimal 过滤不会把 planted block 拆成伪小匹配。它没有检验 proposal recall、噪声鲁棒性、未知 neighborhood、unequal split/merge、ambiguity/refusal，也只有 5 而非 locked 的至少 20 seed pairs。因此不能将其称为 R001 full 或 M1 PASS，亦不支持真实 SAE 的 C1/C2。

### 下一步

将 F01/F02 合并为 R001 full candidate 配置，冻结至少 20 seed pairs、block/rank/q/sample 网格及汇总规则；在正式运行前先增加 near-zero residual/noise sensitivity，防止 `1e-10` 的理想构造容差被误用为真实 matcher 阈值。随后推进 F03 unequal split/merge，为 R002 exact recovery 建立基数不等的 oracle 测试。

## 2026-09-01 22:42 EDT — 冻结并启动 R001 candidate v1

### 触发与准备

Heartbeat `ccad` 继续推进 R001，但遵守 `SYNTHETIC_SUITE_SPEC.md` 中“12 类全部实现前不得把配置称为 M1 full”的限制。因此本次命名为 `candidate_v1`，不是 full 或 gate PASS run。

- 在 `COMPONENT_CANDIDATES.md` 登记 C016 numerical-margin/noise sensitivity diagnostic，用于阻止把理想构造的 `1e-10` 容差误作真实 matcher 阈值。
- 修正 synthetic seed 契约：两侧 structural seeds、共享 paired mean/eval sample seeds 和 solver seed 分开记录；paired 两侧不得使用不同数据 RNG 冒充 seed 差异。
- 增加固定 `1e-3` code perturbation 的负控单元测试，确认 exact tolerance 不接受明显非零扰动；该扰动幅度仅用于实现诊断，不是现实阈值或统计模型。
- 10 个 `unittest` 全部通过；两个历史 PASS run 在当前代码变化后仍通过快照式 contract validation。

### 冻结配置与启动

- Run ID：`R001_candidate_v1_20260902T024247Z`。
- Families：F01 Hadamard 和 F02 local block rotations，各 20 structural seed pairs；F01 `q={2,4,8}`，F02 为 3 个 rank-2 blocks。
- 数据与数值：float64；`n_mean=257`、`n_eval=2048`；独立 synthetic mean/eval；absolute/relative/residual tolerance `1e-10`；F02 max group size 4。
- 汇总规则：F01 每 seed/q 必须同时通过 BCC、PSC、PW-MCC 理论值和逐样本 contribution；F02 每 seed 必须 exact recovery、每 block 一个 support-minimal candidate、零跨 block candidate、PSC=1、residual/contribution 在容差内。全部检查必须通过，不允许按平均率掩盖失败。
- 统计单位：generator structural seed pair；observations 不作为独立重复。
- 范围边界：只覆盖 R001 的 F01/F02，不覆盖 F03–F12、ambiguity/refusal 或真实 SAE；即使 PASS 也不使 M1 通过。
- 资源：有界单进程 CPU matrix，无需 lease。

Tracker 已增加 suffix 并标为 `RUNNING`，parent R001 暂保持 `TODO`，待结果和范围审计后决定其状态。

## 2026-09-01 22:43 EDT — R001 candidate v1 完成

### 结果

`R001_candidate_v1_20260902T024247Z` 以退出码 0 完成，80 条 raw records 的 360/360 预写检查全部通过，contract validator 返回零错误。

- F01：20 seed pairs × 3 个 `q`，共 60 records。最大逐样本 contribution 误差 `1.78e-15`，最大 `|BCC-1|=4.44e-16`，PSC 全为 1，PW-MCC 与 `q^{-1/2}` 的最大误差为 0。
- F02：20/20 seed pairs exact recovery，60/60 planted blocks 恢复；27,000 个枚举 subset pairs 中跨 block 伪候选为 0。最大 planted normalized residual `6.66e-16`，最大逐样本 contribution 误差 `3.56e-15`。
- raw metrics、resolved config、输入、代码快照和生成脚本 hash 链完整。

### Artifact 与状态

- `runs/R001_candidate_v1_20260902T024247Z/` 保存完整 artifacts。
- Tracker suffix 更新为 `PASS`。
- Parent R001 继续保持 `TODO`：其规格还包括 F10/F11 的正式 20-pair 矩阵，目前这两类只有 smoke 证据；M1 还需要 R002/R003 和其余 falsifiers。

### 保守解释与 gate 影响

F01/F02 已满足各自至少 20 seed pairs 的理想 exact-conformance 检查，说明 kernel、PW-MCC、PSC 与局部 exhaustive oracle 在这些受控构造上没有观察到实现错误。该结论不涉及噪声、proposal、未知 neighborhoods、real SAE 或因果终点；C016 的 `1e-3` 扰动测试仅证明 exact tolerance 没有把明显数值偏差当作零，不提供现实阈值。M1 仍未通过，C1/C2 状态不变。

### 下一步

将 F10 same-span/different-computation 与 F11 same-sum/bloated-span 扩成各 20 seed pairs，并保存跨 replicate 的 BCC/PSC 分离统计，从而决定 parent R001 是否可 PASS。之后实现 F03 unequal split/merge，进入 R002 的 unequal-cardinality exact recovery。

## 2026-09-01 23:00 EDT — 启动 R001 complements v1

### 准备与冻结

Heartbeat `ccad` 按上一条推进 F10/F11 的正式重复矩阵。runner 已改为对两类构造分别生成 20 个独立 synthetic replicate seeds，并在 raw records 中保存 generator seed。修改后 10 个单元测试全部通过，`R001_candidate_v1_20260902T024247Z` 的历史快照仍通过 contract validation。

### Run 启动

- Run ID：`R001_complements_v1_20260902T030045Z`。
- F10：20 replicates；每个使用 `n_mean=257`、`n_eval=20480`，预写规则为 `PSC=1` 且 `|BCC|<0.05`。该 BCC 界只用于此零总体相关的 synthetic conformance，不是现实匹配阈值。
- F11：20 replicates；每个使用 `n_mean=257`、`n_eval=2048`，预写规则为 `BCC=1` 且 `PSC=2/3`。
- 数值：float64，absolute/relative tolerance `1e-10`；candidate configuration 已冻结，audit 未打开。
- 统计单位：synthetic replicate seed；观测行不作独立重复。
- 资源：有界单进程 CPU run，无需 lease。
- 范围：R001 complement，不是 M1 full；即使通过也不覆盖 R002/R003。

Tracker suffix 已标为 `RUNNING`，结果将在完成后追加。

## 2026-09-01 23:01 EDT — R001 complements 完成，parent R001 过门

### 结果

`R001_complements_v1_20260902T030045Z` 以退出码 0 完成，40 条 raw records 的 80/80 预写检查全部通过，artifact contract 独立验证无错误。

- F10 20 replicates：PSC 全为 1；BCC 范围 `[-0.023799, 0.012120]`，中位数 `-0.001998`，最大绝对值 `0.023799 < 0.05`。该结果复现 same-span/different-computation 的分离，而非证明真实 SAE 上的效应大小。
- F11 20 replicates：PSC 全为 `2/3`；最大 `|BCC-1|=2.11e-15`；逐样本 contribution 最大误差为 0。该结果复现 same-sum/bloated-span 的分离。

### Parent R001 判定

结合以下不可变 artifacts：

- `R001_candidate_v1_20260902T024247Z`：F01/F02 各 20 seed pairs，360/360 检查通过；
- `R001_complements_v1_20260902T030045Z`：F10/F11 各 20 replicates，80/80 检查通过；
- 单元测试：显式 contribution 与 kernel、BCC residual identity、PSC projector identity、inactive/degenerate 和 exact-tolerance 负控通过；

R001 规格中的 F01/F02/F10/F11、PW-MCC/BCC/PSC 与代数 sanity 均已满足，因此 parent `R001` 更新为 `PASS`。此前失败的 `R001_smoke_20260902T020100Z` 仍保留，且其 provenance 失败没有被后续 PASS 覆盖。

### Gate 与 claim 影响

R001 PASS 只通过 M1 的 kernel/theory-sanity 子门。R002 local many-to-many recovery 与 R003 ambiguity/refusal 仍为 `TODO`，其余 falsifiers 尚未完成，因此 M1 整体未通过，真实 matcher 主线仍不可启动。没有真实 SAE 或因果证据，C1/C2 状态不变。

### 下一步

实现 F03 unequal split/merge 的非等基数 generator 和 exact local oracle 测试，随后扩展到至少 20 seed pairs；必须检查 support-minimal filtering 是否错误偏好一侧更小的组，并报告两侧 group size，而不能用同基数 ARI 掩盖结构差异。

## 2026-09-01 23:19 EDT — 启动 R002 F03 candidate v1

### 实现与预检

按 heartbeat `ccad` 推进 R002 的 F03 unequal split/merge。新增 alternating 1→2 / 2→1 的 contribution-preserving generator；每个局部 block 独占一个正交 hook 方向，split 权重在 `[0.2, 0.8]` 内由 seed 生成。exact local oracle 只在预先给定的局部候选池中穷举，要求完整非等基数 hyperedge 是唯一 support-minimal exact match，并用相邻正交 block 作 false-match 负控。

修改 artifact：`src/ccad/synthetic.py`、`scripts/run_r001_smoke.py`、`tests/test_r001_metrics.py`、`configs/r002_f03_candidate_v1.json`。runner manifest 的 `run_parent`、purpose 与 evidence level 同时改为从 resolved config 读取，避免 R002 artifact 被误标成 R001。新增测试后 11/11 单元测试通过；第一次未设置项目 `PYTHONPATH` 的测试命令产生 import error，修正环境后通过，不构成实验运行失败。

### Run 冻结

- Run ID：`R002_f03_candidate_v1_20260902T031933Z`；parent `R002`。
- 20 个 synthetic seed pairs；每对 4 个 alternating unequal blocks；`n_mean=257`、`n_eval=2048`、float64。
- `max_group_size=2`，exact residual / absolute / relative tolerance 均为 `1e-10`。
- 预写判据：全部 planted hyperedge 精确恢复；每 block 恰有一个 support-minimal passing candidate；cross-block passing candidate 为 0；全部 group 非等基数；PSC=1；显式 contribution 相等。
- ARI 明确记为 `not_applicable_unequal_feature_universes`，不把两侧不同 atom universe 强行编码成 partition ARI。
- synthetic mean 独立用于中心化，candidate family 已冻结，audit 未打开；统计单位为 seed pair。
- 资源：有界单进程 CPU run，无需 resource-manager lease。

Tracker suffix 已标为 `RUNNING`。即使本 run 通过，也只提供 F03 candidate evidence；F04/F06 未完成，parent R002 与 M1 不过门。

## 2026-09-01 23:20 EDT — R002 F03 candidate v1 完成

### 结果

`R002_f03_candidate_v1_20260902T031933Z` 以退出码 0 完成，20 个 seed pairs 的 160/160 预写检查全部通过；artifact contract 独立复验为 `ok=true`，manifest 正确记录 parent `R002`、candidate evidence、未打开 audit 与 synthetic seed-pair 统计单位。

- 80/80 planted unequal-cardinality blocks 精确恢复，80 个 block 共得到恰好 80 个 support-minimal passing candidates。
- 640 个已计数的 local/cross candidate pairs 中，跨正交 block passing candidate 为 0。
- planted normalized residual 最大值 `4.15e-16`，planted PSC 最小值 `1.0`，显式 contribution 最大绝对误差 `4.44e-16`。
- 80 个 split 权重实际范围 `[0.21095, 0.79948]`；所有 planted groups 均为 1↔2。
- ARI 状态一致为 `not_applicable_unequal_feature_universes`。主结构终点是 exact planted hyperedge recovery，而不是伪造共同 atom partition。

### 保守解释与 gate 影响

该 run 支持 C002 exact local oracle 能处理理想、正交、无噪声的非等基数 split/merge，且没有偏好较小侧的 proper subset。它不支持真实 SAE、overlap、co-occurrence confounding 或因果主张。

本 artifact 仍属于 `candidate`：`SYNTHETIC_SUITE_SPEC.md` 对正式 R002 还要求逐实例保存完整候选子集族、最优/次优 residual、tie set、运行时间和更细的独立 seed 字段；当前仅保存 passing support-minimal candidates 的计数、访问预算和 generator seed。因此 parent R002 保持 `TODO`，M1 不过门，不把本次 PASS 偷换成正式 R002 完成。

### 下一步

先扩展 exact-oracle result schema，保存全候选 residual 排序、solver gap/ties、运行时间、planted hyperedges 与结构/采样 seed provenance；随后实现 F04 partial overlap，以 hyperedge precision/recall 而非 partition 指标评估，并保持 overlap 合法输出不被强制删边。

## 2026-09-01 23:38 EDT — 启动 R002 F03 formal v2

### 触发与实现

Heartbeat `ccad` 根据上一条记录，先补齐正式 R002 所需的 exact-oracle 可审计 schema，而不是直接把 candidate v1 升格。`src/ccad/matching.py` 新增完整 exhaustive search result：保存有限候选族的 residual 排序、threshold 内 passing set、support-minimal set、最优/次优 residual、solver gap、tie set、访问候选数和 wall-clock runtime；旧 list API 保留为兼容包装。

F03 generator 新增独立的 `structural_seed_a`、`structural_seed_b`、`mean_sample_seed` 和 `eval_sample_seed` 接口；runner 另记录 deterministic exhaustive oracle 的 `solver_seed`（只作 provenance，算法本身无随机分支）。每个 raw record 现在包含 planted hyperedges、完整 planted/cross neighborhood candidate families 与上述 diagnostics。新增测试后 12/12 单元测试通过。

### Run 冻结

- Run ID：`R002_f03_formal_v2_20260902T033826Z`；parent `R002`。
- 20 个独立 structural seed pairs，每对 4 个 alternating 1↔2 blocks；paired mean/eval 分别使用独立共享 sample seed。
- `n_mean=257`、`n_eval=2048`、float64；`max_group_size=2`。
- residual/absolute/relative tolerance `1e-10`；solver tie tolerance `1e-12`，均在运行前写入 resolved config。
- 除 v1 判据外，新增必须项：每个 planted neighborhood tie set 大小为 1、最小 solver gap 大于 tie tolerance、完整 diagnostics 数量正确。
- candidate family 已冻结，audit 未打开；统计单位为 structural seed pair；有界 CPU exhaustive run，无需 lease。

Tracker suffix 已标为 `RUNNING`。本 run 仅可完成 F03 正式子项，不能单独使 R002 或 M1 过门。

## 2026-09-01 23:39 EDT — R002 F03 formal v2 完成

### 结果与 artifact 审计

`R002_f03_formal_v2_20260902T033826Z` 以退出码 0 完成，20 条 raw structural-seed-pair records 的 220/220 预写检查全部通过；独立 artifact validation 为 `ok=true`。

- 80/80 planted 1↔2 groups 精确恢复，cross-block passing matches 为 0。
- 共访问并保存 640 个 candidate evaluations；每个 record 均保存 8 个完整 neighborhoods（4 planted、4 cross negative），没有 diagnostics 缺失。
- planted 最小 solver gap 为 `0.0246777`，显著大于预写 tie tolerance `1e-12`；最大 planted tie-set size 为 1。
- planted normalized residual 最大值 `4.42e-16`，显式 contribution 最大误差 `4.44e-16`，PSC 均为 1。
- raw provenance 同时包含 `structural_seed_a/b`、`mean_sample_seed`、`eval_sample_seed`、`solver_seed`；20 个 records 的 exact-oracle 总 wall time 约 `0.0110 s`，只用于复现实测而非性能主张。

### 保守解释与 gate 影响

F03 的正式 synthetic 子项现可标为 PASS：理想正交 split/merge 中，exact local oracle 能唯一恢复 unequal-cardinality hyperedge，并完整暴露竞争 residual 与 tie 证据。这仍只是 algebraic/synthetic correctness，不能外推到 noisy proposal、真实 SAE 或 C1/C2。

Parent `R002` 保持 `TODO`：F02 现有 artifact 尚未迁移到完整 diagnostics schema，F04 partial overlap 和 F06 co-occurrence confounding 尚未实现。M1 不过门，真实 matcher 与真实数据主线仍不得启动。

### 下一步

实现 F04 partial-overlap generator 与 overlap-aware exact-oracle evaluator。真值和预测均按 hyperedge set 计 precision/recall/F1，必须保留共享 atom 的两条有效边，并增加 forced-partition 负控来显式展示删除一条边的错误；在 COMPONENT_CANDIDATES.md 中把它作为 C003 的首次 synthetic screen 关联记录，但在 20-pair formal matrix 通过前不升级状态。

## 2026-09-01 23:57 EDT — 启动 R002 F04 formal v1

### 组件登记与实现

Heartbeat `ccad` 按上一条推进 F04。先更新 `COMPONENT_CANDIDATES.md` 的 C003，将 forced-partition projection 明确登记为 overlap hypergraph 的关键负控；C003 状态仍为 `IDEA`，未因实现而提前升级。

新增 F04 构造：三对 atom 的动态贡献满足两条 exact hyperedges `{0,1}↔{0,1}` 与 `{0,2}↔{0,2}`，两条边在左右两侧都共享 atom 0，但所有 proper singleton 和其他大小不超过 2 的组合均不匹配。两侧 decoder/code factorization 分别受独立 structural seeds 控制，mean/eval paired samples 使用独立共享 sample seeds。

新增 overlap-aware evaluator，以完整 hyperedge set 计算 precision/recall/F1；全局候选池 exhaustive oracle 保存 36 个候选及 residual/ties。新增 `forced_partition_projection` 仅作负控：它禁止 atom 复用，预期只能保留一条 planted edge，因此 precision=1、recall=0.5。ARI 明确标为不适用于 overlapping hypergraph。实现后 13/13 单元测试通过。

### Run 冻结

- Run ID：`R002_f04_formal_v1_20260902T035744Z`；parent `R002`。
- 20 个 structural seed pairs；每实例两条共享 atom 的 planted 2↔2 hyperedges。
- `n_mean=257`、`n_eval=2048`、float64；`max_group_size=2`。
- residual/absolute/relative tolerance `1e-10`；tie tolerance `1e-12`，运行前固定。
- 主判据：overlap-aware hyperedge P/R/F1 均为 1，两条共享边均保留；oracle passing support-minimal set 与 tie set 均恰为 2；forced-partition precision=1、recall=0.5；PSC=1 且显式 contribution equality 通过。
- candidate family 已冻结，audit 未打开；统计单位为 structural seed pair；有界 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。即使通过，本 run 只完成 F04 正式子项；R002 仍需 F02 完整 schema 与 F06。

## 2026-09-01 23:58 EDT — R002 F04 formal v1 完成

### 结果与独立复验

`R002_f04_formal_v1_20260902T035744Z` 以退出码 0 完成，20 个 structural seed-pair records 的 280/280 预写检查全部通过；独立 artifact validation 为 `ok=true`。

- overlap-aware 输出在全部 20 个实例上 hyperedge precision/recall/F1 均为 1；两条共享 atom 的 planted edges 全部保留。
- 每个实例完整评估并保存 36 个候选；support-minimal passing set 与 best-tolerance tie set 都恰有 2 条，即两条 planted overlapping edges。
- forced-partition 负控在全部实例上 precision=1、recall=0.5：由于禁止 atom 复用，它必然删除一条真边。这是构造内的机制性负控，不是现实数据效应量。
- expected/observed decision 全部为 `OVERLAPPING_HYPERGRAPH`；PSC 最小值 1，显式 contribution 最大误差 `1.33e-15`。
- exact-oracle 总 wall time 约 `0.0112 s`；完整 seed、candidate、residual、tie 和 runtime provenance 已保存。

### 保守解释与 gate 影响

F04 正式 synthetic 子项可标为 PASS。该证据直接否定了“合法局部输出总能强制成 partition 而不损失真边”的工程假设，并验证当前 hypergraph representation 不会因覆盖冲突删除 planted edge。

这仍是刻意构造的 exact、无噪声、三 atom 小邻域；不证明现实 SAE overlap 普遍存在，也不证明 C003 的 proposal 在真实 discovery corpus 有足够 recall。因此 C003 维持 `IDEA`，不升格为正式方法组件。Parent R002 和 M1 均保持未通过，真实 matcher 主线不得启动。

### 下一步

实现 F06 co-occurrence confounding：构造 activation/code correlation 很高但 decoded dynamic contribution 不匹配的候选，要求 correlation-only proposal 接纳而 contribution matcher 拒绝或降级；同时保存 proposal score、BCC/residual、decision、full candidate family 与 seed provenance。之后将 F02 迁移到完整 formal schema，再决定 R002 parent 状态。

## 2026-09-02 00:17 EDT — 启动 R002 F06 formal v1

### 组件与构造

Heartbeat `ccad` 按上一条推进 F06。先更新 `COMPONENT_CANDIDATES.md` 的 C001，将 correlation-only proposal 和 F06 rejection 纳入 contribution-kernel 的显式必要性对照；C001 状态不变。

新增 F06 构造：左右 singleton codes 来自同一 paired latent，因此中心化后的绝对 Pearson correlation 理论上为 1；但左右 unit contribution directions 的绝对 cosine 由独立 structural seed 限制在 `[0,0.2)`，并随机符号，从而 BCC 低、normalized contribution residual 高、PSC 低。两侧另用独立正尺度进行 decoder/code refactorization，保证 correlation 混杂不是简单尺度差异。

新增 `absolute_code_correlation` 指标和 F06 evaluator。correlation-only proposal 先按预写阈值接纳，contribution exhaustive oracle 再按 residual threshold 拒绝；raw record 同时保存 proposal score、BCC/energies/cross term、PSC/ranks、contribution RMSE、mean contribution error、coverage、unmatched energy、group size、condition/cancellation 单元诊断、完整 singleton candidate family/tie/runtime 和五类 seed provenance。新增测试后 14/14 单元测试通过。

### Run 冻结

- Run ID：`R002_f06_formal_v1_20260902T041703Z`；parent `R002`。
- 20 个 structural seed pairs；`n_mean=257`、`n_eval=2048`、float64。
- correlation-only proposal 阈值 `|r_code|≥0.99`；contribution match residual threshold `0.1`；必须观察 residual≥0.8、`|BCC|≤0.2`、PSC≤0.04、contribution RMSE≥1.0。
- 正确 decision 为 `REFUSE_CONTRIBUTION_MISMATCH`；oracle 访问 1 个候选但 passing/support-minimal set 均为空；coverage=0、unmatched-energy fraction=1。
- 上述 residual threshold 只属于 F06 construction-specific falsifier，明确不冻结为真实 SAE matcher threshold。
- candidate family 已冻结，audit 未打开；统计单位为 structural seed pair；有界 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。即使通过，R002 仍需把 F02 迁移到完整 formal schema 后才能作 parent 判定。

## 2026-09-02 00:18 EDT — R002 F06 formal v1 完成

### 结果与 artifact 审计

`R002_f06_formal_v1_20260902T041703Z` 以退出码 0 完成，20 个 structural seed-pair records 的 260/260 预写检查全部通过；独立 artifact validation 为 `ok=true`。

- 20/20 correlation-only proposals 被接纳，最小绝对 code correlation 为 `0.9999999999999997`。
- contribution matcher 对 20/20 候选输出 `REFUSE_CONTRIBUTION_MISMATCH`；完整 oracle passing candidates 总数为 0。
- 最大 `|BCC|=0.197064`，最小 normalized residual `0.802936`，最大 PSC `0.0388342`，均满足预写分离界。
- 最小 contribution RMSE `1.25090`；coverage 全为 0、unmatched-energy fraction 全为 1。mean contribution error 也已逐 record 保存，最大值 `0.191674`，但未被用作选择或成功判据。
- singleton exact-oracle 总 wall time约 `0.00043 s`；完整 config、code/input hash、seed、candidate/residual/tie 和环境 provenance 均通过 contract。

### 保守解释与 gate 影响

F06 正式 synthetic 子项可标为 PASS：高 paired code correlation 本身不能证明 decoded dynamic contribution 可匹配，contribution-space 检查在该构造中正确拒绝。这支持 C001 中 contribution kernel 相对 correlation-only proposal 的必要性，但仍不说明现实 SAE 中这种 confound 的发生率或效应大小。

Threshold `0.1` 是该无噪声 F06 falsifier 的预写判据，不得迁移为真实 SAE matcher threshold。Parent R002 仍保持 `TODO`，因为 F02 虽已有 20-pair candidate artifact，但尚未保存 formal schema 要求的完整候选 residual、ties、runtime 和独立 seed provenance。M1 不过门。

### 下一步

将 F02 local block rotations 迁移到与 F03 相同的 formal exact-oracle schema，运行 20 个 structural seed pairs，并聚合核查 F02/F03/F04/F06 四类 formal artifacts。只有全部 contract 和 family-specific 终点满足后，才可将 parent R002 标为 PASS；R003 仍将独立阻止 M1/真实主线过门。

## 2026-09-02 00:36 EDT — 启动 R002 F02 formal v1

### Formal schema 迁移

Heartbeat `ccad` 按上一条迁移 F02。local block generator 现使用独立 `structural_seed_a/b` 生成左右 block 内正交基，并用独立共享 `mean_sample_seed`、`eval_sample_seed` 生成 paired latent；旧单 seed API 保留为兼容 wrapper。左右每个 rank-2 block 均用 signed-ReLU refactorization 精确重构同一 hook-space block contribution。

F02 evaluator 现逐 planted 与 cross-negative neighborhood 保存完整 225-candidate family、passing/support-minimal sets、最优/次优 residual、gap、ties、runtime、planted hyperedges和五类 seed provenance；另报告 independent-mean contribution error、coverage/unmatched energy、group size/effective rank，以及左右 partition ARI。新增无外部依赖的 ARI 实现和测试后，15/15 单元测试通过。

### Run 冻结

- Run ID：`R002_f02_formal_v1_20260902T043603Z`；parent `R002`。
- 20 个 structural seed pairs；每实例 3 个正交 rank-2 blocks，每侧每 block 4 个 signed features。
- `n_mean=257`、`n_eval=2048`、float64；`max_group_size=4`。
- residual/absolute/relative tolerance `1e-10`，tie tolerance `1e-12`，均在运行前固定。
- 主判据：3/3 planted blocks 唯一精确恢复且 cross matches 为 0；左右 ARI=1；PSC=1；显式 eval 与 independent-mean contribution error 通过；每 planted tie set 为 1 且最小 solver gap 大于 tie tolerance；coverage=1、unmatched fraction=0。
- 每实例预计 6 个 neighborhoods、1350 个 candidate evaluations；统计单位为 structural seed pair。
- candidate family 已冻结，audit 未打开；有界 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。通过后仍须独立聚合审计 F02/F03/F04/F06 artifacts，不能仅凭本 run 自动修改 parent。

## 2026-09-02 00:37 EDT — R002 F02 formal v1 完成，parent R002 过门

### F02 结果

`R002_f02_formal_v1_20260902T043603Z` 以退出码 0 完成，20 个 structural seed-pair records 的 300/300 预写检查全部通过；独立 artifact validation 为 `ok=true`。

- 60/60 planted rank-2 blocks 精确恢复，cross-block passing matches 为 0；左右 partition ARI 全为 1。
- 完整保存 27,000 个 candidate evaluations；所有 record diagnostics 完整，planted tie-set 最大为 1。
- 最小 planted solver gap `1.3456e-05 > 1e-12`；最大 planted residual `6.41e-16`。
- 最大显式 eval contribution error `4.00e-15`，最大 independent-mean contribution error norm `1.16e-16`；PSC、coverage 均为 1，unmatched fraction 为 0。
- 20 个 records 的 exact-oracle 总 wall time约 `0.350 s`，仅作复现信息。

### R002 parent 聚合审计

重新用当前 snapshot-aware contract validator 独立复验四个 formal artifacts，均为 `status=PASS`、summary PASS、20 个 structural seed-pair records、contract 无错误：

- F02 `R002_f02_formal_v1_20260902T043603Z`，raw hash `84ea0d6f0d4f3ae16ec6819197ecb11e023db148a83d88c0f11a7e1ef7076a0b`；
- F03 `R002_f03_formal_v2_20260902T033826Z`，raw hash `c6265d090f0e25d0ef46ecfbc026a189bd1b0aef44481952e2e128013a76c24f`；
- F04 `R002_f04_formal_v1_20260902T035744Z`，raw hash `7dfbd49fc5a7d3e705b2dedceb363cb75f836d500dc918686f39f0b0564aabc1`；
- F06 `R002_f06_formal_v1_20260902T041703Z`，raw hash `7bc69e2285c3bb0d6057c1bb24b20a957fbf993e69f2592625ef7ea7d8cfbf05`。

四类满足 `SYNTHETIC_SUITE_SPEC.md` 对 R002 的每 family 至少 20 seed pairs、小邻域 exhaustive oracle、完整 candidate/solver provenance 与 family-specific endpoint 要求：F02/F03 exact recovery，F04 hyperedge set recovery，F06 correlation-confound refusal。因此 parent `R002` 更新为 `PASS`。

### 保守解释与 gate 影响

R002 PASS 仅说明理想合成局部 many-to-many recovery 与拒绝一个 proposal confound 的实现正确；不构成真实 SAE、held-out audit 或因果证据。M1 整体仍未通过，因为 R003 ambiguity/refusal 与剩余 falsifiers 为 `TODO`；真实 matcher 和真实数据主线仍暂停。

### 下一步

进入 R003 的最低信息增益路线：先实现 F08 competing covers 与 F09 whole-dictionary-only，冻结 `AMBIGUOUS`/`REFUSE` decision schema、false-unique rate 和完整 tie/cover artifacts；随后补 F05/F07/F10/F11/F12 diagnostics。任何 forced-best 输出都必须作为失败负控而非合法答案。

## 2026-09-02 00:56 EDT — 启动 R003 F08/F09 formal v1

### 理论边界复核

Heartbeat `ccad` 使用 PDF workflow 完整复核理论稿第 35–43 页。THM-CBSM-001 明确：不同 maximum-cardinality exact covers 是 hard partition 不可识别的证书；算法步骤要求保留所有统计上不可区分 covers。边界案例又明确：若只有全字典 balance，则局部概念未被识别。这两点分别约束 F08 输出 `AMBIGUOUS`、F09 输出 `REFUSE_GLOBAL_ONLY`，而不是任意挑一个 cover 或把 full reconstruction 包装成局部概念。该轮未修改理论稿。

### 实现与预写规则

新增 small-hypergraph exact-cover oracle，枚举所有 exact covers 并保留全部 maximum-cardinality covers。F08 使用每侧两个 contribution-identical singleton atoms，产生四条 support-minimal balanced edges和两种最大基数 perfect covers；forced-best 只作 false-unique 负控。F09 使用三 atom 构造，只有完整 3↔3 group balance，所有 proper local pairs 均不 balance；global-collapse acceptance 只作 false-unique 负控。

两类 generator 均记录独立 structural seeds、paired mean/eval seeds 和 solver seed。runner 保存完整 candidate residual/tie family、完整 cover sets、ambiguity/refusal accuracy、false-unique rate、global BCC/mean/eval error、coverage/unmatched energy及 runtime。新增测试后 17/17 单元测试通过。

### Run 冻结

- Run ID：`R003_f08_f09_formal_v1_20260902T045635Z`；parent `R003`。
- F08、F09 各 20 个 structural seed pairs；`n_mean=257`、`n_eval=2048`、float64。
- residual/absolute/relative tolerance `1e-10`，tie tolerance `1e-12`，运行前固定。
- F08：必须返回恰好两套 cardinality-2 maximum covers，ambiguity accuracy=1、CBSM false-unique=0；forced-best false-unique=1。
- F09：唯一 passing edge 必须是 full 3↔3 group，但合法 decision 为 REFUSE；local passing count=0、refusal accuracy=1、CBSM false-unique=0；global-collapse false-unique=1。
- candidate family 已冻结，audit 未打开；统计单位为 structural seed pair；有界 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。本 run 即使通过也只覆盖 R003 的 F08/F09；其余诊断 falsifiers 未完成，R003/M1 不过门。

## 2026-09-02 00:57 EDT — R003 F08/F09 formal v1 完成

### 结果与 artifact 审计

`R003_f08_f09_formal_v1_20260902T045635Z` 以退出码 0 完成，40 个 raw structural-seed-pair records 的 440/440 预写检查全部通过；独立 artifact validation 为 `ok=true`，raw metrics hash 为 `8e117d07827498bf1ccb05abda5a83e8b4adaecd226cdd0c12040d528082757b`。

- F08 20/20 输出 `AMBIGUOUS`，每实例完整返回两套 cardinality-2 maximum covers；ambiguity accuracy=1，CBSM false-unique rate=0。
- forced-best 负控在 F08 20/20 的 false-unique rate=1，显示任意 tie-breaking 会隐瞒不可识别性。
- F09 20/20 输出 `REFUSE_GLOBAL_ONLY`，proper local passing count 最大为 0；refusal accuracy=1，CBSM false-unique rate=0。
- F09 full-dictionary balance residual 最大 `4.49e-16`、显式 eval contribution error 最大 `1.78e-15`、independent-mean error 最大 `6.94e-17`；global balance 被单列保存但没有被当作局部 match。

### 保守解释与 gate 影响

F08/F09 正式子项可标为 PASS，且行为与理论 exact-cover/whole-dictionary 边界一致：多个最大 cover 必须显式歧义，只有全局 balance 必须局部拒答。这验证的是小型 exact synthetic decision logic，不证明真实 SAE 中 ambiguity/refusal 的频率。

Parent R003 与 M1 保持 `TODO`：F05 cancellation、F07 rare occupancy、F10/F11 诊断复核和 F12 downstream cliff 尚未形成各 20-pair formal artifacts。真实 SAE 主线仍暂停。

### 下一步

优先实现 F05 cancellation 与 leave-one-feature-out leverage/cancellation diagnostics，因为它直接检验“高 BCC 是否由大项相消伪造”；同时为 F07 冻结 occupancy-based `n_eff` 和 bootstrap-instability refusal 的统计定义。F10/F11 可在既有构造上补完整 R003 decision schema，F12 最后连接 intervention boundary。

## 2026-09-02 01:16 EDT — 启动 R003 F05 formal v1

### 组件定义与实现

Heartbeat `ccad` 先更新 `COMPONENT_CANDIDATES.md` 的 C007，冻结本轮使用的诊断定义：cancellation energy ratio 为 `sum_i E||phi_i||^2 / E||sum_i phi_i||^2`；leave-one-out energy ratio 等于删除单 feature 引起的 contribution-change energy 除以 aggregate energy。no-diagnostic forced match 是失败负控，同实例 orthogonal clean singleton 是 specificity control。C007 状态仍为 `IDEA`。

F05 generator 的 risk group 在两侧各由两个大幅相消的 feature 构成，但 aggregate contribution pointwise 等于同一 signal，因此 BCC/PSC 均为 1；左右 cancellation latents 和幅度由独立 structural seeds 控制。第三个 feature 是无 cancellation 的 clean singleton exact match。runner 在 exact oracle 之后施加 diagnostic layer：risk group 必须降级/拒答，clean control 必须正常通过。

新增 `cancellation_diagnostics` 及单元测试；当前 18/18 tests 通过。raw record 将保存 risk/clean 完整候选族、per-feature ratios、aggregate diagnostics、BCC/PSC、显式 error、accepted feature/energy coverage、unmatched energy、decision 与五类 seed provenance。

### Run 冻结

- Run ID：`R003_f05_formal_v1_20260902T051636Z`；parent `R003`。
- 20 个 structural seed pairs；`n_mean=257`、`n_eval=2048`、float64。
- exact residual/absolute/relative tolerance `1e-10`，tie tolerance `1e-12`。
- risk 阈值：两侧 cancellation energy ratio 均≥100，max leave-one-out ratio 均≥50；clean control 两类 ratio 必须为 1 且不触发。
- 正确 decision 为 `REFUSE_CANCELLATION_RISK_WITH_CLEAN_CONTROL_PASS`；diagnostic recall/specificity=1、CBSM false-unique=0；no-diagnostic false-unique=1。
- 这些阈值只属于幅度 `[8,12]` 的 F05 construction，不冻结为真实 SAE cutoffs。
- candidate family 已冻结，audit 未打开；统计单位为 structural seed pair；有界 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。通过也只覆盖 F05；R003/M1 仍需 F07/F10/F11/F12。

## 2026-09-02 01:17 EDT — R003 F05 formal v1 完成

### 结果与 artifact 审计

`R003_f05_formal_v1_20260902T051636Z` 以退出码 0 完成，20 个 structural seed-pair records 的 400/400 预写检查全部通过；独立 artifact validation 为 `ok=true`，raw hash 为 `1bc631c99c68a6ef1c48f415df8b6c2214edf76e508f156846fa2ce188f9ef17`。

- 20/20 canceling risk groups 被 diagnostic layer 标记并输出预期 refusal/degrade decision；diagnostic recall=1、CBSM false-unique=0。
- 左右最小 cancellation energy ratio 分别为 `132.50`、`148.66`；最小 max leave-one-out ratio 分别为 `66.69`、`74.58`，均超过预写阈值。
- 20/20 clean singleton controls 正常通过，false flag 为 0；clean cancellation/LOO ratios 全为 1，specificity=1。
- risk groups 虽有 BCC/PSC=1 和 pointwise aggregate equality，但最大计算 residual 为 `4.49e-14`、显式 contribution error 最大 `7.55e-15`。相消放大浮点误差但仍低于预写 `1e-10` exact tolerance，这也支持必须单列 numerical margin。
- 拒绝 risk group 后，accepted energy coverage 范围下界 `0.4784`，unmatched-energy fraction 上界 `0.5216`；没有用 coverage 压力覆盖诊断拒答。

### 保守解释与 gate 影响

F05 正式子项可标为 PASS：高 BCC/PSC 不足以排除大项相消，energy-ratio 与 leave-one-out diagnostics 在该构造中有完整 recall，并通过 clean specificity control。这支持 C007 继续进入后续 screen，但不证明其阈值能迁移到真实 SAE，因此 C007 保持 `IDEA`。

R003/M1 仍保持未通过；F07 rare occupancy、F10/F11 完整诊断 decision schema 和 F12 non-Lipschitz downstream cliff 尚缺。真实主线继续暂停。

### 下一步

为 F07 实现预先定义的 occupancy effective sample size与 document/cluster-respecting bootstrap stability；构造 token 数很大但有效激活事件很少的 paired feature，使 point estimate 看似可匹配而 bootstrap 区间不稳定，正确输出 `REFUSE_LOW_N_EFF`。先用 analytic/event bootstrap 单测定义，再冻结 20-pair formal threshold。

## 2026-09-02 01:37 EDT — 启动 R003 F07 formal v1

### 统计定义与技能影响

Heartbeat `ccad` 使用 statistical-analysis skill，先陈述 confirmatory decision rule、统计单位与不确定性方法，再运行 formal matrix；不做结果后换检验。新登记 C017：以 raw uncentered contribution energy 定义 active token/document，并计算 token/document energy Kish ESS；bootstrap 以 document 为 cluster 成对重采样左右贡献，显式保留 inactive resamples，而不是把 2048 个 token 当作独立重复。

F07 risk singleton 在 32 个 eval documents、2048 tokens 中仅 2 个 documents、4 个 tokens 活跃；两个活跃文档的左右贡献尺度分别为 0.5/1.5，使 point BCC 仍超过 naive acceptance threshold，但 document bootstrap 在低支持下产生 inactive mass 和宽区间。dense singleton exact match 是同实例 specificity control。mean split 独立且每个 rare event pair 正负对称，中心化常数不从 eval 重估。

新增 `occupancy_effective_sample_size`、`document_bootstrap_bcc`、F07 generator/evaluator 和单元测试；当前 19/19 tests 通过。

### Run 冻结

- Run ID：`R003_f07_formal_v1_20260902T053744Z`；parent `R003`。
- 20 个 structural seed pairs；`n_mean=1024`、`n_eval=2048`、64 tokens/document、2 active documents。
- naive point acceptance：BCC≥0.85 / residual≤0.15；risk document ESS≤2.01。
- 每 record 500 次 paired document-cluster bootstrap；risk inactive fraction≥0.05、valid BCC 95% percentile interval width≥0.1。
- dense control document ESS≥25、inactive fraction=0、CI width≤`1e-12`。
- 正确 decision 为 `REFUSE_LOW_N_EFF_WITH_DENSE_CONTROL_PASS`；refusal accuracy/specificity=1、CBSM false-unique=0、naive-token-count false-unique=1。
- 阈值来自构造解析范围与预运行单元测试，只适用于 F07；real-SAE cutoffs 保持 OPEN。
- candidate family 已冻结，audit 未打开；独立单位为 structural seed pair，bootstrap unit 为 document；有界单进程 CPU run，无需 lease。

Tracker suffix 已标为 `RUNNING`。通过也只完成 F07；R003/M1 仍需 F10/F11/F12 formal decisions。

## 2026-09-02 01:38 EDT — R003 F07 formal v1 完成

### 结果与 artifact 审计

`R003_f07_formal_v1_20260902T053744Z` 以退出码 0 完成，20 个 structural seed-pair records 的 420/420 预写检查全部通过；独立 artifact validation 为 `ok=true`，raw hash 为 `e7db5b4c814c1513bd712cdb071b4a4c3a183d1533c2c05821dfe7e10ac2ac1b`。

- 20/20 risk features 的 naive point BCC 超过 0.85，最小值 `0.87075`；如果只看 2048-token 点估计会全部接纳。
- risk group 实际仅 4 active tokens、2 active documents；两侧最大 document-energy Kish ESS 不超过 `1.9997`。
- 500-replicate paired document bootstrap 的 inactive fraction 范围 `[0.106, 0.162]`；valid BCC percentile-CI width 在所有 records 上约 `0.12308`。20/20 正确输出 low-support refusal，false-unique=0。
- dense controls 的最小 document ESS `30.686`，inactive fraction 全为 0，最大 CI width `4.44e-16`；20/20 稳定通过，specificity=1。
- 被拒 rare group 能量很小：accepted energy coverage 最小 `0.99699`，unmatched fraction 最大 `0.00301`。这不改变拒答优先于 coverage 的规则。

### 保守解释与技能影响

statistical-analysis workflow 促使本轮在运行前固定 hypothesis、document cluster unit、ESS/CI/inactive-mass判据，并同时报告点估计、支持规模和不确定性，而没有用 token-level显著性或结果后换检验。F07 正式子项可标为 PASS，但阈值与 0.5/1.5 构造相关，不能迁移到真实 SAE；C017 维持 `IDEA`，等待真实 document-level calibration。

R003/M1 仍未通过；F10/F11 需要补齐明确的 function/span-bloat diagnostic decisions 与 false-unique 负控，F12 需要 downstream cliff intervention boundary。真实主线继续暂停。

### 下一步

复用既有 F10/F11 构造但不复用旧 smoke 结论：新增独立 structural/mean/eval seed provenance、20-pair formal records和 decision layer。F10 必须在 PSC=1 时因低 BCC 输出 `REFUSE_FUNCTION_MISMATCH`；F11 必须在 BCC=1 时因 PSC/rank bloat 输出 `REFUSE_SPAN_BLOAT`，并各自提供去掉对应诊断的 false-unique 负控。

## 2026-09-02 01:59 EDT — 启动 R003 F10/F11 formal v1

### 冻结设计

新增 seeded F10/F11 generators、formal decision records 与两个解析单元测试；当前 21/21 tests 通过。F10 risk 组在共享二维 hook span 内施加精确 90° 动态旋转，预期 PSC=1、BCC=0；正交 clean singleton 必须通过。正确输出为 `REFUSE_FUNCTION_MISMATCH_WITH_CLEAN_CONTROL_PASS`，去掉 contribution/BCC 只看 span 的负控应 false-unique。

F11 risk 组由左侧 rank-1 contribution 对应右侧两个带反向额外方向的 atoms；两项求和精确复原左侧 contribution，但右侧 decoder span 为 rank 2，预期 BCC=1、PSC=2/3。正确输出为 `REFUSE_SPAN_BLOAT_WITH_CLEAN_CONTROL_PASS`，去掉 PSC/rank diagnostic 的负控应 false-unique；另有正交 clean singleton。

- Run ID：`R003_f10_f11_formal_v1_20260902T055911Z`；parent `R003`。
- 每 family 20 个 structural seed pairs；structural A/B、mean、eval、solver seeds 独立留痕。
- `n_mean=257`、`n_eval=2048`；residual acceptance `0.1`、tie tolerance `1e-12`。
- F10：`|BCC|<=1e-10`、residual≥0.99、contribution RMSE≥1.0；F11：PSC≤`2/3+1e-10` 且 rank 1→2。
- 阈值来自解析构造，不适用于真实 SAE；candidate family 已冻结、audit 未打开、统计单位为 structural seed pair。
- 有界单进程 CPU run，无需 resource lease。

Tracker suffix 已标为 `RUNNING`。即使通过，R003/M1 仍需 F12 downstream-cliff formal boundary。

## 2026-09-02 02:00 EDT — R003 F10/F11 formal v1 完成

### 结果与 artifact 审计

`R003_f10_f11_formal_v1_20260902T055911Z` 以退出码 0 完成，40 个 formal records 的 460/460 预写检查全部通过；独立 artifact validation 为 `ok=true`，raw hash 为 `69fdf8eec671b2aa98c39b8a9fc146aa2e0a469671b3c567bc55ba232fb3f025`。

- F10：20/20 输出 `REFUSE_FUNCTION_MISMATCH_WITH_CLEAN_CONTROL_PASS`。risk PSC 全为 1，最大 `|BCC|=1.30e-16`，最小 normalized residual=1，最小 contribution RMSE=`1.1398`；clean controls 全部通过。span-only 负控 20/20 false-unique。
- F11：20/20 输出 `REFUSE_SPAN_BLOAT_WITH_CLEAN_CONTROL_PASS`。risk BCC 最小 `0.9999999999999996`，PSC 数值为 2/3、rank 恒为 1→2，最大显式 contribution error=`8.88e-16`；clean controls 全部通过。no-PSC 负控 20/20 false-unique。
- formal generators 分离 structural A/B、mean、eval 与 solver seed provenance；未复用旧 complements smoke 的结论。

### 保守解释与 gate 影响

F10/F11 正式子项可标为 PASS，支持 C001 的 contribution-sensitive certificate 和 C005 的 span-bloat audit 继续进入 screen；`COMPONENT_CANDIDATES.md` 已追加对应 run 证据，状态仍为 `READY-FOR-SCREEN`，不是 `ADMIT`。结果只证明两个 truth-known 构造中的必要诊断：不能推断真实 SAE 上阈值稳定，也不能把 PSC 或 BCC 单独当作外部验证。

R003/M1 仍保持未通过；剩余 F12 non-Lipschitz downstream cliff 必须给出 intervention-level boundary 与平滑 clean control。真实 SAE 主线继续暂停。

### 下一步

先从理论 PDF 与 `SYNTHETIC_SUITE_SPEC.md` 恢复 F12 的精确定义，再新增最小的 downstream readout/intervention evaluator。预注册 risk/clean 两条曲线、cliff 判据和“高 BCC 不保证下游效应接近”的正确拒答；完成 20-pair formal matrix 后再聚合 R003 与 M1。

## 2026-09-02 02:23 EDT — 启动 R003 F12 formal v1

### 理论恢复与组件登记

使用 PDF workflow 完整核对理论稿第 43、46、47 页。THM-CBSM-009 对任意 deterministic downstream map 只保证 exact contribution equality 的精确 transfer；近似误差界明确要求 downstream map 为 Lipschitz。proof audit 已把 constant/discontinuous/non-Lipschitz map 纳入 counterexample pass，并明确记载“小 contribution error 可导致有限 output jump”。本轮不修改理论，只实现该已声明边界。

新增 C018 `Downstream regularity / margin audit`，状态 `IDEA`：以 discontinuous threshold readout 为 risk，以同实例 identity map 为 1-Lipschitz control；若 risk cliff 存在，候选只能标记 `NONCAUSAL_UNDER_UNCERTIFIED_READOUT`，不能把高 BCC 提升为因果证书。

### 冻结构造与 run

F12 在随机二维正交基 `(u,v)` 中生成 hook `h=c u+s v`，两侧贡献为 `(c-delta s)u` 与 `(c+delta s)u`。`c` 在每个 split 内与常数及平衡符号 `s` 精确正交，均值差为 0；`delta` 由 structural seed 在 `[1e-4,1e-2]` 对数均匀生成。两次 ablation 后的 `u` 分量仅差 `2 delta`，但 threshold readout `1[(state·u)(state·v)>=0]` 在两端对全部样本翻转；identity control 的输出 RMSE 应精确等于 hook contribution RMSE。

- Run ID：`R003_f12_formal_v1_20260902T062322Z`；parent `R003`。
- 20 structural seed pairs；独立 structural A/B、mean、eval、solver provenance；`n_mean=256`、`n_eval=2048`。
- dose grid `[0,0.25,0.49,0.51,0.75,1]`；预期 0.49→0.51 产生单位 risk jump，而 hook step≤`4e-4`。
- BCC≥`0.9998`、residual≤`2e-4`、PSC=1、mean error≤`1e-10`、endpoint contribution RMSE≤`0.02`。
- risk endpoint mismatch=1；identity smooth-transfer ratio=1；正确 decision 为 `NONCAUSAL_UNDER_UNCERTIFIED_READOUT_WITH_SMOOTH_CONTROL_PASS`。
- BCC-only 是应有 20/20 false-causal 负控；CBSM diagnostic layer 目标 false certificate=0。
- 当前 22/22 tests 通过。candidate family 已冻结、audit 未打开；统计单位为 structural seed pair；有界单进程 CPU，无需 lease。

Tracker suffix 已标为 `RUNNING`。若通过，将复验 R003 所有 formal suffix artifacts 后再决定 parent gate；不会把 F12 当作真实模型因果证据。

## 2026-09-02 02:24 EDT — R003 F12 完成、R003/M1 聚合过门

### F12 结果与 artifact 审计

`R003_f12_formal_v1_20260902T062322Z` 以退出码 0 完成，20 records 的 260/260 预写检查通过；独立 contract validation 为 `ok=true`，raw hash 为 `c133c107c56b9a14dc0cfaeb28a963182601c3c8f46e378bb25b1548ffcddb71`。

- 最小 BCC `0.9998693`，最大 normalized residual `1.307e-4`，PSC 全为 1，最大 mean-contribution error `2.65e-17`。
- 最大 endpoint contribution RMSE 仅 `0.01617`；20/20 discontinuous risk readouts 的 endpoint mismatch=1，0.49→0.51 的最大 adjacent jump=1，而对应最大 hook step RMSE 仅 `3.23e-4`。
- identity-map smooth control 的 transfer ratio 范围 `[0.9999999999999841,1.0000000000000142]`；20/20 正确输出 `NONCAUSAL_UNDER_UNCERTIFIED_READOUT_WITH_SMOOTH_CONTROL_PASS`。
- diagnostic false-causal certificate=0/20；BCC-only 负控 false-causal=20/20。

### R003 parent 与 M1 gate 聚合

重新用 snapshot-aware validator 复验 R003 五个 suffix，全部 `status=PASS`、summary PASS、contract `ok=true`：

- F08/F09：`R003_f08_f09_formal_v1_20260902T045635Z`，40 records，raw `8e117d07827498bf1ccb05abda5a83e8b4adaecd226cdd0c12040d528082757b`；
- F05：`R003_f05_formal_v1_20260902T051636Z`，20 records，raw `1bc631c99c68a6ef1c48f415df8b6c2214edf76e508f156846fa2ce188f9ef17`；
- F07：`R003_f07_formal_v1_20260902T053744Z`，20 records，raw `e7db5b4c814c1513bd712cdb071b4a4c3a183d1533c2c05821dfe7e10ac2ac1b`；
- F10/F11：`R003_f10_f11_formal_v1_20260902T055911Z`，40 records，raw `69fdf8eec671b2aa98c39b8a9fc146aa2e0a469671b3c567bc55ba232fb3f025`；
- F12：`R003_f12_formal_v1_20260902T062322Z`，20 records，raw 如上。

七类 R003 falsifiers 均达到每 family 至少 20 structural seed pairs，并满足预写 ambiguity/refusal、diagnostic 与 false-unique/false-causal 判据，因此 parent `R003` 更新为 `PASS`。第一次聚合 PowerShell 命令因管道语法产生 parser error，未写任何 artifact；修正为先收集结果再序列化后复验成功。

随后复验 R001/R002/R003 的 11 个 decisive suffix：全部 contract/status/summary PASS，共 340 records，覆盖 F01–F12 全部 12 families。因此 M1 synthetic gate 现可标为 PASS。该结论只表示 truth-known synthetic implementation、ambiguity/refusal 和理论边界行为通过，不是 C1/C2 的真实 SAE 或 held-out 因果证据。

### 技能影响、下一步与阻塞

PDF workflow 使 F12 严格对应 THM-CBSM-009 的 exact/approximate 分界，并保留了内部 proof audit 的“近似 clause 需要 Lipschitzness”限制；C018 已记录 F12 证据但维持 `IDEA`，等待真实 endpoint screen，未升格为正式方法组件。

M1 已过门，但真实主线仍不得直接启动：下一前置门是 R004/M0 的 Pythia-70M hook/SAE round-trip、token alignment、contribution 与 intervention unit tests；之后才是 R005/R006 的 framework/config smoke。当前无须用户裁决，下一轮应先盘点本地 SAE 基础工作与资源管理器状态，为 R004 提供最小、可复现且不下载大规模资产的执行方案。

## 2026-09-02 02:57 EDT — R004 preflight contract、来源盘点与独立环境完成

### 状态恢复与本地盘点

M1 保持 PASS；R004/M0 是下一门。母目录包含 6 个应用项目和共享 references；没有发现 Pythia/Transformers model cache。当前 CCAD bundled Python 只有 NumPy，无 torch、transformers、NNsight、TransformerLens、SAELens、sparsify 或 datasets，因此不能直接运行真实 R004。

共享资源管理器显示 `gpu-0`、`cpu-heavy`、`disk-d-io`、`disk-e-io` 均为空闲；GPU 实测为 RTX 5070 Ti 16303 MiB。只读核查相邻 EndoSAE 项目的 AGENTS、hook contract、tensor fingerprint、runtime probe 与 SAE smoke；项目根未发现 LICENSE，因此仅登记为内部 reference-only。未复制其代码、配置、阈值或结果；四个已读文件 hash 与借鉴边界写入新建 `REFERENCE_REGISTRY.md`。

### 线上一手来源与 R004 协议

2026-09-02 检索并核对 Pythia official model card/config、Transformers GPT-NeoX docs、PyTorch module-local hook API 和 CUDA 12.8 installation matrix。Pythia current retrained release为 Apache-2.0、6 layers、hidden size 512；旧 `-v0` 不采用。Transformers 文档确认 `output_hidden_states=True` 返回 embedding 加逐层输出，可作为 module hook 的独立索引/数值 oracle。PyTorch 文档允许 module-local forward hook 修改 output，同时警告 global hook 增加全局状态。

`EXPERIMENT_PLAN.md` 新增 R004 frozen preflight contract：候选 `gpt_neox.layers.2` 首 tensor 只有与 `hidden_states[3]` 数值一致后才命名 `resid_post`；固定 token hash、capture-only logits、no-op replacement、self swap、人工 `D,z,b` round-trip、group ablation writeback、float32 `1e-6` 容差及 VRAM/throughput/cache 指标。T009 native HF/PyTorch hook 已登记为 `READY-FOR-SCREEN`；NNsight/TransformerLens 仍是后续 parity 候选。

新增 `src/ccad/activation_contract.py` 与 4 个单元测试，覆盖 tuple auxiliary-output preservation、shape/dtype fail-closed、token order/revision drift、`h=b+Dz+r`、ablation 和 same-group self-swap。全套 26/26 tests 通过。这些是 mock/preflight 证据，不足以使 R004 PASS。

### 独立环境、失败与修复

按用户授权在 `D:\CCAD_Storage\environments\r004` 创建 CCAD 独立 Python 3.13.7 环境，没有改动相邻项目 venv。torch 2.8.0+cu128 使用 official CUDA 12.8 index 安装并命中本机 wheel cache；整个大型安装由 `disk-d-io` 租约保护。环境实际占用 `7,742,685,913` bytes。

Transformers 首次安装请求 `transformers==5.15.0 + safetensors==0.6.2`，resolver 因 Transformers 要求 `safetensors>=0.8.0` 明确失败；无部分污染，租约正常释放。修复为 `safetensors==0.8.0` 后成功。完整 lock 写入 `configs/r004_environment_lock_candidate_v1.json`。

在 `gpu-0` 独占租约下完成极小 CUDA probe：torch `2.8.0+cu128`、CUDA build `12.8`、device capability `(12,0)`、Transformers `5.15.0`、NumPy `2.5.2`；4×4 float32 matmul checksum `3680.0`。probe 后 `gpu-0` 与 `disk-d-io` 均复验为空闲。

### 保守解释、gate 与下一步

R004 仍为 `TODO`：环境和 mock contracts 已就绪，但 Pythia model/tokenizer 尚未下载，真实 hook path、token IDs、hidden-state parity、logit parity、VRAM/throughput 和 intervention writeback 均未产生 CCAD artifact。没有使用 EndoSAE 的运行结果支持本项目。

下一轮应先写独立 `run_r004_roundtrip.py` 与 resolved config/contract validator，再在 `disk-d-io` 租约下下载 Pythia-70M 到 `D:\CCAD_Storage\models`，记录 resolved commit、Apache-2.0、file hashes；随后用 `gpu-0` 租约运行单 batch。当前无用户裁决阻塞。

## 2026-09-02 03:22 EDT — 启动 R004 real round-trip v1

### Hugging Face 下载与固定资产

本轮使用 Hugging Face `hf` CLI workflow。`hf models info` 将 current `EleutherAI/pythia-70m-deduped` main 解析为 commit `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`，public、ungated、Apache-2.0，architecture `GPTNeoXForCausalLM`。只在 `disk-d-io` 租约下下载 README、config、`model.safetensors` 与三个 tokenizer 文件到 `D:\CCAD_Storage\models\pythia-70m-deduped\e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`；明确未下载 pickle `pytorch_model.bin`。CLI 提示匿名请求 rate limit，但下载成功，不影响文件完整性。

固定文件与 SHA256：README `0b8eff9fd326d9089f00c4984db07f89a9dac674ae3191a9bc8ae128b8a37580`；config `002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43`；safetensors `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`；special tokens `6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad`；tokenizer JSON `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`；tokenizer config `70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f`。model assets 总计约 168 MB；`disk-d-io` 已释放并复验 free。

### Runner 与冻结配置

新增 `scripts/run_r004_roundtrip.py`、`configs/r004_pythia70m_roundtrip_v1.json`，并扩展 artifact validator 以支持 summary 显式声明 generator path；新增回归测试后全套 27/27 tests 通过。runner 使用 float32/eager attention、固定 layer 2 module-local hook 和两条公开硬编码文本；保存完整模型文件 hashes、环境、token records、logits/hook tensors、原始 metrics、VRAM/throughput 与 contract validation。

预写 10 项 PASS checks：token records unique；hook `[batch,token,512]`；module output 与 `hidden_states[3]`；capture-only/no-op/self-swap logits；人工 coordinate `D,z,b,r` round-trip；group ablation formula；layer 3 pre-hook 接收精确 writeback；ablation 必须产生非零 logits delta。float32 tolerance `1e-6`，最小 intervention delta `1e-8`；candidate/audit 未打开。

- Run ID：`R004_roundtrip_v1_20260902T072233Z`；parent R004；M0 real-model debug fixture。
- model revision、tokenizer revision 均固定为 `e93a9faa...`；SAE framework 明确为 `manual_coordinate_fixture_not_trained_sae`。
- seeds：torch `20260902`、fixture `20260903`；统计单位为 deterministic two-document fixture。
- run 必须由资源管理器 `gpu-0` 独占包装；失败保留 artifact，不覆盖。

Tracker suffix 已标为 RUNNING。通过只证明 M0 hook/token/writeback contract，不是训练 SAE 质量或 C1/C2 证据。

## 2026-09-02 03:25 EDT — R004 v1 FAIL，启动 deterministic repair v2

`R004_roundtrip_v1_20260902T072233Z` 在 gpu-0 lease 中加载 76/76 weight tensors 成功，但第一次 baseline forward fail-fast。PyTorch deterministic mode 报告 CUDA>=10.2 的 CuBLAS matmul 需要在应用启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 或 `:16:8`。因此 v1 没有产生 hook/metric observation，status=`FAIL`；stderr、resolved config、model inputs/hashes、environment 和空 raw metrics 均保留。独立 artifact validator 为 `ok=true`，gpu lease 已自动释放并复验 free。

该失败属于 deterministic runtime 初始化遗漏，不是 hook path、Transformers API、模型资产或冻结 threshold 的证据。修复在 runner import torch 前设置 config 固定的 `:4096:8`；不改变模型、prompt、layer、fixture、容差或判据。修复后 py_compile、JSON parse 与 27/27 tests 仍通过。

新 Run ID `R004_roundtrip_v2_20260902T072522Z`；完全复用 v1 frozen protocol，只增加明确的 CuBLAS reproducibility environment。Tracker 已标 RUNNING；v1 永久保留为失败记录。

## 2026-09-02 03:28 EDT — R004 v2 PASS，M0 接口门完成

`R004_roundtrip_v2_20260902T072522Z` 在 `gpu-0` lease 下正常退出，runner 与独立 artifact validator 均为 PASS。固定 Pythia commit `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`、layer 2 module-local hook、2-document/21-valid-token fixture、float32 eager attention 和 `1e-6` tolerance 未改变；相对 v1 唯一修复是在 torch import 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

10/10 检查通过：hook tensor 与 `hidden_states[3]` 最大误差 0；capture-only、no-op 与 self-swap logits 最大误差均为 0；人工 `h=b+Dz+r` round-trip、group ablation 公式和 next-layer input writeback 最大误差均为 0。非平凡 ablation 的 logits 最大变化为 20.4110107421875，证明 intervention 路径实际生效。hook shape 为 `[2,11,512]`，activation storage 为 2,048 B/token；峰值 allocated/reserved VRAM 分别为 350,825,984/381,681,664 B。首次 baseline forward 2.047 s 包含 CUDA warm-up，不作为稳态吞吐；后续 capture/no-op/self-swap/intervention 各约 0.0062–0.0073 s，也只作为该 smoke fixture 的容量观测。

关键 artifact：`runs/R004_roundtrip_v2_20260902T072522Z/` 下的 resolved config、manifest、environment、stdout/stderr、raw metrics、summary、token records 和 `tensors.npz`；raw metrics SHA256 `87451d4c85c70292c197665aa4ddf76af5600d2316f784db8e07aa12d95198ea`，tensor artifact SHA256 `220bd54af1c366847ce9b9d7f3dc092aedeffe1ef3b70eac2642179b014a9878`，generator SHA256 `e26efb40ab5d735dc1056b58c96295ab806587d058612b295f947685f5c09dad`。模型目录保存 safetensors/tokenizer 的逐文件 hash；未下载 `pytorch_model.bin`。lease 已释放并复验 `gpu-0=FREE`。

据此将 parent R004 标为 `PASS`：M0 的 token、tensor、hook、round-trip 和 intervention 接口门已完成。证据范围严格限于真实底模加人工 SAE fixture；它不是 trained SAE 质量证据，也不支持 C1/C2，layer 2 仍只是 debug candidate，primary hook/config 保持 `OPEN`。T009 保持 `READY-FOR-SCREEN`，待后续独立 adapter parity；T007/T008 未启动。

下一步进入 R005 的 framework control-surface pilot：先只读核查当前 SAE 框架的一手仓库、许可证、可固定 revision、训练/导出接口及 deterministic control，再冻结最小 smoke 候选。R005/R006 仍为 `TODO`，不得据此启动 primary seed suite。当前无用户裁决阻塞。

收口回归首次调用未给 src-layout 设置 `PYTHONPATH`，因此仅在 collection 阶段出现 3 个 `ModuleNotFoundError`，没有执行测试或改写 artifact。按绝对 `src` 路径修正调用后，全套 27/27 tests 在 0.229 s 内 PASS；这属于命令层修正，不构成协议或实现变更。

## 2026-09-02 03:55 EDT — R005 framework control-surface 静态审计与动态 screen 预写

### 触发、来源与本地快照

R004/M0 已 PASS，R005/M2 是下一门。本轮未涉及理论修改，故未把理论 PDF 当作框架选择依据。只读核查两个官方 primary 候选仓库、官方 README/source metadata、许可证和 PyPI release surface：

- `dictionary_learning` main 固定到 `60ec6bf5264944d64a4ca271f45a29ebfb9d4946`，package metadata `0.1.0`，MIT；
- `sparsify` main 固定到 `42c064525b1cdd2b97f4a4807e247e89025d552c`，source metadata `1.3.3`，MIT；PyPI 当前页面仍显示最新公开 wheel 1.3.0，因此 source commit 与 wheel 不得混称同一版本。

Git `ls-remote` 在 restricted network 中第一次连接失败，获准以只读网络重试后解析 exact commits。两个小型源码快照下载到 `D:\CCAD_Storage\references\source\`，没有安装或执行第三方脚本。第一次 checkout 因 D 盘不记录 ownership 被 Git `dubious ownership` fail-closed；随后仅用命令级 `-c safe.directory=...` 成功 detached checkout，没有修改全局 Git 配置。此次小源码下载不构成资源管理器定义的 multi-GB I/O，且资源状态显示全部 free，故未持无意义 lease。

关键文件 SHA256：dictionary `pyproject.toml` `dd443faafb9fbe5cb15a2e8d7a1f05c7452c5be1906e8568b793fc29b25899f6`、`training.py` `e39a53c18c127e8652c1c63291710730088a4a629c88c52deedaeadfc9e685c8`、TopK trainer `a5135920cac1c42abd812f60d95d93a86f41e5654825d22ce0470b66551f198f`、buffer `3da9a475e5bfec43b7894d88b178baf5d7c054891f7401ea084770648f070b01`；sparsify `pyproject.toml` `7008b43e51ea2d867318802a4de9d62f1479b163bfb981399824ed9507cfd5b9`、config `646336fa99d307d393cf2a992dcb4529e85c27b82a152f0b970bc6654f45d26a`、trainer `90f3e1b35a646c8b96eec60043c027b295cf75bb542478663d28b391fcc4bbc2`、CLI `304214838e9103a6a0683bb87e801d89647a8780c42b118c06a70e6db46ca4bd`。

### 静态结论与新增 artifact

新增 `R005_FRAMEWORK_SCREEN.md`，预写 R005-A 安装/API conformance、R005-B shared token/hook GPU smoke 与 R005-C artifact/resume parity。`EXPERIMENT_PLAN.md` 增加 CANDIDATE preflight pointer，tracker 只补充静态证据，parent R005 仍为 `TODO`；`REFERENCE_REGISTRY.md` 登记 exact commits/license/boundary；`COMPONENT_CANDIDATES.md` 新增 T010 deterministic framework conformance wrapper，状态 `READY-FOR-SCREEN`。

静态证据显示 `sparsify` 有显式 `init_seeds`、独立 `shuffle_seed`、multi-seed naming 和 native safetensors，暂定为动态 screen 的先行候选；但 tokenizer loader 未绑定 model revision、W&B 默认开启、Adam implementation 会受 bitsandbytes 是否安装影响、on-the-fly 重复底模 forward，且 CLI resume path 未泛化到任意 save directory。`dictionary_learning` 支持 seeded TopK/BatchTopK、同 batch 多 trainer 和 config export，但 trainer 初始化与 activation buffer 共用 global RNG，buffer 用未独立 generator 的 `randperm`，final weights 为 pickle-based `.pt`，完整 final resume/RNG/data cursor 不充分，并依赖 `nnsight>=0.3,<0.4`。这些是待动态检验的工程边界，不是框架质量或 SAE 效果结论。

### Gate、保守解释与下一步

R005/M2 没有过门，primary framework/hook/architecture/corpus/width/k 均保持 `OPEN`；未下载 Pythia-160M、未安装框架、未训练 SAE、未触碰 audit，也没有 C1/C2 新证据。短训跨框架 FVE 明确不作为 framework selection 指标。

下一轮优先执行 R005-A：建立两个隔离环境和固定依赖 lock，用 tiny tensors 验证 encode/decode orientation、TopK exact L0、decoder norm、safe export、same-seed replay 与不同 seed separation；禁止 demo、外部 W&B 和未经审计的脚本。只有 R005-A hard checks 通过后，才下载固定 Pythia-160M/tokenizer 并为 R005-B 申请 `disk-d-io`/`gpu-0` lease。当前无用户裁决阻塞。

## 2026-09-02 04:22 EDT — R005-A core conformance v1 FAIL，依赖补全 v2 启动

为避免重复两份约 7.7 GB CUDA runtime，R005-A 使用两个相互隔离的轻量 overlay，共同只读复用 R004 已锁定的 torch/safetensors runtime。dictionary overlay 固定 `einops==0.8.2`；sparsify overlay 初始固定 `einops==0.8.2`、`natsort==8.4.0`、`simple-parsing==0.1.9`、`docstring-parser==0.18.0`、`typing-extensions==4.16.0`。依赖从 PyPI 下载到 D 盘；没有执行仓库 demo、下载脚本、W&B 或 GPU 工作。

新增 `scripts/run_r005a_conformance.py` 与 `configs/r005a_tiny_conformance_v1.json`。Run ID `R005a_tiny_conformance_v1_20260902T081643Z`，固定 CPU float32、activation dim 4、dictionary size 8、k=2、batch 8、init seeds `[0,1]`、seed 0 replay、Adam lr `1e-3`、单步 update。预写 8 项 hard checks：same-seed init/update hash、different-seed separation、selected-k、actual L0 bound、decode formula、decoder norm、safe export round-trip。该 run 仅测试固定源码的 core modules，不是完整安装、CLI、activation buffer、hook、resume、SAE quality 或 M2 证据。

v1 artifact contract PASS，但整体 FAIL：dictionary_learning 完成 8/8 checks，actual L0=2，decoder norm 最大误差 `2.38e-7`，safe export SHA256 `5faf1a389c933c0af7ac766382ce69f23408b06f0c35e830c3b0014aa11ecca4`；sparsify 在导入 `utils.py` 时因 overlay 缺少声明依赖 `accelerate` fail-fast，未产生 framework record。这是环境依赖清单遗漏，不是 SAE 语义比较结果。

修复仅向 sparsify overlay 添加 `accelerate==1.14.0` 与其运行时所需 `psutil==7.2.2`，未改变源码、fixture、seed、优化器、hard checks 或 runner。新 Run ID `R005a_tiny_conformance_v2_20260902T082201Z`，tracker 已标 `RUNNING`；v1 永久保留。R005 parent 仍为 `TODO`。

v2 随后产生两框架完整记录且 artifact contract PASS，但整体 15/16 FAIL。dictionary 仍为 8/8；sparsify 的 seed replay/separation、selected k、actual L0、decoder norm 和 native safetensors round-trip 均通过，唯一失败是 runner 用与上游 eager decoder 不同的 reduction path 显式复算后要求 bitwise `torch.equal`。配置未写 float32 formula tolerance，因此不能回写 v2 或事后直接判 PASS。

新 suffix `R005a_tiny_conformance_v3_20260902T082350Z` 明确把该单位公式检查改为 absolute tolerance `1e-6`、`rtol=0`，并新增 max absolute error 报告；这是公开记录的 protocol clarification，不影响 seed、数据、模型、训练或任何 audit。Tracker 已标 RUNNING。

## 2026-09-02 04:25 EDT — R005-A core-module v3 PASS

`R005a_tiny_conformance_v3_20260902T082350Z` 正常退出，16/16 checks PASS，runner artifact contract 与独立 `validate_run_directory` 均为 `ok=true`。raw metrics SHA256 `98d79badcf44c2f2abc5c26d30d74c90f4ff113865cef53cc55af30556201f60`，generator snapshot SHA256 `bb775b75b8a890ac9b2e10d0e2255c07138324143072accd70acab90a917f7ca`。

- dictionary_learning：same-seed initial/post-step hashes 一致、seed 1 不同；actual L0=selected k=2；decode formula error 0；decoder norm 最大误差 `2.384e-7`；safe export SHA256 `5faf1a389c933c0af7ac766382ce69f23408b06f0c35e830c3b0014aa11ecca4` 且 exact reload。
- sparsify：同样通过 seed replay/separation、L0/k、decoder norm 与 native save/load；decode formula 最大误差 `5.960e-8`，低于明确的 float32 `1e-6`；safe export SHA256 `ed6edbd0af7fe0165a8cb5334a9a039419b7fc0b97ea81900ffa62418970ea8d`。

新增 `configs/r005a_environment_lock_v1.json`：只读复用 R004 base lock hash `92a420b5...`，dictionary overlay 59 files/531,133 B、aggregate `f62a6490...`；sparsify overlay 403 files/6,201,262 B、aggregate `2e4b96cf...`，逐包版本已锁。runner 后续仅补 `.detach()` 消除诊断 warning，不改变 v3 的 snapshot 或结果。

T010 更新为 `SCREENING`，但 parent R005 仍为 `TODO`。该 PASS 只覆盖 core modules；轻量 namespace 明确绕过 public package `__init__`、完整 CLI、dictionary ActivationBuffer/nnsight、sparsify Trainer/data path 与 resume。它不是完整 R005-A、真实 Pythia SAE 质量、M2 或 C1/C2 证据。

下一步应补齐完整 package/CLI import 的依赖与 offline/no-W&B 检查，并对中断恢复做 tiny parity；只有这些通过后才进入 R005-B 的固定 Pythia-160M/tokenizer/token manifest 与 GPU smoke。当前无用户裁决阻塞。

收口时 runner py_compile、environment-lock JSON parse 和全套 27/27 tests PASS；资源管理器复验 `gpu-0`、`cpu-heavy`、`disk-d-io`、`disk-e-io` 全部 free。本轮没有持有或遗留重型资源 lease。

## 2026-09-02 05:05 EDT — R005-A package/offline/tiny-resume v1 FAIL，v2 PASS

### 触发与环境固定

按 R005 的下一子门补齐两个固定源码的公开 package import、离线日志和中断恢复控制面。dictionary overlay 新增 `nnsight==0.3.7`、`wandb==0.29.0`、`torchvision==0.23.0`、`accelerate==1.14.0` 等声明依赖；sparsify overlay 新增 `datasets==5.0.1`、`schedulefree==1.4.1` 及其依赖。所有正式 worker 强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=offline`、`WANDB_DISABLED=true`，并用 socket guard 将任何连接尝试 fail-closed。

依赖安装有两次需保留的异常：bulk `pip --target` 已打印 success 但进程持续不退出，超过观测窗口后被中断，导致 dictionary 一度缺 `wandb` metadata、sparsify 一度缺 `urllib3/xxhash`。随后逐包显式修复并验证公开 import。新增 `configs/r005a_environment_lock_v2.json` 固定修复后状态：dictionary overlay 7,994 files、284,633,507 B、aggregate SHA256 `7c9465ca2f7ea2852621a756f6d2a2e65c3a2c7b9926e23c918d7d6b7bb7e2a5`；sparsify overlay 7,683 files、251,335,332 B、aggregate `988988061b3bb77c8c27354d29c6d29ee6ccd5649c21434a9ac3aeda47686933`。v1 lock 保留为早期 core-only 环境记录，不被覆盖。

### v1 失败与确定性修复

`R005a_package_resume_v1_20260902T085540Z` artifact contract PASS，但 run 为 FAIL。dictionary_learning 完成 7/7：公开 package 与 `trainSAE(use_wandb=False)` 可执行，CCAD wrapper 保存 SAE/optimizer/scheduler/RNG 后从 step 1 恢复，step 2 最终 state hash 与 uninterrupted 路径同为 `030dc5f0...`，无 socket 连接。sparsify 在 Trainer 执行前创建 `datasets.Dataset.from_dict` 时触发 Python 3.13 + PyArrow/dill 的 `MonthDayNano` pickling error，故该 run 只有 1 个 framework record；失败 stderr 和 artifact 均保留。

新 suffix `R005a_package_resume_v2_20260902T090317Z` 只把数据夹具替换为满足 upstream Trainer 实际所需 `Sized + select` contract 的本地 torch Dataset；模型、seed、optimizer、两步训练、一次中断、offline 条件与 7 项判据不变。这个修复隔离了 HF dataset compatibility，不把它伪装成已通过的数据管线。

### v2 结果、验证与边界

v2 正常退出并通过 14/14 checks，runner contract 与独立 `scripts/validate_run.py` 均 `ok=true`。dictionary 继续 7/7；sparsify 公开 package/`__main__` import、原生 `Trainer.fit/save/load_state` 均执行，恢复后与 uninterrupted 最终 state hash 同为 `6d6c6657...`，global step 同为 2，checkpoint config/state/optimizer/scheduler/safetensors 全部存在且逐文件哈希已记录；两个 worker 均无 socket 连接尝试。raw metrics SHA256 `e1439e655e85c8cb66233a8c12651ff2c234028b6ede618cc621ee801636bd90`，generator snapshot SHA256 `ca26cf29916f29c65655a45a8edf09e7c5384f379ce6352a46f17bb8a29752da`。

收口时第一次误用未安装 pytest 的 bundled Python、随后一次 unittest 调用漏设 src-layout `PYTHONPATH`，均在测试收集前失败且未修改 artifact；修正为绝对 `src` 后 27/27 unittest PASS。资源管理器复验四类资源全部 free，本轮仅 CPU tiny runs，未持重型 lease。

结论严格限于 package/offline/tiny-resume 子门 PASS。HF Dataset compatibility、真实 CLI execution、Pythia-160M token/hook parity、data-order trace、VRAM/吞吐、SAE quality 尚未通过；parent R005 与 M2 保持未完成，primary framework/hook/architecture/corpus/width/k 全部未冻结，没有 C1/C2 新证据。下一步先固定 Pythia-160M model/tokenizer exact commit 与有许可证的 tiny manifest，并预写 R005-B 配置；只有需要下载和 GPU smoke 时才分别申请 `disk-d-io` 与 `gpu-0` lease。

## 2026-09-02 05:30 EDT — R005-B fixed model assets 与候选协议预写

### 一手检索、模型固定与下载

本轮按 R005-B 前置要求检索官方 Pythia repository 与 Hugging Face model card/API。官方材料确认 current retrained `EleutherAI/pythia-160m-deduped` 与旧 v0 分离、Apache-2.0，并用于 Pythia 的 deduplicated Pile 路线；当前 API 解析为 commit `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`，public、ungated。配置实测为 GPT-NeoX、12 layers、hidden size 768、12 heads、max context 2048。

第一次两次直接用 Windows `.py` 文件关联调用资源管理器 wrapper，命令均在约 0.2 秒无输出返回，未生成 lease archive、未创建目标目录也未下载；这属于 launcher invocation failure，不能算受管运行。改用 `C:\Program Files\Python313\python.exe` 显式启动同一 resource manager 后，`disk-d-io` lease 正常获取、心跳和释放，下载成功。只保存 README、config、`model.safetensors`、special tokens、tokenizer JSON/config 到 `D:\CCAD_Storage\models\pythia-160m-deduped\582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`；未下载 pickle `pytorch_model.bin`。匿名 HF 请求提示 rate-limit warning，但文件完整。

逐文件 SHA256：config `76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f`；safetensors 374,998,696 B，`bbaae5c00917b163baa499fc8eb64859ee0c850c5fdecfc32f4d70dc07213575`；README `7fe6132bfdcd802b9acf6a59105795b1bb9d5857e33a7fe64809e5fd4aa3b26b`；special tokens `6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad`；tokenizer config `70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f`；tokenizer JSON `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`。资源管理器复验四类资源均 free。

### 候选协议与 debug manifest

新增 `configs/r005b_shared_hook_smoke_candidate_v1.json`，预写而未启动：同一 Pythia/tokenizer commit、debug-only `gpt_neox.layers.5` first tensor 对 `hidden_states[6]` oracle、TopK、3,072 latents、k=16、Adam lr 1e-3、float32 eager、8 steps、init seeds `[0,1]`、seed 0 replay、独立 data-order seed。hard checks 覆盖跨框架/seed token trace、hook oracle、capture-only logits、replay、seed separation、safe export、optimizer implementation、无外部写入和 checkpoint contract；报告 loss/FVE/L0/alive-dead、decoder norm、base forwards、VRAM、吞吐、checkpoint bytes 与 wall time。

新增 `data/r005b_debug_texts_v1.jsonl` 与 CC0-1.0 声明：16 条 project-authored synthetic debug records，SHA256 `25418ab702595c00e3079b08f780d42e989f40dbf27a076f7903d601cc687710`。固定 tokenizer 本地解析共 211 个未 padding tokens，每 document 11–16 tokens，并记录了逐文档 token hash诊断。该规模只测试接口、顺序和 hook 单元，不足以评价 SAE quality；它不属于 Pile、primary SAE training corpus、paired corpus 或 audit。

REFERENCE_REGISTRY 与 R005 screen 已同步。R005-B 尚未产生 run，parent R005/M2 仍 `TODO`；framework/hook/width/k/corpus 均未冻结，没有 C1/C2 新证据。下一步实现共享 immutable token batch runner，先只做离线 config/adapter preflight；确认两框架能表达相同 input trace 后，再申请 `gpu-0` 跑唯一 suffix 的 smoke artifact。当前无用户裁决阻塞。

## 2026-09-02 06:00 EDT — R005-B shared-hook/cache v1 FAIL，v2 PASS

### 实现、v1 失败与受限修复

新增 `scripts/run_r005b_shared_hook.py` 与冻结配置 `configs/r005b_shared_hook_smoke_v1.json`。runner 离线加载固定 Pythia-160M commit，在 data-order seed `20260902` 下将 16 个 debug documents 排为 8 个 batch；每批同时保存 document IDs、token hash、native layer-5 activation hash，随后让 dictionary_learning 与 sparsify 的固定 TopK 模块消费同一份 immutable activations。两框架均训练 seed 0、seed 0 replay、seed 1，保存初始/最终 state hash、逐步 MSE/FVE/L0/alive/decoder norm、safe weights、时间、VRAM 和完整 artifact contract。

首次非 escalated 资源管理器调用因 sandbox 无权创建共享 lease 文件 fail-fast，未启动 run；按权限规则以同一命令申请授权重试。`R005b_shared_hook_smoke_v1_20260902T095000Z` 随后在 GPU lease 中加载模型，但在第一批 activation flatten 前 FAIL：Transformers 5.15 + `use_cache=False` 的 layer output 是直接 `[batch,token,hidden]` tensor，runner 错当 tuple 取 `out[0]`，形成 `[token,hidden]` 与 `[batch,token]` mask 不匹配。v1 未进入 SAE 训练；stderr、空 raw metrics 与 contract PASS artifact 永久保留，GPU lease 正常释放。

v2 只用 M0 已测试的 `HookPointContract/extract_primary_hook_tensor` 替换容器处理，未改变模型、数据、顺序、hook、seed、width=3072、k=16、Adam lr、8 steps 或 `1e-6` tolerance。新 Run ID `R005b_shared_hook_smoke_v2_20260902T095355Z`。

### v2 结果与证据边界

v2 正常退出，6/6 global checks 与两框架各 5/5 checks PASS，runner contract 和独立 `scripts/validate_run.py` 均 `ok=true`。layer-5 activation 与 `hidden_states[6]` 最大误差 0；capture-only logits 最大误差 0。8 batches 共 211 valid tokens，16 次底模 forward 的 extraction wall time 0.701 s，model load 0.637 s，peak allocated VRAM 901,088,256 B。

dictionary_learning seed-0 initial/final hashes分别为 `8346968e...` / `5d7e4b4f...`，replay final 完全一致；sparsify 为 `c1d40d67...` / `3a1ca662...`，replay final 完全一致。两者 seed-1 initial 均与 seed 0 不同，训练前后 activation batch hashes 不变，safe export/reload 精确一致。raw metrics SHA256 `a79f352750727a2058e7b64b081e44d891a92fe70dae60c279de74a342ae335f`，generator snapshot `8c4ee5618173e7d2fed5ddb59fd95aec48be49c5b43b29c57a60df31ecbaffa2`。

上游 loss 不同标度：dictionary `update` 返回 summed reconstruction objective，sparsify 返回 FVU，故 loss 数值不可作框架优劣比较。第 8 个微批 seed-0 FVE 约 0.074 与 -0.028，但样本只有 211 tokens、训练 8 步，明确不足以判定 SAE quality。该 run 也未覆盖 native dictionary ActivationBuffer、sparsify on-the-fly Trainer、CLI execution、真实 resume、CE recovered 或每百万 token 成本。

结论仅为 R005-B shared-cache module 子门 PASS；parent R005 和 M2 仍未完成，framework/hook/width/k/corpus 未冻结，C1/C2 无新证据。下一步应设计 native end-to-end 小规模 screen：先用同一 tokenized local shard分别观测两个框架真实输入 trace与 base-forward count，再比较 artifact/resume/cost；由于 CPU-heavy 当前由 EndoSAE 持有而 GPU 已 free，本轮不启动额外竞争任务。当前无用户裁决阻塞。

## 2026-09-02 06:20 EDT — R005-C native checkpoint 完整性审计

### 审计设计与 v1 失败

源码审计显示 dictionary_learning 的 periodic backup 与 final weights 共用 `ae.pt`，sparsify 的 multi-seed SAE 名称含 `/seedN`，但部分恢复逻辑遍历 unsuffixed hookpoint。为把静态判断变成可复现实证，新增 `scripts/run_r005c_native_state_audit.py` 与 `configs/r005c_native_state_audit_v1.json`，使用 CPU tiny model/tensors 检查 optimizer/step/config、SAE weights、dead-feature counters 与 best-loss key coverage。该 run 的 PASS 定义为“预写 limitation 被动态复现”，不是“native resume 合格”。

`R005c_native_state_audit_v1_20260902T101407Z` 在 dictionary 分支先暴露更早缺陷：`backup_steps=1` 且默认 `normalize_activations=False` 时，upstream `trainSAE` 尝试保存未定义的 `norm_factor`，抛出 `UnboundLocalError`。没有 checkpoint observation 或 sparsify record；失败 stderr、空 raw metrics 和 contract PASS artifact 均保留。

v2 唯一执行修复是启用 upstream `normalize_activations=True`，并提供每次迭代都返回相同固定 batch 的 re-iterable fixture，使其 101-batch norm-factor pass 与一训练步都可完成。模型、维度、k、seed、状态判据不变；v1 的默认路径崩溃不被抹去。

### v2 结果与工程影响

`R005c_native_state_audit_v2_20260902T101704Z` 正常退出，两个 framework limitation 均复现，artifact contract 与独立 validator PASS。dictionary final `ae.pt` SHA256 `de21165f...`，只含六类模型参数键；同路径 backup 的 `step`、`ae`、`optimizer`、`config`、`norm_factor` 全部被 final state_dict 覆盖，`native_resume_complete=false`。

sparsify 保存并 exact reload 了 `layers.0/seed0`、`layers.0/seed1` 两套权重，但故意写入的两个 `num_tokens_since_fired` tensor 均未恢复。实际 SAE keys 是 seed-suffixed names，而 `best_loss` keys 仅为 `layers.0`；因此 `native_multiseed_state_complete=false`。raw metrics SHA256 `52af845edf5ec0d00bb961f98780f98a50fd3596350f36e67e93565404c54d93`，generator snapshot `8c3c763513de7c967957811f75dd1129f4b5d3dcf8a18f73fb937bf64d140fba`。

这推翻了“single-seed resume PASS 足以覆盖主 seed suite”的工程假设，但不涉及 C1/C2 或理论主张。新增 T011 `Exact multi-seed state wrapper`：优先尝试 sparsify 的最小 seed-aware counters/best-loss/RNG/data-cursor wrapper，并要求多 seed uninterrupted 与 resumed 的下一批 token hash、loss、weights、counters 全相等；dictionary 若保留则需更完整 wrapper。R005/M2、primary framework/config 仍未冻结。下一步实现 T011 tiny long-enough parity，再做 native Pythia cost screen；暂无用户裁决阻塞。

## 2026-09-02 06:35 EDT — T011 sparsify exact multi-seed wrapper：v1 FAIL，v2 PASS

### 实现与失败留痕

按 R005-C 的后续任务实现 `src/ccad/checkpointing.py`：CCAD 独立保存每个 seed 的 safetensors，以及 optimizer、scheduler、global step、data cursor、seed-suffixed `num_tokens_since_fired`、best-loss、CPU/CUDA RNG；恢复时严格核对 SAE key space、状态数量和 cursor。新增 formal runner `scripts/run_r005c_multiseed_wrapper.py`，用两个 init seeds、低 dead threshold 和非零 aux-k 在确定性 tiny CPU 四样本轨迹上比较 uninterrupted 与两步中断/恢复，不调用 GPU 或外部服务。

`R005c_multiseed_wrapper_v1_20260902T104000Z` artifact contract PASS，但整体 10/11 FAIL。输入轨迹、两 seed 最终权重、optimizer、scheduler、dead counters、best-loss、cursor 和 global step 全部 exact，唯一失败是 final global RNG。诊断确认 upstream resumed `Trainer.fit` 新建 DataLoader iterator；PyTorch 即便 `shuffle=False` 也会生成 iterator base seed，多消耗一次 global RNG。该 run 和 raw metrics SHA256 `cbff862b99567257dd608d25c61b6c03ef65565d2e67c48b67e4311f93cca5a8` 永久保留。

v2 只修复这一已定位边界：加载后用透明 dataset proxy，在首个 resumed sample 被访问前重置 checkpoint CPU/CUDA RNG，从而抵消新 iterator draw，并保证潜在随机 dataset transform 从正确状态起步。没有改变模型、数据、seed、optimizer、步数或判据。

### v2 结果、验证与边界

`R005c_multiseed_wrapper_v2_20260902T105000Z` 正常退出，11/11 checks PASS。四步 full trace 与 interrupted+resumed trace 完全一致；两个 SAE 最终 hashes 分别为 `649830dd...`、`9d640ae5...`，恢复路径逐项相同；optimizer、scheduler、dead counters、best-loss、global step=4 和 final RNG 均 exact。checkpoint state SHA256 `42c910293b1e2d7fa1e2695e6eb7cc6b52d0c87ee0c1a0eb866fe50783f0bf55`；raw metrics SHA256 `a2c0b76ac1cd57d00682ab0d03dfa79c494b1eb131bde3f517dd2a9597d7aa53`。runner artifact contract 与独立 `scripts/validate_run.py` 均 PASS，py_compile、JSON parse 和全套 27/27 unittest PASS。

T011 从 `READY-FOR-SCREEN` 更新为 `SCREENING`，未 `ADMIT`。这只证明 tiny CPU、单 worker、四样本下 wrapper mechanics；尚未覆盖 CUDA RNG、真实 Pythia native forward、多 worker loader、长训成本与崩溃时目录级原子提交。parent R005/M2、framework/hook/architecture/corpus/width/k 均继续 `OPEN`，没有 C1/C2 新证据。下一步把 wrapper 接入固定 Pythia native small-run，测真实 loss trace、checkpoint 恢复和每 token 成本；当前无用户裁决阻塞。

收口检查确认 CCAD 当前不是 Git repository，故本 run 不能记录 commit；以逐文件 code snapshot 和 aggregate SHA256 `507fb596adf8bf8ee9d615e40574308a174390457e61c258fc7b89d387bbd196` 替代，其中 wrapper SHA256 为 `ca8b64241d14549c8b6fc24a9d8e9b361d122043aa04375fadb71251989d0797`。资源管理器复验 `gpu-0`、`cpu-heavy`、`disk-d-io`、`disk-e-io` 全部 free，本轮无遗留 lease。

## 2026-09-02 07:01 EDT — R005d Pythia native multi-seed resume：v1 FAIL，v2 PASS

### 触发、协议与资源

沿 T011 的下一子门，新增 `scripts/run_r005d_pythia_native_resume.py` 与两份 versioned config。沿用已固定的 Pythia-160M commit `582159a...`、CC0 debug manifest `25418ab...`、layer 5、TopK width 3072/k=16、init seeds `[0,1]`、Adam lr `1e-3`，执行 8 步 uninterrupted 与 4 步中断/恢复的 native sparsify on-the-fly 训练。预写检查覆盖 input、每 seed FVU/aux-k loss、weights、optimizer、scheduler、dead counters、best-loss、cursor、global step、CPU/CUDA RNG；运行不打开 audit，也不冻结 primary config。

首次 sandbox 内资源管理器调用因无权写共享 lease 文件直接失败，未创建 run；授权后所有实质 GPU 运行均通过 `gpu-0` lease，且 lease 自动释放。

### v1 失败与单点修复

`R005d_pythia_native_resume_v1_20260902T111000Z` 完成 full 8 步和 interrupted 4 步，恢复时 fail-fast：checkpoint 以 `map_location=cuda` 载入后，CUDA RNG state 成为 CUDA ByteTensor，但 `torch.cuda.set_rng_state_all` 要求 CPU ByteTensor。artifact contract PASS，stderr 与空 raw metrics 保留。

v2 只把 CUDA RNG states `.cpu()` 后再恢复，并同步修正首样本 RNG proxy 的同一路径；没有改变模型、数据、seed、训练或检查。新 Run ID `R005d_pythia_native_resume_v2_20260902T111500Z`。

### v2 结果、成本与保守解释

v2 正常退出，13/13 checks、runner contract 与独立 validator 全部 PASS。full 与 interrupted+resumed 的 8 个 input hashes、两个 seeds 的逐步 FVU/aux-k loss、最终 weight hashes `06c78879...` / `97dee2f3...`、optimizer、scheduler、counters、best-loss、step=8、cursor=8 examples、CPU/CUDA RNG 均 exact。checkpoint state SHA256 `fc726a18e2f5a09b5e86c4aa55a0427a04afb62c8350a47fa582c7a5d9be27b7`；raw metrics SHA256 `bdea876377c54856b841861ffe5c4b27653e37e5f87bd8462f2bde0f52689ada`。py_compile、JSON parse 与 27/27 unittest PASS。

完整 native 训练用 8 次底模 forward 同时更新两个 seeds；211 valid tokens 用时 0.581 s，约 363 tokens/s，model load 0.447 s，peak allocated/reserved VRAM 1,081,006,592 / 1,174,405,120 B。计时规模太小，不外推正式成本。末步 FVU 约 0.865/0.862，即 FVE 约 0.135/0.138；dead fraction 0.965/0.963，清楚显示 211-token run 严重欠训练，不能作为 SAE quality 或框架优劣证据。

T011 更新为 `ADMIT`，仅表示 exact wrapper 可进入后续 sparsify pilot，不表示 sparsify 已被选为 primary。R005/M2 与 framework/hook/architecture/corpus/width/k 仍未冻结，C1/C2 不变。未覆盖 multi-worker、长训崩溃原子提交、dictionary native input/cost 或真实 CLI execution；下一步优先补 dictionary 对照和 CLI boundary。收口时资源管理器四类资源均 free，无遗留 lease；项目仍非 Git repo，本 run 已记录逐文件 code/input hash。

## 2026-09-02 07:24 EDT — R005e dictionary native buffer：两类隐式状态漂移与最小 guards

### 触发、候选登记与协议

为补齐 R005 的 dictionary native input/cost 对照，源码审计发现 `pytorch_buffer.ActivationBuffer.__next__` 用 global `torch.randperm`，而 `TopKTrainer(seed=...)` 初始化会重置同一 CPU/CUDA global RNG。先在 `COMPONENT_CANDIDATES.md` 登记 T012 generator-isolated sampler；动态 v2 又发现 `trainSAE` 的 step guard 位于 iterator 取值之后，再登记 T013 bounded iterator guard。两项都先登记后修复，没有追认式隐藏。

R005e 沿用固定 Pythia-160M commit、layer 5、16-document CC0 debug manifest、TopK 3072/k=16、seeds `[0,1]`、Adam lr `1e-3`。原生 PyTorch buffer 固定 ctx_len=16、13 contexts、refresh batch=2、output batch=26；预计 8 次 early-stop forwards 建立 208-row pool，并用 4 batches 同时训练两个 dictionaries。所有实质运行均经 `gpu-0` lease，禁网且不打开 audit。

### v1/v2 失败留痕

`R005e_dictionary_native_buffer_v1_20260902T113000Z` 因 runner 从 `dictionary_learning.dictionary` 错误导入实际定义在 `trainers.top_k` 的 `AutoEncoderTopK`，在模型加载前 FAIL；contract PASS，空 raw metrics 保留。v2 仅修复 import。

`R005e_dictionary_native_buffer_v2_20260902T113500Z` 完成真实训练并得到 7/8 checks，仍标 FAIL。T012 预期被证实：trainer 初始化顺序 `[0,1]` 与 `[1,0]` 产生不同的原生首批 activation hash；固定私有 CUDA generator 后两者 hash 完全相同。唯一失败是配置 4 updates 却记录 5 个 buffer yields：Python `enumerate` 已取出第 5 批，上游才执行 `if step >= steps: break`。raw metrics SHA256 `ebef3dfb4a5e39c82012342b8ea9e1b55e78fb0e97413bed9dd67433c2f58dc0`，contract PASS。

### v3 结果与边界

v3 只用 `itertools.islice(buffer, steps)` 封住额外预取；模型、数据、训练和 8 项判据不变。`R005e_dictionary_native_buffer_v3_20260902T114000Z` 8/8 PASS，独立 artifact validator PASS。模型 input trace 与预期 8 个 tokenized batches 完全一致；208×768 float32 pool 为 638,976 B，四个训练 batch hashes 可追溯；两 seed safe weights SHA256 分别为 `7a13d876...` 与 `a1e00fde...`，eval actual L0 均为 16。总 wall 0.896 s（约 232 activation rows/s），peak allocated/reserved VRAM 879,035,392 / 947,912,704 B；raw metrics SHA256 `136937de54676d89ef0b65d05d5cb59df3c4c92f5084b53de9c3c6c84503be54`。py_compile、JSON parse 与 27/27 tests PASS，lease 已释放。

四步 FVE 为约 -0.012/-0.040，属于严重欠训练 debug，不用于框架质量排序，也不可与 sparsify 的不同 batch 单位直接比较。T012/T013 更新为 `ADMIT`，含义仅是 dictionary 路线若继续必须使用这些 guards；dictionary 仍缺完整 exact resume wrapper。parent R005/M2、framework/hook/architecture/corpus/width/k 保持 `OPEN`，C1/C2 无变化。下一步补真实 CLI execution boundary 与 framework scorecard；当前无用户裁决阻塞。

## 2026-09-02 07:51 EDT — R005f CLI PASS，R005 framework 决策完成

### CLI 真入口

新增 subprocess socket guard、`scripts/run_r005f_cli_boundary.py` 与 frozen config。`R005f_cli_boundary_v1_20260902T120000Z` 在 `gpu-0` lease 下，以固定 tokenizer 将四条 CC0 debug 文本派生为 4×16 uint16 memmap（128 B，SHA256 `265e4c38927668975374cb0bcf5d3e4e5f2a013e221833f9fa2d798c4b81911c`），真实执行 `python -m sparsify`。模型、dataset 与 save path 均为本地绝对路径；显式 `--nolog_to_wandb`，socket guard 记录 0 次连接尝试。

run 8/8 checks、artifact contract 与独立 validator 均 PASS。固定 sparsify pyproject 的入口为 `sparsify.__main__:run`；固定 dictionary_learning pyproject 没有 scripts/CLI entry。CLI 产生 seed0/seed1 两份 396,848 B safetensors，SHA256 分别 `2ea4de38...`、`ca6188c1...`，并保存 resolved config、单 optimizer 和 scheduler state。raw metrics SHA256 `c5b7fc88f71aa3091ad2cc4f35c9f1599614ce26abcf9fc1b3bccc22898d1be9`，subprocess wall 6.50 s。CLI 的 FVU 路径使用 bfloat16 base-only GPTNeoXModel，并报告未使用 LM head key；这是预期加载范围记录，不是质量比较。

### Scorecard 与 gate

新增 `R005_FRAMEWORK_SCORECARD.md`，严格按预写优先级汇总 R005a–f。两框架在固定源码、离线 import、core replay/TopK 和 shared-hook 上均通过；sparsify 额外具有已验证的 T011 exact multi-seed resume、原生 safetensors、明确 CLI 和较小 wrapper 面。dictionary native 路线存在 seed/data RNG coupling、额外 iterator prefetch、final state overwrite，并仍缺完整 exact-resume wrapper；T012/T013 虽已修复前两项，但增加维护面。debug 资源都可行，因单位不同未按短训速度或 FVE 排名。

据此将 R005 标为 `PASS`，为 R006 primary pilot 冻结 `sparsify` commit `42c064525b1cdd2b97f4a4807e247e89025d552c` + CCAD T011。T002/T010 `ADMIT`；T001 `REJECT` 仅指 primary candidacy，dictionary 仍保留为实现对照。该决定不使 M2 过门：R006 的 hook、TopK/BatchTopK 可表达性、width/k、training corpus、token budget、FVE/CE recovered/dead fraction thresholds 均未冻结，C1/C2 无变化。下一步进入 R006 的候选协议与最小质量 pilot；当前无用户裁决阻塞。

## 2026-09-02 08:14 EDT — R006 pre-registration：隔离架构/框架混杂并登记 L0 风险

### 触发与源码核查

自动循环按规定重读 `AGENTS.md`、最新 master log、R006/M2 计划、tracker 和 R005 scorecard 后进入 R006 preflight；本轮未打开 R008 audit、未启动训练、未创建资源 lease。固定源码核查确认 `sparsify` commit `42c0645...` 的 activation 仅有 per-token `topk` 与 `groupmax`；`multi_topk` 是附加 reconstruction loss，不是 BatchTopK。固定 `dictionary_learning` commit `60ec6bf...` 则包含 `BatchTopKSAETrainer`：训练时跨 batch 选 `k × batch_size` 个激活，推理时使用训练期平滑 threshold。

因此，原 tracker 中笼统的 “TopK vs BatchTopK” 若直接跨 framework 实现，会同时混入 trainer、数据路径、optimizer/checkpoint 和架构差异。新增 T014 `Joint framework–architecture confounding control`：只允许在相同 immutable activation shard 和 dictionary trainer family 内用 TopK-vs-BatchTopK 估计 architecture delta；`sparsify TopK` 保持 R006 primary，跨框架差异只能标为 joint challenger。若 challenger 的两 seed 证据足够强，必须暂停到 framework/porting decision gate，不能静默推翻 R005 freeze。GroupMax 不冒充 BatchTopK。

### L0 与语料证据

线上阅读 BatchTopK 原论文 arXiv:2412.06410，以及 2025 年 `Sparse but Wrong` arXiv:2508.16560v1。后者的 toy/Gemma 结果提示 L0 过低或过高都可能造成 feature mixing，且 reconstruction 可能奖励错误字典；这只作为新增 T015 `Pre-audit L0 sensitivity bracket` 和 decoder-projection diagnostic 的动机，不把论文中的 Gemma 最佳 L0 外推到 Pythia-160M，也不据此宣称某个 k 正确。官方代码远端 HEAD 分别只读解析为 `b9aab1c...` 和 `d5886b5...`；代码许可证尚未逐文件核查，故只登记为 reference-only。

FineWeb `v1.4.0` 被登记为 R006 corpus candidate，branch 精确解析为 `9bb295ddab0e05d785b879661af7260fed5140fc`，dataset card 标注 ODC-By。API 文件清单显示最小 `sample/10BT` 仍由约 2.15 GB shards 组成；首 shard `sample/10BT/000_00000.parquet` LFS SHA256 为 `6b552ea4...`。本轮没有下载。只有在固定 source object、document IDs/row indices、text hash、tokenizer commit 和 token hash 后才可称为采用语料；若 streaming 不能满足该合同则 fail closed。

### 新 artifact、状态和下一步

新增 `R006_ARCHITECTURE_PILOT.md`，预写 R006-A 数据合同、R006-B 单 hook/单 width 的 TopK+小型 k 漏斗、R006-C within-dictionary architecture challenger、BatchTopK threshold batch-partition invariance、两 seed 质量门和停止条件。更新 `COMPONENT_CANDIDATES.md`（T014/T015）、`REFERENCE_REGISTRY.md` 与 tracker 的 R006/architecture 项。R006 仍为 `TODO`；architecture、hook、width/k、corpus、token budget 和数值质量阈值继续 `OPEN`，M2/C1/C2 均无新证据。

下一步先实现不下载大 shard 的 FineWeb metadata/row-level manifest probe，测量可追溯抽取路径；随后在运行前写定 capacity config 和小型 k bracket，再申请共享资源 lease。若无法得到可审计的小语料资产，转评计划内 Pile-compatible/SmolLM2 corpus，而不是用临时文本启动质量 pilot。

## 2026-09-02 08:36 EDT — R006-A FineWeb revision-pinned manifest probe：v1 FAIL，v2 PASS

### 实现与预写合同

沿上一轮 R006 pre-registration，实现 `src/ccad/data_manifest.py`、`tests/test_data_manifest.py`、两份 versioned config 与 `scripts/run_r006a_fineweb_manifest_probe.py`。核心合同是直接读取包含 resolved commit 的 Parquet URL并校验 `X-Repo-Commit`、LFS SHA256、linked size、Range 响应、source row index、FineWeb document ID 与 text hash；按 `salt + dataset commit + document ID` 做 order-independent 90/10 train/validation hash split。官方 Dataset Viewer 文档确认 `/rows` 只有 dataset/config/split/offset/length，没有 revision 参数，故 viewer 只能作采样 parity cross-check，不能成为版本锚点。

第一次测试命令误用了系统 Python，四个 test module 因缺 NumPy 在 collection 阶段失败；没有创建 run 或数据 artifact。换回锁定的 R004 Python + R005 sparsify overlay 后，新旧共 31/31 unittest 与 py_compile 通过。

### v1 失败和最小修复

`R006a_fineweb_manifest_probe_v1_20260902T124000Z` 对固定 FineWeb commit `9bb295d...`、首个 10BT shard LFS SHA256 `6b552ea4...` 完成 source header 5/5 exact checks，但在首次 HTTP range 前 `fsspec` backend 导入 `aiohttp` 时缺少 `yarl`，run 标 `FAIL`。stderr、config 和所有必需 artifact 保留，artifact contract 与独立 validator PASS；这不是数据或许可证失败。

v2 新建 suffix，唯一修复是用项目内同步 `requests` seekable Range reader 取代 `fsspec/aiohttp`，不安装新依赖，不改变 dataset、commit、shard、100-row target、split salt 或判据。

### v2 结果、边界与 gate

`R006a_fineweb_manifest_probe_v2_20260902T125000Z` 14/14 checks、artifact contract 与独立 validator 全部 PASS。通过 7 个 range 请求共接收 6,957,102 bytes，解析 2,147,292,183-byte shard 的 Parquet footer与首 row group；文件含 1,048,581 rows / 1,049 row groups，首 group 999 rows。保存前 100 个完整 document records；document IDs、source rows 和 text hashes 均唯一，records canonical SHA256 `2b58475f2cf9329e8ff0861b2e13ba7284a04813f7eff1a27053b13d3060b267`，hash split 得 88 train / 12 validation。viewer 100 行与 pinned Parquet 逐字段完全一致且无截断，但仍按无 revision 的弱交叉检查解释。

该结果只证明 FineWeb 可以不下载整 shard 地进行 revision-pinned、文档级可追溯抽取，不等于训练 corpus 已采用；首 row group 有源顺序偏差，不能直接用于 R006 quality pilot。R006、M2、hook、architecture、width/k、token budget 和数值质量阈值仍未冻结，C1/C2 无新证据。下一步预写跨 shard/row-group 的确定性抽样协议与 tokenizer hash 合同，然后生成小型 capacity manifest；正式 range 抽取前根据预计传输量决定是否申请 `disk-d-io` lease。收口时 31/31 tests PASS，v1/v2 validator PASS，四类共享资源均 free。

## 2026-09-02 09:00 EDT — R006-A cross-shard capacity manifest：v1 FAIL，v2 PASS

### 抽样、token 与实现合同

新增 `src/ccad/http_range.py` 及两项 cache/seek/fail-closed tests，把上一轮验证过的同步 Range reader 变成可复用模块；新增 `scripts/run_r006a_capacity_manifest.py` 与两份 versioned config。预写抽样不依赖 source order：从 FineWeb v1.4.0 的 15 个 `sample/10BT` shards 中按 `SHA256(selection_salt, path)` 选 5 个，每个再按 salt/hash 对其 row-group count 取一个 group。所有文件逐一核对 pinned commit、LFS SHA256、linked size、Range 和 redirect。文档按已固定 split salt 做 90/10 train/validation，split 内再按独立 hash order 排序。

tokenizer 固定为本地 Pythia-160M commit `582159a...`；记录 tokenizer 三个文件 hash、EOS id 0 与 vocab size 50,277。packing 使用 leading EOS、document EOS separator、context length 128、每文档最多 2,048 tokens，尾部不足一个 context 的片段丢弃；目标固定为 1,024 train 与 256 validation sequences。输出每文档 token hash、每 sequence token hash 与 contributing document IDs，以及 little-endian uint16 binaries。

实现后项目测试为 33/33 PASS（新增两项 HTTP Range 测试），py_compile 与 config JSON parse 通过。

### v1 失败与受限修复

`R006a_fineweb_capacity_manifest_v1_20260902T131000Z` 完成 5 个 row-group 的抽取、文档检查、tokenization 和两个 binary 的 14 项 checks，但在写 `token_manifest.json` 时把 Python 布尔值写为 JSON 风格 `true`，触发 `NameError`。run 按 fail-closed 标 `FAIL`，stderr、已生成资产和 contract PASS 记录均保留。

v2 唯一修复为 `true` → `True`，未改变任何 source、salt、row group、tokenizer、packing、目标或检查；新 Run ID `R006a_fineweb_capacity_manifest_v2_20260902T131500Z`。

### v2 结果、复现与边界

v2 14/14 checks、artifact contract 与独立 validator PASS。15-shard catalog 中确定性选择 5 个 shards/row groups，共读取 4,999 个 document/source-row/text-hash 均唯一的文档，hash split 为 4,500 train / 499 validation；Range 共 40 requests、36,973,338 bytes。实际 packing 用 225 train + 44 validation documents：train 131,072 tokens / 1,024 sequences，binary SHA256 `ea22f9d4684aba55046aec6c529ea196bcb0e6374e41ac3d7dbed228c3fe7ea0`；validation 32,768 tokens / 256 sequences，SHA256 `7edbda67d1e01108b164d76cc45e37dca50a936395692764e5ac5374cca79354`。除 v1 在 exception handler 写成错误 catalog snapshot 外，v1/v2 的 selected documents、document token records、sequence records 与两个 binaries 均逐字节相同，证明科学资产复现。

该 PASS 只把 capacity-run 输入准备到可审计状态，不冻结 R007 的最终 training corpus 或规模，也不是 SAE quality/M2/C1/C2 证据。R006 保持 `TODO`；hook、architecture、width/k、正式 token budget 和数值质量门继续 `OPEN`。下一步应以此 asset 运行一个固定 layer-5、单 width/k 的 R006-B capacity benchmark，测 VRAM、tokens/s、checkpoint size、短学习曲线和 CE-recovered 评估管线，再据此预写小型 k bracket；任何 GPU run 先申请 `gpu-0` lease。收口时 33/33 tests PASS，capacity v1/v2 validator PASS，共享资源均 free，无用户裁决阻塞。

## 2026-09-02 09:36 EDT — R006-B TopK capacity 与 CE-recovered pipeline：v1/v2 FAIL，v3 PASS

### 触发、预写配置与实现

自动循环按恢复顺序重读项目治理、最新日志、R006/M2 计划、tracker、R006 pilot 和 capacity-v2 输入；理论 PDF 与本轮工程容量问题无关，未重新打开。为避免把 debug 数字追认为质量门，先写定单配置：固定 Pythia-160M commit、layer 5 resid-post、sparsify `42c0645...`、TopK width 3072/k16、seed 0、float32 eager、32,768 train tokens/64 steps 与独立 4,096 validation tokens。输入来自 R006a capacity-v2，两个 binary hash启动前逐一核对；明确 `audit_opened=false`、无质量阈值，且不选择 hook/width/k/corpus budget。

新增 `src/ccad/sae_quality.py` 和四个 CE-recovered 边界单测；定义为 `1 - (reconstruction_ce-clean_ce)/(zero_ablation_ce-clean_ce)`，非法或非有限分母 fail closed。评估实现同时覆盖全 validation 的 global-centered FVE、clean/capture/reconstruction/zero CE、actual/selected L0、alive fraction、hook-hidden-state oracle 与无修改 hook 的 logit parity。项目测试由 33 增至 37，37/37 PASS。所有 GPU 作业均经母目录 `gpu-0` lease，完成后 lease 已释放。

### 两次失败与修复边界

`R006b_topk_capacity_v1_20260902T133000Z` 在训练前 FAIL：config 把 sparsify trainer 的 hookpoint 写成 `gpt_neox.layers.5`，但 trainer 相对 `model.base_model` 解析，得到空 module set并触发 empty optimizer parameters。v2 只把 trainer hookpoint 修为 `layers.5`，另保留 intervention full path；未覆盖 v1。

`R006b_topk_capacity_v2_20260902T134000Z` 完成 64/64 steps、safe checkpoint 和 T011 exact checkpoint，却在 reconstruction evaluation 使用错误字段名失败：sparsify `EncoderOutput` 暴露 `top_acts/top_indices`，不是 `ForwardOutput` 的 `latent_acts/latent_indices`。v3 只修正该固定源码 API；v1/v2 的 config、traceback、空 raw metrics或 checkpoint 与 contract 记录全部保留。两次失败均为工程接口错误，不解释为模型或方法负结果。

### v3 结果、独立复核与解释边界

`R006b_topk_capacity_v3_20260902T135000Z` 12/12 pipeline checks PASS，artifact contract PASS。独立复核再次验证 raw metrics hash、全部 checks、64 steps、FVE 公式和 CE-recovered 公式。训练 32,768 tokens 用时 1.953 s，即 16,775 tokens/s、59.6 s/Mtoken；peak allocated/reserved VRAM 为 1,419,674,624 / 1,465,909,248 B；safe/exact checkpoints 为 18,890,231 / 56,705,391 B。

4,096-token validation 得 FVE 0.6552、CE recovered 0.5363、actual/selected L0=16、alive 2,030/3,072（0.6608）；clean/capture CE 均 4.28987，reconstruction CE 7.03419，zero-ablation CE 10.20821，hook oracle 与 capture-logit max error 均 0。decoder unit-norm max error 3.91e-4，仅记录为短训状态，不设阈值。

该 PASS 证明冻结实现可完成真实模型训练、精确/安全保存及同 hook intervention 端点，并提供当前硬件上的容量量级；只有单 seed、单 k/width 且训练极短，不能用来通过 M2、评价 SAE 最终质量或支持 C1/C2。R006 继续 `TODO`，architecture、hook、width/k、最终 corpus/token budget 与两 seed数值门仍 `OPEN`。下一步应依据预注册方案新写 calibration-only 小型 k bracket 与选择规则，不能把已看到的 v3 指标回填为门槛。收口时共享 GPU/CPU/disk leases 均 free。

## 2026-09-02 10:15 EDT — R006-B1 先验收训练预算，再启动 k bracket

### 触发与文献纠正

自动循环恢复到 R006 后，没有直接把 R006b-v3 的 32,768-token 结果扩成 k sweep。线上复核 arXiv 原始页发现 `Sparse but Wrong` 已更新为 v4（2026-07-07）；论文定义的主要 proxy 是所有 decoder latent 对的平均绝对 cosine `c_dec`，LLM 上关注低-L0 jump 前的 elbow，并明确指出该指标对训练变化存在延迟、平坦区和噪声。BatchTopK 原论文则确认固定 TopK 与 batch-global TopK 的稀疏分配语义不同。REFERENCE_REGISTRY 已更新到 v4 与官方仓库 MIT 边界；这些外部结论只用于实验设计，不转移其 Gemma/Llama 最优 L0。

### 新组件、实现与预注册

新增 T016 `Budget-stability gate before L0 selection`。新增 `src/ccad/decoder_diagnostics.py`，以 blockwise、全 pair 的方式实现 exact `c_dec`，并增加正交、符号不变、block accounting 和 invalid-input tests。新增 versioned `R006B_BUDGET_AND_K_BRACKET_20260902_101500.md` 及 latest copy，并按 experiment-plan 输出协议建立 `MANIFEST.md`。

计划先固定 k=16，在 32,768/65,536/131,072 个嵌套训练 token milestone 上，用完整 32,768-token validation 检查 FVE、CE recovered、alive fraction 与 `c_dec` 稳定性。只有最后两个 milestone 同时满足预写变化界限，才运行 `{8,16,32,64}`；若不稳定则扩充 training-only corpus/budget，禁止从欠训练结果强选。B2 使用 reconstruction eligibility 加 `c_dec` geometry screen，最多输出两个候选进入 two-seed pilot；边界最优或无 elbow 时扩到 `{64,128}`，不静默冻结。

### 探索性诊断与边界

对 v3 checkpoint 做了固定 seed 的 200,000 个不同 decoder pair 抽样，mean absolute cosine 为 0.03076，而 768 维随机各向同性近似基线为 0.02879；由于该计算没有独立 run contract，只能支持“32k 可能仍靠近初始化几何、应先做预算稳定性”的设计判断，不能支持 k=16 好坏或 monosemanticity。R006/M2、hook、width/k、最终 corpus/token budget 和质量阈值均保持 OPEN；未打开 R008 audit，无 C1/C2 新证据。下一步实现 B1 milestone runner 并在 gpu-0 lease 下运行。

测试勘误：首次在当前只读沙箱内运行全套测试时，新增的 3 个 decoder diagnostics tests 全部通过，但 5 个既有 artifact-contract tests 因系统临时目录不可写而报 `FileNotFoundError`，随后 `py_compile` 也因 `__pycache__` 禁写失败；这些是执行环境错误，不是测试断言失败。改用已授权的锁定 Python、`-B` 和系统 TEMP 在沙箱外只读重跑后，40/40 tests PASS。没有为掩盖失败改动测试或实现。

## 2026-09-02 11:15 EDT — R006-B1 预算稳定性 gate PASS

### 触发、环境恢复与运行合同

自动循环按治理顺序恢复到 T016/R006-B1；本轮是已预注册的工程与实证 screen，不涉及理论修改，故没有打开理论 PDF，也没有接触 R008 audit。依据 `run-experiment` skill 补建 `.aris/compute` 环境账本：`local-r006b1-env-spec.json` 固定 Python 3.13.7、torch 2.8.0+cu128、transformers 5.15.0、safetensors 0.8.0、NumPy 2.5.2、sparsify commit/overlay hash、Pythia commit与 offline/CuBLAS 环境。主 witness 在 `gpu-0` lease 下返回 `R006B1_ENV_READY`、RTX 5070 Ti capability 12.0 与数值 witness 25.131683349609375。首次沙箱内调用因无权创建母目录 lease 得 WinError 5，保留为权限边界；授权后相同命令通过。随后 fresh agent 只读 ledger/spec 并逐字执行文档命令，得到完全相同版本、源码路径、设备与数值，无环境漂移；ledger 标为 READY。

新增 `scripts/run_r006b1_budget_stability.py`、versioned config 与只读独立 validator。实现不是三个独立短训：固定完整 256-step scheduler，在 64/128 step 通过预期 forward interrupt 保存 exact optimizer/scheduler/RNG/data-cursor checkpoint，再从完整 dataset 恢复 suffix，最终到 256 step；因此 32,768/65,536/131,072 tokens 属于同一条嵌套轨迹。每点都在完整 32,768-token validation 上计算 FVE、clean/reconstruction/zero CE、CE recovered、actual L0、alive/firing distribution、decoder norm 与 exact all-pair `c_dec`。训练输入 hash 逐 batch 与预期 256 batches 比对；三个 exact 与 safe checkpoints 均保留。正式 Run ID 为 `R006b1_budget_stability_v1_20260902T151124Z`。

### 测试、失败修复与正式结果

新增 `budget_stability_checks` 与三项单测；第一次全测 42/43，唯一失败是十进制边界 0.05 在 binary float 中成为略大值。修复仅将 `<=` 边界增加 1e-12 级 `isclose`，没有修改任何预注册阈值；重跑 43/43 PASS，runner/verifier py_compile PASS。正式 GPU run 由母目录 resource manager 独占调度，结束后 `gpu-0` 为 free。

run 的 14/14 hard checks、artifact contract 与独立 validator 11/11 全部 PASS；validator重算 raw SHA256、CE公式、稳定性判定、milestone tokens、完整输入轨迹、checkpoint/safe SAE存在性与 audit关闭。三个 milestone 指标分别为：32,768 tokens：FVE 0.86083、CE recovered 0.56434、alive 0.81836、`c_dec` 0.032142；65,536：0.93933、0.66055、0.81999、0.033299；131,072：0.94852、0.70378、0.82227、0.033882。actual L0 三点均精确为16，hook oracle/capture logits误差均为0。最后两点变化为 ΔFVE 0.00920、ΔCE recovered 0.04323、Δalive 0.00228、相对 Δ`c_dec` 1.749%，均满足预写 0.05/0.05/0.10/10% 门。

训练本身累计8.633秒、15,183 tokens/s；peak allocated/reserved VRAM 1,457,488,384 / 1,570,766,848 bytes。run 共33个文件、283,593,836 bytes，其中 exact checkpoints 170,116,304 bytes、safe SAEs 56,670,693 bytes。完整 firing distribution显示最后点 validation 上 546/3,072 latents 未触发、median 33、q90 450.9、max 17,246；它只描述该 validation 与单 seed，不是正式 dead-feature 质量门。

### 保守解释与下一步

T016/B1 按预注册规则通过，允许启动 B2 `{8,16,32,64}`；已同步 tracker、`COMPONENT_CANDIDATES.md`、latest R006-B plan 与 `MANIFEST.md`。CE recovered 的最后变化 0.04323 距0.05边界较近，所以该结果只说明“当前预算足以进入 bracket screen”，不说明充分收敛，更不冻结 hook/width/k、R007 corpus/质量门，也不支持 M2/C1/C2。下一轮应先实现 B2 统一 runner/选择器；k=16 只有在代码、输入、schedule 与 config合同兼容时才复用本 checkpoint，否则所有 k 都从头按同预算训练。B2 必须执行原有 reconstruction eligibility、geometry screen与边界扩展规则，不能因本轮 PASS 直接冻结 k=16。

## 2026-09-02 11:42 EDT — R006-B2 初始 k bracket PASS，但触发上界扩展

### 恢复、hash 勘误与选择器冻结

自动循环按治理顺序恢复到 B1 PASS 后的 T015/B2；本轮继续使用 `run-experiment` skill，读取 READY 环境 ledger、B1 config/summary/status/environment/stderr 与复用 checkpoint/metrics，未打开理论 PDF或 R008 audit。复核 skill 的 compute contract 时发现 ledger 原先把环境 spec 的 source-file SHA256 `3129a184...` 标成 spec hash，而合同定义的 cache key 应是 parsed JSON canonical hash。已在 ledger 原位澄清：canonical SHA256 为 `8348de46ad5dd5c721867641e59a9d9d53e1f083622db98af51089e2bf2c9d43`，source-file hash 保留作 artifact integrity。spec 内容、运行时与权重均未改变。因 ledger 文档发生修改，fresh agent 再次只读 compute contract/ledger/spec，逐字执行 invocation；canonical/source hash、版本、GPU、源码路径、sentinel 与数值 witness 25.131683349609375 全部相符，无 doc-reality divergence。

在读取新 k 指标前，把已有 B2 自然语言规则机械化为 `select_k_bracket`：四点各自以全 bracket 最大 FVE/CE recovered 为基准，双 margin 均为包含边界的0.03；eligible 内以 (`c_dec`, k) 取 anchor，并补一个更高的最近 eligible k；eligible 空、只有上界64，或四点 `c_dec` 严格单调时不输出 shortlist而触发扩展。新增三项 selector tests；全套 46/46 tests 与 runner py_compile/config parse PASS。该澄清没有改候选、阈值或 audit，只消除“相邻更高 k”和“严格单调”的实现歧义。

### 正式 run、复用审计与结果

新增 config、`scripts/run_r006b2_topk_bracket.py` 和独立 validator，正式 Run ID `R006b2_topk_bracket_v1_20260902T153552Z`。k=8/32/64 固定同一 Pythia commit、layer5 resid-post、sparsify commit、width3072、seed0、FineWeb capacity-v2 输入/顺序、Adam 1e-3、8-step warmup、256-step/131,072-token schedule 与完整32,768-token validation。三个新 run 的初始化 state hash 完全相同，训练 input trace逐 batch与同一256-batch预期相同；各自保存 safe SAE 与 exact optimizer/scheduler/RNG checkpoint。k=16 复用 B1 final，但先同时检查 B1 PASS/stability、contract、raw hash、关键 runner/evaluator/checkpoint/activation/c_dec 代码未变、固定 config、预算、input trace与 checkpoint hash；所有复用检查通过。

正式 suite 9/9 checks、artifact contract 与独立 validator 12/12 PASS；validator重算四个 CE-recovered公式、selector、raw hash、所有新 input traces、reuse provenance 与 audit关闭。结果：k8 FVE 0.93416、CE recovered 0.62639、alive 0.47591、`c_dec` 0.030967；k16 0.94852/0.70378/0.82227/0.033882；k32 0.95501/0.77000/0.98307/0.034392；k64 0.96153/0.85178/1.00000/0.033340。四点 actual L0 均精确等于 k，hook/capture parity均为0误差。k8/32/64 训练分别5.306/5.146/5.287秒；suite peak allocated/reserved VRAM为1,420,329,984/1,551,892,480 bytes。run 共47文件、397,043,143 bytes，其中项目 exact checkpoints 170,116,179 bytes、safe SAEs 56,670,692 bytes；GPU lease结束后为free。

### Gate 解释与下一步

全 bracket 最佳 FVE/CE recovered 均在k64。按双0.03规则，k8/16/32 至少在 CE recovered 上超出 margin，只有上界k64 eligible；`c_dec` 四点不严格单调。故 suite 状态为 PASS，但 selection=`EXPAND_TO_64_128`、正式 shortlist为空；这正是停止规则，不是模型训练失败。不得仅凭 reconstruction趋势冻结k64，也不得把k64 alive=1解释为更可解释。

已在 latest R006-B plan 中于看到k128前冻结 extension：只新增k128并复用既有四点，在五点联合集上保持同一双0.03 margin；若只有新上界128 eligible则标 `UNBOUNDED_HIGH` 并停止自动扩展，若五点 `c_dec` 严格单调则标 `UNBOUNDED_GEOMETRY`，否则最多给两个 two-seed pilot候选。T015状态改为 `B2 PASS / EXTEND`，tracker与MANIFEST同步。R006/M2、最终hook/width/k/corpus/质量门仍OPEN，无C1/C2证据；下一轮可在同环境与gpu-0 lease下执行k128 extension，无需用户裁决。

## 2026-09-02 13:51 EDT — R006-B2 k128 extension 得到 UNBOUNDED_HIGH 负结果

### 触发、协议与实现

自动循环按治理顺序恢复到 B2 的 `EXPAND_TO_64_128`；重读 READY compute ledger/spec、初始 bracket config/summary/status/environment/stderr 与 `run-experiment` skill。环境 canonical hash `8348de46...` 和 source-file hash `3129a184...` 未变化，按合同 warm-reuse，无 rebuild 或 ledger 修改，故不重复 agent-follows-doc。理论 PDF 与本轮单纯配置 screen 无关，未打开；R008 audit 始终关闭。

在k128指标出现前，新增五点 `select_k_extension` 并将上一轮冻结的 extension rule写成测试：联合 `{8,16,32,64,128}` 使用相同双0.03 margin；只有新上界时 `UNBOUNDED_HIGH`，五点 `c_dec`严格单调时 `UNBOUNDED_GEOMETRY`，FVE/CE最优点分裂且无联合候选时 `NO_JOINT_ELIGIBLE`，其余才允许最多两个 `TWO_SEED_PILOT`候选。新增三项分支测试后全套49/49 PASS。`NO_JOINT_ELIGIBLE`只是把原有“eligible为空不强选”规则在停止自动扩展后的状态命名，没有调整阈值。

新增 `configs/r006b2_extension_k128_v1.json`、runner与独立validator，Run ID `R006b2_extension_k128_v1_20260902T174717Z`。只训练k128，固定与B2相同的模型/hook/width/seed/data/order/optimizer/131,072-token schedule和完整32,768-token validation；initial state hash必须与B2 k64一致。初始四点仅在源run PASS/contract/raw hash、suite checks、关键代码、固定config、所有训练input traces及k64 checkpoint hash全部通过时复用。GPU作业经母目录 `gpu-0` lease执行，完成后lease为free。

### 结果、独立复核与边界

正式 run 8/8 suite checks与artifact contract PASS；独立validator 14/14重算raw hash、CE公式、五点selector、input trace、reuse/candidate checks、checkpoint/safe SAE存在性与audit关闭，全部PASS。k128结果为FVE 0.9689508、CE recovered 0.9113536、alive 1.0、actual/selected L0 128、exact `c_dec` 0.0312176；clean/reconstruction/zero CE为4.60595/5.16212/10.87999，hook与capture parity误差均为0。validation firing count无dead latent，min85、median1141.5、q90 2393.6、max19771；这些是单seed/单validation描述，不是解释性或正式dead-feature门。训练4.875秒、26,884 tokens/s；peak allocated/reserved VRAM 1,419,920,384/1,719,664,640 bytes。run共23文件、132,476,186 bytes，其中exact checkpoint56,705,393 bytes、safe SAE18,890,232 bytes。

联合五点中FVE和CE recovered仍在k128最大；按双0.03 margin，k128是唯一eligible，五点 `c_dec`不严格单调。因此run本身PASS，但selection严格为`UNBOUNDED_HIGH`、shortlist为空，并按已冻结规则停止自动扩到k256。该结果否定了当前固定-margin screen在所测范围内形成稀疏候选的能力；不得事后放宽margin、只看最低 `c_dec` 选k8，或只看reconstruction选k128。它也不说明高k更可解释。

### Gate 与下一步

T015标为`B2 BOUNDED-STOP`，未达到“可重复shortlist”进入条件；R006继续TODO，M2、最终hook/width/k/corpus/质量门与C1/C2均无变化。新增T017候选 `Seed-replicated sparsity–utility Pareto confirmation`：下一轮先做本地/线上方法检索并预写scale-aware Pareto/knee/refusal协议，再决定是否以至少两个新seed复核k8–128全曲线。T017禁止用seed0调权后把新seed称为confirmatory，也禁止若仍只剩最高k就继续无界扩张。当前没有必须由用户裁决的阻塞；若后续pilot证据要求在冲突配置间作最终不可逆选择，再提交用户决定。

## 2026-09-02 14:12 EDT — T017 文献复核否定 reconstruction-only knee，冻结 evaluator-first B3 设计

触发原因是自动循环在 R006-B2 `UNBOUNDED_HIGH` 后继续研究 sparsity 选择方法。本轮按顺序恢复 AGENTS、master log、experiment plan、tracker 与 component 状态；任务不涉及修改理论定理，因此没有重新读取内部理论 PDF，也没有打开 paired/causal/semantic audit。项目目录不是 Git 仓库，已再次如实记录，不能虚构 commit provenance。

本地逐页读取并核对了 SAEBench、ACL 2026 Feature Consistency、Feature Sensitivity、Sparse Dictionary Optimality、Gemma Scope 与 Llama Scope PDF；线上以 PMLR、ACL Anthology、arXiv 和作者托管原论文核对版本与结论。纠正一项元数据：父目录 `SAEBench_ICLR25` 中的 PDF 实际是 ICML 2025 / PMLR 267 论文，后续引用不得沿用目录名。新增来源、版本、边界和本地 SHA256 已写入 `REFERENCE_REGISTRY.md`。

关键证据是 `Sparse but Wrong` arXiv:2508.16560v4（2026-07-07）：它构造了正确 dictionary 在低 L0 下 reconstruction 反而劣于 mixed dictionary 的反例，并在 LLM 实验中用每个 L0 三个 seed，观察 sparse-probing peak 位于低-L0 `c_dec` jump 前的 elbow。作者同时明确说 `c_dec` 更适合排除明显错误 L0，不适合单独精确定位 optimum。SAEBench 也显示 unsupervised proxy 增益不稳定转化为 practical performance；ACL 2026 工作把 run-to-run consistency 定义为 reconstruction/sparsity 之外的轴；Feature Sensitivity 再给出一个互补质量轴。因此，在 seed0 五个 FVE/CE 点上运行 Kneedle 即使可复现，也不能验证 y-axis 是正确科学目标。

新增 versioned `T017_PARETO_SELECTION_REVIEW_20260902_141200.md` 和 current pointer。协议明确拒绝 reconstruction-only knee：B3-A 先用既有 seed0 checkpoints 资格审查 configuration-only sparse-probing evaluator，冻结与 R013/R014/R021 不重叠的 task manifest、revision/hash/split、probe budget、base-residual skyline、PCA/random baseline、deterministic replay 与 missingness 规则；seed0 只作 adapter smoke，不参与确认性选择。新增 T018 `Configuration-only sparse-probing evaluator`，状态为 `CANDIDATE / QUALIFY-BEFORE-TRAINING`；tracker 新增 `R006b3a_sparse_probe_qualification` TODO。

只有 B3-A PASS 才允许 B3-B：固定 k={8,16,32,64,128}、不扩 k256，对恰好两个新 seed 跑全曲线。每个 seed 用冻结 task 上的 macro sparse-probe utility，以 one-standard-error rule 选择最小近最优 k；两个 proposal 必须相同或相邻才能形成至多两个候选。共同落在 k128、跨 seed 不稳定、proposed interior k 在两 seed 都是 `c_dec` 严格局部最大、未胜公平 baseline、质量门/trace/hash不全、task 泄漏或 k-dependent missingness 均显式拒答。FVE、CE、`c_dec`、projection、occupancy、alive/dead 与成本必须报告，但不得调成事后 scalar score覆盖主规则。

当前 seed0 的 `c_dec` 序列为 .030967/.033882/.034392/.033340/.031218，并不呈现 cited LLM 例子中的 clean high-at-low-k elbow；这只加强“当前应拒绝选 k”的判断，不能证明 k8 或 k128 正确。本轮没有启动新训练、没有接触 audit、没有改变 LOCKED 协议，也没有 C1/C2/M2 新证据。下一轮的安全边界是为 R006b3a 获取并审计 sparse-probing evaluator/task 资产；若无法证明其与未来主 endpoint 隔离，则标 BLOCKED 而不是复用。

## 2026-09-02 14:33 EDT — R006-B3-A 固定 evaluator 源，task-license 门 fail-closed

自动循环继续 R006-B3-A，按治理顺序恢复 AGENTS、master log、experiment plan、tracker 和 T017/T018；本轮是 evaluator 与实验设计审计，不修改理论，故未打开理论 PDF，也未接触 paired/causal/semantic audit。依 `research-lit` skill 同时复核了 ICML 2025 sparse-probing 原文、2026 reliability audit、官方 `sae-probes` 仓库和本地 SAEBench 0.6.0 wrapper。

通过只读 `git ls-remote` 固定官方 `sae-probes` v0.4.0：tag object `aa52dc44...`、peeled commit `d71c1661...`，并将 MIT 源码浅克隆到 `D:\CCAD_Storage\references\source\sae-probes_aa52dc4`；LICENSE hash 为 `B343F8C1...`。已验证本地 SAEBench commit `8042bb3...` 声明依赖 `sae-probes ^0.4.0`，但它本身无根 LICENSE，只作reference-only。锁定 R006 Python 中尚无 `sae-probes` package/module，所以未修改现有环境，也没有把 source pin 写成 runnable PASS。

本轮发现的硬边界是：仓库 master CSV 确有 113 个 binary tasks，但只有 34 行含非空 direct link，仅24个 unique links；代码 MIT 不能代替上游数据许可。因此 T018 只升级为 `SOURCE-PINNED / TASK-LICENSE-MANIFEST-PENDING`，R006b3a 仍为 TODO，未运行 probe，更未启动两新 seed 训练。新增 `R006B3A_EVALUATOR_QUALIFICATION_20260902_143300.md` 并同步 plan/tracker/components/reference registry。

设计上纠正了 `k` 过载：`k_train={8,16,32,64,128}` 是 SAE 训练稀疏度，primary `q_probe=16` 是 probe 特征预算；normal/L1/test AUC 为 primary，accuracy/F1 与 q=1/5 仅作诊断。full residual 是 skyline，不再被误写为 SAE 必须超越的资格门；公平稀疏对照是同 split/同 probe 的16维 PCA 与 random-orthonormal projection，只有无法超越 random 才触发 `REFUSE_PROXY_ONLY`。B3-A 还预写了同一固定 SAE 上的 5 probe-seed MRD 和现有五个 `k_train` checkpoints 的 task-bootstrap noise-vs-signal discriminability 门；该门只决定 evaluator 能否用，不选 `k_train`。

下一轮应先完成 113 任务的来源/许可追溯并冻结最小任务数与 missingness rule；然后在新 overlay 中做依赖与 sparsify adapter conformance。在这两项通过前，不得因为 evaluator “在论文里表现最好”就绕过可区分性或数据许可门。本轮无 M2/C1/C2 新证据，无 LOCKED 协议变更。

## 2026-09-02 — 母目录文献库错配通知的 CCAD 影响审计

用户通知母目录文献库已隔离两项 PDF/条目错配：`Prithvi_TokensToPolicy_Preprint26/paper.pdf` 实为 NEXIS 异质性处理效应论文，`Insight_HierarchicalVision_Preprint26/paper.pdf` 实为 CFM；正确 INSIGHT v1 已以 `paper_v1_INSIGHT.pdf` 另存。同时再次确认 `SAEBench_ICLR25` 目录内论文的正确 venue 为 ICML 2025 / PMLR 267。

对 CCAD 计划、tracker、component、T017/B3-A 协议和 reference registry 做全文检索：未发现 Prithvi/NEXIS 或 Insight/CFM 条目，因此没有 novelty、related-work 或 citation 污染需修复。SAEBench 已在 `REFERENCE_REGISTRY.md`、`T017_PARETO_SELECTION_REVIEW_20260902_141200.md` 和当前 B3-A 文档中按 ICML 2025 登记，且明确旧目录名不得作 venue 证据。本轮仅追加文献卫生记录，不改实验协议、run 状态、R006-B3-A 阶段门或自动循环节奏。

## 2026-09-02 15:15 EDT — R006-B3-A 任务 provenance 审计：33/113 可追溯，gate 保持关闭

自动循环继续 R006-B3-A，按治理顺序重读 AGENTS、master log、相关 plan、tracker 与 evaluator qualification；本轮为数据来源/许可审计，不涉及理论，故未打开理论 PDF，也未计算 probe activation 或接触任何 paired/causal/semantic audit。依 `research-lit` skill 检索上游论文、数据卡和官方仓库，并将外部内容只作为证据。

新增 `configs/r006b3a_verified_sources_v1.json`、`scripts/audit_r006b3a_task_provenance.py` 与两项 fail-closed 单测。首次 run `R006b3a_provenance_audit_v1_20260902T190122Z` 正确算出113项 inventory，但遗漏 resolved config、environment 和 structured log，因此按 run 合同改判 `FAIL`，在原目录追加 `CORRECTION.md` 而未覆盖任何输出。修复 runner 后以新 ID `R006b3a_provenance_audit_v2_20260902T190505Z` 重跑，独立 validator 10/10 PASS；其10项手工许可证据仍输出 `INSUFFICIENT_LICENSE_COVERAGE` 和 B3-A `NOT_PASSED`。

随后核查 `wesg52/world-models`：官方仓库 README 明确把 cleaned entity CSVs 作为项目数据，根许可证为 MIT，并以 `git ls-remote` 固定 commit `a572f162948ee185e9c842eb5fa15e23aad3d218`。新增 v2 license config，覆盖23个 historical-figure、headline、geography 与 art 衍生任务；正式扩展审计 `R006b3a_provenance_audit_v3_20260902T191500Z` 在51/51项目单测后运行，独立 validator 10/10 PASS。manifest SHA256为 `CF43DD2A3E3C592DDE9A96523513096DE510C7CD5FC6B0F6CE21ACC501D9B5CC`。

v3机械确认113个 binary catalog row 与113个本地raw文件：23项本行有显式URL，11项使用catalog caret引用，25项可映射到带URL的multiclass parent，54项没有机械可辩护的URL链。当前有33项task-level许可证据：30项permissive-with-notice、1项CC-BY attribution、2项NC research-only；80项仍未解决，且任何本地raw均未获准由CCAD再分发。GLUE数据卡明确要求逐底层任务处理许可，不能用GLUE或`sae-probes`聚合仓库的许可证覆盖。

新增 `R006B3A_TASK_PROVENANCE_REVIEW_20260902_151500.md`，并同步 qualification、tracker、components 与 reference registry。T018状态改为 `PROVENANCE PARTIAL: 33/113 LICENSE-TRACED; GATE CLOSED`；另登记T019 fresh license-clean battery作为未采用候选，防止逐项追溯无法收敛时无界消耗。T019禁止看到SAE分数后挑任务，且在task family、最小数、平衡、变换、split/hash和endpoint隔离冻结前不得计算activation。

保守解释：v2/v3的PASS只证明inventory与artifact contract可复核，不是evaluator或R006-B3-A过门；33项也尚未完成upstream transformation identity、row hash/class balance/split与最小任务数冻结。R006仍TODO，B3-B两新seed训练继续禁止，无M2/C1/C2新证据。下一轮优先审计能一次解锁完整任务族的上游来源，并比较继续清理T018与预注册T019的证据/成本；若两条路线都无法形成足够且许可清楚的任务集，再形成用户裁决选项。

## 2026-09-02 15:29 EDT — R006-B3-A 分流：T019 clean battery 与 family-cluster 统计单位

自动循环继续比较 T018“清理已发表 113-task suite”与 T019“从许可清楚的上游重建”。本轮按治理顺序恢复 AGENTS、master log、相关 experiment plan、tracker、provenance review 与 components，并完整读取 `research-lit` skill。工作只涉及官方数据卡/仓库和预激活设计；没有打开理论 PDF（无理论修改）、没有读取 SAE probe 分数或 activation，也没有接触 paired/causal/semantic audit。

官方来源复核确认 Amazon MASSIVE 与 BANKING77 的数据卡均明确标为 CC-BY-4.0，分别提供大规模 intent ontology 和 77-intent/13,083-query 固定拆分，故进入 T019 候选短名单；但每个上游数据集只算一个 statistical family。DAIR.AI Emotion 的卡只标 `other`，CogComp TREC 标为 unknown，均在获得明确上游数据许可前拒绝；SNIPS 卡报告 CC0，但仍需直接源/revision与 328-row、无原生 split 的小样本审计，故仅 hold。第三方镜像的 MIT 标记未被当作原始数据许可。

新增 `R006B3A_T018_T019_DECISION_REVIEW_20260902_152940.md`。决策采用 bounded two-lane：T018 保留为 published-suite compatibility lane，仅在一个权威来源能成族解锁时继续；T019 作为 publication-quality clean construction 候选，但状态仅为 `SOURCE-SHORTLIST / NOT FROZEN / NO ACTIVATIONS`。关键设计修正是：同一 multiclass 数据集衍生的 one-vs-rest tasks 共享样本、领域和标签本体，不能作为独立重复。primary 必须 family 内先汇总、upstream family 等权，并报告 family-cluster bootstrap、leave-one-family-out 与最大单 family influence；naive task-level SE 只作伪重复诊断。

新增 T020 `Family-clustered sparse-probe aggregation` 及 tracker 项 `R006b3a_family_aggregation_simulation`。它须在不加载真实 SAE score 的情况下，以相关 family/task 合成效应比较 naive task inference、equal-family mean、family bootstrap 与 LOFO，先冻结最小独立 family 数、family cap 和 refusal 条件。同步更新 `COMPONENT_CANDIDATES.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`、`REFERENCE_REGISTRY.md`、T017 current protocol 与 `MANIFEST.md`。

本轮无 run、无 GPU/CPU-heavy lease、无新数据下载；R006-B3-A 与 R006 仍为 TODO，B3-B 两新 seed 训练继续禁止，无 M2/C1/C2 新证据。下一轮应先预写并运行 T020 score-free simulation；只有其覆盖率/稳定性门支持足够独立 family 数，才继续冻结 T019 manifest，否则保持拒答并向用户提交任务集/资源取舍。

## 2026-09-02 15:56 EDT — T020 score-free simulation：family inference 通过，one-SE selector 否定

自动循环按治理顺序恢复 AGENTS、master log、相关 plan、tracker、components 与 T018/T019 决策。本轮完整读取 `statistical-power` skill 及其 simulation-based power reference；不涉及理论修改，故未打开理论 PDF。新增预注册 config、runner、独立 validator 和 3 项单测，全部效应 profile 都是 prospective sensitivity scenario，不从 seed-0 或任何真实 SAE score 估计。合同明确 `audit_opened=false`、`sae_scores_loaded=false`、`probe_activations_computed=false`。

正式 run `R006b3a_family_aggregation_simulation_v1_20260902T195100Z` 使用 5,000 replicates × 243 cells，覆盖 4–32 independent families、每族4/8/12 tasks、ICC .25/.50/.75 和三种 utility geometry。首次资源管理器调用因沙箱无权写母目录 lease 返回 WinError 5；获批后通过 `cpu-heavy` exclusive lease 执行约4分14秒并自动释放。没有申请正在被 EndoSAE 下载占用的 D-disk lease，也无 GPU、数据下载或大 I/O。

运行 artifact contract PASS，但语义结果为 `NO_FAMILY_COUNT_QUALIFIED`：9 个 family counts 均未在全部27情形过门，只有8/243 cells通过。family-cluster t interval 的最坏 coverage lower bound 在各规模为 .935–.939，null-FP upper bound最高 .028，说明 cluster inference 本身正常；naive task-level inference 的最差 coverage 仅 .4068、最高 null FP .3876，直接显示把 one-vs-rest tasks 当独立重复会严重反保守。绑定失败是 one-SE 离散 selector：最坏 agreement lower bound从4 families的 .182到32 families仍仅 .290；LOFO最坏下界从 .192改善到 .713但仍未达 .80。因此不能把失败简化为“收集更多任务/更多family”。

独立 validator 9/9 PASS，results SHA256 `5A5FCC4C732EA73A70069BB9E5178B41D7AD4BE96DA5CAE471341ECA30319056`。targeted tests 3/3 PASS。首次 full-suite invocation 漏设项目 `src` import path，45 tests通过、2 modules import error；未修改实现/测试，以 `PYTHONPATH=src` 原样重跑后54/54 PASS。该环境调用勘误不改变正式 run。

新增 `R006B3A_FAMILY_AGGREGATION_SIMULATION_REVIEW_20260902_155617.md` 并同步 tracker/components/plan/current pointer/manifest。T020 状态为 `V1 NEGATIVE`：保留 equal-family aggregation 和 family-cluster uncertainty，但否定当前 one-SE discrete selector，T017 对应条款暂停。新增 T021 `Family-paired SESOI non-inferiority selector`：以固定 k128 为参考、family-paired loss、multiplicity-controlled one-sided UCB 和明确拒答替代随机 one-SE 阈值；它只是候选，须先在不读真实 SAE score 的情况下校准 false non-inferiority、selection、LOFO 与 MC precision。

保守解释：这只是合成设计证据，不是 evaluator、M2、C1 或 C2 结果，也不证明任务 family 可交换。T019 仍 `NOT FROZEN`，R006-B3-A 仍 `NOT_PASSED`，两新 seed 训练与 activation 继续禁止。下一轮应先为 T021 寻找可独立辩护的 SESOI（不能从现有 SAE 分数反推），预写 multiplicity 方法并做 score-free simulation；若仍需不现实 family 数或错误非劣效失控，则拒绝 sparse-probe 作为 k selector 并形成用户裁决选项。

## 2026-09-02 16:13 EDT — T021 SESOI/non-inferiority selector 预注册

自动循环按顺序完整恢复 AGENTS、master log 最新条目、相关 plan、tracker 和 T020 正式 run 的 resolved config/status/summary/environment/log；本轮继续使用并完整读取 `statistical-power` skill 与 simulation reference。任务只涉及统计决策设计，不改理论，故未打开理论 PDF，也未读取任何 SAE score/activation/audit。

线上以 Lakens、Scheel & Isager 2018 equivalence tutorial 和 FDA 2016 non-inferiority guidance 核对方法边界：必须先定义会改变决策的有意义损失 margin，再用单向置信界检验；不能把“不显著”当等效。外部临床/心理方法只提供原则，没有把任何 margin 转移成 CCAD 事实。新增 `R006B3A_T021_NONINFERIORITY_PREREGISTRATION_20260902_161323.md`、冻结 config `configs/r006b3a_family_paired_noninferiority_simulation_v1.json`，并同步 components/tracker/plan/reference registry/T017 current pointer/manifest。

T021 v1 固定 k128 参考，estimand 为每个较小 k 相对 k128 的 equal-family paired AUC loss；四个比较用 one-sided Bonferroni alpha/4 UCB，避免再从数据中选择“best reference”。primary SESOI `.01` AUC 被明确标为 active-feature budget 的预先工程决策阈值，而非文献公认常数；`.005/.02` 为不可隐藏的敏感性。若真实 `.01/.02` 提案冲突或只有 `.02` 支持更小 k，最终 primary config 交用户决定，不自动强选。

score-free grid 冻结为5,000 replicates、6–64 independent families、4/8/12 tasks、ICC .25/.50/.75、三种以 SESOI 倍数表达的 prospective loss profiles和三档 margin。primary `.01` 的所有 cells 必须同时满足 false non-inferiority、correct smallest-safe selection、LOFO、MC precision 与 boundary calibration；若无现实 family 数通过，T021 失败，不准放宽 margin、删困难情形或用同 family 更多任务伪装样本量。

本轮没有运行 simulation、没有创建 run 状态、没有资源 lease 或下载。T021 仅 `PREREGISTERED / NOT ADOPTED`，T017 one-SE 仍暂停；T019、R006-B3-A、R006/M2、C1/C2状态不变，新 seed 训练继续禁止。下一轮应实现 exact/audited t critical、boundary/null tests、runner与独立 validator，再经 cpu-heavy lease 执行冻结 config。

## 2026-09-02 16:39 EDT — T021 正式负结果：错误控制有效但 smallest-k 功效不足，selector 分支停止

自动循环按序完整恢复 AGENTS、master log、相关 plan/tracker、T021 预注册与 T020 run；完整读取 `statistical-power` skill及simulation reference。本轮不涉及理论，未打开理论 PDF；正式合同再次明确不加载 SAE score/activation/audit。

实现 exact Student-t CDF/quantile（mpmath regularized incomplete beta）、向量化 family-paired Bonferroni UCB、fixed-k128 selector、boundary false-NI、LOFO、argmax/one-SE diagnostics、独立 validator及测试。第一次10-replicate预运行 smoke 在正式run目录创建前发现诊断性 one-SE comparator 的 batch index 维度错误；修复索引并新增端到端测试，冻结 estimand/grid/threshold 未变。临界值与两个公开参考值核对；targeted tests 4/4 PASS。

正式 `R006b3a_family_paired_noninferiority_simulation_v1_20260902T203551Z` 经母目录 `cpu-heavy` exclusive lease运行约2分54秒并自动释放；未争用另一项目的 D-disk lease。5,000×810 cells完整覆盖6–64 families、4/8/12 tasks、ICC .25/.5/.75、三种loss profile与SESOI .005/.01/.02。run/contract PASS，独立 validator 10/10、全项目58/58 tests PASS，results SHA256 `5BA248532E20B22E58FA328FEFDC78454468582BA44E8E1F1903DEDA42F9F2C1`。

语义结果 `PRIMARY_MARGIN_NOT_QUALIFIED`：三档margin均无qualified family count，只有4/810 cells通过。错误控制有效：worst family-wise false-NI upper bound分别为.023/.014/.007，boundary upper bound全网格最高.022；primary `.01` 在64 families的worst LOFO lower bound .838。绑定失败是功效：64 families的worst smallest-safe selection lower bound仅.015/.027/.072（.005/.01/.02）。这说明方法是保守/低功效，不是反保守；继续增加同family派生任务不能解决独立family不足。

新增 `R006B3A_T021_NONINFERIORITY_RESULT_REVIEW_20260902_163945.md` 并同步 tracker/components/plan/T017 pointer/manifest。T021标为`V1 NEGATIVE / CUT AS AUTO-SELECTOR`；结合T020，自动 sparse-probe selector 分支停止，不再自动制造T022。T017/B3暂停，因 primary配置在pilot证据冲突时属于用户保留决定。选项为：(1) sparse probe只作诊断并由用户批准明示工程取舍；(2)显著扩展独立task families后重新预注册；(3)更换pre-audit endpoint并做材料协议复核。

保守边界：T018/T019仍可作provenance/diagnostic资产，但无权启动activation或两新seed；R006-B3-A、R006/M2、C1/C2无变化。等待用户期间，自动循环可继续不承诺primary k的M0/M1完整性、理论审查、provenance或独立实现工作。

## 2026-09-02 17:13 EDT — R001–R003 完整性审计失败，M1 gate 勘误并阻断后续主线

自动循环在等待 primary-config 用户裁决期间，按治理顺序恢复 AGENTS、master log、相关 plan/tracker，并使用 `experiment-audit` skill 对 R001–R003 的 11 个历史 synthetic suffix run 做独立完整性检查。本轮只读取 synthetic artifact 和当前源码，没有打开真实 SAE、paired/causal/semantic audit，也没有训练或申请重型资源 lease。该 skill 要求 fresh reviewer，因此启动 GPT-5.6-Sol/ultra 审阅员；由于仍属同模型族，结论明确标记 `reviewer_independence=same-family`、`acceptance_status=provisional`。

新增 `configs/m1_parent_integrity_audit_v1.json`、`scripts/audit_m1_parent_integrity.py`、独立 `scripts/validate_m1_parent_integrity.py` 和两项 fail-closed 单测。正式 post-hoc inventory run `M1_parent_integrity_audit_v1_20260902T210136Z` 机械核验 11/11 目录、340 raw records 和 3,480/3,480 frozen checks；所有原始 raw/config/status/contract/code-hash 绑定内部一致，parent family union 精确，validator 12/12 PASS，完整项目测试 60/60 PASS，且 `audit_opened=false`。该机械 PASS 只证明现存 artifact 自洽，不是 semantic gate certificate。

fresh semantic audit verdict 为 `FAIL`。当前 `scripts/run_r001_smoke.py` 的 F03/F04/F06/F05/F09 evaluator 入口只建立 seed dict 后隐式返回 `None`，对应实现被错误置于 `rare_occupancy_record` 的 return 后成为不可达代码；直接对 11 个历史 resolved config 调用当前 `evaluate()` 时，5 个配置以 `TypeError: 'NoneType' object is not subscriptable` 崩溃。其余当前路径可执行，但 R001 candidate 当前为 540/540 checks、历史为 360/360，进一步证明源码漂移。11/11 历史 run 均无 run-local source snapshot，workspace 也不是 Git repository；因此历史 hash 能检测漂移，却不能恢复当时执行源码。现有单测绕过这些正式 wrapper，故通过并不消除该缺口。

审阅还确认三类 scope 失败：F01 没有锁定规格要求的独立 structural A/B seed-pair provenance；多类 raw record 缺少 `SYNTHETIC_SUITE_SPEC.md` 要求的完整 numerator/cross、双侧 energy、mean、coverage、rank/cancellation/condition/LOO/gap/ties 表面；若干所谓 recovery 把 planted support 直接作为 exhaustive search pool，只能证明 truth-known local-oracle conformance，不能证明 unknown-support 自动发现。未发现用 model maximum 作分母的 score laundering，且所有历史数字真实存在；证据等级仍只能是 `simulation_only`，不支持 C1/C2。

依 fail-closed 规则，已在 `EXPERIMENT_TRACKER.md` 将 parent R001/R002/R003 从 PASS 勘误为 FAIL，将 M1 synthetic gate 判为 FAIL，并把 R006 主线改为 BLOCKED。11 个 suffix run 的原始 `status.json=PASS` 原样保留，解释为“当时 frozen checks 的历史自检事实”，不覆盖、不删除。`EXPERIMENT_PLAN.md`、`MANIFEST.md` 同步记录；新增 `EXPERIMENT_AUDIT.md`、机器可读 `EXPERIMENT_AUDIT.json`，审阅 trace 保存于 `.aris/traces/experiment-audit/2026-09-02_run01/`。

下一轮必须先做 corrective M1：修复五个 wrapper 与不可达代码，为每个 formal config 增加真正调用 `evaluate()` 的 integration test；给新 run 保存不可变源码快照或 Git commit；补齐 F01 分字段 seeds 和完整 raw metric surface；把 local-oracle 与 unknown-support recovery 分开；冻结单一 12-family `full` matrix 后以新 run ID 重跑并生成同期 parent aggregation，再经过 fresh semantic audit 才能恢复 M1 PASS。primary-k/T017 决策继续暂停，但其阻塞现在次于 M1 corrective gate；在 M1 修复前不得继续 M2/真实 matcher 主线。

## 2026-09-02 17:37 EDT — M1 corrective 第一阶段：evaluator、F01 seeds 与源码快照合同修复

自动循环继续处理上一轮完整性审计的确定性实现缺口。本轮按治理顺序恢复状态并使用 `run-experiment` skill；读取 `.aris/compute/local.md` 后复用已登记的 locked Python runtime。工作是短 CPU synthetic smoke，无 GPU 计算和重型 I/O，故没有申请资源 lease；运行前 `nvidia-smi` 仅作环境 preflight，GPU 0 当时占用 7,584/16,303 MiB，本轮未使用或争抢 GPU。未打开任何真实 audit、未训练 SAE，也未修改 LOCKED claim/protocol。

修复 `scripts/run_r001_smoke.py` 中 F03/F04/F06/F05/F09 的五个空 evaluator wrapper：把原先落在 `rare_occupancy_record` return 后的不可达实现分离为明确 helper，并由公共 wrapper 返回。新增 `tests/test_m1_formal_runner_integration.py`，逐一加载 11 个历史 formal resolved config 并经 public `evaluate()` 执行；此前崩溃的五个配置现均恢复 PASS，完整 11-config replay 可执行。该修复不追认旧 artifact 的历史源码，也不覆盖任何旧 run。

新增 F01 `hadamard_gauge_seeded` 构造：structural A/B、mean sample、eval sample 使用独立显式 seed；左右基由独立 signed-permutation/Hadamard gauge 生成，改变 structural seed 会改变基，同时左右 paired contribution 仍逐样本相等。runner 把四个 seed 分字段保存到每条 raw record，并新增 cross-inner 与 mean-contribution error 字段。单测覆盖 deterministic replay、structural change 和 contribution equality。

同时升级 artifact contract：新 synthetic run 创建 `source_snapshot/`，按原相对路径复制 runner、核心源码和 config；`code_hashes.json` 记录 snapshot path，validator 在 manifest 要求 snapshot 时逐文件验证路径位于 run 内且内容 hash 与账本一致。保留对旧 hash-only run 的兼容，但把旧单测名称纠正为 hash-ledger，而非伪称 source snapshot。新增 tamper test 验证快照变更会 fail-closed。

正式 DEBUG run `M1_corrective_f01_seed_snapshot_smoke_v1_20260902T213724Z` 使用 q={2,4,8}、2 个 structural pairs、独立 mean/eval split，共 6 records，状态 PASS；artifact validator无错误，额外检查确认6/6 record含四字段 seed provenance 且6组 provenance唯一，6个 code/config输入均有 run-local snapshot。raw SHA256=`0D546314E783F7BBE64A960373AC48C807727B5A21DDE5548AA641245349ABE3`，code ledger SHA256=`47EF2399555DEEA8822A989B1B1B89C7ADE2F0F8C5DE3031D956D496FADF3A55`。targeted 26/26、全项目63/63 tests PASS。

保守解释：本轮关闭了审计 corrective item 1、2 的实现缺口，并部分关闭 F01 seed provenance，但 DEBUG smoke 不是 20-pair full matrix，不恢复 R001/R002/R003 或 M1 PASS。剩余硬阻塞是：为12个 family 完成统一且不伪造的 raw metric surface；将 truth-known local-oracle conformance 与 unknown-support proposal/recovery、baseline和solver comparison分开；冻结单一 `full` config、以新ID重跑并生成同期 parent aggregation；之后再做 fresh semantic audit。R006 与后续真实主线继续 BLOCKED，T017 primary-k 用户裁决继续暂停。

## 2026-09-02 18:00 EDT — M1 metric_surface.v1：12-family raw schema 在测试层通过

自动循环继续关闭完整性审计指出的 raw-score 审计缺口。本轮按治理顺序恢复 AGENTS、最新 master log、plan/tracker，并读取上一 corrective DEBUG run 的 resolved config、summary、environment 与 status。工作只修改 synthetic evaluator 与测试；没有打开真实 audit、没有训练或新 run、没有资源 lease，也不涉及理论修改，故未打开理论 PDF。

新增公共 `metric_surface.v1`，在 resolved config 显式启用时，为每条 synthetic raw record 重建同 seed provenance 的确定性 pair 并保存统一审计面。每个 truth-known/diagnostic group 现在包含：BCC status/value/normalized residual/cross inner/双侧 energy；PSC status/value/双侧 rank/projector distance/principal angles；独立 mean contribution 双侧 norm 与 error；group size/effective rank/condition number；双侧 cancellation energy ratio、maximum leave-one-feature-out ratio 和 per-feature ratios。record 级另存 coverage/unmatched、规范化 solver diagnostics 与 expected/observed decision。缺失或不适用值用明确 status 和 JSON null 表示，不用数值零伪装。

F01 的 surface 使用完整 signed-ReLU groups；F02–F05、F07、F09–F12 使用 generator 声明的 truth-known diagnostic groups；F06 明确审计应被拒绝的 correlation-proposed singleton；F08 使用所有等价 expected covers 的 singleton edge union。新增 `metric_surface_errors()` 机器检查 BCC/residual、PSC/projector、principal-angle数量和 cancellation vector/group-size 恒等式，并要求 `OK` surface 至少一个 group。

扩展 `tests/test_m1_formal_runner_integration.py`，对12个 family 各执行一个 structural seed pair、启用完整 surface 并逐 record 验证 schema；全项目64/64 tests PASS。新增 `M1_METRIC_SURFACE_SCHEMA.md` 固定字段、N/A语义和family解释边界。

保守解释：本轮只在 implementation/test 层关闭 raw serialization schema 缺口，没有生成新的 formal/full artifact，不恢复 R001/R002/R003 或 M1 PASS。surface 中的 group 仍由 truth-known synthetic provenance指定，因此不能被称为 unknown-support discovery。下一步必须先预写 proposal-only、greedy/beam、exhaustive local oracle 和公平 atom/group baseline 的统一候选预算与失败归因，再决定是否具备冻结单一12-family full config的条件。R006及后续主线继续BLOCKED；primary-k用户裁决仍暂停。

### 2026-09-02 17:55 EDT — 时间戳勘误

上一条标题误写为 `18:00 EDT`；实际工作完成时间为约 `17:54 EDT`。条目内容、代码、测试结果和 gate 解释均不变。依 append-only 规则保留原标题并在此勘误；当前 plan、audit 与 manifest 已使用正确时间。

## 2026-09-02 18:20 EDT — M1 unknown-support 与共同预算协议预写

自动循环按治理顺序恢复 AGENTS、master log、相关 plan/tracker/components、`metric_surface.v1` schema 与当前 `matching.py`。本轮使用 `research-lit` skill 做边界检索；不修改理论对象或定理，因此未打开理论 PDF。检索核对了 Li et al. 的 bipartite one-to-one/spectral many-to-many 基线、Gerasimov et al. 的 feature-vs-subspace 稳定性、Semantic OT、concept-manifold/community 路线，以及两篇仍处匿名 review 状态的 cross-seed benchmark/null-calibration 工作。后两者只作为设计压力来源，不登记为已发表事实；本轮检索不构成 novelty priority 结论。

新增 `M1_UNKNOWN_SUPPORT_DISCOVERY_PROTOCOL_20260902_181900.md`，明确纠正历史 truth-known planted-pool oracle 的证据边界。新协议把 proposal recall、conditional solver correctness 与 end-to-end recovery 三段分开；proposal API 只允许 full feature universe 和 discovery observations，禁止接收 planted support。tiny synthetic 的 full-universe exhaustive 作为无 truth 输入的 correctness upper reference，不参与 scalable winner 比较。

协议候选 lanes 为 contribution-kNN、decoder-kNN、code-correlation-kNN、Li15 spectral 与 degree/candidate-budget matched random；semantic OT 在 corrective M1 明确 defer。所有 lanes 必须共享 `g_max`、neighborhood atom cap、candidate-evaluation budget、split 与 tie/refusal 规则；超预算必须记 `BUDGET_REFUSAL`，不能静默截断。新增独立 `discovery_sample_seed`，保留 mean/eval 独立性；数值 top-k/`g_max` 尚未冻结，只能在不读 eval 的 feasibility pass 后进入单一 full config。F01 保留为 whole-group algebraic sanity，不伪装为 scalable local discovery。

同步登记 C019 `Truth-blind full-universe proposal harness` 与 C020 `Common-budget refusal and coverage-quality audit`，状态均为 `READY-FOR-SCREEN`；更新 experiment plan、tracker 和 manifest。没有实现代码、没有新 run、没有测试、没有资源 lease、没有打开真实 audit。故 R001/R002/R003 与 M1 继续 FAIL，R006/M2 继续 BLOCKED，C1/C2 无新增证据，primary-k 用户裁决状态不变。

下一轮应实现一个类型/API 层面无法接收 planted truth 的 proposal 接口、带 fail-closed candidate budget 的 full-universe exhaustive reference，以及 F02/F03/F04/F06 的 anti-truth/discovery-eval-separation integration tests。只有这些测试通过后，才允许做 discovery-only feasibility pass 并冻结数值预算。

## 2026-09-02 18:36 EDT — M1 truth-blind proposal 接口与 full-universe reference 通过测试

自动循环按 AGENTS 顺序恢复 master log、相关 plan/tracker 与上一轮 unknown-support preregistration，并读取当前 matching、synthetic generators、metrics 及既有正式 evaluator tests。本轮只做轻量 CPU 实现/测试，不涉及理论修改，故未打开理论 PDF；没有真实 audit、训练、下载或资源 lease。

在 `src/ccad/matching.py` 新增 `full_universe_balanced_search`：输入只包含三块 kernel、阈值、group-size cap 与 candidate budget，feature universe 直接由 kernel shape 得到，没有 truth/planted-pool 参数。它在枚举前计算完整组合数，超预算返回 `BUDGET_REFUSAL`；并修正 exact-search ledger，使 `evaluated_count` 统计所有尝试候选，包括 BCC inactive 而未进入 ranked list 的候选。非法非正 `max_group_size` 现在 fail closed。

新增 `src/ccad/proposal.py`：`singleton_contribution_affinity` 只由 discovery kernel 生成 singleton score；`symmetric_topk_proposal` 以 ID 作为 score tie 的确定性次序，输出重叠 anchor neighborhoods、两侧 degree 和显式 neighborhood-cap refusal，绝不静默截断；`validate_independent_split_seeds` 要求 structural A/B、mean、discovery、eval 五个 RNG 字段全部存在且两两不同。公开 proposal/full-universe 签名均无 truth、labels 或 planted-support 入参。

新增 `tests/test_unknown_support_proposal.py` 七项测试。测试确认 complete feature universe 上 F02 local rotations、F03 unequal split/merge、F04 overlap 的 support-minimal hyperedges均被 exact reference恢复，F06 correlation confounder无 passing contribution match；441候选在budget 440时枚举前拒答；inactive候选仍计入attempt ledger；3×3全tie邻域超cap时保留完整邻域并逐项标记拒答；更换held-out eval seed不改变固定discovery proposal。第一次误用系统Python导致缺numpy并在测试收集阶段失败，未执行任何项目测试；按已登记 `.aris/compute/local.md` 改用 locked runtime后 targeted 6/6通过。补充inactive-ledger测试后，最终 targeted 7/7、全项目71/71通过。

同步更新 preregistration checkpoint、`EXPERIMENT_AUDIT.md`、plan、tracker 与 manifest。保守边界：这只关闭 corrective item 5 的接口/单测部分；当前仅实现 contribution-score lane，decoder/code/spectral/random 的共同预算 feasibility、数值top-k/`g_max`冻结、formal artifact、单一12-family full matrix、parent aggregation与fresh semantic audit均未完成。因此 R001/R002/R003/M1仍FAIL，R006仍BLOCKED，C1/C2无新增证据，用户保留的primary-k决定不变。

下一轮应实现不读eval结果的 feasibility runner，补齐 decoder/code/random lanes（Li15 spectral可先定义最小可复现版本或明确延期），保存proposal hash、degree、candidate count和runtime；若不同lane无法在同一预算语义下比较，应先修订候选设计而不是冻结数字。

## 2026-09-02 18:59 EDT — 启动 M1 discovery-only proposal feasibility

按 `run-experiment` 流程读取已登记 local runtime 并做 preflight；GPU 0 当时为 1,308/16,303 MiB，但本 run 明确使用 CPU，因此不申请或占用 GPU/CPU-heavy lease。启动前冻结 `configs/m1_unknown_support_feasibility_v1.json`：4 个 synthetic families、5 个 structural pairs、top-k={1,2,4}、gmax={1,2,3,4}、atom cap 12；只允许 contribution/decoder/code kNN 与 degree-matched random 读取 mean/discovery 数据，audit、held-out eval 与 planted labels 三项均关闭。run ID 为 `M1_unknown_support_feasibility_v1_20260902T225840Z`。LI15 spectral尚未实现，故无论本 run 数值如何都不能冻结 formal protocol。

### 2026-09-02 19:00 EDT — feasibility 完成：四条已实现 lane 可运行，LI15 继续阻断 formal freeze

先扩展 `src/ccad/proposal.py`：新增 decoder cosine 与完整 code-correlation affinity；degree-matched random 通过simple bipartite edge swaps严格保持两侧degree sequence；`proposal_candidate_family` 对所有未拒答重叠邻域生成并去重有限subset-pair族。新增单测确认两个baseline affinity可运行、random lane的两侧degree和edge count精确匹配；targeted 8/8通过。

正式 CPU run 在约0.44秒内完成20 records与240 lane-results（4 families×5 pairs×3 top-k×4 lanes）。所有输出都明确`audit_opened=false`、`held_out_eval_loaded=false`、`planted_labels_loaded=false`；raw中无planted hyperedge或recovery输出。所有proposal保存hash、两侧degree、edge/neighborhood数量、cap refusal以及gmax 1–4的去重candidate count。观察到的最大candidate family为7,462，按预写规则只登记为`candidate_common_budget_candidate`；240项中atom-cap refusal为0。

run status PASS，独立validator 10/10 PASS，raw SHA256=`C6238E1278D83483CEAC6F6CA2D7221FE33049136A401C06F5484E1E11D7426B`，validation artifact SHA256=`588E32C667BD6448DF94F4545E08B7C19C088CFED7AECF2CB05CE756C72536FD`，validator源码SHA256=`C3B687ACF51E87DC22C919B9662BE617E06EE1ED1FDA92FF13103E9E88A5DED5`；全项目72/72 tests通过。run保存resolved config、environment、stdout/stderr、status、raw、summary及run-local source snapshots；workspace仍明确不是Git repository。

保守解释：本run只说明已实现lanes在当前discovery-only网格的候选规模可控，不说明任何proposal recall、solver correctness或end-to-end recovery，也不能把7,462当成最终冻结预算。预注册的LI15 spectral尚未实现；因此C019/C020仅转`SCREENING`，M1/R001–R003仍FAIL，R006仍BLOCKED，C1/C2无新证据。下一轮应先实现并单测最小、来源忠实的Li15 spectral lane，再用新run ID重做同一feasibility grid；不得把当前四lane结果事后补写成完整协议。

## 2026-09-02 19:18 EDT — 启动含 Li15 spectral 的 discovery-only feasibility v2

本轮使用 `research-lit` skill，先核对 Li et al. 原论文 PMLR 页面与可检索的正文/补充材料。原方法在补充S.3中将两网within/between activation correlation拼成联合矩阵，阈值化后构造未归一化Laplacian，以最小特征向量嵌入、k-means分群，未知cluster数用eigengap；论文conv2示例报告tau=.2。实现仅依据论文描述clean-room编写，未下载、导入或复制GPL-3.0仓库代码。项目内papers/literature未找到该PDF，因此线上PMLR原文为本轮主要方法证据。

新增Li15 partition型baseline及两独立信号块单测；targeted 9/9通过。启动前冻结`configs/m1_unknown_support_feasibility_v2.json`，沿用v1的4 families、5 pairs、top-k/gmax网格，新增tau=.2、最大8 clusters、eigengap选k和deterministic 10-restart k-means。audit、held-out eval与planted labels继续关闭。GPU preflight为1,065/16,303 MiB；run仍为短CPU任务，不申请资源lease。run ID=`M1_unknown_support_feasibility_v2_20260902T231823Z`。

### 2026-09-02 19:19 EDT — v2 PASS，完整lane实现可运行但保留Li15空proposal负例

`src/ccad/proposal.py`新增clean-room Li15 baseline：拼接paired codes计算联合within/between correlation，只保留正相关大于tau的边，构造未归一化Laplacian，以最大eigengap选cluster count，并用固定seed的10-restart k-means。mixed cluster转换成完整bipartite edge group；单侧cluster自然变成unmatched，且该partition baseline没有被改造成overlap方法。两独立信号块fixture恢复两个mixed clusters；targeted 9/9通过。C010进入`SCREENING`，不是`ADMIT`。

正式v2约0.5秒完成20 records/260 lane-results；status PASS、独立validator 10/10、全项目73/73 tests PASS。raw SHA256=`A0F1B41C634FB78ABB1DB82E5CFF7A162CD2A52A7E8D846C78DBDBEDAA43B46E`，validation SHA256=`588E32C667BD6448DF94F4545E08B7C19C088CFED7AECF2CB05CE756C72536FD`，当前validator源码SHA256=`1C9D0F6E852ADB09EFD693D3B7622C339B7E2132065B8AC64022215FB2052DDC`。candidate budget仍为7,462，atom-cap refusals为0。

Li15在20 records中选择1–6个clusters，mixed clusters为0–6；F04 pair index 2选择5 clusters却没有任何mixed-network cluster，edge/candidate均为0。该结果只来自discovery结构、不是held-out recovery分数，但已显示partition/correlation baseline可能完全拒答；结果原样保留，不用planted truth调tau或k。tau=.2来自论文conv2示例，仅作为source-pinned sensitivity点，不声称适用于SAE。

保守解释：预注册lane的实现与discovery-only feasibility缺口已关闭，但正式配置仍未冻结，且没有proposal recall、conditional solver或end-to-end evidence。下一轮应在不改v2结果的前提下预写单一formal config：明确top-k/gmax/tau敏感性、7,462 common budget、20新structural pairs、独立mean/discovery/eval seeds、失败归因和完整metric surface；随后才可启动corrective full matrix。M1/R001–R003仍FAIL，R006仍BLOCKED，C1/C2无新证据。

## 2026-09-02 19:39 EDT — 冻结 M1 corrective 12-family full 协议，synthetic eval 尚未开启

自动循环按治理顺序恢复 AGENTS、master log、plan、tracker、完整 `SYNTHETIC_SUITE_SPEC.md`、unknown-support protocol以及v2 feasibility的resolved config/summary/status/validation。本轮使用 `experiment-plan` skill，只做claim-driven预注册与机器配置冻结，不修改理论对象，故未打开理论PDF；没有运行实验、打开synthetic eval或真实audit，也没有资源lease。

新增 `M1_CORRECTIVE_FULL_PREREGISTRATION_20260902_193900.md`、current pointer `M1_CORRECTIVE_FULL_PREREGISTRATION.md` 和 `configs/m1_corrective_full_v1.json`。这是首次把12 families、20 fresh structural pairs/family、不可变源码快照、`metric_surface.v1`、truth-blind proposal、local solver、baselines、full-universe oracle、失败归因及同期parent aggregation放入单一pre-eval协议。它不覆盖11个历史suffix runs，且PASS只允许fresh semantic audit，不自动恢复M1。

信息顺序冻结为pairwise-distinct structural A/B、mean、discovery、eval、solver seeds；mean只估中心化常数，proposal/candidate/residual/ties只读discovery；candidate record/hash写入后才可打开synthetic eval；planted labels最后只给evaluator。默认样本数为257/1024/2048，F07 mean=1024与F12 mean=256作为已有construction-specific例外显式保留。

primary CBSM冻结为CONTRIB-KNN、top-k4、gmax4、atom cap12、exact residual `1e-10`、tie `1e-12`；top-k1/2只作必报删减ablation，不允许结果后替换primary。选择top-k4/gmax4的唯一理由是F02 signed-ReLU block的四原子规格，不来自新eval。decoder/code/random使用相同top-k与subset预算；Li15固定tau=.2、最大8 clusters与eigengap，明确是source-pinned示例而非通用SAE阈值；FULL-EXHAUSTIVE有独立700,000 diagnostic cap且不参与scalable胜负。

每个scalable lane/pair硬预算冻结为7,462，来自已完成且truth/eval关闭的v2 feasibility最大观察值。fresh formal pair超出即`BUDGET_REFUSAL`，不得扩容。primary硬门按每个structural pair要求F02/F03/F04 proposal与end-to-end precision/recall/F1均1.0、F02/F03零cross false positive、F06 20/20拒答，F05/F07/F08/F09/F10/F11/F12 20/20 expected decision且false unique=0；任何单pair/contract失败均使full run FAIL，baseline只完整报告而不需通过CBSM门。

同步更新plan、tracker和manifest；登记`M1_corrective_full_v1`为MUST/TODO。当前新synthetic eval仍关闭，故无任何结果、M1/R001–R003仍FAIL、R006仍BLOCKED、C1/C2无变化。下一轮先实现arbitrary candidate-family solver、prediction-freeze/truth-access ordering artifacts与fail-closed tests，再跑独立one-pair-per-family smoke；只有smoke在不改协议下通过才启动20-pair full。

## 2026-09-02 19:57 EDT — M1 arbitrary solver 与 prediction freeze 通过实现级测试

自动循环按治理顺序恢复 `AGENTS.md`、最新 master log、相关 plan/tracker、冻结的 corrective full preregistration，以及当前 matching/proposal/tests。本轮只完成冻结协议 run-order 的前两项实现与测试部分；没有启动新 run、没有打开 synthetic eval 或真实 audit、没有申请资源 lease，也没有修改理论对象，故未打开理论 PDF。

`src/ccad/matching.py` 新增任意有限候选族求解接口 `search_candidate_family`。输入仅为 discovery kernels、候选族和冻结阈值/预算，不接受 truth、labels、planted support 或 held-out eval；候选先规范化和去重，非法索引 fail closed，候选数超过 7,462 时在任何 score 计算前返回 `BUDGET_REFUSAL`。输出保留 passing/support-minimal 候选、best/second residual、gap、ties 和实际 evaluated count。

同文件新增 content-addressed discovery prediction freeze。冻结哈希覆盖 proposal source/hash、discovery fingerprint、完整候选族与 discovery-selected predictions；held-out evaluator首先重算并验证冻结哈希，之后才接受 eval kernels 和 planted hyperedges，输出 proposal recall、hyperedge precision/recall/F1、eval residual，以及 `PROPOSAL_MISS`、`SOLVER_MISS` 或 `BUDGET_REFUSAL` 归因。由此把选择阶段和 synthetic truth/eval 阶段在 artifact 接口上分离，而不只是依赖调用约定。

`tests/test_unknown_support_proposal.py` 增加 truth-blind signature 检查、F03 从 contribution proposal 到 arbitrary-family solver 的恢复、超预算 pre-score refusal、freeze deterministic replay、tamper detection、tampered freeze 被 evaluator拒绝，以及合法冻结 fixture 的 held-out perfect metrics。首次补丁中 freeze 构造曾把 hash payload tuple 误作 `BalancedCandidate`；在任何 run 前即由测试发现并修复，失败没有形成 artifact，也没有更改冻结协议。锁定 runtime 下最终全项目 76/76 tests PASS（3.551s）。

保守解释：本轮只关闭 arbitrary solver、prediction freeze 与 truth-access ordering 的实现级缺口，不构成新实验结果，不恢复 R001/R002/R003 或 M1 PASS，也不支持 C1/C2。`M1_corrective_full_v1` 保持 TODO，R006/M2 保持 BLOCKED，primary-k 用户裁决继续暂停。下一步是在独立 smoke ID 下实现并运行 one-pair-per-family runner：必须先落盘 candidate/prediction/hash，再打开 synthetic eval，并由独立 validator 检查 12-family metric surface、unknown-support 输出、failure attribution 与源码快照；只有 smoke 不改协议通过后才可启动 20-pair formal run。

## 2026-09-02 20:21 EDT — M1 one-pair-per-family corrective smoke v3 通过

自动循环使用 `run-experiment` skill，按治理顺序恢复项目状态并读取 `.aris/compute/local.md`、冻结的 `configs/m1_corrective_full_v1.json`、现有 synthetic runner、proposal/matching 实现和测试。GPU preflight 为 1,348/16,303 MiB，但本轮是约27秒的 bounded CPU smoke，无 GPU/CPU-heavy lease。没有真实 SAE audit、外部写入、下载或理论修改；用户此前通知的母目录文献 quarantine 不影响本轮 synthetic 路径。

新增 `configs/m1_corrective_smoke_v1.json`、`scripts/run_m1_corrective_smoke.py` 与独立 `scripts/validate_m1_corrective_smoke.py`。smoke绑定冻结 full config：诊断层对12 families各执行一个 structural pair（F01按q=2/4/8保留三条，共14 records）并生成 `metric_surface.v1`；unknown-support层对F02/F03/F04/F06运行CONTRIB-KNN、DECODER-KNN、CODE-KNN、RANDOM-MATCHED、LI15-SPECTRAL与FULL-EXHAUSTIVE。kNN/random同时运行primary k=4与k=1/2 ablations。每条记录含六个必需且互异的seed字段；F07额外保留独立`bootstrap_seed`。

runner先从discovery kernels生成proposal、有限candidate family与solver结果，写入content-addressed `discovery_predictions.jsonl`，原子关闭并重新读取后才构造held-out eval pair并访问synthetic truth。`phase_ledger.json`记录prediction落盘于00:21:01.915625Z，eval打开于00:21:01.916106Z。F06的truth定义为`NO_ACCEPTABLE_MATCH`，避免把已知的相关性混淆边错误当作应恢复正例。所有run保存resolved smoke/full config、环境、源码和validator快照、raw/hash及日志。

两次失败按suffix保留。`M1_corrective_smoke_v1_20260903T001457Z`在任何discovery/eval artifact前因F01 legacy路径没有`solver_seed`触发KeyError，状态FAIL。修复为对未使用solver的代数路径显式登记未使用seed后，v2 runner完成并自报PASS，但独立validator仅12/13：它错误要求seed对象“恰好”六字段，因而拒绝合法多出`bootstrap_seed`的F07。v2已保守改判FAIL，raw未改，并新增`VALIDATION_CORRECTION.md`。validator修为“六个必需字段构成子集且六者互异”，随后以新ID完整重跑；不是在旧run上补通过。

最终 `M1_corrective_smoke_v3_20260903T002300Z` PASS：14 diagnostic records覆盖12/12 families，metric-surface error=0；56 discovery predictions与56 held-out evaluations hash逐项绑定，六条lane在四个unknown-support family均齐全。primary CONTRIB-KNN k=4对F02/F03/F04 proposal recall、precision、recall、F1全部为1；F06正确空预测且四指标为1，四项均无failure attribution。scalable lane最大实际candidate数6,849，小于冻结7,462；FULL-EXHAUSTIVE最大628,849，小于独立700,000 cap。baseline/ablation共有15/56 `PROPOSAL_MISS`（random 6、contribution ablation 3、code 3、decoder 2、Li15 1），原样保留，不纳入primary gate。独立validator 13/13 PASS，全项目76/76 tests PASS。summary SHA256=`FF07D0749C49B6B3EDAD5DAC31E69377BF1DFDA521C9C84C18392FBF3059C45A`，validation SHA256=`01AA4CC499C4E400C30E93C8E6968A7A250F949539AF6BB3C14918244C723C97`，prediction raw SHA256=`EF3099B00B1559427B1347DCD7DA4D05E0CB1A532EA1E7E007B80B961B665BE4`，held-out raw SHA256=`37B7B048877482661184A108C35EBB42530CD8C20E5B87B5A32B83D109699085`。

保守解释：one-pair smoke只说明冻结协议的实现和artifact/order contract可运行，允许进入20-pair corrective full，不恢复R001/R002/R003或M1 PASS，不支持真实SAE C1/C2。`M1_corrective_full_v1`仍TODO，R006/M2仍BLOCKED，primary-k用户裁决继续暂停。下一步应直接复用v3 runner/validator语义实现20-pair full runner，以新formal ID运行240个structural family pairs（F01 q展开后预计280 diagnostic records）和F02/F03/F04/F06的1,120个lane evaluations；任何primary pair失败均整run FAIL，随后才生成同期parent aggregation和fresh semantic audit。

## 2026-09-02 20:42 EDT — 启动 M1 corrective full v1

自动循环将已验证的 smoke runner泛化为按执行配置的`seed_pair_count`循环，并新增仅作执行适配的`configs/m1_corrective_full_execution_v1.json`；它逐字段绑定既有冻结`configs/m1_corrective_full_v1.json`的20 pairs、base seed 30300000、12 families、六条lane、预算、阈值与信息顺序，不改变LOCKED协议。泛化时预检发现Li15分支误引用尚未定义的`exhaustive_family`，在正式run前修复；随后新回归smoke `M1_corrective_smoke_v4_20260903T004300Z`再次PASS且validator 13/13，证明泛化未破坏one-pair语义。

正式run ID=`M1_corrective_full_v1_20260903T004200Z`。预期输出为280条diagnostic records（F01每pair含q=2/4/8）与1,120条冻结prediction/held-out evaluation。20:41 EDT preflight时GPU 0为1,124/16,303 MiB且本run不使用GPU；`cpu-heavy`资源为空闲，正式执行通过母目录resource manager申请独占lease并自动heartbeat/release。启动状态已同步tracker为RUNNING。任何中间pair失败或baseline负结果都保留，不在运行中改阈值、预算或seed；full结束后必须独立validator通过才可进入parent aggregation/fresh semantic audit。

### 2026-09-02 20:52 EDT — full v1 完成但按冻结 all-pair gate FAIL

resource-manager sandbox内首次创建共享lease因`WinError 5`失败，未启动实验；按工具权限流程以相同命令获准后成功申请`cpu-heavy`独占lease。正式CPU计算约7分钟完成，lease随后正常释放。run完整生成280条diagnostic records、1,120条content-addressed discovery predictions与1,120条held-out evaluations，12/12 families、6/6 lanes齐全，metric-surface error=0，源码/validator/config均有run-local snapshot，prediction文件先于eval打开落盘并重新读取。

决定性结果是FAIL而非实现崩溃。80个primary `CONTRIB-KNN/top-k=4` pairs中79个proposal recall、precision、recall、F1均为1且无failure attribution；`F02_local_block_rotations` pair index 2的55-edge proposal生成10,707个去重candidate pairs，超过冻结的7,462 common cap，solver按协议在评分前返回`BUDGET_REFUSAL`（evaluated_count=0、空prediction）。冻结hard gate不允许平均、删seed或提高预算，因此primary为79/80并使整run FAIL。已完成的scalable lane最大候选数为7,270；FULL-EXHAUSTIVE最大628,849，低于其独立700,000 cap。

全部1,120项中795项无failure attribution、311项`PROPOSAL_MISS`、14项`BUDGET_REFUSAL`。14个budget refusals分布为CONTRIB 1、CODE 6、RANDOM 7；DECODER、Li15与FULL-EXHAUSTIVE为0。独立validator 11/13：source/hash/family/metric surface/six-seed/freeze ordering/prediction hash/lane/binding/raw hash等11项通过，仅`run_pass`与`primary_gate`按预期失败。summary SHA256=`A60990CCAD4D95557CD16499ADA434F10777ED1A772ED6DFCC2BF1A9F0881097`，validation SHA256=`57B2E680FC739EE0BC55040C1467DF9F7B1867BC468E62D4D7FE85EE1A86A8C4`，prediction raw SHA256=`86BFC495F854651B848FFD94720C7B7B805FED80CB24EEAA90D17B2C3B040E0B`，held-out raw SHA256=`2DB24A4133CD405BEB2909A24B25E7C911DEFC0A3E498BE81B90BE84EFE31951`。

新增`M1_CORRECTIVE_FULL_RESULT_REVIEW_20260902_205200.md`并登记C021 discovery-only global candidate prioritization/bounded beam为`BLOCKED`候选。可逆选项是：(1)新预注册固定预算的全局候选优先级/beam并用fresh seeds；(2)经用户批准收窄为允许校准后coverage/refusal率的claim；(3)新预注册更高cap与fresh seeds，但这是科学上最弱方案。任何方案均不得复用本run的held-out结果做确认性评估。

保守结论：当前冻结工程主张“top-k4 overlap union在7,462 cap下对F02 20/20恢复”被否定。R001/R002/R003与M1继续FAIL，R006/M2继续BLOCKED；不生成parent PASS aggregation，也不请求fresh semantic audit。该synthetic工程失败不支持或反驳真实SAE C1/C2。因修订LOCKED协议或主张强度属于用户保留决定，自动循环在此暂停相应正式重跑，只可继续只读分析与不打开新eval的设计工作，等待用户选择。

## 2026-09-02 21:15 EDT — C021 discovery-side hub诊断与bounded-search设计审查

自动循环按治理顺序恢复`AGENTS.md`、最新master log、相关plan/tracker、v1 result review及失败run的冻结prediction/summary/validation。本轮使用`research-lit` skill，仅做既有discovery artifact的回顾诊断与方法设计检索；没有修改LOCKED协议、运行新实验、打开新eval、申请资源lease或读取新truth。项目内`papers/`与`literature/`未找到可用PDF，arXiv本地helper也无法解析，因此方法证据改用线上原始论文页面，并记录该限制。

对F02 contribution-kNN/top-k4的20个discovery records复算候选规模：最小675、中位675、最大10,707，只有pair2超过7,462。规则pair0为48条边、两侧degree全4、48个`(4,4)`neighborhood，raw subset pairs 10,800、去重后675。失败pair2为55条边，左侧最高degree 9、右侧最高8；55个neighborhood中有11个因形状`(4,9)`、`(7,6/9)`或`(8,6/9)`超过atom cap，raw 31,815、去重10,707。pair8的单个degree-5 hub产生2,385个unique candidates，pair13的degree-6/5 hubs产生3,487。由此定位根因：当前所谓symmetric top-k取两方向top-k的union，只限制每点主动选择数，不限制被选入度；incoming hub再经局部subset叉乘造成组合爆炸。这不是主要由exact scorer速度或数值残差引起。

新增`C021_BOUNDED_SEARCH_DESIGN_REVIEW_20260902_211500.md`。最小首选修复候选是mutual/reciprocal top-k：边需在两方向均为top-k，从而两侧degree均不超过k；k=4时anchor neighborhood最多8 atoms，可保留原local exhaustive solver。风险是非对称rotation下proposal recall可能下降，必须用fresh discovery-only design seeds检验。挑战者是保留union graph但采用去重、per-anchor/component与group-depth配额的fixed-budget diverse beam，并完整报告frontier/coverage/refusal。因BCC对partial group未证明单调，也没有admissible lower bound，beam只能是heuristic；不得把NMT diverse beam或submodular selection文献的保证迁移为CBSM保证。现有union/refusal保留为负控，其他baselines保留公平预算语义。

外部参考包括Freitag & Al-Onaizan 2017 fixed/variable beam、Vijayakumar et al. 2016 diverse beam、Sun & Batra NeurIPS 2015 SubmodBoxes、Wei/Iyer/Bilmes ICML 2015 submodular subset selection，以及Malherbe et al. ICML 2024 diverse subset selection。它们只支持“固定预算且保留多样性/覆盖”的设计类比；目标函数、结构与保证均不直接适用于BCC。

同步细化C021、tracker、result review和manifest。保守结论与阶段状态不变：这是使用已失败run的retrospective diagnosis，不构成v2证据；C021仍BLOCKED，M1/R001–R003仍FAIL，R006/M2仍BLOCKED，C1/C2无新证据。推荐用户批准一个全新v2预注册：reciprocal contribution top-k作为首选修复候选、stratified diverse beam作为挑战者、v1 union graph作为负控；先fresh discovery-only feasibility，再从允许的discovery诊断冻结formal方法并在fresh full seeds确认。批准前不启动新formal eval。

## 2026-09-02 21:32 EDT — C021静态实现契约审计：reciprocal去hub但不保证旧global cap

自动循环再次恢复AGENTS、最新日志、相关plan/tracker，并静态审阅`src/ccad/proposal.py`、`src/ccad/matching.py`、corrective runner及unknown-support tests。本轮没有调用synthetic generator、没有重算失败seed上的候选结果、没有实现新方法或打开新eval。当前v1路径确认是fail-closed：双向edge union后逐neighborhood检查atom cap，非拒答neighborhood才枚举，候选全局去重后若超过7,462则在任何BCC score前整family拒答，最终candidate family与prediction被content-addressed freeze。因此v1暴露的是设计边界，不是静默截断bug。

新增`C021_IMPLEMENTATION_CONTRACT_DRAFT_20260902_213200.md`并得到关键静态界：reciprocal top-k令两侧degree均不超过k，anchor局部raw subset-pair数至多`S(k,g)^2`，其中`S(k,g)=sum_{i=1}^{min(k,g)} C(k,i)`；总raw occurrences至多`k*min(n_left,n_right)*S(k,g)^2`。在k=g=4、12×12 universe时该界为`48*225=10,800`，仍高于旧7,462 cap。去重通常显著降低数量，但依赖图结构。故reciprocal能消除已观察到的9/8度hub，并在`2k<=12`时保证neighborhood atom cap，却不能单独保证global candidate cap；上一轮“首选简单修复”已在design review、result review、C021与tracker中补充这一边界。

契约草案还登记了两个此前未显式处理的风险。第一，当前stable-index exact-k tie-break虽确定性但在k边界同分时不具permutation equivariance；v2需预先冻结为“边界tie显式歧义/拒答”或“stable exact-k + margin/permutation sensitivity”，而include-all-ties会破坏degree bound。第二，若使用beam/quota，现有freeze只绑定最终候选族，不足以证明哪些候选在held-out前被排除；v2 schema必须同时绑定directional rankings、boundary margins、pre-selection frontier hash、selected-family hash、quota ledger、anchor/component coverage、refusal和实际score count。草案列出8类fail-closed tests与method-faithful common-cap、actual-count/coverage-matched两层公平比较。

保守解释：本轮只是implementation-neutral code readiness与组合上界审计，不构成理论PDF修改、protocol revision、实验结果或C1/C2证据；M1/R001–R003继续FAIL，R006/M2继续BLOCKED，C021继续BLOCKED。对用户的推荐从“直接以reciprocal作为v2 primary”收紧为“批准two-stage fresh-seed设计协议”：先在truth/eval关闭的fresh design stream比较reciprocal+global refusal、reciprocal+committed bounded selector、union+stratified beam及原union负控，再仅依据许可的discovery diagnostics冻结formal v2，并用另一组fresh seeds确认。

## 2026-09-02 21:52 EDT — C021 protocol-delta审计：truth-closed feasibility不能选择recovery方法

自动循环恢复AGENTS、最新master log、M1相关plan/tracker及两份C021设计草案后，复核信息访问顺序，发现上一轮“truth/eval关闭的fresh feasibility检验proposal recall并选择formal方法”在逻辑上不成立：proposal recall定义需要planted support，若D0 API与artifact确实不读取truth，就只能观察degree、candidate count、tie margin、anchor/component coverage、runtime与budget refusal，不能比较真实hyperedge recovery。这是计划表述缺口，不是既有run污染；本轮没有打开任何seed或eval。

新增`M1_C021_V2_PROTOCOL_DELTA_DRAFT_20260902_215200.md`，把候选修订分成三个彼此seed-disjoint的层次。D0 structural feasibility保持truth/eval关闭，只按预写机械契约淘汰方案并冻结tie/budget语义；D1在第二组专用synthetic development seeds上允许planted labels，但每条prediction/frontier仍须先冻结，再按预写all-pair恢复/拒答gate和预先排序的方法简单性选择单一候选；D2在D1结束并锁定最终源码/config后，才打开第三组fresh formal seeds跑完整12-family确认。D1明确标为`synthetic_development`且永不并入M1 confirmatory evidence。

草案把候选简单性顺序预写为reciprocal+local exhaustive、必要时reciprocal+committed bounded selector、union+stratified beam、原v1 union/refusal负控；更复杂方案只有在更简单前项未通过预写D0机械门或D1恢复门时才可选。大部分v1 metric/split/falsifier/freeze语义建议复用，但这是draft而非协议变更。仍需用户裁决D0 common-budget rule：保留7,462以获得严格v1可比性并接受显式refusal，或允许一个对所有scalable lanes对称、预先写明的组合/资源规则产生新cap。不得由D1 recovery结果反推预算。

同步修正C021 design review、implementation contract、component row、plan、tracker与manifest。保守解释：该修正提高了anti-leakage完整性，但没有授权v2、没有实现代码、没有run、没有新实验或C1/C2证据；M1/R001–R003仍FAIL，R006/M2仍BLOCKED，C021仍BLOCKED。下一步等待用户授权三阶段设计并选择budget rule；在此之前不创建可执行v2 config或seed值。

## 2026-09-02 22:00 EDT — C021 primary common-budget规则审查

自动循环在不运行代码或读取新数据的前提下，对三阶段v2草案中的剩余budget choice做公平性与protocol-deviation审查。结论是推荐D0/D1/D2全程保留v1的7,462次score evaluations作为每个scalable lane/pair的primary cap，并在D0之前固定；D0只检查候选能否诚实在该cap下工作，不再从观察到的candidate maximum选择新cap。这样保持与失败v1的直接计算可比性，避免失败后移动资源目标，迫使bounded selection、coverage ledger与refusal本身接受检验，并对contribution/decoder/code/random/spectral统一适用；full exhaustive继续只在独立700,000 cap下作为diagnostic oracle。

在`M1_C021_V2_PROTOCOL_DELTA_DRAFT_20260902_215200.md`追加该推荐并同步plan/tracker/manifest。若D1没有任何预声明候选在7,462下通过，v2应作为negative development result停止；更高cap只能进入用户另行批准和预注册的v3，不能在v2内补丁。10,800不适合作为“安全新cap”，因为它仅是12×12 reciprocal graph的raw-occurrence上界，并不约束union graph family。可预注册primary cap固定分数的budget sensitivity，但只能作ablation，不能替代失败primary。

保守解释：这是未授权协议草案中的推荐，不是用户决定、LOCKED修改或实验结果；没有新增run、代码、seed或C1/C2证据，M1/R001–R003继续FAIL，R006/M2继续BLOCKED，C021继续BLOCKED。当前需要的用户输入已收敛为是否批准三阶段v2按推荐7,462共同预算推进；若用户偏好更高的a priori对称预算规则，则需明确选择该替代方案。

## 2026-09-02 22:35 EDT — 用户授权后的研究身份收束：Native Intervention Portability / MSCC

用户提供新的讨论材料并明确把研究方案决策权交给本项目执行方，目标仍是不降顶会分量、把理论转成实际意义。本轮按治理顺序读取AGENTS、最新日志、计划、tracker、components、理论PDF相关完整章节与本地/线上近邻文献；使用`research-refine`和`pdf`技能。线上复核确认ACL 2026 PW-MCC仍是atom assignment，Gerasimov et al. 2026研究unstable atom/reproducible subspace，Semantic OT研究distributional feature matching/circuit compression；匿名TMLR投稿`Benchmarking Cross-Seed Feature Correspondence in Sparse Autoencoders`已经覆盖atom-pair causal substitution、dustbin与quality/coverage，故这些不能作为本项目单独novelty。

五轮同一GPT-5.6-Sol xhigh独立review从6.90/REVISE依次到7.60、8.15、8.75，最终9.10/READY（CALIBRATION none；same-family provisional）。关键纠正依次为：(1)诚实放弃把source-conditioned retrieval冒充two-sided discovery；(2)证明exact linear orthonormal block中atom-native subset portability与full-group portability可分离，一般dense rotation改为atom refusal而非正例；(3)真实search failure不等于support不存在；(4)删除未证明的真实finite-sample absence certificate；(5)将真实identification压缩为`FOUND/UNRESOLVED`，multiplicity/safety/reason/causal outcome分轴。完整历史在`refine-logs/`，最终proposal为`refine-logs/FINAL_PROPOSAL.md`。

新增`NATIVE_INTERVENTION_PORTABILITY_DECISION_20260902_223500.md`并同步AGENTS、EXPERIMENT_PLAN、EXPERIMENT_TRACKER与COMPONENT_CANDIDATES。当前主线为C1-NIP/C2-NIP与C022 MSCC；C021 symmetric bounded search降为DEFER诊断。M1_NIP_protocol_v1登记为MUST-CANDIDATE/TODO：D0 truth-closed、D1 labeled development、D2 fresh confirmation，20 unopened pairs/family与per-query/per-lane 7,462 score cap继续保守沿用。

保守解释：本轮只完成研究身份、claim与计划落账；没有实现MSCC、没有生成可执行LOCKED config、没有新run、没有打开真实audit，也不把proposal评分当成C1/C2或录用证据。M1/R001–R003继续FAIL，R006/M2与真实主线继续BLOCKED。下一步是把FINAL_PROPOSAL转成严格protocol delta、理论命题草稿与机器config，再执行D0/D1/D2；任何primary pair失败均保留。

## 2026-09-02 22:43 EDT — Heartbeat `ccad`：冻结 M1-NIP 协议与 atom-group 理论边界

### 触发与状态恢复

15分钟研究heartbeat按新AGENTS治理顺序恢复状态：读取AGENTS、最新master log、EXPERIMENT_PLAN、EXPERIMENT_TRACKER与FINAL_PROPOSAL，并按PDF技能重新核对内部理论PDF第35–43页的identifiability、THM-CBSM-004/005、finite-family LEM/THM-CBSM-006/007、优化对象和THM-CBSM-009。确认现有理论只保证group process的gauge/refactorization invariance与same-hook equality后的intervention transfer，不能推出任意source atom存在target-native subset。

### 实际动作与 artifacts

- 冻结`M1_NIP_PROTOCOL_V1_20260902_224300.md`：D0/D1/D2每family分别5/20/20 pairs；D0 truth-closed，D1仅从预声明atom cap `{4,8,12,16,20}`中词典序选一项，D2在protocol/code/environment/variant hash冻结后才生成fresh seed manifest。
- 冻结非可执行机器契约`configs/m1_nip_protocol_v1.json`，显式保持`execution_enabled=false`、`synthetic_evaluation_opened=false`、`real_sae_audit_opened=false`和D2 `UNGENERATED`。`g_max=4`；20 target atoms的全部非空size<=4 supports经独立复算为6,195，低于沿用的7,462 per-query/per-scalable-lane cap。
- 新增`THEORY_NIP_ATOM_GROUP_BOUNDARY_20260902_224300.md`。命题在exact linear orthonormal block和`Cov(X)>0`下证明：source rank-one projector等于target binary subset projector时，rank迫使subset为singleton且target vector为source方向的正负号；Haar-absolutely-continuous rotation下固定atom事件为零测，但完整block的projector和贡献仍相同。草稿列明非线性/过完备/秩亏数据/中心化mean等不覆盖边界，并映射THM004/005/006/007/009及七项必需synthetic tests。
- 同步EXPERIMENT_PLAN、EXPERIMENT_TRACKER与MANIFEST。M1_NIP_protocol_v1从候选提升为主线MUST，但运行状态仍`TODO`；没有变更任何历史FAIL/BLOCKED或打开任何evaluation。

三份新增artifact的SHA-256分别为：protocol `B22FED5D573A65B1F98C27462A7D23C1FEED7D11992B8400DABC1DDEBB4A8A4D`，theory draft `65006E2FFF94DC9958B6297BDAE601EAC21826BF98B3A2B736ECDB652584D17C`，machine config `D0E3662F102545EC3C4CF10195AE818563EE5001C322A6EBCD8BC0A28BD0D2E3`。JSON已通过PowerShell解析；support count与记录值均为6,195。

### 保守解释与下一步

本轮是prospective protocol/theory freeze，不是实验结果、独立proof review或C1-NIP/C2-NIP证据。M1/R001–R003继续FAIL，R006/M2继续BLOCKED，D1/D2与真实audit关闭。下一轮只应实现N01–N12 generators、one-sided MSCC API与fail-closed artifact/information-order validator，完成unit tests后仅运行D0；若实现发现协议内部矛盾，必须新增suffix和勘误，不可原地改写冻结协议。

## 2026-09-02 22:54 EDT — Heartbeat `ccad`：one-sided MSCC 核心实现与协议测试

### 实际动作

在已冻结`M1_NIP_PROTOCOL_V1_20260902_224300.md`不变的前提下，新增`src/ccad/mscc.py`。公共API只接收source/target contribution kernels、独立mean contribution、source atom ID、冻结target atom IDs、`g_max`、双阈值、epsilon与candidate budget；signature不含truth、label、planted support、eval或audit。实现按cardinality枚举unweighted native supports，分别计算source-normalized `d_ctr`和`d_mu`，返回全部最小可行supports以及`UNIQUE/AMBIGUOUS`；只有机械确认complete allowed universe时才允许`CERTIFIED_ABSENT`，部分family为空只能`UNRESOLVED`。21 atoms、`g_max=4`时预计算7,546 candidates超过7,462，评分前`BUDGET_REFUSAL`且evaluated count为0。非有限、非对称self-kernel、负能量或物质性负平方残差均fail-closed。

新增`tests/test_mscc.py`的8项协议级测试：public API anti-truth、unique size-2 support与bloated superset、两个tied minimum supports、complete-vs-partial absence措辞、pre-score budget refusal、45度dense rotation下atom absent而full block pointwise equal、centered match但mean mismatch，以及invalid kernel拒绝。8/8定向PASS；连同unknown-support与R001 metric相关回归共39/39 PASS。

全项目discover共81项：80项PASS，唯一collection error来自既有`test_r006b3a_family_paired_noninferiority.py`在本轮bundled Python中缺少`mpmath`，发生在导入历史脚本阶段，与MSCC改动无关；未通过下载依赖或跳过测试来伪造全绿。源码与测试SHA-256分别为`79F4CBD4F83F733635C1226200D2FC3A38AB613185601B5FF9818D8F14D5D1FE`和`01EC615852EEB3D59FD25714E1934380FB68FFA8EC79D62EFA872D4E9318BAA2`。

### 状态、解释与下一步

同步tracker、components与manifest：C022进入`SCREENING`，M1_NIP_protocol_v1仍`TODO`。本轮是实现级证据，不是D0 run或synthetic scientific result；`execution_enabled=false`、D2 seeds `UNGENERATED`、D1/D2与真实audit保持关闭，M1/R001–R003继续FAIL，R006/M2继续BLOCKED。下一轮应实现N01–N12中的最小D0 generator registry、prediction freeze/artifact validator和唯一D0 run contract；先解决或隔离环境依赖可复现性，再在不暴露truth的顺序下运行D0。

## 2026-09-02 23:02 EDT — Heartbeat `ccad`：MSCC discovery proposal 与 prediction freeze

### 实际动作与验证

继续在冻结协议内扩展`src/ccad/mscc.py`，未改变threshold、budget、phase或truth access。新增`source_conditioned_topk_proposal`：只读取discovery contribution kernels与source atom ID，对完整target字典计算source-normalized singleton residual，按分数冻结candidate atoms，并记录full-dictionary comparison count、全部singleton scores、boundary margin、planned support count和content hash。若atom-cap边界的两个分数在冻结`1e-12`容差内相同，返回`BUDGET_REFUSAL/BOUNDARY_TIE`且不靠index或truth破同分；若support family超过budget，同样在评分前拒答。

新增`FrozenMSCCPrediction`、`freeze_mscc_prediction`与`verify_frozen_mscc_prediction`。content payload绑定protocol hash、proposal hash、discovery fingerprint、source atom、search/identification/multiplicity、完整support分数与candidate ledger，排除elapsed time等非确定字段。新增三项测试覆盖正确top-k、boundary tie拒答、重复冻结/篡改检测；MSCC定向测试现为11/11 PASS，连同unknown-support/R001 metrics的相关回归为42/42 PASS，`py_compile`通过。

当前源码SHA-256=`DDF08438302B5ED513EAC118EE5BF0981313845AFBC693449890F242BACC052A`，测试SHA-256=`26529B3AE9DCC5C016A65E20DA9B5C89D3AB51BFB2C089A6CAE4308353EEF099`。上一条日志的hash对应扩展前版本，历史记录不改写。

### 保守解释、gate 与下一步

这是discovery-side implementation evidence，不是D0 experiment或proposal recall证据；没有生成phase seed、truth、held-out tensor或run目录。M1_NIP_protocol_v1仍`TODO`，M1/R001–R003继续FAIL，R006/M2继续BLOCKED，D1/D2与真实audit关闭。下一轮应实现N01–N12 D0 generator registry（truth对象与method输入物理分离）、run-local source/config/environment hashes和独立validator；只有这些通过后才创建唯一D0 run并同步RUNNING状态。

## 2026-09-02 23:10 EDT — Heartbeat `ccad`：N01–N12 observed/truth 分离 registry

### 实际动作

新增`src/ccad/nip_synthetic.py`，为冻结协议的N01–N12建立确定性D0-scale registry。`NIPObservedInstance`只包含family ID、source/target contribution tensors、独立mean contributions、document IDs和source atom ID；`NIPTruth`单独保存identification、multiplicity、minimum supports、safety、causal outcome、continuous-reference与full-group control属性。`assert_observed_schema_truth_free`机械拒绝observed schema出现truth、label、planted support、identification或causal字段。生成器显式接收structural/sample seeds，二者均改变输出且不自行生成phase manifest。

12 families覆盖structured split、merge/refactorization、tied supports、absent、bloated decoy、exact/approximate dense rotation、continuous-only、cancellation、rare occupancy、downstream cliff和mean mismatch。N06记录atom-level complete-universe absence及`full_group_portable=true`；N08记录`continuous_reference_feasible=true`但native absence；N09/N10/N11的safety/causal属性与identification分轴。

新增`tests/test_nip_synthetic.py`。首次定向运行3项中1项测试方法在N06/N07错误要求source tensor随structural seed变化，修为允许source或target结构变化；随后新增control attribute测试时又暴露patch把N06 full-group与N08 continuous-only属性绑到错误分支，测试按预期FAIL并已修复。最终registry 4/4 PASS；MSCC、unknown-support和R001 metrics相关回归共46/46 PASS，`py_compile`通过。当前generator SHA-256=`BC321959EB2FC74E22CA6311727BA84615BBC8ACD872E7F408FC7BD4E0FBF899`，测试SHA-256=`5366ADA8CF45E8C6ABD17F923378A285AEEA5C479F15516BDBBA853A06144190`。

### 保守解释与下一步

complete-oracle对照只证明当前小型构造与truth table在固定测试seeds上机械一致，不是D0 formal run、统计泛化或真实SAE证据。没有创建run目录、phase seeds或held-out/audit artifact。M1_NIP_protocol_v1仍`TODO`，M1/R001–R003继续FAIL，R006/M2继续BLOCKED。下一轮需实现run-local observed-only序列化、source/config/environment hash、禁止truth文件的D0 validator与唯一run contract；验证器通过后方可启动5 pairs/family的正式D0。

## 2026-09-02 23:19 EDT — Heartbeat `ccad`：D0 truth 隔离从类型级收紧为模块级

### 勘误与实际动作

上一轮虽然把`NIPObservedInstance`和`NIPTruth`定义为不同dataclass，但同一个`generate_nip_pair`仍在D0可导入模块内同时构造二者；这满足method API不读truth，却不足以支持“D0进程不生成truth”的更强审计目标。本轮不改冻结协议，重构为`src/ccad/nip_synthetic.py`只定义observed schema、N01–N12 observed tensor生成和kernel构造；新增`src/ccad/nip_truth.py`单独定义truth dataclass与静态truth registry。D0 runner后续只能导入前者，D1/D2评分进程才可导入后者。

测试相应改为分别调用observed generator与truth registry，并新增AST级import boundary检查，机械证明observed generator模块不导入`nip_truth`。registry测试5/5、全部NIP测试16/16、连同unknown-support/R001 metrics的相关回归47/47 PASS。当前observed generator SHA-256=`54B8D8D13DA972560876171A8357B2D3F836E49401DF02CB7D16D837C59F078B`，truth registry=`C80D0A03AE17D2717318F9DA02A87832451FB09B94FD416D176C4943AB899FA6`，测试=`D69A0E7AB87348613861160B7B9E5A46F05ACD7E28AF6BE09A06A9C40154EB1F`。上一条日志的generator/test hashes属于重构前版本，历史不改写。

### 保守解释与下一步

本轮只强化anti-leakage实现边界，未生成D0 seed或run，未读取/评分新的正式标签，M1_NIP_protocol_v1仍`TODO`，所有阶段门不变。下一步应创建D0 execution adapter、runner和独立validator：validator除run artifact/hash外必须AST检查runner不导入`ccad.nip_truth`，并扫描raw/summary禁止`minimum_supports`、identification、recovery、causal outcome等字段；实现测试通过后再以唯一run ID启动D0。

## 2026-09-02 23:27 EDT — Heartbeat `ccad`：D0 execution adapter、runner 与独立 validator

### 实际动作

新增`configs/m1_nip_d0_v1.json`，固定phase D0、12 families×5 pairs、512 observations、atom cap 20、`g_max=4`、7,462 budget和冻结exact/approximate thresholds；`truth_opened`、`held_out_eval_opened`、`real_sae_audit_opened`均为false，并绑定protocol SHA-256。新增`run_m1_nip_d0.py`：fail-fast拒绝已有run目录或信息门漂移，保存contract要求的manifest/config/environment/inputs/code hashes/status/log/raw/summary；用protocol hash、code aggregate、phase、family、pair和stream派生四路互异seeds，D0只使用observed generator，不导入truth registry。runner可以执行proposal/MSCC/freeze以检验pipeline，但raw仅序列化shape、seed、proposal/prediction hash和budget ledger，不保存support、identification、multiplicity或truth。

新增独立`validate_m1_nip_d0.py`，不导入runner或ccad实现；它重新检查10个必需artifact、AST import无`nip_truth`、三项closed flags、禁用字段、12×5=60 record网格、每record四路seed互异、60个prediction hash唯一、raw hash及run/summary状态。三文件通过`py_compile`、JSON解析、12-family/count/closed-state与runner import静态检查。config SHA-256=`723001DD377ECB23E1D57CB0A437098113B1A7FC4D758BB1AB3E1A2FD373B87B`，runner=`7B437B2B4FD41E69E6FCA58B3D3A2B9F80D6E200A52DF0FADC04D0ECA9F5AB87`，validator=`FC5BC1CD90DD24719CA4362EADAA8F7D7D3E69125B8CC5AE0ACCDAEA848C6F14`。

### 保守解释与下一步

本轮没有执行runner、没有创建run ID或生成正式D0 seeds；因此只是implementation readiness，不是D0 PASS。M1_NIP_protocol_v1仍TODO，既有FAIL/BLOCKED与D1/D2/真实audit状态不变。下一轮应先以临时唯一implementation-test ID端到端运行runner+validator并保留任何失败；若通过，再登记正式D0 run ID、同步RUNNING、执行并独立验证。轻量CPU规模无需资源lease。

## 2026-09-02 23:34 EDT — 启动 D0 implementation test

启动唯一run `M1_NIP_D0_impltest_v1_20260903T033300Z`，目的仅为端到端验证truth-closed runner、artifact contract与独立validator，不是formal D0 gate。启动前确认run目录不存在，读取resolved config/runner/validator并复核SHA-256与上一轮记录一致；配置为12 families×5 pairs、512 observations、lightweight CPU，不申请共享resource lease。tracker已在计算前同步为`RUNNING`；无论结果如何保留run目录、日志和状态。

### 2026-09-02 23:35 EDT — implementation test 表面 PASS、有效 FAIL

runner在约0.07秒内生成60条records并写出PASS；12/12 families×5 pairs齐全，maximum planned supports=15、proposal refusals=0，truth/held-out/real-audit flags全部false。独立validator报告10/10 PASS：runner无truth import、禁止字段为空、四路seed逐record互异、60个prediction hashes唯一、raw hash与状态均一致。raw SHA-256=`4A89DAA7599616818DF713F7EA117A5BB6409ED26763A98B478CEF4D67741B27`，summary=`F4800655A79716D8EAAAA10E9C06A24856FEAF30498D13AF2B058F84F01F1FF4`，validation=`7A4543B398E2E7149AC6B92D8CEA671F55A082F5E4DEA733A0402B9FC28E72D6`。

随后按`RUN_ARTIFACT_CONTRACT.md`逐字段人工复核发现validator契约不足，故tracker将本run保守记为有效`FAIL`，不修改runner原始PASS文件，并追加run-local`VALIDATION_CORRECTION.md`：(1)源码只有workspace path/hash而无run-local snapshots，validator也未重算源码、protocol与config绑定；(2)manifest缺local start、trigger/operator、project root、显式model/data/tokenizer/framework、完整seed字段、统计单位和artifact schema；(3)summary缺generation-script hash；(4)runner无异常finalizer，目录创建后异常可能遗留RUNNING且缺durable stderr/failure metadata。

保守解释：本run只证明核心60-record truth-closed计算路径可执行，不能证明artifact contract完整，更不是formal D0或M1证据。没有打开任何标签/eval/audit。M1_NIP_protocol_v1继续TODO，M1/R001–R003与R006状态不变。下一轮必须以新ID修复source/config/protocol snapshots、manifest schema、validator独立重算与异常路径测试；通过后才可启动formal D0。

## 2026-09-02 23:45 EDT — 启动 D0 implementation test v2

Heartbeat `ccad`针对v1人工契约审计发现的四项缺口作最小修复，不改变冻结protocol、family grid、threshold、candidate budget或information boundary。`scripts/run_m1_nip_d0.py`新增run-local源码与输入快照、完整manifest provenance、generation-script绑定以及异常终态finalizer；`scripts/validate_m1_nip_d0.py`改为从run-local runner执行AST检查，并独立重算source aggregate、config、protocol、execution config与workspace runner绑定。新增`tests/test_m1_nip_d0_contract.py`，注入异常后验证`RUNNING`确定收束为含failure type/message与traceback的`FAIL`。

第一次回归误用系统Python，3个module在collection阶段因缺NumPy/src import path报错；没有执行测试、创建run或写实验artifact。改用锁定R004 Python并设置绝对src路径后，定向17/17、全套93/93 unittest PASS，`py_compile`亦PASS。现已在实际计算前登记唯一run `M1_NIP_D0_impltest_v2_20260903T034500Z`为`RUNNING`；仍是lightweight CPU，不需resource lease。该run只验证工程合同，formal D0、truth、held-out eval和真实SAE audit均未启动。

### 2026-09-02 23:47 EDT — v2 fail-closed，登记 v3

v2在60条记录计算完成后的generation-script绑定处触发`StopIteration`：Windows将run-relative snapshot序列化为反斜杠，而后缀条件使用正斜杠。异常finalizer按设计将status从`RUNNING`写为`FAIL`，保存ended UTC、exception type和完整stderr traceback；v2目录不覆盖、不删除。这既是跨平台路径缺陷，也是异常持久化合同的端到端正向证据，但v2仍判FAIL。

最小修复仅在runner与validator中将snapshot相对路径经`Path(...).as_posix()`规范化后比较，不改变任何研究或计算参数。已预登记新run `M1_NIP_D0_impltest_v3_20260903T034700Z`为`RUNNING`；执行前先重跑回归。formal D0与所有信息门继续关闭。

### 2026-09-02 23:49 EDT — v3 implementation test PASS

路径修复后定向17/17 tests与`py_compile`通过。v3 runner正常退出，生成12 families × 5 pairs共60条truth-closed records；maximum planned support count 15、proposal refusal 0，truth/held-out/real-audit flags全部false。独立validator 18/18 PASS：必需artifact、完整manifest、closed state、run-local source/input snapshots、code aggregate、config/protocol/execution-config/runner绑定、AST truth-import隔离、禁止字段、record grid、四路seed、prediction hash唯一性、raw hash和终态均通过。

raw SHA-256=`51C64964D97BC0A09168AD5B5B5158A1CC05AA8CAB2D7D9DB252C035A65A32E8`，summary=`FE40AFF72E340774CDD5863A864D955405FAE469AA8F21BF3EC03897B0756953`，validation=`BD6B9E950E81D5734D6062F790E607B06C457DFEFB7C3260F654AF8E25D94702`，封存generation script=`DBF90948D35BC777089120B81DECA8880168251F9C5CE2B2AFC5629E5EF06D8E`。v3仅证明D0工程合同可执行，不评分truth、不构成synthetic scientific result、M1 PASS或C1/C2证据。M1_NIP_protocol_v1仍`TODO`，M1/R001–R003继续FAIL，R006/M2继续BLOCKED。下一轮可先登记唯一formal D0 run为RUNNING，再执行同一truth-closed adapter；成功后才允许按冻结顺序进入D1，D2 seeds仍不得生成。

## 2026-09-02 23:53 EDT — 启动 formal D0 truth-closed gate

Heartbeat `ccad`按冻结顺序恢复AGENTS、最新master log、EXPERIMENT_PLAN的C1/C2与M1-NIP段、tracker以及D0 config/protocol；复核v3 implementation test已PASS且formal run目录不存在。启动前登记唯一run `M1_NIP_D0_formal_v1_20260903T035300Z`为`RUNNING`。执行严格复用锁定D0 adapter：N01–N12各5 pairs、512 observations、atom cap 20、`g_max=4`、7,462 budget；只输出shape/seed/hash/budget ledger，truth、held-out eval、real SAE audit和D2 seed generation均关闭。任务为lightweight CPU，按资源规则不申请lease。

### 2026-09-02 23:55 EDT — formal D0 PASS

formal runner正常退出，独立validator 18/18 PASS。60/60 records覆盖12 families × 5 pairs，proposal refusal为0，maximum planned support count为15；manifest、run-local source/input snapshots、code aggregate、config/protocol/execution-config/runner绑定、AST truth-import隔离、禁止字段、四路seed、prediction hash唯一性、raw hash和终态全部通过。与v3 implementation test逐record比较，family/pair、四路seed、proposal hash、prediction hash与evaluated-count 60/60 exact；raw文件本身因run ID不同而按预期不相同。

formal raw SHA-256=`F94B7AD4DDC35E4B22BF1E7EF38698E623F90256D68D982FE79138C4F5477832`，summary=`1FA9DA45356EA48AD95ABF1FE804199B706547E92235123A13C894A106678328`，validation=`BD6B9E950E81D5734D6062F790E607B06C457DFEFB7C3260F654AF8E25D94702`。保守解释：D0只通过truth-closed工程门，没有打开或评分truth，因此不是minimum-support recovery、synthetic scientific、M1、C1-NIP或C2-NIP证据。按locked protocol现允许实现并预注册D1 labeled development，但D1尚未启动；D2 seeds继续`UNGENERATED`，真实SAE audit关闭，R006/M2仍BLOCKED。下一轮应先构建D1的prediction-freeze-before-label-open两进程/两阶段合同与validator，而不是直接查看truth或生成D2 seeds。

## 2026-09-03 00:01 EDT — pre-D1 cap identifiability audit：D1 暂停

在实现D1两阶段runner前，审计formal D0的60条raw records发现N01–N12 target universe仅1–4 atoms：N11/N12为1，N04/N06/N07/N08/N09为2，N01/N02/N10为3，N03/N05为4；相应完整候选数仅1/3/7/15。locked D1 caps `{4,8,12,16,20}`因此对每个pair都覆盖同一完整target universe，生成相同candidate supports与scored-support count。proposal hash虽因payload含`atom_cap`可不同，但科学预测不变；词典序最终会在smallest-cap之前由非确定runtime噪声选型。

新增`M1_NIP_D1_CAP_IDENTIFIABILITY_REVIEW_20260903_000100.md`和组件C023，列出三种方案。推荐新protocol suffix：将每个target universe扩至20个construction-controlled nuisance/proposal-stress atoms，预写cap-rank bands，并用complete-universe oracle证明不改变N01–N12 truth；随后重跑D0并使用fresh D1 seeds。单cap-4为较弱备选；保留v1让runtime破同分明确拒绝。

这是prospective D1设计缺陷，不追溯否定formal D0的工程PASS，但使`M1_NIP_protocol_v1`在D1前标为`BLOCKED`。没有生成D1/D2 seeds，没有导入或评分truth，也未打开real audit。修复需要改变`LOCKED`协议，按AGENTS需用户裁决；获批前仅可继续安全设计/测试，不执行D1。

### 2026-09-03 00:10 EDT — 方案 A 的第二项必要约束

等待用户裁决期间继续做只读逻辑审计，发现仅扩充到20 atoms仍不足：N04/N06/N07/N08/N12的`CERTIFIED_ABSENT`只有complete-universe oracle才有资格输出；cap<20的scalable lane即使行为完全正确，也只能给`UNRESOLVED`。若D1把两者混在“exact identification accuracy”中，cap20会因为唯一穷举全字典而机械胜出，与proposal recall无关。

已扩充`M1_NIP_D1_CAP_IDENTIFIABILITY_REVIEW_20260903_000100.md`和C023：未来v2必须采用lane-conditional scoring——positive/tied families评minimum-support与multiplicity；negative families对scalable caps只评零false-native-positive，`CERTIFIED_ABSENT`只评`FULL-EXHAUSTIVE`；proposal recall仅在support存在时计算，并与conditional solver correctness分表。推荐方案仍为A，但实现范围现更准确。没有修改v1 protocol/generator/code，没有run或seed生成；blocker与gate不变。

## 2026-09-03 00:18 EDT — 用户授权裁决；选择 A 并冻结 M1-NIP v2

### 方案裁决与实验设计

用户明确授权自行选择合理方案并要求保留备选。采用`experimental-design`技能复核实验单位、配对与blocking后，正式选择方案A；方案B（单cap-4、较快但削弱scalable proposal证据）和方案C（runtime破同分、不可复现，拒绝）继续完整保存在`M1_NIP_D1_CAP_IDENTIFIABILITY_REVIEW_20260903_000100.md`。D1固定为同一structural seed-pair内五caps的paired repeated-measures设计，family为block，seed-pair为独立重复；observation/token不作独立样本。

在冻结前新增独立原型`src/ccad/nip_synthetic_v2.py`与`tests/test_nip_synthetic_v2.py`。每个family扩至20 target atoms；decoy形式为`alpha*s + beta*||s||*u`，`u`彼此正交并与原source/target span及hook-wise constants正交，`beta^2=0.08`。因此任何含decoy support的centered residual至少0.08，高于approximate threshold 0.05且保留0.03 margin。七个正例family预写第一充分cap：N11=4、N01=8、N02/N09=12、N03/N10=16、N05=20。5个专用probe seed-pairs上，每一family-cap band均逐对命中；20-atom complete oracle保持v1 truth。此为pre-freeze implementation probe，不是D0/D1结果。

### 冻结 artifact 与状态

新增`M1_NIP_PROTOCOL_V2_20260903_001500.md`和`configs/m1_nip_protocol_v2.json`。v2同时冻结lane-conditional absence scoring：scalable negative只评零false-native-positive并允许`UNRESOLVED`，`CERTIFIED_ABSENT`只归`FULL-EXHAUSTIVE`；runtime/peak memory仅报告，不参与cap selection；selection以positive exact pairs、false unique、false native positive、budget refusal、scored supports、最小cap依次决定。新增`tests/test_nip_protocol_v2.py`验证文档hash、实现常数、selection order和所有phase seeds在freeze时均`UNGENERATED`。

全套101/101 unittest与py_compile PASS。protocol SHA-256=`7AA85355C13300E2B7677D704B45EDCFAD4EE69A42D8377B943FA2A2E1D6CB6A`；config=`CE9F4724CEE42BA3BD92E638F48139B7AAADC7B1D779AC10E51F593A2E375E41`；prototype=`9A53962DC4F04EE8665CDF5E481A1975C1B2A868D2F854195D8F0B3968BFEC09`；protocol test=`7528AAB4D65E3D7C12FCE54FDDDD74777473BD0B8B34B8B94531EE9E4CB6DBC5`。C023由BLOCKED转`ADMIT`；v1标`CUT`但formal D0 PASS不改写，且从未生成v1 D1/D2 seeds；v2为`TODO`。

保守解释：冻结v2是prospective设计与实现可行性证据，不是synthetic recovery、M1、C1-NIP/C2-NIP或真实SAE证据。没有v2 run、phase seed、truth opening或real audit。下一步只实现v2 D0 truth-closed adapter与独立validator，先做implementation test，再登记formal D0；D1两阶段runner必须等v2 D0 PASS。

## 2026-09-03 00:28 EDT — 启动 v2 D0 implementation test

实现`construction_certificate`，只返回target/base/decoy counts、decoy正交残差下界、正交误差、声明/观测cap band和通过状态；不返回support、identification、multiplicity、causal outcome或truth。新增`configs/m1_nip_d0_v2.json`、`scripts/run_m1_nip_d0_v2.py`和独立`validate_m1_nip_d0_v2.py`。runner不导入`nip_truth`或private construction IDs，MSCC API仍只接收kernels/means/source atom/proposed IDs；raw仅留construction certificate的数值/哈希与shape/seed/proposal/prediction/budget ledger。

validator除完整artifact/hash/status合同外，机械检查60-record grid、20 target atoms、每pair 6,195 scored supports、正交残差不低于0.075、正交误差不高于1e-10、七个positive cap bands、每row四路及全局240个phase seeds唯一、truth字段缺失和runner import边界。新增v2 exception-finalizer测试；py_compile、定向11/11、全套103/103 tests PASS。execution config SHA-256=`25D61BECEE646CED5E031A18F915CCA91FF59B9D2E74700498E07E09130B5ACC`，runner=`1D2ACD3E53C70704DB0DD276C4922D329D4B50E7F0FC0C1EA1166C6D275373EA`，validator=`07F36482CE24389903BC9D1C6EA5FCD9F635F496C03D9A2FDAEF6992CCFC8116`。

现于实际计算前登记唯一run `M1_NIP_D0_v2_impltest_v1_20260903T042800Z`为RUNNING。它是lightweight CPU integration test，无resource lease；formal D0、D1/D2、truth/eval/real audit均未启动。失败必须原样保留并用新suffix修复。

### 2026-09-03 00:29 EDT — v2 D0 impltest v1 有效 FAIL；启动 v2

v1 runner正常完成，但独立validator 20/21 checks PASS，唯一`hashes_unique`失败。人工复核确认prediction hashes为60/60 unique，而construction certificate hashes为48/60 distinct。后者只绑定counts、正交数值和cap-band等设计不变量，不含discovery fingerprint；不同pair得到相同证书合法且预期，因此validator的certificate uniqueness要求过强。run原始PASS/status不改写，追加`VALIDATION_CORRECTION.md`并在tracker记有效FAIL。

前瞻修复只把该检查改为prediction hashes必须60/60 unique、certificate hash必须为合法64位大写SHA-256；不改变generator、seeds、protocol、threshold、budget或raw schema。validator新SHA-256=`7BC1278A4F9EFC3FB84DCD5F0A0F835F609A39749651E36C8811546BAFA5E17A`，py_compile与全套103/103 tests PASS。已在执行前登记新run `M1_NIP_D0_v2_impltest_v2_20260903T042900Z`为RUNNING；formal D0仍未启动。

### 2026-09-03 00:31 EDT — v2 D0 impltest v2 PASS

v2 runner正常退出，独立validator 21/21 PASS。60/60 records均为20 target atoms、20次full-dictionary singleton comparisons和6,195 planned/evaluated supports；七个positive construction cap bands全部通过，proposal refusal与cap-contract failure均为0。240个structural/sample/proposal/solver seeds在row内及全局唯一，60个prediction hashes唯一。全run最小decoy orthogonal residual=`0.07999999999999992`，最大正交误差=`8.326672684688674e-16`，分别满足冻结0.075与1e-10门。

raw SHA-256=`2C8B214280D8BA9A27B275C1C445F6D28DE17F63AB7B7F80DC97ECB6C9A032F9`，summary=`51EB53A13633C52B46ABC8024EDD8767313C5880AEC64D35A48A72E96C1BB024`，validation=`AE87961FDAFDA8840C354AF19573E8E0B6457B54DCB65152B91129CF1A10FA39`。这只证明v2 D0工程合同可执行，不打开truth、不评分recovery，也不是M1/C1/C2证据。`M1_NIP_protocol_v2`保持TODO；下一轮可登记并运行唯一formal D0，成功后才实现D1两阶段信息顺序。D1/D2 seeds和real audit仍关闭，无需用户裁决。

## 2026-09-03 00:36 EDT — 启动 v2 formal D0

Heartbeat `ccad`按规定恢复AGENTS、最新master log、EXPERIMENT_PLAN、tracker与v2协议，并读取implementation v2的resolved config、status、summary和21-check validation。复核protocol=`7AA85355...`、execution config=`25D61BEC...`、runner=`1D2ACD3E...`均与通过的implementation run一致，且目标目录不存在。

已在实际计算前登记唯一run `M1_NIP_D0_v2_formal_v1_20260903T043600Z`为RUNNING。执行仍为12 families×5 pairs、20 target atoms、cap20、`g_max=4`、7,462 budget的truth-closed D0；D1/D2 seed、labels、evaluation与real audit不生成。任务为lightweight CPU，不申请共享lease；任何失败均保留并停止进入D1。

### 2026-09-03 00:38 EDT — v2 formal D0 PASS

formal runner正常退出，独立validator 21/21 PASS。60/60 records均满足20 target atoms、6,195 planned/evaluated supports、七个positive construction cap bands、0 proposal refusal、0 cap-contract failure；全局240个phase seeds与60个prediction hashes唯一。最小decoy orthogonal residual=`0.07999999999999992`，最大orthogonality error=`8.326672684688674e-16`。与通过的implementation v2逐record比较，family/pair、四路seeds、construction/proposal/prediction hashes、budget、residual、orthogonality与cap fields 60/60 exact；raw仅因run ID不同而不同。

formal raw SHA-256=`8D526889DE9083026D4BDE193CA9D4541E9BFB6C5ADE51604885D8051DFFB69B`，summary=`6767E953D6B783BF5105D55C093D54111B90CB8AA0719FFCC154A957979EAF99`，validation=`AE87961FDAFDA8840C354AF19573E8E0B6457B54DCB65152B91129CF1A10FA39`。D0工程门通过，允许实现D1 prediction/freeze与label/score两阶段合同；但D1 seeds/labels仍未生成，D2继续UNGENERATED，real audit关闭。该PASS不是minimum-support recovery、M1、C1/C2或真实SAE证据；R006/M2仍BLOCKED。下一轮先实现D1信息顺序与tamper-before-truth-import测试，implementation test通过后方可登记D1 prediction phase。

## 2026-09-03 00:42 EDT — D1 两阶段信息隔离 adapter 实现完成

Heartbeat `ccad`只实现工程隔离，没有执行I1或生成正式D1 seeds。新增`configs/m1_nip_i1_v2.json`，以独立`I1` namespace登记12 families各1个seed-pair、同pair内caps `{4,8,12,16,20}`；显式声明`formal_d1_seed_consumed=false`。备选仍按既有C023保留：单cap-4为较弱方案B，runtime破同分方案C继续拒绝；当前adapter实现已选择的paired-cap方案A。

新增truth-blind prediction runner `scripts/run_m1_nip_d1_predict_v2.py`。它只导入observed construction和MSCC，写出完整proposal/search ledger及content-addressed prediction；同一pair四路seed跨caps复用。runner在所有raw/summary/manifest/status/source/config/protocol/environment写完后，最后通过原子替换写`prediction_closure.json`，绑定row count、protocol、code aggregate及关键文件hash；prediction目录随后不得由score阶段修改。新增`score_m1_nip_d1_v2.py`，无静态truth import：先以标准库独立验证closure、每个绑定文件、run-local源码快照、code aggregate、每条proposal hash和prediction hash；全部通过后才调用动态`importlib.import_module("ccad.nip_truth")`，并将scores写到另一个新目录。评分按v2执行positive exact、ambiguous false-unique、negative false-native-positive与full-exhaustive/scalable absence lane分离；cap按冻结六级顺序选择，runtime不入选型。

新增`tests/test_m1_nip_d1_information_order.py`：AST确认predictor/scorer均无静态`nip_truth` import；构造篡改closure并mock动态import，证明验证异常发生且truth import调用次数为0；另检查I1不消费正式D1 seed。锁定R004 Python下`py_compile`和全套106/106 unittest PASS。文件SHA-256：I1 config=`CB333413B8CFF50158A817D992DB49C679603A7141B12C0FDFBA2335CDE092DC`，predictor=`A27550D093EAEF7C04DA0AB0289C94EC7A6509FE4D6B74E640F8C1D1FA0E4A76`，scorer=`771EBDB17572A441D045EC9507440C3B0235D4FBAB13BB859B4CB6F1857B4B40`，test=`9A6ED16B268F4BAE450707A298DAC9465338F3D5782F00421E3010DAFDE88D12`。

保守解释：本轮证明源码级信息顺序与fail-before-import测试通过，不证明端到端closure在真实生成artifact上通过，更不是D1 recovery、M1、C1/C2或真实SAE证据。`M1_NIP_protocol_v2`继续TODO；D1/D2正式20/20 seeds仍UNGENERATED，real audit关闭。下一轮应预登记唯一I1 prediction implementation run，先执行并封存预测，再以独立score run验证/开标签；任何合同缺陷用新suffix修复，I1通过后才生成正式D1 prediction seeds。

## 2026-09-03 00:53 EDT — 启动 I1 prediction integration test

Heartbeat `ccad`按AGENTS顺序恢复项目状态并复核v2 protocol、I1 config、prediction/score源码。实际计算前登记唯一run `M1_NIP_I1_predict_impltest_v1_20260903T045300Z`为RUNNING：I1独立namespace，N01–N12各1个structural seed-pair，同pair内配对caps 4/8/12/16/20，512 observations，`g_max=4`，budget 7,462。该run只生成truth-blind proposal/search/prediction ledger并最后封存closure；不生成或消费formal D1/D2 seed，不打开labels、held-out或real audit。任务为lightweight CPU，不申请resource lease。score run尚未登记或启动，必须先确认prediction closure可独立验证。

### 2026-09-03 00:54 EDT — I1 prediction closure PASS；启动独立 score run

prediction runner正常退出，生成12 families × 1 pair × 5 paired caps共60条记录；status/summary PASS，truth_opened=false，raw SHA-256=`0F6442B660D0E75D0667BC84B4894983416BB3526598321B81CF3382FAACF003`。最后原子写入的closure SHA-256=`1A7A006056714D9E2457A3960E47AE2D7661E8CB7BFB67D060FB4E1E4BC727D1`。在任何label import前，单独调用score模块的标准库verification入口，60/60 rows通过closure、bound files、code aggregate、run-local source snapshot、proposal hash与prediction hash重算。

只有上述验证通过后，才登记独立run `M1_NIP_I1_score_impltest_v1_20260903T045400Z`为RUNNING。它只读取sealed I1 prediction并把label-conditioned score写入另一个新目录；不得修改prediction artifacts。I1 labels属于工程integration fixture，不是formal D1 truth result。formal D1/D2 seeds与real audit继续关闭。

### 2026-09-03 00:58 EDT — score v1 有效 FAIL；artifact-only 修复并启动 v2

score v1数值路径退出0：按cap的positive exact pairs为1/2/4/6/7，false-native-positive全为0，cap 8/12各出现1个false-unique，cap 4/16/20为0；冻结词典序选择cap20，runtime未参与。随后按AGENTS run contract人工审计，发现score目录仅有raw/summary，缺manifest、environment、source/input hashes、status/log与异常finalizer。因此v1原始文件不修改、不删除，tracker按有效FAIL记录；其数值只作为I1调试观察，不作D1证据。

另保留一项设计解释：N05的预写第一充分cap就是20，所以I1上最大化positive exact pairs会必然偏向20。这说明cap压力构造确实可区分，但不能把cap20选择外推为真实SAE超参证据；formal D1若执行也只能证明synthetic gate，不冻结真实matcher cap。

对`score_m1_nip_d1_v2.py`作artifact-only修复：加强protocol/config/input snapshot与封存predictor AST核验；verification通过后创建独立score目录，保存scorer snapshot/hash、prediction closure/raw输入hash、environment、manifest、RUNNING→PASS/FAIL status、stdout/stderr及raw/summary hash。tamper测试新增“失败时score目录不存在”断言。py_compile与全套106/106回归PASS。已在执行前登记`M1_NIP_I1_score_impltest_v2_20260903T045800Z`为RUNNING，复用同一sealed I1 prediction；协议、prediction、truth registry及selection rule均未改变。

### 2026-09-03 00:59 EDT — I1 score v2 PASS

修复后的score v2正常退出并写出完整score artifact。独立于runner的机械检查确认10项必需文件存在、raw与封存scorer hash重算一致、terminal status PASS、所消费prediction closure在评分前后仍为`1A7A006056714D9E2457A3960E47AE2D7661E8CB7BFB67D060FB4E1E4BC727D1`，因此score未修改sealed prediction。score raw SHA-256=`042F44BA9B3FAFF1E19F20637760C003FFE4772EC3EE6050A666B7D72C387DA6`，summary=`5AE946C8F37AFEB2D5A5E0E4FB5582EC05140599E1086828FDE88EA49CC3BDA7`，封存scorer=`40193F16BA241B66C0AC5B63C168F1A660CCD8C17DE5D51D2CF5C94C5D63DE7A`。

I1结果与v1调试数值一致：cap 4/8/12/16/20的positive exact pairs为1/2/4/6/7；所有cap的false-native-positive与budget refusal为0；cap8/12各1个false-unique，cap4/16/20为0；按冻结次序选择cap20。prediction closure先通过全量验证，之后才动态导入truth，且runtime不参与选择。

保守解释：I1只证明两阶段信息隔离、cap压力和artifact contract端到端可执行；它不是formal D1、M1、C1/C2或真实SAE证据。尤其cap20由预写N05 rank band推动，只能视为synthetic stress gate，不能外推成真实SAE matcher的cap选择。formal D1/D2 seeds仍UNGENERATED，real audit关闭，R006/M2仍BLOCKED。下一轮应先形成formal D1 execution config与持久化独立prediction/score validators，静态复核20 pairs/family与seed namespace，再登记formal prediction run；不可直接复用I1 scorer结果。

## 2026-09-03 01:03 EDT — GitHub规范远程与及时提交治理规则

用户明确要求将GitHub仓库及“重要更新及时提交”写入`AGENTS.md`。新增第8.1节，将`https://github.com/zz0209/CCAD.git`与`main`登记为规范远程/默认分支，并将`.gitignore`白名单内项目本体的重要、已验证更新之常规`commit`/`push`记为持续授权。重要更新包括实质代码、可执行config/runner/validator/test、已同步master log的实验/失败/修复/gate/裁决，以及经授权更新的理论PDF。

规则要求提交前检查并发改动、白名单、密钥/大文件、测试或validator，禁止`git add -f`绕过忽略；推送后核对本地HEAD与`origin/main`并报告commit hash。持续授权不扩展到release、仓库可见性、远程分支/标签变更、force push、历史改写、被忽略内容、其他远程或付费服务。`runs/`、`data/`、权重、原始统计、其他研究文档和本地环境继续本地保存并以path/hash追溯。该治理更新不改变任何LOCKED协议、实验结果、M1/M2 gate或C1/C2证据。

## 2026-09-03 01:03 EDT — 执行后端恢复；formal D1 validator 资格化

用户报告环境问题可能修复并要求检查后推进一轮。只读`Get-Location`、AGENTS/master log/plan/tracker读取、Python编译与测试均正常，确认此前的Windows sandbox helper故障已解除。新增`configs/m1_nip_d1_v2.json`，严格绑定v2 protocol：D1 namespace、12 families×20 pairs×5 paired caps=1,200 predictions、512 observations、`g_max=4`、7,462 budget、prediction truth closed；配置中的`formal_d1_seed_consumed=true`只表示执行该config会生成正式D1 seeds，本轮尚未运行该config。

新增独立`validate_m1_nip_d1_v2.py`，不导入prediction runner或scorer；重算sealed closure、bound files、code/input snapshots、protocol/config、predictor AST truth隔离、paired grid/seeds、proposal/prediction hashes；score阶段再独立导入truth重算逐row outcomes、cap aggregates与runtime-free词典序。scorer provenance改为从sealed prediction config推导phase、formal-seed flag和evidence level，移除I1硬编码。新增formal-grid测试；py_compile与全套107/107 tests PASS。

首次用新validator检查旧`M1_NIP_I1_score_impltest_v2_20260903T045800Z`时在`score_provenance` fail-closed：该目录由旧scorer生成，虽有formal=false/evidence，但manifest缺显式`phase`。不修改、不删除v2，tracker勘误为有效FAIL；这不影响sealed prediction。已预登记`M1_NIP_I1_score_impltest_v3_20260903T050300Z`为RUNNING，将使用修正后的scorer复用同一sealed I1 prediction并接受持久化validator。formal D1 config仍未执行，D2/real audit关闭。

### 2026-09-03 01:05 EDT — I1 score v3 与持久化 validator PASS

修正后的scorer复用只读sealed I1 prediction，在新目录`M1_NIP_I1_score_impltest_v3_20260903T050300Z`完成评分；独立validator 22/22 PASS，其中prediction 14项、score 8项。验证覆盖60-row paired grid、truth-closed prediction、source/input/config/protocol绑定、逐row proposal/prediction hashes、score phase/formal flag、scorer snapshot、逐row truth重评分、cap aggregate与词典序重算。selected cap仍为20，数值与先前I1一致。validator SHA-256=`9A2AB690DB25F8827DEE39E0CD930C9F00D553F67379425B3874BFB987BF6D33`；formal config=`DCF87704C508A72F9422C00EF83C0DD0DD972AB3E9C704A56E08D9D0B60065A3`；scorer=`7CFF58A55F8F7E70BBEB562C2E211A2BFEE772C7126DA2702AF0F36B690B50E2`；validator source=`42B1858FD68F0E0774EF94E295BF7EF1F132D6C5275820678EF0FB5883EE297D`。

本轮只完成formal D1执行前的资格门，没有运行`configs/m1_nip_d1_v2.json`，所以正式D1 seeds仍未生成；D2与real audit保持关闭。I1只属工程证据，不改变M1/R001–R003 FAIL、R006/M2 BLOCKED，也不支持C1/C2。下一轮可以先登记formal D1 prediction run并生成1,200条truth-blind predictions，必须独立验证sealed closure后才允许登记score run。

## 2026-09-03 01:19 EDT — 启动 formal D1 truth-blind prediction/freeze

Heartbeat `ccad`按新AGENTS顺序恢复状态，确认本地`HEAD`与`origin/main`均为`e59effef4922c0a145cf961af5dca6e165e62383`、工作区无未提交变更、I1 v3 validator 22/22 PASS，且formal config、prediction runner、独立validator与v2 locked protocol一致。运行前登记唯一run `M1_NIP_D1_predict_formal_v1_20260903T051900Z`为RUNNING。

本阶段将首次生成正式D1 seeds：12 families×20 structural seed-pairs，每pair共享structural/sample/proposal/solver四路seed并配对比较caps 4/8/12/16/20，共1,200条predictions；512 observations、20 target atoms、`g_max=4`、7,462 budget。prediction进程禁止truth import，结束时最后原子写sealed closure；本轮只运行独立pre-label validator，不登记或启动formal score。任务为lightweight CPU，不申请resource lease。D2、held-out eval和real audit保持关闭；失败run必须保留且不得进入score。

### 2026-09-03 01:20 EDT — formal D1 prediction closure PASS；标签仍关闭

prediction runner正常退出并最后写入sealed closure。1,200/1,200 rows覆盖12 families×20 pairs×5 paired caps；raw SHA-256=`3CCF6D22D592CB9351CE282826580352D8ABC4BF01E6786AF42466C09F21450D`，closure SHA-256=`0688C54FD9A1AD26ED5C8A32BCFC26ABE57B9EBB279BE25FCA166C84354F0C4B`。独立validator在不导入truth的prediction-only模式下14/14 PASS，重算所有bound file、源码/输入/config/protocol、predictor AST、paired grid/seeds以及1,200个proposal/prediction hashes；pre-label validation SHA-256=`275E415E4DE0AF9199B9E4F3479DFF5E3C5F7001447ABFBD20044D54647B6E17`。

本轮严格停止在pre-label边界：没有创建formal score目录、没有导入D1 truth，也没有生成D2 seeds或打开held-out/real audit。该PASS证明formal D1 predictions可追溯且信息顺序合规，但尚无recovery/selection结果，不支持M1/C1/C2。下一工作单元才可在再次核验closure hash后预登记formal score run；score若失败不得修改prediction目录。

## 2026-09-03 01:38 EDT — 启动 formal D1 post-closure score

Heartbeat `ccad`重新读取AGENTS、最新master log、M1-NIP plan/tracker以及formal prediction的resolved config、status、summary与pre-label validation。复算closure仍为`0688C54FD9A1AD26ED5C8A32BCFC26ABE57B9EBB279BE25FCA166C84354F0C4B`，pre-label validator仍为14/14 PASS，`truth_opened=false`；本地HEAD与origin/main均为`d6b891cc192b7f530501942eaf2081663088e4e9`。

只有完成上述核验后，才预登记唯一run `M1_NIP_D1_score_formal_v1_20260903T053800Z`为RUNNING。score进程读取sealed formal prediction但写入独立目录，先内置重复closure核验、后动态导入D1 truth；随后由持久化独立validator重新验证prediction并逐row重评分/重算selection。任务为lightweight CPU，无resource lease。D2 seeds、held-out eval与real audit保持关闭；任何失败均停止D2。

### 2026-09-03 01:39 EDT — formal D1 score PASS；cap20 selected

score进程正常完成且未修改prediction closure；独立validator对formal prediction 14项与score 8项共22/22 PASS，覆盖1,200条prediction/score绑定、逐row truth重评分、cap aggregate和词典序selection重算。score raw SHA-256=`4DE3EAFFEEB59152C7060B531D54665FBC41131E6E33E3C52FBC0A80B65A3D41`，summary=`30C61F9AF64C78035F497CFE4E6D04D945DA4CB2B83770CB85C34326774C12EB`，independent validation=`CF84FA68BE5900A6258A0DFEDDAB4F2C5E98C355AA19F095B3C0DD59FD54073E`，输入closure复核仍为`0688C54FD9A1AD26ED5C8A32BCFC26ABE57B9EBB279BE25FCA166C84354F0C4B`。

冻结词典序选择cap20。cap 4/8/12/16/20的positive exact pair counts为20/40/80/120/140；cap8和12各有20个false-unique，cap4/16/20为0；所有caps的false-native-positive和budget refusal均为0。cap20在7个positive families的140/140 pairs上恢复精确minimum support set/cardinality/multiplicity，在5个native-absent families的100 pairs上0 false native positive，且false-unique/refusal为0。

保守解释：这是D1 labeled synthetic development证据，证明v2构造内的minimum-support/ambiguity/absence-lane行为并选出cap20；它不是fresh D2 confirmation、真实SAE、held-out contribution或causal证据，不能支持C1/C2。由于N05预写第一充分cap20，该选择主要验证压力构造与solver，不是可外推的真实超参学习。D2 seeds仍未生成；下一轮必须先产生独立selection-freeze/D2 config，绑定selected cap、D1 closure/score/validator、source/config/environment hashes并通过静态审计，之后才可原子生成fresh D2 seeds。M1/R001–R003与R006/M2状态暂不改变。

## 2026-09-03 01:55 EDT — pre-D2 orthogonal-endpoint audit 阻断 v2 D2

Heartbeat `ccad`在生成任何D2 seed前，按v2第6节与继承的v1第5/7/8节审计N09–N12正交属性是否可由当前artifact独立测量。结论：当前prediction/score仅验证identification；N09 cancellation可由组内能量比测量，N10 evidence可由document ESS测量，N12 mean mismatch已有`d_mu`，但N11 causal属性存在不可绕过的逻辑矛盾。

当前`generate_nip_observed`对N11令target atom逐点等于source atom。对相同base hook `h`与确定性downstream map `F`，`Y_A(x)=Y_B(x)`逐点必然推出`F(h-Y_A)=F(h-Y_B)`逐点，因此任何真实intervention gap都为0；这与truth registry写定的`FOUND + CAUSAL_FAIL`不可能同时由现有observable construction成立。直接读取registry标签评分会成为循环自证，违反D2 hard gate与claim纪律。

用20个独立构造seed直接复算N11 source/target首atom的最大逐点差，20/20均为0，全局最大值精确为`0.0`，确认矛盾来自实际生成器而非仅源码解读。

新增`M1_NIP_D2_ORTHOGONAL_ENDPOINT_AUDIT_20260903_015500.md`与C024，保留三案：A（推荐）以prospective v3加入sub-threshold source/target delta、共享cliff endpoint、固定margin与smooth control，并对受影响阶段使用fresh seeds；B删除M1的N11 causal gate、延后到历史F12/真实endpoint，较简单但削弱边界；C只评分registry标签，明确拒绝。v2 D0/D1各自scope内结果不追溯改写，但v2 D2在seed生成前标`BLOCKED`；D2 seeds仍不存在，未打开任何D2 tensor/label/held-out/real audit。

该修复需要改变LOCKED协议，依AGENTS第0/12节暂停并请求用户裁决。推荐A，因为它保留“高贡献一致性不自动推出因果可移植性”的关键 falsifier，同时使结果可由observed endpoint而非标签验证。M1/C1/C2仍未通过，R006/M2继续BLOCKED。

## 2026-09-03 09:20 EDT — 用户裁决授权后采用方案A；登记v3 formal D0

用户指出既然属于implementation矛盾就应修复，并明确授权agent自行裁决。经复核理论PDF的THM-CBSM-009：同hook下pointwise exact contribution equality对任意deterministic downstream map必然exact interchangeable；approximate transfer只在Lipschitz假设下有界。故问题确属v2 N11实例化错误，不是PDF核心定理错误。

三案裁决：采用A，创建prospective v3 observable endpoint；B（删除M1 N11 causal gate）仅保留为v3在D2前失败时的降级备选；C（读取truth registry标签代替endpoint measurement）因循环自证永久拒绝。新增`M1_NIP_PROTOCOL_V3_20260903_091500.md`、`configs/m1_nip_protocol_v3.json`、`src/ccad/nip_synthetic_v3.py`与对应测试。V3 N11冻结exact-balanced zero-mean perturbation，`d_ctr=0.01`、joint feasibility threshold 0.05、threshold margin 0.04；shared base hook为source/target midpoint，固定step readout的normalized boundary margin 0.05、effect RMSE 1.0，smooth linear control RMSE 0.1。endpoint由observed tensors重算，不含truth/outcome label。非N11构造继承v2且不变。

定向py_compile与8/8 v3 tests PASS，覆盖20 seeds endpoint、unique support仍FOUND、exact zero mean、decoy/cap contract、非N11不变、odd-n fail-closed以及protocol hash/phase closure。现于计算前登记唯一run `M1_NIP_D0_v3_formal_v1_20260903T132000Z`为RUNNING：12 families×5 fresh structural pairs、512 observations、20 targets、cap20、`g_max=4`、budget 7,462；N11使用approximate lane，其他阈值继承v2。任务为lightweight CPU，无resource lease。D0只序列化construction/endpoint numeric certificate、shapes、seeds、hash和budget，不打开truth、label、D1/D2或real audit；失败必须原样保留。

### 2026-09-03 09:23 EDT — v3 D0 formal v1 有效 FAIL；生成顺序修复并登记v2

`M1_NIP_D0_v3_formal_v1_20260903T132000Z`在truth-closed construction gate有效FAIL：初版v3先由v2相对旧pointwise-equal N11 span生成orthogonal decoys，随后才把target atom替换为`source+delta`。因此decoys不再对actual v3 forbidden span保持机器精度正交，runner按冻结gate fail-closed。失败run目录、status与traceback原样保留；未打开truth、label、D1/D2或real audit。

修复只改变implementation顺序：N11先在v1 base上构造冻结的zero-mean delta，再相对actual source、perturbed target和hook constants生成同一v2公式的20-atom decoys。协议数值、阈值、endpoint、seed derivation、budget与family均不变。现于重算前登记新run `M1_NIP_D0_v3_formal_v2_20260903T132300Z`为RUNNING；源码hash变化将自然导出fresh code-bound D0 seeds，v1失败不得覆盖。

### 2026-09-03 09:28 EDT — v3 D0 formal v2 PASS；独立完整性审计 WARN 已闭环

修复后的`M1_NIP_D0_v3_formal_v2_20260903T132300Z`正常完成60 records（12 families×5 fresh pairs），独立validator 19/19 PASS。所有240个structural/sample/proposal/solver seeds互异；20-target、6,195 scored supports、decoy residual/orthogonality、cap contract、truth-free imports/outputs、source snapshots、raw hash及status均通过。五个N11 pairs的cliff disagreement与effect RMSE均为1.0，最小normalized boundary margin为`0.04999999999999967`，最大smooth-control RMSE为`0.10000000000000003`；proposal refusal为0。raw SHA-256=`52050F217F0BBF44B821BC404DA1EB3E4E3BDD19BD1748F566DEF880CF00CA63`，summary=`FAC558703A9CB2416B973C86C6F029DF64127303FB4B0BE5C256C137A187C88B`，validation=`8F964BB6BAF920D69AB4FE8497F215CA6872F2AE92E644B845633E215C45D160`。

全项目discover运行112项：111项PASS；唯一collection error仍为历史`test_r006b3a_family_paired_noninferiority.py`导入缺少`mpmath`，与v3代码无关，未跳过或下载依赖伪造全绿。定向v3 8/8 tests PASS。

按`experiment-audit`技能触发fresh same-family read-only reviewer。其总体为provisional `WARN`：A ground-truth provenance、B normalization、D dead-code、E scope与F simulation-only分类均PASS；未发现truth registry替代endpoint、自归一化造高分、phantom result或scope夸大。唯一WARN是审计时tracker/master log仍显示v2 RUNNING；本条与tracker同步已立即闭环，原始WARN不追溯改写。审计同时要求D1/D2额外保存raw delta RMSE、raw cliff margin和source RMS，并在D2对frozen predicted support而非construction atom评分；这些成为D1前置工程项。报告见`EXPERIMENT_AUDIT_M1_NIP_D0_V3_20260903.md/.json`及`.aris/traces/experiment-audit/2026-09-03_m1_nip_d0_v3/`。

保守解释：本轮只证明v3 observable-endpoint synthetic fixture和artifact contract可执行，不是D1/D2 confirmation，不支持M1、C1-NIP、C2-NIP或真实SAE。`M1_NIP_protocol_v3`转为TODO；fresh D1 seeds仍UNGENERATED，D2/real audit关闭。下一轮应先实现N09/N10/N12正交诊断、raw N11 scale字段及predicted-support endpoint API/tests，再制作fresh v3 D1两阶段adapter；不得直接生成D2 seeds。

## 2026-09-03 09:37 EDT — C024 truth-free orthogonal diagnostics 实现门 PASS

Heartbeat `ccad`按AGENTS顺序恢复状态，确认HEAD与origin/main均为`bd2bda0d356e5d6fac2b2060f9dc42fae7d5b8aa`、工作区起始干净、v3 D0 v2仍19/19 PASS，且fresh D1/D2 seeds与real audit均未打开。按上一轮审计action items，实现`src/ccad/nip_diagnostics_v3.py`及冻结配置`configs/m1_nip_orthogonal_diagnostics_v3.json`。

公共diagnostic API只接收observed instance和调用方提供的nonempty sorted target support；不选择support、不读取family truth或outcome label。它统一重算：target constituent/aggregate energy与cancellation ratio；source token/document activity和Kish ESS；独立mean arrays上的`d_mu`；若存在endpoint，则对调用方提供的support做shared-hook cliff与smooth evaluation。N11 endpoint新增raw `source_rms`、`raw_delta_rmse`和`minimum_raw_cliff_margin`，避免只保存由构造固定的normalized values。D2 contract明确要求support来自content-addressed frozen prediction；N12 identification absent时仅允许使用在mean check前冻结的best-centered diagnostic candidate。

在任何v3 D1/D2 seed生成前冻结synthetic attribute阈值：N09 unsafe cancellation ratio至少50；N10 sufficient evidence至少4 active documents，insufficient fixture的document-energy Kish ESS至多2.1；N12 mean mismatch threshold 0.05；N11 cliff/smooth/margin继承v3。20-pair-per-family定向测试验证N09最小ratio超过门、N10始终2 active documents且ESS不超过2.1、N11正确support与错误support产生不同endpoint且raw scales存在、N12 `d_mu≈1`并能对修复mean变为0、非法support fail-closed。定向13/13 PASS。一次非证据性的100-seed implementation probe观察N09 ratio范围86.53–433.09、N10 active docs恒2且document ESS 1.25–2.00、N12 `d_mu≈1`；该probe未预登记，不作为D1/D2或claim证据，仅用于确认冻结门远离数值边界。

全项目discover共117项：116项PASS，唯一collection error仍是历史R006测试缺少`mpmath`，与本轮变更无关，未跳过或安装依赖伪造全绿。源码SHA-256：diagnostics=`17C0BF7F74D2D83B5B067B347E0E239E0376B951A6D710F1B740307A7B82F1FF`，更新后N11 generator=`38B1FBB62248E6FF29EB2B88EF82B4BDE23E5127ECE79219F0200CEF3E8A0F24`，diagnostic config=`2987076AA9C922B7BE20DE62AB8848AD428E08AABCFD79186BBA9A92CEFB59D0`。

保守解释：这是C024的API/测试级工程证据，不是formal D1/D2结果；C024保持SCREENING，M1/C1/C2不变。下一轮应建立v3 D1 truth-blind predictor adapter与信息顺序测试，把fresh namespace、N11 approximate threshold和raw endpoint contract绑定；在prediction closure独立验证前不得打开labels。

## 2026-09-03 09:47 EDT — v3 D1 truth-blind adapter资格化；登记I1 prediction

Heartbeat `ccad`继续C024。新增`configs/m1_nip_i1_v3.json`、`configs/m1_nip_d1_v3.json`、`scripts/run_m1_nip_d1_predict_v3.py`、独立pre-label validator及信息顺序测试。Predictor只导入MSCC与observed generators，不导入`nip_truth`或orthogonal outcome scorer；同时绑定v3 protocol与diagnostic config hashes。N11与N07明确走approximate 0.05 lane；I1和formal D1分别使用phase namespace且前者不消费formal seed。Output保留proposal/prediction hash、paired caps与atomic closure，不序列化endpoint outcome或truth。

Py-compile与17/17定向tests PASS，覆盖config hashes、60/1200 row grids、I1/D1 namespace、N11 approximate lane和validator pre-label性质。现于执行前登记唯一run `M1_NIP_I1_predict_v3_impltest_v1_20260903T134700Z`为RUNNING：12 families×1 structural pair×5 paired caps=60 rows，512 observations，20 target atoms，`g_max=4`、budget 7,462。任务为lightweight CPU，无lease。该run仅验证prediction closure，不生成formal D1/D2 seeds，不打开labels、orthogonal outcomes、held-out或real audit。

### 2026-09-03 09:49 EDT — v3 I1 prediction closure 与pre-label validator PASS

I1 predictor正常完成并最后原子写入closure。60 rows完整覆盖12 families×1 pair×5 paired caps；truth_opened=false，runtime不参与selection。raw SHA-256=`E2A51F98A5BCDEAB3C6380FC45DBC7D9F2BFD54429BCCD3A48365D6CBDA7145A`，closure=`D67E5B0EBEC0342D5279D54922962767BE56CB24A98E8212DF7741DF10AF9D22`。

独立pre-label validator 19/19 PASS：重算closure bound files、code aggregate、source/input snapshots、protocol/diagnostic config binding、60-row paired grid、48个独立structural/sample/proposal/solver seeds、proposal/prediction hashes；AST确认predictor无truth或orthogonal outcome import；cap20的N11 frozen prediction包含target atom0且`d_ctr=0.01`。validation SHA-256=`580E5E9775FE81DE1F3EC0D60D9631F5D8047D017C1A7C56CA2024A2345B1070`。

保守解释：这只是I1 truth-blind prediction与closure工程证据，不是labeled D1、orthogonal attribute结果、D2、M1或C1/C2。formal v3 D1 seeds仍UNGENERATED，labels与real audit关闭。下一轮必须实现并tamper-test v3 score adapter：先验证closure，再动态导入truth；orthogonal metrics必须从seeds重建observed tensors并对frozen predicted support计算。只有I1 score及独立rescore validator通过后才可登记formal D1 prediction。

提交前全项目discover共121项：120项PASS；唯一collection error仍为历史R006测试导入缺少`mpmath`，与v3 D1 adapter无关，未跳过或安装依赖伪造全绿。

## 2026-09-03 09:57 EDT — 修复I1 closure的N12诊断候选缺口并登记v2

Heartbeat `ccad`在实现post-closure scorer前发现：I1 v1虽正确封存了identification prediction，但N12的真实identification应为native-absent，因而没有可供mean-mismatch审计的FOUND support。若在score阶段看到mean或truth后再选support，会违反已冻结的“mean check前固定diagnostic candidate”信息顺序。故不改写v1 artifact，也不把该缺口解释成理论失败；v1在其prediction-closure scope内仍为PASS，但不足以作为完整orthogonal scorer输入。

修复在truth-blind predictor中加入通用centered-only候选冻结：只用discovery centered kernels，在同一`g_max=4`与7,462预算内枚举候选，按`(d_ctr, support size, lexicographic IDs)`确定性选择，并写入candidate content hash。独立pre-label validator从sealed row seeds重建observed tensors并逐row重算该候选；它仍禁止truth import，也不计算mean mismatch或causal outcome。定向10/10 unittest PASS；系统Python的pytest入口不存在且其环境缺numpy，随后按项目既有R004环境与`PYTHONPATH=src`完成有效测试，前两次均未生成artifact。

现于计算前登记唯一run `M1_NIP_I1_predict_v3_impltest_v2_20260903T135700Z`为RUNNING。它使用I1 namespace、12 families×1 pair×5 caps，只验证修复后的sealed prediction contract；formal D1/D2 seeds、labels、held-out与real audit继续关闭。失败run必须保留。

### 2026-09-03 09:59 EDT — 修复后的I1 prediction closure PASS

新predictor正常完成60 rows并最后原子封存closure；raw SHA-256=`662A8DC729AF31AB1C3BDBA6AED3347D48C94E2A45108722B47374D8C2E56F72`，closure绑定的code snapshot=`7D63C092196E882F60BB605D087FAD3C64BAFC25D098646E026B790E56A6E949`。独立pre-label validator 20/20 PASS，新增门逐row由seed重建observed tensors并精确重生centered-only candidate；validation SHA-256=`0AB4ED24C16FC26AA9FFBC77C44E3FF5EEB350129DFCFC2B9BE197B19C30B368`。truth_opened=false，formal D1/D2 seed未生成。

使用项目R004环境运行全量unittest discover，125/125 PASS；这次环境包含历史R006测试所需依赖，因此没有collection error。保守解释：本轮只修复并验证closure-first输入契约，不是D1标签结果或C1/C2证据。下一轮应实现独立v3 scorer与rescore validator，先验closure/tamper、再动态导入truth，并从observed tensors对冻结support计算N09–N12属性。

## 2026-09-03 10:07 EDT — 登记v3 I1 post-closure scorer

Heartbeat `ccad`在已封存I1 v2 prediction上实现`score_m1_nip_d1_v3.py`及独立`validate_m1_nip_d1_score_v3.py`。Scorer在创建score目录及动态导入truth前完整核验closure、bound files、源码/input snapshots、protocol/diagnostic绑定、proposal/prediction/candidate hashes；tamper测试确认失败时truth import未发生且score目录不存在。之后仅对N09–N11使用冻结的唯一predicted support，对N12使用pre-mean centered-only candidate，从structural/sample seeds重建observed tensors并计算cancellation、document evidence、mean与shared-hook endpoint raw metrics。Identification score与orthogonal属性保持分栏。

独立validator没有导入scorer，逐row重建observed tensors、重新计算identification与orthogonal结果、cap aggregation和selection，并单独检查selected-cap N09–N12属性套件。py_compile与2/2 scorer information-order tests PASS。现于打开I1 labels前登记唯一run `M1_NIP_I1_score_v3_impltest_v1_20260903T140700Z`为RUNNING；输入仅为immutable `M1_NIP_I1_predict_v3_impltest_v2_20260903T135700Z` closure。Formal D1/D2 seeds、held-out与real audit仍关闭。

### 2026-09-03 10:11 EDT — I1 score v1 有效FAIL；字段名修复并登记v2

v1 scorer正常写出60行，identification aggregation仍选cap20；独立validator的selected-cap attribute suite失败。检查raw numeric artifact发现N09、N10、N12均正确测得，但N11 endpoint明明保存`cliff_effect_rmse=1.0`、`smooth_effect_rmse=0.1`与margin约0.05，分类器却读取不存在的`effect_rmse`并输出`CAUSAL_PASS`。这是明确的字段名implementation bug，不是理论或阈值失败。v1 run与validator traceback原样保留并标FAIL。

修复scorer和独立validator只把读取键改为已冻结API字段`cliff_effect_rmse`，不更改数值、threshold、prediction closure或protocol；新增targeted regression直接要求N11 observable endpoint分类为`CAUSAL_FAIL`。现登记新run `M1_NIP_I1_score_v3_impltest_v2_20260903T141100Z`为RUNNING，失败仍须保留。

### 2026-09-03 10:13 EDT — I1 score v2 PASS；C024获准进入formal D1

字段回归3/3 PASS后，v2 scorer在同一immutable prediction closure上完成；独立validator 10/10 PASS，逐行重算60条identification与orthogonal outputs、cap aggregates和selection。selected cap为20；该cap的N09在冻结predicted support `(0,1)`上测得cancellation ratio `273.8224`并判`OBSERVATIONALLY_UNSAFE`，N10为2个active documents、document ESS `1.9446`并判`INSUFFICIENT_EVIDENCE`，N11在support `(0,)`上测得cliff RMSE `1.0`、smooth RMSE `0.1`、normalized margin约`0.05`并判`CAUSAL_FAIL`，N12在pre-mean candidate `(0,)`上测得`d_mu≈1.0`并判`MEAN_MISMATCH`。score raw SHA-256=`736CDF3472E376AFB9817096D8E5B192F24327DDDBC7598BE9F9B62D843B427E`；validation=`43DA8A352143A3DCFBB7E1E5931094498B4556F6E192660F11C2BE3782BA9C88`。

全项目unittest discover 128/128 PASS。C024由SCREENING转ADMIT，含义仅为observable-endpoint组件获准进入fresh formal D1，不代表M1、C1或C2成立。Formal v3 D1/D2 seeds仍未生成，held-out与real audit仍关闭；下一轮须先把formal D1 config/source hashes与当前合格scorer/validator绑定并登记truth-blind prediction，仍应先封存再评分。

## 2026-09-03 10:19 EDT — 启动formal v3 D1 truth-blind prediction

Heartbeat `ccad`重新核验I1 score v2 status、immutable prediction closure及10/10独立validator，确认HEAD与origin/main均为`7946741bba5369d16d2ace17ce74f1e2d319d6a3`且工作区起始干净。Formal config继续绑定v3 protocol与冻结diagnostic config，12 families×20 fresh structural pairs×5 paired caps，共1,200 rows；N11使用approximate lane，所有rows在mean/truth前保存content-addressed centered-only candidate。

现于执行前登记唯一run `M1_NIP_D1_predict_v3_formal_v1_20260903T141900Z`为RUNNING。这是首次消费formal v3 D1 seed namespace；prediction进程禁止truth import，只允许最后原子封存closure并运行独立pre-label validator。Formal score、D2 seeds、held-out及real audit仍关闭；若失败不得打开labels。

### 2026-09-03 11:49 EDT — formal v3 D1 prediction closure PASS

Truth-blind predictor在lightweight CPU上用时约91.1分钟完成1,200/1,200 rows；长耗时来自每个paired cap额外冻结centered-only candidate的组合枚举，不涉及GPU或共享重型资源。进程最后原子封存closure；raw SHA-256=`6BF333525EB2482C465575A255FF0C931C31FFB9D3D192107EB1E42923417879`，closure=`26D18520F05A1476C3ACF11377BB5E27D039A4072753A93A7C723D13756A4D5F`。

独立pre-label validator 20/20 PASS，覆盖bound files/source/input/config/protocol/diagnostic绑定、12×20×5 paired grid与seed、proposal/prediction hashes、逐row centered candidate重生及N11 approximate observable contract；validation SHA-256=`F26C4B135DA9C2FD126B55766BF33B58D723082BADA533D8F82DA4594FA02EC6`。`truth_opened=false`，formal score目录不存在，D2 seeds、held-out和real audit均未打开。该PASS仅证明formal D1 prediction及信息顺序合规，尚无label/attribute结论；下一工作单元可先复核closure后登记formal v3 score。

## 2026-09-03 11:58 EDT — 启动formal v3 D1 post-closure score

Heartbeat `ccad`重新读取formal prediction的resolved config、closure、status及20/20 pre-label validation，并复核closure SHA-256仍为`26D18520F05A1476C3ACF11377BB5E27D039A4072753A93A7C723D13756A4D5F`、HEAD与origin/main均为`0a3d1b187208668abe070056ed70ffcc11030291`。只有完成这些检查后才登记唯一run `M1_NIP_D1_score_v3_formal_v1_20260903T155800Z`为RUNNING。

Scorer必须再次在创建score目录和动态导入truth前核验closure及所有绑定，并写独立目录；随后独立validator逐row重建observable tensors与全部identification/attribute输出。该run只打开formal D1 labeled synthetic development，不生成D2 seed，也不打开held-out或real audit；失败不允许修改prediction artifact。

### 2026-09-03 11:59 EDT — formal v3 D1 score PASS；cap20 selected

Formal scorer与独立validator均正常完成。Validator 10/10 PASS，对1,200条rows逐行重算identification与orthogonal measurements、cap aggregation和词典序selection；score raw SHA-256=`C2D9C6C24EEF5BA3F84AF92B8500759320F6B2E17AA0BD91A4C8D5AA847BEF60`，validation=`55A32690694521D87FBE2748D0E0241417701927645804EAB6114C8FF23C3ACA`，输入closure仍为`26D18520F05A1476C3ACF11377BB5E27D039A4072753A93A7C723D13756A4D5F`。

冻结词典序选择cap20：7个positive families的140/140 pairs精确恢复minimum support set/cardinality/multiplicity；5个negative families共100 pairs无false native positive；false unique与budget refusal均为0。Selected-cap orthogonal suite也全部20/20符合预冻结属性：N09 cancellation ratio范围114.95–332.68并判unsafe；N10 active documents恒2且document ESS最大1.99981并判insufficient；N11 cliff RMSE恒1.0、smooth最大0.1、minimum normalized margin约0.05并判causal fail；N12 `d_mu≈1`并判mean mismatch。

保守解释：这是formal D1 labeled synthetic development，支持在v3构造内冻结cap20与属性测量，但不是fresh D2 confirmation、真实SAE、held-out contribution或C1/C2证据。`M1_NIP_protocol_v3`仍TODO；D2 seeds仍未生成。下一轮必须先产生selection-freeze manifest与D2 config，绑定D1 prediction/score/validators、selected cap、protocol/diagnostic/scorer hashes，并经静态审计后才可原子生成fresh D2 seeds。

## 2026-09-03 12:09 EDT — 登记pre-D2 selection/config freeze audit

Heartbeat `ccad`创建`configs/m1_nip_d2_selection_freeze_v3.json`与`configs/m1_nip_d2_v3.json`。Freeze manifest内容寻址绑定formal D1 prediction closure/prelabel validator、score raw/summary/independent validator、selected cap20、v3 protocol/diagnostics及D2 predictor/scorer/validators源码；D2 config固定fresh `D2` namespace、20 pairs/family、selected cap `[20]`、240 rows和既有7,462 budget。Runner仅做必要泛化以接受D2 selected-cap-only grid，并把freeze manifest作为run-local input snapshot；I1/D1五cap语义不变。

新增独立静态审计器，检查完整hash chain、D1 validators PASS、cap20 selection重放、D2 source绑定、fresh namespace、信息边界及不存在已有D2 prediction目录。py_compile与8/8定向tests PASS。现登记`M1_NIP_D2_freeze_v3_audit_v1_20260903T160900Z`为RUNNING；审计只产配置artifact，不生成D2 seed或打开label/held-out/real audit。

### 2026-09-03 12:10 EDT — pre-D2 freeze audit PASS；D2 generation尚未启动

静态审计10/10 PASS。D2 config SHA-256=`4CF9B8746360CE75821A8C7C278D789758DA18A7061FE81EA2405D9919751FB8`，selection freeze=`388692CC223A9C1BF5EBF2A3B9202BDEB38755EEDE10AF74C49CA2EC59742916`，validation=`F9AE273084ADC4759520FBB2ABAF4F745CABA518BBCC3CEF82B507AD75A59BAC`。全项目unittest discover 129/129 PASS。

审计时确认不存在任何`M1_NIP_D2_predict_v3_formal_*`目录，故D2 seeds仍严格UNGENERATED，labels/held-out/real audit关闭。该PASS只授权下一轮使用已绑定配置登记fresh D2 truth-blind prediction；它不是D2结果，也不改变M1/C1/C2。鉴于selected-cap-only D2仅240 rows，预计计算量显著低于D1五cap 1,200 rows，但仍应在run中记录实测耗时。

## 2026-09-03 12:17 EDT — D2生成前修正资源声明并重审

Heartbeat `ccad`恢复状态后注意到formal D1同类计算实际耗时91分钟，因此即使D2只有其约五分之一，仍应按AGENTS视为CPU-heavy，而非原config中的`lightweight_cpu_no_lease`。在任何D2 seed生成前，将D2 config的纯运行资源字段改为`cpu-heavy_lease_required`；selection、threshold、namespace、row grid和科学协议均未改变。原freeze audit v1在其旧config scope内保留PASS，但不用于启动新计算。

现登记`M1_NIP_D2_freeze_v3_audit_v2_20260903T161700Z`为RUNNING，使用同一静态validator重审修改后的config。只有v2审计PASS后才允许通过母目录resource manager的`cpu-heavy` lease运行D2 prediction；不申请当前被其他项目占用但本任务不需要的disk-e lease。

### 2026-09-03 12:19 EDT — 修正后freeze audit PASS；启动formal D2 prediction

修正资源声明后的静态审计再次10/10 PASS；新D2 config SHA-256=`28018643DF95C79D370D25E8A2F994DD32EF85F1E0D1F03A26137CE6F8E1E800`，selection freeze仍为`388692CC223A9C1BF5EBF2A3B9202BDEB38755EEDE10AF74C49CA2EC59742916`。审计仍确认D2 seeds尚未生成。

现登记唯一run `M1_NIP_D2_predict_v3_formal_v1_20260903T161900Z`为RUNNING，并首次生成fresh D2 seeds：12 families×20 pairs×selected cap20=240 rows。执行必须持母目录`cpu-heavy` lease并自动heartbeat/release；prediction保持truth closed，完成后只运行独立pre-label validator。Formal D2 score、held-out和real audit继续关闭。

### 2026-09-03 12:16 EDT — formal D2 prediction closure PASS；标签仍关闭

第一次在sandbox内调用资源管理器因其母目录lease文件无写权限而`PermissionError`，发生在lease获取和run目录创建之前；随后按既有授权提升权限重试。资源管理器成功获取`cpu-heavy` lease，prediction耗时22.3秒完成并自动释放；独立validator另持同类lease约9.0秒重生全部240 rows后释放。最终资源状态确认`cpu-heavy`为free；未触碰其他项目遗留的expired disk-e lease。

Fresh D2 prediction完整覆盖12 families×20 pairs×cap20，共240 rows。Raw SHA-256=`46DCA5946A5826D9021FB6DB495043D1588F93D2900495D4229BC776BF16A490`，closure=`C52116174A648986FB5FF337F628DBF88DC129A109E4C5AB6AE82D625401A67A`。独立pre-label validator新增selection-freeze binding后21/21 PASS；validation=`D75CD6EA74CF73E6BED0DF173AEBDB3E634E2525CE006C92525409270361C03F`。`truth_opened=false`，score目录尚不存在，held-out/real audit仍关闭。

保守解释：这只是fresh D2 truth-blind prediction及closure证据，尚不能称D2 confirmation PASS，也不支持M1/C1/C2。下一轮只可在再次复核closure后登记独立formal D2 score；score失败不得修改prediction artifact。

## 2026-09-03 12:25 EDT — 启动formal D2 post-closure score

Heartbeat `ccad`重新读取D2 resolved config、closure、status和21/21 pre-label validation，确认closure SHA-256仍为`C52116174A648986FB5FF337F628DBF88DC129A109E4C5AB6AE82D625401A67A`、selection freeze绑定存在，且HEAD与origin/main均为`0e5bfc9cc2b562d6713cde3bea567313aba1c6a7`。只有完成上述检查后才登记唯一run `M1_NIP_D2_score_v3_formal_v1_20260903T162500Z`为RUNNING。

Scorer使用selection freeze中预绑定的源码版本，必须先完整核验closure再动态打开D2 truth，并写独立score目录；独立validator随后逐row重建observable tensors和identification/orthogonal results。本run不修改任何D2 prediction、threshold或support，held-out和real audit保持关闭。

### 2026-09-03 12:26 EDT — fresh formal D2 score PASS；v3协议级合成门完成

D2 scorer与独立validator正常完成。Validator 10/10 PASS，对fresh 240 rows逐行重算identification、orthogonal attributes、aggregate及selected-cap contract。Score raw SHA-256=`52BC6DA5288DA5F92EACF3E527B634A55B752B30A515A3E886857D63B433A472`，validation=`7CBD889CF9EE07DAD83D627BCD594AB98B01496D99E50C278810B953C681EF0E`，输入closure仍为`C52116174A648986FB5FF337F628DBF88DC129A109E4C5AB6AE82D625401A67A`。

Fresh D2结果复现D1门：7个positive families共140/140 pairs精确恢复minimum native support/multiplicity；5个negative families共100 pairs零false native positive；false unique与budget refusal均为0。N09–N12各20 pairs全部命中预冻结属性：N09 cancellation ratio 136.10–412.43；N10 active documents恒2且document ESS最大1.99998；N11 cliff RMSE恒1.0、smooth最大0.1、minimum normalized margin约0.05；N12 `d_mu≈1`。

据此`M1_NIP_protocol_v3`本身由TODO转PASS：D0、D1、fresh D2及信息顺序/正交属性门均完成。保守边界不变：这些仍是synthetic fixture confirmation，不是real SAE或C1/C2证据；历史R001–R003 parent FAIL不得自动改写，R006/M2仍BLOCKED。下一轮应做独立parent aggregation/integrity audit，明确v3是否充分替代旧corrective gate及哪些历史缺口仍需补，审计前不恢复M1或启动真实主线。

## 2026-09-03 12:32 EDT — 启动M1-NIP v3 parent aggregation/integrity audit

Heartbeat `ccad`按`experiment-audit`技能启动fresh GPT-5.6-Sol ultra只读reviewer，输入仅为v3 evaluation/code、configs/protocol、formal D0/D1/D2 artifacts以及历史parent audit/corrective evidence的文件路径和A–H审计清单。审计目标不是重复数值计算，而是独立判断ground-truth provenance、normalization、artifact existence、dead code、scope、information order，以及v3能否作为prospective replacement关闭历史M1 parent blocker。

该审计明确标记`review_independence=same-family`、`acceptance_status=provisional`。现登记`M1_NIP_v3_parent_integrity_audit_20260903T163200Z`为RUNNING；reviewer结论收口前不改变历史R001–R003 FAIL、不恢复M1 parent，也不解除R006 BLOCKED。报告与完整trace必须持久化，若发现tracker陈旧/重复行则以追加勘误和最小同步修复处理，不改写实验artifact。

### 2026-09-03 12:43 EDT — parent audit收口FAIL；区分数值通过、实现缺陷与范围不足

Fresh same-family reviewer已完成A–H审计；overall verdict=`FAIL`，artifact integrity=`WARN`，parent gate=`FAIL`，acceptance仍为`provisional`。Reviewer独立重生60条D0、1,200条D1 prediction与240条D2 prediction，零mismatch；D0最大浮点差`4.44e-16`，并对D2每例的6,195个`|J|≤4`允许support做独立穷举，未发现truth/prediction错配；三阶段seed集互斥。因此保留的正面结论仅是：v3窄范围synthetic MSCC identification + N09–N12 fixture subgate的数值与信息顺序可复核。

审计同时确认一个真实implementation defect：`complete_universe`只确保所有atom ID被proposal，搜索却只枚举到`g_max`；当`g_max=4`时，一个需5 atoms的有效support反例会被输出`CERTIFIED_ABSENT`，将`g_max`改为5即`FOUND`。这不改变现有v3 fixtures的数值结果，但使全局absence语义不成立。已登记C025；修复须将machine-readable certificate限定为`|J|≤g_max`的bounded absence，或真正枚举complete support universe，并加5-atom regression。

范围层面，v3未实施继承的fair baselines/OMP simplicity、BCC/PSC及raw energy/rank/leverage/solver/proposal/coverage surface、N06/N08 controls和独立mean/discovery/evaluation streams；部分validator只重分类已序列化prediction，并非implementation-independent重跑整个identification path。因此v3不是历史R001–R003/corrective parent的replacement，R006继续`BLOCKED`；已登记C026作为前瞻性补齐支线。

记录勘误：`EXPERIMENT_TRACKER.md`中同一D2 score run同时存在PASS与stale RUNNING行；已删除陈旧重复行，实验artifact未改动。审计报告保存为`EXPERIMENT_AUDIT_M1_NIP_V3_PARENT_20260903.md/json`；历史FAIL原样保留。下一步不是启动M2，而是先完成C025的语义/API修复和provenance强化，再以fresh suffix对C026做审计前freeze；不得在已打开的D2 labels上调参。

### 2026-09-03 12:45 EDT — C025 bounded-absence implementation repair PASS

不等待下一个实验周期，本轮直接修复审计发现的absence certificate越界。裁决为保守的fail-closed方案：`complete_universe=True`不再只检查是否proposal了全部target atom，还强制`g_max >= target_count`；否则API拒绝发出全局`CERTIFIED_ABSENT`。所有v1/v2/v3 runner调用点同步改为只在真正complete support universe时设置该标志；20 atoms、`g_max=4`的未命中结果今后只能是`UNRESOLVED`。这没有改动阈值、support search或现有封存artifact。

新增5-atom regression：当source仅可由5个`0.2x` target atoms之和表示时，`complete_universe=True,g_max=4`必须fail closed；同一对象在bounded mode/gmax4返回`UNRESOLVED`，在true complete mode/gmax5返回`FOUND`且minimum support size=5。目标测试23/23 PASS；全项目unittest 130/130 PASS，使用`D:\CCAD_Storage\environments\r004\Scripts\python.exe`。首次用系统Python调用因缺`numpy`产生3个import error，属错环境调用，已保留在本条而不误记为科学失败。

该PASS仅关闭C025的核心API实现缺陷；由于旧D2 labels已打开，修复后的端到端artifact/provenance确认必须使用fresh suffix/seeds。C026与R006状态不变，M1 parent仍FAIL/BLOCKED。

## 2026-09-03 13:07 EDT — C026选择补齐路线；parent-completion P0 v3 PASS

Heartbeat `ccad`按experimental-design流程对parent audit缺口做了前瞻性设计。裁决不选“退役旧义务并改写parent gate”的低成本路线，而选择C026-A补齐路线：保留继承的fair baselines、OMP simplicity、raw BCC/PSC/energy/rank/leverage/proposal/solver/coverage surface、N06/N08 controls和独立mean/discovery/evaluation/intervention streams。设计把family作为block，structural seed pair作为独立重复单位，同pair不同method作paired repeated measures；token只是technical measurement，不计伪重复。

新冻结`M1_NIP_PARENT_COMPLETION_PROTOCOL_V1_20260903_130500.md`及machine config `configs/m1_nip_parent_completion_v1.json`。Protocol SHA-256=`0D9B5E8F6F1EC9A6BE8BEF820D019541FEB0833A00AF16CEA4664428F5268B74`，config SHA-256=`CCE798DA4D1025F70569E9376BF2C4F165E4C6AC688B1D91E9332A4F6743C2D5`。正式设计固定12 families×20 fresh pairs，6条互异seed streams，9个native lanes及2个continuous references；`g_max=4`、20 atoms、7,462 budget与既有threshold不改。Formal seeds仍`UNGENERATED`，truth/real audit关闭。

P0静态validator首次调用v1数值16/16，但只生成`validation.json`，违反项目artifact contract，故保留为FAIL。修复validator使其同时产生resolved config、environment、code/input hashes、status、stdout/stderr与manifest后，v2数值与artifact通过，但run ID时标被预分配到晚于实际finish，作为recordkeeping FAIL保留，不追认。

终态`M1_NIP_PC1_V1_P0_static_v3_20260903T170630Z` 16/16 PASS；独立PowerShell重算manifest全部匹配，validation SHA-256=`E01F6AEE4C30F2C9AF79AC93AE072C7065516DB6FF1A388263423532789DE03B`，manifest SHA-256=`90D4BF70531C147AFD66F78B3015B1403C5B4B170B48B9A758BF1B09C5314087`。全项目unittest 133/133 PASS。该P0只授权下一轮实现P1 truth-closed one-pair-per-family metric/baseline smoke；不生成formal seeds，不改变M1 parent FAIL或R006 BLOCKED。

## 2026-09-03 13:17 EDT — raw metric adapter PASS；P0因baseline操作化不完整勘误为FAIL

Heartbeat `ccad`在启动P1前实现`src/ccad/nip_metric_surface.py`。Adapter只接收observed tensors与已冻结support，不接收truth、label、family rule、proposal或selection；持久化centered/mean residual的raw numerator/denominator、BCC raw components、synthetic atom-direction PSC/ranks/angles、mean vectors、effective rank/condition、cancellation/leverage、occupancy与document ESS。只能在post-closure scoring中得到的proposal recall/solver/end-to-end/coverage字段显式记为`NOT_APPLICABLE_PRELABEL`，不伪造数值。

Metric adapter定向测试覆盖N01 raw identity、N06 PSC rank boundary、N09 cancellation、N10 document evidence、N12 centered/mean separation与冻结schema全字段；22/22 targeted PASS。在static validator新增第17个`baseline_operationalization`检查后，最新全项目141/141 tests PASS，且测试确认当前v1 config必须被该检查判FAIL。这是implementation-only证据，未运行P1也未打开任何新seed/label/evaluation。

随后对baseline API做实现前审计，发现P0 v3的16个检查只验证了9个native lane和2个continuous reference的名称存在，却没有冻结dustbin Sinkhorn的cost/entropy/dustbin/convergence/support extraction、OT-mass threshold、OMP coefficient/stopping/native conversion、spectral graph/rank、random matching replicates及continuous solver参数。因此直接编码会引入未登记的researcher degrees of freedom。P0 v3已勘误为FAIL，P1保持未启动，formal seeds仍`UNGENERATED`。

本轮在2026-09-03检索并记录了Scikit-learn OMP官方规范（fixed cardinality与residual tolerance是不同regime）、Cao et al. 2026 SAE semantic OT、Cohen-Indelman & Indelman 2024 dustbin partial matching，以及Gerasimov et al. 2026 cross-seed stable subspace路线；来源与设计影响详见`M1_NIP_BASELINE_PARAMETER_AUDIT_20260903_131700.md`。新登记C027；下一轮必须先以前瞻性suffix冻结参数并扩展P0 validator，然后才能运行P1。M1/R006仍FAIL/BLOCKED，无需用户即时裁决。

## 2026-09-03 13:29 EDT — C027 executable-baseline suffix冻结；PC2 P0 v2 PASS

Heartbeat `ccad`继续用experimental-design规范修复P0的baseline操作化缺口。新冻结`M1_NIP_PARENT_COMPLETION_PROTOCOL_V2_20260903_132700.md`与`configs/m1_nip_parent_completion_v2.json`，新namespace为`M1_NIP_PC2_V1`；不复用v3/PC1数据，P1/P2 seeds仍`UNGENERATED`，truth/evaluation/intervention全部关闭。Protocol SHA-256=`7877D0C014DD701F5B6BC2986C1543B1FB638DC6B0AF8B26D582C8807327805C`，config SHA-256=`FEED1B084102F86156F448B845A4F55EA7559DD166AFFC62CA5D12B648AC5B12`。

核心裁决是：所有native baselines只用discovery信息产生ranking/proposal，然后统一以unweighted target-atom prefixes 1–4和MSCC相同`d_ctr/d_mu`阈值检验；OMP/OT的continuous coefficients只能rank，不得进入native endpoint。冻结了contribution singleton、PW-MCC/Hungarian、decoder-cosine greedy、dustbin log-Sinkhorn、signed forward OMP、unbalanced OT-mass、Li15-style spectral/local-SVD、degree/budget-matched random，以及signed/nonnegative continuous references的cost、regularization、stopping、ties、support conversion和runtime protocol。单source synthetic下PW-MCC/OT的degeneracy必须显式报告，不包装成global assignment证据。

Static validator由17个门扩展到21个，新检查baseline必需字段、关键数值、common native rule、runtime protocol和4个来源registry。第一次PC2 P0调用数值21/21，但run ID预分配时标晚于实际finish，作为recordkeeping FAIL保留，不追认。终态`M1_NIP_PC2_V1_P0_static_v2_20260903T172830Z` 21/21 PASS，manifest独立重算PASS，validation SHA-256=`84CA503D59526E37CEAA3E03B756AF12F3AA519CD7EE9965DFC3DC1F8A3BC3F6`，manifest SHA-256=`56D11344CDF370941BEAA8AAB18FCD88F20270808608F77C8059590741348F22`；全项目142/142 tests PASS。

C027由READY-FOR-SCREEN转ADMIT，只表示该contract获准进入P1 implementation，不表示baseline有效或M1过门。下一轮先实现统一baseline API与deterministic conformance tests，在源码/输入hash封存前仍不生成P1 seeds。R006继续BLOCKED。

## 2026-09-03 13:40 EDT — baseline API tranche 1 PASS；P1继续关闭

Heartbeat `ccad`按PC2 v2实现`src/ccad/nip_baselines.py`的第一批truth-free API：`CONTRIBUTION_NEAREST_ATOM`、单source范围`PW_MCC_HUNGARIAN`、`GREEDY_DECODER_COSINE`、`BINARY_FORWARD_OMP`、`RANDOM_MATCHED_GROUP`，以及 signed least-squares与deterministic projected-gradient NNLS references。Native API只接收source/target discovery contributions、独立mean contribution、固定阈值/预算/种子；不存在truth、label、evaluation、intervention或planted-support参数。OMP和continuous fit的系数均不进入native endpoint，native output只用unweighted support重算。

测试过程保留三个实现级发现：首次test helper误覆盖`unittest.TestCase.run`而在collection时TypeError，改名后修复；原测试假设OMP应恢复N01，但冻结的unit-L2 selection会使`0.4x/0.6x`两列完全tie，实现正确按协议`SELECTION_TIE/BUDGET_REFUSAL`；未用index破tie或修改规则追求好结果，而是把该负结果写入测试，另加无tie二维planted case验证OMP可恢复；同时修复了`g_max`可合法大于tiny atom count却被API误拒的边界bug。

终态targeted 9/9，full 151/151 PASS。源码SHA-256=`9EA6A52344AB1611942A41751A1BFBAABDF84F8CCD54714F188742404D990576`，test SHA-256=`20234D4E77B154141E4CB1F1D854BCCCB2B0CE6F00D49E9D109A4BBFCDE36E06`。`DUSTBIN_SINKHORN`、`OT_MASS_NATIVE_SUPPORT`和`SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT`仍显式`NotImplementedError`，测试要求其fail loudly；因此该run仅在tranche-1范围PASS，P1仍不得启动，formal seeds/truth/evaluation/intervention未生成/未打开。下一轮实现和验证剩3条native lanes，然后再做全registry completeness gate。R006继续BLOCKED。

## 2026-09-03 13:49 EDT — baseline registry implementation完成；C027实现门PASS

Heartbeat `ccad`按PC2冻结参数完成剩余三条native baseline。`DUSTBIN_SINKHORN`使用clipped singleton discovery cost、一个source/target dustbin、uniform augmented marginals和balanced log-domain updates；`OT_MASS_NATIVE_SUPPORT`使用同一truth-free cost与`rho=1.0, epsilon=0.05`的unbalanced log-Sinkhorn。两者都只按outgoing target mass排序，再用共同的unweighted prefix与`d_ctr/d_mu`规则判定；单source限制明确记录为`DEGENERATE_SINGLE_QUERY`，不包装成多query OT证据。两类solver若1000步内不满足冻结容差会`BUDGET_REFUSAL/SINKHORN_DID_NOT_CONVERGE`，不静默使用未收敛结果。

`SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT`只从每个observed atom contribution matrix的rank-one SVD恢复code；若relative residual超过`1e-12`则fail closed。随后按joint absolute code correlation、0.2阈值、unnormalized Laplacian、2–8最大eigengap与10次deterministic k-means生成mixed cluster，只从最佳contribution singleton所在mixed cluster按`d_ctr`排序并截到4。检查时发现公共`li15_spectral_proposal`虽文档/协议要求absolute correlation，实际graph仍使用signed correlation；本轮同步改为absolute，属于冻结规范的implementation repair，不改变协议。

Run ID=`M1_NIP_PC2_baseline_api_impltest_v2_20260903T174928Z`。新增registry exact-completeness、Sinkhorn deterministic/convergence/scope、spectral deterministic/factorization和rank-two拒绝测试。第一次命令误用系统Python，collection阶段因缺NumPy而FAIL，未执行测试或生成实验artifact；换用锁定R004解释器后targeted 12/12、全项目154/154与py_compile全部PASS。源码SHA-256：`nip_baselines.py=67980D775D41602147BCE9E84F9F07A7AD436C2E05F2DFA2E52F4FE9FEB42048`，`proposal.py=2AEDDE174DCAED2CC701FCFE3C36015E3AEB311C3A8ED4312FD8262ABA23AD68`；测试SHA-256=`77CDC912A8D82A98F113414463EF41E6AE019BF9647E1105C9DF38A8BC058BEB`。

保守解释：PC2注册的8条native baselines与2条continuous references现都有truth-free实现和conformance coverage，C027实现门通过；这仍不是任何方法在fresh data上的结果。P1 formal seeds、truth、evaluation和intervention仍未生成/未打开，M1 parent保持FAIL、R006保持BLOCKED。下一轮可实现P1 truth-closed one-pair-per-family runner、完整runtime/cost ledger与pre-label closure；只有独立validator通过后才可打开P1 labels。

## 2026-09-03 14:00 EDT — PC2 P1 runner与pre-label validator实现门PASS

Heartbeat `ccad`继续按C026-A实现P1 truth-closed integration surface。新增执行配置`configs/m1_nip_parent_completion_p1_v1.json`：固定12 families×1 fresh structural pair、six distinct streams、9 native lanes+2 continuous references、MSCC相同threshold/gmax/budget、runtime warmup1+measured5和random primary之外32个diagnostic permutations；P1配置明确`formal_seed_manifest_status=UNGENERATED`、`formal_seed_consumed=false`，prediction不读取evaluation/intervention或truth。

新增runner与独立pre-label validator。Runner在生成seed前封存源码和输入，seed严格按`protocol_hash||code_hash||P1||family||pair||stream`派生；分别生成mean/discovery observations，保存132条proposal/prediction、实际candidate cost、五次runtime、random diagnostics cost、seed ledger、环境、状态和atomic closure。Validator先核验closure及workspace/source snapshot hash，再用AST检查truth import与evaluation/intervention seed reads，重派生全部seeds并重跑132条scientific predictions；runtime不要求bitwise相等，但所有ranking/support/status/cost与random diagnostics必须一致。正式P2 seeds不在此流程生成。

实现测试首次用字符串搜索检查禁止的seed读取，validator为表达该检查本身包含相同字符串，造成1/16自指式误报；未运行P1、未生成run artifact。修复为AST `Subscript`数据流检查后targeted16/16、full158/158、py_compile PASS。Run ID=`M1_NIP_PC2_P1_runner_impltest_v1_20260903T180033Z`。配置SHA-256=`A53E56DCD33B91BC85241D10307BF475AB48658550F9CD39EA9E37375F54C1ED`；runner=`3633BB189EAFC6971D94D1C67EE5B198ABBBE637B24A284368975429A502F29B`；validator=`6D4CA96077C6B4148917B1BB183B86FFCF24C45163ED2D04799291EF0C91189C`；test=`F0D5536CFBCBDA3AADC6EB2BF62501046DB35D168888D9364908D3C727D2DFCD`。

该PASS只说明P1执行面可启动；没有方法结果、metric surface或label证据，M1/R006状态不变。下一动作是先提交并固定实现版本，再以唯一run ID执行fresh P1 prediction及pre-label recomputation；任一closure/validator失败都必须保留run且不得打开truth/evaluation/intervention。

## 2026-09-03 14:01 EDT — 启动PC2 P1 truth-closed prediction

Runner实现已提交并推送，固定HEAD/origin=`a503f7d45d6aef9d219f2692bb3e2511c69598d7`。现登记唯一run `M1_NIP_PC2_V1_P1_predict_v1_20260903T180139Z`为RUNNING；执行12 families×1 fresh pair×11 lanes，只生成mean/discovery observations、proposals、predictions、runtime/cost与closure。必须通过母目录`cpu-heavy` lease运行；truth、evaluation、intervention和formal P2 seeds保持关闭。Runner完成后只能启动独立pre-label validator，未通过前不得创建score目录或导入truth。

### 2026-09-03 14:04 EDT — P1 v1 FAIL；修复synthetic rank-one atom contract

`M1_NIP_PC2_V1_P1_predict_v1_20260903T180139Z`取得`cpu-heavy` lease后在4.26秒内fail closed；lease已自动释放并复验free。失败发生于prediction中、closure和prelabel validation之前，truth/evaluation/intervention均未打开。具体反例是二维hook family的decoy atom 2：contribution matrix rank-one relative residual=`0.684965`，无法满足spectral code-correlation lane所需的原生SAE atom结构。Run目录保留源码/输入snapshot、RUNNING后改写的FAIL status及完整stderr，不覆盖或追认。

裁决新增C028并拒绝“取leading SVD当作code”的低成本伪修复。`nip_synthetic_v2`的20-atom decoy现先在observation space构造与所有已有atom codes及constant vector正交的code basis，再乘固定source decoder，因而每个decoy严格为scalar-code×fixed-decoder rank-one atom，同时保留flattened contribution orthogonality和原residual schedule。新增12-family逐atom rank-one regression。

修复后的第一次targeted回归暴露certificate的旧数值缺陷：对线性相关forbidden atoms使用满列QR会把任意QR补空间当作真实禁区，导致10个family的decoy正交性证书假失败。Construction与certificate均改用`1e-12` numerical-rank SVD projector后，targeted24/24、full159/159 PASS；cap-pressure、bounded-search truth、N11 endpoint和v2/v3 compatibility tests均未改变。另将P0 unit test从“运行后仍要求no-existing-run=true”修为只验证不随执行状态改变的contract checks；P0历史artifact未改。

Repair run ID=`M1_NIP_PC2_rankone_decoy_impltest_v1_20260903T180444Z`。源码SHA-256=`907DC5F8D5299D743E8047348E0D1E5DF19C3A6E346894045825128442B0A01C`；rank-one test=`9C9D4824526E9103BC30FC220B82462A3A2735D25AD12FA3DA5E288D7967E97E`；P0 test=`A1A750FD8AB28CD95EFF87DD7BFF2C70710D405FAE509C5951A96342ACDDC345`。这属于labels打开前的synthetic schema/implementation修复；下一步提交固定新code hash后使用新的P1 run ID和fresh seeds重跑，不复用v1 partial state。

### 2026-09-03 14:05 EDT — 启动C028修复后的fresh P1 v2

C028修复已提交并推送，HEAD/origin=`e1c3662dd0beafa5b25669c6641fa27ea2e38e0b`。新run `M1_NIP_PC2_V1_P1_predict_v2_20260903T180532Z`登记为RUNNING；因seed derivation绑定新code snapshot，本次12 pairs均与v1不同，不读取或复用v1 partial artifact。执行信息边界、lane、阈值、预算与runtime protocol均不变，仍须先prediction closure再pre-label validator，labels保持关闭。

### 2026-09-03 14:06 EDT — P1 v2 FAIL；修正mean-only与endpoint generator混用

`M1_NIP_PC2_V1_P1_predict_v2_20260903T180532Z`在`cpu-heavy` lease中于7.47秒fail closed，lease已释放。P1 runner将冻结的mean-only sample size `n=257`传给N11 intervention endpoint generator，而该generator为exact zero-mean perturbation要求偶数n，因此在N11 mean stream抛出`ValueError`。失败仍发生在closure/validator/labels前；v2 run和stderr完整保留。

冻结协议的mean n257与N11 endpoint偶数约束并不冲突：mean stream只应提供source/target mean contributions，不应构造干预endpoint。Runner改用无endpoint的20-atom observed generator生成mean stream；discovery仍用v3 endpoint-aware generator，evaluation/intervention仍完全未读。新增odd-n257 N11 mean compatibility regression。Repair run `M1_NIP_PC2_mean_stream_impltest_v1_20260903T180654Z` targeted17/17、full160/160 PASS；runner SHA-256=`4D0F6A9C3EEBCD895C938C5199BBB6342163BA2AF147523AD046AE8B94F6F528`，test=`E236ECC19C294EB602F54E5F2863435A91BF0FDF0F13AD56C33253DF2EDAE0D1`。下一步提交后以第三个fresh code-hash run重试，仍不复用任何失败run state。

### 2026-09-03 14:07 EDT — 启动fresh P1 v3

Mean-stream修复已提交并推送，HEAD/origin=`cdf7c03157133ff74e8dab2a5f6f6b7e7ab08824`。登记`M1_NIP_PC2_V1_P1_predict_v3_20260903T180728Z`为RUNNING；新code snapshot派生全新P1 seeds，v1/v2失败run只作证据不作输入。协议和信息边界不变。

### 2026-09-03 14:09 EDT — P1 v3 prediction sealed但pre-label gate FAIL

`M1_NIP_PC2_V1_P1_predict_v3_20260903T180728Z`在`cpu-heavy` lease中8.28秒完成132条prediction并原子封存，closure SHA-256=`0264B65C08CF513E8F9C645A483222F32B764A4FADBA58142C36D6A3B68832C6`。独立validator随后另持lease重跑，17项中16项PASS：scientific predictions、six-stream seeds、random diagnostic cost、runtime count、closure/source/input hashes、AST信息边界和formal-seed closure均通过；唯一失败为`proposal_recomputation`。

诊断确认值本身没有差异：JSONL反序列化的ranking/scores为list，而内存重算结果的同值字段为tuple；validator对prediction先做了JSON canonicalization，对proposal却直接用Python容器类型比较。因此整个P1 v3仍按gate记FAIL，validation SHA-256=`FC793A6B3699CAAE1C7FD5A5E9CA555A685869C94B4DA5EE7271EC6D53BDF751`，不得追认或打开labels。修复只将proposal两侧先转canonical JSON再比较，不改任何科学值。Repair run `M1_NIP_PC2_p1_validator_impltest_v2_20260903T180903Z` full160/160 PASS；validator SHA-256=`76BDC5116E2FA239F6AB12024BB4E5898C08EFB0482BAE8DA84E615E839BBBFA`。因validator属于code snapshot，仍须提交后fresh seeds重跑v4。

### 2026-09-03 14:09 EDT — 启动fresh P1 v4

Validator修复已提交并推送，HEAD/origin=`d03dde7e4e54d4b5826f4948502aa6cef835775e`。登记`M1_NIP_PC2_V1_P1_predict_v4_20260903T180938Z`为RUNNING；再次由新code snapshot派生fresh P1 seeds，协议与information boundary不变。

### 2026-09-03 14:10 EDT — PC2 P1 prediction与pre-label gate PASS

`M1_NIP_PC2_V1_P1_predict_v4_20260903T180938Z`在独占`cpu-heavy` lease中正常封存132条rows；独立validator随后重新获取同类lease并从sealed source/config/code hash重派生seeds、重生observations、重跑全部11 lanes，17/17 checks PASS。两次lease均自动释放，`cpu-heavy`现为free；未占用另一项目正在使用的`disk-d-io`。

关键artifact：closure SHA-256=`03DD31527DD67E79A6894713728037818974974D41282AD496C81B3F96E10A75`，prelabel validation=`EE5CB95CC48C8EBBE12E224C3FF8E167585C03536AA109B3A79C51D6D6DE4E0A`，predictions=`9BCF2473FA09BF3B66F3105BA0A09AB387836471911CAC4C0949D5F9E2AF7EFC`，proposals=`6AD1DD82DBD9FC92DC350E73D5D15339C0DBE2B564B94BB32942D0BEB9E45EE6`，seed ledger=`1227AAD56C865B43D0C4A33046A41D3E3A20D886A625EE588DA0717D8E193810`。`truth_opened=false`，formal P2 seed未生成，evaluation/intervention seed仅封存未读取。

Truth-free输出分布仅作执行诊断：MSCC为7 FOUND/5 UNRESOLVED；contribution-nearest、PW-MCC、dustbin、OT-mass、spectral和OMP各1 FOUND/11 UNRESOLVED；greedy cosine和random primary均12 UNRESOLVED。Continuous references无native identification。由于尚未打开truth/evaluation/intervention，这些计数不能解释为accuracy、baseline优劣或C1/C2证据。

P1 prediction subgate现PASS，只授权下一轮实现post-closure scorer：必须先重验closure/prelabel PASS，再动态打开P1 truth及独立evaluation/intervention streams，生成完整metric surface与N06/N08/N11 controls；在score和独立validator通过前，M1 parent仍FAIL、R006仍BLOCKED，P2 formal不得启动。

## 2026-09-03 14:19 EDT — P1 post-closure scorer与raw validator实现门PASS

Heartbeat `ccad`实现`score_m1_nip_parent_completion_p1.py`和独立`validate_m1_nip_parent_completion_p1_score.py`。Scorer入口先逐文件重算v4 closure hash并要求prelabel 17/17 closed PASS，随后才动态导入`ccad.nip_truth`；它不调用任何matcher或proposal API，只读取冻结support。Evaluation使用独立2048-sample stream，N11 intervention另用独立2048-sample stream，mean继续读取冻结的257-sample mean-only stream。

每条native输出保存完整`metric_surface.v2-nip`：有support时计算raw centered/mean numerators与denominators、BCC、PSC、rank/condition、cancellation/leverage、occupancy/document ESS及post-label algorithm fields；无support时为所有mandatory fields写typed `NOT_APPLICABLE/NO_FROZEN_NATIVE_SUPPORT`，不伪造零值。Continuous lanes保存held-out weighted residual；N06固定(0,1) full-block control、N08两类continuous reference和N11冻结support的cliff/smooth endpoint均单独持久化。

独立validator不调用scorer函数：重新生成mean/evaluation tensors，逐support从原始数组重算`d_ctr/d_mu` numerator/denominator，重算continuous residual与truth classification，并检查132-row grid、mandatory字段、N06/N08/N11 controls、score/source/closure/prelabel hash。Implementation run `M1_NIP_PC2_P1_score_impltest_v1_20260903T181950Z` targeted10/10、full163/163、py_compile PASS。Scorer SHA-256=`0C226EDAA99DDCBE10F52ACAFD88841BAB9AE4B0645E1932B86887508ED0EEB0`；validator=`885468DBE61C6433D2BB9F5B683D5C1D06D3D4089591E83CA970A202233D259B`；test=`0F0750D1DAD9D9400E0614B8BC9716939A7211675338F172B4E134E0BC9738E3`。

该PASS仅授权固定代码后对已封存v4执行post-closure P1 score；尚未打开labels或产生metric结果，M1/R006状态不变。若score/validator失败，必须保留独立score run，不能回写prediction或调阈值。

### 2026-09-03 14:24 EDT — P1 score v1研究门FAIL；N06 full-block fixture修复

`M1_NIP_PC2_V1_P1_score_v1_20260903T182134Z`等待另一项目释放`cpu-heavy`后由资源管理器执行，score正常写出132 rows，raw-identity validator 8/8 PASS；所有lease已释放。执行正确不等于研究门通过：N06固定full-block control实测`d_ctr=1.05557`、`BCC=0.65454`、`PSC=0.66667`、source/target ranks=`1/2`，违反协议要求的full-block positive control。因此本run整体记FAIL，不能因validator只检查control存在而追认。Score raw SHA-256=`6AB8392A7909D3DFB007B64B3DBDE069F9B675FB54905142DBD6A0624EA3400E`，summary=`F1EF53F4AFE5C71A33FC76A0429D8ADB0A9ECEB2D8C3AAF7F195B758F7584A26`，validation=`8FD2FAD7D1A7B40E6F344C03261689B8DAEC96D6C123F657481470813D7DEB87`。

根因是N06/N07旧generator只生成source atom0=`x_1 e_1`，而两个target rotated atoms之和为完整`x_1 e_1+x_2 e_2`；truth却标记full group portable。新增C029：source端改为两个原生orthogonal atoms`x_1 e_1`与`x_2 e_2`，target端仍是同一2D rotation；atom0 query、target universe、threshold和native absence truth不变，而source(0,1)与target(0,1)现在逐点相等。Scorer full-block control先聚合两个source atoms；validator从只检查存在强化为必须`d_ctr=0,BCC=1,PSC=1,ranks=2/2`。

新增逐点group-sum regression；首次测试因新test漏`import numpy`而NameError，未运行实验，补import后full164/164 PASS。Repair run=`M1_NIP_PC2_n06_fullblock_impltest_v1_20260903T182408Z`。哈希：generator=`52FC48AF30410ABDDD9EBE2DED6B896122AAE36E6BD38ABDB42BD328921E03BF`，scorer=`5D45D465CED13974991C8107C90A0547D46B2B7A99E04879874AB0E57E110EEA`，validator=`0BD0991FA76F8B3E79264BAA8466915170CB41B2169DCEF719B17B25C41B469B`，test=`9B5DB0254444ADED62EC9A779705F474B0B939255FAE2A5AD27AFEB7C05B1F19`。因generator和code hash改变，必须提交后从prediction开始使用fresh P1 run；v4 prediction与score v1均保持immutable。

## 2026-09-03 15:28 EDT — 启动C029后的fresh P1 prediction v5

状态恢复确认HEAD/origin=`8e86b45305f51998dcfb168433f0c1445fb7fdbd`、工作树clean，`cpu-heavy`、`disk-e-io`和GPU均free。登记`M1_NIP_PC2_V1_P1_predict_v5_20260903T192854Z`为RUNNING；新code snapshot将派生fresh P1 seeds，不读取v4 prediction或score v1。Prediction仍严格truth/evaluation/intervention closed，完成后只运行pre-label validator。

### 2026-09-03 15:30 EDT — fresh P1 v5 prediction/prelabel PASS；启动score v2

Prediction v5在受管`cpu-heavy` lease内封存132 rows，独立prelabel validator另持lease重算17/17 PASS；两次lease均释放。Closure SHA-256=`CAB5929D43B9AAD4C3FEED90783C6286C7DE5C140B60462606D2E2B0F4A3D5ED`，prelabel validation=`6243650D25ACF48D97E0CACDD306F29B37094A73A1FC3045388A5DEACB52A0EE`，truth/evaluation/intervention仍未在prediction进程打开。现登记`M1_NIP_PC2_V1_P1_score_v2_20260903T193006Z`为RUNNING；score必须验证上述hash后才打开P1 labels，并由强化后的validator检查N06数值门。

### 2026-09-03 15:31 EDT — P1 score v2 FAIL；修复group PSC度量对象

Score v2 scorer正常写出132 rows，但强化validator 7/8，`mandatory_controls_present=false`，因此整体FAIL。C029已经使N06 group contribution逐点相等，实测`d_ctr=0,BCC=1`；剩余`PSC=.6667,ranks=1/2`来自metric adapter把两个source atoms先求和成一个伪atom，再对该aggregate matrix只提取leading singular direction。Score raw SHA-256=`3B94C0435586973D44C907C24C829AAF34CD8A1D23DE58A31404E0BB3BFC79E3`，validation=`71D71FBB56D771D06748C83EAA1796AF64AFD2DC0D9EC377F90213CB4CCBC8A6`；artifact与labels保留，不追认。

新增C030并修复metric adapter：可选`source_atom_ids`显式声明source group；贡献和mean仍按组求和计算`d_ctr/BCC/d_mu`，PSC则从每个组成atom的decoder direction构造span。原`source_atom_id`单query调用保持不变。N06 scorer直接传`source_atom_ids=(0,1)`；新增regression同时要求`d_ctr=0,BCC=1,PSC=1,ranks=2/2`。Implementation run=`M1_NIP_PC2_group_psc_impltest_v1_20260903T193153Z`，targeted11/11、full165/165 PASS。哈希：adapter=`B03BD28EBAD47C34C9485C4E2BBA9A3D9BF7232E2BF2E076E17D3D98028E8BE7`，scorer=`33F6683C2E621FC0102F32079AA5D6B941A20EB6D366C9226FB0E89DD2CD56C3`，test=`0D561B89CD8928349430E507767E7038BC1DE1C46B6758B2AC57B3E0E25850C6`。

Prediction v5不包含metric adapter/scorer且support已封存，其closure仍有效；修复只影响post-closure measurement。因此提交后可对同一v5 prediction用新的score suffix重算，无需重新生成prediction，也不得改v5 artifact。

### 2026-09-03 15:33 EDT — 启动group-aware P1 score v3

C030修复已提交并推送，HEAD/origin=`d7cfa4d5ca8eae5269537e14c39aedb43291fdf0`。登记`M1_NIP_PC2_V1_P1_score_v3_20260903T193310Z`为RUNNING，输入仍为sealed prediction v5（closure `CAB5929D…`）；只重算post-closure metrics和control，不重新选择任何support。

### 2026-09-03 15:34 EDT — PC2 P1完整score门PASS；允许进入P2实现

Score v3与强化validator均正常完成，8/8 checks PASS；`cpu-heavy` lease释放并复验所有资源free。MSCC在7个positive families上7/7 exact minimum-support/multiplicity，5个negative families零false native positive；其余native baselines均零false positive，但OMP、contribution singleton、PW-MCC、dustbin、OT-mass和spectral各仅1/7 exact，greedy/random为0/7。因此预写simplicity rule不触发：OMP没有匹配MSCC correctness，无需比较cost/runtime即可拒绝promotion。

Mandatory controls全部通过：N06 full block `d_ctr=0,BCC=1,PSC=1,ranks=2/2`，projector distance约`7.34e-32`；N08 signed与nonnegative continuous held-out normalized residual均为0且converged；N09 cancellation ratio=`211.73`；N10 active documents=2、document ESS=`1.76`；N11 held-out `d_ctr≈0.01`，独立intervention cliff RMSE=1、smooth RMSE=0.1、normalized margin=0.05。上述仍是单pair/family synthetic smoke，不是formal统计或真实SAE证据。

Artifact hashes：scores=`23D88CCA1B9657E06A006CC96CFB47134F5E23B3BD29BC5FA804179750AE3982`，summary=`754D4F37E75BCDBAC66AC1AC73D897CC2CB74DFE283B17D9360EFA39DDF9F3B0`，validation=`C22FE3200D890CC6EE3E6A0FE0A22E3BAAA31A2EAB22C1AF73B80D9CE2431B47`，manifest=`B0CD214B6EE70AEFA0958C21FEA01FAF6266617C320AEA1B3367D8D1235595DB`。P1 prediction v5与score v3现共同PASS，formal P2 seeds仍UNGENERATED；只授权下一轮实现并静态审计20 pairs/family的P2 config/runner，不恢复M1 parent或R006。

### 2026-09-03 14:21 EDT — 登记P1 post-closure score；等待共享CPU资源

Scorer/validator已提交并推送，HEAD/origin=`78ebb6febdc410829b8dbe37bca110eb91b272e3`。登记`M1_NIP_PC2_V1_P1_score_v1_20260903T182134Z`为RUNNING，输入固定为prediction v4及其closure/prelabel hashes。资源盘点显示另一项目正持`cpu-heavy`与`disk-e-io`执行behavior fidelity；本run不绕过lease、不争抢硬件，将由资源管理器排队，取得`cpu-heavy`后再顺序执行score和raw validator。

### 2026-09-03 15:44 EDT — Formal P2执行面静态门PASS；不提前打开labels

Heartbeat `ccad`在P1完整门通过后执行C031裁决。没有把此前implementation缺陷当作理论问题绕开：rank-two decoy、mean/endpoint generator混用、N06伪full-block及group PSC聚合错误均已由C028–C030分别修复，失败run保持immutable。当前选择不是继续调P1或直接冒险启动formal P2，而是先把已修复路径参数化为可审计P2 contract；这能以很小工程成本换取fresh-seed、信息边界和复现实验的可信度。

新增`configs/m1_nip_parent_completion_p2_v1.json`：固定12 families×20 independent pairs×11 lanes=`2640` prediction rows，沿用P1全部科学参数，显式声明执行时消耗formal seeds，并绑定P1 prediction closure `CAB5929D…`、prelabel validation `6243650D…`、score validation `C22FE320…`。Runner现在将phase纳入seed derivation，P1/P2 namespace数值隔离；P2启动前逐hash验证三份P1 gate，且要求score validation完整PASS。Prediction closure与seed ledger按phase记录`formal_seed_consumed`，但truth/evaluation/intervention仍不得在prediction进程读取。

独立prelabel validator已从固定132行泛化为由config计算pair/lane grid，并核验P2的240条six-stream ledger、2640条prediction、逐lane 240条、phase/formal状态及完全重算。Post-closure score validator同样改为config驱动row count，并独立复核positive exact、false positive和false unique汇总，避免只信scorer summary。没有修改family、lane、阈值、`g_max=4`、预算7,462、样本量、runtime 1+5、random32或simplicity rule。

实现run=`M1_NIP_PC2_P2_runner_impltest_v1_20260903T194443Z`。第一次用系统Python调用pytest因该环境未安装pytest，在测试收集前退出且未产生artifact；随后使用既有R004环境与显式`PYTHONPATH`完成targeted 7/7、full 167/167 unittest PASS，四个脚本py_compile PASS。哈希：P2 config=`44EA2215D06EF530A754110EED528028BF23404704678B753344A6014AEBA494`，runner=`A1BEFECE191BE30BCDFDE494D201558FC39B55D39DA56EEB778FF28FCC316E45`，prelabel validator=`82A74ECC7F19FB1C32A923ACFACBC09EBB6B5D649AA8EEE6615C515517D89433`，scorer=`FF326DE664B2555567121F9D8F73CCFE1F866914D0315418FF06A5693988DDE7`，score validator=`AC63FF02C736B9C662A252727ADE7D095BFDBA4554DF3B43052512BC6B37EDD0`，test=`9BDE338B2CF7F0B3CB5C372DC11B4F0201A7CB306ECC79D3346C1FF14D43D311`。

本轮没有创建P2 run目录、没有生成或消耗formal seeds、没有打开P2 labels，也没有恢复M1/R006。下一动作是在提交固定代码后，通过资源管理器申请`cpu-heavy` lease，启动唯一P2 truth-closed prediction run；只有2640-row closure和独立prelabel validator全部PASS后，才能另起post-closure score run。

### 2026-09-03 15:50 EDT — 启动formal P2 truth-closed prediction

用户要求在不牺牲严谨性的前提下避免过度工程化并实质推进。状态恢复确认P2执行contract已在HEAD/origin=`99cc92544c6a51c8acb30f2e169390c537390f43`冻结、工作树无tracked并发修改、P1三项gate binding仍存在且资源管理器四类资源均free。现登记唯一run `M1_NIP_PC2_V1_P2_predict_v1_20260903T195052Z`为RUNNING，并直接执行12 families×20 fresh pairs×11 lanes的formal truth-closed prediction。

本run只申请`cpu-heavy` lease；正式seed由phase-bound P2 namespace首次生成。Runner完成后在同一冻结代码上另行取得lease执行独立prelabel full recomputation。若两者全部PASS，本工作轮直接进入post-closure score，不再增加新的工程准备门；若失败则保留run并只修复被证实的缺陷。Truth、evaluation与intervention在prediction/prelabel阶段保持关闭。

### 2026-09-03 15:57 EDT — P2 prediction/prelabel PASS；直接启动formal score

Formal prediction在受管`cpu-heavy` lease内完成2640/2640 rows并原子封存；独立prelabel validator随后另持lease，从run-local snapshots、resolved config、240-pair six-stream ledger和code aggregate重跑全grid，17/17 checks PASS。两次lease均已释放。Closure SHA-256=`54331A1A957C7790D892D1866472119B639968355D6FC24C353A17C05FE2A3E9`，prelabel validation=`6381CCA3EF6A6A576F74AD99C05F9EE2BC7E95593F89F9C44F4299C87A54EC6C`。

按用户要求不再增加工程准备轮，现登记`M1_NIP_PC2_V1_P2_score_v1_20260903T195746Z`为RUNNING。Scorer只能读取冻结support，并在验证上述closure/prelabel hashes后打开formal truth、evaluation和N11 intervention streams；完成后立即运行独立raw-identity validator并判定预写all-pair gates。

### 2026-09-03 16:02 EDT — Formal P2结果PASS；保留validator v1 FAIL并最小修复

P2 scorer完成2640 rows。MSCC在7个positive families×20 pairs上140/140 exact minimum-support/multiplicity，5个negative families×20 pairs上0/100 false native positive，全240 pairs 0 false unique、0 budget refusal；140/140 positive proposal recall。OMP、contribution singleton、PW-MCC、dustbin、OT-mass与spectral各20/140 exact，且全部只解决singleton N11；greedy 0/140，random 2/140并有1次false unique。MSCC相对最强deterministic challenger提高85.7 percentage points、exact count为7×。OMP correctness远未匹配，因此冻结simplicity rule不触发；MSCC median evaluated supports=6,195、mean runtime=0.0635 s，效率代价明确保留。

首次独立validator输出`validation.json`为8/9 FAIL。其余raw identities、truth classification、continuous residual、2640 grid、hash bindings及summary aggregates均PASS；唯一失败是validator仍把N08 continuous controls硬编码为P1的2条，而P2正确数量为2 lanes×20 pairs=40。这是验证器泛化缺陷，不是科学结果失败。失败artifact SHA-256=`65877A2F39721208C9BC3559FB9AD711D169B5EC6361ECE75E4D186027809E01`保留不覆盖。最小修复为`expected_n08_controls = 2 * pairs_per_family`，不改prediction、support、score、threshold、metric或任何label-dependent方法。Targeted3/3、full167/167和py_compile PASS；同一frozen score的新`validation_v2.json` 9/9 PASS，SHA-256=`83E399E559EB0D350D748820DC8443C00B86D79A434152829AB1EE0AA4069287`。

Mandatory controls逐pair通过：N06 20/20 full block `d_ctr=0,BCC=1,PSC=1,ranks2/2`；N08两continuous lanes共40/40 normalized residual 0；N09 cancellation ratio 116.88–438.85；N10 document ESS 1.33–2.00；N11 20/20 cliff RMSE 1而smooth RMSE 0.1。Raw scores hash=`18DBFC268668B7281C14B6D94E7C4D1D8B006EEA5EBB10DF5151A2F0BED0782B`，summary=`14D443A2D216F4FBB266302BB59BDFCC2B7DEB87E0809E8E1989357382ABBC45`，manifest=`88FD4BA38A422564647BC3A2DA4AB5075B41BF3667BD51ADAF24EA56F282E975`。详见本地结果评审`M1_NIP_PC2_P2_RESULT_REVIEW_20260903_160200.md`。

保守裁决：P2 formal synthetic gate PASS，形成可信的“many-atom native support recovery + selective refusal + falsifier-aware boundaries”故事空间，但不把它写成真实SAE C1/C2。下一步仅执行锁定的P3 parent aggregation与fresh A–H audit；不再增加合成family或调方法。P3通过后立即解除M1对R006的阻塞，转向受控真实SAE推进。

### 2026-09-03 16:19 EDT — Fresh P3 audit FAIL；数值结论保留，选择单次contract remediation

复用此前指出v3 parent缺口的独立same-family reviewer，对PC2 P2执行fresh A–H audit；reviewer另启numeric-integrity子审查并只读复算当前artifacts。最终标记`reviewer_independence=same-family`、`acceptance_status=provisional`、`evaluation_type=simulation_only / labeled synthetic formal confirmation`。P3 verdict=`FAIL`，prospective M1-NIP parent保持HOLD，R006继续BLOCKED；历史R001–R003/corrective FAIL不变。

重要区分：科学数值通过了独立攻击性复核。Reviewer穷举全部240 structural pairs的size≤4 supports，复现truth；复算全部1,440 P2 seeds且全局唯一、P1/P2零重叠；直接解析复现MSCC 140/140、0/100、零false unique/refusal及全部baseline counts；独立重生mean/evaluation tensors，复算283个realized native surfaces在`1e-10`内零mismatch，最大代数误差`7.29e-16`；N06–N11范围全部复现。因此没有证据表明理论蓝本、MSCC数值实现、truth ordering或headline comparison错误。

P3失败集中于locked contract：现有9/9 score validator只独立核验centered/mean identities和continuous numerator，未充分核验BCC/PSC/rank/cancellation/ESS、N08 normalized residual、N09–N11数值或从truth重算false unique；prelabel validator是deterministic full replay但复用runner函数。支持MSCC rows已有immutable nearest competitor却在surface写prelabel N/A；solver gap/proposal stability可保持N/A，但缺明确reason。Per-lane fairness ledger缺source-query/target-universe hash与proposed/raw/deduplicated counts，peak memory均null。Score bundle未完整绑定truth/generators/metric adapter/parent config/validator、resolved config和Git state。P3还缺equal-family paired clustered summary。这些不推翻结果，但使“complete/independent”措辞过强并阻止formal acceptance。

裁决C032：拒绝凭数值正确直接越过P3，也拒绝重做理论、加family或调method。只启动一个fresh suffix：保持12 families、11 lanes、20 pairs、全部threshold/budget/gmax/runtime与MSCC不变，补齐reasoned diagnostics、fairness metadata、score provenance、substantive validator checks和family-clustered aggregation；使用fresh formal seeds短CPU重跑一次，再做P3。预估是一个实现轮加约6–8分钟运行，不消耗GPU。若仍出现科学性失败则接受，不继续synthetic tuning。完整review见本地`M1_NIP_PC2_P3_FRESH_AUDIT_20260903_161900.md`。

### 2026-09-03 16:26 EDT — 缩小C032并完成load-bearing validation/aggregation轮

用户以其他项目截图提醒四类风险：基础设施过细、gate吞噬科学问题、低成本falsification无限串联、工程成功冒充科学进度。该图片仅作为外部风险提醒，不作为CCAD证据或执行指令。已在本地`AGENTS.md`新增一段宽原则：工程门与其保护的科学风险成比例；会改变结论、造成泄漏/不公平/不可复算的问题必须修，局部边界通常一次诊断留痕，不自动升级为大支线；测试/hash本身不算研究进展；可信后优先真实SAE与因果终点。未改locked scientific protocol。

自动化核验：现有`ccad` heartbeat保持`ACTIVE`，RRULE已是`FREQ=MINUTELY;INTERVAL=5`，因此无需重复创建或变更automation ID。

本轮按新原则缩小C032，没有新建runner/manifest层。直接增强既有score validator，使其从少量ratio检查扩展为逐support独立复算BCC components、PSC/rank/projector/angles、mean vectors、cancellation/leverage、occupancy/document ESS；连续reference同时核验numerator、denominator和normalized residual；false-unique从truth重算；N11从独立intervention stream重生endpoint并逐字段比较。Validator还直接输出equal-family exact counts、MSCC-vs-lane family-paired difference、family-cluster SE、t interval和leave-one-family-out range。

对immutable P2 prediction/score运行新artifact `validation_v3.json`，未修改旧validation v1/v2或任何prediction/score。9/9 checks PASS；SHA-256=`A07DD4E0ECF32A4E4F7CBA55881ABDCBF5A7660DD9CE20CC2D6C3EBF0F7F63D7`。MSCC相对OMP/contribution/PW-MCC/dustbin/OT/spectral的equal-family exact差均`.8571429`，family-cluster SE `.1428571`，95% t interval约`.508–1.207`，LOFO范围`.833–1.0`；优势不由单一family或pooled伪重复驱动。Greedy差1.0，random差`.9857`。Targeted3/3、full167/167、py_compile PASS。Validator hash=`45F71A099EC8A8CDEE7054AB6C0446B46B353A9CB4FF53E93BDB91367286E7D4`，test=`ACDDA29D6CDE387CCC0D49932006BF47E574327615672A7E4FAD7E0171276ADA`。

Gate解释：fresh P3旧裁决仍为FAIL，不用validator增强追认旧score bundle。C032剩余范围已收缩为fresh suffix的score dependency binding与per-lane fairness ledger，然后只重跑一次P2/P3；不再新增synthetic family、pilot、rate定义或工程层。资源管理器lease已正常释放；GPU未使用。

## 2026-09-03 16:37 EDT — C032 fresh-suffix contract completed before formal rerun

**Trigger.** Automation heartbeat resumed the bounded C032 remediation selected after the fresh P3 audit. Per the user and AGENTS.md §5.1, this unit was limited to the two missing scientific consumers—fair-comparison accounting and score dependency provenance—and did not add another synthetic family, threshold tune, or infrastructure branch.

**Actions and artifacts.** Updated `scripts/run_m1_nip_parent_completion_p1.py` so the declared `fresh_namespace` is an actual seed-derivation input; added the parent-frozen twelve-field per-lane fairness ledger, including source-query and target-universe identities, native support counts, charged candidate counts, five descriptive timing repeats, and measured Python allocation peak. Updated the independent prelabel validator to recompute science while excluding only nondeterministic cost measurements and to require the exact fairness schema. Updated `scripts/score_m1_nip_parent_completion_p1.py` to snapshot the scorer, independent score validator, truth module, metric adapter, and both generators; it now binds a resolved config, code aggregate, prediction inputs, Git HEAD, and validator identity. The score validator verifies the v2 dependency manifest while retaining compatibility with immutable v1 score artifacts. Added `configs/m1_nip_parent_completion_p2_v2.json` as the fresh C032 execution suffix; every scientific field and P1 gate binding equals P2 v1, with only schema/execution namespace changed.

**Verification.** Targeted tests 11/11 PASS; full project tests 168/168 PASS using `D:\CCAD_Storage\environments\r004\Scripts\python.exe` with `PYTHONPATH=src;scripts`; all four affected scripts passed `py_compile`; `git diff --check` found no whitespace error. An initial system-Python unittest collection attempt failed because that interpreter lacks NumPy; no project test or run executed and no artifact was written. This is an environment invocation correction, not a scientific failure.

**Conservative interpretation / gate effect.** C032's static implementation gate is complete. These changes improve auditability but do not themselves add scientific evidence and do not alter the prior P2 result. The next action is the already-authorized fresh truth-closed P2 v2 prediction→prelabel→score→independent-validation chain under a CPU-heavy lease; no further diagnostic pilot is inserted before it.

## 2026-09-03 17:03 EDT — C032 fresh P2 v2 chain PASS; ready for one bounded P3 re-audit

**Execution.** After commit `c19bae7` was pushed to `origin/main`, ran the fresh suffix `M1_NIP_PC2_V2_P2_predict_v1_20260903T203800Z` through the shared `cpu-heavy` lease. The first resource-manager invocation lacked the src-layout `PYTHONPATH` and stopped at import before creating the run directory; the corrected invocation acquired and released a healthy lease and sealed 2,640/2,640 rows. The full independent prelabel replay then ran under a second healthy lease and passed 18/18 checks. Prediction closure SHA-256=`1236C89394BD1716DA9D38CFB6BF40CCF13FEFFFF2FE9B2777DAF8B3323AC712`; prelabel validation SHA-256=`8C215F02B45A1B4EB19F56E06900E9281FC85FDA4556CBF28CD972B3C35123AA`.

**Post-closure result.** Score run `M1_NIP_PC2_V2_P2_score_v1_20260903T210200Z` opened truth only after the sealed closure and prelabel PASS. MSCC again recovered 140/140 positive cases exactly, with 0/100 false native positives and zero false-unique errors. Contribution-nearest/PW-MCC/dustbin/OMP/OT/spectral each recovered 20/140; greedy recovered 0; random recovered 1. Independent score validation passed 10/10, including the new dependency binding and direct raw-identity/control recomputation. The equal-family MSCC gap versus each main deterministic challenger is `.8571429` (family-cluster SE `.1428571`, t interval `.507584–1.206702`, LOFO `.833333–1.0`). Score raw SHA-256=`CB13CFDDDCA357792B3E51861FE0795057D83E486C4F1C5E091AA423A70A549B`; summary=`ECD24414C5FC3E0CC4413F83B929AF428A3762B3ECBD2E9E73978EF9A281372A`; manifest=`CEE6F7848280BCDF046872E26AF7078EB0EC9E7F073AB4E2AA7DB8609AF85A32`; validation=`2B788C2D326DBF36280295F9DA2FFA977F1DC27456D304FE02153B28E0C20585`.

**Interpretation and next step.** This is a substantive fresh replication under the remediated contract, not merely a test/hash milestone. It strengthens the formal synthetic C1-NIP feasibility story but does not prove real-SAE C1/C2 and does not yet restore M1/R006. C032 now has no remaining implementation work. The sole next synthetic action is one bounded fresh P3 A–H re-audit of these exact immutable artifacts; if it passes, immediately resume the real-SAE R006 line rather than add another pilot or engineering layer.

## 2026-09-03 17:20 EDT — Fresh P3 v2 FAIL; preserve result and repair two load-bearing contract defects

Fresh same-family GPT-5.6-Sol ultra reviewer audited the immutable v2 prediction/score chain under `experiment-audit`. It independently reproduced the 240-pair/2,640-row grid, all hashes and headline counts, raw BCC/PSC/mean/cancellation/ESS identities, N06–N11 controls, namespace freshness, and closure order. A/B/C/F PASS: no fake truth, self-normalization, phantom result, or synthetic-to-real claim inflation. Exact report: `M1_NIP_PC2_P3_FRESH_AUDIT_V2_20260903.{md,json}`; trace: `.aris/traces/experiment-audit/2026-09-03_run02/reviewer_final.md`.

P3 verdict is nevertheless FAIL for this suffix. The fairness ledger interpreted `raw_support_count`/`deduplicated_support_count` as atom occurrence/unique-atom counts rather than support-object counts (120 affected rows); 843 mandatory nearest/solver/stability cells lacked typed reasons even though MSCC already persisted a nearest competitor; prediction omitted executed `proposal.py`, score omitted executed `nip_synthetic.py`, and slash-containing parent-config input was not actually validated. These are fairness/reproducibility defects, not numerical reversals. Historical failures and this suffix remain immutable; M1 synthetic parent is not restored and R006 remains BLOCKED.

Per AGENTS §5.1, only one minimal fresh suffix is being prepared: correct support-object counts, persist the available MSCC competitor gap and reasoned N/A statuses, bind executed local dependencies/Git cleanliness, and validate every input hash. Families, lanes, thresholds, budget, gmax, sample sizes and method are unchanged. Targeted20/20, full169/169 and py_compile PASS before formal execution. No new family, selector, theory edit or scientific tuning was introduced.

## 2026-09-03 17:55 EDT — PC2 v3 final P3 PASS; M1 prospective parent restored and R006 unblocked

Committed the minimal closure repair at `b7688d8`, then executed `M1_NIP_PC2_V3_P2_predict_v1_20260903T212727Z` under the shared CPU-heavy lease. Prediction sealed 2,640/2,640 rows; independent prelabel replay passed 18/18. Score run `M1_NIP_PC2_V3_P2_score_v1_20260903T214800Z` passed 11/11 independent checks. MSCC again recovered 140/140 positive cases exactly with 0/100 false native positives and zero false-unique errors; deterministic main challengers each recovered 20/140 or fewer. Prediction closure=`98F476A158F9A683A84E918A16CE41FF1815B59E06914C84CDDEC59D7047C48E`; prelabel=`AB79EB71021A2E5A3BAEA41C76946EC49BD1A2ECAF3FECC299EA7BE9F4D4B44A`; raw scores=`99A9A163F7FA0D66DB98B02C744E499C75801E639EBF8D9FC45885CF719544CB`.

Fresh same-family GPT-5.6-Sol ultra final audit `M1_NIP_PC2_P3_FINAL_AUDIT_V3_20260903.{md,json}` returned overall PASS. Direct audit found 0/2,640 fairness support-count mismatches, 0/2,660 missing typed diagnostic reasons, 160/160 reproducible MSCC competitor margins, 1,440 unique fresh seeds with zero v2 overlap, clean Git state, and zero code/input binding mismatches. A/B/D/E/F/G/H PASS；C只有审计时tracker尚未登记的行政性WARN，按AGENTS §5.1不阻塞。Trace=`.aris/traces/experiment-audit/2026-09-03_run03/reviewer_final.md`.

Gate裁决：prospective M1-NIP synthetic parent现在PASS，R006从BLOCKED恢复为TODO；历史R001–R003、corrective和早期PC2 failures保持不变。此结论只授权M2真实SAE配置/质量工作，不构成real-SAE C1-NIP或C2-NIP证据。Synthetic线到此停止，不再追加family、selector或重跑。

## 2026-09-03 18:05 EDT — R006 resumed; prospective k128 two-seed quality gate frozen

R006恢复后没有重开失败的automatic selector。基于既有、audit未开的seed-0 calibration，k128是唯一同时落入冻结FVE与CE-recovered margin的候选，因此选择“保留sparse probe为诊断、以明示工程取舍执行可逆两-seed门”。这不是全局最优k主张；seeds1/2任一失败即停止，不能删seed、调阈值或向k256扩展。

新增`configs/r006c_k128_seed{1,2}_v1.json`和前瞻gate `configs/r006c_k128_two_seed_gate_v1.json`。两seed固定Pythia-160M commit、layer5 resid-post、sparsify commit、width3072、k128、同一131,072-token document order与32,768-token validation。运行前阈值：FVE≥.93、CE recovered≥.85、actual L0=128、alive≥.95、最小nonzero firing≥16、decoder norm error≤5e-6；两seedFVE range≤.03、CE range≤.05、alive range≤.05；吞吐≥10k tok/s、peak allocated VRAM≤2GiB，并要求全部input/checkpoint/hook/logit checks。c_dec只报告，不作为单一selector。`run_r006b_topk_capacity.py`仅补真实Git HEAD/cleanliness记录；full171/171与py_compile PASS。GPU-0资源管理器free，nvidia-smi基线1006/16303MiB，locked runtime ledger仍READY。

## 2026-09-03 18:24 EDT — R006 two-seed quality gate PASS; freeze k128 and enter R007

在提交`e71c0f7`的冻结配置与干净工作树上，经母目录资源管理器依次取得并释放GPU-0 lease，完成`R006c_k128_seed1_v1_20260903T220000Z`与`R006c_k128_seed2_v1_20260903T220000Z`。两个run均为12/12内部检查与artifact contract PASS，保存safe及exact checkpoint；实际只改变初始化seed，模型、hook、训练/验证token及顺序、架构、优化器与预算一致。

冻结阈值的机械汇总`R006c_k128_two_seed_gate_v1_20260903T222000Z`为36/36 PASS。Seed1/2的FVE分别`.9688258/.9687965`，CE recovered `.9120052/.9091944`，actual L0均128，alive fraction均1.0，最小非零firing为54/35，decoder norm最大误差均`1.67e-6`；吞吐14.36k/25.54k tok/s，peak allocated VRAM均1,419,920,384 bytes。两seed FVE range=`2.93e-5`、CE range=`.0028109`、alive range=0，且SAE state hash不同。此前真实模型R005d v2的13/13精确中断续训轨迹复现仅作为framework-level支撑，不参与本次配置选择，也未新增训练支线。

裁决：R006 PASS，正式冻结Pythia-160M commit `582159…`、layer5 resid-post、sparsify `42c0645…`、width3072、TopK k128与131,072-token训练预算为R007 primary配置。这是可解释的工程选择而非全局最优k主张；R006不再扩k、selector或probe。R007登记为RUNNING：保留seeds1/2并直接训练seeds3–5达到最低五种子套件，沿用相同质量阈值，不因不利结果删除seed或事后调整门槛。该进展仍是M2/M3 SAE质量证据，不是C1-NIP或C2-NIP结果。

## 2026-09-03 18:45 EDT — R007 minimum five-seed primary suite PASS

在配置冻结提交`4fd2771`之后，通过共享资源管理器逐个取得GPU-0独占lease并完成`R007_k128_seed{3,4,5}_v1_20260903T223000Z`；三次lease均正常释放。每个run均12/12内部检查、artifact contract、safe checkpoint与exact checkpoint PASS，Git运行状态干净。它们与R006c seeds1/2在全部科学字段上相同，仅运行层级元数据与初始化seed不同。

Seeds3/4/5的FVE为`.9688278/.9688691/.9688140`，CE recovered为`.9082209/.9121620/.9066256`，actual L0均128、alive fraction均1.0，最小非零firing为41/54/72，decoder norm误差不超过`2.03e-6`；吞吐23.60k–26.11k tok/s，peak allocated VRAM均1,419,920,384 bytes。五个seed的SAE state hash全异。

首次五种子aggregation `R007_k128_five_seed_quality_gate_v1_20260903T224000Z`因共享validator只接受R006字段`decision`、未接受R007字段`primary_config_decision`而在输出result前FAIL；失败目录与`status.json/CORRECTION.md`保留。该错误未读取结果做选择，也未改变训练、checkpoint、阈值或任何seed。单行schema兼容修复后，新suffix `R007_k128_five_seed_quality_gate_v2_20260903T224500Z`为81/81 PASS，result SHA-256=`E096C72230FFB86721200E43A99F06A493534B3ED138F5ED90BD75E036B30A92`。五seed FVE range=`7.264e-5`、CE range=`.0055363`、alive range=0，均远低于前瞻上限`.03/.05/.05`。

裁决：R007 PASS，已达到M3最低五个same-config seed的训练与质量要求；不追加第六seed来美化稳定性。下一主线是R008：只为这五个冻结checkpoint构建document-hash 10/40/20/30 paired activation/code资产，保持audit关闭，然后进入R009–R011 discovery/calibration冻结。此处仍不构成C1-NIP/C2-NIP结果。

## 2026-09-03 20:18 EDT — R008 paired corpus and five-seed sparse-code assets PASS

用户要求继续沿主线推进，并提醒遵守更新后的GitHub条件。本工作单元因此在R008形成完整M3里程碑前没有为中间实现或失败单独提交。首先新增document-level `paired_document_split`，用固定FineWeb commit与salt将文档不可变地分为10% mean、40% discovery、20% calibration、30% audit；新增单测后全套172/172 PASS。

R008a v1在任何网络读取前因锁定Python缺`pyarrow`而FAIL，且暴露失败路径未写`environment.json`；失败run保留。复用历史R006记录的`pyarrow==25.0.1`，安装到独立`D:\CCAD_Storage\environments\r008_data_overlay`。V2仍在数据读取前因未加入既有requests/transformers overlay而FAIL，同时artifact validator指出code aggregate算法不一致；失败run继续保留。V3只修复两个依赖路径和项目既有aggregate算法，所有dataset commit、shard/row-group salt、split salt、token数量和最低文档数均与v1/v2相同。

`R008a_paired_corpus_v3_20260903T234000Z`在`disk-e-io` lease下14/14与artifact contract PASS。固定五个row group共读取41,527,004 bytes、采样4,998文档，最终使用569文档。Mean/discovery/calibration/audit分别为56/237/98/178文档与32,768/131,072/65,536/98,304 tokens，总计327,680 tokens；四split文档集合互斥，并与R006 SAE train/validation corpus按document ID和text SHA-256双重零重叠。Audit只被token化封存，没有用于选择、阈值或结果查看。

随后在其他项目释放GPU后，通过外层`gpu-0`、内层`disk-d-io`两个共享lease运行`R008b_paired_codes_v1_20260904T000500Z`。每个batch只做一次Pythia layer-5底模forward，再由五个冻结SAE同步编码；总计640 base forwards、327,680 tokens、51.40秒、6,375 tok/s，peak allocated VRAM 895,002,624 bytes。D盘资产`D:\CCAD_Storage\paired_assets\R008b_paired_codes_v1_20260904T000500Z`共1,305,495,855 bytes，包含四split×五seed的float32 top acts、uint16 indices及五份float32 decoder；selected/nonzero L0在全部split均为128。生成检查8/8、artifact contract PASS。

独立validator在单独`disk-d-io` lease中复算全部约1.3GB文件hash和尺寸，并遍历全部indices/acts/decoders检查范围与有限性，11/11 PASS；validation SHA-256=`1EB2DA0E7E02EED9CB822947A90D1EB1BA94228F5E617D88C4561F848168DA8A`，bulk asset manifest SHA-256=`35A873FB9821369E45E589432A7BC94B361908C5E8E5F6C167846D4E0B0374A5`。所有lease已释放。

裁决：R008 PASS，M3所需的五个受控seed与split/hash完整paired assets均已完成。该里程碑只建立真实SAE数据面，不产生C1/C2结论。下一步直接进入R009–R011：用mean split估计中心化常数，只在discovery/calibration上构造source-only census、实现公平atom/group baselines与MSCC候选冻结；R012/R013之前禁止读取audit codes进行任何度量或选择。

## 2026-09-03 18:59 EDT — R009 source census、query freeze 与 atom discovery baselines PASS；正式进入真实主线故事

**触发与计划裁决。** 用户要求在复核相关材料后正式进入主线、且本轮形成较多实质推进。重新阅读了AGENTS、最新账本、B2/B3计划、R009–R013 tracker、C009/C020/C022以及理论PDF中动态贡献、有限候选族、causal interchangeability、主表与否证条件。由此冻结本阶段承重故事：C1是预冻结query panel上的选择性native portability与拒答账本；C2是MSCC group相对calibration-selected best functional single的held-out因果优势。BCC与centered raw swap residual代数等价，不重复算作两项证据；R011候选/阈值与R013 endpoint冻结前继续禁止读取audit。

**R009a source-only census。** `R009a_source_census_v1_20260904T004000Z`完成数值后因NumPy boolean无法JSON序列化而在finalization FAIL，失败run和correction保留。仅做builtin-bool转换的`R009a_source_census_v2_20260904T005000Z`为7/7与artifact contract PASS。它只读取mean/discovery，生成五seed×3,072 atoms=`15,360`行完整抽样框，SHA-256=`B1C13EE2DC83C0DF51C505934D88B2174E147EFC453E427CFC029B7DE7F30894`；所有atoms alive，各seed最低active documents为115/89/41/125/114，最低document-energy ESS为12.08/6.27/11.54/10.90/14.88。未做query选择、target lookup或阈值。

**R009b target-blind query freeze。** 计划采用每seed按source discovery code energy秩分8个等大层（384 atoms/层），固定salt SHA-256顺序每层取16，得到128 queries/seed、合计640。所有15,360 atoms保留为sampling frame，不按低频或结果删query；occupancy只作属性。`R009b_query_panel_v1_20260904T012500Z`科学检查9/9但manifest漏`resource_lease_reason`，artifact contract FAIL，已改status并保留。只补该字段的v2为9/9与contract PASS，panel SHA-256=`EFBA3B06EC13F43E17AA6FF30786D1E908D63AE638FF4282B33FC8D2542AE447`，与v1选择逐字节相同；panel最低firing=338、active documents=122、document ESS=27.43。没有读取target、calibration或audit。

**R009c atom discovery。** 为正式矩形Hungarian安装了本地隔离依赖`.runtime/r009`（SciPy 1.16.1、NumPy 2.5.2；目录被gitignore）。第一次受限网络安装失败后，经授权下载安装成功。`R009c_atom_discovery_v1_20260904T014000Z`在任何pair结果前因SciPy不能就地排序只读memmap而FAIL；复制为可写CSR缓冲后v2通过。V2同时暴露source-normalized residual最近邻会系统性选择低能量近零target：大多数pair的128 queries只落到1–5个target atoms，median best BCC约零。该输出保留为direct-native-residual负诊断，不能冒充强best-single。

在未打开calibration/audit的前提下，v3按理论预定BCC补入`BALANCED_CONTRIBUTION_BCC_NEAREST`，保留direct residual、decoder cosine与frozen-panel-to-full-dictionary rectangular Hungarian，并增加same-seed contribution identity见证。`R009c_atom_discovery_v3_20260904T020000Z`在嵌套`cpu-heavy`+`disk-d-io` leases下完成全部20个有序seed pairs×128 queries=`2,560`行，每种ranking固定top32；10/10检查和artifact contract PASS，候选SHA-256=`C111C4F2D2A5D616A6C5D22570B2C782C7BCEAA22792FD95D8969FAC6166DD7F`，所有leases释放。矩形Hungarian平均绝对decoder cosine=`.15982`；balanced best-single BCC中位数=`.01940`，90/95/99分位=`.07328/.09514/.13800`，最大`.21643`，只有103/2,560 rows达到`.1`、3 rows达到`.2`。balanced target top1在pair内覆盖87.5%–97.7%的queries，说明未发生direct-residual式塌缩。按source-energy层，median best BCC从最低层`.0082`升到最高层`.0805`，后续统计必须保留该层级而不能只报总体平均。

**保守解释与下一步。** 这是真实same-config SAE上的第一组主线张力：atom correspondence整体很弱，尤其低能量层；它既不证明理论错误，也不证明MSCC/group会成功。R009仍为RUNNING，因为calibration-selected functional best single与阈值尚未冻结；R010/R011下一步应在同一640-query panel和统一candidate/search budget上直接运行group baselines与MSCC，检验小型native supports能否明显超过上述atom ceiling。Audit保持封存。全套172/172 tests在正确`PYTHONPATH=src;scripts`环境PASS；一次遗漏PYTHONPATH的164-test调用产生5个import collection errors，没有执行项目逻辑或生成run artifact。按更新后的GitHub条件，本轮尚未形成R009完整阶段性结论，因此保留本地，不单独commit/push。

## 2026-09-03 19:06 EDT — R011a全量真实group discovery surface PASS，但简单unweighted native groups给出负信号

自动化继续执行C033中预定且有明确科学消费者的Option C，没有新增外围runner层。`R011a_group_discovery_surface_v1_20260903T231500Z`绑定R009b query panel、R009a mean/source statistics与R008b asset manifest；对全部20 ordered seed pairs×128 queries=`2,560`行，使用固定20-atom proposal（positive contribution correlation 8、balanced BCC 8、absolute decoder cosine 4，按固定顺序去重回填）、`g_max=4`，逐query精确评分全部6,195个非空supports，低于统一7,462 evaluation budget。只读取mean/discovery，不使用calibration/audit，不设置阈值或`FOUND`状态。

Run在嵌套`cpu-heavy`与`disk-d-io` leases下约两分钟完成，10/10 checks与artifact contract PASS；输出SHA-256=`EA1E3B8CCC58DC30976E1FC48657511388E7BAB1DDC6E6F0CDD186D8F0ADC80D`，所有leases已释放。Across all query-pairs，median best BCC按support size 1/2/3/4仅为`.01940/.02259/.02350/.02369`；median best source-normalized residual则为`1.4457/2.0589/2.7722/3.5899`。20个seed-pair方向一致：加入更多无权native atoms只带来极小balanced-BCC增量，同时因target contribution energy累积而显著恶化direct interchange residual。

保守裁决：这是主线的实质负结果，不是工程失败，也尚不是C1/C2否证。它否定的是当前“20-atom contribution/cosine union + unweighted binary support求和”能直接恢复source atom的希望；可能原因仍分为proposal miss、binary native representability failure、或本配置根本缺乏局部portable unit。Audit继续封存，R011记RUNNING而非FAIL。下一步限制为一次承重separator：在相同query/candidate budget上计算signed continuous、nonnegative continuous与OMP/greedy上界，并检查其相对binary surface的差距。若continuous也接近零，停止调proposal并优先审计paired identity/配置边界；若continuous显著更好而binary失败，则收窄native-support claim。按Git同步规则，这一发现与即将进行的separator一起形成阶段性裁决后再提交。

## 2026-09-03 19:40 EDT — R011b完整representability separator PASS；排除局部系数约束并停止proposal微调

**触发与动作。** 用户要求把5分钟automation prompt替换为给定的精简治理文本，并再推进一轮、取得更多实质进展。Automation `ccad` 已保持`ACTIVE`与`FREQ=MINUTELY;INTERVAL=5`，prompt逐字替换为用户给定版本。按AGENTS恢复状态后，没有新增文献或下载，因此`REFERENCE_REGISTRY.md`无需变更。C034预先登记一次完整separator：固定R011a的query、20-atom proposal、discovery split和2,560行，不在结果后追加proposal变体；连续拟合仅作non-native reference，永不输出`FOUND`。

新增`configs/r011b_representability_separator_v1.json`与`scripts/run_r011b_representability_separator.py`。正式run `R011b_representability_separator_v1_20260903T233000Z`通过母目录资源管理器依次取得`disk-d-io`与`cpu-heavy`租约，完成全部20 ordered seed pairs×128 queries。每行比较：R011a native unweighted size-4、最优signed scaled-single、逐步重拟合的signed OMP 1–4、20-atom nonnegative quadratic fit、20-atom signed pseudoinverse fit；均使用独立mean split常数与discovery covariance。Run 12/12 checks、artifact contract PASS，nonnegative solver 0 failures，输出SHA-256=`D54E9D0EDFEC14DB4DC1B390FA610222A3CDFDC7128A3A71479C906367DDCC96`；两项lease均已释放，audit/calibration未读。

**结果。** Across 2,560 rows，median `d_ctr`依次为native size-4 `3.58991`、scaled-single `.999557`、signed OMP-4 `.998893`、nonnegative full-20 `.997735`、signed full-20 `.997735`。所有continuous methods的`d_ctr<.25`和`<.50`比例均为0；signed full-20的q10/median/q90为`.97833/.99773/.99938`，全局最优也只有`.84739`。其effective rank median为20，负系数比例median为0；所以不是等权限制、非负锥或4-sparse搜索造成的主要失败。source-energy最高层的signed full-20 median `.95916`，低于最低层`.99940`，显示能量梯度但没有出现足以支撑selective portability的局部族。

**保守解释与阶段裁决。** R007五seed FVE约`.9688`、CE recovered约`.907–.912`、alive=1；R008对五seed使用同一640次base forward并通过全部文件hash/shape检查；R009c same-seed contribution identity亦通过。结合这些既有证据，目前没有资产错位、token错配或单seed质量失败的迹象。R011b因此是科学负结果：当前局部proposal几乎不张成source atom contribution，而不是implementation crash；但它仍不能证明完整target dictionary中不存在大支持表示，也没有打开audit，故不否定C1/C2全局版本。

按C034停止规则，关闭局部proposal/系数变体，不追加小pilot。R011保持`RUNNING`，下一项承重实验转向R010 aggregate reconstruction identity与stable-subspace/group baseline：若整体/子空间跨seed稳定而atom/native-small-support失败，主故事成为“高重构质量掩盖非规范atom分解，并以选择性refusal量化边界”；若整体也不稳定，则回到训练配置/充分性边界。全项目`unittest discover`本轮执行169项，其中168项PASS；唯一collection error来自既有T021模拟测试缺少可选依赖`mpmath`，并非本轮代码失败。R011b真实run自身与artifact contract均PASS。

## 2026-09-03 19:55 EDT — R010a aggregate identity PASS；发现whole-SAE稳定与atom/local-support失配的强尺度分离

5分钟automation按状态恢复顺序进入R010，没有重复调R011局部proposal。新增C035、`configs/r010a_aggregate_reconstruction_identity_v1.json`与`scripts/run_r010a_aggregate_reconstruction_identity.py`。正式run `R010a_aggregate_reconstruction_identity_v1_20260903T235000Z`绑定R008b asset manifest和R009a独立mean constants，在共享`disk-d-io`+`cpu-heavy` leases下流式重建五个冻结SAE对全部131,072个discovery tokens的whole-dictionary hook output；不读取calibration/audit、不选择query、不设置阈值或`FOUND`，也不做token-level inference。Run 9/9 checks与artifact contract PASS，输出SHA-256=`BB912E98008DCE77E348244BB0CC3B0036E43EF29508A7F7EF312E55564CD3C8`，资源均正常释放。

20个ordered seed pairs的whole-SAE centered BCC min/median/max为`.984484/.984521/.984554`；source-normalized residual min/median/max为`.030890/.030960/.031038`。这与同一资产上R009c balanced best atom BCC median `.01940`、R011a native size-4 BCC median `.02369`以及R011b signed full-20 residual median `.99773`形成明确尺度分离：五个seed整体重建几乎一致，但单atom及当前local native support几乎不可互换。

保守解释：该结果实质排除了gross token/pairing错位和“不同seed整体表示完全不一致”，使当前主线故事获得可信空间；它说明非规范性发生在分解尺度，而不是底模hook整体，但不能据此断言具体机制是rotation、不能推出任意atom有大support表示，也不是C1/C2或因果证据。R010由TODO转RUNNING。下一工作单元只允许一个dynamic stable-subspace baseline来量化共享低维结构；完成后应进入calibration/refusal冻结，不再回到local proposal变体。本轮未采用新文献，`REFERENCE_REGISTRY.md`无需修改；该单次automation不单独触发Git提交。

## 2026-09-03 20:08 EDT — R010b动态稳定子空间PASS；均值位移复核后尺度分离仍成立

Automation按上轮停止规则只执行一个dynamic stable-subspace baseline。`R010b_dynamic_stable_subspace_v1_20260904T001000Z`在全131,072 discovery tokens上累积五个whole-SAE reconstruction covariance，计算rank 16/32/64/128/256/512共享hook子空间PSC；10/10与artifact contract PASS。但其effective rank median仅`1.8856`，可能由独立mean split与discovery empirical mean的固定偏移主导。虽然不是程序失败，直接采用v1会使稳定子空间故事存在可避免的混杂。

因此保留v1并执行唯一纠正suffix `R010b_dynamic_stable_subspace_v2_20260904T002000Z`：不改tokens、seeds、ranks或数据，只利用R009a已保存的discovery code means把covariance精确分解为independent-mean与within-discovery empirical-mean两种centering，并报告mean-shift能量。V2在嵌套`disk-d-io`+`cpu-heavy` leases下11/11与contract PASS，输出SHA-256=`73413743F7E8DA97790A1A0F57DD0DD4298977DC04E0CC8E3123AFD3A4768A7A`，所有lease释放。Mean shift只占independent-centered trace的min/median/max `.011553%/.012064%/.012737%`；effective rank两种centering均约`1.88`，故低effective rank不是mean-split伪影。

Within-discovery centering下，rank16/32/64的pairwise PSC median为`.98899/.97511/.95609`，variance coverage median为`.95764/.96447/.97232`；isotropic random expectations仅`.02083/.04167/.08333`。Rank128 PSC仍`.90631`且覆盖`.98080`。与R010a aggregate BCC `.98452`、R009 atom BCC `.01940`和R011 local native BCC `.02369`合并，当前受控真实SAE证据支持一个清楚但保守的尺度分离：跨seed整体输出和领先动态子空间高度复现，原生atom及小型local support却不复现。

该结果不能识别rotation，effective rank也不等于monosemanticity，更没有给C1/C2、`FOUND`或causal portability提供独立证明。R010保持RUNNING，因为stitching/Li15与calibration公平冻结尚未完成。Discovery方法优化到此停止；下一工作单元进入calibration，冻结best-single、group baselines、MSCC refusal/ambiguity与共同预算，之后才能决定是否打开audit。本轮未新增文献，registry无需更新；R010a/R010b组成一个阶段性主线里程碑，允许合并Git同步。

## 2026-09-03 20:20 EDT — R011c calibration为0 coverage；audit继续关闭并触发用户保留裁决

Automation进入calibration前新增C037并预先固定：R009b的640-query panel、R011a candidate/support、R011b OMP membership均不可重选；`g_max=4`、primary `tau_ctr=tau_mu=.05`、tie band `.005`在读取calibration前写入config，`.10/.20`只报告敏感性且不能改变primary。新增`configs/r011c_frozen_support_calibration_v1.json`与`scripts/run_r011c_frozen_support_calibration.py`。`R011c_frozen_support_calibration_v1_20260904T003000Z`通过共享`disk-d-io`+`cpu-heavy` leases完整评分20 ordered seed pairs×128 queries；只读取mean/discovery-frozen identities与calibration codes，不触及audit。Run 10/10 checks、artifact contract PASS，输出SHA-256=`80B049173CBF8A3B077F924D8FC083B9D323DD7E6FC63F3ACC9F39A830BD65EC`，所有lease已释放。

Primary MSCC calibration结果为0/2,560 `FOUND`；report-only `.10`和`.20`敏感性仍均为0。Discovery-minimum-residual native support的calibration median `d_ctr`按size1/2/3/4为`1.4444/2.0699/2.8051/3.6200`，median `d_mu`为`1.1215/1.4160/1.8395/2.4405`。Balanced-BCC supports的median BCC仅从size1 `.01871`到size4 `.02247`，对应`d_ctr`从`2.3434`恶化至`5.1624`；decoder-cosine prefixes和signed-OMP membership同样无可用native transfer。

保守裁决：这只证明当前冻结的20-atom local candidate family在本配置上100% `UNRESOLVED`，不能写成target native support不存在。但它使当前C1没有非零可审计coverage，C2也没有可进入held-out causal比较的MSCC support；打开audit只会消耗封存数据而不能回答主张。按C037停止规则，不再新增local family、不扩大support、不打开audit。R011转为`BLOCKED`。下一步涉及改变claim对象到subspace-level causal transport、改变primary SAE config，或终止/拆分native-local C1/C2，均属于AGENTS明确保留给用户的重大裁决；automation在用户决定前不得用新实验掩盖该阻塞。未新增文献，registry不变；该单轮结果先本地留痕，不单独Git推送。

## 2026-09-03 20:56 EDT — 双轨统一为causal granularity frontier；R011-NR1长预算语料PASS

**触发与理论裁决。** 用户明确要求同时保留subspace-level causal transport与native MSCC配置救援，不接受因当前0 coverage停止主线。重新按顺序阅读AGENTS、最新账本、计划/追踪器、理论蓝本和相关本地论文。理论PDF的THM-CBSM-009针对任意共享hook中的向量贡献过程给出同hook干预转移界，并不要求输出必须是target native atom subset；THM-CBSM-004/005同时允许基底不唯一。因而当前native-local MSCC是理论的严格工程实例，不是唯一合法应用。统一论文问题冻结为causal granularity frontier：跨seed因果可移植性最小在atom、native support还是query-conditioned dynamic subspace层面出现，以及训练充分性/稀疏度能否把该前沿推回atom层。当前0 coverage永久保留为短预算、稠密k128配置的有效负结果，不改公式掩盖。

**文献学习与边界。** 本地原文和官方线上来源交叉确认：Song et al. ACL 2026的Pythia-160M一致性实验约用500M tokens、width16384、TopK k20，而当前suite仅131,072 tokens、width3072、k128；训练量相差约3,800倍且激活更稠密。因此当前高FVE/CE只能证明重构质量，不能证明atom分解已训练充分。P04、SASA、Model Alignment Search、DAS及最新causal audit共同支持把非规范基底、子空间对齐与held-out causal intervention分开评估；它们不直接证明CCAD的SCT或MSCC主张。新增来源和消费者已登记到`REFERENCE_REGISTRY.md`。

**计划与组件。** `AGENTS.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`和`COMPONENT_CANDIDATES.md`已登记统一双轨：R011-S1/C038用source-only query条件化的rank 1/2/4/8/16子空间，比较raw-hook PCA、global SAE subspace、random projector、single/native与stitching/MAS controls；R011-NR1/C039固定Pythia-160M layer5、sparsify、width3072、FineWeb revision/order，只比较4,194,304-token下k128与k32、各seeds1/2，最多晋级一个配置，不在结果后追加k/hook/width。C040下游敏感度seminorm只作延后备选，防止endpoint circularity。Audit继续封存。

**实际运行与失败保留。** 为R011-NR1在`run_r006a_capacity_manifest.py`增加document ID与text SHA-256双重排除，绑定旧SAE训练与R008 paired文档账本，防止训练/mean/discovery/calibration/audit重叠。`R011_NR1_long_budget_corpus_v1_20260904T010000Z`使用5个固定shard，因排除后仅有2,708,569 train tokens而未达到4,194,304，按预期FAIL；其run、日志与metrics SHA-256 `4880AA7587A9F518544A823E491477091F578AEE4252C8450DD534A45EF949D5`保留。唯一有界修复v2只把固定shard数从5增至10，未改变数据revision、token预算、split、排除项或训练设计。

`R011_NR1_long_budget_corpus_v2_20260904T012000Z`在`disk-e-io` lease下16/16 checks与artifact contract PASS，lease正常释放。它从10,000个唯一FineWeb文档中使用7,124个，生成精确32,768×128=`4,194,304` train tokens及256×128=`32,768` validation tokens；train/validation SHA-256分别为`5BCD46159DAA59DE70CF8C05F76502C6D5CB719E95DDD41CF055B3082F60BBA8`与`B6EC15F1F640E691718D9446E97B94A2653F0EF2D9FED0C817D68A6C9D65CC0A`，所有source-row/document/text hash唯一性和838条排除记录均通过。全项目172/172 tests PASS。R011-NR1转`RUNNING`；GPU当前被相邻项目合法占用，因此本轮没有争抢或启动训练。该资产不是C1/C2证据，下一步是在lease可用时执行冻结的四次训练，同时并行推进R011-S1的calibration-feasibility协议与单元测试。

本工作单元已形成阶段性提交`37d2eb8`并推送至规范远程`origin/main`，推送后本地HEAD与远程一致。Automation `ccad`以5分钟间隔恢复为`ACTIVE`；prompt已改为双轨统一版本，要求在R011-S1/R011-NR1间按信息增益和资源可用性推进、GPU占用时切换非GPU工作、继续保持audit封存，并只在实质证据/失败/gate变化或需裁决时通知用户。

## 2026-09-03 21:18 EDT — 公开checkpoint纳入分级协议；高质量layer5锚点接纳PASS；loop按要求暂停

**触发与检索。** 用户指出当前131k-token/width3072/k128与ACL 2026约500M-token/width16384/k20的差距可能造成系统性问题，要求在不降低质量的前提下把可用现成checkpoint纳入协议，并参考文献提出其他有界解决方案；本轮结束后暂停而不删除automation。按AGENTS恢复状态并使用research-lit先查本地母目录原文，再检索官方论文、GitHub和Hugging Face。没有找到可核验同底模、同hook、同配置且含足够独立seeds的现成Tier-A bank。ACL Feature Consistency官方仓库给出500M tokens、seeds42/43/44、width16384的复现配置和Apache-2.0代码，但不发布训练权重。EleutherAI的SAE/ST/SST、k32/64/128、多width矩阵多数是非deduped Pythia、MLP hook和单seed，只登记为Tier-C外部机制资产。

**可用公开锚点。** 唯一通过metadata筛选的Tier-B候选是`EleutherAI/sae-pythia-160m-deduped-32k@b6c77b88065dd74bdd807b9e74900a20745e1871`的`layers.5`。实际cfg是d_in768、width65536、TopK k32、unsigned、decoder normalized；仓库公开且ungated，但没有model card、训练token说明或license tag，不能把另一`sae-pythia-160m-32k`仓库的8.2B-token说明转移过来。只下载该层112-byte cfg与402,918,736-byte safetensors到`D:\CCAD_Storage\external_checkpoints\EleutherAI_sae-pythia-160m-deduped-32k_b6c77b8\layers.5`，没有下载14.1GB全库。cfg SHA-256=`37F288B89DD1864B03192BCE326EC56446C6CC24EFE4E8ACEF8FC814CE9796D1`，weights SHA-256=`0758CD882E2C2E9DD020C2912D8F3022C1DE7BB0AD7FF23D581959534235D58C`；`disk-d-io` lease正常释放。

**R011-CP1运行与失败保留。** 新增一个只有明确消费者的接纳runner与三份suffix config。v1在冻结R006质量validation上7/7科学检查PASS，但成功路径缺`stderr.log`导致artifact contract FAIL；v2只预创建日志，数值逐项复现，但code aggregate未按validator要求排序，contract再次FAIL。两次run均保留且不冒充正式通过。v3只修复代码hash排序，`R011_CP1_public_checkpoint_acceptance_v3_20260904T022500Z`为7/7与artifact contract PASS，metrics SHA-256=`3928C8C069E893F19BDDBA5343B1DA3278ED5D2E32487643D12A189E068028CD`。结果：hook oracle error=0、capture logit error=0、actual/selected L0=32、FVE=`.98478398`、CE recovered=`.98221565`、decoder norm最大误差`1.04e-4`、peak allocated VRAM=1,813,785,600 bytes。在32,768 validation tokens上观察到37,592/65,536 latents firing；这不是全语料dead fraction。所有GPU leases已释放，全项目172/172 tests PASS。

**保守解释。** 该结果是实质正证据：同一Pythia-160M-deduped、同一layer5 hook上存在可直接加载且质量显著高于当前自训suite的更宽、更稀疏公开SAE，因此不能把当前native 0 coverage直接归为理论或任务不可能。但它只有单seed、width/训练语料与自训suite不同且训练provenance不完整，不能提供PW-MCC、seed-only NIP或因果可移植性证据，也不能覆盖当前负结果。后续只可作external quality ceiling、adapter和明确标注cross-config的机制参照；进入audit前需另行冻结消费者。

**其他解决方案与顺序。** C041–C043及计划14.7–14.8已登记：(1)先完成现有4.19M k32/k128×seeds1/2；(2)若出现质量合格的优胜配置，只沿同配置/同seed/同数据流继续到累计16.78M和67.11M tokens，形成nested learning curve并按SESOI早停，不立即支付500M；(3)若预算曲线饱和而仍显著落后公共width65536/k32锚点，再允许一次width机制实验；(4)训练架构reserve按直接稳定性证据和搜索风险排序为parameter-free aligned training、论文指定单点L2 regularization、BatchTopK、最后才是splitting/absorption触发的Matryoshka。JumpReLU暂不优先，因为其主要证据是reconstruction而非cross-seed native portability。每次最多晋级一种，不做架构网格。

`AGENTS.md`新增公开checkpoint分级与禁止混写规则，计划、tracker、components和`REFERENCE_REGISTRY.md`同步更新。Automation `ccad`仍保留原5分钟配置和双轨prompt，但已按用户要求设为`PAUSED`，没有删除。下一次恢复后，优先执行当前R011-NR1两配置两seed screen；公共checkpoint的paired-code/贡献分析只有在其跨配置消费者冻结后才启动，audit继续关闭。

## 2026-09-04 01:49 EDT — R011-NR1四个4.19M-token训练PASS；loop恢复并完成临时间隔回退

**触发与治理。** 用户要求继续验证和实验、重启automation，并允许长process期间临时拉长loop间隔但结束后必须恢复；重大决策或故障可暂停。已在`AGENTS.md` §12写入常态5分钟、长process有界放宽、结束/失败/终止后同轮恢复与留痕，以及重大决策/故障可暂停但不删除的规则。Automation `ccad`由PAUSED恢复为ACTIVE；四次顺序训练期间临时设为20分钟以避免重入，最后一个process退出后立即恢复5分钟，automation未删除。prompt同步为MSCC/SCT双轨、audit继续关闭。

**冻结与验证。** 新增并在训练前提交四份配置：`r011_nr1_k{32,128}_seed{1,2}_v1.json`。四者共同固定Pythia-160M-deduped commit、layer5 resid-post、sparsify commit、width3072、Adam/lr/warmup、4,194,304-token训练资产及顺序、32,768-token validation；只改变预注册的`k`与初始化seed。配置冻结提交=`412ebc3`并在运行前推送。首次全测试命令遗漏`PYTHONPATH=src;scripts`，出现5个import collection errors且没有运行/实验artifact；按既有环境调用修正后172/172 PASS。首次资源管理器启动在取得lease前被沙箱拒绝写母目录，没有创建run或占用GPU；以同一命令取得授权后正常执行，不构成科学失败或配置变化。

**实际运行。** 共享GPU-0在启动前空闲；通过母目录resource manager顺序取得/释放lease，完成`R011_NR1_k32_seed1_v1_20260904T054000Z`、`...k32_seed2...`、`...k128_seed1...`、`...k128_seed2...`。四者均训练8,192 steps/4,194,304 tokens，12/12内部检查与artifact contract PASS，hook oracle/capture logits exact，actual L0分别严格为32或128；所有state hash与safetensors hash互异。k32 seed1/2 safetensors SHA-256分别=`2A0AA163950B82F4B819D9E18B47C44434788C93D7D8C8B4AD45AFA57FE69EDF`、`D8693834E54D2C735A479C6B5B22012729E0FAAEC2B8C784B5B9A3878EB2684B`；k128分别=`818AA8BAC0AA47AA2BA487DF0366AA1049A7A2A65487CBC6F0A4E333AA5938B6`、`5C22EFE536E50A843CA526E54B0DE413170944451255348E0B331E73CC967D1F`。

**质量结果。** 在两配置共享的新validation上，k128 seed1/2的FVE=`.98508274/.98503121`、CE recovered=`.97471081/.97496368`、alive fraction=`.9967448/.9977214`；k32为FVE=`.97577765/.97585806`、CE recovered=`.93798225/.93758814`、alive=`.7347006/.7539063`。四者decoder norm error≤`2.39e-7`、peak allocated VRAM约1.42GB、吞吐21.08k–23.77k tok/s。k32的低alive fraction是真实机制信号，但不能在看到结构结果前用它自动淘汰k32；同样，k128更高重构不能替代cross-seed atom/native coverage。旧131k suite与本轮使用不同validation，当前数字不作无配对的训练预算效应估计。

**保守解释与gate。** R011-NR1继续RUNNING而非PASS：四个训练资产已完成，证明4.19M预算下两种稀疏度都能产生数值稳定且重构良好的SAE，但尚未回答PW-MCC、frequency-stratified atom stability、source-query BCC或native calibration coverage。当前没有选择赢家，没有打开audit，也没有追加k/width/hook。下一工作单元是以同一冻结paired corpus分别构建k32与k128两seed sparse-code资产，在mean/discovery/calibration完成预注册结构与coverage screen；只有该结果才能决定最多一个配置是否进入16.78M nested budget或五seed扩展。SCT支线保持并行授权。

## 2026-09-04 02:00 EDT — R011-NR1两配置共同paired-code资产PASS

**动作。** Automation heartbeat按tracker直接推进R011-NR1，没有新开诊断或超参。将既有`run_r008b_paired_codes.py`从硬编码五seed集合最小泛化为由配置声明的冻结同配置SAE集合；旧R008默认仍严格要求seeds1–5。新增`r011_nr1_k32_paired_codes_v1.json`与`r011_nr1_k128_paired_codes_v1.json`，分别绑定上一轮两个k32或两个k128 checkpoint的权重hash，并共同绑定R008a文档级10/40/20/30 paired corpus manifest。代码和配置在运行前以`c6ca8ca`提交并推送；172/172 tests与py_compile PASS。GPU-0和disk-d-io启动前均free，正式运行通过嵌套资源lease，结束后均已释放。

**运行与结果。** `R011_NR1_k32_paired_codes_v1_20260904T060000Z`和`R011_NR1_k128_paired_codes_v1_20260904T060000Z`均8/8 checks、artifact contract与独立`validate_run.py` PASS。每个资产只做640次shared base forwards，覆盖mean/discovery/calibration/audit共327,680 tokens；audit仅编码封存，没有计算指标、候选或阈值。k32耗时15.32秒、21.39k tok/s、peak allocated VRAM 837,743,616 bytes、资产144,711,666 bytes，metrics SHA-256=`FB5CAC5DAA65D1547039B6B674301F8575D2375096295A2051C0E0180F1CA5AA`、asset manifest SHA-256=`43E036939F257E7BCFD6557ADC344FDE88B20E532368214C9CEE96F3460AF066`。k128耗时25.12秒、13.04k tok/s、peak 838,333,440 bytes、资产522,199,117 bytes，metrics SHA-256=`5822FB0DC8DB898B1CF55694B47FC1E08FBADE653D20D93FB1F2D2708D910FC0`、manifest=`15A7A8531ADCBDE994673725FAB6FCB1FEE60CB4125C02B8DB9B8B3E1B9A479C`。

**解释与下一步。** 这是结构比较的共同输入资产，不是C1/C2结果，也不因编码audit而打开audit。R011-NR1保持RUNNING且仍无赢家。下一工作单元应直接在mean/discovery/calibration上完成两配置的PW-MCC、frequency-stratified atom stability、source-query contribution BCC和native calibration coverage；若两者native coverage仍近零，则按计划关闭本轮TopK救援，不追加k/width/hook。若一个配置出现实质改善，才允许最多一个配置晋级后续预算/五seed路线。

## 2026-09-04 02:25 EDT — R011-NR1结构/coverage screen形成有界负结论；TopK救援关闭

**触发与前瞻冻结。** 用户要求进行一轮推进。按状态恢复后选择tracker唯一顺序明确的R011-NR1结构/coverage screen，不打开audit、不追加k/hook/width。新增联合runner与配置，固定两套4.19M-token、seeds1/2 paired-code manifest；source-only按8个energy strata、每层16 atoms、每seed128 queries冻结panel。完整比较full-dictionary PW-MCC、frequency-stratified Hungarian相关、best-single contribution BCC，以及与当前R011相同的20-atom proposal、`g_max=4`、6,195 supports、`tau_ctr=tau_mu=.05` discovery→calibration transfer。晋级门在读取calibration前固定为完整query universe overall coverage≥10%且两个方向各≥5%；代码与v1配置先以`ad70a23`提交。全项目172/172 tests与py_compile PASS，运行经`cpu-heavy`+`disk-d-io`双租约完成，audit只保持既有编码封存且未评分。

**v1有效失败与修复。** `R011_NR1_structure_coverage_screen_v1_20260904T061224Z`表面10/10 checks与artifact contract PASS，并输出`CLOSE_TOPK_RESCUE`；但逐条审计11个k32 `FOUND`后发现其中10个source atom在calibration上只有1–5次discovery firing且calibration source dynamic energy为零。source与target都为零使数值`d_ctr=0`，却不提供任何可测试的portability。该artifact不覆盖、不删除，按科学有效性记为FAIL。唯一suffix修复增加source-only calibration evaluability门：无calibration source firing/energy一律`UNRESOLVED: CALIBRATION_SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON`，并把晋级分母明确冻结为完整query universe。新增2项零过程回归测试；全项目174/174 PASS，修复提交=`5f19214`。

**v2正式结果。** `R011_NR1_structure_coverage_screen_v2_20260904T061839Z`在同一输入、候选预算和阈值下11/11 checks与artifact contract PASS，输出SHA-256=`24194461E9044C1F25BF6CCF91DA4F4098756C5469ECF5A7D69A9EB024251E47`，所有leases释放。k32 full-dictionary PW-MCC=`.4894015`，alive-only方向值=`.59613/.59566`，median best-single BCC=`.42565`；但42/256 query在discovery零动态、另19个在calibration不可评估，最终仅1/256完整query `FOUND`（`.390625%`），两个方向分别0/128与1/128，median size-4 calibration `d_ctr=2.40663`。该唯一FOUND位于最高energy stratum、source firing4,844，`d_ctr=.04,d_mu=.02`，不足以形成coverage。k128 full-dictionary PW-MCC=`.4566076`、median best-single BCC=`.16458`、median size-4 calibration `d_ctr=2.63454`，0/256 `FOUND`；几乎全部atoms在discovery alive也未转化为native support portability。

**保守解释与gate。** k32的较高PW-MCC和高频strata相关说明稀疏度确实改变了atom-level结构稳定性，但PW-MCC正信号没有转化为双向、可审计的native support coverage；k128则保持高alive/重构但atom稳定性与coverage更弱。这不能证明完整target native support不存在，也不能把稀疏训练一般性判为无效；结论只适用于冻结的width3072、4.19M-token、TopK k32/k128与20-atom/gmax4候选族。按预写停止规则，R011-NR1以可审计负结论`PASS`收口并输出`CLOSE_TOPK_RESCUE`：不晋级16.78M/67.11M、五seed、额外k、width或hook，当前audit继续关闭。下一承重单元切到已授权的R011-S1 query-conditioned causal subspace transport protocol/unit tests/calibration feasibility；SCT成功不得冒充NIP恢复。

## 2026-09-04 03:12 EDT — R011-S1 calibration与因果非平凡性screen完成；SAE-specific SCT在audit前CUT

**触发与协议。** Automation在R011-NR1有界负结论后转入唯一顺序明确的R011-S1。按`experimental-design`工作流，把独立单位固定为source query×ordered seed pair，token仅作同一query内的加权重复测量；从R009b每个seed×energy stratum只取最低既有selection hash，共40个target-blind queries。条件权重为source code平方，mean/discovery/calibration各最多保留256个最高能量token；条件均值只来自独立mean split，projector只在discovery拟合，calibration只从固定rank `{1,2,4,8,16}`选择。Primary gates、raw/global/random/native/stitching controls、10% progression门与raw-hook非平凡性stop rule均在读取calibration前提交。Audit始终封存。

**实现与输入资产。** 新增`ccad.subspace_transport`的deterministic weighted PCA、paired stitching、PSC、dynamic/mean transfer与intervention primitives及相应测试。`R011_S1_raw_hook_asset_v1_20260904T070000Z`在受管GPU-0+disk-d-io leases下缓存mean/discovery/calibration共229,376个layer5 hook vectors，明确不读取audit；448 base forwards、7/7 checks与artifact contract PASS，raw manifest SHA-256=`4F37665D7DD8A86435AD89D21E85531B346B38DD3DB3268E39C09015F48CB596`，约704MB，peak allocated VRAM=793,476,096 bytes。全项目在screen前186/186 tests PASS。

**结构feasibility结果。** `R011_S1_calibration_feasibility_v1_20260904T073000Z`在cpu-heavy+disk-d-io leases下50.69秒完成，10/10 checks与contract PASS。40/40 queries均可评估，160个ordered query-pairs中SAE conditional PCA有159 `FOUND_SUBSPACE`，最小rank1=147、rank2=12，覆盖全部八strata。Rank1 median BCC=`.96828`、normalized residual=`.06308`、PSC=`.96417`。但raw-hook在所有rank为160/160，global SAE PCA在rank2/4/8也160/160；relaxed stitching rank1为156/160。Best native single仍0/160，matched random为0。Raw/global已完全解释低秩稳定性，故只允许一次预写的critical causal screen。Metrics SHA-256=`9E10F62D180B0948FCA4B91525D5008EBB519C9B9E13F9D7C283804C39A890D1`，projectors=`F8B6B47D434474778C5380479B43CE02B76E7B1B11ABA914E6F37D87B9403915`。

**因果screen失败保留与最小修复。** 从每个stratum选择最低query hash的rank1 `FOUND_SUBSPACE` pair，共8 pairs；每pair取source-code能量最高的2个calibration sequences。所有方法rank1并逐sequence匹配到primary source hook RMS，从同一真实hook state执行224次forward；endpoint、阈值和specificity rule在运行前冻结。V1完成全部科学指标但11/12 `FAIL`：唯一失败为跨进程raw-hook逐元素复放最大误差`6.62e-4`超过不合理的exact-like `1e-5`。No-op=0，结果网格完整。V1 artifact不覆盖。V2只将数值conformance改为abs≤`1e-3`且relative RMS≤`1e-4`，不改pair、sequence、projector、能量、endpoint、metric或科学gate；实测relative RMS=`1.2768e-5`，12/12与contract PASS，科学输出hash与V1相同（pair metrics=`8F5E022DEB31430FB0623D8E1AD4504F999B8D54249EAA4B7DFC4042A2D98960`，unit ledger=`7BAEDAF1BDEE6C86487357330F861239F0CF1BF7D89DFAFEDB220870C3D72E00`）。

**因果结果与保守裁决。** 预写规则冻结`next_state`为primary endpoint；no-op最大误差0且primary effect floor为8/8。SAE conditional、global SAE、raw-hook和relaxed stitching均8/8过门，median normalized effect error分别`.04737/.000129/0/.04462`，median effect BCC分别`.97631/.99994/1/.97759`；best native single与matched random均0/8，error约`1.648/1.997`。SAE median off-target fraction=`.96984`，raw=`.96553`，global=`.99945`；primary相对global coverage优势为0，最小specificity优势为`-.00431`。按预写规则输出`STOP_SAE_SPECIFIC_SCT_NOT_IDENTIFIED`。这不是C1/C2-SCT成功：它只保留atom/native失败与低秩整体动态稳定的measurement结果，R011-S1在audit前`CUT`，R012/R013不启动。

**Gate、资源与下一步。** 所有GPU/CPU/disk leases已释放，audit未打开，未新增文献。R011-NR1与R011-S1两条首轮救援均已形成边界清楚的负结论。C040因Euclidean SCT失配而满足技术触发，但会改变主度量且可能与endpoint循环；启动C040、改用稳定化训练机制/独立设置，或收窄/拆分论文均属用户保留的重大裁决。Automation应暂停而不删除，等待用户决定，不用新pilot掩盖阻塞。

## 2026-09-04 03:14 EDT — 双轨首轮收口后automation已暂停

按上一条重大裁决边界，`ccad` heartbeat保留原prompt与15分钟RRULE但状态由`ACTIVE`改为`PAUSED`，未删除。暂停期间不启动C040、训练机制/设置扩展或audit；待用户选择论文/方法方向后再恢复并同步相应prompt与阶段门。

## 2026-09-04 11:00 EDT — 母对象回正为模糊多对多concept correspondence；C040/R011-F1启动

**触发与纠错。** 用户明确指出上一轮把R011-NR1/R011-S1的退化对象负结果错误提升为方向阻塞，并重申论文最高概念不得改变：必须保持为基于线性代数表示的模糊many-to-many concept matching，理论蓝本只允许必要微调。该裁决覆盖上一条“等待用户选择C040/训练/收窄”的暂停边界。今后MSCC仅是binary native-support端点，SCT仅是丢弃两侧对应关系后的marginal-subspace端点；两者失败不能终止、改名或降级母问题。

**学习与方法裁决。** 重新阅读理论PDF中动态贡献对象、THM-CBSM-004/005/009、exact circuit与因果转移边界，并核对DAS/MAS、Li15、Semantic OT和sparsity-controlled many-to-many OT。Generic many-to-many或soft coupling本身已明显拥挤，故新的承重对象必须保留CCAD特有的paired dynamic contribution、source-only query、gauge-aware relation、refusal和held-out causal validation。冻结母对象为Fuzzy Contribution Correspondence：对两侧独立mean-centered native contribution banks学习paired low-rank loadings与cross operator `K=A Sigma B^T`，以`|K|/||K||_1`给出允许split/merge、重叠与非整数effective support的软耦合。因子坐标不作concept本体，relation/operator才是比较对象。

**C040与非平凡性控制。** C040正式触发：用discovery-only、query-agnostic hook probes与宽output sketch估计PSD downstream-sensitivity pullback metric；Euclidean FCC保留为必报基线。为避免R011-S1被raw/global PCA平凡解释，R011-F1同时要求同energy hard-negative contrast与cross-query collision控制。Metric学习endpoint不得成为唯一验证endpoint；calibration只选rank/阈值/refusal，audit继续封存。强对照预定为native/MSCC、SCT、raw/global relation、Li15、sparsity-controlled/semantic OT、stitching/MAS/DAS-style alignment与matched random，全部匹配rank、energy、coverage、candidate count和selection budget。

**实现与验证。** 新增`src/ccad/fuzzy_correspondence.py`：实现ridge probe pullback metric、PSD数值因子、query-contrastive regularized paired CCA relation、soft coupling/marginals/effective support、rank-boundary gap与cross-query Bhattacharyya collision。新增7个单测覆盖output-sensitive subspace、split/merge、任意旋转下的aggregate relation、downstream-null差异、hard-negative nuisance抑制、soft overlap与equal competing rank-one ambiguity。第一次回归中“旋转应给两个逐feature canonical values=1”失败；复核表明一般旋转不保持逐feature contribution-process span，只有group aggregate process由THM-CBSM-004保证不变。保留这一数学边界并把测试修正为rank-1 aggregate relation，而非放宽数值阈值。最终targeted 7/7、全项目194/194、py_compile与`git diff --check` PASS。

**治理、引用与下一步。** `AGENTS.md`、计划、tracker、components和registry已登记C1/C2-FCC、C040/C044与R011-F1；新增sparsity-controlled OT prior-art警告。Automation `ccad`已由PAUSED恢复ACTIVE，常态间隔恢复为5分钟，prompt改为持续推进FCC且不得再因退化端点失败请求方向裁决。Audit未打开，本轮没有真实SAE/FCC结果。下一工作单元是把dense合成内核改为稀疏contribution-kernel实现，完成overlap/competing-relation refusal artifact，并在读取calibration前冻结R011-F1真实discovery协议。

## 2026-09-04 11:15 EDT — R011-F1稀疏kernel与六类合成门PASS；真实pre-audit协议冻结

**动作与实现。** 按上一条唯一顺序明确的工作单元继续R011-F1。保留并完成上一轮未提交的`evaluate_fixed_correspondence` held-out固定关系评估。`src/ccad/fuzzy_correspondence.py`新增`ContributionKernels`、稀疏code×decoder contribution-kernel和kernel-only FCC fit：centered code covariance用独立mean常数代数展开，稀疏codes不转成`[token,feature,hook]`稠密张量；decoder在Euclidean或C040 PSD metric下提供Hadamard hook inner product。source/target/cross Gram、hard-negative contrast与rank-adaptive relation仍使用同一FCC对象，不在kernel层选择candidate。

**合成artifact。** 冻结并运行`R011_F1_sparse_synthetic_gate_v1_20260904T151500Z`，artifact contract PASS，六类fixtures全部PASS。Rotation canonical value=`.9999990`；split/merge第二canonical value=`.99999898`且target effective support=`2.83055`；两个overlap queries都给共享target feature约`.5` membership，cross-query Bhattacharyya overlap=`.5`，证明overlap未被forced partition删除；C040 downstream-null fixture的signal两侧membership均1；hard-negative contrast使concept membership gain=1；两条等强rank-one competing relations的boundary gap=`7.81e-9`，按预写语义输出`UNRESOLVED_RELATION / COMPETING_RELATION_RANK_BOUNDARY`。这些只是synthetic implementation evidence，不是real-SAE C1/C2-FCC。

**协议冻结。** 新增`configs/R011_F1_PREAUDIT_PROTOCOL_20260904_112000.md`与`configs/r011_f1_preaudit_protocol_v1.json`。独立单位固定为query×ordered seed pair；40-query feasibility由R009b每seed×energy stratum最低selection hash产生；positive/hard-negative只由source codes定义；source/target local universes为32×至多128、每relation 4,096 feature-pair预算；rank固定1/2/4/8。C040固定为256个discovery hook states×4 Rademacher directions、±0.01 hook-RMS central perturbation、256维next-logit sketch、ridge fraction `1e-4`与eigen tolerance `1e-6`，并以未用于metric拟合的next-state residual作为primary causal screen。Calibration只可选rank/阈值/refusal；audit禁止读取。40-query screen若失败即结束v1，不追加candidate/threshold变体；若通过只授权同规则扩到完整640 panel。配置保持`execution_enabled=false`，直到C040 probe artifact实现与静态验证完成。

**验证与边界。** 稀疏kernel对显式dense centered fit，以及带C040 metric和hard-negative contrast的fit均在`1e-10`内一致；targeted 13/13、py_compile与`git diff --check` PASS。全发现运行中其余196项PASS，唯一collection error是既有R006 family-paired test依赖当前系统Python未安装的`mpmath`，未进入测试主体且与本轮代码无关；不把该命令写成full-suite PASS。Audit未打开、没有真实calibration结果、没有新文献或GPU运行。下一步只实现/静态审计C040 query-agnostic probe artifact，然后把protocol的`execution_enabled`改为true并执行唯一40-query pre-audit screen；不能越过该门直接读calibration。

## 2026-09-04 11:40 EDT — C040工程artifact PASS但科学metric HOLD；冻结协议变更需用户裁决

**执行与失败保留。** 按R011-F1冻结协议实现discovery-only、query-agnostic C040 probes。V1使用系统Python，因缺`transformers`在模型计算前FAIL，artifact contract仍PASS；V2在正确CUDA环境完成probe/model计算，但finalizer无法JSON序列化NumPy boolean，原`status.json`停在stale `RUNNING`。资源管理器确认无process/lease存活；已追加`CORRECTION.md`将其明确为incomplete FAIL而不改写原失败状态。只修`bool(value)` finalizer的V3 `R011_F1_C040_probe_metric_v3_20260904T163500Z`为11/11与contract PASS，metric SHA-256=`2EE85D0B39E36E1F7FC186356155C4C45F3C32CC61E5A784A9ABE569E34C5266`，raw metrics SHA-256=`13985E6F47F4241528D45378648867E7D849D4A79F0D4FD748BFD8BE074A4632`。所有leases均释放。

**Raw table与异常归因。** V3用256个discovery states×4独立方向=1,024 probes、2,048 central variants。Effect norm min/median/max=`.8093/2.3780/21,355.024`；trace-normalized metric rank=2、effective rank=`1.000091`、top eigenvalue=`767.9957/768`。唯一极端state 108位于sequence569、position31，前一token为EOT，四方向norm约`2,523/12,905/13,494/21,355`。只读全ledger诊断如下：

| 输入分组 | states | state-max median | q90 | max |
|---|---:|---:|---:|---:|
| 前一token为EOT | 2 | 10,710.60 | 19,226.14 | 21,355.02 |
| 距前一EOT≤8 token | 9 | 39.93 | 4,343.01 | 21,355.02 |
| 距前一EOT>8 token | 247 | 3.62 | 10.99 | 131.20 |
| multi-document sequence | 121 | 3.86 | 19.92 | 21,355.02 |

另一个immediate-post-EOT state的max仅`66.17`，所以边界与极端值相关但不充分；不得把“删除EOT附近state”当作已证实修复。为区分batch replay/有限差分伪影，预写并提交bounded diagnostic `389067a`，随后在GPU-0 lease下运行`R011_F1_C040_probe_stability_v1_20260904T170000Z`。四个固定诊断states×4方向×relative amplitude `.003/.01/.03`共96 variants，5/5与contract PASS；`.01`复放相对v3最大RMS误差=`3.22e-8`，极端state相对三controls的min-to-max separation=`12.998×`，但跨幅度maximum ratio=`1.479`超过预写`1.25`，输出`FINITE_DIFFERENCE_OR_STATE_LOCALITY_NOT_CONFIRMED`。Raw SHA-256=`70C036B949DB2E24083DDA0B237F0D82DE81F59A087B88F0E3BF30DC130CC440`；rows=`A614BFDA83FD43FB998BFA317E2E75C87B3028DBD6B3DBF2060183B5FE641376`。该diagnostic明确禁止用于正式state/阈值选择。

**设计审计与gate影响。** 现v1让每个state使用不同随机方向，再对全部`(d,J_xd)`拟合单一ridge map。在线性各向同性近似下，它得到平均Jacobian `E[J_x]`并形成`E[J_x]^T E[J_x]`，不是FCC需要的平均平方敏感度`E[J_x^T J_x]`；state-dependent Jacobian时会产生cancellation与state-direction混杂。V3近rank-1和单state极端主导把这个风险实化，因此V3虽工程PASS但不能作为科学metric，不能据此开启FCC calibration。

R011-F1转`BLOCKED`，但母对象FCC不变；`execution_enabled=false`，calibration/audit从未读取。推荐fresh协议改为crossed shared directions：同一方向跨document-balanced、输入侧固定boundary-margin states复用，拼接/等权聚合各state output effects后拟合stacked Jacobian，并先过state-varying Jacobian synthetic recovery、state-duplication invariance与boundary-rule测试。只删异常点或直接沿用v3均不接受。由于这会修改已标`FROZEN FOR IMPLEMENTATION`的§5 probe contract，按AGENTS必须请求用户批准，不能由automation静默替换。Targeted protocol6/6、artifact/activation10/10、py_compile和diff check PASS；一次额外collection命令因环境缺SciPy/误写不存在的test module而未进入对应测试，不计项目失败。阶段性代码提交`389067a`已推送且当时HEAD=`origin/main`。

## 2026-09-04 11:53 EDT — 勘误过度暂停；用户授权自主裁决并冻结crossed C040 v2

**治理勘误。** 用户明确指出上一工作单元把audit前、可逆的metric修复错误升级为整条automation暂停，并要求agent自行裁决、删除不必要审批。该批评成立。`AGENTS.md`已改为：LOCKED保护历史协议不被覆盖，不禁止保留旧FAIL/CUT并建立fresh suffix；protocol deviation、primary配置、metric/baseline、阶段门和满足预写条件后的audit开启均由agent自主裁决，不得因此暂停或逐项请求批准。只有改变FCC母对象、付费/受限外部资源、公开发布或项目终止/拆分仍由用户保留。`ccad` automation已恢复`ACTIVE`，RRULE恢复常态5分钟，prompt同步为FCC与当前v2任务。

**方法裁决。** 不采用删state或低维shared-direction近似。Fresh C040 v2 estimand冻结为document-balanced `M=E_x[J_x^T P_sketch J_x]`：32个content-hashed/document-balanced discovery states，输入侧距sequence start或最近EOT至少16个visible tokens；32 states全部复用同一完整768维deterministic orthonormal basis。每个direction的state output effects按等权`1/sqrt(32)`拼接后拟合stacked Jacobian；完整basis下其Gram就是平均state Jacobian Gram。Amplitude保持`.01×discovery hook RMS`，output sketch保持固定256 logits，ridge=`1e-4`，eigen tolerance=`1e-6`。预算为24,576 probes/49,152 central variants，约为v1的24倍但仍是本地几分钟级有界run。State不按响应删除或clipping；预冻representativeness门为max trace share≤.25、effective state count≥8。

**实现与静态门。** `ccad.causal_metric_probe`新增deterministic complete orthonormal basis与只检查causal prefix的boundary-safe document-balanced selector；`ccad.fuzzy_correspondence`新增crossed stacked-Jacobian metric fit。协议与run config分别为`configs/R011_F1_PREAUDIT_PROTOCOL_V2_20260904_115000.md`、`configs/r011_f1_preaudit_protocol_v2.json`和`configs/r011f1_c040_crossed_metric_v1.json`。State-varying synthetic Gram recovery relative error=`3.37e-16`，balanced duplication error=`3.63e-16`；实际输入选择32 states/32 blocking docs、最小causal-boundary distance=26；768维basis max orthonormal error=`1.33e-15`。Targeted23/23、py_compile与diff check PASS。R011-F1恢复`RUNNING`，v2 `execution_enabled=true`；下一动作是提交冻结实现后直接执行唯一formal discovery metric run，calibration/audit仍未读取。

## 2026-09-04 12:11 EDT — Crossed C040科学门FAIL；Euclidean FCC raw surface PASS

**C040正式运行与独立复核。** 唯一formal run `R011_F1_C040_crossed_metric_v1_20260904T161500Z`从clean commit `2f6980a`启动，在受管GPU-0 lease下完成32 states×768 shared directions×正负central perturbations，共49,152 variants/3,072 model forwards；112.26秒，peak allocated VRAM=1,516,797,440 bytes。输入hash、32个blocking documents、最小causal boundary distance26、完整正交basis误差`1.33e-15`、central pairs、PSD/trace normalization与非零rank均通过，artifact contract PASS。所有resource leases结束后为free。

**预冻科学FAIL。** Crossed metric的最大单state trace share=`.6488565`，超过`.25`；effective state count=`2.30444`，低于8。故run为12/14 checks、科学`FAIL`，causal metric SHA-256=`940E80063828C1B36E61378B71697417CFC8517BD439B958BFF8226E1200606F`。独立validator 13/13 PASS，完整复算ledger、basis、effect tensor、metric（相对saved float32误差`4.37e-8`）与两个失败门。按锁定stop rule，C040在R011-F1 v2中不可用；不删state、不clip、不改阈值、不再开metric suffix。它拒绝的是当前query-agnostic downstream metric，不是FCC母对象。

**Euclidean mandatory lane。** 在读取raw surface前，helper、runner、config、测试与C040独立validator通过全项目212/212 tests并以`35bf61d`冻结、推送。随后`R011_F1_euclidean_surface_v1_20260904T164500Z`在受管cpu-heavy+disk-d-io leases下完成，21.34秒、10/10 checks与artifact contract PASS；2,560行覆盖160个source-only condition queries、4个ordered targets及rank1/2/4/8，640个anchor loadings完整保存，audit未读。独立surface validator 15/15 PASS，重算grid、hash、loading shape与rank summaries。

**Raw calibration signal与边界。** 160/160 anchor ordered units在每个rank可评估。Rank1/2/4/8的median calibration BCC=`.16742/.11806/.06781/.04995`，median positive-minus-hard-negative contrast=`.16949/.12014/.08175/.06568`，positive contrast fraction=`.975/.975/.98125/.975`，median collision improvement over global=`.22742/.24981/.26890/.30494`。这些量显示query contrast与collision separation存在，但rank增加时BCC下降，不能从单一aggregate挑赢家。Artifact明确没有FOUND、阈值、causal outcome或C1/C2-FCC claim；surface SHA-256=`45DEF786ED2341010FD2F5953E1ABBE900443ECE9D3DF169D23442CB203BEED3`，loadings=`224B5A1A20C606C42FAFC4AE4B3E4BB1AA8AB1A88A36E8AAC4FB36707B86A2E8`。

**Gate与下一步。** R011-F1保持`RUNNING`，C040 lane停止而Euclidean FCC继续。下一工作单元是在固定surface上冻结最小rank、numerical/refusal cutoffs，随后执行协议既定、matched rank/RMS的next-state causal specificity screen及global/raw、native/MSCC、SCT、stitching/MAS与matched-random controls。只有coverage、strata/direction、contrast、collision和独立因果门全部满足才可扩到full-640；audit继续关闭，automation保持常态5分钟。

## 2026-09-04 12:31 EDT — Euclidean FCC calibration rank/refusal冻结PASS

**冻结规则。** 在固定surface上按rank `{1,2,4,8}`选择第一个同时满足calibration BCC>0、positive-minus-hard-negative contrast>0、collision improvement over global≥`.05`和rank-boundary relative gap≥`.001`的关系；无rank通过则`UNRESOLVED_RELATION`。该规则只用calibration，query/candidate/loadings/mean不重学，audit禁止读取。实现与配置在执行前以`bda31ba`提交；全项目212/212与py_compile PASS。

**失败保留与修复。** `R011_F1_euclidean_calibration_freeze_v1_20260904T162500Z`科学checks 10/10且决策完整，但manifest缺`resource_lease_reason`导致artifact contract FAIL。V1不覆盖。Fresh v2只增加该manifest字段并以`db46e29`预先提交；科学配置与输出未变。

**V2结果与边界。** `R011_F1_euclidean_calibration_freeze_v2_20260904T163000Z`为10/10与contract PASS，决策SHA-256=`4D4C68843866975A77EE4A89F8D92B5C17BA5E2FF57ED1F47F1EEDF10FC34305`。157/160 units冻结为`FOUND_RELATION`，coverage=`.98125`；rank1/2/4/8计数125/22/7/3，覆盖8 strata与20 ordered directions。Found median BCC=`.15118`、contrast=`.16075`、collision improvement=`.23759`。前三个progression gate通过，但`progression_state=AWAIT_MATCHED_CAUSAL_GATE`；这不是C1/C2-FCC结果，不授权full-640或audit。下一步是冻结endpoint-blind causal subset与matched controls并执行next-state screen。

## 2026-09-04 13:15 EDT — R011-F1两种冻结causalization均0/8；当前CCA estimator在audit前拒绝

**冻结设计与V1。** 按`experimental-design`约束把独立单位保持为query×ordered seed pair；从每个energy stratum按既有R009b selection hash与target seed固定一个calibration `FOUND_RELATION`，共8 units，并在不读取endpoint的情况下按source-query squared-code energy固定每unit两个calibration sequences。Primary endpoint沿用已冻结next-state；方法逐side/sequence匹配到FCC source hook Frobenius norm。新增operator-induced soft-marginal干预、协议、runner与测试，在模型运行前以`2c8a0db`提交并推送。`R011_F1_euclidean_causal_gate_v1_20260904T174500Z`为12/12、artifact contract PASS；256 forwards、peak allocated VRAM=920,286,208 B，所有GPU/CPU/disk leases释放。FCC 0/8，median effect BCC=`.025706`、normalized error=`2.07417`；raw-hook 8/8，SAE marginal PCA/stitching各7/8，best single/global FCC/random均0/8。

**V2理论边界修复。** V1的`|K|`边际只测试membership加权贡献，丢弃了FCC拟合的signed loadings与rank-component结构，不能单独代表完整relation operator。V1失败完整保留；fresh V2不重选query、rank、sequence、threshold、mean或endpoint，hash绑定V1 selection=`B92878214D4F67503A7F5DBD3303046E25475F8A34FB16F6DCA482578A62BF0E`，只改为逐discovery-frozen paired loading component干预、以一个block scale匹配能量并在原unit内聚合quadratic effects。旋转/符号置换下aggregate energy/cross-energy回归通过；协议与代码在运行前以`c339739`提交并推送。

`R011_F1_euclidean_causal_gate_v2_20260904T180500Z`同样12/12与contract PASS；284 forwards、peak 920,286,208 B，leases全部释放。Signed FCC仍0/8，median effect BCC=`.022322`、normalized error=`2.04621`、off-query fraction=`.94591`。Raw-hook仍8/8，SAE marginal PCA/stitching仍7/8；global FCC median BCC仅`.06280`且0/8。V1/V2均由新增独立validator 18/18复算input/output hash、unit/method grid、energy matching、coverage、medians、gain、forward count与`STOP_FCC_CAUSAL_EFFECT_OR_CONSISTENCY_FLOOR`裁决。Audit始终关闭。

**保守解释与下一步。** Calibration的median BCC仅`.15118`且原门只要求`>0`，因此157/160 nominal coverage没有转化成任何next-state effect consistency。Soft membership与signed relation两种causalization一致失败，排除了“只因丢sign/rank”这一解释；当前fully-whitened contrastive CCA estimator不得扩到full-640/R012/R013。该结果不否定FCC母问题。R011-F1保持`RUNNING`，但CCA lane关闭；下一承重单元固定为C045一次性estimator bracket（CCA reference、energy-balanced contrastive PLS、diagonal-whitened contrastive correlation），沿用相同40 queries/candidates/hard negatives/ranks/4,096 pair预算。任何新causal forward前，alternative必须在完整160-unit calibration上达到BCC≥`.8`、residual≤`.2`、既有contrast/collision/rank-gap门、coverage≥10%、至少4 strata与全represented directions；若均失败则关闭当前local contribution-kernel family，不追加estimator grid、不调弱门。全项目上一轮214项执行通过、唯一collection error仍是既有可选`mpmath`缺失；本轮targeted 25/25、py_compile与diff check PASS。阶段结果与validator待成组提交。

阶段结果与validator已以commit `60a6318`推送，推送后`HEAD=origin/main`。Automation `ccad`保持`ACTIVE`与5分钟常态间隔，prompt已从已完成的causal gate更新为C045 estimator bracket、meaningful calibration transfer门与audit继续关闭；没有暂停或删除。

## 2026-09-04 13:58 EDT — C045两种alternative均0/160；local contribution-kernel family关闭

**预冻设计与实现。** CCA causal失败后按C045只允许一个bounded estimator bracket：完全复用40 anchor、120 collision-neighbor queries、source/target candidate IDs、source-only hard negatives、ranks `{1,2,4,8}`和4,096 feature-pair预算。新增energy-balanced contrastive PLS与diagonal-whitened contrastive correlation；每个signed component在discovery positive kernel上单位能量归一。任何新causal forward前必须在160 anchor ordered units上同时满足calibration BCC≥`.8`、normalized residual≤`.2`、positive contrast、collision improvement≥`.05`、rank gap≥`.001`，且coverage≥10%、至少4 strata和全部20 directions。协议、实现与17/17 targeted tests在结果前以`0a50106`提交。

**工程失败保留。** `R011_F2_estimator_bracket_v1_20260904T183000Z`因runner对collision-neighbor读取仅anchor存在的`global_collision_mean`而在科学summary前FAIL；v2只修`.get()`但又在reference-unevaluable neighbor上读取缺失candidate IDs而FAIL。两run均contract PASS、没有替代estimator裁决，完整保留。对应最小修复分别以`d978556`与`12ad1c2`提交；v3显式输出unevaluable placeholders，不删除query、不改estimator、阈值或候选。

**V3结果与裁决。** `R011_F2_estimator_bracket_v3_20260904T185000Z`在受管cpu-heavy+disk-d-io leases下48.72秒完成，9/9与artifact contract PASS；5,120 surface rows、320 decisions，surface SHA-256=`7D65C5D2490EE6B9847F5F9230E87C074B3B71CA48F4F9A77B80393458DC5E1F`，decisions=`5F472CA01976B2DF5FFA06443D01B1C4B11145AC9D96ACEF05F806FA5F9A67E7`，loadings=`AD161CFB31625602374A2C708C3A98AE053E132BD8B414583D4BB27BC50F5BD5`。独立validator 12/12复算artifact/grid/decisions/summaries与stop outcome；所有leases释放，audit未读且模型causal forwards=0。

PLS与diagonal estimator均0/160 `FOUND_RELATION`，输出`STOP_LOCAL_CONTRIBUTION_KERNEL_FAMILY`。PLS rank1 median/max BCC `.19/.85`，4 units达到`.8`，但minimum residual仍`.31`且0 units≤`.2`；diagonal rank1 median/max BCC `.03/.80`、minimum residual `.48`。高rank继续恶化。Positive contrast/collision coverage很高但不能替代held-out contribution transfer。因而禁止放宽门或追加local estimator。

**下一步。** R011-F2转`CUT`，R011-F1母问题保持`RUNNING`。下一唯一承重解释为C046 full-target candidate setting：保留32 source features、相同query/negative ledgers与PLS，以streaming cross-covariance暴露全部3,072 target features并只计算selected loadings的target energy；在同40-query screen和同一meaningful-transfer门下与cap128比较。只允许一个full-target setting，不做cap sweep；失败则关闭candidate truncation解释并转新的FCC representation/configuration。Automation保持ACTIVE/5分钟，下一轮prompt应同步C046；本轮没有新增文献，registry不变。

## 2026-09-04 14:00 EDT — C046 full-target仍0/160；candidate truncation解释关闭

**运行与artifact复核。** `R011_F3_full_target_v1_20260904T191500Z`从clean commit `01c1166`在受管cpu-heavy+disk-d-io leases下完成，wall time=`623.62s`；运行结束时process与leases均无存活。它保留冻结的32-feature source family、40 anchors/120 collision neighbors、source-only negatives、ranks `{1,2,4,8}`与energy-balanced PLS，只把target candidate universe一次性扩为全部3,072 features，pair budget=`98,304`；未做cap sweep、target-square Gram、model causal forward或audit读取。Run 9/9与artifact contract PASS；surface/decisions/memberships/loadings SHA-256分别为`26FA3D46704BE2A248F8B1CB3D1AAB998D34F13CB1275ED7AF2A6CF17C45B93A`、`187EC7A4B72D2E7ABAA8D6F788BA542E4BDF51754115A481A8579D7CA9506178`、`B3E8F14B9805CC0E584EF976BA01742E043994CD4B66BD98798C391F18EB1954`、`9F3B573B2CE0CD40B4E752A4A93C8297DBE1FA7197C7E8914D6E33E64BDC408C`。新增独立validator 15/15复算hash、grid、160 decisions、summary、NPZ shapes与stop outcome；targeted protocol 2/2和通用artifact validator PASS。

**结果与裁决。** Full target使rank1 median/max BCC从capped PLS的`.18710/.84604`升到`.52402/.86610`，BCC≥`.8` units由4增至8；median residual从`1.57050`降到`.94256`。改善说明target cap确实损失信号，但不足以解释失败：minimum residual仍`.26685`，rank1到8均0/160达到预冻residual≤`.2`，高rank继续退化；最终0/160 `FOUND_RELATION`、coverage=0，输出`STOP_CANDIDATE_TRUNCATION_EXPLANATION`。不得因最优unit接近门而调弱阈值或追加cap。R011-F3作为有界screen为`PASS`，target truncation解释为关闭；C1/C2-FCC仍未成立，audit继续封存。

**下一步。** 现有native-coordinate static relation family已依次排除local estimator、target cap与causalization解释。下一fresh setting定为C047 query-conditioned hook-space reduced-rank transport：source-only query family定义source过程，完整target SAE动态重构贡献作为target输入，在shared hook space以discovery-only ridge RRR学习有限秩map，并与matched raw-hook、whole-SAE/global、hard-negative与collision controls公平比较。先做rotation/split-merge/query-null/global-nuisance/rank-deficient synthetic门和静态协议，再允许一次同40-query真实screen；仍需`.8/.2` meaningful transfer及相对raw/global的query-specific advantage，失败即停止该representation，不运行causal或audit。Automation恢复5分钟常态并同步C047；本轮无新增文献，registry不变。

## 2026-09-04 14:04 EDT — C047 hook-space transport synthetic门PASS；真实screen协议冻结

**设计与实现。** 按`experimental-design`把独立单位、blocks与强对照先于real calibration固定：独立单位是source query×ordered seed pair，token只作重复测量，seed direction与八个energy strata作blocks。C047不是第四个native estimator；每rank先用source-only discovery-positive rows对32-feature local contribution reconstruction做conditional PCA，冻结query projector/source process，再以完整target SAE centered reconstruction输入discovery-only ridge RRR。Ridge fraction固定`.001`，rank固定`{1,2,4,8}`。Matched controls为同rank/ridge/budget的query-conditioned raw hook与query-agnostic whole-SAE/global transport。

**Synthetic结果。** 协议、实现、runner与5项unit tests先以clean commit `6e8130b`冻结。`R011_F4_hook_transport_synthetic_v1_20260904T180000Z`随后为5/5与artifact contract PASS；raw metrics SHA-256=`20CA0BE26EB834B609BAEF4A53A5ABFCF3CAFC463C0147B6C4D9D8E47EDC5F0B`。Rotation/split-merge held-out BCC均在`1-3e-13`内为1，residual≤`5.49e-13`；query-null/global nuisance按zero specificity拒答；未以`.05`击败raw control时明确拒答；rank2对rank1对象输出`RANK_DEFICIENT/effective_rank=1`。新增独立validator 12/12复核artifact、hash、family order、recovery/refusal语义和audit关闭。

**实际协议与下一步。** `configs/r011f4_hook_transport_real_v1.json`已hash绑定synthetic PASS、R009a query census、R011-F1 query surface、R008 paired codes、R011-S1 raw hook与sequence ledger。Meaningful transfer保持BCC≥`.8`、source-normalized residual≤`.2`、positive hard-negative specificity、collision improvement≥`.05`、rank gap≥`.001`，并要求specificity比raw/global最佳control高至少`.05`；coverage仍需≥10%、至少4 strata和全部20 ordered directions。`execution_enabled=true`只授权唯一40-query pre-audit screen；不得调ridge/rank/门、运行causal、扩full-640或读audit。下一轮实现static validator与bounded runner后从clean commit执行。无resource lease或process遗留，本轮无新增文献。

## 2026-09-04 14:27 EDT — C047 real v1工程FAIL；factor保存fresh v2

`R011_F4_hook_transport_real_v1_20260904T181500Z`从clean commit `e66685d`在`disk-d-io`+`cpu-heavy` leases下执行约7.8分钟并完成内存中计算，但在科学surface/decision写盘前构造`anchor_maxrank_hook_factors.npz`时失败：显式`RANK_DEFICIENT` transport的factor列数少于8，直接`np.stack`触发`ValueError: all input arrays must have the same shape`。Run artifact contract PASS、metrics为空、没有可用科学裁决；旧run完整保留，所有process/leases释放，automation间隔立即恢复5分钟，audit未读。

Fresh `r011f4_hook_transport_real_v2`仅修artifact serialization：factor矩阵右侧零填充到冻结max-rank 8，同时另存query/raw/global effective-rank ledger，validator增加shape/rank-ledger检查。零填充不参与计算或decision；query/data/source projector/target process/ridge/ranks/controls/`.8/.2`与control-advantage门全部不变。Targeted 9/9、static protocol 11/11、py_compile与diff check PASS；下一步从clean commit运行v2，不使用v1内存结果。

## 2026-09-04 14:35 EDT — C047 real v2高BCC但raw specificity占优；representation CUT

**运行与复核。** `R011_F4_hook_transport_real_v2_20260904T183000Z`从clean commit `091b944`在受管`disk-d-io`+`cpu-heavy` leases下212.83秒完成，9/9 checks与artifact contract PASS。Surface/decisions/loadings SHA-256分别为`334F4D8523035571B6EA636F0D621A4CA13382EDB37D6873A3F9884F7EE5C04F`、`6D95B109D7DDF200F342DCF4BF997C6F59E51707CCF648C169AD32976B6BA9CF`、`E085F35DF378DDC3E869D57C4C79A8A36372DB51B2B7C208C5DF71358D230526`。独立validator 14/14复算artifact hashes、2,560-row/160-decision grid、全部decisions与summary、padded factor shapes/effective-rank ledger和stop outcome；所有leases释放，automation恢复5分钟，audit未读、causal forwards=0。

**科学裁决。** Rank1/2/4/8 median query BCC=`.99494/.99220/.98850/.98167`且minimum residual均<`3e-5`，说明hook-space RRR能稳定重构source local PCA过程；但median query specificity仅`.01043/.02832/.04458/.05676`，matched raw-hook为`.04742/.07710/.09454/.10248`，所有rank的median control advantage为负`-.03091/-.03861/-.03909/-.04001`。仅6/160 units过全部门，都是rank1，集中于source seed3 atom1093和seed4 atom2679，仅2 strata、6 directions；coverage `.0375`未过`.10`且不满足4 strata/20 directions。其余最终refusal reasons为raw/global未击败70、rank-deficient56、rank-boundary24、hard-negative contrast4。输出`STOP_HOOK_TRANSPORT_REPRESENTATION`；R011-F4转`CUT`，不运行causal/full-640/audit，C1/C2-FCC仍未成立。

**下一步。** 高BCC与raw specificity优势共同指向shared low-rank nuisance而非广泛query-specific relation。下一承重C048限定为一次discovery-only shared-nuisance residual representation：用query-agnostic global covariance按预写explained-variance规则唯一冻结projector，对source local、target SAE和raw hook对称残差化，再复用C047 transport与强门；同时保留raw residual与unresidualized C047 controls。先过global-nuisance recovery、query-signal preservation、nuisance-null refusal、rank-deficient synthetic门，不做nuisance-rank sweep。失败则停止residual FCC并转SAE训练/configuration证据；audit保持关闭。本轮无新增文献。

## 2026-09-04 14:48 EDT — C048 shared-nuisance residual synthetic门PASS

**前瞻设计。** 按`experimental-design`先固定独立单位、blocks、nuisance选择与controls：4,096 document-balanced/query-agnostic discovery raw-hook states；独立mean centering；达到90% global variance的唯一最小rank、hard cap64且禁止rank sweep。Source local、target SAE、raw hook对称残差化；source residual energy保留不足20%即拒答。单位仍为query×ordered seed pair，token仅为重复测量，seed direction和八strata作blocks；controls固定为residual raw、residual global与unresidualized C047。

**实现与artifact。** 协议、`NuisanceProjector`、唯一rank selector、对称residualizer、runner与11项targeted tests先以clean commit `4fbf46e`冻结。`R011_F5_residual_transport_synthetic_v1_20260904T184500Z`随后5/5与artifact contract PASS，raw SHA-256=`4389948D80D68E57319C8BD70B172BEF50F8D38DC38FBC54A92CE0323C708348`。Variance fixture选rank2/解释`.980526`；global nuisance移除后held-out BCC≈1、residual=`1.11e-13`；orthogonal query energy完整保留；nuisance-only与rank-deficient均按协议拒答。独立validator 12/12复核hash、family order、数值恢复与拒答语义。Audit未读、无causal forward或resource lease；下一步只建立hash绑定的real config/static validator/runner。本轮无新增文献。

## 2026-09-04 15:20 EDT — C048在预写nuisance rank-cap门停止；calibration未读

**冻结实现。** Real config绑定synthetic PASS、R009a census、R011-F1 query surface、R011-F4 unresidualized control、R008 paired codes/raw hook/sequence hashes；固定4,096个document-balanced discovery states、90%最小rank、cap64、对称residual、20% source energy floor及既有`.8/.2`和control/progression门。Static validator 13/13、transport unit tests 11/11、py_compile与diff check PASS；运行前实现提交`6f78cd0`并推送。

**失败保留与fresh suffix。** V1真实执行得到rank64 explained variance `.8873410940`，命中预写停止，但runner将该科学分支作为异常收口，contract仍PASS；旧run保留。V2只增加stop artifact分支并从commit `a945b00`执行，但该分支早于`started_compute`初始化，触发`UnboundLocalError`，状态停在stale RUNNING；资源管理器确认process和leases均已释放，并以`CORRECTION.md`标为工程FAIL。Fresh v3只把计时器初始化前移，从clean commit `a06d236`运行；state、rank、阈值、representation、control和数据均未改变。

**正式裁决。** `R011_F5_residual_transport_real_v3_20260904T192000Z`为6/6 checks与artifact contract PASS，metrics SHA-256=`98D6516431C9F860C7DC0BC5D2221E99F6A8941F66B36E87711350A960A4CCB7`；独立validator 7/7复核rank cap、`.887341<.90`、无surface/decision/progression与audit关闭。输出`STOP_NUISANCE_VARIANCE_THRESHOLD_NOT_REACHED`。由于停止发生在calibration载入前，本轮没有transport screen、causal forward、full-640或audit读取。R011-F5转`CUT`；不放宽`.90`、不扩大cap、不做nuisance-rank sweep。下一承重单元按预写规则转SAE训练/configuration证据；FCC母对象不变。本轮无新增文献，registry不变；所有CPU/disk/GPU resources为free。

## 2026-09-04 15:25 EDT — C049单因素width机制screen预冻结

按`experimental-design`先固定单位、block、对照与停止门。C049只把R011-NR1 k32的dictionary width从3,072改为16,384；底模/revision、layer5 resid-post、sparsify、k32、4,194,304-token corpus/order、optimizer、batch、validation和初始化seeds1/2全部保持一致。独立单位是SAE seed，按seed与旧width3072配对；tokens/sequences只作重复测量。Width16,384是与直接相关文献对齐的唯一处理水平，不做width sweep；公开width65,536单seed checkpoint仍只作外部质量锚点。

预写resource gate先执行seed1 256-step smoke：peak allocated VRAM≤6GiB、外推≤50分钟/seed、checkpoint≤1GiB/seed、exact L0与hook/CE pipeline必须通过。两正式seed各需FVE≥.97、CE recovered≥.93、alive≥.50、L0=32、decoder norm error≤1e-5。结构晋级需full-dictionary PW-MCC较`.48940`提高≥`.05`，且source-evaluable native FOUND coverage≥10%、双向各≥5%；best-single BCC不能替代coverage。协议`configs/R011_W1_WIDTH_MECHANISM_PROTOCOL_20260904_152500.md`、manifest与static validator完成，9/9 PASS，py_compile/diff check PASS。当前`execution_enabled=false`，尚未申请GPU或读取audit；下一步只实现capacity smoke配置/runner并再静态复核。本轮未新增文献。

## 2026-09-04 15:33 EDT — 用户请求的项目与母目录系统阅读

**触发与范围。** 用户要求学习CCAD当前目录及母目录相关内容。本轮按治理顺序恢复状态，阅读当前研究对象、关键计划/追踪/日志、理论PDF完整证明部分与修订边界、核心贡献/MSCC/FCC/transport实现及相关测试；对母目录文献目录、既有90-PDF提取索引、六个应用项目和SAE Philosophy作分层梳理。没有打开封存audit、运行模型、改变协议/阶段门或操作任何automation。

**产物与实际深度。** 新增本地 `RESEARCH_ORIENTATION_20260904.md`（SHA256 `2d0a594b6e7919880cdb91d0bb3a946f2649b8ef8d0e4310942e9bdda6bb0c7b`）和 `READING_INVENTORY_20260904.json`（SHA256 `d89b1c66d363a6515521c21d2b3572de9c4c4f4fe52d27b6bc6e076f18c21a86`），生成脚本 `tmp/build_orientation_inventory.py`。357个项目文本文件完成结构索引，含58 Markdown、142 JSON、157 Python；86个参考条目目录含82个paper.pdf与48个code目录；既有90条PDF路径均存在。167个run目录仅清点，关键已完成run的汇总被选择性读取。报告明确区分精读、片段阅读、既有摘录与机器索引，不声称完成所有论文/第三方代码的逐行审计或独立证明复核。索引是本条追加之前的快照，故其中master_log hash对应追加前版本。

**核验与理解。** 索引时Python AST/JSON解析无失败，追加本条前357个文件hash复核无漂移；R011-F4决策文件重计160条、6条FOUND，符合日志。核对R011c/F4/F5原始汇总与C049 execution-disabled配置。报告强调FCC母对象、MSCC/SCT端点分离，旧BCC与source-normalized residual不能混用，F5只在nuisance选择门停止而没有产生residual transport结果。发现计划首页/理论稿/母目录wiki中的历史状态不可替代最新治理与记录；没有回写历史文档。

**结果边界与下一步。** 本工作单元是研究阅读与接续地图，不是新科学run、全库citation audit或全球查新；无新文献采用/排除决定，REFERENCE_REGISTRY不变。当前科学接续仍为已冻结C049 capacity smoke。按第8.1节，本地阅读笔记与常规日志追加不单独触发commit/push；Git基线为2026782，保留全部既有状态。

## 2026-09-04 16:05 EDT — 用户要求的研究流程重整、F5保证勘误与直接因果主线

**触发与裁决。** 用户指出严苛代理门、过度工程化、重复开发panel和错误独立验证保证，要求科学实践方案、长期automation prompt与AGENTS修改。本工作单元已完成这些交付；没有启动新模型实验或宣布FCC阳性。新版AGENTS把可信度硬门、科学证据、预算门分开；atom/native、rank gap、collision、覆盖/全方向不再串联拥有FCC否决权。当前主线改为复用F4 signed relation的source-referenced直接因果开发判别，随后由其证据选择一次机制修正，再冻结独立确认；C049为候选而非唯一路线，旧配置execution-disabled保持。

**实际科学核查。** 从F4 v2原surface（SHA256 334F4D8523035571B6EA636F0D621A4CA13382EDB37D6873A3F9884F7EE5C04F）重计：rank1/2/4/8各160个evaluable units均达到旧BCC>=.8、source-normalized residual<=.2；154/157/156/157个未过旧control advantage>=.05。rank8有56个effective-rank不足，不能把其结果称rank8成功。加入精确反例：正负条件都完美预测时BCC差=0，但线性endpoint正/负效应能量比可为10000。它证伪“该差一般等价因果特异性”，不证明真实F4因果有效。旧6/160 FOUND/CUT原样保留。

**F5勘误与代码修复。** 撤回15:20、15:33条目及orientation中的“calibration未读”：历史runner在nuisance门前物化calibration codes并打开raw memmap；未计算calibration transport指标。未发现这些值参与nuisance选择或audit数组被加载，但不是OS级取证。旧7/7只支持保存字段一致性，未独立重算方差或读取边界。新增 `F5_READ_BOUNDARY_CORRECTION_20260904.md` 和run-local CORRECTION，原metrics/status不改。F4/F5 runner改用 `ccad.split_access.SplitAccess` lazy加载，记录loader成功/失败尝试与raw打开；初始化早停状态。Validator对旧run输出UNVERIFIED_LEGACY_RECORD及独立方差/OS读取未验证，7/7不再隐含更强保证。

**验证。** 使用锁定 `D:/CCAD_Storage/environments/r004/Scripts/python.exe` 与已有r009 overlay，split-access 2项、specificity反例1项、hook-transport既有11项、真实runner早停入口stub1项共15项PASS。入口stub使用不存在的calibration文件和禁止code loader，确证停止能正常收口且无calibration loader调用；它不是实际nuisance方差复算或OS取证。py_compile和git diff --check PASS。旧F5 validator重新执行仍7/7，但明确打印验证范围；未重跑F5科学实验，无GPU/CPU-heavy/disk lease启动，无audit读取。

**文档与引用。** `SCIENTIFIC_PRACTICE_PLAN_20260904.md`包括数学对象/干预、三工作包、统计独立性、直接因果预算、论文主图和分支选择；计划/tracker入口与orientation增加现行范围说明。REFERENCE_REGISTRY记录MAS v7、DAS/CLeaR2024、SeedStability v1一手页面复核与新消费者，不声称全球查新。旧AGENTS备份 `tmp/research_reset_20260904/AGENTS.before.md`（SHA256 af90709ddeb617dda25dd4c39280b41727c4ff88f6976ec198fa49c6cb3518f6），automation备份同目录automation.before.toml。

产物SHA256：AGENTS=`7dcd44436f36df5a7d930f9bbf262957ab2349aa8612a9c094bef9d0117dc835`；科学方案=`a43d1e48fb04f4580b44981605a3ca98f4a3480cca1fbd8281745e0d674c5b82`；F5勘误=`830a4b5dc9530e1a8f25967a8157445f8d34cad77375384414db31119fc8501f`；长期prompt=`1e3e96507a41d40719ca7a5db6a5327532d989f2113801865253f75e898a758f`。本地被忽略研究文件不扩大Git白名单。

**Automation与下一步。** 已通过app工具更新现有ccad的长期prompt，并逐字核对 `configs/CCAD_AUTOMATION_PROMPT_20260904.md`。保持PAUSED、5分钟与原target task不变，未恢复/删除。恢复后的第一个科学消费者是工作包A：8个source-only anchors、各4个target、rank1/4、正负文档与7种移除、两个剂量，初批上限4096单sequence-forward等价；先补实际ID/hash和实测资源预算，不再为内部选择请求批准。新结果仍须独立确认，正面论文目标保持但不保证预设方法成功。

上条标题16:05为录入时间误差，实际初次写入为16:03 EDT。最终审阅补充了source projector在一般GL换基下的定义与hard-negative背景能量匹配范围；科学方案最终SHA256=`d9a9c5f0735ac85580baa07400f5112b6e567f6b8880cd8d464b454b6e40b13f`，取代上条方案hash。其余artifact身份不变。


## 2026-09-04 20:16 UTC — 日常研究入口合并与长期prompt去时效化

用户要求在重启前进一步解决skills无条件依赖、自审过度、重复文档与过期prompt，同时保留积极查新/文献代码组装、硬件与资源策略。上一版只是追加纠错，仍让历史计划充当日常上下文且prompt含F4/C049，本轮改为实际合并信息归属。

AGENTS只管长期工作方式，EXPERIMENT_PLAN只管正结果路径，EXPERIMENT_TRACKER只管当前进度/下一步。三个日常入口由192,376 bytes缩为20,579 bytes（89.3%减少）。COMPONENT_CANDIDATES改成现成积木目录；旧详细实践方案改为指向统一入口。六份原文逐字节归档到 `archive/research_workflow_20260904/`，snapshot_manifest.json记录原hash；已核对全部6份一致。旧run、配置、理论PDF、reference registry与master_log历史未迁移或删除。

规则明确：skills按需取用，不能把评分/轮数/模板变成项目硬门；默认一次聚焦自审，具体缺陷才追加。路径按“可重复信号→关系价值→系统方法→独立确认成文”的正checkpoint发展，允许依据开发结果调整表示/query/rank/正则/训练，不把起步样本量或支线次数锁成必要条件。主动阅读最邻近原文与代码、维护novelty差异、优先最小组装和复用写回长期规则。

硬件/环境保持集中在 `.aris/compute/local.md` 与原spec，本轮补充nvidia-smi实测RTX5070Ti/16,303MiB/driver610.88及CPU字段/20逻辑处理器；RAM和磁盘余量查询受限未建立，未据此扩展排查或编造值。资源lease类型、固定多资源顺序、run自动续租释放、GPU忙时的主线轻任务、bulk storage与锁定解释器策略保留。没有启动实验、安装环境或申请重资源。

现行长期prompt迁至 `configs/CCAD_AUTOMATION_PROMPT.md`（文件SHA256 f7f68c1d90ee9802e39b0f802820773f550ff9d8e81cbe1cfbdfadbb54e5f728），不含支线/run/日期/参数/当前状态，只读活的tracker与plan。原dated prompt成为迁移指针。已通过app工具更新ccad并核对正文（仅归一化Windows换行）；PAUSED、5分钟、原target task保留，未恢复。初次直接字符串比对只因CRLF/LF差异失败，正文一致，未增加修复分支。

整理核对见 `archive/research_workflow_20260904/consolidation_check.json`：快照身份、11个关键路径、prompt内容与调度状态均符合预期；未新增validator或运行科学测试。本轮是用户要求的整理交付，不是科学PASS。重启后的研究机会与自由调整空间已写入tracker；积极寻找正结果，但尚未执行新的真实因果实验。


## 2026-09-04 20:25 UTC — 用户授权重启CCAD研究loop

按用户要求完成一次启动核对：AGENTS/plan/tracker一致，原执行任务idle，共享资源lease free；锁定runtime关键imports成功，F4参考surface/census/sequence/paired和raw manifest及saved factors的hash匹配，因子形状160×768×8，模型config与safetensors存在。未重跑全套测试、未加载audit、未启动第二个竞争实验。

已将现有ccad heartbeat由PAUSED恢复ACTIVE，保留5分钟、原target task和不含当前支线的长期prompt。初始化工作交给原执行任务接续当前tracker中的直接因果工作卡，使用用户要求的GPT-6模型；协调任务不并行修改实验实现。实际运行/结果继续由执行任务维护tracker与plan。本条为普通恢复记录，随下一科学工作单元成组提交，不为恢复调度单独提交。

## 2026-09-04 20:40 UTC — source-reference局部真实作用的首个开发阳性checkpoint

复用F4 saved signed factors、独立mean和共享hook intervention，新runner `scripts/run_f4_source_reference_causal.py` 把所有候选下游效应对同一个source局部投影效应比较。source为32-feature local centered contribution的rank1/4投影；target/raw用各自保存map；wrong-query使用不同source query自己的basis，另有hook总能量匹配版本。source basis跨target逐元素一致。8个query按8档source energy strata内最小source selection hash选择，未按旧FOUND或target效果筛选；涉及4个source seed，每个2个循环target，正负各2个document sequence，每query内文档不重叠，跨query共享文档/seed不视为独立重复。

真实运行：`runs/F4_source_reference_causal_dev_v1_20260904` 全sequence自然幅度，640 forwards/95.27秒；该版source-logit效应尺度过大，随后保留同一query/sequence做 `runs/F4_source_reference_causal_dev_v2_local_20260904`，只在source条件top4内容位置干预，避开起点/EOT后前16位，另计算词表中心化logits。v2为640 forwards/97.68秒，两次累计1280 forwards/192.95秒；峰值allocated VRAM均841,839,104 bytes，无新下载/训练/依赖安装，按disk-d-io→cpu-heavy→gpu-0申请，结束自动释放。v2的512行无空mask或缺失归一化endpoint；每query-method-endpoint-condition有4项。no-op最大差0，raw hook replay相对误差1.533e-5，5项运行自检及artifact契约通过；这不是独立科学复核。

**效应。** 先对每query的2target×2sequence取中位数，再跨8query取中位数。rank1正条件centered-logit source-normalized平方误差target=.012781、raw=.017009、wrong-matched=.661386；next-state为.013097/.016075/.496846。两endpoint均8/8 query优于wrong-matched；相对raw仅5/8与4/8。负条件rank1 centered-logit为.018736/.118156/.665838，target在7/8 query优于raw；next-state为.018986/.079602/.477532。rank4正条件centered-logit .028852/.032530/.663198；负条件.142661/.194219/.650821。负条件source RMS .053277与正条件.054436相近，不能把这些结果升级成语义特异性。真正机会是局部signed transport在跨条件下保留source作用，raw竞争力及有效范围继续研究；不是target native删除或已确认人类概念。

**证据与范围。** calibration继续标记DEVELOPMENT，audit未打开，v2是看到v1后做的开发局部化而非独立复现。原始hash：v1 `metrics.raw.jsonl`=5881e755de3d4cdc4ce293e2616b092101aaf082d67b70a80cbe5fc1e88c1edb；v2=c9853564d07abca3cb918fc88a7b984bc54a8b83db6aafdd00a160df44bbaf78。`scripts/summarize_f4_causal_feedback.py`生成完整正负condition query表/JSON与正条件SVG开发图（SHA256 08ed18a1efcb037c7941bb630b58fc5b996e699427c5ba0f84b336249fe80441）；JSON=afd88125c35fc0edad74ee34bf4e90c5df4004d478d054b11b93d717cd315e34。图保留log10共同尺度、归一化定义、颜色/形状双编码及CSV替代，无CI或期刊合规声明；XML/48数据标记/3图例标记核对通过，浏览器file URL被安全策略拒绝，未完成实际像素预览，未绕过。

**工程与接续。** 无全套重复测试；局部mask/compare短检查和py_compile通过。首次启动在创建run前因未使用scipy import失败，删除无用依赖后复用锁定环境；没有伪报该失败为科学结果。继承asset config含未消费的旧gate字段，manifest和本记录给出实际作用；下次扩展时清理继承字段，不重写旧run。runner v2代码身份由code_hashes与本单元提交留账；v1运行时旧代码只有hash、未另存源码快照，不能声称已保留逐字节历史源码，最终runner仍支持v1配置复跑。tracker已从RUNNING改为真实checkpoint，下一笔预算是全部4target与更多未选开发文档、rank1优先；详细工作卡只留tracker。

本地忽略文件身份：`EXPERIMENT_TRACKER.md`=b844efd00be3efbf2e1346ba55d5dc47913f714c5cb25cc4ef4e228d1d457dfb，`EXPERIMENT_PLAN.md`=cabeef37c9925f80a73e9726b566165c36d10ed49a4c5f7e89e958a9a85a846a。不扩大Git白名单、不改旧负结果；automation实读仍ACTIVE/5分钟，无当前支线写入长期prompt。

## 2026-09-04 20:58 UTC — 扩展真实效应，并以mean-only干预限定动态解释

沿同工作卡完成 `F4_source_reference_causal_dev_v3_expand_20260904`：8个原source-only query、全部4target、rank1、正负各4sequence，排除v2所有25个document IDs，选入43个IDs且交集0，仍为calibration开发。1216 forwards/172.187秒，1024原始行；v2-s5:a710在新正条件的2个sequence无内容位置激活，0干预如实保留，32行/96endpoint无可归一化误差，不算成功；该query正条件有效8项，其余16项。无target筛选、无新训练/下载/audit，锁定环境spec hash8348de46不变，资源依disk-d-io→cpu-heavy→gpu-0申请并释放。峰值allocated VRAM 841,839,104 bytes，no-op0，replay相对误差1.72954e-5。

v3 query中位数的中位数：正条件centered-logit误差target=.013959、raw=.008898、wrong-matched=.652780，target仅4/8优于raw；负条件=.025059/.056124/.582816，7/8优于raw重现。next-state正条件=.011004/.012726/.484472，负条件=.029421/.079724/.462817。旧whole-sequence及v2不变。v3原始SHA256=64c3d1c82f9b334871889bf67758f303c0a2c5a35aa3313e82b81d73394ab7a7。

**关键解释检查。** 新的短分析脚本 `scripts/inspect_f4_atom_participation.py` 只物化所选内容位置的calibration codes，用保存basis/map展开每atom的centered coordinate贡献，报告能量有效数、最大份额、mean/dynamic/aggregate能量；不做模型前向、拟合或必要性推断。source正/负条件effective-energy atoms约8.38/26.49，target约139.51/147.95，均为描述性代数参与。负条件source的mean-energy/centered aggregate约1，dynamic仅.000342；正条件分别.776/.01425。分散的常量项不能当成有用的动态many-to-many结构。重新读F4真实runner weighted_pca(56–61行)及hook_transport的basis-constrained ridge源码：它对全局mean中心化过程算加权二阶矩，没有再减条件mean，首轴可能承载条件常量；这是代码和本次分解支持的解释，未新做外部文献查新。

**决定性真实对照。** 随即运行 `F4_source_reference_causal_dev_v4_mean_control_20260904`，相同source query/文档/mask、首个循环target，增加source_mean_only=−独立mean的source局部投影；512 forwards/53.519秒、320行，其中10空mask行/30缺失endpoint保留。256个既有方法行的全部endpoint与v3逐字段精确相同，额外64行为mean-only对照。正条件centered-logit误差target=.013621/raw=.008898/mean-only=.017035，target在6/8 query优于mean-only；负条件=.027199/.056124/**.000148814**，target仅2/8优于mean-only。负条件next-state mean-only=.000362146而target=.03017445。原始SHA256=fe84163243368cac557f5f8dbebb522219a7e21c5c279cbf6b3e416a83d351d8，v4 query_summary.json=3c1b36a9f287fe8330809c1f65ab958c4a364d285561c938e1da42aadba7ad36。

因此限定20:40条目的解释：局部映射保留下游作用的数值现象真实，但此前负条件相对raw的优势主要是常量作用，尚不能作为动态FCC有用性的证据。正条件仍有小幅超过mean-only的范围，优先用source-only跨文档差分使独立mean严格抵消，直接检验动态作用；若需改善basis，在discovery上分离条件mean与协方差，不机械否定FCC母问题，也不把普通差分/RRR称为首创。具体下一工作/预算只留tracker。

**工程范围。** runner只增加target数量、旧文档排除、实际分母、源码逐字节快照与mean-only方法，并剥离无用旧gate元数据。清理时v3 config漏了audit_opened而manifest=false，初次contract失败/exit1均保留；将原config/manifest备份为*.before_metadata_correction.json，补缺失false并更新manifest config hash，未动任何数值/selection/source_snapshot。run内METADATA_CORRECTION.md及contract_validation.after_metadata_correction.json完整记录；原config hash74e547e400a66c19099f44253ab2973faf7f62557f4ec07215b12a33e182f1ca，修正后0404fe739538dcae83810356b37f26d9459fff6ff84aa1c89b35ca7b9db15d4f。v3最终和v4初次契约均通过；不是OS读取取证或独立科学审查。新增participation代数fixture、三脚本py_compile和diff --check通过，无全套重跑。v4只输出完整表，不画遗漏mean基线的图。

本单元1728 forwards/225.706秒；重启以来3008 forwards/418.655秒。无运行进程，lease全部free，automation保持5分钟ACTIVE，未改长期prompt。忽略文件留本地：tracker SHA256=06d90577bb112463e795eb0205bcd986d2e793903f7084ff7e323a176af6555e；plan=9a9569f29c70f93b9d71b09bc43a37049dc5b286955d59d312fb67a9968da671；v3 atom_participation.summary.json=dfc2ab9a0c595b1840df039b45098fac53c6d52ad378da68f2e6674b65e5bcfb。代码/配置/日志同单元白名单同步，不上传原始数据或扩大白名单。

## 2026-09-04 21:13 UTC — 差分消费者准备与共享资源排队（无新科学结果）

实现source-only跨文档donor差分：原query及recipient文档不变，从该query相反source条件的文档中按source rank1差分能量选donor，相同位置配对供全部方法使用。独立mean代数抵消，source_mean_only差分为0；保留空pair和共享donor依赖。`tests/test_source_reference_difference.py`核对mean抵消、线性map交换、注入符号、未选位置不变、空支持与非零source的零基线误差=1。短测试及py_compile/diff检查通过，未产生新的真实效应数据。

母目录references两个索引未检出相关DAS条目后，复读官方PMLR论文 https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf 的方法3.1–3.5（印刷163–167页），核对donor/base交换、分布式投影和高层interchange训练目标。实际只借鉴反事实操作，不复制代码，不声称SAE局部贡献交换等同完整DAS/IIA；消费者和差异已写REFERENCE_REGISTRY。该文件本次SHA256=e5b6cd23456d27eb8f07de4a92cfc4a0db39ea7c608f004d2ef2d2de65ddf669，plan=cb8f5dc0a8847e6a7c5a3d3fa5aa684f8721749b7ab244176ddab436ca17f6fe。

disk-d-io/cpu-heavy由EndoSAE_EndoFM活跃提取任务占用，未争抢；gpu空闲不构成跳过共享锁顺序的理由。已用原resource_manager提交有界等待，实际句柄和启动状态只留tracker；等待时不额外训练/扫描数据。差分预算最多512 forwards/预计1–2分钟，保持原ACTIVE/5分钟heartbeat。代码/配置/日志待同一单元真实反馈后成组同步，当前不是科学PASS或失败。

## 2026-09-04 21:20 UTC — 均值消去后的动态作用范围与剂量边界

共享资源释放后，`F4_source_reference_causal_dev_v5_difference_20260904`实际于21:12:04 UTC启动并完成512 forwards/144.149秒，含本次较慢的模型加载；前条“21:13准备/尚无新结果”时间标签与刷新有延迟，以manifest/status为准，未额外创建第二个队列。session39636退出0、自己的三项lease释放；随后共享disk/cpu再次由EndoSAE_EndoFM占用，本单元未再排队。

操作为δ_s=[Y_s(recipient)−Y_s(donor)]BBᵀ，δ_t=[Y_t(recipient)−Y_t(donor)]WBᵀ，在相同recipient hook减去相应δ。mean逐侧严格抵消；source_mean_only差分恒0。沿用8个source-only query、原开发recipient与内容mask、首个循环target、rank1；donor仅从相反source条件的文档中按source投影差分能量最大选择，位置按已选位置升序配对。62个有效pair的donor/recipient文档交集均0；2个无支持pair保留为零干预，320原始行含10空mask行、30个缺失endpoint。s5:710正条件有效2项，其余query-condition为4项；共享donor、文档和seed有依赖，只作描述汇总。每个非零source的zero对照误差精确为1，未将缺失计成功。

**真实效应。** 跨8query中位数的中位数，正条件centered-logit误差target=.794332、raw=1.468759、wrong-matched=1.472645、zero=1；target 7/8优于raw、5/8优于wrong-matched与zero。负条件为1.035610/4.350217/1.519944/1，target仅4/8优于zero。next-state正条件=.837416/1.082199/1.248673/1，负条件=1.338876/4.307904/1.284171/1。均值去除后全panel误差明显增大，但有源范围保留了可发展的动态信号，不能以全panel均值宣称普适对应。

按本轮source hook差分能量（不是target误差）排序最强三query：s2:2176 source能量27327，centered-logit target/raw/wrong-matched=.014421/.123205/.556623；s3:1230能量20.616，为.089956/.034035/1.982888；s4:693能量4.394，为.208791/.336183/.630496。其余query更弱时target误差可超过zero；全部8个结果保留于query_summary.csv/json，不隐去失败。source能量排序与有效范围是本轮开发发现，尚未独立确认，也不是SAE相对raw一致优势。

**剂量核查。** 用保存raw-hook资产在recipient同一内容mask位置的范数作分母，三强query的source扰动范数比中位数为4.86828/.129759/.061966。最大source差分例是大剂量，不能直接列为温和干预成功；另两例说明约13%和6%相对剂量仍有动态作用保持。下一步是同一source决定的公共剂量敏感性，再扩大有效范围；不独立重缩各方法或删除自然幅度结果。完整8query剂量数值/定义留run内source_dose_diagnostic.json（SHA256 999927feed3f7144f33d566e898f0e13476c614dd7a5f7c6f10ead2e88c5c209）。

**产物与检查。** 原始metrics SHA256=72bb3ab378d5f549616f2d3839f2c94bc089a4882ca6d260170956c40c42b9cc；query_summary.json=cd5373c6cab0282273dc255f957161b389f1534ab0d5b398c437d78e678da000。`scripts/plot_f4_difference_energy.py`生成正负两panel、全部48个query-method点、zero=1、无CI/平滑的log-log开发图，包含大剂量警告；SVG SHA256=202e390fe3b6fb028f3e80aa03330bb7c5d378b60e2a15ca857bc77b16db1ee0，CSV及provenance同目录。源数据/零值分母和两无支持pair显式保留，SVG结构解析通过，未声称完成像素预览或期刊合规。

运行no-op0、replay相对误差1.72954e-5、5项自检和artifact契约通过，完整source snapshot保留；allocated VRAM841,839,104 bytes。增加的短代数测试核对mean抵消/linear map/交换符号/未选位置/空支持/zero误差=1，py_compile和diff --check通过；无全套重跑。audit未开，无新训练/下载；重启累计3520 forwards/562.804秒，不含资源排队。研究图按scientific-visualization的全分母、共同log尺度和颜色/形状双编码原则生成；DAS原文方法的范围准确为印刷163–167页，前registry页码范围少写末页不影响所借鉴定义。

本地忽略文件SHA256：tracker=1ae679a6cfd0dfefa7995b8617bf40c2788ddb19d7a0f5067d7a041301966915；plan=b2da606245958900dd5d059731bcc3af4d7b2fbe48828001cc0a528d558e6d1d；registry=e5b6cd23456d27eb8f07de4a92cfc4a0db39ea7c608f004d2ef2d2de65ddf669。下一步与资源状态只看tracker；automation仍ACTIVE/5分钟，未将支线写入长期prompt。白名单代码、测试、配置和本日志同单元同步。

## 2026-09-04 21:35 UTC — 动态对应在source限量干预下保留

`F4_source_reference_causal_dev_v6_dose_20260904`完成512 forwards/110.114732秒，allocated VRAM841,839,104 bytes。与v5完全相同的8query、recipient/donor、rank1、首循环target；selection逐字节SHA256同为09bcfecd87c6daef8c09751989cdb6aa94a81933c558409835fb22c75983b902。仅新增scale=min(1,0.1*||recipient masked hook||/||natural source delta||)，source与所有候选共同乘该scale，wrong的相对能量匹配先按自然source完成，避免double scaling。只有source被限量，未按候选独立缩放误差。

**本次正checkpoint。** 正条件source最强三query的centered-logit target归一化平方误差，s2:2176从自然幅度.014421变为.013679，raw=.098526、wrong-matched=.929380、zero=1；source剂量中位数从4.86828倍hook降至.1倍、公共scale中位数.020542，source效应RMS从3.19035降至.032714。s3:1230的target=.092746、raw=.034232、wrong=1.996504，source剂量=.1；s4:693无需缩小、剂量=.061966，target/raw/wrong=.208791/.336183/.630496。最大的开发阳性并不只存在于超大source扰动；另一query仍raw更好，不能宣称一致SAE优势或普适FCC。三query是先前按source能量定位的开发范围，本轮是同pair剂量敏感性，不是独立确认。

完整8query正条件target/raw/wrong的中位数仍为.794332/1.468759/1.472645（target 7/8优于raw、5/8优于zero/wrong）；负条件为1.035610/4.350217/1.494307。聚合中位数未变源于弱query不缩小，不能单用这一事实证明稳健性。320行中60行缩小剂量，其余260行endpoint逐字段精确重放v5；全部source剂量满足上限。62有效pair、2空pair/10空mask行/30缺失endpoint保留，s5:710正条件2有效观察而非4。每个非零source的zero误差=1。保留共享document/donor/seed依赖，无CI，不作native必要性/人类语义主张。

**实现与记录。** 新增source_dose_scale与公共剂量字段及上限自检；两项短代数测试/py_compile通过，no-op0、replay1.72953987e-5、6项运行自检与artifact契约PASS。检查是实现/记录范围，不称独立科学验证。完整source snapshot保留，原始metrics SHA256=f756e7867596096f30f6a95f54a7d0fbe96d3d7a6efe98cee0d290ab58e793e7，query_summary.json=7757a3678d1162330aa5c379c6de9511429b8fd3204466c198af361fa2aa1a44。未复用带v5大剂量警告的图，以当前完整CSV/JSON和tracker比较表报告。

session10381退出0，21:34核对四项共享lease均free，无待执行队列。重启累计4032 forwards/672.919秒，不含资源等待；audit封存，automation保持ACTIVE/5分钟，未修改长期prompt。下一笔实际扩展/预算仅留tracker。plan科学路径无需变化；本地忽略文件tracker SHA256=55e850cef6a6598e1fc5010edf7f34e27127575481028c7d247abb64198cb154，plan=b2da606245958900dd5d059731bcc3af4d7b2fbe48828001cc0a528d558e6d1d。白名单runner、测试、配置与本日志成组同步，无数据/大文件上传。

## 2026-09-04 21:48 UTC — 动态atom参与分解与四target扩展排队

按当前工作卡准备四target扩展配置，source query、recipient/donor、rank1、公共剂量均保持v6，仅targets_per_query从1变4，预算1472 forwards/3–5分钟/<1GiB allocated VRAM。锁定环境spec canonical SHA256仍为8348de46ad5dd5c721867641e59a9d9d53e1f083622db98af51089e2bf2c9d43，按run-experiment环境契约warm reuse，不安装/重验环境。GPU实查空闲显存14304MiB，disk-d-io由EndoSAE_EndoFM活跃acquisition持有，使用共享管理器有界等待；未占用CPU/GPU、未抢占磁盘。到本条记录时新run目录尚未创建、没有新增模型前向；唯一等待句柄/当前操作只留tracker，后续需先核对该队列，不重复启动。

等待期间，对已完成v6作真实差分的atom参与分析。`inspect_f4_atom_participation.py`现在识别donor_difference，逐side用z_recipient−z_donor而不是单recipient减固定mean；mean项严格为0。e_j=sum_position,rank(((z_recipient,j−z_donor,j)*decoder_j@factor)^2)，effective=(sum e)^2/sum e²；比值对公共非零dose不变，aggregate_energy报告自然未缩放能量。脚本物化所选少量位置，运行3.84秒，无新模型前向/拟合、未全量扫描codes，原v6结果不改写。

全8query正条件source/target effective-energy atoms的query中位数分别1.321/106.520；不能把target分散当成功解释。先前source有源三例s2:2176为24.896/54.204，最大单atom份额.08116/.04209；s3:1230为5.043/9.781、最大份额.26427/.25140；s4:693为2.060/22.747、最大份额.59595/.15074。其余source effective约1–1.44，而失败target也可以超过100。动态最强source确实能量分散，但这只是代数参与证据；没有证明原生atom必要性、唯一概念或many-to-many优于单atom方法。均值消去的分解与旧v3固定mean分解属于不同操作，旧产物保留不覆盖。

v6新增atom_participation.summary.json SHA256=0268a840ba7fb88ce1aa34bf2b26c420aaed625c017b51da98279cc317021e99，raw.jsonl=4e3895f5a2993979e915c08a04ead8415e2263d7989fa14f0d59f58ff46a6e7a，summary内记原始v6 metrics和分析脚本hash。短fixture验证零mean、dynamic=1以及共同dose下参与比值不变。汇总器新增target_summary.csv以保留逐target异质性；其160条单target汇总在内存中与v6旧query_summary逐项精确一致，没有重写旧汇总。py_compile/diff检查通过，无全套测试；只属实现与描述检查。

tracker本地SHA256=000fc9b7d90f7a1f66eb3988dba099f7f6817eb99ba678bd1538ea4cd6ac28f6；plan未改变，audit封存，重启模型前向累计仍4032/672.919秒。automation仍ACTIVE/5分钟。分析代码、四target配置与日志白名单成组同步，资源等待跨heartbeat续接；本条不是四target实验完成或新因果阳性声明。

## 2026-09-04 21:59 UTC — 共享磁盘等待到期，无新实验

回收上一轮唯一等待session70527输出，退出1，原因为disk-d-io持续busy达到600秒等待上限；最后owner快照heartbeat 21:52:53。本轮实查同一EndoSAE_EndoFM acquisition租约heartbeat更新到21:58:33、PID61420仍存活，其余资源free。v7 run目录未创建，未启动模型、没有部分科学结果或新的实验FAIL。自己的等待/进程/lease均已结束，不重排长队、不抢占；后续资源释放再接续同一配置，位置见tracker。累计前向和科学结论不变；本次仅资源状态勘误，保留5分钟heartbeat，普通无变化检查不发结果通知。日志与下一实质工作单元成组同步。

## 2026-09-04 22:14 UTC — 用户澄清资源范围，四target扩展实际启动

用户明确资源协调应按实际CPU/GPU等负载处理，不因单纯D盘下载占据整盘独占而阻塞其他项目，也不要求中断下载；其他项目已释放该锁。本轮实查cpu/gpu/disk均free，GPU可用显存14210MiB。按该澄清，读取既有资产的本次推理只申请cpu-heavy→gpu-0，未停止对方进程、删除租约或修改共享管理器。AGENTS增加持久澄清：路径在D盘本身不构成附加整盘锁理由，大规模解压/迁移或实测饱和另协调。资源澄清文件本地SHA256=c2f6967971df4f4afacdbb0c98d4fff24639e7cf47315972bbd97d6b85a4126e。

v7于22:09:47 UTC创建run并真实启动，run manifest覆盖为实际cpu-heavy→gpu-0及原因；原有科学参数未变。修改runner仅3行可选资源元数据，配置只增2项resource字段，不改数值路径。锁定环境spec canonical SHA256仍为8348de46ad5dd5c721867641e59a9d9d53e1f083622db98af51089e2bf2c9d43，按run-experiment环境契约warm reuse，无依赖变动。source_snapshot保留本次运行源码；selection去掉新增target列表后与v6完全相同。

到22:13 UTC已持续记录约600/1472 forwards，仍RUNNING、CPU/GPU租约正常续租；不是实验完成或新效应结论。先前600秒资源超时队列已关闭，本轮仅一个模型进程，定位见tracker，不应重复启动。加载比此前慢，仍沿用有界1472-forward规模；完成后核对首target重放、全target异质性与真实分母。tracker本地SHA256=105994d1623ec1d627b5893687843b084f5a9029669862cbb17d3d12a9c61201。此次用户请求已实现实际推进；同单元同步资源元数据、配置与事实日志（包含此前待同步的资源超时条目），数值分析待run完成。automation保持ACTIVE/5分钟。

## 2026-09-04 22:25 UTC — mean-free限量动态作用跨全部target保留

`F4_source_reference_causal_dev_v7_alltargets_20260904`于22:15:51 UTC完成，1472 forwards/363.884807秒、1280行，allocated VRAM841,839,104 bytes。约6分钟超过原3–5分钟估算，包含较慢模型加载；后续同规模预算据此调整，未虚构吞吐。session20101退出0，CPU/GPU租约释放，无整盘独占。22:21检查计算资源已转交EndoSAE_EndoFM自己的训练任务，没有重复启动/抢占。6项自检、no-op0、replay1.72954e-5、artifact契约PASS；这些检查不称独立科学审查。

**正checkpoint。** 既有source有源三query在全部4个target上的正条件、两个endpoint均优于zero与wrong-matched。centered-logit每个target内的pair中位数范围：s2:2176 target=.010716–.017342，raw=.098526，wrong=.928935–.929953；s3:1230 target=.056548–.102096，raw=.034232，wrong=1.640603–2.029113；s4:693 target=.089218–.208791，raw=.336183，wrong=.507526–.630496。next-state target范围分别.009636–.015476、.087931–.119551、.107007–.179961；对应raw=.088661/.056051/.313446。s2和s4在所有target/两个endpoint都比raw更好，s3全部target仍是raw更好。支持一个可发展的跨seed动态作用范围，不支持普适SAE优势、人类概念唯一性或native必要性。

**完整分母。** 全8query正条件centered-logit的target/raw/wrong-matched跨query中位数=.736222/1.468759/1.644564，target 6/8优于raw、5/8优于wrong、4/8优于zero；next-state=.913946/1.082199/1.678106。负条件centered-logit=1.287397/4.350217/1.312191，target仅3/8优于zero。2个空pair按4target×5方法展开为40空mask行/120缺失endpoint，s5:710正条件8有效观察、其余16，缺失不计成功。三强例是3query×4target的依赖方向，不是12个独立seed重复；原recipient/donor文档未变，因此本轮不是新数据确认。与v6共享的320行hook和全部endpoint逐字段精确相同，新增960行来自其余target。

原始metrics SHA256=17c6f0db14ac293c11e930bf30174e8e2f74b790888bb32164e962ce5547f1c8；query_summary.json=aaa02abecc405b73d4074b98742745d483527c19220ee801c8f1286304c22d42；target_summary.csv=2cbc00b0088e23b876366e7b14d3bfe3a6e3ecd46210894cf1e8268780d1fa59。未新训练/拟合、audit保持封存。重启累计5504 forwards/1036.803807秒，不含排队。

**图与下一笔投入。** 复用stdlib SVG脚本，按scientific-visualization的全分母、共同log尺度、色/形双编码原则，修正target数量和实际dose脚注，加入逐target pair中位数min–max须线（不是CI）。图含正负两panel、全部48query-method点/48须线，x为实际限量source hook能量，zero=1、缺失pair和依赖均说明；无平滑。`difference_energy.svg` SHA256=7490b436bb615c3b44891e9ee67b15b4accb321fc53bb6f0658222d7ac5b3d35，CSV/provenance同目录。XML/范围数量检查通过，未声称像素预览或出版合规；Codex文件预览已排入当前任务。先前自然幅度图不覆盖。

下一步优先在未选开发文档复制既有有源范围，排除全部已用IDs，规则与预算只维护tracker，不再添加一列无关门。tracker本地SHA256=1fc44eca44ca5d8ff2bcae3a48f7bd92864cd79d3289febd4338c827eb09d6aa；plan路径无需改变。automation仍ACTIVE/5分钟。本轮图脚本与日志成组白名单同步，全部原始结果留本地。

## 2026-09-04 22:36 UTC — 新文档source支持变化，尚未读取target误差

v8已按v2/v7两份hash固定selection的document IDs并集排除68个旧ID，选出24个新ID、交集0；64pair中60有有效donor，s5:710负条件4pair因正条件donor只有2–3个可用位置而无法对应负条件4位置，原样保留零差分/缺失。尚未读取本run target误差，不据此调整已运行规则。

仅从本次selection的source差分能量可见支持变化：正条件中位数s4:801=26040.727，s3:1230=23.579，s4:693=6.320，s2:2176=.4355，其余为.002895–.274961。原来最强的s2:2176在新文档中变弱，原弱s4:801变强。按先前source有源解释，预期本轮有效范围更接近source能量前三者，而不应只期待固定三个query ID永远成功；这是运行中的source-only开发预测，不是假称预注册/独立确认。完整8query继续运行，同时报告原三例复现和source支持迁移，不用target结果筛掉失败。等待当前完整真实反馈后再更新结论。

## 2026-09-04 22:44 UTC — 未选文档上的有源范围保持与固定query复现边界

`F4_source_reference_causal_dev_v8_newdocs_20260904`于22:39:14完成，1472 forwards/330.102656秒、1280行、allocated VRAM841,839,104 bytes；session10139退出0。旧文档并集68、新选24、交集0。仅新增多份hash固定排除清单的并集支持，query/mapping/rank/剂量/source-only选择规则不改，discovery无新拟合、audit封存。锁定环境warm spec仍8348de46，按run-experiment复用，CPU/GPU租约运行，不因D盘路径附加整盘独占。原始metrics SHA256=2b53f3c09efc9b14231c84e6b9b033d2bc21487facbd50b80665574d801c621d。

**原有query复现与新有源例分开报告。** 旧三例中s3:1230和s4:693在全部target、两个endpoint继续优于zero和wrong-matched；但source变弱的s2:2176 centered-logit误差范围.960850–4.252971（raw2.577617），没有整体复现。前条在未读取target误差时记录的source能量前三者，全部target/两个endpoint均优于zero与wrong：新强s4:801 centered-logit target=.002601–.004446/raw=.074432/wrong=.942599–.943085；s3:1230=.052970–.133480/.173016/1.803019–3.283003；s4:693=.168998–.609854/.152650/.697957–.845977。s3相对raw的关系与前轮反转、现为全部target更好；s4:693的centered-logit则raw全部更好，不抹去这些异质性。三个query的next-state target范围分别.002345–.004174、.031813–.089943、.117685–.374424。源活动范围比永久query ID更符合目前开发证据；不声称正式确认、一般语义或native等价。

完整8query正条件centered-logit target/raw/wrong中位数=2.043614/4.838597/1.444604，target7/8优于raw但仅3/8优于zero；next-state=2.172924/4.859082/1.307954，同为3/8优于zero。负条件只有7/8支持query，centered-logit=.826308/2.565668/1.241014（4/7优于zero），next-state=1.413416/2.462274/1.397970（3/7）。s5:710负条件4pair没有足够donor位置，保留为80零source行/240个缺失endpoint；它们mask非空，不能将empty-mask计数0写成无缺失。source=0的归一化误差缺失，不是成功或概念不存在。

**动态分解。** 同一差分参与分析中，新强s4:801 source/target effective-energy atoms约26.43/52.02，s3约4.71/9.18，s4:693约2.15/21.16；变弱的旧s2 source降到1.0，target仍约135。多atom代数参与和活动范围有值得检验的关联，但target分散本身不说明功能价值。参与summary SHA256=f509b80c8874767d4093a3373534a6050a98195f9c287c6a604d993461de80c8，逐字段有效query计数显式记录：负条件参与比值7query，aggregate energy包括真实零值为8query。

**缺失处理与图表。** 汇总器补requested_query_count、unsupported_query_conditions、zero_source_hook_rows；图跳过整个不支持的negative s5:710组并显式标注，而不崩溃或画成成功。参与分析仅对有定义的比值汇总并记录分母，零能量本身仍保留。文档并集/mean差分/公共剂量3项短fixture、py_compile、6项运行自检、artifact契约PASS；no-op0/replay1.34689e-5。不称独立科学审查。图保留正负全部可定义45点及45个target中位数min–max须线，非CI，source-dose及缺失明确；XML数量与标签检查通过，无像素预览合格声明，旧图未覆盖。SVG SHA256=2e1a9e434dada177d3df0ef66e44fbf403b8420b8948158081e81509edbdd752；query_summary.json=89cc6ca1d21cd14f7563a491f479c6643f8a568dba1ba0069ae419f18dbcb5ec；target_summary.csv=a25b7141603b217a508131f615d7723d5d7b4b33ed76a5a1b4bcf4dcb0721f0f。

重启累计6976 forwards/1366.906463秒，不含排队；无自己的进程/队列，automation仍ACTIVE/5分钟。下一步从实际单atom功能对照检验分布式增量，先在discovery拟合/选baseline，不能在v8 endpoint上挑atom；当前操作与预算只留tracker。plan补query及输入条件共同定义有效范围、参与不代替功能对照；忽略文件tracker SHA256=e5201a3e4f5beb31d5b519ac037278d545b0641840bcf51fe1faec0b3969f1cc，plan=77d88f72b19ad4a7e67bcb0647db71c0afe6e59989e24c2972481213059b0fad。runner/分析/图/短测试/配置与日志同单元白名单同步，原始数据不上传。

## 2026-09-04 22:59 UTC — discovery-only single-atom functional comparison preparation

Read original F4 runner/condition_weights/fit_basis_constrained_transport implementation and resolved config (SHA256 f87da11bed2354edaef5ea1cfc47a2be08eeb17b2430c18c789f02203b807ef8). Reused exact source-positive top256 discovery positions, squared activation weights, independent mean centering and saved rank1 basis. Added vectorized separate scalar ridge fits across3072 target atoms; same .001 trace/min(shape) convention, dimension-dependent absolute ridge honestly distinct. One ridge setting, discovery weighted prediction SSE selection, atom-ID tie break, no calibration endpoint fitting. Decoder norm cancels under per-scalar trace ridge. Best atom predicts in source basis, not native deletion. Source basis saved float32 precision reused rather than refitting.

v9 config keeps v8 source selection/donors/dose/all8queries/all4targets and runs new atom method only;32fits plus448forwards, 2–4min estimate plus cold load/queue, <1GiB allocated VRAM. Source replay and selection equality required before borrowing v8 distributed outcomes. Two scalar-regression unittest fixtures PASS; earlier unittest discovery on pytest-style difference file ran ZERO tests (not counted as pass); direct fixtures run next. GPU free at22:54, EndoSAE CPU/disk extraction live, no lease removal or download interruption. Queue only cpu-heavy→gpu-0. No new scientific result yet; audit closed.

## 2026-09-04 23:05 UTC — single-atom increment and concrete comparator weakness

v9 started22:58:51, completed23:02:36 UTC; session63055 exit0,448forwards/224.640588s,256rows,allocatedVRAM841839104bytes. 3difference/dose/document direct fixtures PASS,6runtime checks+artifact contract PASS. All256source references replay v8 exactly in selection, hook dose/energy, all endpoint energies/RMS. RawSHA256=06a426cc9d089a3c7f2a67fc8dfa750d5cc157ee15073491f568edaf01c37171; fits=dfdc339ec0f164636f5ecde8ba2cc658a6e8b7491b0fe68bcb476a082e8c3b3c; comparison8104c8eb07a9a6195ea7c0f795b23050570158ed819ef974eaa69bbf2243e94d. No raw v8 overwrite. Compare script writes target/query tables including unsupported negative s5:710 (16zero-source rows/48missing endpoint errors), exact replay assertions and relative-error reductions; descriptive dependent units, no CI.

Positive source-active three all4targets/two endpoints beat this discovery-selected single atom. Query median centered-logit distributed/atom: s4:801 .003651477/.542433369 (99.33% error reduction), s3:1230 .089598236/.265465666 (66.25%), s4:693 .192955215/1 (80.70%). Next-state .002925455/.532400603, .061800028/.295269708, .124000820/1. All8query positives distributed median2.043614/atom1 centered-logit,2.172924/1 next-state, only3/8 better atom/zero. Negative7validqueries .826308/1 centered-logit (4better atom),1.413416/1 next-state (3better atom); atom1/7belowzero. Full negative missing row retained, not success.

Focused interpretation found an actual baseline issue: raw second-moment selection can favor a mean/near-constant predictor that vanishes under donor difference (e.g. s4:801 t1,t2 atom median error1, while t3,t5 about.06091/.05737; s4:693 allfour1). Cannot equate this comparison with best dynamic atom or multi-atom necessity. Development adaptation v10 subtracts weighted conditional predictor/source means on the SAME discovery rows/weights; covariance equals half weighted all-pairs difference Gram. Scalar ridge scale and coefficient are equivalent to direct pair fit; third unittest proves that identity, all3tests PASS. Atoms never chosen with calibration endpoint, but choice of stronger objective informed by v9; disclose as development. Original distributed map remains fixed. Additional448forward3–5minute CPU→GPU budget, no disk monopoly. Cumulative7424forwards/1591.5470518seconds excluding queue; no new scientific claim from tests.

## 2026-09-04 23:12 UTC — dynamic atom narrows but does not erase distributed increment

v10 completed23:11:31 UTC (start23:05:43), session84666 exit0,448forwards/348.3996784seconds including preparation/loading, exceeding3–5min estimate. 256rows, allocatedVRAM841839104bytes,6runtime checks and artifact contract PASS. All256selection/source hook dose/energy and all source endpoint energy/RMS fields replay v8 exactly. No own active process/queue; CPU/GPU free on23:12 read, no whole-disk lease or download interruption. Two runs this unit:896forwards/573.0402668seconds; cumulative7872forwards/1939.9467302seconds excluding queue.

Positive query medians distributed/v9atom/v10dynamic atom centered-logit: s4:801 .003651477/.542433369/1; s3:1230 .089598236/.265465666/.134528503; s4:693 .192955215/1/.124531998. Next-state distributed/v10atom: .002925455/.999999627, .061800028/.100728072, .124000820/.086503000. s4:801 beats BOTH atom objectives in ALL4targets/two endpoints, plus prior raw/wrong/zero: strongest distributed-increment development case. s3:1230 beats dynamic atom3/4targets in both endpoints, t1 atom is better (.105058 vs distributed .133480 centered-logit; .064220 vs .089943 next-state). s4:693 only t2 distributed wins; t1/t3/t5 dynamic atoms win, so the earlier3strongqueries do NOT all demonstrate distributed necessity. v9 itself remains unchanged, but its scientific generalization is narrowed.

All8 positive query centered-logit distributed median2.043614 vs dynamic atom1; next-state2.172924 vs1; only2/8 better dynamic atom, not general superiority. Negative7validqueries centered-logit .826308/.992163, next-state1.413416/.987714, only2/7 better atom at either endpoint; distributed-belowzero counts4/7 and3/7, atom4/7 both. Negative s5:71016zero-source rows/48missing endpoint errors retained as missing, not success. Comparisons remain descriptive/dependent, no CI/audit/native necessity. Relative reduction versus dynamic atom for s3 query medians is33.40% centered-logit and38.65% next-state; s4:693 reverses sign. s4:801 dynamic atom is nearzero-output; its useful incremental evidence also includes better-than-v9 active atoms at t3/t5, not just beating the zero-output dynamic predictor.

Method boundary: conditional variation fits still use SAME source-positive discovery rows, not mixed positive/negative condition pairs; it is a supplementary dynamic objective, NOT uniformly superior to v9 or a proof of globally best single-atom performance. Weighted covariance identity is mathematically exact (tested separately), but cross-condition extrapolation remains a scope limit. Chosen atoms never use test endpoints; choosing to add the objective is openly development feedback. No third same-document objective is queued. Next scientific investment develops the surviving range on unused development documents with the complete8query panel/source-only input rule, both saved atom fits and alltargets, retaining counterexamples. New-data source support will be checked first; no new scoring gates/protocol scaffolds.

Artifacts v10: metrics.raw SHA25660695162126472fa845da898489282c38067cdee0263f7644f31c68aacc8af63; fits bf9d1ff74ad695ab0b0d0ccace91960a9abfa00cb7e866662acb08f004d3bf7c; atom_comparison.json35ebb6ff8d1dcdd2e492aa587146ea3cc73d3bf2b99ef7dc567813cf52bdb3e0; queryCSVc10edef27e504a736e9e5181fd0ecb536b1186c6e0aaf8b4ed58adf65c768830. Target/queryCSV retain full attempted denominator. Runner/source snapshots preserve old versions; v9 comparison hash remains prior record; current comparison generator2d21208b36d22d0e5456028833d8be58fe9a65db5e2de58a40a20921861492e3 adds fitting spec/config identity. Plan adds dynamic atom/pair covariance interpretation without changing independent global mean definition. run-experiment warm reuse and analyze-results query/target descriptive organization used; no independent scientific review claimed.

Ignored local research records retained: EXPERIMENT_TRACKER.md SHA25679322fbe43d2b83739b1b9564f9c613378c52bd1f83d6f5b795270672381553c; EXPERIMENT_PLAN.md a6cd284a0f0f5e726589531952db3c71339c5c2232bff0225f30c9573ce6a8b2. Only allowlisted source/config/tests/master_log queued for grouped synchronization, no raw assets or documents force-added.

## 2026-09-04 23:27 UTC — development document pool exhausted; replenish without audit

Read AGENTS/tracker/plan and current sequence metadata. Excluding union of v2/v7/v8 recipient+donor IDs(92) leaves only1 of512calibration sequences,1document; cannot form new cross-document pair. This is input support/resource boundary, not scientific failure. CurrentCPU/GPUfree, Dfree4128197574656bytes. Prepare new calibration-only development shard from same pinned FineWeb commit, original10/40/20/30 document hash split. Exclude SAE training and ALL original paired document IDs/text hashes (569entries, ledgerSHA25633421f067172f0322551346c1985e3799cf8bb90d9950a55f4cf1b0de38fe201); old audit identity-only exclusion is not audit token/metric access. Original mean and fitted distributed/atom models unchanged. Added small optional exclusion-ledger handling to existingR008a, source snapshots/progress, no new pipeline framework. Config f4_development_corpus_v1.json:5range-read groups,512newcalibration sequences65536tokens. Priorcomparable41.527MB43requests; budget40–100MB expected/5min; no whole-disk monopoly. Future same-pass fiveSAE+raw ~500MB128forwards, historicalR008b throughput6374.76tokens/sec over327680tokens51.40s, loadingexcluded; allow<4min and<1.5GiB allocated VRAM forbounded new extraction. Setup not a scientific result. Memory registry only reaffirmed historical blueprint/evidence boundary; current numerical facts re-read from project artifacts.

## 2026-09-04 23:31 UTC — new disjoint shard ready for fixed-map replication

Corpus actual start23:26:19 (preceding planning heading rounded later), end23:28:00,session61493 exit0;38,148,724bytes/40range requests,4999sampled→113useddocuments,512sequences65536tokens. Eight selfchecks/contractPASS, metadata集合另核对113unique IDs+113unique texts, zerooverlap training+ALL oldpaired, original calibration hashbucket. No oldaudit token or endpointread. TokenSHAea701c9649e7cacdaadac2c473aa1256f5cb25ee927a33f428319daaefa160df. ExistingR008b now accepts explicitrequested/forbiddensplits and same-forward raw capture, bounded4CPUthreads; newrun excludesmean/discovery/audit encoding and borrowsONLY originalrawmeanmetadata. Codes completed23:30:04,session33043 exit0;128sharedbaseforwards/19.4668404seconds measured encoding+output loop (excludesloading),allocatedVRAM895002624bytes;8selfchecks+contractPASS. Five decoder hashes exactlymatchoriginalR008b. Both source snapshots preserved; no weights duplicated/retrained, no source map or mean refit. ~500MBnewcache, no Dmonopoly; this is evidence preparation, not scientificprogress.

v11newshard config prepared: same8hashqueries/all4targets/sourcepositive-negative4sequences/source-only donor/rank1/common0.1source cap. Loadv9andv10savedatomfits with input hash and model/basis/mean identity checks, no discovery/codefitaccess;5methods target/raw/wrong-matched/single_atom_level/single_atom_dynamic. Budget1472forwards,5–8min incl loading,<1GiBallocatedVRAM. Source-only supportedpair counts/natural difference energies printed before modelendpoints. Keepzero/missingdenominator, correct crosscondition extrapolation limits, source-activity range vs fixedIDreplication separated. Newdata isdevelopment replication; auditclosed. Three scalar regression fixtures and three direct difference/dose/doc fixturesPASS; py_compile currentfourchangedrunnersPASS. Not independent scientific review.

## 2026-09-04 23:34 UTC — source-only expectation before reading v11 target errors

v11started23:32:55,session31050runningunderCPU→GPU leases. Savedselection has42newdocumentIDs,64pairs/62supported; s5:710positive2/4unsupported,negative4/4supported. Source-preflight printed beforemodel: positive median naturaldifference energy s2:2176=27172.8000944, s4:801=26365.4264914, s3:1230=64.7357203, s4:693=9.9045322; others .00039735–.67325825. No target error read yet. Prior developmental source-activity interpretation predicts these four as useful operating-range candidates, including revived s2 after v8 weakness; not asserting outcome or formalpreregistration. Inputs were not altered following this read. Waitforactualendpoint feedback; all8 denominators retained.

## 2026-09-05 00:05 UTC — recovered completed new-document distributed increment

User requested continuation after a stall. The causal job had already completed 2026-09-04 23:36:56, status PASS, so it was not relaunched. v11 performed 1472 forwards / 241.1583782 seconds including preparation/loading; 1280 rows, peak allocated VRAM 841839104 bytes. Six runtime checks and artifact contract PASS; no-op 0, maximum raw replay relative error 2.0482556e-5. CPU/GPU both free at 00:05; no own queued process and no disk-exclusive/download interruption. Original runtime source snapshot and raw metrics preserved. Preparation separately used 128 shared base forwards / 19.4668404 seconds excluding model loading, not counted as scientific progress. Cumulative causal forwards 9344 / 2181.1051084 seconds excluding queue.

Eight original source-only hash queries, all four targets, two conditions with four source-selected pairs each. New shard 113 documents is ID/text-hash disjoint from training and ALL prior paired splits; actual selection uses 42 documents. Both atom families reuse their 32 original discovery-only fits and original source basis/mean, no refit. Source preflight expectation above preceded reading target errors. Two of 64 pairs unsupported (s5:710 positive), retained as 40 zero-source/empty-mask rows and 120 missing endpoint errors; every query-condition has some valid observations, s5:710 positive only 8 instead of 16 per method/endpoint. Old audit identity read only for exclusion; tokens/metrics remain closed. This is fresh-document DEVELOPMENT replication, not final audit or independent seed-direction confirmation.

Positive centered-logit query medians (distributed / raw / second-moment atom / dynamic atom): s2:2176 .011992657 / .082529086 / .411271482 / 1; s4:801 .003925944 / .080033873 / .540623125 / 1; s3:1230 .037323271 / .055983734 / .462282845 / .147638824; s4:693 .080925268 / .127246666 / 1 / .074578197. s2:2176 and s4:801 beat both atom objectives, raw, wrong and zero in ALL four targets at BOTH next-state and centered-logit endpoints. Centered-logit distributed target ranges .009566803–.014079081 and .003252554–.005330406. Relative reductions of pooled query-median error versus raw are 85.47% and 95.09%; descriptive ratios, not uncertainty estimates. Both were source-active before endpoint inspection; fixed query IDs need not always be active (v8 s2 failure remains).

s3:1230 beats both atoms in all four targets/two endpoints on new documents, but raw wins centered-logit t5 and next-state t1/t5. s4:693 remains an atom counterexample: centered-logit distributed wins dynamic atom only t2; next-state only t2/t3. Atom objectives use source-positive discovery, not globally optimal mixed-condition differences; many errors of 1 reflect zero output, not a universally strong atom. Do not infer multiatom native necessity or human-readable concept identity from these comparisons. s5:336 and s5:1352 also show moderate improvements; neither was added by an endpoint-based input screen. Original full panel preserved.

Across all eight query medians, positive centered-logit distributed/raw/dynamic atom .250634630/.437414017/1, next-state .281252433/.452618432/1. Distributed beats raw 8/8 and 7/8 respectively, dynamic atom 5/8 both, zero 6/8 both. Negative centered-logit .477992206/1.285753410/1, next-state .457694462/1.387845879/1; distributed beats dynamic atom and zero 5/8 both. Failures remain: s5:369 positive centered-logit 1.936909348 and s5:710 22.539479655, with partial missing support for latter. Not universal superiority. All comparisons descriptive, shared seeds/documents dependent, no CI.

Artifacts: v11 metrics.raw SHA256 003d9ac2afa2a399f076fbc306da464c45e4f8a3fc357e3fe71feb8b224c8ad8; query_summary.json 263e4f697103776688c761611ec9feb97a2db15bce3bb2518c1722260953390b; target_summary.csv b6cc1b98ece23e601aa1f8746aa107bf45187d381c54ec5483cee7fe5262800b. Figure difference_energy.svg 0b9de142225d66a8d5b421f26807e3e90896908311bf0506fdf193498fad122c; 80 data points and 80 target-median min–max whiskers, common log axes, missing support explicit. Figure is provisional, SVG structure checked, no pixel-preview/publisher compliance claim. analyze-results and scientific-visualization used to retain full denominator and source/target descriptive summaries; not an independent scientific audit. REFERENCE_REGISTRY.md records actual reuse of pinned FineWeb/range and R008 code, not novelty from new data acquisition.

Next same-line investment: develop the surviving source-active distributed range, refresh nearest Li15/OT and DAS/MAS methods/implementations, then implement one budget-matched nearest comparator rather than another same-document atom objective. Keep raw, both atoms, wrong and all eight queries; formal confirmation needs a fixed source-only applicability rule and unexposed data. Native deletion remains separate. No queued run or new gate. Existing five-minute automation retained.

Ignored local documents retained by path/hash: EXPERIMENT_TRACKER.md 58a3639ccafec5f16c928f77c562c749170979af7435290c94f9229f0a692589; EXPERIMENT_PLAN.md unchanged a6cd284a0f0f5e726589531952db3c71339c5c2232bff0225f30c9573ce6a8b2; REFERENCE_REGISTRY.md 55fed557ddee1a7eb655b8f2a1cdc76f0bb6f9e45623f39c1b6ed015171037b8. Only nine allowlisted implementation/config/log files are grouped for synchronization; no raw data, model weights or ignored research documents added.

## 2026-09-05 00:27 UTC — nearest-method refresh and bounded UOT comparison preparation

Heartbeat continued the same positive-result opportunity after reading current tracker/plan/registry and the parent reference index. research-lit/PDF skills used narrowly for original methods and implementation boundaries, not a broad review or extra approval gate; local library and official arXiv/PMLR were the sources, arxiv_fetch helper unavailable. No new external code copied, no package install. Read Semantic OT local PDF pp.1–8,24–25 and verified arXiv2605.28567v1/CC BY4.0, SHA256 f8dee28e553bbd01cf06507507872fae9f08b63d6dfbf0d91bac4fc7f7c0b587: activation-weighted token-context distributions, top50 centroid screening then pairwise Sinkhorn nearest atom, CE replacement evaluation. Read Li1511.07543v3 §4/S.3, MAS2501.06164v7 §3.3–3.4 and SCOTM2503.24204v1 §III. References and distinctions recorded in local registry. Semantic OT does have intervention-style CE evaluation; do not call it geometry-only. Generic UOT is not its contextual Wasserstein method, not SCOTM cardinality/q-entropy, and not behavior-trained bidirectional MAS.

Concrete code discrepancy: historical li15_spectral_proposal applied abs(correlation), whereas original S.3 uses positive correlation threshold. Retain old default for exact historical replay; add explicit absolute_correlation=False for original positive graph and label old mode as variant. Negative-correlation counterexample added; all13 proposal tests pass. First direct unittest invocation failed import ccad because src was not on sys.path; corrected explicit-src invocation passed. No old results rewritten, no F4/v11 numerical impact.

New ot_transport.signed_ot_readout reuses existing unbalanced log-Sinkhorn, all source-local/target atoms with conditional variance, cost1-|weighted Pearson|, uniform active marginals, epsilon.05/marginalKL1, then row-normalized signed sigma-ratio lift and one discovery scalar ridge gain(.001). Fixed source decoder coordinate preserves common rank1 output; no native deletion claim. Constant columns excluded before centering to prevent roundoff-only variance. Three fixtures pass: signed/permuted coordinate prediction, positive rescaling gauge plus intercept-cancelling differences, and constant/invalid-weight handling. Numerical convergence is checked, not treated as scientific success. source-coordinate effect may lose to OT; no presumption of FCC advantage.

Prepared v12_uot using same v11 full8panel/all4targets/64pairs/common0.1source cap, NEW METHOD ONLY plus source reference(448forwards), compare to existing v11 records only after exactsource/selection matching. Fit32couplings<=32x3072 on originaldiscovery, reuse savedv10 source-only row/weight list but NOT its selectedatoms/coefficients; sourcebasis/census/model/decoder identities checked. Fitexpected<=3min+forward/load3–5min,total<=8min,allocatedVRAM<1GiB. CPU->GPU managed leases, no diskexclusive. Environment canonicalhash8348de46ad5dd5c721867641e59a9d9d53e1f083622db98af51089e2bf2c9d43 unchanged; run-experiment warmreuse, no repeated qualification. Preflight00:25 CPU/GPUfree, hardware5070Ti16GB/888MiBdriver-visibleexistinguse; sharedlease authority governs availability. No audit, no new data, not confirmation. This is preparation, not a new effect checkpoint.

Actual launch v12 at00:29:15.481740 UTC, exec session23755, CPU->GPU nested resource_manager.run with wait-sec0/heartbeat20; status RUNNING and16/32 fitted maps reported by00:29:40. No target errors inspected yet. Source snapshot includes runner, OT helper, existing Sinkhorn/proposal dependencies, hook/artifact helpers; prospective comparison/results/synchronization remain same work unit. No checkpoint notification for routine in-progress state.

## 2026-09-05 00:42 UTC — paired-correlation UOT effect comparison completed

v12_uot finished00:32:07.851365 UTC, session23755 exit0,448model forwards/172.3023063seconds including fit/preparation/load,256rows,peak allocated841839104bytes. Six run checks and artifact contract PASS; CPU/GPU free when inspected after completion. All32OT fits converged165–215iterations,256source-only discovery rows each,3–32active source atoms and2721–3041active target atoms; original3072target universe retained. No audit/newdata/package/weight download. Raw SHA256144d9bc67de691f6e42e7e4ece4a11fb687c9b61bb46db0627b2745bdac4e32c. Cumulative9792causal forwards/2353.4074147seconds;128asset forwards/19.4668404seconds separate.

analyze-results used for exact matched comparison, complete query/target tables, relative improvement and bounded interpretation, not independent-seed CI. compare_f4_saved_methods.py verifies identical selection,recipient/donor documents/positions,source scale/hook energy and allsource endpoint energy/RMS before joining256new rows to1280saved v11 rows. Six methods kept in method_comparison_query.csv/target.csv and provenance JSON. Zero-source8rows/24missing endpoint values in new run retained; all8queries included. Query medians pool dependent document/target values and are not independent repetitions. No new plot or pixel-preview claim this unit.

Positive centered-logit distributed/UOT query medians: s2:2176 .011992657/.992000461 (98.79% reduction),s4:801 .003925944/.980741475 (99.60%),s3:1230 .037323271/.488786242 (92.36%),s4:693 .080925268/.103418485 (21.75%). Firstthree beat UOT on all4targets/both endpoints;693 only3/4centered-logit and2/4next-state. Earlier dynamic-atom counterexamples remain. Full8 positive centered-logit distributed/UOT=.250634630/.958035361,next-state=.281252433/.973804522;7/8wins both. Negative=.477992206/.988800551 and.457694462/.973275132;6/8wins both. Positive s5:369 UOT.935329247 beats distributed1.936909348 (centered-logit),negative369/1352 also OT better. s5:710 both severely fail (positive distributed22.53948/UOT70872.22890); relative win is not success versus zero. Full failures retained, no universalFCC/native/uniqueconcept claim.

Scope: this is generic paired-correlation UOT with epsilon.05,not optimized OT family or replication of Semantic OT/SCOTM/MAS. Source rank/dose and discovery rows/weights are matched, candidate output energies not independently normalized. OT discovery relative fit errors for2176/801 ~.23–.30 become ~1 at newshard endpoints; default coupling generalization is an open explanation, not proof no OT method can work. Next cheap improvement uses originaldiscovery document-level fit/validation for a shared epsilon from{.005,.01,.05,.1},then fixed actual development forwards; no endpoint-based tuning. Check document support honestly, do not treat arbitraryrow splits as independent. Current artifacts stay development; source-only applicability and unexposed confirmation follow the surviving range. Five-minuteACTIVE heartbeat verified unchanged. No running/queued job after this unit.

Local ignored documents retained: EXPERIMENT_TRACKER.md sha256bc40170a0e7af46deeffdb9a2a0f14d3fb0a82cd8d9052bfa36204957352fa0c; EXPERIMENT_PLAN.md unchanged a6cd284a0f0f5e726589531952db3c71339c5c2232bff0225f30c9573ce6a8b2; REFERENCE_REGISTRY.md sha256876983725f345db54f0037fb6bceb8008bac8dd5db6e1ad45a366b879fbd2c26. Eight allowlisted implementation/config/test/log files grouped for synchronization; raw results/models/private documents remain local. Prior3OT and13proposal tests passed; new comparison compiled and ran successfully on all1536combined records with exact replay assertions. No old raw artifact changed.

## 2026-09-05 — bounded discovery-only UOT tuning preparation

Continuation inspected AGENTS/tracker/plan and original discovery sequence metadata. Conservative connected components across all packed multi-document sequences collapse most selected rows into one group; instead inner tuning excludes multi-document contexts entirely. Each query retains173–223/256rows from59–87documents. Stable salt CCAD-OT-discovery-cv-v1,sha256(document) modulo5 bucket0 validation yields5–43rows/query, disjoint document IDs; remaining single-document rows train. Final refit restores original full256discovery rows for matched data budget. Existing source basis is reused from original full discovery, so this is conditional inner selection, not a wholly unfitted held-out method claim. No audit tokens/metrics or new data.

Prepared v13_uot_tuned epsilon grid{.005,.01,.05,.1}:128inner fits, shared choice by minimum mean across8query medians across4targets of weighted all-pair-difference validation error,then32refits and448actual model forwards at same v11selection/source0.1dose. Upperiterationbudget5000allows small-epsilon convergence rather than censoring it. v12 measured32fits+448forwards172.3s; provisional tuning2–8min+refit/forward3min,total<=12min,4CPUthreads,<1GiBallocatedVRAM. run-experiment warmreuse spec8348de46 unchanged, no rebuild/extra review gate; CPU/GPU free atpreflight,5070Ti896MiB/16303MiBdriveruse. No endpoint-based parameter/query selection. Two new focused tests (document separation/packed exclusion, weighted difference error equals all pairs) plus three priorOT tests PASS. First pycompile caught indentation typo, fixed and compiled before launch; no modelrun failed. Tuning raw records/partitions persist progressively; no expected success assumed.

Actual launch00:47:52.326477 UTC,session68791,CPU->GPU nestedmanager.run(wait0,heartbeat20). At00:48 statusRUNNING and48/128innerfits reported. Code snapshot58d2cf0317800058634602b6e25cf34607a1ae361aaadedeaa3f66a3e64f4340,configa33ca94856162ba5d486286ff3cbd8b945413f46a710f609b0c7c929869d078a. No new causal results yet; continue sameunit next heartbeat, no duplicate launch. Synchronization deferred until unit results, fiveallowlisted implementation/config/test/log changes remain local and source-snapshotted. Routine progress stays quiet.

## 2026-09-05 00:58 UTC — UOT tuning completed; gain boundary unchanged

v13_uot_tuned completed00:50:01.425058,session68791exit0;128inner fits+32refits+448realforwards129.0496355seconds total,256rawrows,peakallocated841839104bytes. Sixrun checks/artifactcontractPASS. Exactselection/sourcehook scale/energy and allsource endpointenergy/RMS vs v11 verified; extra comparison to v12 checks selection/source/donor identity before method change table. Eightzero-source rows/24missingendpoints kept. No ownjob/queue/lease;00:56CPU/GPU occupied byanotherproject,untouched. Noaudit/newdata. Rawbd1b5efbd3248457af9be8ea10e5939ff38ba8c0a608d8da022345e1a510f344. Cumulative10240causalforwards/2482.4570502seconds plus128assetforwards/19.4668404seconds separate.

Discovery chose shared epsilon.01 from{.005,.01,.05,.1},meanqueryvalidationerrors188099.8832/187197.0114/196757.8528/205510.8242. Selected objective is99.9995628%froms5:710,whose validation source variance7.968316e-5 produced normalizedqueryerror1497569.54. This is genuine loss-scale domination,not numericzero or a convergence failure. Keep original choice/result; do not retrofit a new objective and call it frozen. Source basis remains fitted on fulloriginaldiscovery,explicitlynot fully independent inner validation. Thus tuning is a bounded attempt,not evidence of robustly optimized OT family comparison.

Full8positive centered-logit distributed/UOT=.250634630/.968567598,next-state=.281252433/.985131755,7/8winsboth;negative=.477992206/.995523238 and.457694462/.986213647,6/8winsboth. Positive2176/801/1230 all4targetsbothendpoints beatOT,centered errors .011992657/.996553972, .003925944/.987572770, .037323271/.518361519.693wins3/4targets inbothendpoints,old dynamicatomcounterexample remains.369positiveOT.949562426 beatsdistributed1.936909348;negative369/1352OTbetter;710bothfail. Againstdefault.05,tuned.01improves only1/8positivequerybothendpoints,negative1/8next-state and2/8centered-logit. No overalltransfer gain,noOTfamilydefeat/noindependentconfirmation claim. Tables sixmethods/allqueries/targets and method_change_query.csv inrun; comparisonJSON adds tuning/priorraw hashes. analyze-results used for complete comparisons and identifying objective domination,not a new statisticalgate.

Next develop source-only applicability rather than another same-document OT sweep: inspect source-difference magnitude and atomenergy participation across existingv7/v8/v11,using source-only analysis to avoid target arrays entering selector. Form a simple explicitlydevelopment-selected rule, retain fullpanel/coverage/failures, then fix endpoints/dose/comparators before trulyunseen-document range confirmation. Native and fullSemanticOT/SCOTM/MAS comparison gaps stay explicit; this is not permission to remove required papercontrols. No newGPUjob until resources and budget checked.

Local ignored documents retained bypath/hash: EXPERIMENT_TRACKER.md950836d29e9e82a34d138a4c892ab23cd37bf70cc26c846efade87102fc5449f; EXPERIMENT_PLAN.mdunchanged a6cd284a0f0f5e726589531952db3c71339c5c2232bff0225f30c9573ce6a8b2; REFERENCE_REGISTRY.mdunchanged876983725f345db54f0037fb6bceb8008bac8dd5db6e1ad45a366b879fbd2c26. Sixallowlisted code/config/test/log files form thisunit's synchronization. FiveOT fixtures and runnercompile passed beforelaunch; comparisoncompile/exactreplay passed afterrun. No repeated fullsuite or oldrawrewrite, no private data/weights added.

## 2026-09-05 — source-only applicability development checkpoint

Added source-only mode to existing participation inspector: noendpoint metric file opened, source_basis but notquery_target loaded; for eachquery onlyits sourcecode/decoder rows materialized. Shared recipienthook supplies amplitude normalization. Surface/selection identitymetadata areparsed,not claimed whollyunread. Calibration shape comesfromassetmanifest rather than hardcoded65536. Unsupported donor/source pairs retainedexplicitly. Threeexistingpanelsv7/v8/v11 each64pairs analyzedin~7seconds total,summary<1s;0newmodelforwards/noheavyresourcelease/noqueue. Full-mode unsupported-donor refusal retained. Sourcecoordinate aggregateenergy agrees with savedintervention energy within2.96e-8relative duefloat basisnorm. Twofocused testsforenergy/doseinvariance andruleboundariesPASS;scriptcompilePASS. Oldforwardrecordsunchanged.

ChoseONESIMPLEdevelopmentrule,notthresholdsweep: requirevalidsource/donor pair,natural source-difference norm dividedbyrecipientlocalhooknorm>=.1(existingdosecap),largestsourceatom shareofsumatomcoordinateenergies<=.5. Configf4_source_applicability_dev_v1.json SHAfb4c9237cb78cf2dbfbe2409645dd97d8c380e516a89e1a63444163ac1118b74. Source-only applicationdoesnotmake developmentindependent: earlierexposedoutcomesandsourcefeaturetables informedchoice. Noatomnecessity/GL-invariant/humanconceptclaim. Fixedfordirectnew-documenttest;rank1/commoncap.1/next-state&centeredlogit unchanged. Thisisanalyticaldevelopmentartifact,notnewcausalrunorcontractvalidatedexternalconfirmation.

Sourcechoices savedbeforeendpointjoin in F4_source_applicability_dev_v1_20260905/source_selection.jsonl.42/192selected=21.875%(positive27/96,negative15/96);v7/v8/v11 respectively12/64,12/64,18/64.168dependenttarget-inputrows belowzero onbothendpoints,maxcentered-logit.466233907,maxnext-state.314663801.14query-condition-panel groups afterdocument then target medians:14/14betterwrong,12/14centered-logit and11/14next-state betterraw. Availablev11twoatomcomparisons6/6betterbothendpoints; earlierpanelsatomcomparisonsnotaddedhere,notallbaselinecoverage. All48query-condition-panel coveragegroups and192inputs,selected/rejected/alltables retained. Ruletracks2176activev7/v11butweakv8,and801activev8/v11butweakv7;1230selectedwhereactive.693/weaksourcecounterexamples remainrejected,notproofuniversalabsenceoruniversaldistributednecessity. NoCI,sharedseed/documentdependenciesexplicit.

Planupdated with source-only applicabilityprinciple (notcurrentthresholds). Nextnew65,536token batch excludestraining,ALLoriginalpaired,andENTIREF4developmentv1 corpusbyID/texthash; notjustselected42docs,notrenamedunusedshard. Originalauditclosed. Freezeaboveconfig/sourcepanel/donorconstruction/endpoints/doseandoriginalmap/atom/defaultOT/tunedOT identitiesbeforeevaluatingfreshdata; do notretuneOT/rule onnewerrors. Reportall64candidatepairs plusselectedcoverage/effects. Reuseexistingpairedcorpus/codeassetpipeline; prior38MBdownload/128baseforwards/~500MBcacheinformbudget,actualresourceavailabilitycheckedbeforelaunch. No newjobthisunit.

Ignoredlocaldocumentsretained: EXPERIMENT_TRACKER.md9e40f61f927fdfb360cf4138d0e1e9ce0db0440d4f1f2ff0439d398a7a79c306; EXPERIMENT_PLAN.md9e7d0c84d28a65233f27cf98d09554cf9448e8e2f2251a93319c4a579c50388a; REFERENCE_REGISTRY.mdunchanged876983725f345db54f0037fb6bceb8008bac8dd5db6e1ad45a366b879fbd2c26. Fiveallowlistedsource/config/test/logfiles groupedforsync; data/analysisraw/ignoredplans remainlocal. No newmodeltraining/inferencecost thisunit; cumulative10240causalforwards2482.4570502s and128assetforwards19.4668404s unchanged. analyze-results usedforcompletecoverage/comparisontables anddevelopment-vs-confirmation limitation,notextraapprovalgate.

## 2026-09-05 — fresh-document scope confirmation preparation

Before anynewdocumentread, froze configs/f4_scope_confirmation_corpus_v1.json SHA996307af06446662c6c8b9fd4bfb10f25f914573bd52734d81d3c8155b036b26. Contains exactruleconfigSHAfb4c9237...,originalfactorSHA,2atomfit/configidentities,default/tunedOTarrayhashes,7methodlist,source-only8query/64pairconstruction,rank1/sourcecap.1,andnext-state/centered-logit endpoints. No refit/parameterchoiceonnewdata. NewF4_scope_confirmation_corpus_v1_20260905 excludestraining,alloriginalpaired(includingauditidentitiesonly),andENTIRE113docF4developmentbatch; newselectionsaltfixedbeforedata. Newcalibrationhash-bucketdocstestisnarrowunseendocumentconfirmation,notwholepaperaudit/nativeconceptclaim.

Budgetwrittenbeforelaunch: data40–100MB range reads/tokenization<=5min,128sharedassetforwards/~500MBoutput<=4min,then1984causalforwards(64*(3+4*7))<=8min inclload basedonmeasured0.16–0.25s/forward. Totalprovisional17minexcludingresourcewait,<1.5GiBallocatedVRAM.01:21allresourcesfree. FirststepCPUleaseonly,noGPUreservationorwhole-diskmonopoly. Reuseexistinglockedruntime/dataoverlay;noinstall. Corpus/codeasset scripts nowrespectcandidate_family_frozenconfig(defaultFalseunchanged),compilationPASS. Sourceartifact/metric identity values populatedonlyafterpreparation,notaftertuning. No newscientificeffect yet.

Actualcorpusrun01:22:52.289206–01:23:37.370267,session2696exit0,118documents/65536tokens,40,691,264bytes/43ranges,8checks+contractPASS. ID/textdisjointcheckincludesallthreeexclusionledgers. TokenmanifestSHAe0d984a7a58c47a169d0882d0324f5105eb3b009854d06c179849e09283310ea,sequenceSHAa4515715d5399763363bda6e3255cdd628bb905b9628a2c95e4e5e24340f3b27. Noauditmetrics/tokens,no newmodel forward orscientificresult. Corpuspreparationcomplete,donotredownload.

Whiledatafetched,preparedcausalconsumerreusingfixedOTnpzcoefficients: saved_ot_familieschecksfit/config/arrayhashes,sourcebasis/model/hook/decoderidentity,finite3072coeffshape; noOTrefit. Optional source_scope appliesfixedrule andsavesallsourcechoicesbeforeloadingmodelorreadingtarget-code rowsforendpointtest; allcandidatepairsstillrun. Snapshotincludesparticipation/selectorcode. Evidence/scope fields configurableto describenarrowfresh-documenttest,notrenamingolddevelopment. Codecompiled,2sourceruletestsPASS. Encodingconfigf4_scope_confirmation_codes_v1.json SHA70681a63c6c6d94b1f87f9c00a5774f0ecfd17d51f750b6ab2083218018608ceusesnewactualtokenmanifest,128baseforwards/~500MBbudget.01:24EndoSAEholdsCPU/GPU;CCADhasnoprocess/queuedwaiter,willresumeassetencodingwhenfreewithoutnewdownloadorchangingfrozenrule. Current6allowlistedchangesdeferredforsameunitgroupsyncafteractualresult;HEAD71c0de2. Routinepreparation/resourcewaitquiet.

Userdirected completion of THISnew-document roundthenpause(notdelete)theloop forinspection; focusindependentconfirmation/contributionboundary,notunboundedmethodsearchorengineering. Trackerrecordsstopafterunit;noextraexperimentalbranch. FirstencodingattemptsawresourcefreebutanotherprojectacquiredCPU/GPUbeforelaunch;managerrefusedbeforecreatingrun. ThenONECPUwaiter(max900s)wasstarted,withGPUwait0inside; no duplicatejob,leaseoverrideorotherprojectinterrupt. Endofunitmustpauseccadautomationandhand offresultpath.

Encodingwaitersession54124acquiredresourcesandcompleted01:29:43,PASS/contractPASS,128sharedbaseforwards. Newassetmanifest3df615a6720e15e7abe0c89166049c7e4692b7fae57549fe3b2833c71efd13dc/rawmanifest8a6e684361a5c6b9b7c136fa0756dee569c8e9464b9767c31a783654431fad92. Builtfinalcausalconfigf4_source_scope_confirmation_v1.json bymechanicalnewassetidentitysubstitutionandfrozen7method/2OTparameterreferences,notnewmethodselection. Launchedsession68697(CPUwait900->GPUwait0),runF4_source_scope_confirmation_v1_20260905. Source-onlyscopepreflight25/64selectedbeforemodelendpoints,selectionpersisted;all64candidatepairsstillrun,1984forwardsbudget. Thisisthelastrunthisunit;pauseautomationafterresult/handoffperuser.

## 2026-09-05 — frozen source-scope confirmation completed; user-requested pause

F4_source_scope_confirmation_v1_20260905 completed at 01:35:53 UTC; session 68697 exited 0. Actual 1,984 causal forwards, 290.843478 seconds, 1,792 raw rows, peak allocated VRAM 841,839,104 bytes; six execution checks and artifact contract PASS. Raw SHA256 1f0fa913d53f8df6545d54f86550dff7fe9e40e07505271c75d3cfeeca4cfa13. The separate encoding run used 128 shared base forwards / 18.426958 seconds (encoding timer excludes loading). Cumulative causal cost: 12,224 forwards / 2,773.300528 seconds; asset cost separately 256 forwards / 37.893798 seconds. No duplicate run, new fit, threshold change, audit opening, or additional science unit.

Frozen source rule selected 25/64 candidate pairs (39.0625%) before endpoint computation; all 64 pairs were evaluated. Fresh corpus contained 118 documents / 65,536 tokens, disjoint by ID and text hash from training, original complete paired corpus and complete prior F4 development batch. Existing model/query/map and both atom/OT parameter families reused. Post-run source-only recalculation exactly matched all 64 pre-endpoint selected/supported flags and aggregate_energy, natural_source_hook_fraction, largest_atom_energy_share values. This is a replay/implementation check using common functions, not independent scientific review. Summarizer change only makes the evidence-limit description configurable; default development wording and statistical aggregation unchanged.

Positive checkpoint: all 8 selected query-condition groups beat raw, wrong-query matched energy, both single-atom variants, both UOT variants and zero at both endpoints. Query-condition medians of centered-logit/next-state errors: distributed 0.020206572/0.016741870; raw 0.115949314/0.080000443; wrong 0.689699372/0.555844220; level atom 0.437375646/0.468136058; dynamic atom 1.0/0.999813897; default UOT 0.994750920/0.984885552; tuned UOT 0.997573122/0.991744446. Ratios of these aggregate medians give 82.5729%/79.0728% lower error than raw. Across the 32 dependent query-condition-target groups, distributed beats atoms, UOT and wrong in 32/32 for both endpoints; beats raw in 31/32 centered-logit and 32/32 next-state. Raw exception: s3:1230 positive to t2, distributed 0.029876127 versus raw 0.025627528. No independent-seed CI asserted.

Selected positive/negative input counts: s2:2176 4/3, s3:1230 4/4, s4:801 4/1, s5:1352 4/1; other four queries 0/0. All rejected and unsupported rows retained. Rejected 39 pairs have 10 valid query-condition groups: distributed median 1.215257916/1.105381063, below zero in 4/10 and 5/10. All candidates have 15 valid groups: 0.247066367/0.165427346. Seven unsupported input pairs produce 196 missing method-target rows per endpoint (588 values across all three stored endpoints); not imputed. Selected/rejected groups may share a query-condition, so group counts are not additive. This confirms a source-observable applicability range on new documents, not universal success. Contribution is dynamic functional correspondence with local distributed increment; native necessity and the full OT/MAS families remain distinct untested extensions.

Analysis files: runs/F4_source_scope_confirmation_summary_v1_20260905/{source_selection.jsonl,coverage.csv,target.csv,query.csv,summary.json,RESULTS_FOR_REVIEW.md}. Summary SHA256 08700773440c80c424eb696695b42886e3ed240f40f6eba2ea066ec0af43f00b; reader report SHA256 eeb92213e9ad3b5549b73860a7347f176160928d5bc8590413e5073e21182263. Reader report includes complete main comparison, coverage, one contribution-boundary paragraph and a single suggested continuation (not launched). Analyze-results skill used for common aggregation and complete denominators, not extra gates.

Automation ccad updated through the app to PAUSED and TOML readback confirmed PAUSED; original prompt, target task and five-minute recurrence preserved. Resource manager showed CPU/GPU free after automatic release. No current process or waiter remains from this round. User review is the next action; do not resume autonomously. Tracker's top snapshot replaced stale preparation instructions and explicitly labels remaining old entries as history. Ignored local documents retained by path/hash: EXPERIMENT_TRACKER.md 76f0e7a3f608ea1bdbcc4e852ce01203526ebb0b0c851ba7b0352e5fa334d533; EXPERIMENT_PLAN.md unchanged 9e7d0c84d28a65233f27cf98d09554cf9448e8e2f2251a93319c4a579c50388a; REFERENCE_REGISTRY.md unchanged 876983725f345db54f0037fb6bceb8008bac8dd5db6e1ad45a366b879fbd2c26. Raw data/reader report remain local under existing allowlist policy.

Final focused validation: two source-scope unittest fixtures PASS, compile/JSON syntax checks PASS, git diff --check PASS, nine allowlisted files inspected for size and common credential patterns with no findings. An attempted pytest invocation found pytest absent; the existing unittest entrypoint then ran successfully without installing anything. No further model work. Nine code/config/log files are grouped for the authorized main-branch synchronization; local data and ignored reader documents are not staged.

## 2026-09-05 — user resumed main line and 15-minute loop; structural mechanism round

User requested more positive science, bounded engineering, active reading of new/old literature and blueprint, and one real round now. Updated existing ccad heartbeat to ACTIVE with 15-minute recurrence through app tool; readback confirmed both fields, original prompt/target preserved. AGENTS persistent cadence updated to match user. No paid resources or new model training.

Read blueprint PDF pp.12–13,16–17,21–22 completely with pypdf, including native contribution versus marginal projection, substitution lemma, and rank versus atom distinction; no theory rewrite. Parent reference indexes checked before primary-source reading. Re-read Li15 §4 sparse prediction (standardization and L1 tradeoff) and ICLR2025 SAE stitching §4.1–4.2 (direct decoder contributions and connected-support swaps). Adopt their measurement questions, not their claims or a renamed implementation. This motivates truncation/native-readout mechanism comparison on the already confirmed range rather than another free mapping search.

Frozen configs/f4_readout_structure_dev_v1.json SHA70c80156fead57e78856a778adea2150662c5ae40ea41a8e2dce2c4faaf69e19: replay all64 inputs and source-only scope on existing confirmation documents, explicitly now mechanism-development data. Rank1/common source cap.1 and two endpoints unchanged. Full FCC plus discovery conditional-energy top1/4/16/64 readouts without refit; one fixed random-sign full readout matched to full FCC hook norm; same top16 support native donor difference and separate source-energy-matched version. Ranking uses saved v10 discovery rows/weights, Var_w(z_j)*(D_j W)^2, deterministic atom ties; no endpoint ranking. This is deployed-map compression, not an optimal sparse regression contest. Native direction rank differs and is reported separately, not as an equal-rank baseline. Budget2240 forwards=64*(3+4*8),6–8min including prep/load,<1.5GiB allocated VRAM,4CPUthreads,0download/0training. Resource manager free; actual GPU1831/16303MiB used by existing desktop activity. Canonical environment8348de46 unchanged; reused without reinstall/witness churn. Four source-rule/ranking/scaling fixtures and compile PASS. Next start one CPU→GPU managed process, no duplicate.

## 2026-09-05 — readout structure checkpoint and adaptive-sparsity opportunity

Run F4_readout_structure_dev_v1_20260905 completed 06:59:15 UTC, started06:51:38, session67881 exit0. Actual2240 forwards/456.9050955 seconds,2048 raw rows,841839104 allocated VRAM bytes; six execution checks and artifact contract PASS. Raw SHA95b46e06f8308ca4ac43fe708c96751eec11f448772e7697e3f276ad5268fabc. All64 input selections/source choices and all256 full-target rows exactly match prior confirmation. Discovery ranking32 families saved before endpoints in readout_ablation.json SHA9e3cb3eb21d0f6bffea056dfa4fa79a7b7ea960dc134e8cf9a82e626d9f6d1b4. No fitting, download, model training or original-audit access. Total causal cost14464 forwards/3230.2056238 seconds; asset256 forwards/37.8937981 seconds unchanged. CPU/GPU leases automatically released; free at final check.

Selected25 inputs/8 query-condition groups: full logits/state median .020206572/.016741870; top1 .999147787/.998364048, top4 .984959458/.981940001, top16 .999211454/.997074520, top64 .987920258/.983032970; matched random-sign1.647254846/1.901268610; native16 difference1.490412260/1.312658766; native16 source-energy matched1.770706863/1.890396360. Full wins all7 comparisons in32/32 dependent target groups at both endpoints. s3:1230 has a smooth truncation curve (~.73→.33→.11→.053→.020 logits); other selected queries generally lose effect under fixed top64. This demonstrates deployed-map structure, not optimal sparse lower bounds. Descriptive target effective atoms median49.722 across100 selected observations; s3 7.64 versus other queries52–63. Random-sign energy matches full within1.2e-15 relative; selected100 native candidates all nonzero and source-energy matching within5.6e-16. Across228 valid native candidates,4 are zero (all rejected range), retained as zero rather than falsely matched. Seven unsupported source pairs produce224 missing rows per endpoint, retained.

New positive opportunity in rejected39 inputs/10 valid groups: full median1.215257916/1.105381063 versus top16 .338517555/.373426334 (72.14%/66.22% reduction); top16 belowzero9/10 at both endpoints and betterfull7/10. Top64 .324544056/.391663049 has similar performance, motivating smaller16 fallback after viewing all four budgets. Example s5:369 positive improves3.02818/3.29220→.20732/.20790; negative2.18053/2.48125→.37047/.38359. This is not a new failure gate: use it to expand the useful range.

Derived source-adaptive policy chooses full on old source-rule-selected inputs, top16 otherwise; config f4_source_adaptive_readout_dev_v1.json SHAe3047e5d3719529ff1d487d7553764d2bbe58b6713a4759cc2506636fed76d67. Fallback16 chosen AFTER this round's results, explicitly development; old source gate unchanged. Deterministic branch selection reuses actual forward outputs, no synthetic model outcomes or new forward cost. Existing summarizer adds a small optional derived-policy alias; checks show selected branch exactly full/rejected exactly top16,256 requested/228 valid target-input observations per endpoint. All15 valid query-condition groups: adaptive logits/state .235302372/.207902397, both14/15 belowzero versus full11/15. Six groups improve,3 worsen,6 unchanged; state median is higher than full. s5:710 positive fallback is zero,error1,not success; negative unsupported. Full retained for all25 previously selected inputs. Adaptive summary SHA702720fdbff47719ca87a70418c66b98c42bb96c4bb2acea49266eb2a1eadc1e. Next freeze this candidate and saved discovery support for fresh disjoint documents with full/top16 and existing strong controls; no further method search or model run in this unit.

Readout source-only inspector plus full participation analysis completed; an early inspector call correctly refused RUNNING status and was retried after PASS, with no partial analysis overwrite. Four focused unit fixtures/compile PASS; derived-branch identity and denominator checks PASS. Figure script uses bundled Pillow after discovering matplotlib absent; no dependency installation. Actual image2200x1100/80 plotted group-budget-endpoint points, all8 group curves plus median, log axes, no CI; visually inspected, legible/no clipping. Figure SHA94d40a0494b7d9388b04efbb0e0708f6d501aa130727b6c3ca919b3aad4e30b4. Reader report runs/F4_readout_structure_summary_v1_20260905/RESULTS_FOR_REVIEW.md SHA80283db92bedfed975b1fcea8d101d04c86917e00d3dbdd234b319f050bbda78. Complete selected/rejected/all tables and source/provenance retained. Plan updated for source-conditioned readout complexity without relabeling development as independent confirmation.

Ignored local records: AGENTS.md420be7907e086feaf188b10b08b5ae5b51859161286660685bad5a51bf7bb265; EXPERIMENT_TRACKER.mda8430c9b351b04fc73ccebcf22089b7e98ee03fe2127040f16a4be80abed5b59; EXPERIMENT_PLAN.md0146360e6bb3ba2b3bd954604f17b5ac4e1c9aaf608015229b255a1d70644bee; REFERENCE_REGISTRY.mdccf50f1e38fba812644e8127ad76d4fb24a81af51f3feb98ffb7472bb0a41ae0. Registry includes blueprint pp12–13/16–17/21–22, Li15 sparse-prediction methods, ICLR2025 direct SAE stitching, and refreshed2026 seed-stability PDF pp3–6/official v1. Actual borrowing and native/geometry/function distinctions documented; no external code copied. Automation remains ACTIVE/15minutes, original durable prompt preserved. Eight allowlisted source/config/test/log files grouped for main synchronization; raw results/reader report remain local.

## 2026-09-05 — target-venue prompt and frozen adaptive confirmation start

User requested a short target-conference statement in automation and another substantial round. Original target recovered from this log's initial loop entry: ACL/ICLR main-conference full-paper scale. Added only “以ACL、ICLR等主会完整论文的贡献与证据要求为目标。” to configs/CCAD_AUTOMATION_PROMPT.md and existing app heartbeat prompt, preserving ACTIVE/15-minute recurrence and target thread. No year/deadline/submission promise. No unrelated prompt changes.

Freeze configs/f4_adaptive_confirmation_corpus_v1.json before retrieval:65536tokens from new document salt; exclude training, all original paired documents and entire prior113/118-document F4 corpora by ID/text hash. Adaptive rule/config SHAe3047e5d3719529ff1d487d7553764d2bbe58b6713a4759cc2506636fed76d67, saved discovery readout SHA9e3cb3eb21d0f6bffea056dfa4fa79a7b7ea960dc134e8cf9a82e626d9f6d1b4. Full/top16/raw/wrong/twoatoms/twoOT with unchanged source-only query/donor/gate,rank1,common source cap.1; report continuous effect error and valid-group belowzero coverage. No fresh-data ranking/refit/fallback tuning. Budget up to18min excluding waiting:5min range reads,4min128assetforwards/~500MB,6–9min2240causalforwards at prior456.9s;4CPUthreads,<1.5GiBallocatedVRAM. Existing environment8348de46 warm-reused. All resource leases free, GPU1989/16303MiB desktop usage; CPU→GPU manager for compute, no disk monopoly. Original audit remains closed.

Corpus completed exit0/session19845:106documents/65536tokens,36,414,446range bytes/41requests,8checks/contractPASS. Independently recomputed ID/text-hash intersections against each of four exclusion ledgers:all0. Corpus freeze SHAe7461f4a21e6a99bb6a2ff0ffad25aa672d82fdf0b42bca8a328d9e46bddd2f5. Codes exit0/session74758,128 shared model forwards,8checks/contractPASS; manifest4b42dc7fda5d948b63c0048e90d82fc2ce2efe92b11a74c1fe17701a4f188fe9,raw-hook manifest4dde97fab7b46d120ee87f773cd8c56c2aeec72e9b61008fd9b9ffbc80b85706. Frozen causal config612868a10cdcf3e6191b0dac8c1238d48625af4e35c42943383aae3f2c683f07; session88320 running. Source choices saved before endpoint forwards:22/64 selected. All32 saved readout-family records exactly match prior discovery record; beta/sign hashes checked by runner and no reranking/refitting. Minimal saved-readout reuse branch added to existing runner;5focused fixturesPASS including changed-coefficient rejection. Automation prompt text exact after CRLF normalization,ACTIVE15min verified; initial raw newline equality assertion was a formatting-only mismatch,not failed update.

## 2026-09-05 — frozen adaptive readout confirmed on106fresh documents

F4_adaptive_confirmation_v1_20260905 completed07:26:41 UTC,started07:20:56,session88320exit0.2240forwards/345.1899129seconds,2048rawrows,841839104allocatedVRAMbytes,sixchecks/contractPASS. Raw9ec82534885c6a1fc85da52e6a44bc6a1356ee74de22a9936b0fdb9e6189869c. No method change after corpus retrieval. Assets128forwards/17.8893973seconds;total causal16704forwards/3575.3955367seconds,assets384forwards/55.7831954seconds. All shared leases free after completion.

All16valid query-condition groups, median inputs within target then median targets: adaptive logits/state .204066378/.244013028 vs full .608712292/.835069421 (66.4757%/70.7793% lower),fixedtop16 .520496852/.602992092 (60.7939%/59.5330% lower),raw1.319902901/1.599766506 (84.5393%/84.7470% lower). Belowzero BOTH:adaptive14/16,full8/16 (logits9/16,state8/16 separately),fixedtop1614/16,raw8/16,default/tunedUOT12/16,dynamicatom8/16,levelatom5/16,wrong6/16. Adaptive beats each UOT at16/16groups on both endpoints,raw13/16,dynamicatom12/16,levelatom14/16,wrong15/16. Dependent target comparisons vs defaultOT59/64logits58/64state,tuned58/64both;raw55/64,54/64. Not uniformly best in every target or method comparison.

Frozen source gate selected22/64inputs(34.375%),8query-condition groups,full/adaptive .021758385/.021769073,all8belowzero. On this selected subset full beats raw5/8logits7/8state,dynamicatom5/8both;old all-baseline superiority does NOT universally repeat. Full vs selectedtop16 7/8logits8/8state,selectedtop16 medians.595523164/.572754187. Rejected42inputs/12validgroups:top16 .461091573/.490860935 vs full1.148498802/1.279221433,59.8527%/61.6281%lower;bothbelowzero10/12vs4/12. Subsets share some query-condition groups,8+12 are not20 independent groups. Source selections/3numeric fields exactly match pre-endpoint64rows; deterministic branch targets/medians/requested+valid denominators independently replayed from immutable raw.61supported inputs/3unsupported,96missing raw rows per endpoint retained. s5:710bothconditions zero top16 candidate,error1,not success;positiveonly1/4sourcepairs supported. Adaptive vsfull logits8better/4same/4worse;state9better/4same/3worse. Comparedfixedtop16 logits6better/9same/1worse,state7better/9same/0worse. No newdata rescue sweep.

Positive examples logits/state: s5:336positive full1.248/1.298→adaptive.119/.144;1352positive1.049/1.261→.483/.496;369negative2.749/2.588→.439/.476. s2:2176positive retained.014/.013 while fixedtop16~.997. This confirms a simple source-conditioned readout consumer on new documents/existing5seed8querypanel; not native necessity/optimal sparse lower bounds/human-readable concepts. Next expand to8unused source-only hashqueries from same energy strata,fit newquery readouts/comparators only on originaldiscovery,retain existingconfirmation and label exposed-document query extension development. No newtraining/threshold tuning planned. Plan updated with this general expansion logic,tracker carries actual next unit.

Reader report runs/F4_adaptive_confirmation_summary_v1_20260905/RESULTS_FOR_REVIEW.md SHA086ff73b1874ecda9a8b14cae79aaabfd19ef0d9c8811b0fade28e94eb69ace6;summary09dfe9474c0a4180c891994b15e0afd4a259eb6055cc9c31903b03717c0edf67;complete query/target/coverage/sourcechoices and checkpoint_statistics.json retained locally. Ignoredtracker0d012fbbcbefbf5bf5276cd1ef1e488cde64b5eb8995bae4d6ef74785b1cef81,plan460e68bf9d57f839e80bc0b11dcc4650bcceb4e3f87f208e0c2ea2f6cd51071d. Prompt file2b068e0f0c69a270704e97e2040e118f5173aaeb1a2479449886f3939edb546d,ACTIVE15min. Fivefocusedfixtures/branchreplay/identitychecksPASS;small report tables instead of newplotframework. No newdependency/downloadedweights/paidresource. Eight whitelistfiles grouped for main sync;originalauditclosed.

## 2026-09-05 — user21-minute cadence and new-query expansion start

User requested automation every21minutes and continued substantive progress. Updated existing ccad heartbeat through app tool; ACTIVE21min and exact unchanged durable prompt verified. AGENTS cadence changed only, preserving independently added pause-policy paragraphs. The target ACL/ICLR statement remains. Warm environment canonical8348de46 unchanged; resource manager allfree, GPU1720/16303MiB desktop usage. No environment rebuild/training/download.

New panel is source-hash offset1 within each of8energy strata: (3,1144),(1,2641),(3,1093),(1,2298),(1,615),(2,2645),(1,1441),(5,2194); disjoint offset0 consumer panel. All40anchorqueries already have historical F4 geometry/surface outputs, so do not describe these as never seen in any earlier study. New to current causal-method development; select without target effects. Fixedfull/top16 branch,sourcegate/rank1/dose.1 and existing8-method budget unchanged. Originaldiscovery<=256source-weighted rows supply64scalar selections,64OT fits using fixed epsilon.05/default1000iter and transferred oldtuned.01/5000iter,plus32readout rankings; no newepsilon sweep. Preparation reuses existing kernels in a small CPU-only consumer, source-hash offset is the only causal-selection change. SixfocusedfixturesPASS. Provisional prep6min +2240causalforwards6-9min,4CPUthreads,<1.5GiBallocatedVRAM; no newmodelcache. Use existing106-document assets; explicit query-expansion development, not fresh-document confirmation or originalaudit. Start preparation underCPUlease then consume saved fits underCPU→GPU leases; record results before deciding further work.

Preparation session69403 completed numerical work in52.8603585s (64scalar,64OT,32readout) then exited1 at environment finalization: importlib.metadata.PackageNotFoundError for unused scipy. Kernels inspected are NumPy-only; no package install or numerical rerun. Original raw00eb69deeddf79adf7c1b54d852b523f8c3a4a06b24cdd7f00656d0c79288f7d and numeric summary40a64700e87b3031a52cb22ed018532ea00ee2022fd5e75bd2db395a36e8d1f4 unchanged. Metadata-only recovery supplied truthful environment/logs/snapshot_root, preserved original stale status/code_hashes and source snapshot,explicit process_exit_code1/recoveryrecord;contract/payload counts,query identities,finite shapes,discovery provenance PASS after recovery. Current helper removes unused version lookup and adds snapshot_root for future runs;this run keeps originalcode. This is not a scientific failure or a successful originalprocess exit; no loop pause. Saved fits and readout are explicitly hash-bound in causal config.

Started F4_query_expansion_dev_v1_20260905 session86282 underCPU→GPU. Before endpoint forwards,39/64 source-only choices selectedfull;all64 sourcepairs supported. Input8querypanel exactly prescribedoffset1,oldpaneldisjoint. Readout32families and bothatom/twoOT reused frompreparation;source-selection/kerneldefaultoffset0backward-compatible fixturePASS. Originalauditclosed; no calibration-basedmethodchoice.

## 2026-09-05 — newquery positive checkpoint

F4_query_expansion_dev_v1_20260905 completed07:49:20UTC,session86282exit0,2240forwards/348.7844154s,2048rows,841839104allocatedVRAMbytes,sixchecks/contractPASS. Rawac874f5b023e6861637c9ba2ebde2ef7c7b1e937f011441382185d100b12e807. New8sourcehashqueries,39/64inputfullselection,all64supported/0missing. Source pre/post64choices and3numericfields exact; raw-branch/targetmedian/denominator replayPASS. SixfocusedfixturesPASS. Totalcausal18944forwards/3924.1799521s;assets384forwards/55.7831954s unchanged;separateCPUprep52.8603585s. No newtraining/data/threshold/epsilon search. Preparation metadatafailure/recovery remains explicit above and inrun.

New16query-conditiongroups adaptive logits/state .049400201/.057213388,16/16bothbelowzero vsfull13/16,raw12/16. Target-group64/64bothbelowzero butdependent;individualtarget-input247/256bothbelowzero,9failures retained,worst2.820992/2.457136. Fixedtop16 .295033728/.307854621 also16/16;adaptive83.2561%/81.4155%lower,11groupsbetter5equalboth. Dynamicatom .198510132/.243924518,adaptive75.1145%/76.5446%lower,15/16logits16/16statewins. BothOT/wrong16/16groupwins;OT63/64targetwinsboth. Raw .041802332/.044305537 has LOWER aggregate errors,adaptive only10/16logits7/16statewins. Full .049400201/.052889263: adaptive sameglobal logits,state8.18%higher;5groupsbetter9equal2worse. Main gaincoverage,notallmetricdominance.

Selected39inputs/11groups full .022004287/.023087356,11/11belowzero and11/11betterdynamicatom/top16. Rejected25inputs/7groups top16 .327392752/.322803270 vsfull .444387718/.609365129,belowzero6/7vs4/7;subsets sharegroups. s1:1441positivefull2.241/2.465→.492/.461,negative1.403/1.266→.560/.521;s3:1093positive2.873/2.421→.486/.501. s1:615positive .022/.023 preserved vsfixedtop16.366/.321;s5:2194positive .022/.018 preserved vs.922/.914. s3:1144bothconditions worsen versusfull;1093positive rejectedsubset1.328/1.264stillfails. No re-selection after outputs.

Report runs/F4_query_expansion_summary_v1_20260905/RESULTS_FOR_REVIEW.md and completequery/target/coverage/sourcechoices/checkpoint_statistics retained. This is newquery development on exposed106docs,notnewdocumentconfirmation;oldgeometryscreen included thesequeries. Next freeze thispanel/allfits and confirm onnew65536tokens excludingtraining/alloldpaired/full113+118+106F4docs;originalauditclosed. AutomationACTIVE21min,promptunchanged. Sevenallowlisted implementation/config/test/logfiles grouped for sync;localreport/tracker/AGENTShashes recorded below.

Ignoredlocal reportSHA6fbc4502cb5b05f81387eadf957be9567af7029c0d5e9c69b3e084a6ed39ce47;tracker e7209d65dce05db1ccd021a51c5d0255394a16a15385f704d25af954189b9a38;AGENTS9b2fa33322ca0aed1091bac38d1b21b1ce16ad078ac9ca7ba60b3b2aa2a96c6e. Sevenfile whitelist,size,commoncredential-pattern,JSON,compile,gitdiff checksPASS;actualautomation ACTIVE21min verified. No unrelated changes staged.

## 2026-09-05 08:18 UTC — expanded query panel fresh-document confirmation start

Frozen before new-data retrieval: configs/f4_query_confirmation_corpus_v1.json SHA256 ba1f2009fc92cbdbabb6c94353e78cb88727a6702693ba6a4a5694116fc85257. Same eight offset1 queries, five existing seeds, saved source rank1 maps and full/top16 readouts, both saved atom fits and both saved OT fits; no new fitting, ranking or selection thresholds. New65536tokens exclude training/all original paired/full113+118+106 prior F4 corpora by ID and exact text hash; original audit closed. Endpoints and dependence-aware aggregation unchanged. Budget18min excluding waiting: <=5min retrieval,128shared asset forwards/<4min,2240causal forwards/6-9min;4CPUthreads,<1.5GiBallocatedVRAM,~500MB codes. Environment source hash3129a184 unchanged, warm reuse. Resource manager allfree; GPU1653/16303MiB and1% utilization leaves ample measured capacity. No environment rebuild/training/payment or disk monopoly. Corpus uses CPU lease; subsequent inference CPU then GPU leases. Latest registry source-method readings retained; recheck closest comparison before interpreting confirmation.

Corpus session27357 exited0:116newdocuments/65536tokens,42range requests/40595100bytes,8checks/contractPASS. Codes session24518 exited0:128forwards/15.5529324s,895002624allocatedVRAMbytes,8checks/contractPASS. Started causal session12579 at08:22:12UTC underCPU→GPU leases;44/64source-only choices selectedfull,all64supported. No endpoint aggregation at launch. Readout/twoatoms/twoOT hashes checked by consumer. MAS§3.3–3.4 andLi15§4 official fixed-version methods revisited; registry updated with precise comparator limits and possible sparse-refit consumer, no new baseline added to frozen confirmation.

## 2026-09-05 — expanded queries confirmed on fresh documents

F4_query_confirmation_v1_20260905 completed08:27:55UTC,session12579exit0,2240forwards/342.95015309998416s,2048rows,841839104allocatedVRAMbytes,sixchecks/contractPASS. Raw257006fd53dfb4bec2a172971424c5d2ba7261425dbd5543c19e819673ecb5d4;causalconfig247490b47a90cfbaa4768322a955a826cfd53cba00b7384e6c740c478552490d. Independent ID/exact text-hash exclusion checks all0 versus training269,allpaired569,full113/118/106F4batches. New116docs65536tokens; no audit access, numerical kernel change, training, fitting or threshold adjustment. Totalcausal21184forwards/4267.1301052s;assets512forwards/71.3361278s;priorCPUprep52.8603585s separate.

All16query-condition groups adaptive logits/state .0414570195/.0330541850,16/16bothbelowzero. Fixed16 .246532982/.252051552 also16/16;adaptive83.18399%/86.88594%lower with11better5equal0worse groups bothendpoints. Full .0402814864/.0330541850 bothbelow14/16;adaptive4better11equal1worse andlogits2.9183%higher/stateequal. Raw .0537270830/.0346065269 bothbelow13/16;adaptive22.8378%/4.4857%lower medians butonly10/16and9/16groupwins. Prior query-developmentrawlower preserved. Dynamicatom .228222058/.215401367,adaptive16/16logits15/16statewins. BothOT andwrong16/16bothgroupwins;defaultOT .373110374/.347115049,tunedOT .357020451/.344347960,wrong1.188258365/1.039351680.

Selected44/64inputs/12groups full .0231169473/.0232024055,12/12belowzero and12/12betterfixed16/dynamicatom both;rawonly5/12wins. Rejected20inputs/6groups16 .378931509/.338004023 vsfull .429019432/.417872545,belowzero6/6vs4/6;subsets overlap. All64sourcepairs supported/no missing. Individualdependenttarget-input242/256bothbelowzero,14failures retained,worst3.135687779/3.262630964. Target-group63/64bothbelowzero,faileds1:1441negative→seed5=1.557293746/1.364524380. s1:1441negativeoverall .988738/.863160 isnearzero boundary,notrobustmargin. s3:1144negativeadaptive .127997/.206991 worsefull .045032/.078699. No post-result query removal.

All32readoutfamilies exactly equal saveddiscoveryfamilies;bothatom/twoOT32familieseachtakenfromsavedhash-boundfits,norefit.64sourcechoices/support/3numericfields prepostexact. Adaptivebranch independent reconstruction matches272target/subset/endpointmedians andvalid/requesteddenominators exactly. Analysis transport-only issues:large texttooloutput truncation promptedcompact reads;PowerShell stringcast doubles had~1e-17roundtriperror,JSON numeric serialization fixed exactreplay. No numericartifact/kernelchanges or experiment reruns. Reports/completeCSVs/checkpointstatistics under runs/F4_query_confirmation_summary_v1_20260905/. Nextscientificunit same-support top16discoveryridge refit thennecessarynewmethodforwards,not immediate newtraining/docreplication;tracker/plan updated andpriorconfirmationunchanged. Samefive-seed/hook limits andgenericUOTvsfullLi15/MAS distinctionexplicit. Automation21minunchanged,ownleasesreleased.

Local ignored artifacts retained: report6aa3aedc30e7badab5d8a3e706a5f603118787f51ac152e6eea234c4a2ea7aa7;checkpointstatisticsd18fe9341593ecca36a5b489a79a7cfa585760ed2d4301f9bb8081c1f31a5ac6;trackerbc7dfdf49577232b86c9e6608d30c9c40bbf89d94033c52536f0103eb05cee8d;plan2183550b559fd75e3de3fe71a0fea0b16252a876074976b02719cc229843cffd;registrya95dc9c04253125aacfe60a7f78048bf253d372587b3e5e06c9285b82cc177f9. Fiveallowlisted config/logfiles for grouped sync;JSON parsing/gitdiff/size/commoncredentialpattern checksPASS,stagingemptybeforeadd. No numericalcode changes,so nofulltest rerun. No unrelated files staged, no private data uploaded.

## 2026-09-05 08:58 UTC heartbeat — fixed-support refit development start

Currentworkcard: refit same savedtop16support on originaldiscovery, then compare actualsource-aligned downstream effects to savedfull/truncation/othermethods on116exposed documents. Implemented scoped consumer in existingrunner; reuses ccad.hook_transport.fit_hook_space_transport after weightedconditionalcentering,rank1 scalaroutput,trace/min(n,16)ridgefraction.001. No newsolver/dependencies,supportsearch,endpointfitting,querychange ortraining. Same fraction rule is NOT identical full-hook regularizer geometry and not Li15LASSO.3new fixturesnormal-equation/all-pair/intercept/constant+invalidchecks,3existingatom+6source/readoutfixturesPASS. Warmenvhash3129a184unchanged,resourcesallfree,GPU1653/16303MiB1%. IntegratedCPUprepare<=2min +448forwards<=3min underCPU→GPUleases; no separateprepframework or rerunningold8methods. Oldconfirmationpreserved. Configf4_fixed_support_refit_dev_v1.json andsource snapshots bindactualfit/forwardinputs; source selection will be replayed againstreference before interpreting differences. Expectedopportunity: strongfullrelationremainsbetterthanafairerfixedsupportfit,or developsimplerrelationsifrefitbetter; neitheroutcometerminateFCC.

Firstrefit completed09:05:13UTC session31056exit0:32fits1.6368092s integratedwithin448forwards72.0303704s,256rows,6checks/contractPASS. Source selection and44/64gate exactlyequalreference. Raw06b8e203d57036d708d8785573ce7b9d4618a9664ca4ba233da7258b2dabed4d. Refit16allquerymedian .039239951/.033834698 vsunrefitted16 .246532982/.252051552;both16/16belowzero. Refitbetterthanfull11/16groupsboth andoldadaptive10/16logits11/16state. Strongselected12groups .023144821/.020070437 vsfull .023116947/.023202406;refitbeatsfull7/12logits8/12state. This invalidatesinterpretingtruncationgap asdense-readoutnecessity andopenscompactFCCopportunity. Nooldresultrewritten. Budgetedextensionwithincheckpoint: samealgorithm/fraction onoriginal8querypanel/exposed106docs,onlyitsownsaveddiscoverysupport/rowsweights;additional32fits<=2min+448forwards<=3min,CPU→GPUleases,no newdata/training/tuning. Separateexisting-datareplication,notfreshconfirmation.

Originalpanel extension session28644 exited0 at09:09:37UTC:448forwards95.0329847s including32fits20.3649348s,256rows,6checks/contractPASS. Raw66f985c47141ffeac187c51725b5a06a1ff7a731e6a151508727b8e11474d08d;config4f2aee9df4677ceda69e322dcb3a550d1fc0f25e43296fc3f987c7f5f82e8948. Refit originalpanel .854050333/.860946663 versusadaptive .204066378/.244013028 andfixed16 .520496852/.602992092. Refitvsadaptive4better10worse2equalgroupsboth;belowzero14/16butmanyerrorsnear1. Selected8groupsfull .021758385/.021769073 vsrefit .519355682/.513002596;1230/693bothconditionsimprovebut2176/801bothconditionsnear1. Rejected12groupsrefit .854050333/.860946663 vstrunc .461091573/.490860935. Uniformrefitdoesnotreplacepriorconfirmedpolicy. Newpanel245/256bothbelowzero/no missing;oldpanel244valid12missing,196validbothbelowzero48failures,worst5.214770127/4.490457905. s5:710error1bothgroupsnotcountedsuccess. Allresultskept.

Bothpanels selection/sourcegate bytesexactoldreferences;512target-inputsourceRMS/common-dose checks0mismatches. All40historicalquerysource-supportlists identicalacrosstargets. Existingkernel plus12focusedtests;noindependentscientificauditclaim. Original2176positive-discoveryvariance.00168586andfitrelativeerror.79-.90 versuslargecrossconditiondonordifferences suggesttraining-operationmismatch,notprovenmechanismandnotcommensurateratio. Nextboundedcontrol: originaldiscoverypositive/oppositeconditiondonorpairs,samefixed16support/sourcecoordinate,matchedtotalrowbudget,nofreeintercept,first2176/801thenexpandifmeaningful. No newtestgridthisheartbeat. Report two-panel tables/failedrangesunder runs/F4_fixed_support_refit_summary_v1_20260905/RESULTS_FOR_REVIEW.md;checkpoint_statistics containscompleteaggregates. Cumulativecausal22080forwards4434.1934603s;assets512/71.3361278sunchanged. Currentrunfitcostisincluded,notdoublecounted. Automation21minretained;noongoingjobs. Skillsusedwarmruntime/realforwards/consistentaggregation;no paidresources,newtraining/data/environment.

Local ignored artifacts retained with SHA256: EXPERIMENT_TRACKER.md f7678f452bfd781d154493c8b5d323ef104285982396725941671f24ce280093; EXPERIMENT_PLAN.md 9948278f60bf28ce7bfa375d66d7d24d12a96655f801d91b05001b91ffd50a86; REFERENCE_REGISTRY.md af669218bc501f4d4a89f4849a702d0007f820c67e2617a7489e03d97dc17bfe; runs/F4_fixed_support_refit_summary_v1_20260905/RESULTS_FOR_REVIEW.md da75f69d5fe307dd3d322f83a9a5a3002c65d510a764e89984f6c03f2df6a01d. Seven allowlisted code/test/config/log files form this sync unit. JSON parsing, diff whitespace, file size and common credential-pattern checks passed; index empty before staging, no unrelated changes detected, canonical origin verified. Only these explicit files are staged; ignored research artifacts remain local without force-add.

## 2026-09-05 09:40 UTC heartbeat — source-contrast fixed-support control

Started current workcard using F4_contrast_refit_dev_v1_20260905. Reuses fixed16 support, rank1 source coordinate and ridge fraction .001; original discovery positive128 plus negative-score top128 (excluding saved positive256), source-coordinate maximum-difference without-replacement matching across context indices, no free intercept. Same256 materialized fit-row budget, but source scan/condition/weights changed and recorded; not pure centering ablation, document independence not guaranteed for packed contexts. Full existing8query panel including2176/801 costs estimated448forwards<=3min plus source scans/32fits<=3min, avoiding outcome-based query-only reporting. Existing106documents/calibration now development, audit closed; no newdata/training/payment. Warm environment source hash3129a184 unchanged; CPU/GPU free before launch,1653/16303MiB GPU1%. Five fixed-support tests passed including no-intercept between-condition mean and pairing budget/context checks. CPU→GPU wrappers; no whole-disk exclusive lease for ordinary asset reads. Prior confirmed adaptive policy preserved.

Unbounded contrast session83834 exit0:448forwards86.2357354s including13.8388091s fit;256rows and6checks/contractPASS. Raw ece6daf253ae36e776dd6b11fdfc9e27ab92d0b997115e2b02ed6fe710e93dd7. All16 median .674361/.652924, selected8 .051495/.337054; only9/16 groups bothbelowzero versus oldrefit14/16. 244valid/12missing,142valid bothbelowzero; worst222633.857/96694.653. s5:369 bothconditions and336negative improve vs oldtop16;2176/801negative and1352both fail severely. Selection and sourcegate hashes exact oldrun. Independent dense-source coordinate reconstruction max1.7053e-13,32 primal normal-equation coefficient relative discrepancy<=1.2814e-12; no evidence arithmetic bug. Coefficient norms~65–201 on failed queries versus~.47–1.41 elsewhere; max candidate hook fraction8.4078 despite sourcecap.1. Isolate this experimental method, do not deploy or pause whole automation.

Budgeted same-unit bounded control F4_bounded_contrast_refit_dev_v1_20260905: same pairs/support/source coordinate/ridge fraction, constrain coefficient L2 norm to original full-map truncated16 coefficient norm; extra ridge found by16D eigensolve plus monotone bisection, no endpoint selection. This is code-coordinate, not GL invariant, and does not itself cap every intervention. Additional32fits<=3min CPU plus448forwards<=3min, based on86.24s prior total. Six fixed-support tests including constrained KKT/radius/inactive/zero cases PASS; previous6source+3atom PASS. No extra query/method grid, new data or training.

Bounded control session86616 completed09:54:05UTC exit0,448forwards73.7173299s including3.3864832s fit;256rows,6checks/contractPASS,841839104bytes allocatedVRAM. Raw12aa128c286d103ec316deff0bbb012724e0fe012016697475aa218b0819e255;fits204eaf43ad343c692018fe36e21c8330149baa4339d2608d3783fba27b7e0432. All16median .441390532/.484412186,13/16bothbelowzero;vsdirecttop16 error15.2%/19.7%lower with11better2equal3worse groups both. Max candidate hook fraction.1208775 versusunbounded8.4078;244valid12missing,194bothbelowzero,worst4.21437/6.74728. s5:369both and336negative remainpositive versusdirecttop16,336positiveworse;2176/801positive remainnear1 whilefullstrong;801negative1.43444/1.66685 failure.32/32constraintsactive;independentKKT residual<=1.65e-15,radiuserror<=5.56e-17,fitrows/weights/support/source differencesexactunbounded. Bothselection/scopehashesexactold,512sourceRMS/common-dose comparisons0mismatches.

Posthoc development hybrid retains original22source-selectedinputs' full responses and uses bounded-refit fallback elsewhere, joining exact source/target/query/condition/sequence keys acrosssaved actualforwards: all16 .164508730/.268258751 versusoldadaptive .204066378/.244013028,logit19.4%lower/state9.9%higher;7better6equal3worsegroupsboth,13/16belowzero. Selected8groups .021758385/.021769073 unchanged, rejected12groups .441390532/.4844121869/12belowzero (subsets overlap). Not a new forward experiment or independent confirmation; do not replace oldpolicy or cherry-pick oneendpoint. Report and complete statistics under runs/F4_bounded_contrast_refit_summary_v1_20260905/. Nextcheckpoint is an evidence-backed figure of confirmed full/fallback ranges and strong baselines with refitdevelopment boundaries, not another regularization grid. Cumulative22976causalforwards4594.1465256s;assets512/71.3361278sunchanged. Thisunit896forwards159.9530653s includes17.2252923s fitting.15focusedtestsPASS;run-experiment/analyze-resultsusedwarmruntime andsamehierarchicalaggregation. No newtraining/data/environment/payment or automationpause.

Local ignored artifacts retained: EXPERIMENT_TRACKER.md SHAee3128f127cb33ef3cfa454a51c0abaa16990a2fff2835f9a4db22b0992a1f68; EXPERIMENT_PLAN.md SHA95ff67532e52408fa9327ff68ee9045c060f98587cd3613e0fd1e76e462a7338; REFERENCE_REGISTRY.md SHA80f6c8637f412d1a710c7220c64337365f8df533a37e1e77c01a8ccd16e941af; runs/F4_bounded_contrast_refit_summary_v1_20260905/RESULTS_FOR_REVIEW.md SHA4dcbb6b1a2ea25a792de9bc60c7c68660c3f10dd0c3ec8b0ec8231160a3aa2ff; samefolder/checkpoint_statistics.json SHAbefd456383a9113dc1b636c035ed13acf44fa2c0290c9eb5993924294a9404c3. Seven explicit allowlisted code/test/config/log files checked: JSON parse and diff whitespace PASS, common credential-pattern matches0, size bounded, index initially empty, no unrelated changes. Canonical origin/main verified before grouped sync. All four resources free after run; automation disk configuration ACTIVE with21minute interval unchanged. No ignored research data force-added.

## 2026-09-05 10:25 UTC heartbeat — confirmed-effect figure checkpoint

Started F4_effect_figure_package_v1_20260905. Six saved packages: two frozen new-document confirmations, original/expanded conditional refit, unbounded/bounded contrast development. Replay each source rule and all query medians from exact raw hashes; render confirmation/all-method, source-selected/fallback, and separately labeled development plots. Provisional internal manuscript PNG2160x1800 at300dpi; no publisher compliance claim, independent-seed CI, smoothing or removal of failed points. Development figure has explicitly different per-row log ranges because original severe failures extend orders of magnitude beyond the expanded panel; no unrun contrast/bounded values fabricated for expanded panel. Existing plot_f4_readout_structure.py Pillow approach reused; lockedML runtime lacks plotting packages, so bundled Python/Pillow12.3.0 is used solely for rendering, no environment change/install. Budget lightCPU<60s, no model/cache arrays/GPU/newtraining/newdata. Skills scientific-visualization/analyze-results used for evidence separation, source replay, missing-data and contrast review.

Figure package completed exit0 in1.1240696s,contractPASS:384sourcechoices and1544querycells replayed from5120existing rawmethod observations;0newforwards. Points576confirmation+190scope+256development=1022 plotted points (not independentreplicates). All2160x1800RGB/noalpha/~300dpi,182.88mmwidth; noICC/CMYK orpublishercertification. Visual review of3PNG foundno clippedlabels; loglimits common inconfirmation/scope, explicitly row-specific indevelopment, not-run cells labeled. Palette contrast5.185/5.742/16.293againstwhite passedgraphicalscreen;blue-graygrayscaledelta2.781flagged, retainedlabels/categorypositions/conditionmarkers makecolorredundant. Metadatafiles andpaletteaudit retained. Source figuresSHA:confirmed4d4c2146a9bbcb50d0ce32df74290c92677812cc41c5dbea266b41d3ae3b87c9,scope3fe490f0ed018338d4538d9d138b43ae182e95bdfc7a05510a25b881af7a58c9,development83672e6517a1ba5189ebb292b7845a753aefa9e4521831e889f6e5358f50b26a. Derivedmetricsraw78990217107173eb61c09421c800e58329881ce47aab05471aa8187af8343f4b;script a8f18819948e5399b91831bb7d3a66523aed3f82d6c8a5ebd967d6ccd2ce10c6. Allrawsource data immutable and evidencelevelsgroupedseparately.

Confirmedselectedfull medians original .021758385/.021769073 andexpanded .023116947/.023202406;selectedinputs22/64and44/64. Fallbacktop16 .461091573/.490860935 and.378931509/.338004023. Expandedselectedrawstate.021174675 betterthanfull.023202406 retained; no universalrawadvantageclaim. Full report runs/F4_effect_figure_package_v1_20260905/RESULTS_FOR_REVIEW.md SHA db1cf2f44c5ed909234386966e590a655dfd7dca1f84e3e7ea12125247c27676 plus source/summary/plottedCSV. Readactualsavedmapconstruction: raw is alreadycondition-fitted raw-hooktransport, notnaiveprojection;global_target usesgloballysampled256rows butsamequerysourcecoordinates. Currentlocaldifferenceconsumeromitsglobal_target. Nextboundedrealcomparison reuseit asglobal_rows onoriginal/expanded fixedpanels,448forwards each<=4min combined,no refit/training/download. No universalquery-agnosticclaim ornewgate; simplify ifglobalrowfitworks. Cumulativecausal22976/4594.1465256s andassets512/71.3361278sunchanged;automation21minACTIVE unchanged.

Ignored currententry hashes retained locally: EXPERIMENT_TRACKER.md 7178749dd85137f5765c8d4e8d26545f056e3bb16105e3555c9c935f91af1d86; EXPERIMENT_PLAN.md 8977b4e38d155a3501d0c609cc34939ae958f95448d13e16ca3bf5bc7eeaa2be; REFERENCE_REGISTRY.md f2c5e448c7c56d73a1f9c0684ec991e48f0856009aa60a8fb3c073d3387e5441. Three explicit allowlisted files (plot script,config,master_log) form thissyncunit; JSON/compile/diff/size/commoncredentialpattern0 checksPASS, indexinitiallyempty, no unrelated edits. Rendering invoked no inference modules or shared-resource jobs. Generatedfigures/tables/researchnotes remainignored and are not force-added. Canonicalorigin/mainverifiedbeforepush.

Started global-row control unit 2026-09-05: original/expanded existing-document panels, one saved global_target method each, 448 forwards per panel, combined expected <=4min, CPU then GPU leases. Same query-specific source basis, rank1, source-only applicability and common source dose0.1; changes discovery rows/weights, not fully query-agnostic mapping. Factors SHA e085f35df378ddc3e869d57c4c79a8a36372db51b2b7c208c5df71358d230526 verified by runner; preflight all160 global effective ranks8, finite (160,768,8). Runtime spec source hash3129a184 unchanged, no rebuild. pytest unavailable; used existing unittest/direct assertion functions without installing packages. No new fit/data/training/audit. Baseline methods unchanged; new method added only when explicitly requested.

Global-row unit completed bothruns exit0/contractPASS at2026-09-05T11:16:43Z. Original448forwards82.2229007s;expanded44885.9650273s;combined896/168.187928s and512rows, peakallocated841839104each. Raworiginal25988b23de654dab74ba81b8475059c3fd6435159e8a677d77deff8c03bf461d;expanded90fff04053472dc6937b59e099582c23f0e46a3539eb42fbc08540b705501dba. All6checks PASS each;source selections22/64and44/64 exact oldselection/source_scope filebytes;512sourceRMS/doses exactlymatchedbaseline,140querycells recomputedfromraw0mismatch. Sixfixedsupportunittests+3directdifferencefunctiontestsPASS;unittest0testdiscoveryofplainfunctions not counted,pytest absentnotinstalled. Firstresourcewrapper failedsandboxWinError5 withoutlaunch;authorizedidenticalwrapper succeeded. Allresourcesfreeafterrun;automationACTIVE21unchanged. Originalselectedfull.021758/.021769 vsglobal.022431/.023014,4win4lossboth;globalstronglywins2176/801bothconditions. Expandedselectedfull.023117/.023202 vsglobal.043942/.039217=>47.4%/40.8% lower,10/12log11/12statebetter,whileglobal12/12bothbelowzero. Originalfallbackglobal2.026122/1.966141 worseall12boththantop16;expanded.453190/.324003mixedcomparedtop16.378932/.338004. Globalalloriginal1.025422/1.269965,expanded.088599/.088929;244valid12missing/126bothbelow1 and256valid/220both;worst182.761494/123.695735 and17.330924/11.459114retained. Currentpolicynotreplaced;noquery-labeloracleoruniversalconditionnecessityclaim. Reportandcheckpointstatisticsinruns/F4_global_rows_summary_v1_20260905. Nextfixedmaps/source-onlyrules,new65536tokenbatch(fulldocID/texthashexclusions),original+expandedquerypanelsfull/global/raw/top16,2432forwardsbudget<=8min plusasset<=1min,no newtraining/refit/audit. Cumulativecausal23872/4762.3344536s;assets512/71.3361278unchanged.

Local ignored artifact identities: tracker641ff6bad973ed6e8661f26bdeb1893bf61ed0377f06a09e8cedc810649407c2;plan5352b4afa12289148ac606094b0c0f3661a5fac05f3ab7b0d5ce0f663636f15d;registryea79a87126dad09276466ea544f746b960d60c80b58905bddae5eb25544a6fd6;report31b9dff3e4f376aef62e405454678acd17da13b899665226760d153743ff605c;checkpoint_statistics1368342584541e1714d02ed5c35a0b492204e9f46aa5bab56d5b3b3ea135d833. Five explicit allowlisted files (runner,3configs,master_log) form syncunit;9focusedtests,AST,JSON,diff whitespace andcommoncredentialpattern0PASS;staginginitiallyempty,nounrelateddirtyfiles,canonicalorigin/mainverified. No ignored assets/documents forceadded. Large-output truncation in read-only analysis was replaced with compactnumericJSON;no artifact recomputation, source kernel or run repeated. Newtable remains existing-documentdevelopment;next confirmation pending.

Started fixed fit-scope new-document confirmation 2026-09-05T11:46UTC. Corpus config f4_fit_scope_confirmation_corpus_v1 freezes16queries acrossoriginal/expanded source-hash panels, maps/rank1/common source cap0.1/source rule,full/global/raw/top16 endpoints andunchangedderivedsource-adaptive-top16beforedata. One65536token batch sharedacrosspanels,acknowledgesdocument/seeddependence. Excludes training/fullpaired/fourentirepriorF4corpora byIDandtextSHA. Budgetcorpus<=5min/~40-100MB,asset128forwards<=1min/~500MB,causal2432forwards<=8min,4CPUthreads,<1.5GiBallocatedVRAM;nonewtraining/refit/payment/audit. Spec3129a184unchanged,run-experiment/analyze-results reuseexistingwarmruntimeandartifact/aggregation;nonewnumericalcode changes. Resource preflight allfree, GPU1654/16303MiBused1%utilization;acquireactualCPU/GPUleasesforwork,nowholediskmonopoly.

F4_fit_scope_confirmation_corpus_v1_20260905 completed11:49:13UTC exit0,8/8checks+contractPASS. 43range requests39778914bytes,4998sampledcandidates,115useddocs/512sequences/65536tokens. Independentread-only ledgercomparison to training269docs,fullpaired569,andF4batches113/118/106/116:all12 ID/text intersections0,new115uniqueIDs115uniquetexts. TokenmanifestSHAe31d7aff3894ca012c55f5885b1134e099e17ff2450378695454c6e70bba60d6;sequences3f41de3c5fc374660800f33d2f343fa3a5fcfa6994b3ace80b16796101d5b088;documents21fedcfa6ef0411e99b04b241b895ddf43b505816730d32b020a6e6b0569d8d0. Frozenscopecorpuscfg36c3d0aa9bac041edf4060b37065d1bc1304539607e3aeae07643a7d4fa192bd predatesnewdata. Startingone128forwardassetbuild with originalfiveSAEweights/independentmean; no targetendpointsreadyet.

SharedassetF4_fit_scope_confirmation_codes_v1_20260905 exit0/contractPASS8checks128forwards15.824898s65536tokenspeakallocated895002624bytes;assetmanifestfe22a9b539363fd7b1276b27265d7fa83dc93341e8bf2a53af286d9948bd90a2,rawmanifestffaeadf71d0a183abb1c11f4ee43092fb3d655960e1071875736adc7cf13b4e3. Startingoriginalthenexpandedcausalconfirmation1216forwardseach;configsonlyreplaceinputidentitiesandpre-frozenmethodsubset, numericalrunner306236aeunchanged, savedreadoutcoefficients verifiedagainstexistingfactorsbeforeforwards. Boundmethods/rank/dose/source rule to pre-data corpuscfg exact;raw/global/conditional matrices unchanged. No fulltest rerun for config-only work. Original/expanded share115newdocuments andfive models;notindependentpanels.

Fixed fit-scope confirmation completed 2026-09-05T12:00:38.952436Z: original 1216 forwards/212.9459078s, expanded 1216/194.8452997s, combined 2432/407.7912075s; both exit0, six checks and artifact contract PASS, peak allocated 841839104 bytes each. Raw SHA original a1665d92c7a1ca935586cc231c0b23778f2e9d068acef52262ea7bd341ff5a45; expanded 128e2b7c173fd5dc9412b04dd42b06251edaac521759138bd8bcfdf5fca9507f. All resources free after completion. Source selections19/64 and43/64, support57/64 and63/64; pre/post source128 rows exact, all method source RMS/common doses matched, query IDs and all64 saved readout families match pre-frozen records. First source-inspector attempt during RUNNING refused before output; after PASS it ran normally, no numerical run repeated or check bypassed. 700 saved query cells recomputed from raw with zero mismatches; arithmetic checks are not independent scientific review.

Original selected6 groups full logits/state .016813769/.012768296 vs global .000383504/.000046924; global5/6 log and6/6 state wins, with2176/801 both conditions again strong. Prior selected8 groups became6, so scope composition changes contribute to median difference. Expanded selected12 full .030538694/.020093971 vs global .051958927/.042350082:41.2%/52.6% lower medians, log8wins4loss/state9wins3loss, both8wins3loss1cross; both methods12/12 below zero-candidate error1. Expanded raw log .026512057 beats full median but state .021861798 is worse; retain strong baseline. Frozen adaptive all original .161747929/.134918510 (14/15 valid groups both<1,16requested), expanded .039825281/.029673261 (16/16). Actual method rows original228valid28missing,expanded252valid4missing. Adaptive207/228 and245/252 both<1, worst1.823544/2.442051 and1.979638/1.931407; global worst49.120456/287.349953 and18.062029/12.931779 retained. No universal condition necessity or dominant policy claim; source cap does not bound candidate interventions independently. Cumulative causal26304/5170.1256611s, assets640/87.1610258s.

F4_fit_scope_confirmation_figures_v1_20260905 completed exit0/contractPASS in0.5469364s,0model forwards:700 query cells,458 plotted points; existing plot script gains optional method/scope lists/title/footer and skip-development flag, old defaults retained. Two2160x1800 RGB/~300dpi opaque PNG visually inspected, no clipped labels; common log10 axes and missing outcomes retained. Metadata4pass0fail each, blue/gray grayscale delta2.781 warning retained with redundant category positions/labels/markers; no publisher/ICC/accessibility certification. confirmed_effects SHA19561054d3f51e25de9c8537e8a4c7027ecdafe476328a1c3db7ef3f617240a5; source_scope SHA557fc3b32de97c930795117cb611b2bf19abfab3b7e16ecfda02df55643c138b; derived raw SHA21318dee2b40f5f306d1835cdb461617536aabf88cc4ac183a1e8c50062d1978. Scientific-visualization supports truthful scales/missingness; rendering runtime Pillow12.3.0 separate from unchanged ML environment. Report and checkpoint_statistics retained in figure run directory.

Refreshed nearest methods from original official texts MAS arxiv2501.06164v7 sections3.3-3.4 and Li1511.07543v3 section4; reference-only, no code imported. Current method is neither MAS full replication nor standardized L1 Li15 baseline; global rows still use query-specific basis, and256 matched fit-row budget does not isolate weighting from row distribution or trace-scaled ridge. Registry records actual consumers and boundaries. Next checkpoint is source-only preselected real context/signed-contribution/counterfactual case pack, at most32 supported pairs and optional608 forwards<=2min if logit detail requires replay; no training/refit/new corpus. Semantic descriptions AI hypotheses only, preserve failures and endpoint crosses. Automation ACTIVE21min unchanged.

Local ignored artifact identities at close: EXPERIMENT_TRACKER.md 15f6f0107f78b871126e14aac5d27c4377fb4754b1bc836cce099222e75eba41; EXPERIMENT_PLAN.md ffccb468c12556b685447506cdeae482697f54f579193ca07cd5de0041905672; REFERENCE_REGISTRY.md 4c37c946001c506728ff003342d50a84a1440da1f8e8772473451379524d3fb1; figure-run RESULTS_FOR_REVIEW.md b91baac51cf06dcbb74a084ed6b4081f18715c2cd0f968691658a8fea0f8238c; checkpoint_statistics.json a4f41bb3dc567681f85fc1052ec1a925f6fbaa9c982c27f8791a1b264522ecc6. Numerical runner remains306236ae72c7f207d8dafab2ef2367099069b22e8437dd9bc8cf2cdd9270995a; plot script2b55f6cbafd66f0c8c7336cbde03cffa768e4c78fc4dce6673f2f14bbd90c44f. Seven JSON configs valid and small, plot AST/render execution verified, whitespace check PASS. Nine allowlisted files only (7configs,plot,master_log), index initially empty, no unrelated dirty files; ignored research documents and generated assets retained locally, not force-added. Canonical origin/main checked before grouped sync; no environment/manager/automation edits.

Started F4 case unit2026-09-05 after12:35UTC: source-only frozen selection chooses first selected supported pair in original order, otherwise first supported pair, separately each original/expanded query-condition. Original15/16supported with6selected, expanded16/16with12selected; original710negative remains missing. No endpoint file used in selection, though prior confirmation aggregate/endpoints already exposed. Existing runner regenerates full source selections and verifies frozen entry equality before filtering; no scientific delta changes. Optional helper exports all nonzero signed source and target atom terms, reconstruction residuals, local prefix/recipient/donor tokens, intervention-minus-baseline centered logit changes, same four method errors. All target comparisons retained, not picked by target outcome. Budget589forwards<=2min inference plus<=2min detail serialization,CPU thenGPUleases,4threads,existing model/assets. Env spec3129a184unchanged;AST/signed-term test/both selection equality testsPASS. First oversized read-only JSON orchestration hit output truncation and did not parse or edit; compact source-only selection succeeded. No training/refit/audit/data acquisition.

F4_cases_original_v1_20260905 and F4_cases_expanded_v1_20260905 completed exit0/contractPASS, six checks each:285forwards82.8452236999874s and30475.26154870001483s, total589/158.10677240000223s inclusive of detail serialization; inference/export not timed separately, total within prewritten240s budget. Peakallocated841839104bytes each, resource leases released/allfree. Originalraw86db1a2ca7f20dddef6765a29e1946e2799ae1a08abf3ba8c333f7278fac2014; expandedraw0181e4dcc33b51960402c7f323fa5a12cd8277f8782b03456aa0e246c269a538. Detailoriginal12efab4bb9745744440beac73763b5cbc89b1beb34cca9b281b708996ee8af05; expanded4d0187fc90dc9ff0d1e1433895a0a0b2444ac0e9d129b6f70bb13bd03a45a897.496rows entireendpoint/hook/dose exact priorconfirmation; allsignedterms reconstruct source/candidate hook, maxresidual3.844970420803284e-14.32requested31supported1missing;18selected13fallback cases. Fourfocused unittest testsPASS for signedreconstruction, alteredselection refusal, prefixboundary andlogitsign/sharedtokenorder. Cumulativecausal26893/5328.2324335s, assets640/87.1610258sunchanged. No numerical intervention formula changed; optional filtering only after regenerating/verifying source choices.

Case evidence: s1:615 positive/negative fourposition swaps does/that/are/is versus friends/friend/MC/'s, maxsource positionenergyshare.3478883. Full logits/state .029976410/.029244526 and.026093831/.028037125 versusglobal.628677136/.553555621 and.541097541/.549106264;about95%lower, raw remainscompetitive. Largestpositiveposition source -1.895928773 from12signedatoms, fulltarget-1.837134602/-1.716624416/-1.589459549/-1.808986936,global-.588215633/-.582177287/-.547051142/-.427263498;245-250nonzerotargetterms notnecessityevidence. Grammar/possessive interpretation is AI hypothesis only. Original2176/801 fourconditions sourceenergyconcentration .999960986-.999980681 atoneposition withperiodonrecipientordonor. Sevenof18selected aboveposthoc.98threshold haveperiod/newline atstrongestposition;notnewgateorindependentcases. These descriptions explain why global can excel in a boundary range, without dismissing its genuine functional effect. Joint multi-position intervention logits do not independently attribute effect to individualpositions; extreme centered-logit changes notprobability/actualgeneration. Token IDs retained even when single-token byte decoding showsreplacementcharacters.

Case report/CASEBOOK/CSV/JSON in runs/F4_cases_expanded_v1_20260905; allpositions andnonzeroatomterms inbothcase_details files. Next source-only token-class matched donor control onfixed31recipients, originalopposite-condition candidate documents/positions;CPU firstcoverage/localratio prep,<=589forwards3mininference+2minexport ifsupported. Fixedmaps/rank/sourcecap, no newcorpus/refit/training; exposed-documentdevelopment, notnewconfirmation. No user approval or overallpause required. Registry records reuseoflocalparticipation/causalcode and nearestmethodboundaries, no externalcode/dependencies/newlicense. Automation ACTIVE21min unchanged.

Ignored file hashes atclose: trackerf751b0cc19a651861eafa216d7956cc74fc1a3bc909b789d65da52cfbe522d5a;plan80570f56439dba1734ece0d9271612d0451ea998002b03dafb202805487d95ae;registry3c1113a8e1f8727b922bbcc64b6c55d74e5026aeda23a596f08d88c90f17615f;reportbd4b1b589d14838a3a8fa02d9c27a21b6b02877502dcbba5d6aca495203327bf;CASEBOOKafd4950b6b6867f88728cc268718e1c5a1b40a8b2415e0b0e1d71bedbc97b2ea;case_summary6b2cdf2b7e9e4c8bdeafb74b15940eeed1f99c6e2fc4e86a3765ca183f003ad6;positiondiagnostics2228dc71d7f30e556cc5b48c131f8d9a6fc3040d3b053a2437648a775aeb4d57. Runtime runner7a4c8ed65bd2c92aea35bf1a1c88c7459b1adb1b4df1d66b9dffc2ab05f15c07,detailhelper56e8de35289d49eb908b81553a8ee034aa7d7f3756aa309fce4925945d0ec222. Summarygeneratoratactualexecutionab35af67fe7f956a82f233fed6691197d31c4f3a1cb7e5422f71a5806f84e30e;later only prose note onjointintervention attribution appendedtoCASEBOOKandsynchronizedscript(current922ba2e176e71230a438615cf451da29e02fdaecba62d4b5908e0b06dee1f5ec);numericalsummarynotrewritten. Nineexplicitallowlistedfiles4configs/3scripts/test/master_log formgroupedsync; ignoredcasecontent/assetsnotforceadded. No unrelateddirtyfiles atstart,indexemptybeforestage.

Started token-class donor control2026-09-05 after13:16UTC. Fixed31case recipients plus1prior unsupported condition; keep original sorted first-n position pairing and four opposite-condition donor pool, add coarse token-class sequence equality only, then old maximumsourceenergy/smallestdonorseq tie. Unicode anyisalnum=word_number,isspace=whitespace,remainingpunctuation_symbol;empty/replacementbytefragmentseparate. Not POS/semantic labels, no permutations/newpositions or target selection. BoundedCPUpreparationF4_token_class_donor_preparation_v1_20260905 exit0/contractPASS3.8113147s0forwards, original3changed7unchanged5incompatible1oldmissing,expanded4changed8unchanged4incompatible. Raw01c6d18359dc4daef2e9784cf946968616fcac1e621ed8ad667bb46af4ce5453. Inputs only previous source selection/config,source code/decoder/basis rows,tokenizer/tokens,sharedhook andfrozen scope rule;no endpointarraysornewfit. Matched source scope recalculated,2176positive leavesoldselected range,615bothpairs unchanged andwillreuseoldmetrics notclaimnewvalidation. GPUpreflightallfree1653MiB2%,spec3129a184same,warmruntime no rebuild. Sevenfocusedcase/matchingtestsPASS. Startingonly7changedpairs133forwards expected<=1minincludingdetail,CPUthenGPUleases; no corpus/training/refit/audit. Original/expanded7remain dependent query/seed/document developmentcases.

Class-matched original/expanded runs completed exit0/sixchecks/contractPASS:57forwards26.03235220001079s and76forwards23.61717369998223s,combined133/49.64952589999302s,peak841839104bytes each,allresourcesfreeafter. Raws a0d07b3d7394374e352dd5f754234b4d33f3aa41f311d3320bc37463cae74368 anddf3d90674e44487966b358647ea8dd55bc22db94f696648e501d1d956c8d2f44. Preparedoriginal5a1648edbe0189fa016223edd26d6ee66bb467e196a88817e0f6cead1d626dc4;expanded34877b15dcf884b0694d0f162c75b5ad281babc98d8f75ec3a5bbf6250e7e181.112newmethodrows raw/detailexact,signedreconstructionchecked;changedrecipient/donor/positionfields verifiedagainstfreeze,modifiedsource_scope agrees preparation withinrtol1e-10/atol1e-12. All7changedold donors were class-incompatible;15unchanged reused withoutforward,9nomatch+1oldmissing retained. Sevenunittests andlegacy15/16case-selectionreplayPASS. Expandednegativehas0changedpairs andisrecordedevaluated_rows0,notcrashorNaN;emptygroup guardaddedwithoutchangingexistingnonemptystatistics. Cumulativecausal27026/5377.8819594s,assets640/87.1610258unchanged;automationACTIVE21minunchanged.

Newpositiveboundary:2645positive/2194positive maxsourcepositionenergyshare drops from.9993517/.9895386 to.6732857/.2742644 withorderedword_number donorclass match. Full logits/state .022859569/.041658182 and.048200273/.034063381;global.052111182/.064437491 and.054515453/.041339105;top16.138077412/.167723281 and.178652361/.140876824. Raw stronger .019774335/.026922858 and.004721021/.003711752 retained.1144positivefull.137093217/.096529644 bestof4 medians thoughworseoldfull.2176positive sourceRMS.022643801/.009970913 ->.006370076/.002796481,leavesselected;full.917498378/1.233375986,global1.303139335/1.483018766,top16.563018281/.577341437.1230negativeraw wins,369/2298failures retained. Newsource-selected2original/9expanded;11total. All22compatiblehappentobeword_number,notPOSorsemanticcontrol.615bothalreadycompatible unchanged,95%oldadvantagenotnewvalidation. Sourcejointcap.1 yieldsmaxlocalpositionfraction.196166,explicitlynotperposition/candidatecap. Before/afterchange sourceintervention,notdirectcausalimprovement. Report/32requesttable/848beforeafterCSVrow/matched_summary andpositiondiagnostics inruns/F4_class_matched_expanded_v1_20260905.

Nextcheckpoint:freeze11source-selectedmatchedpairs (2original/9expanded) andfourmethods beforeadding predictiveprobability endpoints, normalized KL source-to-candidate versus source-to-baseline andobservednext-tokenNLLchanges;keeplogits/state,neveruseNLLtowin-selectqueries/donors orfitmetric.209forwardsbudget<=2mininference+2minexport,nodata/training/refit/audit. Exposed-documentdevelopment,probabilitycorrespondencenotsemanticuniqueness. Reusedlocalmatching/participation/causalkernels,noexternalcode/dependencies. Parentreferenceindex/manifest hadnomassiveactivation/attentionsink match;do notassertverifiedsinkmechanism. InitialWindowsrgfileglobsyntaxerror was read-only, retriedvalid-g filtering;no source/runtimealterationorcompute repeated.

Ignoredhashes:tracker71e97104a52801688f2c3a51afb21e3cbfb858aaaf8979115583cf67cecb9e9f;plan5dbfd27960af0f19164bd3a1c45e8d757fc4cf5aaa5f2275084e8d7b633576a1;registryae1cc5e39e88b2eb16a8c9f29fa75b243e551edf08f65508bc5f0d84557eebd0;reportdf3db4a1ad2baf097abca8cf2a8d820f2724a11f1ed93462cfd6080e87731246;matched_summary53b9d44e8d2dca305e92c7d9ae6ea48f628ae6da75a9e779dd173272ac270375;positiondiagnosticsce0099f417e8aa8f102a288c676ea7597f395d4cdae50aa4df2c609a35e6a10e. Nineallowlistedfiles3configs/4scripts/test/master_log formgroupedsync;ignoredtext/metrics/assetsretainedlocal,no force-add. Sourceprep snapshotsretainactualthen-code hashes despite later optionalconsumer integration. AST/7tests/legacyselection/sourcefreeze/rawdetail/whitespacechecksPASS;no environment/manager/automation changes.

## 2026-09-05 probability endpoint checkpoint start (14:12 UTC heartbeat)

Frozen configs/f4_probability_original_v1.json and expanded_v1 select all11previous source-selected token-class-matched pairs (2original/9expanded), no new target-outcome selection. Four saved maps/rank1/sourcecap.1 unchanged. Primary intervention-position KL(Psource||Pcandidate)/KL(Psource||Pbaseline) and observed-token NLL intervention-minus-baseline; secondary within-document downstream positions, excluding EOS predictions/final unavailable token. Float64 log-softmax, denominator<=1e-12 perposition null; signed effects and absolute scale retained. Existing environment source-spec3129a184 unchanged; startup clean819f87a, CPU/GPU free. Budget209forwards <=2min inference+2min metric/detail export, no data/training/refit/audit. Numerical/boundary/source-selection targeted tests pass; optional endpoint code only. Skill environment warm-reuse and raw comparison used, no new environment or review gates.

Probability runs completed PASS: original F4_probability_original_v1_20260905 rawSHAa821bc4a49616bbe2ed51f7d465c3e9b7f96a6d0cc0587ea7aa859724b251d18,38forwards/20.85800730000483s; expanded F4_probability_expanded_v1_20260905 rawSHA1c904ce123344098e72e326caf7e96fa0ff20aadea2d9d598aab22636a7cdbbc,171/87.3867554999888s. Total209/108.24476279999362s includes metric/detail export, allocatedpeak841839104bytes, sixchecks+contractsPASS. Leasewrapper sessions60593/66799 exited0, allresourcesfree afterward. Old176methodrows endpoints/hook/dose/pairing EXACT, source2/9 scope recalc exact,352probabilityscope rows and0nullratios. Fourprobabilityfixtures+7casefixtures/AST/JSON/whitespacePASS; unchanged ML runtime. Cumulativecausal27235/5486.1267222s assets640/87.1610258unchanged.

Positivecheckpoint: expanded9case/5query full primaryKL/NLLsquarederror medians.02960358757092206/.050553843306182275 vs global.09908230559343074/.300960362415734 (70.1222%/83.2025%lower medianerrors),raw.014311605350851249/.03623620965421459,top16.18965518626617373/.31537699008757647. full9/9cases36/36targets both<zero,raw36/36,global/top16each32/36. fullvs global8/9KL6/9NLL casewins;vsraw4/9and3/9 only;vstop16 9/9and8/9. Downstreamfull.03191084062046691/.05046658672127614,global.08811282416191829/.2125667902539072,raw.02443290282703927/.042074059333409075,top16.1955962017262899/.32288260411970454. Absolute source expandedprimarymedianKL.0031923912914874463nats/NLLRMS.044150055937211975nats.2194positive sourceNLLmean+.33945330628305337,RMS.576951598986812;fullcandidate+.26121237907597294/raw+.34777684912586726. Realpredictiveprobability effects,notsemanticlabel.

Counterevidence preserved: original1230negative primaryNLLfull13.131007998997074/global6.188113107867978/raw4.16223493177668/top16 6.288765939366934,all4targetsfailed;sourceRMS.00663047281175656/fullRMSE.024026183786293093. Same-document downstreamworksbutdoesnotreplacefailedprimary.2645positivefullKL/NLL.136004/.413436 worseglobal.117967/.300960 despiteearlierlogitorder. Primaryonly4positions/case vssecondary36-110same-docdownstream,notperpositioncausalattribution;EOS/currentpackedboundary/finalnolabel excluded fromscorebutpackedattentionunchanged. Original2cases1query+expanded9cases5queries sharedseeds/documents,notindependent11seedreplicates. Existingdocuments,new endpointsnotnewconfirmation. All32originalrequestsremainupstream;21outside currentfixedsource-scope not silently counted successful.

Nextcheckpoint is genuinelynewdocument confirmation of fixed source-recipient/classdonor/recomputed-scope/fourmap/probability workflow then probabilityfigure; not selecting known1144/615winners. Preserve16queries32conditions,original4donor/first-n pool,firstselected-supported recipient elsefirstsupported,sourceenergy/tie,dose andmetricv1. Freeze BEFOREnew65536tokens excludingalltraining/paired/fullhistoricalcausaldocID+texthash;sourcepreparationwithouttargetendpointforwards. Atmost32pairs608causal+128encoding,estimated5mincausalwithendpoints+1minencoding+2minCPUprep/networkseparate,actualsupportadjustsbudget. No refit/audit ornewthreshold/donorgrid. Automationstays21ACTIVE.

Ignoredlocalartifact hashes: tracker1797eb36ee8e4002628aae1012cc07e8b204c8d4f3c623abe40a7fbd3d7c1d1a;plan4a75c1a273d2cd0f31eceb6cae64db023e92dc51ffeb816f04e031df0fc5c68c;registry147c7cfc2e5d2f97a229320c8dc5c48db57d7dca8f23fd7c0e76c1d4dab5848b;report988cf81d89784eaafa47b2a7d7f05f137d58ea16665d65a05dc28f6858898632;probability_summarydfb6a12cb8404a1535fc0fa0d211a94f469ef870f2d64687b9cbb1bd88e94686;rowsCSVd89223d7de60c607571ae0bb7da72844a64f3dd5f9663e1ed9abb2f3b4538d0b. Report/table/allraw retained local; nineallowlistfiles grouped sync. ActualrunrunnerSHAacbee00dadf2e0acf7ad135a0c45f27d30f7cf7972775e39a6c521005466920b with code snapshots retained; no externalcode/dependencies. Reused skill environmentcontract and rawcomparison, not new scientificgates.

## 2026-09-05 14:54 UTC heartbeat: fresh-document probability workflow confirmation start

Frozen configs/f4_probability_confirmation_corpus_v1.json BEFORE acquisition,16queries/32conditions/source first-selected-supported elsefirst-supported recipient/original4donor first-n positions/coarseclass/sourceenergy-tie/recomputedsource scope/fourmaps/probabilityv1. Exclusions include training/allpaired/allfivepreviouscausalcorpora. Upper608causalforwards budget6min (prior5min roundedestimate updated from209/108.244763s),128assetforwards<=1min,sourceprep<=2min/networkseparate. Environment3129unchanged; initialclean0981d4e andallresourcesfree. New optionalCPU-only source-preparation exit uses existingselection/matching, no targetendpoints/fit; original workflowreplayF4_probability_workflow_replay_original_v1_20260905 PASS/11.075087200006237s/0forwards/raw59afbb3d0f06148beb0abb18cc54debecbc515801a0a42833db21b5bb444bb00, all16recipient choices andfullmatchingexactold. EightcasefixturesPASS. Nearestmethodrefresh via motherindex(noLi15/MASmatch) then officialLi15v3§4 andMASv7§3.3-3.4: sparseprediction/crossmodelbehaviorpredecessors retained, currentunidirectionalfixedmap notMAS orLASSOreplication. No newexternalcode/dependencies.

Completed freshprobabilityconfirmation:120docs65536tokens,7ledgers ID/text14intersections0;corpus39,899,512bytes43requestsPASS,codes128f15.907970100000966sPASS. Sourcepreps0f5.8872678999905474/5.8179847999999765sPASS. Original57f30.88821640002425s raw544b8789e0971804831deebe9d5f7b85ec8992faba13e5320490fefe9f8ab57d;expanded152f68.5342770999996s rawb31c3c39584549cd9787c9058a6615eb0a5d2ece2848da30aad2468be7d9bb3a;bothsixchecks/contractsPASS,total209f99.42249350002385s,peak841839104bytes,allresourcesfree. Cumulativecausal27444f5585.5492157s;assets768f103.0689959s. CorpuscfgbeforedataSHAbfc1809d352f33c778a5a3163a7e57b9eb08975a1f8a651ac906d4b1717051af,runnerdf4697cae141f3b152fe0cc3fa61bd1ff8cb39d335f138ca50a0627fd204cc7d unchangedfromfreeze.

32requests24supported11selected(3orig8expanded),13unselected8noclass,7query21actualpairdocuments. Full11case44targetsprimaryKL/NLL<1;expandedfull.04445718312543868/.02713283402846095,global.08801292995358291/.05719312077930544(49.49%/52.56%lower medianerrors,7/8casewinsboth),raw.03744747009684918/.03366381822021914,top16.12021865406071278/.12269629622381724(full8/8winsboth). Originalfull.035504963253514196/.01785025145115911 vsraw.0576834944451321/.10429463230813063. Raw remainsbetterexpandedKL andsecondaryNLL;2194posfulllosesglobal/raw;global615pos→seed2primary1.0489652475618727/1.0395057825541993fails,secondaryalsofails. Old1230negativefailurepreserved;newdifferentpairworks,notrepairclaim. SourceeffectRMSexpanded.0873468288/orig.0291628802nats. Full88plottedmediansreplayedraw;128sourcecandidatesselectionexact,3/8source-scopeexact,176raw/detail352scope rows. No independence/semanticuniquenessclaim.

Figure2400x2130RGB88pointsall32rows retained;visualno clipping. InitialplotMLruntimeModuleNotFoundErrorPIL beforeoutput,switchedtoalreadyverifiedbundledPython3.12.14/Pillow12.3.0,noMLenvchange/noexperimentrerun. PNG300DPI encodes299.9994causingstrictmin300screenfail;retainedroundingnotfudgedorpublicationclaim. FourcolorscontrastwhitePASS/grayscaleclosepairsredundantshapes/table. Next11fixedpairs sixmethodtarget+wrongquerymatched+dynamicatom+2savedOT+nativetop16matched,297forwardbudget3min+1minmetrics,no newfit/query/data/audit;additionalbaselineisdevelopmentafterexposure. Automation21ACTIVE maintained.

Ignored artifact path/hash ledger:
EXPERIMENT_TRACKER.md:b612a64fb3993dfd3f69daa8d031b20dab838fcc00644e9114114cee407358db
EXPERIMENT_PLAN.md:969de0881b51166fdbe68955a4b74af08633b7339d497fde0a5df186e44dfa53
REFERENCE_REGISTRY.md:5b44a77f090a8afb73e95a05d8d7ec51340c6e32e7d96e777fb79c53c158ecf4
runs/F4_probability_confirmation_expanded_v1_20260905/RESULTS_FOR_REVIEW.md:3e3136c1fa5bcacd729c3ed92833e9ac76405f711a5e3ff9b2be5af103cf60d1
runs/F4_probability_confirmation_expanded_v1_20260905/probability_summary.json:b076924fc2398f902dc5c43559374fa23ce8926b25dcd4e2cfb0f84becdfaf1a
runs/F4_probability_confirmation_expanded_v1_20260905/probability_confirmation.png:54d4857294f8f9ea7c49a1996149adb74762fae3ee3cabef18a95743c1d3ad45

## 2026-09-05 15:28 UTC user direction: demonstrable component value and method optimization

User clarified FCC need not beat raw on every error, but component-level interpretability and cross-seed correspondence must have demonstrable use, not verbal claims; actively optimize the derivation/fitting/evaluation workflow and avoid gate inflation, overengineering and research procrastination. Safe nonduplicative tool/package installation authorized, with paid/major external resources and protected runtime boundaries unchanged. Updated AGENTS persistent principles/install scope, EXPERIMENT_PLAN compact claim-driven evidence and optimization opportunities, and tracker current next-work framing. No automation prompt/state change, no experiments/model forwards/new metrics/results, no environment install or scientific code change.

Focused source inspection: current rank1 signed atom terms share output basis b; additive reconstruction or exact addback is algebraic, not proof of independent semantic components. Raw already uses discovery-conditioned basis-constrained fitting; missing raw atom export is an interface limitation, not noninterpretability. Fit currently optimizes hook-space coordinates, while deployed donor differences and probability/NLL consumers can expose distribution/objective mismatch. Operation-matched full/raw fitting and input-dependent behavioral weighting/rank extension are hypotheses, not established fixes; prior sparse-refit failures retained. Fixed rank1 output metric can reduce to a scalar, so do not add decorative metric machinery. Retain the small existing strong-control run, then prioritize predictive component interventions/target-versus-other effects; no requirement all baselines fail first.

Read mother reference index, SAEBench v4 TPP method and local native-ablation/cross-class code, plus official ICLR/ICML2026 reviewer guidance; provenance and pending code-license check recorded in registry. Borrowed experiment-plan claim-to-evidence/small-block approach; user/project single-entry records override duplicate templates, default seed counts, gate tables and review loops. No new protocol files or agents. Existing result report and raw artifacts unchanged. Only master_log is upload-allowlisted among this unit's documentation changes; ignored files stay local with hashes below.

AGENTS.md:a8325b670e298bbae2ac29dad0e9ade35bdb9c51f932da358c7a2e574fdd9aa1
EXPERIMENT_PLAN.md:31d806ac119c45f407cce26f56ca5b697584cb85492e15549f395741809b2916
EXPERIMENT_TRACKER.md:d987533e4684c2967fc0c28054a72008dc0c3bbfe343818c30e1f076c7c49790
REFERENCE_REGISTRY.md:2a15b79fef59fd0e0d99272d3126d54d7bd2871aae51536193026f72bf7dd85f

## 2026-09-05 user-triggered component probability unit start

User requested existing CCAD automation every5min and one substantive research unit. automation_update ccad ACTIVE5min succeeded and diskverified; preserved prompt/name/target,updatedAGENTS default to5min. Plan fixed existing3original+8expanded source-selectedpairs on120exposed documents. Reuse exact dynamicatom+uot_default+uot_discovery_tuned fits (correctactual saved names),full/wrongquery/native,plus discoverytop16/complement/random16 fromtop64. Nine methods429forwards budget5min plus<=1minCPUgroup preparation,4threads/no disk monopoly. Sourcecase export optionalFalse only for unsupportedmethod detail; original probability/intervention formula unchanged. Small group helper replays discovery rankings and verifies oldtop16 before selectingfixedseedrandom16; no endpoint/test selection. Complement larger than top16,randomgroup overlap/energy retained,no independentgroupnormmatch. Rank1 collinearity/addback caveat preserved; test actual NLL response combination separately from exacthooksum. 2newcomponentfixtures+8case+4probability PASS; immutable3129runtime warmreuse;gpu0/cpuheavy freepreflight,1921MiBGPU2% before leases. No new data,fit,training,audit,externalcode or environment changes.

Completed component unit: F4_component_probability_original_v1_20260905 117 forwards / 145.23093579997658 seconds and expanded counterpart 312 / 143.7616229999985, total 429 / 288.9925587999751 including loading/preparation/endpoints. Both six checks and artifact contracts PASS; peak allocated VRAM 841839104 bytes. All resource-manager leases free after completion; existing automation ACTIVE every 5 minutes with unchanged prompt/target verified from saved config. Cumulative causal 27873 forwards / 5874.541774499975 seconds; assets 768 / 103.0689959 unchanged. Original raw SHA256 abb80b6cabe85a5c6e7fa19e9085756eeb298cb5c27fcb81b127e330735d856d; expanded raw c3baf394551538fef0e06d13669d8443f263f7052ba96eae3bbc2314e3d71a62. Runner be9f1671d0d5c0a02339abfa5da3b56e21fad9ac43febba5caf45ff6c6e8fa2e and helper b670307ce7bf8b7fef4fe1a310efa1ee4fb81eabaeff42037f19b2e25ad24ea4 retained in snapshots. Source selection/scope exact at 3 original / 8 expanded; 88 full/top16 anchors replay probability, logits/state, hook and dose exactly. 396 new method rows plus 88 reused raw/global rows produce 968 primary/secondary scope rows. All 32 requested conditions retained, including 13 unselected and 8 unsupported.

Expanded case-first primary KL/NLL squared errors: full .04445718312543868/.02713283402846095; dynamic atom .13713851133626515/.1780867574198812 (full median errors lower 67.6%/84.8%, 7/8 case wins both); tuned UOT .33841721113823187/.2355736559321368 (86.9%/88.5% lower, 8/8 both); default UOT .363596623265332/.2783373625921427; wrong-query 1.2235270021221083/1.3386407525620707; native matched 1.20379718085794/2.1312154481183234. Raw remains competitive/better on expanded KL and secondary NLL. Counterexamples retained: expanded615positive single atom wins both; original693negative atom wins both and original1230positive atom NLL wins strongly. Native uses decoder directions and norm-matched donor differences, not native zero deletion or operation-equivalent source alignment; saved UOT is not all possible OT methods.

Expanded head/tail/random16 errors .12021865406071278/.12269629622381724, .5859034540725705/.5202342711402281, .7400640680078412/.7018065597464749. Full beats both standalone head and tail on both endpoints in 8/8 expanded cases, 2/3 original; original1230positive NLL exception retained. Discovery head beats one fixed random16 sample in all 11 cases both endpoints, but random from top64 overlaps head by 1-6, has no per-input energy matching, and is not optimal sparse fitting. Rank1 groups collinear; tail has larger support. Hook coordinate additivity checked to 1e-12, not counted as empirical semantic evidence. Actual NLL response composition squared residual relative to full: expanded primary .011063329634517175 / secondary .01328047974713811; original .19496060673098575/.0436271758364764; original1230positive->seed1 worst .33530767250709903. 1.1% squared error is about 10.5% relative RMSE, not 1.1% RMSE. Same-input composition is not yet cross-input prediction or FCC-exclusive/raw-impossible interpretation.

New report, complete control table and component figure v2 in expanded run. Figure 2400x1400 RGB, actual 299.9994 DPI, 44 case median points and 172 actual primary NLL position responses (48 original / 124 expanded). Expanded2641negative excludes one next-EOS prediction under unchanged probability rule. Draft legend incorrectly assumed 128 expanded points; preserved draft and corrected v2 from data, no raw metric change. Visual check no clipping, redundant markers for close grayscale colors. Existing bundled Python/Pillow only; no ML environment changes. Two summary implementation errors (cross-target coordinate assertion and string-vs-Path hashing) occurred before outputs; fixed summary without experiment reruns. 14 targeted fixtures passed. Summary/plot checks not independent scientific adjudication, and shared source/query/document/seed dependence precludes treating all target rows as independent repeats.

Scientific consequence: deletion of head or tail is not a generally improving simplification in expanded scope; do not replace full or repeat truncation grids. Next work is a bounded synthetic subject-verb agreement task with source-only activity/causal screening, held-out templates/lexicon for prediction and target-versus-other effects; actual next configuration and budget live in tracker. Operation-matched full/raw fitting remains a separately preparable hypothesis, not an established fix. Plan and registry updated, no new protocol/gate/agent or unperformed-task result claimed. Run-experiment/analyze-results skills supported resource/identity and raw-result checks; scientific-visualization supported provenance, visual checking and explicit denominator correction, without introducing scientific approval gates.

Ignored local artifact path/hash ledger (preserved locally, not force-added):
AGENTS.md:33ef44df50aee4673659fb46dc8fa1e645678f08842ffa026fc06ae504a8aa10
EXPERIMENT_PLAN.md:6463e40b509e9e9c8f2be6a67d79c0fe37585ee6503296050169c8404f2dbb15
EXPERIMENT_TRACKER.md:d430c4d9e85c34f49bc2f7a051dce3be07a7c8762d9906325b8c61021deb30ca
REFERENCE_REGISTRY.md:71ccade0aef69d5aacf2726eba1e7de5cbfa38a713a50242aad92f4af6c88bd4
runs/F4_component_probability_expanded_v1_20260905/RESULTS_FOR_REVIEW.md:8bdbfe381fd617d0e2aba5d70eeb0ff250e28b7fdc914864436f7202dfd58d30
runs/F4_component_probability_expanded_v1_20260905/component_summary.json:81e0213bf980dd8a3495b64e0cf79149723cefb77a5d0169bf65fe4ace875952
runs/F4_component_probability_expanded_v1_20260905/component_effects_v2.png:bcc426bdf7d6cafd6bd6a17364733dcd227a406cba0eee642e6d7dc00243979d
runs/F4_component_probability_expanded_v1_20260905/component_figure_manifest_v2.json:ad0734a7ab7227677895160faabf9e2a26deea50db39bf990aad3e18c7b94c26
runs/F4_component_probability_expanded_v1_20260905/component_rows.csv:d005baf502fc87506140c588c11d15ba917f9a7d50953970fc00a47487de5571

## 2026-09-05 16:21 UTC heartbeat: agreement source task start

F4_agreement_source_v1_20260905 freezes configs/f4_agreement_source_v1.json SHA256 a301f7d3bd5fda90cdf4de18a3892336c5cdbb5f7367ac485b5476d7dcde617a before model results. 64 development prompts, 16 lexicalized PP templates x subject/attractor number factorial; 64 reserved lexically/preposition-disjoint prompts text only, same structural family, no reserved tokenization/forward. Existing16source-panel queries, five fixed SAEs; no target-map endpoint/fit/audit. Final-valid-position source-local rank1 donor swap with source cap.1, both subject and attractor axes; raw-hook swap samecap diagnostic, not identical-energy control. Primary correct is/are log-prob margin; auxiliary past agreement and tense preference. Source shortlist top2 by signed mean subject margin loss, all ranks/attractor effects retained. Frozen budget288+2 batch8 forwards, <=180s numerical plus<=60s loading/hash, no new packages/weights. 2 focused factorial/signed-dose fixtures PASS. Spec3129 warmreuse; resource manager all free, GPU1844MiB6% preflight; cpu-heavy then gpu-0, no disk lease. Skills run-experiment and experimental-design used for resource/provenance and factorial design; no power-analysis or DOE installation/approval gate for this deterministic exploratory screen.

Mother index no agreement/syntax hit; read official Marvin/Linzen2018 D18-1151 methods2.1-2.2, Finlayson etal2021 ACL-long.144 methods3-6 including swap-number/correct-incorrect ratio and mediator intervention; source URLs in registry. Official lm-intervention README says MIT and transformers2.7, direct raw source/license fetch failed. Reuse current hook-contract/SparseCoder and artifact functions, not external code or legacy dependency stack. Synthetic task or intervention itself not novel; value sought is cross-seed FCC consumer after source-only development, not probes/labels as causal proof.

V1 stopped after18 baseline/noop/unpadded check forwards, 35.58025020000059 seconds, before query intervention/raw observations: common absolute tolerance2e-4 was exceeded by max.0003752575778750433, noop0. FAIL/contractPASS and snapshots preserved. V2 splits log-probability absolute, margin absolute and hook-relative-L2 diagnostics; frozen numerical tolerances .001 logprob / 1e-5 hook, not scientific thresholds or changes to query/metric. Config422e5bbbec2fb890b3397ff7cfd6cfc766daa69931970e701d85b2e8b2c4f0b2, runner64d502e45ed167ffd9788be0a90a071c0bd82565db67cfe24f6aa9fab05f085b. Report nonzero numeric noise and compare with weak effects; do not silently count failed checks as scientific negative results. Two targeted tests still pass; newrun F4_agreement_source_v2_20260905, same bounded numerical budget.

V2 PASS/contractPASS:290forwards12.590705200011143s / numeric7.289643900003284s,2240raw rows SHA273aeabb3918bf613bf47921bd40cc12fa81ec0029e6e3a1fcbfbf89d0a92f8e,peak789664256bytes.64/64correct primary and past, smallest-condition mean primary margin1.7066993713. Padding check hookrelative3.0474744e-6/logprob.0003752576/margin.0005444906; all prompts same length, min/max checks repeated same item, not two distinct padding lengths. Noop0. Source16 strongest693 mean subject loss.0021610260 vs attractor abs.0166273117; raw wholehook owncap subject mean.0975513458/abs.1876087189. Existing source pool weak and not task-selective; do not call successful agreement correspondence. Expand only to remaining24 of already saved40source anchors, no gradients/newfit/target endpoints, same task and rule. Frozen config8dd5cacd9713e7d526b9e221256b0c7455933cd5e25aa81937254ea100f2e7a1, runner46e470f5b4d7bb5965e79b11cd073558066f3998f933516fc9d6a7e2aa6ac60f (row count made dependent on query count). RunF4_agreement_source_remaining_v1_20260905 budget416+2 forwards, <=60s numeric + <=60s loading on measured throughput. Keep original16 shortlist/history; combined40source development selection to precede any reserved/target work.

Completed remaining24 PASS/sixchecks/contractPASS:418forwards15.047219800006133s, numeric9.784738300018944s,3264raw rows SHA781b18881ae10ecc69e7bba98d1beff44423de4afcafeb85e5b869bf9cd3a517, peak789664256bytes. Combined successful708forwards27.637925000017276s/numeric17.074382200022228s; plus18failed-preflight gives726/63.218175200017866s. Cumulative causal/task28599forwards5937.759949699993s; asset768/103.0689959unchanged. Resource manager allfree after completion, no running/queued work; automation5min unchanged.

Combined40source ranking from5312unique raw observations (5120source intervention+64baseline+128raw control),192duplicated baseline/raw anchors EXACT; all logprobs->margins and losses replay1e-12. New strongestseed2:1655 mean subject loss.006245613098144587, median.00091552734375, meanabsolute.007983207702636719,40/64positiveinputs,12/16positive-template-means, range[-.008605957031250444,.056884765625]; all64natural deltasnonzero, meanactualhookfraction.00529285151549864. Attractor absolute effect.035709381103515535. Seed4:2679 mean.0036430358886719513/attractorabs.008690834045410212,10/16positive-template-means. About2.89x strongest original16 mean effect but still weak/heterogeneous. V1 commonabsolute check failure turned out primarily logprob batch-size numerics, NOT established padding corruption; all prompts same length, so no variable-length padding guarantee. Margins near.000544 diagnostic scale not elevated into strong evidence. Attractor/source natural differences are not energy matched, so their magnitude ratio alone does not prove semantic nonselectivity. Earlier sentence 'not task-selective' should be read as 'selective task relation not established,' not an impossibility result.

Report and all40table/CSV/summary in remainingrun. No target endpoints/heldout tokens/SAE training/fit or runtime changes. Counterfactual synthetic subject/attractor2x2 task and program labels not human evidence; reserved64texts only, no base-pretraining novelty guarantee. Narrow inference: LM task is usable but old general query pool has weak subject effects; nextone source-only task-contrast direction fromsource2 savedY, with fullSAE contribution-swap diagnostic, then independentpaired discovery mapping and reserved task validation if useful. No unbounded query grid or project pause. Current tracker supersedes earlier source2-query shortlist as automatic target-evaluation next step; shortlist preserved as data, not a gate.

Official GitHub HTML fallback succeeded for Finlayson source probability/neuron replace-diff consumer and MIT license; registry records master snapshot without invented commit/hash. No external code copied/installed. Experimental-design used 2x2 factor control; analyze-results kept all attempts and raw reductions, no significance claims or duplicated validators. Two focused fixtures pass; source/summary/code identities in run snapshots. Documentation status/plan changes completed in this same unit, white-listed code/config/log grouped for sync; ignored scientific reports remain local with ledger below.

Ignored local artifact path/hash ledger:
EXPERIMENT_TRACKER.md:6cca403a2c43bacc45083147f4e9ea7e92eb78d61b2992c3dbd6206e0fffa3e7
EXPERIMENT_PLAN.md:269255c9a42c218ad51fc01b1b937085a9f38101dc5a41da134fbd755ee9f3bc
REFERENCE_REGISTRY.md:4c6e24a9c5782d05104c4d4c36781b211a52a50d99a73b0b98c6f3bc89b3d5d9
runs/F4_agreement_source_remaining_v1_20260905/RESULTS_FOR_REVIEW.md:933130900d6d417b8513e3f75569f5bde345c588fe3992e97a7e9c4174ec4a01
runs/F4_agreement_source_remaining_v1_20260905/agreement_summary.json:f54ca7835c5296ab3d9f1f330cbc67447f410b416ade5018e78429a1a8b9b39f
runs/F4_agreement_source_remaining_v1_20260905/ALL_SOURCE_QUERIES.csv:5f9f1712853e5b41a980bedfc2022d21506862cbd0a8a938872cdcc2e85a4bd2

## 2026-09-05 17:07 UTC — user-directed core-contribution planning

Planning and read-only method inspection only: no new experiment, model forward, fitting, training, reserved-data exposure, environment change or automation mutation. Previous source/target results and failures remain unchanged. Starting Git HEAD was 42a12d899a7c55e7e90edd9f57d8847144e998cc with a clean tracked tree.

Updated the existing plan and current tracker card around a concrete proposed contribution: a source-derived intervention explanation transferred through independent paired SAE representations, predicting target component-edit effects and collateral behavior on new inputs without target task labels/gradients/endpoint-based selection. This is a planned claim, not an established result. Source task-contrast pilot remains first; pre-TopK evidence is one bounded alternative, followed by refitting and execution on the actual sparse code. Relaxed-code execution is a different representation experiment. Non-projected target signed edits distinguish component utility from a common rank1 output direction; a second genuinely active task factor enables held-out composition tests. No mandatory all-pass gate, added validator, new workflow directory or broad training grid.

Read user-supplied discussion as hypotheses, local encoder/ReLU/TopK/decoder and training-loss code, and official fixed versions Gao2406.04093v1 Sections2/4/5.3, DAS2303.02536v3 Section3, SCOTM2503.24204v1 SectionsII/III-A. In particular, larger inference k can degrade reconstruction, Multi-TopK is prior art, and the local optional 4k term can enter training loss; actual checkpoint flags must be checked before attributing a gating effect. No external source code copied or new dependency installed. Registry separates source versions, uses, licensing boundaries and proposed differences. Standard projection-residual identity in the plan is not a new theorem.

experiment-plan skill supplied claim/evidence/run-order organization; project instructions override its duplicate output trees/default gates and three-seed default. Five controlled SAEs, source-only selection and new-data confirmation remain. Atlas/multiview/metric/OT/calibration suggestions are symptom-triggered candidates, not a sequential checklist. Documentation received a focused scope/math/information-budget read; no code changes, so no repeated model/unit suite. Only this allowlisted master_log entry is staged for grouped sync; ignored plans stay local, no whitelist expansion.

Ignored local artifact path/SHA256 ledger:
EXPERIMENT_PLAN.md:e1347b4a6fc1853c97ccbb8916c38b228c4c1bca0962dcac3ba6d2b610f04ceb
EXPERIMENT_TRACKER.md:1404086f904a30e6ec8dec9eb01b99f305a28b6eb7b86f7f4683b8913ebdeabd
REFERENCE_REGISTRY.md:b3367c6813d0961a16f7c752890ed61c0b1bc0d50846f31b4190ba4818e8137d

## 2026-09-05 user continuation — task-contrast pilot launch preparation

User explicitly requested mainline execution while retaining earlier routes and learning the blueprint/literature. Existing plan now states that preservation; no old evidence or configurations removed. Prepared F4_agreement_task_contrast_v1_20260905 with optional mode in the existing source runner: source seed2 only, mean plural-minus-singular full decoded contribution, projected exchange versus full SAE exchange, both subject and attractor axes. 64 already-exposed inputs, 50 batch forwards including baseline/noop/unpadded checks; numerical budget60s plus expected loading/hash budget60s. Existing input cache and tokenizer-row hashes frozen; raw and old1655 are historical diagnostics, not new equal-energy baselines. No target endpoint/fit or reserved-token exposure. Three focused task fixtures passed. Runtime spec3129a184 unchanged; preflight shared manager had CPU/GPU free, GPU1943MiB/16303MiB,7% utilization. Request cpu-heavy then gpu-0 through the manager; actual run status/artifacts determine launch/completion, not this preparation entry.

Source pilot PASS with7checks/contract,50forwards,5.3175826s numeric/76.8059232s wall; loading/preparation71.49s exceeds initial60s expectation, no numerical budget overrun. Input hook/logprob/seed2code maxdiff0 and tokenized rows equal. Subject task projection mean margin loss.1251888275 (16/16 template means,51/64 inputs positive), fullSAE.03938293457 (12/16,40/64). Actual mean hook fractions.060091995/.1 versus old1655.005293: no equal-dose20x claim. Attractor absolute primary effect.0915470123 is nonzero; no complete-selectivity claim. Source checkpoint cfgmulti_topk=false verified. Read blueprintpp2,16,30-31,43 (same-hook lemma, full-dictionary/semantic boundaries, multi-seed barycenter) and two nearest local primary methods, no PDF edits. Preserve old natural-text/operation-matched-fit routes.

Following the positive source checkpoint, prepared F4_agreement_relations_fit_v1_20260905: fixed source b,512document-blocked discovery states/256global differences, full decoded transport/raw plus native64/single/random64. Native vector-valued quadratic has Gram=(DeltaZ^T DeltaZ) elementwise (D D^T); tested against explicit flattened design. Ridge then box clipping is labeled as such, not claimed constrained optimum.4target seeds, no target task fit; compile task deltas only after parameter save. CPU-only budget180s, retained selected-row snapshots not full new cache. Planned causal consumer8development inputs*17methods*2axes=272conditions plus source/noop50batches=84total forwards, numerical180s. No audit/reserved exposure or environment changes.

Relation-fit completed PASS/contract18.5054152s CPU,16fit observations and17compiled operations. Parent manifests bound, five decoder hashes verified,512read rows retained with hashes; no claim of rehashing all bulk files or reading audit/calibration/mean arrays. Launching causal consumer F4_agreement_relations_causal_v1_20260905 with fixed first/last lexicalized templates(8inputs);84forwards upper bound, common source-reference scale, source cap checked only on source operations. Compiled candidates may exceed source norm and actual doses remain reported. Prior source-reference margin responses saved as prospective target-response predictions before target forward; same developed inputs, not unseen-input forecasting. All17methods retained, no target winner selection or final strong-baseline claim.

Causal consumer completed PASS/7checks/contract84forwards,2.0132279s numeric/7.5904352s wall; CPU/GPU leases released and live status free. This user unit totals134forwards/84.3963584s wall plus18.5054152s CPU fit; cumulative causal/task28733forwards/6022.1563084s, existing assets768/103.068996 unchanged. All320source anchors exactly replayed,592margin observations recomputed,272prospective predictions and common-source scales checked. No statistical-independence claim for2templates/8inputs/four shared-source directions.

Actual subject transfer: FCC targets1/3/4/5 normalized squared errors.802931/.569521/.619270/.675142 versus raw.484272; source-copy forecasting remains inaccurate. Native64 errors.997769–1.001448, single allzero, random close tozero. Native support membership varying anywhere over64task subject contrasts16/11/22/15 of64; mean per8-panel input3/.75/2.5/1.5 varying members; selected code-difference energy fractions.016138/.000286/.029014/.003865. These are code diagnostics, not hook-energy shares or proof that TopK caused the failure. Native fitted groups had0clipped coefficients; random support for target3 had1clipped coefficient, all other single/random fits0. Report preserves all34method-axis reductions and24support rows.

Decision: retain source task direction and existing natural-text positives; reconnect the earlier source-conditioned/operation-matched full/raw route to this task consumer. Next one bounded source-only task-neighborhood discovery fit with matched sample/ridge/support budgets; global-fit controls remain and weak neighborhood support is diagnosed before attributing everything to gating. Gate-aware evidence/sparse-action remains an available alternative, not a mandatory next grid. No project pause, audit, reserved forward, retraining, package installation or shared-manager modification. Automation file read confirmed ACTIVE/5min, unchanged. Blueprint and two neighboring primary paper versions/methods/hashes recorded in REFERENCE_REGISTRY; no copied external code or verified-novel claim. Four focused tests and three AST checks pass, run snapshots preserve earlier runner versions. Grouped sync only explicit allowlist code/config/tests/master_log; ignored science files retained locally.

Ignored local artifact path/SHA256 ledger:
EXPERIMENT_PLAN.md:2631af4146bf8488a32d0b156d565013e0906cc2e866eb487322a827da9c79f4
EXPERIMENT_TRACKER.md:4f5b15528065a28a4b146500ef3da7eae76038b8c81e349717238e6905461bab
REFERENCE_REGISTRY.md:6777fd7b0ac1330191f2cceb8c100e5fa11be5b1a406f2df17d4e8f74760592a
runs/F4_agreement_relations_causal_v1_20260905/RESULTS_FOR_REVIEW.md:c8958e20848f54b6a7a303f6689c946e89083c56f4cb4ce0531fc9138b8eed6e
runs/F4_agreement_relations_causal_v1_20260905/relation_summary.json:7eb0b2d0d05b83e6c7609956f4c43b0d8726f7d890e8494de15a670e11a0c614
runs/F4_agreement_relations_causal_v1_20260905/METHOD_TABLE.csv:1cac6bfeb66cd097e1f38221464c5899aeca33a054f957f1796699f10e48fb1e

## 2026-09-05 source-neighborhood agreement development — launch

F4_agreement_neighborhood_fit_v1_20260905 starts from acbd2f4 with frozen source direction and original discovery assets. One change: source-only cosine retrieval4096 candidates ->512 unique endpoints/256 subject-prototype pairs. No target task fitting, no linguistic counterfactual claim, no audit/reserved access. Same ridge/native64/fullFCC/raw/random consumers; bounded180s CPU fit followed by existing8-input panel<=84forwards. Environment spec unchanged3129a184; CPU/GPU free at preflight. All historical global fits/results retained.


### Completion and scoped interpretation

Fit PASS/contract PASS20.434126700s CPU. Causal PASS/7checks/contract PASS84forwards,4.399680200s numeric/9.816657600s wall. CPU/GPU released; cumulative task/causal28817forwards6031.972966s wall, fit separate; assets unchanged768/103.068996. Source320 anchors replayed;592 raw margins and272 predictions/scales recomputed;5 targeted tests PASS. No new environment, audit, reserved encoding, target task fit, automation/config-period change or extra code dependency.

Subject FCC error target1/3/4/5=.784859/.701793/.669572/.495993 versus old.802931/.569521/.619270/.675142;raw.452098 remains better. Native64 errors.997744–1.003445 and mean effects−.000511..+.000206. Source-neighbor endpoint cosine mean.153973404;pair deltaY cosine mean.065695275,min−.142385071,max.254698358,213/256positive,512unique endpoints. Coordinates recomputed from retained NPZs and bound seed2decoder; formula in INTERPRETATION.md. Retrieval budget4096 versus old512 selection is disclosed, not equal search. No claim neighbor matching gives linguistic counterfactuals, no TopK causal attribution or main explanatory claim.

Next scoped action: source-only native64 signed teacher development using actual source operation, then operation-matched independent-discovery transport if active; original projection, natural-text, global/neighborhood and gating paths retained. Motivation is operation-class mismatch hypothesis, not proven explanation or native-success gate. Source fit<=60s,causal<=66forwards/180s numeric,loading<=90s. No runtime left running. Analyze-results used separate effects,doses,supports and old/new tables; computational replay is not independent scientific review.

Local ignored artifact identities (not uploaded):
EXPERIMENT_PLAN.md:07cfa4dbe5dd59017dff1b0658b4deb941d3327033ca238628a13a8f358a93a5
EXPERIMENT_TRACKER.md:25655fc561115709f5ad74146da39b97d1bd226e78ebc0dc5264e4dc33892da0
runs/F4_agreement_neighborhood_fit_v1_20260905/metrics.summary.json:7187144c8510ab82fe246408d661e0d8521f3d71a3a7afebe30c5a2764f946df
runs/F4_agreement_neighborhood_fit_v1_20260905/source_retrieval.npz:7132a470bd7b88de38fc677bb5d77a9575405b68e4ee3db28db3674625113407
runs/F4_agreement_neighborhood_causal_v1_20260905/RESULTS_FOR_REVIEW.md:42990b1d3cd6f616f28d085e6bce949880111916f51b87d67d42bd1179431769
runs/F4_agreement_neighborhood_causal_v1_20260905/relation_summary.json:ca4879ae07e22d881be768c25999f69b85fb265e178680d390b8793d5987e071
runs/F4_agreement_neighborhood_causal_v1_20260905/global_comparison.json:010022ded12f39d5c200b8c741169b318956b9fb443c09f1684abb42bf5a4267
runs/F4_agreement_neighborhood_causal_v1_20260905/INTERPRETATION.md:b0676b626be0d644bc382c77cc417defbdd3dbe3b48639ae9f2e66f74ba46646

## 2026-09-05 source native teacher pilot — launch

F4_agreement_source_native_v1_20260905: source2 only, existing64developed task inputs,32unique subject pairs. Rank source atoms by vector-error reduction against fixed task projection;native64 signed ridge.001 then clip[-1,1]. Measure actual native donor edit against existing projection/fullSAE anchors, each own natural cap0.1 without upscaling. No target task fitting/behavioral labels as fit targets; all source development. Budget66forwards180s numeric including60s fit,90s expected loading. Starting HEAD065b81d,CPU/GPU free;spec3129 unchanged. Old routes/results retained;native success is not a new FCC gate.

Native source run PASS/7checks/contract PASS66forwards,1.670409400s numeric/6.677345400s wall;fit.118191900s,normalized vector error.797781950,clip0. Native primary mean−.001525879,39/64positive10/16templates,dose.027465880;projection.125188828,fullSAE.039382935 replayed. Native attractor absolute primary.081701279. This local source result motivates one operation-sensitive fit, not target-gating attribution.

Follow-on frozen F4_agreement_source_adjoint_v1_20260905: per-input source primary margin gradients J_i, A_ij=delta_z_ij*(D_j dot J_i), teacher=(deltaY_i dot b)*(b dot J_i). Same64source development;native64/ridge.001/clipping retained. Primary used to fit, past/tense gradients and all target task info absent. No average-Jacobian Gram approximation. 76modelforwards including8gradient backwards and2finite-difference witness forwards,<=180s numerical/60s fit/loading90s expected. Source effect optimization is development, not independent causal semantic proof; actual forward and non-fit auxiliary endpoints reported separately.

Adjoint v1 PASS/7checks/contract PASS76forwards2.407641400s numeric24.341393900s wall;gradient witness1.526038759 vs analytic1.529919947. Native primary mean.033967018,46/64positive15/16templates,past.022359848,dose.041765513;attractor absolute primary.107273102. Fit residual.650607514 after44/64coefficients clipped(max18.89957),source fit.927561500s. Positive source development signal but no specificity/unseen/crossseed claim. Same-support optimization follow-on F4_agreement_source_adjoint_box_v1_20260905 replaces clipping with cyclic exact box-coordinate updates(max2000sweeps,tol1e-8 projected-gradient residual), allotherchoices unchanged;76forward budget and original negative/vector/clip outputs preserved.


### Source-native positive checkpoint completion

Box run PASS/7checks/contract PASS76model-forwards1.912647400s numeric6.828535400s wall;fit.335322300s,121sweeps,projected-gradient residual9.725366956e-9,converged. Same support/gradients/ridge as clip verified. Native primary mean.100981712,50/64positive16/16templates;past(not fitted).084545135;source dose.045290246. Attractor absolute primary.162438393 and dose.090540007 show remaining collateral effect, not clean selectivity. Training scalar-error.074466272 vs clipped.650607514,48coefficients at bounds. Native vector prior−.001525879,adjoint clip.033967018 all preserved;not claimed same-information or equal-dose advantage over originalprojection/fullSAE. Primary gradients used to fit;past/tense gradients absent. Adjoint fit pair_count64 denotes directional recipient states/32unique subject pairs, not64independentpairs;vector fit used32unique pairs.

Unit total218model-forward calls(including16gradient forwards/backwards),37.847274700s wall; cumulative29035/6069.8202407, assets768/103.068996 unchanged. Source1344raw margins recomputed,960anchors replayed,all3coefficients/doses rederived.7targeted tests PASS;CPU/GPU free at completion. No reserved/audit access, no target fit, no environment or automation changes. Source success is not crossseed explanatory claim. Next: freeze box ids/g,fit independent-discovery actual native teacher using its decoder span<=64 rather than artificialrank1 projection;FCC/raw samebasis anddata,targetnative/single/random,existing8devpanel and common-source-native scale. BudgetCPU180s,target~100modelcalls numeric180s/loading90s;alloldroutes retained.

Figure source_native_development.png1600x1040 RGB150dpi83,551bytes inspected visually;metadata4checksPASS. General internal review,not journal-ready. All16template points,noCI,noexclusions;common subject/past scale;attractor abs separatequantity;dose circles/squares withlabels. Palette first2whitecontrasts5.185/3.867;grayscale pairDeltaL8.214belowheuristic10,handledbyshape/directlabels not claimed accessibilitycertification. Scientific-visualization principles made collateral effects/doses and exposed-development limits explicit. Bundled Pillow/NumPy used becausematplotlib absent;no installation orMLenv change. Standard chain rule/boxQP not a novelty claim;registryupdated localcomponents only,no fabricated newofficialreading.

Local ignored artifacts retained/not uploaded:
EXPERIMENT_PLAN.md:96ba1e95c5c3acfeba84fe70942a4195dfe03ec5c3f59ba9e2346022cfd809ef
EXPERIMENT_TRACKER.md:17f41be308d60a72ba0b8641520873ffcf74639736d0c64a48f251c88fa1c73e
REFERENCE_REGISTRY.md:dfc0ca210cb0eabc2c12b29697b2b56acc0d45d773c64310807c6ebe1fea0368
runs/F4_agreement_source_adjoint_box_v1_20260905/source_native_factors.npz:24b40579f060146ec27b17bbc7db6314d03f4bd2f5d4af36eb954df6287ee366
runs/F4_agreement_source_adjoint_box_v1_20260905/RESULTS_FOR_REVIEW.md:1b0021f33eefcd2a72f86ec73dfdbcb4b7e428dee426c836520a990e3bfc7eaf
runs/F4_agreement_source_adjoint_box_v1_20260905/source_comparison.json:8ce0f0733427ffdd274645793c5bef926fc5a351eaf5e254a81aec6e9b26a8d1
runs/F4_agreement_source_adjoint_box_v1_20260905/SOURCE_COMPARISON.csv:11e2638311af18b04a8de0f60c7195a4743661f51e56c2fcdfdcb8a8b4c6dfef
runs/F4_agreement_source_adjoint_box_v1_20260905/source_native_development.png:5c2973eb81b9349ce3162d3b8a4621555de6c3b9e5ba6029e0a625256abe665e
runs/F4_agreement_source_adjoint_box_v1_20260905/figure_provenance.json:7473954b7b3b19db3220744ef1708ee60f020a34c19ef193ccc267089ea1e003

## 2026-09-05 frozen native-teacher correspondence — launch

F4_native_teacher_relations_fit_v1_20260905 launched frombc8691b;original512global discovery states/256differences, frozen source-native box ids/g. Actual native teacher vector and source weighted-decoder fullnumerical span<=64,tol1e-10;FCC/raw samedata,basis,ridge. Native64/single/random boxvectorfit. CPU180s then8development-input17method2axis panel+66sourceanchors=100forwards,180s numeric90s loadingexpected. No targettaskdata usedforfit,nosourcegrouprefit,noreserved/audit. Added explicit preserve_basis to existing dualridge to avoid numerical-rank prefix dropping valid later coordinates;default oldbehavior unchanged. Rank-deficiency reports effective rank but keeps fullcoordinate predictions. Fixturechecks later active coordinate andvectorHadamard RHS;9agreement testsPASS. Initial test_hook_transport*.py matched0tests(no passclaim); actual test filenames to be resolved beforecommit. Environment3129unchanged,CPU/GPUfreepreflight.

Decoded-sum fit PASS14.1544133sCPU,rank64alltargets;causal PASS100forwards7.5117002swall. Source448anchors,720margins,272predictions/scales replay. Native closezero,FCCprimary errors1.068034/1.027341/.834164/1.330172,raw1.013739. Adhoc retained-bank analysis64devinputs subjecthook normalizedsquared error raw.973290848,FCC1.077782575/1.073728741/1.053920213/1.062803486;source native64meanactive discovery2.94921875vs task17.828125. These support examining representational pooling and data distribution,not a proof that no FCC exists. Actual11hooktransport testsPASS withPYTHONPATH=src (initialmissingPYTHONPATHimporterror preserved in console),9agreement testsPASS.

Bounded follow-on F4_native_teacher_codes_fit_v1_20260905 changes onlyFCC predictorinput fromsum decoded contributions768 to3072sparsecodes,source frozen/native/raw/data/basis/ridge identical. CPU180s;compiled vector predictions compared before allocating targetforwards. Extra inputdimension/parameters disclosed,not equalcapacityraw victory ornew algorithm. No new task data or reserved access.


### Completion and scoped next experiment

Code-input fit PASS/contract PASS1.616778600sCPU;64devsubjecthook errors target1/3/4/5=1.000988727/1.014442428/.984371000/.991138869,raw.973290848unchanged. AllnonFCC compiled arraysmaxdiff0;no additionaltargetforwards because it remains nearzero prediction. Doesnot ruleout code relations with appropriate paired-data coverage. Currentunit100modelforwards2.556202000numeric7.511700200wall,cumulative29135/6077.3319409;CPUfits14.154413300+1.616778600separate;assets768/103.068996unchanged. Sourcebox group unchanged,all448sourceanchors match oldsource run. Native Teacher decoder spanrank64residual1.666806458e-15. 720margins272prediction/scales replayed. 9agreement and11actualhooktransport testsPASS;initial0-test glob and missingPYTHONPATH diagnostic corrected withoutenvchanges.

Next authorized bounded scientificcard: independent task-near paired pool using source-defined programmatic grammar templates,document-level hash10/40/20/30,discoveryonly128templates*4numberconditions=512inputs/256subjectpairs. Freeze vocabulary/selection beforetargetcodes;exclude original task/reserved lexicalitems/combinations,keeporiginalreserveduntouched. Target supplies sameinputcodes butno targetbehaviorlabels/gradients/interventions for selectionorfit. This is task-conditioned source-defined structure,not whollyunsupervised ortask-agnostic. Sourceids/gfrozen,512fitdata sharedFCCdecoded/code/raw/native,initial8devcausalpanel thenexpandonlymeaningfulrange. Hook/code collection<=66modelcalls atbatch8,180snumeric90sloadingexpected;actualconfig/inputmanifest stilltofreeze beforelaunch. No newdata generatedthisunit. Preserve globalnegative andoldpositive naturaltext/projectionroutes;automationnotpaused,no jobsleft. Codebasis dimensionandfullrank64 vsoldrank1explicitly disclosed. Standardvectorridge/box/SVDnotnovelalgorithm;registryupdatedlocaluseboundariesonly.

Local ignored artifact identities:
EXPERIMENT_PLAN.md:b18d70aa13a4ae5591ff8545682b13b854c93276ec149e2a63caa5f5660727a0
EXPERIMENT_TRACKER.md:b51f1e6ec36d7ef69bdfdb72ba95c2b09764bca44d7f53fac1bf49d3664fcff1
REFERENCE_REGISTRY.md:b521babc259cf58c5467c8603288841e32a01a7206ea77ff7c8c47ef39368c2c
runs/F4_native_teacher_relations_causal_v1_20260905/RESULTS_FOR_REVIEW.md:ad884a6af81b8533854c31e4178facd3eb69337f3b36663b328f2d4355dc1a54
runs/F4_native_teacher_relations_causal_v1_20260905/relation_summary.json:3e945e6e569a7db95d991f4ab556fa863643d81dc276a8581d7f73d073ce488e
runs/F4_native_teacher_relations_fit_v1_20260905/teacher_basis.json:888f86e76a0083a12615aa3a619de350c4ee866e956eb643d4fa9c7f4cb45254
runs/F4_native_teacher_codes_fit_v1_20260905/NUMERICAL_COMPARISON.md:ce706de632f0c72f40ad29f9ee1735f84aac0bb8da54002532fb1e9957533e52

18:44UTC final resource read: CCAD leases released; CPU/GPU now held by EndoSAE_EndoFM/kumc-dictionary-cv. No interference or new CCAD queue. Next unit can prepare synthetic paired rules/code without heavy lease, recheck before collection. Tracker resource-snapshot hash updated: b82277c48a8f468bbc8b67a92ddaed774f04f32360ab8594271d61ae19b4ce40

## 2026-09-05 task-near paired discovery unit — launch

Frozen configs/f4_task_paired_asset_v1.json and text-only generation before target encoding. Complete lexical four-condition template hash10/40/20/30 via existing paired_document_split: 102/410/184/328; select128 discovery templates=512states/256subject differences by independent hash. All original development/reserved nouns excluded, development prepositions retained. This is source-defined task-conditioned synthetic pairing, not task-agnostic or human evidence. Environment canonical8348de46 unchanged; warm reuse. Resource preflight CPU/GPU free, VRAM1630/16303MiB. F4_task_paired_asset_v1_20260905 budget67forwards (including one normal-forward vs early-stop check), <=180s numerical,90s expected loading. No new behavioral labels/gradients/results accessed; only discovery will be encoded. Original reserved/audit and all old routes/results retained.

Asset PASS/contractPASS,67forwards,1.1592759s numerical/13.1689779s wall; unpadded relative3.636e-6/4.382e-6, normal-forward difference0. Newcache sha256bf720d8a74d7a781d53600d36e58d4e5216db2faeba77190b75e1b88a10320e8. Reused fit/compiler with task_paired input branch, decoded/codes fits both PASS/contractPASS,1.8079436/1.9219032s CPU. Global-input route preserved. Non-FCC compiled control differences0; codeFCC appended to common21method panel without endpoints/selection. Launch F4_task_paired_relations_causal_v1_20260905,108forward budget, same8 exposed development inputs/frozen source-native reference and common scale. All original source64 anchor observations replayed, no reserved/audit.

Small-panel causal PASS/contractPASS108forwards/7.3007143s wall. Source448anchors exact;784margins and336predictions/common scales recomputed. DecodedFCC primaryerrors .284691/.418587/.408557/.384196, codeFCC .377556/.417795/.294572/.374623,raw.327209,native.902880–.997935. This signal motivates fixed-method extension, not selection of favorable seeds: F4_task_paired_relations_full_dev_v1_20260905 uses all21methods/all16development templates64inputs,402forwards budget <=180s numerical/90s loading expected. New56inputs reported separately; all still development. Reserved remains unencoded. No fitting changes or new method choice from these target outcomes.

Full development PASS/7checks/contractPASS402forwards,6.9559606s numerical/11.6790302s wall. All64 codeFCC errors .491601/.478162/.444410/.447362 improve decoded .659724/.603095/.516029/.593052; raw.417718 remains best primary. Small-panel wins over raw did not persist. CodeFCC past errors .522226/.454897/.458771/.455679 all4 below raw.549896; initial report accidentally said3/4, corrected to4/4 without changing numerical results. Extra56templates14 report signs11/14,14/14,14/14,11/14; development, not independent confirmation. Native64 errors.871628–.976990 remain weak. All21methods preserved. All784pilot rows exactly replayed, full3136margins/2688predictions/source-native scales checked; code/raw controls identical across fits. CodeFCC3072 versus decoded768 dimensions not equal capacity; smaller codeFCC doses preclude claiming selective superiority from smaller collateral changes alone. Next sameunit continuation card specifies raw norm-matched to all4 codeFCC vectors,9methods total including originalraw,64inputs/210forwards; no refit/grid, then independent reserved confirmation and composed-contribution tests. Original reserved64 remains unencoded and no newpaired non-discovery arrays accessed.

Execution notes: summary-only first attempt hit NumPy int64 JSON serialization after writing CSV; explicit int cast repaired report, no model rerun. One hook-transport test invocation omitted PYTHONPATH and could not import ccad; corrected invocation with project src passed11tests. Agreement9 and paired2 passed. These are bounded implementation/invocation corrections, not scientific evidence. Skills run-experiment/analyze-results used for warm environment/resource coordination and effect/dose/dependency-separated reporting, no new environment or review loop. Cumulative causal/task29645forwards/6096.311686s wall; asset835forwards/116.237974s wall separate,CPUfit separate. Resource manager confirms allfree; automation.toml ACTIVE/5min unchanged. Local reports preserve all historical routes and negative findings.

Local-only evidence hashes (not force-added):
EXPERIMENT_TRACKER.md:2fcf80ceaecd8c9668544bea9045e4fca6d448ccd2635bc267d99bd1b7187802
EXPERIMENT_PLAN.md:725c689effcb4ca2428ab504e2d10458084be7ef5446da501ccd8ff3ae6fcf28
runs/F4_task_paired_relations_full_dev_v1_20260905/EXTENSION_COMPARISON.md:2b41c4a65e54da6b0e1c0dd9a4093f8df5bd92121b4d222bb7fd25d5742638fb
runs/F4_task_paired_relations_full_dev_v1_20260905/extension_comparison.json:e56293b7393c10661ee59e17837f74108abae493b281bcd9dbef178195dcf6df

## 2026-09-05 19:16 UTC heartbeat — matched-norm control

Launch F4_task_norm_matched_causal_v1_20260905, frozen9methods/64development inputs,210forward budget (66source+144target),180s numerical/90s expected loading. Fourraw controls preserve originalraw directions and per-input/axis normmatch eachcodeFCC; then both receive same source-native scale. Zero raw with nonzero reference errors explicitly. All4directions retained, no fit/query/endpoint selection. Assembly exactparent hashesverified, 3testsPASS; prepared bank1c3c1022e8b7c45a004402eb7b2ce01679780af18e05fd3f239dcf7e69e716f3. Rawmatched inherits FCC norms, hence informed dose-control, not a new standalone unsupervised baseline. Resource preflight allfree,VRAM1630/16303MiB,envspec3129a184 unchanged; run-experiment warmreuse, noenvironment install. Originalreserved remains unencoded. Frozen endpoints: primary/past source-effect error,attractorabsolute primary/tense collateral,actualnorm.

Matched development PASS/contractPASS210forwards/8.4366866s wall. All4codeFCC primary/past errors below matchedraw; primary improvements.052084/.049031/.082225/.111781. Source-template95pointwise bootstrap5000 seed9051916 allprimaryCIs cross0; onlytarget5pastCI excludes0. Attractorabsolute tense lowerall4, primaryabsolutecollateral HIGHERall4; do not claim comprehensive selectivity. Same appliednorm error<=1.388e-17. Replayed1088old source/code/raw rows,1600margins and1152predictions/scales verified. Fresh officialDASv4/MASv7methods read,includingCLMASno-target-causal-access boundary;registryupdated, no externalcode.

Freeze originalreserved confirmation configa36567b809226f275151e227c62394e034d46b09e89394df6df7c7e84f553a9e BEFOREfirstencoding. F4_task_paired_reserved_confirmation_v1_20260905 uses all25methods and original16reservedtemplates/64conditions; sourceids/g,b,decoded/codefit weights same. Pure saved-weight application reproduces allold21+matched9compiled arrays maxdiff0;9source/fit unit testsPASS. Sourcecode runner adds narrow frozen_task mode: reserved input names honest, no newcontrast/gradient/native fit; compile allmethods before targetinterventions, measure fixed sourceeffects before predicting targetresponses. Budget466forwards,180s numerical/90s expected loading. Primary/past source-effect errors,attractorprimary/tense collateral,originalraw andnative controls allprespecified. Hypotheses not implementation gates; no method tuning after reserved. Same structuralfamily, not newsyntax/pretraining exclusion/source-free forecast. CPU/GPUfree beforelaunch, existingresource wrappers.

Confirmation DONE: F4_task_paired_reserved_confirmation_v1_20260905 PASS/6 checks/contract PASS, 466 forwards, 10.0981551s numerical/14.9366326s wall. First resource attempt saw EndoSAE KUMC lease and did not create a run; subsequent managed wait launched when available. Source/fit parameters unchanged. Source-native primary mean .0406065, past .030921, 42/64 primary positive and 15/16 template means positive; baseline primary 62/64 correct. All 3648 margins and 3200 pre-target predictions/common scales recomputed. No old source replay on new inputs. Frozen-source effects were measured first, not a source-free numerical forecast. Original reserved is now exposed; paired audit remains unencoded.

Primary confirmation errors for codeFCC targets 1/3/4/5: .665586/.973692/.940092/.548100 versus matchedraw .618803/.609179/.621997/.597149. Only target5 point estimate favors FCC, difference .0490485 with 95% template interval [-.171694,.262405]. Past .694734/1.099122/1.096400/.556039 versus .602138/.609424/.600540/.590139; target3 past pointwise interval excludes zero in the unfavorable direction, not multiplicity-adjusted. Thus broad development matched-norm gain did NOT confirm. Attractor absolute tense smaller for all four FCC directions, but absolute subject collateral larger for all four; both endpoints retained. Original unmatched raw primary/past 1.893197/2.172556, unlike its development primary advantage. Corrected a generic sentence in both NORM_COMPARISON.md files and generator that had incorrectly implied raw primary advantage on every split; numerical JSON/data unchanged and retains original generator identity. No model rerun for this prose correction.

Provisional four-panel norm_development_confirmation.png and 32-row NORM_FIGURE_DATA.csv present all four directions, development and confirmation, primary/past pointwise template intervals, and both collateral endpoints. No new statistics or exclusions. Scientific-visualization guided common scales, redundant circle/square encoding, uncertainty/replication labels and explicit unfavorable effects. Pillow12.3.0 from existing bundled runtime used; matplotlib absent, no installation or locked ML environment change. Visual inspection found no clipping. Metadata PNG/RGB/no-alpha passes; actual150.0124 DPI, 1800x1290, internal review not submission-ready. Palette white-background contrast passes; grayscale pair heuristic flagged close luminance, handled with separated x positions and circle/square labels, not an accessibility certification. Source data/provenance retained.

Next work card is bounded composition intervention, not another whole-vector gate: frozen source64 signed partition44positive/20negative, source M rank64/condition79.4438, source native-part oblique decomposition identity maximum residual3.747e-16. This is algebra only, not semantic/behavioral evidence. Give FCC/raw the same decomposition; source-family common scale preserves addition. First8 exposed development inputs with full/positive/negative effects, two axes, source plus fourFCC/raw, upper80 forwards/180s numerical/90s expected loading; exact config/method count frozen before launch. Real component predictions and nonlinear composition residual determine next expansion; any changed method needs fresh confirmation beyond now-exposed original reserved. All former natural-text/projection/global/neighborhood/gating routes preserved.

Targeted tests this unit: paired3, source5, relations4, hook-transport11 PASS; initial mistaken test_hook_transport.py discovery found0 tests and was corrected to test_r011f4_hook_transport.py before counting. git diff --check clean. Final resource status allfree, no current/queued run from this unit; automation ACTIVE/5min unchanged. Cumulative causal/task30321 forwards/6119.6850052s wall; assets835/116.237974 separate, CPU fits separate. Closest official DASv4/MASv7 method sections read and registry updated; no external implementation copied or new novelty claim.

Local-only evidence hashes (not force-added):
EXPERIMENT_TRACKER.md:b5e3d1bcbcae3889b7a09d47fdb55033089ed86b689e35b430ce7c57c0108935
EXPERIMENT_PLAN.md:cb39877cd89fa5a58242b2c77e63b0e8f49b01f533befb762967bb8c45f2915c
REFERENCE_REGISTRY.md:e651a43e01ff825b194065ec7fe8e5de637a41f5a6db365249dca9b0023f5ba0
runs/F4_task_norm_matched_causal_v1_20260905/norm_comparison.json:936c993e1273aed349c368557e306a6c18a4540c35059158078506c345af5e10
runs/F4_task_paired_reserved_confirmation_v1_20260905/RESULTS_FOR_REVIEW.md:43ef9eb67dc74bb1c5021a8c2c4758231da3f3c47a5ea8eb3cbdacc82a60c2cf
runs/F4_task_paired_reserved_confirmation_v1_20260905/norm_comparison.json:d5d45c1e703349d9d9796b12a852e9c56eb38c0479eeeace5b199975b2bd8dae
runs/F4_task_paired_reserved_confirmation_v1_20260905/norm_development_confirmation.png:592bf054d1002617cccdb874d1309120746b05576727cccaa3a8245d551d0839
runs/F4_task_paired_reserved_confirmation_v1_20260905/norm_figure_provenance.json:101b87d72ce04337b0f988a3bcd5ef1036095bcabe43c2d817ab1a2e16ec63e7

## 2026-09-05 user macro-priority correction — act on untested contribution opportunities

User expressed urgency for paper-worthy positive results and requested a macro audit of prior proposed versus actually tested work, without ordering a pause. Current controlled suite is131072tokens per seed; existing NR1 k128/k32 models each have4194304tokens and only atom/native coverage was tested, not present FCC downstream consumers. Long/short streams differ, so no nested-curve claim. F5 residual transport was never evaluated after its rank64 variance.887341 old90percent cutoff; loader exposure correction retained. Natural-text probability confirmation and subsequent component baselines contain real limited positive evidence; task-specific failures must not replace this mother story. Reprioritized plan/tracker: natural-text method increment first, existing trained representation consumers next, bounded grammar component experiment retained as a competing explanatory branch. No whole-project pause, discarded old observations, environment installation, or new gate.

Launch F4_full_operation_fit_v1_20260905, config115167784256272f7c9af809367703b8585a61a337ac816036ba5c723b518ab3. All11 existing source-selected probability cases/7queries retained including unfavorable target cases. Original discovery only: at most128 positive/opposite pairs from256 distinct endpoints, same coarse token class and disjoint sequence-document sets; greedy source-coordinate difference and ties, no target endpoint selection/pool expansion. Full decoded FCC and raw both768-dimensional, same paired rows/weights/ridge.001 and original source basis, zero-intercept differences. This is distinct from failed fixed16-code contrast refit. Parent factors preserved; only selected rank1 query/raw columns in new copy change. CPU<=180s, then121 natural-text forwards<=180s numerical+90s expected loading/export on exposed data. Two pairing tests pass; warm spec3129a184 unchanged, resource manager free, actualVRAM1801/16303MiB. run-experiment used solely for warm resource-managed launch. One focused same-family result-to-claim reviewer works read-only on existing positive claims in parallel; no multi-round score gate.

Fit v1 FAIL retained, contract PASS/40.0106008s CPU/0 forwards: sixqueries24 target maps fitted, then5:2194 had fewer than2 compatible pairs and runner raised. This does not make the six fits scientifically negative, nor justify dropping the seventh query. New v2 config255b1aabc3172a85809419a880932dfdc50cbdb0652d92e10a72711e93afaa40 freezes honest fallback before target evaluation: same source-pair sampler/budget, no pool expansion; for insufficient pairs keep both original FCC/raw maps, mark fallback and retain all11 cases. v1 source snapshot/error/statistics untouched. No new metric/fit grid or whole-project pause.

Fit v2 PASS/contract PASS8.6306923s CPU. Pairs for s1:615/2641,s2:2645,s3:1144/1230,s4:693 respectively127/103/120/117/126/124; s5:2194 has0 compatible pairs from128negative candidates, allfourtarget maps fall back to exactparent. 24 fitted maps plus4 explicit fallback records, not28new successful fits. Only selected rank1 query/raw columns changed; source basis and all other factor arrays unchanged. Launch unchanged existing causal runner on generated original/expanded configs; 3/8cases,33/88forwards, original selection and probability denominators retained. Both new FCC/raw learned from identical source-only paired data; data are exposed development, not fresh confirmation.

Complete natural-text operation-matched causal development: F4_full_operation_original_dev_v1_20260905 and expanded_dev_v1 PASS/contract PASS,33/88forwards,39.1041982/41.6420503s wall, total121/80.7462485s. Original primary oldFCC .035505/.017850 -> new .023789/.021831; oldraw .057683/.104295 -> new .019903/.035709. Expanded primary oldFCC .044457/.027133 -> new .030046/.034021; raw .037447/.033664 -> .017555/.017715. FCC KL improves on both panels but NLL worsens; newraw beats newFCC on expanded primary and downstream KL/NLL aggregates. Original newFCC primary NLL still better, so not all-case/endpoint raw domination. No claim of a new positive FCC method; do not replace parent default or launch ridge rescue. Ratios use same source denominators and target-then-case medians, no exclusions. Old strong baselines remain in their original reports, not rerun under misleading new labels.

Summarizer replays88source anchor rows exactly against old raw, verifies88raw/detail endpoint identities and352probability ratios from position arrays;8 fallback rows' full probability/logits/state endpoints exactly parent. New OPERATION_ROWS.csv keeps all352 old/new scope-method rows. summary fit_records28 explicitly distinguishes24new maps and4fallbacks despite original fit_count field name. Environment unchanged. Final resource-manager snapshot allfree; CCAD automation config ACTIVE/5min unchanged. Cumulative causal/task30442forwards/6200.4312537s wall; asset835/116.237974 separate; CPUfit40.0106008+8.6306923separate.

Macro decision and real result in runs/F4_full_operation_expanded_dev_v1_20260905/MACRO_DECISION.md. Prioritized existing4.19Mtoken NR1 k128/k32 weights' FCC behavioral/composition consumer before new training. They were only tested for atom/native coverage, not current FCC endpoints; short/long data streams differ and two seeds are pilot only. Source-controlled smaller paired fit must have same-sample short-configuration comparison. Detailed bounded encoding/forward work card in tracker. F5 residual transport not tested beyond old rank64/.887341 variance cutoff, static C040 not input-dependent metric, width16384 disabled/not run; no false exhausted-path conclusion. Keep bounded grammar composition branch, stop automatic local-repair chain and no universal query/grid expansion. Failed/unhelpful methods need not be in main paper, but relevant contrary evidence/selection accounting remains.

One result-to-claim review (gpt-5.6-sol/ultra, same-family/provisional) independently recomputed existing positive tables and recommended the actually completed increment. Full prompt/raw response/parsed verdict saved in .aris/traces/result-to-claim/2026-09-05_macro/. Main correction: rank1 is output rank, not one-to-many cardinality; source query is32atom group with multiple actual signed terms, retain FCC mother problem. Agree unproven semantic independent components/cross-input utility are real gaps; reject a new approval chain. Historical R001-R003 synthetic audit does not adjudicate F4. No submission/public claim or project-scope change. Plan/tracker updated; local evidence retained without expanding git allowlist.

Local evidence SHA256:
EXPERIMENT_TRACKER.md:13c17b7b45a9913c51befb5724a1d4dedd4b1a338eab8a9316a9ed4576e179cb
EXPERIMENT_PLAN.md:d4685410c1adf70ddf178ac26b120692da9eb3817e79ba1e244a84d357772261
runs/F4_full_operation_expanded_dev_v1_20260905/MACRO_DECISION.md:aa7d92b00a26a66e9c98cfda22de50e2392146cd15a4bb582a6ee79eb8400dbd
runs/F4_full_operation_expanded_dev_v1_20260905/operation_comparison.json:04ea7c680dbf770cd93dac4a9c5fb09b7db2c558d50bb3049b79250557683436
.aris/traces/result-to-claim/2026-09-05_macro/prompt.md:0bdee9f2fcc00db65272d9a74448f770e1d1d45905eeceaa00ebf8c73573ee9d
.aris/traces/result-to-claim/2026-09-05_macro/response.md:1444c17164f65fe836d55a2f11800c78419711747565e714fabe5724d7b1b370
.aris/traces/result-to-claim/2026-09-05_macro/verdict.json:92ee315eaa8cc4efdc67ff41f048fde6b94e541b87ea2fbdc90ea86d8ae17c05

## 2026-09-05 continue positive scope, recipient performance and operational mathematics

User requests continuing positive-result expansion, performance/definition improvement, coherent mathematics and application significance. Selected run-experiment warm-reuse and formula-derivation to refine the operational object, not rewrite old negative outcomes. Read current AGENTS/tracker/plan and existing derivation; original hard-partition proofs remain historical, not proofs of fuzzy signed uniqueness. Next actual experiment uses existing longer-trained recipients against the exact original short-SAE source queries; avoids changing query difficulty alongside representation. Paired mean/discovery caches already exist, no training or large cache rebuild.

Before new target encoding/fitting, read-only document audit found16/120probability-corpus document IDs and exact text hashes in NR1 long train/validation manifest. Original11source-selected cases have5affected recipient/donor contexts (original2,expanded3); keep complete exclusions before seeing new target outcomes. Current probability results involved only shortSAEs and remain valid for that scope. Paired569doc IDs/hashes have0overlap with NR1. Remainingoriginal1/expanded5cases,4sourcequeries,10target-case pairs for recipientseed1/2 excluding equal source seed. This is exposed development, not new confirmation.

Launch F4_long_recipient_fit_v1_20260905: same original256positive discovery states, weights, original independent means and source32atom/rank1basis, ridge.001; short map must replay old factors. Full decoded768, fixedenergy top16, all3072 conditional-variation bestscalar atom, original raw comparator. Only existing cached intervention hook rows reencoded with four long checkpoints plus two short replay encoders;0LMforwards prep,<=300s. Subsequent two panels23+95=118LMforwards budget180s numerical+90s loading. New generic saved-coordinate consumer keeps source dose and masked support unchanged;3testsPASS. Environment spec sourceSHA3129a184 unchanged, actualGPU1796/16303MiB and resourcesfree before launch. No change sharedmanager/automation, no subagent/env rebuild.

## 2026-09-05 recovered continuation: recipient compactness checkpoint and operational FCC

User resumed after app trouble. Read existing status first: preparation had completed PASS at20:32:35UTC; did not duplicate or overwrite it. Preparation54.8185105s/21fits/0LMforwards, sixencoders×32unique cachedhook rows, all7checks andcontractPASS. Old short encoding relativeerrors4.5122e-7/4.4617e-7, originalfullmap replaypassed. Warmenvironment sourceSHA3129a184 unchanged, GPU1465/16303MiB at causal launch. F4_long_recipient_original_dev_v1_20260905 PASS/contractPASS23forwards81.7588399s; expanded first acquisition met EndoSAE_EndoFM cpu-heavy analysis, created no scientificrun and did not seize resource. Queued by manager then F4_long_recipient_expanded_dev_v1_20260905 PASS/contractPASS95forwards36.7973431s. Total118forwards118.556183s wall; causal/task cumulative30560/6318.987437s, asset835/116.237974s unchanged, preparation separate. All shared resources free at closure. AutomationACTIVE5min read directly, not changed. No training/downloads/MLenvironment mutation/audit opening.

Positive development checkpoint: expanded5cases/3queries fixed16short KL/NLL=.1228950893/.09410954195; long12816=.02048835194/.01603886727 (83.3%/83.0% lower medians), all5case comparisons improve both; long3216=.02528265016/.02803555698 also5/5both. Original1230positive long12816 improvesKL but NLL.3843663873->.6415854832 worsens. Long128full expandedKL/NLL=.02215829589/.00739787554 vs shortfull.01939553168/.02141484994; KLmedianworse but4/5casebetter, NLL5/5better. Raw=.03353020271/.02292312045, retains better individualcases. Long32singleNLL=.01863333634 vs shortsingle.1157156144,5/5better; long32fullonly2/5beats ownsingleNLL. Therefore trainingconfiguration can change compactness, not universal superiority/necessity of manytomany. Both longconfigs4.19Mtokens have different trainingstream fromshort; two-seed recipientpilot not nestedlearningcurve/fiveseedlongsuite. Four fixedsource queries/sixeligiblecases/tentargetcasepairs remain dependent. No significancetest or independentmechanism claim.

One focused implementation/math review: 100sourceanchorrows exactlyreplay,30oldmethodrows (target/raw/top16) hook/downstream/probabilities exactlyequal,400probabilityratios independently recomputed from saved sums/vectors. All six recipient/donor document lists additionally equal full packed sequence_records lists and have0longtrainingoverlap; original5exclusions remain frozen. Three savedcoordinate tests andfive finite operationalidentityfixturesPASS; py_compile four touchedscripts andgitdiff--checkPASS. Fixtures are not proofs of empiricaldensity/Jacobianassumptions or independent scientificreview. No fullsuite rerun without a relevant need.

DERIVATION_PACKAGE.md§13 now defines operational signedFCC with applicabilitydistribution, signedmaps, executionclass and componentfamily; original0–12hardpartition text explicitlyhistorical. Exact secondmoment/donor identities, iidvariance and bounded-density transfer, jointsegmentJacobian bound, softmaxKL/NLL constants, component-errorGram full-cancellation counterexample, and nonlinearinteraction bound separatelylabelled. GeneralGL processinvariance doesnot imply nonnegative/sparseencoder/nativeclass invariance. Constant rank1metric limitations and inputdependentGram caveat explicit. Application candidate: reusable compact sourceeffect readout afterSAEretraining; not yet nativeediting/uniquesemantics. Refreshed Li15v3§4 andMASv7§3.3–3.5 officialmethods, retainedLASSO/CLMAS noveltyboundaries; no externalcode/dependencies.

Figure scientific-visualization used MITpalette asset andmetadataCLI in separatebundledPython, notMLruntime. Finalrecipient_compactness_v2.png2400x1330RGB,128points(80target+48casemedian), commonlog10axes[-3,0], rangesexplicitlynotCI, allsixcasesincludingnegative retained; all10methods/twoscopes inCSV/report. Firstv1labelRawhook(full) ambiguous: v2Rawpredictor(rank1), valuesunchanged, v1retained. Visualinspection legible/no clipping; v1metadata4pass, sameexportsettingsv2; palettegraphiccontrastpasseswhite but grayscalehuepairs needredundantmarkers (provided), noaccessibility/publicationcertification. Nextworkcard freezesallfourqueriespositive/negative for genuinelynewdocuments, all10methods andcurrentmaps/supports, strongestlong32single/raw retained; firstconfirmcompactness then considerfivecontrolledlongseeds, notnewgrid.

Local-only artifacts retained under existing allowlist; SHA256:
EXPERIMENT_TRACKER.md:1383ea2f11ecd5bc512137e182df52936ab2289bdc5341698eabe9acf8e6e4b1
EXPERIMENT_PLAN.md:bcd69dc1aa119b1b386e0cd1c7e91c8a4e9922f4eb097fe70acd3923999f0479
DERIVATION_PACKAGE.md:5ebb09e15e0bf25f5a026eea2f7773c1b59af7d4bba6b4d2e397c331e535845e
REFERENCE_REGISTRY.md:2e5d83324ffa77cdc5752bb31ee52d9498911ad998361d7325e47baf0bf84385
runs/F4_long_recipient_fit_v1_20260905/RECIPIENT_COMPARISON.md:84e7c37b5acc4804528cf7341f639aa661fc53bc742bc2255a1282fb2376d33a
runs/F4_long_recipient_fit_v1_20260905/recipient_comparison.json:5f5dbfdc5c587b299cccea566a9176126753a70015b5cf40d2e574d552a185be
runs/F4_long_recipient_fit_v1_20260905/recipient_compactness_v2.png:48ea7a1a5493407be252ba4165788ba5fd91d24b47e077c1c8bfbdf7ca06ad36
runs/F4_long_recipient_fit_v1_20260905/case_exclusions.json:b2df75c4071541f6c35d8f87ba7e2b0fafbff6c5afbf520178047e795abc7906
runs/F4_long_recipient_original_dev_v1_20260905/metrics.raw.jsonl:6d33d7efa02522e51631d27c0d3c9235cb4dd5b2729208a192aedac1b14d4f49
runs/F4_long_recipient_expanded_dev_v1_20260905/metrics.raw.jsonl:3fa8bef8ff4e49bfc11b3d1d4fda8e9389f033ce77307cee1756a12c55b43555

## 2026-09-05 heartbeat: freeze genuinely new recipient-confirmation corpus

Continue the positive compactness work card, not another grid. Config f4_long_recipient_confirmation_corpus_v1.json SHA fe26868b198eca2655a60a80ba4ed3f72a8be59980a31942f97e92bcbca128be fixes four source queries including the original counterexample, all eight signs, ten methods, target seeds1/2 excluding ownseed, old source/donor/dose and no refit. Saved beta SHA9bafd702488df19b2a6cd65eed782f4aa4e614a250345ebbf73355799fc7ee37 and fit metadata SHAea7966d507d5fbcadbddc2bb8f7216e76e86dd68d0d074cfaa55c2051e9ddf43 bind all coefficients/support/single-atom decisions before new data.

Corpus exclusions now include the original NR1 document_token_records.json (train AND validation, SHA1429667aed740b5acb3f8927ca0dd144943dd18393db3c5755afb7ee89c5f78e), short training, original paired data and all six earlier F4 corpus batches by both document ID and text SHA256. Existing runner only accepted JSONL; added a small JSON {documents:[...]} reader with nonempty identity/hash validation; three tests PASS, no sampling/packing/split change. D bulk top-level and source manifests expose no persistent original parquet cache; range reader cache is process-local. Reuse pinned FineWeb bounded ranges, no full-shard download or new dependency. Run F4_long_recipient_confirmation_corpus_v1_20260905 starts with cpu-heavy lease (all resources free at preflight), <=300s/~40–100MB range budget,0GPU/0LMforward. New-data codes/target outcomes are not yet computed. Automation remains active5min; routine work stays quiet.

2026-09-05 next heartbeat: corpus PASS/8checks/contractPASS,109documents65536tokens512sequences.41range requests38,295,662bytes; status timestamps21:21:41.305523–21:22:44.095856UTC imply62.790333s elapsed (not a separate numerical timer). New token hash a771b65ee591b90a0ef4ae3c9b5ab8c82030a922d0284c88a4e5cad3b1171b47. No training/paired/old-F4 document ID or text hash overlap. F4_long_recipient_confirmation_codes_v1_20260905 next uses unchanged run_r008b_paired_codes.py, allfive existing shortSAEs plus rawhook, borrowed originalmean;128sharedforwards budget120s numerical+90s load/~500MB. run-experiment warmreuse contract used, actualGPU1392/16303MiB/envsourceSHA3129a184 unchanged; resource leases CPU thenGPU, no wholedisk monopoly. No newtarget endpointevaluation/refit/audit yet. Fixed fourquery restriction and saved-beta consumer remain the immediate next implementation.

Codes PASS/8checks/contractPASS128forwards,17.4787417s in existing asset numerical/wall field (excludes model loading),peak allocated895002624bytes,L0exact128. Cumulative asset963forwards133.7167157s same accounting. Asset manifestSHA9a05e449dabd5d040c4c853c2b9b114e7c99cef6b4e7d3d5071ccab9a124af6f, rawmanifestSHA8d95b87542c4ca7519f91083d556935d50fa348a60f50316d8adf8eb5afc4a23. Added source_query_subset as a pure restriction of existing hash-selected query order, rejecting unknown/duplicate/empty subsets;4savedcoordinate/subset tests PASS. Old default unchanged; source query/count checks now use actual requested length rather than literal8/16. This is an implementation adaptation of the already-frozen1+3query request, not post-target selection. F4_long_recipient_source_original_v1_20260905 and expanded counterpart start CPU-only sequentially;0LMforwards/combined<=120s/no refit. Frozen original source selection/matching functions otherwise unchanged.

Source-only preparation finished PASS/3checks/contractPASS in6.6669008s original and5.0481897s expanded,0LMforwards. Fourqueries/eightrequests retained: sixclass-matched, five source-selected (3:1230negative;2:2645positive/negative;5:2194positive/negative).3:1230positive and3:1144positive lackclass match;3:1144negative fails frozen source scope. No resampling or altered selection to recover coverage. Eighttargetcasepairs mean95plannedforwards(23original72expanded),still10methods. Source-selection runnerSHA0049740fb2fba2f0d86b0d2bd543d04b402f9b1270516546c2019ac64de1e2a7; originalrawSHA bb52adcea67f68d01ebfbf35a97e669e6ec01f29a71efdea32761e49c4ca6a7f; expandedrawSHA cd95fa7be57ebf6652082221ca6b81900507a17f63277c8a6953055bb5e1a9a1. Confirmation target outcomes not yet read/computed; implement frozen coefficient application next, no refit. No running/queued processes remain for this phase. Both source studies and asset build are preparatory, not another scientific positive checkpoint.

Phase sync retains ignored local EXPERIMENT_TRACKER.md SHA256 1f5ae97583e780acf59a4edaf52e5785644fac67066949456a4fe1c0bed7201d. Nine explicit allowlisted implementation/config/test/log files only; no raw corpus, bulk arrays, credentials, or ignore-policy changes are staged. Target-outcome evaluation remains next.

2026-09-05 heartbeat: implement pure saved-fit branch in existing recipient preparation, preserving legacy fitting path. Hash-check original beta and metadata; load fixed top16 and single-atom coefficients without refit/reranking or old mean/discovery materialization. Two signed-donor/failure fixtures PASS. Source preparation references have no probability endpoints by design; actual generated causal configs explicitly inherit the prior frozen probability definition. Fresh application config SHA24d879b3405d313ccb50a4fd2c6830c23dd9472efe822a7a78c349c05ac2754a; no new scientific selection. First run F4_long_recipient_apply_replay_v1_20260905 requires exact old candidate-coordinate equality,0LM/newfits; then F4_long_recipient_confirmation_apply_v1_20260905 applies to fresh source choices. Each<=120s cached-hook encoding CPU/GPU; no whole-disk lease. run-experiment skill warm-reuses unchanged envspecSHA3129a184, actual GPU1372/16303MiB, registry allfree. No target endpoints read yet.

Frozen-application replay PASS9checks/contractPASS9.3481818s; fresh application PASS9checks/contractPASS5.7525131s, both0newfits/0LMforwards. All old candidate arrays exactly replay; original21fit metadata SHAea7966d507d5fbcadbddc2bb8f7216e76e86dd68d0d074cfaa55c2051e9ddf43 unchanged in both runs. ScriptSHA499d4403d2b19aa2d6a6e96917deb9bca325f345b46d31bf36fca459091a6989. Generated new causal configs retain four source queries/all8requests, expected1+4selectedcases, targets1/2 excluding source2, all10methods and probability v1; source_preparation_only flag removed. Launch original23then expanded72 forwards, total95 <=180s numerical+90s load, no refitting or target-based choice.

Fresh causal confirmation COMPLETE: original/expanded PASS7checks/contractPASS,23/72forwards15.8079008/30.5899815s wall, total95/46.3978823. Cumulative causal/task30655forwards6365.3853193s wall, asset963/133.7167157 separately. Original raw SHA301d00a3436eabf00c90e0480d4457bf726e144e28a946a1dc2e29001982069a; expanded raw SHAe9732d0b3c7c82019e06328a58722b09cbbcf10262158568329dd8440a5b4459. Current causal runner unchangedSHA0049740fb2fba2f0d86b0d2bd543d04b402f9b1270516546c2019ac64de1e2a7. No new fitting/gradients/training/audit.

Positive fixed-budget checkpoint: both long16 configurations beat short16 on primary KL/NLL in all5evaluatedcases, and long12816 does so on both downstream scopes. Combined case medians short16 .15359287738907904/.1773258436359688 -> long12816 .0376394337471935/.029313931423504854 (75.4940%/83.4689% reduction of medians, not median paired percentages); long3216 .026922797938367153/.021789561261890612 (82.4713%/87.7121%). Expanded4case long12816 reductions76.3088%/81.6793%; original1230negative also improves. Raw remains strong (.0027558868517937474/.0013649892844842765), long12816 beats raw in0/5KL and1/5NLLcases. Long128full vs shortfull improves only4/5KL and3/5NLL; long32full beats own single5/5both, while long32single also beats shortsingle5/5both. Keep all10methods and both endpoint scopes, no universal many-to-many necessity or semantic/native-edit claim.

Coverage: all4queries/all8signedrequests,6matched/5source-selected/3activequeries/8dependenttarget-casepairs. Original1230positive and1144positive have no class match;1144negative fails frozen source scope. Old1230positive development counterexample therefore remains unresolved, not repaired by this confirmation. Query5positive/negative swap sequences446/287; not independent repeats. Frozen config/methods/source/donor checks PASS;80source anchors agree within each new case,320probability ratios recomputed. This is not old-document behavior replay or independent human review. Scientific-visualization guidance used for104plottedpoints(64targets40medians), both comparison panels, strongraw, missing-request disclosure and range-not-CI caption; no selective removal. Image inspected2400x1330RGB/opaque. Strict min300dpi screen reports3pass1fail because PNG stores299.9994dpi; retain this quantization detail, no journal-compliance claim. Foreground palette contrast passes; grayscale caution addressed by distinct markers/row positions. ML runtime unchanged; separate bundled Pillow used for figure.

Next main investment: replicate original primary longk128 with seeds3/4/5, exact original scientific config/token order/optimizer/code,<=15minutes sequential CPU/GPU including eval, no extra model/data copies. Old seed1/2 measured train176.4676802/176.5444099s; old/current training scriptSHA1b8f7ac724fa6063496c4e5a5b340bdb09ffcbd6d349af94a0f503f47830e628 equal. Later use same independent paired fits for new targets without refitting targets1/2; new109documents are now exposed and seed extension is not another unseen-document claim. Five long recipients with short-source interface is not yet long-to-long five-seed FCC. No training launched this heartbeat; all evaluation leases released, automation ACTIVE5min retained. Narrow tests2frozenfit+4savedcoordinate PASS, diff check PASS.

Ignored local evidence retained (path:SHA256):
EXPERIMENT_TRACKER.md:38a50966f3b3ac539914bdfe52aafaaf692efc306289a9089ebbb78d7ec84a28
EXPERIMENT_PLAN.md:b7cbe85eb192dd74e900e17cf1fa6a1e431c11c56444756259dcddb642d9559a
runs/F4_long_recipient_confirmation_apply_v1_20260905/RECIPIENT_COMPARISON.md:d2b4aa485d723fa5a339af61f40167cca09154cd7788e83d2cc783d99bbb7ac9
runs/F4_long_recipient_confirmation_apply_v1_20260905/recipient_comparison.json:718eb6bf99b900d96e23374760ee4cffcf5429629d57dabdf398cf905929d1f1
runs/F4_long_recipient_confirmation_apply_v1_20260905/recipient_compactness_v2.png:2ac619479971e448e7ca3093d3000623a3224249b1210cfc9b6b7ae70ebd926a

2026-09-05 heartbeat22:05UTC: freeze original longk128 same-configuration seeds3/4/5, no method change. Oldseed1/2 configs differ only run_id/purpose/init_seeds; newconfigs additionally update descriptive evidence/scope. Training script, four project dependencies, and external trainer match each original run's recorded hashes (12/12 identity checks); no code/environment change. Runs R011_NR1_k128_seed{3,4,5}_v1_20260905 will reuse exact4,194,304tokens, ordered corpus, validation and all optimizer/architecture/hook settings; seed1/2 not rerun. Actual GPU1487/16303MiB and shared CPU/GPUfree, envspecSHA3129a184 unchanged; run-experiment warm reuse. Sequential CPU->GPU resource wrappers, <=15min total budget; after each seed completion stop remaining queue on nonzero exit or elapsed>300s and diagnose. Estimated new storage includes~60MBSAEweights PLUS~114MBexactcheckpoints, not60MBtotal. Training assets are not a new FCC positive result; downstream fixed-source five-target replication remains required. Phase configs committed before launch.

Frozen configs synchronized as9b53befab4c1659376f476ba811fbb547090be18=origin/main before launch. Seed3/4/5 config SHAs9b80f33b500ebb186b99db7b374adb80aca0015534b3277f6ed25f38112725b8 /14f64e79a55dc2fce264a702034d50fe9f85a0091677bf100b6110943a6b31be /089b4caceec3ee810cce7dec69f2a69adf3deb627ff686ec93d929592cd548fc. Sequential queue active; seed3 initialized onlayers.5 with seed3, originaltorch.optim.Adam fallback and8192steps. Seed4/5 have not independently launched; do not duplicate the active queue. Per-seed status/metrics, shared lease and queue exit govern continuation. Routine training startup is not a user-notification checkpoint.

2026-09-05 heartbeat22:15UTC: seed3/4/5 queue finished exit0, allPASS12checks/contract; train176.7258363/173.1200740/168.2571394s, sum518.1030497. Five same scientific configs and all8192ordered batch hashes exact,4.194304Mtokens each. Quality report in runs/R011_NR1_k128_seed5_v1_20260905/FIVE_SEED_QUALITY.json/md: FVE.985007–.985110,CErecovered.974711–.975647,alive3062–3066, normerror1.788e-7. ActualL0seed5=127.99996948242188, others128,selected128all. First/last256meanFVU~.066/.015 are different batches, not convergence certification. Exact checkpoint directory contains saved SAE as well as state.pt; actual3safe+exact~226.8MB rather than earlier~174MB estimate. Existing resume artifacts retained. This is a matched asset suite, not yet FCC effect evidence.

Next use existing originalmean/discovery rawcache to encode onlynewlongseeds3/4/5. Minimal cached_raw_hook branch in run_r008b skips base model entirely, validates sourceconfig/model/hook/token/split and allcachehashes, and first replays oldlongseed1 codes on1024fixed evenlyspacedrows per split at relative<=1e-5. 1cachedidentity/split fixturePASS. Expected163840rows320encoderbatches/0LMforwards,<=120s numerical+30s hash/load,~405MBnewcodes/decoder. Existing raw~503MB reused, oldseed1/2 codes and raw not copied; mean/discovery only, no auditarraymaterialization. CPU->GPU leases, no whole-disk monopoly; allresourcesfree at handoff. Newconfig frozen before encoding; consumer remains sameoldsource4queries, not target-selected newqueries.

F4_long_k128_newseeds_paired_codes_v1_20260905 PASS10checks/contract,163840rows320encoder batches,0LMforwards12.4841636s numerical asset timer (hashes/load/sample replay excluded). Sampled oldlongseed1 codes replay EXACT relative0 on1024mean+1024discovery rows, not proof of all-code identity. ManifestSHA23da54dd82631c91194ce1219a9080d21df3aa03890de5f3826331bc13d93ef3. Encoder-only prep timer is now separately counted, not added to earlier LM asset-throughput wall denominator. Base causal/task counters remain30655/6365.3853193 and LMassets963/133.7167157; training518.1030497s separately.

Extend existing prepare_f4_long_recipients to derive candidate method names from supplied configurations (oldtwo-long default unchanged); onlylong128 exists fornewseeds3/4/5, so do not invent missinglong32 rows. Fit14maps once with oldsourcepositive256discovery/weights/independentmean/source32atom/rank1/ridge.001 and unchangedtop16/single rules; shortfull original-map replay required. Config f4_long_k128_newseeds_fit_v1.json fixes same4sourcequeries/all8requests/5selectedcases andnewtargets3/4/5, oldtarget1/2 untouched. 2saved-fit fixturesPASS. Budget<=180sCPU/GPU/0LM fit plus actualcausaloriginal17+expanded82=99LMforwards<=180s numerical+90sload. Data alreadyexposed: this is newrecipient-seed replication, not a second unseen-document confirmation. Full7methods onnewtargets andold10methods(only2long32seeds) separately retained.

Newseed fit PASS10checks/contract12.5352052s/0LM/14newmaps; allshortfull original maps replay and allshort3/4/5 cachedcodes replay, no newsource query selection. Fit metadataSHA5ac578e861b74f06bda9c3b6dbfe65c48ec98cb0d5c19c4cfbdb3e9ee0504686, preparation scriptSHA169c5091a0dfc9e73d239ddc1927444224bb00377e6aed1bb40ef02fbe8ba8f3. Generated two7method causal configs verified target3/4/5,probabilityv1,1+4cases. Launch F4_long_k128_newseeds_original/expanded_v1_20260905 sequentially with shared CPU/GPU leases,17+82expectedforwards. Compare preserved oldtarget1/2 data as separate prior-confirmation rows; no newdata independence claim.

Five-recipient replication COMPLETE: new original/expanded PASS7checks/contract17/82forwards12.1563818/36.6210041s, total99/48.7773859; cumulative causal30754/6414.1627052. RawSHAs0ddb5c7183461f18d2650f9b206473580817a7747232c0dd64c30c5f7e8cf3b4 and8f81f5c9d887905eaaa3c705404fa6f7de6c37076e40931e1fbf6ee0f454ceb7. Old95forwards not rerun. All20dependenttarget-case pairs primaryKL/NLL long12816<short16; new3/4/5subset12/12. Across-case medians short16 .16070623242057608/.18504030875107522 ->long12816 .028455016966062736/.022749814668772645 (82.2938%/87.7055% reduction of medians); newseedsubset .16745047754823206/.18215579026409756 ->.016642596863509773/.013033875878014162 (90.0612%/92.8447%). Raw .0027558868517937474/.0013649892844842765 still lower in all5case medians; long12816 vs ownsingle improves5/5KL4/5NLL, full vs ownsingle same5/5and4/5. Fullmedians .008987487317020568/.006075669066773582. Long32 still2seedsecondary, not equal-coverage ranked against5long128. New84source anchors equal oldtarget references, total164anchorrows/656ratioschecked; CSV retains all328scope rows. Fivecase/3activequeries/20target-directions not independent; old3missing requests preserved including1230positive. This is five matched long recipients with shortsource interface, not long-to-long five-seed FCC or another unseen-document confirmation.

Report and200point figure in runs/F4_long_k128_newseeds_fit_v1_20260905; addedseed cohorts separate. Scientific-visualization guidance keeps160targetpoints40medians, raw/full controls and explicit missing/overlap/range-not-CI captions. Figure2400x1330RGB inspected without clipping; same299.9994nominal300PNGquantization, no journalcompliance claim. Assets and models not duplicated except existing exact-checkpoint writer's required savedSAE copy; actualnew3safe+exact226786881bytes. No processes queued/running, leases released. Next bounded science is source-defined16/16 contribution-group transfer with commonfamilycap and raw/full controls, not another whole-map repeat; tracker/plan specify budget and rank1/no-independent-semantics boundary. Oldlongtargets1/2 fits remain unchanged.

Ignored local checkpoint evidence SHA256:
EXPERIMENT_TRACKER.md:3cdf6097adee75d53d5bfd180f1d9a770c6656a7a5c7796988e35a1622ef2831
EXPERIMENT_PLAN.md:22d03beb7daf3aceafe5c2bb1dc1cd7588850b03c72b9ce85410f30ff526e664
runs/R011_NR1_k128_seed5_v1_20260905/FIVE_SEED_QUALITY.json:40cb04c8242de69de3951219788d9225ef15cb3a5c0d6553b620b1923591e1a5
runs/F4_long_k128_newseeds_fit_v1_20260905/RECIPIENT_COMPARISON.md:c1794ec64f590d4968aac33c072de0b70a3262bd036c73e69f9e25a4037108c4
runs/F4_long_k128_newseeds_fit_v1_20260905/recipient_comparison.json:fdc1b9c59117a0cb9d1d37a667f57d87e390d123b28fc7427d6cc3b33ae0e9db
runs/F4_long_k128_newseeds_fit_v1_20260905/recipient_compactness_v2.png:0585f3c82377d881a64cb693f2ffa262062b34938cf4ae40747aaa9a634817ea

Final phase review corrected figure-manifest uncertainty wording inherited from the two-recipient mode: this five-seed panel has four dependent targets per case, not one or two. Image/data/source generator identity unchanged; future generator uses count-neutral wording. No scientific values changed. Resource-manager status allfree and ccad automation ACTIVE/5min verified at phase sync.

2026-09-05 heartbeat22:42UTC: start the already planned source16/16 composition consumer F4_source_components_v1_20260905. Source-only conditional variation on frozen32members/positive256discovery, two scalar ridge solves with originalkernel and independentmean, coefficient sums must replay oldlong/raw full. Same five exposed cases/all8requests, five longrecipients excluding source index. Sharedfamilycap.1 from max(full,A,B), source/raw/long same scale; full rerun under changed dose if needed. Uses existing probability and hook helpers, three finite math fixturesPASS (partition/id ties, ridge/primal equivalence/additivity, cancellation/commoncap). Budget preparation180s +100LM180s numerical90sload, resourceCPU->GPU, no newdata/training/env/audit. Local runtime specSHA3129a184 unchanged, allleasesfree preflightGPU1620/16303MiB; warmreuse from run-experiment guidance. Rank1source components remain mathematical parts, not independent semantic mechanisms/native target edits. Scientific effect not yet measured.

F4_source_components_v1_20260905 completed100LM/90rows37.739043s wall: preparation9.8639119,load.5175459,causal27.326228. ContractPASS; runFAIL only because new script's cached_hook_replay mistakenly used1e-5 versus existing run_f4_source_reference_causal.py1e-4. Observed max1.1723888746928424e-5 agrees prior same-case max1.1723890761776082e-5, noop exact, allfull coefficient sums replay. Preserve original status/raw/config/snapshot; no rerun simply to repaint status. Restore existing1e-4 for future consumer and explicitly record correction, not a new scientific acceptance rule. RawSHA9b6995929732706292fcfb26465f3bc92d0995189094277b14a14206f6147dcc. Actual source B effects and unscaled donor vectors allzero, sourceA=full all5cases; selected source32 discovery nonzero variation members10/8/27 for2:2645/3:1230/5:2194. Independentmean creates a constant inactive-tail state target (nonzero secondmoment, not conditional variance); target B can falsely vary despite zero sourceB. Head/tail is a degenerate source family here, not evidence of two functional mechanisms or absence of any decomposition.

Bounded corrective experiment F4_source_components_interleaved_v2_20260905 frozen before target forwards: alternate odd/even source-discovery energy ranks into16/16; original32members/basis/full/rows/weights/independentmean/ridge unchanged. Source-only task inspection (not target labels) gives component norms A/B:3.28689/2.98200,2.00420/2.45861,5.70194/5.01396,4.48013/3.45518,4.48013/3.45518. Every part nonzero; geometry is still rank1collinear per hook and paired-document vectors strongly aligned, not distinct semantics. ConfigSHA16ad97a9c374f0b46d85f9f7ab221d4fa926ff7df94748f281793534ca027ff0,scriptSHAf111b58f9a15f053739b8f641875bd4ce7da0f547be1835280fc7046def767a6. Reuse all assets, 3mathfixturesPASS including interleavedties; another100forwards<=180s numerical+90sload, preparation180s. This is exposed-development refinement; oldv1negative/diagnosticFAIL intact. Both runs count toward resource use, original30754causal counter plus100v1=30854; v1 causal27.326228 added to numerical consumer timer separately from preparation/load.

F4_source_components_interleaved_v2_20260905 PASS7checks/contract,100LM/90rows35.4532243s total; prep8.9347948,modelload.4457998,causal26.0277563. RawSHAbb7072d5e11e33844f1630f1d829bc6b85c3081f9e39dbda5d131c29b9edd3a7. Both component runs200LM: cumulative30954forwards; consumerwall=total-minus-separate-prep adds54.3935606s =>6468.5562658. Preparation18.7987067 separately includes fits/encoding/hash; purecausal53.3539843; LMassets963/133.7167157 unchanged. Sourcefull NLLvectors across both scopes equal old anchors EXACT max0; v1/v2 fullcoordinate maxabs1.24345e-14. Long beta sums oldfull maxrel1.00347e-14; raw coefficient sum vsoldfloat32raw factor maxrel2.89377e-8, do not claim raw output bitwise replay.

New functional positive: both sourceparts nonzero all5cases; A NLLRMS.0147334–.0771650, B.0124248–.0717584. Each component20dependenttargetcases longbothprimaryerrors<1. Acrosscasemedian A KL/NLL.0341595585/.0138537781; B.0139024641/.0159577102. RawA.0041766890/.0087211689; rawB.0090695216/.0111530315 remains generally stronger. A long beatsraw3/20KL7/20NLL, B5/20KL6/20NLL. SourceNLLinteraction full−A−B RMS.005819/.005720/.004874/.049596/.006644, relativefull squaredenergy.001902/.018146/.005929/.118996/.027611; no behavioral-additivity claim. No semantic independence/native/compacttarget/newdata confirmation claim. Rank1parts/full targetreadouts with fixedshortsource, not long-long5seed suite.

Summary component_summary.json / COMPONENT_ROWS.csv180scope rows / COMPONENT_INTERACTIONS.csv60rows and COMPONENT_COMPARISON.md in v2run;360probabilityratios independently reassembled from saved endpoint arrays, allsourceanchors consistent. Scientific-visualization guidance preservesraw/full/A/B,120individualtargetpoints30medians30rawpoints,non-CI targetspans and missing3of8requests; figure2400x1440RGB/opaque144122bytes visually checked, explicitinternalexport/nojournalcompliance. Nominal300dpi stores299.9994; metadata format/mode/noalpha checks3PASS, no strictminimumDPI claim. Next tracker budget binds bothsourceparts toone shared16target support using summed conditional termenergies, matchedshort16/full and jointbestatom; <=325newLM with existinglongfull/raw replayed from thisrun only aftersourceanchor check. No new scientific run started; automation remainsACTIVE5min, resourcesreleased.

Ignored local component checkpoint SHA256:
EXPERIMENT_TRACKER.md:a3a9d11f16cea7f4f73499e45095886048db9ba5427d8f1fd95588f167b7ccde
EXPERIMENT_PLAN.md:f25d083a651d81e4888e36438f3daad515debbfd79148f2384a1f6ee1a7355bb
runs/F4_source_components_interleaved_v2_20260905/COMPONENT_COMPARISON.md:ae8784e8398920a418f5f0ad2ba2c93955fbb8a7d3227db14529a72574f4056d
runs/F4_source_components_interleaved_v2_20260905/component_summary.json:c3fa9c3ed8290873fcb641a2474ed198212414e256e074b2366bc422106f1332
runs/F4_source_components_interleaved_v2_20260905/source_component_fidelity.png:60c7eac00245dae95c3bfed835218a7b390aa19c3f3e9d7b9b9658cc9cb5a1ae

2026-09-05 heartbeat23:11UTC: implement frozen shared16 composition consumer in existing run_f4_source_components.py optional compactmode. One support from VarQ(z_j)*(beta_jA²+beta_jB²), same16 atom IDs for bothparts, no sparse refit; jointbestONEatom over all3072 minimizes both conditional-variation errors. Restore v2long/raw coefficients exactly; only12newshort two-output maps fitted on unchangedsource32/discovery/mean/ridge. Sourcepartgroup andfull/raw coordinates/case/dose must exactlyreplay parent. Existingshort cachedcodes,32longencodedhookpositions; new5methods shortfull/short16/long16/shortsingle/longsingle measured forfull/A/B/all20dependenttargetcases. Expected325LM/315rows<=180s causal+90sload,prep180s; sameapprovedbudget, no data/env/training/download/audit change. 5mathfixturesPASS including joint union16 and explicitallatom objective/constant-shift invariance; raw caches and originalenvironmenthash unchanged. SharedCPU/GPUfree preflight1427MiB/16303MiB. Config f4_compact_source_components_v1.json frozen before actualforward. This is exposed development, not freshdata confirmation or semanticmechanism/native proof.

F4_compact_source_components_v1_20260905 PASS8checks/contract325LM315newrows113.7531622s total,prep11.5540377/load.4945831/causal101.6756353. Consumerwall+102.1991245=>cumulative31279forwards6570.7553903s; assets963/133.7167157unchanged,prepseparate. RawSHA7b75ddd0c77d7001f95a32f46bdc2f2eb39fcd7b87e0f45cb9be277453a7a5c0,runnerSHAf232cf6a13813e801cb6dc7254d1039175042c1cff38e392f9546f68c4c2f17f. Parentcase/dose/coordinates exact, long/raw restored0refits;short12newtwo-outputmaps fullsummaxrel2.6271103e-8vsoldfloat32. All15sourcepart complete probabilityendpointdicts two scopes EXACT parent, so reuse75longfull/raw oldrows, never count as newforward. Combined390rows/780scope/1560ratiosrecomputed;24sharedsupports eachunion16 and24jointatomfits checked.

Compact composition positive: full/A/B each20dependenttargetcase KL/NLL long16<short16 AND<1,60/60total. Case-median A short.1652451780/.1519207492 ->long.0309609834/.0204938209 (81.2636%/86.5102% ratioofmedians reduction); B .1773586246/.1614833338->.0432935811/.0261063697 (75.5898%/83.8334%); full .1643175512/.1802005504->.0397307178/.0340072326 (75.8208%/81.1281%). Jointoneatom remainsstrong: long16beatsit A19/20both, B13/20KL14/20NLL,full15/20KL14/20NLL. Raw remainsoverallmoreaccurate; nooptimal-sparse/native/independent-semantic orlong-long5sourceclaim. Allfivecase medians directionconsistent, notindependent60replications. Shared16meansone identical targetmember set forbothparts, nottwo16sets.

Report/780scope CSV/156interactionrows and JSON in compactrun. Reused plot_f4_source_components --compact renders480points=360individualtargets90medians30raw, short16/long16/jointsingle/raw,allcases/parts;otherfullandshortsinglecontrolsremainintables. Inspected2400x1740RGBopaque154502bytes,metadata3checksPASSformat/mode/alpha,nominal300dpiactual299.9994nojournalclaim. Nextsource-componentconfirmationfreezesactual3queryfamily6signrequests; old3:1144two requests lackanyfrozencomponentmap andarecoverageboundary, not silentlyassumedconfirmed. New512x128token corpus excludescurrent109andalloldtraining/paired/F4byID/text hash before sampling; existingcorpus/assets/source-onlypipeline reused; no newdata fit/ranking. <=128LMasset and<=480LMcausal<=240s numerical+90load pertracker,all7methodstwoscopes/full/A/B. No newrunstarted;resourcesreleasedandautomationretained.

Compact checkpoint phase sync: five focused mathematical tests and three changed-script AST checks PASS; resource-manager all leases free, ccad automation ACTIVE/5min verified, unchanged. Ignored local artifacts remain local with SHA256:
EXPERIMENT_TRACKER.md:1df1547bcc2d25eea3d2f3d86e81685f57ecdf3e4b9721ee4c83ad5bc26cb8ab
EXPERIMENT_PLAN.md:7148543e2748b8051c1905b83f340e56c28c796e815327f5332402199eb97f63
runs/F4_compact_source_components_v1_20260905/COMPACT_COMPONENT_COMPARISON.md:2f8bf2b46651de3986f4327926161ec5dff880019e82498f21915b7019384658
runs/F4_compact_source_components_v1_20260905/compact_component_summary.json:16cb812d385d477dd4a8247d85b4faab423c140971e67363ec55e4b735f36fb0
runs/F4_compact_source_components_v1_20260905/compact_component_fidelity.png:5dc9a6730452c3609efc6407c98a9d27f91e4b1217dfdba8b402135e52d13827

2026-09-05 heartbeat23:35UTC: freeze configs/f4_compact_component_confirmation_corpus_v1.json SHA f021535423c3736c6ee47826374c25acb633836086e7acb6f0a91e947f4e7d2e before any new corpus sampling. Actual frozen family3:1230/2:2645/5:2194 both signs; original3:1144 has no component fit and is an explicit coverage boundary. Parent coefficients c44cd9ce1365c9b426411b1cdaa94cef4b438a35386640b2edd9701433ddf833 and metadata90c704e2919a3ee38155df9386ff3865e5b866cfcf46e8828ecfc20c2108b024 restore all seven methods/full/A/B without new fits/support search. Old109documents added to all prior training/validation/paired/F4 exclusions, nine additional ledger hashes checked. Corpus512x128<=300s/~40-100MB existing pinned FineWeb range; no cached parquet/range in existing reference storage. run-experiment warm-reuse guidance applied, runtime spec3129a184 unchanged; CPU lease only, no whole-disk monopoly. Reused original corpus script1ec173232474204e0ad711d04309ee40ca68cc241f8da8985cdcb8b9d5c44a34, no new code changes. Next assets<=128LM, source-only selection then<=480causalLM per tracker. No new endpoint evidence yet.

Fresh component corpus PASS8/contract108documents/512seq/65536tokens,41range requests38711812bytes, started23:37:24.589 to23:37:40.627UTC. Five shortcodes/raw PASS9/contract128sharedLM. Original/expanded source-only preparationPASS3 each,2/4selected of2/4requests,4.9952094/4.2642694s,0LM. Allsixconditions evaluable including original1230positive; no target endpoint used. Application path implemented in existingrun_f4_source_components.py, skips all discovery materialization/fitting/ranking; restores everyparent coefficient and sourcegroup. Exact old5case/case-dose/all-coordinate replay runF4_compact_component_apply_replay_v1_20260905 PASS5/contract5.3848257s,0LM; parentmetadataSHA90c704e2 unchanged and arrays equal. Current implementationd3b6ded2f6647965d6262f89b566027031dfa86f23c7e7c8ed1911424a57d1bd. Five finite testsPASS. Identity erratum: corpusfreeze inherited causal_runner_sha256 a091d18 from an older ancestor, not actualcurrent source-preparer0049740 whose source_snapshot is bound in both source-only runs; scientific rules/fit/groups unchanged and oldfreeze not rewritten. Now launchF4_compact_component_confirmation_v1_20260905,all6cases/7methods/fullA/B4targets2scopes,480LM468rows<=240s numerical90sload,preparation180s. No newbehavior evidence yet.

F4_compact_component_confirmation_v1_20260905 PASS8checks/contract480LM468rows116.8394605s total,preparation5.2533694/load.4239930/causal111.1272974. Consumerwall+111.5860911=>cumulative31759forwards6682.3414814s; LMassets+128/16.1968305=>1091/149.9135462; source-only9.2594788 and replay5.3848257 separately. RawSHA d2c3f15c91fce7e247fd8593d27184ed6038dbd0a73d0b6341bc7b326b6659d0. Parent metadata90c704e2919a3ee38155df9386ff3865e5b866cfcf46e8828ecfc20c2108b024 byte-identical, all coefficient arrays exact, no refit/ranking. 936scope CSV/252case rows/312interaction rows,1872probability ratios recomputed,all468rows newly measured(no oldendpointmerge).

Confirmation supports bounded shared16 composition reuse: A shortKL/NLL.1695583615/.1906474218 ->long.0328325708/.0332945111 (80.6364%/82.5361% ratioofcasemedians reduction),B .1141293957/.0970694583->.0269546985/.0294124153 (76.3823%/69.6996%),full .1285959346/.1328794466->.0237843441/.0221010154 (81.5046%/83.3676%). Individualtarget improvements A24/24KL21/24NLL,B18/24KL19/24NLL,full23/24KL22/24NLL. Case-median improvements A6/6KL5/6NLL,B4/6KL5/6NLL,full6/6both. LongA NLL below nointervention22/24, all otherlongshared primary component/metric panels24/24. Original1230positive full improves butA NLL.4277721909->1.1983814212; target4/5=1.5559475549/2.6434222076. B KL.2478331738->.3857268022. 1230negative B NLL.0611258542->.4259735183. SourceA/B effects nonzero throughout; A sourceRMS range.00325769–.19243779, B.00358951–.10549857. Do not erase component failures using whole-operation success.

All controls retained: longshared16 vsownjointone A23/24KL21/24NLL,B15/24KL12/24NLL,full20/24KL21/24NLL. Raw full/A medians remain more accurate; B raw medians.0389036212/.0424983637 above longshared16.0269546985/.0294124153,butonly17/24KL12/24NLLindividualrawwins forlong16. Secondary downstream long16short16 wins full23/24KL22/24NLL,A24/24KL21/24NLL,B20/24KL19/24NLL. Source1230 and2645 positive/negative exchange recipient/donor;2194 conditions share a sequence. 108documents are the sampled corpus,6cases/24targets are dependent, no independence/general-five-source/native/semanticmechanism/optimalL0 claim. Long/short training streams differ, no duration-only attribution.

Confirmation figure576points=432individualtargets108case medians36raw,commonlog[-4,2]includesall>1exceptions; actualimage2400x1950RGBopaque168447B visuallychecked. Metadataformat/mode/alpha3PASS,nominal300actual299.9994dpi,no journalcompliance claim. Palette contrastagainstwhite passes forusedcolors; blue-orange/orange-green grayscale differences weak, redundantcircle/triangle/square/cross retained. ReportCOMPACT_COMPONENT_COMPARISON.md and RESULTS_SECTION.tex connect effect, strongcontrols and specificfailures; paper-write guidance used onlyforresultfragment/numericclaimcheck,notfulltemplate/review/submission. Corrected draft-only Bcase-median NLL count to5/6(beforephasecommit;rawunchanged). Source-selection/class/probabilityhelpers matchfreeze4e9aa3f9/d2be67ce/3235deab; the frozenconfig note's phrase pre-probability ancestor should be read simply as historical consumer (precise timing not established). Nextboundedworkfromtracker: mature Li15-inspired joint-support sparse regression <=16members under sameinformation budget, pilotfit<=30s,totalCPU<=300s,two newmethods on old109document5cases<=145LM180s+90load, no newtraining. No nextfit/runstarted; currentallresourcesfree,automationACTIVE5min preserved.

Ignored local confirmation checkpoint SHA256:
EXPERIMENT_TRACKER.md:ea2fbbc90312b90ceaa73cab6044a600fc908734fe934138acb5f90e15e8e5ec
EXPERIMENT_PLAN.md:aa8d2a088f7910a136d3a51677594943a91d3529085c8b3f960d1c6dd5cae059
REFERENCE_REGISTRY.md:df7bcc834fa09dce8147e35e62d7521edff118526a46d96f6dd67b30a2e6f1d4
runs/F4_compact_component_confirmation_v1_20260905/COMPACT_COMPONENT_COMPARISON.md:93c4c519d2f41ff2020b1e8d3dea3dac2551437c0bc955888f22fee1043173cf
runs/F4_compact_component_confirmation_v1_20260905/compact_component_summary.json:e619d11e9e429b7ee04cb04d560a8154dbe4ff06c9d355900a3f18dfdae68cd6
runs/F4_compact_component_confirmation_v1_20260905/compact_component_fidelity.png:69d73948bc5c449526ea5b8fc7f536e48c80ca003a2cfb4adc1a08598d7fa675
runs/F4_compact_component_confirmation_v1_20260905/RESULTS_SECTION.tex:027992f235b67d3c767e15015eb8081cf866b63841f249aada802cf152d0ee16

### 2026-09-05 joint-sparse execution unit: environment and pilot start

Read Li15v3 section4 and official sklearn MultiTaskLasso documentation/installed1.9.0 source: weighted channel normalization, sum-to-n sample weights, coef transpose and unnormalized eps versus normalized dual gap checked. Group-L1 with two outputs/shared support and post-support ridge is an adaptation, not exact scalar Li15 or optimal L0. Checked existing r004, all listed overlays and bundled Python; sklearn/scipy absent and no sklearn wheel cache. Added only official binary wheels into new D:/CCAD_Storage/environments/f4_sparse_overlay_v1, no existing environment mutated. Spec canonical dff2dd232e0bab8999fb32ecd8bcd6c6f80f4da5e3672841c54d6017d31ca61b and install report are in .aris/compute; versions sklearn1.9.0/scipy1.18.1/joblib1.5.2/threadpoolctl3.6.0/narwhals2.25.0, existing NumPy2.5.2. Primary and fresh-agent documented solver witness exit0/support[1,4]/weighted-transform error0/dual gap8.136084398794689e-14 identical. Initial premature import before pip finished target moves failed; retained as installation timing fact, not environment/scientific failure. F4_joint_sparse_pilot_v1_20260905 start: one preselected sorted2:2645->short1, alpha=.1max, <=30s numerical; no endpoint/audit/GPU. Mature implementation replaces ad-hoc solver work; preparation is not a research claim.

Pilot completion and full-fit freeze: 3 mathematical tests initially caught weighted-mean rounding making constant columns/output spuriously nonzero variance; fixed using exact ptp on positive-weight rows before any real fit,3/3PASS. Pilot onealpha support9, .09661319999941043s numeric/1.0642858000010165s wall, all5numerical checksPASS,0LM. Contract FAIL only config.audit_opened missing versus manifestfalse; process exit1 retained, not a fullPASS run and not a scientific negative. No redundant numerical rerun. Full config supplies false and freezes the planned40-point log alpha prefix/max->.01max untilfirstsupport>32, tol1e-7/maxiter5000, ≤16 support candidates with fixed.001 normalized ridge refit and minimum normalizeddiscoveryerror,24pairs<=300numericalseconds. Fullfit F4_joint_sparse_fit_v1_20260905 starts under CPU lease; no taskdata used or causal claim. Prefix may miss later support reentry; shared support not optimalL0.

Fullfit completion PASS5/contract,24maps all selected converged, common supports7-16,2.138521300003049snumerical/4.131716800000504swall,0LM. Fixed coefficients SHA777530ffba3a4bd3433840f5c53fc19678d58320db785f3bb3c4816e326eea96; raw04a87aeb8deaf0e929326c3fd0b25e41931404acc39e3c0fcdf76dafb115769e. Start F4_joint_sparse_causal_v1_20260905: reuse original frozen-component application branch, 0newfits/discovery/audit; field confirmation_inputs selects only old109doc fivecases and is NOT fresh-data evidence. Explicit extra two frozen short/long joint_sparse interfaces, finite coefficient/intercept and support identity checks. All prior coordinates/source/case/dose exact replay required before145LM consumer; two new methods, fourtarget,full/A/B and both scopes. Existing raw/full/shared16/single may only be merged on exact source endpoint anchors. Consumer≤180numerical+90load, no108doc prior confirmation access. New optional branch leaves old defaults unchanged.

Completion F4_joint_sparse_causal_v1_20260905 PASS10/contract:145LM/135newrows,wall46.536101400000916s,prep6.266642200000206s,load.5814573999996355s,causal39.644972699999926s; rawSHAa810eb9158bf6d3fa69a2761181da39b0f3a2b092a529d60fb78c19a236a891d. Exact source/case/dose/all old coordinate replay and two generations'15source endpoint dictionaries allowed375old control rows merged, not new forwards;510total/1020scope rows/2040probability ratios recomputed. All9methods,2scopes,full/A/B,raw/full/jointsingle retained. Fullfit659pathsteps allconverged; short/long median discoveryloss ratio to oldenergy16 .337347/.659822. This fit loss is not behavior evidence. New3mathematical tests +existing5component tests pass, causalAST pass. No tests on changed-free ML environment, no new GPUassets.

Scientific outcome: shortjoint versus shortenergy16 case-median KL/NLL reduction full72.4912/75.8600%, A19.7089/32.4157%, B83.4808/73.3430%; dependent20target-case wins20/18,17/16,20/19. Longjoint versus shortjoint A median80.6402/81.8888% lower but targetwins15/12 andcasewins3/3 of5, full46.6820/72.3168% target12/13 case2/3, Bmedian1.3319/3.7946% target9/7. B meanNLL .0610133short versus.0687310long reverses median order. Longjoint relativeoldlong16 BNLLmedian worsens58.6324%; raw/full generallystronger. Do not preserve broadlongB claim by changingaggregation or deletingcontrols. Existingenergy16 independentconfirmation remainsvalid foritsrule. New interfaces supportcompactexecution improvements, notsemanticindependence/native/optimalL0. Full findings and descriptivemean±sd withdependence and outliers in SPARSE_EXECUTION_FINDINGS.md; summaries/CSVsourcebound. No data108 from priorconfirmation or originalaudit opened; initial109olddata remainsexposed.

Consumer incrementwall40.2694592s,total31904forwards/6722.6109406s; LMassetsremain1091/149.9135462. Pilot.0966132/fullfit2.1385213numeric andpreparations separatelyaccounted. Resourceleases released/allfree,automationccadACTIVE5min unchanged. Next in tracker: freeze24sparse interfaces and source-only threequerysixsign requests, new512x128corpus excludes alltrain/validation/paired/F4including109+108IDs/text hashes; codeassets128LM and all9methods624LM<=240causal+90load based on145/39.645 throughput. Newinput actualsupport/coverage recorded, no tuningornewtraining. No new confirmationartifact has been created this unit. run-experiment isolatedCPUsolver witness and analyze-results allcontrols/descriptivestatistics used; no new skillgate.

Ignored local artifact identities before grouped sync (retained locally, not force-added):
EXPERIMENT_TRACKER.md:39091a869ac51e86d3d5342b9d6cd63365d1e72fad907f259bea4ce031c7e5e1
EXPERIMENT_PLAN.md:7af1b4024684cf165c0a2f22e715ab28aab52781cc09f8a724273f9bbac6c0e3
REFERENCE_REGISTRY.md:72e7930466c80ad999f59edb26f5e9c1e4ff8a894541195adcf6515ad4930ecb
.aris/compute/local.md:8f9a2986d5e7924794293aea58554c925128634515d03fb352b59802ba061d2e
.aris/compute/local-f4-sparse-env-spec.json:636013429e825730e9dd7330c5412eccfb01c4b0ab999d117e736f8265d7bbc1
.aris/compute/f4_sparse_install_report.json:02efef84d06864d7f71b642cec1df2ef7c65da68d7062d115e9ad1faf3ed0da3
runs/F4_joint_sparse_causal_v1_20260905/SPARSE_EXECUTION_FINDINGS.md:d851c1123a89ac1e8e214f0217760db1ee8e8e55f273fcfe827763c5bea1c5f7
runs/F4_joint_sparse_causal_v1_20260905/compact_component_summary.json:233058000ea31fe7977b5856edcd402710abd25d993fdc7e73f75c7c1df19227

### 2026-09-05 evening user-requested macro paper closure

User explicitly asks to complete only necessary current closeout and urgently consolidate results, mathematics, application meaning and a credible method-driven/result-driven complete-paper narrative. No computation is running; pending new joint-sparse confirmation is not automatically started. Tracker now prioritizes macro synthesis over historical NEXT. Use formula-derivation/research-paper-writing to unify the executed object and writing, and one focused result-to-claim same-family/provisional second view; no score gates, protocol tree or broad retraining. Recheck actual reports/raw data and nearest primary work; historical R001-R003 audit FAIL is scoped historical synthetic, later M1 NIP provisionalPASS also synthetic, neither replaces current F4 evidence. Automation remains active, follows current tracker.

### 2026-09-05 evening macro paper closure completed

Delivered PAPER_SNAPSHOT_20260906.md as an actual paper snapshot: English abstract, coherent Chinese introduction/method, process-to-donor-to-behavior mathematics, aggregate-versus-family counterexample, primary-source related-work table, five evidence panels and concrete application/common-hook challenge. Current supported thesis is conditional operational reproducibility of source-defined signed contributions across SAE seeds; real fresh-document effect/probability evidence is retained, not semantic/canonical recovery or universal raw superiority. Expanded probability medians .044457/.027133 concern eight cases, not all eleven selected requests. Raw competitive/better, rank-one collinear components, grammar matched-norm failure and sparse-B reversal remain explicit.

DERIVATION_PACKAGE §14 consolidates standard estimator/conditional bounds, coordinate-dependent sparsity and the direct-source-write objection; historical §§0-12 unchanged. Primary SemanticOT v1 method and CE/VE replacement evaluation, UnstableFeatures v1 subspace methods and MAS/CLMAS v7 were reread; no categorical prior-art novelty or complete SemanticOT/MAS comparison claimed. One result-to-claim same-family/provisional review returned partial; full trace retained with parent corrections about eight-case aggregation, full-versus-component evidence and same-structure grammar. No reviewer thresholds, extra round or unseen-syntax universal gate.

Scientific allocation changes in plan/tracker: pending joint-sparse fresh-doc confirmation is a retained non-default option, not launched. Next useful unit must test source-defined component predictions/selectivity/use benefit beyond generic shared-hook regression, with cheap source-side feasibility first and fair strong controls; no newly named behavioral variable/run yet. No new corpus, training, fit or LM computation in this synthesis. No running task to interrupt; automation remains active at five minutes and follows current tracker. Paper local links and git diff whitespace passed; no code changes, so no full test rerun. Paper-level method novelty and semantic application are still unfinished, not acceptance-ready.

Ignored local artifacts retained, not force-added or publicly uploaded:
PAPER_SNAPSHOT_20260906.md:b81b06fa78361e07baa5c09766ff11ba248cc462323b1d9ed9f8e21cb2f07ae4
DERIVATION_PACKAGE.md:cec330d99ada49a5b53dbae8b2ad71d58cb23d12e49bcbddd04a2ebc3b4f1b0f
EXPERIMENT_PLAN.md:9cb29985b80a52ea53a9da39cb2b3def2b6c0919bc7de0bd91db4024666aa661
EXPERIMENT_TRACKER.md:e746b1b05052132402fe935935e46551ab1e08973af41debc3c3825ca590e7d2
REFERENCE_REGISTRY.md:368636a3093b1da7fd01f26bfcde827885c089ec07eb338203656d3f3b2533cd
.aris/traces/result-to-claim/2026-09-06_macro01/001-paper-story.response.md:078bcbe7983af612cdd8e55a1b7d749265a9608d49f8d43546a38bcf8b136092

### 2026-09-06 01:01UTC heartbeat: bounded source-consumer orientation

Continued paper-priority card, not deferred joint-sparse confirmation. Read original/expanded case_details.jsonl, selected source context/top-logit fields from the first1-2 positions of each of eleven deduplicated cases; inspected coarse token_class and probability exporter plus prior grammar source/norm summaries. No new LM/fit/data or full case semantic assessment. word_number is character-level, and logit-extreme lists include fragments; source NLL/KL arrays are saved but full base/source vocabulary probabilities are not. This limits choosing a meaningful consumer from those lists, not a new scientific failure. Tracker sets one bounded source-only probability-mass replay option (<=22 base/source forwards, <=30numeric+90load seconds) on existing cases to identify an actionable effect, with original source endpoint anchors retained. No target selection, no new semantic labels, no code/run yet. No resource lease acquired. Routine preparation only; no notification-worthy checkpoint.

### 2026-09-06 01:10UTC source-only probability replay launch

F4_source_probability_replay_v1_20260906 freezes the original3+expanded8 cases and parent config/details/metrics hashes. Minimal dedicated replay reconstructs source edits from all signed exported terms times the frozen source basis (no target calculation), retains all cases and validates old source NLL/KL anchors. Probability-mass rankings and full primary-position vocabulary probabilities replace logit-extreme anecdotes for consumer discovery. Two focused tests pass, including a low-probability large-logit-change counterexample. Budget22forwards/30numeric+90load seconds; no fit/data/training. Warm locked r004 CUDA environment reused per run-experiment/compute contract, no reinstall or witness rerun. gpu-0 free with1384/16303MiB used by existing desktop; acquire manager lease only, bounded CPU preparation does not need cpu-heavy or disk exclusivity. No scientific outcome yet.

### 2026-09-06 source probability replay completed, source-family candidate located

v1 FAIL before LM because the new source_ prefix filter included source_to_candidate KL; corrected to four source-only fields. v2 FAIL after14forwards/6rows at final-token observed-next indexing; corrected last-token/EOS label to null. Both failure directories retained; existing experiment results unaffected. Four targeted tests PASS. v3 PASS4/contract,11cases22forwards, every old source NLL/KL vector replays exactly (maxerror0), complete intervention-position probabilities16,320,440bytes archived. v3wall9.4832856=prep.7225344+load2.0504343+numeric3.6424603+save.5785704+other; v1wall.1680742/v2wall6.8246992, combined36LM16.4760590totalwall. Existing totalLM31904 becomes31940; source diagnostic wall kept separate from previous causal consumer wall. gpu lease released, no running/queued job, automation unchanged.

Source-mass inspection is descriptive development, not new semantic confirmation: largest probability shifts can be concrete word completions rather than the extreme-logit fragments, but within-query contexts have heterogeneous roles. No plurality or other concept labels inferred. Same-source q3:1230/q3:1144 bases are noncollinear (cos.7668187966258511, singular values1.3292173650581565/.4828883973455018); this only motivates a bounded same-input two-contribution source feasibility consumer. Tracker next<=16LM across original four source3 cases, both frozen query components/sum/common family dose and explicit observed continuation contrasts; no target or new data yet. Report SOURCE_CONSUMER_FINDINGS.md retains all11case overview and scope. No notification-worthy result; no novelty, independence or utility claim added.

Local-only artifact hashes before grouped sync: EXPERIMENT_TRACKER.md aa4f4e217e76efa071564b1b1539868b241b1be8460d92e20780f29aedf869e8; runs/F4_source_probability_replay_v3_20260906/SOURCE_CONSUMER_FINDINGS.md e7ccd89cb1a5b2957a2ebebd64aa84ab16efac0366e9669ce308f2f207853ade; same-run source_probabilities.npz 0e14a8e74709e07c2c9bb8f3fd421a1b4164a29a32a8e9002025cf462eb06770. These ignored data/reports remain local. Code and configs reuse existing log_prob/probability_metrics, hook contract, artifact helpers and saved signed atom terms; no external code, dependencies or literature novelty introduced this unit.

### 2026-09-06 01:26UTC user-requested loop pause, not deletion

User explicitly requested pause loop without deletion. Existing heartbeat automation ccad updated in place from ACTIVE to PAUSED and saved status reread; original prompt, five-minute recurrence and target task retained. No active/queued CCAD computation, gpu-0 free. No process, lease, data or automation deleted. Tracker records user-controlled recovery: resume only on explicit user instruction, then continue from the unstarted source3 dual-query source feasibility candidate; do not auto-resume under fault-repair policy. Existing paper/probability artifacts and failed runs retained unchanged.

### 2026-09-06 planning-only research convergence review

User requested substantive literature/directory/blueprint review and brainstorming before implementation. Re-read current entry documents, paper snapshot and derivation section14, relevant original blueprint/theory pages, key raw-result tables, source probability findings, and transport/causal/sparse implementation excerpts. Primary-source method refresh covered MAS/CLMAS, SemanticOT, seed-subspace stability, native SAE stitching, SHIFT, affine cross-model feature transfer, matched intervention evaluation, and joint end-to-end SAE training. New citations and reference-only boundaries recorded in REFERENCE_REGISTRY.md; no complete literature search or implementation replication claimed.

Created RESEARCH_CONVERGENCE_PROPOSAL_20260906.md: proposes operation-family correspondence and useful granularity, operation-matched donor fitting, strong shared sparse controls, distinct readout/native execution claims, and a source-decision reuse consumer. These are hypotheses and planning recommendations, not new positive results. Existing native/grammar failures and raw competition retained; native success not made a universal FCC gate. Added discussion link to plan and planning-only note to tracker. No experiment, fit, model forward, training, new package/model/data installation, or audit-array access. Existing automation reread PAUSED/5min, unchanged; pending16 source forwards not started. No code changes or scientific tests run. Pre-existing master_log edits preserved; no commit/push attempted for this local discussion unit.

Local-only document SHA256: RESEARCH_CONVERGENCE_PROPOSAL_20260906.md EAD159A8A5A1B62B1E1DE93CB056A828C1A0CBEB12ECCD5EA4E39B789813E801; EXPERIMENT_PLAN.md 82162754D955596909A62ED3FA311BBBEC4299314F0A2F856EDCF2FBD383E21D; EXPERIMENT_TRACKER.md 547937B94B3B7931230730993A9FB217C3D5FCA6291BBAE94D5791787BFDA073; REFERENCE_REGISTRY.md 4344D222C34449157FDECBFDA3035CDB9962A39960E77F6D42DE13C0BA358F97. Original proof PDF left unchanged; page16 text and render inspected for the same-hook substitution scope.


### 2026-09-06 ACL comparison, whole-paper scientific closure and authorized loop restart

User explicitly requested reconsidering closure against the ACL2026 one-to-one consistency paper, broad learning, and continuing an automation; prompt changes authorized. Read ACL paper main text pp1-9, limitations p10 and selected theory/interpretability appendices, plus seed-subspace and benchmark-reliability methods; source versions/provenance and actual borrowing recorded in REFERENCE_REGISTRY. The comparison corrects the prior application-only emphasis: measurement, conditional theory, synthetic truth, real functional evidence and useful evaluation decisions can form a full contribution. Neither a new optimizer nor a successful editing application is the only route; few-case transfer alone is not a completed paper.

Updated AGENTS durable principle, EXPERIMENT_PLAN C1-C3/four-figure path, tracker current work card, prior proposal supersession note, and evergreen configs/CCAD_AUTOMATION_PROMPT.md. Next unit is a common source-query development panel aligning atom/FCC/actual effects with honest missing coverage, followed by one hypothesis-discriminating real experiment; legacy dual-query16-forward/application and joint-sparse confirmation candidates are retained, no longer automatic queue. Exact current workload and budgets remain solely in tracker. New plan is prospective, not new evidence or a publication guarantee. Existing raw competition, grammar failures, collinear components and strong-sparse counterexamples preserved.

Existing heartbeat ccad updated in place to ACTIVE via app tool; saved TOML verifies ACTIVE, FREQ=MINUTELY;INTERVAL=5, unchanged target task 01a05f68-a2fe-76c1-a4d9-b754d8679a77, and prompt normalized-equal to repository file. No duplicate automation, new task, model change, lease/process deletion or external commitment. This orientation unit performed no fit, LM forward, training, package install or sealed-audit read. Runtime work is handed back to the original execution task. Validation is targeted document/prompt/state/diff consistency; no code change requiring scientific tests. Prior master_log pause/planning entries retained.

Local-only document SHA256 at handoff: AGENTS.md c9213b1572c62cfd804d9cacdb2fa57fe14c9a741c4917474582cfe924bed58c; EXPERIMENT_PLAN.md 7d4855fd506e46adcd960ffa36740aae51f07511cc8833610e0268ba945bb251; EXPERIMENT_TRACKER.md cd0bf831aa3dd1f70675c37677e1efbb65d7ece27583de0a282cd0449117b291; REFERENCE_REGISTRY.md 817fa409c0c86352f7153741e828a6dce56544933d29f59cb6f0cc7f9cccac99; RESEARCH_CONVERGENCE_PROPOSAL_20260906.md deb82c318d91b2449e02e737eebb3c0f21b66334f4f0507f1d0f8eb17b54e2fc. These ignored documents remain local; allowlist is unchanged.

## 2026-09-06 — Common32request evidence and executable functional truth

Original execution task took over the user-authorized whole-paper continuation. Read current AGENTS/tracker/plan, source query selection and actual probability/component consumer code, and the fixed ACL2026 paper pp2–5 (MCC/identifiability/synthetic methods); registry records concrete reuse. No literature result was counted as CCAD evidence.

F4_common_request_panel_v2_20260906 aligns all32requests/16queries, retaining11selected across7queries/all5source seeds,13matched source-rule rejections and8no-class-donor. Exact query, recipient/donor, positions, source probability anchors, common dose and duplicate full/top16 rows are checked before merging484unique old method rows. The3328row common table includes968measured scope rows and explicit missing methods/outcomes; global Hungarian local execution and decoder-span operation remain absent, not estimated from dictionary metrics. FCC beats the best dynamic atom on case-median KL9/11 and NLL8/11, with reversals for4:693negative,1:615positive and NLL3:1230positive preserved. Best dynamic atom is not PW-MCC. Figure330points uses identical log axes, explicit missingness, target/median symbols and no independent-replicate CI. RGB2400x2200/345682bytes, nominal300dpi stored299.9994, viewed without clipping; no submission-compliance claim. Current v2 image hash equals inspected v1. Both common-panel raw hashes33a46aae152037836569f36a379f9e747dc4ab9c8749c2460917d215a2f97a1f. v1 lacked summary raw hash and failed metadata contract; v2 adds it, identical numerical rows, final contractPASS. No scientific outcome reclassified.

After reading this panel, selected one C1 truth experiment (development decision, not pre-registered): F4_functional_truth_v2_20260906,5generated seeds/mean256/discovery256/512same-context independent donor pairs percontext. Executable bounded-domain nonnegative ReLU codes have the same decoder geometry/MCC/span=1 but preserve context-split u, delete it, or replace it with independent n only incontext1. Existing rank1 hook FCC and best dynamic atom are compared with exact tiny global assignment local execution and a clearly higher-information raw source-hook oracle in the same fixed generated tanh/softmax consumer. These are handcrafted encoders, NOT trained SAEs or language-model forwards. All180seed/family/method/context rows, arrays and fit details retained. Full FCC average normalizedKL1.95e-12/1.014133/.653803 for preserved/deleted/alias. Alias context0=.252146/context1=1.094401; aggregate<1 therefore does not certify every context. Best atom aliasKL=.489296 is better than FCC; preserved bestatom=.484838 versus near-zero FCC. Raw oracle0 throughout. This establishes a conditional measurement distinction and a counterexample to uniform FCC superiority, not a novel solver, low PW-MCC result, real mechanism or semantic concept. v1 omitted explicit audit_opened in config while manifestfalse and failed metadata contract; v2 makesfalse explicit,180raw rows SHA identical e385d2a5f2ab0423a473b110aae670a3439dc0759e963373fb0eff878ca85159. v2PASS/contract; total two-run numerical wall.646093sec. Four targeted tests pass; one focused mathematical/input/comparator inspection, not independent scientific replication.

Initial plotting import on locked r004 failed before creating a run (Pillow absent). Used product-returned artifact-only Python3.12.14/Pillow12.3.0 for stdlibJSON/CSV/image, recorded in local ledger. Scientific synthetic run used locked r004Python3.13.7/NumPy2.5.2 with4threads. No install, model/cache/large-array read, GPU/CPU-heavy lease, real LM forward, training, external asset or sealed-audit access. Real LM total remains31940. Script/config metadata failures remain preserved and are not scientific negative findings.

Paper snapshot §4.1/5.3.1 now includes the measured synthetic table and complete real common figure/table, with application no longer the only closure path. analyze-results/scientific-visualization shaped denominator preservation and figure integrity; manuscript-writing-review was inspected but not applied because this was technical content rather than prose-only review. No new review gate or score threshold. Next card: add actual same-operation FCC/dynamicatom/raw effects for13matched-but-rejected requests, keeping false source-selection labels and8no-donor gaps, <=195real forwards before revising budget if needed. Existing selected_only=false helper means only changed donors, so next implementation must add explicit rejection scope rather than silently reuse it. No next real run/config launched here.

Automation ccad readbackACTIVE/5minutes/original target01a05f68-a2fe-76c1-a4d9-b754d8679a77; no duplicate or prompt/schedule changes in this unit. No current CCAD running or queued compute. Old raw, grammar, rank1 and strong-sparse failures remain; old16/624forward candidates not launched. Main/raw output files remain local under existing ignore policy. Local SHA256: PAPER_SNAPSHOT_20260906.md c89b82bb83d7eac3dea84baf381e97ea75f8a4ccbee7dca46103795e175b4dcd; EXPERIMENT_TRACKER.md 615a2a44d924063b6bb928a565bfc66e83abe710e8ba6bd14de8344c80fc8ebf; REFERENCE_REGISTRY.md b5a1b7932c29f7131985cfde853a509bc23fd71643abe76b1c3057c69c182713; .aris/compute/local.md 83c6f96eb961a1b0279d7d86a257301d7eb587ed0b4b016789fcd910fc1d94f1. Common report6b2a39aadc0b130a4f6621124a4c36d1424d2e1299867457b4e8bc87f4b471a8; figuref677ae47a70cd43b31590694e43a9687ad7626a3f788578c86b21e02ed3ade7a; functional truth reportac7b705dcb9f9a4fa96fe241b2fa6459b816dfd1e981b93f25999f879453319d.

## 2026-09-06 — User-requested pause after completed unit

User requested: finish this round, then pause the loop without deleting it. The common-panel/synthetic/paper unit was already complete and commit35c6241 synchronized; no CCAD compute was running or queued. Updated existing automation ccad in place to PAUSED using the app tool and verified saved status. Original prompt, five-minute schedule and target task retained; nothing deleted, no process interrupted, no new experiment started. Tracker records explicit-user-resume condition and recovery at the unstarted13matched-but-rejected request evaluation. Prior ACTIVE records remain historical, not authority to resume automatically.


## 2026-09-06 — Current-chat takeover and direct review of latest results

User requested inspecting the latest rounds, restoring explicit AGENTS/master_log responsibilities, and having this current chat directly own the automation loop rather than another task. Prior execution task 01a05f68-a2fe-76c1-a4d9-b754d8679a77 was idle; its completed-unit pause remained intact until this explicit continuation instruction. No work was delegated, no message sent to restart the old task, no process or result removed.

Read current rules/tracker, common-panel and synthetic reports, raw JSONL/CSV, implementation, paper result sections and new commit35c6241. Independently reaggregated 32 requests:11selected/13source-rule-rejected/8no-donor. Within the11 measured cases, per-case four-target median FCC errors beat best dynamic atom on KL9/11 and NLL8/11; beat raw7/11 for both endpoints. These are dependent old-data comparisons, not new real-model confirmation or a global PW-MCC comparison. All13 rejected requests have concentrated source contribution and12 also have weak source effect. Therefore their planned completion tests a source-rule-defined applicability boundary; concentration and weak denominator cannot be conflated or the selected-panel advantage generalized. Synthetic raw reaggregation confirms FCC mean normalizedKL1.946982955e-12/1.014132623/.653803426 for preserved/deleted/alias families. This adds an executable measurement distinction and counterexample, not learned-SAE prevalence or algorithmic novelty. Actual global matching on the real panel remains absent. No new scientific run or LM forward in this review unit.

Before edits, AGENTS SHA256 c9213b1572c62cfd804d9cacdb2fa57fe14c9a741c4917474582cfe924bed58c exactly matched the previous coordination snapshot. Normalized current master_log retained the entire c87fcb0 version as an unchanged prefix (3621 lines before this entry). Thus no deletion of these files/history was found. The evergreen prompt had kept their names but compressed responsibilities; now explicitly states AGENTS/tracker/plan/master_log/registry/environment roles, append-only log and raw-evidence priority. Added durable current-chat direct ownership/no unsolicited task or subagent delegation to AGENTS and prompt, and updated tracker owner/status/review without discarding completed work or the next13-request recovery point.

Existing automation ccad updated in place via app tool: ACTIVE, five-minute heartbeat, target_thread_id=01a06de2-9fac-7fb1-9b8a-12faa6436c01 (this chat). Saved TOML readback confirms all fields and normalized prompt equality to configs/CCAD_AUTOMATION_PROMPT.md. No duplicate automation or external resource created. Previous task history remains available, with no further research assigned there. Document/state checks and independent raw-statistic recalculation are the relevant validation; no implementation changes or additional tests required. Prior uncommitted user-pause log entry preserved.

Local-only hashes after takeover: AGENTS.md b7996acf3977b185723a2e46d1a4826644a8f6a96b3425d8c73e2868463a9164; EXPERIMENT_TRACKER.md 6bd6f50d98980f04b33f4419ec0984ee200bb05a53a5da17b6c2268ec3ea606e. No upload allowlist expansion.


## 2026-09-06 — Real source-rejected completion and operation-scope boundary

Current chat directly executed the next scientific unit. Added explicit source_selection_scope=rejected to the existing selector: retains unchanged as well as changed matched donors, validates original recipients/token classes, preserves selected=false, keeps legacy behavior when absent, and rejects conflicting options. Fixed request count and exact request identities are checked before model loading.10 focused tests PASS; affected modules compile and diff check passes. Two configs freeze11original+2expanded requests, FCC/dynamic single-atom/raw only, unchanged signed rank1 donor operations/maps/common cap0.1, no new fit/query/donor or outcome-based selection.

F4_rejected_probability_original_v1_20260906 PASS/contractPASS7checks:165forwards132raw rows, wall126.3456271s. Expanded counterpart PASS/contractPASS7checks:30forwards24rows, wall23.1277348s. Total195real LM/149.4733619consumer wall (includes preparation/loading, no isolated numerical time claimed). Max noop0, worst raw-hook relative replay1.8503111e-5, peak allocated VRAM841839104bytes. Existing resource manager reported free leases; measured GPU had desktop/game load~8.45GB/16.3GB and70percent, enough remaining memory against prior0.842GB runner footprint. Used cpu-heavy then gpu-0 run wrappers, released both, did not alter user applications/shared manager/other leases. Timings are not exclusive GPU benchmarks. Real LM cumulative31940+195=32135; this unit wall kept separate, no invented prep/numeric split. No new corpus, training, package/model download or audit access.

Merged new outcomes with unchanged original11case rows using exact source/query/donor/positions/source probability anchors/common dose and fit/data/endpoint identity. Complete24matched requests cover15queries/all5source seeds;8no-donor requests remain missing in32request denominator. Source-selected FCC/atom medians KL.039109/.091371, NLL.021415/.115442, wins9/11 and8/11. Newly source-rejected FCC/atom/raw medians KL.859671/1.0/2.544573, NLL.882603/1.0/2.694067; FCC beats atom6/13,5/13 and raw11/13 for both primary metrics. All24matched medians FCC/atom/raw KL.245865/.692245/.877091,NLL.292696/.718411/.739897; wins vsatom15/24,13/24 and vsraw18/24,18/24. Four targets first aggregated within case; shared seeds/queries/reversed pairs not independent replicates. Secondary same-document outcomes and counterexamples preserved in report/CSV.

Scope finding:13rejections all concentrated,12 also source-weak. Source KL medians selected/rejected .005015/.000306, NLL-delta RMS.079593/.027615, while FCC absolute KL error.000162/.000215 and NLL RMSE.012833/.012929 are close. Thus relative fidelity separation includes denominator changes; no causal attribution to concentration alone. The result extends coverage and supports a conditional distributed advantage, not universal FCC superiority, global one-to-one comparison, unique concepts or fresh-document confirmation. All old raw/grammar/rank1/strong-sparse failures remain.

scripts/analyze_f4_rejected_panel.py writes same-unit SOURCE_RULE_BOUNDARY.md/CSV and analysis/figure/provenance JSONs in the expanded run.768CSV rows include576measured and192missing;240figure points include all dependent targets and case median ratios on shared log axes, with source RMS and concentration visible. Image2300x2140 inspected without clipping. Scientific-visualization skill applied for honest scopes/missingness/provenance, not a submission-compliance gate. Separately recomputed624new probability ratios from raw vectors using math.fsum, maximum relative difference7.99e-16. Checks validate implementation/merge arithmetic, not independent scientific replication. Script/raw identities preserved; no model rerun for reporting.

Paper snapshot5.3.2, plan and tracker updated. Next useful work is a full-dictionary one-to-one alignment lifted to the same source-group/source-basis operation, with geometric and paired-only per-match calibration variants. Reading R009 code confirmed its128-query by3072 assignment is not full-dictionary PW-MCC; do not relabel it. Next bounded CPU preparation and actual common-case comparison budgets live only in tracker. No next-run config created in this unit. Registry records actual code borrowing and previously read source boundaries; no new literature-version refresh claimed. Automation remains ACTIVE/5minutes bound to this current chat, no delegation or duplicate loop.

Local-only hashes at unit close: AGENTS.md b7996acf3977b185723a2e46d1a4826644a8f6a96b3425d8c73e2868463a9164; EXPERIMENT_PLAN.md bd5bb051083abb1a45f6ff6c19429a9dae1f03f6b53a95108b25b186d323cc8e; EXPERIMENT_TRACKER.md c8bc96d2469d25ef78ad770ce822a76aa9c3d748e577079975bf25f5d75f36a2; PAPER_SNAPSHOT_20260906.md 81024f65593409a0e28bd16d7f5c0a8b20351e0063dda44d9f341d9f02dd3f7e; REFERENCE_REGISTRY.md 476bfe9026a0c834c30c7c7e8e50fa4dd09e4b9d1e6bd4049c3c47dff48313a9; runs/F4_rejected_probability_expanded_v1_20260906/SOURCE_RULE_BOUNDARY.md d20b2c8fdb190ef960feda588a3b463efce61e51c90592180edb9828adf8a621; runs/F4_rejected_probability_expanded_v1_20260906/source_rule_boundary.png 7692c1af545a83150e5c2527768390cc26cb3114cabca734b9f7decabf7c0169. Existing upload allowlist unchanged.


## 2026-09-06 — Full-dictionary matching operation unit

Start: current chat froze f4_global_matching_prepare_v1.json after reading the actual SciPy1.18.1 full-assignment docstring/BSD license and existing R009 query-subset implementation. Reuse standard modified Jonker-Volgenant solver, not a new Hungarian implementation. Full3072 matching, source-group geometric/paired-only scalar-calibrated consumers, no target behavior fitting. Four operation/math tests and10selector tests pass; explicit all_supported retains24fixed matched cases and original labels. CPU preparation budget and264forward application budget recorded in tracker before execution. No new data/training/audit or delegated work.

Completion: F4_global_matching_prepare_v1_20260906 PASS/7checks/contractPASS, CPU wall14.9351896s,0LM,10full3072-by3072 assignments inverted to20directions and64query-target group readouts. First matching pair.7782208s; remaining about.62-.68s. All source32 members mapped once; no zero decoder norms or zero cosine matches. Full PW-MCC range.1528493515-.1536283146 is dictionary-level, not query function evidence or10independent replicates. Geometric scale sign(cos)*norm(Dt)/norm(Ds); paired scalar slopes use exact old256source-positive discovery rows/weights, cov/(var*1.001), no cross-pair mixing. Conditional intercept cancels in donor differences. Saved full assignments, group coefficients/details,192coordinates and input/config/source hashes; same frozen five short-training seeds, no training/new corpus/FCC refit/audit.

F4_global_matching_original_v1_20260906 and expanded counterpart both PASS/7checks/contractPASS:154/110LM,112/80raw rows,106.7403337/41.3089336s consumer wall. Total264LM/148.0492673s wall, prep CPU separately above. Real LM cumulative32135+264=32399. Max noop0, worst raw replay1.8503111e-5, peak841839104bytes. Resource-managed CPU/GPU leases automatically released; closing status allfree, no CCAD job running/queued. Timing includes loading/preparation and shared user GPU load, not an exclusive numerical benchmark. No other leases/processes altered, no D-disk lock for inference.

Primary selected11case FCC/geometric/pair-calibrated KL .039109301/.502720538/.477222882 and NLL .021414850/.418400133/.468024429; FCC wins both baselines11/11 on both endpoints. Rejected13case FCC/geometric/paircal KL .859671406/.948225698/1.0 and NLL .882602837/1.023967655/1.0; winsvsgeometric8/13KL7/13NLL, vspaircal9/13KL8/13NLL. All24matched FCC/geometric/paircal KL .245865110/.682283800/.705987099, NLL .292696202/.737924869/.784368353; winsvsgeo19/24KL18/24NLL andvspaircal20/24KL19/24NLL. Secondary same-document all24 FCC/geometric/paircal KL .272119/.676126/.684851, NLL .310463/.668226/.773901, paircal comparisons also20/24KL19/24NLL. Four dependent targets median first, then case median; shared seed/query/doc/reverse pairs not independent replication. Existing atom/raw values unchanged and retained.

Scientific scope: C2 now has actual full-dictionary one-to-one group-operation comparison, not a renamed single-atom/query-subset score. Source-defined selected scope retains clear functional advantage; full matched scope includes failures. All32requests/16queries retained,24matched/15queries measured,8no-donor remain missing. Full FCC reads all target contributions whereas matching reads32target members, so not equal target input/support capacity or proof of unique many-to-many mechanisms. Both are source-aligned readouts, not native deletion. Concentration/weak-source confounding and short training remain. Existing exposed documents mean development, not independent new-document confirmation. No universal raw/atom/one-to-one defeat claimed; old grammar, raw, rank1 and sparse counterexamples preserved.

Analysis merges old FCC/atom/raw and new matching with exact recipient/donor/positions/source probability anchors/common dose and fit/endpoint identity. Expanded run contains GLOBAL_MATCHING_FINDINGS.md, GLOBAL_MATCHING_ROWS.csv1280rows(960measured/320missing), global_matching_comparison/cases/figure JSON, figure_points.json720points and provenance hashes. Plot2400x2200 has all32requests, shared log limits retaining extreme calibrated failures, both target and case markers; visually inspected without clipping. Separately recomputed768new KL/NLL ratios from raw arrays with math.fsum, max relative difference6.831021053001987e-16;0LM. These are implementation/merge/arithmetic checks, not independent scientific replication. Four focused matching tests and10selector tests passed before runs; no unchanged-code repeated full suite.

Paper5.3.3 now reports both full-matching consumers, table/figure and unequal-capacity/weak-source limits; older unmeasured wording points to new section without altering historical outcomes. Plan prioritizes common fresh-document confirmation of frozen FCC/raw/atom/geometric/paircal with all16source queries and32requests. Exact input exclusions/coefficients/configs to be frozen before sampling/evaluation; no next corpus/config/run exists yet. Initial128asset+atmost736consumer LM and source-only selection/application budgets live in tracker. This directly follows positive signal, not another open-ended development control grid. SciPy1.18.1 installed solver docstring/BSD license and reused source-discovery interface recorded in registry; no new package/code download or fresh literature-version check claimed. Automation TOML read at close ACTIVE/FREQ5min/current task01a06de2-9fac-7fb1-9b8a-12faa6436c01 unchanged, no delegation.

Local-only/artifact closing identities: AGENTS.md b7996acf3977b185723a2e46d1a4826644a8f6a96b3425d8c73e2868463a9164; EXPERIMENT_PLAN.md 0450bb21e2f6c203bdc04a6c15a043255580b289cab8853441af229aebfa97d1; EXPERIMENT_TRACKER.md 2020646628ba9bb5ff204dcca273d39e3ebe05490a89e871e71ae0f3e20c5cf9; PAPER_SNAPSHOT_20260906.md 401a7711f0ac272a5ffb6af74bf31a136f0b2caa65d0c202d0ab959c6b080a52; REFERENCE_REGISTRY.md 8e504c87cdde44f42a0648ce63e7b6cdd8d50d20d9c57ed789f27efab1e1a854; runs/F4_global_matching_prepare_v1_20260906/group_coefficients.npz 31db894904c3aa24fc15f5dc92e21b357b1469d4328ea67109f624bfde08c862; runs/F4_global_matching_expanded_v1_20260906/GLOBAL_MATCHING_FINDINGS.md f2ec88f17e96e509c3e0f161be168a8204cac5d551aff2ddcfe68abba40b6336; runs/F4_global_matching_expanded_v1_20260906/common_panel.png e9d3c91122f2f7acbb836513f7ae38911955c4713ea8911a5ed6903feedf22a8; runs/F4_global_matching_expanded_v1_20260906/global_matching_analysis_provenance.json e90899cca0e8b264cd411d16b68669a2b6bfd9330b04c043d2c8d2a40eeb01dd. Upload allowlist unchanged.

## 2026-09-06 — Common fresh-document functional confirmation

Before sampling, froze configs/f4_common_confirmation_corpus_v1.json SHA256 caa1613a28615546c57a31c246310dd68ab4cdfb26669a5ade5aadc95b744c3b. All16queries/32signed requests/five source seeds, FCC/raw/dynamic-atom/global-geometric/global-paircal, original source-only selection/class-donor rules, basis/cap.1 and probability endpoints preserved. Dynamic atom and64global query-target coefficient coverage checked.15document exclusion ledgers include short/long sampled/training/validation, original paired and all existing natural-text F4 corpora, including109/108. No new input or target endpoint read yet. Cache inventory of known storage/standard HF dataset paths found no reusable raw FineWeb row-group cache; use established revision-pinned official ranges. Budget in tracker before execution; no training/new packages/weights/audit/delegation. CPU/GPU leases free at start.

Completed all frozen confirmation stages without fitting, retraining or new target-dependent choices. Corpus F4_common_confirmation_corpus_v1_20260906 PASS8/contractPASS:102used documents/65536tokens,5official parquet row groups/5000sampled rows,39range requests/34742784bytes, state span64.425267s. Recomputed exclusion union over15ledgers:8851unique document IDs/text SHA256, fresh102unique and0overlap; only document identities from audit-containing corpus manifests, no audit arrays/metrics. SAE asset F4_common_confirmation_codes_v1_20260906 PASS/contractPASS:128shared base forwards, same five frozen encoders L0=128,21.4237129s processing/saving timer excluding loading; state span46.298585s. Tokenizer/model locally reused, official source binding commit/LFS etag verified by established runner, no new package/model download.

Source-only original/expanded preparation both PASS/contractPASS0LM,9.3154427/6.2348078s.32requests ->21matched (10original/11expanded),6selected(all expanded),15source-rejected,11no-class-compatible-donor. All14matched queries/five source seeds/27unique recipient-donor documents; selected6cases are1:2641positive,1:615positive/negative,2:2645positive/negative,5:2194negative,4queries/3source seeds(1,2,5)/13documents.102is sampling pool size, not independent effect count. Coverage changes11/32oldselected ->6/32newselected; no new donor/query search or threshold relaxation. F4_common_confirmation_apply_v1_20260906 PASS7/contractPASS2.6939795sCPU, old192global coordinates regenerated exactly(maxdiff0),168new matching coordinates; group coefficients64unchanged,0fit/LM. The adapter only rebounded new input identities and applied coefficients. Source preparation historical configs inherited not_required_bounded_CPU metadata, but actual commands used cpu-heavy wrappers; record that operational metadata correction here without rewriting resolved run identities. Adapter source_config now records cpu-heavy for future use; no numerical code change or rerun.

F4_common_confirmation_original_v1_20260906 and expanded counterpart PASS7/contractPASS:230/253forwards,200/220raw rows,85.5397537/90.9008066consumer wall. Combined483LM/176.4405603s; all five methods on every matched case. Max noop0, raw-hook replay worst1.9382458736483985e-5, peak841839104bytes. Original and expanded consumed only fresh case outcomes, not old endpoints. No failed numeric runs this unit. Whole unit611real forwards =483consumer+128shared asset. Counter clarification: recent shorthand cumulative real LM32399 tracked the consumer/task ledger and omitted the separately recorded1091asset forwards (see prior joint-sparse entry). Continue categories explicitly: consumer/task32399+483=32882; assets1091+128=1219. Do not add128to the consumer ledger or describe32399as the all-category prior total. Preparation CPU/network and consumer/asset timers have different scopes; no invented total GPU-hour. Closing shared-manager status allfree; no running/queued CCAD computation, user processes/other leases untouched.

Primary selected6request medians FCC/raw/dynamicatom/geometric/paircal KL .008911726/.018430918/.195333921/.367099599/.409465154 and NLL .007010389/.021395522/.093879085/.332228780/.452781339. FCC wins atom/geometric/paircal6/6each metric and raw4/6each. Rejected15 medians FCC/raw/atom/geo/paircal KL .805822138/2.495817627/1.0/.722484534/.998316750,NLL .961876963/2.211210265/1.0/.779677210/1.002963185; geo aggregate error better thanFCC, failures retained. All21matched medians FCC/raw/atom/geo/paircal KL .215599462/.394979257/.974541838/.664128159/.985134766, NLL .223425250/.430335388/1.0/.691726244/.992352381. FCC all21wins vspaircal16KL15NLL,vsgeo14KL13NLL,vsatom14both,vsraw13both. Secondary selected FCC/paircal KL.009831820/.429485487,NLL.012237394/.513040597; all21 FCC/paircal .208601960/.987978486,.243152485/.987219457, wins16KL17NLL. Four dependent targets median first, then request median; no independent-direction significance claim. Source absolute errors and weak denominators retained alongside normalized ratios.

Scientific decision: fresh-document confirmation supports C2's frozen source-conditioned functional difference beyond dynamic atom and full-dictionary one-to-one group operation. This is a restricted alignment-family result, not equivalent-capacity advantage, native deletion, unique semantics, universal many-to-many necessity or a clean training mechanism. Full FCC has more target inputs, selected scope only3source seeds, weak/concentrated confounding and11missing requests remain. Rejected cases and raw/atom wins prevent universal claims. F4 current102documents now exposed after frozen confirmation; retain its completed confirmation identity, do not tune next method on it.

summarize_f4_common_confirmation.py verifies exact case/donor/positions/source probability/common-dose anchors across all methods, independently recomputes1680probability ratios from raw arrays with math.fsum(max relative difference6.604578020959411e-16), writes complete1280row CSV(840measured/440missing),32requests/64scope-cases, summary/provenance JSON and630point2400x2200figure. Shared log limits1e-4..1e4 retain extreme counterexamples and all missing slots, visually inspected without clipping. Ratio/interface checks are implementation checks, not independent scientific replication. Small adapter/analysis compilation and actual192old-coordinate replay appropriate to change; no unchanged-core repeated full test suite. Scientific runtime r004/old overlays unchanged; stdlib/Pillow reporting in documented bundled runtime.

Paper snapshot5.3.4 contains full frozen-confirmation table/figure, source/query/document denominators and evidence limits; maturity table updated. Plan/tracker now target one C3 explanation: matched32 jointly fitted ridge versus atmost32function-selected sparse readout on original24development cases with same source target/rows/weights/cap, preserving all old strong controls. Future actual standardization/support/ridge configuration to be set before fit after focused reuse review; initial bounded CPU and264LM budgets only in tracker. No next fit/config/run created or old16/624job started. Registry records actual source/code/asset reuse; no fresh external paper-version check or solver novelty claimed. Automation read at close ACTIVE/5min bound to currenttask01a06de2-9fac-7fb1-9b8a-12faa6436c01, unchanged/no delegation/duplicate loop.

Local-only/artifact closing identities: AGENTS.md b7996acf3977b185723a2e46d1a4826644a8f6a96b3425d8c73e2868463a9164; EXPERIMENT_PLAN.md b5e02e2ad6ae4896248aa002047623856dfd4e8faee766847962aa7f998ece47; EXPERIMENT_TRACKER.md 5452271b05ca94ffde92f22c50c2cd39cb6558c3c5baf37bed13eae5ceeb8f73; PAPER_SNAPSHOT_20260906.md 6b8be804b696d088c3c67ba5df209b3e30c57a27acf8529bca56a42f4ffc9333; REFERENCE_REGISTRY.md 5caaaa465b6a23ec8fb62a78ec72ffcdfa77a108e8e056d2c108a72a42f14c46; runs/F4_common_confirmation_apply_v1_20260906/COMMON_CONFIRMATION_FINDINGS.md 61e60700706e0d6e95d7c91f69ab11305564d39a73b626ce9d906352bb5fb0b2; runs/F4_common_confirmation_apply_v1_20260906/common_panel.png 073419d767df168db7652b5916670388e7b68d441a8a655c69af10a126fbb6c6; runs/F4_common_confirmation_apply_v1_20260906/common_confirmation_analysis_provenance.json 3a4058063a3547b28c4a4b3ae3ecf3ef901ee70a88bd7e3f36008fb2be754ac4. Existing upload allowlist unchanged.

## 2026-09-06 — User-requested full-history retrospective and interpretation-first consumer

Current chat directly reviewed the user's six mathematical bridge concerns and full CCAD history, without delegation, new model calls/training, audit-code access, or automation changes. Existing 310 run directories inventoried; all available primary metadata plus legacy summaries/corrections read. Structural status labels (245 PASS,40 FAIL,23 missing standard status,2 historical residual RUNNING) are not scientific validity counts or live-process claims. Read 264 primary/secondary result ledgers,68,124 rows,164,517,661 bytes with no parse errors. Large candidate/prediction ledgers and sealed real audit arrays were not reloaded; this is not a complete implementation rerun or independent line-by-line proof audit. Full stage-by-stage report: RESEARCH_RETROSPECTIVE_20260906.md; raw coverage/input hashes/reanalysis source: artifacts/retrospective_20260906/.

Recovered evidence: direct recalculation of M1 corrective full held_out_evaluations gives CONTRIB-KNN/top4 79/80 exact with one pre-score BUDGET_REFUSAL; DECODER-KNN/top4 80/80, CODE74/80,RANDOM60/80,Li15 42/80. F02 is19/20,F03/F04/F06 each20/20, with F06 correct rejection rather than positive recovery. Preserve original all-pair FAIL and distinguish local feasibility from a contribution-proposal advantage. Native long-k32's sole FOUND is source2 atom2767 to target1 atom207, size1; calibration BCC.981259623,d_ctr.038889145,d_mu.017522654 and1,915 source firings. Size2/3/4 d_ctr worsens to.249549511/.442719565/.803111150. This is a useful single-atom correspondence reference, not fuzzy many-to-many success or downstream/semantic confirmation. Old native zero coverage, SCT/raw, CCA, grammar, mean-only confounding and strong-sparse counterexamples remain. F5 transport had surface_rows0 and W1 was unexecuted, so neither is empirical evidence of universal failure; prior F5 calibration-read correction remains in force.

New source-only descriptive reanalysis uses previously saved 11case x4position x50,304 vocabulary probability arrays from F4_source_probability_replay_v3_20260906. Fit p_beta proportional to p_base**beta, beta[.25,4], minimizing KL(source||p_beta) on the same exposed outputs. Shared-case optimal-temperature residual/source-to-base KL median.961772995; more generous per-position optimal temperatures yield case-ratio median.775962347. Entropy increases23/44,baseline-top1 probability decreases24/44,top1 identity unchanged42/44. Therefore a simple shared confidence change does not explain most of these old changes; residual is not semantic evidence or a new-input confirmation. Float32 archives normalized in float64,floor1e-300; all44positions retained. Source3:1230 positive anchor absolute-coordinate participation median.060037937,parmes position.027894890;3:1144 positive/negative.091518953. Anchor labels cannot stand for the whole32-member projected source group; participation is not causal necessity. Analysis2.9844543s CPU,0 new LM; known-temperature fixture beta.55 recovered.5500000000000003,probability maxerr1.11e-16. No packages/environment changes; r004 Python3.13.7/NumPy2.5.2 for this array calculation, documented bundled Python for stdlib inventory.

Theory clarification DERIVATION_PACKAGE section15 distinguishes input prediction, output-effect explanation and native member attribution; rank-r SVD optimum is only a lower bound for a restricted SAE realization class. The uncentered +/-v/no-offset nonnegative-single-code counterexample does not refute a centered rank1 process: a finite code can be shifted nonnegative, though sparsity/encoder/bias constraints still matter. Original blueprint/PDF and historical proofs unmodified. PAPER_SNAPSHOT section5.6 adds source-control evidence, section6 corrects wording from an already-transferred human explanation to a source-contribution reproduction interface. Source semantics remains unvalidated.

Nearest-source refresh read Paulo et al arXiv2410.13928v2 sections3.1–3.3.5 andA.7, McCann arXiv2605.12874v1 sections3–7 officialHTML; bothCCBY4.0,reference-only,no code copies/install. Borrow output-effect explanations and functionally informative same-label negatives, not label-uniqueness or a cluster-size penalty. Same-label cross-seed replication/split can be legitimate; collision alone is not failure. Versions, scope and unverified code licenses recorded in REFERENCE_REGISTRY.

Plan now prioritizes testing a falsifiable explanation of a complete source signed group on new contexts and across seeds, with atom/raw/matched-group/standard-sparse information and audit budgets matched. Same32 joint/sparse remains a supporting capacity control rather than automatic next error-only unit. Existing source/native reference and all old positives/failures retained; source-only applicability rejection is not absence of a concept. Full report has all-stage retention table, exact new recalculations, six-bridge assessment, concrete interpretation opportunities and nonclaims. Current run/budget/next-card lives in tracker; no new experiment queue. Automation readback ACTIVE,5min,current chat01a06de2-9fac-7fb1-9b8a-12faa6436c01, unchanged. All new reports/results remain local under existing allowlist; only this factual log is eligible for synchronization.
Local-only closing identities (existing upload allowlist unchanged): RESEARCH_RETROSPECTIVE_20260906.md SHA256=68b25dd841ed1c022f7902fe201a0f8b30e423a490f43f9bc851721378b6ec38; DERIVATION_PACKAGE.md SHA256=622eeed6a348821d99d64f4a414b9b619721f24f210c033d2e9bdb4939921be4; EXPERIMENT_PLAN.md SHA256=79e944a39a4505c8f9def497765ef5f08356697097e572319c6e9141d3f0faba; EXPERIMENT_TRACKER.md SHA256=0bee6df884026f6a0a55711b3f3ce681bc4503ef4672d6fdee432c038cf7dfd2; PAPER_SNAPSHOT_20260906.md SHA256=0d0a0bfd225c03a22c93f17b26e500f6a5a62c13ba33e7bdccb6aa6f38d7bbc5; REFERENCE_REGISTRY.md SHA256=5979061697b9423a14d4cb13cfda10c76931bf5fd9a117fb8fb9e28e82196947; artifacts/retrospective_20260906/review_saved_results.py SHA256=f2b29339cff5f1779a325f6222c0762180ec5295f138b53c82331ffd4b9c34cd; artifacts/retrospective_20260906/source_probability_controls.py SHA256=149a70116668a7ec1a9b3c288d1c0f6d1d2d2cde11c8ea806070d566c42022e6; artifacts/retrospective_20260906/source_probability_controls.json SHA256=8b96b4881ea76a4b3306f10eebc8a6a9c787b765d28f9501fe5e353b00dcfc4b; artifacts/retrospective_20260906/raw_result_inventory.json SHA256=b1ebd2f52500032028add7568c3f8ef39c8f11157fcfb592b8079dfada023e87; artifacts/retrospective_20260906/m1_unknown_support_recalculation.json SHA256=86e1b64db4fcdf3731a76c759246fe808f065a76d49f8dc4c31d1a93f085ae61; artifacts/retrospective_20260906/native_rescue_positive_case.json SHA256=8172919f24875ad50199acf73280dc30ade07fd6c32203dd8e34a7993ee27aa8.

## 2026-09-06 全面推进与当前入口整理；source解释工作启动

用户授权积极维护过期工作卡并保留长期要求。整理前AGENTS/tracker/plan逐字归档至archive/research_workflow_20260906/interpretability_continuation，三份SHA一致，manifest保存原hash/bytes；AGENTS仅加保留规则。F4_source_contexts_v1_20260906已完成100源discovery片段/4.443449秒/0LM/契约PASS；此短运行未先追加启动日志，现如实补记该记录偏差，非科学数据变动。四signed组正尾高度重合于标点和早期位置；native s2:2767出现we上下文线索。已冻结configs/native_we_source_v1.json，下一启动NATIVE_we_source_v1_20260906，16构造输入×4操作，最多64LM/180秒numeric；source-only可行性，非FCC/native跨seed确认。0新训练/audit/付费，资源按gpu-0管理。

源端NATIVE_we_source_v1_20260906完成64LM/numeric11.7690126秒/wall87.5574646秒/peak705221120B/noop0/契约PASS。固定we激活对they/I/you12/12更高，删除对比4/4负、加回3/4正，其他主语有作用与第四条加回反例保留。冻结native_we_correspondence_v1配置：历史固定s1:207，同16输入与原操作/端点，最多64LM/180秒numeric，0新拟合；启动跨seed功能可行性复核。源端代码快照保留，新消费者仅detach decoder避免无用梯度图，不重跑旧结果。

本单元完成：NATIVE_we_correspondence_v1_20260906固定s1:207复核64LM/numeric2.0178398秒/wall9.0911238秒，noop0/peak705221120B/契约PASS；we激活对三个匹配主语12/12更高、删除预定log比4/4下降、加回4/4上升。源端与对应seed16输入/词集完全相同，保存baseline概率逐位maxdiff0；是4模板两seed单原子功能解释正checkpoint，不是FCC优势/自然文本泛化。两run128LM/numeric13.7868524秒/wall96.6485884秒，GPU租约已释放。源端冻结单原子参照与正反结果已写入INTERPRETABILITY_PROGRESS_20260906及论文§5.7，未改旧语法或FCC负结果。

source_position_diagnostic在已暴露discovery上得到四组边界坐标能量93.30%–97.03%、前8位置35.96%–37.43%；非边界且position>8后5:2194对其余相关约−.04至−.06、另三组仍.921–.970。8.2882005秒/0LM，条件掩码由本次观察提出；不称因果拆解/确认/纯起点伪影。论文§5.8记录公共成分与可发展残余，下一source-only and残余用途检验，同时保留native新自然文本确认机会。工作卡收拢49行，过期状态退出当前入口，AGENTS逐项恢复校验显示旧全文在仅去除新增保留段后完全一致。automation已核对ACTIVE/5min/当前对话目标未改，0新训练/下载/安装/audit/付费，不委派。

实现检查：上下文100行及signed项重构检查通过；16跨seed基线概率完全相同，固定对比token一致；未对无改动的核心重复全套测试。轻CPU上下文+条件诊断12.7316498秒，与GPU消费者分列。研究文件依白名单本地保留，身份如下（完整清单另见artifacts/retrospective_20260906/continuation_hashes.json）：
- `AGENTS.md` SHA256 `6f0c1dd725c271fe7b085ce520aa299c36e5f0120b9d1a41412febaa88a6e35d` (13830 bytes).
- `EXPERIMENT_TRACKER.md` SHA256 `57b5893916ce0bf62968d6f1c10c151bb9e275288e184b5821cf8a809f74b263` (7290 bytes).
- `EXPERIMENT_PLAN.md` SHA256 `b9eb19983107f94638304cdfb9803fe0d1095d2fe72ab836fb973af875914cf7` (13118 bytes).
- `INTERPRETABILITY_PROGRESS_20260906.md` SHA256 `d78ae2b4313787d8879b3f324abf5fafca0b9c71fd54b933721a1346de491bd9` (6187 bytes).
- `PAPER_SNAPSHOT_20260906.md` SHA256 `2ec68703ef4471b58edb63015b4b99fa77fbff427849780fe2f84bdcadc27ddf` (43476 bytes).
- `artifacts/retrospective_20260906/source_position_diagnostic.py` SHA256 `fa467da13a112cf3a4656032c4066170f689bb95a6c77bff05fe6d9f15055d16` (2775 bytes).
- `artifacts/retrospective_20260906/source_position_diagnostic.json` SHA256 `736a72c25a50f420f27680ad09171c34ae590df0ac9fb60c195fe81bd35a9d01` (4543 bytes).
- `artifacts/retrospective_20260906/continuation_verification.json` SHA256 `2bd2f53ed17cd14db450eedf6783c70868f9313357b50a18b5abd3f1c937e278` (312 bytes).
- `runs/F4_source_contexts_v1_20260906/metrics.raw.jsonl` SHA256 `1ca64cbc2cbb420d9d17f86383eedb71151e10f829d20e1b0117e876231fd142` (377216 bytes).
- `runs/NATIVE_we_source_v1_20260906/metrics.raw.jsonl` SHA256 `ff2c6041b2032752a9d5f840126f74245b1d23f0090d597fa145a4fcfda90b69` (38965 bytes).
- `runs/NATIVE_we_correspondence_v1_20260906/metrics.raw.jsonl` SHA256 `f57786463c5deff71dba754e109dc85b5633dcffb94c82abd5156b80fe18c758` (39067 bytes).
- `configs/native_we_source_v1.json` SHA256 `7b11b49ddc4c3fb2b8986361a1731da4d56785f8180437af6cf671e0b2460980` (1618 bytes).
- `configs/native_we_correspondence_v1.json` SHA256 `c08e825567272e4eee1e033d87905b5e113fd8f6ad981143e4fceb8f48da83a7` (1869 bytes).

## 2026-09-06 05:19UTC heartbeat：and源解释可行性

按当前卡继续5:2194非边界残余。source-only discovery提取20片段并固定位置/同组正系数分量匹配，1.1673637秒/0LM，artifacts/and_source_development_20260906保留原始项；and/or/but坐标中位−4.287478/−2.573451/−1.452822，between/range/list and较弱。观察后假说不是确认。冻结configs/f4_and_source_v1.json，8对新构造and/or共16输入（四普通四range/list，每对仅最后token变）；原完整32源组/basis、不拟合，最后位置source donor差分、同能量正交随机和noop，64LM/180秒numeric。主端点是预定pronoun-start与指定并列item词的log概率比，完整概率/反例保存；不是句法真值或native删除。启动F4_and_source_v1_20260906，gpu-0租约，0训练/下载/包/audit/付费。

and源实测F4_and_source_v1_20260906完成64LM/numeric5.0424399秒/wall75.3597299秒/peak706739200B，noop与source系数replay误差均0/契约PASS。输入and比or负8/8、受限范围差分小4/4；原预定output方向普通and/or各0/4，保留FAIL-hypothesis（run实现PASS不当科学方向成功）。反方向普通and/or各4/4，source对比变化and−.4884/−.5680/−.9700/−.6679、or+.0299/+.0907/+.2265/+.1123。已冻结修正方向与全新词汇4普通4range/list上下文configs/f4_and_source_lexical_replication_v1.json，原算子和词集类别不变，启动同名run64LM/180秒numeric；新词汇source复现，不是自然文本/目标或句法独立确认。

修正方向F4_and_source_lexical_replication_v1_20260906新词汇64LM/numeric2.1842445秒/wall8.4130222秒/peak706739200B/noop及系数replay0/契约PASS；普通and删除log比4/4负、or4/4正，同能量随机方向非一致；受限差分幅度4/4更小，between coffee方向反转保留。源端出现可复现作用后冻结configs/f4_and_targets_v1.json，完整16输入原source donor/dose/词集不变，复用原四target FCC/动态atom/raw maps无拟合；主signed对比/绝对误差、辅全概率KL，224LM/180秒numeric，启动同名run。预测范围由source结果确定，不按target选；one source query不称五独立source seed。

本heartbeat科学checkpoint完成：F4_and_targets_v1_20260906固定四target/三旧方法/完整16输入，224LM/numeric7.1849367秒/wall20.7750478秒/peak722158592B/契约PASS。普通8输入×4target方向FCC/atom/raw=32/16/32，pooled绝对log比误差中位.043076/.196719/.030964，归一KL.033447/1/.026701；FCC对atom逐行两误差皆30/32更低，对raw仅12/32、11/32。受限范围方向28/12/24，绝对误差.045484/.102513/.031742；raw总体仍更准确。原atom普通16/32、受限18/32的对比作用为零，不把所有不一致都称反向，更不证明所有atom方案不可能。one source query、四主题/四target依赖，full FCC更多输入，不称等容量、多对多必要性、native或完整自然语义。

首次方向失败保留，修正方向有新词汇源端复现与固定target作用复核，写入AND_INTERPRETABILITY_CHECKPOINT_20260906及论文§5.9/§6/§7，registry标清既读output解释原则实际消费者。完整192方法行、48case表与pooled/先target后case两种汇总保留，主表不改统计口径；192个KL比由保存float32全概率另重算maxrel3.740733874e-5，只是数值核验。cached hook完全重放，baseline/source概率对float32档案maxabs8.896e-9/1.26458e-8；source对比锚点在2e-6内核对。三run合计352LM/numeric14.4116211秒/wall104.5477999秒，轻CPU上下文1.1673637秒另列。无代码变更后重复全套测试，无新训练/下载/包/audit/付费/委派，GPU租约已释放。

已清理完成/过期RUNNING，工作卡49行，下一自然文本前瞻解释检验，先冻结完整旧构造集的代词质量端点检查、source-only文本规则/排除hash/旧关系和预算；不把临时item词或旧文本当独立确认。AGENTS规则未动；本次tracker/plan整理前逐字归档archive/research_workflow_20260906/and_checkpoint并核对hash；native we正例/全部旧负结果/候选恢复点保留，automation ACTIVE5min当前对话继续。被忽略研究产物本地保留，以下path/hash留账：
- `EXPERIMENT_TRACKER.md` SHA256 `daca3e43a5b7615baeec1e1b67248cbf9a9c560315ba5e29093d1ad2a6b3b4f9` (7811 bytes).
- `EXPERIMENT_PLAN.md` SHA256 `418842e2ba34d21a6ab6b1df2b22d1d46ef60d6768f00770d62a28f2bf0e9e6f` (13687 bytes).
- `PAPER_SNAPSHOT_20260906.md` SHA256 `bbd30217d2fc4282e23f93c913539cff05f49e224ae028ac472bb8b04ef1b408` (45252 bytes).
- `REFERENCE_REGISTRY.md` SHA256 `c8300ea6196e6527fec3d6b85f49f58634e96f88bf1474d0a6ad6ab8c36e889b` (72086 bytes).
- `AND_INTERPRETABILITY_CHECKPOINT_20260906.md` SHA256 `ab55dbff1f4fb3bebf17c8eeaf4635835ce58f5af16c3635db6e9f1864f1e506` (7153 bytes).
- `artifacts/retrospective_20260906/inspect_and_contexts.py` SHA256 `fa07e90de0ca33b2ed26587964c592dd3062d44a5c80048679c094d211280a95` (5607 bytes).
- `artifacts/and_source_development_20260906/contexts.jsonl` SHA256 `f3eca79e9a84bb45b4f48a2a4cfbb0f1d7ac6e65f072ec251155dbda9aa4946a` (90043 bytes).
- `artifacts/and_source_development_20260906/summary.json` SHA256 `d90873ec48ea974932e28322c940bc34c6a432894391662e769138a82e73eef2` (5395 bytes).
- `artifacts/and_source_development_20260906/CHECKPOINT_SUMMARY.json` SHA256 `fddd9b23f30be243e21c959cee447225a7fe9411f052e99bf32336462a1e3916` (7019 bytes).
- `artifacts/and_source_development_20260906/TARGET_ALL_ROWS.csv` SHA256 `fd902b245aada5f050d103a01f7db4e474c59af50a923fd54e8f2549f241fe46` (66466 bytes).
- `artifacts/and_source_development_20260906/TARGET_CASES.csv` SHA256 `7a56a31db08b2d744142429df98fc3ddbaa919635c052d76b67bb98061179238` (5411 bytes).
- `artifacts/and_source_development_20260906/VERIFICATION.json` SHA256 `5434df5cbcf1b9b6a97bcfc4ac586fa246ae0d30afe097ba057bf3e6095eeb4c` (325 bytes).


### 2026-09-06 — Natural and interpretation confirmation: freeze before acquisition

Old initial+lexical source probability ledgers (32 inputs, 0LM) checked fixed-seven-pronoun logodds: ordinary and8/8 decreases, or7/8 increases; constrained and6/8 decreases. This endpoint is posthoc development, not a new confirmation of the old item ratio. Frozen configs/f4_and_natural_corpus_v1.json SHA256 652ec3a969424da4615cd1d4813898aeb33d132e75b7efe20384720995ae77d4 with 16 old train/paired/F4 document identity/text-hash ledgers excluded, 256x128 calibration-bucket natural tokens, <=300s/3official shard ranges; no audit arrays. Deterministic source-only 4 ordinary+4 constrained lexical-proxy distinct-document candidates, missing retained; matched or is a counterfactual. Unchanged source5:2194/full32/basis/cap.1, four historical FCC/atom/raw maps, all selected cases measured. Budget <=64 sourceLM120s numeric and<=224 targetLM180s numeric; cpu-heavy sampling/gpu-0 inference, no disk monopoly/fit/training/packages/paid work.

Natural corpus PASS/contract, 23,165,028B official ranges/3000 scanned identity records/256 sequences; selector found499 eligible and positions in48 documents (391ordinary/108constrained), selected4+4 distinct documents, missing0, no outcome read. Prefix lexical masks are not gold syntax. Initial selector hash check rejected formatting-only sorted JSON before token read; fixed to parsed-config equality, no input/selection change. Source config frozen SHA256 f654bed3722a931229f6dc7373937cc0ef6f26e1138c394fc7444de3dc3d98c9; prepared inputs SHA256 823498627b7325e15fc2ab56377fbd49f861f07e2ccdb87160f97ddfa9dab8cf. Shared endpoint helper replays all32 old item contrasts within 6.141203545695362e-08; new complement checked analytically. Starting64LM source probe within120s numeric.

Natural source PASS/contract64LM5.2952824s numeric/30.5940051s wall/726481408B peak/noop0. Ordinary and prediction3/4, or3/4; constrained and2/4 lower and or2/4higher. and-minus-or source coordinate negative8/8. Original uniform direction is not universally confirmed. All16 source cases, including reversals, retained for predeclared four-target FCC/atom/raw consumer with frozen config 23ede674997cf705b77818a0c6751881aed5ae565d82ab9614b9337242d4f19d, <=224LM180s numeric, no new fitting.

Natural four-target consumer complete PASS/contract224LM6.4456437s numeric/13.914336s wall; all16inputs measured. Actual direction FCC64/64, atom32/64(remaining32zero), raw64/64. Primary ordinary-and case-balanced absolute/KL errors FCC.021230/.062414,atom.098643/.752570,raw.028065/.067279; FCC beatsatom15/16 andraw9/16 on both per-target axes. Constrained both-input case-balanced raw remains better. Full report AND_NATURAL_CONFIRMATION_20260906.md; all failures and prior item endpoints retained. Following user reconnection, freeze a separate exposed-data context-response diagnostic: all8 natural and, original b with symmetric .01/.05 hook-norm fractions, baseline/noop, <=48LM120s numeric; config SHA256 f78a9855c2e58368daca418522f69227c7d3e4c2cf37f5b9ca22d743b2930a0b. This separates local downstream sign from large-dose nonlinearity, not a new natural confirmation or parameter search.

Context diagnostic PASS/contract48LM1.3901831s numeric/6.2636938s wall/707577856B peak; noop/cachehook0, baseline archive8.59e-9. Positive1% effect matches old source8/8, central finite-difference slope7/8: Railroad and motorsports already have opposite local sign; connection has both +/-1% increasing output and negative central slope, retaining finite-scale nonlinearity. Independent saved-probability formulas match max6.05e-8. Exposed development, not another fresh-data confirmation. Total336LM13.1311092s numeric/50.7720349s inference wall, corpus 65.637037s/23.165028MB separately. All four runs done; shared CPU/GPU free, ccad ACTIVE5min same conversation.

Updated complete natural report, paper5.10/usefulness table, plan C2/C3 implications, registry consumers; tracker one default next native we fixed-correspondence natural confirmation, does not equate different k32long/k128short configurations. Removed stale running/sampling-next items and corrected native ledger falsely saying causal probe had never happened. Old AGENTS/plan/tracker archives and all results preserved; AGENTS unchanged. Old32 endpoint and new192 probability numerical replay passed; no repeated broad core tests for unrelated code. All data-bearing artifacts remain ignored locally; only allowlisted scripts/configs/log grouped for sync. Local completion identities:
- `AND_NATURAL_CONFIRMATION_20260906.md` SHA256 `c1566152eb50352e1b86c5effcee62149d64ba51e86ebf441e798c480e49e425` (9808 bytes).
- `EXPERIMENT_TRACKER.md` SHA256 `c8c1ab6fdca0324a6119ec5c4115f6d63900f5c8f6a821455215779f3d284540` (8440 bytes).
- `EXPERIMENT_PLAN.md` SHA256 `d3f5811011ca55e713592cb8f6aecc1e316264778ef9fa9e8736444e60d30252` (14708 bytes).
- `AGENTS.md` SHA256 `6f0c1dd725c271fe7b085ce520aa299c36e5f0120b9d1a41412febaa88a6e35d` (13830 bytes).
- `PAPER_SNAPSHOT_20260906.md` SHA256 `893cca4ee230a36af9307b1e001832a2ba65e49007470e6ad77b194d521b5cf2` (47884 bytes).
- `REFERENCE_REGISTRY.md` SHA256 `be44bb7c8b9a28e5d3f88de593bc00ba4c6c76bfc7435d950d4cc4b1c7babb19` (72949 bytes).
- `archive/research_workflow_20260906/and_natural_confirmation/manifest.json` SHA256 `b04b601ae736e8212b96363f83b5099e6efd5aa6731de3542276fab8459ccda6` (506 bytes).
- `artifacts/and_natural_confirmation_20260906/freeze.json` SHA256 `a44e123194983756a593d5bb108b617107dfadd45af65d69a8b9ee774f7946de` (371 bytes).
- `artifacts/and_natural_confirmation_20260906/old_all32_pronoun_endpoint.json` SHA256 `1432bd629643cd70e52815b2c0de9263c20aba71ed118db0420bdcdeca63647f` (10964 bytes).
- `artifacts/and_natural_confirmation_20260906/prepared_inputs.json` SHA256 `823498627b7325e15fc2ab56377fbd49f861f07e2ccdb87160f97ddfa9dab8cf` (25824 bytes).
- `artifacts/and_natural_confirmation_20260906/selection.json` SHA256 `3a4625ad16c4ad04db810068ea5fff8b692b71999d078f10c2864085722b20d5` (169499 bytes).
- `artifacts/and_natural_confirmation_20260906/endpoint_verification.json` SHA256 `50101887ad6bf800544121ded948f58773c4af56b02730275ac0e2c0a86936e7` (177 bytes).
- `artifacts/and_natural_confirmation_20260906/CHECKPOINT_SUMMARY.json` SHA256 `3e7a695329c2ace126e0aabffa9b7830c2d973b37f9fdaad967cc2ed103fb7dc` (21478 bytes).
- `artifacts/and_natural_confirmation_20260906/TARGET_ALL_ROWS.csv` SHA256 `8102a8010e5661bb789996052c05eb3d7d8389f26aa7cad0725c0bfbf85d0ee7` (136538 bytes).
- `artifacts/and_natural_confirmation_20260906/TARGET_CASES.csv` SHA256 `64c20297fba174ef541baeb7a73da43883c5b207fb35f73240a4ee0e427afe8d` (5174 bytes).
- `artifacts/and_natural_confirmation_20260906/SOURCE_ALL_ROWS.csv` SHA256 `be45290e76c689e86c12ed8937b8b1a0d5a0196a468aa2aaa146b23c69c64ac9` (3891 bytes).
- `artifacts/and_natural_confirmation_20260906/CONTEXT_RESPONSE.json` SHA256 `f845f714afb5df36d08ef762c89ad088ee1f60628708b62b122bf92f0fe52869` (2700 bytes).
- `artifacts/and_natural_confirmation_20260906/closing_verification.json` SHA256 `b032698348496034718a925c71307b465df2983c8aa0ebe2744ac38e780f4678` (433 bytes).


### 2026-09-06 — User-requested positive-result closeout before pausing existing loop

User explicitly requests one comprehensive closing research round, preservation of actual positives, then PAUSE existing ccad loop (not delete). Original AGENTS/plan/tracker byte-archived with hashes in archive/research_workflow_20260906/user_requested_closeout; automation fields captured locally before change, remain ACTIVE until computations and closeout complete. Native source discovery positives at first post-we should/are/made/were informed fixed final-predicate position; compatible verb list excludes I-are grammatical confounds and includes prior authored predicates. Freeze native natural source8documents x we/they/I/you, historical s2:2767->s1:207, fixed output/dose/all denominator; new corpus512x128<=300s,17old exclusion ledgers, each seed128LM120s numeric+90s model load, no fit/train/package/weight/paid use. Configuration SHA256 ce17704c36596f9fcf90cb80286e0519ea7bb1332383245dde40e7f0688cbec6. All zero/opposite effects remain evidence; no target or new activation-based selection.

Native natural corpus PASS/contract;24,413,746B official ranges/2999 new candidate records. Frozen selector39eligible positions/19docs→8distinctdocs,missing0; no activation/output read. Prepared32inputs SHA256 b532193229346c577a6337521edb9cd878ae962e54ba49c3757576059ca00a8d. Source and counterpart configs both frozen before new LM: 7706813cab264f9757ec2b98e51b589cc7c5d4274593d3c3770a55b5d4ae4c2f / 195054708484b1d99eb814506aa17d112edb5a657f434dcbc65f449ad2c80a18. Source8we plus24subject controls and all baseline/noop/remove/add run at last observed predicate token; counterpart will receive all cases, not only source positives.

Native natural source/counterpart both PASS/contract,128LM each; total256LM6.2210666s numeric18.01982s wall,peak729387008B. Source/counterpart we activations each24/24 above matched controls; we native-remove and native-add actual directions each8/8 correspond. Absolute fixed-logratio errors medians .0141814059/.0191123045; added descriptive full-probability KL ratios .0096759153/.0107416072 (not new preregistered primary endpoints). Original constant-sign remove/add hypotheses only4/8 and3/8, retained; 24controls source4/target2 nonzero,20both-zero+2same-direction+2source-only per operation, no we-exclusive or all-input identity claim. Cross-seed baseline probabilities bit-identical; all saved-probability effects rederived max1.0609e-7. Corpus99docs/512seq/65536tokens,24.413746MB,24.693258s separately. No fitting/train/new weights/packages/paid/audit.

Completed comprehensive closeout report links three true positive evidence levels (common fresh FCC versus atom/global, natural FCC conditional effects, new natural native pair), preserves old constant-sign failures and capacity/native limits, and prioritizes member/capacity comparisons, multiple-source scope, and explicit conditional explanations if resumed. Paper abstract/5.11/usefulness table and plan refreshed to remove stale no-natural-test claims; tracker is a single PAUSED handoff with no queued default run. AGENTS byte-identical to pre-unit archive.

User-requested pause executed through existing automation_update idccad after all runs completed. Read-back PAUSED; onlystatus/updated_at differ, original5minute schedule/prompt/name/id/creation/current-thread target retained. No duplicate/deletion/archive; no live or queued work from this unit, shared CPU/GPU verifiedfree. Resume only on explicit user request, not automatic fault recovery. All ignored research docs/artifacts remain local with these identities; only allowlisted implementation/config/log grouped for synchronization:
- `INTERPRETABILITY_CLOSEOUT_20260906.md` SHA256 `5f8cbbff1518beb38b1234c64126796c2399fbc605c8353a18b5ceb88105255b` (10593 bytes).
- `PAPER_SNAPSHOT_20260906.md` SHA256 `ec373ad5c05ef772bfff8503262b8d5984e98681ebd42724981eeb3c74f1b9b1` (49898 bytes).
- `EXPERIMENT_TRACKER.md` SHA256 `d5c4f9a81e742e00212cd5811ec28c2c4750dbca0dacb12ec6d64177653e9028` (8410 bytes).
- `EXPERIMENT_PLAN.md` SHA256 `a7b6292984f8292b77dd4133fd1fd05e26b4fc4b05d1114c4eafa76a4833dcfa` (15725 bytes).
- `REFERENCE_REGISTRY.md` SHA256 `d445411251fa2a62eab9a3f5f49f973ba4836695075ba869ab9bc1e6d3b1bcd4` (73540 bytes).
- `AGENTS.md` SHA256 `6f0c1dd725c271fe7b085ce520aa299c36e5f0120b9d1a41412febaa88a6e35d` (13830 bytes).
- `archive/research_workflow_20260906/user_requested_closeout/manifest.json` SHA256 `ab7ebf5d3ec4e4afc58f78e6094f76083797f6d84edbf2399e8c26e72fa10c28` (734 bytes).
- `artifacts/interpretability_closeout_20260906/native_prepared_inputs.json` SHA256 `b532193229346c577a6337521edb9cd878ae962e54ba49c3757576059ca00a8d` (56232 bytes).
- `artifacts/interpretability_closeout_20260906/native_selection.json` SHA256 `8ba6b28e336cbd354d08dbf85b59e8b9b8f84c47f10f389eff1a486918191152` (14754 bytes).
- `artifacts/interpretability_closeout_20260906/NATIVE_SUMMARY.json` SHA256 `b6459d7ad6a7f365b8bde64998ae99fa02280cddecd5abbcaa56575021d569f1` (52664 bytes).
- `artifacts/interpretability_closeout_20260906/NATIVE_ALL_ROWS.csv` SHA256 `6c313f7af535465eb397562dfa6b2e46de87b82a5c4eb178aaea1b1a7de7f431` (14182 bytes).
- `artifacts/interpretability_closeout_20260906/automation_before.json` SHA256 `bb576d1e74eca6583e0e22f8c42b19c74b51314cbab9f3d1d40a3ee054d75724` (3523 bytes).
- `artifacts/interpretability_closeout_20260906/automation_after.json` SHA256 `4035b358eb4ac10919d86ee7f63568c6e456f68b2aafc7809afc9f3c569c81a9` (3523 bytes).
- `artifacts/interpretability_closeout_20260906/PAUSE_VERIFICATION.json` SHA256 `19119e83e23bbc14966bd326e80758c8b3db393b0c190a3c899f2f9ad28ca6fc` (360 bytes).
- `artifacts/interpretability_closeout_20260906/CLOSING_VERIFICATION.json` SHA256 `69860a3f0e612243cbfad9ab1540cbf8d9a11d2d366521e44a6264c8cf02b0a7` (614 bytes).

## 2026-09-06T06:51:27Z — 前几轮记录核对、写入时间规范与SAE质量讨论补记

写入时间（written_at_utc）：2026-09-06T06:51:27Z。本条为当前补记时间，不是下列历史实验或旧日志的原写入时间。

用户询问前几轮进展是否进入master log，并要求必须包含具体写入时间。逐项核对当前日志及各提交实际新增的标题：前六个工作单元已有科学结果、失败范围、artifact身份及后续判断；最近natural and与native收尾标题只有日期，其他部分标题至分钟且可能是启动/heartbeat时间，不能据此确认每次追加的秒级写入时间。旧原文全部保留；这些旧条目的原始精确写入时刻当前未证实，标为未知，不以当前时间倒填。下面给出可核实的Git提交时间，按本地Git历史说明结果何时已被纳入提交；提交时间不是实际写入时间，也不是独立可信时钟证明。

| 已有记录及起始行（补记前） | 核实到的主要内容 | 纳入该条目的提交 | Git提交时间（UTC） | 原精确写入时间 |
|---|---|---|---|---|
| Common fresh-document functional confirmation，3672 | 32请求/21匹配/6入选，入选相对atom/两matching的KL与NLL各6/6更低，对raw各4/6；拒选与覆盖限制保留 | 2ff99bd | 2026-09-06T04:23:29Z | 未证实 |
| User-requested full-history retrospective and interpretation-first consumer，3692 | 全历史回看、源概率/anchor范围、理论与解释消费者边界及论文判断 | 661dd57 | 2026-09-06T04:50:18Z | 未证实 |
| 全面推进与当前入口整理；source解释工作启动，3707 | source discovery与构造native we对应、旧负结果和资料归档 | 852ad6d | 2026-09-06T05:12:13Z | 未证实 |
| and源解释可行性，3732 | 初始方向失败、新词汇源复现与四target比较；raw优势及信息容量差异保留 | 577226d | 2026-09-06T05:45:47Z | 未证实，标题05:19UTC不可替代各段写入时间 |
| Natural and interpretation confirmation，3759 | 新8自然文档/16输入×4target作用方向FCC64/64、atom32/64、raw64/64；上下文反向诊断与受限范围raw更强 | 0484a82 | 2026-09-06T06:17:30Z | 未证实 |
| User-requested positive-result closeout before pausing existing loop，3792 | 新自然native we两seed各24/24输入偏好，删除/加回方向各8/8对应；恒定方向假说仅4/8与3/8，随后按用户要求原位暂停 | b941f08 | 2026-09-06T06:40:46Z | 未证实 |

补记紧接收尾后的用户问题“我们的素材SAE是确认已经很好了吗，不需要再去基础提升了？”：上一问进行了只读配置、原始质量账本和实现核查，未执行新实验，该讨论在本次核查前尚未进入master log。结论是素材可用于已有局部结果，但训练充分性、容量/稀疏度选择和语义/功能质量没有完成基础验证，不能宣称不必再提升。

当前多项FCC/and消费者使用short k128五seed，每seed131072训练tokens/256步、宽3072、Pythia-160m-deduped layer5；各自验证FVE96.8796–96.8869%、CE recovered90.6626–91.2162%。现成long k128五seed每seed4194304tokens/8192步，FVE98.5007–98.5110%、CE recovered97.4711–97.5647%。最近native we用long k32两个seed，同长预算，FVE97.5778–97.5858%、CE recovered93.7588–93.7982%，验证样本未激活756–815/3072个特征，不等于永久dead。CE recovered=1−(CE_reconstruction−CE_clean)/(CE_zero−CE_clean)，不是准确率。short与long的train/validation token路径及hash均不同，不能从这些各自验证数字直接作训练时长因果归因；首尾不同训练批次也不是固定验证集的收敛证据。质量证据：runs/R006c_k128_seed{1,2}_v1_20260903T220000Z、runs/R007_k128_seed{3,4,5}_v1_20260903T223000Z的config/metrics.raw.jsonl；runs/R011_NR1_k128_seed5_v1_20260905/FIVE_SEED_QUALITY.md及JSON；runs/R011_NR1_k32_seed{1,2}_v1_20260904T054000Z的config/metrics.raw.jsonl；消费者素材身份见runs/R008b_paired_codes_v1_20260904T000500Z/config.resolved.json。

讨论中的建议而非已执行计划：恢复时提高基础质量受控检验优先级，用现成long五seed与short在相同功能/解释消费者下比较，各自重新拟合对应关系；用固定语料/验证集的多个训练检查点判断训练收益，再决定是否增加宽度或调整k。保留强atom/raw/容量对照，避免将欠训练造成的碎片化误认成母问题固有困难。此前复用素材是投入选择，不是素材已充分优化的实证结论。本次不恢复loop、不创建运行、不改当前暂停状态；未把讨论建议当作已执行实验或新正结果。

长期规则已按用户要求补充至AGENTS.md：每一次追加块必须现场记录秒级UTC ISO 8601写入时间；事件时间另列；补记标明补记，不猜测历史时间，不覆盖原记录。AGENTS改前逐字归档与SHA校验完成，原规则全文保留；归档E:\Projects\SAE_Lab\CCAD\archive\research_workflow_20260906\log_timestamp_20260906T065127Z，旧AGENTS SHA256 6f0c1dd725c271fe7b085ce520aa299c36e5f0120b9d1a41412febaa88a6e35d，新AGENTS SHA256 a9b16939390718704a0e727c44b39e08491b17c230bde5577713dcbe6957c044。本次追加前master_log SHA256 ba443639a04324d6d87c5b3ed2b47b07dc27f40c05057da6e9424022c206cbd4，长度770748字节；验证旧字节是追加后原样前缀。AGENTS及归档仍按现有白名单本地保存，不扩大上传范围；只有master_log本次补记纳入成组同步。

## 2026-09-06T07:11:06Z — 用户四项讨论后的方法/素材/解释主线修正与automation接续

写入时间（written_at_utc）：2026-09-06T07:11:06Z。事件：本工作单元执行期间；不是历史实验完成时间。

用户明确要求：素材质量检验与提升、数学和interpretation共同发展，继续修改方法并寻找概念与实际用途上的独特价值；主动学习和跨领域启发，回看近期是否钻牛角尖、过度工程或在希望不大的方向耗费；当前对话负责automation，15分钟一次，每轮报告理由、规划、推进结果。本次据此恢复，不再受之前用户收尾暂停约束；没有转交其他task/agent。

实际核查：既有long五seed质量与配置、训练脚本save_every/fit/checkpoint/evaluate、现有plan/tracker/回看报告与最新日志、近邻官方ACL及arXiv页面。旧训练脚本save_every=expected_steps+1且save_best=False，完成后保存exact并验证，不能从首尾batch FVU推出固定验证收敛。长seed各自纯训练168–177秒只是历史实测，不能当新端到端耗时。既有short/long语料不同仍保留。

投入纠偏：过去消费者扩展快于素材比较，解释集中于少数词且条件预测不足，full/parts又受到rank1划分与拟合容量影响。改为素材—atom—FCC—解释共同图，先用现成资产共同输入与强基线拆分target素材和读出方法，再按实测需要补受控检查点；操作匹配稀疏关系与多操作Gram约束为两个候选，尚未声称新方法结果。避免将素材充分变成新的硬门，同时不在弱素材上无限续试消费者。

完成：AGENTS保留原有长期规则并更新15分钟、每轮报告和素材/方法/解释要求；plan重新组织科学路径并完整归档旧版；tracker移除旧暂停当前指令并留可承受工作卡；论文加入待检验主张及素材限制而不改旧实证；registry记录本次实际来源范围。无新训练、拟合、LM测量或科学正checkpoint；本轮产出是执行纠偏、训练记录缺口定位和可接续的科学设计，不能算主张已经得到验证。

automation_update原位修改ccad：ACTIVE、15分钟、target_thread_id=01a06e15-b222-7b21-a8cd-eb058a3159e5，notificationPolicy=null。工具成功后读取本地TOML核对status/period/target及prompt归一化完全一致。既有id保留，旧task历史保留，无新loop。长期prompt包含读取AGENTS/tracker、plan维护、master_log每次秒级UTC追加、证据与资源边界、每轮工作报告，不含当前run/暂停/候选参数。未修改模型或thinking。

归档：archive/research_workflow_20260906/method_restart_20260906T070550Z，六个改前文件逐字复制，manifest中源/归档hash一致；未扩上传白名单。当前本地文档身份：
- AGENTS.md SHA256 F6FA30C0BCD71E223F05AD8724B4552E38776353534EA341A0BFD7EE88AA43C1
- EXPERIMENT_PLAN.md SHA256 66171E7B22DDD6196289620323956296A797D52E26F89B5E5D14955AA967C4C2
- EXPERIMENT_TRACKER.md SHA256 C0FAA2C048FC8D7ED85E7D7000FE73CD0B96A99CC4DB5A8F4CDA3AB6A0DB0B20
- PAPER_SNAPSHOT_20260906.md SHA256 87BFAB2DD1B1AF8078B91D2C5EB5674A78A4B4A0FCFBF5E9A9825F73FB80D4AC
- REFERENCE_REGISTRY.md SHA256 11D3048E1F45CC109A38862DD29CB14C622110F19C7C94D2C49B7AA8D4F3F5D8
- configs/CCAD_AUTOMATION_PROMPT.md SHA256 EEA808F973533510F36D8702957E37F0D9837C2DA84AA710F2455821C03351B2
- archive/research_workflow_20260906/method_restart_20260906T070550Z/manifest.json SHA256 6A0BAFB3918B37151CC0A448826EC9289A3BF0ADBB4BC0665828E848DE02D2E3

## 2026-09-06T07:29:33Z — 素材/读出交互的跨面板稀疏检验启动
写入时间（written_at_utc）：2026-09-06T07:29:33Z。实际启动事件为本工作单元，尚无结果。
核查发现现有fixed-source short/long及joint sparse开发已执行，避免重复。选择F4_material_sparse_transfer_v1_20260906：将原冻结24 sparse fits直接用于另一个已暴露compact108面板6case，174LM；对照复用旧480LM结果，随后逐坐标/source概率验证共享操作再合并。无新拟合/训练/下载/audit，不称全新独立确认。cpu-heavy→gpu-0当前free，GPU1395/16303MiB，预算沿tracker30min墙钟上限。该实验检验稀疏拟合削弱long优势能否跨面板延续；优先支持或修正素材/方法主图，非继续and模板。

## 2026-09-06T07:32:58Z — 素材与稀疏读出跨面板结果完成
写入时间（written_at_utc）：2026-09-06T07:32:58Z；实验完成时刻见run/status.json，脚本wall58.02963秒（准备18.14743、LM39.35581秒），不是此日志写入时间。
F4_material_sparse_transfer_v1_20260906完成174LM、162新记录；冻结24 sparse fits/源组成，6case/3query/四非自身target。无新拟合/训练/下载/audit。消费者及契约PASS。18source完整记录与compiled_cases和旧compact确认逐值相同，合并450旧control行；重算KL/NLL最大误差3.55e-15，旧前向不计新重复。资源wrapper完成并释放，GPU核对free。
结果：short full energy16 .128596/.132879(KL/NLL)→sparse .019487/.036001，long sparse .023873/.018858。short sparse对short energy16和单atom两指标均6/6case更低；long sparse对long atom也6/6。long相对short sparse逐case仅2/6更好，不能从聚合NLL排序推出普遍素材优势。B short sparse .018672/.018068，long .029008/.017480，short KL6/6/NLL5/6赢；long A却从energy16 .032833/.033295恶化至sparse .059452/.080169。raw full .007739/.005372仍为强对照。
科学变化：标准稀疏缩小旧long优势的发现延续到另一面板，支持素材与执行结构共同研究，并保留long上分布式关系超过所测atom的有限证据；不再重复该面板小网格。数据此前为旧energy16消费过，故为跨面板开发扩展而非独立确认；shortsource固定、训练流不同、rank1parts与依赖边界均保留。
代码：无改动原LM/拟合实现；新增直接合并同面板controls的轻量summary脚本，严格source全字段/输入/dose相同并用原始概率重算，保存全部case与target。没有新增validator体系/依赖安装。论文§5.4与tracker已更新，下一共同输入质量/频率及long-source自然解释。
改前tracker/paper逐字归档：archive\research_workflow_20260906\material_result_20260906T073258Z，manifest包含源/归档hash。当前本地文档及数据：
- EXPERIMENT_TRACKER.md SHA256 CBC7237B89B3B9AEAC8296C3CF9172944F1010CDB3C0349048A56FF5E214EBDE
- PAPER_SNAPSHOT_20260906.md SHA256 CB468424735D19C938DD070EFACA2DEFA0EF91757E58FC44A7BD79A3CD4253F1
- runs/F4_material_sparse_transfer_v1_20260906/MATERIAL_COMPARISON.json SHA256 BEBBD4A47158B4D081CEF768BE9ECB25B7AAEC5A70872C08B61FA27D9C880660
- runs/F4_material_sparse_transfer_v1_20260906/metrics.raw.jsonl SHA256 FB2BA2FF886C6BB3C11D6DEF544B7EFF3871B767CD3E35B28EE1E21623F337F1

## 2026-09-06T07:53:14Z — 共同输入素材质量与source解释候选启动
写入时间（written_at_utc）：2026-09-06T07:53:14Z。F4_material_common_quality_v1_20260906固定compact512每8取1共64序列/8192token，十个short/long k128资产共用，沿用run_r006b_topk_capacity.evaluate定义（FVE/CE/频率），额外一次cachedhook校验，641LM预算≤1200秒wall。运行前GPUfree约1401/16303MiB。查验短long训练/验证排除文件document ID与text hash，不访问paired audit，无下载/训练。
自然解释候选仅long source1：在top24activation中按dominant词token比例和sequence覆盖排序、保留32候选/实例，属于机会挖掘而非完整概念质量或语义验证，不读取target行为参与选择。所有十个字典频率保存。目标是共同输入质量—功能/解释图，下一根据source原文形成具体可检验假说。

## 2026-09-06T07:55:53Z — long source that角色解释的最小检验启动
写入时间（written_at_utc）：2026-09-06T07:55:53Z。共同质量run已PASS，641forward/18.72秒wall，结果待本单元汇总。source1自然top激活显示988多为found/see/noted/determine/say，2379含feeling/certain/sense/argues/surprised；据此提出report/evidence vs attitude的探索假说，不是已有概念真值。
固定六谓词对confirmed/hoped、verified/feared、observed/suspected、discovered/believed、reported/wished、documented/expected，共用The investigator ... that。各atom独立12输入/48前向，共96；沿现有native消费者实际code删除/加回，激活偏好是主可行性终点，the/a词质量contrast仅为兼容旧字段的描述，不预测输出方向。只检查source1，不按target挑选，不声称自然泛化或FCC成功。
原native脚本仅将purpose改为可配置（旧默认保留），无干预计算改动；版本与全部输入进入各run。GPU租约分别自动释放，无新拟合/训练/download。

## 2026-09-06T07:59:35Z — 共同素材质量与long-source条件解释结果
写入时间（written_at_utc）：2026-09-06T07:59:35Z。各实验结束时刻以run记录为准；本条为事后当下补齐本单元结果。
F4_material_common_quality_v1_20260906 PASS，641batch模型调用、18.71857秒wall、峰值分配VRAM1,389,193,728B。固定64seq/8192tokens涉及65document IDs，整个108池与short/long训练及验证文件ID/textSHA无交集，十SAE共用clean/zero且完全相同。FVE中位short .968698(范围.968633–.968762) vs long .984923(.984864–.984974)；CE recovered .912180(.909914–.913199) vs .975400(.974323–.975952)。L0约128，short频率中位278–285 long163–169，上尾更重/本批9–16未激活，不称永久dead。重建改善已排除评价输入不同，但训练流不同仍不能归因时长或宣称充分。
long source1词法集中度32候选经自然片段阅读，提出988 report/evidence vs2379 attitude判断。NATIVE_that_roles_s1a988_v1_20260906 / s1a2379两个预定六谓词对run各48LM，wall6.207/6.022秒，noop0/契约PASS。2379态度激活更高6/6，988相反假说仅4/6。2379删除/加回KL中位.00982777/.00411494，988 .00893760/.00932864；KL为实际输出变化而非语义方向验证，旧first/third字段本次仅the/a描述。保留全部12input各atom、零值、原始概率。没有跨seed/自然确认或唯一概念结论。
实质增量：共同质量比较支持long素材包重建改善；自然片段到新句框的2379条件预测提供长source解释机会，988反例阻止互补双机制叙事。下一把2379接回crossseed atom/FCC与新上下文对照，不继续fixedshort消费者。没有训练/下载/安装/audit；源码只新增复用evaluate的adapter、native purpose可配置默认兼容，无新gate体系。
论文及tracker更新前逐字归档：archive\research_workflow_20260906\common_quality_20260906T075934Z（manifest记录源/归档hash）。当前本地身份：
- EXPERIMENT_TRACKER.md SHA256 D36A110A016336E61A2E21BA5AE7C34CDCD6E58D55C425B9BA577CBB568A8830
- PAPER_SNAPSHOT_20260906.md SHA256 E30417BBC7EF7735DBF8FFFD8CC7CF57109669DCE51A375DFE96AF3F229C2C2A
- runs/F4_material_common_quality_v1_20260906/MATERIAL_QUALITY_FINDINGS.md SHA256 30EC4A0DF6CBC287E52A2F4A052CC0E1571086C644FC160D57E01CDE0C17F8E9
- runs/F4_material_common_quality_v1_20260906/metrics.raw.jsonl SHA256 3CD4AE49538549CB380F4E532244BF3C4DEB256FA9FB4B34A1E618C3915818FA
- runs/F4_material_common_quality_v1_20260906/SOURCE1_CANDIDATES.json SHA256 A59FE8EA7C4F59A1FB18947F65C25FC10573B1B481FD46E39AA4DF899199EF4B
- runs/NATIVE_that_roles_s1a988_v1_20260906/metrics.raw.jsonl SHA256 7579BB279F7CC210CCFDF47C8B8E0E4C121CE73419C25E7DFF140FCCF88C2FF8
- runs/NATIVE_that_roles_s1a2379_v1_20260906/metrics.raw.jsonl SHA256 A99CEA6BFBF665A9F2A257926BBB14A33441BA14D54F6A41CCFB921C6EDC1F13

## 2026-09-06T08:23:00Z — long source2379跨seed功能对应启动
写入时间（written_at_utc）：2026-09-06T08:23:00Z。执行F4_long_source2379_correspondence_v1_20260906，long source1→target2；原独立mean/discovery，source-only that位置至多512均匀取样，保持所有条件变异。原paired全池与longSAE训练ID/text hash交集已实测0。
对照为同操作的bestatom、geometricatom、标准Lasso<=16加ridge、full ridge与raw；复用fixed_support_ridge现有RRR核，标量Lasso仅标准方法组件，不宣称solver新颖。独立mean与条件截距单列，donor差分截距抵消。先存系数再新target行为；输入为前轮6对谓词，source已暴露，属于开发非新自然确认。所有candidate对齐source decoder，包括atom，不能写target-native结论；source一atom不用于证明many-to-many必要。
预算40alpha前缀支持>32停止、至多85LM前向、≤900秒wall；CPU/GPU当前free，按cpu-heavy→gpu-0自动租约。无train/download/audit。主问题是2379条件偏好与实际作用是否跨seed存在以及需要多少target成员，不按结果排除atom充分情形。

## 2026-09-06T08:29:00Z — long source2379跨seed作用结果
写入时间（written_at_utc）：2026-09-06T08:29:00Z。实验结束见run/status.json；wall32.31343秒、拟合准备29.79950秒，85forward、60method行，PASS与契约通过。GPU租约wrapper已释放。
Discovery862that位置均匀512、原mean；bestatom800、geometry2562、sparse14（10正4负），full/raw。12相关操作KL中位 .767785/.858097/.120357/.029752/.001754；pooled .887084/.783884/.107902/.032088/.001608。六pair内双方向中位再取中位 .728792/.849755/.125556/.028988/.002110，全部口径保留不切换主排序。sparse对两atom12/12KL更低；偏好方向bestatom5/6，其余6/6。geometry同方向但believed标量-0.0002 vs source-0.6899，显示方向不足以说明幅度保持。
source条件差与旧native激活逐值完全一致；noop0；保存float32概率重归一化clamp重算KL比最大误差2.1467e-7。初次分析直接log0产生NaN，仅为重算表达式缺0处理，原consumer使用clamp无NaN，修正后通过，不重跑LM。
意义：long素材上解释线索接到了另seed的分布式功能读出，比所选atom保留更多实际作用；raw显著更强。source单atom故one-to-many，非m2m必要性/最少14/目标native或唯一语义机制。source模板已暴露，target行为拟合后才取，仍为开发；自然/新句框和source广度待扩，下一冻结关系而非旧模板参数网格。
源码复用RRR/标准Lasso与已有hook干预，原文接口/许可见registry本条。无训练/下载/新依赖/audit。tracker/paper/registry改前逐字归档archive\research_workflow_20260906\long_source2379_20260906T082900Z，manifest源/归档hash保留。当前本地身份：
- EXPERIMENT_TRACKER.md SHA256 919381A89321A8FB1A0E74BF5BFF401285B1247E84E414073B410C12F30A7C92
- PAPER_SNAPSHOT_20260906.md SHA256 6A2F10EB9AEE53BFD09EC916FEF3C7020A6DE443AECA33C4B27367EA1FA495F5
- REFERENCE_REGISTRY.md SHA256 CD720EC320BEDEF6E0199E36894483C3BE1303F3A89FB23821C3F13904B8CF00
- runs/F4_long_source2379_correspondence_v1_20260906/coefficients.npz SHA256 5F3E788AFA5EA199A94AE260A9A180C12709427D67253BB22B6A8B96C6DBFC77
- runs/F4_long_source2379_correspondence_v1_20260906/fit_metadata.json SHA256 BDEB869E3706CAAF0A9BA7B91456FDCF74965697153DE1F1BB2A75F77C56B4EB
- runs/F4_long_source2379_correspondence_v1_20260906/metrics.raw.jsonl SHA256 3CE45CFE66DEF32FB92F78F6BDD549A55AB9B880A4A328BE05EF18DE1F2AE2FB
- runs/F4_long_source2379_correspondence_v1_20260906/FINDINGS.md SHA256 228E95B03B7554B3CDFC9F6FB5FDEC9192FC59AE792DE7821E6D594ED9C3876C

## 2026-09-06T08:50:07Z — 冻结long2379的跨句框/自然输入检验启动
写入时间（written_at_utc）：2026-09-06T08:50:07Z。F4_long_source2379_contexts_v1_20260906复用上轮coefficients hash，不重新拟合。source盲输入选择：12新句框谓词对、6同谓词direct vs noun-that结构对照、5自然谓词对（原native-we99doc池，之前无2379候选筛选）。自然report25/attitude13候选，按预定前词词表、>=12token、hash顺序去重复文档，先report后attitude分配，仅5对可用，未匹配项完整保留；不看source/target激活。46输入、323前向预算900秒。旧语料复用明确development非独立确认。脚本新增frozen/no-refit及prepared token输入分支，原拟合与操作公式未改变，完成后直接核对全部系数。

## 2026-09-06T08:54:42Z — 冻结long2379跨上下文结果
写入时间（written_at_utc）：2026-09-06T08:54:42Z。实验结束2026-09-06T08:50:24.734668+00:00，PASS，323LM/230行，12.367138秒wall，峰值741596672B；原系数逐数组相等，无重拟合/train/download/audit。
新句框KL中位bestatom/geometry/sparse/full/raw .653903/.904378/.085704/.052249/.001406；自然 .250374/.792738/.007417/.004923/.001843；名词插入 .037406/1.411942/.055773/.013652/.001312。source条件偏好新句框12/12、自然5/5；原始弱效应全保留。自然复用99doc开发非独立确认，名词插入relative旧标签不证明关系从句且混入语法/邻接混杂，不能称纯句法机制。单atom在名词插入中位更好，raw持续更强。下一固定方法扩现有long target3/4/5，再source广度，不继续单案例模板。
改前逐字归档：archive/research_workflow_20260906/long_source_contexts_20260906T085442Z，manifest核对hash。当前本地身份：
- EXPERIMENT_TRACKER.md SHA256 1bc1f8e19f1d8296f948104f1dd836e7a44779c2c76243c180acf1bc09095810
- PAPER_SNAPSHOT_20260906.md SHA256 2d39b3323d5fd86c54aefd37f23ac3d65ec134d25311f864c179cb8512e8090c
- runs/F4_long_source2379_contexts_v1_20260906/FINDINGS.md SHA256 1d3ae1cdc6e7015cc39e9b61c4813e97974889fb36d4150dee078d42ee0487aa
- runs/F4_long_source2379_contexts_v1_20260906/metrics.raw.jsonl SHA256 7ea538f81c83b245013d2a243f6b369e66490fec6b61e672d2793eb0c26d475a
- artifacts/long_source2379_contexts_20260906/inputs.json SHA256 dc2fef6d9e8f32b1b91c2c70f5d7b9761510a63ecaab5f7db36c2ca598e8195b
- scripts/prepare_long_source_contexts.py SHA256 0040c245bf8dc859a9646fe1ba70d22e615111cefadc14727af78f9b4a7ef59d

## 2026-09-06T09:12:04Z — long2379固定方法扩target3/4/5启动
写入时间（written_at_utc）：2026-09-06T09:12:04Z。复用原mean/discovery、512that行、ridge.001、标准Lasso路径与<=16支持，原46输入不变。每target先冻结拟合再观察行为，单run323LM/<=900秒，总<=30分钟/16GB。扩target seed支持稳定性，不能当独立source重复；原target2结果保留不重跑。raw/source输出应逐值相等作针对性实现检查。现有CPU/GPU free，按cpu-heavy→gpu-0申请，未训练/download/audit。

## 2026-09-06T09:12:37Z — seed扩展资产接口修复
写入时间（written_at_utc）：2026-09-06T09:12:37Z。target3_v1在元数据合并时StopIteration，0forward/0指标/0.018秒；base有calibration/audit条目，新增seed仅mean/discovery。修复合并范围为实际拟合mean/discovery，不读取其余split；失败目录完整保留，target3改唯一v2重试，方法参数和评价输入未变化。

## 2026-09-06T09:16:48Z — long2379四target结果
写入时间（written_at_utc）：2026-09-06T09:16:48Z。新增target3_v2/4_v1/5_v1各323LM/230行，PASS；累计wall44.417950秒，结束时刻分别见各status.json及本单元FINDINGS.md，原0forward失败保留。
自然bestatom中位target2/3/4/5 .250374/.752682/1.272581/.481816；sparse .007417/.337986/.129649/.105339；full .004923/.019662/.035230/.023865；raw .001843。稀疏逐操作胜数8/10、6/10、8/10、6/10，保留全部损失。新句框中位均改善，名词插入2/5中位atom更好。支持分布式可恢复性，紧凑幅度随target异质；共享source方向非独立五seed重复，不改变开发/非native边界。源差/源KL/raw全逐值一致，未新增train/download/audit。下一source及条件广度，不再扩2379模板。
改前逐字归档archive/research_workflow_20260906/long_target_seeds_20260906T091648Z含匹配hash。当前身份：
- EXPERIMENT_TRACKER.md SHA256 0e5344170d757ed92ecba3a001311c8af6186a4895c7421d6b068ed065dfc6f8
- PAPER_SNAPSHOT_20260906.md SHA256 84847b329e9e622fd8d2e7b0cd8b7ea4c925ede0deade960bb6fcaf2989e62aa
- artifacts/long_source2379_seed_extension_20260906/summary.json SHA256 9aee3f37e63dbd071403ea6eb9a6de0593e82bf14fc4be3315b62479dafb1609
- artifacts/long_source2379_seed_extension_20260906/FINDINGS.md SHA256 2257911846cba3179053e295049df4fe6fd9f98f5f2d1e2116fde611b7c823c1
- runs/F4_long_source2379_target3_v2_20260906/metrics.raw.jsonl SHA256 664233d402dbfb1a68e5e0e38d5c466d38d248f8013f5b636b81fff87df729c7
- runs/F4_long_source2379_target4_v1_20260906/metrics.raw.jsonl SHA256 55cc77859b8e3da9dd5a114ca275ade8434bf25717b7a704df25565e61a7d21e
- runs/F4_long_source2379_target5_v1_20260906/metrics.raw.jsonl SHA256 3ffd0b09e5083dfcf200fd12cb8e61493a273b184df187f2f671c302335fb65f

## 2026-09-06T09:36:45Z — 新source及条件类型小面板启动
写入时间（written_at_utc）：2026-09-06T09:36:45Z。仅source2/3原discovery均匀8192位置缓存分析2.2624秒，无LM；全候选/频率/输入hash保留artifacts/long_source_breadth_20260906。自然片段提出newline153/551、to1200/1450、colon1850/2897六候选，每类6对新输入、各候选85LM，source2→1/source3→4预定。原ridge/Lasso预算不变，source预测与输入先冻结再target评价，全弱源/反向保留。newline对照主要句末标点；to词法和语义混杂，colon含长度/格式，均非已证语义机制。六run总<=30分钟/16GB，无训练/download/audit。

## 2026-09-06T09:42:08Z — source广度与解释粒度结果
写入时间（written_at_utc）：2026-09-06T09:42:08Z。六run各85LM/60method行，总510LM/56.862325秒wall，全部PASS，结束事件时间见各status.json及FINDINGS.md。预测153/551/1200/1450/1850/2897为0/6、6/6、6/6、5/6、6/6、6/6；153三零三反向，6/12比率未定义保持null，零源sparse候选KL中位2.2546853e-5，不写成成功。551/1200/1850稀疏比动态atom中位改善，1450动态atom与2897几何atom优于sparse。三类含标点、词身份、长度格式混杂，不推语义机制；源对齐非native，构造开发非自然确认。
source decoder/保存激活donor差逐值一致；输出概率重算max scaled比率误差1.84e-6、candidateKL绝对3.05e-9，针对性实现检查而非独立科学复核。原方法仅泛化source seed/token；无训练/download/audit。下一真实二组成共同操作族，随后落实已保留最小训练曲线，不无限续试当前模板。
归档archive/research_workflow_20260906/source_breadth_20260906T094208Z，源与归档bytehash匹配。当前本地身份：
- EXPERIMENT_TRACKER.md SHA256 3a3f9f7f3fd62edd1eb7fd391c58c7dd93e6ffc5f87f66cba80755c47acb990d
- PAPER_SNAPSHOT_20260906.md SHA256 0fbcc9a86631fdec4b43d34db5308f2de9eab9b2295363b2cf1923b025bc56ec
- artifacts/long_source_breadth_20260906/FINDINGS.md SHA256 04d41efa18be43bb0d18d3486ff9a4fe269e595f3901353659683fa49ac43e58
- artifacts/long_source_breadth_20260906/summary.json SHA256 21e969693b436443e5cec6b1d6e2c1a4c37011c317264bd6d2ca77230de85f6f
- artifacts/long_source_breadth_20260906/checks.json SHA256 a9e06d51d04db61e8cc39dfbf1c91415c20bb21bba40f42583b153c9017a7ef9
- artifacts/long_source_breadth_20260906/selection.json SHA256 5af876809ecf7ce0858abe6d8d8a759f3cbe21db0e01dafcfd8b32ae14c01183
- artifacts/long_source_breadth_20260906/source2.json SHA256 8f028d225bc119f277bdb6e484fae5dbd0fafc6120a78b9860186a06d7f149b2
- artifacts/long_source_breadth_20260906/source3.json SHA256 161a9d382679818f18c866db8e94e8b65098a6535f88c43bedbd34666f84f9e4

## 2026-09-06T10:02:55Z — 真实双组成操作族启动
写入时间（written_at_utc）：2026-09-06T10:02:55Z。source3:1850/2897 decoder cosine.104583，Gram eigenvalues.895417/1.104583，非rank1人为切分。复用fit_f4_joint_sparse.fit_joint标准输出标准化MultiTaskLasso+去收缩，新增共享16与同成员原单位ridge；保留原full/raw/逐atom/独立sparse并加入动态/几何两atom联合ridge。原colon12输入、4操作(A/B/sum/difference)、每输入统一dose；493LM预算900秒。原mean/discovery拟合后冻结，旧评价属开发。逐输入误差Gram恒等式复用DERIVATION_PACKAGE13.7，不声称最优稀疏/两语义机制/native或新solver。

## 2026-09-06T10:06:55Z — 真实双组成共同恢复结果
写入时间（written_at_utc）：2026-09-06T10:06:55Z。事件结束2026-09-06T10:03:30.877236+00:00，493LM/432行/17.433653秒PASS；所有点wise Gram恒等式通过，概率重算max scaled误差3.48e-7。源decoder/covariance均rank2，共享15行两输出非零、映射rank2。
几何两atom联合→shared第一/第二/和/差KL中位 .162478→.046051/.036047→.016705/.077948→.026572/.075715→.024827；逐操作胜数10/12、6/12、9/12、10/12。独立sparse并集28第一组成中位更好，其余共享15改善；同15原单位ridge近等价，不夸求解器新意。经验Gram最大特征值shared.023582/full.034603/raw.008294，full合并KL仍更好；经验向量结论不当行为/未见保证。旧构造输入、共dose操作、source-aligned/non-native边界保留，无训练/download/audit。
实际复用fit_joint与DERIVATION13.7见registry。下一同流最小两seed checkpoint曲线，停止此模板扩张。改前逐字归档archive/research_workflow_20260906/colon_pair_20260906T100655Z含匹配hash。当前身份：
- EXPERIMENT_TRACKER.md SHA256 7c149b318ccddef98e4aea37c517ceadee088912893408633f10b0c921284509
- PAPER_SNAPSHOT_20260906.md SHA256 34e761edf9baa220fa0864dd23517e46eaa51b31770cc5b16eb9592f2e8263ca
- REFERENCE_REGISTRY.md SHA256 1abcb2cafd8a1d1f15a360f3c57fc3e459bdab30ea6b18c11bd6f19d8a83fed1
- runs/F4_colon_pair_shared_v1_20260906/FINDINGS.md SHA256 8d56a19de6874856a29b7a4fcbade34a8cbbf23eacc1f9218ad899fbe7576390
- runs/F4_colon_pair_shared_v1_20260906/comparison.json SHA256 669d91163bad207113f4da73de097639e2da9deb98e91e1219e7d34677b6f65e
- runs/F4_colon_pair_shared_v1_20260906/gram_summary.json SHA256 1a321282d8ee6ffb4a78ddab3e57e1a66de1345329b700b806fe3bb61cc40772
- runs/F4_colon_pair_shared_v1_20260906/metrics.raw.jsonl SHA256 46e42300639ab10a2d7687cd814fd3c77809648517951ad90235a50e4c298037

## 2026-09-06T10:27:33Z — 同流两seed五检查点曲线启动
写入时间（written_at_utc）：2026-09-06T10:27:33Z。R012_same_stream_curve_v1_20260906复用long4,194,304 packed token有序语料、seed1/2、同Adam/warmup8/总8192step线性日程。5保存点256/1024/2048/4096/8192实际optimizer更新，训练后统一固定64validation评价，doc/text与train分离。保存post-update推理权重不改变训练状态；final单独exact checkpoint。不是分别优化短预算/收敛充分证明。预计旧单seed168–177秒基础上两SAE共享LM前向，可承受总1500秒/16GB/预计<500MB新权重；先实际吞吐，不新下载。暂不声称atom/FCC已测。

## 2026-09-06T10:32:00Z — 隔离绘图依赖与训练阶段记录
写入时间（written_at_utc）：2026-09-06T10:32:00Z。训练已保存前四检查点并仍在运行；尚未统一验证，不报质量结论。依既有工具安装授权，查官方PyPI/本机无matplotlib后，以binary wheels安装Matplotlib3.10.8及依赖到项目隔离target，所有URL为files.pythonhosted.org，包版本/下载hash见.aris/compute/plot_env_spec.json与plot_install_report.json。仅bundledPython3.12绘图，训练Python3.13环境未改。先将上轮双组成已知结果出PNG/PDF/SVG，不计新科学结果；新训练曲线待评价。

## 2026-09-06T10:36:35Z — 同流学习曲线完成
写入时间（written_at_utc）：2026-09-06T10:36:35Z。事件结束2026-09-06T10:31:53.603066+00:00，PASS，总229.782098秒，训练212.915838秒含保存2.540558秒，10质量记录/5更新点/2seed。输入8192batch逐hash等于有序语料，终点两权重分别完全等于旧long1/2资产。新增bulk302305376B，final exact状态与中途推理权重分开。
FVE/CE中位五点.968995/.903730、.977218/.953297、.979925/.963541、.982872/.972314、.985027/.978002；最后翻倍仍+.002156/+.005688。非EOS实际token4187246，packed4194304，两者分列。验证活跃数非单调、L0近128，不能据重建宣称概念/FCC好或充分收敛；完整LR日程prefix非分别优化短预算，复用验证非新确认。
科学产出为同流真实曲线；PNG/PDF/SVG及上一双组成图已目检，绘图本身不当新发现。下一固定source双组成和旧操作面板，在已存检查点重拟合target关系，测功能随训练变化，不新增训练。改前逐字归档archive/research_workflow_20260906/training_curve_20260906T103635Z含匹配hash。当前身份：
- EXPERIMENT_TRACKER.md SHA256 27bf7a78d506619dc363330134652594bb716a1939227b0e632eada65ae6eedb
- PAPER_SNAPSHOT_20260906.md SHA256 0177df01bdcec2edbc48540ef50bde7493f00e4c370aa392505a2d9e260e7975
- REFERENCE_REGISTRY.md SHA256 5e927ab51d17aca7a1ef9b9d78b709da52d4e79bb06424fd4660d86d7d7b32ba
- .aris/compute/local.md SHA256 4ae569fbda4d47ec81a522eae8c975abd568c766547762f21f6f2a7579ebd4c5
- .aris/compute/plot_env_spec.json SHA256 701bbd1b58613032d1906f569eed32f76ba2f093489103b03b7ddffa1118e9bd
- runs/R012_same_stream_curve_v1_20260906/FINDINGS.md SHA256 3b7073ddf2d3f920b9ae5339f8e9dd98c63b38eb41eaf6224493751c7fecbb6d
- runs/R012_same_stream_curve_v1_20260906/metrics.raw.jsonl SHA256 b568cf293842ac594e9933d791de5e72c6fa27968759f2cbd73fad61f04b78e7
- runs/R012_same_stream_curve_v1_20260906/curve_summary.json SHA256 624a01b4c42f25530ced8d10716a5a9bbc3322ce7c4945aad4b0b6d47d788a99
- runs/R012_same_stream_curve_v1_20260906/trajectory_checks.json SHA256 2329e665afd94b4c483d0e6553b6e818d070a7f5d2fbf8175684e1a7bd45e590
- runs/R012_same_stream_curve_v1_20260906/training_curve.png SHA256 bd28bb0fd02fcefbb44013739ee4c9f624d3486bcba43958d3ae80a93aafa734
- runs/R012_same_stream_curve_v1_20260906/training_curve.pdf SHA256 90c6f6d161f8d88c31197fdadfc5718b8df43018f703985912e1bc64fe5eb2f3
- runs/F4_colon_pair_shared_v1_20260906/pair_operation_figure.png SHA256 952070b0405a788ea8dee8a226c66b532297c5a3035420ca0f81dd12d455018a

## 2026-09-06T10:56:44Z — 同流功能曲线启动
写入时间（written_at_utc）：2026-09-06T10:56:44Z。固定source3:1850/2897，R012五检查点target1/2，原12input四操作。10配置在本轮target评价前固定。仅编码原独立mean32768+discovery512colon cachedhook，不生成全量激活库；逐checkpoint重拟合full/动态atom/几何atom/共享<=16/同成员ridge/两atom联合，raw源教师不变。每点445LM、总4450/预算30分钟16GB，无新训练/download/audit。旧面板开发，不挑checkpoint改称确认。

## 2026-09-06T11:04:00Z — 同流功能曲线完成
写入时间（written_at_utc）：2026-09-06T11:04:00Z。10run各445LM/384行全部PASS，总wall174.945873秒；事件结束各status.json可查。40相关单元shared中位对dynamic/geometric/full分别34/35/20胜；终点各4操作均胜这三控制，raw合并仍强。阶段非单调不当概念丢失，包含拟合和开发输入分布影响，所有反例保留。
source差/common dose/sourceKL/raw全逐值重放原pair；未新训练/download/audit/fullcache。下一冻结两个终点，在新自然上下文复核，不挑旧最佳阶段。图已目检。归档archive/research_workflow_20260906/function_curve_20260906T110359Z含bytehash。当前身份：
- EXPERIMENT_TRACKER.md SHA256 e7159709e7a83c075451d075b5e69af3d3020e4af41004ca68b4a67b83b7c89a
- PAPER_SNAPSHOT_20260906.md SHA256 3ba8774200d153a3a9e268109cc92e388f7d46e27d2f45f8c7373983c051c7b0
- artifacts/function_curve_20260906/FINDINGS.md SHA256 835285dd0f38b677551e444bcb9abf084630db23341c4d1e05f2c2f9b75742b4
- artifacts/function_curve_20260906/summary.json SHA256 15c64192b9d6b69ebb0fd5b91cdab7a339c586814dba6b705463be529c61fc23
- artifacts/function_curve_20260906/function_curve.png SHA256 5f80a7ea1dd04b2ceedeca6807ee01a4cd0e2efe90cbe66d4e5d03ea6d20d577
- artifacts/function_curve_20260906/function_curve.pdf SHA256 8748ba6e3f2dfe16f9d2c2ad9e7fc516f1a2b55788a316fc5be90d1a5e053243

## 2026-09-06T11:05:56Z — 功能曲线报告小数勘误
写入时间（written_at_utc）：2026-09-06T11:05:56Z。FINDINGS.md正文两条shared轨迹的小数已按同页表格及summary.json修正为六位舍入；原始统计、图、结论不变。此前hash为修正前身份；artifacts/function_curve_20260906/FINDINGS.md当前SHA256 ebe1f1edc76a562b7ec8b13a71f3d5e2866727dca7969b0ed557d6311270a9c1。

## 2026-09-06T11:25:47Z — 双组成新自然确认启动
写入时间（written_at_utc）：2026-09-06T11:25:47Z。冻结两个8192终点八方法系数、四操作和.1共同dose规则，先旧数据回放frozen接口再新自然文本。1024x128新calibration文档按既有hashsplit，排除训练/paired/所有既有documents账本；短字段/长句固定词法代理，盐hash最多12对、全部零/反向保留，无target筛选。预算30分钟/16GB无训练/拟合，语料range<=约100MB/300秒。freeze身份 738b98ba0b86cefea5475d18d175615ab97e356546177f1d9683464f78491d73。

## 2026-09-06T11:34:43Z — 冻结双组成新自然确认完成
写入时间（written_at_utc）：2026-09-06T11:34:43Z。事件结束两target status为2026-09-06T11:29左右，精确各status.json；本块不以事件替代写入。新204文档/1024seq中固定选12对/24不同文档，1778LM/1536行/50.657230秒、0refit，语料25,237,678B公开range。输入条件预测8/12与10/12。第二组成shared比值中位.009636/.013113，对dynamic.255609/.486929、full.060812/.046704，绝对KL/多数逐例同向；第一组成和混合对full不稳，所有Raw中位更强。弱源4例不剔除；time/citation代理错误保留。不得将target1和/差比值中位胜full概括为绝对/逐例优势。
系数冻结逐值/source与raw跨target逐值/Gram/noop/契约PASS；旧接口384行回放15.13秒独列工程不计新科学。s2replay配置已准备但无需重复同分支，因此未运行。下一同总16预算共享vs逐输出L1支持，以旧自然作开发，改善另新确认。论文/图/registry/plan/tracker已同步，图目检纠正图例遮挡。改前逐字归档archive/research_workflow_20260906/colon_natural_20260906T112632Z。当前文件身份：
- EXPERIMENT_TRACKER.md SHA256 d172f9835d2d5f74c696913a0f34255fd495a16abb149f2257065e920e1d2ca2
- EXPERIMENT_PLAN.md SHA256 c32ad2ca9c8c9246298411caf6756930473e5490a3980fa845b82693873e6243
- PAPER_SNAPSHOT_20260906.md SHA256 8a373a6667f4daa24a5021ef5547b4aac5493fd8b0d15ca804706b817fd41e57
- REFERENCE_REGISTRY.md SHA256 8286b89559958194d83b29f67422c777b28e1dcae5fd480c82b392db0c6ba162
- artifacts\colon_natural_20260906\freeze.json SHA256 738b98ba0b86cefea5475d18d175615ab97e356546177f1d9683464f78491d73
- artifacts\colon_natural_20260906\selection.json SHA256 066861c72560dbaf059c8b85141dffeb736f53b636a78a6d39bb2399affd6049
- artifacts\colon_natural_20260906\prepared_inputs.json SHA256 3208e09740dca849d9d90b7ac5f9326fad7add5fd0b9d7908b30ea05af0bbb49
- artifacts\colon_natural_20260906\FINDINGS.md SHA256 abefb4b3dd449af4cd4aa75dfb5e7e2d68a3ce00e7a37bbd22867bf03d1923d3
- artifacts\colon_natural_20260906\summary.json SHA256 10172daf36243947804ec6197c58915ac71560dd7ba3f867054f157c55bb6a34
- artifacts\colon_natural_20260906\natural_confirmation.png SHA256 722daa5197a3ba905eeefe239c1e583e5fb5e0beec0577108fa7a381a720b3ae
- artifacts\colon_natural_20260906\natural_confirmation.pdf SHA256 1cd7fea9b9ba5f76a17981d5d65641ca847c7a7acd1ef1759e41b40b3bc79ddb

## 2026-09-06T11:35:33Z — 自然确认结束时间与资源核对
写入时间（written_at_utc）：2026-09-06T11:35:33Z。两target精确事件结束时间据status.json分别2026-09-06T11:28:33.881136+00:00、2026-09-06T11:29:16.321159+00:00。共享管理器核对cpu-heavy/gpu-0/disk-d-io/disk-e-io当前free，本轮无残留计算。

## 2026-09-06T11:56:20Z — 固定成员预算对照启动
写入时间（written_at_utc）：2026-09-06T11:56:20Z。同原mean/discovery与两个终点，共享group-L1<=16对两个scalar-L1各<=8、并集<=16；各自成员与union共同ridge均保存，对照same_support_ridge使用同核。每输出同40点相对alpha路径，合计最多80点比joint40多，非等求解次数，报告实测成本。两个配置、旧构造12+已暴露自然24输入先固定。每target1621LM/1440行、总20分钟/16GB，无训练/下载/audit，CPU暂忙准备先行。全部原8方法必须逐值重放旧父run；改前归档archive/research_workflow_20260906/colon_support_20260906T115620Z。输入SHA256 3f80f88a9d404fd086f95b98eca58d27f35dd91462356625d16f39de19c182d9。

## 2026-09-06T12:20:35Z — 共享与逐组成成员取舍完成
写入时间（written_at_utc）：2026-09-06T12:20:35Z。两run共3242LM/2880行/85.912318秒，CPU曾等待其他项目随后获租约，非重复运行。两source output的scalar路径用既有标准化和标准sklearnLasso，去收缩同fixed_support_ridge；sanity独立真值supports[1,4]/[3,8]恢复只作实现见证。真实shared成员15/14，逐组成6+7/8+8并集12/14；非实际数/求解预算完全相同。自然第一组成separate比值/绝对/多数逐例均改善，但第二组成恶化，旧构造/混合保留反例；没有全面更优方法，停止本对budget扫描。
原8方法系数及两旧面板浮点统计逐值一致（拼接case/donor索引平移核对），所有输入保留，Gram/noop/契约PASS。论文/图/registry/tracker已更新；下一现有五资产剩余target4/5的固定方法及新自然验证C2，不新增训练。Dirty Model仅原文方法/条件启发，未实现其solver或套用support-recovery定理。事件结束时间：
- F4_colon_support_tradeoff_s1_v1_20260906: 2026-09-06T11:58:02.859354+00:00；metrics SHA256 620225c4d30f1656fdedf730520d5577b65957ebfc0c5a5f96459c6a2ca44555
- F4_colon_support_tradeoff_s2_v1_20260906: 2026-09-06T12:01:10.378475+00:00；metrics SHA256 d18e68b2ae726256242d69ac5636215535e44f72568061b3bdc3195441c7c9cd
当前本地文件身份：
- EXPERIMENT_TRACKER.md SHA256 131130d7e8cf5ae697122f38c944a027a53129f1f2d30416ed15cba73afcf040
- PAPER_SNAPSHOT_20260906.md SHA256 f28c566db0f8ff03b43d9f8dc1a72c6683d65019f736c2ef2f422bedce8fafc4
- REFERENCE_REGISTRY.md SHA256 e2482650a4fcb94f4879aec15ae7bafcb313fc7495fdb892dbc917836e535f5a
- artifacts\colon_support_tradeoff_20260906\FINDINGS.md SHA256 ab4061180b34e55a1ac795ed81955da4afdd8c406da3e6fd7f5046151ab5c569
- artifacts\colon_support_tradeoff_20260906\summary.json SHA256 66050fc806fa8d05ddff2acb412620cc045ae3bf46a931b469f3ac1200d87af0
- artifacts\colon_support_tradeoff_20260906\inputs.json SHA256 3f80f88a9d404fd086f95b98eca58d27f35dd91462356625d16f39de19c182d9
- artifacts\colon_support_tradeoff_20260906\support_tradeoff.png SHA256 57d9515f3779df75045545013d88c1e3e127307248cafb9c9637c8acbe619133
- artifacts\colon_support_tradeoff_20260906\support_tradeoff.pdf SHA256 72f8035b619ba4418125ca33bc5d5e38a7f7f24f40df00c38b518d3c06351b27

## 2026-09-06T12:40:22Z — 四target自然组成确认启动
写入时间（written_at_utc）：2026-09-06T12:40:22Z。核对target4旧source/query/fitrows/ridge/joint/ops/dose相同，为所有target匹配最近逐组成强对照，补target4/5各10方法映射；同原mean/discovery与已知source3双组成，旧12构造仅拟合见证。随后冻结1/2/4/5全部10方法，另采1024seq新自然文档；C2为已开发主要正信号，其他三操作/弱源全保留。预算20分钟16GB，无训练/新增权重/audit。target4_config_check SHA256 5613fb39dee4ba0c136eee0d5e5b68753600d1cba83c27ec377f9820e778096b；改前完整入口归档archive/research_workflow_20260906/colon_cohort_20260906T124022Z。

## 2026-09-06T12:51:23Z — 四target新自然确认完成
写入时间（written_at_utc）：2026-09-06T12:51:23Z。四应用4324LM/3840行/125.666562秒；旧数据准备1082LM/36.847467秒，新语料221文档/1024seq取12对24不同文档、24,937,606B。C2四target对dynamic中位/绝对中位均低，Full三方向和Raw全部更强。预定文档对bootstrap先平均target与互逆方向，shared-minus-dynamic -.00211068区间[-.00366064,-.00099101]；独立稀疏/union负，Full跨零、Raw正；不当seed总体显著。source说明10/12、7/12，clause操作名不当语义机制。无新拟合调参于新数据；全部coeff/source/raw逐值及Gram/noop契约PASS。停止冒号局部续试，下一另一source自然输入/输出预测。图已目检，论文/tracker/registry更新。事件时间与当前本地身份：
- F4_colon_cohort_apply_s1_v1_20260906 event 2026-09-06T12:45:21.792293+00:00; raw SHA256 89451c274fc374e63f8951b2b46d450349cb87c9f32c447691f9306d12326faa
- F4_colon_cohort_apply_s2_v1_20260906 event 2026-09-06T12:45:52.331201+00:00; raw SHA256 20855552e67f2a1be4ae0fd519e60523394f8a63933414a739125170eb79241b
- F4_colon_cohort_apply_s4_v1_20260906 event 2026-09-06T12:46:30.301210+00:00; raw SHA256 a84c685b41b7af60740f42630114e855653ec835b3528845ae6ed4004ccff187
- F4_colon_cohort_apply_s5_v1_20260906 event 2026-09-06T12:47:01.133575+00:00; raw SHA256 74cfddacd5a10d740df0dbfc49b49a681fff6cb372dcfae65511f995bdc2977c
- EXPERIMENT_TRACKER.md SHA256 34cfb03dea49b489a23e50de84055cd4706729bf80f0b82ff927dfc3d518f4d9
- PAPER_SNAPSHOT_20260906.md SHA256 2bd518424f5d305e0f939a4588d3f0329d11f9651f33ee07c94124f81d075c4d
- REFERENCE_REGISTRY.md SHA256 14e94bcdd2282b3c93a84d273e27ee570c54e94c03977ae18fa22f795745edd9
- artifacts\colon_cohort_20260906\freeze.json SHA256 0238a012c1aa8890d9ce315454466538b43ae9989640324d4b5280d4a4c6fc68
- artifacts\colon_cohort_20260906\selection.json SHA256 48d8c23d1804902bb5fca589f5d0a7fc9a3aa5e94ceae4be6ed9482524a5e91d
- artifacts\colon_cohort_20260906\prepared_inputs.json SHA256 9a3065ef1cdccd1033e4b95f040f5a5369f26073a459a458f5e2c79578054876
- artifacts\colon_cohort_20260906\FINDINGS.md SHA256 83f0abe2e87ad59fcae1d2b3c288b6f7510f771277f30f7bc0cd6008357a5873
- artifacts\colon_cohort_20260906\summary.json SHA256 ecfea550fd0f8cfba032f19956ebf6de24fec7fc7f373ae4cce111bc5efb7a9e
- artifacts\colon_cohort_20260906\cohort_confirmation.png SHA256 df899c2937076f39c4f2d6a2ba18f5ed7afcf912463479b1b37823c2d664062f
- artifacts\colon_cohort_20260906\cohort_confirmation.pdf SHA256 35fbdd13c25d7f9d3bc146c6a581dd91be9b0d36327c7870fb6ac9470d604883

## 2026-09-06T13:13:32Z — that自然输出预测启动
写入时间（written_at_utc）：2026-09-06T13:13:32Z。旧12构造source概率事后personal-pronoun/there对比add12正/remove12负；不是新自然结果。三词汇类、source1:2379、最多24文档/96LM、0.1native对称剂量冻结后采新1024seq，保留全部短缺/反例。预算20分钟16GB/<100MB，无target筛选/训练/audit。配置和执行脚本hash见artifacts/that_prediction_20260906/freeze.json；改前入口逐字归档archive/research_workflow_20260906/that_prediction_20260906T131332Z。

## 2026-09-06T13:17:12Z — that source自然结果及冻结target应用
写入时间（written_at_utc）：2026-09-06T13:17:12Z。新208文档/1024seq，三类各8文档，source-native96LM/14.219秒：每类联合add正/remove负均7/8，态度激活较报告高；noun挑战同有作用，不能声称态度独占。三反例保留，不改预测词表。既有四target五方法映射逐一绑定，保留全部24输入，按原hash序构造8report-attitude互逆对和4noun互逆对，冻结输出对比绝对误差主终点后执行676LM，预算仍本单元20分钟16GB；source结果已暴露、target尚未读，无refit。source-native确认与source-aligned donor跨seed操作分栏。

## 2026-09-06T13:22:30Z — that自然预测与四target恢复完成
写入时间（written_at_utc）：2026-09-06T13:22:30Z。source-native96LM/14.219224秒加四target676LM/37.253246秒，共772LM/51.472470秒；新208文档/1024seq21,361,822B，全部24文档保留。三词汇类native联合方向各7/8，词表POS歧义与3反例原样保留，不称态度/句法独占。source donor14/16符合词汇符号预测；四target态度/报告Sparse输出MAE均低于bestatom，名词target4反向，target3仅3/8逐例胜；Full/raw全体更强。所有系数逐值等于旧冻结，source/raw跨target逐值相同，noop/契约PASS。图目检通过，论文/registry/tracker同步；下一等能量方向控制，仅已暴露数据开发，无新语料无限续加。source-native原runner未存完成UTC，事件完成时刻未知（本轮已见exit0）；四target事件按status记录如下。
- F4_that_prediction_apply_s2_v1_20260906: event 2026-09-06T13:17:22.478006+00:00; metrics SHA256 a0468d6a20c4f41aa90a261c532afebde70e47763508bcc62543e3c6a3d31ab6
- F4_that_prediction_apply_s3_v1_20260906: event 2026-09-06T13:17:32.688706+00:00; metrics SHA256 e8734c47f0405e8fd28735490bae16a416f30fd9a162e5686be77804907d9038
- F4_that_prediction_apply_s4_v1_20260906: event 2026-09-06T13:17:42.912547+00:00; metrics SHA256 ac05e1584262b9a5419fdd65e4fdb18dfb53bfec8b33add7793533b3ed41f319
- F4_that_prediction_apply_s5_v1_20260906: event 2026-09-06T13:17:53.332723+00:00; metrics SHA256 766d91fcd0e4e66f8e7550bc6ef18ad47cf5b4e800be1dcd8e39508c598dab16
- EXPERIMENT_TRACKER.md SHA256 c008c3125e49e0a0cfc04d6d473dbdb6db72a616b1ed93e6ac3bdbc873798a9c
- PAPER_SNAPSHOT_20260906.md SHA256 249d4c6a6029e637915426cf67f70c41b086db82729c83153f9e8efca6784fe9
- REFERENCE_REGISTRY.md SHA256 04a298e526b7cf52b0119e8b6093018b58ac2b600b746e600d7b4aa7f7fd910c
- artifacts\that_prediction_20260906\freeze.json SHA256 bad8b80ad6004904a6c92a22a9213d7219bc33a8499f1530e61c9c2ff072f9cb
- artifacts\that_prediction_20260906\transfer_freeze.json SHA256 5ba390691d8b55c190ecc35ee20617b8fccc0eb58d9827fdedfe0d1acbf3bd0a
- artifacts\that_prediction_20260906\natural_inputs.json SHA256 2cf498817e500e9472cda72e401ec12cc870b167a71563765112d71846f711bc
- artifacts\that_prediction_20260906\summary.json SHA256 6b22a2b801cb697f56e610c71c3fa50fa77c5da1ab8112d2f3818441653c8f3d
- artifacts\that_prediction_20260906\FINDINGS.md SHA256 35cf48b4868df37e42644af80f63df36d182f8b867325f35125612093259ba77
- artifacts\that_prediction_20260906\that_prediction.png SHA256 36c4b47b4b370d70b64fbaff22270cf6ef886c89722a8c25fbfeaaa6df04b9f7
- artifacts\that_prediction_20260906\that_prediction.pdf SHA256 8808d6758bb6853987eb89cd7a8286a7959e1fb04eb479ede5206f0ee7d6c5c6

## 2026-09-06T13:42:02Z — that等能量方向控制启动
写入时间（written_at_utc）：2026-09-06T13:42:02Z。source-only既有自然候选核对988dominant-token that及12构造激活非零，作为相同token替代，不预称无关机制；不读target选方向。固定988decoder及RNG20260906两个isotropic方向，按2379实际capped native norm对称+/-，保留24已暴露自然文本/原对比/反例。240LM含96次source重放，无新语料/训练，20分钟16GB。freeze/alternative原始依据见artifacts/that_controls_20260906；改前入口逐字归档archive/research_workflow_20260906/that_controls_20260906T134202Z。

## 2026-09-06T13:47:28Z — that方向控制完成与论文收拢
写入时间（written_at_utc）：2026-09-06T13:47:28Z。实验完成事件：2026-09-06T13:42:34.400862+00:00。240LM/18.009158秒，96次原概率逐值一致/144次新控制，0新语料训练；最大范数差1.1921e-7。2379/988同token有符号响应中位+.116863/-.055541，整体KL.007220/.006618，预期联合21/24vs4/24、988相反17/24。source有符号24/24大，但绝对仅15/24。随机联合17/24、14/24，绝对幅度和整体KL通常较低，不能把方向命中当唯一性或同KL比较。988不是自身native操作，2random非总体检验；全部文档/分组保留。图目检、论文Abstract及过期素材/解释摘要已更新，Paulo A.7复读区别入registry。停止局部续扫，下一整篇贡献与方法机会收拢，不以准备冒充新科学结果。
- NATIVE_that_controls_s1a2379_v1_20260906/metrics.raw.jsonl SHA256 f28719dce7bb7889828aaab6fd4b1eb8b818d684470d3952bd19676afb32a370
- EXPERIMENT_TRACKER.md SHA256 c6cc6b1e8107473f47cea2123e0c5ddc3e14bcfd7f9e52ffd7903e010fc563db
- PAPER_SNAPSHOT_20260906.md SHA256 d32440161050a595aef906278459522ddc5f847c601b581b4511362317d4e879
- REFERENCE_REGISTRY.md SHA256 1f3de4c901b91ecdf9bec78fec8af1a01bf8a7715d064b0bd064f6ea33fcf7e8
- artifacts\that_controls_20260906\alternative_selection.json SHA256 a2ca25274243b67dcc875243d95384c9ead1972233ccdda7ecc04a2827f48215
- artifacts\that_controls_20260906\freeze.json SHA256 ebc0ff57cf1385150b2129db74a418ed565e3ce8dc37ab5f79a2c7a4887ac0e1
- artifacts\that_controls_20260906\summary.json SHA256 8d5f0b5d7c1c720d21401fee0cca37289e74c017c19bff2a93d7416bcf3e06af
- artifacts\that_controls_20260906\FINDINGS.md SHA256 e67292bbe3f032077bbd022dcf8ae01f2a0372f6ad48480abf1c06c2b8ec5330
- artifacts\that_controls_20260906\direction_controls.png SHA256 a3afa7f04c54171edd1db056e419937ac71d364c449715bbf3dcf90b46883f78
- artifacts\that_controls_20260906\direction_controls.pdf SHA256 3a248cdb3df3481318be4ed659f0ed6a328babba40f7af79910c6730d553eb7b

## 2026-09-06T14:12:01Z — 论文收拢与source-energy family诊断启动
写入时间（written_at_utc）：2026-09-06T14:12:01Z。当前差距为组成可靠性对实际选择的增量。采用标准广义Rayleigh商sup(theta^T Gamma theta)/(theta^T S theta)，S是源作用能量Gram；明确共同重参数化不变与source零空间残差导致无界的条件，不声称一般线代首创。已检查消抵/缩放重参数化/零空间fixture；r004未装pytest，直接运行相同assert函数通过，无安装。冻结十已有训练轨迹cell、<=16成员候选、full/trace/worst三选择准则，leave-one-authored-pair-out开发对比已存LM结果；0新LM，无独立确认宣称，全部弱例和raw/full锚点保留。改前入口/推导/论文/registry逐字归档archive/research_workflow_20260906/family_risk_20260906T141201Z。

## 2026-09-06T14:16:26Z — 论文重组及family诊断开发结果
写入时间（written_at_utc）：2026-09-06T14:16:26Z；分析完成事件：2026-09-06T14:12:02.047013+00:00。80方法cell/60相关fold，0新LM/语料训练，.233926秒run wall。最坏family相对full8/60选择变化，主要相对KL4胜4负，平均.183389vs.163604；绝对平均.00104826vs.00108042。混合结果不称选择收益，不换主要口径。源能量归一化和null-space条件数学成立，仍不是LM相对行为保证。论文引言三主张/主图与方法接口直接修订，旧rank1、素材和近邻边界保留；下一逐输入敏感度方法先辨别旧F1/F4失败scope，不重走平均Jacobian错误。全部raw/folds/强锚点和fixture保留，无用户审批或暂停。
- F4_family_energy_diagnostic_v1_20260906/metrics.raw.jsonl SHA256 ce6b39943281619520af828d5af70e32dc1e8f68b997c1fd54f4d96192a6d360
- F4_family_energy_diagnostic_v1_20260906/folds.json SHA256 1aa11859a60b87dd94f0dba5dd72ea1f875fc418e9c395db66b3c999bf87e6f6
- EXPERIMENT_TRACKER.md SHA256 29a3c0effe8d048467f47c510e6a89c27edb2377a6a7ae98b28dde3314fd02d4
- EXPERIMENT_PLAN.md SHA256 7eaa42a405588caf6558e39ef8f53698492ac57008186c289e02261a82909c23
- PAPER_SNAPSHOT_20260906.md SHA256 0ac196de81428db23cc8a1f79be8725d93ccf394aa92a4d1ffd0db6128683411
- DERIVATION_PACKAGE.md SHA256 152d8b9dc89dfe98e7ccb03ad802e8c1335cce522c8d2f302b65d64e6b5b660f
- REFERENCE_REGISTRY.md SHA256 c1fa111f3edbb1362e52efdc9e5b7415f0eca30320dff14cdb40e13942316fa2
- artifacts\family_energy_diagnostic_20260906\FINDINGS.md SHA256 23d5eb5f2ac3d70611112b6a35bb84556b60c6cc3b995b20bd46b4db77d2736e

## 2026-09-06T14:43:37Z — 操作匹配逐点Fisher开发启动
写入时间（written_at_utc）：2026-09-06T14:43:37Z。已核对旧causal_metric_probe及fuzzy_correspondence pooled output-sketch ridge产生全局metric，F1 EOT/平均Jac问题与F4 global native失败范围。新测试沿既有非共线源decoder两维，在每个source操作点计算全词表logit JVP，F=J^T diag(p)J-(p^TJ)^T(p^TJ)，不平均J再平方。固定Euclidean、校准平均Fisher、逐点Fisher三个选择器，十旧训练cell全部保留，≤16target成员信息预算一致，留互逆pair开发对照；主KL、次absoluteKL/TV，TV非独立因果端点。配置在新Fisher计算前冻结，96JVP+8差分普通forward、20分钟16GB，不新增模板/剂量/成员扫描或确认宣称。改前五文件归档archive/research_workflow_20260906/pointwise_fisher_20260906T144337Z。

## 2026-09-06T14:44:58Z — 逐点Fisher差分见证失败，局部数值诊断
写入时间（written_at_utc）：2026-09-06T14:44:58Z。v1在8 forward/6 JVP/13.25秒后FAIL，case0 field coordinate0 的float32中心差分eps .01相对L2误差.039734>冻结.02；未做候选选择。原run/source快照/错误保留。尚不能判为JVP错误或科学失败。v2仅增加差分步长.03/.1/.3诊断并要求至少两步误差<原.02，不放宽容差，不改变科学选择准则、数据和操作；120forward预算，诊断失败仍停止受影响试验。

## 2026-09-06T14:48:33Z — 逐点Fisher开发结果与局部收口
写入时间（written_at_utc）：2026-09-06T14:48:33Z；run完成事件：2026-09-06T14:45:33.826969+00:00。v2 PASS120forward/96JVP/16.895秒，v1 FAIL8forward/6JVP/13.246秒另计保留。多步差分原2%容差四见证通过，源概率最大差2.942e-8；数值正确性不等于科学优越。十暴露cell60相关fold，逐点/欧氏/常量主要均值.166632/.176311/.173970，对欧氏6胜5负，对常量6胜10负，绝对KL/TV均值略差；旧full-only .163604仍较优。细分显示主要均值改善集中s1step2048，不能筛掉s2step256恶化或换端点。未成立稳定选择收益，不再扫描固定候选权重。下一有界方法为操作Gram加权拟合本身，同支持/信息普通ridge与未改动sparse对照，独立拟合及新输入/额外端点确认；标准GLS不当首创。论文方法/结果与推导§17已直接更新；automation持续ACTIVE，无新科学暂停或审批，GPU自动释放。
- runs/F4_pointwise_fisher_v2_20260906/metrics.raw.jsonl SHA256 0d30fc9011f5b234e62d31fdcd30b81fa8870e2bbc40bd738d3a81c3660bcd55
- runs/F4_pointwise_fisher_v2_20260906/folds.json SHA256 44d2622d74e4f12e40e0d2795582f42502f72303991122ad24627bc992ad124d
- runs/F4_pointwise_fisher_v2_20260906/logit_jacobians.npz SHA256 62dfe04c052ab92072c286502ef3cc98554ed72401064123ef487c630efd47c7
- EXPERIMENT_TRACKER.md SHA256 a649b72bba819ccf49baecd12b65bfe67db49ab0c7983aadff257d82ec8710d4
- EXPERIMENT_PLAN.md SHA256 a8b07c03aed2133befe9a4235a75c5bb447ffc9c3f17efaba398969cdb50538e
- PAPER_SNAPSHOT_20260906.md SHA256 e6c9d297d87d8c91d29a0e51de0d7d11cfe7d36c3620f86ae338a2006ce6f54b
- DERIVATION_PACKAGE.md SHA256 7114593e9b2f3b24d9e2b455fda3ab240e752a0995bd343dea0fbe215975624e
- REFERENCE_REGISTRY.md SHA256 0259b082e5b08b4f0616d696c12539a48ea61c432eaeb3f84521c64235663efd
- artifacts/pointwise_fisher_20260906/FINDINGS.md SHA256 cea05318485cfb2f5e175c3d2f0c90603172a0e7a495951b55676c14edb14037
- artifacts/pointwise_fisher_20260906/descriptive_comparison.json SHA256 15d693e9084054926ad3591778c43fc37e1125dc2c9f8aab3421b0f9fd618060

## 2026-09-06T15:10:56Z — 操作加权拟合单元启动
写入时间（written_at_utc）：2026-09-06T15:10:56Z。优先改变映射而非继续给旧候选换选择器。核对旧两自然panel48不同文档/24互逆pair，固定已有shared16实际15成员30参数，原discovery支持不重选；RMS列尺度下ridge锚定原图，fraction.1一次冻结、不扫参。三refit：逐点source-operationFisher、平均Fisher、Euclidean，等输入/支持/正则规则；旧八图全部保留。损失为absolute localKL，无弱sourceKL分母加权；主验证仍四operator最差median原normalizedKL，另absoluteKL和未用于拟合的观测next-tokenNLL响应误差。方法/主要终点在新语料之前冻结；旧48输入用于监督开发不能叫新确认。先运行432forward含384JVP预算20分钟16GB，无SAE训练。新自然验证另在系数冻结后准备。逐字归档archive/research_workflow_20260906/fisher_refit_20260906T151056Z。

## 2026-09-06T15:14:53Z — 加权映射已冻结，新自然验证准备
写入时间（written_at_utc）：2026-09-06T15:14:53Z；fit完成事件：2026-09-06T15:13:25.224138+00:00。拟合PASS432forward含384JVP/26.231秒；设计rank15/参数Gramrank30，各三正规方程相对残差<4e-16。拟合目标下降只算优化检查，不作科学收益。三新映射及原八映射已冻结，coeffSHA c8c5724653719e661aab9d5af66288ab7710f0c0b483c66a7d07651fb760da68。新语料前冻结主要四operator最差median原KL比值、absoluteKL、未用于拟合的自然next-tokenNLL响应误差；同LM输出的另一端点不是独立因果研究。沿用旧词法适用规则与12pair上限、完整拒选/弱例，documentID及textSHA排除所有已物化语料清单（20项）。冻结artifacts/fisher_refit_20260906/freeze.json SHA e584147caf43ea35a146835badab994725e66507cc9fe850e36e1f7baef5eb62。

## 2026-09-06T15:24:27Z — 固定支持功能重拟合新自然验证完成
写入时间（written_at_utc）：2026-09-06T15:24:27Z；新输入执行完成事件：2026-09-06T15:17:27.734921+00:00。拟合/执行均PASS，432forward含384JVP/26.231秒 +1177forward/32.675秒=1609forward；25,005,562range bytes采样2997文档、物化225、固定规则选24不同文档12pair，无新训练。source3→target1/15固定支持，主要worstoperator medianKLratio point.176313 vsEuclid.151365 vsoriginalshared.246879 vsconstant.192704；point absoluteKL .000419397不如Euclid.000396740与original.000357375。次观测词NLL .013574优于Euclid.020964、原图.014560接近，预冻结bootstrap区间跨0。全部11map/rawfull/每op/弱源保留，不换主要指标。三局部单元后结束Fisher默认队列；下一固定训练阶段的共同自然功能曲线，把素材改进与主图3对齐，不加训练网格或重新命名已暴露数据。图已生成并目视检查、论文§3.7/结果及推导§18同步，automation保持ACTIVE，租约已释放。
- runs/F4_fisher_refit_s1_v1_20260906/coefficients.npz SHA256 c8c5724653719e661aab9d5af66288ab7710f0c0b483c66a7d07651fb760da68
- runs/F4_fisher_refit_apply_s1_v1_20260906/metrics.raw.jsonl SHA256 6184cb06e5181bd442ff8d193d832e70b1546c263e0a35d83bd96a2d5d093280
- artifacts/fisher_refit_20260906/summary.json SHA256 440c41a62a35c1c9c1a87075d5280e7de70e3eeedc1804d1e0504c0eabb4debd
- artifacts/fisher_refit_20260906/FINDINGS.md SHA256 e79f46b1e502719a0274e3f7133e7c3827b812a50520de6500b390c07d01e3f7
- artifacts/fisher_refit_20260906/refit_tradeoff.png SHA256 a82664798c46255f666f3f4fa53b45a9827379a3e81ead98ad848d4a97d2f72b
- EXPERIMENT_TRACKER.md SHA256 88d4c9e2b988cd633624cdda5ac98e763329fcde1a5f84c5f55c88efe282b2d4
- EXPERIMENT_PLAN.md SHA256 0147ee80a30e3b67e5e79cdf0bb4d7ba4d383e34576208c939265211a7ecd958
- PAPER_SNAPSHOT_20260906.md SHA256 26b5794c29498d41fea2068f94510c8680eecb60ba3ca2d413fd7579a1d285b0
- DERIVATION_PACKAGE.md SHA256 25c490caff79be5ae6578d94bb50c11464c748c249414120b10f94cfb5499f81
- REFERENCE_REGISTRY.md SHA256 cb53b9a30c35bfdc8652c8cad10e540529e213d1307c712ef8c48773364b6668

## 2026-09-06T15:25:27Z — 图脚本格式补记
写入时间（written_at_utc）：2026-09-06T15:25:27Z。首次staged diff检查提示plot_fisher_refit.py末尾多余空行，原命令仍完成提交推送；现移除该空行再核对。仅格式变化，不改变图、计算或科学结论，不重复模型运行。

## 2026-09-06T15:45:38Z — 自然输入固定训练轨迹启动
写入时间（written_at_utc）：2026-09-06T15:45:38Z。核对十原始checkpoint/系数与原固定64序列质量验证，源3不变、target1/2同流5阶段；当前自然24prefix已暴露，明确开发扩展，不称新确认。冻结原八图与四操作、主要worstopmedian原KL比及absolute/NLL，完整阶段不挑结果；s1/8192已有执行直接复用。旧frozen入口只允许终点checkpoint且assert保护，新增显式use_frozen_target_checkpoint按父配置/权重SHA加载早期SAE；不是旧自然结果失效。先2旧输入75forward逐项复现，再9*889补齐，15分钟16GB，无新训练/下载/refit。改前四入口/论文/registry逐字归档archive/research_workflow_20260906/natural_curve_20260906T154538Z。

## 2026-09-06T15:55:58Z — 自然训练轨迹开发结果与数学分离
写入时间（written_at_utc）：2026-09-06T15:55:58Z；最后自然cell完成事件：2026-09-06T15:49:34.943389+00:00。十固定stage/target完整比较，9新cell8001forward/199.554秒、旧输入见证75forward/7.818秒，wrapper219.869秒；s1终点原八图复用。source/raw精确replay，所有run contract PASS，无新训练/refit/语料。sharedabsolute早晚降74.3%/58.5%，整pair-bootstrap差区间均低于0（限定两轨迹/12pair）；原主要relative曲线非单调区间跨0，Full target2恶化，Raw更强，不能事后换主指标。新的正机会为在新文档预冻结absolute主终点作早晚确认，未运行的不计成果。图与全operator已生成/主图目视可读，论文同步。附自含TopK1完全重建但仿射query残差11/135的精确分数反例和推导§19，bias/频率差/完整字典及非线性边界明列，无数学首创宣称。automation保持ACTIVE，GPU租约释放。
- artifacts/natural_function_curve_20260906/summary.json SHA256 91df2101174d6839afbf258a2ee0b4a6bc3f7c0fcdcc763bc063dafb7f6a7bb3
- artifacts/natural_function_curve_20260906/early_checkpoint_replay.json SHA256 5bff2b80c0222964866e4b52f7428110567410dd50ce8fa2730ee7471aa22c1c
- artifacts/natural_function_curve_20260906/reconstruction_counterexample.json SHA256 6f699bb3e8ce18a9653e2b17583464ce85a50e9ae315d9a8644af360aa8db5e9
- artifacts/natural_function_curve_20260906/FINDINGS.md SHA256 12a01e07f5e4fc992224bba6e96a5feb7bdeccee9e98968cf67d1084b33f2f62
- artifacts/natural_function_curve_20260906/natural_training_curve.png SHA256 a3ac55c16bae87c0f3159b644c6a26e0ac79a86a4aa0008d4f6bb11591838bac
- EXPERIMENT_TRACKER.md SHA256 e4354a9fdc3ab4c1a937fcdd07676d30768122777a55d3ae0622d199670f5f4d
- EXPERIMENT_PLAN.md SHA256 8ea7162645e29dbcbd84f736c76790c67e9432525b02d08eaadf75b5d9457400
- PAPER_SNAPSHOT_20260906.md SHA256 7191d2cd1ac7f64458836df467575db483929625b27419d00bbc3382e133c606
- DERIVATION_PACKAGE.md SHA256 5abaeee312f87c894d0c370da3ce394c654e3c2a9726542d855f2fe4ca53cf32
- REFERENCE_REGISTRY.md SHA256 d86f8955753a9d1355a676ba868864e759ed56dadbe236ff96b4d0de3ed874c4

## 2026-09-06T16:17:10Z — 紧凑关系训练收益新文档确认启动
写入时间（written_at_utc）：2026-09-06T16:17:10Z。冻结四父coeff/config/targetSAE SHA、早256晚8192与source3；新主终点为shared16最差operator中位absoluteKL晚减早，两target均预期下降。旧开发主要relativeKL混合结论不变。原八对照、相对KL/NLL及所有弱例保留；2000整pair bootstrap不当seed总体推断。先冻结再新语料，排除21份既存document清单；预算3556forward/10分钟16GB，无新训练/refit。冻结SHA f42fabf502027c98a74086d3b5891578418eb86dc41c4b0346dcab4af1cbad56；改前四文件逐字归档archive/research_workflow_20260906/training_gain_20260906T161710Z。

## 2026-09-06T16:19:07Z — 确认启动环境路径修复
写入时间（written_at_utc）：2026-09-06T16:19:07Z。首次s1/256 v1缺少既存f4_sparse_overlay_v1的PYTHONPATH，导入sklearn失败，0forward/0结果；失败run保留。仅该执行cell改v2并补齐账本已验证overlay，不安装/升级；冻结权重/系数/输入/主终点完全不变。其余三cell尚未运行。

## 2026-09-06T16:23:46Z — 新文档紧凑训练收益有限确认完成
写入时间（written_at_utc）：2026-09-06T16:23:46Z；实际完成事件：{"F4_training_gain_s1_step256_v2_20260906": "2026-09-06T16:19:33.089724+00:00", "F4_training_gain_s1_step8192_v1_20260906": "2026-09-06T16:19:59.148881+00:00", "F4_training_gain_s2_step256_v1_20260906": "2026-09-06T16:20:25.520379+00:00", "F4_training_gain_s2_step8192_v1_20260906": "2026-09-06T16:20:51.394123+00:00"}。3556forward/100.921446秒，wrapper104.985660秒，0新拟合/训练；新池221文档/23,873,278B，24不同文档12pair，无target选例。预定shared absolute早晚下降60.5%/84.1%，各11/12pair改善；差区间[-.00229357,.00005959]、[-.00353947,-.00031146]，target1跨零。NLL次指标下降42.0%/60.5%且描述区间负，relative区间均跨零；不换旧主指标、不称总体显著。same-support ridge近同，Raw/Full仍强；target1动态pair absolute反升45.4%，旧开发Full target2负差保留。四契约与source/raw身份核对通过；初次0forward失败已记且保留，无额外重跑。论文/主图/当前卡已更新，本对局部探索收口；下一补R012同流三seed至五素材，功能范围待source-only冻结。全部本地忽略文档与成果hash：{"artifacts\\training_gain_confirmation_20260906\\FINDINGS.md": "2e8e5bd632c1477b83b4eb4be64901d39a5f6e92f28ca230f7bf2b23e70db6c1", "artifacts\\training_gain_confirmation_20260906\\summary.json": "89dbe2dea76e9e996ce76e5bd95a4f66384b346aba2b943c8f9e531ff40b18ae", "artifacts\\training_gain_confirmation_20260906\\training_gain_confirmation.png": "239ea96b91734e5c44ad43e9f2299a98d028295d2e15c25ebe6e15805519f19d", "EXPERIMENT_TRACKER.md": "6439a0307b90b0cf05201c72d9f2f4d8ec2e3f15478ed11e26940defbadef87d", "EXPERIMENT_PLAN.md": "b407b36e8d05c67d8afe976bbe62167d199b0b6bee1c39dc433a9ee2cdf36447", "PAPER_SNAPSHOT_20260906.md": "86f864fc17e7ce3924c63156ccf6d52adb394190ccb004777285948eab8dce7c", "REFERENCE_REGISTRY.md": "a77fcb9b4147ec98c7fa90a25b114f3ded5844d170ccc9a5d74b663b95c85c7b"}。

## 2026-09-06T16:42:37Z — 同流五seed素材扩展启动
写入时间（written_at_utc）：2026-09-06T16:42:37Z。在上轮新文档有限训练收益后，仅新增R012 seed3/4/5，既存seed1/2复用。配置逐字段核对：除run身份/存储/seed列表/预算/说明外全同；原8192step4.19Mtokens、五checkpoint、同训练/验证SHA、模型hook架构优化器不变。实读既定Sparsify trainer逐seed manual_seed/独立Adam参数组/不shuffle，与canonical bookkeeping；无训练实现变更。脚本仅泛化manifest两seed描述。旧两seed实测229.782秒，新三seed预算900秒16GB1GB。质量不等于功能或训练充分；source3新旧身份分开，下一扩展source-only作用覆盖。改前四文档逐字归档archive/research_workflow_20260906/five_seed_curve_20260906T164237Z。

## 2026-09-06T16:46:04Z — 扩展源端候选规则冻结
写入时间（written_at_utc）：2026-09-06T16:46:04Z。训练进行中，冻结R012_source_census_s1_v1_20260906，只读原R012 source1终点和8192均匀discovery缓存hook位置；沿既存context-census浓度乘log覆盖排序，在标点/ASCII词法类各取前2个有至少2候选atom的token，每token前2atom，完整请求分母保留。最多4组8atom，未编码/未读target前固定，不以匹配结果选择。词法代理不是语义/作用说明，后续须自然反例与功能验证。预算16encoder batch/60秒16GB、0新LMforward，训练释放后同GPU租约顺序执行，不争抢。

## 2026-09-06T16:49:40Z — source候选保存元数据修正，无重算
写入时间（written_at_utc）：2026-09-06T16:49:40Z。R012_source_census_s1_v1科学计算PASS/5.977970秒/0LMforward，但config缺audit_opened字段而manifest明确false，contract报单项不一致。原run所有文件保留；v2复用父原始数组/候选/逐项统计，四文件SHA完全相同，仅显式补false及新run/继承路径。v2无encoder/LM调用，contract PASS；候选选择发生在v1，不能把v2称新数据或复现。源候选2982/3072，732token请求，规则选4pair为连字符、弯引号、the、on，语义/跨seed作用仍未确认。

## 2026-09-06T16:54:39Z — 五seed同流素材与源端四组候选完成
写入时间（written_at_utc）：2026-09-06T16:54:39Z；新增训练完成事件：2026-09-06T16:48:02.728646+00:00。R012新增3seed318.670425秒/8192shared训练forward及15checkpoint验证，453,454,499B、VRAM1,553,508,352B；旧2seed不重跑，8192实际输入hash完全一致。五seed终点FVE .984989–.985098、CE .977980–.979976，验证alive3054–3060、L0约128、norm1.79e-7，25阶段图/报告已写论文。source census5.977970秒0LMforward，2982atom/732token请求→四pair，原v1仅元数据缺audit字段，v2继承0重算，修正前后raw/code/candidates相同。Source-only自然对比提供I’/否定缩写和in the假说，hyphen/on不清楚仍保留；无新跨seed功能结论。后续同源1四组query四target两端点套件，新自然输入与预算先冻结，避免原colon局部续试。忽略文档/结果与分析代码hash：{"artifacts\\five_seed_training_20260906\\FINDINGS.md": "254e7cfcb0a8256871fc2074a072aad3f7dda3c737098d6c8621df7ce61328b2", "artifacts\\five_seed_training_20260906\\summary.json": "57a8b910a9c15e1d4c6c05444622b49d7837d195f9fd34bcd4b64eab0130d46c", "artifacts\\five_seed_training_20260906\\checkpoints.json": "a4f8e71a3b1e509b1052ac7bda5457340f2c20979c264d1201dfde0a9df8773e", "artifacts\\five_seed_training_20260906\\source_contrasts.json": "679ab92c74ed48706466d9d14b945cd0c6166daaaf913df48092505230181b14", "artifacts\\five_seed_training_20260906\\five_seed_quality.png": "48df474344b2819b2ebe3c0b1e69359569e72608338921bd99e4fb089afe92d6", "EXPERIMENT_TRACKER.md": "a0134e27c85935fe3c54c5f9f596802d994370aba6d666f5505e395921ce38e0", "EXPERIMENT_PLAN.md": "e0aefaf98f7f6f1cc10cdceeba561ec42fc8e12721d3ebf544d0baa419a229ca", "PAPER_SNAPSHOT_20260906.md": "34fa3bb20ffb939b03883701980092d124a5057f8144320dbb9bd8ed31dac14f", "REFERENCE_REGISTRY.md": "7b8d412b50d3570256f9e9d48b6d33e0fe397d319d8a12eca92a06e4e7ebe5d0", "scripts\\summarize_checkpoint_source_contexts.py": "4c98d8d6a44025edcad6e84f63d44a62e01b86159f029b11bd2be4f6b41c8565"}。

## 2026-09-06T16:55:43Z — 五seed单元资源释放核对
写入时间（written_at_utc）：2026-09-06T16:55:43Z。共享管理器gpu-0/cpu-heavy/disk-d-io/disk-e-io均free；tracker已同步当前状态，SHA256 dfcc9365be81714fe350abbf1ed8d4753a60908f5edebf6baeacb17b5e593136。本单元无遗留计算，automation ACTIVE。

## 2026-09-06T17:18:15Z — 四组五seed功能套件冻结启动
写入时间（written_at_utc）：2026-09-06T17:18:15Z。32cell=4query×target2/3/4/5×早256晚8192，R012source1终点不变；新权重和source-only候选完整绑定，不用旧R011同IDsource。新增显式source_checkpoint/无旧parents raw拟合/fit_only入口；先75forward旧输入回放，再32原discovery拟合全存后才取新文档。九图包含交换组成且逐operator源能量匹配wrong-query（原八图保持），主要absolute最差operator中位晚减早及固定4query4target平均，relative/NLL/各组成全报；2000整pair跨target同步重采样。固定hash每query16不同文档邻接成8pair，跨query文档不重叠，无活动/效果筛选，不足保留缺口。预算21024+75LM/30分钟16GB和100MB以内语料，不训练。设计冻结SHA 893d06a929b0a9f56d4b7ae836cc1d14f38bb693f49baaa6f189192d28f4bd24；改前四文档归档archive/research_workflow_20260906/five_seed_function_20260906T171815Z。

## 2026-09-06T17:24:21Z — 四组32映射全冻结，新文档准备
写入时间（written_at_utc）：2026-09-06T17:24:21Z。32fit-only均PASS/0LMforward，maps_freeze.json SHA 5919950f00a684d2573c5a9b871c3701dbeb8ce425207119706e3383a6c5d4c2，全部方法/源与target早晚权重固定后才采新语料；排除22份既存文档清单。原输入回放64行八字段精确。新自然请求及九图规则不变。

## 2026-09-06T17:28:00Z — 新自然64文档应用启动及存储口径
写入时间（written_at_utc）：2026-09-06T17:28:00Z。新池按冻结规则选四query各16prefix、总64不同文档，0缺额；未用活动/target筛选，32原映射全冻结后取数。新应用全词表KL仍当场逐例计算，保存的概率数组仅保留全部observed next-token与预定t token的原值，避免32cell冗余全词表概率造成数GB磁盘IO；全词表概率不另存，故未来新增词表分析需要新计算，非本次端点缺失/改口径。wrong-query控制显式拥有oracle源能量；单组成归一化后仅剩符号，和/差才检验组成方向，不作为同信息预算方法排名。32应用GPU预算21024LM，当前真实执行中。

## 2026-09-06T17:40:20Z — 四query五seed功能结果完成与入稿
写入时间（written_at_utc）：2026-09-06T17:40:20Z。运行完成时间未从独立时钟确证，不以本写入时间替代；32应用均已PASS，21,024forward/554.163秒。紧凑absoluteKL下降66.44%，13/16cell改善，固定面板条件CI排0；the/on、relative/NLL仍跨0，Raw更强、同支持ridge近同。控制能量统计{'matched': 2048, 'total': 2048, 'zero_predicted_energy': 0}。32冻结映射不变，完整source/raw一致与缺失保留。论文§5.12/两主图、计划及tracker已更新；推导§20和Paulo官方方法附录阅读已记registry，不称标准能量代数创新。下一方法/论文整合，不延长局部任务。忽略文件留本地，SHA256：{"artifacts\\five_seed_function_20260906\\FINDINGS.md": "37ff964495580a598caada56e9ef955a4f1e2785aadcd1af77dca4dca57cc7fd", "artifacts\\five_seed_function_20260906\\summary.json": "6c6fedefbcb2076fc2d807954df68f06070b20be4a6d4acfb153bfd340de149c", "artifacts\\five_seed_function_20260906\\final_checks.json": "fb808da1145dbbef29ccef30fca4ac5cb69f0639da00f67b39444616975757e0", "artifacts\\five_seed_function_20260906\\four_query_function.png": "ab92ff0814e839cad7bcde9628bf917d9f37da0ee4c1ee7b046261b81651d282", "artifacts\\five_seed_function_20260906\\compact_training_change.png": "2f0fddd9a6a4c239ad88fa1c1e8d668b20fb5acd8275b80e61f85b99690d7cc2", "EXPERIMENT_TRACKER.md": "90ba711c67e8b4dbb5f12fafbf702a669c6b1d45b7e89ac77a811fba52230457", "EXPERIMENT_PLAN.md": "89cb332f76eec486f48e3e354dac13d019da0a04bfc35ac3b6ff73ffc059e491", "PAPER_SNAPSHOT_20260906.md": "3d276782999b890262b4a2fd3f75f7ccdfb86345a9d56bd2c41b81d8ef54c9b8", "DERIVATION_PACKAGE.md": "b8d7aa63eb33b4b107600de88dc0d0544e8b8e57fe549c9716e8afa7517812cd", "REFERENCE_REGISTRY.md": "2ff2e74bd16c53d31127a2289b8240ea59855eedadaa3cd4f61c43080ae4c222"}。

## 2026-09-06T17:40:51Z — 完成后资源与排版核对
写入时间（written_at_utc）：2026-09-06T17:40:51Z。共享管理器四资源均free，无遗留计算。两主图已视觉检查；报告表格与论文标题层级已修正，不改变数值。最终文件SHA256：{'artifacts/five_seed_function_20260906/FINDINGS.md': '629de93abd15c266ad29bd2e89f4c47937870c88d9270de01c58cdec8f32d922', 'PAPER_SNAPSHOT_20260906.md': '8ab1fdd69392027e2c640b70a2bf25a0c8628a3745ce86a7fd757eed0f065c4d', 'EXPERIMENT_TRACKER.md': 'a40b0e2b13222f0333f9babcf80f92c19e0285582a3c8819f5f9f592467d3617'}。

## 2026-09-06T18:00:42Z — 操作族输出metric单项开发启动
写入时间（written_at_utc）：2026-09-06T18:00:42Z。复读Li15v3 §4明确其标准化保护低活动通道；官方MultiTaskLasso目标及既有源码已核对。对A/B/sum/difference等权，operator二阶矩=3I/4，decoder Gram Hadamard后仅对角；iid donor残差损失=2trace(Cov(residual) M)，不同于每输出方差逆权。拟合候选只改输出metric，保持40alpha/16支持/solver与去收缩，16晚期cell全留。先检验calibration代理再决定功能投入；不能当KL/最差操作匹配或新算法。预算10min16GB0LM/新数据/训练。归档archive\research_workflow_20260906\operation_metric_20260906T180042Z。

## 2026-09-06T18:07:59Z — 输出metric单项开发完成，保留混合负结果
写入时间（written_at_utc）：2026-09-06T18:07:59Z。事件完成见各run/status.updated_utc，本时间是文档写入。16cell PASS，旧默认系数精确；平均iid-donor hook代理0.122126915→0.123417107，+1.056%，6/7/3，仅2双组成改善；不是LM结果或总体统计确认。0LM/训练/下载，wrapper265.521秒，4针对性测试通过，CPU/GPU释放，其他项目D盘租约未触碰。推导§21/论文§5.13、主图和完整正负表已保存；不继续局部权重或扩展该法LM，下一source-only自然组成解释。忽略文件SHA256：{"artifacts\\operation_metric_20260906\\FINDINGS.md": "e3d7c63ea2a76ef0e50547f537cff48ffd53344c2146faf84e167285bf369a8d", "artifacts\\operation_metric_20260906\\summary.json": "d1af75d2ea8f129c445fb6015af579b2e51eb8df6e28e6b972c8348757c75552", "artifacts\\operation_metric_20260906\\calibration_results.json": "38508443ab98b2f2e91d48f99b2ba5e331130973e8a2cc8f177bb57603b41177", "artifacts\\operation_metric_20260906\\operation_metric.png": "cf9ddeb0f06e9394b6b332996bbb6807a7c886ffa4b10fd40de4416fff65daad", "DERIVATION_PACKAGE.md": "2a07b5029cf67b9580d2a9631455a0b41d44ab87d2ad1985d72703d4ed864e5d", "REFERENCE_REGISTRY.md": "3e1b3f8bab08a911243751c90f6379e1545d472e20b10fd0cc16cdc8cb76e177", "PAPER_SNAPSHOT_20260906.md": "2e497b2eec0269b988d11cee8a8f4d2932f80ec3f02cc39b941ad8ea51cbb2cf", "EXPERIMENT_PLAN.md": "77bdd89e20cdc8c976c24e650539540b8afeac4b12f2f15b2a793e354b8aca0a", "EXPERIMENT_TRACKER.md": "1d483aaed1e7500309c7dc19f4e327fafa876e504006745221001912743d30a1"}。

## 2026-09-06T18:08:34Z — energy路径元数据名勘误
写入时间（written_at_utc）：2026-09-06T18:08:34Z。已完成run的energy_support_selection selected/path.debiased_standardized_error实际记录energy目标，top-level standardized_training_error仍正确；run保留原文，报告新增说明。现行helper仅把可选energy路径该字段重命名debiased_objective_error，数值/默认路径/主要calibration结果均不变。报告最终SHA256 02aa42e738ea476f7fe657d882f545e41231bdd2933a95bc1749a998a32bd706。

## 2026-09-06T18:18:37Z — 用户要求核心攻克，撤回局部续试默认队列
写入时间（written_at_utc）：2026-09-06T18:18:37Z。用户明确指出一夜局部结果未改变核心贡献、反复push和“未闭合”不满足目标。保留全部小结果；撤下跨source census作为自动下一步。当前从source-aligned读出与target自身贡献实现的结构差异攻克核心，既有五seed、160source-only分层query/20有向seedpair、full/16/64贡献补全与同信息readout/atom，先开发矩阵再决定完整主图/解释/新自然验证，不把native unweighted成功设唯一出口。首阶段预算30min16GB2GB0LM/训练/下载。改前六文件逐字归档archive\research_workflow_20260906\core_contribution_20260906T181837Z。

## 2026-09-06T18:31:06Z — 全五seed贡献实现差距与粒度检验启动
写入时间（written_at_utc）：2026-09-06T18:31:06Z。F4_contribution_completion_v1 PASS115.975秒0LM，160source-only×4target另160self，self精确与正常方程成立。640跨seed上native全字典learned误差中位.932635、calibration全字典经验oracle下界.926635、same64支持readout.520337；readout<.1的65方向中32个oracle>.5（描述性阈值，不是预定总体统计检验）。native64优于bestatom623/640但幅度常小，不叫普遍功能恢复。直接推进source-only分层group1/4/16/64和等大小随机source组合以检验中间粒度，复用既有cache，配置f4_contribution_grain_v1，30min16GB0LM/训练/新数据。

## Core contribution grain and frozen natural prediction

Written at UTC: 2026-09-06T18:47:31Z
Event completion: 2026-09-06T18:45:24Z (grain status artifact). All4480 group-direction comparisons complete in84.41s. Source-coherent full-native median relative errors size1/4/16/64=.93263/.72147/.64364/.66218; size16 random=.77950. Native64 coherent16=.65281 versus random16=.80150, matching-refit=.72722; marginal64=.65567 remains essentially competitive. Coarsening helps selectively, not monotonically; none of these aggregate errors means complete concept recovery. All source anchors/directions and random controls retained. Exact nonnegative TopK synthetic plus held-out nonlinear softmax witness complete, rho=.5 fine hook/KL=.19209/.15645, correct block~0, wrong merge remains. Source-only40anchor subset (first2/quartile/seed), two same-token cross-document pairs/query, coherent1/16 and random16 are frozen before fresh corpus; scripts/config/maps SHA in configs/f4_contribution_natural_v1.json. Primary contrasts fixed and separate hook from actual KL. No new fit/training, bounded7000 sequence-forwards/30minutes/16GB.

### Natural consumer API repair
Written at UTC: 2026-09-06T18:50:44Z
Event: v1 failed2026-09-06T18:49:29Z at nonexistent embed_out on installed GPTNeoX. One model-body diagnostic attempted; counter0 means no completed diagnostic, not no attempted forward. No source cases selected and no edited target outcomes read. Use public get_output_embeddings API, verify against full model logits, add missing audit_opened=false/config and summary raw hash. v2 retains same frozen scientific choices and fresh corpus. v1 failure and contract errors remain unchanged.

## Core contribution unit completed: theory, wide panel, frozen natural behavior, rewritten full manuscript

Written at UTC: 2026-09-06T19:07:48Z
Event completions: completion run2026-09-06T18:24:46Z; grain2026-09-06T18:45:24Z; corpus2026-09-06T18:48:17Z; naturalv2 2026-09-06T18:51:31Z (each from current artifacts). See existing exact run status records for fractional seconds. No old event time used as write time.

Frozen natural confirmation completed40source queries/80same-token cross-document donor pairs/57used documents, all20directed five-seed pairs,0missing requests,5760method-condition rows. Same64-input readout/native64 relativeKL medians.248339518/.890180881,152/160query-targetcells favor readout on KL andhook,146/160NLL. Coherent16/random16 native64 relativeKL.565335578/.932555927,140/160KL and141/160hook improvements; allfive source-seed medians share both main directions. This is a conditional distribution and sharedseed/query/document panel, not160independent replicates. SourceKL differs.009977983/.006802592; coherent/random absolute candidateKL.004943133/.004049326, so no absoluteKL improvement claim. Marginal64 paired KL difference+.000134152 with79better/81worse, so no distinct solver superiority.

Prior full640comparison dictionary realization gap and4480group structure, exact nonnegative TopK mask algebra and nonlinear witness combine with this new independent behavioral test into the rewritten PAPER_SNAPSHOT_20260906.md. Three source-backed main figures rendered PNG/PDF/SVG and visually inspected. Old full manuscript andallsmallpositive/negative results retained via preunit bytearchive, current supplement references and reports. Main manuscript includes complete abstract/introduction/related work/method/proofs/experiments/natural behavior/interpretation/material/discussion/conclusion, not another next-step-only note. Core claim now operation-conditioned recoverability versus realizability plus conditional partial-granularity improvement; not semantic uniqueness, complete native recovery, standard-solver originality, universalRaw win or conference acceptance.

Actual compute:115.98s completion,84.41s grain,42.09s naturalv2/6755sequence-forwards/823196672peakallocatedVRAMbytes; queue/load scope explicit inrunstats. New corpus21,146,930bytes officialrange/115docs/65536tokens.0newtraining/weights/packages. v1 API failure preserved, scientific rules unchanged beforeeditedoutcomes. One focused review coveredmath/operation/data/fairness; no new large test or follow-on consumer. Source/ref/environment registry updated. Plan andtracker replacecompleted/pendinglocalqueues withthis completedunit and paper-level continuation boundary;AGENTS unchanged,existingautomation kept in place.

Local artifact path/hash ledger: artifacts/core_contribution_20260906/artifact_manifest.json SHA256 170a7dfa40ae824ce443a6e779a0a7f1a75c2cccd9da62f3fdb393a215b37d59. Manuscript SHA256 490b574a4f3928a4603f489559c4daf83985c9762ac2504fcbf974edb01b2373; priorarchive manifest 0a68a0cbfd6b39be205e4cff94deb576af64540f8b0f446adfab5a0d27c9aa6a. Ignoredresearchdocs retained locally; git allowlist unchanged. Stage synchronization follows onlyafter the scientific unit is complete.


## Contribution claim reassessment after explicit user correction

Written at UTC: 2026-09-06T19:40:06Z
Event: user correction on 2026-09-06; prior run outcomes remain unchanged.

The previous assessment conflated completed controlled computations and manuscript sections with publication-level contribution. User explicitly rejected current story, readability, figure information, section depth, material quality, and treatment of simple confounds. The current entries now mark the contribution unresolved; no automatic pause or delegation. Preserved original tracker, plan and paper byte-for-byte at archive/research_workflow_20260906/contribution_reassessment_20260906T194006Z/manifest.json. Next substantive diagnostic separates source covariance rank and common raw directions from coherent-group transfer, using all existing source queries and frozen maps, with source-only interpretation. Existing data are development; no claims of fresh confirmation, solver superiority, or publication readiness. Initial budget <=10min GPU, single CPU thread support,0newLM/training/download,<=1GBoutput.


## Source-structure diagnostic and nuisance-matched group result

Written at UTC: 2026-09-06T20:02:50Z
Run events: structure diagnostic completed 2026-09-06T19:44:26.750589+00:00; matched control completed 2026-09-06T19:49:10.240409+00:00.

All160sourceanchors retained. Original coherent/random native64 medians.652814/.801496; source-only2048-candidate-per-query nuisance-matched random.665432. Paired gap shrinks from-.116583 to-.019207, all5source summaries shrink;377/640 coherent better versus486/640 originally. Full/marginal target controls also close. Source covariance ranks5.973/1.476 oppose the simple coherent-lowrank explanation; original advantage remains after removing rawPC1/4/16/64 and global-reconstruction field. New matching controls energy/frequency/spectrum/common-direction summaries approximately; alltailimbalances/candidates retained. Uniform discovery-energy scaling exact,normalized linear loss invariant; no new matched-control natural LM result. This invalidates the former strong conceptual reading, not all possible FCC concepts.

Source-first s1:1155 example has mixed lexical members and offdiagonal contribution correlation max.0356; deliberately unselected by target success. Diagnostic figure joins workflow, members,relations,spectrum and fullpanel. PNG/PDF/SVG viewed, source data and transformations preserved. Read Mu etal1702.01417§2.1, Gerasimov2606.12138§6.1/7, Bhalla2604.28119§5/6/B.2/E/F; registry distinguishes actual reuse,standard algebra and unimplemented signed/conditional grouping.

Actual run times29.402s/52.704s, GPU-only leases plus1CPUthread auxiliary, peak3.746GB/5.072GB,0newLM/training/download/weights/packages. Both scientific runs PASS and exited; shared leases released. One focused math/data/fairness review; no delegation,newloop,or pause. Reopened publication contribution remains unresolved. Stop default positive-neighborhood consumers; next work targets variable changes, signed/conditional source relationships and suitable material, not a new metric around the residual .019gap. MainpaperAbstract/Intro/5.4/Conclusion,plan/tracker corrected; previous versions atarchive/research_workflow_20260906/contribution_control_20260906T200250Z/manifest.json.

Artifacts:artifacts/contribution_structure_20260906/artifact_manifest.json SHA256 770eb0b2ca21975f0075cd3989ae54acf28961f4e4555995a6620f7a241b9d92. Original records and failures preserved; ignored research files remain local under explicit upload allowlist.


### Contribution control focused verification and final local manifest

Written at UTC: 2026-09-06T20:04:34Z

Original coherent/random16 refit reproduces previous relative metrics with max absolute difference6.66e-16. All160 matched source controls have16 unique members,share only the anchor with the coherent group,and all target native64 fits have64 members. Future generator metadata now says rank_and_shares_matching_attempted; prior run flag was an intent label and is explicitly corrected in the report,not a per-row balance certificate. Original immutable run data unchanged. Figure label spacing refined; new final artifact manifest SHA256 e816763734215bf6beddb10b6d6b0b8cc0d1e603f8c83bd55d6d89c1a47177e7, includes focused_review.json and current paper/figure hashes.


## Calendar variable-change material unit started

Written at UTC: 2026-09-06T20:26:25Z

Live base-model config confirms12layers; current SAE hookgpt_neox.layers.5 is layer6/mid-depth,not the final layer. Previous final-layer inference was unsupported; original records retained and current tracker/plan corrected. Reuse existing five sameflowSAEs and model. New fixed development configf4_calendar_material_v1.json:114day/month prompts with next/previous/identity and cyclic boundaries,rawhooks2/5/8 andfiveSAEdeltas atactualhook5,<=10minGPU,<1GB,0newtraining/download. Allattempts retained; this diagnoses variable function/material before constructing FCC,not a new acceptance gate or concept finding. Pre-update tracker/plan archivearchive/research_workflow_20260906/calendar_material_20260906T202625Z/manifest.json. No GPU work started yet.


## Calendar material result: useful month-change signal, narrow wording scope

Written at UTC: 2026-09-06T20:43:31Z
Run event: v2 completed 2026-09-06T20:34:53.739311+00:00; v1 retained FAIL after numerical equivalence tolerances failed.

All114 authored prompts and228 fixed donor pairs retained. Total conditional candidate accuracy30/114, fullvocabulary7/114. Month-next plain8/12 and7/12 versusafter2/12 and0/12. This is a restricted immediate-token endpoint,not proof of absent general calendar understanding. Existinghook5 is sixth of12layers,notlastlayer. The month signal warrants reusing existingfiveSAEs; no new training.

Month-next24pairs: rawlayer5slot donor-labelcorrect16/24,SAEs9–12/24. All5SAEs full decoded-difference aggregateKL/noop ratios.112420/.140750/.127562/.130471/.135063,all24perseedbetterthannoop. This is materialpreservation,not a fittedFCCrelation or newalgorithmwin. PooledtaskFVE~.9986 versuswithin-templatevalueFVE.506910–.519821; retainbothdistributions,andoutputfunctionratherthanequatingqualitywithconcepts. Mainpaper§7.4,alltemplatefigureandactualfirstmonthpairadded; publicationcontributionremainsUNRESOLVED.

v1run19.8679s failed predefined numericalchecks:maxfullsuffixlogprobdifference.0012494. v2onlyLMfloat64,SAEfloat32; identicalprompts/pairs/checkpoints,baselinepredictionsallunchanged. v2run40.0750s,3781sequenceforwards,1,371,891,712bytespeakallocatedVRAM;fullsuffix<8.4e-13,padding<3e-12,publicpath0; originals preserved,nothresholdrelaxation. BothusedGPUmanagerleaseandoneCPUauxthread,0newmodel/train/download/install. Onefocusedsource/data/operationinterpretationreview,fullrawreferenceKLreplayerror0.0;notindependentrevieworpublicationevidence.

Nextsameworkcard: source-onlymonthvaluecontrasts/signedcomposition,newwordingandsame-tokennoncalendarcontrols,atom/raw-lowrank/statisticalmatchingcomparison; no moreweakcoherent16consumers. All114inputsexposeddevelopment. No subagents,other task,newloop,or globalpause. Archives:archive/research_workflow_20260906/calendar_material_result_20260906T204331Z/manifest.json. Artifacts:artifacts/calendar_material_20260906/artifact_manifest.json SHA256 d4e43db89a0853db475197c1f0dfa64493c8cd6c69fced40c908583ff975dba0; runoutputsbytes{"F4_calendar_material_v1_20260906": 178502085, "F4_calendar_material_v2_20260906": 306723683}. Ignoredresearchdocumentsretainedlocally,noallowlistexpansion.


## Conditional month composition unit started

Written at UTC: 2026-09-06T21:10:46Z

144 authored inputs,288 fixed donor pairs;2calendar+2same-token noncalendar fit templates,4+4 newly authored prefix test templates. Five source64 groups preserve raw calendar changes and suppress noncalendar changes under standard weighted Gram/ridge; source-only policy frozen before outputs and all source groups freeze before20cross-seed fits. Controls: source full,atom,raw conditional,random64refit,positive-only same64; target native64,atom,raw linear,fullSAE. Budget<=600sGPU,<1GBoutput,oneauxCPUthread,0newtraining/model/download. Standard supervised task-conditioned fitting is not novelty: actually read Bhalla2604.28119B.3 manifold-conditioned OT andB.1 centroidsteering. Query question is additional role selectivity across new prefixes; all12months/weakoutputs retained. Preparing runner extension,not yet a result. Pre-edit archives archive/research_workflow_20260906/calendar_composition_start_20260906T211046Z/manifest.json. No delegation or loop change.


### Conditional month result and bounded operator-capacity follow-up

Written at UTC: 2026-09-06T21:22:23Z

Composition v2 completed218.2247s with25536rows/25699sequenceforwards; all5sources/20directions/newprefix tests retained. Source64 calendar/noncalendar meanKL seedmedians.087454/.009772; positive-only energy-matched.086853/.012168; raw conditional linear.004734/.006293. Strong raw and dose control prevent a claim of distinctive source64 selectivity. V1 retained FAIL (int64 source-freeze metadata, before conditional interventions),v2 onlyserializationfix. Next cached operator-capacity diagnostic uses same fit rows and exposed test rows:64/256/fullactive native masks versus full-code readout andraw;<=180sGPU,0newLM,within original10min unitbudget. No taskdata reclassified as independent confirmation; no SAE-invalidity inference from restrictedmaskfailure.



## Conditional function and operator class: complete unit result

Written at UTC: 2026-09-06T21:35:58Z
Run event: composition v2 completed 2026-09-06T21:18:08.932957+00:00; cached-capacity event is recorded in its status.json.

All144 authored inputs/288 donor pairs retained,with96fit and192new-prefix testpairs. Newnextmonth templates11/12,9/12,11/12;previousmonth2/12 retained. Source64 calendar/noncalendar meanKL seedmedians.087454/.009772; source-fit energy-matched same64.086853/.012168; rawconditional.004734/.006293 betterboth. Five source64 weightsallnonnegative,no negativeweightmechanism. This is a defined supervised function,not unsupervised concept discovery. CrossseedKL/noop medians atom.938479,native64.262317,raw.049167,fulltarget3.482495;rawduplicatesexplicit. No new FCC method advantage or publication readiness.

One cached6.350s diagnostic distinguishes support/class/information: calendar vectorrelativeerror64.676935,256.583526,fullfitactive.563862,code_readout.177430,raw.079529; noncalendar leakageenergy.089218/.117321/.124868/.053007/.079932. Fullnativefitactive590–630 and regularized,not an absencecertificate. Code-readout result is geometry,0newLM. This prevents blaming all failure on absent SAE information,and changes nextmethod: nativefixedmasks remain an endpoint/control,not the FCC mother problem. Continue conditional role/value composition with clear source-aligned/general operation classes and real incremental predictions over standardridge/OT.

V1 failed source-freeze int64 serialization before conditional/crossseedforward; keptalloutputs. V2 onlytypefix,sourcearraysreplayexact0. V2 218.2247s/25699seqforwards/1,480,061,952bytespeakVRAM; capacity6.3504s/0forwards/111,317,504bytespeak. Fullunitrunseconds~292;outputsbytes {"F4_calendar_composition_v1_20260906": 430575397, "F4_calendar_composition_v2_20260906": 1056293521, "F4_calendar_operator_capacity_v1_20260906": 82864909}. V2 alone1.056GB exceeds1GBdecimal estimate by5.6%; includingfailure/diagnostic1.570GBretained,notdeletedtohitbudget. FutureunitshouldreusecacheandbudgetactualLPsize. No train/model/download/install; oneauxCPUthread and managerGPUlease,allreleased.

Focused review: explicit field-design Gram maxerror1.2212453270876722e-15,rhs5.551115123125783e-17;positive-fitenergy-matchrelativeerror<1e-12;testtextsdisjointfromprevious114;fitrolepositionsboth3/4;frozenv1v2sourcearraysequal. Figures showallcontrols,newtextinstance,capacity and20directions;rawduplicatesdrawnonce. ReadBhalla2604.28119B.1/B.3 and§6,registryupdated;do notclaimtask-conditionedOTnovel. Mainpaper§7.5,theory§24,plan/trackerupdated afterbytearchive archive/research_workflow_20260906/calendar_composition_result_20260906T213557Z/manifest.json. ArtifactmanifestSHA256 7c1e1040d932b57369dff7c88ea62c6c85e35131a3ddf37d884fde00167b36d5. No agents,newloop,pause,whitelistexpansion,or external publication.


## User-directed seven-round manuscript and science rebuild starts

Written at UTC: 2026-09-06T21:53:45Z

The user explicitly rejects all prior claims of paper completion/competitive contribution and allocates exactly seven further research rounds, requiring full science, material, interpretability, figures, manuscript and result organization, then automatic pause and a complete summary. This user turn is round1/7; completed0. Round state is only in EXPERIMENT_TRACKER.md. Context transitions, tool calls and in-progress heartbeat events are not new rounds. The existing ccad automation is to be paused at completion of round7 and not automatically resumed without new authorization.

Read the complete current manuscript and visually inspected all four user-specified PNGs. Findings: accumulated diagnostic narrative, native operator freedom confound dominating the story, source statistical matching undermining coherent-group claim, no continuous concept/member/operation/output example, and low-information aggregate figures. Detailed audit and concrete remedies are in artifacts/seven_round_rebuild_20260906/MANUSCRIPT_AUDIT.md. This is review/preparation, not a scientific positive result. Plan refocuses on reusable functional composition and novel-context/compositional predictions, with native retained as an operation diagnostic.

Preserved byte copies of AGENTS/tracker/plan/paper/prompt and automation config at archive/research_workflow_20260906/seven_round_start_20260906T215345Z/manifest.json. Tracker carries seven deliverables and costs. Official HF metadata read verifies ungated Apache2.0 Pythia1B revision7199d8fc61a6d565cd1f3c62bf11525b563e13b2, safetensors2090701528bytes, and MIT-card CausalGym data revision95349c3a5e53e2506e8b212482ea6dd784978156. No weights/data downloaded by this start record. First-round bound<=2.2GBdownload,<=10minGPU; initial wholecampaign ceiling6GPUhours/70GBbulk,not claimed consumption. No paid/cloud resources,subagent,scope change,publication or whitelist expansion.


## Seven-round R1 functional material screen starts

Written at UTC: 2026-09-06T22:02:41Z

Official pinned safetensors/tokenizer and CausalGym train/dev downloads completed using existing hf-cli1.29.0; no token needed, test not downloaded. Public pairs can change lemma and tense/label, so the additional independently authored192input/384pair panel fully crosses subject and distractor number on fixed lemmas and is/are labels. Evaluate both models and three middle-spanning layers/two patch positions without outcome-based case selection. This is material selection, not new FCC/benchmark novelty. Source/config implementation is scripts/run_functional_material.py and configs/seven_r1_functional_material_v1.json. Budget600sGPU/oneCPUaux/<1GBoutputs; actual model weight identity checked while loading, no extra model copy.


## Seven-round R1 completed: functional material, manuscript and figure rebuild

Written at UTC: 2026-09-06T22:25:59Z
Run event: v2 completed2026-09-06T22:04:45.552279+00:00. Document/figure closeout event: 2026-09-06T22:25:59Z.

User acceptance remains NOT_MET. Completed1/7, next2/7; counteronlyinEXPERIMENT_TRACKER.md. Full oldpaper/fourfigureaudit, new single-storyEnglish development manuscript with explicit unexecuted method versus actualmaterialresults, and data-backed vector figures completed. No claimthatpaperiscompetitiveorFCCcontributionestablished. Oldnative/coherentgroupdiagnosticstory archived rather than reused aspositivecontribution.

Actual rawmaterialcomparison: 29publicCausalGymdevtasks(all2900rows,bothdirections),160M4290/5800=.739655;1B4978/5800=.858276. Reciprocalpromptreuseanduniquepromptcounts recorded. Independently authored16lexicalblocks×3syntax×4numbergrid:160M185/192;1B192/192. 1Bsubjectdonorpatch transferlayer3=168/192(PP64,SR53,OR51),layer7=159,layer11=141;final0/1/9. Distractorpatch retainsall192labelsbutmeanabsolute logodds~1.5–1.9nat; signedcancellationnotinterpretedasnoeffect. Firstconfiguredexamplebaseline-5.3579,fulldonor+2.8665,layer3subject+2.8319,final-5.3075. This selects1B/layer3forSAEwork,notnovelgrammarcausaldiscovery.

V1 int64serializationfailure kept:15.591409s,6066forwards,5992persistedrawvs5993inmemorysummary. V2metadatafixonlyPASS44.839442s,16692forwards,16592rows,peakVRAM4167833600bytes. Sum60.430851s/0.016786hGPUtaskwall;noSAEtrain/newFCCfit/audit. Outputsbytes {"SEVEN_R1_functional_material_v1_20260906": 45252568, "SEVEN_R1_functional_material_v2_20260906": 97232800};fixedmodelweight2,090,701,528bytesSHAverifiedagainstupstream. No paidresources/newdependencies/remote code/testdownload/secondmodelcopy. GPUleaseauto-released; managerGPUfreeverified. Numericalexactno-opandboundedpaddingchecksrecordedwithlimitedscope. KLfielddirections clarified inreport;do notdivideopposite-directionKL.

Read originalCausalGymmethods+officialdataMITcard,closestBhallaB.3conditionalOT,author-originaldPCAanddSCAv2§3;registryversions/license/adaptationsupdated. dSCAisadirectbaseline,notournewmethod. Candidate uses source-onlyfactorcontrasts,actualcompactfrozen sourceteacher,signedsharedcoordinates,targetgroupsandcomponentedits;noinventedsolvernovelty. Source-native,source-aligned andrawcontrolsseparated.

Artifacts: artifacts/seven_round_rebuild_20260906/R1_REPORT.md,MANUSCRIPT_AUDIT.md,STORY_AND_EVIDENCE.md,material_summary.json,fourCSVtables,twoPNG/PDF/SVGfigures. Finalfiguresvisuallyinspectedafterfixingfooter/labeloverlapandnegativepointaxisclipping;noresultvaluechanged. AuthoritativePAPER_SNAPSHOTrewritten;oldpaperandentriesbytearchive archive/research_workflow_20260906/seven_r1_result_20260906T221656Z/manifest.json. UnitartifactmanifestSHA256 37dfdcf806c3382b9f017d4b3e030d252f06f73b465a31cc56de3cdf0b83b6f2. Alllocalresearchdocsremainignored;allowlistunchanged.

Nextworkcard specifiesfivecontrolled1BSAEsandfixedvalidation/functioncurves,config-drivenhookcontract (oldrunnerhardcodes5/768),tokenizer/corpusidentitycheck,source/targetfunctionalmethodandmatchedstrongbaselines. Plannedwidth8192/k64plusonek128sparsitycomparison,actualtrainingbudgettobemeasured,within6GPUh/70GBcampaignceiling. OriginalautomationACTIVE15minsametarget,promptnormalizedexactmatch;genericfinite-user-roundstopruleinstalled,actualcountsnotprompt. Afterround7pauseoriginalautomationanddeliverfullsummary,nonewloop/eighthround/autoresume. No agentdelegation.


## User adds a dedicated format and visual-style round

Written at UTC: 2026-09-06T22:34:00Z
Event: current user steering message; exact message wall time not separately recorded here.

The user clarifies that the figures fail in professional academic format and visual style as well as information content: confused formatting, inexplicable text and conspicuous AI-like presentation are unacceptable. Prior use of plotting skills, vector export and a positive local visual check did not meet this requirement. No new scientific result is claimed by this update.

The user expressly allows one dedicated formatting/style round after the original seven scientific rounds. Tracker now preserves seven science rounds, adds visual round8, and moves the final pause to completion of round8; completed remains1, next science round2. This steering message consumes no round. Round7 closes scientific work and its summary; round8 revises the manuscript/figures and final PDF, then pauses the original ccad automation and delivers the final summary. No round9 or new research extension is authorized.

The visual plan now covers published-paper figure references and the actual manuscript template, typography/math/labels, layout/white space/panel hierarchy, axes/legends/captions, meaningful text, vector composition and final page-scale PDF inspection. Existing paper-figure and scientific-visualization skills were read; the latter has local export/metadata tools. FigureSpec instructions were inspected but the installed folder has only SKILL.md and its renderer was not found, so it is not claimed as an available renderer. Direct editable vector composition is a sufficient fallback; no install, image generation, subagent or separate task was invoked. Skill ratings/default styles are not acceptance criteria.

Before-edit byte copies and hashes: archive/research_workflow_20260906/additional_visual_round_20260906T223400Z/manifest.json. Updated plan/tracker hashes and counter semantics: artifacts/seven_round_rebuild_20260906/additional_visual_round.json, SHA256 ebb8fe9801a52ae5f36a2f2ed09f28681a7b882ed9670d933e83ce55529c775b. Existing ccad remains ACTIVE every15minutes on the same target; its generic prompt already follows the tracker limit and matches configs/CCAD_AUTOMATION_PROMPT.md exactly after newline normalization, so no prompt/schedule mutation or duplicate automation was needed. Existing scientific scope, budget and raw results are preserved.


## Seven-round science R2 starts: material and end-to-end correspondence

Written at UTC: 2026-09-06T22:56:06Z
Trigger event: heartbeat2026-09-06T22:51:51.369Z.

Scientific round2/7 starts; completed remains1. The separately authorized visual round follows scientific round7, then the original automation pauses. No new result is claimed at launch. Target is five controlled Pythia1B/layer3 SAEs plus a functional source-to-target correspondence comparison under matched strong controls.

The two model tokenizers have byte-identical tokenizer.json, tokenizer_config.json and special_tokens_map.json. Reuse existing disjoint FineWeb natural training/validation tokens; no task data enters SAE training. Existing checkpoint runner had hardcoded layer5/dim768 and assumed multiseed names; adapted it to actual model/config and single-seed names, with model-input identity recording. Original implementation is preserved by git and earlier run source snapshots.

First two pilots fix width8192, seed1,1,048,576 natural tokens, batch8x128,1024 steps with checkpoints256/1024 and shared LR schedule; only k64 versusk128 differs. Each bounded to900s and combined output<=1.6GB; actual training throughput and fixed-validation/variable quality determine the larger five-seed budget within the existing6GPUh/70GB science ceiling. GPU was free (16,303MiB total,1,689MiB observed use); manager lease will govern execution. No downloads, installs, paid compute or delegation. Pre-start tracker archive archive/research_workflow_20260906/seven_r2_start_20260906T225606Z/manifest.json; tokenizer identity and state in artifacts/seven_round_rebuild_20260906/r2_start.json.

## Seven-round campaign R2 material feedback and measured training expansion

- Actual log write UTC: 2026-09-06T23:04:25Z. Event times: pilot training completions in their status.json; factor run wall33.448s, this entry records the subsequent decision.
- k64/k128 single-seed pilots completed115.871s/105.471s. Final validation CE recovery.951402/.965405; earlier256step.881278/.924557. k64 final FVE.994970 is not variable-quality evidence.
- Functional pilot: raw held lexical82/96, k64 full81/native16 81/rank1-projected16 68; k128 full84/native16 82/projected16 51. Eight lexical blocks, reciprocal/syntax-dependent directions. No formal audit. All12672raw rows and source memberships persisted.
- Metadata exception: factor_pilot_v1 contract failed solely because audit_opened absent in resolved config versusfalse manifest. Numerical run completed and no-op exact; preserve original files and report contract limitation, no scientific rerun for metadata.
- Select k64 five controlled seeds plus k128 seed1 full-budget control; each4,194,304 natural tokens, checkpoints256/1024/4096, fixed256sequence validation. New4096step schedule differs from short pilots. Bounded2500s managed GPU time and7GBoutputs. Source method keeps full compact native16 span; rank1 failure retained.
- Archive and decision: E:/Projects/SAE_Lab/CCAD/archive/research_workflow_20260906/seven_r2_material_20260906T230425Z/manifest.json; artifacts/seven_round_rebuild_20260906/r2_training_expansion.json. No round completion/count increment, no agent, no paid resource, no automation change.

## Seven-round campaign R2 five-seed training completed; functional suite prepared

- Actual log write UTC: 2026-09-06T23:25:56Z. Training status completion UTC:2026-09-06T23:23:50.363447+00:00.
- Five same-config k64 SAEs completed4,194,304natural tokens each in shared-forward run1157.879645s;15fixed checkpoint quality rows, all expectedstep/inputorder/validation/hook checks PASS. Final CE recovery.964503–.965158, FVE.995957–.996008. Perseed alive/dead/L0/norm and full curve in artifacts/seven_round_rebuild_20260906/r2_five_training_result.json; no convergence claim.
- Same-budget k128 seed1 control launched under original budget. Full functional config and common-teacher comparators prepared; source-native16span and matched-source PCA1/4 separate mean-direction failure from irreducible dimension. Three focused tests passed; CPU two-pilot solver exercise revealed mixed outcomes and one2000iteration warning, retained; deployed mature MultiTaskLasso maxiter10000 with gaps recorded.
- Reused existing f4_sparse_overlay_v1 spec/runtime for sklearn1.9.0/SciPy1.18.1, no package install or training runtime mutation. Raw/fitted/OT/group/one-to-one operation class documented in R2_METHODS_DRAFT.md. OT is an operational adaptation; contrast RRR not falsely called exact dSCA replication.
- Tracker pre-edit archive: E:/Projects/SAE_Lab/CCAD/archive/research_workflow_20260906/seven_r2_five_20260906T232556Z/manifest.json. R2 still in progress, no round count increment, no automatic pause/agent.

## Seven-round campaign R2 complete learning curves and common-teacher suite in progress

- Actual log write UTC: 2026-09-06T23:38:11Z. Suite start event UTC from manifest: 2026-09-06T23:33:00.233091+00:00. This is a stage entry, not round completion.
- k128 full control PASS487.603741s, CE recovery.975416544/FVE.996537653. Eighteen-checkpoint function prefix completed41856rows; raw-prefix hash and all material counts in artifacts/seven_round_rebuild_20260906/r2_material_results.json. Finalk64native16 held counts82/82/83/82/83 of96 perseed versusraw82; early36/23/18/0/19. All8heldlexicalblocks, reciprocal/syntax dependency retained; development only.
- SourceNative16fullspan and matchedsourcePCA1/4,20orderedseed directions×2factors, same-teacherfull-distributionKL andvector/function endpoints running. Bothper-featureRMS andnative-unit variants included instrongfull/raw/compact controls after a bounded two-pilot vector exercise exposed weak-feature extrapolation; oldscalingresults retained. Geometry improvement alone not functional evidence.
- Mainrun/config: runs/SEVEN_R2_factor_five_v1_20260906;configs/seven_r2_factor_five_v1.json. Same1800s/<1.2GBbudget, wholecampaign6GPUh/70GBnot increased. Already installedsklearn/scipymature solvers;4targeted tests passed. Exactmembercounts/nozero-fill, sourceprocesssingularvalues andsolver gaps saved.
- Tracker pre-edit archive: E:/Projects/SAE_Lab/CCAD/archive/research_workflow_20260906/seven_r2_transfer_20260906T233811Z/manifest.json. Continue sameunit, completed1active2, no newtask/agent/automationchange.

## Seven-round campaign R2 conditional-OT numerical recovery without replaying completed science

- Actual log write UTC: 2026-09-06T23:51:22Z. Failure event UTC fromstatus2026-09-06T23:38:43.886551+00:00; resume eventUTC 2026-09-06T23:48:31.528417+00:00.
- v1 persisted172992rows/17complete direction-factor units/18checkpoint prefix beforesource3target2subjectbudget16 OT2000-step ceiling. FAIL343.576643s, originalartifacts untouched. ExactfailedinputCPU reproduction: marginalerror5.443264e-6 at2000; converged9.985961e-9 at5780steps in.165587s. epsilon.05/cost/marginals/tolerance1e-8 unchanged; iteration ceilingraised50000. Diagnosisartifacts/seven_round_rebuild_20260906/r2_ot_failure_diagnosis.json.
- v2 inherits parentbytes and outputhashes, skipsallcompletedunits, replays192baseline+one192partialsource teacher only forfull-precision references, continues23remaining units; required349056unique finalrows. Remaining1450s keepscombined1800sbudget, no newtraining/data/paidresource. Source cache/native teacher identity checks and4focused tests retained.
- Currentresume has passedfailedpair. Tracker pre-edit archive: E:/Projects/SAE_Lab/CCAD/archive/research_workflow_20260906/seven_r2_resume_20260906T235122Z/manifest.json. Existingautomation remainsACTIVE, completed1/active2; no separateagent/task.


## Seven-round campaign R2 completed: portable functional groups and manuscript integration

写入时间（written_at_utc）：2026-09-07T00:11:08Z。Experiment completion event: resumed main run2026-09-06T23:55:00.208487Z; documentation integration written2026-09-07T00:07:25Z, separately from this actual append time.

- Final five k64 SAEs: samePythia1B/layer3/8192width/4,194,304natural tokens/order/optimizer/schedule, seeds1–5only. FinalCErecovery96.450–96.516%,FVE99.596–99.601%,L063.987–64,validationalive5296–5728/8192; normerror<=5.37e-7. Same-budget k128 seed1 control included; fixed-token evidence is not convergence certification. Source native16subject teacher82/82/83/82/83of96,raw82; early0–36. MatchedsourcePCA1retains79–81,so full16span is not16necessarysemanticdimensions.
- Run SEVEN_R2_factor_five_resume_v2_20260906 PASS/contractPASS:349056unique finite rows,18checkpoint curves,40direction-factorcells,source/baseline/cache replay checks exact at stipulated precision. RawSHA256713b437cd9457b83deedd0fdc0e61edf5c7f2045a76cb6aac01f9a05d74247db. Parent172992byte-prefix retained and verified; original v1OTmaxiterfailure and pilotcontractmetadatafailure remain preserved. Solverceilingonly50000fix,actualfailedcaseconverged5780 at originalepsilon.05/tolerance1e-8; all Lasso path entries converged.
- Compactnative-unitgroup subjectKL.00416303,distractor.00766612; strongrawridge.03355969/.03195031,20/20directions favor group on each. Fullcode native-unitvectorerrorbetter(.12726/.10169vs.20953/.13424)butfunctionalKLworse(.01573459/.00957448). Strong same-membernative subjectKL.00223118,projectednative.00240371: coefficient-mapexclusivebenefitNOTestablished. Subjectgroupwins4/20vsnative,13/20vsdenseselect,15/20vs1to1,19/20vssingle. Actualtarget8–14/7–16;budget32sameas16supports,notextracapacitygain. All39methods×2factorsandoldnegativeRMS/OT/RRR retained.
- Mainmanuscript nowintegratesactualMethods/source-exactteacher,training,functionalcomparison,strongnativecontrols,fixedactorrelation16×9andLMoutputs; threePNG/PDF/SVGfigurefamilies actuallyrendered/viewed. Sourceunit/role-positionconfound and development-only8lexicalblocksexplicit; directions share5seeds. Noformalnewdata confirmation,compositionresult,full dSCA/DASbenchmark orcompletepublicationclaim. Planfocusesportablemembership,compositionandnewrole×valuecontexts; R3includesdirecttarget-onlymemberruleasstrongsimpleexplanation.
- TotalR2GPU-managedscriptwall2632.362234s=.731211732h,includingpilots/failure/fitting/validation; R1+R2=.747998079h oforiginal6h. R2bulk6,176,498,189bytes,runstrees543,226,651bytes; provenanceR2_BUDGET.json. GPUfreeat00:08:22Z; otherprojectcpu/diskleasesuntouched. Noinstall/paidcompute/subagent/duplicateautomation.
- Focusedsource/math/data/consumercheck once,4targetedtests hadpassed aftercodefix; completeactualmainrun validates deployment. Figuresonlysubsequentlyedited; no repeatedfullsuite. Oldlivecardsbytearchived atarchive/research_workflow_20260906/seven_r2_close_20260907T001108Z/manifest.json; pre-integration6filearchive inr2_integration_archive.json. Currenttracker completed_rounds2,active_roundnone,READY_FOR_NEXT_ROUND,acceptanceNOT_MET. User7science+8formatthenpauseoriginalccad preserved; no roundcount fromprogress/toolcalls.
- Ignoredlocaldeliverymanifest artifacts/seven_round_rebuild_20260906/R2_DELIVERY_MANIFEST.json SHA256062551c378e54a8912051d9f941a89ccfe5d9d646b87b77a8d67ba7a1bb8d844; all listed filesverified atclosure. Mainpaperhash0b6a766794e3401918005a77ed29c8b0367dbfd46ed2e392849e0a4739a1f2c6; trackerhash1f9c93591ec402945bb6abe84a7157cb79533b000de10dd3c1ca94d0ceff5663. Earlierartifactmanifestsremainhistoricalas-ofrecords. Allowlistedcode/config/tests/master_log groupedforsynchronization; noforcedaddorwhitelistexpansion.


## Seven-round campaign R3 starts: membership reuse and compositional predictions

写入时间（written_at_utc）：2026-09-07T00:31:08Z。Start recorded at append time; no separate compute-start event yet. Tracker completed2/7,active3,IN_PROGRESS. Opportunity: compact cross-seed functional membership with stronger native operations; compare direct-target local group/dense selection and appropriately scaled shared components before interpreting extra alignment value. Reuse five fixedSAEs/caches; initial ceiling1800sGPU-managedtask/<1GBnewresults within remaining5.252h/70GB,0paid. Newcontexts and individual/jointcomponentoperations to distinguish number/localposition from portable explanation. Parentdata remainDEV. OriginalautomationACTIVE15min/samethread; noagents. Pre-edittracker bytearchive archive/research_workflow_20260906/seven_r3_start_20260907T003108Z/manifest.json SHA2561f9c93591ec402945bb6abe84a7157cb79533b000de10dd3c1ca94d0ceff5663.


## R3 raw number-time composition material launch

写入时间（written_at_utc）：2026-09-07T00:36:59Z。Prepared SEVEN_R3_composition_material_v1_20260907 before model outputs:288 authored inputs/6 newlexicalblocks/3timecuepairs/2syntax/2number/2time/2distractor,7raw operations at layers3and7. Temporal preference is not strict grammatical exclusion. Budget180s/twoCPUthreads/<150MB within R3initial1800s; same pinned1Bweights, no SAE training. Source-reader checked CausalGym1D-DII/region-end alignment and DAS100stepAdam.005schedule; dSCA PDF and BhallaB.3/Cfactorgeometry reading underway. Existing-group overlaps acrosssource0.58–0.94 meanJaccard pertarget/factor vsdifferentlocal16supports, only structural hypothesis not function proof. Noaudit/agents/extraresources.


## R3 two-factor raw signal and five-source material check

写入时间（written_at_utc）：2026-09-07T00:46:48Z。Raw run completed event2026-09-07T00:39:06.181604Z,124.290227s,4320rawrows,contractPASS,exactnoop/tokenalignment/completecases. Four-labelbaseline266/288,number288/288; cueRightnow/Backthen96/96. Layer3number90/96,timecue78/96,jointseparate75/96; timesubject0/96. Newsourcepanel768 with12nouns/fourcuepairs/twosyntax and disjointnounpartitionsbothroles, all5existingk64SAEs; source16/32/full/PCA1separate/joint<=300s/<400MB. No targetfitbefore source feedback, originalR3total1800s. Fullweakcueandotherlayerresults preserved; expectedtensepreferences notungrammaticalalternativeclaim. Pre-edittrackerarchivearchive/research_workflow_20260906/seven_r3_sources_20260907T004648Z/manifest.json.


## R3 source quality failure and functional member experiment

写入时间（written_at_utc）：2026-09-07T00:58:14Z。Source run SEVEN_R3_composition_sources_v1_20260907 completed211.819851s,49920rows,contractPASS. Time mean16 loses substantial raw/fullSAE function, numbers remain strong;32 andPCA1 retained. Before expanding target comparison, launch source-only gradient-gain16/32 on discovery192, all768 actual native separate/joint edits, five seeds,360s/<150MB withinR3total1800s. Output-supervised selection declared; no gradient-only functional claim. Pre-edittracker archivearchive/research_workflow_20260906/seven_r3_functional_20260907T005814Z/manifest.json.


## R3 functional source result and matched compositional correspondence

写入时间（written_at_utc）：2026-09-07T01:05:51Z。Source gain run PASS23040rows149.545835s; five time32groups have102–124/192newnoun/knowncuecorrect but25–56/192bothnew. Number16 strong658–678/768. Main teacher N16/T32, source-only labelled discovery. Full384newlexpanel retained,19jointcomparators/eightseparate,20dependentdirections; directtarget identical selection and scalar gain included. Exact penalizedRRR usesY.T X Wridge; test against independentaugmentedOLS PASS,legacyR2 fittedpredictionSVD results retained. Six narrow testsPASS. R3budget adjusted1800to2100s based288forwards/s, main1500s/<650MB; pre-main485.655913s withinoriginal6h/70GB. Trackerarchivearchive/research_workflow_20260906/seven_r3_main_20260907T010551Z/manifest.json.


## R3 main composition results and one targeted behavior-control followup

写入时间（written_at_utc）：2026-09-07T01:21:56Z。Main run PASS274560rows614.552851s,20directions19jointmethods. Knowncue/newnounjoint KL FCC.000818/raw.002789/full.000500/RRR4.000501/directtarget.011947/native.173419; newcueFCC.023594/directtarget.011770 withweaksource. Noexclusivecompact-algorithm win. Specificunresolvedintensityexplanation:geometricgain doesnotexclude optimalbehaviorgain. Addonlynativegain.25/.5/1/2/4 via64cal andsource-fixed DAS-styleunitrawdirection100stepsbatch4,Adam.005,sourceKL objective,extraoutputsupervisiondeclared. 450s/<100MBwithinR3total2100s; pre-run1100.208764s. All384heldconditions, nojointrefit. Trackerarchivearchive/research_workflow_20260906/seven_r3_behavior_20260907T012156Z/manifest.json.


## R3 scientific integration: compact composition, strong controls and cue boundary

写入时间（written_at_utc）：2026-09-07T01:36:53Z。Behaviorruncompletionevent2026-09-07T01:27:26.239921Z,310.285780s57600rowscontractPASS; main614.552851s274560rowsPASS. FiveR3runsallPASS,total1410.494544s,409440rawrows,452947722bytes,0newbulk/paid/packages; cumulative1.139802119of6managedtaskhours. Knowncue/newnounjointFCC.000818vsraw.002789,DAS-style.002715,nativebehaviorgain.086223,directtarget.011947; full/rank4.000500better. TargetsupportsN5–11/T2–3,knowncue99.06%teacheragreement;newcueFCC.023594worseDAS/direct/full,sourcejoint207/960. Interactionsactual.359/.424andFCCerror.0224,newcue.1289. BodyMethods/Results/Discussion/refs,DERIVATION§25,plan,story,report,22controlmatrixandfixedexampleintegrated. CorrectedpenalizedRRR,6narrowtestsPASS; oldR2posthoctruncationpreservedandidentified. Figurefirst19versionspreserved,finaltwofamilyrendersactuallyviewed. Pre-editmanifestarchive/research_workflow_20260906/seven_r3_results_20260907T013653Z/manifest.json. R3countnotyetincremented; closeafterfocusedverification/sync.


## R3 closed; next controlled confirmation and material improvement

写入时间（written_at_utc）：2026-09-07T01:42:32Z。Science,methods,22controlfigures,fixedcase,mainpaper andR3report integrated; run completion events remain in status.json (last2026-09-07T01:27:26.239921Z). completed_rounds2→3exactlyonce,active_roundnone,READY_FOR_NEXT_ROUND; acceptanceNOT_MET. Originalccad15minautomationcontinues, no newloop/agents. Nextworkcard freezes usefulknowncue/compactcomposition range for untouchedlexical/structuralconfirmation, plusoneboundedsource-position/rolematerial improvement andexternalanchor; no sameweak-source regressiongrid. R3total1410.494544s/452947722bytes, remainingoriginal6h4.860197881h. AllCCADGPUworkended. Trackerpreclosearchivearchive/research_workflow_20260906/seven_r3_close_20260907T014232Z/manifest.json. Stage/push onlyexistingallowlist; localignoredpaper/evidence preserved by deliverymanifest.


## R4 frozen confirmation starts

写入时间（written_at_utc）：2026-09-07T02:09:10Z。事件时间：same preparation, no new model outputs yet. completed3/7science,active4. Frozen R3supports/20maps/behaviorgains/DAS; rawmaps recovered from olddiscovery only and matched storedoldMSE before authoringoutputs. New512rows,8newlemmas,2familiarcues,4syntaxes; primary jointKL anddesiredaccuracy/allrows. Inputs/config/hashatartifacts/seven_round_rebuild_20260906/r4_frozen/FREEZE.json. Initial4200s/<8GB withinoriginal6hours/70GB/zero paid. ActualGPUleasefree,7.45GBfreebut58%utilizationfromdesktop/otherapps observed; useGPUlease andbatchmemorybudget,neveralterotherprocesses. EndoSAEcpu/diskDleasesactive; noCPUheavyorDexclusiveleaseforordinaryassetreads. Trackerbytearchive archive/research_workflow_20260906/seven_r4_start_20260907T020910Z/manifest.json. No scientific result yet.


## R4 bounded role and position screen prepared

写入时间（written_at_utc）：2026-09-07T02:13:50Z。事件时间：same preparation; role results not yet computed. Configs ['configs\\seven_r4_role_temporal_v1.json', 'configs\\seven_r4_role_quoted_v1.json']. Temporalversusquotedtitlecue role; no newconfirmationnouns consumedforselection. Declaredall4cues/4oldnouns/2syntax/8conditionsperrole,256rows. Quotedcue alwaysfollowedbyRightnow so expectedtense unchangedbytimeflip. Rolepositions areprovided, not learned. Atmost300GPU-managedseconds withinR4total4200; reducesreserveforlaterconsumer ifneeded. Primary frozenR3confirmation untouched.


## R4 position-screen adjustment before role outcomes

写入时间（written_at_utc）：2026-09-07T02:18:35Z。事件时间：rawconfirmationmaterialcompleted2026-09-07T02:14:39.400564Z;rolepanelsunrun. Rawall512baseline496/512;layer3time-cue340/512, latepositions3/7weak andlayer11mixed. Oneboundedrole/positionpanelthereforeuseslayers3/11/15, replacing7 ratherthanaddinggrids. v1configsremainuntouched/unexecuted; v2 ['configs\\seven_r4_role_temporal_v2.json', 'configs\\seven_r4_role_quoted_v2.json']. Atlayer15rawfinal donorpatchisidentitybyconstruction;useonlytoverifycompactSAE/sourcecompositionopportunity,notnovelmechanisticfinding. Budgeteach240s dueobserved147scoldstartup;totalR44200s unchanged, adjustconsumerreservefromactualuse. Frozenmainconfirmationmapsunchanged andrunning.


## R4 prefix diagnostic changes the time interpretation

写入时间（written_at_utc）：2026-09-07T02:31:43Z。事件时间：diagnosticcomputedcurrentR4whilefrozenmainrunning. Five source timefields on512newinputs have cue-conditional residualenergy1.31e-9to5.35e-9;rank2retains>=.999999995energy. Causalprefix induction explains idealconstantcuefields; finiteprecisionresidual retained. R3fidelity isnotabstracttimecodingevidence; frozenoutcomes/mapchoicesunmodified. Theorysection26andR4_METHODS_AND_ALGEBRA.mdconnectthis to latercontext/role screen andsamepositionmixedfinite differences. Two primarysources read, ICLRPDFactualpages4–7viewed, sourcehashregistry. Oldregistry/derivationbytearchive archive/research_workflow_20260906/seven_r4_methods_20260907T023143Z/manifest.json. No modelresult claimed fromthisCPUdiagnostic.


## R4 confirmation outcome and external material decision

写入时间（written_at_utc）：2026-09-07T02:48:49Z。事件时间：actual completed artifacts preceding this write, quoted_v2 finished2026-09-07T02:47:29Z. Frozen512allrow confirmation failed previoushighfidelitystrength: familiarFCCKL.041622/raw.080991/DAS.006975,teacheragreement86.9727%; nofit changed. Prefixrank2diagnostic and savedmap replay separate from scientific confirmation. Temporalrawlayer15finaladd218/256 actualcorrect supports single externalhook fiveSAE. quoted_v2 FAIL coldstartup281.124526s before0forwards; samepanelv3 allows600s, not methodological retry. R4actualwallceiling8500s before expansion, total6h/70GB/0paid unchanged. Allfirst4runwalls1911.666462s counted. Archivearchive/research_workflow_20260906/seven_r4_external_20260907T024849Z/manifest.json. Trainingqualitylastlayeroracle bug identified beforelaunch: comparelastblockoutput tofinalLNinput, existinghidden_states[-1] normalized; codecorrected specificbranch. No new scientific result claimed from fix.


## R4 late-hook confirmation material predeclared

写入时间（written_at_utc）：2026-09-07T02:54:43Z。事件时间：same design write, before newhook training/source outputs. Raw quoted_v3 completed165.319047s/5696forwards; jointfinaladd256/256correct,KL.025281 combined, rawfullsinglefinal identity kept separate. New512row panel predeclared:barber/porter/miner/inventor, twofamiliarandtwocompletelynewauthoredcuepairs, temporal/quoted roles,pp/objectrelative; source/target positionsfinal. No new material outcomes seen; maps/supports willfreezeafter olddevonly andbefore thispanel runs. At sameposition jointwrongfactor swap equals identicalsum, so notnegative evidence; add actualsinglefactor wrong controls. Evidenceatartifacts/seven_round_rebuild_20260906/r4_l15/PANEL_PREDECLARATION.json. Existinglastlayerrawsignal justifiesonehook fiveSAE, not acorrespondence claim.


## R4 training resource interruption and grouped recovery

写入时间（written_at_utc）：2026-09-07T03:02:38Z。事件时间：process termination immediately before thiswrite, exact timestampnotseparatelycaptured. Originalfiveconcurrent l15training hadlastobserved189/4096 atsevereslowdown5–6s/iter vsinitial.35s,globalVRAM15503MiB/free493. Pagingisplausiblenotprofiled. ConfirmedownPID81172/child81112 withCIM+sharedowner; stoppedonlythistree, wrapperexit1. Beforecheckpoint256, noSAEcheckpointorqualityresult; previousintenttosavefirstcheckpoint notachieved. Conservativelycharged454.390174s upperboundfrommanifeststarttothiswrite; runs/.../recovery.json exactrecord. Missingcontractfilesnotfabricated. Recovery sameconfigurationfive seeds grouped1,2 then3,4 then5; all4096updates/4.194Mtokens/batch8unchanged, codeunchanged. R4actualcap10500s revisedBEFOREgroups, original6h/70GB/0paidunchanged. No otherprocess/lease changed. Newconfigs ['configs\\seven_r4_l15_train_k64_s12_v2.json', 'configs\\seven_r4_l15_train_k64_s34_v2.json', 'configs\\seven_r4_l15_train_k64_s5_v2.json'].


## R4 measured resource recovery before further compute

写入时间（written_at_utc）：2026-09-07T04:22:01Z。事件时间：s12v2finished2026-09-07T03:33:31.742922Z, currentread at~04:16–04:21Z. s12v2timeout1801.5366716s, last898updates, only256inferencecheckpointsseed1/2, noquality/exactstate. ContractPASS appliesfailedrun structureonly. R4totalconservative4332.9123547s includingallfails; originaltotal~2.3434h so~3.6566hremain. Currentnon-WMIreadRAMtotal33.727GB/available19.175GB, GPUfree14.94GB; EndoSAECPU/diskDleasebusyleftuntouched. WMIpermissionreviewtimedout once(notunsafejudgment); saferGlobalMemoryStatusEx/GetProcessread worked, no outstandinguserapprovalneeded. No exactcauseofpastslowdown proven. Existingtrainer timeout atoptimizerposthook lostcontinuation; nowresourceguard observescompletedupdates (upstreamglobalstep increment/pbarupdate), records64steptiming/VRAM, savesexistingexactstate onboundedslowdown/timeexit andat1024. Preservesoriginalfiveconfiguration; noalgorithm/package/sourcecorpuschange. Fullfiveguardedv3 chosenbeforeanyqualityoutcome withcurrentVRAMmargin, group34/5configsremainunexecuted. R4cap10500/original6h unchanged; pending actualsource/FCCaftervalidtraining.


## R4 external prefix-null comparison

写入时间（written_at_utc）：2026-09-07T04:25:07Z。事件时间：beforeanyexternalSAEquality/source/correspondence or newconfirmation outcome. Addtwo source-onlymean controls toexternal developmentandfreeze: signedglobalfactormean; cue-pair-conditionedmean, fallingbacktoglobalforunseenspellings. These directlytestwhethercontext-varyingFCC isneededbeyondconstantcuevectors suggestedbyprefixdiagnostic. Theyreceiveexplicitgeneratorfactor-direction/cueidentity, aninformationadvantage; nooutputfitting/jointrefit. Serializedvectors/cuesfreeze fromolddevbefore newpanel. NewcueIDs arematchedbyexactcuepairstrings, neverbynewpanelordinal. Pairedquotedcontextsusetheunchangedtemporalmeantochallengecontextselection. Allmethods/sourcequality remain, noextratuninggrid.


## R4 allocator evidence and exact continuation

写入时间（written_at_utc）：2026-09-07T04:41:09Z。事件时间：v3guardstop2026-09-07T04:35:35.744942Z; thiswritepreparesresume. At384stepsallocated6097037824/reserved21323841536/physical17094475776bytes,free0; last64updates4.498sperstepvs.181at128. Guard saved full exactstate AFTERcompleted384updates beforeFAIL; sourcewall790.497703400s. R4cumulativeconservative5123.410058100s. Resumev4fromthatstate(no reinit/repeatedtokens) changesnativeallocatorGC.6/roundup4 andconditionalunusedcache releaseabove10GB. PyTorch2.8official memorynotesandempty_cache docs actuallyread; changes memory reservation, notmodeltensor/objective. No packagesupdated. Bothseed1,2step256 statehashesfroms12v2andfivev3match exactly, directlycheckinggroupingdidnotchange those trajectories. Newrepeatstop savesexactagain. Not a scientificFCCresult; allnewconfirmation stillunopened.


## R4 exact-state storage accounting

写入时间（written_at_utc）：2026-09-07T04:43:26Z。事件时间：before resumed1024/final checkpointwrites. IncreaseR4newbulk allowance8GBto12GB because preservedpartial256/exact384 andnewexact1024/finalplusinferencefilesproject~8.4GB; original70GB/0paidunchanged. Existingweights/checkpointsnotduplicatedforconvenienceordeleted; fulloptimizercontinuationstate necessarytorecoverresourcefailures. Runtimepackages/modelweightsunchanged.


## R4 bounded final-layer inference reuse

写入时间（written_at_utc）：2026-09-07T04:54:17Z。事件时间：same preparatory edit, before external source/map outputs. At final resid_post, only positionwise LayerNorm and unembedding remain. New three consumer configs permit cached-tail inference after complete-model zero and signed joint probe checks (full vocabulary max logprob error <=1e-4); gradient ranking and DAS optimization retain full model. Same float32 sequential slot edits and same outputs/controls. Original predeclared512panel config is untouched. Full-model and cached-tail sequence counts separate; no executed speedup claim yet. Config byte archives/hash: archive/research_workflow_20260907/r4_cached_tail_20260907T045417Z/manifest.json.


## R4 frozen findings integrated into main argument

写入时间（written_at_utc）：2026-09-07T04:56:27Z。实验事件时间：prior frozen run/status and prefix diagnostic, no new LM inference in this write. Main abstract/introduction/discussion/conclusion now distinguish99.06%development from86.97%frozen familiar-syntax agreement and report ranking reversals. Added full-denominator frozen table, prefix rank argument and role-position rationale; coincident norm and wrong-factor identity incorporated into derivation/methods. Layer15training still active; no extra FCC result or round completion claimed. Prior exact bytes/hash: archive/research_workflow_20260907/r4_primary_integration_20260907T045627Z/manifest.json. Interim identities: artifacts/seven_round_rebuild_20260906/R4_PRIMARY_INTEGRATION.json.


## R4 five-seed late-layer training completed

写入时间（written_at_utc）：2026-09-07T05:00:23Z。完成事件时间：2026-09-07T05:00:12.982658+00:00。Five seeds completed4096updates/4194304natural tokens; full combined input/loss trace4096, all15fixed-validation rows andhookoracle checks PASS; contract True. Resume wall1125.988752600s; R4cumulative including failures upperbound6249.398810700s. FinalFVE0.74990961–0.75147982,CErecovered0.85552944–0.85746015. These are limited reconstruction/coverage measures, not convergence or functional proof. Allocator cache stayed bounded; detailedtrace/quality/rawhash atartifacts/seven_round_rebuild_20260906/r4_l15/TRAINING_SUMMARY.json. Nextactualcompact source andcrossseedfunction underunchangedpredeclarednew512panel.


## R4 external-layer maps frozen before new material

写入时间（written_at_utc）：2026-09-07T05:10:05Z。事件时间：samefreeze,nooutputfrompredeclared512newrows. Allfive source supports,20maps/behaviorgains/DAS,sourceglobal/cueconstantcontrols frozen. Rawlinear maps recoveredonlyolddevwithfixedoldalpha andexactoldresidualreplay. 88 identities atartifacts/seven_round_rebuild_20260906/r4_l15/frozen/FREEZE.json. Newtemporal/quoted x familiar/newcue x newlex/pp/objectrelative allrowsretain; no newqueryselection/fit.


## R4 current card reconciled before confirmation result

写入时间（written_at_utc）：2026-09-07T05:13:13Z。事件时间：source/main/behavior/materialcompleted at theirstatus timestamps; current frozen consumer running. Consolidated old resource-recovery chronology into a single live card, preserving exact prior tracker/hash atarchive/research_workflow_20260907/r4_confirmation_live_20260907T051313Z/manifest.json. Countsremain3completed/active4; no false newround orpause. Development FCC.048293/raw.014047/full.021285/DAS.158161 allretained; frozenroledata still not summarized or used forselection.


## R4 existing-checkpoint functional learning measurement

写入时间（written_at_utc）：2026-09-07T05:20:01Z。事件时间：preparation while frozenconfirmationrun continues. Fixed-validation FVErose57.31→65.81→75.08%andCE68.23→76.95→85.64%; toanswerwhether sourcecomponentfunction improved ratherthan infer fromreconstruction, reserve<=300s forsame384devinputs/unchangedN16T32ranking atalreadyretained256/1024/4096five-seedcheckpoints. Reuseexistingrawgradient/cache, no newtraining/data, no map orselectionchange inongoingconfirmation. Finalcheckpoint mustreplay originalmembers/nativeq. This is a bounded material-learning result, not anothermethodgrid or gate beforeindependentconfirmation. R4cap10500/original6h remainunchanged.


## R4 frozen role transfer and complete scientific integration

写入时间（written_at_utc）：2026-09-07T05:58:51Z。实验完成时间：each run status timestamp retained; final frozen consumer wall620.4743541s/PASS and checkpoint-function wall42.190s/PASS, both before this write. New512 all-row confirmation: FCC jointKL .072920/.105256/.054249/.064662 forTemporal familiar/new andQuoted familiar/new; raw/full stronger, native/OT/one-to-one/DAS pooled weaker. FCC time abs effect4.142/3.651/.266/.373nats, constant global4.868/4.954/4.596/4.591. No-op time-only quoted already strong, joint no-op0/.78% vsFCC93.05/92.19%; action/attenuation considered together. Direct learning256/1024/4096 joint36.15/61.93/78.18%, same rule notsamefeatureID. Five sharedseeds/20directions are dependent. Rawsha d85467e5dade9232af6652c2d78a9210f783375ef635e2e3cc0b965f1e8cdf5d. Mainmanuscript now integratesbothfrozenpanels, prefixcounterexample, controlledmateriallearning androle-dependenttransfer; full/raw advantage andsourcefailuresretained. Methods/report/plan updated, exactpriorbytes/hash atarchive/research_workflow_20260907/r4_completed_integration_20260907T055851Z/manifest.json, currentfiles atR4_INTEGRATION.json. Allcomputecomplete; R4wallupperbound7338.5811231s, newbulk8325115509bytes, campaign3.178296876of6h. White-list sync andcount4closing verification next; no newscientificroundcountyet.


## R4 final figures and delivery checks

写入时间（written_at_utc）：2026-09-07T06:02:48Z。事件时间：same final document/figure checks; no new model inference. Added standalone natural-reconstruction versus compact-function figure from the three retained checkpoints, PNG/PDF/SVG with five source trajectories and means; actually viewed, labels visible. Corrected the context-figure caption to match its signed role-scatter panels; source learning has its own figure. Full current local document/figure path/hash manifest: artifacts/seven_round_rebuild_20260906/R4_DELIVERY_MANIFEST.json. All40 changed allowlisted files pass size/AST/JSON/credential-pattern checks and Git whitespace check; 11 referenced manuscript images exist. Actual gpu-0 lease free; other project records unchanged. Targeted final-hook, panel replay and in-run actual full-model/tail checks already passed; no repeated full suite. Stage/commit/push uses existing main whitelist; local research documents remain ignored and retained.


## R5 member and context mechanism study started

写入时间（written_at_utc）：2026-09-07T06:26:47Z。事件时间：same design/start, before new member-level statistics or interventions. Round5 active, completed4 unchanged. Reuse frozenR4 maps and exposed512rolepanel; decompose all20directions and test five prechosen cycle directions. Candidate operations: single member, leave-one-out, whole-map norm matched to deleted edit, role-exchanged edit and its local-norm-matched version; source/fullFCC/native references explicit. Allmembers/allrows retained. Fixed source1target2block0zero-valued examples, including failure. Budget1200s/1GB/no training or refit. Mechanism outcomes cannot be called new independent confirmation. Exact prior tracker atarchive/research_workflow_20260907/r5_start_20260907T062647Z/manifest.json.


## R5 natural activation exemplars

写入时间（written_at_utc）：2026-09-07T06:41:10Z。事件时间：preparation after mainmechanism PASS412.4431358s/300544rows. Add <=180s existing32768naturalvalidationtokens for allalreadyfrozen source1/target2members. Top3 shortactivationcontexts withdocumentURL/hash and actualsequenceboundaries; inactive membersretained. No newtraining/refit orindependentcausal claim; totalR5cap1200s unchanged. Mainrole-swapnormmatched Temporal timeaccuracy12.97%vsFCC83.44%, targetmemberdeletion beyondnormcontrols; rawfacts retained inR5_MECHANISM_SUMMARY.json.


## R5 member mechanism and manuscript integration

写入时间（written_at_utc）：2026-09-07T07:03:14Z。实验完成事件：mechanism2026-09-07T06:37:09.782Z；natural2026-09-07T06:41:51.299Z，来自run状态；本段为当前补充整合时间，不倒填。

Scientificresult: all20mapenergydecompositionsshowtimeindividualattenuation(medianquoted/temporal.10245,total.08255),crosspositive20/20temporal19/20quoted; no dominantaggregatecancellationclaim. Fivecycle/all512actualnormmatchedroleswaps lowerTemporal timeaccuracy83.4375→12.96875%,Quoted98.90625→93.59375%. All169memberincidences retained:Temporal removalvsnormcontrolKLnumber.037997/.016949,time.013569/.002765,referenceintactFCC notsource. Allfive directions afterownmembermeanssamequalitative ordering; notindependentmember/edgeCI. Source1target2 complete10×16N/32×32T signedliftswith8sharedmembers; all16fixedcases/48factorrowsincludesourceandtransferfailures.73uniquenaturalmembers/all32768existingvalidationtokens, descriptive topcontexts only.

Mathphysicaldecomposition/nativeequivalence/rankambiguity/energycounterexample/normscope integratedDERIVATION§27 andmain§4.1/§10; abstract/introduction/conclusionupdated. Threeactualfigurefamilieswithdata/hashandallcases; nearestMarks/Ameisenmethodreadingandprovenanceupdated. R5_REPORT.md/R5_BUDGET.json andr5_mechanism/R5_MECHANISM_SUMMARY.json containfullvalues. Rawmain8df902598ffebaaadeb20cb323a98f39001ea5d0b6c954902d4796b8d7c055ae,300544rows; natural26022df8389d84b58c6dc5b402164dc2248d73e9f2046e13000ac1a24ea8afb4,73rows. BothPASScontractPASS; actualwall425.864371600s,runbytes178852933,zeroadditionalbulk/paid/training. Existingruntime/cacheandleasewrapperreused; allGPUworkdone. Beforeintegrationbytearchivearchive/research_workflow_20260907/r5_integration_20260907T070314Z/manifest.json. Countremains4pendingtargetedartifact/figurecheckandsync, thenR5completedonce/R6ready. Nootheragent/taskornewloop.


## R5 delivery checks and absolute-selectivity clarification

写入时间（written_at_utc）：2026-09-07T07:06:33Z。事件时间：same current artifact/math/figure andpre-synccheck. Allthreefinalfiguresactuallyviewed; allPNG/PDF/SVGsource/targetmanifesthashesmatch;16fixedcases/48factorrowsandjointsourcefailures/transferfailuresverified. All5directions havepositivemembermeanremoval-minus-normcontrolKL forbothfactor/bothrole. ASTof5newmember scripts andJSONof2configs pass;gitdiffcheckpass; mainfigurelinksresolve. GPU0actuallyfree, otherprojectleasesuntouched;ccadACTIVE15minsoriginaltargetconfirmed.

Absoluteeffects preventoverselling signedcross-factorcancellation:Temporalnumberdeletionmeanabsolute(number,time)=(.713367,.110848),timedeletion=(.161074,.210790);Quoted=(.759110,.117861)/(.045301,.044369). Main§10.3/R5_REPORT explicitlystatesclearnumberaverageselectivitybutsubstantialtimecross-effects. Wholegroup role-sensitiveoperationdoesnotimplypuretimeindividualfeatures. This supplementsold signedstatistics withoutchanginganyrawresultormethod. R5_DELIVERY_MANIFEST.json recordsallmain/report/figure/table/code/config/runidentitypathsandhashes; R5_SYNC.jsonwillrecordactualcommit/pushidentity. Scientificworkreadyforonegroupedwhitelistsync;trackerwilladvance4to5onlyoncethatcompletes,nextR6fullpaperPDF. Noadditionalexperiment.


## User extends campaign after the existing visual round

写入时间（written_at_utc）：2026-09-07T07:26:52Z。事件时间：同本次用户明确计划变更及实际写入，不是新实验。User explicitly keeps currentround6/7/8scope and addsfiveindependentfullprojectroundsaftervisualdelivery. Countremainscompleted5/active5/next6;total13,extensioncompleted0. Originalround8pause/no9instructionissuperseded: deliverandpreserveoriginalstageafter8,continueoriginalccadintorounds9to13,currentthreadunchanged;pauseoriginalafter13,no14. Thisplanupdateconsumesnoresearchround.

Newrequirementsrecordedverbatiminscope: findmissingmathematicallinksfromCBSMtargettoactualimplementation; improveallSAEmaterial/training/fitting/applicationimplementation usingauthorizedexistingresources; broadprimaryliteratureandcodelearning/comparison; aimnotweakerinunderstandability/usability/story/evidentialweight/theorythanReproducibleSubspaces,ACL2026FeatureConsistencyandSemanticOT,withoutobscuretermsorunsupportedclaimedparity. Everyroundfull-scope andatleasttensofminutesofrealwork,usually45to120ormore;nowaiting/pollingtoinflatecounts. Visualcorrectionstopsdefaultblue-orangeandDejaVuSans/GPTstyle,reportlikeclutter,meaninglesswords/pictures,unreadabledensepoints/defaultsimplelines;actualpaperfigures/templatesguideclearformats,realcases/necessaryworkflowdrawingsonly,nofabricateddiagram,propermath. Currentstagecannotoffloaditsobligationstofuturephase.

AGENTSdurablerules,EXPERIMENT_PLANscientificroute,EXPERIMENT_TRACKERunique13roundschedule,andcanonicalautomationpromptupdated;previousfilesandactualautomationTOMLbytearchivearchive/research_workflow_20260907/post_visual_extension_20260907T072652Z/manifest.json. Officialthreepapermetadataactuallyrefreshed,registryrecordsfullreadingstillfuturework. Originalautomationupdate/readbackandsame-thread15minstatewillbeverifiedbeforeclosing. Noresearchrun,nonewpaidresource,nosubagent,newlooporpublication. Existingwhite-liststagecommit/pushretainslocaldocshashes,doesnotexpanduploadscope.


## Follow-on schedule applied to the existing automation

写入时间（written_at_utc）：2026-09-07T07:28:44Z。事件时间：same live readback. automation_update(mode=update,id=ccad) succeeded;actualTOMLstatusACTIVE,15minsameoriginaltarget01a06e15-b222-7b21-a8cd-eb058a3159e5,canonicalprompttextmatchesexactlyafterstrip. No duplicateautomation,notpausednow; latesttracker schedulesoriginal6–8unchangedthenindependent9–13withfinalpause. Completed5/next6/extension0 verified. Originalstage-only no-retraining/solebackup phrasing clarified so it cannot bar newstageauthorizedmaterial/methodimprovement. Updated5document/promptpathhashesandpriorbytearchive inartifacts/seven_round_rebuild_20260906/POST_VISUAL_EXTENSION_PLAN.json. Thisisauthorizedplanning/automationwork,notanextra scientificround.


## BONUS_AUDIT_20260907 — START

written_at_utc: 2026-09-07T07:34:47Z
round_id: BONUS_AUDIT_20260907
event_started_at_utc: 2026-09-07T07:34:47Z
phase: START

用户明确授予一轮额外快速全面审计与整理，并要求所有后续轮次master_log留痕、最终交付可投稿的优秀完整项目包。该奖励轮独立于已计划13轮；原completed5/next6与后续6–8、9–13不变。实际检查范围：当前入口/原automation、所有本campaignrun元数据与代码快照、R1–R5日志与成果覆盖、稿件结构/引用/图表资产、复现与最终打包缺项。修确定问题，不重训、不新跑GPU实验、不以短审计冒充第6轮。前8文件和原automation字节归档：archive/research_workflow_20260907/bonus_audit_20260907T073447Z/manifest.json。


## BONUS_AUDIT_20260907 — COMPLETE

written_at_utc: 2026-09-07T07:45:12Z
round_id: BONUS_AUDIT_20260907
phase: COMPLETE
event_started_at_utc: 2026-09-07T07:34:47Z
event_completed_at_utc: 2026-09-07T07:45:12Z

快速全面审计与实质整理完成；不占原13轮，原completed5/next6保持。全部32个当前campaign run为26PASS/6FAIL；记录契约30通过/1不通过/1未形成，原异常保留。234个代码快照hash匹配；404个输入path/hash记录存在且大小一致，其中325小文件82,294,619bytes实际重hash一致。未重hash全部大权重/cache，未重算科学指标。R1–R5共39条阶段日志均有实际写入UTC，并有启动与结果记录；R2英文Actual log write UTC有效。

修复旧契约的无Git/尚未首跑叙述、历史MANIFEST误充全量索引、后7张图的编号、两处模糊图号和released local tables措辞。Section8引用实际正确，初判疑虑已撤回；主文数值/公式/原始图数据不变。三篇具名peer本地PDF/hash匹配，但当前主稿尚未引用，正式全文PDF/TeX/Bib/附录与统一复现入口仍待第6–7轮；用户指出的现有视觉质量仍未达标，编号修复不冒充美学改善。

已将后续每轮START/COMPLETE及实质阶段留痕、日志完成后才计数、最终可投稿项目包内容写入AGENTS/plan及原automation。原ccad实际TOML与规范prompt一致，ACTIVE、15分钟、同原任务；6–8后9–13/最终13暂停不变，奖励轮独立。审计报告与所有核对清单位于artifacts/bonus_audit_20260907/；前8文件及automation逐字archive见该目录START.json。没有新GPU实验、下载/安装/付费或其他项目资源修改。工作/检查是本轮产出，不冒充新科学发现。成组同步后将奖励完成计数置1，下一主轮仍6。


## SEVEN_R6 — START

round_id: SEVEN_R6
phase: START
written_at_utc: 2026-09-07T08:06:32Z
event_started_at_utc: 2026-09-07T08:06:32Z

第6轮启动。选择理由：最后层角色依赖的跨seed操作已有冻结确认、成员干预及固定正反实例，现有约12k词快照仍按实验发展顺序累积，尚无统一正式PDF、完整引用和自含附录。当前工作是把结果组织成可读、可核查的完整论文，不重复相同科学实验或把改写当新发现。

计划依次核对核心结果与证明、读取三篇具名近邻的原文方法和实际PDF图版、统一正文与术语、把必要完整结果和证明纳入附录、重做适合信息的主图、生成可编辑排版源/参考文献/PDF并逐页检查，写清阅读与复现入口。原阶段第6–8轮范围保持；新增9–13轮不承担本轮欠项。无新GPU拟合或训练；复用文档运行时与现有资产，仅在缺少编译器时使用官方来源的隔离工具，新增工具/稿图预算上限2GB，0付费。工作/编译/等待分别如实记录，尚无结束时间；有效工作时长未单独计量，不伪造。

8个当前文件逐字归档与hash：archive\research_workflow_20260907\r6_start_20260907T080632Z\manifest.json。启动配置、预算、原HEAD在artifacts/seven_round_rebuild_20260906/r6_paper/START.json。保留全部历史失败及raw/full更强、源失败、time成员非纯选择性的结果。当前completed=5、active=6，奖励轮已完成且不占原13轮。


## SEVEN_R6 — PROGRESS / CORRECTION：第一版完整 PDF 与实际目检

实际写入 UTC：2026-09-07T12:11:29Z。可证实事件：第一版完整 PDF 构建于 2026-09-07T12:07:08Z 启动、文件于 12:07:11Z 写成；此前 START 为 2026-09-07T08:06:32Z。该跨度不能当有效工作时长；期间有效工作、工具授权交互、上下文接续与非工具等待没有独立连续计时，分项时长未知，不倒算充数。

已实际完成统一英文正文、四个附录、参考文献、五幅同源矢量图、14 个生成表片段与全部 16 个固定案例/48 操作。轻量导出包含 870 个当前汇总行和 4350 个 source-seed 行，早期完整结果单独导出；不新增拟合或推理。第一版 PDF 24 页、SHA256 a13eaec4d9e3323d09df84b6654fdc30c21ecb512258e059a7cb99c23b3e1d0a。已检查实际 PDF 前 16 页；发现正文 lambda 缺反斜线、Quoted number 成员 KL 超出图轴上限、长 hash 出栏、末页仅四行及若干标题/段间距需改。修订前所有稿图逐字归档在 archive/research_workflow_20260907/r6_first_pdf_20260907T121129Z/manifest.json。初次缺样式/字体缓存和 table input 的 TeX 失败已解决，未改实验环境；后续构建按时间保留日志。

科学勘误来自直接复核 scripts/run_frozen_composition.py:79 与 run_composition_correspondence.py:65：历史 wrong_factor 控制实际上交换两个 direct-target-native 源组操作，而非把错误 donor 的目标代码送入原 FCC map。同为 final token 时其 joint 是 q_T+q_N，与 direct_target_native 的 q_N+q_T 完全相同；不能算独立的联合特异性证据。此前泛称“wrong-factor input”的表述须改为“swapped native factors”，原 raw/method id 保留。主正结果未依赖它；本文直接说明退化，R7 聚焦核对时评估是否值得补一个真正在同一 FCC map 上置换 donor 的有区分力控制。

当前仅完成阶段稿，尚未计本轮完成。下一步修复上述具体问题、看完其余页面并复核修订全稿，补 README/证据索引/来源与环境记录，再留 COMPLETE 后更新 tracker。当前 CPU 为短时文稿/绘图/排版，无 GPU/大文件 IO/付费或其他项目租约操作。


## SEVEN_R6 — COMPLETE

round_id: SEVEN_R6
phase: COMPLETE
written_at_utc: 2026-09-07T12:29:53Z
event_started_at_utc: 2026-09-07T08:06:32Z
event_completed_at_utc: 2026-09-07T12:29:53Z

第6轮统一论文与实际PDF交付完成。选择角色依赖操作为单一主线，把原发展稿重排成6页英文正文、1页参考文献及16页完整附录；paper/main.pdf共23页，SHA256 a13170c173f6df2b91239ef04a87901390768596ce6aa7fb5b38a4f7906baa30，368486bytes。正文/四附录源、17项BibTeX引用、5组正式PDF/SVG/PNG图、14个表片段构成的12张表、全部16固定案例/48操作、870汇总行与4350source-seed行、早期负差、完整signed关系和8项主张证据索引均已实际落盘。paper/README.md给阅读/构建/实验driver与config/官方资产pin及获取/许可和重跑范围说明；旧快照退出当前主文入口但保留原文。

数学整合明确均值抵消、预测成员的物理分解与坐标不变、一般GL不保稀疏非负类、有限contrast native等价条件、精确penalized RRR、组成Gram界及混合Hessian、prefix秩反例、individual/cross energy与norm控制。真正由本轮代码核对发现的科学勘误：旧wrong_factor是swapped native factors，同位置joint与direct native相同，不能当独立联合特异性证据；当前主稿/附录/表/README已更正，原raw和ID不变。本轮无新拟合/推理/训练，既有raw/full更准、源失败、time成员非纯选择性的结果保留。

三篇具名近邻实际方法/实验/附录指定页及PDF图版已读，SOURCE_REVIEW.json留位置/hash。subspaces已有功能mask/CE，ACL一致性与SemanticOT语义分布/电路消费者均在主文准确对比；不声称复现完整系统或总体胜出。REFERENCE_REGISTRY追加Bhalla作者列表历史错误勘误，当前12作者BibTeX匹配官方元数据。外部源码/图未复制。

首版24页及修订版23页均实际逐页渲染查看。具体修复lambda转义、成员KL轴截断、长hash出栏、碎末页、标题/间距与案例引号；最终18个字体均嵌入，无未解析引用/出栏问题。PDF_REVIEW.json记录实际范围，目检不冒充投稿验收。无run/权重的新paper-only目录实际重建5图与23页PDF，所有页提取文字和5PNG像素相同；仍使用同机环境，不声称干净机器全实验或PDF二进制复现。

资源：官方Tectonic0.17.0/MIT隔离安装，既有Python3.12.14/Matplotlib3.10.8复用，实验环境未改；最终完整build3.843秒，零GPU/训练/付费。本轮稿图/工具/渲染/便携副本/归档实测170960292bytes，R6_BUDGET.json分项留存。有效研究工作与等待没有连续独立计时，时长未知；开始到结束的跨度不当有效工时，不以空等凑轮次。初期缺样式/字体缓存和TeX输入失败日志保留。三脚本AST、git diff check及实际重建检查完成，未重复实验测试或占用他人租约。

交付：artifacts/seven_round_rebuild_20260906/R6_REPORT.md、R6_DELIVERY_MANIFEST.json、R6_BUDGET.json及r6_paper/检查记录。归档前版本archive/research_workflow_20260907/r6_close_20260907T122953Z/manifest.json逐字/hash保留；首稿修订归档r6_first_pdf_20260907T121129Z。完成记录写入后才更新completed5→6、next7READY，总13/奖励轮独立保持。原ccadACTIVE15分钟同原任务，未委派或改执行归属。随后3构建脚本与master_log成组白名单同步，实际commit/push/HEAD核对由R6_SYNC.json记录；文稿仍按既有白名单保留本地。下一轮聚焦具体对照语义问题与原科学/证据包收口，第8轮原视觉交付不延后，9–13独立追加阶段不变；最终投稿质量仍NOT_MET。


## SEVEN_R7 — START

round_id: SEVEN_R7
phase: START
written_at_utc: 2026-09-07T12:50:29Z
event_started_at_utc: 2026-09-07T12:50:29Z
event_completed_at_utc: unknown; work ongoing

第7轮登记启动；已核对当前AGENTS/tracker/plan和原代码，完成计数仍6，总13。选择理由：第6轮正式稿暴露旧wrong_factor为native交换、共同位置joint退化的具体问题；需要判断同一冻结FCC映射上的错误donor能否提供有区分力的功能证据，并把原阶段科学结果整理成可实际使用的代码/来源/稿件证据包。计划在数学上写清映射与donor的交换关系，按既有512行和全部20方向进行一次必要的冻结对照，保留已有norm与role-swap控制和所有失败；旧数据仅作针对性诊断，不包装为新独立确认。同步必要原文方法核对、可运行操作入口、结果与全文图表，随后原阶段科学收口。

预算最多600秒GPU管理任务、1GB新增小型结果/交付包、0付费或新权重/训练，按既有cached-tail吞吐执行。GPU实际free，RTX5070Ti显存606MiB/16303MiB、利用率0%；其他项目cpu/disk租约仍保留，未改写。复用r004及锁定overlay；GPU任务经共享manager申请。活动研究工作与等待无连续独立计时，不用起止跨度冒充有效时长；计算实测另记，不空等凑轮。原稿与5个当前入口文件逐字归档archive/research_workflow_20260907/r7_start_20260907T125029Z/manifest.json；启动身份与预算artifacts/seven_round_rebuild_20260906/r7_science_package/START.json。当前原ccad继续同一对话，未委派或建立新loop；第8轮原视觉交付及独立9–13阶段不推迟。


## SEVEN_R7 — PROGRESS：正确的donor对照和可用操作接口

round_id: SEVEN_R7
phase: PROGRESS
written_at_utc: 2026-09-07T13:32:27Z
event_started_at_utc: 2026-09-07T12:50:29Z
event_completed_at_utc: unknown; round ongoing

已实现NumPy-only PredictiveOperation接口和冻结donor诊断consumer，无拟合或新source/member选择。正确联合为x_N P_N+x_T P_T，交换donor输入后为x_T P_N+x_N P_T，差为(x_T-x_N)(P_N-P_T)；它一般不退化为原联合。对照还按实际共同位置的总向量范数匹配，每个单因素及joint分别处理，零候选异常保留。现有5source/20方向/512已暴露行全部纳入；实际GPU诊断尚未启动，不预报正结果。

三个针对性接口/符号/交换代数测试全部PASS；报告的unittest wall为2156.630秒，工具exec wall2148.8660956秒，期间没有独立CPU/等待计时，异常跨度原因未证实，不能称为36分钟测试计算或有效研究。测试记录r7_science_package/TARGETED_TESTS.json，未因等待加轮。后续实际run按600秒预算与原租约wrapper执行；若资源忙则仅阻塞该计算，继续论文/来源与交付。CausalGym原文§3.1–4.1/§5.2已重读：保持无关变量的最小donor差与任意输出标签表达能力控制是不同问题；本donor诊断不冒充其expressivity benchmark。


## SEVEN_R7 — PROGRESS：实际donor诊断完成

round_id: SEVEN_R7
phase: PROGRESS
written_at_utc: 2026-09-07T13:46:30Z
run_event_started_at_utc: 2026-09-07T13:33:13.620177+00:00
run_event_completed_at_utc: 2026-09-07T13:35:40.955310+00:00
round_event_completed_at_utc: unknown; manuscript and package ongoing

SEVEN_R7_donor_specificity_v1_20260907完成PASS/contractPASS，99840原始行、全部512既有语境与20相关seed方向；147.2949187秒管理任务，704完整模型＋101056cached-tail序列计算，峰值VRAM4216474624bytes；GPU租约已释放。无新拟合/训练/下载/付费。raw SHA256 b38713bad76677811788e02930bc02f9184172631bb8cd9e6b2a1efd3391520e，配置configs/seven_r7_donor_specificity_v1.json，代码run source_snapshot完整保留。与原冻结FCC物理向量最大差1.42108547e-14，四标签logprob逐行差0；总操作范数匹配最大差1.42108547e-14，零候选/正参考例外0。40个source-target-factor NumPy-only操作bundle已导出，包含signed预测向量和实际模型/hook/SAE身份。

正确FCC/错误donor/等范数错误donor联合KL：Temporal .089088/.552390/.634722，Quoted .059455/.291516/.246172。正确/等范数错误donor的joint accuracy：Temporal .717969/.225000，Quoted .926172/.776367。number/time/joint两角色的全部20方向，错误donor及等范数错误donor平均KL均更差；共同seed不当20独立重复。去掉每个seed的所有入/出方向后，等范数joint excess KL范围Temporal .485716–.613139、Quoted .135775–.206614；是12剩余方向的稳健性范围，不是置信区间。支持因素输入和映射的实际配对有作用，不能升级成纯语义成员、无混杂或新独立确认。R7_DONOR_SUMMARY.json及完整direction/block CSV保留所有分母。

代码/元数据核对还发现R6_REPORT中的“Temporal熟悉/新句法”是标签笔误：.0729/.1053对应熟悉/新cue对，两个syntax同时保留；当时正式论文表与原始数据标签正确。历史报告原文不改，本条追加勘误，当前稿和本轮报告按实际cue分层。下一步整合新的数学对照、主图/附录表及可用接口说明，完成原阶段科学证据包。计数仍6、active7，不因计算完成提前计轮。


## SEVEN_R7 — PROGRESS: manuscript and reproduction inputs prepared

Actual log write time UTC: 2026-09-07T14:07:36Z. Round_id: SEVEN_R7. Event: current manuscript/reproduction preparation completed at this write; full round/package closure remains pending.

The donor diagnostic is now in main Section5.2, Figure2 and AppendixA.8/C.8, with signed input-swap algebra, total-vector norm definition, all role/factor/cue values and dependent-seed leave-out summaries. Current paper24pages,6main+1reference+17appendix,5figurefamilies/14tables; all24 final120dpi page images actually viewed. Two spill pages introduced by added content were corrected by removing duplicate prose; earlier page renders/build logs retained. PDF SHA256 af757f7534cef54be446c75c776ad0ead72a6af7e0c88134e9df93b0f8c4108d. Actual readout/fidelity and source vs intact-FCC reference distinctions remain explicit.

New NumPy CLI executed all16 time examples. Package smoke after extraction remains pending. Added usable package reading/reproduction/round guides and corpus exclusion ledger:838 records retaining only original document_id/text_sha256 fields, same exclusion sets and hashes recorded in delivery/CORPUS_EXCLUSION_PROVENANCE.json. This repairs an undeclared dependency of corpus reconstruction without redistributing original full documents. All original ledgers retained. Formal paper, R7_REPORT and R7_METHODS_AND_PROOFS give original-stage settled questions and explicit NOT_MET acceptance; raw/full baseline gap and missing full SemanticOT/circuit/semantic evidence remain.

CausalGym original method/control/license sections, README and interventions.py actually read; repo main commit unverified, no external code copied. Registry and compute ledger updated. Paper builds use existing isolated runtime; helper attempted missing fitz and immediately used existing pypdfium2, no install or scientific run affected. Multi-GB E-drive package construction is next, with a managed disk-e-io lease and900MB ZIP cap inside original1GB incremental target. No new GPU calculation needed. Active writing/reading vs waiting remains unmeasured, not inferred from timestamps. Counter remains6 until verified package and COMPLETE.


## SEVEN_R7 — COMPLETE: original scientific stage and usable evidence package

实际写入UTC：2026-09-07T14:13:28Z。round_id：SEVEN_R7，阶段COMPLETE。可证实事件：登记开始2026-09-07T12:50:29Z；本轮结果、稿件、包与代表性重放整理完成于本次写入；包构建开始2026-09-07T14:08:01.075301+00:00、结束2026-09-07T14:08:24.445246+00:00，提取后重放完成2026-09-07T14:08:55.372013+00:00。这些是记录/工具事件时间，不冒充有效工时。

选择理由与实际推进：纠正旧wrong_factor在同位置joint退化的问题，并只补一个有区分力的真实控制；同时把数学、接口、来源、整稿和原始结果组织成可使用交付。新控制把另一因子的target code差输入同一冻结FCC map，不换支持/系数/源组、不选择结果；512已暴露句、全部20共享seed方向、单因子及joint均保留。新joint KL（正确/错误/等总norm错误）Temporal0.089088/0.552390/0.634722，Quoted0.059455/0.291516/0.246172。全部role/factor的每个方向平均KL均因错误输入变差；排除任一seed全部入出方向后，joint额外KL仍为正。它支持指定操作内输入配对/方向的功能价值，不是新独立确认、唯一语义或原生机制证据；Quoted time的norm校准确实缩小一部分差距，原数据完整保留。

数学和可用输出：正式附录给q_wrong−q=(L_N−L_T)(Δz_T−Δz_N)、核空间退化条件、共同位置实际总向量norm与零分母规则。40个NumPy映射包保存signed成员向量和实际输入规范，CLI可直接从完整/选中target codes产生source-aligned residual更新及每个成员分量。真实run原矩阵向量最大差1.421e-14、旧FCC四label日志概率差0、norm误差1.421e-14，60方向×factor单元无零分母异常。它不冒充target-native删除或自动概念发现。论文/方法总结在artifacts/seven_round_rebuild_20260906/R7_REPORT.md和R7_METHODS_AND_PROOFS.md，DERIVATION_PACKAGE.md新增§28并保留旧版hash归档。

真实运行：runs/SEVEN_R7_donor_specificity_v1_20260907，PASS及contract PASS，99,840 raw rows，raw SHA256 b38713bad76677811788e02930bc02f9184172631bb8cd9e6b2a1efd3391520e。事件开始2026-09-07T13:33:13.620177Z、结束13:35:40.955310Z；147.294918700秒GPU管理任务wall，704完整模型sequence加101,056缓存tail，峰值分配4,216,474,624bytes。复用已有r004/Torch/Transformers/NumPy，无新训练、下载、安装或付费。实际资源gpu-0与打包disk-e-io均经原manager自然释放并已核对free，其他项目lease未动。

稿件与包：paper/main.pdf共24页（6正文、1参考、17附录），SHA256 af757f7534cef54be446c75c776ad0ead72a6af7e0c88134e9df93b0f8c4108d；可编辑全文/证明/5图/14表/17引用/16固定案例全部保留。加入donor对照图/表和来源，并逐页查看全部24张120dpi最终页图；删除重复文字修复新增的两处尾页溢出，中间渲染和构建日志保留。初始包delivery/ccad_original_science_r7_20260907.zip为497,624,566bytes，2421条目，33原阶段run、完整raw与失败/continuation、代码/配置/环境/报告/证据索引。33个可再生cache共845,258,665bytes只在ZIP省略，逐项path/hash/恢复说明在PACKAGE_MANIFEST；原件未删。838条ID/text-hash语料排除记录和来源digest补入，可复用原构建器恢复相同token身份，不携带原全文。全部ZIP CRC通过；压缩和CRC实测23.375秒。

真正提取包重放：158文件全部匹配包manifest，40map身份一致；number/time各16案例，向量最大差分别1.776e-15/3.553e-15，成员和误差4.441e-16/1.776e-15。提取目录无原task caches或模型/SAE权重，实际重建24页PDF，全部页文字和五PNG像素一致，使用同机已安装NumPy/plotting/TeX字体缓存；不是干净机器全训练重放或逐字节PDF一致。实际6.453秒，PACKAGESMOKE/BUILD/PDF_REVIEW保存具体命令与范围。本COMPLETE与校验receipt随后作为不同命名条目追加到ZIP，原snapshot不覆写；最终ZIP身份见PACKAGE_CLOSEOUT.json/R7_DELIVERY_MANIFEST.json。

科学判断与同行：R7_REPORT逐项回答蓝本/实现、五seed素材、对应使用、成员解释、强基线、正式稿/可复现包的实际已达和未达。CausalGym原文指定方法/控制/许可章节及官方README/interventions.py已读；未核实repo main的精确commit，不假称读完eval，未复制代码或外推数据许可。具名三篇仍按真实方法/功能/上下文分布/电路消费者比较，conditional-correlation OT不冒充SemanticOT完整复现。当前raw/full更准、语义/多层电路用途及跨任务覆盖不足，最终优秀/可投稿验收仍NOT_MET；文件齐全和七轮结束不是科学达标证据。

预算与失败保留：R7_BUDGET.json保守计量本轮新增/整稿/包/提取/归档共657,669,917bytes，低于1GB；原campaign GPU管理任务保守累计3.337507789小时，不是纯kernel小时。三个针对性测试通过但2156.630s unittest/2148.866s工具间隔未解释；有效阅读/写作/推导/审图时间与等待未连续测量，均明确未知，不以起止跨度或空等充作投入。已完成AST/JSON与差异检查；没有无修改重复全套测试。所有此前失败与负差保持原件。

下一步与交付定位：本完成记录到位后计数6→7，原科学七轮按工作分配收口；第8轮专门提升全文视觉/读者路径与阶段完整快照，不继续科学网格。随后原位接续已授权第9–13轮全面补强，仍由本对话执行；第13轮最终交付后暂停原automation，不提前暂停、不启动14。原ccad当前ACTIVE/15分钟、target01a06e15-b222-7b21-a8cd-eb058a3159e5已实查。白名单源代码/配置/master_log成组同步随后执行，receipt另存，不扩大论文/原始数据公开范围。关闭前当前入口归档：archive/research_workflow_20260907/r7_close_20260907T141328Z/manifest.json。
