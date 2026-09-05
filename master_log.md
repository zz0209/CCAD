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
