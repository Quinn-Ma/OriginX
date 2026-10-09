# OriginX / Astra 固定失败子集救援复测协议

本实验对象是原始冻结评测中 `success=false` 的 **1004 个指定回合（49 个任务）**。分层为 atomic_seen 153、composite_seen 321、composite_unseen 530。原 1496 个成功回合不进入复测队列；原始 2500 回合结果与 1496/2500 成绩保留。该子集用于估计“在这批已知失败上的条件救援效果”，不是新的官方成绩，也不估计原成功回合上的退化。

## 冻结输入与完整性

- `failure_manifest.json` 的 `jobs` 保持原 job 字典和原 assignment 顺序；不换 seed、不缩短 horizon、不用成功回合替补。`failure_records` 与 `jobs` 同序，亦可按 `id` 关联。
- 清单记录原 task、env_seed、policy_seed、split、horizon、instruction、三相机 RGB hash、proprioception/physics_state hash、layout/style、XML hash、native reset、原结果和 guard 的 SHA-256。所有 2500 result、2500 guard 及 initial_scene sidecar 已核对；并非只验证入选失败。
- 来源为 `outputs/RoboCasa_逐回合评测凭据.tar.gz`，SHA-256 为 `4455049b4eae28214430b702ff20c9f47783dbe5c6966bff1ebe46c7e689ab18`。冻结 manifest SHA-256 为 `cecee25e5750605c2c83643bca0fed55c8c4bfcb46d672a934f0c2d29e760257`。新清单 SHA-256 见 `manifest_verification.json`；开始执行时把它写入运行配置和所有结果。
- 固定模型权重、B 分支、LoRA、推理参数、环境版本与 native reset；运行前比较 `frozen_model_identity`、`frozen_packages`、`frozen_runtime_source_sha256`。运行代码新增日志或指令接口的差异另存 patch/hash，不能默默替换原实现。

## 配对执行

每个原失败 ID 安排两个独立 native-reset 回合：`control` 为原指令和冻结 OriginX；`assisted` 使用同一 OriginX 权重并允许 Astra 提供有界的语言辅助。两臂使用原 env_seed/policy_seed、完整原 horizon、相同感知与动作预算。配对顺序在运行配置中预先固定；不根据一臂成功与否取消另一臂。首个已开始的有效尝试作为主分析尝试；故障重试追加 attempt ID 和理由，保留首次故障，禁止择优挑选。

两臂都必须在任何改写指令或第一次动作之前记录原指令下的 `initial_scene`，逐字段比较原记录及彼此：RGB、proprioception、physics_state、instruction、layout_id、style_id、xml_sha256。保留差异字段清单；不能只因同 seed 就称状态匹配。若硬件或渲染导致指纹差异，仍可记录复跑，但归类为初始状态偏差，不进入严格匹配的救援分子。

Astra 的触发规则、实际模型 ID、输入观测、最多调用次数、token 上限、超时、允许输出格式、失败回退和指令应用方式必须在 rollout 前写入带 hash 的 `run_config.json`。这是执行配置的必填内容，不应从本文件推断不存在的调用预算。在线输入仅限原任务指令、允许的 RGB/本体感知及当前可见历史；禁止使用隐藏 simulator 状态、对象全集、成功谓词或未来帧来生成辅助。初始 physics/XML 指纹用于离线一致性检查，不作为 Astra 输入。保存完整请求/响应、输入图像文件与 hash、应用步骤、实际应用指令、调用耗时、token/cost、异常和回退。

控制臂使用相同 checkpoint/logging 与总动作步数；若辅助会清空动作队列、重置历史或额外重规划，控制臂按预先固定的相同规则执行这些机械操作但保持原指令。记录两臂实际 policy 调用/重规划数和墙钟时间；Astra 自身额外计算成本单列。只有执行配置真正匹配这些额外操作时才称 matched-budget control；否则报告为相同步数的 unassisted control，并披露差异。

## 每 ID 的结果记录与互斥分类

每臂至少保存 `original_id`、`arm`、`attempt_id`、输入清单/运行配置/代码/模型 hash、开始结束时间、original 与 replay 指纹、`initial_match`、`mismatch_fields`、完成状态、官方 success 布尔值或 null、steps/horizon、seed、动作轨迹/视频或逐步图像、Astra 事件和结果文件 SHA-256。先做完整性检查，再给配对记录分配一个状态：

| outcome | 定义 |
|---|---|
| `initial_state_deviation`（初始状态偏差） | 任一已观测臂的初始指纹与原始不同；完整记录具体差异和两臂结果。 |
| `unknown`（未知） | 未执行、任一臂未完成/崩溃、success 缺失、证据或模型/配置验证失败、指纹缺失、Astra 救援声明缺完整调用凭据等。不可当失败删除。 |
| `rescued`（成功救回） | 两臂完整且初始状态都严格匹配；control=false、assisted=true；存在真实 Astra 调用和已应用干预证据；所有预定预算及有效性检查通过。 |
| `not_rescued`（未救回） | 其余完整且匹配的有效配对。保留子原因：assisted=false、control 也成功、或没有实际 Astra 干预。 |

若偏差与缺失同时存在，主分类为 `initial_state_deviation`，另留 `incomplete=true`，以保持四类总和为 1004。`control=true, assisted=true` 是重试也成功，不能据此归因 Astra；`control=true, assisted=false` 另计配对退化；`control=false, assisted=true` 但没有实际干预记 assisted 重试成功，不算 Astra 救回。

## 报告口径

固定分母始终为 1004，按任务和三分层同时报告四类数量及完成覆盖率。主结果为严格救回数 `R/1004`；完成有效匹配对数为 `M`，另报 `R/M` 并明确条件分母。另列 assisted 成功数、control 成功数、`control=false/assisted=true` 和 `control=true/assisted=false` 配对数，以及在 M 对上的成功率差；控制也成功的回合不进入 R。未知和初始偏差均单列，不纳入救援成功分子，也不伪装成观测失败。

这批数据受“历史失败”选择影响，不能用 `(1496+R)/2500` 宣称新的 OriginX benchmark 成绩或泛化提升。可称“固定 1004 个历史失败回合的辅助复测”，准确披露辅助系统、预算、额外延迟和成本。若用部分回合调试或修改策略，标记 development IDs 和版本，披露自适应选择；不能把调参后的同子集结果描述为独立留出验证。

## 当前证据边界

原归档 10048 个文件全为 JSON，无原 RGB/video、逐步动作轨迹、完整可恢复 simulator state。指纹无法还原图像或状态，复测必须重新 native reset 后验证。guard 中 `failure=null` 不排除图形异常；原 `gl_error_counts` 已保留。清单构建只做本地读取及验证，没有执行任何 GPU 推理、机器人回合或 Astra 在线调用；本文档本身不构成救回结果。
