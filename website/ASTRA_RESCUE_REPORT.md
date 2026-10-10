# OriginX: Astra-assisted replay of 1,004 B2000 failures

Qinzhen Ma · October 9, 2026 · Author-run technical report

**Astra-assisted replay of the 1,004 B2000 failures.**

After the original evaluation, we tested a frozen language-assistance wrapper on exactly its 1,004 B2000 failures, spanning 49 tasks. Each unique case has an unassisted control and an Astra-assisted arm; 2,008 is the number of planned arm attempts, not the number of distinct failure cases. All attempts are terminal. The original 1,496 successes were not retested, and the original 1,496/2,500 (59.84%) result is unchanged.

At the first regular 16-step policy query at or after half of the original horizon, the assisted arm sends the current three RGB views, original instruction, and remaining step budget to `gpt-6-astra` with `high` reasoning effort. One bounded subgoal response is appended as `Immediate next actions: <subgoal>`, followed by `Then complete the original task.` The frozen B2000 policy continues generating actions. We do not train, reset the environment or observation history, change the original action horizon, or provide hidden simulator state, success predicates, or future frames to Astra. Astra introduces extra inference and waiting time; the control matches action steps, not total compute or wall time.

| Final classification | Unique B2000 failure cases |
|---|---:|
| Strictly rescued | **56** |
| Not rescued | 813 |
| Unknown | 28 |
| Initial-state deviation | 107 |
| Fixed total | **1,004** |

A strict rescue requires a failed control and a successful assisted arm, exact initial fingerprints against the historical case in both arms, an exact pre-intervention action prefix, and a linked actual Astra call whose response was applied. The confirmed rescue fraction is **56/1,004 = 5.58%**, a lower-bound count over the fixed cohort with unresolved cases retained. There are **M=874** matched completed replay pairs, of which **E=869** also have verified applied Astra evidence: R/M=6.41% and R/E=6.44%. The five M-minus-E pairs remain unknown. Among M pairs there are 56 assisted-only successes, zero control-only successes, zero both-success pairs, and 818 both-failure pairs. This failure-selected study does not estimate regressions on the original successes or overall benchmark improvement.

| Original failure stratum | Cases | Rescued | Not rescued | Unknown | Initial deviation | Confirmed R / fixed cases |
|---|---:|---:|---:|---:|---:|---:|
| Atomic-Seen | 153 | 12 | 125 | 0 | 16 | 7.84% |
| Composite-Seen | 321 | 25 | 218 | 8 | 70 | 7.79% |
| Composite-Unseen | 530 | 19 | 470 | 20 | 21 | 3.58% |

A recorded example is `native-B-0525-PrepareCoffee-2090900525`: at step 912 of the 1,800-step horizon, the subgoal advised aligning the mug opening under the dispenser and operating the visible front control. The advice is linked to the applied instruction at that policy query. Both historical initial fingerprints and the pre-intervention prefix match; the assisted arm succeeds and the control fails. This observation does not by itself establish which suggested physical action caused success.

The initial v1 partition permanently retains its 36 first-attempt cases (0 rescued, 3 not rescued, 28 unknown, 5 initial deviations). The continuation evaluates the other 968 cases once (56, 810, 0, 102). It only changes infrastructure: same-socket read-only RNG keepalives address an idle timeout, and validated initial-state deviations are recorded without stopping subsequent dispatch. The continuation passed 72-client parity and a 360-second same-socket keepalive probe before replay. There was no resampling of initial states and no outcome-selected retry. Four infrastructure pilot cases, assignment indices 48/97/249/345, had 0/4 rescues and remain disclosed development IDs; their separate pilot outcomes are not added to the primary count.

The main study used **135 unique CLI calls** (134 successful, one quota error), deduplicated by batch and receipt rather than multiplied by the number of cases in a batch. Known counters sum to 2,592,713 input tokens and 195,237 output tokens, including a reported 75,511 reasoning tokens; the error call has no counters, so complete token totals remain unknown. Summed CLI latency is 6,186.44 seconds, median 46.16 seconds per call, range 2.33–77.35 seconds. These are CLI service times, not total replay time or per-case end-to-end waiting. Dollar cost is unavailable. The separate pilot used one call, 19,071 input tokens, 629 output tokens, 137 reasoning tokens, and 24.35 seconds. Complete records are in [the rescue report](https://github.com/Quinn-Ma/OriginX/blob/main/ASTRA_RESCUE_REPORT.md), [case classifications](https://github.com/Quinn-Ma/OriginX/blob/main/evidence/astra-rescue/case-classification.json), and [deduplicated usage](https://github.com/Quinn-Ma/OriginX/blob/main/evidence/astra-rescue/cli-usage.json).

The earlier [three-case offline analysis](https://github.com/Quinn-Ma/OriginX/blob/main/ASTRA_FAILURE_ANALYSIS.md) remains a historical diagnostic artifact. Its hypotheses are not explanations established by this new experiment. The online replay evidence supports 56 recoveries under the stated protocol; it does not create a new 2,500-episode score or an accepted leaderboard position.

## Evidence and reproduction

The original 2,500-episode benchmark release is unchanged. The rescue publication preserves both campaign archives, the separate pilot archive, fixed selection and failure manifests, per-case classifications, model/configuration/source hashes, actual request/response evidence, and deduplicated CLI usage. See [the evidence index](https://github.com/Quinn-Ma/OriginX/blob/main/evidence/astra-rescue/README.md) and [execution source snapshot](https://github.com/Quinn-Ma/OriginX/blob/main/rescue_study/README.md). Unknown results and infrastructure errors are retained. All v1 and continuation workers and model services were independently checked inactive by their recorded process identities.

The [research preprint](https://github.com/Quinn-Ma/OriginX/blob/main/paper/OriginX_arXiv_manuscript.pdf) is a research draft; publication here is not a conference acceptance or an organizer-accepted leaderboard rank.

## 中文结果与方法

**Astra 实际介入：复测 B2000 的 1,004 个失败案例**

原评测结束后，我们只对 B2000 的 1,004 个历史失败案例做救援，覆盖 49 个任务。每个案例各有未干预 control 和 Astra 辅助两臂，因此 2,008 是回合尝试数，案例数始终为 1,004。全部尝试已经结束；原 1,496 个成功案例没有重测，原成绩仍为 1496/2500（59.84%）。

在原 horizon 一半之后的第一个常规 16 步策略查询点，辅助臂把当时的三个 RGB 视角、原任务指令和剩余步数交给 `gpt-6-astra high`。Astra 返回一次有界子目标，包装器追加 `Immediate next actions: <subgoal>` 与 `Then complete the original task.`，然后由冻结的 B2000 继续生成动作。没有训练、环境重置、观测历史清空或动作时限延长；隐藏物理状态、成功谓词、未来图像均不作为 Astra 输入。控制臂匹配动作步数；Astra 的额外计算与等待时间单独报告。

| 最终分类 | B2000 唯一失败案例 |
|---|---:|
| 严格确认救回 | **56** |
| 未救回 | 813 |
| 未知 | 28 |
| 初始状态偏差 | 107 |
| 固定总数 | **1,004** |

严格救回必须同时满足 control 失败、辅助臂成功、两臂初态与历史指纹完全匹配、干预前动作轨迹前缀完全匹配，以及真实 Astra 调用与已应用响应的关联证据。固定分母确认救回比例为 **56/1004=5.58%**；未知和初态偏差仍留在分母，因此这是确认救回的下界统计。匹配且完成的配对 **M=874**，其中实际应用 Astra 且凭证有效的配对 **E=869**，条件比例分别为 R/M=6.41%、R/E=6.44%。M 比 E 多出的 5 对仍为未知。M 中辅助独赢 56、control 独赢 0、双成功 0、双失败 818。由于只选择历史失败，本实验无法衡量原成功案例上的退化，也不估计整体基准提升。

| 原失败分组 | 固定案例 | 救回 | 未救回 | 未知 | 初态偏差 | 救回 / 固定案例 |
|---|---:|---:|---:|---:|---:|---:|
| Atomic-Seen | 153 | 12 | 125 | 0 | 16 | 7.84% |
| Composite-Seen | 321 | 25 | 218 | 8 | 70 | 7.79% |
| Composite-Unseen | 530 | 19 | 470 | 20 | 21 | 3.58% |

一个可核验实例是 `native-B-0525-PrepareCoffee-2090900525`：在 1,800 步时限的第 912 步，Astra 给出将杯口对准出水口、查看并操作前部控制的子目标；实际策略查询记录与已应用指令匹配。两臂历史初态和干预前前缀一致，辅助臂成功而对照失败。这不能单独证明究竟是哪一个动作造成成功。

v1 已开始的 36 个案例永久保留首次结果（救回 0、未救回 3、未知 28、初态偏差 5）；续跑只执行其余 968 个此前未开始的案例（56、810、0、102）。续跑只修复基础设施：在同一 socket 上只读 RNG 保活，避免等待期间连接超时；严格验证的原生初态偏差保留并允许继续派发。开始前通过 72 客户端一致性和 360 秒同 socket 保活验证。没有为匹配初态重采样，没有按结果择优重试。基础设施 pilot 的开发索引 48/97/249/345 单独报告为 0/4；这些开发 ID 仍披露在主队列中，pilot 成绩不加进主研究。

主研究按批次与 CLI receipt 去重后为 **135 次调用**，其中 134 次成功、1 次额度错误。已知计数为输入 2,592,713 tokens、输出 195,237 tokens，报告的 reasoning tokens 为 75,511；错误调用缺少计数，完整 token 总量仍未知。CLI 延迟合计 6,186.44 秒，中位数每调用 46.16 秒，范围 2.33–77.35 秒。这是 CLI 用时，不是整个复测耗时或每案例端到端等待；美元成本不可用。pilot 独立为 1 次调用，输入 19,071、输出 629、reasoning 137 tokens，耗时 24.35 秒。见[救援报告](https://github.com/Quinn-Ma/OriginX/blob/main/ASTRA_RESCUE_REPORT.md)、[逐案例分类](https://github.com/Quinn-Ma/OriginX/blob/main/evidence/astra-rescue/case-classification.json)和[去重用量](https://github.com/Quinn-Ma/OriginX/blob/main/evidence/astra-rescue/cli-usage.json)。

此前的[三个失败案例离线分析](https://github.com/Quinn-Ma/OriginX/blob/main/ASTRA_FAILURE_ANALYSIS.md)保留为历史诊断记录，其原因假设不能自动视为本轮已经证明。现有在线证据支持上述协议内的 56 个救回；不能把它们与旧成功相加，宣称新的 2,500 回合成绩或官方榜首。
