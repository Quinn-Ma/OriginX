# OriginX confirmatory experiment: independent design review

Status: prospective design only; no experiment was launched by this review. Freeze the final executable manifest, prompt, configurations, analysis code, resource cap and this design **before the first confirmatory outcome is inspected**. The previous 1,496/2,500 and 56/1,004 records remain separate historical results.

## Recommended minimum that answers the three requests

Use one new sealed **50-task × 50-seed manifest: 2,500 unique environment cases**. Evaluate the frozen OriginX B endpoint and the unadapted released Xiaomi-Robotics-1-RoboCasa365 endpoint. The second endpoint must exclude both the Action LoRA and the continuous branch, with exact base revision and tensor/configuration hashes recorded. It is a **closely related second policy**, not a different architecture or an independent pretraining lineage.

| Policy | Arms on every one of the 2,500 cases | Planned case–arm outcomes |
|---|---|---:|
| OriginX B, frozen | C no-change/sham; R restatement; G generic; L vision-blind Astra High; V visual Astra High | 12,500 |
| Original Xiaomi, frozen | C no-change/sham; V visual Astra High | 5,000 |
| Total | 5,000 policy–case units, on 2,500 unique environment cases | **17,500** |

Within a policy–case unit, **reuse the same recorded C outcome as the comparator for all its other arms**. Do not run a new C for each comparison. Do not share control trajectories, advice, or outcomes across the two policies: their states at the trigger generally differ. Each policy gets its own visual request at its own current state.

This is smaller than a complete 2-policy × 6-arm grid. C is deliberately the no-change implementation of the same intervention wrapper, so no additional full-manifest bare/no-wrapper arm is needed for the primary question. Verify the wrapper against the bare evaluator on disclosed development cases before freezing it. If this parity gate fails, repair it before sealing; do not launch a known inequivalent C and relabel it “baseline.” If a separate full-manifest bare arm is required, it adds 2,500 runs per included policy and is an explicit scope increase.

**Do not remove L while claiming to have tested visual grounding.** V versus R/G can test whether the full assistant exceeds simple text, but cannot distinguish image-conditioned reasoning from stronger task-text reasoning. Conversely, Xiaomi C/V alone tests transfer of net benefit to the related endpoint; it does not replicate the mechanism controls on that endpoint.

## Case sealing, fairness, and leakage prevention

1. Retain exactly the same published 50-task roster, scene convention, split membership and per-task horizons. Sample 50 new environment/policy seeds per task from a fixed documented generator; check against *all* historical main, pilot and development manifests, not only the 2,500 original seeds. Commit the generator seed, resulting explicit rows and SHA-256 before execution. Do not silently select scenes after seeing outcomes.
2. Do not describe new seeds as held-out tasks, unseen scenes, or an official test server. This is a fresh, sealed seed-level evaluation on the same 50 tasks. Shared training/pretraining task familiarity still applies.
3. The policy/configuration, model service identifier `gpt-6-astra` with `high`, trigger, instruction templates, image handling, fallbacks and analysis are fixed. No training, tuning from confirmatory cases, horizon increases, best-of retries or post-hoc selection of a favorable task subset.
4. Run all assigned cases, including cases the unassisted policy succeeds on. Assistance is scheduled without access to future failure or success labels. The normal terminal-success rule still stops an episode that finishes before the trigger; record such cases as `terminated_before_trigger`, not missing assistance.
5. Randomize execution order with a separate committed scheduling seed and interleave policies/arms/tasks in balanced blocks. Do not execute all controls first and all V arms after a service update. Record GPU, worker, timestamps and service/request receipts. Only GPUs 3/6 are authorized in the inherited experiment scope; GPUs 2/7 remain excluded unless the user changes that scope.
6. Use separate, disclosed old development IDs for infrastructure tests. These must not be rows in the new sealed manifest. Passing those tests is not confirmatory efficacy evidence.

## Arm contract

Keep the original midpoint query trigger `tau = 16 * ceil(H / 32)`, the same frozen weights, action integration, replanning cadence, crop settings, native reset and original action horizon. All arms retain the original instruction prefix unchanged.

| Arm | Instruction after the trigger | External inference | Supported contrast |
|---|---|---|---|
| C | Original instruction, **byte-for-byte unchanged** | None needed; sham event/identity/application trace recorded | Reference for the frozen policy under the same wrapper |
| R | Same append template, with subgoal exactly the original instruction stripped only at its ends | None | Repetition/extra task-text effect |
| G | Same append template, one frozen generic instruction for every task/case | None | Generic continuation wording effect |
| L | Same append template; one Astra High response based on task text and step budget, without current scene information | One request if still active at trigger | Strong task-text-only assistant |
| V | Same append template; one Astra High response based on current three RGB views, task text and budget | One request if still active at trigger | Image-conditioned assistant |

Suggested fixed G text, to freeze before execution: `Reassess the current scene, take the next feasible action toward the original task, and continue until the task is complete.` This is an experimental control proposal, not a claim that this wording has been evaluated. Do not revise it after seeing results.

Use the same new frozen prompt body for L and V, explicitly handling an available/unavailable visual observation. The only intentional information difference is current scene imagery. The conservative implementation is **omit the current RGB from L and mark it unavailable**; keep the task, horizon, remaining steps, identity/response schema, character bound and requested reasoning effort identical. Do not give L task-specific descriptions transcribed from V images, simulator state, historical outcomes or V-generated text. Freeze and publish both rendered templates/diffs. Using a new shared modality-aware prompt means this is a new confirmation experiment, not an exact continuation of the old prompt experiment.

L and V use equal *maximum* response and wait budgets, not a fictional claim of equal actual compute. Image input changes tokens and possibly runtime; account actual usage separately. R/G have no external model cost. C/R/G are action-budget and runtime-contract controls; they do not by themselves establish compute-matched superiority.

At most eight requests per contact sheet/batch. **Keep each batch within one policy, arm and task**, with a committed case grouping/row order. This avoids coupling unrelated task clusters and allows task-block uncertainty calculations. Use the corresponding grouping for L/V, but separate calls and no cross-arm context. Routing IDs/tokens should be newly opaque random values; store their mapping privately. Do not embed task, seed, outcome or treatment name into model-visible routing IDs.

## Sham, matching, and waiting

The first-intervention hash difference in the old study was not sufficient evidence of semantic use. C must therefore traverse the same pause/keepalive/validation/resume wrapper while retaining the original instruction and policy RNG state. A “control” that bypasses all wrapper machinery does not settle that objection.

Prefer a case-level barrier: collect the required generated responses from frozen trigger observations, then resume all arms under one documented pause/keepalive schedule. If arms are replayed sequentially, replay the same recorded schedule without reconnecting/resetting the policy socket. One C can then fairly serve R/G/L/V. If exact pause matching is infeasible, keep the difference explicit, validate service-time invariance on development cases, and restrict the claim to the **deployed intervention package** rather than text semantics alone. Do not conceal this limitation by calling all arms latency-matched.

For each policy–case, use a deterministic reference initial fingerprint and independently check every comparison:

- identical native initial scene, RGB, state and instruction fingerprints across compared arms;
- exact paired state/RGB/instruction/action-query prefix before the intervention;
- equal state/RGB at the intervention query, before the changed instruction is evaluated;
- same policy RNG contract and recorded socket/keepalive events;
- evidence that the specified R/G/L/V instruction, or exact unchanged C instruction, reached the policy query;
- per-arm terminal success, failure, early termination, technical termination and action-step count.

A shared control does **not** require one bad arm to invalidate every other arm. Store pair-specific comparability flags (e.g., V/C and V/L can differ), and never optimize the primary denominator by selecting their intersection after seeing outcomes. The primary analysis below keeps all 2,500 assigned cases per policy.

Do not introduce simulator snapshot cloning merely to reduce cost unless full simulator/controller/RNG restoration has already been validated. The conservative count is 17,500 separately recorded arm executions, naturally stopping on success. Cases terminating before the trigger require no Astra call. If a previously validated common-prefix executor can logically share an early terminal prefix, disclose logical outcomes and physically executed trajectories separately; never label a copied record as another independent rollout.

## Failure and fallback policy: primary system effect versus mechanism evidence

Freeze a **fallback-to-original-instruction** rule for generated arms. On timeout, quota denial, malformed response, prohibited tool event, identity mismatch or unavailable inference, continue the same live episode under its unchanged original instruction when safe/technically possible. Keep the full original error and record `assistance_applied=false` plus its cause. There is no substitute model, changed prompt, extra observation or regenerated response for the same request.

This is an assigned-system/intention-to-assist result: a valid completed fallback episode remains in the main system-effect estimate with its observed success/failure. It must not be removed merely because advice was not applied. A separate eligible-applied analysis asks the narrower mechanism question and reports coverage. This prospective fallback differs from labeling all invalid assistance as unknown in the historical strict-rescue study; do not rewrite the historical ledger to match it.

If an episode cannot safely continue, initial/prefix matching fails, or its terminal outcome cannot be verified, retain a technical/unmatched outcome. **Do not resample its seed or initial scene.** A transport retry may retransmit the same already-produced response from recorded durable state; it may not call Astra again after a CLI output was produced. Keep attempt lineage and hashes. A pre-dispatch infrastructure failure can be resumed only when records prove no action/model call was executed; otherwise the primary first attempt stays and any diagnostic rerun is supplementary.

Automatically drain and stop new dispatch on an unexpected shared infrastructure fault. Preserve ownership checks and first-error evidence. Repair on development cases, version the infrastructure, and continue only the untouched manifest rows. Report execution partitions; no favorable replacement of completed failures. Never start duplicate model owners or kill another user's job.

## Estimands and accounting

For policy p and comparison A versus C, with `N=2500`, let `n01` count control failures/assisted successes and `n10` control successes/assisted failures in verified comparable completed pairs.

- Main endpoint with complete evidence: `Delta_p(A,C) = (n01 - n10)/2500`, in percentage points. Report the full 2×2 table, both arm success counts and rates, and regressions **alongside** rescued failures. The difference of the two paired rates is the same quantity.
- Each of the 50 tasks has 50 seeds, so the complete-cohort pooled difference equals the equal-task mean difference. Do not reweight tasks by their observed failure count.
- Report generated/applied/fallback/unknown/unmatched counts separately. At most 2,500 unique cases per policy; 17,500 is arm outcomes, not 17,500 different failure cases.
- If any target outcomes/comparisons are unknown, do **not** publish a complete-cohort point estimate based only on eligible pairs. Report the fixed-N identification interval. For each arm outcome define `[Ylo,Yhi]=[Y,Y]` when verified or `[0,1]` when unknown; sum `Alo-Chi` and `Ahi-Clo`, then divide by 2,500. Use known one-sided outcomes to tighten the bound. If scene comparability fails, the intended paired comparison is unknown even if each noncomparable rollout has a raw outcome.
- Present the conditional valid-pair estimate separately, with its own denominator and attrition pattern. It cannot replace the fixed-N result or justify a benchmark claim when missingness is outcome-dependent.
- Count outcome-changing regressions in both generated arms, including premature “completion” advice. Report first-query output differences versus C in R/G/L/V and sham repeat parity only as behavioral diagnostics, not proof of semantic reasoning.

The most defensible scientific claims are: (1) whether V changes **net** task success for B over this complete new manifest; (2) whether V exceeds a matched task-text-only assistant L; (3) whether V/C net benefit repeats on the closely related original endpoint. The previous `(1496 + rescued)/2500` construction remains invalid.

## Confirmatory inference and multiplicity

Predeclare a **three-contrast primary family**: B V−C (full-manifest system benefit), B V−L (information from current imagery), original-Xiaomi V−C (related-policy replication). Use two-sided tests and Holm familywise correction at 0.05. Report effects and intervals whether positive, zero or negative. Do not define “replicated” from a positive point estimate alone; report the original-endpoint interval and corrected result, with the close relationship stated.

For uncertainty, retain the same 50 task blocks jointly across arms/policies. Task-local batch formation keeps batching dependence within those blocks. Use a **preimplemented, validated task-cluster procedure** for differences of binary outcomes, with a fixed random seed; a null-imposed wild cluster bootstrap with 9,999 or more repetitions and clustered studentization is suitable. Do not replace it ad hoc with independent-case bootstrapping or claim an exact randomization test merely because execution order was randomized. There are only 50 task clusters, and the sampled task roster is fixed: inferential generalization requires cluster/exchangeability assumptions, so foreground the observed benchmark effect and treat cross-task generalization cautiously.

If a validated cluster-test implementation is not available before launch, freeze the simpler conservative analysis now: primary effects and task-block confidence intervals, no claim of statistical superiority; Holm applies only to correctly calibrated p-values. Exact McNemar from pooled discordances can be a labeled sensitivity analysis, but is not the sole confirmatory evidence when task/batch dependence is present. Do not choose an inference method because it becomes significant.

Secondary B comparisons V−R, V−G, R−C, G−C and L−C are a separately declared five-comparison family with Holm correction, clearly secondary. Split/task-specific analyses, action-hash changes, timing and mechanism narratives are exploratory. A policy-by-treatment interaction is optional exploratory evidence; two policies sharing a base cannot establish broad cross-architecture transfer even if both are significant.

Unknown comparisons preclude a complete fixed-N point test without extra assumptions. Report the identification bounds first; valid-pair tests are conditional. Do not turn all unknowns into observed policy failures. Predeclare a practical threshold if desired before launch (e.g., ≥2 percentage points), but do not retrofit it. At N=2,500, an independent-pair normal approximation with 5–15% discordance, 80% power and a conservative two-sided alpha of 0.05/3 gives an approximately 1.4–2.5-point detectable effect; clustering can worsen power. This is a sensitivity warning, not a powered guarantee or a reason to increase N after inspecting outcomes.

Method references: [Holm, 1979, A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733); [Roodman et al., 2019, Fast and wild: Bootstrap inference in Stata using boottest](https://journals.sagepub.com/doi/10.1177/1536867X19830877); [MacKinnon and Webb, Wild Bootstrap Inference for Wildly Different Cluster Sizes](https://onlinelibrary.wiley.com/doi/10.1002/jae.2508/full). These support the statistical methods, not a claim that this particular experiment automatically satisfies all assumptions.

## Resource ceiling and execution priority

Hard logical workload: **17,500 arm outcomes**, no extra training. Worst-case generated requests: `2500 × (B-L + B-V + Xiaomi-V) = 7,500`; early successful episodes reduce this. Prepartition each task's 50 cases into six groups of eight and one of two. With the three generated policy/arm combinations, the hard no-retry maximum is **1,050 CLI batches**. Do not allow timeout-driven singleton regrouping to silently exceed this count. No paid top-up or second quota reset is implied; the previously authorized single reset has already been used.

Historical completed-batch means suggest an order of **20 million input tokens and 1.5 million output tokens** at 1,050 similarly sized batches, before changes in vision-blind input and early termination. This is a rough extrapolation, not a reservation or reliable upper bound. Token caps must be checked against actual account availability and the supported CLI controls before launch. Set a receipt-based stop threshold and preserve all remaining rows as unexecuted/unknown if the cap is reached. Never switch assistant models to stretch the budget without explicit scope revision.

The GPU cost is also materially larger than the prior failure-only run. Do not promise this finishes in a few hours. Measure development throughput without reading confirmatory efficacy outcomes, multiply by the registered task horizons and arm counts, and freeze a realistic wall-clock/GPU-hour ceiling. Per-request wait stays bounded at 1,200 seconds; model subprocess timeout and transport retry ceilings remain separate logged limits.

Use balanced, outcome-blind dispatch across task/arm blocks rather than spending all resources on V first. If resources require staging, seal the complete 17,500-arm manifest and a fixed staged schedule beforehand; analyze the final fixed cohort only after completion or declared stop. A partial stage is a partial study, not “all controls completed.” The execution coordinator should report the remaining budget and either finish or stop honestly; it should not spend unbounded quota seeking a favorable result.

## Completion requirements

- Frozen manifest/protocol/prompt/configuration hashes and a pre-run timestamped design receipt.
- 2,500 cases × correct arms × correct policies, with a terminal or explicit unexecuted status for every assigned row; immutable error/attempt lineage.
- Per-comparison matching, early termination, actual applied advice, fallbacks, full 2×2 outcomes, net changes, regressions, uncertainty/multiplicity, fixed-denominator unknown bounds.
- Actual batch/request/token/wait/GPU accounting, including unsuccessful calls; no reasoning-token double count.
- Verification that only owned workers/models were terminated; complete evidence synced locally.
- Update the paper with measured outcomes, including null/negative results. Distinguish fresh complete-manifest confirmation, older failure-selected analysis, related-policy replication and untested cross-architecture generalization.
- “Ready for review” is a document status. These experiments cannot guarantee CVPR acceptance, spotlight, official leaderboard acceptance, or first place.
