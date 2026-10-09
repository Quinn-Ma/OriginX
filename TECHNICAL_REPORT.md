# OriginX: An Audited 2,500-Episode RoboCasa365 Evaluation

Qinzhen Ma · October 9, 2026

**Abstract.** We report a frozen-checkpoint evaluation of OriginX, a derivative of Xiaomi-Robotics-1-RoboCasa365 with a previously trained Action LoRA adapter and an additional continuous conditioning branch. Across 50 target tasks with 50 trials per task, the model succeeds in 1,496 of 2,500 episodes, giving a task-weighted success rate of 59.84%. Atomic-Seen, Composite-Seen, and Composite-Unseen success rates are 83.00%, 59.875%, and 33.75%, respectively. All planned trials have terminal records, including 1,004 policy failures; there are no missing or infrastructure-unknown outcomes. We retain the frozen task schedule, seeds, horizons, original reset implementation, and per-episode audit records. These are author-run results, pending organizer review. A separate earlier paired comparison did not demonstrate a statistically reliable improvement over the inherited base, so the present single-arm result does not establish the causal benefit of the conditioning branch.

**Model and attribution.** This model extends the public [Xiaomi-Robotics-1-RoboCasa365 checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365) and [Xiaomi-Robotics-1 code](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1). It is an independent derivative. A2000 is the public release alias of historical A1613; its weights are unchanged and its actual training count remains 1,613 updates. We distinguish three stages. The public Xiaomi checkpoint supplies the base policy. A2000 adds an Action LoRA adapter with rank 16 and alpha 16, trained for 1,613 updates over 80,000 sampled windows. B2000 freezes the complete A2000 policy, including its LoRA adapter, and trains a 4,216,839-parameter continuous conditioning branch. The branch is used during action generation at inference time, rather than serving only as an auxiliary prediction head. A stage-classification objective supplies auxiliary supervision.

B2000 completes 2,000 updates with global batch size 128, totaling 256,000 sampled windows. The frozen-teacher velocity objective has coefficient 1.0 and the auxiliary stage loss has coefficient 0.05. Its recorded training duration is 8,453.98 seconds. A2000 changed batch size during training; multiplying its final batch size by all 1,613 historical updates would misstate its sample count. The leaderboard training fields refer specifically to the B2000 branch-training stage, while this report describes the inherited training chain.

**Training-data scope.** The audited A and B sample plans contain 80,000 and 256,000 entries, respectively, all marked as human data. They cover Human300 tasks and do not include any of the 16 official Composite-Unseen target tasks. B mixes Human300 replay and a 200-trajectory pilot collection in equal proportions. The pilot contains 16 Composite-Seen tasks, with 168 trajectories used for training and 32 held out. This audit covers the bound local sample plans; it does not independently audit the full pretraining history of the inherited Xiaomi checkpoint. No claim of training from scratch or of using only the branch-training data is made.

**Frozen evaluation protocol.** The evaluation uses the 50 target tasks of the RoboCasa365 multi-task benchmark in pretraining kitchen scenes, with 18 Atomic-Seen, 16 Composite-Seen, and 16 Composite-Unseen tasks. Every task receives exactly 50 trials, using its registered horizon. The environment and policy seeds are fixed in advance in the range 2090900000–2090902499; there are no replacement seeds. Checkpoint selection and hyperparameters are frozen before this evaluation. The protocol follows the task and scene split described in the [RoboCasa multi-task evaluation documentation](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html).

Inference uses five Euler steps, replanning every 16 steps, observation history length four, observation interval two, and crop ratio 0.95. The evaluator verifies the original `Counter.get_reset_regions` implementation before and after each episode. Earlier patched-reset development results are not substituted for the new full evaluation. Every failed episode reaches the task's original horizon. The release preserves the manifest and the individual results, so failure records contribute to the denominator exactly as success records do.

**Results.** Because all tasks have 50 trials, the mean of task success rates is equal to pooled successes divided by 2,500. The overall score weights the three splits by their task counts; taking an unweighted mean of the three split percentages would produce a different and inappropriate value for this experiment.

| Evaluation split | Tasks | Successful episodes | Total episodes | Success rate |
|---|---:|---:|---:|---:|
| Atomic-Seen | 18 | 747 | 900 | 83.00% |
| Composite-Seen | 16 | 479 | 800 | 59.875% |
| Composite-Unseen | 16 | 270 | 800 | 33.75% |
| Overall | 50 | 1,496 | 2,500 | 59.84% |

Atomic-Seen performance is substantially higher than performance on unseen composite tasks within this evaluation. That descriptive gap is consistent with remaining difficulty in the unseen composite setting, but the present experiment does not isolate its cause. No task-level cherry-picking, discarded failures, or completed-subset score is used.

At the October 9 check of the [public leaderboard](https://robocasa.ai/leaderboard.html), the page, marked updated October 8, displayed Paimon-0 first with 58.1 overall and Xiaomi-Robotics-1 with 57.4. Our author-run score is numerically above those displayed aggregate values. This is an unmatched comparison across different evaluations, not evidence of statistical superiority, an organizer-validated rank, or a matched baseline improvement. The current result has not been accepted onto the leaderboard.

**Earlier paired comparison and negative evidence.** An earlier development evaluation covered 30 tasks with five seeds per task: the original Xiaomi model scored 72% and B2000 scored 76%. A subsequent confirmation used 20 new seeds per task, yielding 600 episodes for each model. The original model succeeded in 413/600 episodes (68.83%), and B2000 succeeded in 419/600 (69.83%). There were 53 pairs won by B, 47 pairs lost, and 500 ties. The estimated difference was one percentage point, with a reported 95% interval of [-2.17, 4.17] percentage points and McNemar p=0.6173.

That confirmation failed the predeclared requirements of at least a five-point gain and a positive interval lower bound. We retain this result and do not describe the branch as a proven improvement. The older 30-task evaluation is not a substitute for the new 50-task native-reset experiment. Conversely, the new single-arm experiment does not overturn the earlier negative evidence: no matched 2,500-episode Xiaomi baseline or branch ablation was run with the new protocol.

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

The main study used **135 unique CLI calls** (134 successful, one quota error), deduplicated by batch and receipt rather than multiplied by the number of cases in a batch. Known counters sum to 2,592,713 input tokens and 195,237 output tokens, including a reported 75,511 reasoning tokens; the error call has no counters, so complete token totals remain unknown. Summed CLI latency is 6,186.44 seconds, median 46.16 seconds per call, range 2.33–77.35 seconds. These are CLI service times, not total replay time or per-case end-to-end waiting. Dollar cost is unavailable. The separate pilot used one call, 19,071 input tokens, 629 output tokens, 137 reasoning tokens, and 24.35 seconds. Complete records are in [the rescue report](ASTRA_RESCUE_REPORT.md), [case classifications](evidence/astra-rescue/case-classification.json), and [deduplicated usage](evidence/astra-rescue/cli-usage.json).

The earlier [three-case offline analysis](ASTRA_FAILURE_ANALYSIS.md) remains a historical diagnostic artifact. Its hypotheses are not explanations established by this new experiment. The online replay evidence supports 56 recoveries under the stated protocol; it does not create a new 2,500-episode score or an accepted leaderboard position.

**Infrastructure and audit.** Simulation uses CPU OSMesa with previously validated rendering decimation that preserves policy-query pixels, plus context-lifetime and readback guards. These departures from a default graphics stack are disclosed for organizer review. Eighteen frozen model services, six on each of three H100 GPUs, serve six isolated-RNG clients each, allowing up to 108 simulation processes. Models initialize sequentially. GPU 2 is excluded. This configuration is an execution detail, not a claim that scaling parallelism improves the policy.

The evaluation retains each episode's initial-scene metadata, reset implementation proofs, RNG and request checks, and readback guard receipt. Final auditing verifies all 2,500 result hashes and all 2,500 guard-record hashes against the frozen records; it also checks task membership, seed identity, horizon completion, and the aggregate scores. All 2,500 planned trials complete, with zero infrastructure-unknown outcomes and zero unattempted trials. Process identity checks confirm that the episode processes, all 18 model services, and the campaign driver have ended.

The campaign driver records about 3 hours 35 minutes, including service initialization; the rollout phase records about 3 hours 13 minutes. The previously completed training is separate from these times. A proposed C continuation was not executed. We make no claim that the complete base model, adapters, training, and evaluation were produced from scratch within a five-hour period.

**Reproducibility and submission limitations.** The released A adapter and B branch retain the following SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| A2000 Action LoRA | `5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4` |
| B2000 branch | `41988b92391953687b39a0bc5a35bce962fbfcd08fbf6e550108430850d9944f` |
| Evaluation configuration | `a3b0dbed4fb26ddcc472645fb9cc13008694293c23c49c325cc7524b4a93cd04` |
| Evaluation manifest | `cecee25e5750605c2c83643bca0fed55c8c4bfcb46d672a934f0c2d29e760257` |

**Current model distribution.** The complete model assets are available from [Qinzhen3/OriginX on Hugging Face](https://huggingface.co/Qinzhen3/OriginX): three base safetensors shards totaling 10,106,433,336 bytes, the tokenizer/configuration assets, `adapter-originx-2000.pt`, and `branch-00002000.pt`. The full snapshot is approximately 10.15 GB. Implementation code is distributed through [the OriginX GitHub repository](https://github.com/Quinn-Ma/OriginX), including three pinned original model Python files. Its `prepare_originx.py` helper combines these sources and verifies the complete original base identity and both adaptation weight hashes. A2000 remains the unchanged adapter release alias; packaging does not change the historical training or evaluation. The original per-episode evidence remains in the versioned GitHub evaluation release. Preparation is not GPU action-parity validation or a new benchmark run.

Source and artifact packaging performed after evaluation is distinguished from the original evaluation entry point. The artifact documentation describes what has and has not been retested. The original evaluation remains unchanged; publishing an inference entry point is not itself a repeat of the 2,500 episodes.

The [official submission requirements](https://github.com/robocasa-benchmark/leaderboard) ask derivatives to attribute the base and demonstrate a measurable improvement from the new contribution. Although this model adds a conditioning branch used by the policy, its contribution has not been established beyond statistical noise in the available paired comparison. We explicitly request organizer review of eligibility and reproducibility rather than presuming acceptance. The verified contribution of this report is the complete frozen-model evaluation and its retained evidence; stronger claims require additional controlled experiments.
<!-- CONFIRMATORY_50_START -->
## Fresh-seed follow-up: the amended 50-case study

At the user's three-hour delivery request, the running 2,500-case prospective plan was reduced to one case per task: the minimum pre-existing fresh environment seed for each of 50 tasks. Selection did not inspect outcomes, but the reduction occurred after launch and is not a new untouched preregistration. B2000 has five arms (C/R/G/L/V), and the original Xiaomi policy has C/V: **50 distinct environment cases, 350 planned arm outcomes**. C is unchanged control; R repeats the original instruction; G appends a fixed generic continuation; L receives text-only Astra guidance; V receives image-conditioned Astra guidance. We retain the original horizon, model weights, seeds, native reset, and Astra High settings.

The terminal audit has **350/350 valid arm records**, with **350/350 claimed**. 191 earlier arm records outside this fixed subset remain separately archived. Incomplete or incomparable pairs remain unresolved within the full 50-case denominator.

| Comparison | Fixed cases | Matched | Rescues | Regressions | Unresolved | Full-cohort net-effect bounds (pp) |
|---|---:|---:|---:|---:|---:|---|
| B/G_vs_C | 50 | 44 | 1 | 1 | 6 | [-12.0, 0.0] |
| B/L_vs_C | 50 | 45 | 0 | 1 | 5 | [-12.0, -2.0] |
| B/R_vs_C | 50 | 47 | 0 | 1 | 3 | [-8.0, -2.0] |
| B/V_vs_C | 50 | 44 | 1 | 0 | 6 | [-8.0, 4.0] |
| B/V_vs_L | 50 | 42 | 2 | 0 | 8 | [-6.0, 16.0] |
| base/V_vs_C | 50 | 47 | 0 | 1 | 3 | [-6.0, 0.0] |

**The follow-up does not establish a net benefit.** B control and visual-assistance arms both record 29/50 successes (58%); repetition records 28/50, generic and text-only advice 27/50 each. These are descriptive arm totals, not a paired causal estimate. V versus C has 44 matched pairs, one treatment-only success and no regression; six initial-state mismatches remain, giving a full-cohort net-effect bound of **−8 to +4 percentage points**. All pairwise technical-unknown and prefix-deviation counts are zero; unresolved counts in the table are initial-state mismatches.

Advice delivery is a separate quantity. The 44 matched B V-versus-C pairs include 18 early-terminal pairs, 18 with visual advice actually delivered, and eight with fallback to the unchanged original instruction. Across all B visual assignments there are 20 delivered interventions, ten fallbacks and 20 early endings. V-versus-L has two V-only successes among 42 matched pairs, but one V arm used fallback and merely retained the success also observed in C; only the other received visual advice. Thus these two events must not both be attributed to visual reasoning. The original Xiaomi policy records 26/50 control and 25/50 visual-arm successes; its 47 matched pairs contain no rescue and one regression.

These are identification bounds that include unresolved comparisons, not confidence intervals. A rescue in this table is a verified reference-arm failure paired with a treatment-arm success; a regression is the reverse. The reference is L in V-versus-L and C otherwise. The B V-versus-L contrast tests the incremental effect of supplying images under the specified interface. Small samples, initial-state mismatches, incomplete advice delivery, and wide bounds limit conclusions; these data do not establish a general visual-reasoning benefit or a benchmark improvement.

**Verified example: WashFruitColander.** At step 1,584, the saved visual advice suggested operating the faucet control to the right of the spout, keeping the colander under the water, and checking whether the mango, peach and apple were in the basket. The V arm applied this instruction and succeeded at step 2,347; matching C and L arms reached their 3,150-step horizon without success. This links actual advice to the successful paired continuation, but does not isolate which suggested physical action caused success.

One continuation stopped when the broker treated an intermediate CLI reconnection event as a terminal error, although the saved invocation eventually completed. The original error, raw events, usage and affected records were retained. The separately audited infrastructure continuation processed only never-claimed work; it never replaces a claimed outcome or regenerates an executed CLI call. The recovery check requires successful terminal completion and validates the saved response, identities and absence of tool calls.

72 unique CLI invocations; known input 1,244,218, output 24,566. Complete input/output accounting: True. Summed invocation latency is 871.88 seconds. Reasoning tokens are included in output, not added again. All original and continuation main-study calls, including out-of-cohort work and failures, are included. Development use is separate: six B/base calls used 102,367 input and 1,011 output tokens (56.25 seconds summed invocation latency); one GR00T admission call used 17,203 input and 213 output tokens (10.85 seconds). These seven calls do not overlap the 72 main-study calls. Dollar cost is unavailable. No additional training, quota reset or paid top-up was used.

GR00T formal cross-architecture evaluation was **not run** within this delivery; development admission is a systems check, not efficacy evidence. The historical records remain **1,496/2,500** and **56/1,004**. These follow-up results are a separate supplementary study and do not produce an accepted leaderboard score.

### Future directions

A larger fresh-seed evaluation should estimate benefit and regression jointly, compare image-conditioned advice against instruction repetition, generic advice and text-only advice, and retain all unresolved pairs. Independent GR00T evaluation is needed for architectural transfer. Fixed-budget comparisons of intervention timing and a stronger stall detector should be prospectively registered. Compute-matched controls, latency accounting, and real-robot evaluation remain open; no outcome-driven retuning is inferred from the present observations.

Terminal report SHA-256: `d437095e785fd603cb6e99a2ae0d9556dbc2254a212ebef138e4a07c592c8270`. Machine-readable evidence: [50-case audit](evidence/confirmatory-50/report.json).
