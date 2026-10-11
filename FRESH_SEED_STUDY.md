# OriginX: fresh-seed policy and assistance study

The new study completed all 50 trials for the frozen base policy (X), adapter-only policy (A), full OriginX policy (B), and B with visual Astra assistance (BV). They recorded 30, 32, 33, and 32 successes, respectively. Strict B/BV matching found one rescue and one regression: **no observed net gain among comparable pairs**. A separate GR00T extension ended with 31 of its planned 50 cases completed; its remaining cases stay unknown.

This study does not replace the original B2000 result of 1,496/2,500, the historical-failure rescue result of 56/1,004, or the earlier independent 50-case result of 29/50 for both control and visual assistance.

## Fixed protocol

The cohort contains one new seed for each of 50 RoboCasa365 tasks: `sprint-fresh-00`–`49`, environment and policy seeds 2027101100–2027101149. Seeds were fixed without selecting outcomes or replacing failed trials. Native reset and each task's registered horizon were retained. All model weights were frozen; there was no new training or real-robot experiment. X is the public Xiaomi base, A is the Action LoRA policy, and B adds the continuous conditioning branch. A2000 remains the release name for weights trained for 1,613 updates; B has 2,000 updates.

For assistance, `gpt-6-astra` at `high` receives the original instruction and three current camera views once at the fixed midpoint query, then the frozen policy continues with the returned subgoal. Control preserves the original instruction. Action horizons are matched; wall-clock latency is not. Initial states must match for a valid policy comparison. B/BV and GR00T control/visual comparisons additionally check shared pre-intervention traces and advice evidence; different X/A/B policies are not required to produce identical action prefixes. The separate engineering case `sprint-dev-00` (seed 2027201100) is excluded from this fresh cohort.

## Completed OriginX arms

| Frozen policy | Success | Failure | Unknown | Planned |
|---|---:|---:|---:|---:|
| X: public base | 30 | 20 | 0 | 50 |
| A: Action LoRA | 32 | 18 | 0 | 50 |
| B: continuous branch | 33 | 17 | 0 | 50 |
| BV: B + visual Astra | 32 | 18 | 0 | 50 |

These are individual recorded outcomes, not a fully matched causal comparison: only 42 cases share the same initial state across X/A/B; eight retain initial-state deviations. The raw differences do not establish policy superiority.

B/BV has 44 matched pairs and six initial-state deviations. The matched table is 27 both-success, one BV-only success, one B-only success, and 15 both-failure. Astra was applied in 25 BV trials, including 22 matched pairs; the other matched pairs terminated before intervention. The verified rescue was `CloseToasterOvenDoor` (`sprint-fresh-06`), and the verified regression was `PortionHotDogs` (`sprint-fresh-28`). Net change among matched pairs is zero. The conditional task-block bootstrap interval is −6.82 to +6.82 percentage points; the fixed-cohort identification bounds are −10 to +2 points. Identification bounds describe unresolved comparability, not sampling confidence. Neither supports a claim of established benefit.

## GR00T extension: retain all 50 planned cases

GR00T N1.5 RoboCasa `checkpoint-120000` was tested with unchanged-instruction control and one visual Astra intervention. Execution ended with 31 complete two-arm cases (62 outcomes); 19 cases (38 outcomes) were not run. The denominator remains 50 per arm.

| GR00T arm | Success | Failure | Unknown / unrun | Planned |
|---|---:|---:|---:|---:|
| Control | 6 | 25 | 19 | 50 |
| Visual Astra | 7 | 24 | 19 | 50 |

There are 29 matched pairs, two initial-state deviations, and 19 unknown pairs. Among matched pairs, six succeed in both arms, one succeeds only with assistance, none succeeds only in control, and 22 fail in both. The rescue is `GarnishPancake` (`sprint-fresh-10`); no matched regression was observed. Astra was applied 28 times, including 26 matched pairs. Success identification bounds are 12–50% for control and 14–52% for visual assistance; full-cohort net-effect bounds are −36 to +44 percentage points. The incomplete study cannot establish a population benefit or be described as a completed 50-case evaluation.

## Cost and what the diagnostics show

| New-study scope | CLI calls | Input tokens | Output tokens | Reasoning tokens within output | Summed CLI latency |
|---|---:|---:|---:|---:|---:|
| BV | 25 | 430,501 | 7,548 | 3,743 | 306.53 s |
| GR00T visual | 28 | 482,162 | 7,608 | 3,401 | 322.55 s |
| Total | 53 | 912,663 | 15,156 | 7,144 | 629.08 s |

Calls are deduplicated by invocation and receipt, with complete counters in the sealed inventories. Reasoning tokens are included in output and must not be added again. Latency is summed CLI service time, not elapsed study time; monetary cost was not recorded. Historical and engineering usage is separate.

All eight X/A/B initial-state deviations contain real free-object position differences; none is merely a pixel or texture discrepancy. This diagnosis preserves the eight exclusions from strict matching and does not establish their cause. It motivates tracing reset stages before causal comparisons. The proposed intervention gate passed 15 synthetic checks, with zero training runs and zero rollouts: no empirical gate benefit is claimed.

## Evidence

[Machine-readable summary](evidence/fresh-seed-study/summary.json) · [Full audits and usage](evidence/fresh-seed-study/) · [中文报告](FRESH_SEED_STUDY_ZH.md)

Source hashes, case classifications, terminal outcome records, and unknowns are retained. These are author-run research results, not an organizer-assigned leaderboard rank.
