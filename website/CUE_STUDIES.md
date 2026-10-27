# Cue-precision studies: point cue, fine-region cue, contour masks

Last updated: 2026-10-27. Companion note for the "05 / Cue-precision studies" section of the OriginX project page. 中文版本：[CUE_STUDIES_ZH.md](CUE_STUDIES_ZH.md).

## The question

Does a frozen robot policy benefit from a more precise description of what to grasp and where to place it, without any change to its weights, its native action outputs or the original placement target?

## Executor, task and what stays fixed

- Executor: the frozen public Xiaomi-Robotics-1-RoboCasa365 policy (XR1). This is the public base policy, not the OriginX continuous branch; these studies are separate from the OriginX leaderboard entry and from the Astra recovery study.
- Task: RoboCasa CoffeeSetupMug (one task, one simulator).
- Planner and verifier: frozen Qwen3-VL-4B-Instruct.
- Fixed in every arm: policy weights, planner, verifier, native action interface, stage-transition rules, the original placement destination. Only the visual cue drawn into the camera views differs between arms.

## Three cue designs

1. **Point cue (baseline).** The existing placement marker, a point-style cue drawn at the destination, with the stage text. Every comparison is paired against this arm.
2. **Fine-region "neighborhood" cue (Study 1, finished).** The same marker augmented with a lattice of neighbouring reference points in the parent frame, an object-centre proxy and a fixed text description. The destination itself is not moved. The comparison therefore tests this bundle, and cannot attribute an effect to finer identity alone.
3. **Exterior contour masks + destination ghost (Study 2, task outcomes not measured).** Exterior contours of the target object and a subject-sized destination ghost box (derived from the held object's root-local bounding box and the parent-frame axes), painted with supervision 0.27.0 `MaskAnnotator`. Supervision is the painter library, not a segmentation model; masks use simulator-provided geometry, so this is an oracle-assisted presentation study rather than RGB-only perception. Object interiors are required to stay pixel-unchanged.

## Paired, preregistered protocol (shared by both studies)

- 12 seed pairs fixed before any outcome (Study 1: seeds 20261031–20261042; Study 2: seeds 20261044–20261055, manifest sha256 prefix `94d4721e1452`).
- Each seed is run once per arm; arm order alternates (counterbalanced); matched initial states and independently generated natural plans are required for a pair to be admissible.
- Each episode is capped at 600 steps. No seed is replaced and no case is rerun because of a failure.
- UNKNOWN accounting: an infrastructure outcome (for example a verifier timeout or an interrupted case with an opened trace) stays UNKNOWN. It is not recounted as a failure, and its pair is excluded from paired inference.
- Primary contrast: exact two-sided McNemar test on discordant pairs. The reported 95% interval for the success-rate difference is conservative: the difference of 97.5% Clopper–Pearson intervals for the win and loss probabilities (Bonferroni, simultaneous coverage at least 95%). The interval and the conditional test are not inversions of one another; neither is cherry-picked.
- A worst-case envelope over all 12 planned seeds (known arms keep their result, UNKNOWN arms take success or failure) is also reported. It contains no sampling uncertainty and is not a confidence interval.
- Secondary descriptive endpoint: final horizontal (XY) distance between the object root and the receiving location at episode end. This is not mug-mouth-to-nozzle precision and not a trajectory error at a shared time point.

## Study 1 result: fine-region cue vs point cue (finished)

Run on a local RTX 5090 Laptop GPU (all model inference on CUDA; cloud cost 0). The run was interrupted once by a host shutdown after 6 episodes (3 complete pairs); the remaining 18 episodes were executed by a resume driver with the same frozen manifest, configuration, ports and models. One case (seed 20261034, regions arm) had opened a trace but executed zero environment steps when the host went down; it was re-executed after registration. Under the registered rule an interrupted case with an opened trace counts as UNKNOWN, so the primary analysis records it as UNKNOWN and the re-executed result enters only a supplementary analysis.

### Registered rule (primary analysis, 10 admissible pairs of 12)

| Quantity | Value |
|---|---|
| Point cue: success / known failure / UNKNOWN | 7 / 4 / 1 |
| Fine-region cue: success / known failure / UNKNOWN | 6 / 5 / 1 |
| Paired success | point 6/10, regions 6/10 |
| Discordant pairs | 0 regions-only, 0 point-only (6 both succeed, 4 both fail) |
| Difference, regions − point | 0.0 percentage points |
| Conservative 95% interval | [−35.5, 35.5] percentage points |
| Exact two-sided McNemar p | 1.000000 |
| Worst-case envelope over 12 seeds (not a CI) | [−16.7, 0.0] percentage points |
| Final XY distance, paired subset median | point 3.43 cm, regions 2.00 cm (10 episodes each) |
| Paired distance difference, regions − point (10 pairs) | median −0.31 cm, mean −0.57 cm (negative favours regions) |

Excluded pairs: seed 20261042 (point case UNKNOWN: infrastructure timeout, `TimeoutError: timed out`, after 544 of 600 steps; not rerun) and seed 20261034 (regions case interrupted at zero steps; counted as UNKNOWN under the registered rule).

### Deviation-inclusive analysis (supplementary, 11 pairs)

Adds the re-executed seed-20261034 pair. This is a disclosed post-registration deviation.

| Quantity | Value |
|---|---|
| Point cue: success / known failure / UNKNOWN | 7 / 4 / 1 |
| Fine-region cue: success / known failure / UNKNOWN | 6 / 6 / 0 |
| Paired success | point 7/11, regions 6/11 |
| Discordant pairs | 0 regions-only, 1 point-only (6 both succeed, 4 both fail) |
| Difference, regions − point | −9.1 percentage points |
| Conservative 95% interval | [−45.7, 32.7] percentage points |
| Exact two-sided McNemar p | 1.000000 |
| Worst-case envelope over 12 seeds (not a CI) | [−16.7, −8.3] percentage points |
| Final XY distance, paired subset median | point 3.39 cm, regions 2.13 cm (11 episodes each) |
| Paired distance difference, regions − point (11 pairs) | median −0.30 cm, mean 0.27 cm (negative favours regions) |

The entire difference between the two analyses comes from the added pair: seed 20261034, point success, regions failure.

### Per-seed ledger

S = success, F = known task failure, U = UNKNOWN. XY is the final horizontal distance in cm.

| Seed | Point cue | Fine-region cue | Admissible (registered / deviation-inclusive) |
|---|---|---|---|
| 20261031 | F, 600 steps, 16.29 | F, 600 steps, 17.67 | yes / yes |
| 20261032 | F, 600 steps, 11.81 | F, 600 steps, 7.40 | yes / yes |
| 20261033 | F, 600 steps, 19.43 | F, 600 steps, 19.18 | yes / yes |
| 20261034 | S, 319 steps, 3.39 | F, 600 steps, 12.10 (re-executed after a zero-step interruption) | no / yes |
| 20261035 | F, 600 steps, 6.78 | F, 600 steps, 8.22 | yes / yes |
| 20261036 | S, 256 steps, 0.67 | S, 258 steps, 0.37 | yes / yes |
| 20261037 | S, 281 steps, 3.14 | S, 290 steps, 0.31 | yes / yes |
| 20261038 | S, 295 steps, 1.51 | S, 289 steps, 0.97 | yes / yes |
| 20261039 | S, 280 steps, 3.72 | S, 296 steps, 2.13 | yes / yes |
| 20261040 | S, 274 steps, 2.03 | S, 295 steps, 1.71 | yes / yes |
| 20261041 | S, 262 steps, 0.19 | S, 290 steps, 1.87 | yes / yes |
| 20261042 | U (timeout after 544/600 steps) | F, 600 steps, 15.99 | no / no |

### What Study 1 shows

- Under the registered rule the paired success rates are identical. The validation does not establish a benefit of the fine-region cue, and it does not establish equivalence either (the interval spans ±35.5 points).
- The deviation-inclusive analysis also does not support a success benefit of the fine-region cue.
- The earlier three-seed development comparison (2/3 and 3.64 cm) is reported separately and is not pooled with these 12 seeds.
- Separately, 74 fixed-frame GPU diagnostics showed persistent physical-judgement errors in the visual verifier despite controller and prompt repairs; the evaluated presentation therefore excludes experimental alignment feedback, and no claim is made that feedback was repaired.

## Study 2 status: contour masks vs point cue (task outcomes not measured)

### What has been done

- Implementation finished, with a CPU test suite covering resume, budget, control interfaces, geometry and lifecycle, policy-input checks, analysis and task-time accounting. These tests verify software contracts on synthetic inputs; they are not rollouts and say nothing about task success.
- Rendering amendment registered on 2026-10-25, before any pilot or prospective outcome: because native 4-sample RGB varies across repeated renders at fixed physics, both arms use a declared zero-sample policy-RGB rendering cohort on the H800 (`mujoco/visual/quality/@offsamples` only; physics, seeds and actions unchanged). Older 5090 results and other rendering cohorts are not pooled.
- Fidelity checks (T3, sim-in-the-loop, no model services) finished on an AutoDL H800 (EGL, NVIDIA 595.71.05) on 2026-10-26. Result: `all_passed: false`, 11 of the 12 declared checks pass.
  - Pass: depth convention (`legacy` everywhere), frozen extractor rule vs registry (189,225 / 189,225 equal), production single-sample context, independent single-sample reference render (0 mismatched pixels), policy RGB byte-identical with sites hidden, non-ghost pixels of the ghost render equal the plain render, 0 pixels inside masks changed by painting, ghost/subject/ring disjointness and supervision = NumPy painter, per-step overhead medians 74.6 / 72.7 / 70.2 ms within the 100 ms budget.
  - Pass (registered pilot gate G4): ghost GL cross-check IoU ≥ 0.98, minimum 0.9925, median 1.0 over 115 evaluated of 117 masks.
  - **Fail (stricter engineering check):** all mismatched pixels within a 1-px boundary band. 3 pixels fall outside the band in 1 of 117 masks (seed 20261026 scripted-motion trajectory; analytic 2767 px vs GL 2753 px, 14 mismatched px, IoU 0.9949). The other 116 masks have 0 pixels outside the band. No code or threshold was changed in response; the failure is recorded and is disclosed in the paper's fidelity table.
- Registered runs: a six-case engineering pilot (seeds 20261025–20261027) and the formal 12-pair, 24-case run (seeds 20261044–20261055), 2 workers on one H800.

### Task outcomes

**Not measured.** The six-case engineering pilot was launched on the H800 at 2026-10-26 23:06:50 UTC and the instance became unreachable about 20 minutes later (compute budget exhausted); no episode result was retrieved, so it is unknown whether any pilot case finished. The 12-pair formal run was never started, and the instance was not restarted. There is no mask success count, no paired difference, no interval and no p-value. Nothing here should be read as a mask result, positive or negative. The frozen formal manifest was re-frozen locally after the shutdown and reproduces the registered hash prefix `94d4721e1452`; the preregistered protocol and the offline analyzer (`analyze_masks_validation.py`, same schema as the Study 1 report with `masks` in place of `regions`) remain available for a future run.

## Evidence boundaries

1. Single task, single simulator, single frozen policy. Nothing here is a cross-model or cross-benchmark claim.
2. Both arms share simulator identity, semantic position and geometry; the receiving location is simulator-provided. None of the cues is pure RGB or privilege-free.
3. The fine-region comparison tests a bundle (lattice, centre proxy, fixed text) and cannot isolate finer identity.
4. The original placement target and the native action interface are preserved; there is no hand-written action correction. The full identity-cue versus native-policy contrast is not what these studies measure.
5. These studies do not evaluate the OriginX continuous branch, do not touch the RoboCasa365 leaderboard entry, and do not change the Astra recovery study.

## Files

- Fine-region cue study, anonymous paper (PDF): https://github.com/Quinn-Ma/OriginX/blob/main/experiments/cue-studies/regions-20261024/regions_validation_paper.pdf
- Cue-study evidence folder: https://github.com/Quinn-Ma/OriginX/tree/main/experiments/cue-studies
- Project page section: https://robocasa.originxairobotics.com/#cue-studies
