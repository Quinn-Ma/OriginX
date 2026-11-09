# PaTH-VLA: Progress-and-Target Context for Long-Horizon Robot Policy Adaptation

Qinzhen Ma (Rice University), Harry Yang and Changsheng Lu (The Hong Kong University of Science and Technology),
Xiaogang Xu (Zhejiang University).
Preprint: [`paper/PaTH-VLA_preprint.pdf`](paper/PaTH-VLA_preprint.pdf) ·
supplementary material: [`paper/PaTH-VLA_supplementary.pdf`](paper/PaTH-VLA_supplementary.pdf) ·
LaTeX source: [`paper/PaTH-VLA_source.zip`](paper/PaTH-VLA_source.zip) ·
project page: <https://robocasa.originxairobotics.com/path-vla/>

A controlled, pre-registered ablation of two cheap auxiliary context signals for pretrained generalist
manipulation policies on RoboCasa365: **progress** (predict the current subtask stage) and **target**
(see a box around the current target and predict its location). Three official RoboCasa365 checkpoints
(Xiaomi Robotics 1 / XR-1, Awomo-0.5I, Awomo-0.5V) are adapted with rank-8 LoRA for 1,000 updates in
three arms (S = progress only, BL = target only, F = both, the full PaTH-VLA recipe) and evaluated with
ordinary inputs on the same 200 seeded episodes (10 tasks x 20 seeds) per policy: 2,400 episodes in total,
plus 600 episodes of a secondary feedback study.

Study dates: 2026-11-08 to 2026-11-09. Pre-registration: [`PREREG.md`](PREREG.md), frozen before the first
optimizer update and the first scored episode. SHA-256 of the complete file:
6e1d95234730406cb5b06996fd1079edb7040c8d1e17eadb972704b40f047c8d. One administrative scheduling line is
withheld in this public copy, so the public file does not reproduce that hash.

## Headline numbers (successes out of 200)

| model | official | S | BL | F |
|---|---|---|---|---|
| XR-1 | 68 (34.0%) | 81 (40.5%) | 81 (40.5%) | 83 (41.5%) |
| Awomo-0.5I | 89 (44.5%) | 79 (39.5%) | 69 (34.5%) | 76 (38.0%) |
| Awomo-0.5V | 88 (44.0%) | 83 (41.5%) | 73 (36.5%) | 66 (33.0%) |

Pre-registered primary contrasts (paired, pooled over 200 pairs; task-stratified bootstrap 95% CI; exact McNemar):

| model | F − BL | F − S |
|---|---|---|
| XR-1 | +1.0 pp [−6.0, +8.0], p = 0.89 | +1.0 pp [−5.5, +7.5], p = 0.89 |
| Awomo-0.5I | +3.5 pp [−3.5, +10.5], p = 0.43 | −1.5 pp [−9.5, +6.5], p = 0.81 |
| Awomo-0.5V | −3.5 pp [−11.0, +4.0], p = 0.46 | −8.5 pp [−15.5, −1.5], p = 0.030 |

Holm correction runs over the two primary contrasts of a model. For Awomo-0.5V, F − S has the smaller p; its
Holm-corrected (97.5%) interval [−16.5, −1.0] excludes zero, but its Holm-corrected McNemar p is 0.060. Under the
pre-registered rule (corrected interval excludes zero and the gain is at least +5 pp) no contrast is supported,
and the upper 95% bounds (+6.5 to +10.5 pp in four of the six contrasts) do not exclude gains of that size.

Secondary, uncorrected contrasts against the official checkpoints: XR-1 gains 6.5–7.5 pp in every arm (F: +7.5 pp
[+0.5, +14.5], p = 0.049), mostly on the five seen tasks (F: +13 pp seen, +2 pp unseen; S and BL: +9 and +4).
Both Awomo models lose. Every nominally significant loss is in an arm that drew boxes during training: Awomo-0.5I BL −10.0 pp
(p = 0.033; −18 pp on the unseen tasks) and Awomo-0.5V F −11.0 pp (p = 0.015; pooled only with the crash reruns
below, but −14 pp, p = 0.029, on the seen tasks regardless).
There is no matched action-only arm (by design), so these differences cannot be attributed to either signal.
Full tables: [`results/tables.md`](results/tables.md); per-episode logs: `results/eval/<model>-<arm>-q*/results.jsonl`.

## Deviations and sensitivity analyses

Deviations from the pre-registration and post-hoc analyses (none of these analyses was pre-registered):

- **Scene identity.** A seed is meant to fix the kitchen, objects and initial state. Hashing the first left
  agent-view frame shows it byte-identical (and the initial state identical) across all twelve policies for 191 of the 200 seeds; SearingMeat seeds 1, 4, 6, 11 and 14
  and StirVegetables seeds 1, 11, 14 and 16 came in two first-frame variants, and for StirVegetables 1 and 11 the
  initial base pose also differed. All 200 pairs are kept. On the 191 identical seeds no contrast becomes supported;
  Awomo-0.5V F − S is −9.4 pp (p = 0.015, Holm p = 0.031) and Awomo-0.5I BL − official is −9.4 pp (p = 0.054).
  Output: `code/analysis/out/sensitivity.json`.
- **Crashed episodes.** 18 Awomo-0.5V official-checkpoint episodes, all on unseen tasks, crashed with a GPU
  out-of-memory error and were rerun from scratch. That decision was taken after partial results had been seen. The released logs contain the
  reruns; the original shards with the error rows are in `results/eval_crashed/`, and their 132 completed rows are
  unchanged in the release. 5 of the 18 reruns succeeded. Counting the crashes as failures gives 83 instead of 88
  official successes and changes only the contrasts against this checkpoint: F − official −8.5 pp [−17.0, −0.5],
  p = 0.068 (released: −11.0, p = 0.015); BL − official −5.0 pp [−13.0, +2.5], p = 0.275 (released: −7.5,
  p = 0.086); S − official 0.0 pp [−8.0, +8.0], p = 1.00 (released: −2.5, p = 0.63).
  Output: `code/analysis/out/crash_sensitivity.json`.
- **Unreleased duplicates.** Nine official-checkpoint episodes are not among the 2,400. One is from a smoke test
  (XR-1, DeliverStraw, seed 0) and eight are from an aborted first launch. Each matches its released counterpart
  in outcome, length and first frame (`results/eval_unreleased/raw_rows.txt`).
- **Task screen.** Five seen and six unseen tasks passed the published-success screen (20–60% for all three
  models). PortionHotDogs (exactly 20% for all three) was left out.
- **Head initialisation.** XR-1's new heads did not start from the same random draw in the three arms. The action
  pathway still starts identical, because LoRA B starts at zero; details are in the paper's limitations.
- **Feedback study.** The study fed back the box only, not the stage, although the pre-registration mentions
  stage/box feedback.

## Secondary study: feeding the policy's own target back (600 episodes)

Each F checkpoint replays the same 200 episodes, with one change. The location head's prediction from the previous
query is drawn as a fixed 32×32-pixel box on every camera's current frame. There is no box at the first query, the
instruction is unchanged, and the stage head is not used.

| model | official | F | FB | FB − F | FB − official |
|---|---|---|---|---|---|
| XR-1 | 68 | 83 | 63 | −10.0 pp [−17.5, −2.5], p = 0.013 | −2.5 pp [−9.0, +4.0], p = 0.56 |
| Awomo-0.5I | 89 | 76 | 66 | −5.0 pp [−12.5, +2.5], p = 0.26 | −11.5 pp [−19.5, −3.5], p = 0.013 |
| Awomo-0.5V | 88 | 66 | 58 | −4.0 pp [−11.5, +3.5], p = 0.38 | −15.0 pp [−22.5, −7.5], p < 0.001 |

All p-values are uncorrected, and Awomo-0.5V's FB − official uses the released official run (88/200). Within an
episode the predicted centre has a median RMS spread of 32 px (XR-1), 11 px (Awomo-0.5I) and 12 px (Awomo-0.5V).
The labelled target spreads 59 px within a training demonstration. Logs are in
`results/eval_fb/<model>-FB-*/results.jsonl`, with the per-query predicted centres in `location_trace`; the
analysis is `code/analysis/feedback.py`, which writes `code/analysis/out/feedback.json`. In these shards'
`meta.json`, the `inputs` string is a static default of the evaluation runner and wrongly reads "no feedback".
The `condition` field (`FB`) and the location traces are authoritative. An earlier Awomo-0.5V run (73 episodes)
read the location head from the wrong feature; it was discarded without analysis and is not included.

## Layout

```
PREREG.md                  pre-registration (one venue line withheld, see above)
code/ann/                  target definitions, frame extraction, Grounding DINO + SAM 2.1 proposals,
                           Qwen3-VL-4B set-of-mark instance selection, panel builder, audit sheets
code/train/                decoded-frame window loader (exactness-checked), prefetching trainer wrapper
code/eval/                 closed-loop RoboCasa365 evaluation runner, training->eval chain daemon,
                           feedback-mode worker and adapter
code/analysis/             pre-registered analysis (analyze.py); secondary, per-task, sensitivity, crash,
                           head-diagnostic and feedback analyses; table and macro generators;
                           out/ holds their JSON outputs
code/figs/                 figure scripts
results/eval/              2,400 primary episode records (JSON lines) + per-shard meta
results/eval_fb/           600 feedback-study episode records with per-query location traces
results/eval_crashed/      the three original Awomo-0.5V official shards, including the 18 crash rows
results/eval_unreleased/   the nine duplicate episodes that are not part of the 2,400
results/summary.json       primary analysis output; results/tables.md human-readable tables
results/secondary.json     secondary contrasts (official checkpoint, seen/unseen splits)
results/training_logs/     per-microbatch loss terms of the nine LoRA runs
labels/                    label panel (panel-train.json), its statistics, the 48-window audit sheets and the
                           per-view audit decisions
figures/                   overview, label examples, main result, per-task heatmap, rollouts, training curves
paper/                     preprint, supplementary material, LaTeX source
```

The analysis scripts compute every number in the paper from these logs, the training logs and the label files.
The exceptions are the label-audit counts, the training wall-clock times and per-task window coverage (read from the
training logs), and the number of discarded feedback episodes, the number of queued windows and the
trainable-parameter counts, which come from cluster run logs. The scripts were run in the study's
working layout (analysis/, results/ and train/runs/ side by side, with the label panel at results/panel-train.json
and the primary summary at results/analysis/summary.json); to rerun them here, point their root paths at the
corresponding folders of this directory. The training framework these scripts plug
into (LeRobot 0.7 + native XR-1/Awomo workers, LoRA `training_core.py`, `train_prepared.py`) and the frozen
100 + 25 demonstration panel are documented in the `lerobot-training-ready-20261106` preparation package. Model
weights are the public RoboCasa365 checkpoints.

## Labels

Target boxes are automatic pseudo-labels. 198 task-specific stage descriptions are mapped to a detector class and
an instance hint (`code/ann/targets.py`). Grounding DINO (base) proposes boxes, SAM 2.1 tightens them to masks,
and Qwen3-VL-4B picks the described instance per camera or says "not visible". 8,174 of 8,931 training windows
have a visible target in at least one camera and form the shared panel. An audit of 48 random windows (103 boxes) was done
twice. The first pass, before any training result, counted 10 wrong boxes; the second, view by view after the
experiments and checked independently, found 26 clearly wrong (12 on a robot link or the gripper, 8 on the wrong
instance among duplicates, 3 on another object, 2 on a chair, 1 on the carried pan instead of the destination
burner) and one more probably wrong, so 76 to 77 (74–75%) bound the described target, 10 of them only coarsely (the
whole toaster, stove top or drawer front instead of the named part). Sheets: `labels/audit-sheets/`; per-view decisions of the second pass:
`labels/label_audit_second_pass.json`. The panel itself (8,174 windows with stage index,
stage description and per-camera target box and centre) is `labels/panel-train.json`.

## What is and is not claimed

- Claimed:
  - Within this budget, neither signal measurably adds to the other in any of the three models.
  - The stage-only arm is never significantly below the official checkpoint.
  - Every nominally significant loss against the official checkpoints is in an arm that drew boxes during
    training.
  - Feeding the policy's own predicted target back as a box helps no model.
- Not claimed:
  - That XR-1's gain or the Awomo losses are caused by either signal; there is no matched action-only arm, by
    design.
  - That the effects hold at other budgets, adapter placements, loss weights or training seeds; there is one
    training seed per arm.
  - Anything beyond the ten screened tasks or beyond simulation.
