# PaTH-VLA: Progress-and-Target Context for Long-Horizon Robot Policy Adaptation

A controlled, pre-registered ablation of two cheap auxiliary context signals for pretrained generalist
manipulation policies on RoboCasa365: **progress** (predict the current subtask stage) and **target**
(see a box around the current target and predict its location). Three official RoboCasa365 checkpoints
(Xiaomi Robotics 1 / XR-1, Awomo-0.5I, Awomo-0.5V) are adapted with rank-8 LoRA for 1,000 updates in
three arms (S = progress only, BL = target only, F = both, the full PaTH-VLA recipe) and evaluated with
ordinary inputs on the same 200 seeded episodes (10 tasks x 20 seeds) per policy. 2,400 episodes in total.

Study date: 2026-11-08. Pre-registration: [`PREREG.md`](PREREG.md) (SHA-256 6e1d9523…, frozen before the
first optimizer update and the first scored episode).

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
| Awomo-0.5V | −3.5 pp [−11.0, +4.0], p = 0.46 | −8.5 pp [−15.5, −1.5], p = 0.030 (Holm 0.060) |

No primary contrast meets the pre-registered support rule. Against the official checkpoints XR-1 gains
6.5–7.5 pp in every arm (F: p = 0.049; seen tasks +13 pp, unseen +2 pp); both Awomo models lose, most in the
arms that saw boxes during training (Awomo-0.5I BL −10.0 pp, p = 0.033; Awomo-0.5V F −11.0 pp, p = 0.015).
Full tables: [`results/tables.md`](results/tables.md); per-episode logs: `results/eval/<model>-<arm>-q*/results.jsonl`.

## Layout

```
PREREG.md                 pre-registration (frozen 2026-11-08 before any run)
code/ann/                 target definitions, frame extraction, Grounding DINO + SAM 2.1 proposals,
                          Qwen3-VL-4B set-of-mark instance selection, panel builder, audit sheets
code/train/               decoded-frame window loader (exactness-checked), prefetching trainer wrapper
code/eval/                closed-loop RoboCasa365 evaluation runner, training->eval chain daemon,
                          secondary feedback-mode worker/adapter
code/analysis/            pre-registered analysis (analyze.py), secondary analyses, table/macro generators
code/figs/                figure scripts
results/eval/             2,400 episode records (JSON lines) + per-shard meta
results/summary.json      analysis output; results/tables.md human-readable tables
results/training_logs/    per-microbatch loss terms of the nine LoRA runs
labels/panel-summary.json label-panel statistics; labels/audit-sheets/ the 48-window visual audit
figures/                  main result figure, label examples, rollouts, training curves
paper/                    preprint PDF, supplementary PDF, LaTeX source
```

The training framework these scripts plug into (LeRobot 0.7 + native XR-1/Awomo workers, LoRA
`training_core.py`, `train_prepared.py`) and the frozen 100 + 25 demonstration panel are documented in the
`lerobot-training-ready-20261106` preparation package; model weights are the public RoboCasa365 checkpoints.

## Labels

Target boxes are automatic pseudo-labels: 198 stage descriptions -> detector class + instance hint
(`code/ann/targets.py`); Grounding DINO (base) proposals -> SAM 2.1 tight mask boxes -> Qwen3-VL-4B picks the
described instance per camera or says "not visible". 8,174 of 8,931 training windows have a visible target
in at least one camera and form the shared panel. An audit of 48 random windows (103 boxes) found 90% correct;
errors are mostly robot links and wrong instances among duplicates (`labels/audit-sheets/`).

## What is and is not claimed

- Claimed: within this budget, progress and target supervision are not complementary in any of the three
  models; progress-only adaptation is never significantly below the official checkpoint; the arms that draw
  boxes during training carry every significant loss, on the two world-action models.
- Not claimed: that XR-1's gain is caused by either signal (there is no matched action-only arm, by design);
  that the effects hold at other budgets, adapter placements or training seeds (one training seed per arm).

Secondary study (feedback of the policy's own predicted box at inference) is reported separately when complete.
