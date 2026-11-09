# Pre-registration: Stage / Box+Location / Full adaptation of frozen RoboCasa365 policies

Frozen on 2026-11-08 (UTC) before any training run and before any scored evaluation episode of this study.
Executor: Claude (Fable 5.1) on behalf of Qinzhen Ma; cluster hkust-cluster (8x H100 PCIe, GPUs 2/3 faulted and unused).

## Question
Does lightweight auxiliary supervision about *where the policy is in the task* (stage prediction) and *what it should
attend to* (bounding-box prompts + target-location prediction) improve a pretrained generalist manipulation policy on
long-horizon RoboCasa365 tasks, and which component carries the effect?

## Models (official RoboCasa365 checkpoints, pinned in lerobot-three-model-adapters-20261106/receipts/weights-index.json)
- Xiaomi Robotics 1 (XR1)
- Awomo-0.5I
- Awomo-0.5V  (Awomo-I/V are related variants: two model families, three models)

## Tasks (10, fixed; selected 2026-11-05 from published per-task success in [20%, 60%] for all three models)
- composite-seen (training demos available): DeliverStraw, GetToastedBread, PackIdenticalLunches, SearingMeat, StirVegetables
- composite-unseen (never used for training or tuning): ArrangeTea, CuttingToolSelection, RecycleBottlesByType, WaffleReheat, WeighIngredients

## Training conditions (per model, identical everything except the switches)
| id | stage loss | box prompts in training input | location loss |
|----|-----------|-------------------------------|---------------|
| S  | on        | off                           | off           |
| BL | off       | on                            | on            |
| F  | on        | on                            | on            |
Shared: same official checkpoint init; frozen encoders/base weights; rank-8 LoRA on the same four late DiT projections
plus stage head (16 classes) and location head (3 cameras x 2); identical demonstrations (20 train episodes per seen
task = 100, SHA256-ranked, fixed on 2026-11-06), identical annotated window panel, identical sampling order (seed
20261106), microbatch 1 x accumulation 8, AdamW lr 1e-4 wd 0.01, identical optimizer update count within a model
(set from a timing calibration *before* any run; the same count is then used for S, BL and F of that model).
Loss = action flow-matching MSE (+ Awomo future-latent MSE) + 0.1 stage CE (if on) + 0.1 location smooth-L1 (if on).
No stage text is added to the policy instruction in any condition. The box is a 2-px red rectangle drawn on the current
frame only; history and future-prediction targets stay clean. No action-only retrained baseline (user decision).

## Annotations
Automatic pseudo-labels: Grounding DINO (base) proposals -> SAM 2.1 (hiera-small) tight mask boxes -> Qwen3-VL-4B-Instruct
set-of-mark selection of the described target instance per camera (or "not visible"). No human review. Windows whose
target is not visible in any camera (and the 'task complete' stage) are excluded from the panel for ALL arms.
A random audit sample of the final labels is inspected visually and its hit rate reported.

## Evaluation (primary)
- Ordinary inputs for every condition: original task instruction, clean RGB, state. No feedback of predicted heads.
- Conditions evaluated per model: official checkpoint (reference), S, BL, F -> 12 policies.
- 10 tasks x 20 fixed seeds (0-19) per policy = 200 episodes per policy, 2,400 total. Same seed => same initial scene
  (checked by first-frame hash); official per-task horizon and official success function; policy query every 16 env
  steps (native chunking); policy RNG seed 4242 + replan index.
- Primary endpoints (per model, paired by (task, seed), pooled over the 10 tasks):
  (a) F - BL  : contribution of stage supervision given box+location,
  (b) F - S   : contribution of box+location given stage.
  Secondary: F - official (total short-budget adaptation, confounded with extra action training), S - official,
  BL - official; seen vs unseen subsets; per-task tables.
- Statistics: paired success-rate difference with 95% CI from a stratified (by task) paired bootstrap (10,000 draws),
  exact two-sided McNemar/sign test on discordant pairs, Holm correction over the two primary contrasts per model.
  A contrast counts as "supported" only if the Holm-corrected interval excludes 0 and the point estimate is >= +5 pp.
- All episodes are reported (success / failure / error / timeout). No task, seed or checkpoint is replaced after seeing
  results. Only the final checkpoint of each run is evaluated.

## Optional secondary study (only if time permits, reported separately)
Same F checkpoint with self-generated stage/box feedback vs ordinary inputs; not part of the primary claims.

## Compute window
Training and evaluation on hkust-cluster GPUs 0,1,4,5,6,7 (user authorized exclusive use on 2026-11-08).
[One administrative scheduling line is withheld in this public copy; the SHA-256 in the README refers to the complete file.]
