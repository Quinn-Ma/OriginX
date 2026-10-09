# XR1-Continuous-B2000

A continuous conditioning branch on top of an Action LoRA adaptation of [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365). Author: Qinzhen Ma. This is an independent derivative and is not affiliated with or endorsed by Xiaomi.

**Author-run RoboCasa365 result: 59.84% across all 2,500 episodes.** This result is prepared for organizer review; it is not an accepted leaderboard rank. The available earlier paired comparison did not establish a statistically reliable improvement from our changes.

| Split | Successes / episodes | Success rate |
|---|---:|---:|
| Atomic-Seen | 747 / 900 | 83.00% |
| Composite-Seen | 479 / 800 | 59.875% |
| Composite-Unseen | 270 / 800 | 33.75% |
| Overall | 1,496 / 2,500 | 59.84% |

**Read the experiment.** The [English technical report](TECHNICAL_REPORT.md) explains the inherited training chain, evaluation protocol, data audit, earlier negative result, and limitations. The [Chinese article](TECHNICAL_REPORT.zh-CN.md) presents the same experiment. The [release](https://github.com/Quinn-Ma/xr1-continuous-b2000/releases/tag/v1.0.0) contains the two adaptation weight files and the original per-episode evidence. The base model must be downloaded from Xiaomi's original model repository; it is not repackaged here.

**Training chain.** The original Xiaomi policy is adapted with Action LoRA (A1613: rank 16, alpha 16, 80,000 sampled windows over 1,613 updates). B2000 freezes that complete policy and trains a 4,216,839-parameter continuous conditioning branch for 2,000 updates, global batch 128, or 256,000 sampled windows. This branch affects action generation at inference time. Its objectives include frozen-teacher velocity retention and auxiliary stage classification. This repository does not claim to have trained the base model from scratch.

**What was evaluated.** One frozen B2000 checkpoint, 50 target tasks, 50 episodes per task, pretrain scenes, registered task horizons, and seeds 2090900000–2090902499. All 1,004 policy failures remain in the denominator. There are no missing, unattempted, or infrastructure-unknown episodes. Reset implementation proofs and individual result hashes are retained. The run uses CPU OSMesa and disclosed rendering/readback safeguards; it is not presented as an unmodified default graphics stack.

**What the result does not show.** A separate earlier 600-pair confirmation found a one-point difference with 95% interval [-2.17, 4.17] percentage points and McNemar p=0.6173. We have not run a matched 2,500-episode original-base baseline or a conditioning-branch ablation under the new native-reset protocol. No statistical superiority or accepted first-place rank is claimed. The C continuation mentioned in historical planning was not trained.

**Inspect and reproduce.** See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for loading the released weights, exact source provenance, artifact checks, and the distinction between the original evaluated entry point and the portable publication entry point. The `evidence/` directory contains the complete aggregate report, final audit, and the original fixed manifest and configuration. Evidence includes original experiment paths for provenance; those paths are not credentials.

**Attribution and license.** The public base checkpoint and upstream Xiaomi code retain their Apache-2.0 license and attribution. Original files in this derivative release are provided under Apache-2.0. Third-party source snapshots retain their own notices; see [NOTICE](NOTICE). Dataset assets and the Xiaomi base checkpoint are not redistributed.
