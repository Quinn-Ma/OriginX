# OriginX

[Model weights on Hugging Face](https://huggingface.co/Qinzhen3/OriginX) · [Project website](https://originx.thumbnodirosving.chatgpt.site) · [Evaluation evidence](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0) · [Reproduction guide](REPRODUCIBILITY.md) · [Technical report](TECHNICAL_REPORT.md)

**Naming.** OriginX is the public model name. The unchanged internal experiment and checkpoint identifier is B2000; earlier publication drafts used the name XR1-Continuous-B2000. Renaming does not change any weights, experiment records, or provenance hashes. A2000 is the public release alias of historical A1613: the adapter weights are unchanged and were trained for 1,613 actual updates.

A continuous conditioning branch on top of an Action LoRA adaptation of [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365). Author: Qinzhen Ma. This is an independent derivative.

**Author-run RoboCasa365 result: 59.84% across all 2,500 episodes.** [Official submission PR #26](https://github.com/robocasa-benchmark/leaderboard/pull/26) is open for organizer review; it is not an accepted leaderboard rank. The available earlier paired comparison did not establish a statistically reliable improvement from our changes.

| Split | Successes / episodes | Success rate |
|---|---:|---:|
| Atomic-Seen | 747 / 900 | 83.00% |
| Composite-Seen | 479 / 800 | 59.875% |
| Composite-Unseen | 270 / 800 | 33.75% |
| Overall | 1,496 / 2,500 | 59.84% |

**Read the experiment.** The [English technical report](TECHNICAL_REPORT.md) explains the inherited training chain, evaluation protocol, data audit, earlier negative result, and limitations. The [Chinese article](TECHNICAL_REPORT.zh-CN.md) presents the same experiment. The complete model assets are distributed through [Qinzhen3/OriginX on Hugging Face](https://huggingface.co/Qinzhen3/OriginX): the original base in three safetensors shards, tokenizer/configuration assets, `adapter-originx-2000.pt`, and `branch-00002000.pt`. The three base shards total 10,106,433,336 bytes; the complete snapshot is approximately 10.15 GB. Implementation code is distributed through this GitHub repository, including the three pinned upstream model Python files. Original per-episode evidence remains in the [versioned evaluation release](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0).

**Prepare the complete model.** Clone this repository, then run the hash-verifying preparation helper from its root:

```bash
python -m pip install huggingface_hub
python prepare_originx.py --revision main --output-dir assets
```

For a reproducible download, replace `main` with the recorded Hugging Face commit SHA. The helper assembles `assets/base/` from the Hugging Face assets and the pinned model code in this repository, and places both adaptation weights in `assets/weights/`. It verifies the exact original base identity and both weight hashes. It performs no training or GPU validation. See [PORTABLE_INFERENCE.md](PORTABLE_INFERENCE.md) for offline preparation and serving. [SHA256SUMS](SHA256SUMS) lists the three published base shards and two adaptation weight files; the full asset manifest is [base-model-assets.json](base-model-assets.json).

**Training chain.** The original Xiaomi policy is adapted with Action LoRA (A2000: rank 16, alpha 16, 80,000 sampled windows over 1,613 updates). B2000 freezes that complete policy and trains a 4,216,839-parameter continuous conditioning branch for 2,000 updates, global batch 128, or 256,000 sampled windows. This branch affects action generation at inference time. Its objectives include frozen-teacher velocity retention and auxiliary stage classification. This repository does not claim to have trained the base model from scratch.

**What was evaluated.** One frozen B2000 checkpoint, 50 target tasks, 50 episodes per task, pretrain scenes, registered task horizons, and seeds 2090900000–2090902499. All 1,004 policy failures remain in the denominator. There are no missing, unattempted, or infrastructure-unknown episodes. Reset implementation proofs and individual result hashes are retained. The run uses CPU OSMesa and disclosed rendering/readback safeguards; it is not presented as an unmodified default graphics stack.

**What the result does not show.** A separate earlier 600-pair confirmation found a one-point difference with 95% interval [-2.17, 4.17] percentage points and McNemar p=0.6173. We have not run a matched 2,500-episode original-base baseline or a conditioning-branch ablation under the new native-reset protocol. No statistical superiority or accepted first-place rank is claimed. The C continuation mentioned in historical planning was not trained.

**Inspect and reproduce.** See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for loading the released weights, exact source provenance, artifact checks, and the distinction between the original evaluated entry point and the portable publication entry point. The `evidence/` directory contains the complete aggregate report, final audit, and the original fixed manifest and configuration. Evidence includes original experiment paths for provenance; those paths are not credentials.

**Attribution and license.** The public base checkpoint and upstream Xiaomi code retain their Apache-2.0 license and attribution. Original files in this derivative release are provided under Apache-2.0. Third-party source snapshots retain their own notices; see [NOTICE](NOTICE). The redistributed Xiaomi base assets retain their original identity, Apache-2.0 license, and source credit. Dataset assets are not redistributed.

**Website source.** The public project page is mirrored in [website/](website/). It contains the editorial release article and original desktop/mobile architecture diagrams; it does not change the frozen model artifacts or evaluated release tag.
