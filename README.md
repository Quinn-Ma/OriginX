# OriginX

*Continuous Conditioning of a Frozen Robot Policy for RoboCasa365*

Qinzhen Ma · Harry Yang · Jialin Wu · Shichen Tang · Gordon Dai

OriginX adds a compact continuous conditioning branch to an Action LoRA adaptation of [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365). The branch uses visual-language features and robot state to modulate action generation while the adapted policy remains frozen.

[Paper](paper/OriginX_arXiv_manuscript.pdf) · [TeX source](paper/OriginX_arXiv_source.zip) · [Model weights](https://huggingface.co/Qinzhen3/OriginX) · [Project website](https://robocasa.originxairobotics.com) · [Reproduction guide](REPRODUCIBILITY.md)

## Method

Training has two stages:

1. **Action adaptation.** An Action LoRA adapter adapts the Xiaomi policy with rank 16 and alpha 16, using 80,000 sampled windows over 1,613 updates. A2000 is the public name of this checkpoint; the training records identify it as A1613.
2. **Continuous conditioning.** The adapted policy is frozen, and a 4,216,839-parameter branch is trained for 2,000 updates with global batch 128, totaling 256,000 sampled windows. The branch conditions action generation on the observed scene and proprioceptive history. Training includes frozen-teacher velocity retention and auxiliary stage classification.

B2000 is the evaluated checkpoint. The [English technical report](TECHNICAL_REPORT.md) and [Chinese article](TECHNICAL_REPORT.zh-CN.md) describe the architecture, training recipe, and experiments.

## RoboCasa365 evaluation

OriginX achieves **59.84% success** in our evaluation of **50 tasks × 50 episodes**, using one frozen B2000 checkpoint and native resets.

| Split | Successes / episodes | Success rate |
|---|---:|---:|
| Atomic-Seen | 747 / 900 | 83.00% |
| Composite-Seen | 479 / 800 | 59.875% |
| Composite-Unseen | 270 / 800 | 33.75% |
| Overall | 1,496 / 2,500 | 59.84% |

The evaluation uses pretrain scenes, registered task horizons, and seeds 2090900000–2090902499. All 2,500 episodes have known outcomes, including all 1,004 failures. The rendering setup uses CPU OSMesa and the readback safeguards described in the reproduction guide. Configurations, reset checks, and individual results are available in the [evaluation release](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0) and [evidence directory](evidence/).

The result was submitted for organizer review through [leaderboard PR #26](https://github.com/robocasa-benchmark/leaderboard/pull/26); it is an author-run score, not an official ranking. An earlier 600-pair comparison found a one-percentage-point difference with a 95% interval of [−2.17, 4.17] points and McNemar p=0.6173. A matched 2,500-episode base-policy comparison and conditioning-branch ablation under this native-reset protocol remain necessary to establish the contribution of each stage.

## Astra-assisted recovery

We tested Astra High (`gpt-6-astra`, high reasoning effort) on the fixed **1,004 B2000 failure cases**. It receives the task instruction and current camera views, proposes a short subgoal, and passes that instruction to the frozen robot policy. Each case has an unassisted control and an assisted replay.

The study confirmed **56 rescues (5.58%)**: 813 cases were not rescued, 28 remain unknown, and 107 had initial-state deviations. A confirmed rescue requires matching initial states and pre-intervention action prefixes, a failed control, a successful assisted replay, and evidence that the Astra response was applied. These results are conditional on the original failures and are reported separately from the 59.84% benchmark score. See the [recovery study](ASTRA_RESCUE_REPORT.md) and [per-case records](evidence/astra-rescue/).

## Fresh-seed follow-up

A supplementary study covers **50 fresh-seed environment cases and 350 arm outcomes**. B2000 control and visual-assistance assignment both succeed on **29/50** cases. Among 44 matched V-versus-C pairs, there is **1 rescue and 0 regressions**; the other 6 pairs have initial-state deviations. Full-cohort net-effect bounds are **−8 to +4 percentage points**, so the study does not establish an overall benefit.

The visual-versus-text comparison includes a fallback success and cannot be interpreted as two visual rescues. The original Xiaomi policy has 0 rescues and 1 regression among 47 matched pairs. Formal GR00T evaluation was not run. The cohort was reduced after launch to the smallest pre-existing fresh seed per task, without selecting on outcomes. The [paper and appendices](paper/OriginX_arXiv_manuscript.pdf) and [study records](evidence/confirmatory-50/) report matching, advice delivery, costs, and limitations.

## Use the model

The [Hugging Face repository](https://huggingface.co/Qinzhen3/OriginX) contains the complete model assets: the original base in three safetensors shards, tokenizer and configuration files, `adapter-originx-2000.pt`, and `branch-00002000.pt`. The complete snapshot is approximately 10.15 GB. Inference code and the pinned upstream model Python files are hosted here.

Clone this repository and run:

```bash
python -m pip install huggingface_hub
python prepare_originx.py --revision main --output-dir assets
```

For reproducibility, replace `main` with a Hugging Face commit SHA. The helper prepares `assets/base/` and `assets/weights/` and verifies the base and adaptation weight hashes. See [portable inference](PORTABLE_INFERENCE.md) for serving and offline use, [SHA256SUMS](SHA256SUMS) for weight hashes, and the [asset manifest](base-model-assets.json) for the full file inventory.

## Reproducibility and license

The [reproduction guide](REPRODUCIBILITY.md) documents the evaluated entry point, portable inference entry point, source revisions, and artifact checks. Historical [analysis scripts and execution-contract templates](https://github.com/Quinn-Ma/OriginX/tree/1c97321fbf568f8816faee014c7717d9cdbc4d30/paper/revision-v3-20261009/) remain available in Git history. The project website source is in [website/](website/).

The Xiaomi base checkpoint and upstream code retain their Apache-2.0 license and attribution. Original code in this repository is also provided under Apache-2.0. Third-party snapshots retain their notices; see [NOTICE](NOTICE). Dataset assets are not redistributed.
