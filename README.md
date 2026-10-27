# OriginX

*Continuous Conditioning of a Frozen Robot Policy for RoboCasa365*

Qinzhen Ma · Harry Yang · Jialin Wu · Shichen Tang · Gordon Dai

[📄 Paper](paper/OriginX_arXiv_manuscript.pdf) · [🤗 Models](https://huggingface.co/Qinzhen3/OriginX) · [🌐 Website](https://robocasa.originxairobotics.com) · [🔁 Reproduce](REPRODUCIBILITY.md) · [📦 TeX](paper/OriginX_arXiv_source.zip)

**59.84% in a complete 2,500-episode author-run evaluation. [Paper](paper/OriginX_arXiv_manuscript.pdf) · [Evaluation evidence](evidence/).**

OriginX adapts [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365) with Action LoRA, then freezes the policy and trains a **4.22M-parameter conditioning branch**. Visual-language features and robot state guide action generation.

**RoboCasa365 success rate (%)**

| Model | Atomic-Seen | Composite-Seen | Composite-Unseen | Overall |
|---|---:|---:|---:|---:|
| **OriginX (ours)** | **83.00** | **59.875** | 33.75 | **59.84** |
| [Xiaomi-Robotics-1](https://robocasa.ai/leaderboard.html) | 80.2 | 57.1 | 32.1 | 57.4 |

Xiaomi: published leaderboard result. OriginX: separate author-run evaluation with auxiliary subtask-stage supervision. The [current benchmark rules](https://robocasa.ai/docs/build/html/use_cases/subtask_annotations.html) exclude those annotations from standard-data leaderboard training. These numbers do not establish comparable supervision, a matched improvement, or an official rank. [Training details](TECHNICAL_REPORT.md).

- **Astra-assisted recovery:** 56 confirmed rescues among 1,004 historical failures, reported separately from the benchmark. [Study](ASTRA_RESCUE_REPORT.md).
- **50-case follow-up:** control and visual assistance both scored 29/50; no overall benefit established. [Results](evidence/confirmatory-50/).
- **Fresh-seed policy study:** X/A/B/BV scored 30/32/33/32 out of 50; matched assistance had one rescue and one regression. The partial GR00T extension retains 19 unknown cases per arm. [Study](FRESH_SEED_STUDY.md).
- **Cue-precision studies:** fine-region cues vs point cues scored 6/10 vs 6/10 under the registered rule (point 7/11 vs fine-region 6/11 with a disclosed rerun); no benefit established. Mask cues: pixel fidelity verified on an H800, task outcomes not measured (compute exhausted). [Studies](experiments/cue-studies/) · [CVPR 2027 submission (anonymous PDF)](paper/OriginX_Masks_CVPR2027_submission_v1.pdf)

```bash
pip install huggingface_hub
python prepare_originx.py --revision main --output-dir assets
```

Use a pinned Hugging Face revision for reproducibility. See [inference instructions](PORTABLE_INFERENCE.md), the [technical report](TECHNICAL_REPORT.md), and [evaluation evidence](evidence/) for setup and experimental details.

[Apache-2.0](LICENSE) · [Third-party notices](NOTICE)
