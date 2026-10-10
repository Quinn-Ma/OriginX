# OriginX

*Continuous Conditioning of a Frozen Robot Policy for RoboCasa365*

Qinzhen Ma · Harry Yang · Jialin Wu · Shichen Tang · Gordon Dai

[Paper](paper/OriginX_arXiv_manuscript.pdf) · [Models](https://huggingface.co/Qinzhen3/OriginX) · [Website](https://robocasa.originxairobotics.com) · [Reproduce](REPRODUCIBILITY.md) · [TeX](paper/OriginX_arXiv_source.zip)

OriginX adapts [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365) with Action LoRA, then freezes the policy and trains a **4.22M-parameter conditioning branch**. Visual-language features and robot state guide action generation.

- **RoboCasa365:** 1,496/2,500 successes (**59.84%**) across 50 tasks in our evaluation. [Submitted for organizer review](https://github.com/robocasa-benchmark/leaderboard/pull/26); no official ranking confirmed.
- **Astra-assisted recovery:** 56 confirmed rescues among 1,004 historical failures, reported separately from the benchmark. [Study](ASTRA_RESCUE_REPORT.md).
- **50-case follow-up:** control and visual assistance both scored 29/50; no overall benefit established. [Results](evidence/confirmatory-50/).

```bash
pip install huggingface_hub
python prepare_originx.py --revision main --output-dir assets
```

Use a pinned Hugging Face revision for reproducibility. See [inference instructions](PORTABLE_INFERENCE.md), the [technical report](TECHNICAL_REPORT.md), and [evaluation evidence](evidence/) for setup and experimental details.

[Apache-2.0](LICENSE) · [Third-party notices](NOTICE)
