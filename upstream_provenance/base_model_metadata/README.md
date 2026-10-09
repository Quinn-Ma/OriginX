---
license: apache-2.0
library_name: transformers
pipeline_tag: robotics
tags:
- robotics
- vision-language-action
- robocasa365
- qwen3-vl
- custom_code
- vla
---

# Xiaomi-Robotics-1-RoboCasa365

This repository contains the Hugging Face checkpoint used by
Xiaomi-Robotics-1 for RoboCasa365 evaluation. It includes the model weights,
custom Transformers model and processor code, tokenizer files, and
RoboCasa365 action normalization statistics.

## Requirements

The reference environment uses:

```text
Python 3.11
PyTorch 2.8.0
Transformers 4.57.1
FlashAttention 2
```

The custom model and processor must be loaded with `trust_remote_code=True`.

## Loading

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False,
)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
).cuda()
```

For reproducible multi-GPU evaluation, use the standard eight-server launcher in
the companion source repository.

## RoboCasa365 Evaluation

Use the companion Xiaomi-Robotics-1 source repository and follow
`eval_robocasa365/README.md`. The server launcher accepts either this Hugging
Face repository ID after downloading it locally or a local snapshot path.

Reference configuration:

```text
split: pretrain
task set: target50
tasks: 50
episodes per task: 50
observation history: 4
observation interval: 2
actions per query: 16
environment base seed: 7
crop ratio: 0.95
```

Reference result:

| Metric | Value |
| --- | ---: |
| Episodes | 2500 |
| Successes | 1432 |
| Episode success rate | 57.28% |

## License

Apache License 2.0. See `LICENSE`.