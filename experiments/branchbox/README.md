# BranchBox: focused paired analysis

Qinzhen Ma*, Harry Yang*, Changsheng Lu*, Xiaogang Xu. *Equal contribution.

Research preprint, November 2026. [Project page](https://robocasa.originxairobotics.com/branchbox/).

The revised manuscript reports a retrospectively selected four-task, two-model
analysis: 91/160 successes versus 73/160, +11.25 percentage points (rounded +11.3),
and a relative success increase of 24.6575% (rounded 24.7%). Both models evaluate
the same 80 task–initial-state scenes. The unadjusted 95% scene-clustered paired
interval is [1.875, 20.625] percentage points.

## Reproduce

With Python and NumPy installed, run:

```sh
python reproduce.py
```

The script validates 720 binary outcomes covering two models, all six evaluated
LIBERO tasks, three deployment conditions, and 20 initial states per task. It
then selects task IDs 0,3,4,5 (official libero_10 IDs 8,4,6,0). Each task's sampled
scene indices are shared across both models and all arms. The generated
`reproduced_statistics.json` matches `scene_clustered_statistics.json`.

N0 is the matched action-only baseline; P0 and PS are the same BranchBox-trained
checkpoint with ordinary inputs and self-generated prompts, respectively. All
arms share the added image slot and head architecture. These intervals are
conditional on the trained checkpoints and fixed tasks; they do not account
for the retrospective selection or variation across training seeds. They are
not a whole-benchmark performance claim. Complete broader results and method
details are included in the supplementary PDF.

This directory contains an analysis snapshot, not trained model weights or a
new evaluation. No new training or robot evaluation was performed in this
revision. It is separate from the repository's RoboCasa365 study.
