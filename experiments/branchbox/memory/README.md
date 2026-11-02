# BranchBox: full-weight memory development study

We adapt [MemBodied](https://arxiv.org/abs/2609.28256) to the existing
BranchBox-trained X-VLA-P, retaining associative reads, dedicated state/memory
tokens, an episode-first anchor and execution-conditioned writes. The full policy
receives 6,000 additional updates with master seed 1. This is an attributed X-VLA
adaptation, not a new memory architecture or the official reference benchmark run.

| Condition | Successes | Success rate |
|---|---:|---:|
| Existing BranchBox-trained X-VLA-P | 13/20 | 65% |
| Final memory EMA | 12/20 | 60% |
| Same weights, added-memory reset | 11/20 | 55% |

Memory minus existing policy is **-5 percentage points** (pointwise 95% CI
[-30, +20]); memory minus reset is **+5 points** (CI [-15, +25]). Both two-sided
exact McNemar tests have Holm-adjusted p=1.0. This panel does not establish an
overall improvement. All four tasks and all 20 paired scene outcomes are retained.

The 20 scenes (four tasks, official initial-state IDs 40–44) appeared in earlier
pilots. Estimates condition on one trained checkpoint and this development panel.
There is no equally fine-tuned memory-free control. Reset changes both added
recurrent and anchor history at test time; the original stage/target prompt
controller remains active in all three conditions. This experiment is separate
from the historical two-model +24.7% relative result.

## Files

- [Mechanism code, tests and licenses](core/README.md): the public X-VLA port.
- [All paired outcomes](analysis/paired_outcomes.csv) and [statistics](analysis/analysis.json).
- [Reproduction instructions](analysis/README.txt): recompute the complete panel.

```bash
python analysis/reproduce_memory.py --reference analysis/analysis.json
```

The statistics reproduce exactly from the released CSV. They do not rerun the
simulator or retrain the policy. This public source release contains the model
mechanism, not checkpoint weights, training videos or private orchestration.
See the paper supplement for the complete training recipe, backbone mapping,
single-seed limitations and retained 6,000-update recovery lineage.
