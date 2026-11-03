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

## Later diagnostics and separate memory studies

The final manuscript retains this 6,000-update result and adds its monitoring diagnostic: action loss on 43 fixed monitoring windows rose overall from **0.05133 at update 200 to 0.08169 at update 6,000 (+59.1%)**. The curve is not monotonic. These are raw-weight measurements; rollouts used the final EMA. The monitoring demonstrations were excluded from this additional training but were part of the starting policy’s original training data. There is no equally fine-tuned memory-free control, so neither this curve nor the success comparison identifies a causal explanation. Small readout weights do not establish zero functional memory use.

Two additional experiments remain separate:

- **[One-query target cache](../memory_cache/):** reads disabled **19/40**, cache enabled **14/40**; paired change **−12.5 points**, 95% CI **[−27.5, +2.5]**. This controller uses an alignment gate and its control is not the original PS arm.
- **[Revised learned-memory study](../memory_redo/):** original policy **118/180**, selected LoRA memory **103/180**, same-weight state reset with first-frame anchor **107/180**, joint state/anchor reset **113/180**. The primary history effect is **−2.2 points**, 95% CI **[−8.9, +4.4]**, with Holm-adjusted exact McNemar **p=0.783**. The evaluated checkpoint is validation-selected LoRA at update **100**, distinct from this directory’s final 6,000-update full-weight model.

The follow-up selected checkpoint has near-identical validation action loss with reads enabled and zeroed (0.01490071 vs 0.01489845); this describes aggregate loss sensitivity, not proof of unused memory in every action or checkpoint. It does not establish an overall memory gain. Counts and intervals across these different experiments must not be pooled. Complete training amendments, all task-level increases and decreases, and the limits of reset interventions are reported in the final supplementary material.
