Reproducible memory comparison

Requirements: Python 3.10+ and NumPy. This script reproduces statistics from
recorded outcomes; it does not run the robot simulator or load model weights.

Input panel
The final paired_outcomes.csv contains exactly 20 rows and 60 binary outcomes:
four tasks crossed with official initial-state IDs 40, 41, 42, 43 and 44.

Internal task ID    LIBERO-Long task ID    Task
0                   8                      Two moka pots
3                   4                      Two mugs
4                   6                      Mug and pudding
5                   0                      Soup and sauce

CSV columns are task,init,baseline,memory,reset. Outcomes are true/false
(case-insensitive); numeric 0/1 are rejected. Missing or duplicate scenes,
unknown values and malformed rows stop analysis. No partial panel is analyzed.

Run from this directory:
  python reproduce_memory.py
Or supply filenames:
  python reproduce_memory.py --input paired_outcomes.csv --output memory_reproduced.json
To check against the supplied audited reference:
  python reproduce_memory.py --reference analysis.json

The reference check requires exact agreement in counts, paired effects,
intervals, discordant scenes, exact and adjusted p-values, and task summaries.

Statistical definitions
- Every arm has 20 outcomes on the same 20 initial scenes.
- The two fixed contrasts are memory minus baseline and memory minus reset.
- Bootstrap uses NumPy default_rng(seed=1) and 100,000 replicates. Within each
  task, draw five paired scenes with replacement. Share the draws across arms
  and contrasts, then average equally over tasks. Report the 2.5th and 97.5th
  percentiles of the success-rate difference in percentage points.
- A rescue is a scene where memory succeeds and its comparator fails; a
  regression is the reverse. McNemar uses the two-sided exact conditional
  binomial test. Holm adjustment covers the two tests; intervals are unadjusted.
- Relative success change is defined only when comparator success is nonzero.

Interpretation
baseline is the existing BranchBox-trained X-VLA-P checkpoint, distinct from
the original action-only N0 arm. memory is the final policy after full-weight
fine-tuning. reset uses identical final weights, reinitializes the added
associative matrices, clears pending transitions, and replaces the episode-first
anchor with the current frame at each query. The original outer stage filter and
previous-target feedback remain enabled in all arms, using each policy's own
predictions. Reset therefore concerns the added memory paths, rather than all
history in the controller.

The baseline contrast includes additional training and changed architecture.
The same-weight contrast tests the combined use of associative and initial-scene
history; it does not separate the two components. These previously explored
scenes form a repeated development panel. One training seed and this fixed panel
do not establish generalization across new tasks, scenes or training seeds.
Separate trajectory and checkpoint audits establish execution and provenance;
this CSV-only script checks the resulting statistics.

Training record
Evaluation uses only the final 6,000-update EMA. The training trajectory has
800 + 1,200 + 4,000 retained updates, with recorded budget and schedule amendments
made during training before the new rollout evaluation. A save-space failure at
update 1,200 required restoration from the complete 1,000-update checkpoint.
The unsaved 200-update tail was preserved as incident evidence and replayed from
the complete model, Adam, EMA and random states on the original schedule. It is
not counted twice: 6,000 updates are retained, and 6,200 were executed including
the discarded tail. The 800- and 2,000-update ancestors are not evaluated as
alternative final models.

CPU self-checks:
  python -m unittest -v test_reproduce_memory
These use synthetic fixtures in temporary directories. They do not generate
experimental outcomes. SELF_CHECKS.json distinguishes the historical synthetic
checks from the final exact comparison on the real, complete panel.
