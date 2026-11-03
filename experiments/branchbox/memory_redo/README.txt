Learned-memory follow-up: anonymized outcome-level reproduction

Run: python reproduce.py
Optional: python reproduce.py --output reproduced.json
Dependencies: Python3 and NumPy; no model, simulator, training libraries or GPU.

paired_outcomes.csv contains180matched scenes, each with four binary outcomes.
Task IDs0-5 correspond respectively to: two moka pots on the stove; mug in the
microwave and close it; bowl in the lower drawer and close it; two mugs on two
plates; mug on the plate and pudding to its right; soup and sauce in a basket.
Each task uses official initial-state IDs20-49. Training and rollout master seed
is1. These scenes include previously used development states, not a new held-out
test set. Every scene is present in all four conditions; no errors or exclusions.

Conditions:
baseline: original X-VLA-P checkpoint, no added memory module or episode anchor.
memory: selected LoRA checkpoint, recurrent state retained, first-frame anchor.
sreset: same checkpoint, recurrent state restored to its learned initialization
at every action query, first-frame anchor preserved.
reset: same checkpoint, recurrent state reset and anchor rebuilt from current
frame at every action query.
The forward-only external stage/target controller is retained in every arm.

Primary contrast: memory minus sreset, holding weights and episode anchor fixed.
All five reported contrasts use pooled paired differences.10,000percentile
bootstrap draws (NumPy default_rng seed1) resample30scenes within each of6tasks;
the indices are shared across arms/contrasts. These CIs are pointwise. Exact
two-sided McNemar tests use the discordant pairs; Holm corrects the family of
five tests.95%CI excluding zero and exact-test p<.05 need not coincide. In
particular memory-baseline has a nominal negative bootstrap interval but its
exact p=.0627 and Holm p=.3137. No corrected test rejects equality.

The final evaluated model is LoRA run a at update100, selected for lowest
memory-enabled validation action loss among23saved LoRA EMA checkpoints.
Frozen-backbone and full-finetuning attempts were discontinued without rollout
evaluation. Strategy changes and early stopping were documented before rollout
evaluation, after inspecting training/validation diagnostics. This is distinct
from the earlier6,000-update full-finetuning study; results must not be pooled.
The43validation demonstrations were excluded from the added training but were
part of the original policy's203demonstration training set.

Interpretation: this selected recipe/checkpoint did not establish a benefit
from retaining history. The checkpoint's aggregate validation loss was almost
unchanged when reads were disabled. This does not establish equality of all
predicted actions, general ineffectiveness of memory, or a particular cause.
The baseline comparison also changes adapters,newtokens,anchor,andpromptheads.
State reset is a test-time intervention; elementwise read dropout during
training does not guarantee it is in-distribution.

analysis.json is an anonymized canonical aggregate derived from the original
episode records. reproduce.py calculates all statistics from CSV before
comparing with that file. provenance.json gives original source hashes and
package hashes. No raw prompts, author names, host names or private paths are
included. This package reproduces outcome-level statistics, not model training.
