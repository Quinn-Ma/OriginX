Bounded target reuse pilot: anonymous outcome-level reproduction

Files
  paired_outcomes.csv: all 40 paired scenes (80 evaluated episodes).
  statistics.json: aggregate and per-task counts, paired confidence intervals,
    exact two-sided McNemar tests and the prespecified screening decision.
  reproduce_memory_cache.py: standalone statistical reproduction using NumPy.
  provenance.json: source hashes and design, with private paths removed.
  MANIFEST.json: SHA-256 hashes of these release files.

Run
  python reproduce_memory_cache.py
Requires Python 3 and NumPy. No model, simulator, GPU or checkpoint is needed.
The script verifies all 40 unique expected scenes and binary outcomes, then
checks its full numerical output against statistics.json. It does not write
files or perform additional experiments.

Design
  One frozen X-VLA-P checkpoint; one evaluation seed, 1.
  Four tasks (internal IDs 0,3,4,5); initial states 20 through29 for each.
  Each paired scene compares off and hold1 with fresh environments.
  off: omit a visual mark when the latest raw-stage prediction does not align
    with the updated filtered stage.
  hold1: under the same alignment rule, allow one extra use of a still-valid
    same-stage cached target. At most60 environment steps old with30 actions
    executed per query. Stage transitions invalidate the cached target.
  Both arms share the alignment gate and stage filter. The original BranchBox
  PS controller does not use this alignment gate; these are distinct controls.
  The four task IDs map to official LIBERO-Long task IDs8,4,6,0.

Statistics
  Paired percentile bootstrap,100000 replicates,NumPy default_rng seed1.
  Ten paired initial states are resampled within each of four fixed tasks;
  task means receive equal weights. Each per-task interval resets the RNG.
  Intervals are conditional on the fixed checkpoint,task set and one seed.
  The screening rule requires net>=4 successes and>=3 nondecreasing tasks
  after all40 pairs and audits pass. It is a resource-screening decision,
  not a confirmatory significance threshold.

Result and scope
  off19/40 (47.5%);hold1 14/40 (35.0%);difference-12.5 percentage points,
  95% interval[-27.5,+2.5];3 rescues,8 regressions;exact p=.2265625.
  Screening failed, so the conditional SmolVLA stage was not run.
  There is no demonstrated improvement for this bounded-target-cache design.
  The confidence interval includes zero; these data do not establish that
  memory in general is harmful. Existing scenes were reused and are not a
  fresh test set. Do not pool these observations with the original BranchBox
  main study or with separately trained memory experiments.

This is an outcome-level statistical reproduction, not the full policy or
simulator implementation. Protocol chronology and mechanism details are
documented in the supplementary manuscript.
