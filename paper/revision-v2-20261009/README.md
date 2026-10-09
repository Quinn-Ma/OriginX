# OriginX research manuscript — revision 2

Title: OriginX: Measuring One-Shot Recovery in a Frozen Robot Policy.
Revision date: October 9, 2026.

## Build the PDFs

From this directory, run Tectonic 0.17.0:

```
tectonic --keep-logs main.tex
tectonic --keep-logs supplement.tex
```

The main PDF has 9 pages: its content ends on page 8, followed by references. The supplement has 8 pages. The unmodified official cvpr-org/author-kit style currently identifies its annual revision as CVPR 2026 and is used provisionally for 2027. Final conference-policy and anonymity checks are still required. No acceptance, submission, or spotlight is asserted.

## What changed

The revision separates valid replay coverage, changed action outputs, and actual task recovery. New retrospective analyses check the first action output at intervention for all 869 eligible pairs, the timing of all 56 observed rescues, shared-batch deletion sensitivity, and the full task scatter. Future Directions specifies falsifiable contrasts and endpoints rather than claiming planned results.

Historical outcomes remain 1496/2500. The fixed failure cohort remains 56 rescued, 813 not rescued, 28 unknown, and 107 initial-state deviations over 1004 cases. No new training, simulator rollout, or experimental assistant call was performed for this paper revision.

## Contents and reproducibility

- The TeX files and figures compile locally without fetching project data or visiting a project website.
- `contract/` contains the fixed executed prompt, placeholder request projection, real response-schema factory, documentary request schema, and append rule. Placeholders do not represent real new calls. Personal source-path receipts are excluded.
- `analysis/` contains all 869 eligible-case timing rows, a separate 56-rescue table, task/split counts, source hashes, and read-only analysis scripts. These scripts are snapshots from the experiment workspace, whose expected layout is documented in their source. Re-running raw verification requires the original campaign evidence at those paths; the CSV/JSON files preserve the already-computed plotting data.
- `analysis_statistics.json` retains the original audit aggregates. `audit_analysis.py` is the first-revision raw-evidence checker, requiring the documented original archive layout.
- `figures/cases.png` retains the expanded uncropped input-view panel. The compact panel in the main paper shows the same actual intervention observations. These are not final-state images.

Public release paths and references must be re-reviewed against final double-blind submission rules. This public source bundle is not represented as a completed anonymous submission package.
