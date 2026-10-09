# Coverage factorization and difficulty: verified retrospective figures

## Recommended main figure

`coverage_decomposition.pdf` is sized 7.1 x 2.15 inches for a CVPR double-column figure.

Suggested caption: **Evidence coverage and rescue efficacy describe different limitations.** Each row factorizes the confirmed rescue fraction of the fixed historical failure cohort as $(E/N)(R/E)=R/N$. Counts are shown beside the observed percentages; these are exact cohort summaries, not fitted estimates. Composite-Unseen has the highest evidence coverage (489/530, 92.26%) yet the lowest rescue fraction among verified pairs (19/489, 3.89%). Its lower confirmed fraction therefore persists after restricting to verified pairs. Composite-Seen has lower coverage (243/321, 75.70%) and higher conditional rescue (25/243, 10.29%). This is an accounting identity and retrospective comparison, not a causal attribution to task novelty.

## Recommended supplementary figure

`task_difficulty_coverage.pdf` is sized 7.1 x 2.6 inches and includes all tasks for which each metric is defined.

Suggested caption: **Task-level heterogeneity does not imply a monotonic difficulty relationship.** Historical success across 50 original episodes per task serves only as a policy-specific difficulty proxy. Each point is one task, colored by split, with unweighted rank associations shown descriptively. CloseFridge contributes no original failures and is excluded from all rescue plots; OpenStandMixerHead has no verified pair and is additionally excluded from $R/E$. Identical coordinates overlap without jitter. Despite low recovery among the five hardest original tasks, the overall task-level association between historical success and conditional rescue is weak ($\rho=0.12$, 48 tasks). No regression, p-value, causal trend, or missing-at-random assumption is asserted.

## Validity details

- Source hashes are verified against the previous detailed audit; case counts and all split/task numerators and denominators are recomputed from the final per-case joint ledger.
- Task failure counts are checked against the independent historical 2,500-episode report.
- The full-cohort identity is 869/1004 x 56/869 = 56/1004. Displayed rounded percentages are illustrative; exact counts define the identity.
- For E=0, R/E is undefined and is never silently encoded as zero.
- Case evidence verification comes from the existing detailed audit; this script checks its source hashes and independently recomputes every plotted aggregate.
- Baseline task difficulty and rescue rates have shared finite-sample denominators; these exploratory task correlations do not identify a mechanism.
- No experiments, policy updates, model calls, or source evidence edits were performed.
