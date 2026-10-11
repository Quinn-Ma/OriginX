# Fresh-seed study evidence

[English report](../../FRESH_SEED_STUDY.md) · [中文报告](../../FRESH_SEED_STUDY_ZH.md) · [Compact summary](summary.json)

The fixed cohort has 50 unique task/seed cases. X, A, B and BV each have 50 terminal outcomes. GR00T control and visual assistance each have 31 terminal outcomes and 19 unknown/unrun cases. Missing cases and initial-state deviations are retained; the original benchmark and earlier follow-ups are unchanged.

- `originx-audit.json`: terminal X/A/B and strict B/BV analysis, including every case.
- `originx-cli-usage.json`: deduplicated BV invocation receipts and complete token counters.
- `gr00t-analysis.json`: fixed-denominator GR00T results and paired classifications.
- `gr00t-cli-usage.json`: deduplicated GR00T invocation receipts and counters.
- `initial-state-diagnostic.json`: unchanged classification of eight real X/A/B initial-state deviations; cause unestablished.
- `gate-synthetic-checks.json`: 15 synthetic tests only; zero training and zero rollouts.
- `gr00t-cleanup-and-evidence.json`: independent terminal archive and cleanup verification, with machine paths normalized.

`summary.json` records SHA-256 identities of the original source files and the public path-normalized copies separately. Absolute machine-specific prefixes are replaced by `project/` or `workspace/`; scientific values, per-case ledgers and source-code hashes are unchanged. Native-reset initial states are checked independently from single-arm task outcomes. GR00T action-trace hashes come from the source-pinned producer; full later action arrays were not archived. No model superiority or general intervention benefit is established.
