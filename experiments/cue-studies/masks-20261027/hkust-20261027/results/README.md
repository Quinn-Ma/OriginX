# Results of the HKUST cluster execution (2026-10-27)

- `validation-results-local-wsl-v1.json` — authoritative analysis of the formal 12-pair run, produced offline in the frozen WSL sim runtime (Python 3.11.13) from the complete run directory with traces; `analyzer-local-wsl.log` is its log.
- `validation-results-server-v1.json` — the same frozen analyzer run on the cluster right after the run; identical paired statistics (the only differences are path fields and one floating-point last digit).
- `pilot-engineering-local-wsl-v1.json` / `pilot-engineering-server-v1.json` — engineering-only analysis of the 6-case pilot (gates G9/G5/sources; success counts are never evidence).
- `per-case-ledger.csv|json` — one row per case (seed, arm, status, steps, elapsed seconds, worker, case_id). The 30 per-case result files (about 3.5 MB each) and the NPZ traces (formal 7.1 GB, pilot 1.8 GB) are kept offline in `runs/hkust-mirror/` of the harness workspace and are available on request.
- `id60-masks-validation-hkust-w3-v1/`, `id60-masks-pilot-hkust-w3-v1/` — run-level receipts: `local-execution.json` (layout, GPUs, deadline), `site.json`, the manifest copy, launch plan/receipt, case summary and per-worker policy/planner/worker identities.
