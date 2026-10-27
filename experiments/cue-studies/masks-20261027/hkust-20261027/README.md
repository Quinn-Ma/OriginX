# HKUST cluster execution of the mask-cue study (2026-10-27)

The registered mask-versus-point comparison was executed on the HKUST lab cluster (`h800-8-1`, 8x NVIDIA
H100 PCIe 80 GB, driver 570.195.03, native NVIDIA EGL) after the AutoDL H800 instance of the first attempt
was lost with no outcome retrieved. Same frozen source snapshot, vendor archives, seed assets, policy and
planner weights (all hash-verified), same zero-sample policy-RGB cohort for both arms, same frozen analyzer.

## Result (registered paired analysis; 12/12 pairs admissible, 0 UNKNOWN)

| | masks | point |
|---|---|---|
| success / 12 | **5** (41.7 %, Wilson 95 % [19.3, 68.0]) | **8** (66.7 %, Wilson 95 % [39.1, 86.2]) |
| 600-step horizon failures | 7 | 4 |

Discordant pairs: masks-only 2, point-only 5; both success 3, both failure 2. Risk difference masks − point
= **−25.0 percentage points**, conservative 95 % interval **[−74.1, +39.7]**, exact paired McNemar p = **0.453**.
No success benefit from exact contours was found; the point estimate favors the point cue; the result does not
establish equivalence. Secondary, descriptive only: terminal XY distance median masks 9.83 cm vs point 2.14 cm
(n = 11 each); time to success median masks 277 s vs point 282 s. The mask cue was drawn on 5,281 of 5,630
presentation events (86–99 % of the steps of every stage and camera); gate G5 (zero interior pixels changed)
held on all 5,630. Per case 217–630 s wall time (budget 1,600 s); no infrastructure timeout; no resume;
138/138 registered source files matched.

Authoritative numbers: `results/validation-results-local-wsl-v1.json` (frozen analyzer, offline, WSL sim
runtime, complete run directory with traces); the cluster-side `validation-results-server-v1.json` is identical
in every statistic. Per-case ledger: `results/per-case-ledger.csv`. Chinese summary: `实验结论.txt`.

## Deviations from the registration (all disclosed; see `site-change-note-hkust-v1.md`)

1. **Execution site**: AutoDL H800 → HKUST H100 cluster (`linux_egl` is an allowed site of the manifest). Python
   environments rebuilt from the same pinned lists (CPython 3.12.3, torch 2.8.0+cu128, flash-attn 2.8.3, transformers
   4.57.1 / 5.5.0); installed freezes in `evidence/*-installed-freeze.txt`. T3 pixel fidelity re-run on the cluster
   before the pilot: `t3/exactness-checks-hkust-h100-v1.json`, 11/12 checks, the same single failing mask
   (seed 20261026, step 128, wrist camera, 3 px outside the 1-px band), overhead 58.4/46.0/45.9 ms.
2. **Excluded GPUs**: GPUs 2 and 3 report `GPU Recovery Action: Reset` and return corrupt depth buffers (first T3
   attempt aborted; `probes/harness_probe_gpu2.out`, `harness_probe_gpu3.out`; the MuJoCo-only probe segfaults
   there). GPUs 5, 6, 7 are clean (`harness_probe_gpu5.out`, `harness_probe_gpu6.out`). EGL device index ==
   CUDA index was verified (`probes/egl_cuda_map.py`, `egl_memtest.py`).
3. **Worker-count re-freeze** (user instruction: use GPUs 5/6/7 fully): the frozen drivers re-froze both manifests
   with 3 workers, one per GPU, before any case completed. Pilot digest prefix `df91a4d28570` → `7c89c95a12e7`
   (`manifests/pilot-manifest-hkust-w3.json`), formal `94d4721e1452` → `8266cc633102`
   (`manifests/manifest-hkust-w3.json`; derived `max_local_seconds` 21,000 → 14,600 s). Seeds, pair indices,
   positions, arms, task, horizon and pair keys are identical case by case
   (`evidence/preexecution-worker-amendment-v1.json`); only `workers`, `worker_assignment`, `execution_budget`,
   the case-id hashes (which embed the worker fields) and `manifest_sha256` differ.
4. **Launches that produced no episode** (archived unchanged, nothing pooled): the 2-worker pilot stopped after
   3 minutes to apply item 3 (0 results); a 3-worker pilot launch and the first formal launch refused by the driver
   because a port of a stopped service was still occupied (all cases `unattempted`;
   `evidence/runs-receipts/`, `logs/`). The chain then waits for the ports to clear before a launch.
5. **Pilot gates**: G1, G5, G6–G7 (pair admission 3/3), G9 passed on the cluster pilot (`results/pilot-engineering-*`);
   G2–G4 and the overhead part of G8 come from the cluster T3; the per-case memory part of G8 was not
   evaluated; G10 (author sign-off on contact sheets) was not re-recorded for the cluster sheets.
6. The three workers shared their GPUs with another user's training job (~50 GB and 100 % utilisation each);
   nothing of other users was touched. Timeline (UTC, 2026-10-27): T3 16:22–16:30; pilot 16:38–16:55;
   formal 16:58–18:10.

## Layout

- `results/` — analyzer reports (local WSL = authoritative, server copy), pilot engineering reports, per-case
  ledger, run-level receipts and identities (`results/README.md`).
- `manifests/` — the two re-frozen manifests used on the cluster.
- `t3/` — cluster fidelity report, launch decision, logs (GPU 5 run; failed GPU 3 attempt).
- `probes/` — GPU/EGL probes and their outputs.
- `evidence/` — site files, backend config, worker-count amendment, transfer receipt, environment freezes,
  mechanical smoke receipt (GPU 7), archive extraction markers, run receipts of the refused launches.
- `scripts/` — deployment and execution scripts (`hkust_build_envs.sh`, `prepare_hkust_project.py`,
  `stage2_hkust.sh`, `chain_hkust.sh`, `relaunch_w3.sh`, `chain_watch_hkust.sh`, `pull_results_hkust.sh`, smoke).
- `logs/` — chain logs (including the stopped/refused attempts), launcher logs, environment build log.
- `site-change-note-hkust-v1.md` — the disclosure note; `实验结论.txt` — Chinese summary.

Full per-case result files (30 x ~3.5 MB) and NPZ traces (formal 7.1 GB, pilot 1.8 GB) are kept offline in the
harness workspace (`runs/hkust-mirror/`) and are available on request.
