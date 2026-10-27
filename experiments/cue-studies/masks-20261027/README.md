# Mask-cue study (exterior contour masks + destination ghost box) — status: done, no benefit found

This folder holds the evidence for the mask-cue arm of the cue-precision studies: a pixel-exact
object-mask prompt for the frozen XR1 policy (Xiaomi-Robotics-1-RoboCasa365) on RoboCasa
`CoffeeSetupMug`, drawn with `supervision` 0.27.0, compared against the frozen point cue under the
same preregistered paired protocol as the fine-region study in `../regions-20261024/`.

**Result (2026-10-27, HKUST H100 cluster; registered paired analysis, 12/12 pairs admissible, 0 UNKNOWN):**
masks 5/12 vs point 8/12 (2 mask-only, 5 point-only, 3 both, 2 neither); difference masks − point
= −25.0 percentage points, conservative 95% interval [−74.1, +39.7], exact paired McNemar p = 0.453.
No success benefit from exact contours; the point estimate favors the point cue; equivalence is not
established. Everything about that execution — analyzer reports, re-frozen manifests (pilot
`7c89c95a12e7`, formal `8266cc633102`), cluster T3 (11/12, same failing mask), GPU probes, scripts,
logs and the disclosure note — is in [`hkust-20261027/`](hkust-20261027/).

**History:** the first pilot launch on a rented H800 (2026-10-26 23:06:50 UTC) was lost with the
instance about 20 minutes later; no episode result was retrieved, and nothing from it is counted.
The material below (implementation, CPU test suite, H800 T3, rendering amendment, seed audit, the
originally frozen manifests `df91a4d28570` / `94d4721e1452`, AutoDL deployment scripts and the v1
manuscript) is kept as registered.

## T3 pixel-fidelity result (H800, EGL, zero-sample policy-RGB cohort)

`fidelity/exactness-checks-autodl-h800-v1.json` (summary in `t3-autodl-h800-note-v1.md`):
11 of 12 pre-declared checks pass. Depth convention `legacy`; frozen extractor rule 189,225/189,225;
independent single-sample reference render 0 mismatched pixels; policy RGB byte-identical with
gripper sites hidden; ghost-box GL cross-check IoU min 0.9925 / median 1.0 over 115 evaluated masks
(registered gate IoU ≥ 0.98 passes); pixels inside masks changed 0; supervision painter equals the
numpy reference; per-step overhead 70–75 ms (budget 100 ms). The stricter check "all mismatched
pixels within the 1-px boundary band" fails: 3 pixels outside the band in 1 of 117 masks
(analytic 2,767 px, GL 2,753 px, IoU 0.9949). No code or threshold was changed in response.

`fidelity/contact/` holds the 12 contact sheets (3 seeds × {grasp-initial, place-at_site,
place-above_site, place-in_gripper}; rows: raw render, masks arm, XR1 exact crop/resize raw,
XR1 exact crop/resize masks; the place-like states are test-only injected mug positions).
`exactness-checks-local-wsl-v1.json` is the earlier local D3D12/MSAA run (2026-10-24) that failed the
ghost gate and motivated the single-sample segmentation context.

## Preregistration (frozen, not executed)

- `unused-seeds-v1.json`: seed audit selecting 12 unused seeds 20261044–20261055 (disjoint from the
  development seeds 20261025–27 and the regions seeds 20261031–42).
- Formal manifest: 12 pairs / 24 cases, counterbalanced order, 2 workers on one GPU,
  `max_local_seconds` 21000, manifest sha256 prefix `94d4721e1452` (frozen on the instance on
  2026-10-26 before the pilot to permit dry runs; no case was executed). Pilot manifest: 6 cases
  (3 pairs, seeds 20261025–27). `manifests/` holds both manifests re-frozen locally after the
  instance shut down; the formal manifest reproduces the server-registered hash prefix
  `94d4721e1452` exactly (`manifests/manifest-refreeze-receipt-v1.json`).
- `rendering-amendment-h800-rgb0-v1.json`: zero-sample policy-RGB cohort declared for both arms
  before any registered episode (four-sample RGB was frame-to-frame nondeterministic on the H800).
- `pilot-g10-signoff-v1.json`: the user's sign-off on the contact sheets (pilot gate G10).
- `design/harness-map-v1.md`: the implementation plan with the pre-declared gates G1–G10;
  `design/implementation-report-v1.json`: the engineering report.

## Code and deployment

- `code/autodl-source-v2.tar.gz` (+ `.json` inventory with sha256): the source snapshot deployed to
  the instance (masks package, drivers, tests, analyzer).
- `code/analyze_masks_validation.py`, `test_masks_analysis.py`, `audit_unused_seeds.py`,
  `check_mask_exactness.py`, `contact_sheets.py`: offline analyzer (exact McNemar, Clopper–Pearson
  paired CI, Wilson intervals, UNKNOWN accounting, pilot-engineering mode), its tests, the seed audit,
  the T3 checker and the contact-sheet tool.
- `cloud/`: AutoDL deployment, stage-2 (renderer pin + cohort + T3 + freeze + dry run), launch,
  pull, and the post-reboot chain scripts.

Full model weights, simulator assets and the 5 GB trace archives of the regions study are not in
this repository; see `../README.md`.

`SHA256SUMS` lists every file in this folder.
