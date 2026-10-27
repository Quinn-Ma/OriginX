# T3 on the AutoDL H800 (zero-sample policy-RGB cohort) — disclosure note

Source: `autodl-t3/t3/exactness-checks-autodl-h800-v1.json` (site label `autodl_h800_egl_rgb0`,
renderer `NVIDIA H800 PCIe/PCIe/SSE2`, EGL, finished 2026-10-26 ~22:09 UTC server clock),
pulled unchanged from `/root/autodl-tmp/originx-harness-v2/work/autodl/t3/` on instance port 40137.

## Outcome

`all_passed: false` — 11 of the 12 checks the script declares pass; one fails.

| Check | Result |
|---|---|
| Depth convention | `legacy` selected everywhere, equals the frozen constant |
| Frozen extractor rule vs registry (with sites) | 189,225 / 189,225 equal |
| Production single-sample context | ok; registry `inconsistent: 0`; `superset_violation_px: 0`; `gained_px_not_site: 0` |
| Independent single-sample reference render | 0 mismatched pixels (segmentation, object masks, depth, analytic ghost, with-sites) |
| Policy RGB with sites hidden | byte-identical (`policy_rgb_unchanged: true`), `policy_offscreen_samples: [0]` |
| Ghost GL cross-check IoU ≥ 0.98 (registered pilot gate G4, harness-map §10.4) | **pass** — min 0.9925, median 1.0 over 115 evaluated / 117 masks |
| Ghost GL cross-check: all mismatched pixels within the 1-px boundary band | **FAIL** — 3 pixels outside the band, all in one mask (`results[1].gl_cross_check_trajectory[26]`, seed 20261026 scripted-motion trajectory: analytic 2767 px, GL 2753 px, 14 mismatched px, IoU 0.9949). The other 116 masks have 0 pixels outside the band. |
| Non-ghost pixels of the ghost render equal the plain render | true |
| Pixels inside masks changed by painting | 0 |
| Ghost ∩ subject, rings disjoint and exterior, supervision = numpy painter | all true |
| Overhead per step (median, 3 seeds) | 74.6 / 72.7 / 70.2 ms ≤ 100 ms budget |

## Treatment

- The registered pilot gate G4 is "GL cross-check IoU ≥ 0.98 at prepare steps" and passes.
- The stricter 1-px-band check is an additional engineering check declared by the T3 script.
  It fails by 3 pixels (0.11 % of the mask) on one of 117 masks. No code or threshold is changed
  in response; the failure is recorded here and will be disclosed in the paper's fidelity table.
- Decision (2026-10-26 23:10 UTC): proceed to the pilot and the formal run on this cohort.
  The analytic ghost (what the policy sees) is the registered cue; the GL render is a cross-check.
