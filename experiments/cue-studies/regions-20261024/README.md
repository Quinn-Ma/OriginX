# Fine-region cues vs point cues for a frozen policy: preregistered 12-pair validation (2026-10-24)

**Result: no benefit of fine-region cues was found, and equivalence was not established.** Under the registered accounting rule, 10 of 12 preregistered seed pairs were admissible; the fine-region ("neighborhood") cue succeeded in 6/10 and the point cue in 6/10, with 0 region-only and 0 point-only successes. The paired success difference (regions minus point) is 0.0 percentage points, conservative 95% interval [-35.5, 35.5], exact two-sided paired McNemar p = 1.0. A deviation-inclusive supplementary analysis that admits one case re-executed after a host interruption has 11 pairs: point 7/11 vs regions 6/11, difference -9.1 percentage points, conservative 95% interval [-45.7, 32.7], p = 1.0; the entire difference between the two analyses comes from that one added pair (seed 20261034: point success, regions failure). One further pair (seed 20261042) contains an UNKNOWN point-arm outcome (infrastructure timeout) and enters neither paired analysis. Neither analysis supports a success benefit from the region presentation, and a non-significant result on 12 pairs is not evidence of equivalence.

Study scope: RoboCasa365 `CoffeeSetupMug`, the frozen Xiaomi-Robotics-1-RoboCasa365 policy (XR1) and a frozen Qwen3-VL-4B-Instruct planner/verifier, one task, one model, one fixed implementation. Executed locally on an RTX 5090 Laptop GPU with all model inference on CUDA; no cloud spend. Wall time: about 2.09 h for the original run until the interruption and 3.85 h (including startup and cleanup) for the resumed portion.

The anonymous paper in this folder (`regions_validation_paper.pdf`, "Persistent Entity Handles and Local Reference Marks for a Frozen Robot Policy: A Preregistered Paired Simulation Study", Anonymous Authors, October 2026) reports the same numbers. It is kept anonymous here as it is. The Chinese conclusions file `实验结论.txt` is the authoritative summary from which the numbers above are taken; `validation-results.json` is the analyzer report.

## Design

- **Preregistered seeds and slots.** Twelve fixed seeds, 20261031-20261042, each run once per arm: 24 registered slots. These seeds were audited as unused in prior local experiments (`audit_unused_seeds.py`). The three development seeds (20261025-20261027) are excluded and never pooled with this cohort; their historical 1/3 vs 2/3 development comparison is recorded separately in `development-context-and-power-v1.json`.
- **Arms.** `point`: the frozen v1 controller/verifier with the original exact destination-point presentation. `regions`: the same frozen system plus the unchanged subject-anchor and local-region presentation package (local neighbour points, an object-centre proxy and a fixed legend). Both arms keep the original Place target, the native 16x12 action chunks, the 600-step horizon, the shared verifier/parser and the same geometry extractor; no action modifications, no feedback repair, no gripper gating, zero policy parameter updates.
- **Counterbalanced order.** Even pair indices run point then regions; odd pair indices run regions then point (6 point-first, 6 regions-first pairs).
- **Pairing admission.** The first arm of a pair establishes a fresh natural initial state and a non-empty, independently generated natural plan. Before its first action, the second arm must match the first arm's full initial signature and natural plan; the plan is not injected. A mismatch makes the pair UNKNOWN/invalid, evidence is preserved, and there is no retry or seed substitution.
- **Primary endpoint.** Official benchmark success within 600 unchanged control steps; paired contrast regions minus point.
- **UNKNOWN accounting.** Exhausting the 600-step horizon is a known task failure. Infrastructure interruption or timeout, invalid pairing and unattempted slots are UNKNOWN; they are never converted to successes or failures, there is no outcome replacement, and excluded-pair counts stay visible. Paired inference is conditional on the admitted complete pairs.
- **Statistics (frozen before execution, `analysis-plan-v1.json`).** Two-sided exact McNemar test conditional on discordant pairs; per-arm Wilson 95% intervals on the same eligible pairs; paired risk difference with a conservative Bonferroni-simultaneous Clopper-Pearson 95% interval (difference of 97.5% Clopper-Pearson intervals for the win and loss probabilities). The conservative interval and the conditional McNemar test are not inverses of each other; neither is selected after seeing the data.
- **Secondary, descriptive endpoint.** Terminal horizontal (XY) distance from the object root to the naturally selected receiving site at each episode's own termination, including known failures. It is not mug-to-spout precision and not a common-time trajectory error.
- **Power.** A 6-0 discordant split would be the smallest one-sided count with two-sided exact p < 0.05 (p = 0.03125). Twelve pairs have low power for modest effects (illustrative calculations in `development-context-and-power-v1.json`).

## Deviation: host interruption and resume

The original run completed six episodes (the three complete pairs for seeds 20261031-20261033) and was then interrupted when a Codex desktop application restart closed WSL. The seventh episode (seed 20261034, regions arm, first arm of its pair) had recorded no environment step (trace with 0 records). The six completed episodes were kept and not re-run. The remaining 18 episodes (seeds 20261034-20261042, including the zero-step episode) were executed by a resume driver written after registration, using the same frozen manifest, configuration, ports and models. Under the registered rule an interrupted case whose trace had been opened is UNKNOWN, so the main analysis marks that case UNKNOWN and excludes its pair (10 admissible pairs). The re-executed case is reported only in the labelled deviation-inclusive supplementary analysis (11 pairs). Both analyses are reported; the registered rule is primary.

## UNKNOWN cases

- Seed 20261034, regions arm: zero-step interruption, re-executed in the resume run; UNKNOWN under the registered rule, admitted only in the deviation-inclusive analysis (outcome there: failure, 600 steps, 12.10 cm).
- Seed 20261042, point arm: infrastructure timeout (`TimeoutError: timed out`, a verifier HTTP timeout) after 544 of 600 steps; category `infrastructure_timeout`; UNKNOWN, not re-run; its pair enters neither paired analysis.

## Per-seed ledger

S = success, F = known task failure (horizon exhausted), U = unknown; XY in cm at each episode's termination.

| Seed | Point | Regions | Admitted (registered rule) | Admitted (deviation-inclusive) |
|---|---|---|---|---|
| 20261031 | F, 600 steps, 16.29 | F, 600 steps, 17.67 | yes | yes |
| 20261032 | F, 600 steps, 11.81 | F, 600 steps, 7.40 | yes | yes |
| 20261033 | F, 600 steps, 19.43 | F, 600 steps, 19.18 | yes | yes |
| 20261034 | S, 319 steps, 3.39 | F (re-executed after zero-step interruption), 600 steps, 12.10 | no | yes |
| 20261035 | F, 600 steps, 6.78 | F, 600 steps, 8.22 | yes | yes |
| 20261036 | S, 256 steps, 0.67 | S, 258 steps, 0.37 | yes | yes |
| 20261037 | S, 281 steps, 3.14 | S, 290 steps, 0.31 | yes | yes |
| 20261038 | S, 295 steps, 1.51 | S, 289 steps, 0.97 | yes | yes |
| 20261039 | S, 280 steps, 3.72 | S, 296 steps, 2.13 | yes | yes |
| 20261040 | S, 274 steps, 2.03 | S, 295 steps, 1.71 | yes | yes |
| 20261041 | S, 262 steps, 0.19 | S, 290 steps, 1.87 | yes | yes |
| 20261042 | U (infrastructure timeout at 544 steps), XY missing | F, 600 steps, 15.99 | no | no |

Counts (success / known failure / unknown), registered rule: point 7 / 4 / 1, regions 6 / 5 / 1; paired (10): both success 6, both failure 4, regions-only 0, point-only 0. Deviation-inclusive: point 7 / 4 / 1, regions 6 / 6 / 0; paired (11): both success 6, both failure 4, regions-only 0, point-only 1.

Worst-case envelopes over all 12 registered seeds (known arms keep their outcome; each unknown arm set to success or failure): registered rule [-16.7, 0.0] percentage points; deviation-inclusive [-16.7, -8.3] percentage points. These envelopes carry no sampling uncertainty and are not confidence intervals.

## Secondary endpoint (terminal XY distance, descriptive only)

- Registered rule: point median 3.43 cm over the 10 paired episodes (3.39 cm over all 11 known episodes); regions median 2.00 cm over the 10 paired episodes (2.13 cm over all 11 known episodes). Paired difference regions minus point over 10 pairs: median -0.31 cm, mean -0.57 cm (negative favours regions).
- Deviation-inclusive: point median 3.39 cm (11 paired; 3.39 cm over 11 known); regions median 2.13 cm (11 paired; 4.76 cm over all 12 known). Paired difference over 11 pairs: median -0.30 cm, mean 0.27 cm.

These distances include failed episodes, are measured at unequal termination times, and are a descriptive secondary endpoint, not a tested hypothesis.

## Evidence boundaries

1. The earlier development figures (2/3 successes, 3.64 cm median) come from three development seeds and are reported separately; they are never merged with this cohort.
2. The regions arm is a combined presentation package (local neighbour points, object-centre proxy and fixed legend text); any effect cannot be attributed to finer identity partitioning alone.
3. Both arms share simulator identity, semantic location and geometry information; the receiving site is supplied by the simulator asset. No pure-RGB or no-privileged-information claim is made.
4. The original Place target and native action interface are preserved; there are no hand-written action corrections. The full identity-interface vs native-policy difference is not this contrast.
5. ALIGN/EXECUTE feedback and later repair candidates are excluded. The historical 74 fixed-frame GPU diagnostics still show physical-judgement errors by the visual verifier; valid output format or fewer nulls does not mean the feedback is repaired.
6. This is not a cross-model or cross-benchmark conclusion and carries no acceptance, ranking or benchmark claim.

The execution-evidence audit (`execution-evidence-audit-combined-v1.json`, status `PASS`, execution evidence status `PARTIAL_EXECUTION_EVIDENCE_VERIFIED`) verified all 138 registered source files, 1,376 persisted planner generations, identical service identities across the two launches and the integrity of prior archives. It does not claim a complete physics replay and did not instrument the native encoded input tensors; those limits are stated inside the file.

## What is included and what is not

Included here: the anonymous paper PDF and its LaTeX sources, the Chinese conclusions, the analyzer report, the frozen manifest, the frozen analyzer and its combined (prior + resume) wrapper, the audit wrappers and their principal outputs, the independent read-only reviews (runtime, archive packaging, paper facts), the frozen analysis plan, the development-context/power note, the paper updater and the PDF build script. Every file is byte-identical to the harness copy; see the SHA256 list below and `SHA256SUMS`.

Not included:

- The full trace archive `id60-regions-validation-combined-full.zip` (about 5 GB; 5,451,533,385 bytes: per-step traces, NPZ arrays, CUDA/renderer identities, launcher journals and the raw run directories `runs/id60-regions-validation-local-v1` and `runs/id60-regions-validation-local-v1-resume1`) together with its inventory and verification receipts. It is kept offline and is available on request.
- `unused-seeds-v1.json` (6.2 MB, the output of `audit_unused_seeds.py`; over the 5 MB per-file limit for this folder) and `treatment-exposure-audit-combined-v1.json` (1.1 MB, the output of `audit_treatment_exposure_combined.py`), both available on request.
- The frozen originals that the combined wrappers import: `audit_treatment_exposure.py` (needed by `audit_treatment_exposure_combined.py`), `package_regions_validation.py` and `scripts/run_id60_regions_validation_resume_local.py` (needed by `audit_package_combined.py`), and the harness runtime package itself (`originx_harness/id60_regions_validation/`, hashes listed in `independent-runtime-review-v1.json` and `manifest.json`).
- Model weights, simulator assets, synthetic-test directories, dry-run receipts, paper build intermediates and backup inventories.

The scripts are archived as executed, not as a portable package: they resolve paths relative to the harness layout (`work/`, `runs/`, `outputs/` under one root) and need the raw run directories from the offline archive to re-run. They are provided for audit and reading. Line-by-line correspondence between the paper and the report is recorded in `paper-results-insertion-receipt.json` and the paper-facts review.

## Frozen identities

- Registered manifest identity (`manifest_sha256` field inside `manifest.json`): `9414bd6546022392a2c813f1b8999a050a2c5230592f6f67fcc61971964eb9e4`
- `manifest.json` file: `ec686401d0300bbb0740d009a7bf9b5570ab3e60f709835fdb00cb991d638bbe`
- Frozen analyzer `analyze_regions_validation.py`: `d9cb8f468b9c493432c5a291b96355f0f693da001296a562b810de2ee57b4aff` (as pinned in `analysis-plan-v1.json`)
- Final analysis (`validation-results.json`, byte-identical to `final-analysis-combined-v1.json`): `fa85d64d082508fcc1b3c091d93653dd2dbd45b114ecac008b3b1ada04381325`
- Execution-evidence audit: `617e14e80e6db706f5b78051ffd9025e9cdd30e891b463f8292eeaac4f40bbf9`
- Paper source archive `paper-sources.zip` (22 members): `3ad26e68633e40fb52b70dff846daefa89c00fc00d8a38c87633e3ffdfb06d38`

## Files (SHA-256)

| File | Bytes | SHA-256 |
|---|---:|---|
| `regions_validation_paper.pdf` | 684,457 | `8e466638f7ef63fc1615d05ab3a7fe5dde8ea14e1c9883e54e4ab6709b213894` |
| `paper-sources.zip` | 1,154,064 | `3ad26e68633e40fb52b70dff846daefa89c00fc00d8a38c87633e3ffdfb06d38` |
| `实验结论.txt` | 7,105 | `eb33418d4a72303272cbfab2860c6218c7caa4483aa2bad0b8600542071c864f` |
| `validation-results.json` | 288,435 | `fa85d64d082508fcc1b3c091d93653dd2dbd45b114ecac008b3b1ada04381325` |
| `manifest.json` | 30,678 | `ec686401d0300bbb0740d009a7bf9b5570ab3e60f709835fdb00cb991d638bbe` |
| `analysis-plan-v1.json` | 1,414 | `42e2421c8fbd6164de6ed63bf2e7fe5d3488d763658601e8c430c30073161c86` |
| `development-context-and-power-v1.json` | 5,829 | `f6b29156cedead3f1b14f555121232caf98487a1fd35a7110d3ae6ae7f7939d2` |
| `analyze_regions_validation.py` | 29,386 | `d9cb8f468b9c493432c5a291b96355f0f693da001296a562b810de2ee57b4aff` |
| `analyze_regions_validation_combined.py` | 5,688 | `3c1cb1e7f85991473fda1baf29cac564a4871154a8bab11ba5dfe9b840955ff9` |
| `audit_unused_seeds.py` | 6,242 | `b3253152201ed92c4dfcd850628e1b776048b6aaf36a631464253394dd6b7657` |
| `audit_package_combined.py` | 15,953 | `f8857e8fabc0e7f3a8e0cef9116d01b0cd1b8e5a6945d691e2a3affeb740125a` |
| `audit_treatment_exposure_combined.py` | 8,752 | `93c62d1c352d7aa21d3fe28757847fd8aff02649231067e1cd5d33ea906a08b0` |
| `execution-evidence-audit-combined-v1.json` | 3,250,535 | `617e14e80e6db706f5b78051ffd9025e9cdd30e891b463f8292eeaac4f40bbf9` |
| `execution-evidence-audit-combined-v1.sha256` | 109 | `44276e605a9ab846192cfb2ebd982bdd497d4ab31b8294600c6db3907f226ba8` |
| `final-analysis-combined-v1.json` | 288,435 | `fa85d64d082508fcc1b3c091d93653dd2dbd45b114ecac008b3b1ada04381325` |
| `independent-package-review-v1.json` | 612 | `0cb6ff2dfe2449310175149caed86b39d5cf69217c3bd35f789003f20c26893c` |
| `independent-package-review-v1.txt` | 3,517 | `b2319afa90b1852a1a4ad2455ebc029b353056eabbe560bbbcc137f02386c8b8` |
| `independent-runtime-review-v1.json` | 1,450 | `c69382d8ba024f00ee1050a61748e56da34b20571829fa1a3951dda04628a47d` |
| `independent-runtime-review-v1.txt` | 3,330 | `4382f021246ca0dc8372d4efc7d519c4471dc366afed668555985d833c8bec73` |
| `paper-facts-review-v1.json` | 6,241 | `7c293e6edcc1e1377e156e95575ae81e305dfe8a7278faaa0fdddaff09f6ee04` |
| `paper-facts-review-v1.txt` | 3,086 | `c91fe2c444d8abcdddf5a0bf5bfd372290bcda7d8849bc0f77224f2908d45ce0` |
| `paper-results-insertion-receipt.json` | 721 | `b719be8b5f4c0458296df8363ec4b768596ed16994c9db33bb38411597e4caba` |
| `update_paper_results.py` | 23,344 | `070524d3fb7d5903722c19ef1d99b8ef43b3cb4718959b5bd5f9b9244706e8a1` |
| `build_paper_pdf.sh` | 477 | `2baaa71f36255fb048cc911d22b0a42372100d6fa9dc0fa80203f11e5e2cb49a` |

`SHA256SUMS` in this folder lists the same digests (plus this README) in `sha256sum -c` format.
