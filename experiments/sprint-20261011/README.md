# 2026-10-11 fresh-seed sprint: exact source snapshots

This directory preserves the source used in the bounded X/A/B, B-with-Astra, and GR00T C/V study. It is an audit snapshot, **not a portable or one-command experiment package**. No experiment was executed while preparing this release. No weights, raw trajectories/images, CLI transcripts, credentials, personal workstation paths, or live process-owner records are included.

The study reused the same 50 fresh environment cases. Its **300 planned arm slots** comprise:

| Campaign | Planned | Completed | Not started |
|---|---:|---:|---:|
| X/A/B: 50 cases × three frozen policies | 150 | 150 | 0 |
| B/V: Astra visual assistance, paired with the existing B control | 50 | 50 | 0 |
| GR00T: 50 cases × C/V | 100 | 62 | 38 |
| Total | 300 | 262 | 38 |

These are arm slots over 50 shared cases, not 300 independent environment cases. Engineering admission runs are outside this denominator. The 38 unstarted GR arms remain unknown administrative omissions; they are not policy failures, and seeds were not replaced. Initial-state and prefix deviations remain separate from strict paired effects. This small, incomplete sprint does not establish official benchmark performance or statistical superiority.

## Files and provenance

- `runtime/originx_sprint_20261011_v1/`: config-pinned X/A/B and B/V runtime, service guard, director, source-pinned terminal audit, and the **actually executed BV broker v3** under `bv_broker/`.
- `runtime/originx_sprint_gr_20261011_v2/`: selected required modules from the **actually executed GR v2** runtime, broker, strict transport-recovery verifier, and service guard. Unlaunched GR v1 candidates and source generators are excluded.
- `control/`: executed GR manager and terminal usage finalizer/host waiter. `xab_finalize_wait_v2.py` is the executed terminal watcher snapshot; its audit dependencies are explicitly hash-pinned in that source. It is a terminal evidence controller, not a new policy experiment.
- `manifests/`: byte-exact **actual prepared** manifests, preserving all 150/50/100 assignments and the shared 50-case roster. The GR Linux manifest uses LF; its earlier Windows staged copy differed only in CRLF line endings, not in any parsed field.
- `freezes/`: original source/service freeze metadata. Some original GR freeze entries name development utilities omitted from this selected source release; `dependency-inventory.json` lists them. Do not interpret a retained freeze document as a claim that every named file is bundled here.
- `configs/`: a byte-exact GR budget and clearly labelled configuration excerpts. The excerpts retain protocol, dependency-version and budget fields plus original config hashes; they are **not executable copies of the private launch configuration**.
- `source-inventory.json`: original logical source, provenance authority, and original SHA-256 for every exact copied file. Copy hashes were verified without newline normalization or code rewriting.
- `dependency-inventory.json`: hash-matched references to support code already in this repository, and explicit unbundled dependencies.
- `terminal-accounting.json`: denominator accounting and original configuration hashes. This is metadata derived from terminal records, not a replacement for the raw evidence archive.

Existing support sources live in [confirmatory-20261009/runtime](../confirmatory-20261009/runtime) and [gr00t-confirmatory-20261009/runtime](../gr00t-confirmatory-20261009/runtime). The inventory resolves 17 exact support-source matches and identifies 43 additional environment/dependency files not bundled here. The missing material includes the experiment-specific native-reset/evaluator and policy-loader packages, patched simulator components, model/checkpoint assets, OSMesa, and historical admission/parity evidence. Checkpoints, official simulator/model installations, authorized CLI authentication, and the original admitted environment must be supplied separately. No dependency installation or admission is performed by this release.

## Protocol and operational boundaries

Declared UTC schedule on **2026-10-11**: start 15:45:18, stop new rollout dispatch no later than 17:45:18, compute/cleanup deadline 18:00:18, delivery 18:45:18. The GR manager additionally required a service start by 17:28:18, fresh admission by 17:40:18, and began its outer cleanup by 17:58:18. Subsequent finalizers only read and aggregate existing evidence; they do not add rollouts or CLI calls. Those historical absolute deadlines are intentionally preserved in these source bytes.

Before any deliberately redesigned rerun, an operator must establish a new authorized namespace and schedule, reproduce all pinned dependency/checkpoint identities, freeze the complete roster before outcomes, and pass the source/runtime admission checks. Services require the specified GPU identity, ECC/memory checks, no conflicting owner, exact process identity tracking, and successful socket/RNG parity. GR also requires the real six-stream 360-second keepalive probe and verified termination of prior XAB/BV workers and models. The manager drains its own workers before the model owner. Missing cleanup evidence is not permission to force-start a replacement.

Teacher calls use the preserved `gpt-6-astra` / `high` prompt and public task/current RGB/step-budget interface. Both BV and GR have separate declared allowances of at most 50 CLI batches, 1,000,000 input tokens and 100,000 output tokens, batch size at most eight, no paid top-up and no account quota reset. Unknown usage stops further calls. Token thresholds are receipt-based: the last in-flight batch can exceed a token threshold; they are not a guarantee of a provider-enforced hard cap.

The source retains historical cluster roots, ports, GPU UUIDs, host aliases and a generic Windows D-drive requirement because changing them would invalidate the executed source hashes. Those identifiers are environment bindings, not credentials or a claim of portability. Simply changing paths or dates does not reproduce the original admission. Review all dependencies and create a separately reviewed configuration before attempting any rerun. This snapshot does not launch automatically.

The offline intervention-value gate prototype was not trained, calibrated, or used in these 300 arms and is not represented as an executed model here.
