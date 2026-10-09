# GR00T formal service guard (prepared, not executed)

`gr00t_formal_service_guard.py` is an external infrastructure supervisor. It does not import Torch, alter the frozen study, generate admission, start scored rollouts, start a broker, or call Astra. Its only Python children are the existing frozen GR00T `server` and `parity_probe` modules. It must be reviewed and copied to `/ephemeral/qinzhen/robocasa-xr1-20261003/gr00t_formal_service_guard.py` before any future execution. None of this preparation launches a remote process.

## Admission and fixed scope

A future operator supplies a SHA256-pinned JSON plan with exactly these fields:

```json
{
  "schema": "originx_gr00t_formal_service_plan_v1",
  "development_admission": {"path": "<absolute remote path>", "sha256": "<actual digest>"},
  "main_finished": {"path": "<absolute remote path>", "sha256": "<actual digest>"},
  "main_cleanup": {"path": "<absolute remote path>", "sha256": "<actual digest>"},
  "main_services": {"path": "<absolute remote path>", "sha256": "<actual digest>"},
  "shared_budget": {"path": "<absolute remote path>", "sha256": "<actual digest>"}
}
```

This is a schema illustration, not approval or a runnable plan. The four non-budget paths are fixed by the helper: independent GR development-v2 `admission-review.json`, main full `campaign-finished.json`, main full `cleanup.json`, and main v2 `services-ready.json`. The helper rehashes admission evidence and checks current Linux owner identities. Main must be terminal, including a honestly recorded terminal failure; its campaign, workers, and all recorded service owners must be inactive. An incomplete main campaign cannot pass.

GPU 6 must have the pinned UUID `GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17`, zero reported volatile uncorrected ECC errors, at least 16 GiB free, and no compute occupants before startup. It does not permit coexistence with main models or other users. All other GPUs are hidden. The service is fixed to localhost port 27903, six slots, the original seven service-file SHA256 values, the original official GR00T source commit, and all checkpoint/config/metadata hashes. It retains official four denoising steps and 16-step replanning. No adapters are loaded. The previous direct-v2 reference and its native fixtures are rehashed; every new service must complete fresh six-stream socket parity and at least 360 seconds of same-socket keepalive.

Port-27903 profile directory, formal parity directory, and guard output directory must all be absent. Existing output is never overwritten or treated as permission to restart. Service profile and parity result hashes are published in `results/originx-gr00t-formal-service-guard-20261009-v1/ready.json` only after the checks pass. This ready receipt does not approve a scientific experiment; the separately admitted study remains a separate operation.

## Shared budget evidence

The companion `build_gr00t_shared_debit.py` prepares the evidence from actual durable states and CLI receipts only after all four studies and their brokers terminate:

1. `originx-confirmatory-development-20261009-v2`
2. `originx-confirmatory-20261009-v1`
3. `originx-gr00t-confirmatory-development-20261009-v1`
4. `originx-gr00t-confirmatory-development-20261009-v2`

The debit schema is `originx_shared_budget_debit_v1`, with `all_prior_brokers_inactive: true`, `complete_input_output_token_accounting: true`, `prior_usage` containing nonnegative integer `cli_calls`, `input_tokens`, and `output_tokens`, and exactly four `studies` entries. Each entry supplies its exact `study` name and SHA-bound `usage` and `terminal` JSON references. Terminal schema is `originx_confirmatory_broker_terminal_audit_v1`; its full directory inventory is rehashed, excluding only the new audit itself and `broker.lock`. Inventory paths use forward slashes.

Original aggregate usage files remain unchanged, including original `source_paths`. When copied into a new evidence bundle, a study may supply `raw_receipts` mapping each original receipt SHA256 to `{ "path": "<absolute copied receipt path>", "sha256": "<same SHA256>" }`. Every mapped receipt must occur inside that study's sealed terminal directory and match both the inventory and aggregate token counters. Invocations are deduplicated across all four studies. Technical failures are not exempted from costs: status need not be `ok`. Missing counters, unfinished CLI starts, duplicate invocations, inconsistent receipts, or transcript-integrity problems stop admission rather than reduce the debit.

The only explicit zero-call exception is failed GR development-v1. Its original usage contains null totals and false completeness; these bytes are retained. Independently proving zero requires a sealed terminal inventory with no batch/start/receipt files, explicit integer `broker_status.cli_calls == 0`, zero terminal invocations, and a SHA-bound `zero_invocation_evidence` referencing the original `originx_gr00t_failed_development_terminal_audit_v1` with all recorded campaign/worker/model owners inactive. Unknown counters in any other study are never interpreted as zero.

All three remaining caps must be strictly positive before model start: 1,500 CLI calls, 25,000,000 input tokens, and 2,000,000 output tokens minus actual prior usage. Reasoning tokens are already included in output tokens and are never added again. The broker retains responsibility for enforcing the remaining run-time budget; this helper does not invoke it or refill allowances.

## Ownership and lifetime

The embedded `TrackedChild` implementation was copied from the tested GR development-v2 `processes.py` (SHA256 `89bf3178e1e71ff0cbb19e0d3bbc1c6bc880b27450a12e9ec424ff6b41d2c6d8`). A child is registered immediately after `Popen`, before any process metadata reads. The guard holds both the unreaped direct child and its kernel pidfd. A bounded five-second handshake requires two identical PID/start-tick/full-argv/cwd observations, tolerating transient empty argv. A failed handshake still leaves the child registered for drain and reap. Signals require the direct parent and original start ticks and use pidfd, including before full argv binding; foreign processes are never signalled. An owner with the same PID and start ticks but changed argv/cwd is rejected as unproven termination; only absence, zombie/exit state, or a different birth identity proves the old owner inactive.

The requested lease is between 600 and 435,600 seconds (five days plus one hour). Sixty seconds are reserved from that lease for draining at most two owned children. Each drain permits SIGTERM for 15 seconds, then SIGKILL and a ten-second reap deadline; failures are recorded rather than masked. Service readiness and parity each have independent bounded waits. On completion, signal, timeout, or error, the guard preserves the first error and drains parity before the model. On normal separately run campaign cleanup, the guard reaps its own terminated model. A SHA-matching `release-request.json` may request an early stop only if port 27903 has no active or closing client sockets.

Future read-only verification (replace placeholders with real independently verified paths/digests):

```text
python3 /ephemeral/qinzhen/robocasa-xr1-20261003/gr00t_formal_service_guard.py --check --plan <real-plan.json> --plan-sha256 <actual-sha256>
```

Future explicit infrastructure execution uses `--run` instead of `--check`, optionally `--lease-seconds 435600`. The helper itself neither writes the supplied plan nor creates a shared debit or scientific approval.

## CPU verification

```text
python -m unittest discover -s work/confirmatory_20261009/tests -p test_gr00t_formal_service_guard.py -v
wsl -d Ubuntu-22.04 --cd /mnt/d/codex/2026-10-08/new-chat python3 -m unittest discover -s work/confirmatory_20261009/tests -p test_gr00t_formal_service_guard.py -v
```

Tests cover incomplete/duplicate/exhausted accounting, sealed inventory mutation, unchanged raw-receipt relocation, the narrowly defined zero-call proof, forbidden GPU occupants, exact child commands, registration before identity failure, and foreign-parent rejection. Three Linux-only tests use disposable sleeping Python children to verify actual pidfd lifecycle, transient argv binding, unbound-child timeout cleanup, and timed phase drain/reap. They import no model library and run no simulation or Astra call.
