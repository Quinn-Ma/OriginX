# Preserved-claim continuation v2

This continuation retains the original fixed 50 minimum-seed task cases. It excludes every claim in both the original campaign and multigpu-v1, including failed, missing, or unknown records. If the original cohort has seven claims and v1 has 99 new claims, only 244 of the 350 fixed arms remain eligible. The 191 original claims outside this cohort remain preserved separately.

Sources are separate new files; no v1 source or result is rewritten:

- `continue_multigpu_v2.py`: `bdd8e17ac5bb62cdf61264843ce06d2453476a77b8d77aa33cea2e6d69dd2d9d`
- `multigpu_services_v2.py`: `ed9284bf00aae2aba10e07f5a30a6e5a1cc11684939bc07a3a936b303020ffd1`
- `aggregate_reduced_v2.py`: `51609688703d19e6b51e4c428738553ea88ce5292a08f3f3c669d9e71e05b788`

New services use `originx-confirmatory-multigpu-services-20261009-v2/gpu3` and `/gpu6`, ports 28020â€“28025 and 28030â€“28035 respectively. The same six-model, 36-stream, 108-forward, 360-second exact action/RNG gate applies. GPU3/GPU6 must actually be free of all prior and foreign models before new loading. All ownership and active-socket guards from the tested v1 guardian are preserved. Release schema is unchanged.

The parent launcher must wait for original and v1 terminal campaign/worker/model cleanup, sealed complete broker inventories, and completed v1 evidence archiving before campaign launch. The `prepare` and `debit` gates independently verify both prior batches' process identities, models, claims, actual CLI start/receipt coverage, stdout counters, and token completeness. No command below launches models, a broker, or any episode until the separate `campaign` command is explicitly invoked.

Create the budget debit with actual Linux paths and terminal hashes:

```
python -B continue_multigpu_v2.py debit \
  --broker-terminal ORIGINAL_TERMINAL --broker-terminal-sha256 ORIGINAL_SHA \
  --broker-state ORIGINAL_STATE \
  --v1-broker-terminal V1_TERMINAL --v1-broker-terminal-sha256 V1_SHA \
  --v1-broker-state V1_STATE --output NEW_EXCLUSIVE_DEBIT_PATH
```

`prepare` uses all the v1 arguments plus `--v1-broker-terminal`, `--v1-broker-terminal-sha256`, and `--v1-broker-state`. It requires the generated debit and exact combined service admission. Its config additionally pins `recover_cli_transport_v1.py`, `aggregate_transport_v2.py`, and `multigpu_services_v2.py`. Fixed shared caps are never reset. The intended remaining duration must be passed by the parent using authoritative current UTC and the 22:40 UTC evaluation stop; remote wall-clock skew is not a new allowance.

`aggregate_reduced_v2.py` takes the original `--old-broker-*` and newest `--new-broker-*` terminal/state arguments plus `--v1-broker-*`. It joins all three complete normalized inventories by irrevocable claim precedence, verifies every claimed raw record again, and writes only a fresh `originx-three-hour-amendment-20261009-v1/analysis-reduced-v2`. It keeps original and v1 transport-error outcomes unchanged; only new v2 receipts can pass the separately audited completed-reconnection exception. It never rewrites `analysis-reduced` or selects a better result from another batch.

Validation: 35 Windows CPU tests passed, including actual generated-debit binding with the real v2 broker lifecycle. Actual WSL Linux service tests passed 14/14; reused parent-owned child tests passed 21 with one Windows-only skip, including real pidfd/empty-argv/reap behavior. WSL had no pytest package, so scheduler/join pytest tests were run on Windows rather than represented as Linux tests. No remote process, CLI, rollout, or CUDA model was started while implementing these files.

Pre-launch debit-interface repair: the generated debit now includes verified `prior_broker_inactive: true` and SHA-bound `authority_files` covering both prior evidence sets and both sealed broker terminals. The real broker accepts this exact output; omission, fewer than three authority files, or changed evidence is rejected in contract tests. Earlier uploaded unexecuted helper bytes must be retained by the parent in the infrastructure archive.
