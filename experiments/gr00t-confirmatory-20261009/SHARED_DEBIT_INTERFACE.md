# Sealed shared-budget debit builder

`build_gr00t_shared_debit.py` is an external accounting tool. It does not modify
the frozen runtime, invoke Astra, start models, run trials, reset quota, upload
files, or grant GR00T admission. Its only network action is a read-only SSH
rehash and owner audit. Do not run it until the main study has terminated and
its final evidence and complete broker inventory have been synchronized.

The tool has exactly four input studies: main development v2, main full v1,
failed GR00T development v1, and GR00T development v2. The durable local broker
state directories are fixed in `LOCATIONS`; copied inventories cannot stand in
for their original ledgers. Each actual start must have one terminal receipt
with matching source, config, manifest, batch, request, prompt, schema, stdout,
stderr, and optional image hashes. Every completed stdout usage event must have
explicit integer input and output counters and match the receipt. Actual
invocations are deduplicated; receipt copies never increase counts. Reasoning
tokens are part of output, not an additional charge.

The failed first GR00T development retains its original incomplete aggregate
with null totals and false completion. Its separate, strictly zero-call,
terminal owner proof permits a zero debit without rewriting that evidence.
Other unknown/incomplete accounting blocks generation. All four Windows broker
owners, scoped CLI processes, and recorded remote workers/models must be dead.
No completed trial or CLI is retried to make accounting pass.

## Invocation after termination

```powershell
python -B work/confirmatory_20261009/build_gr00t_shared_debit.py --plan <sealed-plan.json> --output-directory <new-D-drive-directory> --remote-directory /ephemeral/qinzhen/robocasa-xr1-20261003/results/<new-shared-budget-bundle>
```

The plan schema is `originx_shared_debit_build_plan_v1`. It has `studies`, an
array of four objects `{study, files}`. Each `files` object supplies `config`,
`manifest`, `source_freeze`, `campaign_finished`, `completion`, `usage`, and
`authority`; the failed GR00T development additionally supplies
`failure_terminal`. Each reference is `{local: <absolute D: path>, remote:
<absolute original remote path>}`. Use the actual final analysis directories:
main development `analysis-final-v2`, GR00T failed development
`analysis-final-failed`, GR00T development v2 `analysis-final`, and the main full
finalizer's verified final analysis directory. Local evidence must be exact
copies of those files. Source and checkpoint hashes from authority are freshly
checked remotely; reading them may take several minutes.

All checks precede creation of the exclusive output directory. The output
preserves each original broker inventory and original usage bytes, adds a new
terminal audit, and writes `debit.json` plus `local-build-receipt.json`. Upload
this complete bundle to its declared remote directory, verify its hashes, then
provide the remote debit path and SHA to the separate formal-service guard.
`raw_receipts` relocates unchanged raw receipt bytes inside the bundle while the
original usage's `source_paths` remain unchanged. The builder does not do the
upload or start the guard.

## Cap interpretation

The frozen main broker's cap applies to its own receipts; it does not subtract
the already executed development overhead. This tool reports the actual sum
across all four studies, rather than claiming a strict retrospective shared
cap. In-flight call cost is unknown until terminal accounting. Nonpositive
remaining allowance blocks future GR00T formal calls, and unknown accounting
blocks debit generation. No second allowance, reset, or paid top-up is created.

## Offline verification

```powershell
python -B -m unittest discover -s work/confirmatory_20261009/tests -p test_shared_debit.py -v
```

The 19 isolated fixture tests cover live owner rejection, incomplete receipts,
unknown counters, source/byte changes, ledger copies and duplicate invocations,
the strict failed zero-call exception, exclusive output, exhausted allowance,
and an end-to-end schema check using the real formal-service guard with fake
file/owner callbacks. They perform no SSH, CLI, GPU, or experiment action.
