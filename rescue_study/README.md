# Executed rescue-study source snapshot

These are byte-preserved Python sources from the two executed campaign packages and the local broker/finalizer tools. `source-sha256.json` identifies each source file. Historical package README/protocol files retain their pre-execution wording; final results are in [the rescue report](../ASTRA_RESCUE_REPORT.md).

This is an execution/audit snapshot, not a one-command portable deployment. The runner depends on the frozen simulator, policy service, runtime source files, and environment paths named in the original configuration. Those dependencies are identified in the campaign archives and original `source_snapshot/`. A researcher must reconstruct and verify that environment before scheduling another experiment. Importing or invoking launch/broker code may start paid inference or GPU jobs; no new run is required to inspect the published evidence.

`build_failure_manifest.py` selects all original failures; `runner.py` verifies initial fingerprints, pre-intervention prefixes and linked actual Astra application; `aggregate_campaigns.py` projects the v1 36-case partition and v2 968-case partition into exactly 1,004 records. `aggregate_cli_usage.py` deduplicates calls by batch/receipt. The continuation adds same-socket read-only keepalives and a narrow native-reset-deviation dispatch rule, without resampling or policy changes.

The online intervention uses the original instruction, current three camera views, and remaining action steps. Offline fingerprints can contain simulator state for verification; they are not model inputs. No weights or credentials are distributed in this source directory. Model weights are hosted at https://huggingface.co/Qinzhen3/OriginX.

## Recompute the fixed-cohort classification

After verifying the release archive hashes, extract v1 and v2 into separate directories. The path passed to each `--*-dir` must contain its campaign `report.json` and `case-classification.json` directly (preserve all sibling evidence). The preserved CLI enforces the original D: output location. On Windows, from the GitHub repository root, use explicit evidence paths and a new output filename (shown below in a shell with backslash continuations):

```bash
python rescue_study/astra_rescue_20261009/aggregate_campaigns.py \
  --failure-manifest evidence/astra-rescue/failure-manifest.json \
  --selection evidence/astra-rescue/continuation-selection.json \
  --selection-receipt evidence/astra-rescue/continuation-selection-receipt.json \
  --v1-dir D:/evidence/extracted-v1-campaign \
  --continuation-dir D:/evidence/extracted-v2-campaign \
  --continuation-config-sha256 7a7ee33a82a4af4f65d19bb62e1c5ec892b0cd2409d77f0ad17105cddd998cd9 \
  --output D:/evidence/joint-analysis.json
```

The expected classification is 56/813/28/107, M=874 and E=869. This command does not invoke Astra or schedule GPU work. Creation times and local source paths can differ from the publication snapshot; compare case identities, source hashes and statistics, not the regenerated JSON's whole-file hash.
