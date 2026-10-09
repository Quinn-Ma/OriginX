# Reproducing and auditing OriginX

**Distribution.** Model weights and processor/configuration assets are distributed through [Qinzhen3/OriginX on Hugging Face](https://huggingface.co/Qinzhen3/OriginX). All Python implementation code, including the three original Xiaomi model files needed by `trust_remote_code`, is distributed through this GitHub repository. The full runtime is assembled locally; Hugging Face does not host Python code. Evaluation evidence remains in the [OriginX v1.0.0 GitHub release](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0).

**Naming and immutable history.** The public adapter name is A2000, released as `adapter-originx-2000.pt`. This is an alias for the unchanged historical A1613 adapter, which completed 1,613 updates over 80,000 sampled windows. The subsequent B2000 branch completed 2,000 updates with that policy frozen. Renaming does not change any weights, historical metadata, or the 2,500-episode result.

**Prepare the complete local runtime.** Clone this GitHub repository and run the helper from its root. It downloads only the 15 original non-code base assets and the two adaptation files from OriginX, then copies the three exact original model Python files from this checkout. The default download revision is `main`; for a fixed reproduction, pass a recorded Hugging Face commit SHA.

```bash
python -m pip install huggingface_hub
python prepare_originx.py --revision main --output-dir assets
```

If the OriginX snapshot has already been downloaded, reuse it without another network download:

```bash
python prepare_originx.py --snapshot /path/to/OriginX-snapshot --output-dir assets
```

The selected snapshot must contain every file named by `base-model-assets.json` with source `huggingface`, plus `adapter-originx-2000.pt` and `branch-00002000.pt`. The helper rejects incomplete uploads, hash mismatches, and a snapshot of only the old adaptation files. It produces exactly 18 original identity assets in `assets/base/` and the two adapters in `assets/weights/`. It checks every size and SHA-256 before copying and verifies copied bytes and the original aggregate base identity. Existing matching files are reused; different or unexpected destination files are never overwritten. No model execution, training, or GPU access is performed. When downloading, its Hugging Face cache is under `assets/.hf-cache/`; copying the 10.1 GB base into `assets/base/` requires additional disk space.

The base contents are the original Xiaomi checkpoint at revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`. The expected aggregate base identity is `ccaa65176cf4294b7c0c618e1cc3be7b65617359630c35d3a18dd75c406fc184`. Do not add README, helper, manifest, or modified processor/config files inside `assets/base/`. The portable loader verifies this identity and the original loaded-tensor identity before serving.

**Audit the recorded evaluation.** Download `evaluation-evidence.tar.gz` from the versioned GitHub release, then run:

```bash
python verify_evidence.py evaluation-evidence.tar.gz
```

`verify_evidence.py` needs only the Python standard library. It validates the original archive, per-episode result hashes, complete denominator, fixed assignment identities, reset proofs, failed-episode horizons, and reconstructed counts. The expected output is 2,500 episodes, 1,496 successes, and 1,004 failures. This evidence check does not perform simulation or model inference.

**Environment.** The original inference environment was Linux, Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.57.1, and FlashAttention 2.8.3. [requirements-inference.txt](requirements-inference.txt) records the inspected relevant package versions, rather than claiming to be a complete operating-system lockfile. Install the matching CUDA PyTorch build before building FlashAttention; follow the upstream Xiaomi and FlashAttention installation instructions for the GPU platform. The simulator was a separate Python 3.11 environment; its package pins are in `evidence/config.json`. OSMesa is a system dependency and is not redistributed here.

The original code checkouts were:

| Component | Commit |
|---|---|
| Xiaomi-Robotics-1 | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| RoboCasa | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |

**Portable entry point.** See [PORTABLE_INFERENCE.md](PORTABLE_INFERENCE.md) for CPU artifact verification, localhost GPU RPC serving, and connection through the original multiplex evaluation client. `b2000_portable.py` is a new publication loader. It strictly verifies both sidecars, base assets, tensor identity, and the original core source hashes. It preserves FP32 adapter/branch master tensors, BF16 base forward/autocast, disabled TF32, deterministic algorithms, original action generation, and the original per-client RNG implementation. It avoids the old loader's hardcoded training-checkpoint directory requirements.

The new loader has passed CPU structural checks using both actual sidecars and an explicitly synthetic base actor. The preserved historical receipt is `cpu-validation.json`; it predates the A2000 error-label rename and the new download/materialization helper. The naming edit changes only an error-message label. The preparation helper passed synthetic file-safety checks and a real offline materialization of all 18 original base assets plus two sidecars, reconstructing the pinned base identity. No model execution or rollout was performed by that materialization check. **The new loader has not been validated by real-model GPU action parity or a new 2,500-episode evaluation.** Its CPU synthetic action check is not a benchmark score. Original evaluation results remain tied to their original entry point and configuration, rather than being retroactively attributed to this new wrapper.

**Original evaluation source.** `source_snapshot/` preserves the inference, training-support, and evaluation source bytes used for the audit. `evidence/source-snapshot-sha256.json` hashes the published snapshot. The native evaluation entry point is `source_snapshot/native_reset_b2500_v2/runner.py`; related renderer and evaluation support live under `source_snapshot/official_b2500_v1/`. Those archival supervisors retain absolute project paths and historical prerequisite checks. They are provided for exact-source inspection, not advertised as a portable one-command benchmark installer. Re-executing that archival entry point requires restoring its pinned environment, assets, bindings, and original path layout.

The historical `continuous_branch_v3/reset_fix.py` file appears in the source inventory, but the native-reset entry point explicitly rejects importing it. Presence in the archive is not evidence that the patch was executed. The published per-episode reset proofs refer to the original `counter.py` implementation.

**Provenance and limitations.** The individual evidence archive includes original absolute project paths, service identities, initial-scene metadata, and guard receipts. The final-audit timestamp and `submitted_to_leaderboard=false` describe the completed experiment before this publication request; they are preserved as historical records, not rewritten after submission. The paper and README describe current submission status. Raw private access credentials and unrelated project data are not part of this release.

The older paired comparison and the current unmatched author-run score do not establish a reliable improvement over the base. Organizer eligibility and evaluation decisions remain pending. The documentation and portable-loader checks enable review; they do not substitute for independent reproduction.
