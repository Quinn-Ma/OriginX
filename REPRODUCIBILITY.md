# Reproducing and auditing OriginX

**Release contents.** The prepared OriginX release package contains `adapter-step-00001613.pt`, `branch-00002000.pt`, `evaluation-evidence.tar.gz`, and `SHA256SUMS`. The adaptation files are the exact artifacts used by the frozen evaluation, not newly trained weights. The base is obtained separately from Xiaomi. Neither training datasets nor base-model weight shards are republished.

Download the exact adaptation files and evidence from the [OriginX v1.0.0 release](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0). Place both adaptation `.pt` files in `weights/`. A Hugging Face mirror is planned under the name `OriginX`, but no Hugging Face URL has been verified; this release does not claim that mirror is available.

```bash
mkdir -p weights
python verify_evidence.py evaluation-evidence.tar.gz
```

`verify_evidence.py` needs only the Python standard library. It validates the original archive, per-episode result hashes, complete denominator, fixed assignment identities, reset proofs, failed-episode horizons, and reconstructed counts. The expected output is 2,500 episodes, 1,496 successes, and 1,004 failures. The unchanged original evidence can also be inspected directly; the verification command does not perform simulation or run model inference.

**Frozen upstream assets.** Download the complete, clean snapshot of `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` at revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`. Do not add or edit model-directory Python, JSON, tokenizer, or processor files. The portable loader reconstructs the same base-asset identity used in training and verifies the model tensor identity after injection.

```bash
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4 \
  --local-dir assets/base
```

**Environment.** The original inference environment was Linux, Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.57.1, and FlashAttention 2.8.3. [requirements-inference.txt](requirements-inference.txt) records the inspected relevant package versions, rather than claiming to be a complete operating-system lockfile. Install the matching CUDA PyTorch build before building FlashAttention; follow the upstream Xiaomi and FlashAttention installation instructions for the GPU platform. The simulator was a separate Python 3.11 environment; its package pins are in `evidence/config.json`. OSMesa is a system dependency and is not redistributed here.

The original code checkouts were:

| Component | Commit |
|---|---|
| Xiaomi-Robotics-1 | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| RoboCasa | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |

**Portable entry point.** See [PORTABLE_INFERENCE.md](PORTABLE_INFERENCE.md) for CPU artifact verification, localhost GPU RPC serving, and connection through the original multiplex evaluation client. `b2000_portable.py` is a new publication loader. It strictly verifies both sidecars, base assets, tensor identity, and the original core source hashes. It preserves FP32 adapter/branch master tensors, BF16 base forward/autocast, disabled TF32, deterministic algorithms, original action generation, and the original per-client RNG implementation. It avoids the old loader's hardcoded training-checkpoint directory requirements.

The new loader has passed CPU structural checks using both actual sidecars and an explicitly synthetic base actor. The receipt is `cpu-validation.json`. **The new loader has not been validated by real-model GPU action parity or a new 2,500-episode evaluation.** Its CPU synthetic action check is not a benchmark score. Original evaluation results remain tied to their original entry point and configuration, rather than being retroactively attributed to this new wrapper.

**Original evaluation source.** `source_snapshot/` preserves the inference, training-support, and evaluation source bytes used for the audit. `evidence/source-snapshot-sha256.json` hashes the published snapshot. The native evaluation entry point is `source_snapshot/native_reset_b2500_v2/runner.py`; related renderer and evaluation support live under `source_snapshot/official_b2500_v1/`. Those archival supervisors retain absolute project paths and historical prerequisite checks. They are provided for exact-source inspection, not advertised as a portable one-command benchmark installer. Re-executing that archival entry point requires restoring its pinned environment, assets, bindings, and original path layout.

The historical `continuous_branch_v3/reset_fix.py` file appears in the source inventory, but the native-reset entry point explicitly rejects importing it. Presence in the archive is not evidence that the patch was executed. The published per-episode reset proofs refer to the original `counter.py` implementation.

**Provenance and limitations.** The individual evidence archive includes original absolute project paths, service identities, initial-scene metadata, and guard receipts. The final-audit timestamp and `submitted_to_leaderboard=false` describe the completed experiment before this publication request; they are preserved as historical records, not rewritten after submission. The paper and README describe current submission status. Raw private access credentials and unrelated project data are not part of this release.

The older paired comparison and the current unmatched author-run score do not establish a reliable improvement over the base. Organizer eligibility and evaluation decisions remain pending. The documentation and portable-loader checks enable review; they do not substitute for independent reproduction.
