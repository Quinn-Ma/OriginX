# Portable frozen B2000 inference

`b2000_portable.py` loads the published A2000 LoRA adapter followed by the B2000
online stage branch. It imports the original core classes unchanged and replaces
only the deployment-specific artifact locator. It needs no optimizer state,
training data, historical checkpoints, or original absolute project directory.

## Prepare assets from Hugging Face and GitHub

The model snapshot is [Qinzhen3/OriginX](https://huggingface.co/Qinzhen3/OriginX). Python code stays in this GitHub checkout. `prepare_originx.py` combines 15 pinned non-code assets from the model snapshot with three pinned original Python model files in `upstream_model_code/`. It verifies all 18 sizes and hashes and reconstructs the exact original base identity. It also verifies and copies the two adaptation files. Install `huggingface_hub` for download mode, or use the offline snapshot option:

```bash
python prepare_originx.py --revision main --output-dir assets
# Or reuse an already downloaded complete snapshot, with no network call:
python prepare_originx.py --snapshot /path/to/OriginX-snapshot --output-dir assets
```

For a reproducible download, replace `main` with the recorded OriginX Hugging Face commit SHA. Preparation fails if that revision is incomplete. Outputs are `assets/base/`, containing exactly the original 18 identity assets, and `assets/weights/`, containing `adapter-originx-2000.pt` and `branch-00002000.pt`. Different existing files are never overwritten. Download mode caches under `assets/.hf-cache/`; allow disk space for both cached and materialized base weights. No GPU or training is used by preparation.

A2000 is the release alias for the unchanged historical A1613 adapter (1,613 actual optimizer updates, 80,000 sampled windows). B2000 is the subsequent 2,000-update continuous branch. The underlying Xiaomi base remains revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`; do not substitute later assets or edit the assembled model files. Do not place this preparation helper, release manifest, README, or new configuration files inside `assets/base/`.

The original project uses CUDA, PyTorch, Transformers and FlashAttention 2. Use
the captured original environment requirements supplied with the release. GPU
serving keeps base weights in BF16, LoRA and branch master tensors in FP32, and
uses BF16 autocast during the original one-VLM/five-Euler forward. Never cast the
whole model to BF16 after injecting the adapter. Requests must be serialized.

Use `source_snapshot/` as the runtime root. The adjacent
`runtime-source-provenance.json` pins every required original source byte and
records which files were present in the old endpoint source inventory.

## CPU validation

```bash
python b2000_portable.py verify \
  --runtime-root source_snapshot \
  --adapter assets/weights/adapter-originx-2000.pt \
  --branch assets/weights/branch-00002000.pt \
  --output local-cpu-validation.json
```

This optional CPU command validates both exact sidecars, all tensor names/shapes/dtypes, finite values,
the adapter's internal checksum, A/B shared base identity, and real B branch
weights. A synthetic base actor exercises the unchanged StageInferenceView's
one-VLM/five-Euler hook contract, `[1,16,60]` action shape, repeated deterministic
results, cleanup and rejection of oracle stage inputs. It is not a real-model
GPU action-parity test and does not repeat the 2,500 rollout benchmark. The existing
`cpu-validation.json` is the preserved historical receipt; write any new check to
`local-cpu-validation.json` instead of overwriting it.

## GPU serving (explicit user execution)

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python b2000_portable.py serve \
  --runtime-root source_snapshot --base-model assets/base \
  --adapter assets/weights/adapter-originx-2000.pt \
  --branch assets/weights/branch-00002000.pt --device cuda:0 --port 18000
```

The server binds only `127.0.0.1`, reusing the original multiplex server and its
isolated per-client Python/NumPy/Torch CPU/CUDA RNG states. It uses the multiplex
protocol, not the upstream raw-pickle client's protocol without its adapter.
The existing `multiplex_inference.client.MultiplexEvalClient` provides the
upstream official image preprocessing and action decoding plus this wire adapter:

```python
import sys
sys.path.insert(0, "source_snapshot")
from multiplex_inference.client import MultiplexEvalClient

client = MultiplexEvalClient("path/to/Xiaomi-Robotics-1", "assets/base", 18000)
client.reset(seed=2090900000)  # once, after the environment's natural reset
actions = client.eval.infer(state_history, image_history, instruction)
client.close()
```

Use a fresh client/reset for each episode. The original XR-1 code checkout is
pinned to commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`. A full benchmark
reproduction must also retain the published per-episode manifest, native reset,
RoboCasa version/assets, pretrain split, official horizons, history 4/interval 2,
crop 0.95 and replan 16. The helper above is a portable inference interface, not
a claim that the old absolute-path benchmark supervisor itself is portable.

## GPU parity strategy

The `infer-fixture` mode loads one previously captured official
`first_request.json`/`.npz` fixture with its original checksums, seeds the original
RNG bank, and writes the full action tensor plus final RNG hashes:

```bash
python b2000_portable.py infer-fixture \
  --runtime-root source_snapshot --base-model assets/base \
  --adapter assets/weights/adapter-originx-2000.pt \
  --branch assets/weights/branch-00002000.pt \
  --fixture fixtures/first_request.json --seed 940001 --output action-check.pt
```

For a new GPU validation, compare all 60 action dimensions and each RNG hash
against the original frozen endpoint's first request for that exact fixture and
seed. Then compare both original three-request sequences with seeds 940001 and
940002, and run native-reset smoke episodes. No such new GPU execution was
performed while preparing this release; the archived original endpoint parity
evidence and new CPU structural test are different evidence.
