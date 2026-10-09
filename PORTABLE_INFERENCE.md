# Portable frozen B2000 inference

`b2000_portable.py` loads the published A1613 LoRA adapter followed by the B2000
online stage branch. It imports the original core classes unchanged and replaces
only the deployment-specific artifact locator. It needs no optimizer state,
training data, historical checkpoints, or original absolute project directory.

Use the released base model `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, pinned
to Hugging Face revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`. Download that
complete snapshot into a clean directory: the original full base-asset identity
and original loaded-tensor identity are both checked, in addition to sidecar SHA256.
Do not replace the base with a later revision or alter its processor/config files.

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
  --adapter weights/adapter-step-00001613.pt \
  --branch weights/branch-00002000.pt \
  --output cpu-validation.json
```

This validates both exact sidecars, all tensor names/shapes/dtypes, finite values,
the adapter's internal checksum, A/B shared base identity, and real B branch
weights. A synthetic base actor exercises the unchanged StageInferenceView's
one-VLM/five-Euler hook contract, `[1,16,60]` action shape, repeated deterministic
results, cleanup and rejection of oracle stage inputs. It is not a real-model
GPU action-parity test and does not repeat the 2,500 rollout benchmark.

## GPU serving (explicit user execution)

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python b2000_portable.py serve \
  --runtime-root source_snapshot --base-model assets/base \
  --adapter weights/adapter-step-00001613.pt \
  --branch weights/branch-00002000.pt --device cuda:0 --port 18000
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
  --adapter weights/adapter-step-00001613.pt \
  --branch weights/branch-00002000.pt \
  --fixture fixtures/first_request.json --seed 940001 --output action-check.pt
```

For a new GPU validation, compare all 60 action dimensions and each RNG hash
against the original frozen endpoint's first request for that exact fixture and
seed. Then compare both original three-request sequences with seeds 940001 and
940002, and run native-reset smoke episodes. No such new GPU execution was
performed while preparing this release; the archived original endpoint parity
evidence and new CPU structural test are different evidence.
