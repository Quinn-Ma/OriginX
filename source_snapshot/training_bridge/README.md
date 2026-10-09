# HF-compatible RoboCasa365 training bridge

This is a minimal **training-readiness component**, not a trained policy or a completed reproduction. It uses the released RoboCasa365 HF model directly. It does not modify its inference source, add Choice heads, use the generic dual-arm trainer, download datasets, or start data training.

The local CPU suite passes. Real checkpoint gradients and HF round-trip compatibility must still be tested on an available H100 using `smoke_gpu.py`. The GPU script uses synthetic images/states and **zero optimizer steps**. Passing it says nothing about policy success.

## Files

- `layout.py`: exact controller/state conversions, official history and crop/prompt, and a single-episode sample adapter.
- `bridge.py`: a separate flow-matching loss with frozen VLM and trainable DiT/projections, gradient checks, and HF export.
- `smoke_gpu.py`: real HF forward/backward on synthetic observations; optional HF save, reload and exact seeded inference parity.
- `tests/`: lightweight CPU checks, including direct comparisons against extracted official evaluator functions.

## Run checks

From the workspace root, without installing this directory:

```bash
PYTHONPATH=work python3 -m pytest -q work/training_bridge/tests
```

After copying this directory to the server, activate the existing XR-1 deployment environment with compatible Torch, Transformers **4.57.1**, and Flash Attention. Use one GPU confirmed idle by the parent. Provide the **complete local checkpoint**, including tokenizer, video processor and custom Python files; `work/hf_robocasa_metadata/` alone is not complete.

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> python /path/training_bridge/smoke_gpu.py \
  --model /path/Xiaomi-Robotics-1-RoboCasa365 \
  --report /path/results/gradient_smoke.json
```

To additionally prove that a saved artifact loads through the official HF interface, choose a new export directory with enough space for another checkpoint copy:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> python /path/training_bridge/smoke_gpu.py \
  --model /path/Xiaomi-Robotics-1-RoboCasa365 \
  --export-dir /path/checkpoints/bridge_smoke_export \
  --report /path/results/gradient_and_reload_smoke.json
```

These are command templates; replace angle-bracket/path placeholders. The script requires local files and refuses to overwrite a report or nonempty export. `--attn-implementation sdpa` is available for troubleshooting but is not the official runtime. The script does not install packages, contact a remote model hub intentionally, or claim benchmark equivalence from synthetic data.

Success requires finite loss, finite nonzero gradient norm in **every** trainable component, no VLM gradients, and (when requested) identical processor tensors and exactly equal generated actions after HF save/reload with the same seed. A real GPU run can expose version-specific issues not covered by the toy CPU model; its JSON report is the evidence, not this README.

## Exact input contract

The HF processor statistics select an action tensor of `(B,16,60)`. The first 12 dimensions have mean 0/std 1; remaining dimensions have std 0. This bridge checks those statistics and refuses an incompatible processor.

- Raw state is EE position 3, EE axis-angle 3, gripper qpos 2, base position 3, base axis-angle 3, then 46 zeros. There is no quantile normalization.
- Standard LeRobot state is 16D, base-first, with XYZW quaternions. `lerobot_state_to_policy` normalizes/sign-canonicalizes both quaternions exactly like the official evaluator.
- Standard LeRobot action is base motion 4, control mode 1, EE position command 3, EE rotation command 3, gripper 1. `lerobot_action_to_policy` reorders it to EE6/gripper1/base4/mode1 and pads to 60. These are controller commands; do not subtract the current pose or convert them into a second relative action representation.
- History is `[max(0,t-6), max(0,t-4), max(0,t-2), t]` for state and left/right/wrist cameras. The prompt, `/no_cot`, empty `<cot></cot>` assistant turn, 0.95 center crop and resize are copied in behavior from the official evaluator.
- Future actions are `t:t+16`. At an episode boundary the final action is repeated for storage, but only actual future timesteps receive loss. Histories never cross episodes.

`make_sample(states, actions, frame_reader, timestep, instruction)` adapts one episode-local step. `states` and `actions` are aligned `(T,16)` and `(T,12)` arrays. `frame_reader(camera_key, indices)` returns four RGB `uint8` arrays or RGB PIL images in the requested order, including repeated indices. Its output is `EpisodeSample`: messages, state `(1,4,60)`, target `(1,16,60)`, and boolean future validity `(1,16)`.

`encode_sample` uses the real HF processor and moves only tensors to the selected device, preserving integer token/grid types. Floating inputs follow the official server's cast: BF16 on CUDA by default, FP32 on CPU, with an explicit `floating_dtype` override if needed. Action targets remain FP32. It is a **single-sample** adapter. This release does not implement multi-episode batching or packed video sequences. The loss supports correctly assembled B-sized batches, but rejects the common bug where the processor produces only a single action mask for several text examples.

## Dataset reader specification

No dataset reader or training manifest is selected implicitly. A production reader must:

1. Build and persist a manifest using `split=pretrain`, source `human`, and the exact `PRETRAINING_TASKS['pretrain300']` task allowlist. Reject `target` paths and composite-unseen task names. Record archive/task/episode IDs and source hashes. The official downloader defaults to **target**, so never rely on its defaults.
2. Verify each archive's `meta/modality.json` with `validate_modality` before applying the fixed state/action conversions. Resolve parquet, video paths, timestamps and task descriptions through that archive's metadata; do not assume future LeRobot formats retain old paths.
3. Read episode-local aligned low-dimensional arrays. Convert `annotation.human.task_description` through the metadata mapping to the actual instruction string. Use videos already rendered by the dataset; no rerender or re-encoding is needed for this bridge.
4. Cache video readers and use indexed frame access; avoid materializing all Human300 images or per-frame Python dictionaries. Preserve frame ordering and timestamps. Confirm observation at `t` precedes action at `t` via one recorded demonstration playback.
5. Pass the episode-local arrays and frame callback to `make_sample`, then `encode_sample`. Keep training and development episode/seed manifests separate. The baseline must finish and the development protocol must be approved by the parent before actual data training.

## Loss, masks and precision

For a normalized controller target `a` (identity normalization here), sample standard normal `z` and **uniform** `t ∈ [0,1)`. Input `(1-t)z + t*a`, supervise velocity `a-z`, and take mean squared error only over valid future timesteps × the first 12 action dimensions. Uniform time is an explicit new bridge default; it is **not asserted to be the original Xiaomi training schedule**. Optional `noise` and `times` tensors make tests deterministic.

Baseline inference samples and evolves all 60 action coordinates but multiplies noisy action by the fixed 12-coordinate mask before projecting it. This bridge also samples 60-dimensional noise and uses that unchanged mask in `dit_forward`. It excludes inactive coordinates from loss rather than diluting loss over 48 padding dimensions. The temporal validity mask affects **loss only**; it does not change the HF input mask or causal structure. Because DiT queries are causal, padded future queries cannot influence preceding valid queries.

The VLM is frozen, forced to evaluation mode and executed under `torch.no_grad()` (not `inference_mode`, because DiT attention backward saves its constant K/V tensors). Calling `vlm.model` avoids allocating vocabulary logits. State projection remains trainable. The exact HF query layout is sink + 4 state tokens + 16 action tokens, with contiguous positions after VLM positions and a causal lower-triangular DiT attention mask joined to the VLM padding mask. No extra action-token turn or +10 position offset is inserted.

Trainable modules are `dit`, `state_projector`, `action_projector`, `action_output_layer`, `t_embedder`, `t_projector`, and `sink`. They use FP32 master parameters by default with BF16 CUDA autocast; the loss and velocity target subtraction use FP32. This allows ordinary AdamW to keep FP32 optimizer state when real training is later implemented. Optimizers must consume `bridge.trainable_parameters()`, not all model parameters. This directory supplies **no optimizer loop**, DDP implementation, checkpoint resumption, recovery-data collection or learning-rate recipe.

The wrapper owns the original model but does not add model-internal parameters. `export_hf` calls the underlying model's `save_pretrained`, so inference weights have their original names, without a `model.` wrapper prefix. It preserves processor/tokenizer/custom-code assets and does not copy a stale shard index after resharding. For the gradient-only smoke, heads are explicitly cast back to BF16 before reference inference/export. For actual training, preserve FP32 master weights and optimizer state separately; derive a BF16 inference export without casting away the live training master's precision.

## Scope and provenance

The source audit is `work/training_readiness.md`; this bridge targets the locally downloaded official RoboCasa365 HF source and the official evaluation client at Xiaomi repository commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`. It deliberately adds no frequency loss, Choice loss, asynchronous prefix objective, learned memory, or task-value mechanism. Those require separate experiments after baseline and data validation. No benchmark score or leaderboard gain has been measured by this work.
