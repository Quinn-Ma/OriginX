# Xiaomi RoboCasa365 baseline

Prepared 2026-10-03 from Xiaomi commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`. These helpers do not change the model or official evaluator. They were checked locally; actual model execution requires the remote GPU host.

## Fast path on the existing host

Reuse `/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365` after checking its three weight shards, index, model code and processor files. Reuse existing environments only after checking package versions and importing the model/processor. Put new code, logs, caches and results in `/ephemeral/qinzhen/robocasa-xr1-20261003`; the root disk is full. Do not alter existing experiments or allocate occupied GPUs. Human300 training data is unnecessary for baseline evaluation.

| Process | Verified upstream requirement / recommended compatible setup |
|---|---|
| Inference | Python 3.12, torch 2.8.0 CUDA 12.8, torchvision 0.23.0, torchaudio 2.8.0, transformers **4.57.1**, flash-attn 2.8.3 |
| Simulator / processor | Python 3.11, RoboCasa 1.0.1, robosuite master, transformers **4.57.1**, imageio[ffmpeg], scipy; torch 2.7.1 + torchvision 0.22.1 satisfy LeRobot 0.3.3's requirements |
| Rendering | MUJOCO_GL=egl, libegl1/libgl1/libgles2; official launcher deliberately leaves MUJOCO_EGL_DEVICE_ID unset |

The simulator's `lerobot==0.3.3` requires `torch<2.8` and `torchvision<0.23`. Keep it separate from inference. Local existing RoboCasa environment uses torch 2.7.1, torchvision 0.22.1, torchcodec 0.5, mujoco 3.3.1, numpy 2.2.5 and numba 0.61.2; this is dependency evidence, not a verified XR-1 run. Local RoboCasa SHA is `456174f62b89b8fca99eaaf33949c29fec9cfc2a`; local robosuite SHA is `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Capture actual remote SHAs before evaluating.

Check model size and available GPU memory with `nvidia-smi`; published checkpoint is approximately 10.1 GB. Do not download it again if existing files validate. Assets are approximately 10 GB compressed according to RoboCasa docs, and must include textures, generative textures, lightwheel fixtures, and three object families. Asset download scripts can swallow download failures, so a rendering smoke test is essential.

## Start owned services and run

Copy these helpers and the pinned official checkout to the new workspace. Set absolute paths, and select currently free physical GPU IDs. Example below uses five GPUs, as only these were free at the last root-agent check; recheck immediately before launch.

```bash
export XR_REPO=/ephemeral/qinzhen/robocasa-xr1-20261003/code/Xiaomi-Robotics-1
export MODEL_PATH=/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365
export SERVER_PYTHON=/absolute/path/to/inference/bin/python
export SIM_PYTHON=/absolute/path/to/simulator/bin/python
export OUT_ROOT=/ephemeral/qinzhen/robocasa-xr1-20261003/results
export BASE_PORT=11086 WORKERS=5

python3 start_servers.py --repo "$XR_REPO" --model "$MODEL_PATH" \
  --python "$SERVER_PYTHON" --gpus 1,3,5,6,7 --base-port "$BASE_PORT" \
  --run-dir /ephemeral/qinzhen/robocasa-xr1-20261003/server-runs/baseline-001

bash run_baseline.sh smoke
# Launch full evaluation in an owned tmux session or a detached job with its log.
# Opt in to low-frame-rate failure videos for diagnosis, or omit for maximum throughput.
SAVE_FAILURE_VIDEOS=1 bash run_baseline.sh full
```

`start_servers.py` checks port availability, launches one official server per GPU, uses localhost only, logs each server, and waits at most 900 seconds for readiness. The original `scripts/deploy.sh` unconditionally kills the `model_servers` tmux session and hardcodes its environment; this helper avoids that interference. Failed startup stops only the processes it just created.

The smoke episode has horizon 20 and tests end-to-end inference/rendering, **not** task success. Full evaluation uses 50 tasks × 50 episodes, pretrain split, target50, history 4, interval 2, replanning 16, crop 0.95, seed 7, and unmodified registry horizons. Xiaomi reports 1432/2500 = 57.28% with its public code, while the leaderboard entry is 57.4%. Do not replace missing episodes with failures or count a partial result as the baseline.

The official scheduler can use fewer than eight workers without changing task/episode seed assignments. The checkpoint's `modeling_mibot.py` seeds a persistent PyTorch RNG during import: `MIBOT_SERVER_SEED` if set, otherwise `7 + first visible numeric CUDA device` (UUID selectors fall back to 7). This helper explicitly assigns seed `7 + logical server index` and records it, keeping seeds stable across physical GPU allocation. `--seed-base` overrides that declared base. This matches the official initial seeds for the standard eight-worker GPU 0..7 mapping, but dynamic scheduling still changes which jobs consume each stream. Preserve worker count and process/package metadata for comparison. Stop owned servers with `python3 stop_owned_servers.py /path/to/servers.json`, then start a fresh group before the full baseline so the smoke test does not consume its RNG stream. The stop helper verifies process starttime and exact command, and uses pidfds where available to prevent PID reuse errors.

## Monitoring and failures

```bash
RUN_ID="$(cat "$OUT_ROOT/latest-full.txt")"
python3 status.py "$OUT_ROOT/scheduler/$RUN_ID"
tail -n 30 "$OUT_ROOT/scheduler/$RUN_ID/logs/worker-0.log"
```

Queue paths: `pending/`, `running/`, `results/`, `errors/`, `logs/`. `status.py` validates job IDs and episode seeds, prints per-task and category counts, and marks incomplete output clearly. An interim overall rate is biased by which tasks finished first. Only the official final `summary.json` plus 2500 valid results is a completed baseline. Review all errors and workers; a server never becoming available can otherwise leave the official client retrying forever.

Each full run gets a new ID and queue. Existing result directories are preserved. If a worker dies, do not relaunch the official initialization against the same queue. Retain its evidence, diagnose the cause, and explicitly recover missing infrastructure-failed jobs or start a clearly identified new run. Never rerun an ordinary policy failure and substitute its better result. Save failures and successes alike in the official denominator.

The evaluator already emits per-task horizon, steps, success, global episode ID and seed; failure videos are optional. A later failure taxonomy must be based on observations/video, not success flags alone.

## New environment fallback

In new dedicated environments, follow the official setup. Download wheels/caches to the ephemeral disk. For FlashAttention on Python 3.12 / torch 2.8, the official deployment guide links a prebuilt wheel, avoiding a slow build:

```text
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Install the simulator's torch/torchvision matching pair first, then robosuite and RoboCasa, then transformers 4.57.1 and imageio[ffmpeg]. Run `pip check`; do not silently upgrade the server's torch or the simulator's pinned physics stack. Capture the resolved HF revision if a new checkpoint download is needed (`HfApi().model_info(...).sha` then `snapshot_download(..., revision=sha)`).

## Sources

- [Official Xiaomi evaluation guide](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/main/eval_robocasa365/README.md)
- [Deployment versions and FlashAttention wheel](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/main/docs/DEPLOYMENT.md)
- [Official checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365)
- [RoboCasa installation](https://robocasa.ai/docs/build/html/introduction/installation.html)
