# Execution-site change for the mask-cue study: AutoDL H800 -> HKUST cluster (disclosure note, v1)

Registered before any pilot or formal episode on the new site (2026-10-27, UTC).

## What changed and why

- The registered 6-case engineering pilot and the 12-pair formal run were to execute on a rented AutoDL
  H800 instance (`linux_egl` site, renderer `NVIDIA H800 PCIe/PCIe/SSE2`, driver 595.71.05). The pilot was
  launched there on 2026-10-26 23:06:50 UTC and the instance became unreachable about 20 minutes later
  (balance exhausted); no episode result was retrieved and the formal run never started.
- On 2026-10-27 the user asked for the study to be completed on the HKUST lab cluster (`hkust-cluster`,
  host `h800-8-1`, 8x NVIDIA H100 PCIe 80 GB, driver 570.195.03, CUDA 12.8). The manifest's
  `execution_sites.allowed` already lists `linux_egl`; the site is recorded at execution in
  `local-execution.json` / `site.json`, so the frozen pilot manifest (sha256 prefix `df91a4d28570`) and the
  frozen formal manifest (`94d4721e1452`) are used unchanged.

## Deployment identity on the cluster

- Code: the same frozen source snapshot `autodl-source-v2.tar.gz` (sha256 in `autodl-source-v2.json`)
  plus the two frozen manifests, under `/ephemeral/qinzhen/originx-harness-v2`.
- Vendor sources and assets: the same hash-verified archives as the AutoDL deployment
  (`stage-a-upstreams.tar.gz` 57cf637a..., `stage-a-assets.tar.gz` 6f7ed1c3..., `dev-seeds-assets-delta-v1.tar.gz`
  5dfabac2..., `formal-seeds-assets-delta-v1.tar.gz` 93a9b3f9...), extracted by `prepare_hkust_project.py`.
- Python environments: rebuilt with `uv` from the same frozen lists (`autodl-sim-requirements.lock`,
  `policy-cloud.requirements.txt`, `planner-cloud.requirements.txt`) on the system CPython 3.12.3 (the same
  minor/patch as the AutoDL base), torch 2.8.0+cu128 / torchvision 0.23.0+cu128 (the AutoDL base versions),
  the identical official flash-attn 2.8.3 cp312 wheel (sha256 f25da186...); installed freezes in
  `work/hkust/*-installed-freeze.txt`. The environments live on the RAM tier `/dev/shm/qinzhen/originx-envs`
  (regenerable; the disk `/ephemeral` was at 99 % with 115 GB free, below the lab's 150 GB floor).
- Model weights: policy `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` @ 3a6d0293 downloaded from
  huggingface.co with per-file SHA-256 admission against `models_manifest.json`; planner
  `Qwen/Qwen3-VL-4B-Instruct` @ ebb281ec reused from the cluster copy of 2026-10-22 after per-file SHA-256
  verification against the same manifest (`verified_reuse`). `download_receipt.json`: complete, 30/30 files.
  Aggregate `checkpoint_sha256` values are the manifest's (policy ccaa6517..., planner 6efa4249...), as
  checked by the drivers' identity admission at startup.
- Rendering: native NVIDIA EGL, renderer string probed and pinned as `NVIDIA H100 PCIe/PCIe/SSE2`
  (`work/hkust/site-overrides-hkust.json`); the registered zero-sample policy-RGB cohort
  (`policy_rgb_samples: 0`, amendment `rendering-amendment-h800-rgb0-v1.json`) applies to both arms.
  T3 (`check_mask_exactness.py`) is re-run on this site before the pilot
  (`work/hkust/t3/exactness-checks-hkust-h100-v1.json`); the H800 T3 result is not reused.
- EGL device <-> CUDA device mapping was verified empirically (`egl_cuda_map.py`, `egl_memtest.py`): EGL
  device index i is CUDA/nvidia-smi index i for i in 0..7, so `MUJOCO_EGL_DEVICE_ID == CUDA_VISIBLE_DEVICES`
  as the driver assumes.

## GPU selection (deviation from "2 workers on one GPU")

- GPUs 2 and 3 report `GPU Recovery Action: Reset` (pending since at least 2026-10-21). On them the
  depth readback is corrupt: the first T3 attempt on GPU 3 failed with
  `depth buffer must be finite and within [0, 1] (min -7.17e+21)`; `probe_hkust_depth.py` shows exactly one
  invalid depth pixel (OpenGL row 0, column 1) in every render on GPU 3 and thousands of invalid pixels in
  the lower-left 128x128 quadrant on GPU 2, in both the policy context and the single-sample mask context;
  the minimal MuJoCo-only probe (`probe_mujoco_min.py`) segfaults on GPUs 2 and 3 and is clean on GPU 5
  (0 invalid pixels, single-sample and MSAA). GPUs 0, 1, 5, 7 report `Recovery Action: None`.
- The two frozen workers are therefore placed one per healthy GPU (`--gpus 5,7 --workers-per-gpu 1`:
  worker 0 on GPU 5, worker 1 on GPU 7) instead of two on one GPU; each had about 26-30 GB free with
  other users' jobs resident (GPU 0/1 had more free memory but a 61/49 GB job of another user had just
  left them and may return). The manifest's worker count (2), case-to-worker assignment, order,
  per-case budget and `max_local_seconds` are unchanged. Nothing of other users was touched.
- Both execution GPUs were at 100 % utilisation from another user's training job; policy/planner inference
  shares them. Per-case wall time is reported from the run; the registered per-case budget (1,600 s) and
  watchdog were not changed.

## Pre-execution worker-count amendment (user instruction, 2026-10-27 ~16:30 UTC)

- The user instructed: use GPUs 5, 6 and 7 fully, break the 2-worker setting, run as many workers as fit
  (no other GPUs; other users' processes are never killed, but the user's jobs take priority over sharing).
  GPU 6 passed the same depth-readback probe as GPU 5 (0 invalid pixels in 18 renders, Recovery Action: None).
- Capacity: one policy+planner replica is ~20 GiB and each of GPU 5/6/7 had ~28-30 GB free beside another
  user's training job, so one worker per GPU: **3 workers** (`--gpus 5,6,7 --workers-per-gpu 1`).
- The 2-worker pilot launched at 16:31:53 UTC was stopped at ~16:35 UTC before any case completed (archived as
  `runs/id60-masks-pilot-hkust-v1.interrupted-*`, 0 results). The pilot and formal manifests were re-frozen with
  `--workers 3` by the frozen drivers themselves: pilot `df91a4d28570` -> `7c89c95a12e7`
  (`pilot-manifest-hkust-w3.json`, max 14,400 s unchanged), formal `94d4721e1452` -> `8266cc633102`
  (`manifest-hkust-w3.json`, derived `max_local_seconds` 21,000 -> 14,600 s because the busiest worker now has
  8 cases). Verified identical between old and new: seeds, pair_index, pair_position, comparison arm, arm,
  task, model, max_steps and pair_key for every case; only `workers`, `worker_assignment`, `execution_budget`,
  the case_id hashes (which include the worker fields by construction) and `manifest_sha256` changed
  (`work/hkust/preexecution-worker-amendment-v1.json`). The README/paper hash references must be updated to
  the new prefixes when the results are reported.
- First 3-worker launch (16:36:39 UTC) was refused by the driver because planner port 20292 was still held by a
  planner service of the stopped 2-worker chain; all 6 pilot cases were recorded `unattempted` (archived as
  `runs/id60-masks-pilot-hkust-w3-v1.unattempted-*`); the lingering service (ours) ended, and the chain was
  relaunched at 16:38:16 UTC (`PILOT_START` 16:38:24 UTC).

## Controls kept

No source file, threshold, prompt, precision or seed was changed; the frozen manifests reproduce the
registered hashes on the cluster; GPU placement, environment rebuild and host are execution-site details
recorded here, in `transfer-and-site-receipt.json`, `site.json` and `local-execution.json` of each run.
Outcomes of the AutoDL attempt (none retrieved) are not pooled with anything.
