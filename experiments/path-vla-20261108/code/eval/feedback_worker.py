"""Native worker for the SECONDARY feedback study: the F adapter's own location head draws a box on the current frame
of the NEXT query (one-chunk delay). Inference only; no optimizer. Protocol identical to trained_inference_worker.py,
plus a 'predicted_locations' field in each response and optional 'feedback_boxes' in each request."""
from pathlib import Path
import argparse, importlib.metadata, pickle, sys, time, traceback
ROOT = Path('/dev/shm/qinzhen/lerobot-three-model-adapters-20261106')
TRAIN = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
sys.path.insert(0, str(TRAIN / 'scripts'))
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
HALF = 16  # half side of the drawn box in pixels (32x32 box on a 256x256 frame)


def paint(img, cxy):
    """Draw a 2-px red rectangle of fixed size around the predicted centre (normalized xy) on a HWC uint8 frame."""
    import numpy as np
    im = np.array(img, copy=True); h, w = im.shape[:2]
    cx, cy = int(round(cxy[0] * (w - 1))), int(round(cxy[1] * (h - 1)))
    x1, y1, x2, y2 = max(0, cx - HALF), max(0, cy - HALF), min(w - 1, cx + HALF), min(h - 1, cy + HALF)
    im[y1:y1 + 2, x1:x2 + 1] = [255, 0, 0]; im[max(y1, y2 - 1):y2 + 1, x1:x2 + 1] = [255, 0, 0]
    im[y1:y2 + 1, x1:x1 + 2] = [255, 0, 0]; im[y1:y2 + 1, max(x1, x2 - 1):x2 + 1] = [255, 0, 0]
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', choices=['xr1', 'awomo_i', 'awomo_v'], required=True)
    ap.add_argument('--adapter', required=True); args = ap.parse_args(); variant = args.variant
    protocol_out = sys.stdout.buffer; sys.stdout = sys.stderr
    import torch, numpy as np
    torch.set_num_threads(2)
    start = time.perf_counter()
    try:
        from training_core import TrainingCore
        core = TrainingCore(variant); core.load_adapter(args.adapter)  # hook stays registered: features are captured at inference
        if variant == 'xr1':
            sys.path.insert(0, str(ROOT / 'src/xiaomi'))
            from eval_robocasa365.entry import EvalClient, observation_to_state
            processor = core.processor; model = core.native; native = model
            builder = EvalClient.__new__(EvalClient); builder.crop_ratio = .95
        else:
            native = core.policy
            if variant == 'awomo_i':  # the training-time hook needs the token layout that encode() produces
                _enc = core.native.encode
                def _enc_rec(sample):
                    c = _enc(sample); core.history_len = c['history'].shape[1]; core.reference_len = c['reference'].shape[1]; return c
                core.native.encode = _enc_rec
            if variant == 'awomo_v':  # training captured the head feature only during the observation-only warm-up pass
                _wu = core.native.warm_up
                def _wu_capture(*a, **k):
                    core.capture_clean = True
                    try:
                        return _wu(*a, **k)
                    finally:
                        core.capture_clean = False
                core.native.warm_up = _wu_capture
        mods ={'policy': core.native} if variant == 'xr1' else {k: v for k, v in vars(core.native).items() if isinstance(v, torch.nn.Module)}
        devices = {n: sorted(set(str(p.device) for p in m.parameters())) for n, m in mods.items()}
        receipt = {'ok': True, 'variant': variant, 'load_seconds': time.perf_counter() - start, 'devices': devices,
                   'loading': {'adapter': args.adapter, 'feedback': 'own location head, one-chunk delay, 32px box'}, 'training_started': False,
                   'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                   'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers']}}
        pickle.dump(receipt, protocol_out, protocol=5); protocol_out.flush()
        while True:
            try:
                request = pickle.load(sys.stdin.buffer)
            except EOFError:
                break
            if request.get('command') == 'close':
                break
            try:
                tic = time.perf_counter(); seed = int(request['seed']); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                fb = request.get('feedback_boxes')  # list of 3 normalized centres or None
                core.feature = None
                with torch.inference_mode():
                    if variant == 'xr1':
                        obs = [dict(x) for x in request['observations']]
                        if fb is not None:
                            for i, c in enumerate(CAMERAS):
                                obs[-1]['video.' + c] = paint(obs[-1]['video.' + c], fb[i])
                        if variant == 'xr1':
                            if hasattr(core, 'capture_clean'):
                                pass
                        states = np.stack([observation_to_state(x) for x in obs])
                        state = np.zeros((1, 4, 60), dtype=np.float32); state[0, :, :14] = states
                        images = {k: np.stack([x[k] for x in obs]) for k in obs[0] if k.startswith('video.')}
                        inputs = processor.apply_chat_template(builder._build_messages(images, request['instruction']), tokenize=True,
                                                               return_dict=True, return_tensors='pt', do_resize=False, state=state, robot_type='robocasa365')
                        inputs['task_id'] = 'robocasa365'
                        inputs = {k: (v.to(device='cuda:0', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0')) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        raw = model(**inputs).actions
                        actions = processor.decode_action(raw.cpu(), robot_type='robocasa365')[0, :, :12].float().numpy()[:16]
                    else:
                        req = dict(request)
                        if fb is not None:
                            frames = np.array(req['camera_frames'], copy=True)  # (n_new, 3, H, W, 3)
                            for i in range(3):
                                frames[-1, i] = paint(frames[-1, i], fb[i])
                            req['camera_frames'] = frames
                        out = native.predict(req); actions = out['actions']
                    loc = None
                    if core.feature is not None:
                        loc = core.location_head(core.feature.float()).sigmoid().reshape(-1, 3, 2)[0].detach().cpu().numpy().tolist()
                actions = np.asarray(actions, dtype=np.float32)
                assert actions.shape == (16, 12) and np.isfinite(actions).all(), actions.shape
                torch.cuda.synchronize()
                out = {'ok': True, 'actions_native': actions, 'elapsed_seconds': time.perf_counter() - tic, 'predicted_locations': loc,
                       'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9}
            except Exception:
                out = {'ok': False, 'error': traceback.format_exc()}
            pickle.dump(out, protocol_out, protocol=5); protocol_out.flush()
    except Exception:
        pickle.dump({'ok': False, 'error': traceback.format_exc()}, protocol_out, protocol=5); protocol_out.flush(); return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
