"""Adapter for the secondary feedback study: feeds the policy's own predicted target locations back as a drawn box."""
from pathlib import Path
import os, pickle, sys, uuid
import numpy as np
import torch
ROOT = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
NATIVE = Path('/dev/shm/qinzhen/lerobot-three-model-adapters-20261106')
SBL = Path('/dev/shm/qinzhen/sbl')
sys.path.insert(0, str(NATIVE))
from adapter import NativeRoboCasaAdapter, CAMERAS, PERM, decode_observation


class FeedbackAdapter(NativeRoboCasaAdapter):
    def __init__(self, variant, adapter_path, physical_gpu=0, seed=4242):
        if variant not in ['xr1', 'awomo_i', 'awomo_v'] or physical_gpu == 3:
            raise ValueError('Unsupported model/GPU')
        adapter_path = Path(adapter_path).resolve(strict=True)
        self.variant = variant; self.seed = seed; self.physical_gpu = physical_gpu
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HOME=str(NATIVE / 'cache' / variant),
                   XDG_CACHE_HOME=str(ROOT / 'tmp/cache'), TMPDIR=str(ROOT / 'tmp'), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
        self.log = (SBL / 'logs' / ('feedback-worker-%s-%s.log' % (variant, uuid.uuid4().hex[:8]))).open('wb')
        self.worker = __import__('subprocess').Popen([str(NATIVE / ('env-' + variant) / 'bin/python'), '-u', str(SBL / 'eval/feedback_worker.py'),
                                                      '--variant', variant, '--adapter', str(adapter_path)],
                                                     stdin=__import__('subprocess').PIPE, stdout=__import__('subprocess').PIPE, stderr=self.log, env=env, cwd=ROOT)
        try:
            self.load_receipt = self.receive(600)
        except Exception:
            self.close(); raise
        self.reset()

    def reset(self):
        super().reset(); self.last_locations = None; self.location_trace = []

    @torch.inference_mode()
    def select_action(self, batch):
        obs, state = decode_observation(batch)
        task = batch['task']; task = task[0] if isinstance(task, (list, tuple)) else task
        if self.instruction is None:
            self.instruction = str(task)
        self.history.append(obs)
        if self.step_index % 4 == 0:
            self.new_frames.append((self.step_index, np.stack([obs['video.' + c] for c in CAMERAS])))
        if not self._action_queue:
            replan = self.step_index // 16
            request = {'seed': self.seed + replan, 'instruction': self.instruction, 'session_id': self.session_id, 'replan_index': replan,
                       'feedback_boxes': self.last_locations}
            if self.variant == 'xr1':
                h = list(self.history); request['observations'] = [h[max(0, len(h) - 1 - (3 - i) * 2)] for i in range(4)]
            else:
                request.update(observation_steps=[x[0] for x in self.new_frames], camera_frames=np.stack([x[1] for x in self.new_frames]), state=state)
            self.last_request = request
            pickle.dump(request, self.worker.stdin, protocol=5); self.worker.stdin.flush()
            response = self.receive(240); native = response.pop('actions_native')
            self.last_locations = response.get('predicted_locations'); self.location_trace.append(self.last_locations)
            self.last_native_actions = native.copy(); self._action_queue.extend(native[:, PERM].copy())
            self.query_receipts.append(response); self.new_frames = []
        action = self._action_queue.popleft(); self.step_index += 1
        return torch.from_numpy(action.copy())[None]
