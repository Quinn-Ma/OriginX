"""One observation-only intervention at a fixed policy-query boundary.

This module never accesses the simulator, success signal, state vectors, original
results, task implementation, or reset fingerprints. Its queue contains only the
original language instruction, three current RGB images, and remaining budget.
"""
import hashlib
import json
import math
from pathlib import Path
import secrets
import time

CAMERAS = (
    ('left', 'video.robot0_agentview_left'),
    ('right', 'video.robot0_agentview_right'),
    ('wrist', 'video.robot0_eye_in_hand'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_response(value, request_id, request_sha256, request_token):
    if not isinstance(value, dict):
        raise ValueError('Response must be an object')
    expected = dict(schema='astra_observation_response_v1', request_id=request_id,
                    request_sha256=request_sha256, request_token=request_token, model='gpt-6-astra')
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('Response identity/model/request digest differs')
    if value.get('status') not in ('ok', 'error'):
        raise ValueError('Response status must be ok/error')
    if value['status'] == 'ok':
        text = value.get('subgoal_instruction')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1500:
            raise ValueError('Response needs a nonempty bounded subgoal_instruction')
        if any(ord(c) < 32 and c not in '\n\t' for c in text):
            raise ValueError('Instruction contains unsupported control characters')
    return value


class AssistanceClient:
    def __init__(self, client, *, request_root, episode_output, request_id,
                 case_id, arm, horizon, wait_seconds=1200, intervention_fraction=0.5):
        if arm not in ('control', 'astra'):
            raise ValueError('Unknown intervention arm')
        if intervention_fraction != 0.5:
            raise ValueError('This protocol fixes the intervention at half horizon')
        if wait_seconds <= 0:
            raise ValueError('Response wait must be positive')
        self.client = client
        self.request_root = Path(request_root)
        self.episode_output = Path(episode_output)
        self.request_id, self.case_id, self.arm = request_id, case_id, arm
        self.horizon, self.wait_seconds = horizon, wait_seconds
        self.trigger_step = math.ceil(horizon * intervention_fraction / 16) * 16
        if self.trigger_step >= horizon:
            raise ValueError('Intervention must leave some official horizon')
        self.calls = 0
        self.instruction = None
        self.record = dict(arm=arm, trigger_step=self.trigger_step, requested=False,
                           status='control' if arm == 'control' else 'not_reached',
                           applied=False, wait_seconds=0.0,
                           intervention_rule='first fixed 16-step query at or after half horizon')

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _request(self, images, instruction, step):
        import numpy as np
        from PIL import Image
        folder = self.request_root / self.request_id
        folder.mkdir(parents=True, exist_ok=False)
        cameras = []
        for name, key in CAMERAS:
            array = np.asarray(images[key][-1])
            if array.shape != (256, 256, 3) or array.dtype != np.uint8:
                raise ValueError('Expected current official uint8 256x256 RGB')
            image_path = folder / (name + '.png')
            Image.fromarray(np.ascontiguousarray(array)).save(image_path, format='PNG')
            cameras.append(dict(name=name, file=image_path.name, sha256=sha(image_path)))
        request = dict(schema='astra_observation_request_v1', request_id=self.request_id,
                       request_token=secrets.token_hex(16),
                       case_id=self.case_id, step=step, horizon=self.horizon,
                       remaining_steps=self.horizon-step, original_instruction=instruction,
                       cameras=cameras,
                       permitted_inputs=['original_instruction', 'current_rgb', 'step_budget'],
                       oracle_inputs_included=False)
        request_path = folder / 'request.json'
        write(request_path, request)
        digest = sha(request_path)
        self.record.update(requested=True, status='waiting', request_path=str(request_path),
                           request_sha256=digest, request_token=request['request_token'],
                           step=step, original_instruction=instruction)
        write(self.episode_output / 'assistance.json', self.record)
        started = time.monotonic()
        response_path = folder / 'response.json'
        response = None
        while time.monotonic() - started < self.wait_seconds:
            if response_path.exists():
                try:
                    response = validate_response(json.loads(response_path.read_text(encoding='utf-8')),
                                                 self.request_id, digest, request['request_token'])
                    self.record.update(response_sha256=sha(response_path), response=response)
                    self.record['status'] = 'applied' if response['status'] == 'ok' else 'broker_error'
                except Exception as error:
                    self.record.update(status='invalid_response', error=repr(error))
                break
            time.sleep(0.5)
        else:
            self.record['status'] = 'timeout'
        self.record['wait_seconds'] = time.monotonic() - started
        if response is not None and self.record['status'] == 'applied':
            self.instruction = (instruction + '\nImmediate next actions: '
                                + response['subgoal_instruction'].strip()
                                + '\nThen complete the original task.')
            self.record.update(applied=True, policy_instruction=self.instruction)
        write(self.episode_output / 'assistance.json', self.record)

    def infer(self, state_history, image_history, instruction):
        import numpy as np
        def array_sha(value):
            array = np.asarray(value)
            return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()
        step = self.calls * 16
        if self.arm == 'astra' and step == self.trigger_step:
            self._request(image_history, instruction, step)
        self.calls += 1
        actual_instruction = self.instruction or instruction
        action = self.client.infer(state_history, image_history, actual_instruction)
        trace = dict(step=step, state_sha256=array_sha(state_history),
                     image_sha256={key:array_sha(image_history[key]) for _, key in CAMERAS},
                     instruction_sha256=hashlib.sha256(actual_instruction.encode('utf-8')).hexdigest(),
                     action_sha256=array_sha(action))
        with (self.episode_output / 'query_trace.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(trace, sort_keys=True, allow_nan=False) + '\n')
        return action

    def finish(self):
        self.record['policy_queries'] = self.calls
        write(self.episode_output / 'assistance.json', self.record)
        return self.record
