"""Original observation-only assistance plus serialized same-wire RNG keepalive.

No reconnect, reset, background thread, simulator access or additional inference.
The broker request and response schemas, 1200-second budget and policy instruction
construction are unchanged. Protocol receipts stay outside the broker request.
"""
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
import time

from astra_rescue_20261009.assistance import (
    AssistanceClient as OriginalAssistanceClient, CAMERAS, sha, write, validate_response,
)

INTERVAL = 120.0
STATE_FIELDS = ('server_instance', 'connection_id', 'protocol', 'model_config_sha256',
                'seed', 'requests_since_reset', 'python_rng_sha256', 'numpy_rng_sha256',
                'torch_cpu_rng_sha256', 'torch_cuda_rng_sha256')


def normalized_state(response):
    state = {key: response[key] for key in STATE_FIELDS}
    if state['protocol'] != 'robocasa-multiplex-rng-v1':
        raise RuntimeError('Unexpected keepalive protocol')
    if type(state['seed']) is not int or type(state['requests_since_reset']) is not int:
        raise RuntimeError('Keepalive seed/count must be exact integers')
    for key in ('server_instance', 'connection_id'):
        if not isinstance(state[key], str) or not state[key]:
            raise RuntimeError('Missing keepalive connection identity')
    hashes = [state[k] for k in ('model_config_sha256', 'python_rng_sha256',
                                'numpy_rng_sha256', 'torch_cpu_rng_sha256')]
    if not isinstance(state['torch_cuda_rng_sha256'], list) or not state['torch_cuda_rng_sha256']:
        raise RuntimeError('Missing CUDA RNG fingerprints')
    hashes += state['torch_cuda_rng_sha256']
    if any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None for x in hashes):
        raise RuntimeError('Malformed keepalive RNG/model digest')
    return state


class AssistanceClient(OriginalAssistanceClient):
    # Inherit the frozen constructor, infer(), trace serialization and finish().
    def _keepalive(self, phase):
        thread = threading.get_ident()
        if getattr(self, '_wait_thread', thread) != thread:
            raise RuntimeError('Keepalive must use the waiting/inference thread')
        self._wait_thread = thread
        wire = self.client.policy.wire
        if getattr(self, '_wait_wire', wire) is not wire:
            raise RuntimeError('Keepalive wire changed; reconnect forbidden')
        self._wait_wire = wire
        started = time.monotonic()
        row = dict(schema='astra_same_socket_keepalive_v1', phase=phase,
                   sequence=getattr(self, '_keepalive_count', 0), control='rng_state',
                   same_thread=True, reconnect=False, reset=False)
        try:
            response = wire.control('rng_state')
            state = normalized_state(response)
            for key in ('connection_id', 'server_instance', 'model_config_sha256', 'protocol'):
                if state[key] != wire.hello[key]:
                    raise RuntimeError('Keepalive identity differs from admitted hello: ' + key)
            if state['seed'] != self.client.seed or state['requests_since_reset'] != self.client.infer_calls:
                raise RuntimeError('Keepalive changed policy seed/request count')
            baseline = getattr(self, '_keepalive_baseline', None)
            if baseline is not None and state != baseline:
                raise RuntimeError('Keepalive changed same-connection RNG/count/identity')
            if baseline is None:
                self._keepalive_baseline = state
            row.update(status='ok', state=state, response=response,
                       response_sha256=hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest())
            self._keepalive_count = row['sequence'] + 1
            self.record['keepalive'] = dict(status='ok', interval_seconds=INTERVAL,
                                           calls=self._keepalive_count, same_socket=True,
                                           rng_and_count_unchanged=True, reconnect=False, reset=False)
        except BaseException as error:
            row.update(status='error', error=repr(error))
            self.record['keepalive'] = dict(status='error', error=repr(error),
                                           interval_seconds=INTERVAL, reconnect=False, reset=False)
            self.record['status'] = 'keepalive_error'
            raise
        finally:
            row['wall_seconds'] = time.monotonic() - started
            with (self.episode_output/'policy-keepalive.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            write(self.episode_output/'assistance.json', self.record)

    def _request(self, images, instruction, step):
        self._keepalive('enter')
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
                       request_token=secrets.token_hex(16), case_id=self.case_id, step=step,
                       horizon=self.horizon, remaining_steps=self.horizon-step,
                       original_instruction=instruction, cameras=cameras,
                       permitted_inputs=['original_instruction', 'current_rgb', 'step_budget'],
                       oracle_inputs_included=False)
        request_path = folder/'request.json'
        write(request_path, request)
        digest = sha(request_path)
        self.record.update(requested=True, status='waiting', request_path=str(request_path),
                           request_sha256=digest, request_token=request['request_token'],
                           step=step, original_instruction=instruction)
        write(self.episode_output/'assistance.json', self.record)
        started = time.monotonic()
        next_keepalive = started + INTERVAL
        response_path = folder/'response.json'
        response = None
        while time.monotonic() - started < self.wait_seconds:
            if time.monotonic() >= next_keepalive:
                self._keepalive('periodic')
                next_keepalive = time.monotonic() + INTERVAL
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
        # Always verify after waiting and before the next policy forward, including
        # immediate replies, broker errors and timeouts. No socket request overlaps.
        self._keepalive('exit')
        if response is not None and self.record['status'] == 'applied':
            self.instruction = (instruction + '\nImmediate next actions: '
                                + response['subgoal_instruction'].strip()
                                + '\nThen complete the original task.')
            self.record.update(applied=True, policy_instruction=self.instruction)
        write(self.episode_output/'assistance.json', self.record)
