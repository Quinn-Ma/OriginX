"""One persistent GR00T episode connection; LazyClient/EntryBridge compatible."""
import hashlib
import json
from pathlib import Path
import secrets
import socket
import threading
import time
import numpy as np
from .adapter import CAMERAS,STATE_DIMS,ACTION_DIMS,LANGUAGE,PROTOCOL
from .core import CONTROL,POLICY_ID
from . import wire

IDENTITY_KEYS=('protocol','policy_id','server_instance','model_config_sha256','model_assets_sha256',
               'model_config_runtime_sha256','asset_source_identity_sha256','denoising_steps','replan_steps','observation_frames',
               'adapters_loaded','serial_forward','isolated_rng_stream','rng_isolation','gpu_uuid')

def validate_hello(actual,expected):
    if actual.get('protocol')!=PROTOCOL or actual.get('policy_id')!=POLICY_ID:
        raise ValueError('Wrong independent GR00T identity')
    if any(actual.get(k)!=expected.get(k) for k in IDENTITY_KEYS):raise ValueError('Pinned service identity differs')
    if (actual.get('adapters_loaded') is not False or actual.get('denoising_steps')!=4 or
        actual.get('replan_steps')!=16 or actual.get('observation_frames')!=1 or
        actual.get('serial_forward') is not True or actual.get('isolated_rng_stream') is not True):
        raise ValueError('Unexpected policy execution semantics')

def load_profile(path,digest):
    from .server import BASE,GPU_UUID,sha,process_identity,validate_gr00t_owner,PINS
    path=Path(path)
    if not path.resolve().is_relative_to(BASE/'services') or path.is_symlink() or sha(path)!=digest:
        raise ValueError('Profile path/hash outside admitted service namespace')
    if (path.parent/'draining.json').exists():raise RuntimeError('Service is draining')
    profile=json.loads(path.read_text())
    if profile.get('ready')is not True or profile.get('schema')!='originx_gr00t_service_profile_v1':raise ValueError('Unready profile')
    hello=profile['hello'];validate_hello(hello,hello)
    if hello['gpu_uuid']!=GPU_UUID or hello['model_assets_sha256']!={k:v[1] for k,v in PINS.items()}:
        raise ValueError('Wrong checkpoint/GPU profile')
    validate_gr00t_owner(profile['owner'],process_identity(profile['owner']['pid']))
    return dict(profile,server_manifest=str(path),server_manifest_sha256=digest)

class GR00TClient:
    def __init__(self,profile,timeout=1200,telemetry_path=None):
        self.profile=profile;self.wire=self;self.seed=None;self.infer_calls=0;self.closed=False
        self._thread=threading.get_ident();self.telemetry=Path(telemetry_path).open('x') if telemetry_path else None
        self.socket=socket.create_connection(('127.0.0.1',profile['port']),timeout=timeout);self.socket.settimeout(timeout)
        try:
            self.hello=self.control('hello');validate_hello(self.hello,profile['hello'])
        except BaseException:self.close();raise
    def _request(self,command,**kwargs):
        if self.closed or threading.get_ident()!=self._thread:raise RuntimeError('Closed or different-thread socket')
        nonce=secrets.token_hex(16);request=dict(kwargs,**{CONTROL:command,'protocol':PROTOCOL,'request_nonce':nonce})
        start=time.monotonic();payload=wire.encode(request)
        try:
            wire.send(self.socket,payload);response_bytes=wire.receive(self.socket);response=wire.decode(response_bytes)
            if not isinstance(response,dict) or response.get('ok')is not True or response.get('request_nonce')!=nonce:raise RuntimeError('Invalid acknowledgment')
            if hasattr(self,'hello'):
                validate_hello(response,self.hello)
                if response.get('connection_id')!=self.hello['connection_id']:raise RuntimeError('Connection identity changed')
            if self.telemetry:
                self.telemetry.write(json.dumps(dict(command=command,wall_seconds=time.monotonic()-start,request_bytes=len(payload)+4,response_bytes=len(response_bytes)+4))+'\n');self.telemetry.flush()
            return response
        except BaseException:self.close();raise
    def control(self,command,**kwargs):
        if command not in ('hello','reset_rng','rng_state'):raise ValueError('Unknown control')
        return self._request(command,**kwargs)
    def reset(self,seed):
        if self.seed is not None:raise RuntimeError('One reset per episode connection')
        if type(seed)is not int or not 0<=seed<2**63:raise ValueError('Invalid seed')
        ack=self.control('reset_rng',seed=seed)
        if ack['seed']!=seed or ack['requests_since_reset']!=0:raise RuntimeError('Bad seed acknowledgment')
        self.seed=seed;return ack
    def get_action(self,observation):
        if self.seed is None:raise RuntimeError('Seed before inference')
        if set(observation)!=set(CAMERAS)|set(STATE_DIMS)|{LANGUAGE}:raise ValueError('Only official named GR00T inputs')
        native={}
        for key in set(CAMERAS)|set(STATE_DIMS):
            v=np.asarray(observation[key])
            if v.shape[0]!=1:raise ValueError('Official single current frame only')
            native[key]=np.ascontiguousarray(v[0])
        text=observation[LANGUAGE]
        if len(text)!=1 or not isinstance(text[0],(str,np.str_)):raise ValueError('One task description required')
        response=self._request('infer',observation=native,instruction=str(text[0]))
        if response['seed']!=self.seed or response['requests_since_reset']!=self.infer_calls+1:
            self.close();raise RuntimeError('Inference stream count/seed changed')
        actions=response['actions']
        if actions.shape!=(16,12) or actions.dtype!=np.float32 or not np.isfinite(actions).all():
            self.close();raise RuntimeError('Malformed decoded actions')
        self.infer_calls+=1;self.last_rng=response;result={};start=0
        for key,dim in ACTION_DIMS.items():result[key]=actions[:,start:start+dim].copy();start+=dim
        return result
    def close(self):
        if not self.closed:
            self.closed=True
            if hasattr(self,'socket'):self.socket.close()
            if self.telemetry:self.telemetry.close()
