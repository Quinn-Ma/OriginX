"""One model thread and per-connection RNG, independent GR00T protocol."""
from concurrent.futures import Future
import hashlib
import json
import pickle
import queue
import random
import re
import threading
import numpy as np
from .adapter import CAMERAS, STATE_DIMS, PolicyAdapter, PROTOCOL
from .wire import encode,decode

POLICY_ID='gr00t_n15_robocasa_multitask120000'
CONTROL='_gr00t_control'

class RngBank:
    """Same capture/restore algorithm as admitted multiplex_inference.core."""
    def __init__(self,torch):
        self.torch=torch
        self.cuda=torch.cuda.is_available()
    def capture(self):
        t=self.torch
        return (random.getstate(),np.random.get_state(),t.get_rng_state().clone(),
                [s.clone() for s in t.cuda.get_rng_state_all()] if self.cuda else [])
    def restore(self,state):
        random.setstate(state[0]);np.random.set_state(state[1]);self.torch.set_rng_state(state[2])
        if self.cuda:self.torch.cuda.set_rng_state_all(state[3])
    def seed(self,seed):
        if type(seed)is not int or not 0<=seed<2**63:raise ValueError('Invalid stream seed')
        random.seed(seed);np.random.seed(seed%(2**32));self.torch.manual_seed(seed)
        if self.cuda:self.torch.cuda.manual_seed_all(seed)
    @staticmethod
    def hashes(state):
        h=lambda b:hashlib.sha256(b).hexdigest()
        return dict(python_rng_sha256=h(pickle.dumps(state[0],protocol=5)),
            numpy_rng_sha256=h(pickle.dumps(state[1],protocol=5)),
            torch_cpu_rng_sha256=h(state[2].cpu().numpy().tobytes()),
            torch_cuda_rng_sha256=[h(s.cpu().numpy().tobytes()) for s in state[3]])

class SerialExecutor:
    def __init__(self,policy_factory,torch,identity,capacity=12):
        if identity.get('policy_id')!=POLICY_ID or identity.get('protocol')!=PROTOCOL:
            raise ValueError('Wrong policy family or protocol')
        self.identity=dict(identity,serial_forward=True,isolated_rng_stream=True,
                           rng_isolation='python_numpy_torch_cpu_all_cuda_per_connection')
        self.torch=torch;self.factory=policy_factory;self.queue=queue.Queue(capacity)
        self.sessions={};self.closed=False;self.started=threading.Event();self.startup_error=None
        self.thread=threading.Thread(target=self._work,name='gr00t-serial-forward',daemon=True)
        self.thread.start()
        if not self.started.wait(900):raise RuntimeError('Model thread startup timeout')
        if self.startup_error:raise RuntimeError('Model thread startup failed') from self.startup_error

    def submit(self,connection,payload=None,*,close=False):
        if self.closed:raise RuntimeError('Executor closed')
        future=Future();self.queue.put((connection,payload,close,future),timeout=30);return future
    def close(self):
        if not self.closed:
            self.closed=True;self.queue.put(None,timeout=30);self.thread.join(120)
            if self.thread.is_alive():raise RuntimeError('Model thread did not drain')
    def _work(self):
        try:
            self.bank=RngBank(self.torch);self.policy=self.factory();self.adapter=PolicyAdapter(self.policy)
            self.identity.update(model_execution_thread_name=threading.current_thread().name)
            config=getattr(getattr(self.policy,'model',None),'config',None)
            self.identity['model_config_runtime_sha256']=(hashlib.sha256(
                json.dumps(config.to_dict(),sort_keys=True,default=str).encode()).hexdigest() if config is not None else None)
        except BaseException as error:self.startup_error=error
        finally:self.started.set()
        if self.startup_error:return
        while True:
            item=self.queue.get()
            if item is None:return
            cid,payload,close,future=item
            try:
                if close:self.sessions.pop(cid,None);result=None
                else:result=self._handle(cid,payload)
                future.set_result(result)
            except BaseException as error:future.set_exception(error)
    def _handle(self,cid,payload):
        if threading.current_thread() is not self.thread:raise RuntimeError('Model thread changed')
        session=self.sessions.setdefault(cid,dict(state=None,count=0,poisoned=False))
        if session['poisoned']:raise RuntimeError('Connection poisoned; no retry')
        ambient=self.bank.capture()
        try:
            if session['state'] is not None:self.bank.restore(session['state'])
            r=decode(payload)
            if not isinstance(r,dict) or r.get('protocol')!=PROTOCOL:raise ValueError('Wrong protocol')
            command=r.get(CONTROL)
            if command not in ('hello','reset_rng','rng_state','infer'):raise ValueError('Unknown command')
            extra={'seed'} if command=='reset_rng' else {'observation','instruction'} if command=='infer' else set()
            if set(r)!={CONTROL,'protocol','request_nonce'}|extra:raise ValueError('Unexpected command arguments')
            if not isinstance(r['request_nonce'],str) or not re.fullmatch('[0-9a-f]{32}',r['request_nonce']):raise ValueError('Invalid nonce')
            response=dict(self.identity,ok=True,connection_id=cid,request_nonce=r['request_nonce'])
            if command=='reset_rng':
                if session['state'] is not None:raise RuntimeError('Exactly one reset per connection')
                self.bank.seed(r['seed']);session.update(state=self.bank.capture(),seed=r['seed'])
            elif command=='infer':
                if session['state'] is None:raise RuntimeError('Seed before inference')
                if not isinstance(r['observation'],dict) or set(r['observation'])!=set(CAMERAS)|set(STATE_DIMS):
                    raise ValueError('Only named policy observations are accepted')
                actions=self.adapter.infer_native(r['observation'],r['instruction'])
                session.update(state=self.bank.capture(),count=session['count']+1)
                response['actions']=actions
            if command in ('reset_rng','rng_state','infer'):
                if session['state'] is None:raise RuntimeError('Seed before RNG state')
                response.update(seed=session['seed'],requests_since_reset=session['count'],**self.bank.hashes(session['state']))
            return encode(response)
        except BaseException:
            session['poisoned']=True
            raise
        finally:self.bank.restore(ambient)
