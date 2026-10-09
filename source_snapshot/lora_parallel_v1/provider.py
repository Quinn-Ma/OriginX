"""Ordered CPU preprocessing attached to the original FixedHumanProvider instance.

The object's class and serialized state are unchanged. Only delivered samples
commit parquet hashes/cursor; queued worker results are disposable on resume.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
from pathlib import Path
import re
import time
from types import MethodType

from lora_retention_run_v1.plan import SLOTS, require, sha256_file

_WORKER_PROOF = None


def _initialize_worker(config):
    """Spawn-only worker initializer; hide CUDA before importing the CPU reader."""
    global _WORKER_PROOF
    from cpu_prefetch_mechanical_v1.run import cpu_environment, initialize
    cpu_environment()
    for path, digest in config.get('processor_asset_pins', {}).items():
        require(sha256_file(path) == digest, 'Processor asset changed in worker: ' + path)
    _WORKER_PROOF = initialize(config)


def _prepare_worker(reference):
    from cpu_prefetch_mechanical_v1.run import prepare
    require(_WORKER_PROOF is not None, 'CPU worker was not independently initialized')
    result = prepare(reference)
    result['worker_initializer'] = _WORKER_PROOF
    return result


def _processor_pins(path):
    root = Path(path).resolve(strict=True)
    # Never load/hash5B weight shards to construct a CPU processor worker.
    files = sorted(p for p in root.iterdir() if p.is_file()
                   and p.suffix in ('.py', '.json', '.jinja', '.txt', '.model'))
    require((root / 'config.json') in files and files, 'Need original local processor assets')
    return {str(p): sha256_file(p) for p in files}


class _OrderedPrefetch:
    def __init__(self, provider, config, *, workers, prefetch, timeout_seconds,
                 executor_factory=None, prepare_fn=None, initialize_fn=None):
        require(type(workers) is int and 1 <= workers <= 16, 'CPU workers must be1..16')
        require(type(prefetch) is int and workers <= prefetch <= 32,
                'Prefetch must be between worker count and32 samples')
        require(timeout_seconds > 0, 'Positive worker timeout required')
        self.provider, self.config = provider, copy.deepcopy(config)
        self.workers, self.limit, self.timeout = workers, prefetch, timeout_seconds
        self.executor_factory = executor_factory
        self.prepare_fn = prepare_fn or _prepare_worker
        self.initialize_fn = initialize_fn or _initialize_worker
        self.pool = None
        self.pending = deque()
        self.next_submit = provider.cursor
        self.failed = None
        self.closed = False
        self.peak_pending = 0
        self.delivered = 0
        self.worker_pids = set()
        self.wait_seconds = 0.0
        self.original_load = provider.load_state_dict
        self.original_next = provider.next_batch

    def _start(self):
        require(not self.closed, 'Prefetch provider is closed')
        if self.pool is None:
            if self.executor_factory is not None:
                self.pool = self.executor_factory()  # CPU fixtures only.
            else:
                # Never fork a process that owns a CUDA context/model.
                self.pool = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=self.initialize_fn, initargs=(self.config,))
            self.next_submit = self.provider.cursor

    def _fill(self):
        self._start()
        while len(self.pending) < self.limit and self.next_submit < SLOTS:
            index = self.next_submit
            row = self.provider.slots[index]
            item = {k: row[k] for k in ('source','partition','task','episode','episode_length','start')}
            reference = {'dispatch_index':index, 'item':item}
            future = self.pool.submit(self.prepare_fn, reference)
            self.pending.append((index, future))
            self.next_submit += 1
            self.peak_pending = max(self.peak_pending, len(self.pending))

    def _validate_result(self, result, row):
        from cpu_prefetch_mechanical_v1.core import encoded_description
        import torch
        p = self.provider
        require(result['dispatch_index'] == p.cursor == row['slot_index'], 'Out-of-order delivery')
        require(result.get('cuda_initialized') is False, 'Worker initialized CUDA')
        require(type(result.get('pid')) is int and result['pid'] > 0, 'Invalid worker PID proof')
        identity = result['sample_identity']
        for k in ('source','partition','task','episode','episode_length','start'):
            require(identity.get(k)==row[k], 'Worker sample identity changed: ' + k)
        require(identity['archive_sha256']==row['human_archive_sha256']
                and identity['data_binding_sha256']==p.dataset.binding_sha256
                and identity['source_receipt_sha256']==p.dataset.source_receipt_pins['human/'+row['task']],
                'Worker source/archive/binding differs')
        digest = identity['parquet_sha256']
        require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest), 'Invalid parquet digest')
        key = f"human/{row['task']}/{row['episode']}"
        require(key not in p.dataset.parquet_hashes or p.dataset.parquet_hashes[key]==digest,
                'Worker read differs from an already committed parquet pin')
        data = result['data']
        require(encoded_description(data)==result['encoded'], 'IPC tensor bytes changed')
        inputs, actions, valid = data
        require(all(not isinstance(x,torch.Tensor) or x.device.type=='cpu' for x in inputs.values())
                and actions.device.type==valid.device.type=='cpu', 'Worker returned nonCPU tensors')
        require(tuple(actions.shape)==(1,16,60) and actions.dtype==torch.float32
                and tuple(valid.shape)==(1,16) and valid.dtype==torch.bool, 'Target/mask geometry changed')
        require(all(not x.is_floating_point() or x.dtype==torch.bfloat16
                    for x in inputs.values() if isinstance(x,torch.Tensor)), 'Processor inputs must match CUDA BF16 encoding')
        # Clone dictionary/identity; do not expose mutable worker bookkeeping.
        return key, digest, dict(inputs), actions, valid, copy.deepcopy(identity)

    def next_batch(self):
        p = self.provider
        require(self.failed is None, 'Prefetch failed; no implicit retry: ' + str(self.failed))
        require(p.cursor < SLOTS, 'Fixed80000 sample schedule exhausted')
        try:
            # Refill before consumption only. A submit failure can never occur
            #after committing a sample but before returning it to the trainer.
            self._fill()
            index, future = self.pending[0]
            require(index == p.cursor, 'Queue cursor differs from committed provider cursor')
            started = time.monotonic()
            result = future.result(timeout=self.timeout)
            future_wait = time.monotonic() - started
            self.wait_seconds += future_wait
            verify_started = time.monotonic()
            row = p.slots[index]
            key,digest,inputs,actions,valid,identity = self._validate_result(result,row)
            verify_seconds = time.monotonic() - verify_started
            import torch
            copy_started = time.monotonic()
            inputs = {k:v.to(device=p.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            actions, valid = actions.to(device=p.device), valid.to(device=p.device)
            copy_seconds = time.monotonic() - copy_started
            timing = dict(result.get('timings', {}), future_wait_seconds=future_wait,
                          IPC_verify_seconds=verify_seconds, H2D_host_wall_seconds=copy_seconds,
                          H2D_scope='Host call wall time; not CUDA event kernel time')
            # All worker validation and device copies complete before mutation.
            identity.update(slot_index=p.cursor,update=row['update_one_based'],micro_slot=row['micro_slot'],
                slot_seed=row['slot_seed'],task_visit=row['task_visit'],episode_visit=row['episode_visit'],
                sampling_config_sha256=p.config_sha256,slots_sha256=p.slots_sha256,arm=p.arm)
            p.dataset.parquet_hashes[key] = digest
            p.cursor += 1
            p.source_counts['human'] += 1
            p.last_sample = identity
            p.last_prefetch_timing = timing
            self.pending.popleft()
            self.delivered += 1
            self.worker_pids.add(result['pid'])
            return inputs,actions,valid,identity
        except BaseException as error:
            self.failed = repr(error)
            for _, future in self.pending:future.cancel()
            raise

    def load_state_dict(self, state):
        # Invalid input must not mutate either committed state or the queue.
        self.provider.validate_state_dict(state)
        self._shutdown(wait=True)
        self.original_load(state)
        self.provider.last_prefetch_timing = None
        self.next_submit = self.provider.cursor
        self.failed = None
        self.closed = False

    def _shutdown(self, *, wait):
        for _, future in self.pending:future.cancel()
        self.pending.clear()
        if self.pool is not None:
            self.pool.shutdown(wait=wait,cancel_futures=True)
            self.pool = None

    def close(self, *, wait=True):
        self._shutdown(wait=wait)
        self.closed = True

    def status(self):
        return {'workers':self.workers,'prefetch_limit':self.limit,'peak_pending':self.peak_pending,
                'submitted_cursor':self.next_submit,'committed_cursor':self.provider.cursor,
                'pending':len(self.pending),'delivered_this_controller':self.delivered,
                'observed_worker_pids':sorted(self.worker_pids),'wait_seconds':self.wait_seconds,
                'failed':self.failed,'closed':self.closed,'checkpoint_state_contains_prefetch':False}


def _next(self):return self._ordered_prefetch.next_batch()
def _load(self,state):return self._ordered_prefetch.load_state_dict(state)
def _close(self,*,wait=True):return self._ordered_prefetch.close(wait=wait)
def _status(self):return self._ordered_prefetch.status()


def attach_prefetch(provider, worker_config, *, workers=8, prefetch=16, timeout_seconds=900,
                    _executor_factory=None, _prepare_fn=None, _initialize_fn=None):
    """Attach to the exact original object; queue starts only on first next_batch.

    Original state_dict/validate_state_dict/class stay unchanged for old strict
    checkpoints. Record new source/worker settings in separate runtime provenance.
    """
    from lora_retention_run_v1.provider import FixedHumanProvider
    require(type(provider) is FixedHumanProvider, 'Require original exact FixedHumanProvider class')
    require(not hasattr(provider,'_ordered_prefetch'), 'Prefetch already attached')
    require(provider.cursor % 8 == 0, 'Attach only at an optimizer boundary')
    require(worker_config['data_binding']['sha256']==provider.dataset.binding_sha256, 'Wrong worker data binding')
    controller = _OrderedPrefetch(provider,worker_config,workers=workers,prefetch=prefetch,
        timeout_seconds=timeout_seconds,executor_factory=_executor_factory,prepare_fn=_prepare_fn,
        initialize_fn=_initialize_fn)
    provider._ordered_prefetch = controller
    provider.next_batch = MethodType(_next,provider)
    provider.load_state_dict = MethodType(_load,provider)
    provider.close = MethodType(_close,provider)
    provider.prefetch_status = MethodType(_status,provider)
    return provider


def make_provider(arm,binding_path,binding_sha256,processor,device,*,processor_path,
                  workers=8,prefetch=16,start_method='spawn',work_root=None,
                  timeout_seconds=900,processor_asset_pins=None,**kwargs):
    """Original make_provider arguments plus explicit processor_path and CPU window."""
    require(start_method=='spawn','Only spawn is permitted for CUDA trainer parent')
    from lora_retention_run_v1.provider import make_provider as serial_make_provider
    provider = serial_make_provider(arm,binding_path,binding_sha256,processor,device,**kwargs)
    config={'work_root':str(Path(work_root).resolve() if work_root else Path(__file__).resolve().parent.parent),
            'data_binding':{'path':str(Path(binding_path).resolve()),'sha256':binding_sha256},
            'processor':str(Path(processor_path).resolve()),
            'processor_asset_pins':processor_asset_pins or _processor_pins(processor_path)}
    return attach_prefetch(provider,config,workers=workers,prefetch=prefetch,timeout_seconds=timeout_seconds)
