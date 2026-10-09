"""Two independent CUDA replicas, one AdamW owner, explicit8/32-slot updates.

No model creation, GPU selection, thread launch, or optimizer step on import.
The caller owns loading/health, exact slot order, per-slot draws, and checkpoint
identity. This execution changes floating-point reduction order; it is not a
bitwise continuation of the serial-eight implementation.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
import hashlib
import json
import random
import threading
import time

import numpy as np
import torch


def require(value, message):
    if not value:
        raise ValueError(message)


def rng_fingerprint(devices):
    """Read-only global states; local per-slot Generators are not these streams."""
    def digest(value):return hashlib.sha256(value).hexdigest()
    numpy=np.random.get_state()
    numpy=[numpy[0],numpy[1].tolist(),int(numpy[2]),int(numpy[3]),float(numpy[4])]
    return dict(python=digest(repr(random.getstate()).encode()),
        numpy=digest(json.dumps(numpy,separators=(',',':')).encode()),
        torch_cpu=digest(torch.random.get_rng_state().numpy().tobytes()),
        torch_cuda={str(d):digest(torch.cuda.get_rng_state(d).cpu().numpy().tobytes()) for d in devices})


def adapter_parameters(bridge):
    """Only the declared trainable FP32 LoRA tensors, in stable name order."""
    bridge._assert_trainable_contract()
    named=dict(bridge.model.named_parameters())
    names=sorted(bridge.handle.adapter_names)
    require(names and set(names)=={n for n,p in named.items() if p.requires_grad},'Only adapter parameters may be trainable')
    result=[(n,named[n]) for n in names]
    require(all(p.dtype==torch.float32 and not p.is_meta for _,p in result),'Materialized FP32 adapter masters required')
    require(len({p.device for _,p in result})==1,'One device per independent replica')
    return result


def check_pair(master, replica):
    require([n for n,_ in master]==[n for n,_ in replica],'Replica adapter names differ')
    require(all(a.shape==b.shape and a.dtype==b.dtype==torch.float32 for (_,a),(_,b) in zip(master,replica)),'Replica adapter geometry/dtype differs')
    require(all(a is not b for (_,a),(_,b) in zip(master,replica)),'Replicas cannot share Parameter objects')


@torch.no_grad()
def sync_adapters(master, replica):
    """One contiguous transfer, followed by local copies; no original weights."""
    check_pair(master,replica)
    flat=torch.cat([p.detach().reshape(-1) for _,p in master])
    copied=flat.to(device=replica[0][1].device,non_blocking=False)
    offset=0
    for _,p in replica:
        p.copy_(copied[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    return flat.numel()*flat.element_size()


@torch.no_grad()
def sum_replica_gradients(master, replica):
    """Add already globally-scaled replica gradients; do NOT average again."""
    check_pair(master,replica)
    require(all(p.grad is not None for _,p in master+replica),'Every declared adapter needs a gradient')
    require(all(p.grad.dtype==torch.float32 and p.grad.shape==p.shape for _,p in master+replica),'Bad adapter gradient dtype/shape')
    flat=torch.cat([p.grad.detach().reshape(-1) for _,p in replica])
    require(torch.isfinite(flat).all(),'Nonfinite replica gradient')
    copied=flat.to(device=master[0][1].device,non_blocking=False)
    offset=0
    for _,p in master:
        p.grad.add_(copied[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    return flat.numel()*flat.element_size()


def run_rank(bridge,batches,device,compute_fn,global_batch=8):
    """Single replica worker; caller must not mutate global RNG concurrently."""
    device=torch.device(device)
    context=torch.cuda.device(device) if device.type=='cuda' else nullcontext()
    results=[]
    with context:
        for batch in batches:
            count=int(batch['actions'].shape[0])
            # One equal weight per sample, independent of chunk/rank grouping.
            result=compute_fn(bridge,batch['inputs'],batch['actions'],batch['valid'],
                noise=batch['noise'],times=batch['times'],loss_scale=count/global_batch)
            require(result['samples']==count,'Gradient helper sample count differs')
            results.append(result)
    return results


class TwoDeviceEngine:
    """A synchronous step on two separately loaded models, each on one GPU.

    step([rank0_batches, rank1_batches]); each rank consumes global_batch//2
    samples in B2/4/8/16 dictionaries. Each dictionary contains inputs,
    actions, valid, noise, times, already on that rank's device. Rank0 precedes
    rank1 in original slot order. Per-sample seeded draws are generated BEFORE
    calling step. No provider/global RNG work is permitted while workers run.

    Failure is fail-stop: wait for both workers, preserve their exception and
    do not continue this engine. No retry, checkpoint, model/CPU-worker cleanup
    or process signalling is hidden here. Checkpoints belong to master only,
    after a successful step; caller records the explicit two-replica identity.
    """
    def __init__(self,master_bridge,replica_bridge,optimizer,*,global_batch=8,max_grad_norm=1.0,compute_fn=None):
        require(type(global_batch) is int and global_batch in (8,32),'Explicit global batch must be8 or32')
        self.global_batch=global_batch;self.rank_batch=global_batch//2
        require(master_bridge is not replica_bridge and master_bridge.model is not replica_bridge.model,'Independent bridge/model objects required')
        self.bridges=(master_bridge,replica_bridge)
        self.parameters=(adapter_parameters(master_bridge),adapter_parameters(replica_bridge))
        check_pair(*self.parameters)
        self.devices=tuple(params[0][1].device for params in self.parameters)
        require(all(d.type=='cuda' for d in self.devices) and self.devices[0]!=self.devices[1],'Two distinct CUDA devices required')
        require(master_bridge.retention_coefficient==replica_bridge.retention_coefficient,'Replicas must implement the same arm objective')
        require(master_bridge.handle.metadata['base_state_sha256']==replica_bridge.handle.metadata['base_state_sha256'],'Replicas must use identical original base weights')
        expected={id(p) for _,p in self.parameters[0]}
        actual=[p for group in optimizer.param_groups for p in group['params']]
        require(type(optimizer) is torch.optim.AdamW and len(actual)==len(expected) and {id(p) for p in actual}==expected,'Single master AdamW must own exactly all master adapters')
        require(type(max_grad_norm) in (int,float) and 0<max_grad_norm<float('inf'),'Finite positive clipping norm required')
        self.optimizer=optimizer;self.max_grad_norm=max_grad_norm
        if compute_fn is None:
            from .batched_bridge import compute_microbatch_gradients
            compute_fn=compute_microbatch_gradients
        self.compute_fn=compute_fn;self.failed=False;self.closed=False
        self._lock=threading.Lock();self.pool=None
        # This deliberately discards replica adapter initialization/old state;
        # the caller has already restored the exact master checkpoint.
        self.adapter_bytes=sync_adapters(*self.parameters)
        for device in self.devices:torch.cuda.synchronize(device)
        # Original checkpoint stores master CUDA RNG only. Recreate secondary
        # from master on each construction, then require neither is consumed.
        torch.cuda.set_rng_state(torch.cuda.get_rng_state(self.devices[0]),self.devices[1])
        self.initial_rng_fingerprint=rng_fingerprint(self.devices)
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='lora-independent-replica')

    def _batches(self,groups):
        require(len(groups)==2,'Exactly two rank groups required')
        prepared=[]
        for rank,group in enumerate(groups):
            batches=[group] if isinstance(group,dict) else list(group)
            require(batches and all(set(b)=={'inputs','actions','valid','noise','times'} for b in batches),'Each batch must explicitly carry independent noise/time')
            counts=[int(b['actions'].shape[0]) for b in batches]
            require(sum(counts)==self.rank_batch and all(n in (2,4,8,16) and n<=self.rank_batch for n in counts),'Each replica must consume exactly half global_batch as B2/4/8/16 chunks')
            for b in batches:
                require(all(v.device==self.devices[rank] for v in b['inputs'].values() if isinstance(v,torch.Tensor)),'Inputs on wrong replica device')
                require(all(b[k].device==self.devices[rank] for k in ('actions','valid','noise','times')),'Target/draws on wrong replica device')
                require(b['noise'].shape==b['actions'].shape and b['times'].shape==(b['actions'].shape[0],1,1),'Independent slot draws have wrong batched shape')
                self.bridges[rank]._validate_batch(b['inputs'],b['actions'],b['valid'])
            prepared.append(batches)
        return prepared

    def step(self,rank_batches):
        require(not self.closed and not self.failed,'Closed/failed engine cannot be reused')
        require(self._lock.acquire(blocking=False),'Engine is non-reentrant')
        futures=[];started=time.perf_counter()
        try:
            groups=self._batches(rank_batches)
            self.optimizer.zero_grad(set_to_none=True)
            self.bridges[1].model.zero_grad(set_to_none=True)
            for bridge in self.bridges:bridge.train()
            before_rng=rng_fingerprint(self.devices)
            before=time.perf_counter()
            for bridge,batches,device in zip(self.bridges,groups,self.devices):
                futures.append(self.pool.submit(run_rank,bridge,batches,device,self.compute_fn,self.global_batch))
            # Join both ranks even when one throws: no healthy thread is abandoned.
            wait(futures)
            results=[future.result() for future in futures]
            for device in self.devices:torch.cuda.synchronize(device)
            require(rng_fingerprint(self.devices)==before_rng,'Forward/backward consumed a global or replica CUDA RNG stream')
            fb_seconds=time.perf_counter()-before;reduce_start=time.perf_counter()
            copied=sum_replica_gradients(*self.parameters)
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in self.parameters[0]],self.max_grad_norm,error_if_nonfinite=True)
            self.optimizer.step()
            sync_adapters(*self.parameters)
            for device in self.devices:torch.cuda.synchronize(device)
            self.optimizer.zero_grad(set_to_none=True);self.bridges[1].model.zero_grad(set_to_none=True)
            after_rng=rng_fingerprint(self.devices)
            require(after_rng==before_rng,'Two-replica update consumed a global RNG stream')
            rows=[row for rank in results for row in rank]
            report={key:sum(row[key]*row['samples'] for row in rows)/self.global_batch for key in ('loss','flow_loss','retention_loss')}
            report.update({key:[value for row in rows for value in row[key]] for key in ('per_sample_loss','per_sample_flow_loss','per_sample_retention_loss')})
            report.update(samples=self.global_batch,global_batch=self.global_batch,active_values=sum(row['active_values'] for row in rows),gradient_norm=float(norm),
                forward_backward_wall_seconds=fb_seconds,reduce_optimizer_sync_wall_seconds=time.perf_counter()-reduce_start,
                wall_seconds=time.perf_counter()-started,gradient_transfer_bytes=copied,adapter_transfer_bytes=self.adapter_bytes,
                original_weights_synchronized=False,optimizer_owners=1,replicas=2,
                rng_unchanged=True,rng_before=before_rng,rng_after=after_rng,
                rank_samples=[self.rank_batch,self.rank_batch],sample_loss_global_weight=1/self.global_batch)
            report['per_replica']=[dict(device=str(device),samples=self.rank_batch,
                microbatch_sizes=[row['samples'] for row in rank],
                **{key:sum(row[key]*row['samples'] for row in rank)/self.rank_batch
                   for key in ('loss','flow_loss','retention_loss')})
                for device,rank in zip(self.devices,results)]
            return report
        except BaseException:
            self.failed=True
            if futures:wait(futures)
            raise
        finally:self._lock.release()

    def close(self):
        # The caller closes only after step has returned/raised, never mid-step.
        require(self._lock.acquire(blocking=False),'Cannot close during an active step')
        try:
            if not self.closed and self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=False)
            self.closed=True
        finally:self._lock.release()
