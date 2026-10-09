"""Two physical GPUs per existing arm, global batch32, ordered CPU prefetch."""
import argparse, collections, fcntl, hashlib, json, os, signal, subprocess, time, traceback
from pathlib import Path
from .train import sha, write_json, read_pinned, validate_sources, BoundaryStop


def admission(rows):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
    found={int(x[0]):[v.strip() for v in x] for line in raw.splitlines() if (x:=line.split(','))}
    report=[]
    for row in rows:
        physical=row['physical_gpu']; actual=found[physical]
        if physical not in (0,1,5,6) or actual[1]!=row['gpu_uuid'] or actual[3]!='0' or int(actual[2])<24576:
            raise ValueError('Need eligible matching physical GPU, zero volatile uncorrected ECC and24GiB free')
        report.append(dict(row,free_MiB=int(actual[2]),volatile_uncorrected_ecc=0))
    if len(rows)!=2 or len({x['gpu_uuid'] for x in rows})!=2:raise ValueError('Exactly two distinct GPUs required')
    return report


def validate_progress(step,cursor,prefix_step,prefix_samples,global_batch,target_samples):
    if (any(type(x) is not int for x in (step,cursor,prefix_step,prefix_samples,global_batch,target_samples))
            or prefix_step<0 or prefix_samples!=prefix_step*8 or step<prefix_step
            or cursor!=prefix_samples+(step-prefix_step)*global_batch
            or not 0<=cursor<=target_samples or (target_samples-cursor)%global_batch):
        raise ValueError('Optimizer count, original batch8 prefix and batch32 sample cursor disagree')
    return step+(target_samples-cursor)//global_batch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--resume');ap.add_argument('--stop-after-updates',type=int)
    args=ap.parse_args()
    if args.stop_after_updates is not None and args.stop_after_updates<1:raise ValueError('Positive engineering length required')
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['root']).resolve();arm=binding['arm']
    if arm not in ('L','A'):raise ValueError('Unknown arm')
    config=read_pinned(root/'lora_retention_run_v1/training_config.json',binding['config_sha256'])
    old_sources=read_pinned(root/'lora_retention_run_v1/runtime_sources.json',config['runtime_sources_sha256']);validate_sources(root,old_sources)
    sources=read_pinned(root/binding['execution_sources'],binding['execution_sources_sha256']);validate_sources(root,sources)
    for f in ['train_dual.py','replicas.py','batched_bridge.py','batching.py','provider.py','train.py','__init__.py']:
        if 'lora_parallel_v1/'+f not in sources:raise ValueError('Unpinned execution source '+f)
    if config['batch']!=8 or config['updates']!=10000:raise ValueError('Unexpected parent numerical recipe')
    batch_config=read_pinned(root/'lora_parallel_v1/training_config32.json',binding['batch32_config_sha256'])
    global_batch=batch_config['global_batch'];target_samples=batch_config['target_samples'];microbatch=binding['microbatch']
    if global_batch!=32 or target_samples!=80000 or microbatch not in (4,8,16):raise ValueError('Explicit batch32/micro4,8,16/80000 sample contract required')
    workers=binding.get('workers',16);prefetch=binding.get('prefetch',32)
    if type(workers) is not int or type(prefetch) is not int or not 1<=workers<=16 or not workers<=prefetch<=32:raise ValueError('Invalid bounded CPU workers/prefetch')
    if batch_config['learning_rate']!=config['optimizer']['lr']:raise ValueError('Batch32 plan cannot silently change original learning rate')
    devices=binding['devices'];health=admission(devices)
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(x['gpu_uuid'] for x in devices),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    parent=Path(binding['parent_checkpoint']);parent_identity=read_pinned(binding['parent_run_identity'],binding['parent_run_identity_sha256'])
    if sha(parent)!=binding['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    out=Path(binding['output']).resolve()
    if out==parent.parent or out.is_relative_to(parent.parent):raise ValueError('Separate execution namespace required')
    if args.resume:
        if not out.is_dir() or (out/'completion.json').exists():raise ValueError('Need unfinished execution')
    else:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=str(time.time_ns());stop=BoundaryStop();signal.signal(signal.SIGUSR1,stop.handler)
    owner={'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'devices':devices,'binding_sha256':sha(args.binding),'started_unix':time.time()}
    write_json(out/'owner.json',owner);write_json(out/('owner-'+attempt+'.json'),owner);write_json(out/('gpu-'+attempt+'.json'),health)
    provider=engine=None
    try:
        import random,numpy as np,torch
        from transformers import AutoModel,AutoProcessor
        from training_bridge.train_single import checkpoint_identity,runtime_identity
        from human_mg_training_v1.bridge import slot_draws,describe
        from route_reassessment_20261004.action_lora.action_lora import inject_action_lora,save_adapter,_tensor_digest
        from route_reassessment_20261004.action_lora.training_checkpoint import load_training_checkpoint,save_training_checkpoint,_digest,_rng
        from .provider import make_provider
        from .batching import collate_encoded_samples,pad_token_id_from_processor
        from .batched_bridge import BatchedLoRAFlowBridge
        from .replicas import TwoDeviceEngine
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        if torch.cuda.device_count()!=2:raise ValueError('Need exactly two bound GPUs')
        write_json(out/'status.json',{'stage':'verifying_and_loading','arm':arm,'time':time.time()})
        assets=checkpoint_identity(config['model'])
        if assets!=json.loads((root/config['base_assets']).read_text()):raise ValueError('Released base changed')
        base_id=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()
        processor=AutoProcessor.from_pretrained(config['model'],trust_remote_code=True,use_fast=False,local_files_only=True)
        pad=pad_token_id_from_processor(processor)
        provider=make_provider(arm,root/config['data_binding'],config['data_binding_sha256'],processor,'cpu',processor_path=config['model'],expected_config_sha256=config['sampling_config_sha256'],workers=workers,prefetch=prefetch)
        if provider.slots_sha256!=batch_config['sampling_plan_sha256']:raise ValueError('Batch32 config refers to a different fixed sample sequence')
        handles=[];bridges=[]
        for i in range(2):
            write_json(out/'status.json',{'stage':'loading_replica','replica':i,'arm':arm,'time':time.time()})
            with torch.cuda.device(i):
                model=AutoModel.from_pretrained(config['model'],trust_remote_code=True,attn_implementation='flash_attention_2',dtype=torch.bfloat16,local_files_only=True).to(f'cuda:{i}').to(torch.bfloat16).eval()
                handle=inject_action_lora(model,base_identity=base_id,**config['lora']);bridge=BatchedLoRAFlowBridge(handle,retention_coefficient=config['retention_coefficients'][arm])
                handles.append(handle);bridges.append(bridge)
        primary=handles[0];params=list(bridges[0].trainable_parameters());opt=config['optimizer']
        optimizer=torch.optim.AdamW(params,lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'],foreach=False)
        random.seed(config['train_seed']);np.random.seed(config['train_seed']);torch.manual_seed(config['train_seed'])
        numerical={'config_sha256':binding['config_sha256'],'source_sha256':old_sources,'arm':arm,'base_identity':base_id,'base_tensor_sha256':primary.metadata['base_state_sha256'],'sampling_config_sha256':config['sampling_config_sha256'],'runtime':runtime_identity()}
        if numerical!=parent_identity:raise ValueError('Reconstructed legacy identity differs')
        execution={'kind':'two_replica_global32_v1','source_sha256':sources,'parent_checkpoint_sha256':binding['parent_checkpoint_sha256'],'parent_run_identity_sha256':binding['parent_run_identity_sha256'],'devices':devices,'workers':workers,'prefetch':prefetch,'global_batch':global_batch,'microbatch_per_device':microbatch,'accumulation_calls_per_device':global_batch//2//microbatch,'checkpoint_accumulate':global_batch,'checkpoint_accumulate_units':'global sample contribution count per optimizer update','target_samples':target_samples,'per_sample_equal_weight':True,'precision':'BF16 compute, FP32 LoRA/AdamW','optimizer_recipe_unchanged':True,'sampling_unchanged':True,'batch_kernel_rounding_may_differ':True,'stop_after_updates':args.stop_after_updates}
        identity={'config_sha256':binding['batch32_config_sha256'],'source_sha256':dict(old_sources,**sources),'arm':arm,'base_identity':base_id,'base_tensor_sha256':primary.metadata['base_state_sha256'],'parent_numerical_identity':numerical,'runtime':runtime_identity(),'execution':execution}
        common=dict(handle=primary,optimizer=optimizer,provider=provider,device='cuda:0',max_grad_norm=config['max_grad_norm'])
        if args.resume:
            latest=json.loads((out/'latest_checkpoint.json').read_text());resume=Path(args.resume).resolve()
            if resume.parent!=out or str(resume)!=latest['path'] or sha(resume)!=latest['file_sha256'] or json.loads((out/'run_identity.json').read_text())!=identity:raise ValueError('Resume lineage changed')
            step=load_training_checkpoint(resume,identity=identity,accumulate=global_batch,**common)
            migration=json.loads((out/'migration.json').read_text())
            if (migration['parent_checkpoint_sha256']!=binding['parent_checkpoint_sha256']
                    or migration['execution']!=execution or migration.get('strict_old_identity_verified') is not True):raise ValueError('Resume migration lineage changed')
            prefix_step=migration['restored_step'];prefix_samples=migration['restored_samples']
        else:
            step=load_training_checkpoint(parent,identity=numerical,accumulate=8,**common)
            prefix_step=step;prefix_samples=provider.cursor
            write_json(out/'run_identity.json',identity);write_json(out/'numerical_identity.json',numerical)
            write_json(out/'migration.json',{'parent_checkpoint':str(parent),'parent_checkpoint_sha256':binding['parent_checkpoint_sha256'],'restored_step':step,'restored_samples':provider.cursor,'strict_old_identity_verified':True,'execution':execution,'time':time.time()})
        planned_final_step=validate_progress(step,provider.cursor,prefix_step,prefix_samples,global_batch,target_samples)
        engine=TwoDeviceEngine(bridges[0],bridges[1],optimizer,max_grad_norm=config['max_grad_norm'],global_batch=global_batch)
        common.update(identity=identity,accumulate=global_batch)
        start_step=step;started=time.monotonic();recent=collections.deque(maxlen=30)
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        def save_boundary():
            validate_progress(step,provider.cursor,prefix_step,prefix_samples,global_batch,target_samples)
            p=out/f'checkpoint-{step:08d}.pt'
            if p.exists():
                x=json.loads((out/'latest_checkpoint.json').read_text())
                if x['step']!=step or sha(p)!=x['file_sha256']:raise ValueError('Existing boundary mismatch')
                return x
            x=save_training_checkpoint(p,step=step,**common);x.update(file_sha256=sha(p),execution_identity_sha256=_digest(identity),parent_checkpoint_sha256=binding['parent_checkpoint_sha256'],secondary_state_reconstructed_from_primary=True,secondary_global_rng_not_used=True)
            if step==start_step+1:
                saved=torch.load(p,map_location='cpu',weights_only=True);payload=saved['payload']
                checks={'payload_digest':_digest(payload)==saved['sha256'],
                    'adapters_exact':_digest(payload['adapters'])==_digest({n:v for n,v in primary.model.named_parameters() if n in primary.adapter_names}),
                    'optimizer_exact':_digest(payload['optimizer'])==_digest(optimizer.state_dict()),
                    'provider_exact':_digest(payload['provider'])==_digest(provider.state_dict()),
                    'master_rng_exact':_digest(payload['rng'])==_digest(_rng('cuda:0')),
                    'global_batch_metadata':payload['metadata']['accumulate']==global_batch,
                    'step_exact':payload['step']==step}
                if not all(checks.values()):raise ValueError('Saved actual CUDA checkpoint mismatch: '+repr(checks))
                write_json(out/'first_checkpoint_integrity.json',{'passed':True,'checks':checks,'step':step,'samples':provider.cursor,'secondary_reconstructed_from_master':True})
            write_json(out/'latest_checkpoint.json',x);return x
        def pause(reason):
            cp=save_boundary();s={'stage':'engineering_pause' if args.stop_after_updates else 'paused_at_optimizer_boundary','arm':arm,'step':step,'samples':provider.cursor,'reason':reason,'checkpoint':cp,'candidate_selected':False,'time':time.time()}
            write_json(out/'status.json',s);write_json(out/('pause-'+attempt+'.json'),s)
        if stop.requested:pause('SIGUSR1');return
        with (out/('metrics-'+attempt+'.jsonl')).open('x') as stream:
            while provider.cursor<target_samples:
                begin=time.monotonic();data_begin=begin;start_cursor=provider.cursor;samples=[provider.next_batch() for _ in range(global_batch)];data_seconds=time.monotonic()-data_begin
                if [x[3]['slot_index'] for x in samples]!=list(range(start_cursor,start_cursor+global_batch)):raise ValueError('Slot schedule changed')
                chunks=[];ids=[]
                for rank in range(2):
                    rank_chunks=[]
                    for offset in range(rank*(global_batch//2),(rank+1)*(global_batch//2),microbatch):
                        inp,actions,valid,ident=collate_encoded_samples(samples[offset:offset+microbatch],pad_token_id=pad)
                        device=f'cuda:{rank}'
                        inp={k:v.to(device) for k,v in inp.items()};actions=actions.to(device);valid=valid.to(device)
                        ns=[];ts=[]
                        for j,sample_id in enumerate(ident):
                            n,t=slot_draws(inp['action_mask'][j:j+1],sample_id['slot_seed']);ns.append(n);ts.append(t)
                            sample_id['tensor_sha256']={k:describe(v)['sha256'] for k,v in [('actions',actions[j:j+1]),('valid',valid[j:j+1]),('noise',n),('times',t)]};ids.append(sample_id)
                        rank_chunks.append(dict(inputs=inp,actions=actions,valid=valid,noise=torch.cat(ns),times=torch.cat(ts)))
                    chunks.append(rank_chunks)
                result=engine.step(chunks);step+=1
                seconds=time.monotonic()-begin;recent.append(seconds)
                row={'step':step,'arm':arm,'loss':result['loss'],'flow_loss':result['flow_loss'],'retention_loss':result['retention_loss'],'gradient_norm':result['gradient_norm'],'wall_seconds':seconds,'provider_wall_seconds':data_seconds,'engine':result,'samples':ids,'completed_unix':time.time()}
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                status={k:v for k,v in row.items() if k not in ('samples','engine')};status.update(stage='training',samples=provider.cursor,planned_steps=planned_final_step,target_samples=target_samples,devices=devices,global_batch=global_batch,microbatch_per_device=microbatch,recent_seconds_per_step=sum(recent)/len(recent),estimated_remaining_seconds=(planned_final_step-step)*sum(recent)/len(recent),prefetch=provider.prefetch_status(),peak_allocated_GiB=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)])
                write_json(out/'status.json',status)
                if step<=start_step+3 or step%25==0:print(json.dumps(status),flush=True)
                if step==start_step+1 or step%250==0 or provider.cursor==target_samples:save_boundary()
                if stop.requested or args.stop_after_updates and step-start_step>=args.stop_after_updates:pause('SIGUSR1' if stop.requested else 'bounded_engineering_update_count');return
        if provider.cursor!=80000 or provider.source_counts!={'human':80000,'mg':0}:raise ValueError('Terminal coverage differs')
        adapter=save_adapter(primary,out/f'adapter-step-{step:08d}.pt')
        completion={'completed':True,'arm':arm,'updates':step,'samples':provider.cursor,'global_batch':global_batch,'target_samples':target_samples,'adapter':adapter,'adapter_file_sha256':sha(adapter['path']),'identity_sha256':sha(out/'run_identity.json'),'parent_checkpoint_sha256':binding['parent_checkpoint_sha256'],'migration_sha256':sha(out/'migration.json'),'elapsed_seconds':time.monotonic()-started,'completed_unix':time.time()}
        write_json(out/'completion.json',completion);write_json(out/'status.json',dict(completion,stage='complete'))
    except BaseException as e:
        x={'stage':'failed','error':repr(e),'traceback':traceback.format_exc(),'time':time.time(),'no_uncontrolled_retry':True};write_json(out/('failure-'+attempt+'.json'),x);write_json(out/'status.json',x);raise
    finally:
        if engine is not None:engine.close()
        if provider is not None:provider.close()

if __name__=='__main__':main()
