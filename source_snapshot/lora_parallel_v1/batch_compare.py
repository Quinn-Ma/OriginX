"""Real checkpoint training-gradient engineering check; no optimizer update/eval.

Loads fixed subsequent training slots once. Compares legacy serialB1 to
selectableB1/B2/B4/B8/B16 with equal-window means and identical slot draws.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .train import sha,read_pinned,validate_sources,write_json


THRESHOLDS = {'batched_loss_relative_max':0.005,'per_sample_loss_relative_max':0.02,
              'gradient_relative_L2_max':0.05,'gradient_cosine_min':0.999,
              'batch1_loss_relative_max':1e-6,'batch1_gradient_relative_L2_max':1e-5}


def compare_gradients(reference,actual):
    import torch
    if set(reference)!=set(actual):raise ValueError('Gradient tensor names differ')
    aa=bb=dd=dot=0.
    max_abs=0.
    for name in reference:
        left,right=reference[name].double(),actual[name].double()
        if left.shape!=right.shape or not torch.isfinite(right).all():raise ValueError('Gradient shape/value differs')
        aa+=left.square().sum().item();bb+=right.square().sum().item()
        dd+=(right-left).square().sum().item();dot+=(left*right).sum().item()
        max_abs=max(max_abs,(right-left).abs().max().item())
    return {'relative_L2':(dd/max(aa,1e-30))**.5,
            'cosine':dot/max((aa*bb)**.5,1e-30),
            'reference_L2':aa**.5,'actual_L2':bb**.5,'max_absolute':max_abs}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binding',required=True)
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,4,8,16])
    args=parser.parse_args()
    if args.samples not in (8,16,32) or not args.batch_sizes or len(set(args.batch_sizes))!=len(args.batch_sizes):
        raise ValueError('Require8/16/32 fixed samples and unique requested batch sizes')
    if any(size not in (1,2,4,8,16) or args.samples%size for size in args.batch_sizes):
        raise ValueError('Requested batch sizes must divide fixed sample count')
    sample_count=args.samples
    binding=json.loads(Path(args.binding).read_text());root=Path(binding['root']).resolve()
    config=read_pinned(root/'lora_retention_run_v1/training_config.json',binding['config_sha256'])
    old_sources=read_pinned(root/'lora_retention_run_v1/runtime_sources.json',config['runtime_sources_sha256'])
    validate_sources(root,old_sources)
    new_sources=read_pinned(root/binding['execution_sources'],binding['execution_sources_sha256'])
    validate_sources(root,new_sources)
    for name in ['lora_parallel_v1/batch_compare.py','lora_parallel_v1/batched_bridge.py',
                 'lora_parallel_v1/batching.py','lora_parallel_v1/provider.py']:
        if name not in new_sources:raise ValueError('Unpinned new source '+name)
    checkpoint=Path(binding['parent_checkpoint'])
    if sha(checkpoint)!=binding['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    parent_identity=read_pinned(binding['parent_run_identity'],binding['parent_run_identity_sha256'])
    arm=binding['arm']
    if arm not in ('L','A') or binding['physical_gpu'] not in (0,1,5,6):raise ValueError('Arm/GPU not admitted')
    os.environ.update(CUDA_VISIBLE_DEVICES=binding['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    out=Path(binding['output']);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'owner.json',{'pid':os.getpid(),'gpu_uuid':binding['gpu_uuid'],
        'physical_gpu':binding['physical_gpu'],'process_start_ticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        'binding_sha256':sha(args.binding),'started_unix':time.time(),'kind':'training_numerics_only'})
    provider=None
    try:
        import random
        import numpy as np
        import torch
        from transformers import AutoModel,AutoProcessor
        from training_bridge.train_single import checkpoint_identity,runtime_identity
        from online_flow_training_gate_v1.check import gpu_inventory
        from human_mg_training_v1.bridge import slot_draws,describe
        from route_reassessment_20261004.action_lora.action_lora import inject_action_lora,_tensor_digest
        from route_reassessment_20261004.action_lora.lora_flow_bridge import LoRAFlowBridge
        from route_reassessment_20261004.action_lora.training_checkpoint import load_training_checkpoint,_digest,_rng,_restore_rng
        from .provider import make_provider
        from .batching import collate_encoded_samples,pad_token_id_from_processor
        from .batched_bridge import BatchedLoRAFlowBridge,compute_microbatch_gradients
        write_json(out/'gpu.json',gpu_inventory(binding['physical_gpu'],binding['gpu_uuid']))
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        write_json(out/'status.json',{'stage':'source_model_loading','time':time.time()})
        assets=checkpoint_identity(config['model'])
        if assets!=json.loads((root/config['base_assets']).read_text()):raise ValueError('Base assets changed')
        base_id=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()
        processor=AutoProcessor.from_pretrained(config['model'],trust_remote_code=True,use_fast=False,local_files_only=True)
        provider=make_provider(arm,root/config['data_binding'],config['data_binding_sha256'],processor,'cuda:0',
            expected_config_sha256=config['sampling_config_sha256'],processor_path=config['model'],workers=8,prefetch=16)
        model=AutoModel.from_pretrained(config['model'],trust_remote_code=True,attn_implementation='flash_attention_2',
            dtype=torch.bfloat16,local_files_only=True).to('cuda:0').to(torch.bfloat16).eval()
        handle=inject_action_lora(model,base_identity=base_id,**config['lora'])
        legacy=LoRAFlowBridge(handle,retention_coefficient=config['retention_coefficients'][arm])
        parameters=list(legacy.trainable_parameters());opt=config['optimizer']
        optimizer=torch.optim.AdamW(parameters,lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],
            weight_decay=opt['weight_decay'],foreach=False)
        identity={'config_sha256':binding['config_sha256'],'source_sha256':old_sources,'arm':arm,
                  'base_identity':base_id,'base_tensor_sha256':handle.metadata['base_state_sha256'],
                  'sampling_config_sha256':config['sampling_config_sha256'],'runtime':runtime_identity()}
        if identity!=parent_identity:raise ValueError('Parent numerical identity differs')
        step=load_training_checkpoint(checkpoint,handle=handle,optimizer=optimizer,provider=provider,identity=identity,
            device='cuda:0',accumulate=config['batch'],max_grad_norm=config['max_grad_norm'])
        if step<=0 or provider.cursor!=step*8:raise ValueError('Require a trained nonzero-step checkpoint')
        start_cursor=provider.cursor
        write_json(out/'status.json',{'stage':'preparing_training_slots','count':sample_count,'parent_step':step,'time':time.time()})
        samples=[provider.next_batch() for _ in range(sample_count)]
        provider.close()
        draws=[slot_draws(row[0]['action_mask'],row[3]['slot_seed']) for row in samples]
        sample_receipts=[]
        for row,(noise,times) in zip(samples,draws):
            sample_receipts.append({'identity':row[3], 'input_sha256':{k:describe(v)['sha256'] for k,v in row[0].items() if isinstance(v,torch.Tensor)},
                'actions_sha256':describe(row[1])['sha256'],'valid_sha256':describe(row[2])['sha256'],
                'noise_sha256':describe(noise)['sha256'],'times_sha256':describe(times)['sha256']})
        write_json(out/'sample-receipts.json',sample_receipts)
        adapter_before=_tensor_digest((n,p) for n,p in model.named_parameters() if n in handle.adapter_names)
        optimizer_before=_digest(optimizer.state_dict());rng_before=_rng('cuda:0')
        grouped=BatchedLoRAFlowBridge(handle,retention_coefficient=config['retention_coefficients'][arm])
        pad_token_id=pad_token_id_from_processor(processor)
        results={};reference_gradients=None
        def measure(mode):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic();losses=[];flows=[];retentions=[]
            if mode=='legacy1':
                for row,(noise,times) in zip(samples,draws):
                    result=legacy(row[0],row[1],row[2],noise=noise,times=times)
                    if not torch.isfinite(result['loss']).item():raise FloatingPointError('Legacy nonfinite loss')
                    (result['loss']/sample_count).backward()
                    losses.append(result['loss'].detach().item());flows.append(result['flow_loss'].detach().item())
                    retentions.append(result['retention_loss'].detach().item())
            else:
                for index in range(0,sample_count,mode):
                    inputs,actions,valid,ids=collate_encoded_samples(samples[index:index+mode],pad_token_id=pad_token_id)
                    noise=torch.cat([x[0] for x in draws[index:index+mode]],dim=0)
                    times=torch.cat([x[1] for x in draws[index:index+mode]],dim=0)
                    result=compute_microbatch_gradients(grouped,inputs,actions,valid,noise=noise,times=times,loss_scale=mode/sample_count)
                    losses+=result['per_sample_loss'];flows+=result['per_sample_flow_loss'];retentions+=result['per_sample_retention_loss']
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if name in handle.adapter_names}
            row={'mode':str(mode),'samples':sample_count,'loss':sum(losses)/sample_count,'flow_loss':sum(flows)/sample_count,
                 'retention_loss':sum(retentions)/sample_count,'per_sample_loss':losses,'per_sample_flow_loss':flows,
                 'per_sample_retention_loss':retentions,'wall_seconds':seconds,
                 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                 'gradient_sha256':_tensor_digest(gradients.items())}
            return row,gradients

        for mode in ['legacy1']+args.batch_sizes:
            try:
                row,gradients=measure(mode)
            except torch.cuda.OutOfMemoryError as error:
                if mode=='legacy1':raise
                row={'mode':str(mode),'status':'unavailable_capacity','passed':False,'error':str(error),
                     'samples':sample_count,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                results[str(mode)]=row
                write_json(out/('mode-'+str(mode)+'.json'),row)
                print(json.dumps(row),flush=True)
                error.__traceback__=None
                optimizer.zero_grad(set_to_none=True)
                import gc
                gc.collect();torch.cuda.empty_cache()
                continue
            row['status']='completed'
            losses=row['per_sample_loss'];seconds=row['wall_seconds']
            if mode=='legacy1':reference_gradients=gradients
            else:
                reference=results['legacy1']
                row['loss_relative_error']=abs(row['loss']-reference['loss'])/max(abs(reference['loss']),1e-8)
                row['max_sample_loss_relative_error']=max(abs(a-b)/max(abs(b),1e-8) for a,b in zip(losses,reference['per_sample_loss']))
                row['gradient_difference']=compare_gradients(reference_gradients,gradients)
                loss_limit=THRESHOLDS['batch1_loss_relative_max'] if mode==1 else THRESHOLDS['batched_loss_relative_max']
                grad_limit=THRESHOLDS['batch1_gradient_relative_L2_max'] if mode==1 else THRESHOLDS['gradient_relative_L2_max']
                row['passed']=bool(row['loss_relative_error']<=loss_limit and row['gradient_difference']['relative_L2']<=grad_limit
                    and row['gradient_difference']['cosine']>=THRESHOLDS['gradient_cosine_min']
                    and row['max_sample_loss_relative_error']<=THRESHOLDS['per_sample_loss_relative_max'])
                row['compute_speedup_vs_legacy']=reference['wall_seconds']/seconds
            results[str(mode)]=row
            write_json(out/('mode-'+str(mode)+'.json'),row)
            print(json.dumps({k:v for k,v in row.items() if not k.startswith('per_sample')}),flush=True)
            del gradients
        optimizer.zero_grad(set_to_none=True)
        unchanged={'adapter':adapter_before==_tensor_digest((n,p) for n,p in model.named_parameters() if n in handle.adapter_names),
                   'optimizer':optimizer_before==_digest(optimizer.state_dict()),
                   'global_RNG':_digest(rng_before)==_digest(_rng('cuda:0'))}
        legacy._assert_trainable_contract();grouped._assert_trainable_contract()
        _restore_rng(rng_before,'cuda:0')
        usable=[x for x in args.batch_sizes if results[str(x)]['passed']]
        numerically_complete=[x for x in args.batch_sizes if results[str(x)]['status']=='completed']
        completed_modes_pass=bool(usable) and all(results[str(x)]['passed'] for x in numerically_complete) and all(unchanged.values())
        report={'kind':'real5B_training_microbatch_numeric_check_v1',
                'passed':all(results[str(x)]['passed'] for x in args.batch_sizes) and all(unchanged.values()),
                'completed_modes_passed':completed_modes_pass,'usable_batch_sizes':usable,
                'capacity_unavailable_batch_sizes':[x for x in args.batch_sizes if results[str(x)]['status']=='unavailable_capacity'],
                'parent_checkpoint':str(checkpoint),'parent_checkpoint_sha256':binding['parent_checkpoint_sha256'],'parent_step':step,
                'arm':arm,'training_slots':[start_cursor,start_cursor+sample_count-1],'compared_batch_sizes':args.batch_sizes,'model_base_sha256':base_id,'source_sha256':new_sources,
                'thresholds_frozen_before_run':THRESHOLDS,'results':results,'unchanged':unchanged,
                'optimizer_updates':0,'training_steps_committed':0,'rollouts':0,'evaluation_success_read':False,
                'limits':'Fixed real training windows; compute-only cold/order-biased timing. No training efficacy or end-to-end throughput claim.',
                'completed_unix':time.time()}
        write_json(out/'report.json',report);write_json(out/'status.json',{'stage':'complete','all_requested_passed':report['passed'],
            'completed_modes_passed':completed_modes_pass,'usable_batch_sizes':usable,'time':time.time()})
        if not completed_modes_pass:raise RuntimeError('Completed batch numerical check failed; do not promote changed execution')
    except BaseException as error:
        failure={'error':repr(error),'traceback':traceback.format_exc(),'time':time.time(),'optimizer_updates':0,'no_uncontrolled_retry':True}
        write_json(out/'failure.json',failure);write_json(out/'status.json',dict(failure,stage='failed'))
        raise
    finally:
        if provider is not None:provider.close()


if __name__=='__main__':main()
