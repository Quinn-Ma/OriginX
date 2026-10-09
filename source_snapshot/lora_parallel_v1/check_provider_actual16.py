"""Explicit CPU-only16slot byte check from a real step500 provider checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def encode_gpu_dtype_on_cpu(sample,processor,device):
    import torch
    from training_bridge.bridge import encode_sample
    if str(device)!='cpu':raise ValueError('This small actual-data check is CPU-only')
    return encode_sample(sample,processor,'cpu',floating_dtype=torch.bfloat16)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha256',required=True)
    p.add_argument('--processor',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--prefetch',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoProcessor
    from lora_retention_run_v1.provider import make_provider as serial_make
    from lora_parallel_v1.provider import make_provider as parallel_make
    from cpu_prefetch_mechanical_v1.core import encoded_description
    from route_reassessment_20261004.action_lora.training_checkpoint import _digest
    from human_mg_training_v1.bridge import describe
    torch.set_num_threads(1)
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('CheckpointSHA differs')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if set(saved)!= {'payload','sha256'} or _digest(saved['payload'])!=saved['sha256']:
        raise ValueError('Training checkpoint content differs')
    payload=saved['payload'];state=payload['provider'];arm=state['arm']
    if payload['step']!=500 or state['cursor']!=4000 or arm not in ('L','A'):
        raise ValueError('Require an actual step500 / cursor4000 checkpoint')
    expected_config=state['config_sha256'];del payload,saved
    processor=AutoProcessor.from_pretrained(args.processor,trust_remote_code=True,use_fast=False,local_files_only=True)
    timings={};serial_rows=[]
    started=time.monotonic()
    serial=serial_make(arm,args.binding,args.binding_sha256,processor,'cpu',
                       expected_config_sha256=expected_config,encoder=encode_gpu_dtype_on_cpu)
    serial.load_state_dict(state);timings['serial_setup_seconds']=time.monotonic()-started
    started=time.monotonic()
    for slot in range(4000,4016):
        inputs,actions,valid,ident=serial.next_batch()
        if ident['slot_index']!=slot:raise ValueError('Serial cursor changed')
        serial_rows.append({'identity':ident,'encoded':encoded_description((inputs,actions,valid))})
    timings['serial16_decode_encode_hash_seconds']=time.monotonic()-started
    serial_terminal=serial.state_dict();serial.dataset.clear_caches();del serial
    print(json.dumps({'stage':'serial16_complete','seconds':timings['serial16_decode_encode_hash_seconds']}),flush=True)
    started=time.monotonic()
    parallel=parallel_make(arm,args.binding,args.binding_sha256,processor,'cpu',processor_path=args.processor,
            workers=args.workers,prefetch=args.prefetch,expected_config_sha256=expected_config,
            encoder=encode_gpu_dtype_on_cpu)
    parallel.load_state_dict(state);timings['parallel_parent_setup_seconds']=time.monotonic()-started
    rows=[];started=time.monotonic()
    try:
        for index in range(16):
            inputs,actions,valid,ident=parallel.next_batch()
            encoded=encoded_description((inputs,actions,valid));expected=serial_rows[index]
            checks={'slot_exact':ident['slot_index']==4000+index,'identity_exact':ident==expected['identity'],
                    'all_encoded_bytes_exact':encoded==expected['encoded']}
            if not all(checks.values()):raise ValueError('Serial/parallel difference: '+str(checks))
            rows.append({'slot_index':4000+index,**checks,'slot_seed':ident['slot_seed'],
                         'encoded':encoded,'timings':parallel.last_prefetch_timing})
        terminal=parallel.state_dict()
        if terminal!=serial_terminal:raise ValueError('Committed provider states differ')
        parallel.validate_state_dict(terminal)
        status=parallel.prefetch_status()
        timings['parallel16_including_spawn_processing_IPC_hash_seconds']=time.monotonic()-started
    finally:
        parallel.close()
    if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
    report={'kind':'lora_ordered_prefetch_actual16_v1','passed':True,'slots':[4000,4015],
        'checkpoint_sha256':args.checkpoint_sha256,'checkpoint_step':500,'arm':arm,
        'provider_class':'lora_retention_run_v1.provider.FixedHumanProvider',
        'restored_cursor':4000,'terminal_cursor':terminal['cursor'],'state_exact':True,
        'CPU_BF16_inputs_match_original_GPU_input_arithmetic':True,
        'actual_GPU_or_model_forward':False,'noise_time_redrawn':False,'CUDA_initialized':False,
        'same_slot_seeds':True,'worker_status':status,'timings':timings,'rows':rows,
        'limits':'CPU originalserial vs parallelencode bytes; not newGPUtraining equivalence or speed claim. Serial-first cache bias and worker startup included. Restored actual4000slot source evidence, no fabricated parquet pins.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'slots':16,'terminal_cursor':4016,'report':str(args.output),'sha256':sha(args.output)}),flush=True)


if __name__=='__main__':main()
