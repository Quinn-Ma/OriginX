"""Authorized GPU0 engineering check only; one disposable branch update, no rollout."""
import argparse,json,os,time,traceback,hashlib
from pathlib import Path


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent.parent;out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTORCH_NVML_BASED_CUDA_CHECK='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    write(out/'owner.json',dict(pid=os.getpid(),process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        command=Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str(Path.cwd()),gpu_uuid=args.gpu_uuid,
        physical_gpu=1,started_unix=time.time(),scope='engineering only; no candidate or rollout'))
    try:
        import torch
        from transformers import AutoProcessor
        from training_bridge.bridge import encode_sample,FrozenVLMFlowBridge
        from training_bridge.train_core import rng_state,restore_rng
        from teacher_anchor_pilot_v1.bridge import equal_rng
        from human_mg_data_bridge_v1.readers import SampleReader
        from human_mg_training_v1.bridge import slot_draws,describe
        from lora_parallel8_v1.batching import collate_encoded_samples,pad_token_id_from_processor
        from route_reassessment_20261004.action_lora.action_lora import _tensor_digest
        from continuous_model_v3.loader import load_frozen_a_bridge,branch_identity,export_branch,load_branch
        from types import SimpleNamespace
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        print(json.dumps({'stage':'loading_frozen_A'}),flush=True)
        b,view,anchor,binding=load_frozen_a_bridge(root/'results/lora-dev150-final-v1/A-endpoint.json',device='cuda:0',
            stage_count=7,supervised=True,retention_coefficient=1.,auxiliary_coefficient=.05)
        print(json.dumps({'stage':'A_loaded_and_verified'}),flush=True)
        processor=AutoProcessor.from_pretrained(binding['base_model'],trust_remote_code=True,use_fast=False,local_files_only=True)
        datafile=root/'human_mg_data_bridge_v1/data-binding.ready.json';reader=SampleReader(datafile,sha(datafile),partition='train')
        row=json.loads((root/'human_mg_data_bridge_v1/train-slots.jsonl').open().readline())
        item=row.get('H600',row.get('arms',{}).get('H600',row));assert item['source']=='human'
        sample,sampleid=reader.sample(item);i,a,v=encode_sample(sample,processor,'cuda:0')
        n,t=slot_draws(i['action_mask'],2026100401)
        before_digest=_tensor_digest(b.model.state_dict().items())
        print(json.dumps({'stage':'full_A_digest_before_complete'}),flush=True)
        torch.manual_seed(20261004);initial=rng_state('cuda:0')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            base=b.model(**i,logits_to_keep=1).actions.detach().cpu()
            baseline=FrozenVLMFlowBridge._loss(SimpleNamespace(model=b.model),i,a,v,n,t)['loss'].detach()
        after=rng_state('cuda:0');restore_rng(initial,'cuda:0')
        adapted=view(**i,logits_to_keep=1).actions.detach().cpu()
        assert torch.equal(base,adapted),'Zero stage branch changed A action'
        assert equal_rng(after,rng_state('cuda:0')),'Zero stage changed actor RNG'
        rng=rng_state('cuda:0');result=b(i,a,v,noise=n,times=t,stage_target=torch.tensor([3],device='cuda'),stage_confidence=torch.zeros(1,device='cuda'))
        assert torch.equal(result['flow_loss'],baseline) and result['retention_loss'].item()==0,'Zero flow/teacher mismatch'
        grads=torch.autograd.grad(result['phase_aux_loss'],list(b.branch.parameters()),retain_graph=True,allow_unused=True)
        assert all(g is None or not g.count_nonzero().item() for g in grads),'Low confidence CE produced gradient'
        result['loss'].backward();first=b.branch.to_delta.weight.grad.norm().item();assert first>0
        assert equal_rng(rng,rng_state('cuda:0')),'Training changed ambient RNG'
        optimizer=torch.optim.AdamW(b.trainable_parameters(),lr=1e-4,weight_decay=0,foreach=False)
        norm=torch.nn.utils.clip_grad_norm_(list(b.trainable_parameters()),1.,error_if_nonfinite=True).item()
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        del result,grads
        changed=b(i,a,v,noise=n,times=t,stage_target=torch.tensor([3],device='cuda'),stage_confidence=torch.zeros(1,device='cuda'))
        assert changed['retention_loss'].item()>0,'One update did not change actual velocity'
        changed['loss'].backward();encoder_norm=b.branch.encoder[0].weight.grad.norm().item();assert encoder_norm>0,'Action-only latent branch gradient is dead'
        after_retention=changed['retention_loss'].item();optimizer.zero_grad(set_to_none=True);del changed
        restore_rng(initial,'cuda:0');nonzero=view(**i,logits_to_keep=1).actions.detach().cpu()
        assert not torch.equal(base,nonzero),'Nonzero stage branch did not change actions'
        identity=branch_identity(b,anchor,vocabulary_sha256='e3b7cbef486268961c5e660e0d50a7397d5c3d3ad9caf2e8b41663cd15fa52b0',
            training_data_sha256='7b174f21348c148a4f94ab5240625b1d2e79a3375e940282b5a4edce72e7c6a2')
        checkpoint=out/'mechanical-only-branch.pt';cp_sha=export_branch(checkpoint,b,identity)
        with torch.no_grad():b.branch.to_delta.bias.add_(.1)
        load_branch(checkpoint,b,identity,expected_sha256=cp_sha);restore_rng(initial,'cuda:0')
        reloaded=view(**i,logits_to_keep=1).actions.detach().cpu();assert torch.equal(nonzero,reloaded)
        print(json.dumps({'stage':'B32_capacity'}),flush=True)
        # Same historical engineering window repeated32 times; real geometry and
        # full VLM+DiT/backward, not a throughput claim for variable-length data.
        samples=[(i,a,v,dict(sampleid,slot_index=j,slot_seed=2026100401+j)) for j in range(32)]
        bi,ba,bv,ids=collate_encoded_samples(samples,pad_token_id=pad_token_id_from_processor(processor))
        draws=[slot_draws(bi['action_mask'][j:j+1],ids[j]['slot_seed']) for j in range(32)]
        bn=torch.cat([d[0] for d in draws]);bt=torch.cat([d[1] for d in draws])
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();timings=[];scales=[]
        for j in range(2):
            start=time.perf_counter();r=b(bi,ba,bv,noise=bn,times=bt,stage_target=torch.full((32,),3,device='cuda',dtype=torch.long),
                stage_confidence=torch.ones(32,device='cuda'));r['loss'].backward();torch.cuda.synchronize()
            timings.append(time.perf_counter()-start);scales.append({k:r[k].detach().item() for k in ('loss','flow_loss','retention_loss','phase_aux_loss','phase_aux_rows')})
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.trainable_parameters())
            optimizer.zero_grad(set_to_none=True);del r
        peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
        print(json.dumps({'stage':'B32_passed_final_frozen_A_digest'}),flush=True)
        after_digest=_tensor_digest(b.model.state_dict().items());assert before_digest==after_digest,'Frozen A tensors changed'
        b.guard.check()
        report=dict(passed=True,physical_gpu=1,gpu_uuid=args.gpu_uuid,real_A_5B=True,zero_actions_exact=True,
            zero_flow_exact=True,zero_actor_rng_exact=True,low_conf_CE_gradient_zero=True,action_projection_gradient_norm=first,
            second_action_encoder_gradient_norm=encoder_norm,after_one_update_velocity_retention_loss=after_retention,
            nonzero_actions_changed=True,branch_reload_actions_exact=True,full_frozen_A_digest_before=before_digest,
            full_frozen_A_digest_after=after_digest,frozen_A_all_weights_exact=True,trainable_tensors=len(list(b.branch.parameters())),
            trainable_parameters=sum(p.numel() for p in b.branch.parameters()),branch_settings=b.branch.settings,
            B32_forward_backward_wall_seconds=timings,B32_peak_allocated_GiB=peak,B32_peak_reserved_GiB=reserved,
            B32_loss_scales=scales,sample=sampleid,anchor=anchor,mechanical_optimizer_steps=1,
            candidate=False,rollouts=0,branch_checkpoint_sha256=cp_sha,completed_unix=time.time())
        write(out/'report.json',report);print(json.dumps(report),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
