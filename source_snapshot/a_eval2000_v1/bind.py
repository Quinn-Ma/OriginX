from pathlib import Path
import json,torch
from .loader import sha,read_binding
from route_reassessment_20261004.action_lora.training_checkpoint import _digest
R=Path(__file__).resolve().parent.parent

def main():
 out=R/'results/continued-ab2000-v1';out.mkdir(exist_ok=True);run=R/'training/a-continue2000-v1/A'
 def pin(p):return dict(path=str(p),sha256=sha(p))
 def write(p,v):
  with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
 c=json.loads((run/'completion.json').read_text());cp=run/'checkpoint-00002000.pt';a=run/'adapter-step-00002000.pt'
 assert c['completed'] and c['updates']==2000 and c['additional_updates']==387 and c['samples']==129536
 e=torch.load(cp,map_location='cpu',weights_only=True);payload=e['payload'];ad=torch.load(a,map_location='cpu',weights_only=True)
 assert _digest(payload)==e['sha256'] and payload['step']==2000 and payload['provider']['cursor']==49536
 assert _digest(payload['adapters'])==_digest(ad['tensors']) and all(v['step'].item()==2000 for v in payload['optimizer']['state'].values())
 assert c['adapter_file_sha256']==sha(a)
 audit=dict(passed=True,parent_step=1613,all_AdamW_steps=2000,adapter_equals_checkpoint=True,checkpoint=pin(cp),adapter=pin(a));ap=out/'A-audit.json';write(ap,audit)
 old=R/'results/lora-dev150-final-v1/A-endpoint.json';base=json.loads(old.read_text());sources=dict(base['source_sha256'])
 for f in (R/'a_eval2000_v1').glob('*.py'):sources[str(f)]=sha(f)
 b=dict(schema='a_continue2000_endpoint_v1',ready=True,arm='A',root=str(R),endpoint_updates=2000,endpoint_samples=129536,global_batch=128,base_model=base['base_model'],inference_arithmetic=base['inference_arithmetic'],original_A_binding=pin(old),completion=pin(run/'completion.json'),run_identity=pin(run/'run_identity.json'),adapter=pin(a),audit=pin(ap),source_sha256=sources)
 path=out/'A-endpoint.json';write(path,b);read_binding(path,sha(path));print(json.dumps(pin(path)))
if __name__=='__main__':main()
