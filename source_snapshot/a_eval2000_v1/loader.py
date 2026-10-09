"""Exact A continuation endpoint, preserving original A1613 as a separate teacher."""
import json
from pathlib import Path
from lora_dev150_final_v1.loader import sha,require,pinned

def read_binding(path,expected_sha):
 require(sha(path)==expected_sha,'A2000 binding changed');b=json.loads(Path(path).read_text())
 require(b['schema']=='a_continue2000_endpoint_v1' and b['ready'] and b['arm']=='A','Wrong A2000 endpoint')
 require((b['endpoint_updates'],b['endpoint_samples'],b['global_batch'])==(2000,129536,128),'Wrong A2000 budget')
 c=pinned(b['completion']);audit=pinned(b['audit']);identity=pinned(b['run_identity'])
 require(c['completed'] and c['updates']==2000 and c['additional_updates']==387 and c['samples']==129536,'Continuation incomplete')
 require(c['adapter_file_sha256']==b['adapter']['sha256']==sha(b['adapter']['path']) and c['identity_sha256']==b['run_identity']['sha256'],'A2000 weights/lineage changed')
 require(audit['passed'] and audit['all_AdamW_steps']==2000 and audit['adapter_equals_checkpoint'] and audit['parent_step']==1613,'Incomplete actual terminal audit')
 require(audit['checkpoint']['sha256']==sha(audit['checkpoint']['path']) and audit['adapter']==b['adapter'],'Audited checkpoint changed')
 require(identity['execution']['to_step']==2000 and identity['execution']['additional_updates']==387,'Wrong continuation identity')
 for p,h in b['source_sha256'].items():require(sha(p)==h,'A2000 source changed: '+p)
 return b

def load_bound_model(path,expected_sha,*,device='cuda:0'):
 b=read_binding(path,expected_sha)
 from lora_dev150_final_v1.loader import load_bound_model as load_original
 from route_reassessment_20261004.action_lora.action_lora import load_adapter
 from continuous_model_v3.loader import A_BINDING_SHA
 view,handle,wire,base=load_original(b['original_A_binding']['path'],A_BINDING_SHA,device=device)
 loaded=load_adapter(handle,b['adapter']['path']);handle.assert_integrity();view.model.eval()
 old=dict(wire['lora_serving_identity']);old.update(binding_sha256=expected_sha,adapter_file_sha256=b['adapter']['sha256'],adapter_payload_sha256=loaded['payload_sha256'],endpoint_updates=2000,endpoint_samples=129536,completion_audit_sha256=b['audit']['sha256'],source_sha256=b['source_sha256'])
 wire.update(lora_serving_identity=old,loader_kind='original_base_then_continued_A2000_adapter')
 require(read_binding(path,expected_sha)==b,'A2000 changed during load')
 return view,handle,wire,b
