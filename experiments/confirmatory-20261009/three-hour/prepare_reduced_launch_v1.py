"""Single-use preparation bridge; no rollouts, model starts, or CLI invocations."""
from pathlib import Path
import hashlib, json, os, subprocess, sys

ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
OLD=ROOT/'results/originx-confirmatory-20261009-v1'
SERVICES=ROOT/'results/originx-confirmatory-multigpu-services-20261009-v1'
OUT=ROOT/'results/originx-three-hour-amendment-20261009-v1'
PINS={'multigpu_services_v1.py':'9504e17868783952a17181c95102ff224b7c92b1f558185ac8ce9714741864e5',
 'continue_multigpu_v1.py':'142b0ac2bdccd4edffc7515134227d2961d9be7796734e52f62601fcf0b18968',
 'broker_multigpu_v1.py':'201b896300a93f9d41ad9d449c488f8a589c2afa7a9a4df8a8d89555605346e3'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)

def main():
 assert not (OUT/'continuation-debit.json').exists(),'Prior preparation attempt exists; inspect rather than repeat'
 for name,digest in PINS.items():assert sha(ROOT/name)==digest,(name,'changed')
 sys.path.insert(0,str(ROOT));from continue_multigpu_v1 import owner_inactive,verify_broker_terminal
 config=read(OLD/'config.json')
 assert read(OLD/'cleanup.json')['all_owned_services_stopped'] is True
 assert read(OLD/'completion.json')['all_children_drained'] is True
 assert owner_inactive(read(OLD/'campaign.owner.json'))
 terminal=OLD/'broker_inventory_final/terminal-inventory.json'
 audit=verify_broker_terminal(terminal,terminal.parent,OLD,sha(OLD/'config.json'),config['manifest_sha256'])
 usage=read(OLD/'analysis-final/cli_usage.json')
 assert usage['complete_input_output_token_accounting'] is True
 assert usage['inventory_complete'] is True and not usage['invalid_receipts'] and not usage['interrupted_cli_starts_without_receipt']
 assert audit['cli_starts']==usage['unique_observed_cli_invocations']
 prior=dict(cli_calls=usage['unique_observed_cli_invocations'],input_tokens=usage['known_usage_totals']['input_tokens'],output_tokens=usage['known_usage_totals']['output_tokens'])
 assert all(type(v) is int and v>=0 for v in prior.values())
 profiles=[];groups=[]
 for gpu in (3,6):
  g=SERVICES/('gpu'+str(gpu));gate=read(g/'ready.json')
  assert gate['passed'] is True and gate['same_socket_action_rng_parity_passed'] is True and gate['hold_seconds']>=360
  assert gate['helper_sha256']==PINS['multigpu_services_v1.py']
  assert sha(g/'services-ready.json')==gate['services_ready_sha256']
  assert sha(Path(gate['parity_path']))==gate['parity_sha256']
  assert not (g/'draining.json').exists() and not owner_inactive(read(g/'owner.json'))
  ms=read(g/'services-ready.json')['models'];assert len(ms)==6
  for m in ms:
   assert not owner_inactive(m['owner']) and m['gpu_index']==gpu
   assert sha(Path(m['server_manifest']))==m['server_manifest_sha256']
  profiles.extend(ms);groups.append(ref(g/'ready.json'))
 assert len({m['port'] for m in profiles})==12
 debit=OUT/'continuation-debit.json'
 write(debit,dict(schema='originx_main_continuation_debit_v1',prior_usage=prior,
  complete_input_output_token_accounting=True,prior_broker_inactive=True,
  authority_files=[ref(p) for p in [OLD/'analysis-final/cli_usage.json',terminal,OLD/'config.json',OLD/'cleanup.json']],
  scope='Original main batch consumption; earlier development remains separately accounted; no main-cap reset',
  no_paid_topup=True,no_quota_reset=True))
 ready=SERVICES/'combined-services-ready.json'
 write(ready,dict(ready=True,models=profiles,model_count=12,clients=72,groups=groups,namespace='originx_confirmatory_20261009'))
 admission=SERVICES/'combined-service-admission.json'
 write(admission,dict(schema='originx_multigpu_service_admission_v1',passed=True,services_ready_sha256=sha(ready),
  same_socket_action_rng_parity_passed=True,hold_seconds=360,groups=groups,helper_sources=PINS))
 cmd=[str(ROOT/'envs/training/bin/python'),'-B',str(ROOT/'continue_multigpu_v1.py'),'prepare','--duration-seconds','5400']
 for flag,p in [('services-ready',ready),('service-admission',admission),('broker-terminal',terminal),('broker-source',ROOT/'broker_multigpu_v1.py'),('debit',debit)]:
  cmd+=['--'+flag,str(p),'--'+flag+'-sha256',sha(p)]
 result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=300)
 (OUT/'prepare-continuation.stdout.log').write_text(result.stdout)
 (OUT/'prepare-continuation.stderr.log').write_text(result.stderr)
 assert result.returncode==0,result.stderr[-2500:]
 print(result.stdout)

if __name__=='__main__':main()
