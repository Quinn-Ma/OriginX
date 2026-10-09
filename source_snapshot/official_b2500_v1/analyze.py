"""All2500 complete and provenance checked before any score; one frozen B arm."""
from pathlib import Path
from collections import Counter
import json,math,time
import numpy as np
from official_b2500_v1 import runner
R=Path(__file__).resolve().parent.parent;D=R/'results/official-b2500-v1';P=R/'official_b2500_v1'
read=runner.read;sha=runner.sha

def wilson(s,n):
 z=1.959963984540054;p=s/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,c-h),min(1,c+h)]
def statistic(rows):
 n=len(rows);s=sum(int(r['success']) for r in rows)
 return dict(successes=s,episodes=n,rate=s/n,descriptive_wilson95=wilson(s,n))
def intervals(rows,task_names,seed=2080702600):
 rng=np.random.default_rng(seed);v=np.array([[int(x['success']) for x in rows if x['task']==t] for t in task_names]);assert v.shape==(len(task_names),50)
 draws=[];clusters=[]
 for _ in range(100):
  idx=rng.integers(0,50,size=(1000,len(task_names),50));draws.extend(v[np.arange(len(task_names))[None,:,None],idx].mean((1,2)));clusters.extend(v.mean(1)[rng.integers(0,len(task_names),size=(1000,len(task_names)))].mean(1))
 return dict(method='stratified percentile bootstrap over50episodes within each fixed task',draws=100000,seed=seed,confidence=.95,rate_ci95=np.quantile(draws,[.025,.975]).tolist(),task_cluster_sensitivity_ci95=np.quantile(clusters,[.025,.975]).tolist())

def main():
 m=runner.design(P/'manifest.json');protocol=read(P/'protocol.json');assert sha(P/'protocol.json')==runner.PROTOCOL_SHA
 cohort=D/'cohort/B';completion=read(cohort/'completion.json');summary=read(cohort/'summary.json');records=read(cohort/'records.json')
 assert summary['complete'] and summary['completed']==summary['planned']==summary['attempted']==2500 and summary['infrastructure_unknown']==0 and summary['not_attempted']==0
 assert completion['selected_assignments_complete'] and completion['active_children']==0 and completion['stop_reasons']==[] and completion['dispatch_order']==list(range(2500))
 assert len(records)==len({x['id'] for x in records})==2500
 jobs={j['id']:j for j in m['jobs']};assert set(jobs)=={x['id'] for x in records}
 # Never aggregate if any own evaluation process is still active.
 owners=[read(cohort/'launch.json')]+[read(p) for p in (cohort/'sessions').glob('*.json')]+[read(p) for p in (cohort/'processes').glob('*.json')]
 assert len(list((cohort/'processes').glob('*.json')))==2500 and not any(runner.process_active(o) for o in owners)
 binding_path=D/'B-bindings.json';binding=read(binding_path);bsha=sha(binding_path)
 assert binding['runner_pins_sha256']==sha(P/'runner-source-pins.json')
 proofs={}
 def pin(p,expected=None):
  p=Path(p);h=sha(p);assert expected is None or h==expected;assert str(p) not in proofs or proofs[str(p)]==h;proofs[str(p)]=h;return p
 pin(P/'manifest.json',runner.MANIFEST_SHA);pin(P/'protocol.json',runner.PROTOCOL_SHA);pin(binding_path,bsha)
 for name,digest in read(P/'runner-source-pins.json').items():pin(P/name,digest)
 for path,digest in binding['source_sha256'].items():pin(path,digest)
 ref=binding['reference_B'];pin(ref['endpoint_path'],ref['endpoint_sha256']);assert ref['endpoint_sha256']==protocol['frozen_B_endpoint_sha256'];ep=read(ref['endpoint_path']);pin(ep['branch']['path'],protocol['frozen_B_branch_sha256'])
 for p,d in ep.get('source_sha256',{}).items():pin(p,d)
 for x in ep.values():
  if isinstance(x,dict) and x.get('path') and x.get('sha256'):pin(x['path'],x['sha256'])
 rows=[]
 for pub in records:
  p=pin(pub['sealed_result_path'],pub['sealed_result_sha256']);row=read(p);j=jobs[pub['id']]
  assert all(row[k]==j[k] for k in ('id','arm','task','env_seed','policy_seed','pair_id','pair_index','assignment_index'))
  assert row['status']=='completed' and type(row['success']) is bool and row['guard_passed'] and row['guard_close_proof']['closed']
  assert row['bindings_sha256']==bsha and row['manifest_sha256']==runner.MANIFEST_SHA
  episode=row['stats']['episodes'][0];assert len(row['stats']['episodes'])==1 and episode['success']==row['success'] and episode['seed']==j['env_seed'] and episode['episode']==episode['global_episode_index']==0
  assert 1<=episode['steps']<=j['horizon'] and (episode['success'] or episode['steps']==j['horizon']) and row['stats']['horizon']==j['horizon']
  assert row['rng_after']['requests_since_reset']==(episode['steps']+15)//16
  g=read(pin(row['guard_close_proof']['path'],row['guard_close_proof']['sha256']));assert runner.guard_complete(g)
  assert g['source_sha256']==sha(P/'readback_guard.py') and g['backend']=='osmesa' and g['infrastructure_guard'] is True
  initial=read(pin(p.parent/'initial_scene.json'));assert initial==row['initial_fingerprints'];pin(p.parent/'initial.xml',initial['xml_sha256'])
  model=binding['models'][row['endpoint_key']];h=read(pin(model['server_manifest'],model['server_manifest_sha256']))['hello'];parity=read(pin(model['parity_report'],model['parity_sha256']))
  for k in ['server_instance','model_assets_sha256','official_server_sha256','multiplex_sources_sha256','model_config_runtime_sha256','lora_serving_identity','stage_serving_identity']:assert row['server_identity'].get(k)==h.get(k)
  assert runner.canonical(h['model_assets_sha256'])==protocol['original_base_assets_sha256'] and h['stage_serving_identity']['arm']=='B' and h['stage_serving_identity']==parity['model']['stage_serving_identity']
  assert parity['passed'] and parity['ambient_rng_unchanged'] and len(parity['trace'])==6 and all(t['full_action_exact'] and t['all_rng_exact'] for t in parity['trace'])
  rows.append(row)
 # All integrity checks complete; aggregate exactly the preregistered2500.
 assert Counter(x['task'] for x in rows)=={t['task']:50 for t in m['tasks']}
 overall=statistic(rows);bytask={t['task']:dict(statistic([r for r in rows if r['task']==t['task']]),stratum=t['stratum']) for t in m['tasks']};strata={}
 for i,s in enumerate(['atomic_seen','composite_seen','composite_unseen']):
  ts=[t['task'] for t in m['tasks'] if t['stratum']==s];subset=[r for r in rows if r['task'] in ts];strata[s]=dict(statistic(subset),interval=intervals(subset,ts,2080702601+i))
 assert abs(overall['rate']-sum(protocol['score']['stratum_weights'][s]*strata[s]['rate'] for s in strata))<1e-12
 for path,digest in proofs.items():assert sha(path)==digest,'Artifact changed before publication'
 report=dict(complete=True,episodes=2500,model='frozen B continuous2000',overall=overall,interval=intervals(rows,[t['task'] for t in m['tasks']]),strata=strata,by_task=bytask,
  target=dict(at_least59percent=overall['successes']>=1475,at_least60percent=overall['successes']>=1500,above_current_reference58point1=overall['rate']>.581,reference_is_not_contemporaneous_baseline=True),
  episode_results=[dict(id=x['id'],task=x['task'],seed=x['env_seed'],success=x['success'],steps=x['stats']['episodes'][0]['steps']) for x in rows],
  source_artifacts=proofs,manifest_sha256=runner.MANIFEST_SHA,protocol_sha256=runner.PROTOCOL_SHA,completed_unix=time.time(),officially_verified=False,leaderboard_submission_started=False,reset_compatibility=protocol['reset'],limits=protocol['limits'])
 out=D/'analysis';out.mkdir(exist_ok=False);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 ci=report['interval']['rate_ci95'];text=f"# B2000：完整50任务本地评测\n\n总体成功：{overall['successes']}/2500，{overall['rate']:.2%}。固定任务分层bootstrap95%区间：[{ci[0]:.2%}, {ci[1]:.2%}]。\n\n"
 for s,v in strata.items():text+=f"- {s}：{v['successes']}/{v['episodes']}，{v['rate']:.2%}。\n"
 text+='\n使用已披露的stable-reset保序修复；官方未核准此修复，结果属于本地完整口径，不代表已获榜单资格。无同期原始模型2500回合，不计算相对官网57.4%的配对显著性。没有自动提交榜单。\n'
 (out/'RESULTS.md').write_text(text);print(text)
if __name__=='__main__':main()
