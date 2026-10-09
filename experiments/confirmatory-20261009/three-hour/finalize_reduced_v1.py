"""No-LLM join after both batch inventories and model cleanup are verified."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, tarfile, time, traceback
from broker import ProcessLock, local_owner, write_json
from sync_multigpu_v1 import ssh

HERE=Path(__file__).resolve().parent
STATE=HERE/'finalization_reduced_v1'
OUTPUTS=HERE.parents[1]/'outputs'
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def status(phase,**extra):
 write_json(STATE/'status.json',dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra))

def finalize():
 attempt=STATE/'attempt.json'
 if attempt.exists():raise RuntimeError('Prior reduced finalization attempted; inspect without rerunning')
 write_json(attempt,dict(started_at=datetime.now(timezone.utc).isoformat(),no_new_trials=True))
 code=r'''
from pathlib import Path
import json,hashlib,subprocess,os,tarfile
r=Path('/ephemeral/qinzhen/robocasa-xr1-20261003');final=r/'results/originx-three-hour-amendment-20261009-v1';old=r/'results/originx-confirmatory-20261009-v1';new=r/'results/originx-confirmatory-multigpu-20261009-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
script=r/'aggregate_reduced_v1.py'
assert sha(script)=='e5ef204cd61f19726d26afbb4a1424fe876f27d85dac15fc775a7a61d31f6151'
config=json.loads((new/'config.json').read_text());cmd=[config['python'],'-B',str(script)]
for label,out in [('old',old),('new',new)]:
 state=out/'broker_inventory_final';terminal=state/'terminal-inventory.json'
 cmd+=['--'+label+'-broker-state',str(state),'--'+label+'-broker-terminal',str(terminal),'--'+label+'-broker-terminal-sha256',sha(terminal)]
with (final/'aggregate-reduced-command.json').open('x') as f:json.dump(dict(command=cmd,script_sha256=sha(script)),f,indent=2)
p=subprocess.run(cmd,cwd=r,env=dict(os.environ,PYTHONPATH=str(r),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=1200)
(final/'aggregate-reduced.stdout.log').write_text(p.stdout);(final/'aggregate-reduced.stderr.log').write_text(p.stderr)
assert p.returncode==0,p.stderr[-3000:]
a=final/'analysis-reduced.tar.gz'
with tarfile.open(a,'x:gz') as t:
 for name in ['analysis-reduced','aggregate-reduced-command.json','aggregate-reduced.stdout.log','aggregate-reduced.stderr.log']:
  t.add(final/name,arcname=name)
print(json.dumps(dict(path=str(a),sha256=sha(a),size=a.stat().st_size,summary=json.loads(p.stdout))))
'''
 meta=ssh(code,1300);artifact=STATE/'analysis-reduced.tar.gz'
 subprocess.run(['scp','-q','hkust-cluster:'+meta['path'],str(artifact)],check=True,timeout=900)
 if artifact.stat().st_size!=meta['size'] or sha(artifact)!=meta['sha256']:raise RuntimeError('Reduced archive SHA mismatch')
 target=STATE/'evidence';target.mkdir(exist_ok=True)
 with tarfile.open(artifact) as t:
  for m in t.getmembers():
   if m.issym() or m.islnk() or not (target/m.name).resolve().is_relative_to(target.resolve()):raise RuntimeError('Unsafe reduced archive')
  t.extractall(target,filter='data')
 write_json(STATE/'archive-receipt.json',dict(**meta,local=str(artifact),verified_at=datetime.now(timezone.utc).isoformat()))
 report=read(target/'analysis-reduced/report.json');assert report['N_per_policy']==50 and report['amended_arm_outcomes']==350
 usage=read(target/'analysis-reduced/cli_usage.json')
 write_json(OUTPUTS/'OriginX_50案例补充实验结果.json',dict(report,local_verified_archive=read(STATE/'archive-receipt.json')))
 write_json(OUTPUTS/'OriginX_50案例补充实验CLI用量.json',usage)
 lines=['# OriginX 50案例补充实验','',
  '按用户三小时交付要求，在原始新seed清单中对每个任务固定选择最小seed，共50个不同案例、350个分组结果。缩减发生在原全量运行开始后，选择规则不读取结果。',
  '原1496/2500基准和56/1004历史失败救援保持不变。本报告不代表原2500案例确认实验完成。',
  f"有效分组记录：{report['normalized_amended_valid_arm_records']}/350；已认领分组：{report['amended_claimed_arm_outcomes']}/350。旧批次范围外记录单独保留：{report['outside_cohort_prior_arm_records_retained']}。",'',
  '|策略/对照|案例分母|匹配|救回|退化|未解决|完整分母净效应界限|','|---|---:|---:|---:|---:|---:|---|']
 for policy,v in report['policies'].items():
  for name,c in v['comparisons'].items():
   bounds=c['full_cohort_net_effect_bounds']
   lines.append(f"|{policy}/{name}|{c['N']}|{c['categories']['matched']}|{c['rescues']}|{c['regressions']}|{c['unresolved_comparisons']}|[{bounds['low']:.4f}, {bounds['high']:.4f}]|")
 lines+=['','界限包含未知与不匹配，不是置信区间。小样本、启动后缩减及完整分母结果均须披露。',
  '全部回合和模型已在汇总前核验结束；原始两批证据和联合汇总已按SHA回传D盘。',
  'CLI使用按真实调用去重，计入旧批次范围外开销；计量缺失保持未知。',
  'GR00T正式跨架构评测是否执行须按独立证据说明，开发测试不能代替正式效果。',
  '论文、补充材料、文章和公开网站仍需代理依据此报告更新。','']
 (OUTPUTS/'OriginX_50案例补充实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
 status('completed',summary=meta['summary'],publications_pending=True,archive_sha256=meta['sha256'])

def main():
 STATE.mkdir(exist_ok=True)
 with ProcessLock(STATE/'finalizer.lock'):
  if (STATE/'owner.json').exists():raise RuntimeError('Prior reduced finalizer owner; inspect instead of duplicate')
  write_json(STATE/'owner.json',local_owner());end=time.monotonic()+10000
  while time.monotonic()<end:
   phases=[]
   for name in ['finalization_v3','finalization_multigpu_v1']:
    p=HERE/name/'status.json';phases.append(read(p).get('phase') if p.exists() else None)
   if 'failed' in phases:status('blocked_batch_finalizer_failed',batch_phases=phases);return
   if phases==['completed','completed']:
    try:status('joining');finalize()
    except BaseException:status('failed',error=traceback.format_exc(),no_automatic_retry=True);raise
    return
   status('waiting_batch_finalizers',batch_phases=phases);time.sleep(30)
  status('deadline',new_results_claimed=False,reason='Local finalizer time limit')

if __name__=='__main__':main()
