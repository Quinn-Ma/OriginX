"""No-LLM terminal inventory/aggregation for the frozen prospective campaign.

No trials, model loads, CLI generations, publication or training are started.
An interrupted invocation remains unknown; it is never retried here.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, tarfile, time, traceback
from datetime import datetime, timezone
from broker import ProcessLock, local_owner, local_owner_active, write_json
from admit_full import live_cli_for_batches

HERE=Path(__file__).resolve().parent
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
NAME='originx-confirmatory-20261009-v1'
REMOTE=ROOT+'/results/'+NAME
STATE=HERE/'broker_full_v1'
OUT=HERE/'finalization_v2'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()
def ssh(code,timeout=90):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hkust-cluster','python3','-'],
        input=code.encode(),capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-4000:]+p.stdout.decode(errors='replace')[-2000:])
    return json.loads(p.stdout)
def status(phase,**extra):
    write_json(OUT/'status.json',dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra))

def terminal_ready(snapshot):
    data=snapshot.get('remote',{}).get(NAME,{})
    return 'campaign-finished.json' in data

def markdown(report, audit):
    lines=['# OriginX 前瞻性补充实验结果','',
        '原生 B 基准 1496/2500、历史条件救援 56/1004 保持不变。以下为独立的新 seed 队列，未获官方榜单验证。','',
        f"固定 {report['N_per_policy']} 个不同环境案例，计划 {report['planned_arm_outcomes']} 个分臂结果；记录 {report['present_arm_records']}，原始证据有效 {report['raw_evidence_valid']}。",'',
        '| 策略/比较 | 分母 | 配对有效 | 新增成功 | 新增失败 | 未解决比较 | 完整分母净效应边界 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for policy, value in report['policies'].items():
        for name,c in value['comparisons'].items():
            bounds=c['full_cohort_net_effect_bounds']
            lines.append(f"| {policy}/{name} | {c['N']} | {c['categories']['matched']} | {c['rescues']} | {c['regressions']} | {c['unresolved_comparisons']} | [{bounds['low']:.4f}, {bounds['high']:.4f}] |")
    lines+=['','边界是包含未知与不匹配案例的识别区间，不是置信区间；有效子集的任务分块 bootstrap 单独保留在 JSON。',
        '开发结果不纳入以上分母，未运行/技术故障没有被删除或计作确定的策略失败。',
        f"自有模型清理核验：{audit['owned_models_stopped']}。完整原始证据本地归档已验证：{audit['raw_archive_verified']}。",'',
        '实际 token、延迟、未完成 CLI 与缺失计量详见结果 JSON 的 cli_usage；无新增重置或付费。',
        '此自动收尾文件保留真实结果；论文、GitHub 和项目网站的更新仍由代理核对后执行。','']
    return '\n'.join(lines)

def live_owned_cli(starts, state):
    if starts:
        return live_cli_for_batches([str(p.parent) for p in starts])
    # No empty-string process match: the durable broker ledger must agree that
    # no CLI was invoked before the empty batch inventory is accepted.
    statuses = sorted(Path(state).glob('**/*status*.json'))
    if not statuses or any(read(p).get('cli_calls', 0) != 0 for p in statuses):
        raise RuntimeError('Empty CLI starts conflict with broker accounting')
    if list((Path(state)/'batches').glob('*/receipt.json')):
        raise RuntimeError('Receipt exists without a durable CLI start')
    return []


def finalize(snapshot):
    data=snapshot['remote'][NAME]
    if local_owner_active(read(STATE/'owner.json')):raise RuntimeError('Broker still active; do not freeze changing evidence')
    starts=list((STATE/'batches').glob('*/cli_started.json'))
    if live_owned_cli(starts, STATE):raise RuntimeError('Owned CLI batch still active')
    inventory={str(p.relative_to(STATE)).replace('\\','/'):sha(p) for p in STATE.rglob('*') if p.is_file() and p.name!='broker.lock'}
    write_json(OUT/'broker-terminal.json',dict(owner=read(STATE/'owner.json'),owner_inactive=True,
        cli_scoped_processes_absent=True,per_cli_pid_start_cwd_available=False,
        identity=read(STATE/'identity.json'),files_sha256=inventory))
    archive=OUT/'broker-terminal.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for name in sorted(inventory):t.add(STATE/name,arcname='broker_inventory_final/'+name)
        t.add(OUT/'broker-terminal.json',arcname='broker_inventory_final/terminal-inventory.json')
    digest=sha(archive)
    subprocess.run(['scp','-q',str(archive),'hkust-cluster:'+REMOTE+'/broker-terminal-final.tar.gz'],check=True,timeout=300)
    response=ssh('''from pathlib import Path
import hashlib,json,tarfile,subprocess,os
d=Path(%r);root=Path(%r);a=d/'broker-terminal-final.tar.gz'
if hashlib.sha256(a.read_bytes()).hexdigest()!=%r:raise RuntimeError('Broker archive hash mismatch')
if (d/'broker_inventory_final').exists() or (d/'analysis-final').exists():raise RuntimeError('Final evidence already exists; preserve it')
with tarfile.open(a) as t:
 for m in t.getmembers():
  if not m.isfile() or m.issym() or m.islnk() or not m.name.startswith('broker_inventory_final/') or '..' in Path(m.name).parts:raise RuntimeError('Unsafe archive')
 t.extractall(d,filter='data')
c=json.loads((d/'config.json').read_text())
p=subprocess.run([c['python'],'-B','-m','originx_confirmatory_20261009.aggregate','--config',str(d/'config.json'),'--broker-state',str(d/'broker_inventory_final'),'--output-directory',str(d/'analysis-final')],cwd=root,env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=3600)
(d/'final-aggregate.stdout.log').write_text(p.stdout);(d/'final-aggregate.stderr.log').write_text(p.stderr)
if p.returncode:raise RuntimeError('Final aggregate failed: '+p.stderr[-2000:])
a=d/'final-analysis-evidence.tar.gz'
with tarfile.open(a,'x:gz') as t:
 for name in ['analysis-final','broker_inventory_final','final-aggregate.stdout.log','final-aggregate.stderr.log']:t.add(d/name,arcname=name)
print(json.dumps(dict(path=str(a),size=a.stat().st_size,sha256=hashlib.sha256(a.read_bytes()).hexdigest())))
'''%(REMOTE,ROOT,digest),timeout=3700)
    local=OUT/'final-analysis-evidence.tar.gz'
    subprocess.run(['scp','-q','hkust-cluster:'+response['path'],str(local)],check=True,timeout=1800)
    if sha(local)!=response['sha256'] or local.stat().st_size!=response['size']:raise RuntimeError('Final analysis archive verification failed')
    with tarfile.open(local) as t:
        for m in t.getmembers():
            if m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts:raise RuntimeError('Unsafe final analysis archive')
        t.extractall(OUT/'evidence',filter='data')
    write_json(OUT/'archive-receipt.json',response)
    report=read(OUT/'evidence/analysis-final/report.json')
    if report.get('development') is not False or report['N_per_policy']!=2500:raise RuntimeError('Wrong study report')
    raw=HERE/'sync'/NAME/'archive-receipt.json'
    audit=dict(owned_models_stopped=data.get('cleanup.json',{}).get('all_owned_services_stopped') is True,
        raw_archive_verified=raw.exists(),raw_archive_receipt=read(raw) if raw.exists() else None,
        broker_owner_inactive=True,final_analysis_archive=response,generated_at=datetime.now(timezone.utc).isoformat(),
        report_source_sha256=sha(OUT/'evidence/analysis-final/report.json'))
    outputs=HERE.parents[1]/'outputs'
    write_json(outputs/'OriginX_补充实验结果.json',dict(report,finalization=audit))
    (outputs/'OriginX_补充实验报告.md').write_text(markdown(report,audit),encoding='utf-8')
    status('completed',audit=audit)

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    with ProcessLock(OUT/'finalizer.lock'):
        if (OUT/'owner.json').exists():raise RuntimeError('Existing finalizer owner; inspect rather than duplicate')
        write_json(OUT/'owner.json',local_owner());deadline=time.monotonic()+6*86400
        while time.monotonic()<deadline:
            snapshot=read(HERE/'sync/snapshot.json')
            if terminal_ready(snapshot):
                if (STATE/'owner.json').exists() and not local_owner_active(read(STATE/'owner.json')):
                    if not (HERE/'sync'/NAME/'archive-receipt.json').exists():status('waiting_verified_raw_archive')
                    else:
                        try:status('finalizing');finalize(snapshot)
                        except BaseException:status('failed',error=traceback.format_exc(),no_automatic_retry=True);raise
                        return
                else:status('waiting_broker_exit',no_processes_signalled=True)
            else:status('waiting_campaign')
            if a.once:return
            time.sleep(120)
        status('wait_deadline_reached',new_result_claimed=False)

if __name__=='__main__':main()
