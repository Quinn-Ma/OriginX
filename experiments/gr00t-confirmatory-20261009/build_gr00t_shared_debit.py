"""Build a sealed shared-budget bundle only after all four studies terminate.

Explicit invocation: python -B build_gr00t_shared_debit.py --plan PLAN.json
  --output-directory NEW_LOCAL_DIRECTORY --remote-directory ABS_REMOTE_DIRECTORY

PLAN schema originx_shared_debit_build_plan_v1 has exactly four `studies` rows:
{study, files:{config, manifest, source_freeze, campaign_finished, completion,
usage, authority[, failure_terminal]}}. Each file is {local:absolute Windows
path, remote:absolute path under the fixed remote project}. The local durable
broker directories and source files are fixed below, never accepted from an
archive-copy path. `failure_terminal` is required only for failed GR00T dev-v1.

No model, rollout, Astra, reset or paid service is invoked. The only network
operation is a read-only SSH rehash/owner audit. No output is written until all
local ledgers and remote terminal/source evidence pass. The output directory is
exclusive. Upload the resulting directory at its declared remote path before
using debit.json; this tool does not upload or start the later experiment.

The main frozen broker enforces its own receipt cap without pre-debiting the
development calls. This later debit records the actual combined expenditure;
it is not a claim that historical combined expenditure was strictly capped.
In-flight CLI usage is only known after terminal accounting. Exhausted or
unknown combined allowance cannot authorize any new GR00T formal calls.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
HELPER_SHA='a9cc287845d188be23f2bf26cbb94b82d4915be5134669cd5415537b9a4fa170'
STUDIES=(
 'originx-confirmatory-development-20261009-v2',
 'originx-confirmatory-20261009-v1',
 'originx-gr00t-confirmatory-development-20261009-v1',
 'originx-gr00t-confirmatory-development-20261009-v2')
FAILED=STUDIES[2]
LOCATIONS={
 STUDIES[0]:('confirmatory_20261009','broker_development_v2'),
 STUDIES[1]:('confirmatory_20261009','broker_full_v1'),
 STUDIES[2]:('originx_gr00t_confirmation_20261009','broker_development_v1'),
 STUDIES[3]:('originx_gr00t_confirmation_20261009_v2','broker_development_v2')}
CAPS=dict(cli_calls=1500,input_tokens=25000000,output_tokens=2000000)
FIELDS=('input_tokens','cached_input_tokens','output_tokens','reasoning_tokens')


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Missing/symlink evidence: '+str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def scoped_remote(path):
    p=PurePosixPath(path)
    require(str(p)==path and p.is_absolute() and '..' not in p.parts
            and p.is_relative_to(PurePosixPath(ROOT)),'Unscoped remote evidence path')
    return path


def inventory(directory):
    rows={}
    for p in sorted(Path(directory).rglob('*')):
        require(not p.is_symlink(),'Symlink in durable broker ledger')
        if p.is_file() and p.name!='broker.lock':rows[p.relative_to(directory).as_posix()]=sha(p)
    return rows


def usage_rows(stdout,receipt):
    rows=[]
    for line in Path(stdout).read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        event=json.loads(line);require(isinstance(event,dict),'Non-object CLI event')
        if event.get('type')=='turn.completed':
            row=event.get('usage');require(isinstance(row,dict),'Completed CLI turn lacks usage')
            for key in ('input_tokens','output_tokens'):
                require(type(row.get(key)) is int and row[key]>=0,'Unknown CLI input/output usage cannot be debited as zero')
            rows.append(row)
    require(rows and receipt.get('usage_raw')==rows,'Missing/inconsistent raw usage rows')
    totals={k:None for k in FIELDS}
    for row in rows:
        for key in FIELDS:
            value=row.get(key)
            if key=='reasoning_tokens' and value is None:
                value=row.get('reasoning_output_tokens')
                if value is None and isinstance(row.get('output_tokens_details'),dict):value=row['output_tokens_details'].get('reasoning_tokens')
            if value is not None:
                require(type(value) is int and value>=0,'Malformed CLI usage counter')
                if key=='reasoning_tokens':require(value<=row['output_tokens'],'Reasoning cannot exceed inclusive output')
                if key=='cached_input_tokens':require(value<=row['input_tokens'],'Cached input cannot exceed input')
                totals[key]=(totals[key] or 0)+value
    require(all(receipt.get('usage_totals',{}).get(k)==v for k,v in totals.items()),'Receipt/stdout usage mismatch')
    return totals


def verify_call(folder,config_sha,manifest_sha,source_sha,jobs):
    folder=Path(folder);started=read(folder/'cli_started.json');batch=read(folder/'batch.json');r=read(folder/'receipt.json')
    require(batch.get('batch_id')==folder.name and batch.get('broker_source_sha256')==source_sha
            and batch.get('config_sha256')==config_sha and batch.get('manifest_sha256')==manifest_sha,'Batch source/study authority mismatch')
    require(r.get('schema')=='astra_cli_receipt_v1' and r.get('cli_executed') is True
            and type(r.get('returncode')) is int and r.get('finished_at') and r.get('process_exception') is None,
            'CLI start lacks a terminal wait-return receipt')
    require(r.get('started_at')==started.get('started_at') and r.get('request_ids')==started.get('request_ids'), 'CLI start/receipt mismatch')
    require(r.get('model')==r.get('model_requested')=='gpt-6-astra' and r.get('reasoning_effort')=='high','Wrong executed CLI model/effort')
    command=r.get('command',[])
    require(command.count('--model')==1 and command[command.index('--model')+1]=='gpt-6-astra'
            and 'model_reasoning_effort=high' in command,'CLI argv model/effort mismatch')
    require(sha(folder/'batch.json')==r.get('batch_manifest_sha256') and sha(folder/'stdout.jsonl')==r.get('stdout_sha256'),
            'Batch/CLI stdout hash mismatch')
    for filename,key in [('prompt.txt','prompt_sha256'),('schema.json','output_schema_sha256'),('stderr.txt','stderr_sha256')]:
        require(sha(folder/filename)==r.get(key),'CLI input/output hash mismatch: '+filename)
    if r.get('mosaic_sha256') is not None:require(sha(folder/'mosaic.png')==r['mosaic_sha256'],'CLI image hash mismatch')
    bindings=[{k:x[k] for k in ('request_id','request_token','request_sha256')} for x in batch['requests']]
    require(bindings==r.get('requests') and [x['request_id'] for x in bindings]==r.get('request_ids'),'Batch/receipt request binding mismatch')
    require(bindings and all(x['request_id'] in jobs for x in bindings),'CLI request outside manifest')
    require(len({x['request_id'] for x in bindings})==len(bindings),'Duplicate request in CLI batch')
    for binding in bindings:
        request=folder/'requests'/binding['request_id']/'request.json'
        require(sha(request)==binding['request_sha256'],'Durable downloaded request hash mismatch')
    totals=usage_rows(folder/'stdout.jsonl',r)
    key=json_sha({k:r.get(k) for k in ('started_at','command','requests','stdout_sha256','batch_manifest_sha256')})
    return dict(invocation_key=key,batch_id=folder.name,receipt_sha256=sha(folder/'receipt.json'),
                receipt=r,usage=totals,folder=folder)


def default_owner_active(source,owner):
    return module(source,'debit_owner_'+sha(source)[:16]).local_owner_active(owner)


def read_only_remote_check(payload):
    code='''from pathlib import Path
import json,hashlib,os
p=json.loads(PAYLOAD);root=Path(p['root']);checked=0
for name,expected in p['sha256'].items():
 f=Path(name)
 if not f.is_file() or f.is_symlink() or not f.resolve().is_relative_to(root):raise RuntimeError('Invalid scoped evidence '+name)
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8388608),b''):h.update(b)
 if h.hexdigest()!=expected:raise RuntimeError('Changed terminal/source evidence '+name)
 checked+=1
def inactive(owner):
 d=Path('/proc')/str(owner['pid'])
 try:
  x=(d/'stat').read_text().rsplit(')',1)[1].split()
  return int(x[19])!=owner['process_start_ticks'] or x[0] in ('Z','X')
 except FileNotFoundError:return True
owners=[]
for name in p['studies']:
 out=root/'results'/name;c=json.loads((out/'config.json').read_text());finished=json.loads((out/'campaign-finished.json').read_text())
 if type(finished.get('failed')) is not bool:raise RuntimeError('Missing terminal campaign marker')
 paths=list((out/'processes').glob('*.json'))+list(out.glob('*.owner.json'))
 owners.extend(json.loads(x.read_text()) for x in paths)
 owners.extend(m['owner'] for m in c['models'])
 completion=json.loads((out/'completion.json').read_text());owners.extend(completion.get('remaining_owned_workers',[]))
 if name==p['failed']:
  fail=json.loads((out/'failure-terminal-audit.json').read_text());owners.extend(x['recorded_owner'] for x in fail['checks'])
for owner in owners:
 if not inactive(owner):raise RuntimeError('Prior owned worker/model remains live: '+str(owner['pid']))
print(json.dumps(dict(passed=True,remote_files_verified=checked,owner_records_checked=len(owners),all_prior_remote_owners_inactive=True)))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    completed=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hkust-cluster','python3','-'],
        input=code.encode(),capture_output=True,timeout=900)
    require(completed.returncode==0,'Read-only remote source/terminal audit failed: '+completed.stderr.decode(errors='replace')[-2500:])
    result=json.loads(completed.stdout);require(result.get('passed') is True and result.get('all_prior_remote_owners_inactive') is True,'Remote owners not terminal')
    return result


def validate(plan,*,work=WORK,owner_active=default_owner_active,process_scan=None,remote_check=read_only_remote_check):
    require(plan.get('schema')=='originx_shared_debit_build_plan_v1','Wrong build-plan schema')
    rows=plan.get('studies',[]);require(len(rows)==4 and {x['study'] for x in rows}==set(STUDIES),'Exactly four fixed prior studies required')
    helper_path=Path(work)/'confirmatory_20261009/admit_full.py'
    require(sha(helper_path)==HELPER_SHA,'Frozen audit helper changed')
    h=module(helper_path,'shared_debit_frozen_helper')
    process_scan=h.live_cli_for_batches if process_scan is None else process_scan
    remote_hashes={};local_hashes={};verified=[];all_invocations=set();all_starts=[]
    def bind(local,remote=None):
        local=Path(local).resolve();value=sha(local);local_hashes[str(local)]=value
        if remote:
            scoped_remote(remote)
            require(remote not in remote_hashes or remote_hashes[remote]==value,'Contradictory remote evidence bindings')
            remote_hashes[remote]=value
        return value
    for item in sorted(rows,key=lambda x:STUDIES.index(x['study'])):
        name=item['study'];subdir,state_name=LOCATIONS[name];state=(Path(work)/subdir/state_name).resolve();source=(Path(work)/subdir/'broker.py').resolve()
        require(state.is_dir() and not state.is_symlink(),'Original durable broker directory unavailable')
        files=item['files'];needed={'config','manifest','source_freeze','campaign_finished','completion','usage','authority'}
        if name==FAILED:needed.add('failure_terminal')
        require(set(files)==needed,'Study file plan incomplete or unexpected')
        parsed={};file_hashes={}
        for key,ref in files.items():
            require(set(ref)=={'local','remote'},'Evidence plan references require local and remote path')
            file_hashes[key]=bind(ref['local'],ref['remote']);parsed[key]=read(ref['local'])
        output=ROOT+'/results/'+name;config=parsed['config'];cs=file_hashes['config'];ms=file_hashes['manifest']
        for key,filename in [('config','config.json'),('manifest','manifest.json'),('completion','completion.json'),('campaign_finished','campaign-finished.json')]:
            require(files[key]['remote']==output+'/'+filename,'Study file points outside its actual output')
        require(config.get('output')==output and config.get('root')==ROOT and config.get('manifest')==output+'/manifest.json'
                and config.get('manifest_sha256')==ms,'Configuration/manifest identity mismatch')
        expected_schema='originx_gr00t_confirmatory_config_v1' if name.startswith('originx-gr00t-') else 'originx_confirmatory_config_v1'
        require(config.get('schema')==expected_schema and config.get('development') is (name!=STUDIES[1]),'Wrong study configuration schema/scope')
        for key,filename in [('usage','cli_usage.json'),('authority','authority.json')]:
            require(PurePosixPath(files[key]['remote']).is_relative_to(PurePosixPath(output))
                    and PurePosixPath(files[key]['remote']).name==filename,'Unscoped study accounting evidence')
        if name==FAILED:require(files['failure_terminal']['remote']==output+'/failure-terminal-audit.json','Unscoped failed-development proof')
        require(config.get('source_freeze')==files['source_freeze']['remote'] and config.get('source_freeze_sha256')==file_hashes['source_freeze'],'Frozen source authority mismatch')
        require(sha(source)==config['broker_source_sha256']==parsed['source_freeze']['broker_source_sha256'],'Actual frozen broker source differs')
        bind(source)
        for path,digest in config['source_sha256'].items():
            scoped_remote(path);require(path not in remote_hashes or remote_hashes[path]==digest,'Conflicting frozen source identity');remote_hashes[path]=digest
        require(config['source_sha256'] and parsed['source_freeze']['source_sha256'],'Empty source authority')
        for filename,digest in parsed['source_freeze']['source_sha256'].items():
            path=str(PurePosixPath(config['source_freeze']).parent/filename)
            require(PurePosixPath(filename).name==filename and config['source_sha256'].get(path)==digest,'Freeze/module source binding absent')
        authority=parsed['authority']
        require(authority.get('config_sha256')==cs and authority.get('manifest_sha256')==ms and authority.get('source_checks'),'Stale source authority report')
        for entry in authority['source_checks']:
            require(entry.get('valid') is True and entry['actual_sha256']==entry['expected_sha256'],'Invalid prior source authority')
            path=scoped_remote(entry['path']);digest=entry['expected_sha256']
            require(path not in remote_hashes or remote_hashes[path]==digest,'Conflicting audited source identity');remote_hashes[path]=digest
        identity=read(state/'identity.json');owner=read(state/'owner.json');status=read(state/'broker_status.json')
        require(identity.get('remote_output')==output and identity.get('config_sha256')==cs and identity.get('manifest_sha256')==ms
                and identity.get('model')=='gpt-6-astra' and identity.get('reasoning_effort')=='high','Broker belongs to another study')
        argv=owner.get('command',[])
        require(argv and Path(argv[0]).resolve()==source and argv.count('--local-output')==1
                and Path(argv[argv.index('--local-output')+1]).resolve()==state,'Archived broker copy cannot count as original durable ledger')
        require(not owner_active(source,owner),'A prior broker owner is still alive')
        require(status.get('state') in ('finished','stopped') and type(status.get('cli_calls')) is int,'Broker status not terminal with explicit count')
        require(type(parsed['campaign_finished'].get('failed')) is bool,'Campaign terminal evidence missing')
        if name!=FAILED:require(parsed['completion'].get('all_children_drained') is True,'Prior workers are not drained')
        starts=sorted((state/'batches').glob('*/cli_started.json'));receipts=sorted((state/'batches').glob('*/receipt.json'))
        require({p.parent for p in starts}=={p.parent for p in receipts} and len(starts)==status['cli_calls'], 'Durable starts/terminal receipts/accounting are not one-to-one')
        all_starts.extend(str(p.parent) for p in starts)
        jobs={j['request_id']:j for j in parsed['manifest']['jobs']}
        require(jobs and len(jobs)==len(parsed['manifest']['jobs']),'Empty/duplicate manifest request identities')
        calls=[verify_call(p.parent,cs,ms,config['broker_source_sha256'],jobs) for p in starts]
        usage=parsed['usage'];observed=usage.get('calls',[])
        require(type(usage.get('unique_observed_cli_invocations')) is int and usage['unique_observed_cli_invocations']==len(calls)
                and len(observed)==len(calls),'Published aggregate/durable call count disagreement')
        require(usage.get('broker_inventory_provided') is True and usage.get('invalid_receipts')==[]
                and usage.get('interrupted_cli_starts_without_receipt')==[],'Unknown or invalid prior CLI inventory')
        require(all(usage.get('observed_calls_missing_counters',{}).get(k)==0 for k in ('input_tokens','output_tokens')),
                'Published inventory has unknown input/output counters')
        if name==FAILED:
            zero=parsed['failure_terminal']
            require(not calls and observed==[] and status['cli_calls']==0 and parsed['completion'].get('all_children_drained') is False
                    and parsed['campaign_finished']['failed'] is True,'Failed GR v1 requires explicit zero-call special case')
            require(zero.get('schema')=='originx_gr00t_failed_development_terminal_audit_v1'
                    and all(zero.get(k) is True for k in ('campaign_failed','all_recorded_worker_and_model_identities_inactive','original_completion_preserved'))
                    and zero.get('request_count')==0 and zero.get('checks'),'Failed GR v1 zero-call owner proof incomplete')
            require(all(x.get('inactive') is True and x.get('recorded_owner') for x in zero['checks']),
                    'Failed development terminal proof has nonterminal owners')
        else:require(usage.get('inventory_complete') is True and usage.get('complete_input_output_token_accounting') is True,'Prior study has unknown CLI accounting')
        usage_by_key={r['invocation_key']:r for r in observed};require(len(usage_by_key)==len(calls),'Duplicate aggregate invocations')
        sums=dict(cli_calls=len(calls),input_tokens=0,output_tokens=0)
        for call in calls:
            key=call['invocation_key'];require(key not in all_invocations,'Duplicated actual invocation across studies');all_invocations.add(key)
            aggregate_call=usage_by_key.get(key)
            require(aggregate_call and aggregate_call.get('receipt_sha256')==call['receipt_sha256']
                    and aggregate_call.get('usage')==call['usage'],'Aggregate does not match exact durable receipt')
            require(aggregate_call.get('transcript_problems')==[],'Prior CLI transcript has unresolved accounting problems')
            for counter in ('input_tokens','output_tokens'):sums[counter]+=call['usage'][counter]
        if calls:
            for counter in ('input_tokens','output_tokens'):
                require(usage['known_usage_totals'].get(counter)==sums[counter] and status['usage_totals'].get(counter)==sums[counter], 'Ledger/aggregate token totals disagree')
        inv=inventory(state);local_hashes.update({str(state/rel):digest for rel,digest in inv.items()})
        verified.append(dict(study=name,state=state,source=source,files=files,parsed=parsed,identity=identity,
            owner=owner,calls=calls,inventory=inv,usage=sums,zero_invocation_derived=name==FAILED))
    require(not all_starts or not process_scan(all_starts),'An owned CLI process remains active')
    remote=remote_check(dict(root=ROOT,studies=list(STUDIES),failed=FAILED,sha256=remote_hashes))
    require(remote.get('passed') is True and remote.get('all_prior_remote_owners_inactive') is True,'Remote terminal audit failed')
    for row in verified:require(not owner_active(row['source'],row['owner']),'Broker revived during debit audit')
    for path,digest in local_hashes.items():require(sha(path)==digest,'Local evidence changed during debit audit')
    totals={k:sum(r['usage'][k] for r in verified) for k in CAPS}
    return verified,totals,remote,local_hashes,remote_hashes


def build(plan,output,remote_directory,**kwargs):
    output=Path(output).resolve();remote_directory=scoped_remote(remote_directory)
    require(not output.exists(),'Output bundle must be new; do not overwrite a sealed debit')
    require(PurePosixPath(remote_directory).parent==PurePosixPath(ROOT)/'results','Remote budget bundle must be a direct results child')
    rows,totals,remote,local_hashes,remote_hashes=validate(plan,**kwargs)
    output.mkdir(parents=True,exist_ok=False);studies=[]
    for row in rows:
        folder=output/row['study'];folder.mkdir();broker=folder/'broker_inventory';broker.mkdir()
        for name,digest in row['inventory'].items():
            destination=broker/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['state']/name,destination)
            require(sha(destination)==digest,'Inventory changed while copying')
        remote_base=remote_directory+'/'+row['study']
        copied={}
        for key,ref in row['files'].items():
            destination=folder/(key+'.json');shutil.copyfile(ref['local'],destination)
            require(sha(destination)==local_hashes[str(Path(ref['local']).resolve())],'Evidence changed while copying')
            copied[key]=dict(path=remote_base+'/'+destination.name,sha256=sha(destination))
        terminal=dict(schema='originx_confirmatory_broker_terminal_audit_v1',checked_at=datetime.now(timezone.utc).isoformat(),
            broker_state_original_path=str(row['state']),broker_identity=row['identity'],broker_source_sha256=sha(row['source']),
            broker_owner=row['owner'],broker_owner_inactive=True,broker_owner_sha256=sha(row['state']/'owner.json'),
            cli_starts=len(row['calls']),verified_cli_wait_returns=[dict(batch_id=c['batch_id'],receipt_sha256=c['receipt_sha256'],
                returncode=c['receipt']['returncode'],finished_at=c['receipt']['finished_at'],evidence='Complete original durable wait receipt, source/batch/stdout and raw usage verified') for c in row['calls']],
            scoped_cli_process_matches=[],all_cli_wait_returns_terminal=True,all_owned_cli_processes_absent=True,
            process_check='Fresh Windows owner identity and exact nonempty batch-path scan; an explicitly proven empty scope has no matching CLI',
            per_cli_pid_start_cwd_available=False,files_sha256=row['inventory'],excluded_non_evidence_files=['broker.lock'],
            audit_tool_sha256=sha(__file__),no_cli_invoked_by_audit=True)
        terminal_path=broker/'shared-budget-terminal-audit.json';write_new(terminal_path,terminal)
        study=dict(study=row['study'],usage=copied['usage'],terminal=dict(path=remote_base+'/broker_inventory/'+terminal_path.name,sha256=sha(terminal_path)),
            raw_receipts={c['receipt_sha256']:dict(path=remote_base+'/broker_inventory/batches/'+c['batch_id']+'/receipt.json',sha256=c['receipt_sha256']) for c in row['calls']},
            verified_usage=row['usage'],source_authority=copied['authority'])
        if row['zero_invocation_derived']:study.update(zero_invocation_derived=True,zero_invocation_evidence=copied['failure_terminal'])
        studies.append(study)
    debit=dict(schema='originx_shared_budget_debit_v1',prior_usage=totals,all_prior_brokers_inactive=True,
        complete_input_output_token_accounting=True,studies=studies,root=ROOT,created_at=datetime.now(timezone.utc).isoformat(),
        budget_caps_unchanged=CAPS,remaining_budget={k:CAPS[k]-totals[k] for k in CAPS},
        all_remaining_strictly_positive=all(totals[k]<CAPS[k] for k in CAPS),
        reasoning_tokens_included_in_output_not_added_again=True,remote_terminal_source_audit=remote,
        budget_interpretation='The main frozen broker cap excludes development overhead. This is an actual combined debit, not a guarantee of a historical shared cap; in-flight call costs become known at terminal accounting. Nonpositive remaining allowance blocks new GR00T formal calls. No second allowance, reset or paid top-up is created.',
        source_evidence_sha256=remote_hashes,builder_sha256=sha(__file__),frozen_audit_helper_sha256=HELPER_SHA,
        no_new_calls_models_or_training=True,copy_deduplication='Only four original durable state directories; each batch start has one exact receipt and invocation key; published copies never add calls.')
    write_new(output/'debit.json',debit);write_new(output/'local-build-receipt.json',dict(local_source_sha256=local_hashes,
        debit_sha256=sha(output/'debit.json'),remote_debit_path=remote_directory+'/debit.json',all_copies_hash_verified=True))
    return debit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--remote-directory',required=True)
    args=p.parse_args();result=build(read(args.plan),args.output_directory,args.remote_directory)
    print(json.dumps(dict(debit=str(args.output_directory/'debit.json'),prior_usage=result['prior_usage'],
        all_remaining_strictly_positive=result['all_remaining_strictly_positive'])))


if __name__=='__main__':main()
