"""Isolated fake ledgers only: never SSH, run a broker, invoke CLI, or inspect live results."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('shared_debit_under_test',DIRECTORY/'build_gr00t_shared_debit.py')
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')


class SharedDebitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.temp.name).resolve();self.work=self.root/'work'
        helper=self.work/'confirmatory_20261009/admit_full.py';helper.parent.mkdir(parents=True)
        shutil.copyfile(DIRECTORY/'admit_full.py',helper)
        self.plan=dict(schema='originx_shared_debit_build_plan_v1',studies=[])
        self.states={};self.files={};self.callfolders={}
        for i,study in enumerate(d.STUDIES):self.fixture(i,study)
        self.scan=[];self.remote=[]
        self.kw=dict(work=self.work,owner_active=lambda source,owner:False,
            process_scan=lambda paths:self.scan,remote_check=self.remote_check)

    def tearDown(self):self.temp.cleanup()

    def remote_check(self,payload):
        self.remote.append(payload)
        return dict(passed=True,all_prior_remote_owners_inactive=True,remote_files_verified=len(payload['sha256']))

    def fixture(self,i,study):
        package,state_name=d.LOCATIONS[study];state=self.work/package/state_name;state.mkdir(parents=True)
        source=self.work/package/'broker.py';source.write_text('# fixed fixture broker: never imported\n')
        self.states[study]=state
        output=d.ROOT+'/results/'+study;local=self.root/'evidence'/study
        sourcepath=d.ROOT+'/'+package+'/runner.py';digest='a'*64
        freeze=dict(source_sha256={'runner.py':digest},broker_source_sha256=d.sha(source))
        write(local/'source_freeze.json',freeze)
        request_id=f'{i+1:032x}';manifest=dict(jobs=[dict(request_id=request_id)])
        write(local/'manifest.json',manifest)
        config=dict(schema='originx_gr00t_confirmatory_config_v1' if i>=2 else 'originx_confirmatory_config_v1',
            development=i!=1,output=output,root=d.ROOT,manifest=output+'/manifest.json',
            manifest_sha256=d.sha(local/'manifest.json'),source_freeze=d.ROOT+'/'+package+'/source-freeze.ready.json',
            source_freeze_sha256=d.sha(local/'source_freeze.json'),source_sha256={sourcepath:digest},
            broker_source_sha256=d.sha(source),models=[dict(owner=dict(pid=i+100,process_start_ticks=1))])
        write(local/'config.json',config);cs=d.sha(local/'config.json');ms=config['manifest_sha256']
        write(local/'authority.json',dict(config_sha256=cs,manifest_sha256=ms,
            source_checks=[dict(path=sourcepath,expected_sha256=digest,actual_sha256=digest,valid=True)]))
        write(local/'campaign_finished.json',dict(failed=i==2))
        write(local/'completion.json',dict(all_children_drained=i!=2))
        write(state/'identity.json',dict(remote_output=output,config_sha256=cs,manifest_sha256=ms,model='gpt-6-astra',reasoning_effort='high'))
        write(state/'owner.json',dict(pid=i+10,creation_filetime=10000+i,command=[str(source),'--local-output',str(state)]))
        calls=[];totals=dict(input_tokens=0,cached_input_tokens=0,output_tokens=0,reasoning_tokens=0)
        if i!=2:
            folder=state/'batches'/('fixture-batch-'+str(i));folder.mkdir(parents=True);self.callfolders[study]=folder
            request=folder/'requests'/request_id/'request.json';write(request,dict(fixture=True,request_id=request_id))
            binding=dict(request_id=request_id,request_token='t'+str(i),request_sha256=d.sha(request))
            batch=dict(batch_id=folder.name,broker_source_sha256=d.sha(source),config_sha256=cs,manifest_sha256=ms,requests=[binding])
            write(folder/'batch.json',batch)
            started=dict(started_at=f'fixture-{i}',request_ids=[request_id]);write(folder/'cli_started.json',started)
            raw=dict(input_tokens=100+i,cached_input_tokens=20,output_tokens=30,reasoning_output_tokens=7)
            (folder/'stdout.jsonl').write_text(json.dumps(dict(type='turn.completed',usage=raw))+'\n',encoding='utf8')
            for filename in ('prompt.txt','schema.json','stderr.txt'):(folder/filename).write_text('fixture '+filename,encoding='utf8')
            totals=dict(input_tokens=100+i,cached_input_tokens=20,output_tokens=30,reasoning_tokens=7)
            receipt=dict(schema='astra_cli_receipt_v1',cli_executed=True,returncode=0,finished_at=f'finished-{i}',
                process_exception=None,**started,model='gpt-6-astra',model_requested='gpt-6-astra',reasoning_effort='high',
                command=['fixture-codex','exec','--model','gpt-6-astra','-c','model_reasoning_effort=high'],requests=[binding],
                batch_manifest_sha256=d.sha(folder/'batch.json'),stdout_sha256=d.sha(folder/'stdout.jsonl'),
                prompt_sha256=d.sha(folder/'prompt.txt'),output_schema_sha256=d.sha(folder/'schema.json'),stderr_sha256=d.sha(folder/'stderr.txt'),
                usage_raw=[raw],usage_totals=totals)
            write(folder/'receipt.json',receipt)
            key=d.json_sha({k:receipt.get(k) for k in ('started_at','command','requests','stdout_sha256','batch_manifest_sha256')})
            calls=[dict(invocation_key=key,receipt_sha256=d.sha(folder/'receipt.json'),usage=totals,source_paths=['original immutable fixture'],transcript_problems=[])]
        write(state/'broker_status.json',dict(state='finished',cli_calls=len(calls),usage_totals=totals))
        write(local/'usage.json',dict(unique_observed_cli_invocations=len(calls),calls=calls,
            invalid_receipts=[],interrupted_cli_starts_without_receipt=[],inventory_complete=i!=2,
            broker_inventory_provided=True,observed_calls_missing_counters={k:0 for k in totals},
            complete_input_output_token_accounting=i!=2,known_usage_totals=totals if i!=2 else {k:None for k in totals}))
        files={}
        for key in ('config','manifest','source_freeze','campaign_finished','completion','usage','authority'):
            remote=output+'/'+{'campaign_finished':'campaign-finished.json','usage':'analysis-final/cli_usage.json','authority':'analysis-final/authority.json'}.get(key,key+'.json')
            if key=='source_freeze':remote=config['source_freeze']
            files[key]=dict(local=str(local/(key+'.json')),remote=remote)
        if i==2:
            write(local/'failure_terminal.json',dict(schema='originx_gr00t_failed_development_terminal_audit_v1',campaign_failed=True,
                all_recorded_worker_and_model_identities_inactive=True,original_completion_preserved=True,request_count=0,
                checks=[dict(inactive=True,recorded_owner=dict(pid=102,process_start_ticks=1,command=['fixture'],cwd='/fixture'))]))
            files['failure_terminal']=dict(local=str(local/'failure_terminal.json'),remote=output+'/failure-terminal-audit.json')
        self.files[study]=files;self.plan['studies'].append(dict(study=study,files=files))

    def mutate(self,path,fn):
        obj=d.read(path);fn(obj);write(path,obj)

    def test_complete_four_study_debit_and_reasoning_subset(self):
        rows,totals,*_=d.validate(self.plan,**self.kw)
        self.assertEqual(totals,dict(cli_calls=3,input_tokens=304,output_tokens=90))
        self.assertEqual(len(rows),4);self.assertEqual(len(self.remote),1)
        self.assertTrue(rows[2]['zero_invocation_derived'])

    def test_active_main_broker_rejected_before_network(self):
        kw=dict(self.kw,owner_active=lambda source,owner:owner['pid']==11)
        with self.assertRaisesRegex(RuntimeError,'still alive'):d.validate(self.plan,**kw)
        self.assertEqual(self.remote,[])

    def test_remote_owner_active_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Remote terminal'):
            d.validate(self.plan,**dict(self.kw,remote_check=lambda payload:dict(passed=False)))

    def test_nonterminal_main_rejected(self):
        self.mutate(self.states[d.STUDIES[1]]/'broker_status.json',lambda x:x.update(state='running'))
        with self.assertRaisesRegex(RuntimeError,'not terminal'):d.validate(self.plan,**self.kw)

    def test_missing_terminal_receipt_rejected(self):
        (self.callfolders[d.STUDIES[1]]/'receipt.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'one-to-one'):d.validate(self.plan,**self.kw)

    def test_unknown_raw_input_rejected(self):
        folder=self.callfolders[d.STUDIES[1]];r=d.read(folder/'receipt.json')
        del r['usage_raw'][0]['input_tokens'];(folder/'stdout.jsonl').write_text(json.dumps(dict(type='turn.completed',usage=r['usage_raw'][0])))
        r['stdout_sha256']=d.sha(folder/'stdout.jsonl');write(folder/'receipt.json',r)
        with self.assertRaisesRegex(RuntimeError,'Unknown CLI'):d.validate(self.plan,**self.kw)

    def test_changed_stdout_and_source_rejected(self):
        folder=self.callfolders[d.STUDIES[1]];(folder/'stdout.jsonl').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'stdout hash'):d.validate(self.plan,**self.kw)
        (self.work/d.LOCATIONS[d.STUDIES[0]][0]/'broker.py').write_text('# different source')
        with self.assertRaisesRegex(RuntimeError,'broker source differs'):d.validate(self.plan,**self.kw)

    def test_foreign_or_copied_state_cannot_count(self):
        self.mutate(self.states[d.STUDIES[0]]/'owner.json',lambda x:x.update(command=[x['command'][0],'--local-output',str(self.root/'archive')]))
        with self.assertRaisesRegex(RuntimeError,'Archived broker copy'):d.validate(self.plan,**self.kw)

    def test_duplicate_aggregate_call_rejected(self):
        self.mutate(self.files[d.STUDIES[1]]['usage']['local'],lambda x:x['calls'].append(x['calls'][0]))
        with self.assertRaisesRegex(RuntimeError,'count disagreement'):d.validate(self.plan,**self.kw)

    def test_duplicate_actual_invocation_across_studies_rejected(self):
        original=d.verify_call
        def samekey(*args,**kwargs):
            value=original(*args,**kwargs);value['invocation_key']='f'*64;return value
        for study in (d.STUDIES[0],d.STUDIES[1],d.STUDIES[3]):
            self.mutate(self.files[study]['usage']['local'],lambda x:x['calls'][0].update(invocation_key='f'*64))
        with patch.object(d,'verify_call',samekey):
            with self.assertRaisesRegex(RuntimeError,'Duplicated actual invocation'):d.validate(self.plan,**self.kw)

    def test_boolean_zero_not_explicit_accounting(self):
        self.mutate(self.states[d.FAILED]/'broker_status.json',lambda x:x.update(cli_calls=False))
        with self.assertRaisesRegex(RuntimeError,'explicit count'):d.validate(self.plan,**self.kw)

    def test_missing_failed_zero_owner_proof_rejected(self):
        self.mutate(self.files[d.FAILED]['failure_terminal']['local'],lambda x:x.update(checks=[]))
        with self.assertRaisesRegex(RuntimeError,'zero-call owner proof'):d.validate(self.plan,**self.kw)

    def test_unknown_nonfailed_accounting_rejected(self):
        self.mutate(self.files[d.STUDIES[1]]['usage']['local'],lambda x:x.update(inventory_complete=False))
        with self.assertRaisesRegex(RuntimeError,'unknown CLI accounting'):d.validate(self.plan,**self.kw)

    def test_all_four_studies_required(self):
        self.plan['studies'].pop()
        with self.assertRaisesRegex(RuntimeError,'Exactly four'):d.validate(self.plan,**self.kw)

    def test_invalid_source_authority_rejected(self):
        self.mutate(self.files[d.STUDIES[0]]['authority']['local'],lambda x:x['source_checks'][0].update(valid=False))
        with self.assertRaisesRegex(RuntimeError,'Invalid prior source'):d.validate(self.plan,**self.kw)

    def test_owned_cli_active_rejected(self):
        self.scan.append(dict(pid=999))
        with self.assertRaisesRegex(RuntimeError,'CLI process remains'):d.validate(self.plan,**self.kw)

    def test_exclusive_bundle_preserves_usage_and_relocates_receipts(self):
        output=self.root/'sealed';remote=d.ROOT+'/results/fixture-budget'
        before=Path(self.files[d.FAILED]['usage']['local']).read_bytes()
        result=d.build(self.plan,output,remote,**self.kw)
        self.assertEqual(result['prior_usage']['output_tokens'],90)
        for row in result['studies']:
            terminal=d.read(output/row['study']/'broker_inventory/shared-budget-terminal-audit.json')
            self.assertEqual(terminal['cli_starts'],row['verified_usage']['cli_calls'])
            for digest,ref in row['raw_receipts'].items():
                local=output/PureRel(ref['path'],remote);self.assertEqual(d.sha(local),digest)
        self.assertEqual(before,(output/d.FAILED/'usage.json').read_bytes())
        with self.assertRaisesRegex(RuntimeError,'must be new'):d.build(self.plan,output,remote,**self.kw)

    def test_exhausted_shared_allowance_never_becomes_second_allowance(self):
        with patch.object(d,'CAPS',dict(cli_calls=3,input_tokens=303,output_tokens=90)):
            result=d.build(self.plan,self.root/'exhausted',d.ROOT+'/results/fixture-exhausted',**self.kw)
        self.assertEqual(result['remaining_budget'],dict(cli_calls=0,input_tokens=-1,output_tokens=0))
        self.assertFalse(result['all_remaining_strictly_positive'])
        self.assertIn('not a guarantee',result['budget_interpretation'])

    def test_bundle_satisfies_actual_formal_service_guard(self):
        guard=d.module(DIRECTORY/'gr00t_formal_service_guard.py','shared_debit_guard_fixture')
        output=self.root/'guard-compatible';remote=d.ROOT+'/results/fixture-budget'
        result=d.build(self.plan,output,remote,**self.kw)
        def translate(value):return output/PureRel(str(value).replace('\\','/'),remote)
        def load(ref):
            path=translate(ref['path']);self.assertEqual(d.sha(path),ref['sha256']);return d.read(path)
        def inv(ref,terminal):
            path=translate(ref['path']);actual=d.inventory(path.parent);actual.pop(path.name)
            self.assertEqual(actual,terminal['files_sha256']);return actual
        with patch.object(guard,'ROOT',d.PurePosixPath(d.ROOT)),patch.object(guard,'owner_inactive',lambda owner:True):
            verified=guard.validate_budget(result,load=load,inventory=inv)
        self.assertEqual(verified['prior_usage'],result['prior_usage'])
        self.assertEqual(verified['zero_calls_proven_by_sealed_inventory'],[d.FAILED])


def PureRel(path,root):return Path(*d.PurePosixPath(path).relative_to(d.PurePosixPath(root)).parts)


if __name__=='__main__':unittest.main()
