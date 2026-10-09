"""CPU-only guard checks. Linux cases spawn only disposable sleeping Python children."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'gr00t_formal_service_guard.py'
spec=importlib.util.spec_from_file_location('gr00t_guard_tests',FILE)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


class FakeChild:
    def __init__(self,code=None):self.pid=101;self.returncode=code;self.waited=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        self.waited+=1
        if self.returncode is None:self.returncode=-15
        return self.returncode


class BudgetFixture:
    def __init__(self):
        self.docs={};self.studies=[];total={k:0 for k in g.CAPS}
        for i,name in enumerate(g.STUDIES):
            root=g.ROOT/'results'/name;tr=root/'broker_inventory';n=0 if i==2 else 1
            receipt_digest=hashlib.sha256(name.encode()).hexdigest()
            rawpath=tr/'batches'/'b0'/'receipt.json';usagepath=root/'analysis'/'cli_usage.json';terminalpath=tr/'terminal-audit.json'
            usage=dict(input_tokens=100+i,output_tokens=20+i,reasoning_tokens=10)
            calls=[] if n==0 else [dict(invocation_key=receipt_digest,receipt_sha256=receipt_digest,
                source_paths=[str(rawpath)],usage=usage,transcript_problems=[])]
            raw=dict(schema='astra_cli_receipt_v1',cli_executed=True,model='gpt-6-astra',reasoning_effort='high',usage_totals=usage)
            files={'broker_status.json':'b'*64}
            if n:files['batches/b0/receipt.json']=receipt_digest
            terminal=dict(schema='originx_confirmatory_broker_terminal_audit_v1',broker_owner_inactive=True,
                all_cli_wait_returns_terminal=True,all_owned_cli_processes_absent=True,no_cli_invoked_by_audit=True,
                scoped_cli_process_matches=[],broker_identity=dict(remote_output=str(root)),cli_starts=n,
                verified_cli_wait_returns=[] if not n else [dict(receipt_sha256=receipt_digest)],files_sha256=files)
            observed=dict(unique_observed_cli_invocations=n,calls=calls,broker_inventory_provided=True,
                inventory_complete=bool(n),complete_input_output_token_accounting=bool(n),invalid_receipts=[],
                interrupted_cli_starts_without_receipt=[],known_usage_totals={k:(usage[k] if n else None) for k in ('input_tokens','output_tokens')},
                observed_calls_missing_counters=dict(input_tokens=0,output_tokens=0))
            self.docs[str(rawpath)]=raw;self.docs[str(usagepath)]=observed;self.docs[str(terminalpath)]=terminal
            self.docs[str(tr/'broker_status.json')]=dict(cli_calls=n)
            row=dict(study=name,usage=dict(path=str(usagepath),sha256='a'*64),terminal=dict(path=str(terminalpath),sha256='c'*64))
            if not n:
                proofpath=root/'failure-terminal-audit.json';row['zero_invocation_evidence']=dict(path=str(proofpath),sha256='d'*64)
                self.docs[str(proofpath)]=dict(schema='originx_gr00t_failed_development_terminal_audit_v1',campaign_failed=True,
                    original_completion_preserved=True,all_recorded_worker_and_model_identities_inactive=True,
                    checks=[dict(inactive=True,recorded_owner=dict(pid=99112233,process_start_ticks=1,command=['owned'],cwd='owned'))])
            self.studies.append(row);total['cli_calls']+=n
            for key in ('input_tokens','output_tokens'):total[key]+=usage[key] if n else 0
        self.debit=dict(schema='originx_shared_budget_debit_v1',all_prior_brokers_inactive=True,
            complete_input_output_token_accounting=True,prior_usage=total,studies=self.studies)
    def load(self,ref):return copy.deepcopy(self.docs[ref['path']])
    def inventory(self,ref,terminal):return terminal['files_sha256']
    def check(self):
        with patch.object(g,'owner_inactive',return_value=True):return g.validate_budget(self.debit,self.load,self.inventory)


class BudgetTests(unittest.TestCase):
    def test_complete_four_studies_and_strict_zero_case_use_actual_counters(self):
        f=BudgetFixture();answer=f.check();self.assertEqual(answer['prior_usage'],f.debit['prior_usage'])
        self.assertEqual(answer['prior_usage']['output_tokens'],64) # reasoning is not added
        self.assertEqual(answer['zero_calls_proven_by_sealed_inventory'],[g.STUDIES[2]])
    def test_missing_study_or_incomplete_usage_blocks(self):
        f=BudgetFixture();f.debit['studies'].pop()
        with self.assertRaisesRegex(RuntimeError,'four prior'):f.check()
        f=BudgetFixture();f.docs[f.studies[0]['usage']['path']]['inventory_complete']=False
        with self.assertRaisesRegex(RuntimeError,'Incomplete actual'):f.check()
    def test_unknown_counter_or_receipt_disagreement_blocks(self):
        f=BudgetFixture();path=f.docs[f.studies[0]['usage']['path']]['calls'][0]['source_paths'][0]
        f.docs[path]['usage_totals']['output_tokens']=None
        with self.assertRaises(RuntimeError):f.check()
    def test_duplicate_invocation_across_studies_blocks(self):
        f=BudgetFixture();a=f.docs[f.studies[0]['usage']['path']]['calls'][0]
        f.docs[f.studies[1]['usage']['path']]['calls'][0]['invocation_key']=a['invocation_key']
        with self.assertRaisesRegex(RuntimeError,'Duplicate/invalid'):f.check()
    def test_any_exhausted_dimension_blocks(self):
        f=BudgetFixture()
        with patch.dict(g.CAPS,dict(cli_calls=3,input_tokens=25000000,output_tokens=2000000)):
            with self.assertRaisesRegex(RuntimeError,'exhausted'):f.check()
    def test_zero_claim_with_a_hidden_start_or_absent_status_blocks(self):
        f=BudgetFixture();t=f.docs[f.studies[2]['terminal']['path']];t['files_sha256']['batches/b/cli_started.json']='x'
        with self.assertRaises(RuntimeError):f.check()
        f=BudgetFixture();root=Path(f.studies[2]['terminal']['path']).parent;f.docs[str(root/'broker_status.json')]['cli_calls']=None
        with self.assertRaisesRegex(RuntimeError,'not explicit'):f.check()
    def test_relocated_raw_receipt_preserves_original_usage(self):
        f=BudgetFixture();row=f.studies[0];call=f.docs[row['usage']['path']]['calls'][0];old=call['source_paths'][0]
        new=Path(row['terminal']['path']).parent/'batches'/'copied'/'receipt.json'
        f.docs[str(new)]=copy.deepcopy(f.docs[old]);row['raw_receipts']={call['receipt_sha256']:dict(path=str(new),sha256=call['receipt_sha256'])}
        f.docs[row['terminal']['path']]['files_sha256'][new.relative_to(new.parents[2]).as_posix()]=call['receipt_sha256']
        self.assertEqual(f.check()['prior_usage'],f.debit['prior_usage'])
    def test_sealed_inventory_detects_added_files_and_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'ROOT',Path(tmp).resolve()):
            root=Path(tmp).resolve();audit=root/'audit.json';audit.write_text('{}');p=root/'broker_status.json';p.write_text('{"cli_calls":0}')
            expected={'broker_status.json':g.sha(p)};ref=dict(path=str(audit),sha256=g.sha(audit))
            terminal=dict(files_sha256=expected,excluded_non_evidence_files=['broker.lock'])
            self.assertEqual(g.terminal_inventory(ref,terminal),expected)
            (root/'cli_started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'inventory incomplete'):g.terminal_inventory(ref,terminal)


class OwnershipTests(unittest.TestCase):
    def birth(self,pid):return dict(pid=pid,parent_pid=os.getpid(),process_start_ticks=99,state='R')
    def identity(self,command):return dict(pid=101,process_start_ticks=99,command=command,cwd=str(g.BASE))
    def test_empty_argv_race_needs_two_matching_reads_and_one_handle(self):
        child=FakeChild();command=['expected'];good=self.identity(command)
        with patch.object(g,'birth',side_effect=self.birth),patch.object(g,'identity',side_effect=[self.identity(['']),good,good]), \
             patch.object(g,'open_pidfd',return_value=44) as opened,patch.object(g.time,'sleep'):
            tracked=g.TrackedChild(child,command,g.BASE);self.assertEqual(tracked.await_identity(),good)
            self.assertEqual(len(tracked.samples),3);opened.assert_called_once_with(101)
    def test_early_exit_reaped_without_signal(self):
        child=FakeChild(3);tracked=g.TrackedChild(child,['expected'],g.BASE)
        with patch.object(g,'open_pidfd') as opened,patch.object(g,'send_pidfd') as sent:
            with self.assertRaises(RuntimeError):tracked.await_identity()
            self.assertEqual(tracked.drain(),3)
        self.assertTrue(tracked.reaped);opened.assert_not_called();sent.assert_not_called()
    def test_foreign_parent_never_signalled(self):
        child=FakeChild();tracked=g.TrackedChild(child,['expected'],g.BASE)
        foreign=dict(self.birth(101),parent_pid=os.getpid()+1)
        with patch.object(g,'birth',return_value=foreign),patch.object(g,'open_pidfd',return_value=44),patch.object(g,'send_pidfd') as sent:
            with self.assertRaisesRegex(RuntimeError,'direct unreaped'):tracked.send(signal.SIGTERM)
        sent.assert_not_called()
    def test_same_birth_owner_drift_is_not_termination(self):
        owner=self.identity(['expected'])
        with patch.object(g,'birth',side_effect=self.birth):
            for actual in (dict(owner,command=['different']),dict(owner,cwd='different')):
                with patch.object(g,'identity',return_value=actual):
                    with self.assertRaisesRegex(RuntimeError,'same birth identity'):g.owner_inactive(owner)
            with patch.object(g,'identity',return_value=owner):self.assertFalse(g.owner_inactive(owner))
            with patch.object(g,'identity',return_value=dict(owner,process_start_ticks=100)):
                self.assertTrue(g.owner_inactive(owner))
    def test_wrong_gpu_foreign_pid_or_low_memory_blocks(self):
        sample=dict(gpu_uuid=g.GPU_UUID,ecc=0,free_mib=20000,occupants=[])
        g.validate_gpu(sample)
        for bad in (dict(sample,occupants=[123]),dict(sample,gpu_uuid='other'),dict(sample,free_mib=100)):
            with self.assertRaises(RuntimeError):g.validate_gpu(bad)
    def test_spawn_registered_before_identity_failure_and_then_reaped(self):
        child=FakeChild(2)
        with tempfile.TemporaryDirectory() as tmp,patch.object(g.subprocess,'Popen',return_value=child), \
             patch.object(g.TrackedChild,'await_identity',side_effect=TimeoutError('handshake')):
            supervisor=g.Supervisor(tmp,600)
            with self.assertRaises(TimeoutError):supervisor.spawn('service',g.commands()[0],True)
            self.assertEqual(len(supervisor.children),1)
            self.assertTrue((Path(tmp)/'service/handshake.json').exists())
            receipt=supervisor.drain()
            self.assertTrue(receipt['all_children_reaped']);self.assertGreaterEqual(child.waited,1)
    def test_commands_are_exact_and_other_spawn_refused(self):
        service,parity=g.commands('a'*64)
        self.assertEqual(service[-4:],['--port','27903','--max-clients','6'])
        self.assertNotIn('--coexist-services',service)
        self.assertEqual(parity[-4:],['--hold-seconds','360','--interval-seconds','120'])
        with tempfile.TemporaryDirectory() as tmp,patch.object(g.subprocess,'Popen') as spawn:
            s=g.Supervisor(tmp,600)
            with self.assertRaises(RuntimeError):s.spawn('service',['python','other.py'],True)
            with self.assertRaises(RuntimeError):s.spawn('service',service,False)
            spawn.assert_not_called()


@unittest.skipUnless(os.name=='posix' and hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux CPU-only pidfd lifecycle')
class RealLinuxTests(unittest.TestCase):
    def child(self,cwd):
        command=[sys.executable,'-c','import time; time.sleep(60)']
        process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return g.TrackedChild(process,command,cwd)
    def test_actual_child_transient_identity_then_pidfd_drain(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd=Path(tmp).resolve();tracked=self.child(cwd);original=g.identity;seen=[0]
            def delayed(pid):
                row=original(pid);seen[0]+=1
                if seen[0]<3:row['command']=['']
                return row
            try:
                with patch.object(g,'identity',side_effect=delayed):tracked.await_identity()
                self.assertGreaterEqual(seen[0],4);self.assertIsNotNone(tracked.pidfd)
            finally:tracked.drain()
            self.assertTrue(tracked.reaped);self.assertIsNotNone(tracked.child.returncode)
    def test_actual_identity_timeout_unbound_child_is_drained(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracked=self.child(Path(tmp).resolve());original=g.identity
            def unready(pid):
                row=original(pid);row['command']=[''];return row
            try:
                with patch.object(g,'identity',side_effect=unready):
                    with self.assertRaises(TimeoutError):tracked.await_identity(timeout=.06,interval=.01)
                self.assertIsNone(tracked.owner)
            finally:tracked.drain()
            self.assertTrue(tracked.reaped)
    def test_actual_phase_timeout_reaps_registered_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd=Path(tmp).resolve();tracked=self.child(cwd);supervisor=g.Supervisor(cwd,600);supervisor.children.append(('parity',tracked))
            try:
                tracked.await_identity()
                with self.assertRaises(TimeoutError):supervisor.wait_phase(tracked,.03)
            finally:receipt=supervisor.drain()
            self.assertTrue(receipt['all_children_reaped']);self.assertIsNotNone(tracked.child.returncode)


if __name__=='__main__':unittest.main()
