"""Offline launcher gates only; never SSH, load a model, or start a campaign."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

PATH = Path(__file__).parents[1] / 'start_full_admitted_v2.py'
spec = importlib.util.spec_from_file_location('admitted_start_v2_tests', PATH)
a = importlib.util.module_from_spec(spec); sys.modules[spec.name] = a; spec.loader.exec_module(a)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def model(policy, index):
    return dict(service_id=f'{policy}-gpu6-r{index}', policy_id=policy, gpu_index=6,
                gpu_uuid=a.GPU_UUID, namespace=a.NAMESPACE, slots=6,
                port=27800+index if policy=='B' else 27805-index,
                owner={'pid':100+index+(10 if policy=='base' else 0)})


def ready(models):
    return dict(schema='originx_confirmatory_services_v1', namespace=a.NAMESPACE,
                ready=True, gpu_index=6, gpu_uuid=a.GPU_UUID, model_count=len(models),
                clients=6*len(models), models=models, profiles=copy.deepcopy(models))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.temp.name); self.out=self.root/'results'/a.DEV
        self.services=self.root/'results'/a.SERVICES
        self.old=ready([model('B',0),model('base',0)])
        self.new=ready([model('B',i) for i in range(4)]+[model('base',i) for i in range(2)])
        (self.root/'admit_full.py').write_text('fixture')
        self.old_admit_sha=a.ADMIT_SHA; a.ADMIT_SHA=a.sha(self.root/'admit_full.py')
        self.source=self.root/'frozen.py';self.source.write_text('frozen')
        self.config=dict(models=self.old['models'],manifest_sha256='m'*64,source_freeze_sha256='s'*64,broker_source_sha256=a.BROKER_SHA)
        write(self.out/'config.json',self.config)
        self.approval=dict(schema='originx_confirmatory_admission_review_v1',passed=True,no_confirmatory_outcomes_seen=True,
                           raw_verified_outcomes=14,complete_cli_usage=True,admission_tool_sha256=a.ADMIT_SHA,
                           broker_source_sha256=a.BROKER_SHA,manifest_sha256='m'*64,source_freeze_sha256='s'*64,
                           config_sha256=a.sha(self.out/'config.json'),owned_development_processes=[],
                           bindings_sha256={str(self.source):a.sha(self.source),str(self.out/'config.json'):a.sha(self.out/'config.json')})
        write(self.out/'admission-review.json',self.approval)
        write(self.services/'services-ready.json',self.old);write(self.services/'lease.owner.json',{'pid':42})
        self.check=dict(broker_source_sha256=a.BROKER_SHA,launcher_source_sha256='self',performed_on_local_host=True)

    def tearDown(self):
        a.ADMIT_SHA=self.old_admit_sha;self.temp.cleanup()

    def precheck(self,**kwargs):
        return a.precheck(self.root,self.check,'self',owner_active=kwargs.get('active',lambda _:True),owner_stopped=lambda _:True)

    def test_check_read_only_no_model_side_effects(self):
        before={str(p):a.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result=self.precheck();self.assertTrue(result['passed']);self.assertEqual(result['duration_seconds'],432000)
        self.assertEqual(before,{str(p):a.sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse((self.root/'results'/a.FULL).exists())

    def test_exact_gpu6_four_plus_two_extension_passes(self):
        a.validate_extension(self.old,self.new)

    def test_replacement_old_profile_is_rejected(self):
        self.new['models'][0]['owner']['pid']=999
        self.new['profiles']=copy.deepcopy(self.new['models'])
        with self.assertRaisesRegex(RuntimeError,'replaced'):a.validate_extension(self.old,self.new)

    def test_foreign_gpu_is_rejected(self):
        self.new['models'][1]['gpu_index']=3
        self.new['profiles']=copy.deepcopy(self.new['models'])
        with self.assertRaisesRegex(RuntimeError,'GPU6'):a.validate_extension(self.old,self.new)

    def test_changed_immutable_source_is_rejected(self):
        self.source.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'immutable'):self.precheck()

    def test_actual_local_broker_hash_required(self):
        self.check['broker_source_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'Local broker'):self.precheck()

    @unittest.skipUnless(a.os.name=='nt','Windows local launch driver')
    def test_windows_local_driver_serializes_posix_remote_paths(self):
        with patch.object(a.sys,'argv',['start_full_admitted_v2.py','--check']), \
             patch.object(a,'sha',side_effect=lambda p:'self-digest' if Path(p).name=='start_full_admitted_v2.py' else a.BROKER_SHA), \
             patch.object(a.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute:
            self.assertEqual(a.main(),0)
        command=execute.call_args.args[0]
        remote=command[command.index('hkust-cluster')+1:]
        self.assertTrue(remote[0].startswith('/ephemeral/'))
        self.assertTrue(remote[2].startswith('/ephemeral/'))
        self.assertNotIn('\\',remote[0]+remote[2])

    def test_existing_full_manifest_or_owner_is_rejected(self):
        write(self.root/'results'/a.FULL/'config.json',{})
        with self.assertRaisesRegex(RuntimeError,'already contains'):self.precheck()

    def test_prior_external_attempt_is_not_restarted(self):
        (self.services/'full-launch-admission-v1').mkdir()
        with self.assertRaisesRegex(RuntimeError,'Prior external'):self.precheck()

    def test_original_owner_or_lease_must_be_live(self):
        with self.assertRaisesRegex(RuntimeError,'lease'):self.precheck(active=lambda _:False)

    def test_only_ready_binding_can_change_after_exact_extension(self):
        ready_path=self.services/'services-ready.json'
        self.approval['bindings_sha256'][str(ready_path)]=a.sha(ready_path)
        write(ready_path,self.new)
        a.validate_bindings(self.approval,self.root,previous_ready=self.old,current_ready=self.new)
        self.source.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'immutable'):
            a.validate_bindings(self.approval,self.root,previous_ready=self.old,current_ready=self.new)

    def test_failed_admission_cannot_reach_any_spawn(self):
        self.approval['passed']=False;write(self.out/'admission-review.json',self.approval)
        with patch.object(a.subprocess,'Popen',side_effect=AssertionError('No spawn permitted')):
            with self.assertRaisesRegex(RuntimeError,'approval'):self.precheck()


class FakeChild:
    def __init__(self,pid=101,returncode=None):self.pid=pid;self.returncode=returncode;self.waited=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        self.waited+=1
        if self.returncode is None:self.returncode=-15
        return self.returncode


class SpawnLifecycleTests(unittest.TestCase):
    def birth(self,pid):return dict(pid=pid,parent_pid=a.os.getpid(),process_start_ticks=99,state='R')
    def owner(self,command):return dict(pid=101,process_start_ticks=99,command=command,cwd=str(a.ROOT))

    def test_empty_argv_then_two_stable_matches_bind_same_pidfd(self):
        child=FakeChild();command=['expected','model'];good=self.owner(command)
        with patch.object(a,'child_birth',side_effect=self.birth),patch.object(a,'open_pidfd',return_value=10) as opened, \
             patch.object(a,'child_identity',side_effect=[self.owner(['']),good,good]),patch.object(a.time,'sleep'):
            tracked=a.TrackedChild(child,command,a.ROOT);self.assertEqual(tracked.await_identity(),good)
            self.assertEqual(len(tracked.samples),3);opened.assert_called_once_with(101)

    def test_early_exit_is_reaped_without_pid_signal(self):
        child=FakeChild(returncode=3);tracked=a.TrackedChild(child,['expected'],a.ROOT)
        with patch.object(a,'open_pidfd') as opened,patch.object(a,'send_pidfd') as signalled:
            with self.assertRaisesRegex(RuntimeError,'exited'):tracked.await_identity()
            self.assertEqual(tracked.drain(),3)
        self.assertTrue(tracked.reaped);opened.assert_not_called();signalled.assert_not_called()

    def test_permanent_empty_argv_is_drained_through_same_handle(self):
        child=FakeChild();tracked=a.TrackedChild(child,['expected'],a.ROOT)
        with patch.object(a,'child_birth',side_effect=self.birth),patch.object(a,'open_pidfd',return_value=12), \
             patch.object(a,'child_identity',return_value=self.owner([''])),patch.object(a.time,'sleep'), \
             patch.object(a.time,'monotonic',side_effect=[0,1,2,11]),patch.object(a,'send_pidfd') as signal, \
             patch.object(a,'close_pidfd') as closed:
            with self.assertRaises(TimeoutError):tracked.await_identity()
            tracked.drain()
        self.assertIsNone(tracked.owner);self.assertTrue(tracked.reaped)
        signal.assert_called_once_with(12,a.signal.SIGTERM);closed.assert_called_once_with(12)

    def test_foreign_argv_delegated_without_registry_or_shared_module_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            calls=[];original=SimpleNamespace(Popen=lambda cmd,*ar,**kw:calls.append(cmd) or 'unrelated')
            services=SimpleNamespace(subprocess=original,layout=lambda *_:[{'service_id':'B-gpu6-r1'}],
                                     service_command=lambda _:['exact','model'])
            guard=a.OwnedServiceSpawns(services,Path(tmp),[])
            with guard:
                self.assertIsNot(services.subprocess,original)
                self.assertEqual(services.subprocess.Popen(['foreign','argv']),'unrelated')
            self.assertIs(services.subprocess,original);self.assertEqual(guard.children,[])
            self.assertEqual(calls,[['foreign','argv']])

    def test_expansion_error_retains_pending_child_and_restores_original_class(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            child=FakeChild();original=SimpleNamespace(Popen=lambda *args,**kwargs:child)
            services=SimpleNamespace(subprocess=original,layout=lambda *_:[{'service_id':'B-gpu6-r1'}],
                                     service_command=lambda _:['exact','model'])
            guard=a.OwnedServiceSpawns(services,Path(tmp),[])
            with patch.object(a.TrackedChild,'await_identity',side_effect=RuntimeError('early startup error')):
                with self.assertRaisesRegex(RuntimeError,'early startup'):
                    with guard:services.subprocess.Popen(['exact','model'],cwd=a.ROOT)
            self.assertIs(services.subprocess,original);self.assertIs(guard.children[0][1].child,child)
            child.returncode=7;guard.drain()
            self.assertTrue(guard.children[0][1].reaped)
            self.assertTrue((Path(tmp)/'spawns/B-gpu6-r1/handshake.json').exists())

    def test_campaign_owner_partial_json_is_waited_for_not_restarted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'owner.json';path.write_text('{')
            owner=self.owner(['expected']);child=FakeChild()
            with patch.object(a,'read',side_effect=[json.JSONDecodeError('partial','{',1),owner]),patch.object(a.time,'sleep'):
                self.assertEqual(a.await_campaign_owner(child,path),owner)

    def test_campaign_stop_is_observed_before_child_enumeration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            process=FakeChild();states=iter(['R','T','T']);observed=[];sent=[]
            def birth():
                state=next(states);observed.append(state);return {'state':state}
            def enumerate_proc(path):
                self.assertEqual(observed,['R','T','T']);return iter([])
            tracked=SimpleNamespace(child=process,pid=101,_birth=birth,send=lambda sig:sent.append(sig),
                 drain=lambda:setattr(process,'returncode',-15),receipt=lambda:{'fixture':True})
            with patch.object(a.Path,'iterdir',enumerate_proc),patch.object(a.time,'sleep'), \
                 patch.object(a.signal,'SIGSTOP',19,create=True),patch.object(a.signal,'SIGCONT',18,create=True), \
                 patch.object(a,'require_no_live_prebroker_children') as final_scan:
                a.drain_campaign(tracked,Path(tmp))
            self.assertEqual(sent,[19,18]);final_scan.assert_called_once()

    def test_parent_exit_is_not_enough_when_orphan_child_remains(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            process=FakeChild(returncode=1)
            tracked=SimpleNamespace(child=process,drain=lambda:None,receipt=lambda:{'fixture':True})
            with patch.object(a,'require_no_live_prebroker_children',side_effect=RuntimeError('orphan remains')):
                with self.assertRaisesRegex(RuntimeError,'orphan'):a.drain_campaign(tracked,Path(tmp))
            self.assertTrue((Path(tmp)/'campaign-drain-error.json').exists())

    def test_partial_spawn_preserved_without_fake_owner_while_known_models_cleanup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);out=root/'services-output';location=root/'launch';location.mkdir()
            specs=[dict(service_id=n,directory=str(out/'services'/n),port=28000+i)
                   for i,n in enumerate(('B-gpu6-r0','base-gpu6-r0','B-gpu6-r1'))]
            originals=[];states={}
            for i,spec in enumerate(specs[:2]):
                owner=dict(pid=201+i,process_start_ticks=99,command=['exact'],cwd=str(a.ROOT))
                originals.append(dict(service_id=spec['service_id'],owner=owner));states[201+i]='R'
                write(Path(spec['directory'])/'owner.json',owner)
            partial=Path(specs[2]['directory']);partial.mkdir(parents=True)
            pending=SimpleNamespace(reaped=True,receipt=lambda:dict(reaped=True,identity_bound=False))
            spawned=SimpleNamespace(children=[(specs[2],pending)])
            def birth(pid):return dict(pid=pid,process_start_ticks=99,state=states[pid])
            def identity(pid):return next(x['owner'] for x in originals if x['owner']['pid']==pid)
            checked=[]
            services=SimpleNamespace(SERVICE_OUTPUT=out,layout=lambda *_:specs,
                 service_lock=lambda _:contextlib.nullcontext(),process_identity=identity,
                 tcp_connections=lambda *_:[],assert_cleanup_safe=lambda owner,spec,actual,connections:checked.append(owner['pid']))
            with patch.object(a,'child_birth',side_effect=birth),patch.object(a,'open_pidfd',side_effect=lambda pid:pid), \
                 patch.object(a,'send_pidfd',side_effect=lambda fd,sig:states.__setitem__(fd,'Z')), \
                 patch.object(a,'close_pidfd'),patch.object(a.time,'sleep'):
                receipt=a.cleanup_known_services(services,spawned,location,originals)
            self.assertTrue(receipt['all_owned_services_stopped']);self.assertEqual(set(receipt['signalled_owned_pids']),{201,202})
            self.assertTrue(partial.exists());self.assertFalse((partial/'owner.json').exists());self.assertEqual(set(checked),{201,202})

    @unittest.skipUnless(a.os.name=='posix' and hasattr(a.os,'pidfd_open'),'Linux CPU-only child-handle check')
    def test_real_linux_child_survives_transient_empty_argv_then_is_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd=Path(tmp).resolve();command=[a.sys.executable,'-c','import time; time.sleep(60)']
            process=a.subprocess.Popen(command,cwd=cwd,stdin=a.subprocess.DEVNULL,
                                       stdout=a.subprocess.DEVNULL,stderr=a.subprocess.DEVNULL)
            tracked=a.TrackedChild(process,command,cwd)
            original=a.child_identity;seen=[0]
            def transient(pid):
                value=original(pid);seen[0]+=1
                if seen[0]<=2:value['command']=['']
                return value
            try:
                with patch.object(a,'child_identity',side_effect=transient):owner=tracked.await_identity()
                self.assertEqual(owner['command'],command);self.assertGreaterEqual(seen[0],4)
            finally:tracked.drain()
            self.assertTrue(tracked.reaped);self.assertIsNotNone(process.returncode)


if __name__=='__main__':unittest.main()
