"""Offline infrastructure checks; no SSH, CUDA, rollout, or Astra calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

FILE = Path(__file__).parents[1]/'multigpu_services_v2.py'
SPEC = importlib.util.spec_from_file_location('multigpu_services_under_test', FILE)
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


class MultiGpuTests(unittest.TestCase):
    def services(self):
        return SimpleNamespace(validate_owner=Mock(), memory_required_mib=lambda n:max(16384,n*12288+4096) if n else 4096)

    def snapshot(self, gpu=3):
        return dict(gpu_index=gpu, gpu_uuid=m.GPU_UUIDS[gpu], ecc_uncorrected=0, free_mib=81920, occupants=[])

    def test_layout_capacity_unique_across_gpu_groups(self):
        both = m.layout(3)+m.layout(6)
        self.assertEqual(len({s['service_id'] for s in both}), 12)
        self.assertEqual(len({s['port'] for s in both}), 12)
        self.assertEqual({s['port'] for s in m.layout(3)}, set(range(28020,28026)))
        self.assertEqual({s['port'] for s in m.layout(6)}, set(range(28030,28036)))
        for gpu in (3,6):
            self.assertEqual(sum(s['policy_id']=='B' for s in m.layout(gpu)), 4)
            self.assertEqual(sum(s['slots'] for s in m.layout(gpu)), 36)
            self.assertTrue(all(s['gpu_index']==gpu and s['gpu_uuid']==m.GPU_UUIDS[gpu] for s in m.layout(gpu)))

    def test_forbidden_gpu_and_unsupported_layout_rejected(self):
        for gpu in (2,7,0,True):
            with self.assertRaises(RuntimeError): m.layout(gpu)
        with self.assertRaises(RuntimeError): m.layout(3,5,1)

    def test_first_load_requires76GiB(self):
        snapshot=self.snapshot(); snapshot['free_mib']=76*1024-1
        with self.assertRaisesRegex(RuntimeError,'needs77824|needs 77824'): m.admit_gpu(self.services(),3,snapshot,[],6)
        snapshot['free_mib']=76*1024
        m.admit_gpu(self.services(),3,snapshot,[],6)

    def test_foreign_occupancy_rejected_even_with_free_memory(self):
        snapshot=self.snapshot(); snapshot['occupants']=[dict(pid=12,gpu_uuid=m.GPU_UUIDS[3])]
        with self.assertRaisesRegex(RuntimeError,'Foreign'): m.admit_gpu(self.services(),3,snapshot,[],6)

    def test_exact_own_occupant_validates_full_owner(self):
        snapshot=self.snapshot(); snapshot['occupants']=[dict(pid=12,gpu_uuid=m.GPU_UUIDS[3])]
        owner=dict(pid=12,service_id='B-gpu3-r0'); services=self.services()
        m.admit_gpu(services,3,snapshot,[owner],5)
        services.validate_owner.assert_called_once_with(owner,m.layout(3)[0])
        services.validate_owner.side_effect=RuntimeError('owner drift')
        with self.assertRaisesRegex(RuntimeError,'drift'): m.admit_gpu(services,3,snapshot,[owner],5)

    def test_uuid_ecc_and_remaining_memory_fail_closed(self):
        for key,value in [('gpu_uuid',m.GPU_UUIDS[6]),('ecc_uncorrected',1),('free_mib',16000)]:
            snapshot=self.snapshot(); snapshot[key]=value
            with self.assertRaises(RuntimeError): m.admit_gpu(self.services(),3,snapshot,[],1)

    def test_private_adapter_corrects_literal_gpu6_profile(self):
        original=SimpleNamespace(profile_from_manifest=lambda *a:dict(gpu_index=6,other='checked'))
        with patch.object(m,'load_pinned',return_value=original):
            services=m.services_adapter(3)
        profile=services.profile_from_manifest({}, {}, {}, 'sha', {})
        self.assertEqual(profile['gpu_index'],3)
        self.assertEqual(profile['gpu_uuid'],m.GPU_UUIDS[3])
        self.assertEqual(profile['other'],'checked')
        self.assertEqual(profile['frozen_profile_source_sha256'],m.SERVICES_SHA)
        self.assertEqual(services.GPU_INDEX,3)

    def test_service_commands_pin_unchanged_server_and_explicit_gpu_wrapper(self):
        services=SimpleNamespace(B_BINDING=Path('/binding'),B_BINDING_SHA='binding-sha',B_PARITY=Path('/parity'),B_PARITY_SHA='parity-sha')
        commands=[m.service_command(services,s) for s in m.layout(3)]
        self.assertIn('continuous_eval2000_v3.server',commands[0])
        self.assertIn('binding-sha',commands[0])
        self.assertIn('--_serve-base',commands[-1])
        self.assertEqual(commands[-1][commands[-1].index('--gpu')+1],'3')
        self.assertEqual(commands[-1][commands[-1].index('--port')+1],'28024')
        self.assertFalse(any('runner' in str(c) or 'broker' in str(c) for c in commands))

    def test_source_tampering_rejected_before_import(self):
        with tempfile.TemporaryDirectory(dir=FILE.parent) as td:
            path=Path(td)/'module.py';path.write_text('raise AssertionError("MUST NOT EXECUTE")')
            with self.assertRaisesRegex(RuntimeError,'changed'): m.load_pinned(path,'never_imported','0'*64)

    def test_release_requires_exact_inventory_and_worker_attestation(self):
        request=dict(schema='originx_multigpu_service_release_v1',services_ready_sha256='s',all_assigned_workers_stopped=True)
        m.validate_release(request,'s')
        for key,value in [('services_ready_sha256','wrong'),('all_assigned_workers_stopped',False),('schema','other')]:
            bad=dict(request);bad[key]=value
            with self.assertRaises(RuntimeError):m.validate_release(bad,'s')

    def test_cleanup_does_not_signal_model_with_active_connections(self):
        with tempfile.TemporaryDirectory(dir=FILE.parent) as td:
            guardian=m.Guardian.__new__(m.Guardian);guardian.output=Path(td);guardian.gpu=3;guardian.deadline=0
            child=SimpleNamespace(pid=123,reaped=False,child=SimpleNamespace(poll=lambda:None),receipt=lambda:dict(pid=123),drain=Mock())
            owner=dict(pid=123,process_start_ticks=17,command=['real'],cwd='/p')
            guardian.children=[('B-gpu3-r0',child)]
            guardian.records=[(m.layout(3)[0],owner,child)]
            guardian.services=SimpleNamespace(process_identity=lambda pid:owner,tcp_connections=lambda *a:[dict(state_hex='01')])
            result=guardian.cleanup()
            child.drain.assert_not_called()
            self.assertFalse(result['all_children_reaped'])
            self.assertTrue(result['blocked_connections'])
            self.assertTrue((Path(td)/'draining.json').exists())

    def test_cleanup_refuses_live_owner_identity_drift(self):
        with tempfile.TemporaryDirectory(dir=FILE.parent) as td:
            guardian=m.Guardian.__new__(m.Guardian);guardian.output=Path(td);guardian.gpu=3;guardian.deadline=0
            child=SimpleNamespace(pid=123,reaped=False,child=SimpleNamespace(poll=lambda:None),receipt=lambda:dict(pid=123),drain=Mock())
            owner=dict(pid=123,process_start_ticks=17,command=['real'],cwd='/p')
            guardian.children=[('B-gpu3-r0',child)];guardian.records=[(m.layout(3)[0],owner,child)]
            guardian.services=SimpleNamespace(process_identity=lambda pid:dict(owner,command=['changed']),tcp_connections=lambda *a:[])
            result=guardian.cleanup();child.drain.assert_not_called()
            self.assertTrue(result['errors']);self.assertFalse(result['all_children_reaped'])

    def test_probe_inherits_cpu_only_and_rejects_cuda_environment(self):
        with patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':m.GPU_UUIDS[3]}):
            with self.assertRaisesRegex(RuntimeError,'CPU-only'):m.probe_entry(3)

    def test_lease_range_checked_before_any_live_reads(self):
        with patch.object(m,'check',side_effect=AssertionError('should not call')):
            for duration in (1199,435601):
                with self.assertRaisesRegex(RuntimeError,'Lease'):m.run(3,duration)


if __name__=='__main__':unittest.main()
