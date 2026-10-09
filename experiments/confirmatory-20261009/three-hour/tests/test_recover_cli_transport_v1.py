"""No-network recovery checks, including the preserved real v1 failed CLI."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def module(name, file):
    spec=importlib.util.spec_from_file_location(name, HERE/file)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


h=module('recovery_test','recover_cli_transport_v1.py')
b=module('broker_recovery_test','broker_multigpu_v2.py')
ad=module('aggregate_adapter_test','aggregate_transport_v2.py')
REAL = HERE/'broker_multigpu_v1/batches/20261009T211401-24fa091b6040'


@pytest.fixture
def saved():
    return (json.loads((REAL/'receipt.json').read_text(encoding='utf-8')),
            (REAL/'stdout.jsonl').read_bytes(), (REAL/'response.json').read_bytes())


def hypothetical_new_receipt(old):
    # In-memory candidate only. The historical v1 receipt is NEVER rewritten.
    value=copy.deepcopy(old)
    value.update(recovered_event_errors=value['event_errors'], event_errors=[], recovered_transport_candidate=True,
                 status='ok', error_kind=None, error_message=None, output_validated=True)
    return value


def test_real_cli_output_and_usage_are_recoverable_but_old_receipt_stays_error(saved):
    old, raw, answer=saved
    assert old['returncode']==0 and old['turn_completed'] and old['status']=='error'
    review=h.recovered_transport_trace(h.parse_events(raw))
    assert review['usage_totals']['input_tokens']==17205 and review['usage_totals']['output_tokens']==299
    original_sha=hashlib.sha256((REAL/'receipt.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Unvalidated'):
        h.validate_completed_recovery(old,raw,answer,h.validate_output_contract)
    proof, parsed=h.validate_completed_recovery(hypothetical_new_receipt(old),raw,answer,h.validate_output_contract)
    assert proof['cli_reinvoked'] is False and len(parsed)==1
    assert hashlib.sha256((REAL/'receipt.json').read_bytes()).hexdigest()==original_sha
    assert old['status']=='error'


@pytest.mark.parametrize('mutation', ['fatal','quota','after_terminal','tool','incomplete','second_turn','wrong_model'])
def test_recovery_rejects_fatal_or_nontransport_conditions(saved,mutation):
    _,raw,_=saved;events=h.parse_events(raw)
    if mutation=='fatal':events.insert(-1,dict(type='turn.failed',error='failed'))
    elif mutation=='quota':events[2]['message']='Rate limit reached 429'
    elif mutation=='after_terminal':events.append(events[2])
    elif mutation=='tool':events.insert(-1,dict(type='item.completed',item={'type':'command_execution'}))
    elif mutation=='incomplete':events.pop()
    elif mutation=='second_turn':events.insert(-1,dict(type='turn.started'))
    elif mutation=='wrong_model':events[0]['model']='different'
    with pytest.raises(ValueError):h.recovered_transport_trace(events)


def test_new_broker_preserves_recovered_errors_and_malformed_stdout_fails_closed(saved):
    old,raw,_=saved; inspected=b.inspect_events(raw)
    assert inspected['event_errors']==[] and inspected['recovered_event_errors']==old['event_errors']
    assert inspected['recovered_transport_candidate'] is True
    malformed=b.inspect_events(raw+b'not-json\n')
    assert malformed['event_errors'] and not malformed.get('recovered_transport_candidate')


def test_nonzero_process_or_changed_artifacts_cannot_be_recovered(saved):
    old,raw,answer=saved;value=hypothetical_new_receipt(old);value['returncode']=1
    with pytest.raises(ValueError,match='terminate successfully'):
        h.validate_completed_recovery(value,raw,answer,h.validate_output_contract)
    value=hypothetical_new_receipt(old)
    with pytest.raises(ValueError,match='hashes differ'):
        h.validate_completed_recovery(value,raw,answer+b' ',h.validate_output_contract)


def test_proof_revalidation_rejects_token_or_receipt_tamper(saved):
    old,raw,answer=saved;value=hypothetical_new_receipt(old)
    proof,parsed=h.validate_completed_recovery(value,raw,answer,h.validate_output_contract)
    assert h.verify_published_recovery(value,proof,raw,answer,h.validate_output_contract)==parsed
    broken=copy.deepcopy(proof);broken['usage_totals']['input_tokens']=1
    with pytest.raises(ValueError,match='independent revalidation'):
        h.verify_published_recovery(value,broken,raw,answer,h.validate_output_contract)


def test_real_run_cli_integration_mock_process_never_invokes_cli(tmp_path,monkeypatch,saved):
    old,stdout,answer=saved;batch=tmp_path/'newbatch';batch.mkdir()
    shutil.copytree(REAL/'requests',batch/'requests')
    original_manifest=json.loads((REAL/'batch.json').read_text(encoding='utf-8'))
    (batch/'batch.json').write_text(json.dumps(dict(original_manifest,broker_source_sha256=b.digest(Path(b.__file__).read_bytes()))))
    stderr=(REAL/'stderr.txt').read_bytes()
    called=[]

    class NoProcess:
        returncode=0
        def communicate(self,prompt,timeout):
            assert isinstance(prompt,bytes)
            (batch/'response.json').write_bytes(answer)
            return stdout,stderr
    def fake_popen(command,**kwargs):
        called.append(command);return NoProcess()
    monkeypatch.setattr(b.subprocess,'Popen',fake_popen)
    records=[(f,b.validate_request(f)) for f in (batch/'requests').iterdir()]
    receipt=b.run_cli(batch,records,['codex.exe'])
    assert len(called)==1 and receipt['status']=='ok' and receipt['output_validated']
    assert receipt['recovered_event_errors']==old['event_errors']
    proof=json.loads((batch/'transport_recovery_audit.json').read_text(encoding='utf-8'))
    assert b.digest((batch/'transport_recovery_audit.json').read_bytes())==receipt['recovery_audit_sha256']
    h.verify_published_recovery(receipt,proof,stdout,answer,h.validate_output_contract)


def fake_aggregate():
    def strict(events):
        problems=['cli_trace_error'] if any(e['type']=='error' for e in events) else []
        return dict(problems=problems)
    obj=SimpleNamespace(_inspect_events=strict)
    obj.read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    obj.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def original(output,job,*_):
        raw=(Path(output)/'requests'/job['request_id']/'cli_stdout.jsonl').read_bytes()
        result=obj._inspect_events(h.parse_events(raw))
        if result['problems']:raise ValueError('cli_trace_error')
        return {'accepted':True}
    obj._generated_evidence=original
    return obj


def test_adapter_only_accepts_v2_verified_proof_never_v1(tmp_path,saved):
    old,raw,answer=saved;receipt=hypothetical_new_receipt(old)
    proof,_=h.validate_completed_recovery(receipt,raw,answer,h.validate_output_contract)
    proofbytes=json.dumps(proof).encode();receipt.update(recovery_audit_file='transport_recovery_audit.json',recovery_audit_sha256=hashlib.sha256(proofbytes).hexdigest())
    job={'request_id':old['request_ids'][0]}
    for namespace in ('originx-confirmatory-multigpu-20261009-v1',ad.OUTPUT_NAME):
        folder=tmp_path/namespace/'requests'/job['request_id'];folder.mkdir(parents=True)
        (folder/'cli_receipt.json').write_text(json.dumps(receipt));(folder/'transport_recovery_audit.json').write_bytes(proofbytes)
        (folder/'cli_stdout.jsonl').write_bytes(raw);(folder/'cli_final_output.json').write_bytes(answer)
    ag=ad.install(fake_aggregate());strict=ag._inspect_events
    with pytest.raises(ValueError,match='cli_trace_error'):
        ag._generated_evidence(tmp_path/'originx-confirmatory-multigpu-20261009-v1',job,{},'original',1,{})
    assert ag._generated_evidence(tmp_path/ad.OUTPUT_NAME,job,{},'original',1,{})=={'accepted':True}
    assert ag._inspect_events is strict
    folder=tmp_path/ad.OUTPUT_NAME/'requests'/job['request_id']
    (folder/'transport_recovery_audit.json').write_text('{}')
    with pytest.raises(ValueError,match='proof hash'):
        ag._generated_evidence(tmp_path/ad.OUTPUT_NAME,job,{},'original',1,{})
