"""No SSH, real process spawning, or real signals."""
import importlib.util
import json
import signal
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deadline_test', HERE/'deadline_multigpu_v2.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

@pytest.fixture(autouse=True)
def linux_signal_constant(monkeypatch):
    monkeypatch.setattr(signal,'SIGKILL',9,raising=False)

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')

def setup(tmp_path):
    out=tmp_path/'results'/d.NAME
    cfg=out/'config.json';put(cfg,dict(output=str(out)));digest=d.sha(cfg)
    command=['python','-u',str(tmp_path/'continue_multigpu_v2.py'),'campaign','--config',str(cfg),'--config-sha256',digest]
    campaign=dict(pid=100,process_start_ticks=400,cwd=str(tmp_path),command=command)
    worker=dict(pid=101,process_start_ticks=401,cwd=str(tmp_path),command=['python','-u',str(tmp_path/'originx_confirmatory_20261009/runner.py'),'episode','--config',str(cfg),'--config-sha',digest,'--index','0','--model-index','0'])
    put(out/'campaign.owner.json',campaign);put(out/'processes/job.json',worker)
    put(out/'claims/job.json',dict(config_sha256=digest,command=worker['command']))
    return out,digest,campaign,worker

def call(out,digest,root,live):
    calls=[];sleeps=[]
    result=d.enforce(out,digest,'2026-10-09T22:40:12Z',root=root,observe=lambda p:live.get(p),open_fd=lambda p:p+1000,send=lambda fd,s:calls.append((fd,s)),close=lambda fd:None,sleep=sleeps.append)
    return result,calls,sleeps

def test_terminal_never_opens_or_signals(tmp_path):
    out,digest,c,w=setup(tmp_path);put(out/'campaign-finished.json',{'phase':'completed'})
    result,calls,sleeps=call(out,digest,tmp_path,{c['pid']:c,w['pid']:w})
    assert result['already_terminal'] and not calls and not sleeps
    assert (out/'external-deadline-receipt.json').exists()

def test_exact_owned_campaign_term_workers_term_and_kill(tmp_path):
    out,digest,c,w=setup(tmp_path)
    result,calls,sleeps=call(out,digest,tmp_path,{c['pid']:c,w['pid']:w})
    assert calls==[(1100,signal.SIGTERM),(1101,signal.SIGTERM),(1101,signal.SIGKILL)]
    assert sleeps==[12,15] and not result['errors']

@pytest.mark.parametrize('field,value',[('process_start_ticks',999),('command',['foreign']),('cwd','/foreign')])
def test_live_identity_mismatch_never_signalled(tmp_path,field,value):
    out,digest,c,w=setup(tmp_path);changed=dict(w);changed[field]=value
    result,calls,_=call(out,digest,tmp_path,{w['pid']:changed})
    assert not calls and result['errors']

def test_model_or_foreign_config_argv_rejected(tmp_path):
    out,digest,c,w=setup(tmp_path)
    w['command'][3]='serve-model';put(out/'processes/job.json',w)
    result,calls,_=call(out,digest,tmp_path,{w['pid']:w})
    assert not calls and result['errors']

def test_pid_reuse_after_open_is_rejected(tmp_path):
    out,digest,c,w=setup(tmp_path);seen=0;calls=[]
    def observe(pid):
        nonlocal seen
        if pid==c['pid']:return None
        seen+=1
        return w if seen==1 else dict(w,process_start_ticks=999)
    result=d.enforce(out,digest,'2026-10-09T22:40:12Z',root=tmp_path,observe=observe,open_fd=lambda p:p,send=lambda *args:calls.append(args),close=lambda fd:None,sleep=lambda _:None)
    assert result['errors'] and not calls

def test_before_deadline_refused(tmp_path):
    out,digest,c,w=setup(tmp_path)
    with pytest.raises(RuntimeError,match='before fixed deadline'):
        d.enforce(out,digest,'2026-10-09T22:40:11Z',root=tmp_path)
    assert not (out/'external-deadline-intent.json').exists()
