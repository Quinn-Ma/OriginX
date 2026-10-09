import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
s=importlib.util.spec_from_file_location('finalizer_v3',HERE/'finalize_full_v3.py')
f=importlib.util.module_from_spec(s);s.loader.exec_module(f)

def test_no_cli_inventory_does_not_scan_unrelated_codex(tmp_path,monkeypatch):
    (tmp_path/'broker_status.json').write_text(json.dumps({'cli_calls':0}))
    monkeypatch.setattr(f,'live_cli_for_batches',lambda paths:pytest.fail('Empty paths cannot be scanned'))
    assert f.live_owned_cli([],tmp_path)==[]

def test_missing_and_conflicting_zero_inventory_fail(tmp_path):
    with pytest.raises((RuntimeError,FileNotFoundError)):f.live_owned_cli([],tmp_path)
    (tmp_path/'broker_status.json').write_text(json.dumps({'cli_calls':1}))
    with pytest.raises((RuntimeError,FileNotFoundError)):f.live_owned_cli([],tmp_path)

def test_real_started_batches_use_scoped_scan(tmp_path,monkeypatch):
    p=tmp_path/'batches'/'unique'/'cli_started.json'
    monkeypatch.setattr(f,'live_cli_for_batches',lambda paths:[{'batch':paths[0]}])
    assert f.live_owned_cli([p],tmp_path)==[{'batch':str(p.parent)}]

@pytest.mark.parametrize("value",[{}, {"cli_calls":False}, {"cli_calls":None}, {"cli_calls":"0"}])
def test_missing_or_untyped_zero_is_not_evidence(tmp_path,value):
    (tmp_path/"broker_status.json").write_text(json.dumps(value))
    with pytest.raises(RuntimeError):f.live_owned_cli([],tmp_path)
