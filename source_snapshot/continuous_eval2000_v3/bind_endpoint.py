"""CPU/file-only binder; completion and branch bytes must already exist. No parity claim."""
import argparse
import json
import os
from pathlib import Path
from .loader import A_BINDING_SHA, RUNTIME_FILES, sha, require, pinned, read_binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run-output',type=Path,required=True)
    p.add_argument('--base-A-binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);run=a.run_output.resolve(strict=True);output=a.output.resolve()
    require(not output.exists() and not output.with_suffix('.pending').exists(),'Endpoint binding already exists; no overwrite')
    arm=run.name;require(arm in ('B',) and run==root/'training/continuous-pilot-2000-v3'/arm,'Wrong fixed pilot output')
    base_item=dict(path=str(a.base_A_binding.resolve(strict=True)),sha256=A_BINDING_SHA);base=pinned(base_item)
    completion=run/'completion.json';run_identity=run/'run_identity.json';complete=json.loads(completion.read_text())
    branch=complete['branch'];require(Path(branch['path']).resolve()==run/'branch-00002000.pt','Sidecar must be this run terminal100 export')
    paths=set(RUNTIME_FILES)|{str(Path(x).resolve().relative_to(root)) for x in base['source_sha256']}
    source={str(root/name):sha(root/name) for name in sorted(paths)}
    binding=dict(schema='continuous_eval2000_v3_endpoint_v1',ready=True,arm=arm,root=str(root),endpoint_updates=2000,endpoint_samples=256000,
        base_A_binding=base_item,branch=branch,branch_identity=complete['branch_identity'],
        completion=dict(path=str(completion),sha256=sha(completion)),run_identity=dict(path=str(run_identity),sha256=sha(run_identity)),
        source_sha256=source,inference_arithmetic='frozen_A_bf16_forward_fp32_stage_master_autocast',
        online_stage_conditioning=True,no_label_actor_inputs=True,actual_direct_socket_parity_passed=False)
    output.parent.mkdir(parents=True,exist_ok=True);pending=output.with_suffix('.pending')
    with pending.open('x') as f:json.dump(binding,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    read_binding(pending,sha(pending));os.link(pending,output);pending.unlink()
    print(json.dumps(dict(output=str(output),sha256=sha(output),arm=arm,metadata_valid=True,GPU_started=False,parity_not_yet_run=True)))

if __name__=='__main__':main()
