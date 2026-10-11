"""No-model terminal evidence watcher, with a bounded CPU audit child."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from campaign import ROOT,tracker
from episode import read,write,sha,require,identity,environment,active

def run(seconds):
    state=HERE/'final-audit-wait-v2';state.mkdir(exist_ok=False);write(state/'owner.json',identity(),exclusive=True)
    cutoff=time.monotonic()+seconds;seal=HERE/'bv_broker_terminal_state.seal.json';broker=HERE/'bv_broker_terminal_state'
    core=ROOT/'results/originx-sprint-xab-20261011-v1';bv=ROOT/'results/originx-sprint-bv-20261011-v1'
    try:
        director=HERE/'director'
        while True:
            gates=dict(broker_seal=seal.exists(),core_terminal=(core/'completion.json').exists(),BV_terminal=(bv/'completion.json').exists(),
                       director_terminal=(director/'completion.json').exists(),transfer_cleanup=(director/'transfer-cleanup.json').exists())
            write(state/'status.json',dict(state='audit_waiting_incomplete_usage' if not gates['broker_seal'] else 'audit_waiting_terminal_cleanup',gates=gates))
            if all(gates.values()):
                transfer=read(director/'transfer-cleanup.json');workers=read(director/'worker-cleanup.json')
                require(transfer.get('all_workers_inactive') is True and transfer.get('all_models_inactive') is True and
                        transfer.get('service_finished',{}).get('all_children_reaped') is True,'Transferred cleanup incomplete')
                require(all(not active(x['owner']) for x in transfer['models']) and all(not active(o) for o in workers['owners']),'Owned worker/model still active')
                if not active(read(director/'owner.json')) and not active(read(ROOT/'results/originx-sprint-xab-services-20261011-v1/owner.json')):break
            require(time.monotonic()<cutoff,'audit_waiting_incomplete_usage_or_cleanup at audit delivery deadline; no inferred zero')
            time.sleep(10)
        receipt=read(seal);require(sha(HERE/'bv_broker_terminal_state.tar.gz')==receipt['archive_sha256'],'Sealed archive differs');actual={str(p.relative_to(broker)):sha(p) for p in broker.rglob('*') if p.is_file()}
        require(actual==receipt['files'],'Sealed broker inventory differs')
        require(sha(HERE/'final_audit.py')=='ff6d374a3d5a92b45f689185ba917046c77e403585505fedab949bfa223d7265','Postprocessor changed')
        require(sha(HERE/'transport_audit.py')=='257a3c10d107faa1ee44465137d7dffbe0e1083bdb84832c98049b9ab4a8289d','Transport adapter changed')
        c=read(core/'config.json');destination=ROOT/'results/originx-sprint-final-audit-20261011-v1'
        command=[c['python'],'-B','-u',str(HERE/'final_audit.py'),'--output',str(destination),'--broker-state',str(broker)]
        log=(state/'audit.log').open('x');child=subprocess.Popen(command,cwd=ROOT,env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        tracked=tracker()(child,command,ROOT);write(state/'spawn-pending.json',tracked.receipt(),exclusive=True)
        try:
            write(state/'audit-owner.json',tracked.await_identity(),exclusive=True)
            until=time.monotonic()+300
            while child.poll() is None and time.monotonic()<until:time.sleep(1)
            if child.poll() is None:tracked.drain(grace=10,kill_timeout=5)
            else:tracked.close()
            require(child.returncode==0,'Final audit CPU child failed')
            write(state/'complete.json',dict(report=str(destination/'report.json'),report_sha256=sha(destination/'report.json'),broker_seal_sha256=sha(seal),audit_child_reaped=True),exclusive=True)
        finally:
            if child.poll() is None:tracked.drain(grace=10,kill_timeout=5)
            log.close();write(state/'audit-exit.json',tracked.receipt(),exclusive=True)
    except BaseException:
        write(state/'first-error.json',dict(error=traceback.format_exc()),exclusive=True);return 1
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-seconds',type=float,required=True);a=p.parse_args();sys.exit(run(a.wait_seconds))
