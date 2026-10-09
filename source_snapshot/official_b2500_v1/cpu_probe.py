"""Import/runtime admission before GPU services; never construct env or query GPU."""
import importlib.metadata,json,os,sys
from pathlib import Path
from official_b2500_v1 import runner
R=Path(__file__).resolve().parent.parent;P=R/'official_b2500_v1';D=R/'results/official-b2500-v1'
def main():
 b=runner.read(R/'results/continued-ab2000-v1/B-bindings.json')
 assert Path(b['root']).resolve()==R and Path(b['python']).resolve()==Path(sys.executable).resolve()
 for key,value in runner.environment(b).items():
  if key in ['MUJOCO_GL','PYOPENGL_PLATFORM','CUDA_VISIBLE_DEVICES','LP_NUM_THREADS','LD_LIBRARY_PATH','PYTHONHASHSEED','NUMBA_CACHE_DIR']:assert os.environ.get(key)==value,key
 for p,h in b['source_sha256'].items():assert runner.sha(p)==h,p
 for n,h in runner.read(P/'runner-source-pins.json').items():assert runner.sha(P/n)==h,n
 assert {n:importlib.metadata.version(n) for n in b['packages']}==b['packages']
 sys.path[:0]=[str(P),str(R/'remote_processor_candidate_v1/deps'),str(R)]
 import dev_randomized_broad_alpha as legacy,dev_repeat_audit as audit,readback_guard,context_lifetime,candidate
 for mod in (legacy,audit,legacy.common,audit.reference_tools,readback_guard,context_lifetime,candidate):
  source=Path(mod.__file__).resolve();assert source.parent==P and runner.sha(source)==runner.read(P/'runner-source-pins.json')[source.name]
 import torch,gymnasium,robocasa
 assert not torch.cuda.is_initialized() and torch.version.cuda is None
 assert Path(robocasa.__file__).resolve().parent==Path(b['package_root']).resolve()
 from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
 from robocasa.utils.dataset_registry_utils import get_task_horizon
 m=runner.design(P/'manifest.json')
 assert {t['task'] for t in m['tasks']}==set(TASK_SET_REGISTRY['target50'])
 for t in m['tasks']:
  assert t['task'] in TASK_SET_REGISTRY[t['stratum']] and get_task_horizon(t['task'])==t['horizon']
 from continuous_branch_v3.reset_fix import install
 reset=install()
 p=D/'cpu-runtime-preflight.json'
 with p.open('x') as f:json.dump(dict(passed=True,python=sys.executable,source_inventory_sha256=runner.sha(P/'runner-source-pins.json'),tasks_verified=50,packages=b['packages'],reset_patch=reset,environment_constructed=False,GPU_initialized=False),f,indent=2);f.write('\n')
 print('CPU imports,50 task/horizon registry,source identities and reset-patch compatibility passed; no environment or CUDA initialized')
if __name__=='__main__':main()
