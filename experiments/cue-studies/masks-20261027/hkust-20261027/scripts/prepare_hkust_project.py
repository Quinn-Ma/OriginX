"""HKUST cluster deployment of the mask study: verify + extract the frozen vendor archives and
write the T3 site file.  Same archives and site shape as the AutoDL deployment
(prepare-cloud-project.py), with cluster paths; envs/models live on the RAM tier."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import tarfile

ROOT = Path('/ephemeral/qinzhen/originx-harness-v2')
WORK = ROOT / 'work/hkust'
VENDOR = ROOT / 'vendor'
ENVS = Path('/dev/shm/qinzhen/originx-envs')
GPU = '5'  # GPU 2/3 carry 'GPU Recovery Action: Reset' and corrupt or segfault EGL depth readback; 5 and 7 are healthy
archives = {
    'stage-a-upstreams.tar.gz': '57cf637ab54242b424d68660ad817d7ef20ce8c8aceb783c9f35252f69f2fdad',
    'stage-a-assets.tar.gz': '6f7ed1c3b74b3933e00e56e87838ef9ed286f083bf854af256c43aa83acfd425',
    'dev-seeds-assets-delta-v1.tar.gz': sys.argv[1],
    'formal-seeds-assets-delta-v1.tar.gz': sys.argv[2],
}
receipts = []
for name, expected in archives.items():
    path = WORK / name
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f'Archive hash mismatch: {name}')
    marker = WORK / (name + '.extracted.json')
    if not marker.exists():
        with tarfile.open(path) as tar:
            tar.extractall(VENDOR, filter='data')
        marker.write_text(json.dumps({'sha256': digest, 'bytes': path.stat().st_size}) + '\n')
    receipts.append({'archive': name, 'sha256': digest, 'bytes': path.stat().st_size})
site = {
    'site_name': 'linux_egl', 'placement': 'linux_egl', 'benchmark': 'robocasa', 'model': 'xr1',
    'sequential_gpu': False, 'model_residency': 'resident', 'native_device': 'cuda:0',
    'python': {'sim': str(ENVS / 'sim/bin/python'), 'policy': str(ENVS / 'policy/bin/python'),
               'planner': str(ENVS / 'planner/bin/python')},
    'pythonpath': {'sim': [str(VENDOR / 'robocasa'), str(VENDOR / 'robosuite')]},
    'env': {'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'CUDA_VISIBLE_DEVICES': GPU,
            'MUJOCO_EGL_DEVICE_ID': GPU, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HOME': str(WORK / 'hf-cache'), 'MPLBACKEND': 'Agg',
            'MPLCONFIGDIR': str(WORK / 'mpl-cache')},
    'renderer': {'kind': 'native_nvidia_egl', 'gl_vendor': 'NVIDIA Corporation',
                 'gl_renderer_prefix': 'NVIDIA ', 'gl_renderer': 'NVIDIA H100 PCIe/PCIe/SSE2',
                 'forbidden_env': ['MESA_D3D12_DEFAULT_ADAPTER_NAME']},
    'scope': 'T3 engineering only; policy/planner paths are not yet admitted for task runs',
    'policy_rgb_samples': 0,
}
path = WORK / 'site-t3.json'
if path.exists() and json.loads(path.read_text()) != site:
    raise ValueError('Existing T3 site differs')
path.write_text(json.dumps(site, indent=2) + '\n')
(WORK / 'transfer-and-site-receipt.json').write_text(json.dumps({
    'host': 'hkust-cluster h800-8-1 (NVIDIA H100 PCIe x8)', 'archives': receipts, 'site': str(path),
    'python_original': '3.11.13', 'python_deployment': sys.version, 'same_python_patch_claimed': False,
    'env_tier': 'RAM tier /dev/shm/qinzhen (regenerable from the lock files under work/hkust)',
    'asset_delta_validation': 'pending', 'policy_calls': 0, 'task_rollouts': 0,
}, indent=2) + '\n')
print(json.dumps({'site': str(path), 'archives_verified_and_extracted': len(receipts)}))
