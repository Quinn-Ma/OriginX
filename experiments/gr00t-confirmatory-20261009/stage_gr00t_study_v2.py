"""Seal a distinct infrastructure-corrected study without replacing v1."""
from pathlib import Path
import hashlib,json,subprocess,tarfile
HERE=Path(__file__).resolve().parent
LOCAL=HERE.parent/'originx_gr00t_confirmation_20261009_v2'
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
BASE=ROOT+'/originx_gr00t_confirmation_20261009_v2'
SERVICE=ROOT+'/originx_gr00t_confirmation_20261009'
FILES=['__init__.py','adapter.py','aggregate.py','assistance.py','broker.py','campaign.py',
       'capture_native_fixtures.py','client.py','core.py','parity_probe.py','processes.py',
       'runner.py','server.py','wire.py']
SERVICE_FILES=['__init__.py','adapter.py','wire.py','core.py','server.py','client.py','parity_probe.py']
if {p.name for p in LOCAL.glob('*.py')}!=set(FILES):raise RuntimeError('Unexpected source inventory')
pins={name:hashlib.sha256((LOCAL/name).read_bytes()).hexdigest() for name in FILES}
archive=HERE/'gr00t-study-v2-source.tar.gz'
with tarfile.open(archive,'x:gz') as t:
    for name in FILES:t.add(LOCAL/name,arcname=name)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
remote_archive=ROOT+'/originx_confirmatory_20261009/gr00t-study-v2-source.tar.gz'
subprocess.run(['scp','-q',str(archive),'hkust-cluster:'+remote_archive],check=True,timeout=60)
code='''from pathlib import Path
import hashlib,json,tarfile,datetime
b=Path(%r);service=Path(%r);a=Path(%r);pins=%r;service_files=%r
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(a)!=%r:raise RuntimeError('Source archive mismatch')
if b.exists():raise RuntimeError('v2 namespace already exists; inspect rather than overwrite')
if sha(service/'source-freeze.json')!='b2518de83d2370f8456248f51b962e0d63f41b54db23046f23f0e91492d7bd51':raise RuntimeError('Prior v1 freeze changed')
for name,expected in json.loads((service/'source-freeze.json').read_text())['source_sha256'].items():
 if sha(service/name)!=expected:raise RuntimeError('Prior frozen source changed: '+name)
for name in service_files:
 if sha(service/name)!=pins[name]:raise RuntimeError('Actual model service bytes changed: '+name)
b.mkdir(exist_ok=False)
with tarfile.open(a) as t:
 if {m.name for m in t.getmembers()}!=set(pins):raise RuntimeError('Unexpected members')
 for m in t.getmembers():
  if not m.isfile() or m.issym() or m.islnk() or Path(m.name).name!=m.name:raise RuntimeError('Unsafe member')
 t.extractall(b,filter='data')
for name,expected in pins.items():
 if sha(b/name)!=expected:raise RuntimeError('Extracted source mismatch')
receipt=dict(schema='originx_gr00t_source_freeze_v1',namespace=b.name,service_namespace=service.name,
 source_sha256=pins,broker_source_sha256=pins['broker.py'],created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 confirmation_outcomes_seen=False,model_training=False,
 prior_failure_retained='results/originx-gr00t-confirmatory-development-20261009-v1',
 change_scope='Process startup ownership handshake and distinct study namespace; actual seven service modules unchanged')
with (b/'source-freeze.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')
print(json.dumps(dict(source_freeze_path=str(b/'source-freeze.json'),source_freeze_sha256=sha(b/'source-freeze.json'),**receipt)))
'''%(BASE,SERVICE,remote_archive,pins,SERVICE_FILES,digest)
p=subprocess.run(['ssh','-o','BatchMode=yes','hkust-cluster','python3','-'],input=code.encode(),capture_output=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
receipt=json.loads(p.stdout)
with (LOCAL/'source-freeze.remote.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}))
