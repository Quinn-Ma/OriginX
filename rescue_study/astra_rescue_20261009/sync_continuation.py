"""Read-only, bounded campaign synchronization. Never starts model/rollout jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time
import uuid

PROJECT = '/ephemeral/qinzhen/robocasa-xr1-20261003'
HOST = 'hkust-cluster'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
NAME = re.compile(r'astra-native-reset-[A-Za-z0-9_-]+\Z')
POLL_SECONDS = 60
MAX_FAILURES = 6
EVIDENCE_INDEX = '__sync_evidence_manifest__.json'
SNAPSHOT_MEMBER = '__sync_remote_snapshot__.json'

# This program runs over SSH with Python source on stdin. It only reads the
# selected study directory and /proc for its recorded driver; tar is stdout-only.
REMOTE_PROGRAM = r'''
import hashlib,io,json,os,re,stat,sys,tarfile,time
from pathlib import Path,PurePosixPath
PROJECT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAME=re.compile(r'astra-native-reset-[A-Za-z0-9_-]+\Z')
ROOT_FILES={'config.json','manifest.json','preflight.json','progress.json','completion.json',
 'cleanup.json','cleanup-error.json','supervisor-error.json','supervisor-owner.json',
 'launch.json','warmup.json','owner.json','drain.request.json','stop.json','failure.json','events.jsonl',
 'campaign-status.json','campaign-finished.json','campaign-owner.json',
 'continuation-exclusion-ledger.json','protocol-diff.json','campaign-cleanup.json','campaign-error.json',
 'campaign-started.json','broker-ready.json'}
JSON_DIRS={'analysis','episodes','events','processes','claims','errors'}
IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
def encode(v):return (json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
def allowed(relative):
 p=PurePosixPath(relative)
 if p.is_absolute() or '..' in p.parts or 'logs' in p.parts or p.name=='initial.xml':return False
 if len(p.parts)==1:return p.name in ROOT_FILES
 if p.parts[0]=='requests':return p.suffix.lower() in IMAGE_SUFFIXES|{'.json','.jsonl'}
 return p.parts[0] in JSON_DIRS and p.suffix.lower() in {'.json','.jsonl'}
def checked_root(raw):
 p=PurePosixPath(raw)
 if str(p)!=raw or p.parent!=PurePosixPath(str(PROJECT))/'results' or not NAME.fullmatch(p.name):
  raise ValueError('Remote output must be a named direct study child of approved results')
 d=Path(raw).resolve(strict=True)
 if d.parent!=(PROJECT/'results').resolve(strict=True) or d.name!=p.name or not d.is_dir():
  raise ValueError('Remote output resolves outside the approved study directory')
 return d
def safe_file(root,relative):
 p=root/relative
 if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):return None
 return p
def owner_state(owner):
 out={'active':None,'identity_checked':False}
 if isinstance(owner,dict) and isinstance(owner.get('pid'),int) and isinstance(owner.get('process_start_ticks'),int):
  try:
   proc=Path('/proc')/str(owner['pid']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
   match=int(fields[19])==owner['process_start_ticks'] and fields[0] not in ('Z','X')
   if match and 'command' in owner:match=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')==owner['command']
   if match and 'cwd' in owner:match=str((proc/'cwd').resolve())==owner['cwd']
   out={'active':match,'identity_checked':True,'recorded_pid':owner['pid']}
  except (FileNotFoundError,ProcessLookupError):out={'active':False,'identity_checked':True,'recorded_pid':owner['pid']}
  except (OSError,ValueError,IndexError) as e:out['error_type']=type(e).__name__
 return out
def collect(root):
 out={'schema':'astra_campaign_remote_snapshot_v1','remote_output':str(root),'remote_unix':time.time(),
      'files':{},'file_errors':{},'driver_tails':{},'read_only':True}
 for name in ('progress.json','completion.json','analysis/report.json','preflight.json','cleanup.json',
              'cleanup-error.json','supervisor-error.json','supervisor-owner.json','launch.json','stop.json','failure.json',
              'campaign-status.json','campaign-finished.json','campaign-owner.json',
 'continuation-exclusion-ledger.json','protocol-diff.json','campaign-cleanup.json','campaign-error.json',
 'campaign-started.json','broker-ready.json'):
  p=safe_file(root,name)
  if p is None:continue
  try:
   raw=p.read_bytes();out['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
  except (ValueError,OSError) as e:out['file_errors'][name]=type(e).__name__
 for name in ('driver.log','run.log','runner.log'):
  p=safe_file(root,name)
  if p is not None:
   try:
    with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-4000));out['driver_tails'][name]=f.read().decode('utf-8','replace')
   except OSError as e:out['file_errors'][name]=type(e).__name__
 owner=out['files'].get('supervisor-owner.json',out['files'].get('launch.json',{})).get('data')
 out['driver']=owner_state(owner)
 out['campaign_driver']=owner_state(out['files'].get('campaign-owner.json',{}).get('data'))
 return out
class HashReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256()
 def read(self,n=-1):b=self.f.read(n);self.h.update(b);return b
def archive(root):
 files=[];skipped_symlinks=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d!='logs' and not (Path(current)/d).is_symlink())
  for name in sorted(names):
   p=Path(current)/name;relative=p.relative_to(root).as_posix()
   if not allowed(relative):continue
   if p.is_symlink():skipped_symlinks.append(relative);continue
   if not p.resolve().is_relative_to(root):raise ValueError('Archive file escapes study root')
   files.append((relative,p))
 rows=[]
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
  for relative,p in sorted(files):
   before=p.stat()
   if not stat.S_ISREG(before.st_mode):continue
   with p.open('rb') as f:
    opened=os.fstat(f.fileno())
    if (opened.st_ino,opened.st_dev)!=(before.st_ino,before.st_dev):raise RuntimeError('Source changed before read')
    reader=HashReader(f);info=tarfile.TarInfo(relative);info.size=opened.st_size;info.mtime=opened.st_mtime;info.mode=0o600
    tar.addfile(info,reader)
   after=p.stat()
   if (after.st_size,after.st_mtime_ns,after.st_ino)!=(before.st_size,before.st_mtime_ns,before.st_ino):raise RuntimeError('Source changed while archiving')
   rows.append({'path':relative,'bytes':info.size,'sha256':reader.h.hexdigest()})
  snapshot=encode(collect(root));info=tarfile.TarInfo('__sync_remote_snapshot__.json');info.size=len(snapshot);info.mode=0o600
  tar.addfile(info,io.BytesIO(snapshot));rows.append({'path':info.name,'bytes':len(snapshot),'sha256':hashlib.sha256(snapshot).hexdigest()})
  index=encode({'schema':'astra_campaign_evidence_index_v1','remote_output':str(root),'created_unix':time.time(),
    'files':rows,'skipped_symlinks':skipped_symlinks,'read_only_remote':True,
    'exclusions':['initial.xml','logs directories','full driver/run/runner logs (snapshot includes bounded tails)',
                  'files outside the explicit study evidence allowlist']})
  info=tarfile.TarInfo('__sync_evidence_manifest__.json');info.size=len(index);info.mode=0o600;tar.addfile(info,io.BytesIO(index))
if __name__=='__main__':
 try:
  root=checked_root(sys.argv[1]);mode=sys.argv[2]
  if mode=='status':print(json.dumps(collect(root),ensure_ascii=False,allow_nan=False))
  elif mode=='archive':archive(root)
  else:raise ValueError('Unknown read-only mode')
 except Exception as error:
  print('Read-only collector failed: '+type(error).__name__,file=sys.stderr);raise SystemExit(2)
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_remote(value):
    path = PurePosixPath(value)
    if str(path) != value or path.parent != PurePosixPath(PROJECT) / 'results' or not NAME.fullmatch(path.name):
        raise ValueError('--remote-output must be an astra-native-reset-* direct child of the approved results directory')
    return str(path)


def validate_local(value):
    path = Path(value).expanduser().resolve()
    if path.drive.upper() != 'D:' or path == Path(path.anchor) or not path.is_absolute():
        raise ValueError('--local-output must be an absolute folder on D:')
    return path


def redact(value):
    if isinstance(value, dict):
        sensitive = {'api_key', 'apikey', 'access_token', 'refresh_token', 'password', 'authorization', 'client_secret'}
        return {key: '[REDACTED]' if str(key).lower() in sensitive else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b', '[REDACTED]', value)
    value = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+', r'\1[REDACTED]', value)
    value = re.sub(r'(?i)((?:api[_-]?key|password|access[_-]?token|refresh[_-]?token|client[_-]?secret)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', value)
    value = re.sub(r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----',
                   '[REDACTED PRIVATE KEY]', value)
    return value


def save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append(path, data):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(redact(data), ensure_ascii=False, allow_nan=False) + '\n')


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def payload(snapshot, name):
    return snapshot.get('files', {}).get(name, {}).get('data', {})


def cleanup_status(snapshot):
    if (payload(snapshot, 'cleanup-error.json')
            or payload(snapshot, 'campaign-error.json').get('cleanup_error')
            or payload(snapshot, 'campaign-finished.json').get('cleanup_error')):
        return 'recorded_error'
    if any(name in snapshot.get('files', {}) for name in ('cleanup.json', 'campaign-cleanup.json')):
        return 'record_present_not_independently_verified'
    return 'pending_or_not_requested'


def terminal_reason(snapshot):
    if snapshot.get('driver', {}).get('active') is True:
        return None
    completion = payload(snapshot, 'completion.json')
    files = snapshot.get('files', {})
    if 'campaign-owner.json' in files or 'campaign-finished.json' in files:
        # The outer supervisor may still be preparing, analyzing or cleaning up
        # after an inner runner ends. Its finished record is a separate contract.
        if snapshot.get('campaign_driver', {}).get('active') is not False:
            return None
        finished = payload(snapshot, 'campaign-finished.json')
        owner = payload(snapshot, 'campaign-owner.json')
        if (not isinstance(finished, dict)
                or finished.get('schema') not in ('astra_full_campaign_finished_v1',
                                                  'astra_continuation_campaign_finished_v2')
                or finished.get('stage') not in ('completed', 'drained', 'incomplete')
                or type(finished.get('complete')) is not bool
                or finished.get('owner') != owner):
            return None
        if finished['schema'] == 'astra_continuation_campaign_finished_v2':
            if (type(finished.get('attempt_coverage_complete')) is not bool
                    or (finished['complete'] and finished['stage'] != 'completed')):
                return None
            report = payload(snapshot, 'analysis/report.json')
            coverage = (finished['stage'] == 'completed'
                        and finished['attempt_coverage_complete'] is True
                        and finished.get('planned_episodes') == 1936
                        and completion.get('all_children_drained') is True
                        and completion.get('attempt_coverage_complete') is True
                        and completion.get('episodes') == 1936
                        and report.get('attempt_coverage_complete') is True
                        and report.get('planned_episodes') == 1936)
            if coverage:
                if (finished['complete'] is True and completion.get('complete') is True
                        and report.get('complete') is True):
                    return 'completed'
                return 'completed_all_attempts_with_unknowns'
            return 'campaign_finished_incomplete'
        if finished['complete'] != (finished['stage'] == 'completed'):
            return None
        if (finished['complete'] is True and completion.get('all_children_drained') is True
                and completion.get('complete') is True
                and payload(snapshot, 'analysis/report.json').get('complete') is True):
            return 'completed'
        return 'campaign_finished_incomplete'
    if completion.get('all_children_drained') is True and type(completion.get('complete')) is bool:
        return 'completed' if completion['complete'] else 'stopped_incomplete'
    if snapshot.get('driver', {}).get('active') is False:
        if payload(snapshot, 'supervisor-error.json') or payload(snapshot, 'failure.json'):
            return 'recorded_failure'
        progress = payload(snapshot, 'progress.json')
        if progress.get('stop_reasons') and progress.get('active') == 0:
            return 'stopped_without_completion'
    return None


class SSHReader:
    def __init__(self, remote_output):
        self.remote_output = validate_remote(remote_output)
        self.ssh = r'C:\Windows\System32\OpenSSH\ssh.exe' if os.name == 'nt' else 'ssh'

    def command(self, mode):
        remote = 'python3 - ' + shlex.quote(self.remote_output) + ' ' + shlex.quote(mode)
        return [self.ssh, *SSH_OPTIONS, HOST, remote]

    def status(self, timeout):
        result = subprocess.run(self.command('status'), input=REMOTE_PROGRAM.encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RuntimeError('SSH status read failed (exit %d): %s' %
                               (result.returncode, redact(result.stderr.decode('utf-8', 'replace')[-1500:])))
        snapshot = json.loads(result.stdout)
        if snapshot.get('schema') != 'astra_campaign_remote_snapshot_v1':
            raise ValueError('Unexpected remote snapshot schema')
        return snapshot

    def archive(self, destination, timeout):
        with destination.open('xb') as stream:
            result = subprocess.run(self.command('archive'), input=REMOTE_PROGRAM.encode(), stdout=stream,
                                    stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RuntimeError('SSH evidence read failed (exit %d): %s' %
                               (result.returncode, redact(result.stderr.decode('utf-8', 'replace')[-1500:])))


def verify_archive(path, remote_output):
    seen = {}
    index = snapshot = None
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or str(name) != member.name or member.name in seen:
                raise ValueError('Unexpected evidence archive member')
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            raw = bytearray() if member.name in (EVIDENCE_INDEX, SNAPSHOT_MEMBER) else None
            count = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
                count += len(chunk)
                if raw is not None:
                    raw.extend(chunk)
            seen[member.name] = {'path': member.name, 'bytes': count, 'sha256': digest.hexdigest()}
            if member.name == EVIDENCE_INDEX:
                index = json.loads(raw)
            elif member.name == SNAPSHOT_MEMBER:
                snapshot = json.loads(raw)
    if not index or index.get('remote_output') != remote_output or not snapshot:
        raise ValueError('Evidence index/root/snapshot missing or different')
    expected = {item['path']: item for item in index['files']}
    if len(expected) != len(index['files']) or expected != {name: item for name, item in seen.items() if name != EVIDENCE_INDEX}:
        raise ValueError('Evidence file inventory or SHA-256 mismatch')
    return index, snapshot


def acquire_evidence(reader, local, timeout, reason):
    archive = local / 'study-evidence.tar.gz'
    receipt_path = local / 'evidence_receipt.json'
    if archive.exists() or receipt_path.exists():
        if not archive.exists() or not receipt_path.exists():
            raise RuntimeError('Existing unpaired evidence file/receipt retained; inspect before synchronization')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('remote_output') != reader.remote_output or receipt.get('archive_sha256') != file_sha(archive):
            raise RuntimeError('Existing evidence receipt differs; preserved without replacement')
        return receipt
    partial = local / ('study-evidence-' + uuid.uuid4().hex + '.partial.tar.gz')
    reader.archive(partial, timeout)
    index, snapshot = verify_archive(partial, reader.remote_output)
    receipt = dict(schema='astra_campaign_local_evidence_receipt_v1', collected_utc=utc(),
                   remote_output=reader.remote_output, archive_path=str(archive), archive_bytes=partial.stat().st_size,
                   archive_sha256=file_sha(partial), evidence_files=len(index['files']),
                   uncompressed_bytes=sum(item['bytes'] for item in index['files']),
                   individual_file_hashes_verified=True, remote_writes=False,
                   terminal_observation=reason, cleanup_status=cleanup_status(snapshot),
                   cleanup_independently_verified=False, skipped_symlinks=index.get('skipped_symlinks', []),
                   exclusions=index.get('exclusions', []), publicly_uploaded=False)
    partial.replace(archive)
    save(receipt_path, receipt)
    return receipt


def monitor(reader, local, run_seconds, *, clock=time.monotonic, sleeper=time.sleep):
    local.mkdir(parents=True, exist_ok=True)
    started = clock()
    deadline = started + run_seconds
    failures = 0
    archive_attempts = 0
    stop_reason = 'local_time_limit'
    terminal_seen = None
    last_snapshot = None
    while clock() < deadline:
        event = {'observed_client_utc': utc(), 'remote_output': reader.remote_output}
        try:
            snapshot = reader.status(min(45, max(1, deadline - clock())))
            last_snapshot = snapshot
            reason = terminal_reason(snapshot)
            event.update(sync_ok=True, snapshot=snapshot, terminal_observation=reason,
                         cleanup_status=cleanup_status(snapshot), cleanup_independently_verified=False)
            append(local / 'sync_status.jsonl', event)
            save(local / 'sync_snapshot.json', event)
            failures = 0
            # Two successful observations, at least one normal poll apart, allow
            # analysis/cleanup records to follow completion without a partial tar.
            settled = reason is not None and reason == terminal_seen
            terminal_seen = reason
            if settled and ('analysis/report.json' in snapshot.get('files', {}) or reason != 'completed'):
                archive_attempts += 1
                receipt = acquire_evidence(reader, local, min(900, max(1, deadline - clock())), reason)
                end = dict(status='evidence_collected', terminal_observation=reason, remote_complete=reason == 'completed',
                           observed_client_utc=utc(), cleanup_status=receipt['cleanup_status'],
                           cleanup_independently_verified=False, publicly_uploaded=False)
                save(local / 'sync_finished.json', end)
                print(json.dumps(end), flush=True)
                return 0
            progress = payload(snapshot, 'progress.json')
            print(json.dumps(dict(sync_ok=True, terminal_observation=reason,
                                  completed=progress.get('completed'), active=progress.get('active'),
                                  pending=progress.get('pending'), cleanup_status=cleanup_status(snapshot))), flush=True)
            delay = POLL_SECONDS
        except Exception as error:
            failures += 1
            terminal_seen = None
            event = dict(observed_client_utc=utc(), sync_ok=False, consecutive_failures=failures,
                         error_type=type(error).__name__, error=redact(str(error)),
                         archive_attempts=archive_attempts, remote_output=reader.remote_output,
                         remote_completion_unknown=True)
            append(local / 'sync_status.jsonl', event)
            save(local / 'sync_error.json', event)
            print(json.dumps({key: event[key] for key in ('sync_ok', 'consecutive_failures', 'error_type')}), flush=True)
            if archive_attempts >= 3:
                stop_reason = 'evidence_transfer_limit'
                break
            if failures >= MAX_FAILURES:
                stop_reason = 'read_failure_limit'
                break
            delay = min(300, POLL_SECONDS * (2 ** (failures - 1)))
        remaining = deadline - clock()
        if remaining > 0:
            sleeper(min(delay, remaining))
    end = dict(status='synchronization_stopped_incomplete', reason=stop_reason,
               observed_client_utc=utc(), remote_complete=None, campaign_failure_inferred=False,
               last_remote_snapshot_utc=last_snapshot.get('remote_unix') if last_snapshot else None,
               cleanup_independently_verified=False, publicly_uploaded=False)
    save(local / 'sync_finished.json', end)
    print(json.dumps(end), flush=True)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-output', required=True)
    parser.add_argument('--local-output', required=True)
    parser.add_argument('--run-seconds', required=True, type=int)
    args = parser.parse_args(argv)
    remote = validate_remote(args.remote_output)
    local = validate_local(args.local_output)
    if not 0 < args.run_seconds <= 7 * 24 * 3600:
        parser.error('--run-seconds must be positive and at most seven days')
    raise SystemExit(monitor(SSHReader(remote), local, args.run_seconds))


if __name__ == '__main__':
    main()
