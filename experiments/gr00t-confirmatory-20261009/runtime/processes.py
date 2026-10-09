"""Parent-owned Linux child handles, including the interval before argv is ready.

Construction has no OS reads and must immediately follow Popen registration.
Only an unreaped Popen child may acquire a pidfd. Signals never depend on a
partially populated cmdline or a process-name match.
"""
from pathlib import Path
import os
import signal
import time
import subprocess


def require(value,message):
    if not value:raise RuntimeError(message)


def birth(pid):
    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,parent_pid=int(fields[1]),process_start_ticks=int(fields[19]),state=fields[0])


def identity(pid):
    p=Path('/proc')/str(pid);b=birth(pid)
    return dict(pid=pid,process_start_ticks=b['process_start_ticks'],
        command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str((p/'cwd').resolve(strict=True)))


def open_pidfd(pid):
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd support required')
    return os.pidfd_open(pid)


def send_pidfd(fd,sig):signal.pidfd_send_signal(fd,sig)


def close_pidfd(fd):os.close(fd)


class TrackedChild:
    def __init__(self,child,command,cwd):
        self.child=child;self.command=list(command);self.cwd=str(cwd);self.pid=child.pid
        self.parent_pid=os.getpid();self.start=None;self.pidfd=None;self.owner=None
        self.samples=[];self.signals=[];self.reaped=False;self.startup_error=None

    def _birth(self):
        current=birth(self.pid)
        require(current['pid']==self.pid and current['parent_pid']==self.parent_pid,
                'Spawn is no longer the direct unreaped child; no signal permitted')
        if self.start is None:self.start=current['process_start_ticks']
        require(current['process_start_ticks']==self.start,'Spawn start ticks changed; no signal permitted')
        return current

    def _attach(self):
        if self.child.poll() is not None:return False
        if self.pidfd is None:
            # Popen has not reaped this still-live direct child, preventing PID
            # reuse while the kernel handle is acquired and its birth checked.
            self.pidfd=open_pidfd(self.pid)
        self._birth()
        return True

    def await_identity(self,timeout=5.0,interval=.02):
        deadline=time.monotonic()+timeout;previous=None;matches=0
        try:
            if not self._attach():raise RuntimeError('Child exited before startup identity binding')
            while time.monotonic()<deadline:
                if self.child.poll() is not None:raise RuntimeError('Child exited before startup identity binding')
                self._birth()
                try:
                    observed=identity(self.pid);self._birth()
                    valid=(observed['pid']==self.pid and observed['process_start_ticks']==self.start
                        and observed['command']==self.command and observed['cwd']==self.cwd)
                    self.samples.append(dict(observed=observed,expected_identity=valid))
                    self.samples=self.samples[-16:]
                    matches=matches+1 if valid and previous==observed else 1 if valid else 0
                    previous=observed
                    if matches>=2:self.owner=observed;return observed
                except OSError as error:
                    matches=0;previous=None
                    self.samples.append(dict(read_error=str(error)));self.samples=self.samples[-16:]
                time.sleep(interval)
            raise TimeoutError('Bounded child identity handshake expired; pending spawn must be drained')
        except BaseException as error:
            self.startup_error=repr(error)
            raise

    def send(self,sig):
        if not self._attach():return False
        self._birth()
        send_pidfd(self.pidfd,sig)
        self.signals.append(dict(signal=int(sig),pidfd=True,bound_owner=self.owner is not None))
        return True

    def drain(self,grace=15,kill_timeout=10):
        """Wait/reap every spawn, even if the complete identity was never bound."""
        try:
            if self.child.poll() is None:self.send(signal.SIGTERM)
            try:code=self.child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL);code=self.child.wait(timeout=kill_timeout)
            self.reaped=True
            return code
        finally:
            if self.child.returncode is not None:self.close()

    def close(self):
        require(self.child.poll() is not None,'Cannot release a live tracked child')
        self.child.wait(timeout=0);self.reaped=True
        if self.pidfd is not None:close_pidfd(self.pidfd);self.pidfd=None

    def receipt(self):
        return dict(schema='originx_gr00t_spawn_handshake_v2',pid=self.pid,parent_pid=self.parent_pid,
            process_start_ticks=self.start,requested_command=self.command,requested_cwd=self.cwd,
            owner=self.owner,identity_bound=self.owner is not None,samples=self.samples,
            signals=self.signals,reaped=self.reaped,returncode=self.child.returncode,startup_error=self.startup_error)
