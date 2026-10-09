"""Process-local private OSMesa RGB guard; never changes images, actions, or RNG.

Install before gym.make; call check() after each official evaluator returns,
before publishing a result, and close() finally. Integrity errors are sticky
infrastructure errors, never task failures. Known material draw1281 is counted.
Candidate only: uniform RGB requires independent same-frame confirmation;
uniform content alone is not evidence of a readback failure.
"""
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
import time

BINDING_SHA = 'a3d2526fb2e60875e2c3a87b8828b8e3b512781708c80d0c96ee6024980896c8'


class ReadbackIntegrityError(RuntimeError):
    """Infrastructure failure: do not publish a normal unsuccessful episode."""


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class ReadbackGuard:
    """Injectable implementation; install_guard supplies the actual native API."""

    def __init__(self, output, *, original_render, original_read, read_again,
                 gl_errors, interval=100, context_check=None):
        if type(interval) is not int or interval != 100:
            raise ValueError('This fixed guard checks first and every 100th raw frame')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.original_render, self.original_read = original_render, original_read
        self.read_again, self.gl_errors, self.interval = read_again, gl_errors, interval
        self.context_check = context_check
        self.context_identities = {}
        self.frames = self.comparisons = 0
        self.errors = Counter()
        self.failure = None
        self.started = time.monotonic()
        self.restorations = []
        self.closed = False
        self._events = (self.output / 'events.jsonl').open('x')
        self._save()

    def _save(self):
        _write_json(self.output / 'summary.json', dict(
            status='failed' if self.failure else ('closed' if self.closed else 'active'),
            raw_frames_checked=self.frames, sentinel_checks=self.comparisons,
            backend='osmesa', context_identities=self.context_identities,
            interval=self.interval, gl_error_counts=dict(self.errors), failure=self.failure,
            wall_seconds=time.monotonic()-self.started, infrastructure_guard=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def _event(self, kind, **values):
        self._events.write(json.dumps(dict(kind=kind, raw_frame=self.frames,
            wall_seconds=time.monotonic()-self.started, **values), allow_nan=False)+'\n')
        self._events.flush()

    def check(self):
        if self.failure:
            raise ReadbackIntegrityError(self.failure['reason'])
        if self.closed:
            raise ReadbackIntegrityError('Readback guard already closed')

    def _fail(self, reason, **arrays):
        import numpy as np
        # Latch first: disk exhaustion must never turn a caught render fault into a score.
        self.failure = dict(reason=reason, frame=self.frames, evidence=None)
        if arrays:
            evidence = f'failure-{self.frames:08d}.npz'
            try:
                np.savez_compressed(self.output / evidence, **arrays)
                self.failure['evidence'] = evidence
            except Exception as error:
                self.failure['evidence_write_error'] = repr(error)
        try:
            self._event('integrity_failure', **self.failure)
            self._save()
        except Exception as error:
            self.failure['metadata_write_error'] = repr(error)
        raise ReadbackIntegrityError(reason)

    def _phase(self, phase):
        try:
            if self.context_check is not None:
                identity = self.context_check(phase)
                if identity is not None:
                    self.context_identities[str(identity['context'])] = identity
            errors = list(self.gl_errors())
        except Exception as error:
            self._fail(f'GL diagnostic raised {type(error).__name__}: {error}')
        for error in errors:
            self.errors[f'{phase}:{int(error)}'] += 1
        unexpected = [int(x) for x in errors if not (phase == 'after_draw' and int(x) == 1281)]
        if unexpected:
            self._fail(f'Unexpected GL error in {phase}: {unexpected}')

    def render(self, context, *args, **kwargs):
        self.check()
        self._phase('before_draw')
        try:
            result = self.original_render(context, *args, **kwargs)
        except Exception as error:
            self._fail(f'Native render raised {type(error).__name__}: {error}')
        self._phase('after_draw')
        return result

    def read(self, context, width, height, depth=False, segmentation=False):
        import numpy as np
        self.check()
        if depth or segmentation:
            self._fail('Guard only supports original RGB camera reads')
        self._phase('before_read')
        try:
            result = self.original_read(context, width, height, depth=depth, segmentation=segmentation)
        except Exception as error:
            self._fail(f'Native read raised {type(error).__name__}: {error}')
        self._phase('after_read')
        self.frames += 1
        if (not isinstance(result, np.ndarray) or result.dtype != np.uint8
                or result.shape != (256, 256, 3) or (width, height) != (256, 256)):
            self._fail('Unexpected RGB shape or dtype', original=np.asarray(result))
        # A legitimately occluded/background-only camera may be spatially uniform.
        # Confirm every uniform frame with both native reads; never alter pixels.
        uniform = not np.any(result != result[0, 0])
        if uniform or self.frames == 1 or self.frames % self.interval == 0:
            a = np.full(result.shape, 17, dtype=np.uint8)
            b = np.full(result.shape, 239, dtype=np.uint8)
            distinct = not any(np.shares_memory(left, right) for left, right in
                               ((result, a), (result, b), (a, b)))
            if not distinct:
                self._fail('Same-frame RGB buffers are not independent',
                           original=result, sentinel17=a, sentinel239=b)
            for buffer, label in ((a, 'sentinel17'), (b, 'sentinel239')):
                self._phase('before_' + label)
                try:
                    self.read_again(context, width, height, buffer)
                except Exception as error:
                    self._fail(f'Sentinel read raised {type(error).__name__}: {error}',
                               original=result, sentinel17=a, sentinel239=b)
                self._phase('after_' + label)
            self.comparisons += 1
            equal = distinct and np.array_equal(result, a) and np.array_equal(result, b)
            self._event('same_frame_sentinel', equal=bool(equal), distinct=distinct, uniform_rgb=bool(uniform))
            if not equal:
                self._fail('Same-frame RGB readback differs across independent buffers',
                           original=result, sentinel17=a, sentinel239=b)
            self._save()
        return result

    def close(self):
        if self.closed:
            return
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)
        self.restorations.clear()
        self.closed = True
        self._save()
        self._events.close()


def install_guard(output_dir, interval=100, *, library_path):
    """Original full RGB rendering through the already inspected private OSMesa."""
    import os
    required = dict(MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', CUDA_VISIBLE_DEVICES='',
        LIBGL_ALWAYS_SOFTWARE='true', GALLIUM_DRIVER='llvmpipe', MESA_DEBUG='1', LP_NUM_THREADS='4')
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError('Require the previously measured private OSMesa environment')
    for name in ('MESA_NO_ERROR', 'MESA_GL_VERSION_OVERRIDE', 'MESA_GLSL_VERSION_OVERRIDE',
                 'MESA_EXTENSION_OVERRIDE', 'LP_NO_RAST', 'LP_PERF', 'GALLIUM_HUD'):
        if os.environ.get(name):
            raise ValueError('Unset renderer-altering override: '+name)
    library = Path(library_path).resolve(strict=True)
    if hashlib.sha256(library.read_bytes()).hexdigest() != '40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a':
        raise ValueError('Private OSMesa library differs from inspected binary')
    native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    native.OSMesaGetCurrentContext.argtypes = []
    native.OSMesaGetCurrentContext.restype = ctypes.c_void_p
    native.OSMesaGetProcAddress.argtypes = [ctypes.c_char_p]
    native.OSMesaGetProcAddress.restype = ctypes.c_void_p
    def function(name, result, *parameters):
        address = native.OSMesaGetProcAddress(name.encode())
        if not address:
            raise ValueError('Native OSMesa GL function missing: '+name)
        return ctypes.CFUNCTYPE(result, *parameters)(address)
    get_error = function('glGetError', ctypes.c_uint)
    get_string = function('glGetString', ctypes.c_char_p, ctypes.c_uint)
    import mujoco
    import mujoco.osmesa as context_module
    from robosuite.utils import binding_utils
    if (mujoco.__version__ != '3.3.1'
        or hashlib.sha256(Path(binding_utils.__file__).read_bytes()).hexdigest() != BINDING_SHA
        or hashlib.sha256(Path(context_module.__file__).read_bytes()).hexdigest() != '3f3dd69f62d0362a5832387f5171ac56bdf0fc0c4e33e862c1bb5086fe0a26c0'
        or binding_utils._MUJOCO_GL != 'osmesa'):
        raise ValueError('Unreviewed MuJoCo/OSMesa source')
    owner = binding_utils.MjRenderContext
    if getattr(owner, '_readback_guard_v1_installed', False):
        raise ValueError('Guard already installed')
    identities = {}
    def context_check(phase):
        failure = getattr(binding_utils, '_osmesa_lifetime_failure', None)
        if failure:
            raise ReadbackIntegrityError('OSMesa context-lifetime failure: '+failure)
        current = int(native.OSMesaGetCurrentContext() or 0)
        if not current:
            if phase == 'before_draw':
                return None
            raise ReadbackIntegrityError('No current OSMesa context at '+phase)
        if current not in identities:
            vendor, renderer, version = [(get_string(k) or b'').decode() for k in (0x1F00,0x1F01,0x1F02)]
            if 'llvmpipe' not in renderer.lower():
                raise ReadbackIntegrityError('Require the inspected llvmpipe renderer')
            identities[current] = dict(context=current,vendor=vendor,renderer=renderer,version=version,
                library_sha256='40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a')
        return identities[current]
    def errors():
        if not native.OSMesaGetCurrentContext():
            return []
        values = []
        for _ in range(32):
            value = int(get_error())
            if not value:
                return values
            values.append(value)
        return values+[-1]
    def read_again(context,width,height,buffer):
        mujoco.mjr_readPixels(rgb=buffer,depth=None,viewport=mujoco.MjrRect(0,0,width,height),con=context.con)
    guard = ReadbackGuard(output_dir,original_render=owner.render,original_read=owner.read_pixels,
        read_again=read_again,gl_errors=errors,interval=interval,context_check=context_check)
    def render(context,*args,**kwargs):
        return guard.render(context,*args,**kwargs)
    def read(context,width,height,depth=False,segmentation=False):
        return guard.read(context,width,height,depth=depth,segmentation=segmentation)
    guard.restorations = [(owner,'render',owner.render),(owner,'read_pixels',owner.read_pixels),
        (owner,'_readback_guard_v1_installed',False)]
    owner.render,owner.read_pixels,owner._readback_guard_v1_installed = render,read,True
    return guard

