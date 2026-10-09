"""Free in the owning OSMesa context; restore the interrupted surviving context."""
import ctypes
import hashlib
from pathlib import Path
import weakref

BINDING_SHA = 'a3d2526fb2e60875e2c3a87b8828b8e3b512781708c80d0c96ee6024980896c8'
CONTEXT_SHA = '3f3dd69f62d0362a5832387f5171ac56bdf0fc0c4e33e862c1bb5086fe0a26c0'

def install():
    from robosuite.utils import binding_utils as b
    import mujoco.osmesa as context_module
    from OpenGL import osmesa
    if hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest() != BINDING_SHA or hashlib.sha256(Path(context_module.__file__).read_bytes()).hexdigest() != CONTEXT_SHA:
        raise ValueError('Unreviewed OSMesa binding/context')
    if b._MUJOCO_GL != 'osmesa':
        raise ValueError('Require OSMesa backend')
    pointer = lambda x: int(ctypes.cast(x, ctypes.c_void_p).value or 0)
    if pointer(osmesa.OSMesaGetCurrentContext()):
        raise ValueError('Install OSMesa lifetime hook before any contexts are created')
    contexts = weakref.WeakValueDictionary()
    original_init = context_module.GLContext.__init__
    def register(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        contexts[pointer(self._context)] = self
    context_module.GLContext.__init__ = register
    original_free = b.MjRenderContext.__del__
    b._osmesa_lifetime_failure = None
    def release_owned_context(context):
        previous = pointer(osmesa.OSMesaGetCurrentContext())
        owned = context.gl_ctx
        restore = previous != 0 and previous != pointer(owned._context)
        surviving = contexts.get(previous) if restore else None
        try:
            if restore and surviving is None:
                raise RuntimeError('Surviving OSMesa context was not registered')
            owned.make_current()
            try:
                return original_free(context)
            finally:
                if surviving is not None:
                    surviving.make_current()
                    if pointer(osmesa.OSMesaGetCurrentContext()) != previous:
                        raise RuntimeError('OSMesa context restoration failed')
        except BaseException as error:
            b._osmesa_lifetime_failure = repr(error)
            raise
    b.MjRenderContext.__del__ = release_owned_context
    return {'patch': 'owning OSMesa context destruction with surviving-context restoration',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
