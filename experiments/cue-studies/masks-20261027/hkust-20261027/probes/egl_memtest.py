"""Allocate a 1 GiB texture on EGL device argv[1] and report which nvidia-smi GPU's memory grew."""
import ctypes as C, ctypes.util, os, subprocess, sys, time
idx = int(sys.argv[1])
def smi():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(',') for l in out.strip().splitlines())}
egl = C.CDLL(ctypes.util.find_library('EGL') or 'libEGL.so.1')
egl.eglGetProcAddress.argtypes = [C.c_char_p]; egl.eglGetProcAddress.restype = C.c_void_p
def fn(name, restype, *types):
    a = egl.eglGetProcAddress(name.encode()); assert a, name
    return C.CFUNCTYPE(restype, *types)(a)
query = fn('eglQueryDevicesEXT', C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))
platform = fn('eglGetPlatformDisplayEXT', C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_int))
initialize = fn('eglInitialize', C.c_uint, C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int))
choose = fn('eglChooseConfig', C.c_uint, C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_int, C.POINTER(C.c_int))
bind = fn('eglBindAPI', C.c_uint, C.c_uint)
surface = fn('eglCreatePbufferSurface', C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int))
context = fn('eglCreateContext', C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int))
current = fn('eglMakeCurrent', C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p)
getstring = fn('glGetString', C.c_char_p, C.c_uint)
gen = fn('glGenTextures', None, C.c_int, C.POINTER(C.c_uint)); bindtex = fn('glBindTexture', None, C.c_uint, C.c_uint)
teximg = fn('glTexImage2D', None, C.c_uint, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint, C.c_uint, C.c_void_p)
finish = fn('glFinish', None); geterr = fn('glGetError', C.c_uint)
devs = (C.c_void_p * 16)(); n = C.c_int(); query(16, devs, C.byref(n))
print('device_count', n.value, 'CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
base = smi()
display = platform(0x313F, devs[idx], None)
major, minor = C.c_int(), C.c_int(); assert initialize(display, C.byref(major), C.byref(minor))
conf, nconf = C.c_void_p(), C.c_int()
attrs = (C.c_int * 13)(0x3033, 0x0001, 0x3040, 0x0008, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3025, 24, 0x3038)
assert choose(display, attrs, C.byref(conf), 1, C.byref(nconf)) and nconf.value >= 1
bind(0x30A2)
surf = surface(display, conf, (C.c_int * 5)(0x3057, 64, 0x3056, 64, 0x3038))
ctx = context(display, conf, None, (C.c_int * 1)(0x3038))
assert current(display, surf, surf, ctx)
t = C.c_uint(); gen(1, C.byref(t)); bindtex(0x0DE1, t.value)
teximg(0x0DE1, 0, 0x8058, 16384, 16384, 0, 0x1908, 0x1401, None)  # GL_RGBA8 16384^2 = 1 GiB
finish(); print('gl_error', hex(geterr()), 'renderer', (getstring(0x1F01) or b'').decode())
time.sleep(1.5)
after = smi()
delta = {k: after[k] - base[k] for k in base}
print('delta_MiB', delta, '-> egl', idx, 'renders on GPU', max(delta, key=delta.get))
