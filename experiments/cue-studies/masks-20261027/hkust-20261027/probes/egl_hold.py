"""Create a GL context on EGL device N (argv[1]) and report which nvidia-smi GPU shows this PID."""
import ctypes as C, ctypes.util, os, subprocess, sys, time
idx = int(sys.argv[1])
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
devs = (C.c_void_p * 16)(); n = C.c_int(); query(16, devs, C.byref(n))
display = platform(0x313F, devs[idx], None)
major, minor = C.c_int(), C.c_int(); assert initialize(display, C.byref(major), C.byref(minor))
conf, nconf = C.c_void_p(), C.c_int()
attrs = (C.c_int * 13)(0x3033, 0x0001, 0x3040, 0x0008, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3025, 24, 0x3038)
assert choose(display, attrs, C.byref(conf), 1, C.byref(nconf)) and nconf.value >= 1
bind(0x30A2)
surf = surface(display, conf, (C.c_int * 5)(0x3057, 64, 0x3056, 64, 0x3038))
ctx = context(display, conf, None, (C.c_int * 1)(0x3038))
assert current(display, surf, surf, ctx)
print('renderer', (getstring(0x1F01) or b'').decode(), 'pid', os.getpid())
time.sleep(4)
out = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
print([l for l in out.splitlines() if str(os.getpid()) in l])
