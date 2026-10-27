"""Map EGL device indices (what MUJOCO_EGL_DEVICE_ID selects) to CUDA device numbers and PCI bus ids."""
import ctypes as C, ctypes.util, json, os, subprocess
egl = C.CDLL(ctypes.util.find_library('EGL') or 'libEGL.so.1')
egl.eglGetProcAddress.argtypes = [C.c_char_p]; egl.eglGetProcAddress.restype = C.c_void_p
def fn(name, restype, *types):
    a = egl.eglGetProcAddress(name.encode()); assert a, name
    return C.CFUNCTYPE(restype, *types)(a)
query = fn('eglQueryDevicesEXT', C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))
attrib = fn('eglQueryDeviceAttribEXT', C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t))
devstr = fn('eglQueryDeviceStringEXT', C.c_char_p, C.c_void_p, C.c_int)
devs = (C.c_void_p * 16)(); n = C.c_int(); query(16, devs, C.byref(n))
rows = []
for i in range(n.value):
    ext = (devstr(devs[i], 0x3055) or b'').decode()
    v = C.c_ssize_t(-1); ok = attrib(devs[i], 0x323A, C.byref(v))  # EGL_CUDA_DEVICE_NV
    drm = (devstr(devs[i], 0x3233) or b'').decode()  # EGL_DRM_DEVICE_FILE_EXT
    pci = None
    if drm:
        card = os.path.basename(drm)
        p = f'/sys/class/drm/{card}/device'
        if os.path.exists(p):
            pci = os.path.basename(os.path.realpath(p))
    rows.append({'egl_index': i, 'cuda_device_attr': v.value if ok else None, 'drm': drm, 'pci': pci,
                 'nvidia': 'EGL_NV_device_cuda' in ext})
smi = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], capture_output=True, text=True).stdout
print(json.dumps({'egl_devices': rows, 'nvidia_smi': smi.strip().splitlines()}, indent=1))
