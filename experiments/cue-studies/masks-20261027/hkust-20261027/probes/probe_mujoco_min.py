"""Minimal MuJoCo-only depth readback probe (no robosuite): does pixel (0,1) of the depth
readback come back invalid on this host?  Tries single-sample and MSAA offscreen buffers,
depth-only reads, a viewport offset by one pixel, and a direct glReadPixels."""
import json
import os
import sys
import numpy as np
import mujoco
from mujoco.egl import GLContext

XML = """<mujoco><visual><global offwidth="640" offheight="480"/><quality offsamples="%d"/></visual>
<worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/><geom type="box" size=".1 .1 .1" pos="0 0 .1"/>
<camera name="c" pos="1 1 1" xyaxes="-1 1 0 -1 -1 2"/></worldbody></mujoco>"""


def bad(z):
    return ~np.isfinite(z) | (z < 0) | (z > 1)


def run(samples):
    m = mujoco.MjModel.from_xml_string(XML % samples)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctx = GLContext(640, 480)
    ctx.make_current()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    scn = mujoco.MjvScene(m, 100)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = 0
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    rows = []
    for (w, h) in [(640, 480), (256, 256), (128, 128)]:
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, d, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, con)
        rgb = np.empty((h, w, 3), np.uint8)
        z = np.full((h, w), -5.0, np.float32)
        mujoco.mjr_readPixels(rgb, z, vp, con)
        b = bad(z)
        row = {'samples': samples, 'size': [w, h], 'invalid': int(b.sum()), 'at': np.argwhere(b)[:3].tolist(),
               'vals': [float(v) for v in z[b][:3]], 'z0_first4': z[0, :4].tolist(), 'z1_first4': z[1, :4].tolist(),
               'untouched_prefill': int((z == -5.0).sum())}
        z2 = np.full((h, w), -5.0, np.float32)
        mujoco.mjr_readPixels(None, z2, vp, con)
        row['depth_only_invalid'] = int(bad(z2).sum())
        row['depth_only_z0_first4'] = z2[0, :4].tolist()
        vp1 = mujoco.MjrRect(1, 0, w - 1, h)
        z3 = np.full((h, w - 1), -5.0, np.float32)
        rgb3 = np.empty((h, w - 1, 3), np.uint8)
        mujoco.mjr_readPixels(rgb3, z3, vp1, con)
        b3 = bad(z3)
        row['offset1_invalid'] = int(b3.sum())
        row['offset1_at'] = np.argwhere(b3)[:3].tolist()
        vp2 = mujoco.MjrRect(0, 1, w, h - 1)
        z4 = np.full((h - 1, w), -5.0, np.float32)
        rgb4 = np.empty((h - 1, w, 3), np.uint8)
        mujoco.mjr_readPixels(rgb4, z4, vp2, con)
        b4 = bad(z4)
        row['bottom_offset1_invalid'] = int(b4.sum())
        row['bottom_offset1_at'] = np.argwhere(b4)[:3].tolist()
        try:
            from OpenGL import GL
            raw_u = np.empty((h, w), np.uint32)
            GL.glReadPixels(0, 0, w, h, GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_INT, raw_u)
            raw_f = np.empty((h, w), np.float32)
            GL.glReadPixels(0, 0, w, h, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, raw_f)
            row['gl_direct'] = {'uint_first4': [int(v) for v in raw_u[0, :4]], 'float_first4': raw_f[0, :4].tolist(),
                                'float_invalid': int(bad(raw_f).sum()), 'gl_error': int(GL.glGetError()),
                                'read_fbo': int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING)),
                                'pack_alignment': int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT)),
                                'pack_row_length': int(GL.glGetIntegerv(GL.GL_PACK_ROW_LENGTH)),
                                'pack_skip_pixels': int(GL.glGetIntegerv(GL.GL_PACK_SKIP_PIXELS)),
                                'pack_skip_rows': int(GL.glGetIntegerv(GL.GL_PACK_SKIP_ROWS))}
        except Exception as exc:  # noqa: BLE001
            row['gl_direct_error'] = repr(exc)
        rows.append(row)
    con.free()
    ctx.free()
    return rows


out = {'mujoco': mujoco.__version__, 'MUJOCO_EGL_DEVICE_ID': os.environ.get('MUJOCO_EGL_DEVICE_ID'),
       'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'python': sys.version.split()[0]}
out['single_sample'] = run(0)
out['msaa4'] = run(4)
print(json.dumps(out, indent=1))
