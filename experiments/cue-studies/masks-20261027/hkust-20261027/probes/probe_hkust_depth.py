"""Read-only probe on the HKUST site: raw depth from the dedicated single-sample mask
context vs the policy context, repeated, with invalid-pixel locations."""
import json
import sys
from pathlib import Path

ROOT = Path('/ephemeral/qinzhen/originx-harness-v2')
sys.path.insert(0, str(ROOT))
import numpy as np
from originx_harness.id60_masks_validation.site_rendering import make_environment
from originx_harness.id60_masks_validation.geometry import SingleSampleMaskRenderer

site = json.loads((ROOT / 'work/hkust/site-t3.json').read_text())
env = make_environment('CoffeeSetupMug', 600, site)
raw = env.reset(20261025, 'probe-depth')
sim = env.extractor.sim
_, h, w, _ = raw['rgb'].shape
cams = list(env.camera_names)


def stats(z):
    z = np.asarray(z, dtype=np.float64)
    bad = ~np.isfinite(z) | (z < 0) | (z > 1)
    r = {'invalid': int(bad.sum()), 'min': float(np.nanmin(z)), 'max': float(np.nanmax(z)), 'eq1': int((z == 1).sum())}
    if bad.any():
        ys, xs = np.nonzero(bad)
        r.update(rows=[int(ys.min()), int(ys.max())], cols=[int(xs.min()), int(xs.max())],
                 sample_values=[float(v) for v in z[bad][:6]], n_rows_touched=int(len(set(ys.tolist()))))
    return r


owner = sim._render_context_offscreen
out = {'shape': [h, w], 'cams': cams,
       'policy_con': {'offSamples': int(owner.con.offSamples), 'readDepthMap': int(owner.con.readDepthMap),
                      'offWidth': int(owner.con.offWidth), 'offHeight': int(owner.con.offHeight)},
       'model_offsamples': int(sim.model._model.vis.quality.offsamples),
       'model_offwidth': int(sim.model._model.vis.global_.offwidth), 'model_offheight': int(sim.model._model.vis.global_.offheight),
       'rows': []}
r = SingleSampleMaskRenderer(sim)
out['mask_con'] = r.audit()
for rep in range(3):
    for cam in cams:
        seg, z = r.render(cam, h, w, sites=False, depth=True)
        out['rows'].append({'rep': rep, 'cam': cam, 'src': 'mask_ctx', **stats(z)})
        _, zp = sim.render(width=w, height=h, camera_name=cam, depth=True, segmentation=True)
        out['rows'].append({'rep': rep, 'cam': cam, 'src': 'policy_ctx', **stats(zp)})
# a second read of the same mask-context render without re-rendering
import mujoco
seg, z = r.render(cams[0], h, w, sites=False, depth=True)
z2 = np.empty((h, w), dtype=np.float32)
rgb2 = np.empty((h, w, 3), dtype=np.uint8)
owner.gl_ctx.make_current()
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, r._con)
mujoco.mjr_readPixels(rgb=rgb2, depth=z2, viewport=mujoco.MjrRect(0, 0, w, h), con=r._con)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, owner.con)
out['reread_same_frame'] = {'first': stats(z), 'second': stats(z2), 'identical': bool(np.array_equal(z, z2))}
r.close()
env.close()
print(json.dumps(out, indent=1))
