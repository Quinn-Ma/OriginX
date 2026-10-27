"""T3 v2: production single-sample mask/controller checks, no model services.

At every zero-action/scripted-motion state and four diagnostic injected states:
compare production object masks, depth and analytic ghosts pixel-for-pixel to
an independent single-sample context using official robosuite render/read_pixels;
check production site-hiding subset/gained-SITE invariants; run the real
MaskFocusController.compute_masks/_paint, including all protected role masks;
check analytic ghosts against GL rasterized boxes (IoU >= .98 and <=1px boundary
band), depth convention, and unchanged native policy RGB. The native MSAA
context is never rebuilt or given different options by the mask implementation.

Source-resolution zero-interior-change fidelity is separate from the actual
XR1 crop/resize diagnostics (T2). The original 100ms overhead gate is retained
and a failure stays a failure. Scratch simulation is engineering evidence,
never task success or evidence of a policy improvement. Outputs are new v2
receipts/contact sheets, never overwritten historical results or runs/.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
OVERLAY = ROOT / 'work/id60-masks-validation/deps/sim-overlay'
DEFAULT_OUTPUT = ROOT / 'work/id60-masks-validation/exactness-checks-local-wsl-v2.json'
DEFAULT_CONTACT = ROOT / 'work/id60-masks-validation/contact-local-v2'
SEEDS = (20261025, 20261026, 20261027)
TASK = 'CoffeeSetupMug'
SUBJECT_LABEL = 'mug'
DESTINATION_LABEL_PREFIX = 'dispenser receiving region'
IOU_MIN, IOU_MIN_PIXELS = 0.98, 50
OVERHEAD_BUDGET_MS = 100.


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    import numpy as np
    if not values:
        return {'n': 0, 'median': None, 'p90': None, 'max': None, 'mean': None}
    a = np.asarray(values, dtype=float)
    return {'n': int(a.size), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90)),
            'max': float(a.max()), 'mean': float(a.mean())}


class RobosuiteReference:
    """Independent test-only MjrContext using official render and ID decoding.

    No inheritance from, or calls to, SingleSampleMaskRenderer/decode_segmentation.
    It shares only the episode's GL context and model/data. The reference owns
    all MuJoCo rendering objects; official methods operate on that private shim.
    """
    def __init__(self, sim):
        import copy
        import inspect
        import mujoco
        from robosuite.utils.binding_utils import MjRenderContext, _MjSim_render_lock
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.owner = sim._render_context_offscreen
        self.gl_ctx = self.owner.gl_ctx
        self.con = None
        self._official = MjRenderContext
        source = inspect.getfile(MjRenderContext)
        self.source = {'path': str(source), 'sha256': sha256_file(source),
                       'render': 'MjRenderContext.render', 'decode': 'MjRenderContext.read_pixels'}
        with _MjSim_render_lock:
            self.gl_ctx.make_current()
            saved = int(self.model.vis.quality.offsamples)
            try:
                self.model.vis.quality.offsamples = 0
                self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
            finally:
                self.model.vis.quality.offsamples = saved
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.owner.con)
        self._audit = {'kind': 'independent_official_robosuite_reference',
                       'offSamples': int(self.con.offSamples), 'model_offsamples_restored':
                       int(self.model.vis.quality.offsamples) == saved,
                       'independent_render_and_decode': True, 'source': self.source}
        try:
            if self.con.offSamples != 0:
                raise ValueError('reference context must be single-sample')
            self.scn = mujoco.MjvScene(self.model._model, maxgeom=int(self.owner.scn.maxgeom))
            self.cam, self.pert = mujoco.MjvCamera(), mujoco.MjvPerturb()
            self.pert.active, self.pert.select, self.pert.skinselect = 0, 0, -1
            self.vopt = copy.deepcopy(self.owner.vopt)
        except BaseException:
            self.close()
            raise

    def audit(self):
        return dict(self._audit)

    def render(self, camera, h, w, *, sites, depth):
        import copy
        import mujoco
        from robosuite.utils.binding_utils import _MjSim_render_lock
        if self.con is None:
            raise RuntimeError('reference context already closed')
        if w > self.con.offWidth or h > self.con.offHeight:
            raise ValueError('reference must not resize/mutate shared model')
        with _MjSim_render_lock:
            self.gl_ctx.make_current()
            self.vopt = copy.deepcopy(self.owner.vopt)
            if not sites:
                self.vopt.sitegroup[:] = 0
            self.scn.flags[:] = self.owner.scn.flags
            try:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
                self._official.render(self, w, h, camera_id=self.model.camera_name2id(camera), segmentation=True)
                value = self._official.read_pixels(self, w, h, depth=depth, segmentation=True)
            finally:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.owner.con)
        return value if depth else (value, None)

    def close(self):
        import mujoco
        from robosuite.utils.binding_utils import _MjSim_render_lock
        if self.con is None:
            return
        with _MjSim_render_lock:
            self.gl_ctx.make_current()
            con, self.con = self.con, None
            try:
                con.free()
            finally:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.owner.con)


# --------------------------------------------------------------------------- #
# test-only GL ghost render (copy of robosuite MjRenderContext.render/read_pixels)
# --------------------------------------------------------------------------- #

def render_with_ghost(renderer, camera, h, w, box, site_id, hide_sites=True):
    """Segment-mode render with the ghost box appended as one extra scene geom.

    The scene is rebuilt by mjv_updateScene on every render, so nothing here
    persists into any later (policy) render.
    """
    import numpy as np
    import mujoco
    def append_box(scene):
        n = scene.ngeom
        if n >= scene.maxgeom:
            raise RuntimeError('scene has no room for the ghost geom')
        geom = scene.geoms[n]
        mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_BOX),
                            np.asarray(box.half_extents, dtype=np.float64),
                            np.asarray(box.center, dtype=np.float64),
                            np.asarray(box.rotation, dtype=np.float64).reshape(9),
                            np.array([1., 1., 1., 1.], dtype=np.float32))
        geom.objtype = int(mujoco.mjtObj.mjOBJ_SITE)
        geom.objid = int(site_id)
        geom.segid = n
        scene.ngeom = n + 1
    seg, z01 = renderer.render(camera, h, w, sites=not hide_sites, depth=True, scene_hook=append_box)
    return np.asarray(seg)[::-1].copy(), np.asarray(z01)[::-1].copy()


def gl_cross_check(renderer, aug, camera, h, w, box, site_id, subject_mask, analytic):
    import numpy as np
    from originx_harness.id60_masks_validation.masks import dilate_chebyshev
    seg, _ = render_with_ghost(renderer, camera, h, w, box, site_id)
    ghost_any = (seg[:, :, 0] == 6) & (seg[:, :, 1] == int(site_id))
    gl = ghost_any & ~subject_mask
    union, inter = int((gl | analytic).sum()), int((gl & analytic).sum())
    iou = 1. if union == 0 else inter / union
    # pixels within Chebyshev distance 1 of the analytic mask boundary (both sides)
    band = dilate_chebyshev(analytic, 1) & dilate_chebyshev(~analytic, 1)
    mismatch = gl ^ analytic
    outside_band = int((mismatch & ~band).sum())
    # non-ghost pixels of the ghost render must equal the plain render (scene untouched elsewhere)
    plain = aug.cameras[camera].seg
    others_equal = bool(np.array_equal(seg[~ghost_any], plain[~ghost_any]))
    return {'analytic_px': int(analytic.sum()), 'gl_px': int(gl.sum()), 'iou': float(iou),
            'mismatch_px': int(mismatch.sum()), 'mismatch_outside_1px_band': outside_band,
            'evaluated': bool(max(analytic.sum(), gl.sum()) >= IOU_MIN_PIXELS),
            'non_ghost_pixels_equal_plain_render': others_equal}


# --------------------------------------------------------------------------- #
# cue construction mirroring the planned MaskFocusController.render
# --------------------------------------------------------------------------- #

def presentation_controller(aug, registry, *, stage, subject_id, destination_id, painter):
    """Real controller computation with a fixed test-only binding, no planner calls.

    Stage selection is injected solely for T3 engineering coverage; no task
    outcome or policy result is produced. Role resolution, small-component
    filtering, protected union, drawability and painting are production code.
    """
    from types import SimpleNamespace
    from originx_harness.id60_masks_validation import masks as M
    from originx_harness.id60_masks_validation.controller import MaskFocusController
    if painter not in (M.paint_rings, M.paint_rings_reference):
        raise ValueError('T3 requires the registered painter or its numpy reference')
    ctrl = MaskFocusController(None, 'full', arm='masks',
                               painter='supervision' if painter is M.paint_rings else 'numpy_reference')
    stages = [SimpleNamespace(operation='grasp', related_entity_ids=(), primary_target_id=subject_id,
                              completion_predicate='test-only grasp coverage'),
              SimpleNamespace(operation='place', related_entity_ids=(), primary_target_id=destination_id,
                              completion_predicate='test-only place coverage')]
    index = 1 if stage == 'place' else 0
    ctrl.manager = SimpleNamespace(index=index, stage=stages[index], stages=stages,
        snapshot=SimpleNamespace(target_id=destination_id if index else subject_id,
                                 related_entity_ids=(), mode='bound', status='active'))
    ctrl._previous_bound_grasp = {'stage_index': 0, 'target_id': subject_id} if index else None
    ctrl.set_geometry(aug, registry.frame)
    return ctrl


def present(rgb, aug, registry, camera_names, *, stage, subject_id, destination_id, painter,
            policy_diagnostics=False):
    """Paint with production compute_masks/_paint; measure against *all* role masks."""
    import numpy as np
    from originx_harness.id60_masks_validation import masks as M
    ctrl = presentation_controller(aug, registry, stage=stage, subject_id=subject_id,
                                  destination_id=destination_id, painter=painter)
    raw = {'rgb': rgb, 'camera_names': list(camera_names)}
    per_camera, info = ctrl.compute_masks(raw, registry, stage == 'place')
    if info['status'] != 'computed' or len(per_camera) != len(camera_names):
        raise AssertionError('T3 production mask computation did not produce every camera')
    out, records = np.array(rgb, copy=True), {}
    for index, (camera, data) in enumerate(zip(camera_names, per_camera)):
        if data is None:
            raise AssertionError(f'T3 missing mask frame: {camera}')
        rings, protected = data['color_rings'], data['protected']
        out[index] = ctrl._paint(rgb[index], rings)
        entries = data['entries']
        records[camera] = {
            'presentation_path': 'MaskFocusController.compute_masks/_paint',
            'roles': [entry['role']['class_id'] for entry in entries if entry['drawable']],
            'role_details': [dict(entry['leaf']) for entry in entries],
            'mask_px': [int(entry['mask'].sum()) for entry in entries if entry['mask'] is not None],
            'ring_px': {str(c): int(r.sum()) for c, r in rings.items()},
            'protected_px': int(protected.sum()),
            'inside_mask_pixels_changed': M.changed_pixels(rgb[index], out[index], protected),
            'outside_ring_pixels_changed': M.changed_pixels(rgb[index], out[index], ~data['ring_union']),
            'rings_disjoint': sum(int(r.sum()) for r in rings.values()) == int(data['ring_union'].sum()),
            'rings_touch_masks': bool((data['ring_union'] & protected).any()),
            'site_gate_failures': sum(e['leaf'].get('site_gate', 'pass') != 'pass' for e in entries)}
        if policy_diagnostics:
            # Outside the timed production measurements. No zero-boundary-bleed
            # assumption: bilinear preprocessing is measured separately.
            from contact_sheets import policy_input_diagnostics
            records[camera]['actual_xr1_preprocessing'] = policy_input_diagnostics(
                rgb[index], out[index], protected, data['ring_union'])
    return out, records


# --------------------------------------------------------------------------- #
# per-seed run
# --------------------------------------------------------------------------- #

def find_ids(extractor):
    subject = [s for s in extractor.specs if s.kind == 'object' and s.label == SUBJECT_LABEL]
    destination = [s for s in extractor.specs if s.kind == 'region' and s.label.startswith(DESTINATION_LABEL_PREFIX)]
    if len(subject) != 1 or len(destination) != 1:
        raise ValueError(f'expected one mug and one receiving region, got {len(subject)}/{len(destination)}')
    return subject[0], destination[0]


STATUSES = ('equal', 'contains', 'inconsistent', 'registry_hidden', 'both_hidden')


def registry_checks(sim, aug, registry, specs, camera_names, subject_id):
    """Augmenter masks (sites hidden) versus the frozen registry, plus a strict replica.

    A test-only with-sites seg render (exactly the extractor's call) checks that
    the extractor rule reproduces every registry bbox; the augmenter masks must be
    pixel supersets of those masks, gaining only pixels that were site pixels.
    """
    import numpy as np
    from originx_harness.id60_masks_validation import masks as M
    from originx_harness.id60_masks_validation.geometry import entity_mask
    h, w = aug.cameras[camera_names[0]].shape
    row = {'comparisons': 0, 'registry_visible': 0, 'strict_with_sites_equal': 0,
           'status': {k: 0 for k in STATUSES}, 'subject_status': {k: 0 for k in STATUSES},
           'superset_violation_px': 0, 'gained_px_total': 0,
           'gained_px_not_site': 0, 'entity_cameras_gaining_px': 0, 'subject_gained_px': {},
           'mismatches': []}
    # Frozen MSAA extractor replication is a diagnostic; the production
    # pixel-level site gate uses the *same single-sample context* as each mask.
    frozen_with_sites = {c: sim.render(width=w, height=h, camera_name=c, segmentation=True)[::-1]
                         for c in camera_names}
    for spec in specs:
        entity = registry.get(spec.id)
        for camera in camera_names:
            seg = aug.cameras[camera].seg_with_sites
            if seg is None:
                raise AssertionError('production frame lacks the with-sites reference')
            mask = entity_mask(aug, spec.id, camera)
            strict = M.object_mask(seg, spec.geoms)
            registered = entity.views.get(camera, {}).get('bbox') if entity.visible else None
            row['comparisons'] += 1
            row['registry_visible'] += registered is not None
            box = M.bbox_from_mask(M.object_mask(frozen_with_sites[camera], spec.geoms))
            same = (box is None and registered is None) or (
                box is not None and registered is not None and tuple(box) == tuple(registered))
            row['strict_with_sites_equal'] += same
            status = M.bbox_consistency(mask, registered)
            row['status'][status] += 1
            gained = mask & ~strict
            row['superset_violation_px'] += int((strict & ~mask).sum())
            row['gained_px_total'] += int(gained.sum())
            row['gained_px_not_site'] += int((gained & (seg[:, :, 0] != M.SITE_OBJTYPE)).sum())
            row['entity_cameras_gaining_px'] += bool(gained.any())
            if spec.id == subject_id:
                row['subject_gained_px'][camera] = int(gained.sum())
                row['subject_status'][status] += 1
            if (not same or status == 'inconsistent') and len(row['mismatches']) < 10:
                row['mismatches'].append({'id': spec.id, 'camera': camera, 'status': status, 'strict_equal': same,
                                          'mask_bbox': M.bbox_from_mask(mask), 'strict_bbox': box,
                                          'registry_bbox': list(registered) if registered else None,
                                          'mask_px': int(mask.sum()), 'strict_px': int(strict.sum())})
    return row


def merge_checks(total, row):
    if total is None:
        total = {k: 0 for k in row if k not in ('status', 'subject_status', 'subject_gained_px', 'mismatches')}
        total.update(status={k: 0 for k in STATUSES}, subject_status={k: 0 for k in STATUSES},
                     subject_gained_px={}, mismatches=[])
    for key, value in row.items():
        if key in ('status', 'subject_status'):
            for k, v in value.items():
                total[key][k] += v
        elif key == 'subject_gained_px':
            for k, v in value.items():
                total['subject_gained_px'].setdefault(k, []).append(v)
        elif key == 'mismatches':
            total['mismatches'] = (total['mismatches'] + value)[:20]
        else:
            total[key] += value
    return total


def inject_subject(sim, subject_root, aabb_low, aabb_high, center_xy=None, bottom_z=None, center_xyz=None):
    """Test-only: move the subject's free joint so its root-local AABB lands as requested."""
    import numpy as np
    import mujoco
    model = sim.model._model
    jnt = int(model.body_jntadr[subject_root])
    if jnt < 0 or int(model.jnt_type[jnt]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError('subject root has no free joint')
    adr = int(model.jnt_qposadr[jnt])
    rotation = np.asarray(sim.data.body_xmat[subject_root]).reshape(3, 3)
    center_local = (np.asarray(aabb_low) + np.asarray(aabb_high)) / 2
    if center_xyz is not None:
        target_local, target_world = center_local, np.asarray(center_xyz, dtype=float)
    else:
        target_local = np.array([center_local[0], center_local[1], aabb_low[2]])
        target_world = np.array([center_xy[0], center_xy[1], bottom_z])
    sim.data.qpos[adr:adr + 3] = target_world - rotation @ target_local
    sim.data.qvel[model.jnt_dofadr[jnt]:model.jnt_dofadr[jnt] + 6] = 0
    sim.forward()


def contact_sheet(path, rows, labels):
    """Show both source RGB and the exact frozen XR1 crop/resize view."""
    from contact_sheets import write_contact_sheet
    return write_contact_sheet(path, rows[0], rows[1], labels=[
        labels[0], labels[1], 'XR1 exact crop/resize: raw', 'XR1 exact crop/resize: masks'])


def debug_tint(rgb, aug, camera_names, subject_id, destination_id):
    """Inspection only: 50% tint of the raw subject (cyan) and visible ghost (orange) masks."""
    import numpy as np
    from originx_harness.id60_masks_validation import masks as M
    from originx_harness.id60_masks_validation.geometry import entity_mask, destination_mask
    out = np.array(rgb, copy=True)
    for i, camera in enumerate(camera_names):
        subject = entity_mask(aug, subject_id, camera)
        ghost = destination_mask(aug, destination_id, subject_id, camera, subject_mask=subject)
        for mask, color in ((subject, M.CYAN), (ghost, M.ORANGE)):
            if mask is not None and mask.any():
                out[i][mask] = (out[i][mask].astype(np.uint16) + np.asarray(color, np.uint16)) // 2
    return out


def single_sample_reference_check(reference, aug, specs, subject_id, destination_id):
    """Pixel equality to a second zero-sample MjrContext at the identical state.

    The reference uses official robosuite render/read_pixels, independently of
    production scene construction and ID decoding. Analytic-versus-GL ghost
    correctness is a separate IoU gate.
    """
    import numpy as np
    from dataclasses import replace
    from originx_harness.id60_masks_validation.geometry import render_camera_frame, entity_mask, destination_mask, depth_planes
    reference_frames = {}
    for camera, frame in aug.cameras.items():
        h, w = frame.shape
        reference_frames[camera] = render_camera_frame(camera, h, w,
            renderer=lambda c, hh, ww: reference.render(c, hh, ww, sites=False, depth=True),
            site_renderer=lambda c, hh, ww: reference.render(c, hh, ww, sites=True, depth=False)[0],
            matrices=lambda c, hh, ww, f=frame: (f.K, f.cam_to_world),
            planes=depth_planes(reference.sim),
            convention=aug.audit['mask_frame']['depth_convention'])
    other = replace(aug, cameras=reference_frames)
    row = {'object_mask_mismatch_px': 0, 'analytic_ghost_mismatch_px': 0,
           'segmentation_mismatch_px': 0, 'with_sites_mismatch_px': 0, 'depth_mismatch_px': 0,
           'object_camera_comparisons': 0, 'reference_offSamples': reference.audit()['offSamples']}
    for camera, frame in aug.cameras.items():
        ref = reference_frames[camera]
        row['segmentation_mismatch_px'] += int(np.any(frame.seg != ref.seg, axis=-1).sum())
        row['with_sites_mismatch_px'] += int(np.any(frame.seg_with_sites != ref.seg_with_sites, axis=-1).sum())
        row['depth_mismatch_px'] += int((frame.depth_m != ref.depth_m).sum())
        for spec in specs:
            row['object_mask_mismatch_px'] += int((entity_mask(aug, spec.id, camera) !=
                                                   entity_mask(other, spec.id, camera)).sum())
            row['object_camera_comparisons'] += 1
        a = destination_mask(aug, destination_id, subject_id, camera)
        b = destination_mask(other, destination_id, subject_id, camera)
        if (a is None) != (b is None):
            raise AssertionError('ghost reference availability differs')
        row['analytic_ghost_mismatch_px'] += int((a != b).sum()) if a is not None else 0
    return row


def policy_rgb_unchanged(sim, rgb, cameras):
    """Re-render the native MSAA policy context at the same state, byte for byte."""
    import numpy as np
    h, w = rgb.shape[1:3]
    return [bool(np.array_equal(sim.render(width=w, height=h, camera_name=c)[::-1], rgb[i]))
            for i, c in enumerate(cameras)]


def mask_components(mask):
    import numpy as np
    from scipy.ndimage import label
    labels, count = label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())[1:] if count else np.zeros(0, dtype=int)
    return {'components': int(count), 'largest_px': int(sizes.max()) if count else 0,
            'stray_px_in_components_le_2px': int(sizes[sizes <= 2].sum()) if count else 0}


def injected_states(*, sim, extractor, augmenter, aug, shape, cams, subject, destination, geom_specs, anchor,
                    contact_dir, seed, reference):
    """Test-only free-joint injections of the subject; restores the saved state after each."""
    import numpy as np
    from originx_harness.id60.entities import EntityRegistry
    from originx_harness.id60_masks_validation import masks as M
    from originx_harness.id60_masks_validation.geometry import destination_ghost, destination_mask, entity_mask
    _, h, w, _ = shape
    state = sim.get_state()
    bounds = aug.subject_bounds[subject.id]
    geometry = augmenter._geometry[subject.id]
    if aug.region_anchors[destination.id] is not None:
        anchor_center, parent_rotation, z_support = aug.region_anchors[destination.id]
    else:
        anchor_center, parent_rotation, z_support = anchor, np.eye(3), float(anchor[2])
    side = float(min((bounds[1] - bounds[0])[:2]))
    beside_xy = (np.asarray(anchor_center) + parent_rotation @ np.array([side + .04, 0., 0.]))[:2]
    grip_site = sim.model.site_name2id('gripper0_right_grip_site')
    injections = {
        'at_site': dict(center_xy=anchor_center[:2], bottom_z=z_support),
        'above_site': dict(center_xy=anchor_center[:2], bottom_z=z_support + .08),
        'beside_site': dict(center_xy=beside_xy, bottom_z=z_support),
        'in_gripper': dict(center_xyz=np.asarray(sim.data.site_xpos[grip_site]).copy()),
    }
    injected = {}
    for k, (name, target) in enumerate(injections.items()):
        inject_subject(sim, geometry.root_body, bounds[0], bounds[1], **target)
        registry = EntityRegistry()
        registry.update(extractor.observe(shape), 100000 + k)
        aug_i = augmenter.observe(shape, registry)
        rgb = np.stack([sim.render(width=w, height=h, camera_name=c)[::-1] for c in cams])
        checks = registry_checks(sim, aug_i, registry, geom_specs, cams, subject.id)
        p_rgb, p_rec = present(rgb, aug_i, registry, cams, stage='place', subject_id=subject.id,
                               destination_id=destination.id, painter=M.paint_rings, policy_diagnostics=True)
        r_rgb, _ = present(rgb, aug_i, registry, cams, stage='place', subject_id=subject.id,
                           destination_id=destination.id, painter=M.paint_rings_reference)
        reference_check = single_sample_reference_check(reference, aug_i, geom_specs, subject.id, destination.id)
        row = {'registry_checks': checks, 'single_sample_reference': reference_check,
               'policy_rgb_unchanged': policy_rgb_unchanged(sim, rgb, cams),
               'cues': p_rec, 'supervision_equals_reference': bool(p_rgb.tobytes() == r_rgb.tobytes()),
               'subject_root_xpos_m': np.asarray(sim.data.body_xpos[geometry.root_body]).tolist()}
        box = destination_ghost(aug_i, destination.id, subject.id)
        exclusion, gl = {}, []
        for camera in cams:
            subject_mask = entity_mask(aug_i, subject.id, camera)
            ghost = destination_mask(aug_i, destination.id, subject.id, camera, subject_mask=subject_mask)
            unexcluded = destination_mask(aug_i, destination.id, subject.id, camera,
                                          subject_mask=np.zeros_like(subject_mask))
            exclusion[camera] = {
                'ghost_px': int(ghost.sum()) if ghost is not None else None,
                'subject_px': int(subject_mask.sum()),
                'ghost_and_subject_px': int((ghost & subject_mask).sum()) if ghost is not None else None,
                'subject_px_inside_unexcluded_ghost': (int((unexcluded & subject_mask).sum())
                                                       if unexcluded is not None else None),
                'subject_components': mask_components(subject_mask)}
            if box is not None:
                gl.append({'state': name, 'camera': camera,
                           **gl_cross_check(augmenter.mask_renderer, aug_i, camera, h, w, box, destination.site, subject_mask, ghost)})
        row['exclusion'] = exclusion
        row['gl_cross_check'] = gl
        injected[name] = row
        if contact_dir is not None and name in ('above_site', 'at_site', 'in_gripper'):
            contact_sheet(contact_dir / f'seed-{seed}-place-{name}.png',
                          [rgb, p_rgb, debug_tint(rgb, aug_i, cams, subject.id, destination.id)],
                          [f'seed {seed} place-like ({name}, injected): raw render',
                           'masks arm (orange destination ghost ring, cyan held-object ring)',
                           'inspection only: cyan=subject mask, orange=visible ghost'])
        sim.set_state(state)
        sim.forward()
    return injected


def run_seed(seed, args, contact_dir):
    """One scratch episode inside the drivers' ``episode_resource_guard``.

    The guard defers cyclic GC for the whole env lifecycle, exactly like the
    registered drivers.  Without it, a later collection of the stale EGL
    context left by the RoboCasa reset calls ``eglReleaseThread`` and every
    following GL readback (RGB, segmentation and depth) silently stops being
    written; see tools/probe_gc_egl.py.
    """
    from originx_harness.id60_revision.resource_guard import episode_resource_guard
    with episode_resource_guard():
        return _run_seed(seed, args, contact_dir)


def _run_seed(seed, args, contact_dir):
    import numpy as np
    from originx_harness.id60.entities import EntityRegistry
    from originx_harness.id60_hierarchical_revision.geometry import HierarchicalGeometryAugmenter
    from originx_harness.id60_masks_validation.site_rendering import make_environment, rendering_identity
    from originx_harness.id60_masks_validation import masks as M
    from originx_harness.id60_masks_validation.geometry import (
        MaskedGeometryAugmenter, MaskedGeometryAugmentation, destination_ghost, destination_mask, entity_mask)

    result = {'seed': seed}
    augmenter = reference = None
    t0 = time.perf_counter()
    site = json.loads(Path(args.site_json).read_text(encoding='utf-8')) if args.site_json else {}
    env = make_environment(TASK, 600, site)
    result['env_build_s'] = time.perf_counter() - t0
    try:
        t0 = time.perf_counter()
        raw = env.reset(seed, f'masks-t3-{seed}')
        result['policy_rendering'] = getattr(env, 'rendering_receipt', rendering_identity(site))
        result['reset_s'] = time.perf_counter() - t0
        from scripts.run_id60_grasp_holdout_renderer_recovery import current_gl_identity
        result['gl'] = current_gl_identity()
        sim, extractor = env.extractor.sim, env.extractor
        shape = raw['rgb'].shape
        _, h, w, _ = shape
        cams = tuple(env.camera_names)
        subject, destination = find_ids(extractor)
        result['subject'] = {'id': subject.id, 'label': subject.label, 'geoms': len(subject.geoms)}
        result['destination'] = {'id': destination.id, 'label': destination.label, 'site': destination.site,
                                 'parent': destination.parent}
        # policy RGB equals a direct render with the same flip (validates the flip/resolution)
        options = sim._render_context_offscreen.vopt
        saved_sites = np.array(options.sitegroup, copy=True)
        options.sitegroup[:] = 0
        try:
            result['rgb_with_sites_hidden_equals_policy_rgb'] = [
                bool(np.array_equal(sim.render(width=w, height=h, camera_name=c)[::-1], raw['rgb'][i]))
                for i, c in enumerate(cams)]
        finally:
            options.sitegroup[:] = saved_sites
        result['rgb_rerender_equals_policy_rgb'] = [
            bool(np.array_equal(sim.render(width=w, height=h, camera_name=c)[::-1], raw['rgb'][i]))
            for i, c in enumerate(cams)]
        t0 = time.perf_counter()
        augmenter = MaskedGeometryAugmenter(extractor)
        reference = RobosuiteReference(sim)
        result['augmenter_construction_ms'] = 1e3 * (time.perf_counter() - t0)
        aug = augmenter.observe(shape, env.registry)
        assert isinstance(aug, MaskedGeometryAugmentation)
        result['depth_check_frame0'] = augmenter.depth_check
        result['production_mask_context'] = augmenter.mask_context_audit()
        result['reference_mask_context'] = reference.audit()
        result['policy_rgb_unchanged_after_context_creation'] = policy_rgb_unchanged(sim, raw['rgb'], cams)
        support = aug.audit['mask_frame']['region_support'][destination.id]
        if support.get('hit_geom') is not None:
            support = {**support, 'hit_geom_name': sim.model.geom_id2name(support['hit_geom']),
                       'hit_geom_group': int(sim.model.geom_group[support['hit_geom']])}
        result['destination_support'] = support
        # visual-group downward ray, audit comparison only
        import mujoco
        hit = np.array([-1], dtype=np.int32)
        anchor = np.asarray(sim.data.site_xpos[destination.site], dtype=float)
        group = np.asarray(sim._render_context_offscreen.vopt.geomgroup, dtype=np.uint8).copy()
        dist = mujoco.mj_ray(sim.model._model, sim.data._data, anchor, np.array([0., 0., -1.]), group, 1, -1, hit)
        result['destination_support_visual_ray'] = {
            'distance_m': float(dist), 'hit_geom': int(hit[0]),
            'hit_geom_name': sim.model.geom_id2name(int(hit[0])) if hit[0] >= 0 else None}
        box0 = destination_ghost(aug, destination.id, subject.id)
        result['ghost_box'] = box0.to_dict() if box0 is not None else None
        geom_specs = [s for s in extractor.specs if s.kind in ('object', 'part') and s.geoms]
        result['object_part_specs'] = len(geom_specs)

        # contact sheet: grasp stage at the initial state, on the true policy RGB
        grasp_rgb, grasp_records = present(raw['rgb'], aug, env.registry, cams, stage='grasp',
                                           subject_id=subject.id, destination_id=destination.id,
                                           painter=M.paint_rings, policy_diagnostics=True)
        contact_sheet(contact_dir / f'seed-{seed}-grasp-initial.png',
                      [raw['rgb'], grasp_rgb, debug_tint(raw['rgb'], aug, cams, subject.id, destination.id)],
                      [f'seed {seed} grasp initial: raw policy RGB', 'masks arm (orange target ring)',
                       'inspection only: cyan=subject mask, orange=visible ghost'])
        result['grasp_initial_cues'] = grasp_records

        bbox = {'zero': None, 'motion': None}
        timing = {'masked_observe_ms': [], 'base_observe_ms': [], 'grasp_cue_ms': [], 'place_cue_ms': [],
                  'reference_place_cue_ms': [], 'frozen_extractor_observe_ms': []}
        components = {camera: {'max_components': 0, 'max_stray_px': 0, 'steps_visible': 0} for camera in cams}
        inside_changed, painter_equal, steps_checked = 0, True, 0
        ring_violations = bbox_inconsistent = outside_ring_changed = 0
        gl_checks, reference_checks, policy_rgb_checks = [], [], []
        visible_cams = {'subject': 0, 'ghost': 0}
        steps = args.zero_steps + args.motion_steps
        for step in range(steps + 1):
            if step:
                t0 = time.perf_counter()
                aug = augmenter.observe(shape, env.registry)
                timing['masked_observe_ms'].append(1e3 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                HierarchicalGeometryAugmenter.observe(augmenter, shape, env.registry)
                timing['base_observe_ms'].append(1e3 * (time.perf_counter() - t0))
                if step % 8 == 0:
                    t0 = time.perf_counter()
                    env.extractor.observe(shape)
                    timing['frozen_extractor_observe_ms'].append(1e3 * (time.perf_counter() - t0))
            for camera in cams:
                comp = mask_components(entity_mask(aug, subject.id, camera))
                row = components[camera]
                row['steps_visible'] += int(comp['components'] > 0)
                row['max_components'] = max(row['max_components'], comp['components'])
                row['max_stray_px'] = max(row['max_stray_px'], comp['stray_px_in_components_le_2px'])
            reference_checks.append(single_sample_reference_check(reference, aug, geom_specs, subject.id, destination.id))
            policy_rgb_checks.append(policy_rgb_unchanged(sim, raw['rgb'], cams))
            phase = 'zero' if step <= args.zero_steps else 'motion'
            checks = registry_checks(sim, aug, env.registry, geom_specs, cams, subject.id)
            checks['mismatches'] = [{**m, 'step': step} for m in checks['mismatches']]
            bbox[phase] = merge_checks(bbox[phase], checks)
            t0 = time.perf_counter()
            g_rgb, g_rec = present(raw['rgb'], aug, env.registry, cams, stage='grasp', subject_id=subject.id,
                                   destination_id=destination.id, painter=M.paint_rings)
            timing['grasp_cue_ms'].append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            p_rgb, p_rec = present(raw['rgb'], aug, env.registry, cams, stage='place', subject_id=subject.id,
                                   destination_id=destination.id, painter=M.paint_rings)
            timing['place_cue_ms'].append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            r_rgb, _ = present(raw['rgb'], aug, env.registry, cams, stage='place', subject_id=subject.id,
                               destination_id=destination.id, painter=M.paint_rings_reference)
            timing['reference_place_cue_ms'].append(1e3 * (time.perf_counter() - t0))
            g_ref, _ = present(raw['rgb'], aug, env.registry, cams, stage='grasp', subject_id=subject.id,
                               destination_id=destination.id, painter=M.paint_rings_reference)
            painter_equal &= bool(p_rgb.tobytes() == r_rgb.tobytes() and g_rgb.tobytes() == g_ref.tobytes())
            for rec in (*g_rec.values(), *p_rec.values()):
                inside_changed += rec['inside_mask_pixels_changed']
                outside_ring_changed += rec['outside_ring_pixels_changed']
                ring_violations += int((not rec['rings_disjoint']) or rec['rings_touch_masks'])
                bbox_inconsistent += rec['site_gate_failures']
            visible_cams['subject'] += sum(1 in r['roles'] for r in p_rec.values())
            visible_cams['ghost'] += sum(0 in r['roles'] for r in p_rec.values())
            steps_checked += 1
            if step % 16 == 0:
                box = destination_ghost(aug, destination.id, subject.id)
                for camera in cams:
                    subject_mask = entity_mask(aug, subject.id, camera)
                    analytic = destination_mask(aug, destination.id, subject.id, camera, subject_mask=subject_mask)
                    if box is not None:
                        gl_checks.append({'step': step, 'state': 'trajectory', 'camera': camera,
                                          **gl_cross_check(augmenter.mask_renderer, aug, camera, h, w, box, destination.site,
                                                           subject_mask, analytic)})
            if step == steps:
                break
            action = np.zeros(12, dtype=np.float32)
            if step >= args.zero_steps:
                k = step - args.zero_steps
                action[0] = .5 * np.sin(2 * np.pi * k / 40)
                action[1] = .5 * np.cos(2 * np.pi * k / 40)
                action[2] = -.4 if k < args.motion_steps // 2 else .4
                action[5] = .3
            raw, _, _ = env.step(action)
        for phase, value in bbox.items():
            if value is not None:
                gained = value.pop('subject_gained_px')
                value['subject_gained_px_by_camera'] = {c: stats(v) for c, v in gained.items()}
        result['registry_checks'] = bbox
        result['steps_checked'] = steps_checked
        result['zero_steps'], result['motion_steps'] = args.zero_steps, args.motion_steps
        result['inside_mask_pixels_changed_total'] = inside_changed
        result['outside_ring_pixels_changed_total'] = outside_ring_changed
        result['ring_violations_total'] = ring_violations
        result['cue_site_gate_failures_total'] = bbox_inconsistent
        result['single_sample_reference_checks'] = reference_checks
        result['policy_rgb_unchanged_all_steps'] = policy_rgb_checks
        result['supervision_equals_reference_all_steps'] = painter_equal
        result['place_stage_camera_steps_with'] = visible_cams
        result['timing_ms'] = {k: stats(v) for k, v in timing.items()}
        result['subject_mask_components_trajectory'] = components
        result['gl_cross_check_trajectory'] = gl_checks
        result['depth_check_extended_final_state'] = augmenter.check_depth_convention(
            h, w, samples_per_camera=args.depth_samples)

        # ---------------- state injection (test only) ---------------- #
        common = dict(sim=sim, extractor=extractor, augmenter=augmenter, aug=aug, shape=shape, cams=cams,
                      subject=subject, destination=destination, geom_specs=geom_specs, anchor=anchor, reference=reference)
        result['state_injection'] = injected_states(**common, contact_dir=contact_dir, seed=seed)
        result['offscreen_samples_production'] = int(sim._render_context_offscreen.con.offSamples)
    finally:
        if reference is not None:
            reference.close()
        if augmenter is not None:
            augmenter.close()
        env.close()
    return result


def summarize(results):
    import numpy as np
    from originx_harness.id60_masks_validation import masks as M
    rows = [r['registry_checks'][p] for r in results for p in ('zero', 'motion') if r['registry_checks'][p]]
    rows += [v['registry_checks'] for r in results for v in r['state_injection'].values()]
    registry_all = {k: sum(x[k] for x in rows) for k in (
        'comparisons', 'registry_visible', 'strict_with_sites_equal', 'superset_violation_px',
        'gained_px_total', 'gained_px_not_site', 'entity_cameras_gaining_px')}
    registry_all['bbox_diagnostic_status'] = {k: sum(x['status'][k] for x in rows) for k in STATUSES}
    refs = [x for r in results for x in r['single_sample_reference_checks']] + [
        v['single_sample_reference'] for r in results for v in r['state_injection'].values()]
    reference_totals = {k: sum(x[k] for x in refs) for k in (
        'object_mask_mismatch_px', 'analytic_ghost_mismatch_px', 'segmentation_mismatch_px',
        'with_sites_mismatch_px', 'depth_mismatch_px', 'object_camera_comparisons')}
    gl = [g for r in results for g in r['gl_cross_check_trajectory']] + [
        g for r in results for v in r['state_injection'].values() for g in v['gl_cross_check']]
    evaluated = [g for g in gl if g['evaluated']]
    ious = [g['iou'] for g in evaluated]
    selected = [r['depth_check_frame0']['selected'] for r in results] + [
        r['depth_check_extended_final_state']['selected'] for r in results]
    exclusion_ok = all(e['ghost_and_subject_px'] in (0, None) for r in results
                       for v in r['state_injection'].values() for e in v['exclusion'].values())
    inside = sum(r['inside_mask_pixels_changed_total'] for r in results) + sum(
        c['inside_mask_pixels_changed'] for r in results for v in r['state_injection'].values()
        for c in v['cues'].values())
    rings_ok = all(not c['rings_touch_masks'] and c['rings_disjoint'] and c['outside_ring_pixels_changed'] == 0
                  for r in results for v in r['state_injection'].values() for c in v['cues'].values()) and all(
        r['ring_violations_total'] == 0 and r['outside_ring_pixels_changed_total'] == 0
        and r['cue_site_gate_failures_total'] == 0 for r in results)
    painter_ok = all(r['supervision_equals_reference_all_steps'] for r in results) and all(
        v['supervision_equals_reference'] for r in results for v in r['state_injection'].values())
    extra = [r['timing_ms']['masked_observe_ms']['median'] - r['timing_ms']['base_observe_ms']['median']
             for r in results]
    place = [r['timing_ms']['place_cue_ms']['median'] for r in results]
    overhead = [e + p for e, p in zip(extra, place)]
    rgb_ok = all(all(r['rgb_rerender_equals_policy_rgb']) and
                 all(r['policy_rgb_unchanged_after_context_creation']) and
                 all(all(x) for x in r['policy_rgb_unchanged_all_steps']) and
                 all(all(v['policy_rgb_unchanged']) for v in r['state_injection'].values()) for r in results)
    contexts = [r[key] for r in results for key in ('production_mask_context', 'reference_mask_context')]
    context_ok = bool(contexts) and all(c['offSamples'] == 0 and c['model_offsamples_restored'] for c in contexts)
    reference_ok = bool(refs) and reference_totals['object_camera_comparisons'] > 0 and all(
        reference_totals[k] == 0 for k in reference_totals if k.endswith('_px'))
    gates = {
        'scope': 'source_resolution_only; actual XR1 crop/resize diagnostics are separate T2 evidence',
        'production_single_sample_registry_checks': registry_all,
        'single_sample_reference_pixel_comparison': reference_totals,
        'reference_scope': 'independent official robosuite render/read_pixels in a private single-sample MjrContext',
        'independent_reference_render_and_decode': all(r['reference_mask_context']['independent_render_and_decode'] for r in results),
        'depth_convention_selected': sorted(set(map(str, selected))),
        'depth_convention_frozen': M.DEPTH_CONVENTION,
        'gl_iou_min_evaluated': min(ious) if ious else None,
        'gl_iou_median_evaluated': float(np.median(ious)) if ious else None,
        'gl_evaluated_masks': len(evaluated), 'gl_total_masks': len(gl),
        'gl_mismatch_outside_1px_band_total': sum(g['mismatch_outside_1px_band'] for g in evaluated),
        'gl_non_ghost_pixels_equal_plain_render': all(g['non_ghost_pixels_equal_plain_render'] for g in gl),
        'policy_rgb_unchanged': rgb_ok,
        'policy_offscreen_samples': sorted({r['offscreen_samples_production'] for r in results}),
        'mask_offscreen_samples': sorted({c['offSamples'] for c in contexts}),
        'ghost_subject_exclusion_ok': exclusion_ok, 'inside_mask_pixels_changed_total': inside,
        'rings_disjoint_and_exterior': rings_ok, 'supervision_equals_reference': painter_ok,
        'overhead_ms_median_per_seed': overhead,
        'augmenter_extra_ms_median_per_seed': extra, 'place_cue_ms_median_per_seed': place,
        'all_gates_are_predeclared': True,
    }
    gates['passed'] = {
        'G2_production_context_single_sample': context_ok,
        'G2_strict_frozen_extractor_rule_100pct': bool(rows) and registry_all['strict_with_sites_equal'] == registry_all['comparisons'],
        'G2_pixel_identical_single_sample_reference': reference_ok,
        'G2_site_hiding_superset_and_only_site_pixels_gained_production': bool(rows)
            and registry_all['superset_violation_px'] == 0 and registry_all['gained_px_not_site'] == 0,
        'policy_rgb_byte_identical': rgb_ok,
        'G3_depth_convention': gates['depth_convention_selected'] == [M.DEPTH_CONVENTION],
        'G4_gl_iou_ge_098_production_single_sample': bool(ious) and min(ious) >= IOU_MIN,
        'G4_gl_mismatch_within_1px_band_production_single_sample': bool(evaluated)
            and gates['gl_mismatch_outside_1px_band_total'] == 0,
        'G4_non_ghost_pixels_unchanged': bool(gl) and gates['gl_non_ghost_pixels_equal_plain_render'],
        'G5_masks_source_inside_unchanged': inside == 0,
        'exclusion_rings_painter_production_controller': exclusion_ok and rings_ok and painter_ok,
        'G8_overhead_le_100ms': bool(overhead) and all(o <= OVERHEAD_BUDGET_MS for o in overhead),
    }
    gates['all_passed'] = all(gates['passed'].values())
    return gates


def child(args):
    import numpy as np
    import mujoco
    import cv2
    from originx_harness.id60_masks_validation import masks as M
    contact_dir = Path(args.contact_dir)
    output = Path(args.output)
    for path in (output, contact_dir):
        if 'runs' in Path(os.path.relpath(path.resolve(), ROOT)).parts[:1]:
            raise ValueError('T3 never writes under runs/')
    if output.exists():
        raise FileExistsError(f'T3 receipts are immutable; select a fresh --output: {output}')
    if args.zero_steps + args.motion_steps < 1:
        raise ValueError('at least one step is required to measure the registered overhead gate')
    started = datetime.now(timezone.utc).isoformat()
    results = [run_seed(seed, args, contact_dir) for seed in args.seeds]
    receipt = {
        'schema': 'id60-masks-exactness-checks-v2', 'kind': 'T3_sim_in_the_loop', 'task': TASK,
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': list(args.seeds), 'host': {'site': args.site_label, 'gl': results[0]['gl'] if results else None,
                                            'renderer_env': {k: os.environ.get(k) for k in (
                                                'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'CUDA_VISIBLE_DEVICES',
                                                'MUJOCO_EGL_DEVICE_ID', 'MESA_D3D12_DEFAULT_ADAPTER_NAME')}},
        'versions': {'numpy': np.__version__, 'mujoco': mujoco.__version__, 'cv2': cv2.__version__,
                     'supervision': M.supervision_version(), 'python': sys.version.split()[0]},
        'sources_sha256': {name: sha256_file(ROOT / name) for name in (
            'originx_harness/id60_masks_validation/masks.py',
            'originx_harness/id60_masks_validation/geometry.py',
            'originx_harness/id60_masks_validation/controller.py',
            'originx_harness/id60_masks_validation/runtime.py',
            'originx_harness/id60_masks_validation/site_rendering.py',
            'work/id60-masks-validation/contact_sheets.py',
            'work/id60-masks-validation/replay_trace_checks.py',
            'work/id60-masks-validation/check_mask_exactness.py')},
        'constants': {'ring_width': M.RING_WIDTH, 'min_pixels': M.MIN_PIXELS, 'depth_tol_m': M.DEPTH_TOL_M,
                      'depth_convention': M.DEPTH_CONVENTION, 'iou_min': IOU_MIN,
                      'iou_min_pixels': IOU_MIN_PIXELS, 'overhead_budget_ms': OVERHEAD_BUDGET_MS},
        'no_model_services': True, 'writes_under_runs': False,
        'actual_xr1_preprocessing_identity': __import__('contact_sheets').preprocessing_identity(),
        'oracle_simulator_geometry_track': True,
        'actual_xr1_metrics_scope': 'sampled initial grasp and injected place states; separate from source-resolution gates',
        'state_injection_is_test_only': True,
        'results': results,
    }
    receipt['summary'] = summarize(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False)
    output.write_text(text + '\n', encoding='utf-8')
    print(json.dumps(receipt['summary'], indent=1, sort_keys=True))
    return 0 if receipt['summary']['all_passed'] else 3


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--zero-steps', type=int, default=100)
    parser.add_argument('--motion-steps', type=int, default=40)
    parser.add_argument('--depth-samples', type=int, default=256)
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--contact-dir', default=str(DEFAULT_CONTACT))
    parser.add_argument('--site-label', default='local_wsl')
    parser.add_argument('--site-json', default=None,
                        help='site.json-shaped config (python/pythonpath/env); default: the local regions site_config()')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    if args.child:
        return child(args)
    from scripts.launch_id60_cell import process_environment
    if args.site_json:
        site = json.loads(Path(args.site_json).read_text(encoding='utf-8'))
    else:
        from scripts.run_id60_regions_validation_local import site_config
        site = site_config()
    environment = process_environment(site, 'sim')
    environment['PYTHONPATH'] = str(OVERLAY) + os.pathsep + environment['PYTHONPATH']
    environment['MPLBACKEND'] = 'Agg'
    environment['MASKS_REQUIRE_SUPERVISION'] = '1'
    command = [site['python']['sim'], str(Path(__file__).resolve()), '--child', *sys.argv[1:]]
    return subprocess.call(command, env=environment, cwd=str(ROOT))


if __name__ == '__main__':
    sys.exit(main())
