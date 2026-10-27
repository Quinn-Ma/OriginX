"""CPU-only inspection of the exact frozen XR1 image transform.

This module never imports a policy, simulator or GPU runtime. Source-image
protection and post-interpolation boundary mixing are different quantities.
The latter is measured, not silently converted into a zero-interior claim.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_RELATIVE = 'work/id60-revision/asset-pack/upstream-validation/Xiaomi-Robotics-1/eval_robocasa365/entry.py'
UPSTREAM_SHA256 = '18d70fade4c990210dc5b9c34285c8030eac142d9c332066c73335e1c0aaa28e'
ADAPTER_RELATIVE = 'originx_harness/adapters.py'
ADAPTER_SHA256 = '1b436e133569381e2b767ad69307b4c73636b9f442d5b33cc912cf137517ef13'
CROP_RATIO = 0.95
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
SCHEMA = 'id60-masks-policy-input-v2'


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


@lru_cache(maxsize=8)
def _compile_crop(source: bytes):
    # The whole source hash is admitted before this function is called. Only
    # this dependency-free function is compiled; upstream module imports and
    # model clients are never executed.
    node = next((n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'center_crop'), None)
    if node is None:
        raise ValueError('Frozen XR1 center_crop function is missing')
    namespace = {'np': np, 'Image': Image}
    exec(compile(ast.Module(body=[node], type_ignores=[]), UPSTREAM_RELATIVE, 'exec'), namespace)
    return namespace['center_crop'], _sha(ast.dump(node, include_attributes=False).encode())


def preprocessing_identity(upstream_path=None, *, adapter_path=None) -> dict:
    """Bind diagnostics to both frozen crop code and the adapter's 0.95 call."""
    path = Path(upstream_path) if upstream_path is not None else ROOT / UPSTREAM_RELATIVE
    adapter = Path(adapter_path) if adapter_path is not None else ROOT / ADAPTER_RELATIVE
    source, adapter_source = path.read_bytes(), adapter.read_bytes()
    if _sha(source) != UPSTREAM_SHA256:
        raise ValueError('XR1 entry.py differs from the frozen preprocessing identity')
    if _sha(adapter_source) != ADAPTER_SHA256:
        raise ValueError('XR1 adapters.py differs from the frozen preprocessing identity')
    _, function_sha = _compile_crop(source)
    return {'source': UPSTREAM_RELATIVE, 'source_sha256': UPSTREAM_SHA256,
            'function': 'center_crop', 'function_ast_sha256': function_sha,
            'adapter': ADAPTER_RELATIVE, 'adapter_sha256': ADAPTER_SHA256,
            'crop_ratio': CROP_RATIO, 'resampling': 'PIL.Image.Resampling.BILINEAR',
            'pillow_version': PIL.__version__, 'numpy_version': np.__version__,
            'scope': 'Exact frozen XR1 client center_crop stage; subsequent model processor is not inspected.'}


def crop_box(shape, crop_ratio=CROP_RATIO):
    if not np.isfinite(crop_ratio) or not 0 < crop_ratio <= 1:
        raise ValueError('crop_ratio must be in (0, 1]')
    height, width = int(shape[0]), int(shape[1])
    crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def _rgb(value):
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
        raise ValueError('Expected uint8 HxWx3 RGB')
    return result


def xr1_crop(image, *, upstream_path=None, crop_ratio=CROP_RATIO):
    """Execute the admitted production function, without importing its module."""
    image = _rgb(image)
    if crop_ratio != CROP_RATIO:
        raise ValueError('Only the frozen production crop_ratio=0.95 is admitted')
    preprocessing_identity(upstream_path)
    path = Path(upstream_path) if upstream_path is not None else ROOT / UPSTREAM_RELATIVE
    function, _ = _compile_crop(path.read_bytes())
    return np.asarray(function(image, crop_ratio)).copy()


def reference_crop(image, crop_ratio=CROP_RATIO):
    """Independent geometry implementation used by CPU equivalence checks."""
    image = _rgb(image)
    h, w = image.shape[:2]
    return np.asarray(Image.fromarray(image).crop(crop_box(image.shape, crop_ratio)).resize(
        (w, h), Image.Resampling.BILINEAR)).copy()


def _mask(value, shape):
    value = np.asarray(value)
    if value.dtype != bool or value.shape != shape:
        raise ValueError('Expected boolean mask with the RGB spatial shape')
    return value


def _mask_transform(mask, *, nearest=False):
    h, w = mask.shape
    # Float coverage avoids confusing uint8 rounding to 255 with complete
    # interpolation support inside an object. Nearest is a separate diagnostic.
    value = mask.astype(np.uint8) if nearest else mask.astype(np.float32)
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(value).crop(crop_box(mask.shape)).resize((w, h), method))


def _max_interior_depth(mask, changed):
    """Chebyshev distance in pixels from the exterior, 1 at the inner boundary."""
    current = mask.copy()
    depth = maximum = 0
    while np.any(current & changed):
        depth += 1
        maximum = depth
        pad = np.pad(current, 1, constant_values=False)
        current = np.logical_and.reduce([pad[i:i + mask.shape[0], j:j + mask.shape[1]]
                                        for i in range(3) for j in range(3)])
    return maximum


def policy_input_diagnostics(raw_hwc, painted_hwc, protected_mask_hw, ring_mask_hw,
                             *, upstream_path=None, crop_ratio=CROP_RATIO):
    raw, painted = _rgb(raw_hwc), _rgb(painted_hwc)
    if raw.shape != painted.shape:
        raise ValueError('Raw and painted images must have the same shape')
    protected = _mask(protected_mask_hw, raw.shape[:2])
    ring = _mask(ring_mask_hw, raw.shape[:2])
    identity = preprocessing_identity(upstream_path)
    before = xr1_crop(raw, upstream_path=upstream_path, crop_ratio=crop_ratio)
    after = xr1_crop(painted, upstream_path=upstream_path, crop_ratio=crop_ratio)
    source_changed = np.any(raw != painted, axis=-1)
    changed = np.any(before != after, axis=-1)
    nearest = _mask_transform(protected, nearest=True).astype(bool)
    coverage = _mask_transform(protected)
    full = coverage >= 1.0 - 1e-7
    ring_support = _mask_transform(ring) > 0
    left, top, right, bottom = crop_box(raw.shape, crop_ratio)
    total = int(ring.sum())
    retained = int(ring[top:bottom, left:right].sum())
    bleed = changed & nearest
    return {'schema': SCHEMA, 'preprocessing': identity,
            'source_shape': list(raw.shape), 'crop_box_xyxy': [left, top, right, bottom],
            'source_inside_changed_px': int((source_changed & protected).sum()),
            'source_non_ring_changed_px': int((source_changed & ~ring).sum()),
            'source_ring_intersects_protected_px': int((ring & protected).sum()),
            'source_changed_px': int(source_changed.sum()),
            'ring_pixels_before_crop': total, 'ring_pixels_retained_by_crop': retained,
            'ring_retention_fraction': retained / total if total else None,
            'ring_pixels_lost_to_crop': total - retained,
            'crop_ring_support_px': int(ring_support.sum()),
            'crop_ring_support_with_visible_change_px': int((ring_support & changed).sum()),
            'crop_ring_has_visible_change': bool(np.any(ring_support & changed)) if total else None,
            'crop_changed_px': int(changed.sum()),
            'crop_changed_inside_nearest_mask_px': int(bleed.sum()),
            'crop_changed_full_coverage_interior_px': int((changed & full).sum()),
            'crop_boundary_bleed_max_interior_depth_px': _max_interior_depth(nearest, bleed),
            'crop_boundary_bleed_definition': 'Changed pixels inside nearest-resized protected mask; Chebyshev depth is 1 at its inner boundary.',
            'full_coverage_definition': 'Float-mode bilinear mask coverage >= 1 - 1e-7; excludes mixed boundary pixels.',
            'raw_crop_sha256': _sha(before.tobytes()), 'painted_crop_sha256': _sha(after.tobytes()),
            'claims': {'source_protection': not bool(np.any(source_changed & protected)),
                       'post_crop_nearest_interior_unchanged': not bool(np.any(bleed)),
                       'post_crop_full_coverage_interior_unchanged': not bool(np.any(changed & full)),
                       'task_improvement_evaluated': False}}


def write_contact_sheet(path, raw_3hwc, painted_3hwc, *, camera_names=None, labels=None,
                        protected_masks=None, ring_masks=None, upstream_path=None,
                        scale=2, overwrite=False):
    """Save raw/painted/exact-client-crop rows; return provenance and metrics.

    Optional masks are sequences matching cameras. Scale affects presentation
    only, using nearest-neighbor; all metrics use the true client resolution.
    """
    raw, painted = np.asarray(raw_3hwc), np.asarray(painted_3hwc)
    if raw.ndim != 4 or painted.shape != raw.shape or raw.shape[-1] != 3:
        raise ValueError('Expected matching NxHxWx3 image stacks')
    names = list(camera_names or CAMERAS[:len(raw)])
    if len(names) != len(raw) or not isinstance(scale, int) or scale < 1:
        raise ValueError('Camera count or presentation scale invalid')
    identity = preprocessing_identity(upstream_path)
    cropped_raw = np.stack([xr1_crop(x, upstream_path=upstream_path) for x in raw])
    cropped_painted = np.stack([xr1_crop(x, upstream_path=upstream_path) for x in painted])
    metrics = None
    if (protected_masks is None) != (ring_masks is None):
        raise ValueError('Supply both protected_masks and ring_masks, or neither')
    if protected_masks is not None:
        if len(protected_masks) != len(raw) or len(ring_masks) != len(raw):
            raise ValueError('Mask camera count differs from images')
        metrics = {name: policy_input_diagnostics(raw[i], painted[i], protected_masks[i], ring_masks[i],
                                                upstream_path=upstream_path)
                   for i, name in enumerate(names)}
    rows = [raw, painted, cropped_raw, cropped_painted]
    row_labels = list(labels or ('Source raw', 'Source painted', 'XR1 client raw crop', 'XR1 client painted crop'))
    if len(row_labels) != 4:
        raise ValueError('Four row labels are required')
    h, w = raw.shape[1:3]
    header = 24
    canvas = Image.new('RGB', (len(raw) * w * scale, 4 * (h * scale + header)), '#181c24')
    draw = ImageDraw.Draw(canvas)
    for row_index, (stack, label) in enumerate(zip(rows, row_labels)):
        y = row_index * (h * scale + header)
        for col, (im, name) in enumerate(zip(stack, names)):
            draw.text((col * w * scale + 5, y + 5), f'{label} | {name}', fill='white')
            tile = Image.fromarray(_rgb(im)).resize((w * scale, h * scale), Image.Resampling.NEAREST)
            canvas.paste(tile, (col * w * scale, y + header))
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Preserving existing contact sheet: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {'schema': SCHEMA, 'path': str(path), 'png_sha256': _sha(path.read_bytes()),
            'preprocessing': identity, 'camera_names': names, 'rows': row_labels,
            'presentation_scale': scale, 'presentation_resize': 'nearest',
            'source_sha256': _sha(raw.tobytes()), 'painted_sha256': _sha(painted.tobytes()),
            'policy_input_metrics': metrics, 'task_improvement_evaluated': False}
