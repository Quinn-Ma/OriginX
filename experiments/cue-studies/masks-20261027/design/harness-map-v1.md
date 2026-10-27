# MAP: presentation

**How cues are drawn into the policy inputs (id60 regions validation, CoffeeSetupMug, XR1)**

Everything below comes from reading the code statically, plus one read-only numpy look at the existing trace in `runs/id60-regions-validation-local-v1`. Nothing was modified and nothing was run in WSL. All paths are under `D:/codex/originx-harness-v2`.

**1. The presentation function F and call order**

F is `HierarchicalFocusController.render(raw, registry)` in `originx_harness/id60_hierarchical_revision/controller.py:102-130`. It returns `{**raw, 'rgb': marked[3,256,256,3] uint8, 'instruction': text}`.

Class chain (MRO), from the subclass down to the base:
1. `HierarchicalFocusController`
2. `PlacementContractController` (`id60_placement_revision/controller.py:9`)
3. `GraspFeasibilityController` (`id60_grasp_revision/feasibility_controller.py:102`)
4. `SelectorPatchedController` (`id60_revision/parser_controller.py:17`)
5. `FocusController` (`id60/controller.py:23`)

Only `FocusController` and `HierarchicalFocusController` define `render`.

Per step, in `originx_harness/id60_regions_validation/runtime.py::run_case`:
1. `env.step` → `ID60RoboCasa.step` (`id60/robocasa.py:140-144`). This does `canonical_observation`, then `extractor.observe(shape)`, then `registry.update(..., frame=steps)`.
2. Line 119: `augmentation = augmenter.observe(raw['rgb'].shape, env.registry)`, using `HierarchicalGeometryAugmenter` (`geometry.py:223`).
3. Line 121: `ctrl.set_geometry(augmentation, env.registry.frame)`. At `controller.py:33` this keeps the augmentation with `audit={}` stripped.
4. Line 124: `public = {rgb, proprio, instruction, step, episode_id, camera_names}`.
5. Branch on the step:
   - When `steps % 16 == 0` (line 129): `ctrl.prepare(public, registry, steps)`. This runs `Hierarchical.prepare:85`, then `Placement.prepare:21` (sets `_placement_registry`), then `GraspFeasibility.prepare:115`, then `FocusController.prepare` (`id60/controller.py:158-260`). That handles the planner plan/step calls, `BindingManager`, and `compile_instruction` (line 253), then calls `self.render(raw, registry)` at line 257, which dispatches to F.
   - Otherwise (line 158): `ctrl.render(public, registry)`. If the instruction changes between boundaries, line 160 raises.
6. Lines 141-152: when the instruction changes, the policy is reset with episode id `case_id/segment-k`.
7. Lines 162-165: the policy observation is `{rgb, proprio, instruction, step - segment_start, episode_id}`. `camera_names` is dropped here.
8. Line 174: `policy.observe`. Line 176: `propose` every 16 steps.

Inside F:
- **Base call:** `base = super().render(raw, registry)`. This is `FocusController.render` (`id60/controller.py:262-277`): `render_focus(raw['rgb'], registry, snapshot, camera_names=raw['camera_names'])` plus `compile_instruction(stage, registry, snapshot)['text']`.
- **Early return:** lines 104-106 return `base` unchanged unless `stage.operation == 'place'`. So the grasp stage is pixel- and text-identical in both arms.
- **Place-stage context:** lines 109-112 temporarily swap `_placement_registry = registry`, compute `placement = self._context()`, then `hierarchy = build_hierarchical_context(...)`.
- **Regions arm:** lines 115-117: `if self.arm in ('regions','feedback'): output, overlay = render_hierarchical_overlay(output, hierarchy, camera_names)` and `text += ' ' + HIERARCHICAL_LEGEND`.
- **Receipt:** lines 122-129 store `_last_hierarchical_render`. `prepare` (lines 93-96) copies it into `receipt['hierarchical_focus']` and `receipt['rgb_changed']`.

**2. Cues drawn in both arms: `render_focus` (`id60/focus.py:48-75`)**

- Colours: `PRIMARY_COLOR = (255,176,0)` orange (line 13), `RELATED_COLOR = (0,220,255)` cyan (line 14).
- Nothing is drawn if `binding.mode == 'native'` or `status == 'complete'` (line 65).
- For each camera, related entities are drawn first in cyan with width `max(1, ow-1) = 1` (lines 70-71). The target is drawn afterwards in orange with width 2 (lines 72-73).
- `_draw_entity` (lines 24-45):
  - A `bbox` view draws a rectangle for objects/parts and an ellipse for regions (lines 30-37).
  - A `point` view draws a ring of radius `max(3, round(min(W,H)*.025))` (6 px at 256), plus a crosshair of half-length radius+2 (lines 38-45).
- What this means for CoffeeSetupMug:
  - Grasp stage (target is the mug object, bbox only): a 2 px orange **rectangle**, not a crosshair.
  - Place stage (target region `R00a128035d`, "dispenser receiving region of coffee machine", point only): the orange **ring and crosshair**.
- In the actual plan (`initial_plan` in the result JSONs), both stages have `related_entity_ids: []`, so no cyan related-entity boxes are drawn at all. The place-stage subject (the mug) is unmarked in the point arm.

**3. Cues drawn only in the regions arm: `render_hierarchical_overlay` (`id60_hierarchical_revision/focus.py:16-56`)**

- Colours: `ANCHOR_COLOR` cyan (line 7), `REGION_COLOR` orange (line 8).
- Input checks: `[C,H,W,3]` uint8, one distinct camera name per frame, then `validate_hierarchical_context`.
- If `hierarchy['valid']` is false, it draws nothing and returns `status='unavailable'`.
- **Neighbour lattice:** each visible `region_cells` row in this camera with `grid_index != (0,0)` is drawn as **one pixel** with `draw.point` (lines 43-47). The (0,0) centre is not redrawn; the original crosshair stands in for it.
- **Cyan object proxy:** the `subject_anchor` is an outline circle of radius 3, width 1 (lines 48-51).
- **Legend:** `HIERARCHICAL_LEGEND` (lines 9-11) is appended to the text by the controller.
- The geometry comes from `geometry.py`:
  - `GRID` is 3x3 (line 29). The lattice spacing is 0.15 × the subject parent's shortest root-local xy AABB extent (`_read_object_geometry`, lines 125-149).
  - The centre cell copies the original region point views (`_original_center_views`, lines 208-221).
  - Neighbour cells are projected (`_projection`, lines 151-164) and occlusion-tested with `mj_ray` (`_visible_views`, lines 179-206).
  - The anchor is the subject's root-local geom-AABB centre (lines 232-244).
  - `build_hierarchical_context` (lines 303-358) exports only current visible normalized points.

**4. How instruction text is built per stage (`compile_instruction`, `id60/focus.py:85-120`)**

The parts, in order:
1. The planner's `stage.instruction`.
2. `"Current subtask: {operation}."`
3. `"Focus on the {label}[ on the {parent label}] ({kind})."`
4. The fixed orange-marker sentence.
5. Optionally, `"Related context, marked in cyan where visible: …"`.

It is empty or base-only when native/complete. The regions arm adds `' ' + HIERARCHICAL_LEGEND` in the place stage only. The text depends only on the stage and binding, never on visibility; that is what keeps it constant between prepare boundaries.

Text observed in the run (regions arm):
- Grasp: "Pick up the mug from the counter. Current subtask: grasp. Focus on the mug (object). An orange marker identifies this target …"
- Place: "Place the mug under the coffee machine dispenser. Current subtask: place. Focus on the dispenser receiving region of coffee machine on the coffee machine (region). An orange marker … When visible, cyan marks the object's geometric center proxy; discrete orange neighboring reference points …"

**5. Coordinate conventions**

- **Camera order** is fixed everywhere: `('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')`. It appears in `id60/robocasa.py:14`, `geometry.py:26` and `evaluation.py:37` (`video.*` keys). `canonical_observation` (`evaluation.py:270-280`) stacks them to rgb `[3,256,256,3]` uint8. XR1 `_observe_native` (`adapters.py:458-465`) indexes `rgb[i]` by `entry.CAMERA_KEYS` in the same order, with a deque of 7 and `sample_history(…, 4, 2)` (lines 455 and 467-471).
- **Orientation:** RGB is flipped vertically by the RoboCasa gym wrapper (`img[::-1]`, `gym_wrapper.py:270-273` in the asset-pack copy), so it is upright. The extractor's segmentation is flipped the same way (`robocasa.py:100`, `sim.render(..., segmentation=True)[::-1]`). Projection uses robosuite `get_camera_transform_matrix`, with an OpenCV-style axis correction, so u is the column and v is the row from the top of the upright image.
- **Check against the trace:** in the first case, step-0 trace (`000007-observation.npz`), the pixels changed in camera 0 span x 175-197, y 162-186. The logged segmentation of geom 1357 (the mug) spans x 176-197, y 163-186. So segmentation and RGB are aligned to about 1 px.
- **Stored geometry** is normalized to [0,1] as `[x/w, y/h]`:
  - Region points: `robocasa.py:124`.
  - Object bboxes from the segmentation mask: `[xmin/w, ymin/h, (xmax+1)/w, (ymax+1)/h]` (`robocasa.py:108`).
  - Lattice points: `geometry.py:160`, rejected when ≥1.
- **Drawing** converts back with `round(v*(W-1))`, so there is up to 1 px of drift near the right/bottom edge.

**6. What is available per camera at draw time**

Available:
- `raw` (`public`): rgb, proprio, instruction, step, episode_id, `camera_names`.
- `registry` (`EntityRegistry`): `Entity.views[camera]` holds only `{'bbox'}` and/or `{'point'}` (`entities.py:85-96`), plus `visible`. Visibility rules:
  - Objects and parts are visible if their segmentation mask has at least 3 pixels (line 107).
  - Regions are visible if the point projects in-frame, passes a ray-occlusion test with 0.06 m tolerance, and the parent is visible in that camera (lines 110-124).
- `self._augmentation` (`GeometryAugmentation`: `frame`, `entities`, `region_children`, `object_anchors`; `audit` stripped). It is used only if its frame equals `registry.frame`.
- `self.manager.stage` / `self.manager.snapshot`.
- `self._context()`: subject comes from `previous_bound_grasp` (`placement/controller.py:44-74`); destination is `snapshot.target_id`.

Not available:
- **No segmentation masks reach the controller.** The masks built at `robocasa.py:105` (`(seg[:,:,0]==5) & np.isin(seg[:,:,1], spec.geoms)`) are reduced to a bbox and thrown away.
- `DiagnosticRoboCasa.segmentation()` (`id60_revision/environment.py:28-33`) returns `{cam: (256,256,2) int32 [objtype, objid]}` but is logging-only. `run_case` lines 167-168 call it only when `steps % 16 == 0`, and only for the trace.
- No depth.

Region specs (`_Spec`) have `site`/`body+offset` and `geoms=()`. The receptacle site in the coffee machine XML has size 0.01 and rgba alpha 0. **So there is no native segmentation mask for the target region; one would have to be built from geometry.**

**7. How the arm is selected and registered**

- **Runtime:** `ARMS = ('point','regions')` (`id60_regions_validation/runtime.py:14`); `run_case` lines 49-51 reject any other `case['comparison_arm']`. Lines 65-70 default `controller_factory` to `HierarchicalFocusController` and `augmenter_factory` to `HierarchicalGeometryAugmenter`. Both are injectable, but the driver never passes them (`scripts/run_id60_regions_validation_local.py:203-207`). Line 101 builds the controller with `controller_factory(planner, 'full', arm=arm)`.
- **Controller:** `ARMS = ('point','regions','feedback')` (`controller.py:13`); `__init__` lines 17-19 raise on anything else.
- **Manifest** (`id60_regions_validation/registration.py`):
  - `cases()` lines 56-71 build 12 pairs with alternating order (line 62: `ARMS` or its reverse).
  - `treatments` dict at lines 100-101 (`point`, `regions`).
  - `NEW_REQUIRED` at lines 15-19.
  - `source_files()` at lines 74-84 makes every parent-manifest source frozen by hash. That list includes `id60/focus.py`, `id60/robocasa.py`, `id60/controller.py` and all of `id60_hierarchical_revision/*`.
  - `SHARED_CONTROLS` at lines 20-29 (includes `runtime_success_threshold_access: False`).
- **Verifier identity:** `presentation_identity()` (`scripts/serve_id60_hierarchical_verifier.py:140-160`) hashes `id60_hierarchical_revision/controller.py` (`CONTROLLER_MODULE`, line 19). The driver's `check_services` (lines 96-111) checks that identity.
- **Consequence:** editing any frozen file (`controller.py`, `focus.py`, `robocasa.py`, …) breaks both the source-hash check and the verifier-identity check. A new arm has to go in new files.
- The verifier always sees raw, unmarked RGB (`_planner_raw`, `id60/controller.py:153-156`; `server_modifies_policy_markers: False`).

**8. Where a new per-stage segmentation-mask arm would plug in**

All of this is new files; nothing frozen changes.

**(a) New augmenter** — `MaskedGeometryAugmenter(HierarchicalGeometryAugmenter)`. This is the only component with simulator access: it gets `extractor.sim` and `extractor.specs` through `augmenter_factory(env.extractor)` at `run_case:87`.
- Override `observe(rgb_shape, registry)`: call `super().observe`, then for each camera `seg = self.sim.render(width=w, height=h, camera_name=cam, segmentation=True)[::-1]`.
- Object masks: `(seg[...,0]==5) & np.isin(seg[...,1], spec.geoms)`, the same rule as `robocasa.py:105`, giving bool `[H,W]` per object id per camera.
- Return a frozen dataclass subclass, e.g. `MaskedGeometryAugmentation(GeometryAugmentation)` with an extra `masks` field. This survives `set_geometry`'s `replace(..., audit={})` (`controller.py:33`) and stays out of `geometry_record`/`shared_augmentation_sha256` unless a mask hash is added to `audit` on purpose.
- Use the same augmenter in both arms, so pairing signatures and the verifier context stay the same.
- The target-region mask has to be built geometrically, because the site has no geoms. One way: project the subject mug's root-local AABB (`self._geometry[mug].low_local` / `high_local`, already computed at `geometry.py:118-123`) placed at `site_xpos`, take the convex hull, fill it, then occlusion-test with depth (`sim.render(..., depth=True)`) or rays.
- **Do not size the region from the scorer threshold.** `check_receptacle_placement_for_pouring` uses `xy_thresh=0.04` and `|dz|<0.10` on the mug root body, and success also requires `gripper_obj_far`. Using those values would break `runtime_success_threshold_access: False`.

**(b) New controller** — `MaskFocusController(HierarchicalFocusController)`.
- `__init__`: accept `arm` in `('point','regions','masks')`. For `'masks'`, set `self.arm` and call `PlacementContractController.__init__(self, planner, variant, confirmations)` directly, to get past the check at line 18. With `arm='masks'`, the regions/feedback branches (lines 115-121) and the feedback consume path (line 67) stay off.
- Override `render(raw, registry)`:
  1. `out = super().render(raw, registry)`, which keeps the point-arm cues and text.
  2. Return early if not `_render_enabled`, the snapshot or stage is `None`, or the augmentation frame is not `registry.frame`.
  3. Grasp: mask the snapshot's `target_id`, the mug object.
  4. Place: swap `_placement_registry` like lines 109-111, then mask `placement['subject']['id']` (held mug) and the synthesized `placement['destination']['id']` region.
  5. Draw on `out['rgb']` per camera index, in `raw['camera_names']` order.
  6. Append a fixed per-operation legend to `out['instruction']`. It must not depend on per-frame visibility, or `run_case` line 160 raises.
  7. Store `_last_mask_render` and copy it into the receipt in an overridden `prepare`, mirroring lines 93-96.
- To *replace* rather than add to the orange cues, build the text yourself with `compile_instruction(...)` and draw on `raw['rgb']` instead of calling `super().render`. The base text still says "An orange marker identifies this target", so it would need adjusting.
- `ExternalMarkerController` (`id60_grasp_revision/focus.py:101-113`) is an existing example of exactly this kind of render-only override.
- A filled tint changes the object pixels the policy grasps from; the earlier external-bracket study protected bbox interiors for that reason. A mask *contour* is the safer choice.

**(c) New runtime, registration and driver**
- Copy `id60_regions_validation/runtime.py` with its own `ARMS` (for example `('point','masks')` or `('regions','masks')`) and `PURPOSE`.
- Copy `registration.py` with new `NEW_REQUIRED`, a new `treatments` entry and a new unused-seed audit. The 12 seeds in `work/id60-regions-validation/unused-seeds-v1.json` are now used.
- Copy the driver, passing `controller_factory=MaskFocusController, augmenter_factory=MaskedGeometryAugmenter` at the call that corresponds to lines 203-207.
- The verifier and policy services stay unchanged and `check_services` still passes. `presentation_identity`'s descriptive `policy_marker_treatment` will not list the new arm.
- Rendering 3 more segmentation frames per step (and depth, if used) adds cost. Both arms pay it if the augmenter is shared.

# MAP: geometry-seg

# Simulator geometry and segmentation for the id60 regions-validation path (CoffeeSetupMug, XR1, 3 cams at 256x256)

**Bottom line:** the harness already renders a per-geom MuJoCo segmentation image for all 3 cameras on every step, at the policy-input size, flipped to match the policy RGB. It then throws the pixels away and keeps only a bbox. Depth is not produced anywhere, but the same render call can return it. The receiving region has no pixels of its own: it is only a site point, so its 3D footprint has to be built from geometry and occlusion-tested against a depth image.

## 0. Call order (what exists, and when)

**Reset**
1. `run_case` (`originx_harness/id60_regions_validation/runtime.py:82`) calls `env.reset`.
2. `DiagnosticRoboCasa.reset` (`id60_revision/environment.py:15`). The driver builds the env with `part_geometry=False` (`scripts/run_id60_regions_validation_local.py:196`), so this falls through to `ID60RoboCasa.reset` (`id60/robocasa.py:133-138`):
   - `RoboCasaEntities(self.env.env.unwrapped.env)` takes the robosuite Kitchen env; `.sim` is the `MjSim`.
   - `registry.update(extractor.observe(raw['rgb'].shape), 0)`.
3. `runtime.py:87-88` constructs `HierarchicalGeometryAugmenter(env.extractor)` and calls `.observe(raw['rgb'].shape, env.registry)`.

**Each step `s`** (`runtime.py:114-198`)
1. If `s > 0`: `augmenter.observe(...)` (line 119). This does projection and `mj_ray` only, no rendering.
2. `ctrl.set_geometry` (121), then trace event `shared_hierarchical_geometry` (122).
3. If `s % 16 == 0`: `ctrl.prepare(public, registry, s)` (129). Otherwise `ctrl.render(public, registry)` (158). Overlays are drawn here.
4. If `s % 16 == 0`: `env.segmentation()` is logged (167-168).
5. `policy.observe(observation)` (174), then `policy.propose` every 16 steps (176).
6. `env.step` (186) runs `ID60RoboCasa.step` (`robocasa.py:140-144`), which does `env.step`, then `canonical_observation`, then `registry.update(extractor.observe(...), steps)`. Segmentation is rendered here, after physics and with the same sim state as the obs RGB.

**Where new masks would go:** they must be computed after `env.step`/`env.reset` and before `ctrl.render`/`prepare`. The natural places are inside `observe`, or next to `runtime.py:119`.

## 1. Entity identities

`RoboCasaEntities.__init__` (`id60/robocasa.py:42-92`):
- **Objects:** for every `base.fixtures` / `base.objects` entry, `spec.geoms` is every geom whose name starts with `obj.naming_prefix` (line 54). The ID is `opaque_id('object', collection+'/'+name)` = `'O' + sha256[:10]`.
- **Parts:** regex groups over geom-name suffixes (lines 61-68).
- **Regions:** a site whose name contains `place_site` or `receptacle` gets `_Spec(..., site=sid)` (lines 71-76). Otherwise, `get_reset_regions` offsets give `body=root, offset=...` (lines 81-92).

IDs checked against the logged traces:

| Entity | ID | Source | Geoms in seed-20261034 episodes |
|---|---|---|---|
| mug | `O970ccd9b5f` | `objects/obj` | ids 1750–1770 (21 geoms) |
| coffee machine | `O3c752db840` | `fixtures/coffee_machine_right_group_1` | ids 1402–1458 (57 geoms) |
| receiving region | `Rc56bdb0edb` | `coffee_machine_right_group_1_receptacle_place_site` | n/a (site) |

`Entity.views` only accepts the normalized keys `bbox` / `point` (`id60/entities.py:85-86`). Metadata must be JSON-like (line 39). So masks cannot go through `Entity` or `EntityRegistry`; they need a separate side channel.

`HierarchicalGeometryAugmenter` (`id60_hierarchical_revision/geometry.py:77-288`) adds:
- per object, `_ObjectGeometry`: `root_body`, root-local AABB `low`/`high`/`center` from `geom_aabb` + `geom_xpos`/`xmat` (lines 125-149), `spacing = 0.15 * min planar extent`, and the `geoms` tuple;
- `object_anchors` (`P…`) and `region_children` (9 `R…` cells per region, `GRID` 3x3 in the parent root x/y axes, lines 108-111, 259-268).

## 2. What is rendered today

**Segmentation, every step, 3 cams, at policy resolution** (`id60/robocasa.py:99-108`):
```python
seg=self.sim.render(width=w,height=h,camera_name=cam,segmentation=True)[::-1]
mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],spec.geoms)   # 5 = mjOBJ_GEOM
```
- **Shape:** int32 `(256,256,2)` holding `[objtype, objid]`. Background is `(-1,-1)` and sites are objtype 6. This comes from `MjRenderContext.read_pixels`, `vendor/robosuite/robosuite/utils/binding_utils.py:169-193`; `MjSim.render` is at 1109-1146.
- **What's kept:** only a bbox (needs at least 3 pixels). The mask is discarded.
- **Logged copy:** `DiagnosticRoboCasa.segmentation()` (`environment.py:28-33`) renders the same thing again. It is stored every 16 steps in the trace npz as `payload.raw.segmentation.<cam>`, `<i4 [256,256,2]`. Example: `runs/id60-regions-validation-local-v1-resume1/worker-0/run/traces/<case>/NNNNNN-observation.npz`.
- **Geom groups:** collision geoms (group 0) are not rendered, because `vopt.geomgroup[0]=0` (`vendor/robosuite/robosuite/environments/base.py:381-382`). Visual meshes (group 1) are rendered.
- **Alpha-0 geoms and sites:** checked in the traces, they do not appear. At step 304 of case `2bab…`, the mug sits under the dispenser inside the coffee machine's alpha-0 `reg_main` box and is still visible (258 px, left camera). The receiving site (alpha 0) never appears; only robot sites (objtype 6) do.

**Depth is not produced anywhere.**
- `RoboCasaEnvironment.__init__` calls `gym.make("robocasa/"+task, split="pretrain", seed=7)` (`originx_harness/evaluation.py:372`) without `camera_depths`, so `create_env` defaults it to False (`vendor/robocasa/robocasa/utils/env_utils.py:106-112`).
- Even if enabled, `canonical_observation` (`evaluation.py:270-280`) passes through only the `video.*` RGB keys.

**How to get depth cheaply:** add `depth=True` to the existing render, which returns both buffers from one pass:
```python
seg, z01 = sim.render(width=256, height=256, camera_name=cam, depth=True, segmentation=True)
seg, z01 = seg[::-1], z01[::-1]                       # both are OpenGL bottom-up
from robosuite.utils.camera_utils import get_real_depth_map
D = get_real_depth_map(sim, z01)                      # (256,256) float32, camera z-depth in metres
```
- `get_real_depth_map` is at `vendor/robosuite/robosuite/utils/camera_utils.py:106-125`. It uses `near = vis.map.znear*extent` with `znear=0.001` (`models/assets/base.xml:11`).
- **Depth caveat:** MuJoCo here is 3.3.1. Check the depth conversion once (see §3a) before trusting depth tests.
- **Do not** turn on robosuite `camera_segmentations`: it sets every `site_size` to 1e-8 (`robot_env.py:370-371`), which changes the frozen policy RGB.

## 3. Recipes

### (a) Exact pixel mask of an object entity, all its geoms, per camera, current step

```python
sim = env.extractor.sim
spec = next(s for s in env.extractor.specs if s.id == obj_id)      # e.g. 'O970ccd9b5f'
seg = sim.render(width=W, height=H, camera_name=cam, segmentation=True)[::-1]
mask = (seg[...,0] == 5) & np.isin(seg[...,1], spec.geoms)          # bool (256,256)
```
- **Occlusion:** the mask is occlusion-correct by construction (visible pixels only). The gripper, sites, other objects and transparent visual meshes all occlude.
- **Ownership:** it is by name prefix. A more robust alternative is body-tree ownership (`geom_bodyid` descends from `obj.root_body`), as `part_geometry._descendant` does (`id60_revision/part_geometry.py:39-49`).
- **Full outline including occluded parts (amodal):** this needs a custom render. Replicate `binding_utils.py:129-193`, keep only the target object's geoms in `scn.geoms` after `mjv_updateScene`, then `mjr_render`.
- **Depth sanity check:** use the same render with `depth=True`, then compare `D[mask]` against `p[2]` of that object's `geom_xpos` projections.

### (b) Receiving region: 3D footprint and an occlusion-correct mask

**What the region actually is.** It is only a site, read as `sim.data.site_xpos[spec.site]` (3,). The site is defined at `vendor/robocasa/robocasa/models/assets/fixtures/coffee_machines/<asset>/model.xml`; for CoffeeMachine063 it is line 45:
```xml
<site rgba="0 0 0 0" size="0.02 0.02" pos="0.013 -0.238 -0.187" name="receptacle_place_site"/>
```
That is a sphere of radius 0.02 m, invisible, with no area and no geom. There is no other receiving-area geometry:
- `get_reset_regions()['bottom']` is the same position with size `(0.01,0.01)` (`models/fixtures/coffee_machine.py:48-57`). The extractor skips it because `has_receiving_site` is set (`robocasa.py:76,81`).
- `reg_main` (model.xml:86) is a whole-machine bounding box.

**The scorer's tolerance.** `check_receptacle_placement_for_pouring` (`coffee_machine.py:95-115`) passes when the mug body `body_xpos` is within 0.04 m of the site in xy and within 0.10 m in z. In the success trace `2bab…` at step 319, the mug was 0.0339 m from the site in xy and 0.0455 m above it. Using the 0.04 m radius would read a scorer tolerance, which `geometry.py`'s docstring says the harness never does. Task-neutral alternatives:
- the site's own radius, `sim.model.site_size[sid][0]`;
- a fixed constant;
- the mug's own footprint placed at the site (option 4 below).

**Support surface height.** Cast a ray straight down: `mujoco.mj_ray(model, data, site_xpos, [0,0,-1], None, 1, -1, hit)` gives `z_support = site_z - dist` and the tray geom id in `hit`.

**Occlusion-correct mask, analytic per-pixel method** (exact, numpy only). Pixel centres are at `(j+.5, i+.5)` in the flipped, top-down image:
```python
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
K  = get_camera_intrinsic_matrix(sim, cam, H, W)        # f = .5*H/tan(fovy/2), cx=W/2, cy=H/2
Tc = get_camera_extrinsic_matrix(sim, cam)              # cam->world, OpenCV axes (x right, y down, z fwd)
u, v = np.meshgrid(np.arange(W)+.5, np.arange(H)+.5)
ray = np.stack([(u-K[0,2])/K[0,0], (v-K[1,2])/K[1,1], np.ones_like(u)], -1) @ Tc[:3,:3].T
o = Tc[:3,3]; t = (z0 - o[2]) / ray[...,2]              # t equals camera z-depth, because ray_cam.z = 1
P = o + t[...,None]*ray
inside  = (t > 0) & (np.linalg.norm(P[...,:2] - c_xy, axis=-1) <= r)   # or a polygon in the parent root frame
visible = inside & (t <= D + tol)                        # D from §2; tol a few mm (z-fighting at the surface)
```
- **Ghost of the mug at the target (option 4):** replace the disc with a ray-vs-box slab test against the mug's root-local AABB (`augmenter._geometry[mug].low/high`, `geometry.py:68-75,144-149`). Translate it to the site's xy with its bottom at `z_support`.
- **Rendering alternative:** in a custom copy of `binding_utils.py:129-167`, after `mjv_updateScene`, add a thin disc with `mujoco.mjv_initGeom(scn.geoms[scn.ngeom], …)`, set `segid` and `scn.ngeom += 1`, then render in segment mode. The GL depth test then gives the occlusion. The scene is rebuilt on every call, so the policy RGB is never touched.

**Do not use `mj_ray` for pixel visibility.**
- The extractor's region visibility (`robocasa.py:116-124`, tolerance 0.06 m) and the lattice rays (`geometry.py:166-206`, tolerance `spacing*1e-3`) use `mj_ray` with `geomgroup=None`. That tests the semi-transparent collision boxes, which are not rendered. Ray visibility therefore does not match what the camera sees; use the depth buffer from the same render.
- Once the mug is placed, the site is occluded by the mug. In case `2bab…` at step 304 the region's `views` were `{}`, so `build_hierarchical_context` returns `destination_center_hidden` (`geometry.py:354-356`) and the regions overlay goes `'unavailable'`.

## 4. Orientation, flips and crop the masks must match

1. **Policy RGB:**
   - robosuite renders bottom-up with `IMAGE_CONVENTION="opengl"`, so no flip in the sensor (`macros.py:28`, `robot_env.py:405,417-429`).
   - `RoboCasaGymEnv.get_basic_observation` flips it vertically with `img[::-1,:,:]` (`vendor/robocasa/robocasa/wrappers/gym_wrapper.py:271-278`).
   - The size is 256x256 (`gym_wrapper.py:46`).
   - `canonical_observation` stacks the cameras in order left, right, wrist into uint8 `(3,256,256,3)` (`evaluation.py:37,272`).
   - No horizontal flip, resize or crop at this stage.
2. **Segmentation and depth:** apply `[::-1]` (rows only). That is exactly what `robocasa.py:100`, `environment.py:31-32` and robosuite's `get_camera_segmentation` (`camera_utils.py:103`) do.
3. **Projection:** `get_camera_transform_matrix(sim,cam,H,W)` (`camera_utils.py:67-87`, includes the y/z axis correction at 60-63) gives `u = p0/p2` as the column and `v = p1/p2` as the row from the top. Both are already in the flipped frame; no extra flip is needed. Two small pixel offsets exist in the current code:
   - The harness stores `u/w`, but the overlay maps it back with `round(p*(w-1))` (`id60_hierarchical_revision/focus.py:43,49`; `id60/focus.py:32-40`). That can be off by up to 1 px from `floor(u)`.
   - `entities.project_parent_geometry` divides by `(w-1)` (`entities.py:284`).
   For pixel-exact masks, index the 256x256 arrays directly.
4. **The model's own crop, a hidden transform:**
   - The XR1 adapter builds `EvalClient(..., "robocasa365", 0.95)` (`originx_harness/adapters.py:443`).
   - Every frame is passed through `center_crop`: box `[6, 249)` on both axes, then a bilinear resize back to 256 (`work/id60-revision/asset-pack/upstream-validation/Xiaomi-Robotics-1/eval_robocasa365/entry.py:92-104,136`).
   - So the model sees `x' = (x-6)*256/243`. The 6–7 px border strip is lost, and 1-px marks such as the `draw.point` lattice dots (`focus.py:46`) get blurred.
   - The history it samples is frames `t, t-2, t-4, t-6` (`entry.py:81-89`; `adapters.py:468-469`).
5. **Eye-in-hand camera:** it moves with the arm, so recompute `cam_xpos`/`xmat` and K/T every step, as the existing code already does.

## 5. Checks run on logged traces (read-only, Windows Python, no numpy)

- **Alignment:** I compared segmentation edges against RGB gradients at step 304 of case `2bab…`. The ratio of RGB gradient on seg edges to the mean gradient was 5.7–8.7 for the segmentation as logged, against 1.0–1.9 when it was flipped vertically or horizontally. The logged segmentation lines up with the raw and policy RGB with no further flip.
- **Projection:** the projected receiving-site points at step 0 of case `285e…` are left (167,156), right (108,159) and wrist (162,127). All three land on coffee-machine geoms 1449/1435 (tray/base).

The scratch scripts are in `C:\Users\LUCKYQ~1\AppData\Local\Temp\claude\D--codex-2026-10-08-new-chat\e6bc7a5c-e11e-439d-a833-cf3dbdfdb40e\scratchpad\` (`inspect_seg4.py`, `align_check.py`, `pix.py`).

Run context: MuJoCo 3.3.1, EGL through WSL D3D12/NVIDIA, robosuite/robocasa loaded from `runs/gpu3-local5090-20261022/vendor` via `site.json` pythonpath.

# MAP: pipeline

# Regions-validation pipeline: how it was registered and run, and how a point-vs-masks study can copy it

Everything below comes from reading files and from read-only scans with the Windows python. I modified nothing and ran no WSL, simulator or model. All paths are relative to `D:/codex/originx-harness-v2`.

## 1. Registration: `originx_harness/id60_regions_validation/registration.py`

**Constants**
- `PARENT='runs/id60-hierarchical-local-resident-v1/manifest.json'` and `PARENT_HASH='8d364a…'` (L9-10).
- `SEED_AUDIT='work/id60-regions-validation/unused-seeds-v1.json'` and `SEED_AUDIT_SHA='4b1ac191…'` (L11-12). The audit's SHA is written into the source by hand, so the seed audit had to run before the registration file was final.
- `MAX_SECONDS=28800` (L13) and `PAIR_COUNT=12` (L14).
- `NEW_REQUIRED` (L15-19) lists five files: `id60_regions_validation/{__init__,runtime,registration}.py`, `scripts/run_id60_regions_validation_local.py` and `tests/test_id60_regions_validation.py`.
- `SHARED_CONTROLS` (L20-29) includes `native_action_shape [16,12]`, `control_replan_steps 16`, `model_inference_device cuda:0`, `model_residency resident`, the verifier `frozen_v1_shared_hierarchical_verifier`, `feedback_enabled False`, and no automatic retry or seed replacement.

**`seed_audit(root)` (L36-53)**
- Checks the file SHA against `SEED_AUDIT_SHA`.
- Requires `status=='PASS'`, `all_selected_absent is True`, 12 distinct non-negative ints, `ordered_seeds==selected_seeds`, and `selected_seed_prior_matches[str(s)]==[]` for every seed.

**`cases(seeds)` (L56-71)**
- `order = ARMS if pair_index%2==0 else reversed(ARMS)`. Even pairs run point then regions; odd pairs run regions then point, giving 6 of each order.
- Each case row is `{task:'CoffeeSetupMug', benchmark:'robocasa', model:'xr1', arm:'full', comparison_arm, seed, max_steps:600, pair_index, pair_position (0|1), order_in_worker=len(rows), worker:0, pair_key}`.
- `pair_key = canonical_hash({task, seed, model:'xr1', purpose:PURPOSE})`.
- `case_id = canonical_hash(row)`. Because `PURPOSE` is inside `pair_key`, a new study gets new case IDs automatically.

**`source_files(root, parent)` (L74-84)**
- Every parent `source_files` entry (133) must still match its sha256, otherwise "Frozen v1 source changed".
- Each `NEW_REQUIRED` file is then hashed from its current bytes, giving 138 entries.
- Any name containing `feedback_repair` is rejected.

**`build_manifest(root)` (L87-114)**
- Order: seed audit → parent read, with `manifest_sha256` and the canonical hash re-checked (L91) → manifest dict → `manifest_sha256=canonical_hash(...)` → `validate`.
- The manifest dict contains: cases, seeds, seed_audit, source_files, historical_source_count=133, parent `{path, sha256, manifest_sha256}`, `checkpoint_sha256` copied from the parent, `trace.max_bytes_per_case=2 GiB`, shared_controls, treatments, pairing, endpoints and limitations.

**`validate` (L117-131) and `load_manifest` (L134-147)**
- `load_manifest` re-runs `seed_audit`, re-hashes the parent, and checks `source_files(root, original) == value['source_files']` (L143-144).
- Consequence: any edit to a `NEW_REQUIRED` file after freezing, including the test file, makes the run impossible to start.

**Frozen artifacts**
- `work/id60-regions-validation/manifest.json`: canonical `9414bd65…`, file sha `ec686401…`.
- All 138 registered sources still match today (checked).

## 2. Seed audit: `work/id60-regions-validation/audit_unused_seeds.py`

**Scope**
- `EXCLUDE` (L15-19) lists cache, asset and vendor directories, **plus `id60_regions_validation` and `id60-regions-validation`**, so the study cannot contaminate its own audit.
- `KEYS` (L20) are JSON seed-like keys. `TOKEN = (?<!\d)(20\d{6})(?!\d)` (L21) conservatively treats any date-like 8-digit integer as "used".
- `candidates()` (L37-55) shells out to `rg --files`. `rg` is not on the Windows PATH in this shell. It scans json/py/yaml/yml/toml under runs, work, outputs, configs, scripts, originx_harness and tests.
  - The namespace filter is at L48.
  - Under `work`, only `work/id60*` is scanned (L49).
  - Under `runs`, only manifest, registration, amendment, results and `trace.json` files are scanned (L50-53).
- Files larger than 8 MiB are skipped; 7 were skipped.

**Selection (L77-80):** the first 12 integers ≥ 20261028 that have no evidence.
- 20261028-30 were rejected; they appear only as mutation values in `tests/test_id60_grasp_holdout_registration.py:71,105,118`.
- The result was 20261031-20261042. The receipt recorded 1761 scanned files and 83 history manifests.

**Receipt quirk:** L83 `{str(s):evidence[s] ...}` on a `defaultdict` inserts keys. As a result, `excluded_prior_seed_or_token_values` (L86) lists the 12 selected seeds as well. I confirmed that this list contains 20261031-42 while `previous_seed_fields` stops at 20261027. Use `evidence.get(s, [])` in any copy.

**Choosing 12 new seeds**
- A copy that keeps the old `EXCLUDE` and the L48 filter will **not see** the regions runs or work files, and will re-select 20261031-42.
- The copy must remove `id60_regions_validation` and `id60-regions-validation` from the exclusions and exclude its own namespace instead (`id60_masks_validation`, `id60-masks-validation`).
- My read-only token scan covered 21,140 json/py/yaml/toml/txt/tex/md files under the same roots and exclusions:
  - 20261031-42 appear only in regions-named paths.
  - 20261043 appears once, as the upper bound in `tests/test_id60_regions_validation.py:19 SEEDS = tuple(range(20261031, 20261043))`, so the conservative token rule excludes it.
  - Nothing appears from 20261044 upward.
- Expected selection under the same rule: **20261044-20261055**. This is a prediction; the audit itself must be run to produce the receipt.
- Optionally add an explicit assertion that the selection does not intersect {20261025..27} or {20261031..42}.

## 3. Runtime: `originx_harness/id60_regions_validation/runtime.py`

`PURPOSE='regions_point_unseen_seed_validation'` (L13), `ARMS=('point','regions')` (L14), `PairAdmissionError(ValueError)` (L17).

**Signature (L46-48)**
```python
run_case(case, env, policy, planner, *, trace, expected_initial=None, expected_plan=None,
         contact_reader_factory=None, deadline=None, engineering_steps=None, on_proposal=None,
         controller_factory=None, augmenter_factory=None, renderer_check=None)
```

**Entry guards (L49-64)**
- Rejects any arm outside `ARMS`.
- Requires `replan_steps==16` and `max_steps==600`.
- Position 0 must not receive a reference; position 1 must receive both `expected_initial` and a non-empty `expected_plan`.
- Real runs require both `contact_reader_factory` and `renderer_check`.

**Defaults (L65-70):** `HierarchicalFocusController` and `HierarchicalGeometryAugmenter`. The same augmenter is used for every arm.

**Call order**
1. `policy.identity()`; reject fixture policies.
2. `env.reset(seed, case_id)`, then `renderer_check(raw)`, which checks the NVIDIA GL identity and the visual health of the first RGB frame.
3. `env.initial_fingerprint()`, then `augmenter.observe(raw['rgb'].shape, env.registry)`.
4. `initial_signature` builds keys {initial_fingerprint, rgb, proprio, instruction, camera_names, step, candidates_visible_sha256, candidates_all_sha256}, and L90 adds `shared_augmentation_sha256`.
5. Position 1 only: `assert_initial_match`. A mismatch raises `PairAdmissionError` before any observation or action (L92-98).
6. `ctrl = controller_factory(planner, 'full', arm=arm)` (L101), then `ctrl.reset` and `policy.reset`.
7. Per step:
   - `augmenter.observe` and `ctrl.set_geometry`, plus a `shared_hierarchical_geometry` event.
   - Build `public = {rgb, proprio, instruction, step, episode_id, camera_names}`.
   - Every 16 steps: `capture_state` and `ctrl.prepare` (planner calls). At step 0 the natural plan must be non-empty, and in the second arm must equal the first arm's plan, otherwise `PairAdmissionError` (L131-140).
   - Between boundaries: `ctrl.render`, and the instruction must not change (L157-160).
   - `trace.capture_observation`; segmentation is logged every 16 steps (L167-168).
   - `policy.observe`; every 16 steps `policy.propose(seed)` must return `np.float32 (16,12)` and finite values (L178), and the chunk is captured.
   - `env.step(actions[steps%16])`.
   - Events `action_executed`, `contact_diagnostic` and `placement_geometry_diagnostic` on every step (L188-197).
   - Break on success or done.

**Data shapes**
- `raw['rgb']` is uint8 `[3,256,256,3]` for cameras `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`.
- proprio is f4 `[16]`.
- Entity views are normalized `bbox [x0,y0,x1,y1]` for objects and `point [x,y]` for regions.
- The natural plan is `[grasp O… (mug), place R… (coffee dispenser receiving region)]`.

**What differs per arm:** only `arm` passed into the controller, which changes the policy RGB and the legend text. The frozen controller's `render` (`controller.py` L102-130) adds the overlay and `HIERARCHICAL_LEGEND` only when `arm in ('regions','feedback')` and only during place stages.
- The planner and verifier receive raw RGB in both arms (`training_free.PlannerClient.call` L53 sends only `rgb`, `instruction`, `candidates` and kwargs).
- The hierarchical place context is built in all arms (`controller.py` L42-57).

## 4. Driver and execution

**`scripts/run_id60_regions_validation_local.py`**
- `--freeze` (L247-252) writes the manifest and refuses to overwrite.
- Otherwise:
  1. Requires 120..28800 s.
  2. Requires `os.name=='posix'` (WSL) (L257), `load_manifest`, `site_config`, and **at least 60 GiB free** (L260).
  3. Real runs must write inside `runs/`; dry runs may write anywhere (L263). The output directory must be new (`mkdir exist_ok=False`).
  4. Writes `local-execution.json`, `site.json`, `manifest.json` and `launch-plan.json`.
  5. `--dry-run` returns at this point (L285-286).
  6. `assert_port_free(20172, 20192)`, then `graphics_preflight`.
  7. Starts the policy service: `policy-python serve_id60_hierarchical_policy.py --config configs/id60-backend-local-xr1.json --port 20172`.
  8. Starts the planner/verifier: `planner-python serve_id60_hierarchical_verifier.py --weights runs/gpu3-local5090-20261022/weights/awomo/qwen --port 20192`.
  9. Starts the worker (`sim-python <driver> --worker`) and polls once per second.
  10. `finally`: `stop_owned_processes`, `launch-receipt.json`, `account_missing` (missing cases recorded as UNKNOWN), `case-summary.json`.
- `worker()` (L150-234):
  1. `validate_execution` (L130-147) checks the driver sha, site hash and watchdog.
  2. `PolicyClient(timeout=600)`, `FrozenPlanner`, then `check_services` (L96-111). This compares the verifier identity against `presentation_identity()`, which hashes `id60_hierarchical_revision/controller.py`.
  3. For each case in manifest order: `reference_for` (L73-93) reads the first arm's result for `expected_initial`/`expected_plan`; then `DiagnosticRoboCasa(task, 600, part_geometry=False)`, `TraceRecorder(traces/<case_id>, max_bytes=2 GiB)`, and `run_case`.
  4. Any error other than `PairAdmissionError` is re-raised and stops the whole run.

**Recorded launch command** (from `work/id60-regions-validation/runtime-launcher-checks-v1.json`):
`wsl.exe -d Ubuntu-22.04 --cd /mnt/d/codex/originx-harness-v2 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 …/runtime/bin/sim-python scripts/run_id60_regions_validation_local.py --manifest work/id60-regions-validation/manifest.json --output runs/id60-regions-validation-local-v1 --duration-seconds 28800 --residency resident`

**What actually happened**
- The run was a child of the Codex desktop app; when the app restarted at about 13:16 CDT, WSL stopped and the run was killed after 6 cases (`runs/id60-regions-validation-local-v1/interruption-record-20261024.json`).
- An unregistered resume driver was written afterwards: `scripts/run_id60_regions_validation_resume_local.py`. Its `prior_state` (L31-62) allows a re-attempt only when the trace has `records==0` and `last_step==-1`, and only for whole pairs. It ran the remaining 18 cases in `runs/…-resume1`.
- The last case (20261042, point arm) failed on the planner HTTP timeout (180 s, `training_free.py:55`) during `ctrl.prepare`, giving `unknown/timeout`.
- Results:
  - Point: 7 success, 4 failure, 1 unknown. Regions: 6 success, 6 failure.
  - 11 eligible pairs: 6 both succeeded, 4 both failed, 1 point-only success, 0 regions-only. McNemar p = 1.0.
- Results were then analyzed by `analyze_regions_validation_combined.py`.

**Wall-clock per episode** (23 completed cases)
- Successes (256-319 steps): 425-640 s, median 578 s.
- 600-step failures: 1064-1368 s, median about 1169 s.
- That is roughly 2 s per control step; the 23 cases summed to about 19,030 s (5.3 h).
- Service startup takes 5.5-7 min.
- Worst case for 24 failures is about 24 × 1370 s ≈ 9.1 h, which exceeds `MAX_SECONDS=28800`.

**Disk:** the traces used 179-372 MiB per case and 7.05 GiB in total. D: currently has about 71.9 GiB free, which is just above the 60 GiB check.

## 5. Frozen analyzer: `work/id60-regions-validation/analyze_regions_validation.py`

Its sha `d9cb8f46…` was frozen in `analysis-plan-v1.json` five seconds before launch.

**Hard-coded values**
- `PURPOSE` (L18) and `ARMS=('point','regions')` (L19).
- `GATE` is `work/id60-targeted-parser/analyze_targeted.py` (L20).
- `manifest_check` (L79-116) requires 12 seeds / 24 cases and the counterbalanced order derived from `ARMS`, and **hard-codes the parent `manifest_sha256` `8d364a…` (L111)**.
- `make_pairs` and `summaries` (L282-336) emit keys named `regions_case_id`, `regions_outcome`, `success_delta_regions_minus_point` and `final_xy_delta_cm_regions_minus_point`.

**Evidence assumptions**
- `initial_checks` (L179-196) rebuilds the initial signature from the journal using exactly the nine current keys, including `shared_augmentation_sha256` from the step-0 `shared_hierarchical_geometry` event. Adding any key to the signature turns every case into `invalid_evidence`.
- `final_xy` (L198-225) needs `contact_diagnostic` and `placement_geometry_diagnostic` events on every step from 1 to `steps`, and a plan of grasp followed by exactly one place.
- `gate.trace_integrity_issues` (`analyze_targeted.py` L126-166) needs a complete trace, matching counts, and observation, action, proposal and prepare steps at the expected cadence.
- Trace payloads must not contain `{'captured': False}` or non-finite floats (`tracing.py` L58-69); either makes the capture incomplete.

**Reusing it:** because the functions read the module globals `PURPOSE` and `ARMS` when called, a wrapper could set them to new values. That only works if the new manifest keeps parent `8d364a…`, and the output would still label masks results as `regions_*`. A copy is cleaner.

## 6. Tests and checks that were run

- **`tests/test_id60_regions_validation.py`: 35 tests.** The count is 4 parity, 4 initial-mismatch, 1 plan-mismatch, 1 empty plan, 1 matrix, 2 reference, 12 `validate_execution` mutations, 3 manifest mutations, 1 lineage (asserts 138 sources and the seed list), 1 UNKNOWN accounting, 1 spawn-failure cleanup, and 4 service identity.
  - It uses `h.Env`, `h.Augmenter`, `h.Controller`, `h.case` and `h.execute` from `tests/test_id60_hierarchical_runtime.py`, which loads `tests/test_id60_grasp_runtime.py`, with `engineering_steps=48`.
  - Receipt: `runtime-launcher-checks-v1.json` records 35 passed on Windows (13.0 s) and in WSL (15.0 s), freeze exit 0 and dry-run exit 0.
- **Dry runs:** `work/id60-regions-validation/dry-run-v1/` (`--dry-run` in WSL; writes 4 JSON files and starts no services) and `resume-dry-run-v1/`.
- **Analyzer tests:** `work/id60-regions-validation/test_regions_analysis.py` has 12 unittest cases. `verify_analysis_preparation.py` wrote `analysis-cpu-checks-v1.json` (10 tests plus 2 smoke checks against existing development episodes) and `development-context-and-power-v1.json`; `analysis-cpu-checks-v2.json` covers the 2 added tests.
- **Power at 12 pairs** (from `development-context-and-power-v1.json`): 0.18 at a risk difference of 1/3, 0.33 at 0.4, 0.61 at 0.5.
- **Independent review:** `independent-runtime-review-v1.json` returned NO_BLOCKER. `runtime-v1-parity.diff` diffs the frozen hierarchical runtime against the regions runtime.

## 7. Recommended minimal new files for point vs masks (no frozen file modified)

**Why new code is unavoidable**
- The frozen controller only knows `ARMS=('point','regions','feedback')` and raises on anything else (`controller.py:13,18`).
- The regions runtime rejects any arm outside `ARMS` (`runtime.py:50`).

**Files to add to `NEW_REQUIRED`, all written before freezing**
1. `originx_harness/id60_masks_validation/__init__.py`
2. `originx_harness/id60_masks_validation/masks.py` (can be merged into item 3)
   - Pure functions only.
   - Object mask: `(seg[...,0]==5) & np.isin(seg[...,1], spec.geoms)`, the same rule as `id60/robocasa.py:100-108`.
   - Destination-region mask: the region is only a site point, so a footprint has to be defined explicitly (for example, the subject object's footprint projected at the site with an occlusion test).
   - An overlay function for `[3,256,256,3]` uint8 images and a constant `MASK_LEGEND`.
3. `originx_harness/id60_masks_validation/controller.py`
   - `MaskFocusController(HierarchicalFocusController)` with its own `ARMS=('point','masks')`. Call `super().__init__(..., arm='point')`, then set `self.arm`.
   - Add `set_masks(masks, frame)`, mirroring `set_geometry`.
   - `render`: grasp stage shows the bound object's mask; place stage shows the held object and destination masks. Append the constant legend so the instruction does not change between 16-step boundaries.
   - The point arm must stay byte-identical to the frozen controller.
4. `originx_harness/id60_masks_validation/runtime.py`
   - Copy of the regions runtime with `PURPOSE='masks_point_unseen_seed_validation'` and `ARMS=('point','masks')`.
   - Keep `HierarchicalGeometryAugmenter` for both arms; the verifier's place context and the initial signature depend on it.
   - Call `env.segmentation()` on every step in **both** arms (`id60_revision/environment.py:28-33`) and pass the result through `ctrl.set_masks`, not through `public`, so traces do not grow every step.
   - Keep the initial-signature keys and every diagnostic event exactly as they are.
5. `originx_harness/id60_masks_validation/registration.py`
   - New `SEED_AUDIT` path and SHA, `PURPOSE`/`ARMS` imported from the new runtime, the new `NEW_REQUIRED` list, treatments `{point, masks}`, and a revised `MAX_SECONDS`.
   - Parent choice:
     - Recommended: `work/id60-regions-validation/manifest.json` with `PARENT_HASH 9414bd65…`. This re-verifies all 138 regions sources.
     - Alternative: keep the hierarchical parent `8d364a…` if you want to reuse the frozen analyzer through the global-patching wrapper.
6. `scripts/run_id60_masks_validation_local.py`
   - Copy with new imports. Ports 20172/20192 can be reused because runs are sequential.
7. `scripts/run_id60_masks_validation_resume_local.py`
   - Register the zero-step, whole-pair resume rule in advance, unlike last time.
8. `tests/test_id60_masks_validation.py`
   - Mirror the 35 tests (lineage count becomes 138 + N), and add mask tests: exact pixels against a synthetic segmentation, nothing drawn for invisible entities, constant text, point arm parity, and a synthetic Env that provides `segmentation()`.

**Work directory (`work/id60-masks-validation/`), frozen by hash in receipts**
9. `audit_unused_seeds.py` (fixed exclusions as in section 2) → `unused-seeds-v1.json`.
10. `analyze_masks_validation.py` and `test_masks_analysis.py` (copy with masks key names; parent hash matched to the chosen parent).

**Order of steps**
1. Write the code and tests.
2. Run the seed audit and write its SHA into `registration.py`.
3. Run the tests on Windows and in WSL.
4. `--freeze`.
5. `--dry-run` inside WSL.
6. Freeze the analyzer (CPU-check receipts and analysis plan).
7. Launch inside WSL to a fresh `runs/id60-masks-validation-local-v1`, detached from the desktop app.

**Risks to record**
- The verifier's `presentation_identity()['policy_marker_treatment']` (`serve_id60_hierarchical_verifier.py:145`) does not list `masks`. The verifier is unchanged and arm-agnostic, so `check_services` still passes, but the manifest should document this.
- The masks come from the simulator's own segmentation (oracle perception), which is a limitation the paper should state.
- An 8 h watchdog is shorter than the 24-failure worst case of about 9.1 h.
- Free disk is close to the 60 GiB check.
- The planner HTTP timeout of 180 s is what produced the last UNKNOWN case in the regions run.

The scan script I wrote is at `C:/Users/LUCKYQ~1/AppData/Local/Temp/claude/D--codex-2026-10-08-new-chat/e6bc7a5c-e11e-439d-a833-cf3dbdfdb40e/scratchpad/scan_seed_tokens.py`.

# MAP: deps

## Dependencies and assets: static inspection, read-only

**Answers in brief:** `supervision` is not installed in any of the three runtimes, in any env they link to, or anywhere under D:/codex. A CVPR kit exists only as one `cvpr.sty`, labelled 2026, with 11 byte-identical copies. There is no `cvpr.cls` and no `.bst` file anywhere under D:/codex.

### 1. Python environments

**Launchers.** All three files are under `D:/codex/originx-harness-v2/work/id60-revision/deployment/runtime/bin/`: `sim-python`, `policy-python` and `planner-python`. They are the same 12-line bash wrapper, differing only in the package directory:
```
export PYTHONNOUSERSITE=1; export PYTHONHOME="$BUNDLE_ROOT/python"
export PYTHONPATH="$BUNDLE_ROOT/packages/<sim|policy|planner>..."
exec "$BUNDLE_ROOT/python/bin/python3.11" -S "$@"
```
- `runtime/python` is a symlink to `/d/codex/originx-attrib-local-20261021/cache/uv-python/cpython-3.11.13-linux-x86_64-gnu`.
- Because of `-S`, the base site-packages (which holds only pip 24.3.1 and setuptools 80.9.0) is never on `sys.path`. Only `PYTHONPATH` entries are importable. A `pip install` into the base interpreter would therefore not be visible. Any new package has to be placed in `runtime/packages/<role>/` (those entries are symlinks) or added to the site.json pythonpath.

**Which process uses which launcher.** Source: `runs/id60-regions-validation-local-v1-resume1/launch-plan.json` and `site.json`.

| Process | Launcher | Script | Extra paths |
|---|---|---|---|
| worker (sim) | `sim-python` | `scripts/run_id60_regions_validation_resume_local.py --worker` | `site.json` `pythonpath.sim` adds `runs/gpu3-local5090-20261022/vendor/robocasa` and `.../vendor/robosuite` |
| native (policy) | `policy-python` | `scripts/serve_id60_hierarchical_policy.py --port 20172` | — |
| planner (verifier) | `planner-python` | `scripts/serve_id60_hierarchical_verifier.py --port 20192` | — |

**Where the cue overlay is drawn.** `originx_harness/id60_hierarchical_revision/focus.py:16` `render_hierarchical_overlay(rgb, hierarchical_context, camera_names)` runs in the sim worker.
- It takes `uint8` arrays of shape `(N_cam, H, W, 3)` and draws with `PIL.ImageDraw`. Neighbor points are `draw.point` single pixels in orange `(255,176,0)`. The anchor is a radius-3 cyan outline circle.
- There are no cv2 or supervision imports in `id60_regions_validation`, `id60_revision`, `id60_hierarchical_revision` or the four scripts.
- A supervision-based overlay would therefore have to import in the sim env.

**Ground-truth segmentation is already available in the worker.** `originx_harness/id60_revision/environment.py:28` `DiagnosticRoboCasa.segmentation()` returns, per camera, `self.extractor.sim.render(width=w, height=h, camera_name=camera, segmentation=True)[::-1].copy()`. Its docstring says "Logging only; never passed to planner or policy." It is called at `id60_regions_validation/runtime.py:167`, every 16 steps. I believe the array is `(H, W, 2)` holding (objtype, objid), but that comes from the MuJoCo/robosuite convention and was not checked here.

**Lock files** (all in `.../work/id60-revision/deployment/`): `{sim,policy,planner}.requirements.lock`, `*.missing.lock`, `dependency-lock-amendment.json` (adds imageio 2.38.0 to policy only) and `deployment-freeze.json` (frozen 2026-10-23T05:46:50Z, `"passed": true`). None of them lists supervision.

**Versions per role** (from `packages/<role>` dist-info METADATA, `cv2/version.py` and the lock files):

| Package | sim | policy | planner |
|---|---|---|---|
| numpy | 2.2.5 | 2.1.3 | 2.2.5 |
| opencv | opencv-python-headless 5.0.0.93 (`headless=True`) | dist-info says opencv-python 5.0.0.93, but `cv2` symlinks to the sim env's headless build | none |
| pillow | 12.3.0 | 11.3.0 | 11.3.0 |
| scipy | 1.15.3 | none | none |
| matplotlib | none | 3.11.2 | none |
| pyyaml / tqdm | yes / yes | yes / yes | yes / yes |
| requests | no | yes | yes |
| defusedxml | no | no | no |
| torch | — | 2.8.0+cu128 | 2.8.0+cu128 |
| transformers | — | 4.57.1 | 5.5.0 |
| mujoco | 3.3.1 | — | — |

**Search for `supervision`:**
- `runtime/packages/*/supervision*`: none.
- Source envs behind the symlinks: `originx-attrib-local-20261021/envs/{sim,training}`, `originx-harness-v2/runs/tf-recovery-20261022T1500Z/envs/awomo` and `visual-cue-local-5090/restore/envs/{server,sim}`. None has it.
- `deployment/wheels` holds only the flash_attn wheel. The uv-caches contain nothing named supervision.
- A full `find /d/codex -maxdepth 9` for supervision dirs, dist-info, wheels or sdists returned nothing.
- The Windows python (3.14.2) has no supervision and no cv2.
- WSL-native paths such as `~/.local` were not inspected, because running WSL was off-limits.

**What adding supervision to sim would involve.** This part is from upstream supervision metadata, not checked against a local copy.
- Its declared dependencies include matplotlib, defusedxml and requests, none of which are in `packages/sim`.
- It also declares `opencv-python`, which would clash with the sim env's `opencv-python-headless`; installing with `--no-deps` avoids that.
- OpenCV 5.0.0.93 meets the `>=4.5.5` version floor, but supervision has not been tested against OpenCV 5 here.
- Changing the env breaks the recorded `deployment-freeze.json` sha state.

### 2. CVPR LaTeX kit

**Primary location:** `D:/codex/2026-10-08/new-chat/outputs/OriginX_visual_cue_paper_v1/template_src/cvpr.sty`, with an identical copy in `.../build/cvpr.sty`.
- Line 28: `\ProvidesPackage{cvpr}[2026 LaTeX class for IEEE CVPR]`.
- 508 lines, 17969 bytes, sha256 `2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153`.
- It loads `natbib[numbers,sort&compress]` and supports the `review`, `final` and `rebuttal` options.
- `template_src/README.txt` calls it the "unmodified official CVPR 2026 style". It also notes that the final 2027 author kit was not yet available and has to be checked.

**`common.tex`:** `\documentclass[10pt,twocolumn,letterpaper]{article}`, `\usepackage[review]{cvpr}`, `\def\confYear{2027}`, `\def\paperID{Anonymous}`. That is the 2026 style with the year relabelled to 2027.

**No `.bst` and no `.bib` files.** References are written by hand in a `\begin{thebibliography}` block: `build/references.tex:12` and `template_src/references.tex:1`.

**Other copies.** A broad search (`find /d/codex -maxdepth 10` for `cvpr*.sty`, `cvpr*.cls`, `ieeenat*.bst`, `ieee_fullname.bst`, `*author*kit*`, `*cvpr*template*`) found 11 `cvpr.sty` files, all byte-identical (same sha prefix `2602473285d1a7df`, all labelled 2026):
- `D:/codex/2026-09-29/3-astra-action-expert-4/work/cvpr.sty`
- `D:/codex/2026-10-08/new-chat/work/final_submission_20261015_10h/paper/{source,named_source,fresh_package_check/anonymous,fresh_package_check/author}/cvpr.sty`
- `D:/codex/2026-10-08/new-chat/work/final_submission_20261015_10h/review/final_package/anonymous_source/cvpr.sty`
- `D:/codex/2026-10-08/new-chat/work/{paper_extreme_20261009,paper_reduced_v3_20261009,paper_revision_20261009}/source/cvpr.sty`
- the two OriginX_visual_cue_paper_v1 copies above.

**TeX toolchain.** There is no pdflatex, latexmk, bibtex or tectonic on the Windows/Git-Bash PATH. Past compiles ran under WSL TinyTeX. `.../final_submission_20261015_10h/paper/fresh_package_check/anonymous/main.log` shows "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)" from `/home/luckyqinzhen/.TinyTeX/texmf-dist/...`, with "Package: cvpr 2026 LaTeX class for IEEE CVPR".

# SYNTHESIS

# Plan: "masks" cue arm for OriginX id60 (CoffeeSetupMug, XR1), point vs masks prospective study

Sources: the four maps, plus a read-only re-check of the key lines in `D:/codex/originx-harness-v2`. Nothing has been modified or run. All paths below are relative to that root unless they are absolute.

## 0. Guardrails (apply to every step)

1. **Frozen files stay untouched.** This covers all 138 sources in `work/id60-regions-validation/manifest.json` (canonical `9414bd65…`). Among them are `id60/focus.py`, `id60/robocasa.py`, `id60/controller.py`, `id60_hierarchical_revision/*`, `id60_regions_validation/*` and `scripts/serve_id60_hierarchical_verifier.py`. Any edit breaks `registration.source_files` (`id60_regions_validation/registration.py:74-84`) and the verifier identity (`serve_id60_hierarchical_verifier.py:140-160`). Everything new goes in new files.
2. **The outcome is not assumed.** The user expects masks to "definitely improve" success. Treat that as a pre-registered hypothesis:
   - Pilot success counts are never used to tune the cue.
   - The registered result is reported whatever it is.
   - There is no re-running on fresh seeds until a positive result appears.
   - Power at 12 pairs is low (0.18 at a risk difference of 1/3, 0.61 at 0.5; `development-context-and-power-v1.json`). Prior evidence: regions vs point was 6/12 vs 7/12, McNemar p = 1.0.
3. **User-approval gates:**
   - downloading the supervision and defusedxml wheels (§8);
   - launching the registered run;
   - the CVPR submission itself. The user submits through their own account; the agent never logs in.
4. **Hosts and reporting:** non-training eval runs on the local RTX 5090 through WSL, as in the regions study. Never kill processes we did not start; if a port or GPU is busy, stop and ask. User-facing reports go in Chinese, per the user's memory rule.

## 1. Treatment definition (fixed before the pilot, frozen at registration)

**Masks arm policy RGB.** Start from the unmarked `raw['rgb']`. The point arm's bbox/crosshair is not drawn. Then add exterior outlines ("rings") of exact masks on all three cameras.

**Stage rule:**
- `operation != 'place'` (grasp): one **orange** ring around the exact visible mask of `snapshot.target_id`, an object or part.
- `operation == 'place'`:
  - a **cyan** ring around the exact visible mask of the held subject. This is `placement['subject']['id']`, resolved through `previous_bound_grasp` (`id60_placement_revision/controller.py:44-74`).
  - an **orange** ring around the occlusion-correct visible mask of the destination, `snapshot.target_id`, which is region `R…` "dispenser receiving region".
  - If the destination is an object or part rather than a region, its segmentation mask is used instead.

**Colours and width** reuse the frozen constants:
- `ORANGE = PRIMARY_COLOR (255,176,0)` (`id60/focus.py:13`); `CYAN = RELATED_COLOR (0,220,255)` (`id60/focus.py:14`).
- `RING_WIDTH = 2`, matching the `outline_width` default in `id60/focus.py:49`.
- `MIN_PIXELS = 3`, the extractor's visibility floor (`robocasa.py:107`).

**Ring geometry, so outlines fit exactly:**
- `ring_k(M) = {p : 1 ≤ chebyshev_dist(p, M) ≤ k} \ U`, where `U` is the union of all masks drawn in that camera.
- No pixel of the grasp target, held object or destination interior is ever changed. The external-bracket study protected bbox interiors for the same reason.
- Precedence when rings overlap: the target/destination (orange) wins, matching `render_focus` drawing the target last (`id60/focus.py:70-73`).
- Clipped at the image border with no wrap-around.

**Destination "ghost" (the receiving region has no pixels):** the site is alpha-0 with no geoms (`model.xml:45`; `robocasa.py:71-76`, `geoms=()`). Its mask is therefore defined from task-neutral geometry:
- **Box frame:** axes are the parent fixture root axes (`COORDINATE_FRAME='parent_root_x_y'`, `geometry.py:28`). Require the parent z axis to be within 0.999 of world z, otherwise the ghost is marked unavailable.
- **Box size:** let `low, high` be the subject's root-local AABB (`_ObjectGeometry.low_local/high_local`, `geometry.py:67-75,125-149`).
  - `s = min((high-low)[:2])`, the same short planar extent as `geometry.py:145`; for the mug this is roughly the body diameter.
  - `h = (high-low)[2]`.
  - Half-extents are `(s/2, s/2, h/2)`.
- **Box placement:**
  - xy is the region anchor `c`: `site_xpos`, or `body_xpos + R·offset` (`geometry.py:252-256`).
  - The bottom sits at `z_support`, computed once at augmenter construction (frame 0, fixtures are static). It is the first hit of `mj_ray(m, d, c, [0,0,-1], None, 1, -1, hit)`, the same call signature as `robocasa.py:120-121`.
  - The ghost is valid only if the hit geom belongs to the parent fixture's geoms; otherwise it is unavailable (fail-closed, audited).
  - If there is no hit, `z_support = c_z`; the audit records which rule fired.
- **Never use scorer tolerances.** That rules out `xy_thresh=0.04` and `|dz|<0.10` (`coffee_machine.py:95-115`), keeping `runtime_success_threshold_access: False`.
- **Visible ghost mask:** `{pixel: ray hits box, t_entry ≤ D + τ, and the pixel is not in the subject's seg mask}`.
  - `t` is camera z-depth and `D` is the depth buffer of the same segmentation render.
  - `τ = DEPTH_TOL_M = 0.003`.
  - Excluding subject pixels avoids z-fighting when the mug sits inside the ghost.

**Rejected supervision styles:**
- `PolygonAnnotator`: its line straddles the boundary (findContours plus polylines), so it covers object pixels.
- `HaloAnnotator`: blurred, so not exact.
- `MaskAnnotator` at its default `opacity=0.5`: tints the object pixels the policy grasps from.
- What is used instead: `sv.MaskAnnotator(opacity=1.0)` applied to precomputed ring masks. This is byte-exact.

## 2. New files (nothing frozen is modified)

| # | Path | Role |
|---|---|---|
| 1 | `originx_harness/id60_masks_validation/__init__.py` | package |
| 2 | `originx_harness/id60_masks_validation/masks.py` | pure numpy, plus supervision painter; constants, legends |
| 3 | `originx_harness/id60_masks_validation/geometry.py` | `MaskedGeometryAugmenter` (the only sim access), dataclasses |
| 4 | `originx_harness/id60_masks_validation/controller.py` | `MaskFocusController` |
| 5 | `originx_harness/id60_masks_validation/runtime.py` | copy of `id60_regions_validation/runtime.py` with masks defaults and one extra diagnostic event |
| 6 | `originx_harness/id60_masks_validation/registration.py` | copy of the regions registration, plus `build_pilot_manifest` |
| 7 | `scripts/run_id60_masks_validation_local.py` | driver copy (factories, site overlay, `MAX_SECONDS`) |
| 8 | `scripts/run_id60_masks_pilot_local.py` | pilot driver for the 6 dev cases |
| 9 | `scripts/run_id60_masks_validation_resume_local.py` | resume rule, registered in advance (zero-step, whole-pair, as in `run_id60_regions_validation_resume_local.py:31-62`) |
| 10 | `tests/test_id60_masks_validation.py` | the 35 regions tests mirrored, plus controller/runtime mask tests |
| 11 | `tests/test_id60_masks_exactness.py` | T1 pure-CPU pixel-exactness tests |
| 12 | `work/id60-masks-validation/check_mask_exactness.py` | T3 sim-in-the-loop checks (WSL, no model services) |
| 13 | `work/id60-masks-validation/replay_trace_checks.py` | T2 checks on recorded regions traces |
| 14 | `work/id60-masks-validation/contact_sheets.py` | pilot visual sheets, including the XR1 crop view |
| 15 | `work/id60-masks-validation/audit_unused_seeds.py` | seed audit copy with fixed exclusions |
| 16 | `work/id60-masks-validation/analyze_masks_validation.py`, `test_masks_analysis.py` | analyzer copy |
| 17 | `work/id60-masks-validation/deps/sim-overlay/`, `dependency-lock-amendment-masks-v1.json` | supervision overlay (§8) |

`NEW_REQUIRED` contains items 1-11. Items 12-16 are frozen by sha in receipts, following the regions precedent.

## 3. Signatures

### `masks.py`

Pure functions; the only external import is supervision, used for painting.

```python
ORANGE=(255,176,0); CYAN=(0,220,255); RING_WIDTH=2; MIN_PIXELS=3; DEPTH_TOL_M=0.003
DEPTH_CONVENTION: str   # 'legacy' | 'reversed_z', frozen from the T3 check
MASK_LEGEND={'target': "...", 'place': "..."}          # see §7
def object_mask(seg: np.ndarray, geoms: Sequence[int]) -> np.ndarray      # (seg[...,0]==5)&isin(seg[...,1],geoms), robocasa.py:105
def bbox_from_mask(mask: np.ndarray) -> list[float] | None                # exact robocasa.py:106-108 formula; None if <3 px
def buffer_to_camera_depth(z01, near, far, convention) -> np.ndarray       # legacy: n*f/(f - z01*(f-n)); reversed: n*f/(n + z01*(f-n))
def pixel_rays(K, cam_to_world, h, w) -> tuple[np.ndarray, np.ndarray]     # origin (3,), dirs (h,w,3) world, camera-z component = 1
@dataclass(frozen=True) class GhostBox: center: np.ndarray; rotation: np.ndarray; half_extents: np.ndarray
def ghost_box(anchor_xyz, parent_rotation, z_support, subject_low, subject_high) -> GhostBox | None
def ray_box_entry(origin, dirs, box: GhostBox) -> np.ndarray                # slab test in box frame; t_entry (camera z), +inf on miss, 0 if camera inside
def ghost_visible_mask(frame: "CameraMaskFrame", box: GhostBox, subject_mask, tol=DEPTH_TOL_M) -> np.ndarray
def outline_rings(ordered: Sequence[tuple[int, np.ndarray]], width=RING_WIDTH) -> dict[int, np.ndarray]
    # class_id -> ring; ring = dilate_chebyshev(mask,width) & ~union(all masks) & ~rings_already_assigned (earlier entries win)
    # dilation = OR of (2w+1)^2 shifted windows over a False-padded copy (no np.roll, so no wrap-around)
def paint_rings(rgb_hw3: np.ndarray, rings: Mapping[int, np.ndarray]) -> np.ndarray   # supervision path, returns a new array
def paint_rings_reference(rgb_hw3, rings) -> np.ndarray                    # numpy oracle: out[ring_c] = COLOR[c]
```

`paint_rings` steps:
1. Drop empty rings, then build `bgr = np.ascontiguousarray(rgb[..., ::-1])`.
2. `det = sv.Detections(xyxy=sv.mask_to_xyxy(m), mask=m, class_id=ids)`.
3. `sv.MaskAnnotator(color=sv.ColorPalette([sv.Color(*ORANGE), sv.Color(*CYAN)]), opacity=1.0, color_lookup=sv.ColorLookup.CLASS).annotate(scene=bgr, detections=det)`.
4. Return `bgr[..., ::-1].copy()`.

Supervision quirks this has to handle:
- Colours are converted with `as_bgr()`, so the image must be BGR on the way in and flipped back after.
- `annotate` mutates `scene` in place.
- `mask_to_xyxy` fails on empty masks, hence the filtering in step 1.
- With `opacity=1.0`, `cv2.addWeighted` gives exactly the coloured copy.

### `geometry.py`

```python
@dataclass(frozen=True)
class CameraMaskFrame: seg: np.ndarray; depth_m: np.ndarray; K: np.ndarray; cam_to_world: np.ndarray  # arrays set read-only
@dataclass(frozen=True, eq=False)
class MaskedGeometryAugmentation(GeometryAugmentation):            # geometry.py:57-64
    cameras: Mapping[str, CameraMaskFrame]
    object_geoms: Mapping[str, tuple[int, ...]]                       # every object/part spec id -> spec.geoms
    region_anchors: Mapping[str, tuple]                               # region id -> (center xyz, parent R 3x3, z_support) | None
    subject_bounds: Mapping[str, tuple]                               # object id -> (low_local, high_local) from self._geometry
class MaskedGeometryAugmenter(HierarchicalGeometryAugmenter):      # geometry.py:77
    def __init__(self, extractor, **kw)   # super(); compute z_support per region; run check_depth_convention() at frame 0
    def observe(self, rgb_shape, registry) -> MaskedGeometryAugmentation
    def check_depth_convention(self) -> dict   # raises ValueError before the first action on mismatch
```

What `observe` does:
1. `base = super().observe(...)`.
2. For each `cam` in `CAMERAS`: `seg, z01 = self.sim.render(width=w, height=h, camera_name=cam, depth=True, segmentation=True)`, then flip both with `seg[::-1]` and `z01[::-1]`.
   - This returns a `(ret_img, depth_img)` tuple; verified in `runs/gpu3-local5090-20261022/vendor/robosuite/robosuite/utils/binding_utils.py`, `render` around lines 129-167 and `read_pixels` around 169-193.
3. `D = buffer_to_camera_depth(z01, near=vis.map.znear*extent, far=vis.map.zfar*extent, DEPTH_CONVENTION)`.
4. `K = get_camera_intrinsic_matrix`, `T = get_camera_extrinsic_matrix` (OpenCV axes, axis correction applied).
5. Add to `audit['mask_frame']` per camera: `seg_sha256`, `K`, `cam_to_world`, and the deterministic integer `seg` hash only.
6. Return `MaskedGeometryAugmentation(**vars(base), cameras=..., ...)`.

Why this is safe:
- `dataclasses.replace(..., audit={})` in `set_geometry` (`controller.py:33`) rebuilds the subclass and keeps the new fields.
- `geometry_record` (`runtime.py:23-27`) sees only the base fields plus the extra audit keys.
- The depth hash goes to a separate diagnostic event, not into the step-0 signature, to avoid any float-determinism pairing risk.

### `controller.py`

```python
ARMS=('point','masks')
class MaskFocusController(HierarchicalFocusController):
    def __init__(self, planner, variant='full', *, arm='point', confirmations=2):
        if arm not in ARMS: raise ValueError('Unknown masks comparison arm')
        super().__init__(planner, variant, arm='point', confirmations=confirmations)   # passes the controller.py:18 check
        self.arm = arm                       # 'masks' never matches the controller.py:67/89/115/118 branches; labels only (73,88,122)
    def reset(self, instruction): super().reset(instruction); self._last_mask_render = None
    def compute_masks(self, raw, registry) -> tuple[list[tuple[str,int,dict[str,np.ndarray]]], dict]   # both arms, diagnostic
    def render(self, raw, registry) -> Mapping
    def prepare(self, raw, registry, step)   # mirrors controller.py:85-100: self._last_mask_render=None; super().prepare;
                                             # receipt['mask_focus']=deepcopy(...); receipt['rgb_changed']; replace history[-1]
```

`render` logic:
1. `base = super().render(raw, registry)`. This is the frozen point path, and it still sets `_last_hierarchical_render` in the place stage.
2. Gating identical to the point arm. Return `base` unchanged if any of these hold:
   - `variant == 'native'`, `not _render_enabled`, or `manager`, `stage` or `snapshot` is None (`id60/controller.py:268-270`);
   - `snapshot.mode == 'native'` or `status == 'complete'` (`id60/focus.py:65,100-101`);
   - `snapshot.target_id is None` (`focus.py:104-106`).
3. Compute masks in a fail-closed `try`: on an exception, record `status='mask_error'` and draw nothing. Steps inside:
   - Use `aug = self._augmentation` only if `self._augmentation_frame == registry.frame` and it is a `MaskedGeometryAugmentation`; otherwise `status='stale_or_missing'`.
   - In the place stage, swap `_placement_registry` exactly as `controller.py:109-111` does, then call `self._context()`.
   - Object/part masks are drawn per camera only if `cam in registry.get(id).views` and the mask has at least 3 pixels.
   - Check consistency: `bbox_from_mask(mask) == registry.get(id).views[cam]['bbox']`. On mismatch, skip that entity in that camera and audit it.
4. If `self.arm == 'point'`, store a diagnostic `_last_mask_render` with `drawn=False` and return `base`. The point arm stays byte-identical.
5. Masks arm:
   - `rgb = raw['rgb'].copy()`.
   - For each index `i` and camera name in `raw['camera_names']`: `rgb[i] = paint_rings(rgb[i], outline_rings(ordered))`. `ordered` is `[(0, target)]` in grasp and `[(0, destination), (1, subject)]` in place.
   - `text = base['instruction'] + ' ' + MASK_LEGEND['place' if op == 'place' else 'target']`.
   - Return `{**base, 'rgb': rgb, 'instruction': text}`.
6. `_last_mask_render` holds JSON-safe data only: per camera and role, `mask_px`, `ring_px`, mask and ring sha256, bbox, ghost status, `inside_mask_pixels_changed` (must be 0), and np.packbits as base64 at prepare steps. No inf or NaN; `tracing.py:58-69` rejects them.

`ExternalMarkerController` (`id60_grasp_revision/focus.py:101-113`) is the existing precedent for a render-only override.

### `runtime.py` (copy of `id60_regions_validation/runtime.py`)

- `PURPOSE='masks_point_unseen_seed_validation'`, `ARMS=('point','masks')`, its own `PairAdmissionError`.
- Defaults at lines 65-70 become `MaskFocusController` and `MaskedGeometryAugmenter`. Add `assert isinstance(augmentation, MaskedGeometryAugmentation)`.
- Everything else stays line-for-line: the guards (49-64), the 9-key initial signature (89-90), plan admission (131-140), the line-160 instruction check, the line-167 segmentation log, and the contact/placement diagnostics (192-197).
- One addition: after the observation (line 173), record `trace.record_event(steps, {'kind':'mask_presentation', 'arm':arm, **ctrl._last_mask_render_or_render_receipt, 'depth_sha256':…, 'logging_only':True, 'used_by_policy_or_planner':False})`.

### `registration.py` (copy)

| Constant | Value |
|---|---|
| `PARENT` | `'work/id60-regions-validation/manifest.json'` |
| `PARENT_HASH` | the `9414bd65…` canonical, read in full from the file |
| `SEED_AUDIT` | `'work/id60-masks-validation/unused-seeds-v1.json'` |
| `SEED_AUDIT_SHA` | filled in after the audit |
| `PAIR_COUNT` | 12 |
| `MAX_SECONDS` | 36000 (worst case of 24 failures ≈ 9.1 h, which exceeds 28800) |

Other contents:
- `SHARED_CONTROLS`: the regions values plus `shared_mask_augmenter_for_all_arms: True`, `mask_source`, `supervision_version`, `supervision_wheel_sha256`, `depth_convention`, `ring_width_px: 2` and `depth_tol_m: 0.003`.
- `treatments`: `point` and `masks`, the latter worded as in §1.
- `limitations`: oracle simulator masks; the ghost is a subject-sized geometric proxy, not the scorer region; the legend is part of the treatment; one task and one model.
- `build_pilot_manifest(root)`: seeds `[20261025, 20261026, 20261027]`, `purpose='masks_point_dev_pilot'`, 6 counterbalanced cases, not a prospective study.

### Drivers

`run_id60_masks_validation_local.py` differs from the regions driver in these places:
- It imports from `id60_masks_validation`.
- `site_config` (copy of lines 24-32) appends `str(ROOT/'work/id60-masks-validation/deps/sim-overlay')` to `site['pythonpath']['sim']` and sets `MPLBACKEND=Agg` and `MPLCONFIGDIR=…/mpl-cache`.
- `case_count` comes from `manifest['planned_cases']` instead of the hard-coded 24 (lines 132, 269).
- The `run_case(...)` call (copy of lines 203-207) adds `controller_factory=MaskFocusController, augmenter_factory=MaskedGeometryAugmenter`.
- `check_services` (96-111) is unchanged. The verifier is untouched, so `presentation_identity()` still matches.

## 4. Data flow from env to policy, per step

1. `env.step` runs `ID60RoboCasa.step` (`robocasa.py:140-144`): physics, then `canonical_observation`, then `extractor.observe`. The extractor renders seg and keeps only a bbox (lines 100, 105-108).
2. `runtime:119`: `augmenter.observe` runs the frozen lattice/anchors, then one seg+depth render per camera at 256×256, same sim state, no physics change. This yields `MaskedGeometryAugmentation(frame=registry.frame)`.
3. `runtime:121`: `ctrl.set_geometry` strips the audit but keeps `cameras`, `object_geoms`, `region_anchors` and `subject_bounds`.
4. `runtime:124`: `public` is built. Masks never travel through `public`, `Entity` or the registry; `Entity.views` accepts only `bbox`/`point` (`entities.py:85-86`).
5. Every 16 steps, `ctrl.prepare` runs: the planner sees raw RGB (`id60/controller.py:153-156`), then `compile_instruction`, then `self.render`. Otherwise `ctrl.render` runs. Both paths reach `MaskFocusController.render`.
6. `runtime:162-165`: the policy observation is `{rgb (masked), proprio, instruction (base+legend), step - segment_start, episode_id}`.
7. XR1 `_observe_native` (`adapters.py:458-465`) takes `rgb[i]` in `CAMERA_KEYS` order, then applies `center_crop [6,249)` and a bilinear resize to 256 (`…/eval_robocasa365/entry.py:92-104,136`), with history t, t-2, t-4, t-6.

## 5. Coordinate and flip conventions

- **Camera order** is fixed: `('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')` (`robocasa.py:14`, `geometry.py:26`, `evaluation.py:37`). Masks are keyed by camera name and drawn on `rgb[i]` for `raw['camera_names'][i]`.
- **Vertical flip:** `sim.render` returns OpenGL bottom-up buffers. Apply `[::-1]` (rows only) to both seg and depth, exactly as `robocasa.py:100` and `environment.py:31-32` do. The policy RGB was flipped the same way by `gym_wrapper.py:270-278`. There is no horizontal flip, resize or crop before XR1.
- **Pixel indexing:** pixel `(row i, col j)` has its centre at continuous coordinates `(u, v) = (j+0.5, i+0.5)`.
  - Intrinsics are `K = [[f,0,W/2],[0,f,H/2],[0,0,1]]` with `f = 0.5·H/tan(fovy/2)`.
  - `cam_to_world` uses OpenCV axes (x right, y down, z forward; `diag(1,-1,-1)` correction in `camera_utils.get_camera_extrinsic_matrix`).
  - World ray: `d = R_c2w @ [(u-cx)/f, (v-cy)/f, 1]`. Its camera-z component is 1, so the ray parameter `t` equals the depth-buffer z.
- **Index arrays directly.** Never go through the normalized `[x/w, y/h]` registry values or `round(v*(W-1))` (`id60/focus.py:32-40`), which drift by up to 1 px.
- **Eye-in-hand** K/T are recomputed every step.
- **Depth:** metric camera z, with `near = znear·extent` and `far = zfar·extent`. MuJoCo here is 3.3.1, which may use reversed-Z, so robosuite's `get_real_depth_map` formula is not trusted. `DEPTH_CONVENTION` is frozen from T3, and `check_depth_convention()` re-asserts it at every reset.
- **What XR1 actually sees:** `x' = (x-6)·256/243`. Rings in the outer 6 px are lost, and 2 px rings come out about 2.1 px.

## 6. How the arm is selected

1. **Manifest:** `case['comparison_arm'] ∈ ('point','masks')`, built by `cases()` with alternating order (copy of `registration.py:56-71`). `pair_key` includes the new `PURPOSE`, so case IDs are new.
2. **Runtime:** the guard at `runtime:49-51` checks against the new `ARMS`.
3. **Controller:** `controller_factory(planner, 'full', arm=arm)` (`runtime:101`).
4. **Both arms share** the same augmenter (identical renders and step-0 signature) and the same controller class. Only `self.arm` gates drawing and the legend.
5. **Engineering runs:** `run_case(..., engineering_steps=48, controller_factory=MaskFocusController, augmenter_factory=MaskedGeometryAugmenter)`.

## 7. Instruction text

- **Point arm:** unchanged. This is `compile_instruction` (`id60/focus.py:85-120`), including "An orange marker identifies this target when visible…" (lines 111-112).
- **Masks arm:** the same base text plus one fixed sentence per operation. It depends only on stage and binding, never on visibility, so `runtime:159-160` cannot fire. It is omitted when rendering is gated off.
  - `MASK_LEGEND['target']` (grasp): "A thin orange outline drawn just outside this target traces its exact visible boundary; the outline never covers the target itself."
  - `MASK_LEGEND['place']`: "A thin cyan outline traces the exact visible boundary of the held object, and a thin orange outline traces the visible space that the held object occupies when placed at the destination; neither outline covers the objects themselves."
- The legend is descriptive only, with no added behavioural command (unlike `ALIGN_INSTRUCTION`). It is part of the treatment and is stated as such in the manifest.

## 8. Getting supervision into the sim runtime

**Current state:**
- supervision is not installed anywhere.
- The sim env (`work/id60-revision/deployment/runtime/packages/sim`) already has numpy 2.2.5, opencv-python-headless 5.0.0.93, pillow 12.3.0, scipy 1.15.3, pyyaml, tqdm, six and packaging.
- It lacks matplotlib, defusedxml and requests. The policy env already has matplotlib 3.11.2 (plus contourpy, cycler, fonttools, kiwisolver, pyparsing, dateutil) and requests 2.34.2 with urllib3, idna, certifi and charset_normalizer.

**Steps:**
1. **Ask the user before downloading.** Pin `supervision==0.25.1`, or the newest release at least two weeks old, and download `defusedxml==0.7.1` from PyPI. Report exact filenames and sizes in the request. Fetch with `--no-deps` so `opencv-python` cannot shadow the headless cv2.
2. **Build the overlay** as a new directory: `work/id60-masks-validation/deps/sim-overlay/`. Contents:
   - supervision and defusedxml;
   - symlinks to the policy env's matplotlib-stack and requests-stack packages, so nothing else is downloaded. The numpy 2.x ABI is forward compatible from 2.1.3 to 2.2.5.

   Do not touch `packages/sim`, which is frozen by `deployment-freeze.json`. PYTHONPATH order is `packages/sim` first, then `ROOT`, then the site paths (`sim-python:9`, `launch_id60_cell.process_environment:182-185`). That keeps the sim env's numpy, cv2 and PIL in charge.
3. **Verify** with `sim-python -I -c "import supervision, cv2; print(supervision.__version__, cv2.__version__)"` and add only what is actually missing. Then write `dependency-lock-amendment-masks-v1.json` (per-file sha256), following the imageio amendment precedent.
4. **Fallback** if approval is declined or the import fails: `paint_rings_reference` produces identical bytes, as T1 shows. In that case the paper must not claim supervision was used.

## 9. Offline pixel-exactness testing

### T1: pure CPU (`tests/test_id60_masks_exactness.py`)

Runs in WSL `sim-python` with the overlay. The supervision tests may only skip on Windows when the import fails; `MASKS_REQUIRE_SUPERVISION=1` makes skipping illegal in WSL.

1. **`object_mask`** on a synthetic seg: objtype-6 (site) ids that collide with geom ids are excluded, and the `(-1,-1)` background is excluded.
2. **`bbox_from_mask`** equals the `robocasa.py:108` formula exactly (float `==`).
3. **Flip:** a fake bottom-up renderer must be flipped identically for seg and depth (row 0 is the top).
4. **Depth conversions:** round-trip for both conventions, and the selector picks the right one from synthetic samples.
5. **`ray_box_entry`, axis-aligned head-on case:** the box mask equals a closed-form integer rectangle on pixel centres.
6. **`ray_box_entry`, random poses** (fixed RNG, 200 cases, box fully in front of the camera): the mask equals point-in-convex-hull of the 8 projected corners, an independent oracle.
7. **Occlusion:** a synthetic depth occluder removes exactly the covered pixels, and ties at `±τ` behave as documented.
8. **Rings:** `outline_rings` equals `(scipy.ndimage.distance_transform_cdt(~M, 'chessboard') ≤ 2) & ~U` on 200 random masks, including masks touching the border (no wrap-around), with disjoint precedence.
9. **Painter:** `paint_rings` equals `paint_rings_reference` byte-for-byte on random uint8 images. Also:
   - ring pixels are exactly `(255,176,0)` and `(0,220,255)` in RGB (this catches a BGR swap);
   - every other pixel is unchanged;
   - the input is not mutated (`np.shares_memory` is False);
   - empty masks leave the image unchanged.
10. **Dataclass:** `replace(aug, audit={})` keeps the subclass type and fields.
11. **JSON safety:** receipts and audits contain no inf or NaN.

### Controller and runtime tests (`tests/test_id60_masks_validation.py`)

- Mirror the 35 regions tests on the `tests/test_id60_hierarchical_runtime.py` fixtures with 48 engineering steps. The lineage test asserts `138 + len(NEW_REQUIRED)`.
- **Point-arm parity:** `MaskFocusController(arm='point')` must produce policy rgb, instruction and planner call payloads byte-identical to `HierarchicalFocusController(arm='point')`.
- **Masks arm:**
  - grasp draws only the orange target ring; place draws orange destination plus cyan subject;
  - `inside_mask_pixels_changed == 0`;
  - text stays constant when masks vanish between boundaries;
  - nothing is drawn when rendering is disabled, complete or native, or when the frame is stale.

### T2: recorded traces (CPU)

Input: `runs/id60-regions-validation-local-v1*/worker-0/run/traces/*/NNNNNN-observation.npz`, using `payload.raw.segmentation.<cam>` and the raw rgb.

1. **Alignment:** the ratio of RGB gradient on seg edges to mean gradient must be ≥ 3 as stored, and < 2 for flipped variants. Map 2 measured 5.7-8.7 vs 1.0-1.9.
2. **Painter on real frames:** zero interior pixels changed, and rings still detectable after a reimplemented XR1 `center_crop` and resize.

### T3: sim in the loop (`check_mask_exactness.py`)

Runs in WSL with EGL and no model services, on a scratch env. It writes `exactness-checks-v1.json`.

1. **Bbox equality:** for every visible object/part, augmenter mask bbox equals `registry.views[cam]['bbox']` over 100 steps of zero actions on seeds 20261025-27.
2. **Depth convention:** 64 seg pixels per camera; `mj_ray` through each pixel centre with `geomgroup = ctx.vopt.geomgroup`; keep only samples whose hit geom equals the seg id. Convert to camera z. Pass if the chosen convention has median |Δ| ≤ 2 mm and p90 ≤ 5 mm while the other is clearly off. This freezes `DEPTH_CONVENTION`.
3. **GL ghost cross-check:** a copy of `binding_utils` `render`/`read_pixels` adds the same box via `mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mjGEOM_BOX, half, center, R.ravel(), rgba)`, with `objtype=mjOBJ_SITE`, `objid=site_id`, `segid=ngeom`, then `ngeom += 1`. It renders in segment mode, so the GL depth test supplies occlusion. Pass if IoU ≥ 0.98 for masks of at least 50 px and every mismatching pixel is within 1 px of the analytic boundary. This is a test-only diagnostic; the scene is rebuilt on the next render, so policy RGB is unaffected.
4. **State injection:** test-only free-joint `qpos` writes plus `forward` put the mug at, above and beside the site and in the gripper. Check ghost and subject exclusion and ring disjointness.
5. **Cost:** per-step mask overhead in ms (budget ≤ 100 ms median against about 2 s per step).

## 10. Pilot on dev seeds 20261025-20261027

1. **Order:** the pilot runs after T1-T3 pass and before the seed audit and freeze.
2. **Matrix:** `build_pilot_manifest` gives 3 pairs and 6 cases: pair 0 point→masks, pair 1 masks→point, pair 2 point→masks.
3. **Launch:** `wsl.exe -d Ubuntu-22.04 --cd /mnt/d/codex/originx-harness-v2 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 work/id60-revision/deployment/runtime/bin/sim-python scripts/run_id60_masks_pilot_local.py --manifest work/id60-masks-validation/pilot-manifest-v1.json --output runs/id60-masks-pilot-local-v1 --duration-seconds 14400 --residency resident`.
   - Ports 20172/20192 must be free.
   - Launch detached from the desktop app (for example with `Start-Process` and `setsid`). The previous run died when the app restarted (`interruption-record-20261024.json`).
4. **Gates (fidelity only, declared in advance):**
   - G1: frozen-source check passes (138 files).
   - G2: 100% bbox consistency.
   - G3: depth-convention check passes at every reset.
   - G4: GL cross-check IoU ≥ 0.98 at prepare steps.
   - G5: `inside_mask_pixels_changed == 0` everywhere.
   - G6: no line-160 error; traces `complete`; no non-finite payloads.
   - G7: all 3 second arms pass `assert_initial_match` and the plan gate (proves render determinism with the new audit).
   - G8: overhead ≤ 100 ms/step; ≤ 450 MiB/case.
   - G9: the copied analyzer reads all 6 pilot traces (the trace-integrity cadence of `analyze_targeted.py:126-166` still holds with the extra `mask_presentation` event).
   - G10: the user signs off on contact sheets covering 3 cameras × {grasp start, pre-grasp, place start, pre-release}, each raw and through the XR1 crop.
5. **Pilot outcomes:** success counts are logged but are not a gate and not evidence. Only engineering bugs may be fixed afterwards, and then T1-T3 and the pilot must be re-run.

## 11. Registering the prospective study (12 new seeds)

1. **Seed audit:** copy `work/id60-regions-validation/audit_unused_seeds.py`.
   - Remove `id60_regions_validation` and `id60-regions-validation` from `EXCLUDE` (L15-19); exclude `id60_masks_validation` and `id60-masks-validation` instead, by path substring.
   - Fix the L83 receipt bug with `evidence.get(s, [])`.
   - Keep the `TOKEN` rule (L21) and the selection "first 12 ≥ 20261028 with no evidence" (L77-80).
   - Assert disjointness from {20261025-27} and {20261031-42}.
   - Run in WSL (`rg` is not on the Windows PATH). The expected result is 20261044-20261055, but that is only a prediction; the receipt is authoritative.
   - No new file may contain 2026104x/105x literals before the audit runs.
2. Put the audit sha into `SEED_AUDIT_SHA`.
3. Run all tests in WSL (mandatory) and on Windows (subset). Record `runtime-launcher-checks-v1.json`.
4. `sim-python scripts/run_id60_masks_validation_local.py --freeze --manifest work/id60-masks-validation/manifest.json`. This refuses to overwrite (copy of lines 247-252), and `load_manifest` re-verifies 138 + N sources.
5. Run `--dry-run` in WSL to `work/id60-masks-validation/dry-run-v1/`.
6. **Analyzer copy:**
   - `ARMS=('point','masks')` and the new `PURPOSE`.
   - The parent manifest hash in `manifest_check` (regions analyzer L111) changes to the regions canonical `9414bd65…`.
   - Key names become `masks_*`.
   - Keep `initial_checks` with exactly the 9 keys.
   - Primary: exact two-sided McNemar on discordant pairs, plus the paired risk difference with an exact CI. Secondary: final xy delta (cm) and steps. Fidelity: draw coverage per stage and camera.
   - Test it on the pilot traces, then freeze `analysis-plan-v1.json` with the analyzer sha before launch.
7. **Launch (user go-ahead required)**, detached:
   - `… sim-python scripts/run_id60_masks_validation_local.py --manifest work/id60-masks-validation/manifest.json --output runs/id60-masks-validation-local-v1 --duration-seconds 36000 --residency resident`.
   - About 71.9 GiB is free now; the pilot uses about 2 GiB, which leaves enough above the 60 GiB check at line 260.
   - If interrupted, use only the pre-registered resume driver.
   - If a bug is found mid-run, stop and register a v2 with new seeds; never patch and continue.

## 12. CVPR

- **Kit:** only a 2026-labelled `cvpr.sty` exists (`template_src/cvpr.sty`, sha `2602473285…`, relabelled 2027 in `common.tex`). Check the official 2027 author kit and the deadline before writing.
- **Build:** compile with WSL TinyTeX.
- **Content rules:**
  - Results come only from the frozen analyzer outputs.
  - State that the masks are oracle simulator masks: an upper-bound cue, not perception.
  - Report a null or negative result as it comes out.
  - Keep the submission anonymous.
- **Submission:** the user does this themselves.

## 13. Risks and unknowns

1. **Dependencies:** downloads are gated on user approval; the deployment freeze is extended by an overlay rather than edited; supervision has not been tested on OpenCV 5.0; the BGR and in-place quirks are covered by T1 test 9.
2. **Depth convention** in MuJoCo 3.3.1 (possible reversed-Z) and depth readback under Mesa D3D12 are unverified until T3.
3. **The ghost is a design choice:** a square prism of side `s` centred at the site, which may differ from where success is actually scored. The `z_support` ray could hit a non-parent geom, in which case the ghost is marked unavailable.
4. **Modal vs registry visibility:** masks are modal (gripper and sites occlude). The ≥3 px registry rule and the per-pixel ghost rule differ in the masks arm; this is inherent to the treatment and audited.
5. **The policy may be hurt** by out-of-distribution outlines. The regions study showed no gain, and 12 pairs give low power.
6. **Watchdog and failures:** worst case is about 9.1 h, inside the 10 h watchdog. The planner's 180 s HTTP timeout (`training_free.py:55`) can still produce UNKNOWN cases. A desktop-app restart killed the last run.
7. **Pairing determinism** with the extra renders: covered by G7, and the depth hash is kept out of the signature.
8. **Verifier label:** `presentation_identity()['policy_marker_treatment']` does not list `masks`. The verifier is unchanged and arm-agnostic; document this in the manifest.
9. **Timeline:** code and tests about 1 day; T3 and pilot half a day; registered run overnight; then analysis and paper. Feasible only if the CVPR 2027 deadline is mid-November or later, so check the official dates.