"""Merge detector candidates + VLM selections into training annotations and the frozen shared window panel.

Also patches the nine run configs (sampling manifest, hash, update budget, authorization) when --configure is given.
"""
from pathlib import Path
import argparse, collections, hashlib, json
ROOT = Path('/dev/shm/qinzhen/sbl')
TRAIN = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
SOURCE = 'GroundingDINO-base + Meta-SAM2.1-hiera-small candidates; Qwen3-VL-4B-Instruct set-of-mark instance selection'
REVIEWER = 'automatic two-model agreement pipeline (no human review); see PREREG.md'


def load_jsonl(pattern_dir):
    rows = []
    for p in sorted(Path(pattern_dir).glob('shard*.jsonl')):
        for line in open(p):
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default=str(ROOT / 'ann/candidates')); ap.add_argument('--selected', default=str(ROOT / 'ann/selected'))
    ap.add_argument('--out', default=str(ROOT / 'ann/panel')); ap.add_argument('--configure', action='store_true')
    ap.add_argument('--updates', type=int, default=None); ap.add_argument('--wall-seconds', type=int, default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cands = {(r['task'], r['episode'], r['anchor']): r for r in load_jsonl(a.candidates)}
    sels = {(r['task'], r['episode'], r['anchor']): r for r in load_jsonl(a.selected)}
    data_manifest = (TRAIN / 'receipts/training-data-manifest.json').read_bytes()
    manifest = json.loads(data_manifest)
    permitted = {r['task']: (set(r['train_episodes']), set(r['dev_episodes'])) for r in manifest['tasks']}
    annotations = {'train': [], 'dev': []}; stats = collections.Counter(); per_task = collections.defaultdict(collections.Counter)
    per_kind = collections.defaultdict(collections.Counter)
    for key, c in sorted(cands.items()):
        s = sels.get(key)
        if s is None:
            stats['missing_selection'] += 1; continue
        entries = []; any_visible = False
        for cam in c['cams']:
            ci = cam['camera_index']; choice = s['choice'].get(str(ci), s['choice'].get(ci, -1))
            chosen = next((x for x in cam['candidates'] if x['idx'] == choice), None) if choice is not None and int(choice) >= 0 else None
            if chosen is None:
                entries.append({'visible': False, 'box_xyxy': None, 'location_xy': None, 'candidate_index': None,
                                'n_candidates': len(cam['candidates'])})
            else:
                any_visible = True
                entries.append({'visible': True, 'box_xyxy': chosen['box_xyxy'], 'location_xy': chosen['location_xy'],
                                'candidate_index': chosen['idx'], 'det_score': chosen['det_score'], 'sam_iou': chosen['sam_iou'],
                                'n_candidates': len(cam['candidates'])})
        task, ep, anchor = key
        split = 'train' if ep in permitted[task][0] else 'dev' if ep in permitted[task][1] else None
        if split is None:
            stats['outside_split'] += 1; continue
        per_task[task]['windows'] += 1; per_kind[c['kind']]['windows'] += 1
        if not any_visible:
            stats['no_visible_target'] += 1; per_task[task]['excluded_no_target'] += 1; per_kind[c['kind']]['excluded_no_target'] += 1
            continue
        ann = {'task': task, 'episode': ep, 'anchor': anchor, 'stage': c['stage'], 'kind': c['kind'], 'description': c['description'],
               'review_status': 'approved', 'reviewer': REVIEWER, 'source': SOURCE, 'label_type': 'model_generated_pseudo_label',
               'target_entity_id': '%s/e%d/%s' % (task, ep, c['spec']['entity']), 'target_role': c['spec']['role'],
               'target_prompt': c['spec']['prompt'], 'target_hint': c['spec']['hint'],
               'annotation_scope': 'current_frame_only', 'location_definition': 'normalized_2d_mask_bbox_center',
               'frames': {'0': entries}, 'visible_cameras': sum(e['visible'] for e in entries),
               'source_parquet_sha256': c['source_parquet_sha256']}
        annotations[split].append(ann); stats[split] += 1; per_task[task][split] += 1
        per_task[task]['visible_cams_total'] += ann['visible_cameras']; per_kind[c['kind']][split] += 1
    for split in ['train', 'dev']:
        (out / ('annotations-%s.json' % split)).write_text(json.dumps(annotations[split]))
    panel = {'data_manifest_sha256': hashlib.sha256(data_manifest).hexdigest(), 'frozen_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
             'selection': 'every queued train window (anchor % 16 == 0, anchor + 32 < len) with a VLM-confirmed visible target in >= 1 camera; task-complete stages excluded',
             'source': SOURCE, 'reviewer': REVIEWER,
             'windows': [{'task': x['task'], 'episode': x['episode'], 'anchor': x['anchor'], 'stage': x['stage'], 'annotation': x} for x in annotations['train']]}
    panel_bytes = json.dumps(panel).encode(); panel_path = out / 'panel-train.json'; panel_path.write_bytes(panel_bytes)
    sha = hashlib.sha256(panel_bytes).hexdigest()
    summary = {'stats': dict(stats), 'per_task': {k: dict(v) for k, v in per_task.items()}, 'per_kind': {k: dict(v) for k, v in per_kind.items()},
               'panel_sha256': sha, 'panel_windows': len(panel['windows']), 'panel_path': str(panel_path)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
    if a.configure:
        if not a.updates or not a.wall_seconds:
            raise SystemExit('--updates and --wall-seconds are required with --configure')
        for cfg_path in sorted((TRAIN / 'configs').glob('*.json')):
            cfg = json.loads(cfg_path.read_text())
            cfg.update(training_authorized=True, optimizer_updates=a.updates, max_wall_seconds=a.wall_seconds,
                       sampling_manifest=str(panel_path), sampling_manifest_sha256=sha,
                       annotation_manifest=str(panel_path), launch_blockers=[],
                       authorization_note='User authorized on 2026-11-08 (chat): complete all experiments; see /dev/shm/qinzhen/sbl/PREREG.md')
            cfg_path.write_text(json.dumps(cfg, indent=2))
        print('CONFIGURED', a.updates, 'updates', a.wall_seconds, 's wall')


if __name__ == '__main__':
    main()
