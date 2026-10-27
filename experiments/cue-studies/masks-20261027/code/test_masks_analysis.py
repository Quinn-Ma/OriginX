"""CPU tests for the point/masks prospective analyzer and seed audit; synthetic evidence only.

Runs with `python -m unittest` (Windows) or pytest (WSL sim-python). Never touches
runs/, model services or the simulator; every synthetic study lives in a temp dir.
"""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / 'analyze_masks_validation.py'
AUDIT = HERE / 'audit_unused_seeds.py'
CAMS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
SEED_BASE = 700000  # synthetic seeds only; real seeds are chosen by the audit


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding='utf-8')


def rehash(manifest):
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    return {**body, 'manifest_sha256': canonical(body)}


SUBJECT_NAME = 'mug'
SUBJECT_ID = 'O' + hashlib.sha256(('objects/' + SUBJECT_NAME).encode()).hexdigest()[:10]
PLAN = [{'operation': 'grasp', 'primary_target_id': SUBJECT_ID}, {'operation': 'place', 'primary_target_id': 'R1'}]
GEOMETRY = {'entities': [], 'region_children': {}, 'object_anchors': {}, 'audit': {'mask_frame': {'note': 'synthetic'}}}


def mask_event(arm, step, operation, *, inside=0, wrist_visible_in_grasp=False, layout='camera_major',
               field='inside_mask_pixels_changed'):
    """Synthetic event in the schema the masks runtime is planned to log.

    ``field`` names the inside-pixel leaf: inside_mask_pixels_changed (masks arm, and the older point-arm
    name) or point_cue_pixels_inside_masks (renamed point-arm quantity)."""
    roles = ['target'] if operation == 'grasp' else ['destination', 'subject']
    cameras = {}
    for cam in CAMS:
        visible = not (operation == 'grasp' and cam == CAMS[2] and not wrist_visible_in_grasp)
        cameras[cam] = {role: {'mask_px': 50 if visible else 0, 'ring_px': 40 if visible else 0,
                               'mask_sha256': 'm' * 64, 'ring_sha256': 'r' * 64,
                               field: inside if visible else 0} for role in roles}
    event = {'kind': 'mask_presentation', 'arm': arm, 'status': 'drawn' if arm == 'masks' else 'point_control',
             'operation': operation, 'drawn': arm == 'masks', 'logging_only': True,
             'used_by_policy_or_planner': False, 'depth_sha256': {c: 'd' * 64 for c in CAMS}}
    if layout == 'camera_major':
        event['cameras'] = cameras
    elif layout == 'role_major':
        event['roles'] = {role: {'cameras': {cam: cameras[cam][role] for cam in CAMS}} for role in roles}
    elif layout == 'list':
        event['detail'] = [{'camera': cam, 'role': role, **cameras[cam][role]} for cam in CAMS for role in roles]
    return event


class Study:
    """Synthetic registered study whose manifest passes manifest_check after patching
    the analyzer's parent constants to the synthetic parent."""

    def __init__(self, a, base, pairs=12, workers=1, seed_base=SEED_BASE, purpose=None):
        self.a, self.base = a, Path(base)
        purpose = a.PURPOSE if purpose is None else purpose
        self.root = self.base / 'root'; self.run = self.base / 'run'
        for rel, text in (('src/frozen.py', 'frozen = 1\n'), ('src/new.py', 'new = 2\n')):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True); (self.root / rel).write_text(text)
        frozen_sources = {'src/frozen.py': sha(self.root / 'src/frozen.py')}
        parent = rehash({'purpose': 'synthetic_parent', 'seeds': [seed_base - 1], 'source_files': frozen_sources,
                         'checkpoint_sha256': {'robocasa/xr1': 'c' * 64}})
        write_json(self.root / 'parent/manifest.json', parent)
        a.PARENT_PATH = 'parent/manifest.json'; a.PARENT_MANIFEST_SHA256 = parent['manifest_sha256']
        self.seeds = [seed_base + i for i in range(pairs)]
        audit = {'status': 'PASS', 'ordered_seeds': self.seeds, 'selected_seeds': self.seeds, 'all_selected_absent': True,
                 'selected_seed_prior_matches': {str(s): [] for s in self.seeds}}
        write_json(self.root / 'audit/unused-seeds-v1.json', audit)
        rows, slots = [], Counter()
        for index, seed in enumerate(self.seeds):
            worker = index % workers
            order = a.ARMS if index % 2 == 0 else tuple(reversed(a.ARMS))
            key = canonical({'task': 'CoffeeSetupMug', 'seed': seed, 'model': 'xr1', 'purpose': purpose})
            for position, arm in enumerate(order):
                row = {'task': 'CoffeeSetupMug', 'benchmark': 'robocasa', 'model': 'xr1', 'arm': 'full',
                       'comparison_arm': arm, 'seed': seed, 'max_steps': 600, 'pair_index': index,
                       'pair_position': position, 'order_in_worker': slots[worker], 'worker': worker, 'pair_key': key}
                slots[worker] += 1
                rows.append({**row, 'case_id': canonical(row)})
        self.manifest = rehash({'purpose': purpose, 'cases': rows, 'seeds': self.seeds, 'planned_pairs': pairs,
            'planned_cases': 2 * pairs,
            'seed_audit': {'path': 'audit/unused-seeds-v1.json', 'sha256': sha(self.root / 'audit/unused-seeds-v1.json'),
                           'ordered_seeds': self.seeds},
            'source_files': {**frozen_sources, 'src/new.py': sha(self.root / 'src/new.py')},
            'parent': {'path': 'parent/manifest.json', 'sha256': sha(self.root / 'parent/manifest.json'),
                       'manifest_sha256': parent['manifest_sha256']},
            'checkpoint_sha256': parent['checkpoint_sha256'],
            'treatments': {'point': 'point control', 'masks': 'exact mask outlines'}})
        if purpose == a.PILOT_PURPOSE:
            body = {k:v for k,v in self.manifest.items() if k not in ('seed_audit','manifest_sha256')}
            body.update(prospective_validation=False, seed_role='development_seeds_already_used_not_prospective_evidence')
            self.manifest = rehash(body)
        write_json(self.run / 'manifest.json', self.manifest)

    def case(self, seed, arm):
        return next(c for c in self.manifest['cases'] if c['seed'] == seed and c['comparison_arm'] == arm)

    def pair(self, seed):
        return sorted((c for c in self.manifest['cases'] if c['seed'] == seed), key=lambda c: c['pair_position'])

    def write_case(self, source, case, outcome, *, steps=32, reference=None, mask=True, inside=0, xy=0.02,
                   worker=None, partial=False, point_field='inside_mask_pixels_changed'):
        """outcome: 'success' | 'failure' | 'timeout' (UNKNOWN infrastructure) ; returns result path.

        ``inside``: int for both arms or {arm: int}; ``point_field``: the point arm's inside-pixel field name."""
        a = self.a; source = Path(source); worker = case['worker'] if worker is None else worker
        if isinstance(inside, dict):
            inside = inside.get(case['comparison_arm'], 0)
        field = point_field if case['comparison_arm'] == 'point' else 'inside_mask_pixels_changed'
        folder = source / f'worker-{worker}' / 'run'
        cid, arm, position, seed = case['case_id'], case['comparison_arm'], case['pair_position'], case['seed']
        if outcome == 'timeout':
            path = folder / 'results' / f'{cid}.json'
            write_json(path, {**{k: case[k] for k in case}, 'purpose': self.manifest['purpose'], 'manifest_sha256': self.manifest['manifest_sha256'],
                              'status': 'unknown', 'outcome_category': 'timeout', 'reason': 'TimeoutError: planner HTTP'})
            write_json(folder / 'traces' / cid / 'trace.json', {'status': 'partial', 'records': 5, 'last_step': 3})
            return path
        success = outcome == 'success'
        fp = hashlib.sha256(str(seed).encode()).hexdigest()
        desc = {'sha256': hashlib.sha256(b'rgb' + str(seed).encode()).hexdigest(), 'dtype': '|u1', 'shape': [3, 256, 256, 3]}
        raw = {'rgb': desc, 'proprio': {'sha256': 'p' * 64, 'dtype': '<f4', 'shape': [16]}, 'instruction': 'Place the mug.',
               'camera_names': list(CAMS), 'step': 0}
        obs = {'raw': raw, 'candidates_visible': [{'id': SUBJECT_ID}], 'candidates_all': [{'id': SUBJECT_ID}, {'id': 'R1'}]}
        sig = {'initial_fingerprint': fp, 'rgb': desc, 'proprio': raw['proprio'], 'instruction': raw['instruction'],
               'camera_names': raw['camera_names'], 'step': 0,
               'candidates_visible_sha256': canonical(obs['candidates_visible']),
               'candidates_all_sha256': canonical(obs['candidates_all']), 'shared_augmentation_sha256': canonical(GEOMETRY)}
        rows, receipts = [], []
        def add(kind, step, payload): rows.append({'sequence': len(rows), 'kind': kind, 'step': step, 'payload': payload})
        add('event', 0, {'kind': 'initial_comparison', 'actual': sig,
                         'gate': {'passed': True if position == 1 else None, 'injected': False}})
        add('event', 0, {'kind': 'initial_reset', 'instruction': raw['instruction'], 'initial_fingerprint': fp})
        for step in range(steps):
            add('event', step, {'kind': 'shared_hierarchical_geometry', **GEOMETRY, 'same_extractor_for_all_arms': True})
            if step == 0:
                add('event', 0, {'kind': 'initial_plan_comparison', 'role': arm, 'passed': True if position == 1 else None,
                                 'pair_position': position, 'expected_plan': PLAN if position == 1 else None,
                                 'natural_plan': PLAN, 'injected': False})
            if step % 16 == 0:
                receipt = {'step': step, 'plan': PLAN, 'status': 'focused', 'calls': [], 'segment': 0, 'segment_reset': False}
                receipts.append(receipt); add('event', step, {'kind': 'prepare', 'receipt': receipt})
            add('observation', step, obs if step == 0 else {'raw': {'step': step}})
            if mask:
                operation = 'grasp' if step < steps // 2 else 'place'
                add('event', step, mask_event(arm, step, operation, inside=inside if step == 1 else 0, field=field))
            if step % 16 == 0:
                add('action_chunk', step, {'actions': {'sha256': 'x'}})
            add('event', step, {'kind': 'action_executed', 'next_step': step + 1, 'action_modified': False,
                                'scorer_success': success and step == steps - 1, 'environment_done': False})
            add('event', step + 1, {'kind': 'contact_diagnostic', 'logging_only': True, 'used_by_policy_or_planner': False,
                                    'diagnostic': {'object_name': SUBJECT_NAME, 'object_position': [xy, 0.0, 0.9]}})
            add('event', step + 1, {'kind': 'placement_geometry_diagnostic', 'logging_only': True,
                                    'used_by_policy_or_planner': False, 'region_positions_world_m': {'R1': [0.0, 0.0, 0.9]}})
        add('event', steps, {'kind': 'terminal', 'raw_observation': {'step': steps}, 'success_observed': success,
                             'environment_done': not success})
        trace = folder / 'traces' / cid; trace.mkdir(parents=True, exist_ok=True)
        (trace / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        metadata = {'status': 'partial' if partial else 'complete', 'all_requested_fields_captured': True,
                    'records': len(rows), 'counts': dict(Counter(r['kind'] for r in rows)), 'last_step': steps,
                    'failures': [], 'coverage_gaps': [], 'unsupported_fields': 0}
        write_json(trace / 'trace.json', metadata)
        chunks = len(range(0, steps, 16))
        result = {**{k: case[k] for k in case}, 'purpose': self.manifest['purpose'], 'manifest_sha256': self.manifest['manifest_sha256'],
                  'status': outcome, 'steps': steps, 'success_observed': success, 'events': receipts,
                  'initial_fingerprint': fp, 'initial_signature': sig, 'initial_plan': PLAN, 'proposal_count': chunks,
                  'native_chunk_sha256': ['x'] * chunks, 'parameter_updates': 0, 'action_modifications': 0,
                  'trace': metadata, 'pair_reference': reference, 'elapsed_seconds': 1.0,
                  'natural_plan_matches_paired_reference': True if position == 1 else None, 'comparison_arm': arm}
        path = folder / 'results' / f'{cid}.json'; write_json(path, result)
        return path

    def write_pair(self, source, seed, outcomes, **kw):
        """outcomes: {'point': ..., 'masks': ...}; writes first arm then the second with its reference."""
        first, second = self.pair(seed); fixed = kw.pop('steps', None)
        steps = {arm: fixed or (16 * (2 + i) if outcomes[arm] == 'success' else 32) for i, arm in enumerate(self.a.ARMS)}
        path = self.write_case(source, first, outcomes[first['comparison_arm']], steps=steps[first['comparison_arm']], **kw)
        reference = {'case_id': first['case_id'], 'comparison_arm': first['comparison_arm'], 'result_sha256': sha(path),
                     'injected': False, 'expected_initial': json.loads(path.read_text())['initial_signature']
                     if outcomes[first['comparison_arm']] != 'timeout' else None, 'expected_plan': PLAN}
        self.write_case(source, second, outcomes[second['comparison_arm']], steps=steps[second['comparison_arm']],
                        reference=reference, **kw)


def pattern(n, wins, losses, both_success):
    """Outcome dicts per pair: masks-only wins, point-only wins, both success, rest both failure."""
    rows = []
    for i in range(n):
        if i < wins: rows.append({'point': 'failure', 'masks': 'success'})
        elif i < wins + losses: rows.append({'point': 'success', 'masks': 'failure'})
        elif i < wins + losses + both_success: rows.append({'point': 'success', 'masks': 'success'})
        else: rows.append({'point': 'failure', 'masks': 'failure'})
    return rows


class Base(unittest.TestCase):
    def setUp(self):
        self.a = load_module(ANALYZER, 'masks_analysis_under_test_' + self._testMethodName)
        self.tmp = tempfile.TemporaryDirectory(prefix='masks-analysis-test-'); self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def study(self, pairs=12, workers=1, outcomes=None, **kw):
        s = Study(self.a, self.base, pairs, workers)
        for seed, out in zip(s.seeds, outcomes or pattern(pairs, 0, 0, pairs)):
            s.write_pair(s.run, seed, out, **kw)
        return s


class PilotEngineeringTests(Base):
    def pilot(self, outcomes=None):
        s = Study(self.a, self.base, 3, 1, seed_base=self.a.DEVELOPMENT_SEEDS[0],
                  purpose=self.a.PILOT_PURPOSE)
        for seed, outcome in zip(s.seeds, outcomes or pattern(3, 1, 1, 0)):
            s.write_pair(s.run, seed, outcome, steps=16, point_field='point_cue_pixels_inside_masks')
        return s

    def test_explicit_pilot_reads_six_traces_without_effect_estimation(self):
        s = self.pilot()
        prospective_purpose = self.a.PURPOSE
        with self.assertRaisesRegex(ValueError, 'prospective purpose'):
            self.a.analyze(s.run, s.root)
        report = self.a.analyze_pilot(s.run, s.root)
        self.assertEqual(report['status'], 'PILOT_ENGINEERING_OFFLINE_ANALYSIS')
        self.assertEqual(report['purpose'], self.a.PILOT_PURPOSE)
        self.assertEqual((report['planned_pairs'], report['planned_cases']), (3, 6))
        self.assertTrue(report['g9_complete_trace_gate']['passed'])
        self.assertEqual(report['g9_complete_trace_gate']['complete_valid_traces'], 6)
        self.assertEqual(report['g9_complete_trace_gate']['invalid_or_missing_case_ids'], [])
        self.assertEqual(report['descriptive_outcome_counts'],
                         {'point': {'failure': 2, 'success': 1}, 'masks': {'failure': 2, 'success': 1}})
        for key in ('prospective_evidence', 'formal_freeze_performed', 'effect_estimation_performed'):
            self.assertIs(report[key], False)
        for key in ('paired', 'arms', 'risk_difference_masks_minus_point', 'mcnemar_exact_two_sided_p'):
            self.assertNotIn(key, report)
        self.assertEqual(self.a.PURPOSE, prospective_purpose)
        self.assertEqual(report['new_env_steps'], 0)

    def test_pilot_rejects_other_purpose_seeds_count_or_prospective_flags(self):
        s = self.pilot()
        for change in (lambda m: m.update(purpose=self.a.PURPOSE),
                       lambda m: m.update(seeds=[20261025, 20261026, 999999]),
                       lambda m: m.update(planned_cases=5),
                       lambda m: m.update(prospective_validation=True),
                       lambda m: m.update(seed_role='prospective'),
                       lambda m: m.update(seed_audit={'status':'PASS'})):
            value = copy.deepcopy(s.manifest); change(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.a.pilot_manifest_check(rehash(value), s.root)
        write_json(s.run / 'resume-plan.json', {})
        with self.assertRaisesRegex(ValueError, 'does not admit'):
            self.a.analyze_pilot(s.run, s.root)

    def test_pilot_partial_trace_keeps_case_unknown_and_fails_g9(self):
        s = self.pilot()
        first = s.pair(s.seeds[0])[0]
        path = s.run / f"worker-{first['worker']}/run/traces/{first['case_id']}/trace.json"
        value = json.loads(path.read_text()); value['status'] = 'partial'; write_json(path, value)
        report = self.a.analyze_pilot(s.run, s.root)
        self.assertFalse(report['g9_complete_trace_gate']['passed'])
        self.assertEqual(report['g9_complete_trace_gate']['complete_valid_traces'], 5)
        row = next(c for c in report['cases'] if c['case_id'] == first['case_id'])
        self.assertEqual(row['outcome'], 'unknown')
        self.assertIn('saved_trace_not_complete', row['integrity_issues'])

    def test_pilot_wrong_result_purpose_and_missing_mask_cadence_fail_closed(self):
        s = self.pilot()
        first, second = s.pair(s.seeds[1])
        path = s.run / f"worker-{first['worker']}/run/results/{first['case_id']}.json"
        value = json.loads(path.read_text()); value['purpose'] = self.a.PURPOSE; write_json(path, value)
        report = self.a.analyze_pilot(s.run, s.root)
        row = next(c for c in report['cases'] if c['case_id'] == first['case_id'])
        self.assertEqual(row['outcome'], 'unknown')
        self.assertIn('result_purpose_or_manifest_mismatch', row['integrity_issues'])
        self.assertFalse(report['g9_complete_trace_gate']['passed'])
        # The scorer trace may be complete while presentation events are absent.
        s.write_pair(s.run, s.seeds[1], {'point':'success','masks':'success'}, steps=16, mask=False)
        report = self.a.analyze_pilot(s.run, s.root)
        self.assertFalse(report['g9_complete_trace_gate']['passed'])
        self.assertEqual(set(report['g9_complete_trace_gate']['mask_event_cadence_issues']),
                         {first['case_id'], second['case_id']})


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = load_module(ANALYZER, 'masks_analysis_statistics')

    def test_exact_mcnemar_equals_bruteforce_binomial(self):
        for w in range(21):
            for l in range(21 - w):
                d = w + l
                tail = sum(Fraction(math.comb(d, i), 2 ** d) for i in range(min(w, l) + 1)) if d else Fraction(1, 2)
                want = min(Fraction(1), 2 * tail) if d else Fraction(1)
                self.assertAlmostEqual(self.a.exact_mcnemar(w, l), float(want), places=15)
                self.assertEqual(self.a.exact_mcnemar(w, l), self.a.exact_mcnemar(l, w))
        self.assertEqual(self.a.exact_mcnemar(6, 0), .03125); self.assertEqual(self.a.exact_mcnemar(5, 0), .0625)
        self.assertEqual(self.a.exact_mcnemar(0, 0), 1.)

    def test_wilson_closed_form_and_boundaries(self):
        z = 1.959963984540054
        for n in (12, 24, 36):
            for k in range(n + 1):
                lo, hi = self.a.wilson(k, n)
                centre = (k + z * z / 2) / (n + z * z); half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
                self.assertAlmostEqual(lo, 0. if k == 0 else max(0., centre - half), places=12)
                self.assertAlmostEqual(hi, 1. if k == n else min(1., centre + half), places=12)
                self.assertLessEqual(lo, k / n); self.assertGreaterEqual(hi, k / n)
        self.assertIsNone(self.a.wilson(0, 0))

    def test_clopper_pearson_exact_tails(self):
        for n in (12, 24, 36):
            self.assertAlmostEqual(self.a.clopper_pearson(0, n)[1], 1 - .025 ** (1 / n), places=10)
            self.assertAlmostEqual(self.a.clopper_pearson(n, n)[0], .025 ** (1 / n), places=10)
            for k in (1, n // 3, n - 1):
                lo, hi = self.a.clopper_pearson(k, n)
                self.assertAlmostEqual(self.a.binom_cdf(k, n, hi), .025, places=9)
                self.assertAlmostEqual(1 - self.a.binom_cdf(k - 1, n, lo), .025, places=9)

    def test_paired_ci_conservative_and_contains_estimate(self):
        for n in (12, 24, 36):
            for w in range(0, n + 1, 3):
                for l in range(0, n - w + 1, 3):
                    lo, hi = self.a.paired_effect_ci(w, l, n)
                    self.assertTrue(-1 <= lo <= (w - l) / n <= hi <= 1)
        # Bonferroni CP is never narrower than the 95% CP of either margin alone.
        lo, hi = self.a.paired_effect_ci(0, 0, 12)
        self.assertLess(lo, -self.a.clopper_pearson(0, 12)[1] + 1e-12)
        self.assertIsNone(self.a.paired_effect_ci(0, 0, 0))

    def test_design_power_matches_direct_sum(self):
        # p_loss = 0: rejection iff wins >= 6 at n = 12 (2/2**6 = .03125 <= .05; 2/2**5 > .05).
        want = sum(math.comb(12, w) * .5 ** 12 for w in range(6, 13))
        self.assertAlmostEqual(self.a.exact_design_power(12, .5, 0.), want, places=12)
        self.assertLess(self.a.exact_design_power(12, .3, .1), self.a.exact_design_power(36, .3, .1))
        with self.assertRaises(ValueError): self.a.exact_design_power(12, .7, .4)


class UnknownHandlingTests(Base):
    def test_classify_table(self):
        c = self.a.classify; e = {'scorer_true_steps': []}; s = {'terminals': [{'environment_done': False}]}
        self.assertEqual(c({}, [], {}, {}), ('unknown', 'not_yet_recorded'))
        self.assertEqual(c({'status': 'unknown', 'reason': 'TimeoutError: deadline'}, [], {}, {}), ('unknown', 'infrastructure_timeout'))
        self.assertEqual(c({'status': 'unknown', 'outcome_category': 'timeout'}, [], {}, {}), ('unknown', 'infrastructure_timeout'))
        for category in ('invalid_pair', 'prerequisite_failed', 'unattempted', 'infrastructure_error'):
            self.assertEqual(c({'status': 'unknown', 'outcome_category': category}, [], {}, {}), ('unknown', category))
        self.assertEqual(c({'status': 'unknown'}, [], {}, {}), ('unknown', 'invalid_or_incomplete_execution'))
        self.assertEqual(c({'status': 'success'}, ['trace_partial'], {}, {}), ('unknown', 'invalid_evidence'))
        self.assertEqual(c({'status': 'failure', 'steps': 600, 'success_observed': False}, [], s, e), ('failure', 'task_horizon_timeout'))
        self.assertEqual(c({'status': 'failure', 'steps': 100, 'success_observed': False}, [], s, e), ('unknown', 'inconsistent_task_termination'))
        self.assertEqual(c({'status': 'success', 'steps': 50, 'success_observed': True}, [], s, e), ('unknown', 'inconsistent_task_termination'))

    def test_complete_synthetic_study_all_eligible(self):
        s = self.study(outcomes=pattern(12, 3, 1, 4))
        r = self.a.analyze(s.run, s.root)
        self.assertEqual([c['integrity_issues'] for c in r['cases']], [[]] * 24)
        p = r['paired']
        self.assertEqual((p['planned_pairs'], p['eligible_pairs'], p['masks_only_success'], p['point_only_success'],
                          p['both_success'], p['both_failure']), (12, 12, 3, 1, 4, 4))
        self.assertEqual(p['mcnemar_exact_two_sided_p'], self.a.exact_mcnemar(3, 1))
        self.assertAlmostEqual(p['risk_difference_masks_minus_point'], 2 / 12)
        self.assertEqual(p['risk_difference_conservative95'], self.a.paired_effect_ci(3, 1, 12))
        self.assertEqual(r['arms']['masks']['paired_eligible_success_rate_wilson95'], self.a.wilson(7, 12))
        self.assertEqual(r['arms']['point']['success'], 5)
        self.assertEqual(p['final_xy_paired_delta_cm']['n'], 12)
        self.assertAlmostEqual(p['final_xy_paired_delta_cm']['median'], 0.)
        self.assertIn('masks_case_id', r['pairs'][0]); self.assertNotIn('regions_case_id', r['pairs'][0])

    def test_unknown_pair_excluded_but_accounted(self):
        outcomes = pattern(12, 0, 0, 12); outcomes[4] = {'point': 'success', 'masks': 'timeout'}
        s = self.study(outcomes=outcomes)
        r = self.a.analyze(s.run, s.root)
        self.assertEqual(r['paired']['eligible_pairs'], 11); self.assertEqual(r['paired']['excluded_unknown_or_mismatched_pairs'], 1)
        masks = r['arms']['masks']
        self.assertEqual((masks['success'], masks['unknown'], masks['planned']), (11, 1, 12))
        self.assertEqual(masks['accounting'], {'benchmark_success': 11, 'infrastructure_timeout': 1})
        self.assertEqual(masks['registered_success_fraction_identification_bounds'], [11 / 12, 12 / 12])
        self.assertEqual(r['arms']['point']['paired_eligible_denominator'], 11)
        self.assertFalse(next(p for p in r['pairs'] if p['seed'] == s.seeds[4])['eligible'])

    def test_missing_result_and_partial_trace_never_known(self):
        outcomes = pattern(12, 0, 0, 12)
        s = self.study(outcomes=outcomes)
        os.remove(s.run / 'worker-0/run/results' / (s.case(s.seeds[2], 'point')['case_id'] + '.json'))
        first, second = s.pair(s.seeds[3])
        s.write_case(s.run, second, 'success', steps=48, partial=True,
                     reference=json.loads((s.run / 'worker-0/run/results' / (second['case_id'] + '.json')).read_text())['pair_reference'])
        r = self.a.analyze(s.run, s.root)
        missing = next(c for c in r['cases'] if c['case_id'] == s.case(s.seeds[2], 'point')['case_id'])
        partial = next(c for c in r['cases'] if c['case_id'] == second['case_id'])
        self.assertEqual((missing['outcome'], missing['category']), ('unknown', 'not_yet_recorded'))
        self.assertEqual((partial['outcome'], partial['category']), ('unknown', 'invalid_evidence'))
        self.assertIn('saved_trace_not_complete', partial['integrity_issues'])
        self.assertEqual(r['paired']['eligible_pairs'], 10)

    def test_reference_tamper_excludes_pair(self):
        s = self.study()
        _, second = s.pair(s.seeds[0])
        path = s.run / 'worker-0/run/results' / (second['case_id'] + '.json')
        value = json.loads(path.read_text()); value['pair_reference']['result_sha256'] = '0' * 64; write_json(path, value)
        r = self.a.analyze(s.run, s.root)
        self.assertFalse(r['pairs'][0]['eligible']); self.assertEqual(r['paired']['eligible_pairs'], 11)

    def test_modified_registered_source_invalidates_all_outcomes(self):
        s = self.study()
        (s.root / 'src/new.py').write_text('new = 3\n')
        r = self.a.analyze(s.run, s.root)
        self.assertEqual(r['source_mismatches'], ['src/new.py'])
        self.assertTrue(all(c['outcome'] == 'unknown' for c in r['cases']))
        self.assertEqual(r['paired']['eligible_pairs'], 0); self.assertIsNone(r['paired']['mcnemar_exact_two_sided_p'])


class MultiWorkerTests(Base):
    def test_cases_read_from_registered_worker_directories(self):
        s = self.study(pairs=12, workers=3, outcomes=pattern(12, 2, 1, 5))
        self.assertEqual({c['worker'] for c in s.manifest['cases']}, {0, 1, 2})
        r = self.a.analyze(s.run, s.root)
        self.assertEqual(r['paired']['eligible_pairs'], 12)
        for c in r['cases']:
            self.assertIn(f"worker-{c['worker']}", Path(c['result_path']).parts)
            self.assertEqual(c['integrity_issues'], [])

    def test_copy_under_unregistered_worker_is_integrity_issue(self):
        s = self.study(pairs=12, workers=2)
        case = s.case(s.seeds[1], 'masks'); self.assertEqual(case['worker'], 1)
        src = s.run / 'worker-1/run/results' / (case['case_id'] + '.json')
        dst = s.run / 'worker-0/run/results' / src.name; shutil.copy(src, dst)
        r = self.a.analyze(s.run, s.root)
        row = next(c for c in r['cases'] if c['case_id'] == case['case_id'])
        self.assertIn('case_present_under_unregistered_worker:worker-0', row['integrity_issues'])
        self.assertEqual(row['outcome'], 'unknown'); self.assertEqual(r['paired']['eligible_pairs'], 11)

    def test_result_only_under_wrong_worker_is_not_found(self):
        s = self.study(pairs=12, workers=2)
        case = s.case(s.seeds[3], 'point')
        shutil.move(str(s.run / 'worker-1/run/results' / (case['case_id'] + '.json')), str(s.run / 'worker-0/run/results'))
        row = next(c for c in self.a.analyze(s.run, s.root)['cases'] if c['case_id'] == case['case_id'])
        self.assertEqual(row['outcome'], 'unknown')

    def test_invalid_worker_index_rejected(self):
        s = Study(self.a, self.base, 12, 1)
        bad = copy.deepcopy(s.manifest); row = bad['cases'][0]; row['worker'] = -1
        row['case_id'] = canonical({k: v for k, v in row.items() if k != 'case_id'})
        with self.assertRaisesRegex(ValueError, 'worker'): self.a.manifest_check(rehash(bad), s.root)
        dup = copy.deepcopy(s.manifest); row = dup['cases'][1]; row['order_in_worker'] = 0
        row['case_id'] = canonical({k: v for k, v in row.items() if k != 'case_id'})
        with self.assertRaisesRegex(ValueError, 'slot'): self.a.manifest_check(rehash(dup), s.root)


class PairCountTests(Base):
    def test_supported_pair_counts(self):
        for n, (w, l, b) in ((12, (2, 1, 4)), (24, (6, 1, 9)), (36, (9, 3, 12))):
            with self.subTest(pairs=n):
                base = self.base / f'n{n}'
                s = Study(self.a, base, n, workers=4 if n > 12 else 1)
                for seed, out in zip(s.seeds, pattern(n, w, l, b)): s.write_pair(s.run, seed, out, steps=16)
                r = self.a.analyze(s.run, s.root)
                p = r['paired']
                self.assertEqual((r['planned_pairs'], p['planned_pairs'], p['eligible_pairs']), (n, n, n))
                self.assertEqual((p['masks_only_success'], p['point_only_success'], p['both_success']), (w, l, b))
                self.assertEqual(p['mcnemar_exact_two_sided_p'], self.a.exact_mcnemar(w, l))
                self.assertEqual(r['arms']['masks']['planned'], n)
                self.assertTrue(any(f'fixed {n} new seed pairs' in x for x in r['limits']))
                self.assertTrue(any(f'complete {2 * n}-case' in x for x in r['limits']))

    def test_unsupported_or_inconsistent_counts_rejected(self):
        s = Study(self.a, self.base, 12, 1)
        for change, message in ((lambda m: m.update(planned_pairs=13), 'planned_pairs'),
                                (lambda m: m.update(planned_pairs='12'), 'planned_pairs'),
                                (lambda m: m.update(planned_cases=23), 'planned_cases'),
                                (lambda m: m.update(planned_pairs=24, planned_cases=48), 'pairs'),
                                (lambda m: m.update(seeds=m['seeds'][:11] + [m['seeds'][0]]), 'pairs'),
                                (lambda m: m['cases'].pop(), 'cases')):
            bad = copy.deepcopy(s.manifest); change(bad)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.a.manifest_check(rehash(bad), s.root)

    def test_counterbalance_and_identity_enforced(self):
        s = Study(self.a, self.base, 12, 1)
        bad = copy.deepcopy(s.manifest)
        for row in bad['cases']:
            if row['pair_index'] == 1:
                row['pair_position'] = 1 - row['pair_position']
                row['case_id'] = canonical({k: v for k, v in row.items() if k != 'case_id'})
        with self.assertRaisesRegex(ValueError, 'Counterbalanced'): self.a.manifest_check(rehash(bad), s.root)
        tampered = copy.deepcopy(s.manifest); tampered['cases'][0]['max_steps'] = 600; tampered['cases'][0]['seed'] = s.seeds[0]
        tampered['cases'][0]['task'] = 'Other'
        with self.assertRaises(ValueError): self.a.manifest_check(rehash(tampered), s.root)


class LineageTests(Base):
    def test_registration_lineage_checks(self):
        s = Study(self.a, self.base, 12, 1)
        self.assertEqual(self.a.manifest_check(s.manifest, s.root), [])
        mutations = [
            (lambda m: m.update(purpose='regions_point_unseen_seed_validation'), 'purpose', False),
            (lambda m: m.update(treatments={'point': 'x', 'regions': 'y'}), 'treatments', False),
            (lambda m: m['parent'].update(manifest_sha256='0' * 64), 'parent', False),
            (lambda m: m['parent'].update(path='other/manifest.json'), 'parent', False),
            (lambda m: m['parent'].update(sha256='0' * 64), 'Parent lineage', False),
            (lambda m: m['source_files'].pop('src/frozen.py'), 'Frozen source', False),
            (lambda m: m.update(checkpoint_sha256={'robocasa/xr1': 'd' * 64}), 'Checkpoint', False),
            (lambda m: m['seed_audit'].update(sha256='0' * 64), 'Seed audit', False),
            (lambda m: m.update(manifest_sha256='0' * 64), 'canonical', True)]
        for change, message, raw in mutations:
            bad = copy.deepcopy(s.manifest); change(bad)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.a.manifest_check(bad if raw else rehash(bad), s.root)
        self.assertEqual(self.a.PARENT_PATH, 'parent/manifest.json')
        fresh = load_module(ANALYZER, 'masks_analysis_constants')
        self.assertEqual((fresh.PURPOSE, fresh.ARMS), ('masks_point_unseen_seed_validation', ('point', 'masks')))
        self.assertEqual(fresh.PARENT_PATH, 'work/id60-regions-validation/manifest.json')
        self.assertEqual(fresh.PARENT_MANIFEST_SHA256, '9414bd6546022392a2c813f1b8999a050a2c5230592f6f67fcc61971964eb9e4')

    def test_seed_reuse_and_prior_evidence_rejected(self):
        s = Study(self.a, self.base, 12, 1)
        audit = json.loads((s.root / 'audit/unused-seeds-v1.json').read_text())
        audit['selected_seed_prior_matches'][str(s.seeds[0])] = ['runs/x/manifest.json']
        write_json(s.root / 'audit/unused-seeds-v1.json', audit)
        m = rehash({**s.manifest, 'seed_audit': {**s.manifest['seed_audit'], 'sha256': sha(s.root / 'audit/unused-seeds-v1.json')}})
        with self.assertRaisesRegex(ValueError, 'prior evidence'): self.a.manifest_check(m, s.root)
        self.a.DEVELOPMENT_SEEDS = (s.seeds[5],)
        with self.assertRaisesRegex(ValueError, 'Development seed'): self.a.manifest_check(s.manifest, s.root)

    def test_parent_seed_reuse_rejected(self):
        s = Study(self.a, self.base, 12, 1, seed_base=SEED_BASE)
        parent = json.loads((s.root / 'parent/manifest.json').read_text()); parent['seeds'] = [s.seeds[0]]
        parent = rehash(parent); write_json(s.root / 'parent/manifest.json', parent)
        self.a.PARENT_MANIFEST_SHA256 = parent['manifest_sha256']
        m = rehash({**s.manifest, 'parent': {'path': 'parent/manifest.json', 'sha256': sha(s.root / 'parent/manifest.json'),
                                            'manifest_sha256': parent['manifest_sha256']}})
        with self.assertRaisesRegex(ValueError, 'Parent study seed'): self.a.manifest_check(m, s.root)


class ResumeTests(Base):
    def split_study(self, done_pairs=4, zero_step=True, chained=False):
        s = Study(self.a, self.base, 12, 1)
        outcomes = pattern(12, 2, 1, 5)
        for seed, out in zip(s.seeds[:done_pairs], outcomes[:done_pairs]): s.write_pair(s.run, seed, out)
        owed = [c for c in s.manifest['cases'] if c['seed'] in s.seeds[done_pairs:]]
        zero = {}
        if zero_step:
            first = s.pair(s.seeds[done_pairs])[0]
            meta = s.run / 'worker-0/run/traces' / first['case_id'] / 'trace.json'
            write_json(meta, {'status': 'partial', 'records': 0, 'last_step': -1, 'started_utc': 'synthetic'})
            zero = {first['case_id']: {'trace_sha256': sha(meta), 'records': 0, 'last_step': -1, 'started_utc': 'synthetic'}}
        completed = {c['case_id']: sha(s.run / 'worker-0/run/results' / (c['case_id'] + '.json'))
                     for c in s.manifest['cases'] if c['seed'] in s.seeds[:done_pairs]}
        resume = self.base / 'resume1'
        write_json(resume / 'manifest.json', s.manifest)
        plan = {'prior_output': str(s.run), 'prior_completed_result_sha256': completed,
                'prior_interrupted_zero_step': zero, 'owed_case_ids': [c['case_id'] for c in owed]}
        write_json(resume / 'resume-plan.json', plan)
        stop = 8 if chained else 12
        for seed, out in zip(s.seeds[done_pairs:stop], outcomes[done_pairs:stop]): s.write_pair(resume, seed, out)
        dirs = [resume]
        if chained:
            resume2 = self.base / 'resume2'; write_json(resume2 / 'manifest.json', s.manifest)
            done = {c['case_id']: sha((s.run if c['seed'] in s.seeds[:done_pairs] else resume) / 'worker-0/run/results' / (c['case_id'] + '.json'))
                    for c in s.manifest['cases'] if c['seed'] in s.seeds[:8]}
            write_json(resume2 / 'resume-plan.json', {'prior_output': str(resume), 'prior_completed_result_sha256': done,
                'prior_interrupted_zero_step': {}, 'owed_case_ids': [c['case_id'] for c in s.manifest['cases'] if c['seed'] in s.seeds[8:]]})
            for seed, out in zip(s.seeds[8:], outcomes[8:]): s.write_pair(resume2, seed, out)
            dirs.append(resume2)
        return s, dirs, plan

    def test_resume_combines_and_reports_sensitivity(self):
        s, dirs, plan = self.split_study()
        r = self.a.analyze(s.run, s.root, dirs)
        self.assertEqual(r['paired']['eligible_pairs'], 12)
        self.assertEqual(Counter(c['source_run'] for c in r['cases']), {'prior': 8, 'resume-1': 16})
        sens = r['registered_rule_sensitivity']
        self.assertEqual(sens['reattempted_case_ids'], sorted(plan['prior_interrupted_zero_step']))
        self.assertEqual(sens['paired']['eligible_pairs'], 11)
        self.assertTrue(r['limits'][0].startswith('1 registered case(s)'))
        self.assertEqual(r['resume_plans'][0]['owed_cases'], 16)

    def test_chained_resumes(self):
        s, dirs, _ = self.split_study(chained=True)
        r = self.a.analyze(s.run, s.root, dirs)
        self.assertEqual(Counter(c['source_run'] for c in r['cases']), {'prior': 8, 'resume-1': 8, 'resume-2': 8})
        self.assertEqual(r['paired']['eligible_pairs'], 12)

    def test_without_resume_owed_cases_stay_unknown(self):
        s, dirs, _ = self.split_study()
        r = self.a.analyze(s.run, s.root)
        self.assertEqual(r['paired']['eligible_pairs'], 4); self.assertNotIn('registered_rule_sensitivity', r)

    def test_resume_rule_violations_rejected(self):
        cases = []
        def changed_prior(s, d, p): (s.run / 'worker-0/run/results' / (next(iter(p['prior_completed_result_sha256'])) + '.json')).write_text('{}')
        def not_partition(s, d, p): write_json(d[0] / 'resume-plan.json', {**p, 'owed_case_ids': p['owed_case_ids'][:-1]})
        def split_pair(s, d, p):
            moved = p['owed_case_ids'][-1]; fake = s.run / 'worker-0/run/results' / (moved + '.json'); write_json(fake, {'status': 'failure'})
            completed = {**p['prior_completed_result_sha256'], moved: sha(fake)}
            write_json(d[0] / 'resume-plan.json', {**p, 'owed_case_ids': p['owed_case_ids'][:-1], 'prior_completed_result_sha256': completed})
        def other_prior(s, d, p): write_json(d[0] / 'resume-plan.json', {**p, 'prior_output': str(self.base / 'elsewhere')})
        def other_manifest(s, d, p): write_json(d[0] / 'manifest.json', {**s.manifest, 'note': 'changed'})
        def progressed(s, d, p):
            cid = next(iter(p['prior_interrupted_zero_step'])); meta = s.run / 'worker-0/run/traces' / cid / 'trace.json'
            write_json(meta, {'status': 'partial', 'records': 7, 'last_step': 2, 'started_utc': 'synthetic'})
            zero = {cid: {**p['prior_interrupted_zero_step'][cid], 'trace_sha256': sha(meta)}}
            write_json(d[0] / 'resume-plan.json', {**p, 'prior_interrupted_zero_step': zero})
        def unregistered_trace(s, d, p):
            cid = p['owed_case_ids'][-1]; write_json(s.run / 'worker-0/run/traces' / cid / 'trace.json', {'status': 'partial'})
        def owed_has_result(s, d, p):
            cid = p['owed_case_ids'][-1]; write_json(s.run / 'worker-0/run/results' / (cid + '.json'), {'status': 'unknown'})
        for mutate, message in ((changed_prior, 'Prior result changed'), (not_partition, 'partition'),
                                (split_pair, 'whole pairs'), (other_prior, 'different prior'),
                                (other_manifest, 'manifests differ'), (progressed, 'progressed'),
                                (unregistered_trace, 'unregistered prior trace'), (owed_has_result, 'already has a recorded result')):
            with self.subTest(message=message):
                shutil.rmtree(self.base); self.base.mkdir()
                s, dirs, plan = self.split_study(); mutate(s, dirs, plan)
                with self.assertRaisesRegex(ValueError, message): self.a.analyze(s.run, s.root, dirs)

    def test_only_the_regions_resume_schema_is_accepted(self):
        s, dirs, plan = self.split_study()
        zero = {cid: {'prior_result_sha256': None, 'prior_trace_present': True, 'prior_trace_sha256': v['trace_sha256']}
                for cid, v in plan['prior_interrupted_zero_step'].items()}
        variants = {
            'superseded masks schema': {'prior_output': plan['prior_output'], 'prior_site': 'local_5090',
                'prior_launch_receipt_status': None,
                'prior_task_outcome_result_sha256': plan['prior_completed_result_sha256'], 'prior_zero_step': zero,
                'prior_not_resumable': {}, 'owed_case_ids': plan['owed_case_ids'], 'not_owed_zero_step_case_ids': []},
            'extra key': {**plan, 'prior_site': 'local_5090'},
            'missing key': {k: v for k, v in plan.items() if k != 'prior_interrupted_zero_step'}}
        for label, bad in variants.items():
            with self.subTest(variant=label):
                write_json(dirs[0] / 'resume-plan.json', bad)
                with self.assertRaisesRegex(ValueError, 'Malformed resume plan'): self.a.analyze(s.run, s.root, dirs)
        write_json(dirs[0] / 'resume-plan.json', plan)
        self.assertEqual(self.a.analyze(s.run, s.root, dirs)['paired']['eligible_pairs'], 12)

    def test_wsl_prior_path_mapping(self):
        mapped = self.a.local('/mnt/d/codex/run')
        if Path('C:/').exists() and os.name == 'nt':
            self.assertEqual(mapped, Path('D:/codex/run'))
        else:
            self.assertEqual(mapped, Path('/mnt/d/codex/run'))
        self.assertEqual(self.a.local('/mnt/data/x'), Path('/mnt/data/x'))


class FidelityTests(Base):
    def test_coverage_per_stage_camera_and_role(self):
        s = self.study(outcomes=pattern(12, 0, 0, 12), steps=32)
        r = self.a.analyze(s.run, s.root)
        f = r['fidelity']
        self.assertTrue(f['descriptive_only']); self.assertFalse(f['conditions_primary_analysis'])
        masks = f['arms']['masks']; point = f['arms']['point']
        grasp = masks['by_operation']['grasp']; place = masks['by_operation']['place']
        self.assertEqual(masks['events'], sum(c['steps'] for c in r['cases'] if c['comparison_arm'] == 'masks'))
        self.assertEqual(grasp['cameras'][CAMS[0]]['drawn_fraction_of_stage_steps'], 1.)
        self.assertEqual(grasp['cameras'][CAMS[2]]['drawn_fraction_of_stage_steps'], 0.)
        self.assertEqual(grasp['cameras'][CAMS[2]]['mask_available_steps'], 0)
        self.assertEqual(set(grasp['cameras'][CAMS[0]]['roles']), {'target'})
        self.assertEqual(set(place['cameras'][CAMS[2]]['roles']), {'destination', 'subject'})
        self.assertEqual(place['cameras'][CAMS[2]]['roles']['subject']['drawn_fraction_when_mask_available'], 1.)
        self.assertEqual(sum(row['cameras'][c]['drawn_steps'] for row in point['by_operation'].values() for c in CAMS), 0)
        self.assertGreater(point['by_operation']['place']['cameras'][CAMS[0]]['mask_available_steps'], 0)
        self.assertTrue(masks['inside_mask_pixels_changed']['all_drawn_events_reported_zero'])
        self.assertEqual(masks['cases_with_mask_events'], 12); self.assertEqual(masks['cases_event_steps_differ_from_observation_steps'], [])
        case = next(c for c in r['cases'] if c['comparison_arm'] == 'masks')
        self.assertTrue(case['mask_fidelity']['event_steps_match_observation_steps'])
        self.assertEqual(case['mask_fidelity']['camera_names'], list(CAMS))

    def test_fidelity_never_conditions_primary_analysis(self):
        outcomes = pattern(12, 3, 2, 4)
        clean = self.study(outcomes=outcomes); reference = self.a.analyze(clean.run, clean.root)
        variants = {}
        for label, kw in (('nonzero_inside', {'inside': 5}), ('no_mask_events', {'mask': False})):
            shutil.rmtree(self.base); self.base.mkdir()
            s = self.study(outcomes=outcomes, **kw); variants[label] = self.a.analyze(s.run, s.root)
        for label, r in variants.items():
            with self.subTest(variant=label):
                strip = lambda rows: [{k: v for k, v in c.items() if k not in ('mask_fidelity', 'result_path', 'result_sha256',
                    'journal_sha256', 'journal_record_count', 'pair_reference')} for c in rows]
                self.assertEqual(strip(r['cases']), strip(reference['cases']))
                self.assertEqual(r['paired'], reference['paired'])
                self.assertEqual([{k: v for k, v in p.items() if k != 'native_chunk_comparison'} for p in r['pairs']],
                                 [{k: v for k, v in p.items() if k != 'native_chunk_comparison'} for p in reference['pairs']])
                self.assertEqual({a: {k: v for k, v in arm.items()} for a, arm in r['arms'].items()}, reference['arms'])
        bad = variants['nonzero_inside']['fidelity']['arms']['masks']
        self.assertEqual(len(bad['cases_with_nonzero_inside_mask_pixels_changed']), 12)
        self.assertEqual(bad['inside_mask_pixels_changed']['max'], 5)
        self.assertFalse(bad['inside_mask_pixels_changed']['all_reported_zero'])
        self.assertEqual(variants['no_mask_events']['fidelity']['arms']['masks']['events'], 0)

    def test_g5_zero_requirement_is_evaluated_for_masks_arm_only(self):
        outcomes = pattern(12, 2, 1, 5)
        for point_field in ('inside_mask_pixels_changed', 'point_cue_pixels_inside_masks'):
            with self.subTest(point_field=point_field):
                shutil.rmtree(self.base); self.base.mkdir()
                s = self.study(outcomes=outcomes, inside={'point': 9}, point_field=point_field)
                r = self.a.analyze(s.run, s.root); f = r['fidelity']
                gate = f['inside_mask_gate']
                self.assertEqual((gate['gate'], gate['arm'], gate['field']), ('G5', 'masks', 'inside_mask_pixels_changed'))
                self.assertTrue(gate['all_reported_zero']); self.assertTrue(gate['all_drawn_events_reported_zero'])
                self.assertEqual(gate['cases_with_nonzero'], []); self.assertIsNone(gate['control_arm_requirement'])
                masks, point = f['arms']['masks'], f['arms']['point']
                self.assertTrue(masks['inside_mask_pixels_changed']['all_reported_zero'])
                self.assertEqual(masks['cases_with_nonzero_inside_mask_pixels_changed'], [])
                self.assertEqual(masks['inside_mask_pixels_changed']['source_fields'],
                                 {'inside_mask_pixels_changed': masks['events']})
                # The point arm carries no zero requirement and no G5 verdict, whichever field name it used.
                self.assertNotIn('inside_mask_pixels_changed', point)
                self.assertNotIn('cases_with_nonzero_inside_mask_pixels_changed', point)
                cue = point['point_cue_pixels_inside_masks']
                self.assertIsNone(cue['requirement']); self.assertNotIn('all_reported_zero', cue)
                self.assertEqual((cue['max'], cue['events_nonzero']), (9, 12))
                self.assertEqual(cue['source_fields'], {point_field: point['events']})
                self.assertEqual(len(point['cases_with_point_cue_pixels_inside_masks']), 12)
                case = next(c for c in r['cases'] if c['comparison_arm'] == 'point')
                self.assertEqual(case['mask_fidelity']['point_cue_pixels_inside_masks_nonzero_steps'], [1])
                self.assertNotIn('inside_mask_pixels_changed_nonzero_steps', case['mask_fidelity'])
                self.assertEqual(r['paired']['eligible_pairs'], 12)

    def test_masks_arm_nonzero_or_renamed_field_fails_g5(self):
        s = self.study(outcomes=pattern(12, 0, 0, 12), inside={'masks': 3})
        gate = self.a.analyze(s.run, s.root)['fidelity']['inside_mask_gate']
        self.assertFalse(gate['all_reported_zero']); self.assertEqual(len(gate['cases_with_nonzero']), 12)
        self.assertEqual(gate['max'], 3)
        # A masks-arm event that carries only the point-arm field is a missing G5 value, never a silent zero.
        e = self.a.mask_event_summary(0, mask_event('masks', 0, 'grasp', field='point_cue_pixels_inside_masks'))
        counts, nonzero = self.a.fidelity_counts([e], 'masks', CAMS)
        inside = self.a.with_fractions(counts)['inside_mask_pixels_changed']
        self.assertEqual((inside['events_missing'], inside['events_reporting'], nonzero), (1, 0, []))
        self.assertFalse(inside['all_reported_zero'])

    def test_point_field_preferred_over_older_name(self):
        event = mask_event('point', 2, 'place', inside=4, field='point_cue_pixels_inside_masks')
        event['cameras'][CAMS[0]]['inside_mask_pixels_changed'] = 0  # stale old-name value is ignored
        counts, nonzero = self.a.fidelity_counts([self.a.mask_event_summary(2, event)], 'point', CAMS)
        cue = counts['point_cue_pixels_inside_masks']
        self.assertEqual((cue['max'], cue['source_fields'], nonzero), (4, {'point_cue_pixels_inside_masks': 1}, [2]))
        self.assertNotIn('inside_mask_pixels_changed', counts)

    def test_schema_variants_read_identically(self):
        results = []
        for layout in ('camera_major', 'role_major', 'list'):
            events = [self.a.mask_event_summary(step, mask_event('masks', step, op, layout=layout))
                      for step, op in enumerate(['grasp'] * 3 + ['place'] * 2)]
            counts, nonzero = self.a.fidelity_counts(events, 'masks', CAMS)
            results.append(counts); self.assertEqual(nonzero, [])
            self.assertEqual(counts['events_without_mask_leaves'], 0)
        self.assertEqual(results[0], results[1]); self.assertEqual(results[0], results[2])
        self.assertEqual(results[0]['by_operation']['grasp']['cameras'][CAMS[1]]['roles']['target']['drawn_steps'], 3)

    def test_labels_scope_and_unrecognised_events_counted(self):
        e1 = self.a.mask_event_summary(0, {**mask_event('point', 0, 'grasp'), 'logging_only': False})
        e2 = self.a.mask_event_summary(1, {'kind': 'mask_presentation', 'arm': 'masks', 'status': 'gated_native',
                                                 'logging_only': True, 'used_by_policy_or_planner': False})
        counts, _ = self.a.fidelity_counts([e1, e2], 'masks', CAMS)
        self.assertEqual((counts['arm_label_mismatch_events'], counts['scope_issue_events'], counts['events_without_mask_leaves']), (1, 1, 1))
        self.assertEqual(counts['by_operation']['no_active_stage']['steps'], 1)
        self.assertEqual(counts['inside_mask_pixels_changed']['events_missing'], 1)
        # Point-labelled event (drawn False) never counts as drawn even with ring_px > 0.
        self.assertEqual(counts['by_operation']['grasp']['cameras'][CAMS[0]]['drawn_steps'], 0)
        invalid = self.a.mask_event_summary(2, {**mask_event('masks', 2, 'grasp'), 'inside_mask_pixels_changed': -1})
        counts, _ = self.a.fidelity_counts([invalid], 'masks', CAMS)
        self.assertEqual(counts['inside_mask_pixels_changed']['invalid_values'], 1)
        self.assertFalse(self.a.with_fractions(counts)['inside_mask_pixels_changed']['all_reported_zero'])


class SeedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = HERE / 'unused-seeds-v1.json'
        cls.before = cls.receipt.stat().st_mtime_ns if cls.receipt.exists() else None
        cls.s = load_module(AUDIT, 'masks_seed_audit_under_test')

    def receipt_state(self):
        return self.receipt.stat().st_mtime_ns if self.receipt.exists() else None

    def test_import_has_no_side_effect(self):
        self.assertEqual(self.receipt_state(), self.before, 'importing the audit must never write its receipt')
        self.assertEqual(self.s.OUT, self.receipt)

    def test_selection_rule_and_counts(self):
        start = self.s.START_SEED
        evidence = defaultdict(list, {start: ['a'], start + 1: ['b'], start + 5: ['c']})
        before = dict(evidence)
        picked = self.s.select(evidence, 12, start)
        self.assertEqual(picked[:4], [start + 2, start + 3, start + 4, start + 6]); self.assertEqual(len(picked), 12)
        self.assertEqual(dict(evidence), before)
        for n in (24, 36): self.assertEqual(len(self.s.select(evidence, n, start)), n)
        with self.assertRaises(ValueError): self.s.select(evidence, 13, start)
        with self.assertRaisesRegex(ValueError, 'overlap'): self.s.select(evidence, 12, start, forbidden={start + 2})

    def test_receipt_bug_fix_does_not_insert_keys(self):
        evidence = defaultdict(list, {5: ['x']})
        self.assertEqual(self.s.receipt_matches(evidence, [5, 6, 7]), {'5': ['x'], '6': [], '7': []})
        self.assertEqual(sorted(evidence), [5])

    def test_fields_and_tokens(self):
        value = {'seed': 3, 'nested': [{'Seeds': [4, True, -1, 2 ** 32]}, {'episode_seed': 5}], 'other': 6}
        self.assertEqual(sorted(f['seed'] for f in self.s.fields(value)), [3, 4, 5])
        self.assertEqual(self.s.TOKEN.findall('x20261099y 120261099 2026109'), ['20261099'])

    def test_scope_filter_excludes_only_masks_namespace(self):
        keep = self.s.keep
        for name in ('tests/test_id60_masks_validation.py', 'originx_harness/id60_masks_validation/runtime.py',
                     'work/id60-masks-validation/manifest.json', 'scripts/run_id60_masks_validation_local.py',
                     'work/other/manifest.json', 'runs/x/worker-0/run/traces/c/records.json', 'runs/x/site.json'):
            self.assertFalse(keep(name), name)
        for name in ('work/id60-regions-validation/manifest.json', 'work/id60-regions-validation/unused-seeds-v1.json',
                     'originx_harness/id60_regions_validation/registration.py', 'tests/test_id60_regions_validation.py',
                     'runs/id60-regions-validation-local-v1/manifest.json', 'runs/x/worker-0/run/results/c.json',
                     'runs/x/worker-0/run/traces/c/trace.json', 'scripts/run_id60_masks_pilot_local.py',
                     'runs/id60-masks-pilot-local-v1/manifest.json'):
            self.assertTrue(keep(name), name)
        windows = lambda text: text.replace('/', chr(92))
        self.assertTrue(keep(windows('work/id60-regions-validation/manifest.json')))
        self.assertTrue(keep(windows('runs/x/worker-0/run/results/c.json')))
        self.assertFalse(keep(windows('tests/test_id60_masks_validation.py')))
        self.assertFalse(keep(windows('runs/x/worker-0/run/traces/c/records.json')))
        command = ' '.join(self.s.rg_command())
        self.assertIn('!**/id60_masks_validation/**', command); self.assertIn('!**/id60-masks-validation/**', command)
        self.assertNotIn('regions', command)

    def test_known_used_seeds_from_regions_receipt(self):
        used = self.s.known_used_seeds()
        self.assertEqual(used['development'], [self.s.START_SEED - 3, self.s.START_SEED - 2, self.s.START_SEED - 1])
        self.assertEqual(used['regions_validation'], list(range(self.s.REGIONS_FIRST_SEED, self.s.REGIONS_FIRST_SEED + 12)))

    def test_help_runs_without_auditing(self):
        out = subprocess.run([sys.executable, '-B', str(AUDIT), '--help'], capture_output=True, text=True, timeout=60, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(out.returncode, 0); self.assertIn('--count', out.stdout)
        self.assertEqual(self.receipt_state(), self.before)


class AnalyzerCliTests(unittest.TestCase):
    def test_help_and_self_test(self):
        out = subprocess.run([sys.executable, '-B', str(ANALYZER), '--help'], capture_output=True, text=True, timeout=60, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(out.returncode, 0); self.assertIn('--resume', out.stdout)
        out = subprocess.run([sys.executable, '-B', str(ANALYZER), '--self-test'], capture_output=True, text=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(out.returncode, 0, out.stderr); self.assertIn('CPU_SELF_TEST_PASS', out.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
