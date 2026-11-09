"""Target definition for every RoboCasa365 stage description in the five seen tasks.

Returns a dict with:
  prompt   - open-vocabulary detector class text (Grounding DINO)
  hint     - natural-language description of the exact instance for the VLM selector
  entity   - short stable entity name used in target_entity_id
  role     - manipulated_object | destination | interaction_part | navigation_target | watch_target
or None when the stage has no spatial target (task complete).
"""
import re


def _spec(prompt, hint, entity, role):
    return {'prompt': prompt, 'hint': hint, 'entity': entity, 'role': role}


def target_spec(task, description, kind):
    d = description.lower().strip()
    if kind == 'done' or d == 'task complete':
        return None
    if kind == 'navigate':
        m = re.match(r'navigate (?:back )?to (?:the |a |an |either )?(.+)$', d)
        obj = m[1] if m else d
        if 'cup' in obj:
            return _spec('glass cup', 'the glass cup on the dining counter (navigation target)', 'glass_cup', 'navigation_target')
        if 'plate' in obj:
            return _spec('plate', 'the plate on the dining counter (navigation target)', 'plate', 'navigation_target')
        if 'fridge' in obj:
            return _spec('refrigerator', 'the fridge (navigation target)', 'fridge', 'navigation_target')
        if 'tupperware' in obj:
            return _spec('plastic food container', obj + ' (navigation target)', 'tupperware', 'navigation_target')
        if 'stove' in obj:
            return _spec('stove', 'the stove top with burners (navigation target)', 'stove', 'navigation_target')
        if 'counter' in obj:
            return _spec('kitchen counter', 'the kitchen counter the robot is heading to (navigation target)', 'counter', 'navigation_target')
        return _spec(obj, obj + ' (navigation target)', obj.replace(' ', '_'), 'navigation_target')
    if kind == 'pick':
        m = re.match(r'pick up (?:the |a |an )?(.+?) from (?:the )?(.+)$', d)
        obj, src = (m[1], m[2]) if m else (d, '')
        prompt = {'pan': 'frying pan', 'bread': 'slice of bread', 'spatula': 'spatula'}.get(obj, obj)
        return _spec(prompt, 'the ' + obj + ' that is about to be picked up from the ' + src, obj.replace(' ', '_'), 'manipulated_object')
    if kind == 'place':
        m = re.match(r'place (?:the |a |an )?(.+?) (?:in|on|into|onto) (?:the |a |an |either )?(.+)$', d)
        obj, dest = (m[1], m[2]) if m else (d, d)
        if 'tupperware' in dest:
            return _spec('plastic food container', 'the destination container: ' + dest + ' (where the ' + obj + ' must be placed)', 'tupperware', 'destination')
        if 'burner' in dest:
            return _spec('stove burner', 'the destination: ' + dest + ' of the stove (where the pan must be placed)', dest.replace(' ', '_').replace('-', '_'), 'destination')
        if 'pan' in dest:
            return _spec('frying pan', 'the frying pan (destination where the ' + obj + ' must be placed)', 'pan', 'destination')
        if 'pot' in dest:
            return _spec('cooking pot', 'the cooking pot (destination where the ' + obj + ' must be placed)', 'pot', 'destination')
        if 'plate' in dest:
            return _spec('plate', 'the plate (destination where the ' + obj + ' must be placed)', 'plate', 'destination')
        if 'cup' in dest:
            return _spec('glass cup', 'the glass cup (destination where the ' + obj + ' must be placed)', 'glass_cup', 'destination')
        return _spec(dest, 'the destination: ' + dest, dest.replace(' ', '_'), 'destination')
    if kind == 'press' or 'lever' in d:
        return _spec('toaster lever', 'the lever of the toaster', 'toaster_lever', 'interaction_part')
    if kind == 'wait':
        return _spec('toaster lever', 'the lever of the toaster being waited on', 'toaster_lever', 'watch_target')
    if kind == 'execute':
        if 'open the drawer' in d:
            return _spec('drawer handle', 'the handle of the drawer in front of the robot', 'drawer_handle', 'interaction_part')
        m = re.match(r'turn on the (.+?) burner', d)
        if m:
            return _spec('stove knob', 'the knob that controls the ' + m[1] + ' burner of the stove', 'knob_' + m[1].replace('-', '_'), 'interaction_part')
        if 'stir' in d:
            return _spec('cooking pot', 'the cooking pot with vegetables being stirred', 'pot', 'interaction_part')
    return _spec(d, d, d.replace(' ', '_'), 'unresolved')


if __name__ == '__main__':
    import json, sys, collections
    seen = collections.OrderedDict()
    for split in ['train', 'dev']:
        for line in open(sys.argv[1] + '/' + split + '.jsonl'):
            r = json.loads(line)
            key = (r['task'], r['description'], r['kind'])
            if key not in seen:
                seen[key] = target_spec(*key)
    for k, v in seen.items():
        print(k, '->', None if v is None else (v['prompt'], v['entity'], v['role']))
    print(len(seen), 'distinct stage descriptions;', sum(v is None for v in seen.values()), 'without target;',
          sum(v is not None and v['role'] == 'unresolved' for v in seen.values()), 'unresolved')
