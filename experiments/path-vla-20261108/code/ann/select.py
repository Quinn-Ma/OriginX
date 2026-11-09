"""Qwen3-VL-4B-Instruct selects, per camera, the Grounding DINO/SAM2 candidate that is the described target instance.

Set-of-Mark style: candidate boxes are drawn with numbers on a 2x upscaled copy of each camera image. The model answers
with one candidate index per camera, or -1 when the target is not visible / no candidate is correct. Offline labeling only.
"""
from pathlib import Path
import argparse, json, os, re, time
ROOT = Path('/dev/shm/qinzhen/sbl')
QWEN = Path('/dev/shm/qinzhen/lerobot-three-model-adapters-20261106/weights/Qwen3-VL-4B-Instruct')
COLORS = [(255, 60, 60), (60, 220, 90), (70, 140, 255), (255, 190, 40), (210, 70, 240), (40, 220, 220)]
SYSTEM = ('You verify object annotations for OFFLINE robot demonstration data. You are not a robot controller. '
          'Each image is a camera view with numbered colored boxes drawn around candidate objects proposed by a detector. '
          'For EACH camera, pick the single candidate number whose box bounds the TARGET described below, or -1 if the target '
          'is not visible in that camera or none of the candidates is the target. Choose the specific instance the description '
          'refers to (e.g. the empty container, the front-left burner, the knob for a given burner). Boxes around robot arm parts, '
          'fingers, or wrong objects must be rejected. Return JSON only, like {"views":[{"camera_index":0,"choice":2},{"camera_index":1,"choice":-1}]}.')


def draw(im, cands):
    from PIL import ImageDraw, ImageFont
    big = im.resize((512, 512), resample=3); d = ImageDraw.Draw(big)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    for c in cands:
        col = COLORS[c['idx'] % len(COLORS)]; x1, y1, x2, y2 = [v * 512 for v in c['box_xyxy']]
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        tx, ty = x1 + 2, max(0, y1 - 24)
        d.rectangle([tx, ty, tx + 26, ty + 24], fill=col); d.text((tx + 5, ty), str(c['idx']), fill='black', font=font)
    return big


def parse(answer, cams):
    s, e = answer.find('{'), answer.rfind('}')
    if s < 0 or e < 0:
        raise ValueError('no json')
    obj = json.loads(answer[s:e + 1]); choice = {}
    for v in obj['views']:
        ci = int(v['camera_index']); ch = int(v['choice'])
        valid = {c['idx'] for c in cams[ci]['candidates']}
        choice[ci] = ch if ch in valid else -1
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--shard', type=int, required=True); ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--candidates', default=str(ROOT / 'ann/candidates')); ap.add_argument('--frames', default=str(ROOT / 'frames'))
    ap.add_argument('--out', default=str(ROOT / 'ann/selected')); ap.add_argument('--batch', type=int, default=6)
    a = ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                      TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'), HF_HOME=str(ROOT / 'cache/hf'), TOKENIZERS_PARALLELISM='false')
    (ROOT / 'ann/selected').mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(2)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(QWEN, local_files_only=True); processor.tokenizer.padding_side = 'left'
    records = []
    for p in sorted(Path(a.candidates).glob('shard*.jsonl')):
        for line in open(p):
            records.append(json.loads(line))
    records.sort(key=lambda r: (r['task'], r['episode'], r['anchor']))
    mine = records[a.shard::a.nshards]
    out_path = Path(a.out) / ('shard%02d.jsonl' % a.shard); done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line); done.add((r['task'], r['episode'], r['anchor']))
    todo = [r for r in mine if (r['task'], r['episode'], r['anchor']) not in done]
    print('shard', a.shard, 'windows', len(mine), 'todo', len(todo), flush=True)
    t0 = time.time(); n = 0
    with torch.inference_mode(), out_path.open('a') as out:
        for i in range(0, len(todo), a.batch):
            batch = todo[i:i + a.batch]; messages = []; metas = []
            for r in batch:
                cams_with = [c for c in r['cams'] if c['candidates']]
                if not cams_with:
                    rec = {**{k: r[k] for k in ['task', 'episode', 'anchor', 'stage', 'split', 'description', 'kind', 'spec']},
                           'choice': {c['camera_index']: -1 for c in r['cams']}, 'selector': 'none:no_candidates', 'raw': ''}
                    out.write(json.dumps(rec) + '\n'); n += 1
                    continue
                content = []
                for c in cams_with:
                    im = Image.open(Path(a.frames) / r['task'] / ('e%03d' % r['episode']) / ('f%05d-%s.png' % (r['anchor'], c['camera']))).convert('RGB')
                    content.append({'type': 'text', 'text': 'Camera %d (%s):' % (c['camera_index'], c['camera'].replace('robot0_', ''))})
                    content.append({'type': 'image', 'image': draw(im, c['candidates'])})
                content.append({'type': 'text', 'text': json.dumps({'task': r.get('instruction', ''), 'current_subtask': r['description'],
                                                                   'target_to_annotate': r['spec']['hint'], 'detector_class': r['spec']['prompt'],
                                                                   'cameras_shown': [c['camera_index'] for c in cams_with]})})
                messages.append([{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}])
                metas.append((r, cams_with))
            if not messages:
                continue
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors='pt', padding=True).to('cuda')
            gen = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            answers = processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for (r, cams_with), ans in zip(metas, answers):
                try:
                    choice = parse(ans, r['cams']); err = None
                except Exception as exc:
                    choice = {}; err = str(exc)
                full = {c['camera_index']: choice.get(c['camera_index'], -1) for c in r['cams']}
                rec = {**{k: r[k] for k in ['task', 'episode', 'anchor', 'stage', 'split', 'description', 'kind', 'spec']},
                       'choice': full, 'selector': 'Qwen3-VL-4B-Instruct set-of-mark selection', 'raw': ans, 'parse_error': err}
                out.write(json.dumps(rec) + '\n'); n += 1
            if (i // a.batch) % 20 == 0:
                out.flush(); print('shard', a.shard, 'done', n, '/', len(todo), 'elapsed', round(time.time() - t0), flush=True)
    print('SELECT_DONE shard', a.shard, n, round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
