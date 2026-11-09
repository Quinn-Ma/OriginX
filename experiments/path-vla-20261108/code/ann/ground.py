"""Grounding DINO -> SAM2.1 candidate proposals for every queued window (GPU, inference only)."""
from pathlib import Path
import argparse, json, os, sys, time, hashlib
ROOT = Path('/dev/shm/qinzhen/sbl')
WEIGHTS = Path('/dev/shm/qinzhen/branchbox-sam-labels/weights')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import target_spec
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--shard', type=int, required=True); ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--queue', default='/dev/shm/qinzhen/branchbox-sam-labels/queue'); ap.add_argument('--splits', nargs='+', default=['train', 'dev'])
    ap.add_argument('--frames', default=str(ROOT / 'frames')); ap.add_argument('--out', default=str(ROOT / 'ann/candidates'))
    ap.add_argument('--box-threshold', type=float, default=0.2); ap.add_argument('--text-threshold', type=float, default=0.2)
    ap.add_argument('--max-candidates', type=int, default=6)
    a = ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                      TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'), HF_HOME=str(ROOT / 'cache/hf'), TOKENIZERS_PARALLELISM='false')
    for d in ['tmp', 'cache', 'ann/candidates']:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from torchvision.ops import nms
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
    torch.set_num_threads(2); torch.manual_seed(20261108)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(WEIGHTS / 'grounder', local_files_only=True)
    detector.config.disable_custom_kernels = True; detector = detector.cuda().eval().requires_grad_(False)
    dp = AutoProcessor.from_pretrained(WEIGHTS / 'grounder', local_files_only=True)
    sam = Sam2Model.from_pretrained(WEIGHTS / 'sam2', local_files_only=True).cuda().eval().requires_grad_(False)
    sp = Sam2Processor.from_pretrained(WEIGHTS / 'sam2', local_files_only=True)
    # windows
    windows = []
    for split in a.splits:
        for line in open(Path(a.queue) / (split + '.jsonl')):
            r = json.loads(line)
            if r['kind'] == 'done':
                continue
            windows.append(r)
    windows.sort(key=lambda r: (r['task'], r['episode'], r['anchor']))
    mine = windows[a.shard::a.nshards]
    out_path = Path(a.out) / ('shard%02d.jsonl' % a.shard)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line); done.add((r['task'], r['episode'], r['anchor']))
    print('shard', a.shard, 'windows', len(mine), 'already', len(done), flush=True)
    t0 = time.time(); n = 0
    with torch.inference_mode(), out_path.open('a') as out:
        for r in mine:
            key = (r['task'], r['episode'], r['anchor'])
            if key in done:
                continue
            spec = target_spec(r['task'], r['description'], r['kind'])
            if spec is None:
                continue
            paths = [Path(a.frames) / r['task'] / ('e%03d' % r['episode']) / ('f%05d-%s.png' % (r['anchor'], c)) for c in CAMERAS]
            images = [Image.open(p).convert('RGB') for p in paths]
            text = spec['prompt'].strip().rstrip('.') + '.'
            inputs = dp(images=images, text=[text] * 3, return_tensors='pt', padding=True).to('cuda')
            det = detector(**inputs)
            found = dp.post_process_grounded_object_detection(det, inputs.input_ids, threshold=a.box_threshold,
                                                              text_threshold=a.text_threshold, target_sizes=[im.size[::-1] for im in images])
            cams = []
            for ci, (im, f) in enumerate(zip(images, found)):
                scores, boxes = f['scores'], f['boxes']
                cands = []
                if len(boxes):
                    keep = nms(boxes, scores, 0.65)[:a.max_candidates]
                    boxes = boxes[keep].clamp(0, 255); scores = scores[keep]
                    valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                    boxes = boxes[valid]; scores = scores[valid]
                if len(boxes):
                    batch = sp(images=im, input_boxes=[boxes.tolist()], return_tensors='pt').to('cuda')
                    pred = sam(**batch, multimask_output=False)
                    masks = sp.post_process_masks(pred.pred_masks, batch['original_sizes'])[0]
                    for j, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
                        mask = mask.squeeze().bool(); ys, xs = torch.where(mask); area = int(mask.sum())
                        if area < 4:
                            continue
                        tight = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        if tight[2] <= tight[0] or tight[3] <= tight[1]:
                            continue
                        norm = [v / 255 for v in tight]
                        cands.append({'idx': j, 'det_score': round(float(score), 4), 'sam_iou': round(float(pred.iou_scores[0, j, 0]), 4),
                                      'det_box_xyxy': [round(v / 255, 4) for v in box.tolist()], 'box_xyxy': [round(v, 4) for v in norm],
                                      'location_xy': [round((norm[0] + norm[2]) / 2, 4), round((norm[1] + norm[3]) / 2, 4)], 'area_px': area})
                cams.append({'camera_index': ci, 'camera': CAMERAS[ci], 'candidates': cands})
            rec = {'task': r['task'], 'episode': r['episode'], 'anchor': r['anchor'], 'stage': r['stage'], 'split': r['split'],
                   'description': r['description'], 'kind': r['kind'], 'spec': spec, 'prompt_text': text, 'cams': cams,
                   'source_parquet_sha256': r['source_parquet_sha256']}
            out.write(json.dumps(rec) + '\n'); n += 1
            if n % 100 == 0:
                out.flush(); print('shard', a.shard, 'done', n, '/', len(mine), 'elapsed', round(time.time() - t0), flush=True)
    print('GROUND_DONE shard', a.shard, n, 'windows', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
