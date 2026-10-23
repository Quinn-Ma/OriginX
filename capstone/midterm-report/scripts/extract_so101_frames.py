"""Build the SO101 demonstration figure from a local LeRobot v3 dataset.

Usage: python extract_so101_frames.py <dataset_root> <episode_index> <out.jpg>
Picks four front-camera frames at fixed fractions of one episode and one
wrist-camera frame, then writes a single-row montage.
"""
import glob
import os
import sys

import av
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FRACTIONS = (0.05, 0.40, 0.65, 0.95)
LABELS = ("start", "grasp", "transport", "cube in cup")


def episode_row(root, episode):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    eps = pd.concat(pd.read_parquet(f) for f in files)
    return eps[eps["episode_index"] == episode].iloc[0]


def frame_at(root, row, key, frac):
    chunk = int(row[f"videos/{key}/chunk_index"])
    fidx = int(row[f"videos/{key}/file_index"])
    t0 = float(row[f"videos/{key}/from_timestamp"])
    t1 = float(row[f"videos/{key}/to_timestamp"])
    target = t0 + frac * (t1 - t0)
    path = os.path.join(root, "videos", key, f"chunk-{chunk:03d}", f"file-{fidx:03d}.mp4")
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(max(target - 1.0, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time >= target:
                return frame.to_image()
    raise RuntimeError(f"no frame at {target:.2f}s in {path}")


def main():
    root, episode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    row = episode_row(root, episode)
    tiles = [(frame_at(root, row, "observation.images.front", f), f"front: {name}") for f, name in zip(FRACTIONS, LABELS)]
    tiles.append((frame_at(root, row, "observation.images.wrist", FRACTIONS[1]), f"wrist: {LABELS[1]}"))
    w, h = 320, 240
    canvas = Image.new("RGB", (w * len(tiles) + 4 * (len(tiles) - 1), h), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(tiles):
        x = i * (w + 4)
        canvas.paste(img.resize((w, h)), (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([x, h - 26, x + w, h], fill=(0, 0, 0))
        draw.text((x + 6, h - 24), label, fill="white", font=font)
    canvas.save(out, quality=90)
    print(f"episode {episode}: length {int(row['length'])} frames -> {out}")


if __name__ == "__main__":
    main()
