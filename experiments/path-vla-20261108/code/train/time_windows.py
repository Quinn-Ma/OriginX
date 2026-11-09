"""CPU timing of the demonstration window loader per model variant (no GPU, no training)."""
import sys, time, json, random
sys.path.insert(0, '/dev/shm/qinzhen/lerobot-training-ready-20261106/scripts')
from training_data import DemonstrationWindows

def main():
    out = {}
    for variant in ['xr1', 'awomo_i', 'awomo_v']:
        ds = DemonstrationWindows('SearingMeat', 'train'); rng = random.Random(1)
        times = []
        for _ in range(6):
            ep = rng.choice(ds.episodes); n = len(ds.rows[ep]); anchor = rng.randrange(0, (n - 40) // 16) * 16
            t = time.time(); w = ds.window(variant, ep, anchor); times.append(time.time() - t)
        out[variant] = {'seconds_per_window': [round(x, 2) for x in times], 'mean': round(sum(times) / len(times), 2)}
        print(variant, out[variant], flush=True)
    print(json.dumps(out))

if __name__ == '__main__':
    main()
