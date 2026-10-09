#!/usr/bin/env python3
"""Inspect explicit pretrain episodes, optionally decoding one sample; no GPU."""

import argparse
import json
from pathlib import Path

from lerobot_reader import LeRobotPretrainReader
from provenance import provenance_from_acquisition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--box-links", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--decode-timestep", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    provenance = provenance_from_acquisition(args.dataset, args.acquisition, args.registry, args.box_links)
    reader = LeRobotPretrainReader(args.dataset, provenance, args.registry, args.box_links,
                                  episode_ids=args.episodes)
    report = reader.inventory()
    report["episodes"] = {}
    for episode_id in args.episodes:
        episode = reader.load_episode(episode_id)
        data = {"frames": len(episode.states), "instruction": episode.instructions[0],
                "parquet_sha256": episode.parquet_sha256,
                "state_shape": list(episode.states.shape), "action_shape": list(episode.actions.shape),
                "timestamps": [float(episode.timestamps[0]), float(episode.timestamps[-1])]}
        if args.decode_timestep is not None:
            sample = reader.sample(episode_id, args.decode_timestep)
            data["decoded_sample"] = {"timestep": args.decode_timestep,
                                      "state_shape": list(sample.state.shape),
                                      "action_shape": list(sample.action.shape),
                                      "valid_future_steps": int(sample.valid_steps.sum())}
        report["episodes"][episode_id] = data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.output}")


if __name__ == "__main__":
    main()
