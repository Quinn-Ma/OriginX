#!/usr/bin/env python3
"""Build an explicit Human300 or pilot manifest; never launch training."""

import argparse
import ast
import json
import random
from pathlib import Path

try:
    from .provenance import _literal, provenance_from_acquisition
except ImportError:
    from provenance import _literal, provenance_from_acquisition


def canonical_human100(episode_ids):
    """Match GR00T's seed-0 membership, preserving metadata iteration order.

    Its get_subset_demos_filter_key shuffles with seed 0, chooses the first 100,
    and its reader then filters the original episodes.jsonl order by membership.
    This is declared GR00T-compatible selection, not a claim about Xiaomi's data.
    """
    original = list(episode_ids)
    if len(set(original)) != len(original):
        raise ValueError("Duplicate episode IDs")
    if len(original) <= 100:
        return original
    shuffled = original.copy()
    random.Random(0).shuffle(shuffled)
    selected = set(shuffled[:100])
    return [index for index in original if index in selected]


def human300_tasks(registry):
    for node in ast.parse(Path(registry).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PRETRAINING_TASKS" for target in node.targets):
            tasks = _literal(node.value)["pretrain300"]
            if len(tasks) != 300 or len(set(tasks)) != 300:
                raise ValueError("Registry does not define exactly 300 unique pretraining tasks")
            return tasks
    raise ValueError("Human300 registry list not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Contains <Task>/lerobot and acquisition records")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--box-links", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--human300", action="store_true")
    selection.add_argument("--tasks", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=193)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    tasks = human300_tasks(args.registry) if args.human300 else args.tasks
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate tasks")
    datasets = []
    for task in tasks:
        root = args.data_root / task / "lerobot"
        acquisition = args.data_root / task / "download_complete.json"
        identity = provenance_from_acquisition(root, acquisition, args.registry, args.box_links)
        episodes = [json.loads(line)["episode_index"] for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
        datasets.append({"task": task, "role": "train", "root": str(root.resolve()),
                         "acquisition": str(acquisition.resolve()), "episode_ids": canonical_human100(episodes),
                         "archive_sha256": identity["archive_sha256"]})
    manifest = {
        "format_version": 1, "purpose": "pretrain_human_training",
        "scope": "human300" if args.human300 else "explicit_pilot_tasks",
        "base_model": str(args.base_model.resolve()), "registry": str(args.registry.resolve()),
        "box_links": str(args.box_links.resolve()), "seed": args.seed,
        "sampler": "task_uniform_episode_uniform_frame",
        "episode_selection": "groot_seed0_up_to100; original metadata order after filtering",
        "microbatch": 1, "gradient_accumulation": 8, "max_grad_norm": 1.0,
        "planned_optimizer_steps": 200,
        "optimizer": {"type": "AdamW", "lr": 1e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0},
        "recipe_status": "New declared control defaults; not asserted as original Xiaomi training settings",
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(datasets)} explicit tasks, {sum(len(x['episode_ids']) for x in datasets)} episodes to {args.output}")


if __name__ == "__main__":
    main()
