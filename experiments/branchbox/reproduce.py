#!/usr/bin/env python3
"""Reproduce the focused paired task-initial-state bootstrap from episodes.csv.

Usage: python reproduce.py
Dependency: NumPy. The input and output are next to this script.

All 720 rows are validated before selecting the four focused tasks. Each task's
20 initial-state indices are resampled once per bootstrap replicate; the same
indices are used for both models and all three arms. Tasks and models have equal
weights. These percentile intervals condition on the recorded checkpoints and
tasks. They do not account for training-seed variation or post hoc selection of
the task/model subset.

The output reproduces counts, rates, paired effect estimates, percentile 95%
intervals, and discordant-pair counts for pooled, model, and task summaries.
Intervals/effects are rounded to 12 decimals to remove floating-point noise;
paper formatting rounds exact-count percentages to one decimal, half up.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np


MODELS = ("smolvla", "xvla")
ARMS = ("N0", "P0", "PS")
ALL_TASKS = tuple(range(6))
FOCUSED_TASKS = (0, 3, 4, 5)
INITIAL_STATES = tuple(range(20, 40))
CONTRASTS = (("PS", "N0"), ("P0", "N0"), ("PS", "P0"))
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20261101
COLUMNS = ("model", "task_id", "initial_state", "arm", "success")


def load_episodes(path: Path) -> dict[tuple[str, int, int, str], int]:
    """Require exactly one valid binary result for each specified condition."""
    observations: dict[tuple[str, int, int, str], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or (
            len(reader.fieldnames) != len(COLUMNS)
            or set(reader.fieldnames) != set(COLUMNS)
        ):
            raise ValueError(f"CSV columns must be exactly {COLUMNS}.")

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row at line {line_number}.")
            model = row["model"].strip()
            arm = row["arm"].strip()
            try:
                task_id = int(row["task_id"])
                initial_state = int(row["initial_state"])
            except ValueError as exc:
                raise ValueError(
                    f"Task and initial-state IDs must be integers (line {line_number})."
                ) from exc
            if model not in MODELS or arm not in ARMS:
                raise ValueError(f"Unknown model or arm at line {line_number}.")
            if task_id not in ALL_TASKS or initial_state not in INITIAL_STATES:
                raise ValueError(f"Out-of-range task or state at line {line_number}.")
            success_text = row["success"].strip().lower()
            if success_text not in ("0", "1", "false", "true"):
                raise ValueError(f"Success must be binary at line {line_number}.")
            success = int(success_text in ("1", "true"))
            key = (model, task_id, initial_state, arm)
            if key in observations:
                raise ValueError(f"Duplicate condition at line {line_number}: {key}.")
            observations[key] = success

    expected = set(itertools.product(MODELS, ALL_TASKS, INITIAL_STATES, ARMS))
    if set(observations) != expected:
        missing = sorted(expected - set(observations))
        raise ValueError(
            f"Expected all {len(expected)} conditions; found {len(observations)}. "
            f"Missing examples: {missing[:5]}"
        )
    return observations


def reproduce(observations: dict[tuple[str, int, int, str], int]) -> dict:
    """Compute all summaries using a single shared set of bootstrap draws."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cell_values = {}
    cell_replicates = {}

    # Ordering is part of reproducibility: tasks 0,3,4,5; states 20,...,39.
    # No independent draw is made for a second model or another deployment arm.
    for task_id in FOCUSED_TASKS:
        indices = rng.integers(
            0, len(INITIAL_STATES),
            size=(BOOTSTRAP_REPLICATES, len(INITIAL_STATES)),
        )
        for arm in ARMS:
            values = np.array([
                [observations[model, task_id, state, arm] for state in INITIAL_STATES]
                for model in MODELS
            ])
            for model_index, model in enumerate(MODELS):
                key = (model, task_id, arm)
                cell_values[key] = values[model_index]
                cell_replicates[key] = values[model_index, indices].mean(axis=1)

    def summarize(models: tuple[str, ...], tasks: tuple[int, ...]) -> dict:
        cells = [(model, task_id) for model in models for task_id in tasks]
        values = {
            arm: np.concatenate([cell_values[model, task_id, arm] for model, task_id in cells])
            for arm in ARMS
        }
        replicates = {
            arm: np.mean(
                [cell_replicates[model, task_id, arm] for model, task_id in cells],
                axis=0,
            )
            for arm in ARMS
        }
        result = {
            "n_per_arm": len(values["N0"]),
            "unique_task_initial_states": len(tasks) * len(INITIAL_STATES),
            "counts": {arm: int(values[arm].sum()) for arm in ARMS},
            "rates_percent": {arm: float(100 * values[arm].mean()) for arm in ARMS},
            "contrasts": {},
        }
        for first, second in CONTRASTS:
            paired_difference = values[first] - values[second]
            interval = 100 * np.quantile(
                replicates[first] - replicates[second], [0.025, 0.975]
            )
            result["contrasts"][f"{first}-{second}"] = {
                "effect_pp": float(np.round(100 * paired_difference.mean(), 12)),
                "ci95_pp": np.round(interval, 12).tolist(),
                "discordant_wins": int((paired_difference > 0).sum()),
                "discordant_losses": int((paired_difference < 0).sum()),
            }
        return result

    return {
        "pooled": summarize(MODELS, FOCUSED_TASKS),
        "by_model": {model: summarize((model,), FOCUSED_TASKS) for model in MODELS},
        "by_task": {str(task): summarize(MODELS, (task,)) for task in FOCUSED_TASKS},
    }


def main() -> None:
    directory = Path(__file__).resolve().parent
    observations = load_episodes(directory / "episodes.csv")
    output = directory / "reproduced_statistics.json"
    output.write_text(json.dumps(reproduce(observations), indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(observations)} episode rows; wrote {output.name}.")


if __name__ == "__main__":
    main()
