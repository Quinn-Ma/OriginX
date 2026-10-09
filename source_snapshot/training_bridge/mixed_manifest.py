"""Build/verify a matched-budget Human300 plus corrective experiment manifest.

No model load, GPU initialization or training is performed. An empty or
unverified corrective set is an error. The candidate starts from the same
released base as the human control, never from the control's updated weights.
"""

import argparse
import copy
import json
from pathlib import Path

from .corrective_reader import open_corrective_reader
from .lerobot_reader import LeRobotPretrainReader
from .mixture import SCHEDULE, CorrectivePool, ThreeToOneProvider
from .provenance import provenance_from_acquisition, sha256_file
from .train_single import SamplePool, _json_hash, checkpoint_identity, read_manifest


RECIPE_KEYS = ("base_model", "seed", "microbatch", "gradient_accumulation", "max_grad_norm",
               "planned_optimizer_steps", "optimizer")


def _human_control(path):
    human = read_manifest(path)
    if human["scope"] != "human300" or human["microbatch"] != 1 or human["gradient_accumulation"] != 8:
        raise ValueError("Require the full Human300 control with microbatch one and accumulation eight")
    if human["planned_optimizer_steps"] != 200:
        raise ValueError("This matched-budget interface requires the declared 200-update control")
    return human


def build_manifest(human_manifest, corrective_directories, *, human_run_identity):
    human_manifest = Path(human_manifest).resolve()
    human = _human_control(human_manifest)
    paths = [Path(directory).resolve() for directory in corrective_directories]
    if not paths:
        raise ValueError("Zero verified corrective branches: no mixed training manifest can be built")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate corrective branch directories")
    readers = [open_corrective_reader(directory) for directory in paths]
    registry_sha256 = sha256_file(human["registry"])
    human_tasks = {entry["task"] for entry in human["datasets"]}
    if any(reader.inventory()["registry_sha256"] != registry_sha256 or reader.task not in human_tasks for reader in readers):
        raise ValueError("Corrective source registry/task differs from the Human300 control")
    human_run_identity = Path(human_run_identity).resolve()
    control = json.loads(human_run_identity.read_text())
    if control.get("manifest") != human or control.get("identity", {}).get("manifest_sha256") != _json_hash(human):
        raise ValueError("Human control run identity belongs to a different manifest")
    base_identity = checkpoint_identity(human["base_model"])
    if control.get("identity", {}).get("base_checkpoint") != base_identity:
        raise ValueError("Released base weights/assets differ from the Human-only control")
    return {
        "format_version": 1, "purpose": "pretrain_human_corrective_training",
        "scope": "human300_plus_verified_pretrain_corrections",
        "human_manifest": str(human_manifest), "human_manifest_sha256": sha256_file(human_manifest),
        "human_run_identity": str(human_run_identity), "human_run_identity_sha256": sha256_file(human_run_identity),
        "base_checkpoint": base_identity,
        "initialization": "same_released_base_as_human_control",
        "recipe": {key: copy.deepcopy(human[key]) for key in RECIPE_KEYS},
        "mixture": {"schedule": list(SCHEDULE), "human_to_corrective": [3, 1],
                    "human_sampler": human["sampler"],
                    "corrective_sampler": "task_uniform_recording_uniform_eligible_action_step",
                    "human_microbatches_per_update": 6, "corrective_microbatches_per_update": 2},
        "corrective": [reader.inventory() for reader in readers],
        "data_disclosure": "Additional success-filtered simulation data, not original human demonstrations; live intervention data is self-imitation, not counterfactual recovery evidence",
        "method_status": "Pilot interface; no claim of leaderboard submission eligibility or measured gain",
    }


def verify_manifest(path):
    """Re-run the source gate and compare the entire declared experiment."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("format_version") != 1 or manifest.get("purpose") != "pretrain_human_corrective_training":
        raise ValueError("Unsupported mixed training manifest")
    human_path = manifest.get("human_manifest")
    if sha256_file(human_path) != manifest.get("human_manifest_sha256"):
        raise ValueError("Human control manifest changed")
    if sha256_file(manifest.get("human_run_identity")) != manifest.get("human_run_identity_sha256"):
        raise ValueError("Human control run identity changed")
    expected = build_manifest(human_path, [entry["root"] for entry in manifest.get("corrective", [])],
                              human_run_identity=manifest["human_run_identity"])
    if manifest != expected:
        raise ValueError("Mixed manifest differs from the verified sources or matched-budget recipe")
    return manifest


def build_sample_provider(manifest_path, processor, device):
    """Assemble the verified 3:1 provider; does not load a model or train.

    The caller must separately use the declared released base initialization,
    pin the training code/runtime and apply the manifest's matched recipe.
    Human metadata are checked against the actual control run, not just the
    current dataset files at the same paths.
    """
    manifest = verify_manifest(manifest_path)
    human = _human_control(manifest["human_manifest"])
    readers = []
    for entry in human["datasets"]:
        if entry.get("role") != "train":
            raise ValueError("Human dataset must be explicitly assigned role=train")
        provenance = provenance_from_acquisition(entry["root"], entry["acquisition"], human["registry"], human["box_links"])
        if entry["task"] != provenance["task"] or entry.get("archive_sha256") != provenance["archive_sha256"]:
            raise ValueError("Human source identity changed")
        readers.append(LeRobotPretrainReader(entry["root"], provenance, human["registry"], human["box_links"],
                                            episode_ids=entry["episode_ids"]))
    control = json.loads(Path(manifest["human_run_identity"]).read_text())
    if [reader.inventory() for reader in readers] != control["identity"]["data"]:
        raise ValueError("Human data metadata differ from the actual Human-only control run")
    corrective = [open_corrective_reader(entry["root"]) for entry in manifest["corrective"]]
    if [reader.inventory() for reader in corrective] != manifest["corrective"]:
        raise ValueError("Corrective recording changed while constructing the provider")
    human_pool = SamplePool(readers, processor, device, human["seed"])
    corrective_pool = CorrectivePool(corrective, processor, device, (human["seed"] + 1) % 2**32)
    return ThreeToOneProvider(human_pool, corrective_pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--human-run-identity", type=Path, required=True)
    parser.add_argument("--corrective-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = build_manifest(args.human_manifest, args.corrective_dir, human_run_identity=args.human_run_identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(manifest['corrective'])} successful simulation recordings; wrote manifest only: {args.output}")


if __name__ == "__main__":
    main()
