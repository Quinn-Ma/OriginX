"""Explicit pretrain-human provenance; no simulator imports or network access."""

import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

METADATA_FILES = (
    "meta/info.json", "meta/modality.json", "meta/episodes.jsonl",
    "meta/tasks.jsonl", "extras/dataset_meta.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_file(root, relative_path):
    """Metadata paths may not escape the chosen dataset, even through symlinks."""
    root = Path(root).resolve()
    relative_path = str(relative_path)
    if Path(relative_path).is_absolute():
        raise ValueError(f"Dataset metadata contains an absolute path: {relative_path}")
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {relative_path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _literal(node):
    """Only literal containers and dict/OrderedDict keyword constructors."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in {"dict", "OrderedDict"} or node.args or any(k.arg is None for k in node.keywords):
            raise ValueError("Unsupported registry expression")
        return {key.arg: _literal(key.value) for key in node.keywords}
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(value) for value in node.elts]
    return ast.literal_eval(node)


def registry_entry(registry_path, box_links_path, task):
    """Read the official allowlist without importing/initializing RoboCasa."""
    desired = {"ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS", "PRETRAINING_TASKS", "TARGET_TASKS"}
    values = {}
    for node in ast.parse(Path(registry_path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in desired:
                    values[target.id] = _literal(node.value)
    if values.keys() != desired:
        raise ValueError("Could not resolve the expected literal RoboCasa task registry")
    if task not in values["PRETRAINING_TASKS"]["pretrain300"]:
        raise ValueError(f"Task is not in Human300 pretraining: {task}")
    if task in values["TARGET_TASKS"]["composite_unseen"]:
        raise ValueError(f"Refusing composite-unseen training data: {task}")
    all_tasks = {**values["ATOMIC_TASK_DATASETS"], **values["COMPOSITE_TASK_DATASETS"]}
    human_path = all_tasks[task]["pretrain"]["human_path"]
    parts = PurePosixPath(human_path).parts
    if len(parts) != 5 or parts[:2] != ("v1.0", "pretrain") or parts[3] != task:
        raise ValueError(f"Unexpected registry human_path: {human_path}")
    archive_key = str(PurePosixPath(*parts[1:]) / "lerobot.tar")
    links = json.loads(Path(box_links_path).read_text())
    if archive_key not in links:
        raise ValueError(f"Official archive mapping missing: {archive_key}")
    targets = set(sum(values["TARGET_TASKS"].values(), []))
    return {"task": task, "split": "pretrain", "source": "human",
            "human_path": human_path, "archive_key": archive_key,
            "archive_shared_url": links[archive_key], "outside_target50": task not in targets}


def make_provenance(dataset_root, task, registry_path, box_links_path, *,
                    archive_path=None, archive_sha256=None, archive_bytes=None):
    """Record extraction identity after acquisition; caller saves returned JSON.

    If an archive is still present, hash it here. If acquisition already removed
    it, require the hash and byte count recorded by acquisition and label that
    fact honestly; metadata inspection cannot prove what bytes were downloaded.
    """
    root = Path(dataset_root).resolve()
    expected = registry_entry(registry_path, box_links_path, task)
    if archive_path is not None:
        archive_sha256 = sha256_file(archive_path)
        archive_bytes = Path(archive_path).stat().st_size
        hash_source = "computed_from_archive"
    else:
        hash_source = "provided_by_acquisition"
    if not isinstance(archive_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise ValueError("A recorded source archive SHA256 is required")
    if not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise ValueError("A positive source archive byte count is required")
    metadata = {name: sha256_file(dataset_file(root, name)) for name in METADATA_FILES}
    environment = json.loads(dataset_file(root, "extras/dataset_meta.json").read_text())
    if environment["env_args"]["env_name"] != task:
        raise ValueError("Dataset extras environment name disagrees with the pretrain task")
    return {"format_version": 1, "dataset_root": str(root), **expected,
            "archive_sha256": archive_sha256, "archive_bytes": archive_bytes,
            "archive_hash_source": hash_source,
            "registry_sha256": sha256_file(registry_path),
            "box_links_sha256": sha256_file(box_links_path),
            "metadata_sha256": metadata}


def verify_provenance(dataset_root, manifest, registry_path, box_links_path):
    """Verify manifest identity and metadata before reading policy data."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported provenance manifest version")
    if (manifest.get("split"), manifest.get("source")) != ("pretrain", "human"):
        raise ValueError("Only explicitly identified pretrain human datasets are allowed")
    root = Path(dataset_root).resolve()
    if str(root) != manifest.get("dataset_root"):
        raise ValueError("Provenance belongs to a different dataset root")
    expected = registry_entry(registry_path, box_links_path, manifest["task"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Provenance differs from official registry: {key}")
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("Registry has changed since provenance was recorded")
    if manifest.get("box_links_sha256") != sha256_file(box_links_path):
        raise ValueError("Archive URL mapping has changed since provenance was recorded")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("archive_sha256", ""))):
        raise ValueError("Missing archive SHA256 provenance")
    if not isinstance(manifest.get("archive_bytes"), int) or manifest["archive_bytes"] <= 0:
        raise ValueError("Missing archive size provenance")
    for name in METADATA_FILES:
        if manifest.get("metadata_sha256", {}).get(name) != sha256_file(dataset_file(root, name)):
            raise ValueError(f"Dataset metadata changed: {name}")
    environment = json.loads(dataset_file(root, "extras/dataset_meta.json").read_text())
    if environment["env_args"]["env_name"] != manifest["task"]:
        raise ValueError("Dataset environment identity disagrees with provenance")
    return manifest


def provenance_from_acquisition(dataset_root, acquisition_path, registry_path, box_links_path):
    """Accept the downloader's recorded identity only if it matches the registry."""
    record = json.loads(Path(acquisition_path).read_text())
    if (record.get("split"), record.get("source")) != ("pretrain", "human"):
        raise ValueError("Acquisition record is not pretrain human data")
    expected = registry_entry(registry_path, box_links_path, record["task"])
    if record.get("key") != expected["archive_key"] or record.get("box_shared_url") != expected["archive_shared_url"]:
        raise ValueError("Acquisition archive identity differs from the official pretrain registry")
    manifest = make_provenance(
        dataset_root, record["task"], registry_path, box_links_path,
        archive_sha256=record["archive_sha256"], archive_bytes=record["archive_bytes"])
    manifest["acquisition_record_sha256"] = sha256_file(acquisition_path)
    return manifest
