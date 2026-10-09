"""Materialize the frozen OriginX base and sidecars without running model code.

Model assets come from Hugging Face; the three pinned Python model files come
from this GitHub checkout. No training, deserialization, or GPU access occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO_ID = "Qinzhen3/OriginX"
BASE_ID = "ccaa65176cf4294b7c0c618e1cc3be7b65617359630c35d3a18dd75c406fc184"
SIDECARS = {
    "adapter-originx-2000.pt": {
        "size_bytes": 14207135,
        "sha256": "5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4",
    },
    "branch-00002000.pt": {
        "size_bytes": 16873301,
        "sha256": "41988b92391953687b39a0bc5a35bce962fbfcd08fbf6e550108430850d9944f",
    },
}
CHUNK = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def aggregate(hashes):
    # Match the exact original training identity algorithm (default JSON spaces).
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def load_manifest(repo_root):
    data = json.loads((repo_root / "base-model-assets.json").read_text(encoding="utf-8"))
    require(data["base_identity"] == BASE_ID, "Unexpected frozen base identity")
    assets = data["assets"]
    require(len(assets) == 18, "Expected exactly 18 original base identity assets")
    names = set()
    for item in assets:
        name = item["name"]
        require(isinstance(name, str) and name not in {"", ".", ".."}
                and not any(c in name for c in "/\\:\0")
                and Path(name).name == name, "Unsafe base asset name")
        require(name not in names, "Duplicate base asset name")
        names.add(name)
        require(item["source"] in {"huggingface", "github"}, "Unknown asset source")
        require(type(item["size_bytes"]) is int and item["size_bytes"] > 0,
                "Invalid asset size")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                "Invalid asset hash")
        require(item.get("included_in_base_identity") is True,
                "Only original base identity assets may be materialized")
        require((item["source"] == "github") == name.endswith(".py"),
                "Python model code must come from the GitHub checkout")
    require(sum(x["source"] == "github" for x in assets) == 3,
            "Expected exactly three original Python model files")
    require(aggregate({x["name"]: x["sha256"] for x in assets}) == BASE_ID,
            "Manifest does not reconstruct the frozen base identity")
    return assets


def is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def verify(path, item):
    require(path.is_file(), "Missing regular file: " + str(path))
    require(path.stat().st_size == item["size_bytes"], "Size mismatch: " + str(path))
    value = digest(path)
    require(value == item["sha256"], "SHA-256 mismatch: " + str(path))
    return value


def check_destination(directory, expected):
    require(not is_link(directory), "Destination must not be a symlink or junction: " + str(directory))
    if not directory.exists():
        return
    require(directory.is_dir(), "Destination is not a directory: " + str(directory))
    for path in directory.iterdir():
        require(path.name in expected and not is_link(path),
                "Unexpected destination entry; refusing to change it: " + str(path))
        verify(path, expected[path.name])


def copy_verified(source, destination, item):
    """Reuse an identical file; exclusive creation never overwrites a collision."""
    require(not is_link(destination), "Refusing destination link: " + str(destination))
    if destination.exists():
        return verify(destination, item), "reused"
    try:
        output = destination.open("xb")
    except FileExistsError:
        require(not is_link(destination), "Destination changed to a link")
        return verify(destination, item), "reused"
    identity = None
    try:
        with output:
            import os
            identity = os.fstat(output.fileno())
            value = hashlib.sha256()
            count = 0
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    output.write(block)
                    value.update(block)
                    count += len(block)
            require(count == item["size_bytes"] and value.hexdigest() == item["sha256"],
                    "Source changed during copy: " + str(source))
        return value.hexdigest(), "copied"
    except BaseException:
        # Remove only this operation's incomplete file, never an existing file.
        if identity is not None and destination.exists() and not is_link(destination):
            now = destination.stat()
            if (now.st_dev, now.st_ino) == (identity.st_dev, identity.st_ino):
                destination.unlink()
        raise


def materialize(snapshot, output_dir, repo_root, assets):
    snapshot = Path(snapshot).resolve(strict=True)
    require(snapshot.is_dir(), "Snapshot must be a directory")
    output_dir = Path(output_dir).expanduser().absolute()
    require(not is_link(output_dir), "Output directory must not be a symlink or junction")
    base = output_dir / "base"
    weights = output_dir / "weights"
    expected = {x["name"]: x for x in assets}
    check_destination(base, expected)
    check_destination(weights, SIDECARS)
    jobs = []
    for item in assets:
        origin = repo_root / "upstream_model_code" if item["source"] == "github" else snapshot
        jobs.append((origin / item["name"], base / item["name"], item))
    jobs += [(snapshot / name, weights / name, item) for name, item in SIDECARS.items()]
    # Preflight every source before creating any output base/weight files.
    for source, _, item in jobs:
        verify(source, item)
    base.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    hashes = {}
    counts = {"copied": 0, "reused": 0}
    for source, destination, item in jobs:
        value, mode = copy_verified(source, destination, item)
        counts[mode] += 1
        if destination.parent == base:
            hashes[destination.name] = value
    require({p.name for p in base.iterdir()} == set(expected), "Base directory gained unexpected files")
    require({p.name for p in weights.iterdir()} == set(SIDECARS), "Weight directory gained unexpected files")
    require(aggregate(hashes) == BASE_ID, "Materialized base identity mismatch")
    return {"base_model": str(base.resolve()), "weights": str(weights.resolve()),
            "base_identity": BASE_ID, "base_files": 18, "sidecar_files": 2,
            "copied": counts["copied"], "reused": counts["reused"],
            "training_performed": False, "gpu_validation_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--snapshot", type=Path, help="Existing OriginX HF snapshot directory (offline)")
    source.add_argument("--revision", help="OriginX HF commit or revision; defaults to main")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"),
                        help="Write base/ and weights/ here; default: assets")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    assets = load_manifest(repo_root)
    for item in assets:
        if item["source"] == "github":
            verify(repo_root / "upstream_model_code" / item["name"], item)
    snapshot = args.snapshot
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub or supply --snapshot for offline use") from exc
        allow = [x["name"] for x in assets if x["source"] == "huggingface"] + list(SIDECARS)
        snapshot = snapshot_download(repo_id=REPO_ID, revision=args.revision or "main",
                                     allow_patterns=allow,
                                     cache_dir=str(args.output_dir / ".hf-cache"))
    result = materialize(snapshot, args.output_dir, repo_root, assets)
    result["source"] = "existing_snapshot" if args.snapshot else REPO_ID
    result["requested_revision"] = None if args.snapshot else args.revision or "main"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("OriginX preparation failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
