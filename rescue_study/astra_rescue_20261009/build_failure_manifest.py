"""Derive the fixed failure subset; read original evidence without extracting it."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "outputs/RoboCasa_逐回合评测凭据.tar.gz"
AUDIT = ROOT / "work/robocasa365-model/evidence/final-audit.json"
LOCAL_MANIFEST = ROOT / "work/robocasa365-model/evidence/manifest.json"
LOCAL_REPORT = ROOT / "work/robocasa365-model/evidence/evaluation-report.json"
PREFIX = "results/native-reset-b2500-20261009-v1/"
REMOTE_ROOT = "/ephemeral/qinzhen/robocasa-xr1-20261003/"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def main():
    originals = {p: sha(p.read_bytes()) for p in (ARCHIVE, AUDIT, LOCAL_MANIFEST, LOCAL_REPORT)}
    audit = json.loads(AUDIT.read_bytes())
    assert originals[ARCHIVE] == audit["archive_sha256"]
    report = manifest = config = None
    jobs = {}
    results = {}
    scenes = {}
    guards = {}
    evidence_bytes = {}
    archive_extensions = Counter()
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            archive_extensions[Path(member.name).suffix] += 1
            name = member.name
            if name in {PREFIX + "manifest.json", PREFIX + "config.json", PREFIX + "analysis/report.json"}:
                raw = tar.extractfile(member).read()
                evidence_bytes[name] = raw
                obj = json.loads(raw)
                if name.endswith("manifest.json"):
                    manifest = obj
                    jobs = {j["id"]: j for j in manifest["jobs"]}
                elif name.endswith("config.json"):
                    config = obj
                else:
                    report = obj
            elif name.startswith(PREFIX + "episodes/"):
                relative = name[len(PREFIX + "episodes/"):]
                job_id, suffix = relative.split("/", 1)
                if suffix not in {"result.json", "initial_scene.json", "readback-guard/summary.json"}:
                    continue
                raw = tar.extractfile(member).read()
                obj = json.loads(raw)
                if suffix == "result.json":
                    assert job_id not in results
                    assert sha(raw) == report["result_sha256"][REMOTE_ROOT + name]
                    assert obj["job"] == jobs[job_id]
                    assert obj["status"] == "completed" and type(obj["success"]) is bool
                    assert obj["config_sha256"] == report["config_sha256"]
                    assert obj["manifest_sha256"] == report["manifest_sha256"]
                    assert obj["native_reset_before"] == obj["native_reset_after"]
                    assert obj["native_reset_before"]["native_reset"] is True
                    assert obj["native_reset_before"]["reset_patch_applied"] is False
                    episode = obj["stats"]["episodes"][0]
                    assert len(obj["stats"]["episodes"]) == 1
                    assert episode["success"] == obj["success"]
                    assert episode["seed"] == jobs[job_id]["env_seed"]
                    assert obj["success"] or episode["steps"] == jobs[job_id]["horizon"]
                    results[job_id] = {
                        "success": obj["success"],
                        "stats": obj["stats"],
                        "initial_scene": obj["initial_scene"],
                        "native_reset_before": obj["native_reset_before"],
                        "local_rng_before": obj["local_rng"]["before"],
                        "guard_report_sha256": obj["guard_report_sha256"],
                        "result_sha256": sha(raw),
                        "result_archive_member": name,
                    }
                elif suffix == "initial_scene.json":
                    assert job_id not in scenes
                    scenes[job_id] = {"value": obj, "sha256": sha(raw), "archive_member": name}
                else:
                    assert job_id not in guards
                    guards[job_id] = {"value": obj, "sha256": sha(raw), "archive_member": name}

    assert manifest and report and config
    assert report["complete"] is True and report["smoke"] is False
    assert len(jobs) == len(manifest["jobs"]) == len(results) == len(scenes) == len(guards) == 2500
    assert set(jobs) == set(results) == set(scenes) == set(guards)
    assert sha(evidence_bytes[PREFIX + "manifest.json"]) == report["manifest_sha256"] == originals[LOCAL_MANIFEST]
    assert sha(evidence_bytes[PREFIX + "config.json"]) == report["config_sha256"]
    report_sha256 = sha(evidence_bytes[PREFIX + "analysis/report.json"])
    assert report_sha256 == audit["audit"]["remote_report_file_sha256"]
    assert report == json.loads(LOCAL_REPORT.read_bytes())
    task_counts, task_wins = Counter(), Counter()
    for job_id, job in jobs.items():
        result = results[job_id]
        assert result["initial_scene"] == scenes[job_id]["value"]
        assert result["guard_report_sha256"] == guards[job_id]["sha256"]
        task_counts[job["task"]] += 1
        task_wins[job["task"]] += result["success"]
    assert len(task_counts) == 50 and set(task_counts.values()) == {50}
    assert sum(task_wins.values()) == report["successes"] == 1496
    for name in task_counts:
        assert task_wins[name] == report["tasks"][name]["successes"]

    selected = [j for j in manifest["jobs"] if results[j["id"]]["success"] is False]
    selected_ids = [j["id"] for j in selected]
    assert len(selected) == len(set(selected_ids)) == 1004
    assert not any(results[job_id]["success"] for job_id in selected_ids)
    task_by_name = {t["task"]: t for t in manifest["tasks"]}
    failures_per_task = Counter(j["task"] for j in selected)
    failures_per_stratum = Counter(task_by_name[j["task"]]["stratum"] for j in selected)
    records = []
    for job in selected:
        job_id = job["id"]
        result = results[job_id]
        scene = scenes[job_id]
        guard = guards[job_id]
        records.append({
            "id": job_id,
            "stratum": task_by_name[job["task"]]["stratum"],
            "original_job": job,
            "original_instruction": scene["value"]["instruction"],
            "original_initial_scene": scene["value"],
            "original_stats": result["stats"],
            "original_success": False,
            "original_native_reset": result["native_reset_before"],
            "original_local_rng_before": result["local_rng_before"],
            "original_readback_guard": guard["value"],
            "source": {
                "result_archive_member": result["result_archive_member"],
                "result_sha256": result["result_sha256"],
                "initial_scene_archive_member": scene["archive_member"],
                "initial_scene_sha256": scene["sha256"],
                "readback_guard_archive_member": guard["archive_member"],
                "readback_guard_sha256": guard["sha256"],
                "report_hash_verified": True,
                "manifest_job_verified": True,
                "initial_scene_matches_result": True,
                "guard_hash_verified": True,
            },
        })

    model_hello = config["models"][0]["hello"]
    for endpoint in config["models"]:
        assert endpoint["hello"]["stage_serving_identity"] == model_hello["stage_serving_identity"]
        assert endpoint["hello"]["model_assets_sha256"] == model_hello["model_assets_sha256"]
    output = {
        "schema": "originx_astra_fixed_failure_subset_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_subset_prepared_no_retests_in_this_artifact",
        "selection": "Exactly status=completed and success=false in the frozen original 2500-episode evaluation; preserve original manifest order.",
        "no_seed_replacements": True,
        "no_original_result_mutation": True,
        "split": manifest["split"],
        "original_episodes": 2500,
        "original_successes_excluded": 1496,
        "selected_failures": 1004,
        "selected_task_count": len(failures_per_task),
        "original_task_count": 50,
        "failure_counts_by_stratum": dict(sorted(failures_per_stratum.items())),
        "failure_counts_by_task": dict(sorted(failures_per_task.items())),
        "ordered_failure_ids_canonical_sha256": canonical_sha(selected_ids),
        "canonical_hash_format": "UTF-8 JSON, ensure_ascii=false, sorted object keys, separators comma/colon, no trailing newline",
        "provenance": {
            "archive_path": ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_sha256": originals[ARCHIVE],
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_file_extensions": dict(archive_extensions),
            "frozen_manifest_archive_member": PREFIX + "manifest.json",
            "frozen_manifest_sha256": report["manifest_sha256"],
            "frozen_config_archive_member": PREFIX + "config.json",
            "frozen_config_sha256": report["config_sha256"],
            "frozen_report_archive_member": PREFIX + "analysis/report.json",
            "frozen_report_sha256": report_sha256,
            "local_report_copy_sha256": originals[LOCAL_REPORT],
            "local_report_copy_semantically_identical": True,
            "all_original_result_hashes_verified": 2500,
            "all_original_guard_hashes_verified": 2500,
            "all_initial_scene_sidecars_match_result": 2500,
        },
        "frozen_protocol": config["protocol"],
        "frozen_packages": config["packages"],
        "frozen_model_identity": {
            "model_config_sha256": model_hello["model_config_sha256"],
            "model_assets_sha256": model_hello["model_assets_sha256"],
            "stage_serving_identity": model_hello["stage_serving_identity"],
        },
        "frozen_runtime_source_sha256": config["source_sha256"],
        "tasks": [t for t in manifest["tasks"] if t["task"] in failures_per_task],
        "jobs": selected,
        "failure_records": records,
        "evidence_limits": [
            "Archive entries are JSON; no original RGB/video/action trajectory or restorable simulator state bytes are present.",
            "Initial RGB, proprioception and physics state are one-way fingerprints; regenerate from original native reset and compare before intervention.",
            "Readback guard failure=null does not establish absence of graphics anomalies; preserve gl_error_counts.",
            "A fixed historical-failure subset estimates conditional rescue performance; it cannot measure regression on 1496 original successes or establish a new 2500-episode benchmark score.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "failure_manifest.json"
    destination.write_bytes(json_bytes(output))
    assert all(sha(p.read_bytes()) == digest for p, digest in originals.items())
    receipt = {
        "schema": "originx_astra_failure_manifest_verification_v1",
        "passed": True,
        "failure_manifest_sha256": sha(destination.read_bytes()),
        "selected_failures": len(selected),
        "selected_task_count": len(failures_per_task),
        "excluded_original_successes": 1496,
        "failures_per_stratum": dict(sorted(failures_per_stratum.items())),
        "verified_original_result_hashes": 2500,
        "verified_original_guard_hashes": 2500,
        "original_source_hashes_unchanged": True,
        "retests_executed_by_manifest_builder": 0,
    }
    (OUT / "manifest_verification.json").write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
