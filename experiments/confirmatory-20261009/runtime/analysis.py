"""Independent, conservative analysis of the sealed confirmation cohort.

Input contract (all records are normalized by the caller, never old raw outcomes):
  manifest: list of {case_id, task_name, policy_id, arm, seed, horizon,
                    episode_id?: str, query_interval?: int (=16)}.
  records: list of matching identities plus
    {success: bool | None, valid: bool, steps: int, terminal_reason: str,
     initial_fingerprint: nonempty dict[str, str],
     trace: [{step: int, state_hash: str, raw_observation_hash: str,
              instruction_hash: str, action_hash: str}],
     assistance: {triggered: bool, delivered: bool, fallback: bool,
                  evidence_valid: bool}}.
  trace_path may replace trace: normalized JSON array or JSONL, relative to
  base_dir when supplied. raw_observation_hash must cover the exact current
  RGB/ordinary-observation contract consistently in every arm. It is not a
  digest of a generated caption or a model response.

valid=True attests a valid native-reset, endpoint-identified terminal outcome;
it is not synonymous with task success. For post-trigger non-C arms,
evidence_valid=True attests the frozen instruction application OR the declared
unchanged-instruction fallback. A delivered Astra response must be linked to
its real request/response/CLI/application evidence by the adapter. Static R/G
instructions must be checked against their frozen text. This module checks the
normalized evidence and its agreement, not cryptographic contents of raw files.

An episode legitimately ending at/before the trigger, without a trigger query,
needs no assistance receipt. The entire early trajectory must match its pair,
including steps and terminal success/failure. C uses exact unchanged instruction
bytes even when it traverses sham bookkeeping. Fallback is a valid system
outcome when the runtime contract and terminal outcome remain valid.

Every policy and every assigned arm must cover the same N cases. Missing records
are unknown; missing manifest assignments or duplicate outcomes are errors.
No retries, winning-attempt selection, raw-outcome adapters, remote access, or
historical-evidence mutation are performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


ARMS = {"C", "R", "G", "L", "V"}
HASH_FIELDS = ("state_hash", "raw_observation_hash", "instruction_hash", "action_hash")
INITIAL = "initial_deviation"
PREFIX = "prefix_deviation"
UNKNOWN = "unknown"
MATCHED = "matched"


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        key = tuple(row[k] for k in ("policy_id", "arm", "case_id"))
    except (KeyError, TypeError) as exc:
        raise ValueError("Each row requires policy_id, arm and case_id") from exc
    if not all(isinstance(x, str) and x for x in key):
        raise ValueError("Identity fields must be nonempty strings")
    return key  # type: ignore[return-value]


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _fingerprint(value: Any) -> bool:
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(k, str) and k and isinstance(v, str) and v
                    for k, v in value.items()))


def _trigger(row: Mapping[str, Any]) -> int:
    interval = row.get("query_interval", 16)
    return interval * math.ceil(row["horizon"] / (2 * interval))


def _load_trace(record: Mapping[str, Any], base_dir: Path | None) -> list[Any]:
    if "trace" in record:
        value = record["trace"]
    elif isinstance(record.get("trace_path"), str) and record["trace_path"]:
        path = Path(record["trace_path"])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        text = path.read_text(encoding="utf-8-sig")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError("missing_trace")
    if not isinstance(value, list):
        raise ValueError("trace_not_array")
    return value


def _prepare(row: Mapping[str, Any], record: Mapping[str, Any] | None,
             base_dir: Path | None) -> dict[str, Any]:
    """Validate one normalized endpoint, without imposing paired equality."""
    out = {"record": record, "usable": False, "reason": "missing_record",
           "known": None, "initial": None, "prefix": [], "trigger": None,
           "early": False, "assistance": {}, "steps": None}
    if record is None:
        return out
    out["assistance"] = record.get("assistance", {})
    if not isinstance(out["assistance"], dict):
        out["assistance"] = {}
    if _fingerprint(record.get("initial_fingerprint")):
        out["initial"] = record["initial_fingerprint"]
    for field in ("episode_id", "task_name", "seed", "horizon", "query_interval"):
        if field in record and field in row and record[field] != row[field]:
            out["reason"] = "record_manifest_mismatch:" + field
            return out
    if record.get("valid") is not True:
        out["reason"] = "invalid_terminal_record"
        return out
    if type(record.get("success")) is not bool:
        out["reason"] = "missing_boolean_outcome"
        return out
    steps = record.get("steps")
    if type(steps) is not int or not 0 <= steps <= row["horizon"]:
        out["reason"] = "invalid_steps"
        return out
    if not isinstance(record.get("terminal_reason"), str) or not record["terminal_reason"].strip():
        out["reason"] = "missing_terminal_reason"
        return out
    if out["initial"] is None:
        out["reason"] = "missing_initial_fingerprint"
        return out
    try:
        trace = _load_trace(record, base_dir)
    except (OSError, ValueError, TypeError) as exc:
        out["reason"] = "unreadable_trace:" + type(exc).__name__
        return out
    last = -1
    for event in trace:
        if (not isinstance(event, dict) or type(event.get("step")) is not int
                or not last < event["step"] < steps
                or not all(isinstance(event.get(k), str) and event[k] for k in HASH_FIELDS)):
            out["reason"] = "invalid_query_trace"
            return out
        last = event["step"]
    interval = row.get("query_interval", 16)
    if [q["step"] for q in trace] != list(range(0, steps, interval)):
        out["reason"] = "incomplete_query_trace"
        return out
    tau = _trigger(row)
    prefix = [{k: q[k] for k in ("step",) + HASH_FIELDS} for q in trace if q["step"] < tau]
    trigger = next((q for q in trace if q["step"] == tau), None)
    early = steps <= tau and trigger is None
    assistance = out["assistance"]
    if early:
        if any(assistance.get(k) is True for k in ("triggered", "delivered", "fallback")):
            out["reason"] = "early_terminal_with_intervention_claim"
            return out
    else:
        if trigger is None:
            out["reason"] = "missing_trigger_observation"
            return out
        if row["arm"] == "C":
            if assistance.get("delivered") is True or assistance.get("fallback") is True:
                out["reason"] = "control_claims_intervention"
                return out
            original_hash = prefix[0]["instruction_hash"] if prefix else None
            if original_hash is None or any(q["instruction_hash"] != original_hash for q in trace):
                out["reason"] = "control_instruction_changed"
                return out
        else:
            if assistance.get("triggered") is not True or assistance.get("evidence_valid") is not True:
                out["reason"] = "unverified_intervention_or_fallback"
                return out
            delivered, fallback = assistance.get("delivered"), assistance.get("fallback")
            if type(delivered) is not bool or type(fallback) is not bool or delivered == fallback:
                out["reason"] = "invalid_delivery_fallback_flags"
                return out
            if fallback and any(q["instruction_hash"] != prefix[0]["instruction_hash"] for q in trace):
                out["reason"] = "fallback_instruction_changed"
                return out
    if len({q["instruction_hash"] for q in prefix}) > 1:
        out["reason"] = "pretrigger_instruction_changed"
        return out
    out.update(usable=True, reason="valid", known=record["success"], steps=steps,
               prefix=prefix, trigger=trigger, early=early)
    return out


def _pair_status(reference: Mapping[str, Any], treatment: Mapping[str, Any]) -> tuple[str, str]:
    # Confirmed initial deviations get their own category even if a later record
    # is also incomplete. No arbitrary arm is selected as a replacement scene.
    if reference["initial"] is not None and treatment["initial"] is not None:
        if reference["initial"] != treatment["initial"]:
            return INITIAL, "initial_fingerprints_differ"
    if not reference["usable"] or not treatment["usable"]:
        return UNKNOWN, "reference=" + reference["reason"] + ";treatment=" + treatment["reason"]
    if reference["prefix"] != treatment["prefix"]:
        return PREFIX, "pretrigger_queries_differ"
    if reference["early"] or treatment["early"]:
        if not (reference["early"] and treatment["early"]
                and reference["steps"] == treatment["steps"]
                and reference["known"] == treatment["known"]):
            return PREFIX, "early_terminal_trajectories_differ"
        return MATCHED, "matched_early_terminal"
    for field in ("state_hash", "raw_observation_hash"):
        if reference["trigger"][field] != treatment["trigger"][field]:
            return PREFIX, "trigger_" + field + "_differs"
    return MATCHED, "matched_posttrigger"


def _percentile(values: Sequence[float], q: float) -> float:
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _bootstrap(task_differences: Mapping[str, list[int]], samples: int,
               seed: int) -> dict[str, Any]:
    blocks = [(sum(ds), len(ds)) for _, ds in sorted(task_differences.items()) if ds]
    result = {"estimand": "conditional_valid_pair_net_effect",
              "method": "task_block_percentile_bootstrap", "confidence": 0.95,
              "seed": seed, "samples": samples, "task_count": len(blocks),
              "pair_count": sum(n for _, n in blocks), "interval": None,
              "generalization_assumption": "task blocks are exchangeable; tasks and within-task batches are not independent episodes",
              "confirmatory_superiority_test": False}
    if len(blocks) < 2:
        result["unavailable_reason"] = "fewer_than_two_tasks_with_valid_pairs"
        return result
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        draws.append(sum(s for s, _ in chosen) / sum(n for _, n in chosen))
    draws.sort()
    result["interval"] = [_percentile(draws, .025), _percentile(draws, .975)]
    return result


def _mcnemar(rescues: int, regressions: int) -> dict[str, Any]:
    n = rescues + regressions
    if n == 0:
        p = 1.0
    else:
        # Integer summation avoids cancellation and remains stable for n=2500.
        lower_tail = sum(math.comb(n, k) for k in range(min(rescues, regressions) + 1))
        p = min(1.0, (2 * lower_tail) / (1 << n))
    return {"label": "unadjusted_McNemar_independent_pair_sensitivity_only",
            "discordant_pairs": n, "two_sided_exact_p": p,
            "confirmatory_superiority_test": False,
            "limitation": "assumes independent discordant pairs; task/batch dependence is not corrected; no multiplicity-adjusted claim"}


def _assistance_counts(prepared: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {k: sum(p["assistance"].get(k) is True for p in prepared)
              for k in ("triggered", "delivered", "fallback", "evidence_valid")}
    result["valid_early_terminal"] = sum(p["usable"] and p["early"] for p in prepared)
    result["valid_delivered"] = sum(p["usable"] and p["assistance"].get("delivered") is True
                                    for p in prepared)
    result["valid_fallback"] = sum(p["usable"] and p["assistance"].get("fallback") is True
                                   for p in prepared)
    return result


def analyze(manifest: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
            N: int = 2500, *, bootstrap_samples: int = 2000,
            bootstrap_seed: int = 20261009, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Return JSON-serializable full-denominator results, never selecting retries.

    Every comparison reports exhaustive matched/unknown/initial/prefix counts.
    Known one-sided valid endpoints tighten bounds. When paired initial/prefix
    evidence deviates, the canonical C endpoint remains known if usable; other
    endpoints are retained only if independently matched to usable C. With no C
    anchor, both endpoints of a confirmed deviation are conservatively unknown.
    Missing evidence in one arm does not erase usable evidence from the other.
    All task-bootstrap CIs are explicitly conditional on valid paired outcomes.
    """
    if not _positive_int(N) or not _positive_int(bootstrap_samples) or type(bootstrap_seed) is not int:
        raise ValueError("N and bootstrap_samples must be positive integers; bootstrap_seed must be an integer")
    if not manifest:
        raise ValueError("Empty manifest")
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    env: dict[str, tuple[Any, ...]] = {}
    episode_ids: set[str] = set()
    by_policy: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in manifest:
        key = _key(row)
        if key in rows:
            raise ValueError("Duplicate manifest identity: " + repr(key))
        policy, arm, case = key
        if arm not in ARMS or not isinstance(row.get("task_name"), str) or not row["task_name"]:
            raise ValueError("Unknown arm or missing task_name")
        if type(row.get("seed")) is not int or not _positive_int(row.get("horizon")):
            raise ValueError("Manifest seed/horizon must be integers and horizon positive")
        if not _positive_int(row.get("query_interval", 16)):
            raise ValueError("query_interval must be a positive integer")
        identity = (row["task_name"], row["seed"], row["horizon"], row.get("query_interval", 16))
        if case in env and env[case] != identity:
            raise ValueError("Conflicting environment specification for " + case)
        env[case] = identity
        if "episode_id" in row:
            eid = row["episode_id"]
            if not isinstance(eid, str) or not eid or eid in episode_ids:
                raise ValueError("episode_id must be nonempty and unique")
            episode_ids.add(eid)
        rows[key] = row
        by_policy[policy][arm].add(case)
    if len(env) != N:
        raise ValueError(f"Manifest has {len(env)} unique cases, expected N={N}")
    for policy, arms in by_policy.items():
        if "C" not in arms or arms["C"] != set(env):
            raise ValueError("Each policy requires C on every manifest case: " + policy)
        if any(cases != set(env) for cases in arms.values()):
            raise ValueError("Every assigned arm requires the complete cohort: " + policy)
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = _key(record)
        if key not in rows:
            raise ValueError("Outcome outside sealed manifest: " + repr(key))
        if key in indexed:
            raise ValueError("Duplicate outcome; choose/record primary attempt in the adapter, not analysis: " + repr(key))
        indexed[key] = record
    root = Path(base_dir) if base_dir is not None else None
    prepared = {key: _prepare(row, indexed.get(key), root) for key, row in rows.items()}
    report: dict[str, Any] = {
        "schema": "originx_confirmation_analysis_v1", "N_per_policy": N,
        "manifest_case_count": len(env), "policy_count": len(by_policy),
        "planned_arm_outcomes": len(rows), "present_arm_records": len(indexed),
        "primary_inference": "effects_and_bounds; conditional_task_block_CI; no_claim_of_statistical_superiority",
        "historical_scores_modified": False, "policies": {}}
    case_order = sorted(env)
    for policy, arm_cases in sorted(by_policy.items()):
        arms: dict[str, Any] = {}
        for arm in sorted(arm_cases):
            ps = [prepared[(policy, arm, case)] for case in case_order]
            valid = [p for p in ps if p["usable"]]
            successes = sum(p["known"] is True for p in valid)
            missing = N - len(valid)
            arms[arm] = {"planned": N, "present": sum(p["record"] is not None for p in ps),
                         "individually_valid_terminal_outcomes": len(valid),
                         "successes": successes, "failures": len(valid) - successes,
                         "unknown": missing,
                         "success_rate_if_individually_complete": successes / N if not missing else None,
                         "success_rate_bounds": [successes / N, (successes + missing) / N],
                         "scope": "individual recorded arm outcomes; pairwise comparability is checked separately",
                         "assistance": _assistance_counts(ps),
                         "invalid_reasons": dict(sorted(Counter(p["reason"] for p in ps if not p["usable"]).items()))}
        comparisons: dict[str, Any] = {}
        pairs = [(arm, "C") for arm in sorted(arm_cases) if arm != "C"]
        if "V" in arm_cases and "L" in arm_cases:
            pairs.append(("V", "L"))
        for treatment_arm, reference_arm in pairs:
            ledger: list[dict[str, Any]] = []
            categories = {MATCHED: 0, UNKNOWN: 0, INITIAL: 0, PREFIX: 0}
            table = {"both_success": 0, "treatment_only_success": 0,
                     "reference_only_success": 0, "both_failure": 0}
            task_differences: dict[str, list[int]] = defaultdict(list)
            low_sum = high_sum = 0
            paired_treatments = []
            paired_references = []
            for case in case_order:
                ref = prepared[(policy, reference_arm, case)]
                trt = prepared[(policy, treatment_arm, case)]
                control = prepared[(policy, "C", case)]
                category, reason = _pair_status(ref, trt)
                # For V-L, both arms must agree with the canonical initial C
                # when that fingerprint exists, even if they agree with each other.
                if control["initial"] is not None and category != INITIAL:
                    for item in (ref, trt):
                        if item["initial"] is not None and item["initial"] != control["initial"]:
                            category, reason = INITIAL, "initial_differs_from_canonical_control"
                            break
                categories[category] += 1
                kr, kt = ref["known"], trt["known"]
                if category in (INITIAL, PREFIX):
                    def anchored(item: Mapping[str, Any], arm: str) -> bool | None:
                        if arm == "C":
                            return item["known"]
                        if control["usable"] and _pair_status(control, item)[0] == MATCHED:
                            return item["known"]
                        return None
                    kr, kt = anchored(ref, reference_arm), anchored(trt, treatment_arm)
                lo = (int(kt) if kt is not None else 0) - (int(kr) if kr is not None else 1)
                hi = (int(kt) if kt is not None else 1) - (int(kr) if kr is not None else 0)
                low_sum += lo
                high_sum += hi
                difference = None
                if category == MATCHED:
                    difference = int(trt["known"]) - int(ref["known"])
                    cell = ("both_success" if trt["known"] and ref["known"] else
                            "treatment_only_success" if trt["known"] else
                            "reference_only_success" if ref["known"] else "both_failure")
                    table[cell] += 1
                    task_differences[env[case][0]].append(difference)
                    paired_treatments.append(trt)
                    paired_references.append(ref)
                ledger.append({"case_id": case, "task_name": env[case][0],
                               "category": category, "reason": reason,
                               "reference_usable_outcome": kr, "treatment_usable_outcome": kt,
                               "paired_difference": difference, "difference_bounds": [lo, hi],
                               "reference_early_terminal": ref["early"] if ref["usable"] else None,
                               "treatment_early_terminal": trt["early"] if trt["usable"] else None})
            matched = categories[MATCHED]
            rescued = table["treatment_only_success"]
            regressed = table["reference_only_success"]
            d = rescued - regressed
            name = treatment_arm + "_vs_" + reference_arm
            key_seed = int.from_bytes(hashlib.sha256((str(bootstrap_seed) + "|" + policy + "|" + name).encode()).digest()[:8], "big")
            comparisons[name] = {
                "treatment": treatment_arm, "reference": reference_arm, "N": N,
                "categories": categories, "valid_paired_2x2": table,
                "rescues": rescued, "regressions": regressed, "valid_pair_net_count": d,
                "unresolved_comparisons": N - matched,
                "complete_cohort_net_effect": d / N if matched == N else None,
                "full_cohort_net_effect_bounds": {"low": low_sum / N, "high": high_sum / N,
                                                  "kind": "identification_bounds_not_confidence_interval"},
                "conditional_valid_pair_net_effect": d / matched if matched else None,
                "conditional_task_bootstrap": _bootstrap(task_differences, bootstrap_samples, key_seed),
                "mcnemar_sensitivity": _mcnemar(rescued, regressed),
                "matched_assistance": {"treatment": _assistance_counts(paired_treatments),
                                       "reference": _assistance_counts(paired_references)},
                "task_effects": [{"task_name": task, "valid_pairs": len(ds), "net_count": sum(ds),
                                  "conditional_net_effect": sum(ds) / len(ds)}
                                 for task, ds in sorted(task_differences.items())],
                "case_ledger": ledger}
        report["policies"][policy] = {"arms": arms, "comparisons": comparisons}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    parser.add_argument("--base-dir", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.manifest.read_text(encoding="utf-8-sig")),
                     json.loads(args.records.read_text(encoding="utf-8-sig")), N=args.n,
                     bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
                     base_dir=args.base_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
