"""Independent pre-label validator for the PC2 P1 prediction closure."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from ccad.nip_synthetic import FAMILIES

import run_m1_nip_parent_completion_p1 as runner


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _scientific_prediction(row: dict) -> dict:
    value = json.loads(json.dumps(row, sort_keys=True))
    value["cost"].pop("runtime_seconds", None)
    value["cost"].pop("median_runtime_seconds", None)
    return value


def _json_canonical(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _forbidden_seed_reads(tree: ast.AST) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name) or node.value.id != "seeds":
            continue
        if isinstance(node.slice, ast.Constant) and node.slice.value in {"evaluation", "intervention"}:
            found.add(str(node.slice.value))
    return found


def validate(run_dir: Path) -> dict:
    checks: dict[str, bool] = {}
    closure = json.loads((run_dir / "prediction_closure.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    code_hashes = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
    seed_ledger = json.loads((run_dir / "seed_ledger.json").read_text(encoding="utf-8"))
    checks["sealed_pass_truth_closed"] = closure["state"] == "SEALED" and status["status"] == "PASS" and not closure["truth_opened"] and not status["truth_opened"]
    checks["closure_file_hashes"] = all((run_dir / name).is_file() and sha(run_dir / name) == value for name, value in closure["files"].items())
    checks["source_snapshot_hashes"] = all(
        (run_dir / item["snapshot"]).is_file()
        and sha(run_dir / item["snapshot"]) == item["sha256"]
        and sha(ROOT / item["path"]) == item["sha256"]
        for item in code_hashes["files"]
    )
    checks["code_aggregate"] = runner.digest(code_hashes["files"]) == code_hashes["aggregate_sha256"] == closure["code_snapshot_hash"]
    source_text = "\n".join((run_dir / item["snapshot"]).read_text(encoding="utf-8") for item in code_hashes["files"] if item["path"].endswith(".py"))
    tree = ast.parse(source_text)
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    checks["no_truth_import"] = "ccad.nip_truth" not in imported
    checks["evaluation_intervention_unused"] = not _forbidden_seed_reads(tree)
    checks["config_and_parent_binding"] = sha(ROOT / config["parent_config_path"]) == config["parent_config_sha256"] and sha(ROOT / config["protocol_path"]) == config["protocol_sha256"]
    phase = config["phase"]
    pair_count = 12 * config["pairs_per_family"]
    expected_rows = pair_count * (len(config["native_lanes"]) + len(config["continuous_references"]))
    expected_formal = phase == "P2"
    checks["grid_contract"] = phase in {"P1", "P2"} and tuple(config["families"]) == FAMILIES and config["pairs_per_family"] == (1 if phase == "P1" else 20) and config["expected_prediction_rows"] == expected_rows
    checks["seed_ledger_complete_distinct"] = (
        seed_ledger["phase"] == phase and len(seed_ledger["rows"]) == pair_count and seed_ledger["formal_seed_consumed"] == expected_formal
        and all(set(item["seeds"]) == set(config["required_seed_streams"]) and len(set(item["seeds"].values())) == 6 for item in seed_ledger["rows"])
    )

    observed_proposals = read_jsonl(run_dir / "proposals.jsonl")
    observed_predictions = read_jsonl(run_dir / "predictions.jsonl")
    recomputed_proposals, recomputed_predictions, recomputed_seeds = runner.build_records(config, code_hashes["aggregate_sha256"])
    checks["proposal_recomputation"] = _json_canonical(observed_proposals) == _json_canonical(recomputed_proposals)
    checks["prediction_recomputation"] = [ _scientific_prediction(row) for row in observed_predictions ] == [ _scientific_prediction(row) for row in recomputed_predictions ]
    checks["seed_recomputation"] = seed_ledger["rows"] == recomputed_seeds
    lane_counts = {}
    for row in observed_predictions:
        lane_counts[row["lane"]] = lane_counts.get(row["lane"], 0) + 1
    checks["lane_grid_complete"] = len(observed_predictions) == expected_rows and set(lane_counts) == set(config["native_lanes"]) | set(config["continuous_references"]) and all(count == pair_count for count in lane_counts.values())
    checks["native_outputs_unweighted"] = all(
        row["kind"] != "NATIVE" or row["lane"] == "MSCC"
        or all(list(support) == sorted(set(support)) for support in row["prediction"]["supports"])
        for row in observed_predictions
    )
    checks["random_diagnostics_charged"] = all(
        row["lane"] != "RANDOM_MATCHED_GROUP"
        or (len(row["random_diagnostics"]) == 32 and row["cost"]["diagnostic_evaluated_candidate_count"] == sum(item["evaluated"] for item in row["random_diagnostics"]))
        for row in observed_predictions
    )
    checks["runtime_protocol_complete"] = all(len(row["cost"]["runtime_seconds"]) == 5 and row["cost"]["median_runtime_seconds"] >= 0.0 for row in observed_predictions)
    checks["formal_seed_state"] = config["formal_seed_manifest_status"] == "UNGENERATED" and not config["formal_seed_consumed"] and closure["formal_seed_consumed"] == expected_formal
    result = {
        "schema_version": f"pc2.{phase.lower()}.prelabel_validation.v1",
        "run_id": run_dir.name,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "truth_opened": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    result = validate(run_dir)
    runner.write_json(run_dir / "prelabel_validation.json", result)
    print(json.dumps({"status": result["status"], "passed": result["passed_count"], "checks": result["check_count"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
