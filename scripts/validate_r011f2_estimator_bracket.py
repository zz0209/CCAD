"""Independent integrity and decision validation for the C045 bracket."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "estimator_surface.jsonl").read_text(encoding="utf-8").splitlines() if line]
    decisions = [json.loads(line) for line in (run_dir / "estimator_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    anchors = [row for row in rows if row["query_role"] == "anchor"]
    all_directions = {(row["source_seed"], row["target_seed"]) for row in anchors}
    expected_decisions = []
    summaries = {}
    for estimator in cfg["estimators"]:
        groups = {}
        for row in anchors:
            if row["estimator"] == estimator:
                groups.setdefault((row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"]), []).append(row)
        found = []
        for key, values in sorted(groups.items()):
            ordered = sorted(values, key=lambda row: cfg["candidate_ranks"].index(row["rank"]))
            passing = [row for row in ordered if row["evaluable"] and row["calibration_positive_bcc"] >= cfg["minimum_calibration_bcc"] and row["calibration_positive_residual"] <= cfg["maximum_calibration_normalized_residual"] and row["calibration_bcc_contrast"] > cfg["minimum_calibration_contrast"] and row["collision_improvement_over_global"] >= cfg["minimum_collision_improvement_over_global"] and row["rank_boundary_relative_gap"] is not None and row["rank_boundary_relative_gap"] >= cfg["minimum_rank_boundary_relative_gap"]]
            selected = passing[0] if passing else None
            expected_decisions.append((estimator, *key, "FOUND_RELATION" if selected else "UNRESOLVED_RELATION", selected["rank"] if selected else None))
            if selected: found.append(selected)
        coverage = len(found) / cfg["anchor_units"]
        directions = {(row["source_seed"], row["target_seed"]) for row in found}
        strata = {row["energy_stratum"] for row in found}
        progression = coverage >= cfg["minimum_progression_coverage"] and len(strata) >= cfg["minimum_covered_strata"] and directions == all_directions
        summaries[estimator] = {"found": len(found), "coverage": coverage, "rank_counts": dict(Counter(row["rank"] for row in found)), "covered_strata": sorted(strata), "covered_directions": len(directions), "progression_pass": bool(progression)}
    actual_decisions = [(row["estimator"], row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"], row["decision"], row["selected_rank"]) for row in decisions]
    outcome = "PROCEED_WITH_SINGLE_QUALIFIED_ESTIMATOR" if sum(value["progression_pass"] for value in summaries.values()) == 1 else ("STOP_LOCAL_CONTRIBUTION_KERNEL_FAMILY" if not any(value["progression_pass"] for value in summaries.values()) else "FAIL_MULTIPLE_ESTIMATORS_QUALIFIED_NO_SELECTION_RULE")
    saved = np.load(run_dir / "estimator_loadings.npz", allow_pickle=False)
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "status_pass": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "surface_hash": sha256(run_dir / "estimator_surface.jsonl") == record["surface_sha256"],
        "decision_hash": sha256(run_dir / "estimator_decisions.jsonl") == record["decisions_sha256"],
        "loadings_hash": sha256(run_dir / "estimator_loadings.npz") == record["loadings_sha256"],
        "surface_grid": len(rows) == len(cfg["estimators"]) * cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]),
        "decision_grid": len(decisions) == len(cfg["estimators"]) * cfg["anchor_units"],
        "decisions_recomputed": actual_decisions == expected_decisions,
        "summaries_recomputed": all(summaries[name][field] == record["estimator_summaries"][name][field] for name in summaries for field in summaries[name]),
        "outcome_recomputed": outcome == record["screen_decision"],
        "loading_shapes": saved["source_loadings"].shape[0] == saved["target_loadings"].shape[0] == int(np.sum(saved["ranks"] > 0)),
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
    }
    print(json.dumps({"run_id": run_dir.name, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "screen_decision": outcome, "estimator_summaries": summaries}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
