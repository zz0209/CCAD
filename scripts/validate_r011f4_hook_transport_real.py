"""Independent artifact and decision validation for the C047 real screen."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.hook_transport import decide_transport_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8")); record = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    residual_mode = "nuisance_state_count" in cfg
    if record.get("screen_decision") == "STOP_NUISANCE_VARIANCE_THRESHOLD_NOT_REACHED":
        checks = {
            "artifact_contract": validate_run_directory(run_dir).ok,
            "status_pass": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "PASS",
            "residual_mode": residual_mode and record.get("residual_mode") is True,
            "rank_cap_reached": record.get("nuisance_rank") == cfg["nuisance_maximum_rank"],
            "threshold_not_reached": record.get("nuisance_explained_variance_fraction", 1.0) < cfg["nuisance_explained_variance_threshold"],
            "no_screen_or_progression": record.get("surface_rows") == 0 and record.get("decision_rows") == 0 and record.get("progression_pass") is False,
            "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
        }
        print(json.dumps({"run_id": run_dir.name, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "screen_decision": record["screen_decision"], "nuisance_rank": record["nuisance_rank"], "nuisance_explained_variance_fraction": record["nuisance_explained_variance_fraction"]}, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1
    rows = [json.loads(line) for line in (run_dir / "hook_transport_surface.jsonl").read_text(encoding="utf-8").splitlines() if line]
    decisions = [json.loads(line) for line in (run_dir / "hook_transport_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    anchors = [row for row in rows if row["query_role"] == "anchor" and row["evaluable"]]
    groups = {}
    for row in anchors: groups.setdefault((row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"]), []).append(row)
    expected = []; found = []
    for key, values in sorted(groups.items()):
        selected, reason = None, "NO_RANK_PASSED_HOOK_TRANSPORT_GATE"
        for row in sorted(values, key=lambda value: cfg["candidate_ranks"].index(value["rank"])):
            if row["query_status"] != "OK" or row["raw_status"] != "OK" or row["global_status"] != "OK": reason = "RANK_DEFICIENT"; continue
            if residual_mode and row["source_residual_energy_fraction"] < cfg["minimum_source_residual_energy_fraction"]:
                reason = "SOURCE_RESIDUAL_ENERGY_BELOW_FLOOR"; continue
            raw_control = max(row["raw_specificity"], row["unresidualized_query_specificity"]) if residual_mode else row["raw_specificity"]
            gate = decide_transport_gate(SimpleNamespace(**row["query_positive"]), SimpleNamespace(**row["query_negative"]), rank_boundary_relative_gap=row["source_rank_boundary_relative_gap"], collision_improvement_over_global=row["collision_improvement_over_global"], raw_control_specificity=raw_control, global_control_specificity=row["global_specificity"], minimum_bcc=cfg["minimum_calibration_bcc"], maximum_normalized_residual=cfg["maximum_calibration_normalized_residual"], minimum_specificity=cfg["minimum_calibration_specificity"], minimum_control_advantage=cfg["minimum_control_specificity_advantage"], minimum_collision_improvement=cfg["minimum_collision_improvement_over_global"], minimum_rank_gap=cfg["minimum_rank_boundary_relative_gap"])
            reason = gate.reason
            if gate.decision == "FOUND_RELATION": selected = row; break
        expected.append((*key, "FOUND_RELATION" if selected else "UNRESOLVED_RELATION", None if selected else reason, selected["rank"] if selected else None))
        if selected: found.append(selected)
    actual = [(row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"], row["decision"], row["reason"], row["selected_rank"]) for row in decisions]
    coverage = len(found) / cfg["anchor_units"]; directions = {(row["source_seed"], row["target_seed"]) for row in found}; all_directions = {(row["source_seed"], row["target_seed"]) for row in anchors}; strata = {row["energy_stratum"] for row in found}
    progression = coverage >= cfg["minimum_progression_coverage"] and len(strata) >= cfg["minimum_covered_strata"] and directions == all_directions
    outcome = ("PROCEED_RESIDUAL_TRANSPORT_TO_MATCHED_CAUSAL_GATE" if progression else "STOP_RESIDUAL_FCC_REPRESENTATION") if residual_mode else ("PROCEED_HOOK_TRANSPORT_TO_MATCHED_CAUSAL_GATE" if progression else "STOP_HOOK_TRANSPORT_REPRESENTATION")
    saved = np.load(run_dir / "anchor_maxrank_hook_factors.npz", allow_pickle=False); max_rank = max(cfg["candidate_ranks"])
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "status_pass": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "surface_hash": sha256(run_dir / "hook_transport_surface.jsonl") == record["surface_sha256"],
        "decision_hash": sha256(run_dir / "hook_transport_decisions.jsonl") == record["decisions_sha256"],
        "loadings_hash": sha256(run_dir / "anchor_maxrank_hook_factors.npz") == record["loadings_sha256"],
        "surface_grid": len(rows) == cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]),
        "decision_grid": len(decisions) == cfg["anchor_units"],
        "decisions_recomputed": actual == expected,
        "summary_recomputed": record["found"] == len(found) and record["coverage"] == coverage and record["rank_counts"] == {str(key): value for key, value in Counter(row["rank"] for row in found).items()} and record["covered_strata"] == sorted(strata) and record["covered_directions"] == len(directions) and record["progression_pass"] is bool(progression),
        "outcome_recomputed": record["screen_decision"] == outcome,
        "strong_controls_present": all(all(key in row for key in (("query_positive", "raw_positive", "global_positive", "query_specificity", "raw_specificity", "global_specificity", "unresidualized_query_specificity") if residual_mode else ("query_positive", "raw_positive", "global_positive", "query_specificity", "raw_specificity", "global_specificity"))) for row in anchors),
        "loading_shapes": saved["source_basis"].shape == saved["query_target"].shape == saved["raw_target"].shape == saved["global_target"].shape == (cfg["anchor_units"], cfg["hook_hidden_size"], max_rank),
        "effective_rank_ledger": all(saved[name].shape == (cfg["anchor_units"],) and np.all((saved[name] >= 0) & (saved[name] <= max_rank)) for name in ("query_effective_rank", "raw_effective_rank", "global_effective_rank")),
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
    }
    print(json.dumps({"run_id": run_dir.name, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "screen_decision": outcome, "found": len(found), "coverage": coverage}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
