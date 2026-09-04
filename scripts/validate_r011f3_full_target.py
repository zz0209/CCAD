"""Independent artifact and decision validation for the C046 full-target screen."""
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
    rows = [
        json.loads(line)
        for line in (run_dir / "full_target_surface.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    decisions = [
        json.loads(line)
        for line in (run_dir / "full_target_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    anchors = [row for row in rows if row["query_role"] == "anchor"]
    all_directions = {(row["source_seed"], row["target_seed"]) for row in anchors}

    groups: dict[tuple[int, int, int, int], list[dict]] = {}
    for row in anchors:
        key = (
            row["source_seed"],
            row["source_atom"],
            row["target_seed"],
            row["energy_stratum"],
        )
        groups.setdefault(key, []).append(row)

    expected_decisions = []
    found = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda row: cfg["candidate_ranks"].index(row["rank"]))
        passing = [
            row
            for row in ordered
            if row["evaluable"]
            and row["calibration_positive_bcc"] >= cfg["minimum_calibration_bcc"]
            and row["calibration_positive_residual"] <= cfg["maximum_calibration_normalized_residual"]
            and row["calibration_bcc_contrast"] > cfg["minimum_calibration_contrast"]
            and row["collision_improvement_over_global"] >= cfg["minimum_collision_improvement_over_global"]
            and row["rank_boundary_relative_gap"] >= cfg["minimum_rank_boundary_relative_gap"]
        ]
        selected = passing[0] if passing else None
        expected_decisions.append(
            (
                *key,
                "FOUND_RELATION" if selected else "UNRESOLVED_RELATION",
                selected["rank"] if selected else None,
            )
        )
        if selected:
            found.append(selected)

    actual_decisions = [
        (
            row["source_seed"],
            row["source_atom"],
            row["target_seed"],
            row["energy_stratum"],
            row["decision"],
            row["selected_rank"],
        )
        for row in decisions
    ]
    coverage = len(found) / cfg["anchor_units"]
    directions = {(row["source_seed"], row["target_seed"]) for row in found}
    strata = {row["energy_stratum"] for row in found}
    progression = (
        coverage >= cfg["minimum_progression_coverage"]
        and len(strata) >= cfg["minimum_covered_strata"]
        and directions == all_directions
    )
    outcome = (
        "PROCEED_FULL_TARGET_TO_MATCHED_CAUSAL_GATE"
        if progression
        else "STOP_CANDIDATE_TRUNCATION_EXPLANATION"
    )

    memberships = np.load(run_dir / "target_memberships.npz", allow_pickle=False)
    loadings = np.load(run_dir / "anchor_maxrank_loadings.npz", allow_pickle=False)
    evaluable_rows = [row for row in rows if row["evaluable"]]
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "status_pass": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "surface_hash": sha256(run_dir / "full_target_surface.jsonl") == record["surface_sha256"],
        "decision_hash": sha256(run_dir / "full_target_decisions.jsonl") == record["decisions_sha256"],
        "membership_hash": sha256(run_dir / "target_memberships.npz") == record["memberships_sha256"],
        "loadings_hash": sha256(run_dir / "anchor_maxrank_loadings.npz") == record["loadings_sha256"],
        "surface_grid": len(rows) == cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]),
        "decision_grid": len(decisions) == cfg["anchor_units"],
        "full_target_exact": all(row["target_candidate_count"] == cfg["num_latents"] for row in evaluable_rows),
        "decisions_recomputed": actual_decisions == expected_decisions,
        "summary_recomputed": (
            record["found"] == len(found)
            and record["coverage"] == coverage
            and record["rank_counts"] == {str(key): value for key, value in Counter(row["rank"] for row in found).items()}
            and record["covered_strata"] == sorted(strata)
            and record["covered_directions"] == len(directions)
            and record["progression_pass"] is bool(progression)
        ),
        "outcome_recomputed": record["screen_decision"] == outcome,
        "membership_shapes": memberships["query"].shape == memberships["global_control"].shape == (len(evaluable_rows), cfg["num_latents"]),
        "loading_shapes": loadings["source"].shape == (cfg["anchor_units"], cfg["source_candidate_count"], max(cfg["candidate_ranks"])) and loadings["target"].shape == (cfg["anchor_units"], cfg["num_latents"], max(cfg["candidate_ranks"])),
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
    }
    payload = {
        "run_id": run_dir.name,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "screen_decision": outcome,
        "found": len(found),
        "coverage": coverage,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
