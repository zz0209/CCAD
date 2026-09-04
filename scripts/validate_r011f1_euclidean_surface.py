"""Independent integrity validation for the R011-F1 Euclidean FCC surface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    raw = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))["inputs"]
    rows = [
        json.loads(line)
        for line in (run_dir / "euclidean_fcc_surface.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    loadings = np.load(run_dir / "anchor_loadings.npz")

    anchor_rows = [row for row in rows if row["query_role"] == "anchor"]
    evaluable_anchor_rows = [row for row in anchor_rows if row["evaluable"]]
    expected_rows = cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"])
    expected_anchor_rows = cfg["anchor_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"])
    expected_summaries = []
    for rank in cfg["candidate_ranks"]:
        selected = [row for row in evaluable_anchor_rows if row["rank"] == rank]
        collisions = [row["collision_improvement_over_global"] for row in selected]
        expected_summaries.append(
            {
                "rank": rank,
                "evaluable_units": len(selected),
                "median_calibration_bcc": float(np.median([row["calibration_positive_bcc"] for row in selected])),
                "median_calibration_bcc_contrast": float(np.median([row["calibration_bcc_contrast"] for row in selected])),
                "positive_calibration_contrast_fraction": float(np.mean([row["calibration_bcc_contrast"] > 0 for row in selected])),
                "median_collision_improvement_over_global": float(np.median(collisions)),
            }
        )

    loading_indices = sorted(int(row["loading_index"]) for row in evaluable_anchor_rows)
    loading_count = len(evaluable_anchor_rows)
    finite_names = (
        "discovery_positive_bcc",
        "discovery_negative_bcc",
        "calibration_positive_bcc",
        "calibration_negative_bcc",
        "calibration_positive_residual",
    )
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "run_passed": status["status"] == "PASS" and all(raw["checks"].values()),
        "all_declared_inputs_bound": all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"] for row in inputs),
        "surface_hash": sha256(run_dir / "euclidean_fcc_surface.jsonl") == raw["surface_sha256"],
        "loadings_hash": sha256(run_dir / "anchor_loadings.npz") == raw["anchor_loadings_sha256"],
        "complete_unique_grid": len(rows) == expected_rows and len({(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]) for row in rows}) == expected_rows,
        "anchor_grid": len(anchor_rows) == expected_anchor_rows,
        "calibration_without_audit": cfg["splits"] == ["discovery", "calibration"] and cfg["forbidden_splits"] == ["audit"] and not cfg["audit_opened"],
        "no_posthoc_decisions": cfg["threshold_source_split"] == "none_raw_surface_only" and all("decision" not in row for row in rows),
        "evaluable_metrics_finite": all((not row["evaluable"]) or all(np.isfinite(row[name]) for name in finite_names) for row in rows),
        "anchor_collision_metrics_finite": all((not row["evaluable"]) or np.isfinite(row["collision_improvement_over_global"]) for row in anchor_rows),
        "rank_summaries_recomputed": expected_summaries == raw["rank_summaries"],
        "loading_indices_contiguous": loading_indices == list(range(loading_count)),
        "loading_shapes": loadings["source_loadings"].shape == (loading_count, cfg["source_candidate_count"], max(cfg["candidate_ranks"])) and loadings["target_loadings"].shape == (loading_count, cfg["target_candidate_cap"], max(cfg["candidate_ranks"])) and loadings["target_counts"].shape == (loading_count,) and loadings["ranks"].shape == (loading_count,),
        "loading_rank_and_count_bounds": bool(np.all(np.isin(loadings["ranks"], cfg["candidate_ranks"]))) and bool(np.all(loadings["target_counts"] <= cfg["target_candidate_cap"])),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
        "surface_rows": len(rows),
        "evaluable_anchor_rows": loading_count,
        "rank_summaries": expected_summaries,
        "interpretation": "RAW_EUCLIDEAN_FCC_SURFACE_INTEGRITY_ONLY_NO_FOUND_OR_CAUSAL_CLAIM",
    }
    (run_dir / "independent_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
