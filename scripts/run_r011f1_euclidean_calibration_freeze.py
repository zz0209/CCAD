"""Freeze minimum-rank Euclidean FCC decisions from the fixed calibration surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    value = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve()]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    surface = ROOT / cfg["surface_path"]
    loadings = ROOT / cfg["loadings_path"]
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD", "license_or_access_boundary": "internal", "role": "run_protocol"},
        {"path": str(surface.resolve()), "sha256": sha256(surface), "bytes": surface.stat().st_size, "source": cfg["surface_run"], "license_or_access_boundary": "internal", "role": "calibration_surface"},
        {"path": str(loadings.resolve()), "sha256": sha256(loadings), "bytes": loadings.stat().st_size, "source": cfg["surface_run"], "license_or_access_boundary": "internal", "role": "frozen_loadings"},
    ]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": "Freeze minimum-rank Euclidean FCC calibration decisions", "milestone": "M4-pre-audit-calibration-freeze",
        "evidence_level": "real_sae_calibration_only", "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows),
        "audit_opened": False, "candidate_family_frozen": True, "mean_constants_source_split": "mean",
        "threshold_source_split": "calibration", "statistics_unit": "source_query_and_ordered_seed_pair",
        "device": "cpu", "seeds": [1, 2, 3, 4, 5], "resource_lease": "not_required_lightweight",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record = None
    error = None
    status = "FAIL"
    try:
        if sha256(surface) != cfg["surface_sha256"] or sha256(loadings) != cfg["loadings_sha256"]:
            raise ValueError("frozen input hash mismatch")
        if cfg["audit_opened"] or cfg["forbidden_splits"] != ["audit"]:
            raise ValueError("audit boundary drift")
        rows = [json.loads(line) for line in surface.read_text(encoding="utf-8").splitlines() if line]
        anchors = [row for row in rows if row["query_role"] == "anchor"]
        groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
        for row in anchors:
            groups[(row["source_seed"], row["source_atom"], row["energy_stratum"], row["target_seed"])].append(row)
        decisions = []
        for key, candidates in sorted(groups.items()):
            ordered = sorted(candidates, key=lambda row: cfg["candidate_ranks"].index(row["rank"]))
            passing = [row for row in ordered if row["evaluable"] and row["calibration_positive_bcc"] > cfg["minimum_calibration_bcc"] and row["calibration_bcc_contrast"] > cfg["minimum_calibration_contrast"] and row["collision_improvement_over_global"] >= cfg["minimum_collision_improvement_over_global"] and row["rank_boundary_relative_gap"] >= cfg["minimum_rank_boundary_relative_gap"]]
            selected = passing[0] if passing else None
            decisions.append({
                "source_seed": key[0], "source_atom": key[1], "energy_stratum": key[2], "target_seed": key[3],
                "decision": "FOUND_RELATION" if selected else "UNRESOLVED_RELATION",
                "reason": None if selected else "NO_RANK_PASSED_CALIBRATION_RULES",
                "selected_rank": selected["rank"] if selected else None,
                "loading_index": selected["loading_index"] if selected else None,
                "calibration_positive_bcc": selected["calibration_positive_bcc"] if selected else None,
                "calibration_bcc_contrast": selected["calibration_bcc_contrast"] if selected else None,
                "collision_improvement_over_global": selected["collision_improvement_over_global"] if selected else None,
                "rank_boundary_relative_gap": selected["rank_boundary_relative_gap"] if selected else None,
            })
        output = run_dir / "calibration_decisions.jsonl"
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
        found = [row for row in decisions if row["decision"] == "FOUND_RELATION"]
        coverage = len(found) / len(decisions)
        directions = {(row["source_seed"], row["target_seed"]) for row in found}
        all_directions = {(row["source_seed"], row["target_seed"]) for row in decisions}
        checks = {
            "frozen_inputs_bound": True,
            "complete_unit_grid": len(decisions) == 160 and len(groups) == 160,
            "minimum_rank_rule_exact": all(row["selected_rank"] in cfg["candidate_ranks"] for row in found),
            "coverage_gate": coverage >= cfg["minimum_screen_coverage"],
            "strata_gate": len({row["energy_stratum"] for row in found}) >= cfg["minimum_covered_strata"],
            "direction_gate": (directions == all_directions) if cfg["require_all_ordered_seed_directions"] else True,
            "contrast_gate": np.mean([row["calibration_bcc_contrast"] > 0 for row in found]) >= cfg["minimum_positive_contrast_fraction"],
            "collision_gate": float(np.median([row["collision_improvement_over_global"] for row in found])) >= cfg["minimum_collision_improvement_over_global"],
            "audit_not_opened": not cfg["audit_opened"],
            "causal_gate_not_evaluated": True,
        }
        record = {"checks": {k: bool(v) for k, v in checks.items()}, "units": len(decisions), "found": len(found), "coverage": coverage, "rank_counts": dict(Counter(row["selected_rank"] for row in found)), "covered_strata": sorted({row["energy_stratum"] for row in found}), "covered_directions": len(directions), "median_found_bcc": float(np.median([row["calibration_positive_bcc"] for row in found])), "median_found_contrast": float(np.median([row["calibration_bcc_contrast"] for row in found])), "median_found_collision_improvement": float(np.median([row["collision_improvement_over_global"] for row in found])), "decisions_sha256": sha256(output), "progression_state": "AWAIT_MATCHED_CAUSAL_GATE", "scope_limit": cfg["scope_limit"]}
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011f1_euclidean_calibration_freeze.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
