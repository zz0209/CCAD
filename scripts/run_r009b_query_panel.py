"""Freeze a deterministic, source-only, energy-stratified R009 query panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def selection_hash(salt: str, seed: int, atom: int) -> str:
    return hashlib.sha256(f"{salt}\0{seed}\0{atom}".encode()).hexdigest()


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
    code_rows = [{"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size}]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    census_path = ROOT / cfg["source_census_path"]
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(census_path.resolve()), "sha256": sha256(census_path), "bytes": census_path.stat().st_size, "source": cfg["source_census_run"], "license_or_access_boundary": "internal", "role": "source_census"},
    ]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "audit_opened": False,
        "candidate_family_frozen": False, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": "cpu",
        "seeds": cfg["source_seeds"], "resource_lease": "none_lightweight_deterministic_selection", "git_head_at_run": git_head,
        "resource_lease_reason": "640-row deterministic selection from a 15,360-row local JSONL census; no heavy CPU, GPU, or bulk I/O",
        "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        if sha256(census_path) != cfg["source_census_sha256"]:
            raise ValueError("source census hash mismatch")
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        expected_grid = {(seed, atom) for seed in cfg["source_seeds"] for atom in range(cfg["num_latents"])}
        actual_grid = {(row["seed"], row["atom"]) for row in census}
        if actual_grid != expected_grid or len(census) != len(expected_grid):
            raise ValueError("source census is not the complete unique seed-atom grid")
        selected: list[dict] = []
        summaries: list[dict] = []
        for seed in cfg["source_seeds"]:
            rows = [row for row in census if row["seed"] == seed]
            ranked = sorted(rows, key=lambda row: (row[cfg["stratification_field"]], row["atom"]))
            if len(ranked) != cfg["num_latents"]:
                raise ValueError(f"seed {seed} census size mismatch")
            for stratum in range(cfg["strata_per_seed"]):
                begin = stratum * cfg["atoms_per_stratum"]
                end = begin + cfg["atoms_per_stratum"]
                members = ranked[begin:end]
                chosen = sorted(
                    members,
                    key=lambda row: (selection_hash(cfg["selection_salt"], seed, row["atom"]), row["atom"]),
                )[: cfg["queries_per_stratum"]]
                for row in chosen:
                    selected.append({
                        **row,
                        "energy_stratum": stratum,
                        "energy_rank_within_seed": ranked.index(row),
                        "stratum_population": len(members),
                        "stratum_sample_size": cfg["queries_per_stratum"],
                        "inclusion_fraction": cfg["queries_per_stratum"] / len(members),
                        "selection_hash": selection_hash(cfg["selection_salt"], seed, row["atom"]),
                    })
                summaries.append({
                    "seed": seed,
                    "energy_stratum": stratum,
                    "population": len(members),
                    "selected": len(chosen),
                    "minimum_energy": members[0][cfg["stratification_field"]],
                    "maximum_energy": members[-1][cfg["stratification_field"]],
                })
        panel_path = run_dir / "query_panel.jsonl"
        with panel_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(selected, key=lambda item: (item["seed"], item["energy_stratum"], item["selection_hash"])):
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        counts = {(seed, stratum): 0 for seed in cfg["source_seeds"] for stratum in range(cfg["strata_per_seed"])}
        for row in selected:
            counts[(row["seed"], row["energy_stratum"])] += 1
        checks = {
            "source_census_hash_bound": sha256(census_path) == cfg["source_census_sha256"],
            "complete_sampling_frame": len(census) == len(expected_grid) == 15360,
            "panel_size": len(selected) == len(cfg["source_seeds"]) * cfg["queries_per_seed"] == 640,
            "unique_queries": len({(row["seed"], row["atom"]) for row in selected}) == len(selected),
            "balanced_strata": all(value == cfg["queries_per_stratum"] for value in counts.values()),
            "all_atoms_eligible": cfg["eligibility_rule"] == "all_source_atoms_no_target_dependent_exclusion",
            "source_only_information": set(cfg["allowed_splits"]) == {"mean", "discovery"} and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "audit_not_opened": not cfg["audit_opened"],
            "no_target_fields": all(not any(key.startswith("target") for key in row) for row in selected),
        }
        record = {
            "checks": checks, "sampling_frame_rows": len(census), "query_panel_rows": len(selected), "seed_stratum_summaries": summaries,
            "minimum_panel_firing": min(row["discovery_firing_count"] for row in selected),
            "minimum_panel_active_documents": min(row["active_document_count"] for row in selected),
            "minimum_panel_document_ess": min(row["document_energy_ess"] for row in selected),
            "query_panel_sha256": sha256(panel_path),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r009b_query_panel.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
        "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
