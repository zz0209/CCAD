"""Run the C048 shared-nuisance residual-transport synthetic gate."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.hook_transport import (  # noqa: E402
    decide_transport_gate,
    fit_hook_space_transport,
    fit_nuisance_projector,
    residualize_hook_process,
    transport_metrics,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    return hashlib.sha256("".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"])).encode()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "source": "CCAD frozen artifact", "license_or_access_boundary": "internal", "role": role}


def run_fixtures(cfg: dict) -> list[dict]:
    rows = []
    rng = np.random.default_rng(cfg["seeds"][0]); process = rng.normal(size=(4000, 3)) * np.sqrt([8.0, 2.0, .2])
    nuisance = fit_nuisance_projector(process, np.ones(len(process)), explained_variance_threshold=cfg["nuisance_explained_variance_threshold"], maximum_rank=3)
    passed = nuisance.status == "OK" and nuisance.rank == 2 and nuisance.explained_variance_fraction >= .9
    rows.append({"family": "nuisance_rank", "passed": bool(passed), "status": nuisance.status, "rank": nuisance.rank, "explained_variance_fraction": nuisance.explained_variance_fraction})

    rng = np.random.default_rng(cfg["seeds"][1]); n = 1200
    global_process = np.column_stack([10 * rng.normal(size=n), np.zeros(n), np.zeros(n)])
    nuisance = fit_nuisance_projector(global_process, np.ones(n), explained_variance_threshold=.9, maximum_rank=3)
    signal = rng.normal(size=n); nuisance_value = 4 * rng.normal(size=n)
    source = np.column_stack([nuisance_value, signal, np.zeros(n)]); target = np.column_stack([nuisance_value, 1.7 * signal, np.zeros(n)])
    source_residual = residualize_hook_process(source, nuisance); target_residual = residualize_hook_process(target, nuisance)
    fit = fit_hook_space_transport(target_residual[:600], source_residual[:600], np.ones(600), rank=1, ridge_fraction=1e-6)
    metric = transport_metrics(source_residual[600:], fit.predict(target_residual[600:]), np.ones(600))
    passed = nuisance.rank == 1 and metric.bcc is not None and metric.bcc > .999 and metric.normalized_residual is not None and metric.normalized_residual < .002
    rows.append({"family": "global_nuisance_recovery", "passed": bool(passed), "nuisance_rank": nuisance.rank, "bcc": metric.bcc, "normalized_residual": metric.normalized_residual})

    before = float(np.mean(np.sum(source * source, axis=1))); orthogonal = np.column_stack([np.zeros(n), signal, np.zeros(n)]); after_values = residualize_hook_process(orthogonal, nuisance); retained = float(np.mean(np.sum(after_values * after_values, axis=1)) / np.mean(np.sum(orthogonal * orthogonal, axis=1)))
    passed = retained > .999999 and before > 0
    rows.append({"family": "query_signal_preservation", "passed": bool(passed), "retained_energy_fraction": retained})

    nuisance_only = np.column_stack([rng.normal(size=n), np.zeros(n), np.zeros(n)]); residual = residualize_hook_process(nuisance_only, nuisance)
    inactive = transport_metrics(residual, np.zeros_like(residual), np.ones(n)); decision = decide_transport_gate(inactive, inactive, rank_boundary_relative_gap=.5, collision_improvement_over_global=.2, raw_control_specificity=0, global_control_specificity=0)
    passed = decision.decision == "UNRESOLVED_RELATION" and decision.reason == "POSITIVE_PROCESS_INACTIVE"
    rows.append({"family": "nuisance_only_refusal", "passed": bool(passed), "decision": decision.decision, "reason": decision.reason, "residual_energy": inactive.source_energy})

    latent = rng.normal(size=(500, 1)); source = np.column_stack([np.zeros(500), latent[:, 0], np.zeros(500)]); target = np.column_stack([np.zeros(500), 2 * latent[:, 0], np.zeros(500)])
    fit = fit_hook_space_transport(residualize_hook_process(target, nuisance), residualize_hook_process(source, nuisance), np.ones(500), rank=2)
    passed = fit.status == "RANK_DEFICIENT" and fit.effective_rank == 1
    rows.append({"family": "rank_deficient", "passed": bool(passed), "status": fit.status, "requested_rank": 2, "effective_rank": fit.effective_rank})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args(); cfg = json.loads(args.config.read_text(encoding="utf-8")); run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/hook_transport.py"]; code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)}); protocol = ROOT / cfg["protocol_document"]; write_json(run_dir / "inputs.json", {"inputs": [file_entry(args.config.resolve(), "run_protocol"), file_entry(protocol, "protocol_document")]})
    write_json(run_dir / "manifest.json", {**{key: cfg[key] for key in ("schema_version", "run_id", "run_parent", "purpose", "milestone", "evidence_level", "audit_opened", "candidate_family_frozen", "mean_constants_source_split", "threshold_source_split", "statistics_unit", "device", "seeds")}, "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows), "resource_lease": "none; bounded CPU synthetic run", "resource_lease_reason": "deterministic pre-calibration nuisance-residual representation gate", "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()})
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started}); status, error, rows = "FAIL", None, []
    try:
        rows = run_fixtures(cfg); status = "PASS" if [row["family"] for row in rows] == cfg["families"] and all(row["passed"] for row in rows) else "FAIL"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
    raw = run_dir / "metrics.raw.jsonl"; raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"); write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "families_passed": sum(row.get("passed", False) for row in rows), "families_total": len(cfg["families"]), "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011f5_residual_transport_synthetic.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}); write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error}); (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir); write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)}); print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error})); return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
