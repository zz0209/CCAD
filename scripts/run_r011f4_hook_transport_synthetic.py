"""Run the pre-calibration C047 hook-transport synthetic/refusal gate."""
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
    transport_metrics,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": "CCAD frozen artifact", "license_or_access_boundary": "internal", "role": role,
    }


def run_fixtures(cfg: dict) -> list[dict]:
    rows: list[dict] = []

    rng = np.random.default_rng(cfg["seeds"][0])
    source_basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    target_basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    discovery = rng.normal(size=(512, 2)); calibration = rng.normal(size=(512, 2))
    fit = fit_hook_space_transport(discovery @ rotation @ target_basis.T, discovery @ source_basis.T, np.ones(512), rank=2, ridge_fraction=1e-6)
    metrics = transport_metrics(calibration @ source_basis.T, fit.predict(calibration @ rotation @ target_basis.T), np.ones(512))
    passed = fit.status == "OK" and metrics.bcc is not None and metrics.bcc > .999 and metrics.normalized_residual is not None and metrics.normalized_residual < .002
    rows.append({"family": "rotation", "passed": bool(passed), "status": fit.status, "bcc": metrics.bcc, "normalized_residual": metrics.normalized_residual})

    rng = np.random.default_rng(cfg["seeds"][1])
    discovery = rng.normal(size=(600, 2)); calibration = rng.normal(size=(600, 2))
    source_mix = np.array([[1, 0], [0, 1], [.4, .6], [0, 0]], dtype=float)
    target_mix = np.array([[.4, 0], [.6, 0], [0, 1], [.3, .7]], dtype=float)
    fit = fit_hook_space_transport(discovery @ target_mix.T, discovery @ source_mix.T, np.ones(600), rank=2, ridge_fraction=1e-6)
    metrics = transport_metrics(calibration @ source_mix.T, fit.predict(calibration @ target_mix.T), np.ones(600))
    passed = fit.status == "OK" and metrics.bcc is not None and metrics.bcc > .999 and metrics.normalized_residual is not None and metrics.normalized_residual < .002
    rows.append({"family": "split_merge", "passed": bool(passed), "status": fit.status, "bcc": metrics.bcc, "normalized_residual": metrics.normalized_residual})

    rng = np.random.default_rng(cfg["seeds"][2])
    discovery = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
    positive = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
    negative = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
    fit = fit_hook_space_transport(discovery, discovery, np.ones(500), rank=2, ridge_fraction=cfg["ridge_fraction"])
    pos = transport_metrics(positive, fit.predict(positive), np.ones(500)); neg = transport_metrics(negative, fit.predict(negative), np.ones(500))
    decision = decide_transport_gate(pos, neg, rank_boundary_relative_gap=.5, collision_improvement_over_global=.2, raw_control_specificity=0, global_control_specificity=0)
    passed = decision.decision == "UNRESOLVED_RELATION" and decision.reason == "HARD_NEGATIVE_CONTRAST"
    rows.append({"family": "query_null_global_nuisance", "passed": bool(passed), "decision": decision.decision, "reason": decision.reason, "specificity": decision.specificity})

    source = np.ones((20, 1)); zero = np.zeros((20, 1))
    pos = transport_metrics(source, np.full_like(source, .9), np.ones(20)); neg = transport_metrics(zero, np.full_like(zero, .01), np.ones(20))
    decision = decide_transport_gate(pos, neg, rank_boundary_relative_gap=.5, collision_improvement_over_global=.2, raw_control_specificity=1.0, global_control_specificity=.1)
    passed = decision.decision == "UNRESOLVED_RELATION" and decision.reason == "RAW_OR_GLOBAL_CONTROL_NOT_BEATEN"
    rows.append({"family": "raw_control_rejection", "passed": bool(passed), "decision": decision.decision, "reason": decision.reason, "specificity": decision.specificity, "raw_control_specificity": 1.0})

    rng = np.random.default_rng(cfg["seeds"][4])
    latent = rng.normal(size=(300, 1))
    target = np.column_stack([latent[:, 0], 2 * latent[:, 0], np.zeros(300)])
    source = np.column_stack([3 * latent[:, 0], -latent[:, 0], np.zeros(300)])
    fit = fit_hook_space_transport(target, source, np.ones(300), rank=2, ridge_fraction=cfg["ridge_fraction"])
    passed = fit.status == "RANK_DEFICIENT" and fit.effective_rank == 1
    rows.append({"family": "rank_deficient", "passed": bool(passed), "status": fit.status, "requested_rank": 2, "effective_rank": fit.effective_rank})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/hook_transport.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    protocol = ROOT / cfg["protocol_document"]
    write_json(run_dir / "inputs.json", {"inputs": [file_entry(args.config.resolve(), "run_protocol"), file_entry(protocol, "protocol_document")]})
    write_json(run_dir / "manifest.json", {
        **{key: cfg[key] for key in ("schema_version", "run_id", "run_parent", "purpose", "milestone", "evidence_level", "audit_opened", "candidate_family_frozen", "mean_constants_source_split", "threshold_source_split", "statistics_unit", "device", "seeds")},
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows),
        "resource_lease": "none; bounded CPU synthetic run", "resource_lease_reason": "deterministic pre-calibration representation and refusal gate",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    status, error, rows = "FAIL", None, []
    try:
        rows = run_fixtures(cfg)
        status = "PASS" if [row["family"] for row in rows] == cfg["families"] and all(row["passed"] for row in rows) else "FAIL"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
    raw = run_dir / "metrics.raw.jsonl"; raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "families_passed": sum(row.get("passed", False) for row in rows), "families_total": len(cfg["families"]), "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011f4_hook_transport_synthetic.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir); write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
