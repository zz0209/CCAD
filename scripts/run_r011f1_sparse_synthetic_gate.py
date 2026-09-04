"""Run the bounded sparse-kernel FCC synthetic/refusal gate."""
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
sys.path.insert(0, str(ROOT / ".runtime" / "r009"))
sys.path.insert(0, str(ROOT / "src"))

from scipy import __version__ as scipy_version  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.fuzzy_correspondence import (  # noqa: E402
    fit_fuzzy_correspondence,
    fit_fuzzy_correspondence_from_kernels,
    sparse_contribution_kernels,
    soft_membership_overlap,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_entry(path: Path, source: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": "internal", "role": role}


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def fit_sparse(source_codes, target_codes, source_decoders, target_decoders, weights, rank=1, metric=None, negative=None, contrast=0.0):
    kernels = sparse_contribution_kernels(
        csr_matrix(source_codes), csr_matrix(target_codes), source_decoders, target_decoders, weights,
        source_mean_codes=np.zeros(source_codes.shape[1]), target_mean_codes=np.zeros(target_codes.shape[1]),
        metric=metric, negative_weights=negative,
    )
    return fit_fuzzy_correspondence_from_kernels(kernels, rank=rank, contrast_strength=contrast)


def run_families(cfg: dict) -> list[dict]:
    rows = []
    uniform = lambda n: np.full(n, 1.0 / n)

    rng = np.random.default_rng(cfg["seeds"][0])
    latent = rng.normal(size=(2000, 2))
    angle = 0.63
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = latent @ rotation
    source_codes = latent
    target_codes = rotated
    source_decoders = np.eye(2)
    target_decoders = rotation.T
    result = fit_sparse(source_codes, target_codes, source_decoders, target_decoders, uniform(len(latent)), rank=1)
    passed = result.canonical_values[0] >= cfg["minimum_canonical_value"] and np.all(result.source_membership > 0.05) and np.all(result.target_membership > 0.05)
    rows.append({"family": "rotation", "passed": bool(passed), "canonical_value": float(result.canonical_values[0]), "decision": "FOUND_RELATION" if passed else "UNRESOLVED_RELATION"})

    rng = np.random.default_rng(cfg["seeds"][1])
    latent = rng.normal(size=(2000, 2))
    source_codes = latent
    target_codes = np.column_stack([0.35 * latent[:, 0], 0.65 * latent[:, 0], latent[:, 1]])
    result = fit_sparse(source_codes, target_codes, np.eye(2), np.asarray([[1, 0], [1, 0], [0, 1]]), uniform(len(latent)), rank=2)
    passed = result.canonical_values[-1] >= cfg["minimum_canonical_value"] and result.target_effective_support > 2.4
    rows.append({"family": "split_merge", "passed": bool(passed), "minimum_canonical_value": float(result.canonical_values[-1]), "target_effective_support": result.target_effective_support, "decision": "FOUND_RELATION" if passed else "UNRESOLVED_RELATION"})

    rng = np.random.default_rng(cfg["seeds"][2])
    n = 2000
    first, second = rng.normal(size=(2, n))
    first[n // 2:] = 0.0
    second[:n // 2] = 0.0
    source_codes = np.column_stack([first, second])
    target_codes = np.column_stack([0.5 * first, 0.5 * (first + second), 0.5 * second])
    weights_first = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    weights_second = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    first_relation = fit_sparse(source_codes, target_codes, np.ones((2, 1)), np.ones((3, 1)), weights_first)
    second_relation = fit_sparse(source_codes, target_codes, np.ones((2, 1)), np.ones((3, 1)), weights_second)
    overlap = soft_membership_overlap(first_relation.target_membership, second_relation.target_membership)
    passed = first_relation.target_membership[1] >= cfg["minimum_overlap_membership"] and second_relation.target_membership[1] >= cfg["minimum_overlap_membership"] and overlap > 0
    rows.append({"family": "overlap", "passed": bool(passed), "shared_target_memberships": [float(first_relation.target_membership[1]), float(second_relation.target_membership[1])], "cross_query_overlap": overlap, "decision": "FOUND_OVERLAPPING_RELATIONS" if passed else "UNRESOLVED_RELATION"})

    rng = np.random.default_rng(cfg["seeds"][3])
    signal, nuisance = rng.normal(size=(2, 2000))
    source_codes = np.column_stack([signal, nuisance])
    target_codes = np.column_stack([signal, -nuisance])
    causal = fit_sparse(source_codes, target_codes, np.eye(2), np.eye(2), uniform(len(signal)), metric=np.diag([1.0, 0.0]))
    passed = causal.source_membership[0] > 0.99 and causal.target_membership[0] > 0.99
    rows.append({"family": "downstream_null", "passed": bool(passed), "signal_memberships": [float(causal.source_membership[0]), float(causal.target_membership[0])], "decision": "FOUND_RELATION" if passed else "UNRESOLVED_RELATION"})

    rng = np.random.default_rng(cfg["seeds"][4])
    n = 2000
    concept, nuisance = rng.normal(size=(2, n))
    source_codes = np.column_stack([concept, nuisance])
    target_codes = source_codes.copy()
    source_codes[n // 2:, 0] = 0.0
    target_codes[n // 2:, 0] = 0.0
    positive = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    negative = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    plain = fit_sparse(source_codes, target_codes, np.eye(2), np.eye(2), positive)
    contrasted = fit_sparse(source_codes, target_codes, np.eye(2), np.eye(2), positive, negative=negative, contrast=1.0)
    gain = float(contrasted.source_membership[0] - plain.source_membership[0])
    passed = gain > 0 and contrasted.source_membership[0] > plain.source_membership[0]
    rows.append({"family": "hard_negative", "passed": bool(passed), "concept_membership_gain": gain, "decision": "FOUND_CONTRASTIVE_RELATION" if passed else "UNRESOLVED_RELATION"})

    rng = np.random.default_rng(cfg["seeds"][5])
    latent = rng.normal(size=(5000, 2))
    competing = fit_sparse(latent, latent, np.eye(2), np.eye(2), uniform(len(latent)), rank=1)
    gap = competing.rank_boundary_relative_gap
    passed = gap is not None and gap <= cfg["maximum_competing_rank_boundary_gap"]
    rows.append({"family": "competing_relation", "passed": bool(passed), "rank_boundary_relative_gap": gap, "decision": "UNRESOLVED_RELATION", "reason": "COMPETING_RELATION_RANK_BOUNDARY"})
    return rows


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
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/fuzzy_correspondence.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    write_json(run_dir / "inputs.json", {"inputs": [file_entry(args.config.resolve(), "CCAD frozen config", "protocol")]})
    write_json(run_dir / "manifest.json", {
        **{key: cfg[key] for key in ("schema_version", "run_id", "run_parent", "purpose", "milestone", "evidence_level", "audit_opened", "candidate_family_frozen", "mean_constants_source_split", "threshold_source_split", "statistics_unit", "device", "seeds")},
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "resource_lease": "none; bounded CPU synthetic run",
        "resource_lease_reason": "small deterministic implementation gate",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    status, error, rows = "FAIL", None, []
    try:
        rows = run_families(cfg)
        status = "PASS" if len(rows) == len(cfg["families"]) and all(row["passed"] for row in rows) and [row["family"] for row in rows] == cfg["families"] else "FAIL"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "families_passed": sum(row.get("passed", False) for row in rows), "families_total": len(cfg["families"]), "metrics_raw_sha256": sha256(raw_path), "generator_script_path": "scripts/run_r011f1_sparse_synthetic_gate.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
