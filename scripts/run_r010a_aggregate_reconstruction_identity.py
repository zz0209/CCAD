"""Measure full-SAE reconstruction agreement across frozen seeds on discovery data."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "r009"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import __version__ as scipy_version  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


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
    code_paths = [Path(__file__).resolve(), ROOT / "scripts" / "run_r009c_atom_discovery.py"]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    census_path = ROOT / cfg["source_census_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(census_path.resolve()), "sha256": sha256(census_path), "bytes": census_path.stat().st_size, "source": "R009a", "license_or_access_boundary": "internal", "role": "independent_mean_constants"},
        {"path": str(asset_manifest.resolve()), "sha256": sha256(asset_manifest), "bytes": asset_manifest.stat().st_size, "source": "R008b", "license_or_access_boundary": "internal", "role": "bulk_asset_hash_ledger"},
    ]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows), "audit_opened": False,
        "candidate_family_frozen": False, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": "cpu",
        "seeds": cfg["source_seeds"], "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "stream all five 131,072-token sparse-code matrices from D: and reconstruct aggregate hook outputs",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {
            "census": sha256(census_path).lower() == cfg["source_census_sha256"].lower(),
            "asset_manifest": sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower(),
        }
        if not all(bound.values()):
            raise ValueError(f"frozen input hash mismatch: {bound}")
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        stats = {(row["seed"], row["atom"]): row for row in census}
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["discovery_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float32, copy=False) for seed in cfg["source_seeds"]}
        mean_vectors = {}
        for seed in cfg["source_seeds"]:
            means = np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
            mean_vectors[seed] = means @ decoders[seed].astype(np.float64)
        energies = {seed: 0.0 for seed in cfg["source_seeds"]}
        crosses = {(left, right): 0.0 for left in cfg["source_seeds"] for right in cfg["source_seeds"] if left < right}
        elapsed_batches = []
        for start in range(0, cfg["discovery_tokens"], cfg["batch_tokens"]):
            batch_started = time.perf_counter()
            stop = min(start + cfg["batch_tokens"], cfg["discovery_tokens"])
            centered = {}
            for seed in cfg["source_seeds"]:
                reconstruction = np.asarray(matrices[seed][start:stop] @ decoders[seed], dtype=np.float64)
                centered[seed] = reconstruction - mean_vectors[seed][None, :]
                energies[seed] += float(np.einsum("ij,ij->", centered[seed], centered[seed]))
            for left, right in crosses:
                crosses[(left, right)] += float(np.einsum("ij,ij->", centered[left], centered[right]))
            elapsed_batches.append(time.perf_counter() - batch_started)
        ordered_rows = []
        for source in cfg["source_seeds"]:
            for target in cfg["source_seeds"]:
                if source == target:
                    continue
                cross = crosses[(min(source, target), max(source, target))]
                residual = max(0.0, energies[source] + energies[target] - 2.0 * cross)
                mean_residual = float(np.sum((mean_vectors[source] - mean_vectors[target]) ** 2))
                source_mean_energy = float(np.sum(mean_vectors[source] ** 2))
                ordered_rows.append({
                    "source_seed": source, "target_seed": target,
                    "source_centered_energy": energies[source], "target_centered_energy": energies[target], "centered_cross": cross,
                    "bcc": 2.0 * cross / (energies[source] + energies[target]),
                    "source_normalized_residual": residual / energies[source],
                    "symmetric_energy_normalized_residual": 2.0 * residual / (energies[source] + energies[target]),
                    "mean_source_normalized_residual": mean_residual / max(source_mean_energy, np.finfo(np.float64).tiny),
                })
        output_path = run_dir / "aggregate_identity.jsonl"
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_rows), encoding="utf-8")
        bcc = np.asarray([row["bcc"] for row in ordered_rows])
        residual = np.asarray([row["source_normalized_residual"] for row in ordered_rows])
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "five_seed_assets": set(matrices) == set(cfg["source_seeds"]),
            "complete_ordered_pair_grid": len(ordered_rows) == 20,
            "pair_symmetry": all(abs(next(r["bcc"] for r in ordered_rows if r["source_seed"] == a and r["target_seed"] == b) - next(r["bcc"] for r in ordered_rows if r["source_seed"] == b and r["target_seed"] == a)) < 1e-12 for a in cfg["source_seeds"] for b in cfg["source_seeds"] if a < b),
            "metrics_finite": bool(np.isfinite([list(row.values())[2:] for row in ordered_rows]).all()),
            "discovery_only": cfg["split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "independent_mean_constants": cfg["mean_constants_source_split"] == "mean",
            "no_found_decision": cfg["threshold_source_split"] == "none_aggregate_control_only",
            "audit_not_opened": not cfg["audit_opened"],
        }
        record = {
            "checks": checks, "ordered_pair_count": len(ordered_rows),
            "bcc_summary": {"min": float(np.min(bcc)), "median": float(np.median(bcc)), "max": float(np.max(bcc))},
            "source_normalized_residual_summary": {"min": float(np.min(residual)), "median": float(np.median(residual)), "max": float(np.max(residual))},
            "median_batch_seconds": float(np.median(elapsed_batches)), "batch_count": len(elapsed_batches),
            "output_sha256": sha256(output_path),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r010a_aggregate_reconstruction_identity.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
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
