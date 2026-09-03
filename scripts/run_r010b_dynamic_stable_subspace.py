"""Compute dynamic whole-reconstruction covariance subspace stability across SAE seeds."""
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
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    aggregate_path = ROOT / cfg["aggregate_identity_path"]
    census_path = ROOT / cfg["source_census_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(aggregate_path.resolve()), "sha256": sha256(aggregate_path), "bytes": aggregate_path.stat().st_size, "source": cfg["aggregate_identity_run"], "license_or_access_boundary": "internal", "role": "aggregate_identity_anchor"},
        {"path": str(census_path.resolve()), "sha256": sha256(census_path), "bytes": census_path.stat().st_size, "source": "R009a", "license_or_access_boundary": "internal", "role": "independent_mean_constants"},
        {"path": str(asset_manifest.resolve()), "sha256": sha256(asset_manifest), "bytes": asset_manifest.stat().st_size, "source": "R008b", "license_or_access_boundary": "internal", "role": "bulk_asset_hash_ledger"},
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
        "seeds": cfg["source_seeds"], "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "stream full discovery sparse codes and accumulate five 768x768 dynamic reconstruction covariance matrices",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {
            "aggregate_identity": sha256(aggregate_path).lower() == cfg["aggregate_identity_sha256"].lower(),
            "census": sha256(census_path).lower() == cfg["source_census_sha256"].lower(),
            "asset_manifest": sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower(),
        }
        if not all(bound.values()):
            raise ValueError(f"frozen input hash mismatch: {bound}")
        if sorted(set(cfg["subspace_ranks"])) != cfg["subspace_ranks"] or max(cfg["subspace_ranks"]) >= cfg["hook_hidden_size"]:
            raise ValueError("subspace ranks must be unique, sorted, and below hook dimension")
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        stats = {(row["seed"], row["atom"]): row for row in census}
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["discovery_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float32, copy=False) for seed in cfg["source_seeds"]}
        independent_means, discovery_means, covariances = {}, {}, {}
        for seed in cfg["source_seeds"]:
            code_mean = np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
            discovery_code_mean = np.asarray([stats[(seed, atom)]["discovery_signed_code_mean"] for atom in range(cfg["num_latents"])], dtype=np.float64)
            independent_means[seed] = code_mean @ decoders[seed].astype(np.float64)
            discovery_means[seed] = discovery_code_mean @ decoders[seed].astype(np.float64)
            covariances[seed] = np.zeros((cfg["hook_hidden_size"], cfg["hook_hidden_size"]), dtype=np.float64)
        batch_seconds = []
        for start in range(0, cfg["discovery_tokens"], cfg["batch_tokens"]):
            batch_started = time.perf_counter()
            stop = min(start + cfg["batch_tokens"], cfg["discovery_tokens"])
            for seed in cfg["source_seeds"]:
                reconstruction = np.asarray(matrices[seed][start:stop] @ decoders[seed], dtype=np.float64)
                centered = reconstruction - independent_means[seed][None, :]
                covariances[seed] += centered.T @ centered
            batch_seconds.append(time.perf_counter() - batch_started)
        centerings = cfg.get("reported_centerings", ["independent_mean"])
        spectra = {centering: {} for centering in centerings}
        bases = {centering: {} for centering in centerings}
        seed_rows = []
        for seed in cfg["source_seeds"]:
            independent_covariance = 0.5 * (covariances[seed] + covariances[seed].T) / cfg["discovery_tokens"]
            displacement = discovery_means[seed] - independent_means[seed]
            covariance_by_centering = {"independent_mean": independent_covariance}
            if "within_discovery_empirical_mean" in centerings:
                covariance_by_centering["within_discovery_empirical_mean"] = independent_covariance - np.outer(displacement, displacement)
            centering_rows = {}
            for centering in centerings:
                covariance = 0.5 * (covariance_by_centering[centering] + covariance_by_centering[centering].T)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues = np.maximum(0.0, eigenvalues[order])
                eigenvectors = eigenvectors[:, order]
                spectra[centering][seed], bases[centering][seed] = eigenvalues, eigenvectors
                total = float(np.sum(eigenvalues))
                probabilities = eigenvalues[eigenvalues > 0] / total
                centering_rows[centering] = {
                    "trace": total,
                    "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
                    "retained_numerical_rank": int(np.sum(eigenvalues > cfg["eigenvalue_relative_tolerance"] * eigenvalues[0])),
                    "variance_coverage_by_rank": {str(rank): float(np.sum(eigenvalues[:rank]) / total) for rank in cfg["subspace_ranks"]},
                }
            seed_rows.append({
                "seed": seed,
                "independent_to_discovery_mean_shift_energy": float(displacement @ displacement),
                "mean_shift_fraction_of_independent_trace": float((displacement @ displacement) / np.trace(independent_covariance)),
                "by_centering": centering_rows,
            })
        pair_rows = []
        for left_index, left in enumerate(cfg["source_seeds"]):
            for right in cfg["source_seeds"][left_index + 1:]:
                by_centering = {}
                for centering in centerings:
                    rank_rows = []
                    for rank in cfg["subspace_ranks"]:
                        singular_values = np.linalg.svd(bases[centering][left][:, :rank].T @ bases[centering][right][:, :rank], compute_uv=False)
                        psc = float(np.sum(singular_values**2) / rank)
                        random_expectation = rank / cfg["hook_hidden_size"]
                        rank_rows.append({
                            "rank": rank, "psc": psc, "random_isotropic_expectation": random_expectation,
                            "psc_excess_over_random": psc - random_expectation,
                            "principal_cosine_min": float(np.min(singular_values)),
                            "principal_cosine_median": float(np.median(singular_values)),
                            "principal_cosine_max": float(np.max(singular_values)),
                        })
                    by_centering[centering] = rank_rows
                pair_rows.append({"left_seed": left, "right_seed": right, "by_centering": by_centering})
        output_path = run_dir / "dynamic_stable_subspace.json"
        write_json(output_path, {"seed_spectra": seed_rows, "pair_subspaces": pair_rows})
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "five_seed_covariances": len(seed_rows) == 5,
            "complete_unordered_pair_grid": len(pair_rows) == 10,
            "all_centerings_and_ranks_reported": all(set(pair["by_centering"]) == set(centerings) and all([row["rank"] for row in pair["by_centering"][centering]] == cfg["subspace_ranks"] for centering in centerings) for pair in pair_rows),
            "psc_in_unit_interval": all(0.0 <= row["psc"] <= 1.0 + 1e-10 for pair in pair_rows for centering in centerings for row in pair["by_centering"][centering]),
            "spectra_finite": bool(np.isfinite([row["by_centering"][centering]["effective_rank"] for row in seed_rows for centering in centerings]).all()),
            "mean_shift_fraction_valid": all(0.0 <= row["mean_shift_fraction_of_independent_trace"] <= 1.0 + 1e-8 for row in seed_rows),
            "discovery_only": cfg["split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "independent_mean_constants": cfg["mean_constants_source_split"] == "mean",
            "no_found_decision": cfg["threshold_source_split"] == "none_stable_subspace_baseline_only",
            "audit_not_opened": not cfg["audit_opened"],
        }
        rank_summary = {centering: {} for centering in centerings}
        for centering in centerings:
            for rank_index, rank in enumerate(cfg["subspace_ranks"]):
                values = np.asarray([pair["by_centering"][centering][rank_index]["psc"] for pair in pair_rows])
                coverage = np.asarray([row["by_centering"][centering]["variance_coverage_by_rank"][str(rank)] for row in seed_rows])
                rank_summary[centering][str(rank)] = {
                    "psc_min": float(np.min(values)), "psc_median": float(np.median(values)), "psc_max": float(np.max(values)),
                    "variance_coverage_min": float(np.min(coverage)), "variance_coverage_median": float(np.median(coverage)), "variance_coverage_max": float(np.max(coverage)),
                    "random_isotropic_expectation": rank / cfg["hook_hidden_size"],
                }
        record = {
            "checks": checks, "seed_count": len(seed_rows), "unordered_pair_count": len(pair_rows),
            "effective_rank_summary_by_centering": {centering: {"min": float(min(r["by_centering"][centering]["effective_rank"] for r in seed_rows)), "median": float(np.median([r["by_centering"][centering]["effective_rank"] for r in seed_rows])), "max": float(max(r["by_centering"][centering]["effective_rank"] for r in seed_rows))} for centering in centerings},
            "mean_shift_fraction_summary": {"min": float(min(r["mean_shift_fraction_of_independent_trace"] for r in seed_rows)), "median": float(np.median([r["mean_shift_fraction_of_independent_trace"] for r in seed_rows])), "max": float(max(r["mean_shift_fraction_of_independent_trace"] for r in seed_rows))},
            "rank_summary_by_centering": rank_summary, "median_batch_seconds": float(np.median(batch_seconds)), "batch_count": len(batch_seconds),
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
        "generator_script_path": "scripts/run_r010b_dynamic_stable_subspace.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
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
