"""Build discovery-only atom baseline rankings for frozen source queries."""
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

from scipy import __version__ as scipy_version  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def sparse_codes(asset_dir: Path, split: str, seed: int, tokens: int, k: int, latents: int) -> csr_matrix:
    base = asset_dir / split / f"seed_{seed}"
    indices = np.memmap(base / "top_indices.uint16.bin", dtype="<u2", mode="r", shape=(tokens, k))
    acts = np.memmap(base / "top_acts.float32.bin", dtype="<f4", mode="r", shape=(tokens, k))
    indptr = np.arange(0, (tokens + 1) * k, k, dtype=np.int64)
    writable_acts = np.asarray(acts, dtype=np.float32).reshape(-1).copy()
    writable_indices = np.asarray(indices, dtype=np.int32).reshape(-1).copy()
    matrix = csr_matrix((writable_acts, writable_indices, indptr), shape=(tokens, latents), copy=False)
    matrix.sort_indices()
    return matrix


def decoder(asset_dir: Path, seed: int, latents: int, hidden: int) -> np.ndarray:
    return np.asarray(np.memmap(asset_dir / "decoders" / f"seed_{seed}.float32.bin", dtype="<f4", mode="r", shape=(latents, hidden)))


def top_ids(values: np.ndarray, count: int, *, descending: bool) -> np.ndarray:
    atom_ids = np.arange(values.size)
    primary = -values if descending else values
    return np.lexsort((atom_ids, primary))[:count]


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
    panel_path = ROOT / cfg["query_panel_path"]
    census_path = ROOT / cfg["source_census_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(panel_path.resolve()), "sha256": sha256(panel_path), "bytes": panel_path.stat().st_size, "source": cfg["query_panel_run"], "license_or_access_boundary": "internal", "role": "frozen_query_panel"},
        {"path": str(census_path.resolve()), "sha256": sha256(census_path), "bytes": census_path.stat().st_size, "source": cfg["source_census_run"], "license_or_access_boundary": "internal", "role": "mean_and_source_statistics"},
        {"path": str(asset_manifest.resolve()), "sha256": sha256(asset_manifest), "bytes": asset_manifest.stat().st_size, "source": cfg["paired_codes_run"], "license_or_access_boundary": "internal", "role": "bulk_asset_hash_ledger"},
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
        "resource_lease_reason": "sparse cross-seed covariance and Hungarian ranking over 20 ordered pairs backed by D: assets",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    output_rows: list[dict] = []
    try:
        if sha256(panel_path) != cfg["query_panel_sha256"] or sha256(census_path) != cfg["source_census_sha256"]:
            raise ValueError("frozen query/census hash mismatch")
        if sha256(asset_manifest).lower() != cfg["asset_manifest_sha256"].lower():
            raise ValueError("paired asset manifest hash mismatch")
        panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line]
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        stats = {(row["seed"], row["atom"]): row for row in census}
        queries = {seed: sorted((row for row in panel if row["seed"] == seed), key=lambda row: row["atom"]) for seed in cfg["source_seeds"]}
        if any(len(rows) != cfg["queries_per_seed"] for rows in queries.values()):
            raise ValueError("query panel seed count mismatch")
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["discovery_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]) for seed in cfg["source_seeds"]}
        pair_summaries = []
        for source_seed in cfg["source_seeds"]:
            query_rows = queries[source_seed]
            query_ids = np.asarray([row["atom"] for row in query_rows], dtype=np.int64)
            zs = matrices[source_seed][:, query_ids]
            ds = decoders[source_seed][query_ids].astype(np.float64, copy=False)
            source_means = np.asarray([stats[(source_seed, int(atom))]["mean_code"] for atom in query_ids], dtype=np.float64)
            source_second = np.asarray([stats[(source_seed, int(atom))]["discovery_code_energy"] / cfg["discovery_tokens"] for atom in query_ids], dtype=np.float64)
            source_variance = np.maximum(0.0, source_second - source_means * source_means)
            source_decoder_norm_sq = np.einsum("ij,ij->i", ds, ds)
            source_energy = source_variance * source_decoder_norm_sq
            self_raw_code_cross = (zs.T @ matrices[source_seed]).toarray().astype(np.float64) / cfg["discovery_tokens"]
            self_code_covariance = self_raw_code_cross - source_means[:, None] * np.asarray(
                [stats[(source_seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64
            )[None, :]
            self_contribution_cross = self_code_covariance * (ds @ decoders[source_seed].astype(np.float64, copy=False).T)
            self_identity_cross = self_contribution_cross[np.arange(cfg["queries_per_seed"]), query_ids]
            if not np.allclose(self_identity_cross, source_energy, rtol=2e-5, atol=2e-7):
                raise ValueError(f"same-seed contribution identity failed for seed {source_seed}")
            for target_seed in cfg["source_seeds"]:
                if target_seed == source_seed:
                    continue
                pair_started = time.perf_counter()
                zt = matrices[target_seed]
                dt = decoders[target_seed].astype(np.float64, copy=False)
                target_means = np.asarray([stats[(target_seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
                target_second = np.asarray([stats[(target_seed, atom)]["discovery_code_energy"] / cfg["discovery_tokens"] for atom in range(cfg["num_latents"])], dtype=np.float64)
                target_variance = np.maximum(0.0, target_second - target_means * target_means)
                target_decoder_norm_sq = np.einsum("ij,ij->i", dt, dt)
                target_energy = target_variance * target_decoder_norm_sq
                raw_code_cross = (zs.T @ zt).toarray().astype(np.float64) / cfg["discovery_tokens"]
                code_covariance = raw_code_cross - source_means[:, None] * target_means[None, :]
                decoder_dot = ds @ dt.T
                contribution_cross = code_covariance * decoder_dot
                residual_sq = np.maximum(0.0, source_energy[:, None] + target_energy[None, :] - 2.0 * contribution_cross)
                source_normalized = residual_sq / np.maximum(source_energy[:, None], np.finfo(np.float64).tiny)
                balanced_denom = source_energy[:, None] + target_energy[None, :]
                bcc = np.divide(2.0 * contribution_cross, balanced_denom, out=np.full_like(contribution_cross, np.nan), where=balanced_denom > 0)
                decoder_norm = np.sqrt(source_decoder_norm_sq[:, None] * target_decoder_norm_sq[None, :])
                absolute_cosine = np.divide(np.abs(decoder_dot), decoder_norm, out=np.full_like(decoder_dot, -np.inf), where=decoder_norm > 0)
                mean_dot = (source_means[:, None] * target_means[None, :]) * decoder_dot
                source_mean_energy = source_means * source_means * source_decoder_norm_sq
                target_mean_energy = target_means * target_means * target_decoder_norm_sq
                mean_residual_sq = np.maximum(0.0, source_mean_energy[:, None] + target_mean_energy[None, :] - 2.0 * mean_dot)
                mean_source_normalized = mean_residual_sq / np.maximum(source_mean_energy[:, None], np.finfo(np.float64).tiny)
                hungarian_rows, hungarian_cols = linear_sum_assignment(-absolute_cosine)
                hungarian = {int(row): int(col) for row, col in zip(hungarian_rows, hungarian_cols)}
                nearest_targets = []
                balanced_targets = []
                for query_index, query in enumerate(query_rows):
                    contribution_ids = top_ids(source_normalized[query_index], cfg["candidate_top_k"], descending=False)
                    balanced_ids = top_ids(bcc[query_index], cfg["candidate_top_k"], descending=True)
                    cosine_ids = top_ids(absolute_cosine[query_index], cfg["candidate_top_k"], descending=True)
                    assigned = hungarian[query_index]
                    nearest_targets.append(int(contribution_ids[0]))
                    balanced_targets.append(int(balanced_ids[0]))
                    output_rows.append({
                        "source_seed": source_seed, "source_atom": int(query["atom"]), "energy_stratum": int(query["energy_stratum"]),
                        "target_seed": target_seed,
                        "contribution_candidate_ids": [int(value) for value in contribution_ids],
                        "contribution_source_normalized_residuals": [float(source_normalized[query_index, value]) for value in contribution_ids],
                        "contribution_bcc": [float(bcc[query_index, value]) for value in contribution_ids],
                        "contribution_mean_source_normalized_residuals": [float(mean_source_normalized[query_index, value]) for value in contribution_ids],
                        "balanced_candidate_ids": [int(value) for value in balanced_ids],
                        "balanced_bcc": [float(bcc[query_index, value]) for value in balanced_ids],
                        "balanced_source_normalized_residuals": [float(source_normalized[query_index, value]) for value in balanced_ids],
                        "balanced_mean_source_normalized_residuals": [float(mean_source_normalized[query_index, value]) for value in balanced_ids],
                        "cosine_candidate_ids": [int(value) for value in cosine_ids],
                        "cosine_scores": [float(absolute_cosine[query_index, value]) for value in cosine_ids],
                        "hungarian_target_atom": assigned,
                        "hungarian_absolute_cosine": float(absolute_cosine[query_index, assigned]),
                        "hungarian_source_normalized_residual": float(source_normalized[query_index, assigned]),
                        "hungarian_bcc": float(bcc[query_index, assigned]),
                        "hungarian_mean_source_normalized_residual": float(mean_source_normalized[query_index, assigned]),
                        "discovery_source_energy": float(source_energy[query_index]),
                        "target_universe_size": cfg["num_latents"],
                        "candidate_top_k": cfg["candidate_top_k"],
                    })
                pair_rows = output_rows[-cfg["queries_per_seed"] :]
                pair_summaries.append({
                    "source_seed": source_seed, "target_seed": target_seed, "queries": len(pair_rows),
                    "median_best_contribution_residual": float(np.median([row["contribution_source_normalized_residuals"][0] for row in pair_rows])),
                    "median_best_contribution_bcc": float(np.median([row["contribution_bcc"][0] for row in pair_rows])),
                    "median_best_balanced_bcc": float(np.median([row["balanced_bcc"][0] for row in pair_rows])),
                    "median_balanced_source_normalized_residual": float(np.median([row["balanced_source_normalized_residuals"][0] for row in pair_rows])),
                    "mean_hungarian_cosine": float(np.mean([row["hungarian_absolute_cosine"] for row in pair_rows])),
                    "median_hungarian_contribution_residual": float(np.median([row["hungarian_source_normalized_residual"] for row in pair_rows])),
                    "nearest_target_unique_fraction": len(set(nearest_targets)) / len(nearest_targets),
                    "balanced_target_unique_fraction": len(set(balanced_targets)) / len(balanced_targets),
                    "elapsed_seconds": time.perf_counter() - pair_started,
                })
        candidates_path = run_dir / "atom_discovery_candidates.jsonl"
        with candidates_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        expected_rows = len(cfg["source_seeds"]) * (len(cfg["source_seeds"]) - 1) * cfg["queries_per_seed"]
        checks = {
            "query_panel_hash_bound": sha256(panel_path) == cfg["query_panel_sha256"],
            "asset_manifest_hash_bound": sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower(),
            "complete_ordered_pair_grid": len(output_rows) == expected_rows == 2560,
            "unique_query_pair_rows": len({(row["source_seed"], row["source_atom"], row["target_seed"]) for row in output_rows}) == len(output_rows),
            "candidate_budget_exact": all(len(row["contribution_candidate_ids"]) == len(row["balanced_candidate_ids"]) == len(row["cosine_candidate_ids"]) == cfg["candidate_top_k"] for row in output_rows),
            "hungarian_injective_per_pair": all(len({row["hungarian_target_atom"] for row in output_rows if row["source_seed"] == s and row["target_seed"] == t}) == cfg["queries_per_seed"] for s in cfg["source_seeds"] for t in cfg["source_seeds"] if s != t),
            "metrics_finite": all(np.isfinite(row["hungarian_source_normalized_residual"]) and np.isfinite(row["contribution_source_normalized_residuals"]).all() for row in output_rows),
            "discovery_only": cfg["split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "no_found_decision": cfg["threshold_source_split"] == "none_discovery_rankings_only",
            "audit_not_opened": not cfg["audit_opened"],
        }
        record = {
            "checks": checks, "row_count": len(output_rows), "ordered_pair_count": len(pair_summaries), "pair_summaries": pair_summaries,
            "overall_median_best_contribution_residual": float(np.median([row["contribution_source_normalized_residuals"][0] for row in output_rows])),
            "overall_median_best_contribution_bcc": float(np.median([row["contribution_bcc"][0] for row in output_rows])),
            "overall_median_best_balanced_bcc": float(np.median([row["balanced_bcc"][0] for row in output_rows])),
            "overall_median_balanced_source_normalized_residual": float(np.median([row["balanced_source_normalized_residuals"][0] for row in output_rows])),
            "overall_mean_hungarian_cosine": float(np.mean([row["hungarian_absolute_cosine"] for row in output_rows])),
            "overall_median_hungarian_contribution_residual": float(np.median([row["hungarian_source_normalized_residual"] for row in output_rows])),
            "atom_candidates_sha256": sha256(candidates_path),
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
        "generator_script_path": "scripts/run_r009c_atom_discovery.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
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
