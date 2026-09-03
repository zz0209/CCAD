"""Compute a discovery-only exact local support surface for real SAE queries."""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
from run_r009c_atom_discovery import decoder, sparse_codes, top_ids  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def ordered_unique(groups: list[np.ndarray], cap: int) -> list[int]:
    result: list[int] = []
    for group in groups:
        for value in group:
            atom = int(value)
            if atom not in result:
                result.append(atom)
                if len(result) == cap:
                    return result
    return result


def support_metrics(combos: np.ndarray, ktt: np.ndarray, kst: np.ndarray, mtt: np.ndarray, mst: np.ndarray, source_energy: float, source_mean_energy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_energy = np.sum(np.diag(ktt)[combos], axis=1)
    target_mean_energy = np.sum(np.diag(mtt)[combos], axis=1)
    for left in range(combos.shape[1]):
        for right in range(left + 1, combos.shape[1]):
            target_energy += 2.0 * ktt[combos[:, left], combos[:, right]]
            target_mean_energy += 2.0 * mtt[combos[:, left], combos[:, right]]
    cross = np.sum(kst[combos], axis=1)
    mean_cross = np.sum(mst[combos], axis=1)
    residual = np.maximum(0.0, source_energy + target_energy - 2.0 * cross)
    mean_residual = np.maximum(0.0, source_mean_energy + target_mean_energy - 2.0 * mean_cross)
    d_ctr = residual / max(source_energy, np.finfo(np.float64).tiny)
    d_mu = mean_residual / max(source_mean_energy, np.finfo(np.float64).tiny)
    bcc = np.divide(2.0 * cross, source_energy + target_energy, out=np.full_like(cross, np.nan), where=(source_energy + target_energy) > 0)
    return d_ctr, d_mu, bcc


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
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    panel_path = ROOT / cfg["query_panel_path"]
    census_path = ROOT / cfg["source_census_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(panel_path.resolve()), "sha256": sha256(panel_path), "bytes": panel_path.stat().st_size, "source": cfg["query_panel_run"], "license_or_access_boundary": "internal", "role": "frozen_query_panel"},
        {"path": str(census_path.resolve()), "sha256": sha256(census_path), "bytes": census_path.stat().st_size, "source": "R009a", "license_or_access_boundary": "internal", "role": "mean_and_energy_statistics"},
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
        "resource_lease_reason": "exact 6,195-support discovery surface for 2,560 real-SAE query pairs backed by D: sparse codes",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    output_rows: list[dict] = []
    try:
        if sha256(panel_path) != cfg["query_panel_sha256"] or sha256(census_path) != cfg["source_census_sha256"] or sha256(asset_manifest).lower() != cfg["asset_manifest_sha256"].lower():
            raise ValueError("frozen input hash mismatch")
        if sum(len(list(itertools.combinations(range(cfg["proposal_atom_cap"]), size))) for size in range(1, cfg["g_max"] + 1)) != cfg["support_family_count"]:
            raise ValueError("support family count mismatch")
        if cfg["support_family_count"] > cfg["candidate_evaluation_budget"]:
            raise ValueError("support family exceeds budget")
        combos = {size: np.asarray(list(itertools.combinations(range(cfg["proposal_atom_cap"]), size)), dtype=np.int16) for size in range(1, cfg["g_max"] + 1)}
        panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line]
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        stats = {(row["seed"], row["atom"]): row for row in census}
        queries = {seed: sorted((row for row in panel if row["seed"] == seed), key=lambda row: row["atom"]) for seed in cfg["source_seeds"]}
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["discovery_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        pair_summaries = []
        for source_seed in cfg["source_seeds"]:
            query_rows = queries[source_seed]
            query_ids = np.asarray([row["atom"] for row in query_rows], dtype=np.int64)
            zs = matrices[source_seed][:, query_ids]
            ds = decoders[source_seed][query_ids]
            source_means = np.asarray([stats[(source_seed, int(atom))]["mean_code"] for atom in query_ids], dtype=np.float64)
            source_variance = np.maximum(0.0, np.asarray([stats[(source_seed, int(atom))]["discovery_code_energy"] / cfg["discovery_tokens"] for atom in query_ids]) - source_means**2)
            source_norm_sq = np.einsum("ij,ij->i", ds, ds)
            source_energy = source_variance * source_norm_sq
            source_mean_vectors = source_means[:, None] * ds
            source_mean_energy = np.einsum("ij,ij->i", source_mean_vectors, source_mean_vectors)
            for target_seed in cfg["source_seeds"]:
                if source_seed == target_seed:
                    continue
                pair_started = time.perf_counter()
                zt = matrices[target_seed]
                dt = decoders[target_seed]
                target_means = np.asarray([stats[(target_seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
                target_variance = np.maximum(0.0, np.asarray([stats[(target_seed, atom)]["discovery_code_energy"] / cfg["discovery_tokens"] for atom in range(cfg["num_latents"])]) - target_means**2)
                target_norm_sq = np.einsum("ij,ij->i", dt, dt)
                target_energy = target_variance * target_norm_sq
                raw_cross = (zs.T @ zt).toarray().astype(np.float64) / cfg["discovery_tokens"]
                code_cov = raw_cross - source_means[:, None] * target_means[None, :]
                decoder_dot = ds @ dt.T
                cross = code_cov * decoder_dot
                bcc = np.divide(2.0 * cross, source_energy[:, None] + target_energy[None, :], out=np.full_like(cross, -np.inf), where=(source_energy[:, None] + target_energy[None, :]) > 0)
                rho = np.divide(cross, np.sqrt(source_energy[:, None] * target_energy[None, :]), out=np.full_like(cross, -np.inf), where=(source_energy[:, None] * target_energy[None, :]) > 0)
                cosine = np.divide(np.abs(decoder_dot), np.sqrt(source_norm_sq[:, None] * target_norm_sq[None, :]), out=np.full_like(decoder_dot, -np.inf), where=(source_norm_sq[:, None] * target_norm_sq[None, :]) > 0)
                for query_index, query in enumerate(query_rows):
                    rule = cfg["proposal_union_rule"]
                    proposal = ordered_unique([
                        top_ids(rho[query_index], rule["positive_contribution_correlation"], descending=True),
                        top_ids(bcc[query_index], rule["balanced_bcc"], descending=True),
                        top_ids(cosine[query_index], rule["absolute_decoder_cosine"], descending=True),
                        top_ids(rho[query_index], cfg["num_latents"], descending=True),
                        top_ids(bcc[query_index], cfg["num_latents"], descending=True),
                    ], cfg["proposal_atom_cap"])
                    if len(proposal) != cfg["proposal_atom_cap"]:
                        raise ValueError("proposal backfill failed")
                    ids = np.asarray(proposal, dtype=np.int64)
                    zsub = zt[:, ids]
                    code_tt = (zsub.T @ zsub).toarray().astype(np.float64) / cfg["discovery_tokens"] - np.outer(target_means[ids], target_means[ids])
                    dsub = dt[ids]
                    ktt = code_tt * (dsub @ dsub.T)
                    kst = cross[query_index, ids]
                    mean_vectors = target_means[ids, None] * dsub
                    mtt = mean_vectors @ mean_vectors.T
                    mst = source_mean_vectors[query_index] @ mean_vectors.T
                    by_size = []
                    all_rows = []
                    for size, local_combos in combos.items():
                        d_ctr, d_mu, group_bcc = support_metrics(local_combos, ktt, kst, mtt, mst, source_energy[query_index], source_mean_energy[query_index])
                        best_ctr_index = int(np.lexsort((np.arange(len(d_ctr)), d_ctr))[0])
                        best_bcc_index = int(np.lexsort((np.arange(len(group_bcc)), -group_bcc))[0])
                        by_size.append({
                            "size": size,
                            "best_residual_support": [proposal[int(v)] for v in local_combos[best_ctr_index]],
                            "best_source_normalized_residual": float(d_ctr[best_ctr_index]),
                            "best_residual_bcc": float(group_bcc[best_ctr_index]),
                            "best_residual_d_mu": float(d_mu[best_ctr_index]),
                            "best_bcc_support": [proposal[int(v)] for v in local_combos[best_bcc_index]],
                            "best_bcc": float(group_bcc[best_bcc_index]),
                            "best_bcc_source_normalized_residual": float(d_ctr[best_bcc_index]),
                            "best_bcc_d_mu": float(d_mu[best_bcc_index]),
                        })
                        all_rows.extend((float(d_ctr[i]), size, tuple(proposal[int(v)] for v in local_combos[i]), float(d_mu[i]), float(group_bcc[i])) for i in range(len(local_combos)))
                    top_supports = sorted(all_rows, key=lambda item: (item[0], item[1], item[2]))[: cfg["saved_top_supports_per_ranking"]]
                    output_rows.append({
                        "source_seed": source_seed, "source_atom": int(query["atom"]), "energy_stratum": int(query["energy_stratum"]), "target_seed": target_seed,
                        "proposal_target_ids": proposal, "proposal_atom_cap": cfg["proposal_atom_cap"], "support_family_count": cfg["support_family_count"],
                        "best_by_size": by_size,
                        "top_residual_supports": [{"support": list(item[2]), "size": item[1], "d_ctr": item[0], "d_mu": item[3], "bcc": item[4]} for item in top_supports],
                    })
                pair_rows = output_rows[-cfg["queries_per_seed"] :]
                pair_summaries.append({
                    "source_seed": source_seed, "target_seed": target_seed, "queries": len(pair_rows),
                    "median_best_d_ctr_size1": float(np.median([row["best_by_size"][0]["best_source_normalized_residual"] for row in pair_rows])),
                    "median_best_d_ctr_size4": float(np.median([row["best_by_size"][3]["best_source_normalized_residual"] for row in pair_rows])),
                    "median_best_bcc_size1": float(np.median([row["best_by_size"][0]["best_bcc"] for row in pair_rows])),
                    "median_best_bcc_size4": float(np.median([row["best_by_size"][3]["best_bcc"] for row in pair_rows])),
                    "elapsed_seconds": time.perf_counter() - pair_started,
                })
        output_path = run_dir / "group_discovery_surface.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        expected = len(cfg["source_seeds"]) * (len(cfg["source_seeds"]) - 1) * cfg["queries_per_seed"]
        checks = {
            "frozen_inputs_bound": sha256(panel_path) == cfg["query_panel_sha256"] and sha256(census_path) == cfg["source_census_sha256"],
            "complete_ordered_pair_grid": len(output_rows) == expected == 2560,
            "unique_rows": len({(row["source_seed"], row["source_atom"], row["target_seed"]) for row in output_rows}) == len(output_rows),
            "proposal_cap_exact": all(len(row["proposal_target_ids"]) == len(set(row["proposal_target_ids"])) == cfg["proposal_atom_cap"] for row in output_rows),
            "support_budget_exact": all(row["support_family_count"] == cfg["support_family_count"] <= cfg["candidate_evaluation_budget"] for row in output_rows),
            "all_sizes_reported": all([item["size"] for item in row["best_by_size"]] == [1, 2, 3, 4] for row in output_rows),
            "metrics_finite": all(np.isfinite([item["best_source_normalized_residual"] for item in row["best_by_size"]]).all() for row in output_rows),
            "discovery_only": cfg["split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "no_found_decision": cfg["threshold_source_split"] == "none_discovery_surface_only",
            "audit_not_opened": not cfg["audit_opened"],
        }
        record = {
            "checks": checks, "row_count": len(output_rows), "pair_summaries": pair_summaries,
            "overall_median_best_d_ctr_by_size": {str(size): float(np.median([row["best_by_size"][size - 1]["best_source_normalized_residual"] for row in output_rows])) for size in range(1, 5)},
            "overall_median_best_bcc_by_size": {str(size): float(np.median([row["best_by_size"][size - 1]["best_bcc"] for row in output_rows])) for size in range(1, 5)},
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
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011a_group_discovery_surface.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
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
