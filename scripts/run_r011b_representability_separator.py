"""Run the bounded R011b coefficient/representability separator on R011a families."""
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
from scipy.optimize import minimize  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def quadratic_metrics(
    coefficients: np.ndarray,
    ktt: np.ndarray,
    kst: np.ndarray,
    mtt: np.ndarray,
    mst: np.ndarray,
    source_energy: float,
    source_mean_energy: float,
) -> dict[str, float]:
    target_energy = float(coefficients @ ktt @ coefficients)
    cross = float(coefficients @ kst)
    residual = max(0.0, source_energy + target_energy - 2.0 * cross)
    target_mean_energy = float(coefficients @ mtt @ coefficients)
    mean_cross = float(coefficients @ mst)
    mean_residual = max(0.0, source_mean_energy + target_mean_energy - 2.0 * mean_cross)
    tiny = np.finfo(np.float64).tiny
    return {
        "d_ctr": residual / max(source_energy, tiny),
        "d_mu": mean_residual / max(source_mean_energy, tiny),
        "bcc": (2.0 * cross / (source_energy + target_energy)) if source_energy + target_energy > 0 else float("nan"),
        "target_energy_ratio": target_energy / max(source_energy, tiny),
    }


def solve_signed(ktt: np.ndarray, kst: np.ndarray, rcond: float) -> tuple[np.ndarray, int, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(ktt)
    scale = max(float(np.max(eigenvalues)), np.finfo(np.float64).tiny)
    keep = eigenvalues > rcond * scale
    coefficients = np.zeros_like(kst)
    if np.any(keep):
        projected = eigenvectors[:, keep].T @ kst
        coefficients = eigenvectors[:, keep] @ (projected / eigenvalues[keep])
    positive = eigenvalues[eigenvalues > rcond * scale]
    condition = float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    return coefficients, int(np.sum(keep)), condition


def solve_omp(ktt: np.ndarray, kst: np.ndarray, max_support: int, rcond: float) -> list[tuple[list[int], np.ndarray]]:
    selected: list[int] = []
    coefficients = np.empty(0, dtype=np.float64)
    results: list[tuple[list[int], np.ndarray]] = []
    diagonal = np.maximum(np.diag(ktt), np.finfo(np.float64).tiny)
    for _ in range(max_support):
        residual_correlation = kst.copy()
        if selected:
            residual_correlation -= ktt[:, selected] @ coefficients
        scores = np.abs(residual_correlation) / np.sqrt(diagonal)
        scores[selected] = -np.inf
        chosen = int(np.lexsort((np.arange(len(scores)), -scores))[0])
        selected.append(chosen)
        kss = ktt[np.ix_(selected, selected)]
        coefficients, _, _ = solve_signed(kss, kst[selected], rcond)
        results.append((selected.copy(), coefficients.copy()))
    return results


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

    surface_path = ROOT / cfg["group_surface_path"]
    panel_path = ROOT / cfg["query_panel_path"]
    census_path = ROOT / cfg["source_census_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(surface_path.resolve()), "sha256": sha256(surface_path), "bytes": surface_path.stat().st_size, "source": cfg["group_surface_run"], "license_or_access_boundary": "internal", "role": "frozen_local_candidate_families"},
        {"path": str(panel_path.resolve()), "sha256": sha256(panel_path), "bytes": panel_path.stat().st_size, "source": "R009b", "license_or_access_boundary": "internal", "role": "frozen_query_panel"},
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
        "resource_lease_reason": "full 2,560-row coefficient/representability separator on fixed R011a local families",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})

    record, error, status = None, None, "FAIL"
    output_rows: list[dict] = []
    try:
        bound = {
            "surface": sha256(surface_path).lower() == cfg["group_surface_sha256"].lower(),
            "panel": sha256(panel_path).lower() == cfg["query_panel_sha256"].lower(),
            "census": sha256(census_path).lower() == cfg["source_census_sha256"].lower(),
            "asset_manifest": sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower(),
        }
        if not all(bound.values()):
            raise ValueError(f"frozen input hash mismatch: {bound}")
        surface = [json.loads(line) for line in surface_path.read_text(encoding="utf-8").splitlines() if line]
        panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line]
        census = [json.loads(line) for line in census_path.read_text(encoding="utf-8").splitlines() if line]
        stats = {(row["seed"], row["atom"]): row for row in census}
        queries = {seed: sorted((row for row in panel if row["seed"] == seed), key=lambda row: row["atom"]) for seed in cfg["source_seeds"]}
        surface_map = {(row["source_seed"], row["source_atom"], row["target_seed"]): row for row in surface}
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["discovery_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        pair_summaries: list[dict] = []
        solver_failures = 0

        for source_seed in cfg["source_seeds"]:
            query_rows = queries[source_seed]
            query_ids = np.asarray([row["atom"] for row in query_rows], dtype=np.int64)
            zs = matrices[source_seed][:, query_ids]
            ds = decoders[source_seed][query_ids]
            source_means = np.asarray([stats[(source_seed, int(atom))]["mean_code"] for atom in query_ids], dtype=np.float64)
            source_variance = np.maximum(0.0, np.asarray([stats[(source_seed, int(atom))]["discovery_code_energy"] / cfg["discovery_tokens"] for atom in query_ids]) - source_means**2)
            source_energy = source_variance * np.einsum("ij,ij->i", ds, ds)
            source_mean_vectors = source_means[:, None] * ds
            source_mean_energy = np.einsum("ij,ij->i", source_mean_vectors, source_mean_vectors)

            for target_seed in cfg["source_seeds"]:
                if source_seed == target_seed:
                    continue
                pair_started = time.perf_counter()
                zt = matrices[target_seed]
                dt = decoders[target_seed]
                target_means = np.asarray([stats[(target_seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
                raw_cross = (zs.T @ zt).toarray().astype(np.float64) / cfg["discovery_tokens"]
                code_cov = raw_cross - source_means[:, None] * target_means[None, :]
                cross = code_cov * (ds @ dt.T)

                pair_rows: list[dict] = []
                for query_index, query in enumerate(query_rows):
                    surface_row = surface_map[(source_seed, int(query["atom"]), target_seed)]
                    proposal = surface_row["proposal_target_ids"]
                    if len(proposal) != len(set(proposal)) or len(proposal) != cfg["proposal_atom_cap"]:
                        raise ValueError("invalid frozen proposal")
                    ids = np.asarray(proposal, dtype=np.int64)
                    zsub = zt[:, ids]
                    code_tt = (zsub.T @ zsub).toarray().astype(np.float64) / cfg["discovery_tokens"] - np.outer(target_means[ids], target_means[ids])
                    dsub = dt[ids]
                    ktt = code_tt * (dsub @ dsub.T)
                    ktt = 0.5 * (ktt + ktt.T)
                    kst = cross[query_index, ids]
                    mean_vectors = target_means[ids, None] * dsub
                    mtt = mean_vectors @ mean_vectors.T
                    mst = source_mean_vectors[query_index] @ mean_vectors.T
                    es = float(source_energy[query_index])
                    em = float(source_mean_energy[query_index])

                    diagonal = np.diag(ktt)
                    scaled_scores = np.divide(kst, diagonal, out=np.zeros_like(kst), where=diagonal > np.finfo(np.float64).tiny)
                    scaled_metrics = [quadratic_metrics(np.eye(len(ids))[i] * scaled_scores[i], ktt, kst, mtt, mst, es, em) for i in range(len(ids))]
                    scaled_index = min(range(len(ids)), key=lambda i: (scaled_metrics[i]["d_ctr"], proposal[i]))
                    scaled_single = {
                        "support": [proposal[scaled_index]], "coefficients": [float(scaled_scores[scaled_index])],
                        **scaled_metrics[scaled_index], "semantics": cfg["continuous_output_semantics"],
                    }

                    omp_rows = []
                    for selected, coefficients in solve_omp(ktt, kst, cfg["omp_max_support"], cfg["solver_rcond"]):
                        full = np.zeros(len(ids), dtype=np.float64)
                        full[selected] = coefficients
                        omp_rows.append({
                            "size": len(selected), "support": [proposal[i] for i in selected], "coefficients": coefficients.tolist(),
                            **quadratic_metrics(full, ktt, kst, mtt, mst, es, em), "semantics": cfg["continuous_output_semantics"],
                        })

                    signed_coefficients, effective_rank, condition = solve_signed(ktt, kst, cfg["solver_rcond"])
                    signed_full = {
                        "support": proposal, "coefficients": signed_coefficients.tolist(), "effective_rank": effective_rank,
                        "condition_number_retained": condition, "negative_coefficient_fraction": float(np.mean(signed_coefficients < -1e-12)),
                        **quadratic_metrics(signed_coefficients, ktt, kst, mtt, mst, es, em), "semantics": cfg["continuous_output_semantics"],
                    }

                    objective = lambda c: float(c @ ktt @ c - 2.0 * c @ kst)
                    gradient = lambda c: 2.0 * (ktt @ c - kst)
                    initial = np.maximum(0.0, signed_coefficients)
                    nonnegative_result = minimize(
                        objective, initial, jac=gradient, method="L-BFGS-B", bounds=[(0.0, None)] * len(ids),
                        options={"maxiter": cfg["nonnegative_max_iterations"], "gtol": cfg["nonnegative_gradient_tolerance"], "ftol": 1e-14},
                    )
                    if not nonnegative_result.success:
                        solver_failures += 1
                    nonnegative_coefficients = np.maximum(0.0, np.asarray(nonnegative_result.x, dtype=np.float64))
                    nonnegative_full = {
                        "support": proposal, "coefficients": nonnegative_coefficients.tolist(), "solver_success": bool(nonnegative_result.success),
                        "solver_status": int(nonnegative_result.status), "solver_iterations": int(nonnegative_result.nit),
                        "active_coefficient_count": int(np.sum(nonnegative_coefficients > 1e-8)),
                        **quadratic_metrics(nonnegative_coefficients, ktt, kst, mtt, mst, es, em), "semantics": cfg["continuous_output_semantics"],
                    }

                    unweighted = surface_row["best_by_size"][cfg["omp_max_support"] - 1]
                    row = {
                        "source_seed": source_seed, "source_atom": int(query["atom"]), "energy_stratum": int(query["energy_stratum"]),
                        "target_seed": target_seed, "proposal_target_ids": proposal,
                        "unweighted_native_size4": {
                            "support": unweighted["best_residual_support"], "coefficients": [1.0] * cfg["omp_max_support"],
                            "d_ctr": unweighted["best_source_normalized_residual"], "d_mu": unweighted["best_residual_d_mu"],
                            "bcc": unweighted["best_residual_bcc"], "semantics": "native_support_reference",
                        },
                        "scaled_single": scaled_single, "signed_omp": omp_rows,
                        "nonnegative_full20": nonnegative_full, "signed_full20": signed_full,
                    }
                    output_rows.append(row)
                    pair_rows.append(row)
                pair_summaries.append({
                    "source_seed": source_seed, "target_seed": target_seed, "queries": len(pair_rows),
                    "median_d_ctr_unweighted_size4": float(np.median([r["unweighted_native_size4"]["d_ctr"] for r in pair_rows])),
                    "median_d_ctr_scaled_single": float(np.median([r["scaled_single"]["d_ctr"] for r in pair_rows])),
                    "median_d_ctr_omp4": float(np.median([r["signed_omp"][3]["d_ctr"] for r in pair_rows])),
                    "median_d_ctr_nonnegative_full20": float(np.median([r["nonnegative_full20"]["d_ctr"] for r in pair_rows])),
                    "median_d_ctr_signed_full20": float(np.median([r["signed_full20"]["d_ctr"] for r in pair_rows])),
                    "elapsed_seconds": time.perf_counter() - pair_started,
                })

        output_path = run_dir / "representability_separator.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        expected = len(cfg["source_seeds"]) * (len(cfg["source_seeds"]) - 1) * cfg["queries_per_seed"]
        omp_monotone = all(all(row["signed_omp"][i + 1]["d_ctr"] <= row["signed_omp"][i]["d_ctr"] + 1e-8 for i in range(3)) for row in output_rows)
        dominance = all(
            row["signed_full20"]["d_ctr"] <= min(row["signed_omp"][3]["d_ctr"], row["scaled_single"]["d_ctr"], row["nonnegative_full20"]["d_ctr"]) + 1e-7
            for row in output_rows
        )
        finite = all(np.isfinite([
            row["unweighted_native_size4"]["d_ctr"], row["scaled_single"]["d_ctr"], row["signed_omp"][3]["d_ctr"],
            row["nonnegative_full20"]["d_ctr"], row["signed_full20"]["d_ctr"], row["signed_full20"]["d_mu"],
        ]).all() for row in output_rows)
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "complete_ordered_pair_grid": len(output_rows) == expected == 2560,
            "unique_rows": len({(r["source_seed"], r["source_atom"], r["target_seed"]) for r in output_rows}) == len(output_rows),
            "proposal_identity_preserved": all(r["proposal_target_ids"] == surface_map[(r["source_seed"], r["source_atom"], r["target_seed"])]["proposal_target_ids"] for r in output_rows),
            "all_methods_reported": all(len(r["signed_omp"]) == 4 for r in output_rows),
            "continuous_outputs_non_native": all(r["scaled_single"]["semantics"] == r["signed_full20"]["semantics"] == cfg["continuous_output_semantics"] for r in output_rows),
            "metrics_finite": finite,
            "omp_residual_monotone": omp_monotone,
            "signed_full_dominates_nested_references": dominance,
            "discovery_only": cfg["split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "no_found_decision": cfg["threshold_source_split"] == "none_separator_only",
            "audit_not_opened": not cfg["audit_opened"],
        }
        methods = {
            "unweighted_native_size4": [r["unweighted_native_size4"]["d_ctr"] for r in output_rows],
            "scaled_single": [r["scaled_single"]["d_ctr"] for r in output_rows],
            "signed_omp4": [r["signed_omp"][3]["d_ctr"] for r in output_rows],
            "nonnegative_full20": [r["nonnegative_full20"]["d_ctr"] for r in output_rows],
            "signed_full20": [r["signed_full20"]["d_ctr"] for r in output_rows],
        }
        method_summary = {
            name: {
                "median_d_ctr": float(np.median(values)), "q10_d_ctr": float(np.quantile(values, 0.1)), "q90_d_ctr": float(np.quantile(values, 0.9)),
                "fraction_below_0_25": float(np.mean(np.asarray(values) < 0.25)), "fraction_below_0_50": float(np.mean(np.asarray(values) < 0.50)),
            }
            for name, values in methods.items()
        }
        record = {
            "checks": checks, "row_count": len(output_rows), "pair_summaries": pair_summaries, "method_summary": method_summary,
            "nonnegative_solver_failures": solver_failures,
            "median_signed_full_negative_coefficient_fraction": float(np.median([r["signed_full20"]["negative_coefficient_fraction"] for r in output_rows])),
            "median_signed_full_effective_rank": float(np.median([r["signed_full20"]["effective_rank"] for r in output_rows])),
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
        "generator_script_path": "scripts/run_r011b_representability_separator.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
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
