"""Run the frozen R011-NR1 k32-vs-k128 structure and native-coverage screen."""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import __version__ as scipy_version  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from run_r009c_atom_discovery import decoder, sparse_codes, top_ids  # noqa: E402
from run_r011a_group_discovery_surface import ordered_unique, support_metrics  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def selection_hash(salt: str, config_name: str, seed: int, atom: int) -> str:
    return hashlib.sha256(f"{salt}\0{config_name}\0{seed}\0{atom}".encode()).hexdigest()


def moments(matrix, mean: np.ndarray, tokens: int) -> tuple[np.ndarray, np.ndarray]:
    second = np.asarray(matrix.multiply(matrix).sum(axis=0)).reshape(-1).astype(np.float64) / tokens
    variance = np.maximum(0.0, second - mean * mean)
    firing = np.asarray((matrix != 0).sum(axis=0)).reshape(-1).astype(np.int64)
    return variance, firing


def mean_codes(matrix, tokens: int) -> np.ndarray:
    return np.asarray(matrix.sum(axis=0)).reshape(-1).astype(np.float64) / tokens


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def calibration_metrics(
    support: list[int],
    z_source,
    d_source: np.ndarray,
    source_mean: float,
    z_target,
    d_target: np.ndarray,
    target_means: np.ndarray,
    tokens: int,
    source_energy_epsilon: float,
    zero_mean_tolerance: float,
) -> dict:
    ids = np.asarray(support, dtype=np.int64)
    zt = z_target[:, ids]
    dt = d_target[ids]
    source_second = float(z_source.multiply(z_source).sum()) / tokens
    source_variance = max(0.0, source_second - source_mean * source_mean)
    source_norm_sq = float(d_source @ d_source)
    source_energy = source_variance * source_norm_sq
    source_firing_count = int(z_source.nnz)
    calibration_evaluable = source_energy > source_energy_epsilon and source_firing_count > 0
    target_cov = (zt.T @ zt).toarray().astype(np.float64) / tokens - np.outer(target_means[ids], target_means[ids])
    target_energy = float(np.sum(target_cov * (dt @ dt.T)))
    raw_cross = (z_source.T @ zt).toarray().reshape(-1).astype(np.float64) / tokens
    code_cross = raw_cross - source_mean * target_means[ids]
    contribution_cross = float(np.sum(code_cross * (d_source @ dt.T)))
    residual = max(0.0, source_energy + target_energy - 2.0 * contribution_cross)
    d_ctr = residual / max(source_energy, source_energy_epsilon)
    source_mean_vector = source_mean * d_source
    target_mean_vector = np.sum(target_means[ids, None] * dt, axis=0)
    mean_residual = float(np.sum((source_mean_vector - target_mean_vector) ** 2))
    source_mean_energy = float(source_mean_vector @ source_mean_vector)
    mean_gate_applicable = source_mean_energy > zero_mean_tolerance
    d_mu = mean_residual / source_mean_energy if mean_gate_applicable else None
    mean_pass = d_mu <= 0.05 if mean_gate_applicable else mean_residual <= zero_mean_tolerance
    return {
        "d_ctr": finite_or_none(d_ctr),
        "d_mu": finite_or_none(d_mu) if d_mu is not None else None,
        "mean_residual_absolute": mean_residual,
        "mean_gate_applicable": mean_gate_applicable,
        "mean_pass_default": bool(mean_pass),
        "bcc": finite_or_none(2.0 * contribution_cross / (source_energy + target_energy)) if source_energy + target_energy > 0 else None,
        "source_energy": source_energy,
        "source_firing_count": source_firing_count,
        "calibration_evaluable": bool(calibration_evaluable),
    }


def config_screen(cfg: dict, item: dict, combos: dict[int, np.ndarray]) -> tuple[dict, list[dict], list[dict]]:
    name = item["name"]
    asset_dir = Path(item["bulk_asset_dir"])
    manifest_path = asset_dir / "asset_manifest.json"
    if sha256(manifest_path).lower() != item["asset_manifest_sha256"].lower():
        raise ValueError(f"{name}: asset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_meta = {row["split"]: row for row in manifest["splits"]}
    expected_tokens = {
        cfg["mean_split"]: cfg["mean_tokens"],
        cfg["discovery_split"]: cfg["discovery_tokens"],
        cfg["calibration_split"]: cfg["calibration_tokens"],
    }
    if any(split_meta[split]["tokens"] != count for split, count in expected_tokens.items()):
        raise ValueError(f"{name}: split token contract mismatch")

    seeds = cfg["source_seeds"]
    k = item["k"]
    latents = cfg["num_latents"]
    hidden = cfg["hook_hidden_size"]
    mean_matrices = {seed: sparse_codes(asset_dir, cfg["mean_split"], seed, cfg["mean_tokens"], k, latents) for seed in seeds}
    discovery = {seed: sparse_codes(asset_dir, cfg["discovery_split"], seed, cfg["discovery_tokens"], k, latents) for seed in seeds}
    calibration = {seed: sparse_codes(asset_dir, cfg["calibration_split"], seed, cfg["calibration_tokens"], k, latents) for seed in seeds}
    decoders = {seed: decoder(asset_dir, seed, latents, hidden).astype(np.float64, copy=False) for seed in seeds}
    means = {seed: mean_codes(mean_matrices[seed], cfg["mean_tokens"]) for seed in seeds}
    variances, firing, energy = {}, {}, {}
    for seed in seeds:
        variances[seed], firing[seed] = moments(discovery[seed], means[seed], cfg["discovery_tokens"])
        energy[seed] = variances[seed] * np.einsum("ij,ij->i", decoders[seed], decoders[seed])

    panels: dict[int, list[dict]] = {}
    panel_rows: list[dict] = []
    for seed in seeds:
        ranked = np.lexsort((np.arange(latents), energy[seed]))
        selected = []
        for stratum in range(cfg["strata_per_seed"]):
            members = ranked[stratum * cfg["atoms_per_stratum"] : (stratum + 1) * cfg["atoms_per_stratum"]]
            chosen = sorted(
                (int(atom) for atom in members),
                key=lambda atom: (selection_hash(cfg["selection_salt"], name, seed, atom), atom),
            )[: cfg["queries_per_stratum"]]
            for atom in chosen:
                row = {
                    "configuration": name,
                    "seed": seed,
                    "atom": atom,
                    "energy_stratum": stratum,
                    "discovery_firing_count": int(firing[seed][atom]),
                    "discovery_firing_rate": float(firing[seed][atom] / cfg["discovery_tokens"]),
                    "source_contribution_energy": float(energy[seed][atom]),
                    "selection_hash": selection_hash(cfg["selection_salt"], name, seed, atom),
                    "dynamic_eligible": bool(energy[seed][atom] > cfg["source_energy_epsilon"] and firing[seed][atom] > 0),
                }
                selected.append(row)
                panel_rows.append(row)
        panels[seed] = sorted(selected, key=lambda row: (row["energy_stratum"], row["selection_hash"]))

    full_pw = []
    directional_pw = []
    for source_seed, target_seed in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        raw = (discovery[source_seed].T @ discovery[target_seed]).toarray().astype(np.float64) / cfg["discovery_tokens"]
        covariance = raw - np.outer(means[source_seed], means[target_seed])
        denom = np.sqrt(variances[source_seed][:, None] * variances[target_seed][None, :])
        corr = np.divide(np.abs(covariance), denom, out=np.zeros_like(covariance), where=denom > cfg["source_energy_epsilon"])
        corr = np.clip(corr, 0.0, 1.0)
        row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
        assigned = np.zeros(latents, dtype=np.float64)
        assigned[row_ind] = corr[row_ind, col_ind]
        full_pw.append({
            "source_seed": source_seed,
            "target_seed": target_seed,
            "pw_mcc_all_atoms": float(np.mean(assigned)),
            "pw_mcc_source_alive": float(np.mean(assigned[firing[source_seed] > 0])) if np.any(firing[source_seed] > 0) else 0.0,
            "source_alive_fraction": float(np.mean(firing[source_seed] > 0)),
            "target_alive_fraction": float(np.mean(firing[target_seed] > 0)),
        })
        ranked = np.lexsort((np.arange(latents), firing[source_seed]))
        for stratum in range(cfg["strata_per_seed"]):
            ids = ranked[stratum * cfg["atoms_per_stratum"] : (stratum + 1) * cfg["atoms_per_stratum"]]
            directional_pw.append({
                "source_seed": source_seed,
                "target_seed": target_seed,
                "frequency_stratum": stratum,
                "minimum_firing": int(firing[source_seed][ids].min()),
                "maximum_firing": int(firing[source_seed][ids].max()),
                "mean_hungarian_abs_correlation": float(np.mean(assigned[ids])),
                "median_hungarian_abs_correlation": float(np.median(assigned[ids])),
            })

    result_rows: list[dict] = []
    for source_seed, target_seed in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        query_rows = panels[source_seed]
        query_ids = np.asarray([row["atom"] for row in query_rows], dtype=np.int64)
        zs = discovery[source_seed][:, query_ids]
        zt = discovery[target_seed]
        ds = decoders[source_seed][query_ids]
        dt = decoders[target_seed]
        source_means = means[source_seed][query_ids]
        target_means = means[target_seed]
        raw_cross = (zs.T @ zt).toarray().astype(np.float64) / cfg["discovery_tokens"]
        code_cov = raw_cross - source_means[:, None] * target_means[None, :]
        decoder_dot = ds @ dt.T
        contribution_cross = code_cov * decoder_dot
        source_energy = energy[source_seed][query_ids]
        target_energy = energy[target_seed]
        denom = source_energy[:, None] + target_energy[None, :]
        bcc = np.divide(2.0 * contribution_cross, denom, out=np.full_like(contribution_cross, -np.inf), where=denom > cfg["source_energy_epsilon"])
        rho_denom = np.sqrt(source_energy[:, None] * target_energy[None, :])
        rho = np.divide(contribution_cross, rho_denom, out=np.full_like(contribution_cross, -np.inf), where=rho_denom > cfg["source_energy_epsilon"])
        source_norm = np.einsum("ij,ij->i", ds, ds)
        target_norm = np.einsum("ij,ij->i", dt, dt)
        cosine_denom = np.sqrt(source_norm[:, None] * target_norm[None, :])
        cosine = np.divide(np.abs(decoder_dot), cosine_denom, out=np.full_like(decoder_dot, -np.inf), where=cosine_denom > 0)

        for qi, query in enumerate(query_rows):
            base = {
                "configuration": name,
                "source_seed": source_seed,
                "target_seed": target_seed,
                "source_atom": int(query["atom"]),
                "energy_stratum": int(query["energy_stratum"]),
                "source_firing_count": int(query["discovery_firing_count"]),
                "dynamic_eligible": bool(query["dynamic_eligible"]),
            }
            if not query["dynamic_eligible"]:
                result_rows.append({**base, "best_single_bcc": None, "proposal_target_ids": [], "discovery_best_by_size": [], "calibration_by_size": [], "identification": "UNRESOLVED", "reason": "SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON"})
                continue
            rule = cfg["proposal_union_rule"]
            proposal = ordered_unique([
                top_ids(rho[qi], rule["positive_contribution_correlation"], descending=True),
                top_ids(bcc[qi], rule["balanced_bcc"], descending=True),
                top_ids(cosine[qi], rule["absolute_decoder_cosine"], descending=True),
                top_ids(rho[qi], latents, descending=True),
                top_ids(bcc[qi], latents, descending=True),
            ], cfg["proposal_atom_cap"])
            ids = np.asarray(proposal, dtype=np.int64)
            zsub = zt[:, ids]
            dsub = dt[ids]
            code_tt = (zsub.T @ zsub).toarray().astype(np.float64) / cfg["discovery_tokens"] - np.outer(target_means[ids], target_means[ids])
            ktt = code_tt * (dsub @ dsub.T)
            kst = contribution_cross[qi, ids]
            mean_vectors = target_means[ids, None] * dsub
            mtt = mean_vectors @ mean_vectors.T
            source_mean_vector = source_means[qi] * ds[qi]
            mst = source_mean_vector @ mean_vectors.T
            source_mean_energy = float(source_mean_vector @ source_mean_vector)
            discovery_best, calibration_by_size = [], []
            for size in range(1, cfg["g_max"] + 1):
                local = combos[size]
                d_ctr, d_mu, group_bcc = support_metrics(local, ktt, kst, mtt, mst, float(source_energy[qi]), source_mean_energy)
                best_index = int(np.lexsort((np.arange(len(d_ctr)), d_ctr))[0])
                support = [proposal[int(value)] for value in local[best_index]]
                discovery_best.append({"size": size, "support": support, "d_ctr": float(d_ctr[best_index]), "d_mu": finite_or_none(d_mu[best_index]), "bcc": finite_or_none(group_bcc[best_index])})
                cal = calibration_metrics(
                    support,
                    calibration[source_seed][:, int(query["atom"])],
                    ds[qi],
                    float(source_means[qi]),
                    calibration[target_seed],
                    dt,
                    target_means,
                    cfg["calibration_tokens"],
                    cfg["source_energy_epsilon"],
                    cfg["zero_mean_absolute_tolerance"],
                )
                ctr_pass = cal["calibration_evaluable"] and cal["d_ctr"] is not None and cal["d_ctr"] <= cfg["primary_tau_ctr"]
                mean_pass = (cal["d_mu"] <= cfg["primary_tau_mu"]) if cal["mean_gate_applicable"] and cal["d_mu"] is not None else cal["mean_residual_absolute"] <= cfg["zero_mean_absolute_tolerance"]
                calibration_by_size.append({"size": size, "support": support, **cal, "passes_primary": bool(ctr_pass and mean_pass)})
            calibration_evaluable = calibration_by_size[0]["calibration_evaluable"]
            accepted = next((row for row in calibration_by_size if row["passes_primary"]), None) if calibration_evaluable else None
            result_rows.append({
                **base,
                "best_single_bcc": finite_or_none(np.max(bcc[qi])),
                "proposal_target_ids": proposal,
                "discovery_best_by_size": discovery_best,
                "calibration_by_size": calibration_by_size,
                "identification": "FOUND" if accepted else "UNRESOLVED",
                "support": accepted["support"] if accepted else [],
                "support_size": accepted["size"] if accepted else None,
                "reason": None if accepted else ("CALIBRATION_SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON" if not calibration_evaluable else "NO_DISCOVERY_FROZEN_SUPPORT_PASSES_CALIBRATION_GATES"),
            })

    eligible = [row for row in result_rows if row["dynamic_eligible"]]
    found = [row for row in eligible if row["identification"] == "FOUND"]
    direction_summaries = []
    for source_seed, target_seed in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        rows = [row for row in result_rows if row["source_seed"] == source_seed and row["target_seed"] == target_seed]
        eligible_rows = [row for row in rows if row["dynamic_eligible"]]
        found_rows = [row for row in eligible_rows if row["identification"] == "FOUND"]
        direction_summaries.append({
            "source_seed": source_seed,
            "target_seed": target_seed,
            "queries": len(rows),
            "eligible_queries": len(eligible_rows),
            "found": len(found_rows),
            "found_fraction_all_queries": len(found_rows) / len(rows) if rows else 0.0,
            "eligible_found_fraction": len(found_rows) / len(eligible_rows) if eligible_rows else 0.0,
            "median_best_single_bcc": float(np.median([row["best_single_bcc"] for row in eligible_rows])) if eligible_rows else None,
        })
    rule = cfg["meaningful_coverage_rule"]
    eligible_found_fraction = len(found) / len(eligible) if eligible else 0.0
    all_found_fraction = len(found) / len(result_rows) if result_rows else 0.0
    meaningful = all_found_fraction >= rule["minimum_overall_found_fraction"] and all(
        row["found_fraction_all_queries"] >= rule["minimum_each_direction_found_fraction"] for row in direction_summaries
    )
    summary = {
        "configuration": name,
        "k": k,
        "query_rows": len(result_rows),
        "eligible_queries": len(eligible),
        "zero_dynamic_energy_refusals": len(result_rows) - len(eligible),
        "calibration_unevaluable_refusals": sum(row["reason"] == "CALIBRATION_SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON" for row in result_rows),
        "found": len(found),
        "found_fraction_all_queries": all_found_fraction,
        "found_fraction_eligible_queries": eligible_found_fraction,
        "meaningful_native_coverage": bool(meaningful),
        "median_best_single_bcc": float(np.median([row["best_single_bcc"] for row in eligible])) if eligible else None,
        "median_calibration_d_ctr_size4": float(np.median([row["calibration_by_size"][3]["d_ctr"] for row in eligible])) if eligible else None,
        "full_dictionary_pw_mcc": full_pw,
        "frequency_stratified_pw_mcc": directional_pw,
        "direction_summaries": direction_summaries,
        "asset_manifest_sha256": sha256(manifest_path),
    }
    return summary, panel_rows, result_rows


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
    code_paths = [Path(__file__).resolve(), ROOT / "scripts" / "run_r009c_atom_discovery.py", ROOT / "scripts" / "run_r011a_group_discovery_surface.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    inputs = [{"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"}]
    for item in cfg["configurations"]:
        manifest = Path(item["bulk_asset_dir"]) / "asset_manifest.json"
        inputs.append({"path": str(manifest.resolve()), "sha256": sha256(manifest), "bytes": manifest.stat().st_size, "source": item["paired_codes_run"], "license_or_access_boundary": "internal", "role": f"{item['name']}_paired_asset_ledger"})
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "audit_opened": False,
        "candidate_family_frozen": True, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": "cpu",
        "seeds": cfg["source_seeds"], "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "full-dictionary sparse covariance, Hungarian matching, and frozen local-support calibration over D: paired assets",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    panel_rows: list[dict] = []
    screen_rows: list[dict] = []
    try:
        if cfg["support_family_count"] != sum(len(list(itertools.combinations(range(cfg["proposal_atom_cap"]), size))) for size in range(1, cfg["g_max"] + 1)):
            raise ValueError("support family count mismatch")
        if cfg["support_family_count"] > cfg["candidate_evaluation_budget"]:
            raise ValueError("candidate evaluation budget exceeded")
        if set(cfg["forbidden_splits"]) != {"audit"} or cfg["audit_opened"]:
            raise ValueError("audit must remain closed")
        combos = {size: np.asarray(list(itertools.combinations(range(cfg["proposal_atom_cap"]), size)), dtype=np.int16) for size in range(1, cfg["g_max"] + 1)}
        summaries = []
        for item in cfg["configurations"]:
            summary, config_panel, config_rows = config_screen(cfg, item, combos)
            summaries.append(summary)
            panel_rows.extend(config_panel)
            screen_rows.extend(config_rows)
        panel_path = run_dir / "query_panel.jsonl"
        panel_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in panel_rows), encoding="utf-8")
        results_path = run_dir / "screen_results.jsonl"
        results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in screen_rows), encoding="utf-8")
        passing = [row["configuration"] for row in summaries if row["meaningful_native_coverage"]]
        decision = "ADVANCE_" + passing[0] if len(passing) == 1 else ("CLOSE_TOPK_RESCUE" if not passing else "NO_SINGLE_WINNER")
        expected_panel = len(cfg["configurations"]) * len(cfg["source_seeds"]) * cfg["queries_per_seed"]
        expected_results = expected_panel
        checks = {
            "asset_manifests_bound": all(row["asset_manifest_sha256"].lower() == next(item["asset_manifest_sha256"].lower() for item in cfg["configurations"] if item["name"] == row["configuration"]) for row in summaries),
            "complete_panel": len(panel_rows) == expected_panel,
            "balanced_panel_strata": all(sum(row["configuration"] == item["name"] and row["seed"] == seed and row["energy_stratum"] == stratum for row in panel_rows) == cfg["queries_per_stratum"] for item in cfg["configurations"] for seed in cfg["source_seeds"] for stratum in range(cfg["strata_per_seed"])),
            "complete_ordered_pair_results": len(screen_rows) == expected_results,
            "unique_results": len({(row["configuration"], row["source_seed"], row["source_atom"], row["target_seed"]) for row in screen_rows}) == len(screen_rows),
            "zero_energy_refused": all(row["dynamic_eligible"] or (row["identification"] == "UNRESOLVED" and row["reason"] == "SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON") for row in screen_rows),
            "calibration_unevaluable_refused": all(row["identification"] == "UNRESOLVED" and row["reason"] == "CALIBRATION_SOURCE_DYNAMIC_ENERGY_BELOW_EPSILON" for row in screen_rows if row["dynamic_eligible"] and row["calibration_by_size"] and not row["calibration_by_size"][0]["calibration_evaluable"]),
            "fixed_candidate_budget": all(not row["dynamic_eligible"] or len(row["proposal_target_ids"]) == cfg["proposal_atom_cap"] for row in screen_rows),
            "all_support_sizes_scored": all(not row["dynamic_eligible"] or [entry["size"] for entry in row["calibration_by_size"]] == list(range(1, cfg["g_max"] + 1)) for row in screen_rows),
            "audit_not_opened": not cfg["audit_opened"] and set(cfg["forbidden_splits"]) == {"audit"},
            "decision_rule_applied": decision in {"ADVANCE_k32_long4m", "ADVANCE_k128_long4m", "CLOSE_TOPK_RESCUE", "NO_SINGLE_WINNER"},
        }
        record = {"checks": checks, "configuration_summaries": summaries, "screen_decision": decision, "query_panel_sha256": sha256(panel_path), "screen_results_sha256": sha256(results_path)}
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
        "generator_script_path": "scripts/run_r011_nr1_structure_coverage_screen.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
        "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "decision": record["screen_decision"] if record else None}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "decision": record["screen_decision"] if record else None}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
