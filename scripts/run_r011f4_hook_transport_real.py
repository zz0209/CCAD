"""Run the single bounded C047 hook-space transport screen."""
from __future__ import annotations

import argparse
from collections import Counter
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
from ccad.causal_metric_probe import select_document_balanced_states  # noqa: E402
from ccad.hook_transport import (  # noqa: E402
    decide_transport_gate,
    fit_basis_constrained_transport,
    fit_nuisance_projector,
    residualize_hook_process,
    transport_metrics,
    transport_prefix,
    transport_subspace_overlap,
)
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402
from run_r011f1_euclidean_surface import condition_weights  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "source": "CCAD frozen artifact", "license_or_access_boundary": "internal", "role": role}


def centered_reconstruct(matrix, rows: np.ndarray, dec: np.ndarray, mean_codes: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[rows] @ dec, dtype=np.float64) - (mean_codes @ dec)[None, :]


def weighted_pca(process: np.ndarray, weights: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    root = np.sqrt(weights / np.sum(weights))[:, None]
    _, singular, right_t = np.linalg.svd(root * process, full_matrices=False)
    if singular.size < max_rank + 1 or singular[0] <= 0:
        raise ValueError("source query process does not expose the frozen rank family and boundary")
    return right_t[:max_rank].T, singular


def rank_gap(singular: np.ndarray, rank: int) -> float:
    return float((singular[rank - 1] - singular[rank]) / singular[0])


def serial_metrics(value) -> dict:
    return {key: getattr(value, key) for key in value.__dataclass_fields__}


def specificity(positive, negative) -> float | None:
    if positive.bcc is None:
        return None
    return positive.bcc - (0.0 if negative.bcc is None else negative.bcc)


def pad_columns(matrix: np.ndarray, columns: int) -> np.ndarray:
    """Pad a rank-deficient factor matrix without changing its effective rank."""

    if matrix.ndim != 2 or matrix.shape[1] > columns:
        raise ValueError("factor matrix cannot be padded to requested columns")
    padded = np.zeros((matrix.shape[0], columns), dtype=np.float32)
    padded[:, :matrix.shape[1]] = matrix.astype(np.float32, copy=False)
    return padded


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/hook_transport.py", ROOT / "scripts/run_r009c_atom_discovery.py", ROOT / "scripts/run_r011f1_euclidean_surface.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    residual_mode = "nuisance_state_count" in cfg
    paths = {
        "protocol": ROOT / cfg["protocol_document"], "synthetic_status": ROOT / cfg["synthetic_gate_status_path"], "synthetic_metrics": ROOT / cfg["synthetic_gate_metrics_path"],
        "reference": ROOT / cfg["reference_surface_path"], "census": ROOT / cfg["source_census_path"], "sequences": ROOT / cfg["sequence_records_path"],
        "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json", "raw_manifest": Path(cfg["raw_hook_asset_dir"]) / "raw_hook_manifest.json",
    }
    if residual_mode:
        paths["unresidualized_surface"] = ROOT / cfg["unresidualized_transport_surface_path"]
    write_json(run_dir / "inputs.json", {"inputs": [file_entry(args.config.resolve(), "run_protocol")] + [file_entry(path, role) for role, path in paths.items()]})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True,
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": "cpu", "seeds": cfg["source_seeds"],
        "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run", "resource_lease_reason": "bounded sparse reconstruction, dual ridge solves, and hook-space transport evaluation",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {
            "synthetic_status": sha256(paths["synthetic_status"]).lower() == cfg["synthetic_gate_status_sha256"], "synthetic_metrics": sha256(paths["synthetic_metrics"]).lower() == cfg["synthetic_gate_metrics_sha256"],
            "reference": sha256(paths["reference"]).lower() == cfg["reference_surface_sha256"], "census": sha256(paths["census"]).lower() == cfg["source_census_sha256"], "sequences": sha256(paths["sequences"]).lower() == cfg["sequence_records_sha256"],
            "asset_manifest": sha256(paths["asset_manifest"]).lower() == cfg["asset_manifest_sha256"], "raw_manifest": sha256(paths["raw_manifest"]).lower() == cfg["raw_hook_manifest_sha256"],
        }
        if residual_mode:
            bound["unresidualized_surface"] = sha256(paths["unresidualized_surface"]).lower() == cfg["unresidualized_transport_surface_sha256"]
        synthetic_status = json.loads(paths["synthetic_status"].read_text(encoding="utf-8"))["status"]
        if not all(bound.values()) or synthetic_status != "PASS" or not cfg["execution_enabled"] or cfg["audit_opened"] or cfg["forbidden_splits"] != ["audit"]:
            raise ValueError(f"frozen input, synthetic, execution, or audit boundary mismatch: {bound}")
        reference = [json.loads(line) for line in paths["reference"].read_text(encoding="utf-8").splitlines() if line]
        base_rows = [row for row in reference if row["rank"] == 1]
        census = [json.loads(line) for line in paths["census"].read_text(encoding="utf-8").splitlines() if line]
        stats = {(int(row["seed"]), int(row["atom"])): row for row in census}
        means = {seed: np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64) for seed in cfg["source_seeds"]}
        asset_dir = Path(cfg["bulk_asset_dir"]); asset_manifest = json.loads(paths["asset_manifest"].read_text(encoding="utf-8")); split_tokens = {row["split"]: int(row["tokens"]) for row in asset_manifest["splits"]}
        matrices = {split: {seed: sparse_codes(asset_dir, split, seed, split_tokens[split], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]} for split in cfg["splits"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        raw_manifest = json.loads(paths["raw_manifest"].read_text(encoding="utf-8")); raw_meta = {row["split"]: row for row in raw_manifest["splits"]}
        raw = {split: np.memmap(raw_meta[split]["path"], dtype="<f4", mode="r").reshape(raw_meta[split]["shape"]) for split in ("mean", "discovery", "calibration")}
        raw_mean = np.mean(raw["mean"], axis=0, dtype=np.float64)
        sequence_payload = json.loads(paths["sequences"].read_text(encoding="utf-8"))["sequences"]
        nuisance = None
        unresidualized = {}
        if residual_mode:
            nuisance_states = select_document_balanced_states(sequence_payload, split="discovery", count=cfg["nuisance_state_count"], token_positions=tuple(cfg["nuisance_state_positions"]), salt=cfg["nuisance_state_salt"])
            nuisance_rows = np.asarray([int(row["sequence_index"]) * cfg["context_length"] + int(row["token_position"]) for row in nuisance_states], dtype=np.int64)
            nuisance_process = np.asarray(raw["discovery"][nuisance_rows], dtype=np.float64) - raw_mean[None, :]
            nuisance = fit_nuisance_projector(nuisance_process, np.full(len(nuisance_rows), 1 / len(nuisance_rows)), explained_variance_threshold=cfg["nuisance_explained_variance_threshold"], maximum_rank=cfg["nuisance_maximum_rank"])
            if nuisance.status != "OK":
                raise ValueError(f"frozen nuisance variance threshold not reached: rank={nuisance.rank}, fraction={nuisance.explained_variance_fraction}")
            unresidualized_rows = [json.loads(line) for line in paths["unresidualized_surface"].read_text(encoding="utf-8").splitlines() if line]
            unresidualized = {(int(row["source_seed"]), int(row["source_atom"]), int(row["target_seed"]), int(row["rank"])): row for row in unresidualized_rows if row.get("query_role") == "anchor" and row.get("evaluable")}
        global_states = select_document_balanced_states(sequence_payload, split="discovery", count=cfg["global_control_tokens"], token_positions=tuple(cfg["global_control_state_positions"]), salt=cfg["global_control_state_salt"])
        global_rows = np.asarray([int(row["sequence_index"]) * cfg["context_length"] + int(row["token_position"]) for row in global_states], dtype=np.int64); global_weights = np.full(len(global_rows), 1 / len(global_rows))
        query_cache = {}; raw_cache = {}; factor_map = {}; anchor_payload = []; output_rows = []; started_compute = time.perf_counter()

        for base in base_rows:
            source_seed, target_seed, atom = int(base["source_seed"]), int(base["target_seed"]), int(base["source_atom"])
            common = {"source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": int(base["energy_stratum"]), "query_role": base["query_role"]}
            if not base.get("evaluable", False):
                for rank in cfg["candidate_ranks"]: output_rows.append({**common, "rank": rank, "evaluable": False, "reason": base.get("reason")})
                continue
            query_key = (source_seed, atom)
            if query_key not in query_cache:
                source_ids = np.asarray(base["source_candidate_ids"], dtype=np.int64); negative_atoms = [int(value) for value in base["negative_source_atoms"]]
                drows, dweights = condition_weights(matrices["discovery"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                dnrows, dnweights = condition_weights(matrices["discovery"][source_seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                crows, cweights = condition_weights(matrices["calibration"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                cnrows, cnweights = condition_weights(matrices["calibration"][source_seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                source_dec = decoders[source_seed][source_ids]; source_mean = means[source_seed][source_ids]
                source_disc_unresidualized = centered_reconstruct(matrices["discovery"][source_seed][:, source_ids], drows, source_dec, source_mean)
                source_disc = residualize_hook_process(source_disc_unresidualized, nuisance) if residual_mode else source_disc_unresidualized
                retained_energy = float(np.sum(dweights[:, None] * source_disc * source_disc) / max(np.sum(dweights[:, None] * source_disc_unresidualized * source_disc_unresidualized), np.finfo(np.float64).eps))
                basis, singular = weighted_pca(source_disc, dweights, max(cfg["candidate_ranks"]))
                query_cache[query_key] = {"source_ids": source_ids, "negative_atoms": negative_atoms, "drows": drows, "dweights": dweights, "dnrows": dnrows, "dnweights": dnweights, "crows": crows, "cweights": cweights, "cnrows": cnrows, "cnweights": cnweights, "basis": basis, "singular": singular,
                    "source_disc_coord": source_disc @ basis,
                    "source_cal_pos_coord": (residualize_hook_process(centered_reconstruct(matrices["calibration"][source_seed][:, source_ids], crows, source_dec, source_mean), nuisance) if residual_mode else centered_reconstruct(matrices["calibration"][source_seed][:, source_ids], crows, source_dec, source_mean)) @ basis,
                    "source_cal_neg_coord": (residualize_hook_process(centered_reconstruct(matrices["calibration"][source_seed][:, source_ids], cnrows, source_dec, source_mean), nuisance) if residual_mode else centered_reconstruct(matrices["calibration"][source_seed][:, source_ids], cnrows, source_dec, source_mean)) @ basis,
                    "source_global_coord": (residualize_hook_process(centered_reconstruct(matrices["discovery"][source_seed][:, source_ids], global_rows, source_dec, source_mean), nuisance) if residual_mode else centered_reconstruct(matrices["discovery"][source_seed][:, source_ids], global_rows, source_dec, source_mean)) @ basis,
                    "residual_energy_fraction": retained_energy}
            q = query_cache[query_key]; basis = q["basis"]
            if query_key not in raw_cache:
                raw_disc = np.asarray(raw["discovery"][q["drows"]], dtype=np.float64) - raw_mean[None, :]
                if residual_mode: raw_disc = residualize_hook_process(raw_disc, nuisance)
                fitted_raw = fit_basis_constrained_transport(raw_disc, q["source_disc_coord"], basis, q["dweights"], ridge_fraction=cfg["ridge_fraction"])
                raw_pos_input = np.asarray(raw["calibration"][q["crows"]], dtype=np.float64) - raw_mean[None, :]
                raw_neg_input = np.asarray(raw["calibration"][q["cnrows"]], dtype=np.float64) - raw_mean[None, :]
                if residual_mode:
                    raw_pos_input = residualize_hook_process(raw_pos_input, nuisance); raw_neg_input = residualize_hook_process(raw_neg_input, nuisance)
                raw_cache[query_key] = {}
                for rank in cfg["candidate_ranks"]:
                    fit_r = transport_prefix(fitted_raw, rank); source_pos = q["source_cal_pos_coord"][:, :rank] @ basis[:, :rank].T; source_neg = q["source_cal_neg_coord"][:, :rank] @ basis[:, :rank].T
                    pos = transport_metrics(source_pos, fit_r.predict(raw_pos_input), q["cweights"]); neg = transport_metrics(source_neg, fit_r.predict(raw_neg_input), q["cnweights"])
                    raw_cache[query_key][rank] = {"transport": fit_r, "positive": pos, "negative": neg, "specificity": specificity(pos, neg)}
            target_dec = decoders[target_seed]; target_mean = means[target_seed]
            target_disc = centered_reconstruct(matrices["discovery"][target_seed], q["drows"], target_dec, target_mean)
            target_cal_pos = centered_reconstruct(matrices["calibration"][target_seed], q["crows"], target_dec, target_mean)
            target_cal_neg = centered_reconstruct(matrices["calibration"][target_seed], q["cnrows"], target_dec, target_mean)
            target_global = centered_reconstruct(matrices["discovery"][target_seed], global_rows, target_dec, target_mean)
            if residual_mode:
                target_disc = residualize_hook_process(target_disc, nuisance); target_cal_pos = residualize_hook_process(target_cal_pos, nuisance); target_cal_neg = residualize_hook_process(target_cal_neg, nuisance); target_global = residualize_hook_process(target_global, nuisance)
            fitted_query = fit_basis_constrained_transport(target_disc, q["source_disc_coord"], basis, q["dweights"], ridge_fraction=cfg["ridge_fraction"])
            fitted_global = fit_basis_constrained_transport(target_global, q["source_global_coord"], basis, global_weights, ridge_fraction=cfg["ridge_fraction"])
            for rank in cfg["candidate_ranks"]:
                query_fit = transport_prefix(fitted_query, rank); global_fit = transport_prefix(fitted_global, rank); raw_result = raw_cache[query_key][rank]
                source_pos = q["source_cal_pos_coord"][:, :rank] @ basis[:, :rank].T; source_neg = q["source_cal_neg_coord"][:, :rank] @ basis[:, :rank].T
                query_pos = transport_metrics(source_pos, query_fit.predict(target_cal_pos), q["cweights"]); query_neg = transport_metrics(source_neg, query_fit.predict(target_cal_neg), q["cnweights"])
                global_pos = transport_metrics(source_pos, global_fit.predict(target_cal_pos), q["cweights"]); global_neg = transport_metrics(source_neg, global_fit.predict(target_cal_neg), q["cnweights"])
                factor_map[(source_seed, atom, target_seed, rank)] = (query_fit, global_fit)
                unresidualized_specificity = unresidualized.get((source_seed, atom, target_seed, rank), {}).get("query_specificity") if residual_mode else None
                output_rows.append({**common, "rank": rank, "evaluable": True, "source_candidate_ids": q["source_ids"].tolist(), "negative_source_atoms": q["negative_atoms"], "query_status": query_fit.status, "raw_status": raw_result["transport"].status, "global_status": global_fit.status, "source_residual_energy_fraction": q["residual_energy_fraction"], "unresidualized_query_specificity": unresidualized_specificity,
                    "source_rank_boundary_relative_gap": rank_gap(q["singular"], rank), "query_positive": serial_metrics(query_pos), "query_negative": serial_metrics(query_neg), "query_specificity": specificity(query_pos, query_neg),
                    "raw_positive": serial_metrics(raw_result["positive"]), "raw_negative": serial_metrics(raw_result["negative"]), "raw_specificity": raw_result["specificity"],
                    "global_positive": serial_metrics(global_pos), "global_negative": serial_metrics(global_neg), "global_specificity": specificity(global_pos, global_neg)})
                if base["query_role"] == "anchor" and rank == max(cfg["candidate_ranks"]): anchor_payload.append((source_seed, atom, target_seed, basis, query_fit.target_factors, raw_result["transport"].target_factors, global_fit.target_factors, query_fit.effective_rank, raw_result["transport"].effective_rank, global_fit.effective_rank))

        row_map = {(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]): row for row in output_rows if row["evaluable"]}
        anchors = [row for row in output_rows if row["query_role"] == "anchor" and row["evaluable"]]
        for row in anchors:
            query_fit, global_fit = factor_map[(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"])]
            query_overlap, global_overlap = [], []
            for neighbor in row["negative_source_atoms"]:
                key = (row["source_seed"], neighbor, row["target_seed"], row["rank"])
                if key not in factor_map or key not in row_map: continue
                other_query, other_global = factor_map[key]
                query_overlap.append(transport_subspace_overlap(query_fit, other_query)); global_overlap.append(transport_subspace_overlap(global_fit, other_global))
            if not query_overlap: raise ValueError("anchor has no evaluable collision neighbor")
            row["query_collision_mean"] = float(np.mean(query_overlap)); row["global_collision_mean"] = float(np.mean(global_overlap)); row["collision_improvement_over_global"] = row["global_collision_mean"] - row["query_collision_mean"]

        groups = {}
        for row in anchors: groups.setdefault((row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"]), []).append(row)
        decisions = []; found = []
        for key, values in sorted(groups.items()):
            selected, last_reason = None, "NO_RANK_PASSED_HOOK_TRANSPORT_GATE"
            for row in sorted(values, key=lambda value: cfg["candidate_ranks"].index(value["rank"])):
                if row["query_status"] != "OK" or row["raw_status"] != "OK" or row["global_status"] != "OK": last_reason = "RANK_DEFICIENT"; continue
                qpos = type("Metrics", (), row["query_positive"])(); qneg = type("Metrics", (), row["query_negative"])()
                if residual_mode and row["source_residual_energy_fraction"] < cfg["minimum_source_residual_energy_fraction"]:
                    last_reason = "SOURCE_RESIDUAL_ENERGY_BELOW_FLOOR"; continue
                raw_control = max(row["raw_specificity"], row["unresidualized_query_specificity"]) if residual_mode else row["raw_specificity"]
                gate = decide_transport_gate(qpos, qneg, rank_boundary_relative_gap=row["source_rank_boundary_relative_gap"], collision_improvement_over_global=row["collision_improvement_over_global"], raw_control_specificity=raw_control, global_control_specificity=row["global_specificity"], minimum_bcc=cfg["minimum_calibration_bcc"], maximum_normalized_residual=cfg["maximum_calibration_normalized_residual"], minimum_specificity=cfg["minimum_calibration_specificity"], minimum_control_advantage=cfg["minimum_control_specificity_advantage"], minimum_collision_improvement=cfg["minimum_collision_improvement_over_global"], minimum_rank_gap=cfg["minimum_rank_boundary_relative_gap"])
                last_reason = gate.reason
                if gate.decision == "FOUND_RELATION": selected = row; break
            decisions.append({"source_seed": key[0], "source_atom": key[1], "target_seed": key[2], "energy_stratum": key[3], "decision": "FOUND_RELATION" if selected else "UNRESOLVED_RELATION", "reason": None if selected else last_reason, "selected_rank": selected["rank"] if selected else None})
            if selected: found.append(selected)
        coverage = len(found) / cfg["anchor_units"]; directions = {(row["source_seed"], row["target_seed"]) for row in found}; all_directions = {(row["source_seed"], row["target_seed"]) for row in anchors}; strata = {row["energy_stratum"] for row in found}
        progression = coverage >= cfg["minimum_progression_coverage"] and len(strata) >= cfg["minimum_covered_strata"] and directions == all_directions
        surface_path = run_dir / "hook_transport_surface.jsonl"; surface_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        decision_path = run_dir / "hook_transport_decisions.jsonl"; decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
        loading_path = run_dir / "anchor_maxrank_hook_factors.npz"; maximum_rank = max(cfg["candidate_ranks"]); np.savez_compressed(loading_path, source_seed=np.asarray([x[0] for x in anchor_payload]), source_atom=np.asarray([x[1] for x in anchor_payload]), target_seed=np.asarray([x[2] for x in anchor_payload]), source_basis=np.stack([x[3] for x in anchor_payload]).astype(np.float32), query_target=np.stack([pad_columns(x[4], maximum_rank) for x in anchor_payload]), raw_target=np.stack([pad_columns(x[5], maximum_rank) for x in anchor_payload]), global_target=np.stack([pad_columns(x[6], maximum_rank) for x in anchor_payload]), query_effective_rank=np.asarray([x[7] for x in anchor_payload]), raw_effective_rank=np.asarray([x[8] for x in anchor_payload]), global_effective_rank=np.asarray([x[9] for x in anchor_payload]))
        decision = ("PROCEED_RESIDUAL_TRANSPORT_TO_MATCHED_CAUSAL_GATE" if progression else "STOP_RESIDUAL_FCC_REPRESENTATION") if residual_mode else ("PROCEED_HOOK_TRANSPORT_TO_MATCHED_CAUSAL_GATE" if progression else "STOP_HOOK_TRANSPORT_REPRESENTATION")
        rank_summaries = {}
        for rank in cfg["candidate_ranks"]:
            values = [row for row in anchors if row["rank"] == rank]
            rank_summaries[str(rank)] = {"units": len(values), "median_query_bcc": float(np.median([row["query_positive"]["bcc"] for row in values])), "minimum_query_residual": float(np.min([row["query_positive"]["normalized_residual"] for row in values])), "median_query_specificity": float(np.median([row["query_specificity"] for row in values])), "median_raw_specificity": float(np.median([row["raw_specificity"] for row in values])), "median_global_specificity": float(np.median([row["global_specificity"] for row in values])), "median_control_advantage": float(np.median([row["query_specificity"] - max(row["raw_specificity"], row["global_specificity"], row.get("unresidualized_query_specificity") if row.get("unresidualized_query_specificity") is not None else -np.inf) for row in values]))}
        checks = {"frozen_inputs_bound": all(bound.values()), "synthetic_pass": synthetic_status == "PASS", "complete_surface_grid": len(output_rows) == cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]), "complete_anchor_decisions": len(decisions) == cfg["anchor_units"], "strong_controls_present": all(all(key in row for key in ("query_positive", "raw_positive", "global_positive")) for row in output_rows if row["evaluable"]), "all_anchor_collisions": all(np.isfinite(row["collision_improvement_over_global"]) for row in anchors), "finite_anchor_metrics": all(np.isfinite([row["query_positive"]["bcc"], row["query_positive"]["normalized_residual"], row["query_specificity"], row["raw_specificity"], row["global_specificity"]]).all() for row in anchors), "no_causal_forward": True, "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"]}
        record = {"checks": {key: bool(value) for key, value in checks.items()}, "screen_decision": decision, "residual_mode": residual_mode, "nuisance_rank": nuisance.rank if nuisance is not None else None, "nuisance_explained_variance_fraction": nuisance.explained_variance_fraction if nuisance is not None else None, "found": len(found), "coverage": coverage, "rank_counts": dict(Counter(row["rank"] for row in found)), "covered_strata": sorted(strata), "covered_directions": len(directions), "progression_pass": bool(progression), "surface_rows": len(output_rows), "decision_rows": len(decisions), "rank_summaries": rank_summaries, "surface_sha256": sha256(surface_path), "decisions_sha256": sha256(decision_path), "loadings_sha256": sha256(loading_path), "wall_seconds": time.perf_counter() - started_compute, "scope_limit": cfg["scope_limit"]}
        status = "PASS" if all(checks.values()) else "FAIL"; write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8"); write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw_path = run_dir / "metrics.raw.jsonl"; raw_path.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path), "generator_script_path": "scripts/run_r011f4_hook_transport_real.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error}); (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "screen_decision": record.get("screen_decision") if record else None}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir); write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)}); print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "screen_decision": record.get("screen_decision") if record else None, "error": error})); return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
