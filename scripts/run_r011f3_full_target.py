"""Run the single scalable full-target C046 FCC screen."""
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
from ccad.fuzzy_correspondence import soft_membership_overlap  # noqa: E402
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402
from run_r011f1_euclidean_surface import condition_weights  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "source": "CCAD frozen artifact", "license_or_access_boundary": "internal", "role": role}


def centered_code_cross(left, right, weights: np.ndarray, left_mean: np.ndarray, right_mean: np.ndarray) -> np.ndarray:
    weighted_right = right.multiply(weights[:, None])
    raw = left.T @ weighted_right
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    left_weighted = np.asarray(left.T @ weights, dtype=np.float64).reshape(-1)
    right_weighted = np.asarray(right.T @ weights, dtype=np.float64).reshape(-1)
    return np.asarray(raw, dtype=np.float64) - np.outer(left_weighted, right_mean) - np.outer(left_mean, right_weighted) + np.outer(left_mean, right_mean)


def contribution_cross(left, right, left_decoder: np.ndarray, right_decoder: np.ndarray, weights: np.ndarray, left_mean: np.ndarray, right_mean: np.ndarray) -> np.ndarray:
    return centered_code_cross(left, right, weights, left_mean, right_mean) * (left_decoder @ right_decoder.T)


def component_process(matrix, dec: np.ndarray, mean: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    outputs = []
    for component in range(loadings.shape[1]):
        coefficients = loadings[:, component]
        raw = matrix.multiply(coefficients[None, :]) @ dec
        mean_vector = (mean * coefficients) @ dec
        outputs.append(np.asarray(raw, dtype=np.float64) - mean_vector[None, :])
    return np.stack(outputs, axis=1)


def component_metrics(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict:
    source_energy = float(np.sum(weights[:, None, None] * source * source))
    target_energy = float(np.sum(weights[:, None, None] * target * target))
    cross_energy = float(np.sum(weights[:, None, None] * source * target))
    residual = max(0.0, source_energy + target_energy - 2.0 * cross_energy)
    return {"source_energy": source_energy, "target_energy": target_energy, "cross_energy": cross_energy, "bcc": 2.0 * cross_energy / (source_energy + target_energy) if source_energy + target_energy > 1e-12 else None, "normalized_residual": residual / source_energy if source_energy > 1e-12 else None}


def fit_pls(source_matrix, target_matrix, source_decoder, target_decoder, source_mean, target_mean, positive_rows, positive_weights, negative_rows=None, negative_weights=None, rank=8):
    source_positive = source_matrix[positive_rows]
    target_positive = target_matrix[positive_rows]
    positive_cross = contribution_cross(source_positive, target_positive, source_decoder, target_decoder, positive_weights, source_mean, target_mean)
    objective = positive_cross.copy()
    if negative_rows is not None:
        objective -= contribution_cross(source_matrix[negative_rows], target_matrix[negative_rows], source_decoder, target_decoder, negative_weights, source_mean, target_mean)
    left, singular, right_t = np.linalg.svd(objective, full_matrices=False)
    left = left[:, :rank]
    right = right_t.T[:, :rank]
    source_process = component_process(source_positive, source_decoder, source_mean, left)
    target_process = component_process(target_positive, target_decoder, target_mean, right)
    source_energy = np.sum(positive_weights[:, None, None] * source_process * source_process, axis=(0, 2))
    target_energy = np.sum(positive_weights[:, None, None] * target_process * target_process, axis=(0, 2))
    if np.any(source_energy <= 1e-12) or np.any(target_energy <= 1e-12):
        raise ValueError("full-target PLS selected negligible-energy component")
    left /= np.sqrt(source_energy)[None, :]
    right /= np.sqrt(target_energy)[None, :]
    source_process = component_process(source_positive, source_decoder, source_mean, left)
    target_process = component_process(target_positive, target_decoder, target_mean, right)
    canonical = np.sum(positive_weights[:, None, None] * source_process * target_process, axis=(0, 2))
    flip = canonical < 0
    right[:, flip] *= -1
    canonical = np.abs(canonical)
    return left, right, canonical, singular


def membership(left: np.ndarray, right: np.ndarray, canonical: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    operator = left[:, :rank] @ np.diag(canonical[:rank]) @ right[:, :rank].T
    magnitude = np.abs(operator)
    coupling = magnitude / np.sum(magnitude)
    return np.sum(coupling, axis=1), np.sum(coupling, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/fuzzy_correspondence.py", ROOT / "scripts/run_r009c_atom_discovery.py", ROOT / "scripts/run_r011f1_euclidean_surface.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    paths = {"protocol": ROOT / cfg["protocol_document"], "reference": ROOT / cfg["reference_surface_path"], "capped": ROOT / cfg["capped_pls_surface_path"], "census": ROOT / cfg["source_census_path"], "sequences": ROOT / cfg["sequence_records_path"], "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json"}
    write_json(run_dir / "inputs.json", {"inputs": [file_entry(args.config.resolve(), "run_protocol")] + [file_entry(path, role) for role, path in paths.items()]})
    write_json(run_dir / "manifest.json", {"schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True, "mean_constants_source_split": "mean", "threshold_source_split": "calibration_with_frozen_meaningful_transfer_gate", "statistics_unit": cfg["statistics_unit"], "device": "cpu", "seeds": cfg["source_seeds"], "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run", "resource_lease_reason": "streamed full-target sparse cross-covariance and selected rank-component energy evaluation", "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()})
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {"reference": sha256(paths["reference"]).lower() == cfg["reference_surface_sha256"], "capped": sha256(paths["capped"]).lower() == cfg["capped_pls_surface_sha256"], "census": sha256(paths["census"]).lower() == cfg["source_census_sha256"], "sequences": sha256(paths["sequences"]).lower() == cfg["sequence_records_sha256"], "asset_manifest": sha256(paths["asset_manifest"]).lower() == cfg["asset_manifest_sha256"]}
        if not all(bound.values()) or cfg["audit_opened"] or cfg["target_candidate_count"] != cfg["num_latents"]: raise ValueError(f"frozen input or full-target boundary mismatch: {bound}")
        reference = [json.loads(line) for line in paths["reference"].read_text(encoding="utf-8").splitlines() if line]
        base_rows = [row for row in reference if row["rank"] == 1]
        census = [json.loads(line) for line in paths["census"].read_text(encoding="utf-8").splitlines() if line]
        stats = {(int(row["seed"]), int(row["atom"])): row for row in census}
        means = {seed: np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64) for seed in cfg["source_seeds"]}
        asset_dir = Path(cfg["bulk_asset_dir"]); manifest = json.loads(paths["asset_manifest"].read_text(encoding="utf-8")); split_tokens = {row["split"]: int(row["tokens"]) for row in manifest["splits"]}
        matrices = {split: {seed: sparse_codes(asset_dir, split, seed, split_tokens[split], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]} for split in cfg["splits"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        sequence_payload = json.loads(paths["sequences"].read_text(encoding="utf-8"))["sequences"]
        global_states = select_document_balanced_states(sequence_payload, split="discovery", count=cfg["global_control_tokens"], token_positions=tuple(cfg["global_control_state_positions"]), salt=cfg["global_control_state_salt"])
        global_rows = np.asarray([int(row["sequence_index"]) * cfg["context_length"] + int(row["token_position"]) for row in global_states], dtype=np.int64); global_weights = np.full(len(global_rows), 1 / len(global_rows))
        target_ids = np.arange(cfg["num_latents"], dtype=np.int64); output_rows = []; memberships = []; anchor_loadings = []; started_compute = time.perf_counter()
        for base in base_rows:
            source_seed, target_seed, atom = int(base["source_seed"]), int(base["target_seed"]), int(base["source_atom"])
            if not base.get("evaluable", False):
                for rank in cfg["candidate_ranks"]: output_rows.append({"source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": int(base["energy_stratum"]), "query_role": base["query_role"], "rank": rank, "evaluable": False, "reason": base.get("reason")})
                continue
            source_ids = np.asarray(base["source_candidate_ids"], dtype=np.int64); negative_atoms = [int(value) for value in base["negative_source_atoms"]]
            drows, dweights = condition_weights(matrices["discovery"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"]); dnrows, dnweights = condition_weights(matrices["discovery"][source_seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
            left, right, canonical, singular = fit_pls(matrices["discovery"][source_seed][:, source_ids], matrices["discovery"][target_seed], decoders[source_seed][source_ids], decoders[target_seed], means[source_seed][source_ids], means[target_seed], drows, dweights, dnrows, dnweights, rank=max(cfg["candidate_ranks"]))
            gleft, gright, gcanonical, _ = fit_pls(matrices["discovery"][source_seed][:, source_ids], matrices["discovery"][target_seed], decoders[source_seed][source_ids], decoders[target_seed], means[source_seed][source_ids], means[target_seed], global_rows, global_weights, rank=max(cfg["candidate_ranks"]))
            crows, cweights = condition_weights(matrices["calibration"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"]); cnrows, cnweights = condition_weights(matrices["calibration"][source_seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
            source_cal = component_process(matrices["calibration"][source_seed][crows][:, source_ids], decoders[source_seed][source_ids], means[source_seed][source_ids], left); target_cal = component_process(matrices["calibration"][target_seed][crows], decoders[target_seed], means[target_seed], right)
            source_neg = component_process(matrices["calibration"][source_seed][cnrows][:, source_ids], decoders[source_seed][source_ids], means[source_seed][source_ids], left); target_neg = component_process(matrices["calibration"][target_seed][cnrows], decoders[target_seed], means[target_seed], right)
            if base["query_role"] == "anchor": anchor_loadings.append((left, right, source_seed, atom, target_seed))
            for rank in cfg["candidate_ranks"]:
                pos = component_metrics(source_cal[:, :rank], target_cal[:, :rank], cweights); neg = component_metrics(source_neg[:, :rank], target_neg[:, :rank], cnweights); smem, tmem = membership(left, right, canonical, rank); _, gtmem = membership(gleft, gright, gcanonical, rank)
                membership_index = len(memberships); memberships.append((tmem.astype(np.float32), gtmem.astype(np.float32)))
                output_rows.append({"source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": int(base["energy_stratum"]), "query_role": base["query_role"], "rank": rank, "evaluable": True, "source_candidate_ids": source_ids.tolist(), "target_candidate_count": cfg["target_candidate_count"], "negative_source_atoms": negative_atoms, "calibration_positive_bcc": pos["bcc"], "calibration_positive_residual": pos["normalized_residual"], "calibration_negative_bcc": neg["bcc"], "calibration_bcc_contrast": pos["bcc"] - neg["bcc"], "rank_boundary_relative_gap": float((singular[rank-1] - singular[rank]) / max(singular[0], np.finfo(float).eps)), "source_membership": smem.tolist(), "membership_index": membership_index})
        row_map = {(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]): row for row in output_rows}; anchors = [row for row in output_rows if row["query_role"] == "anchor"]
        for row in anchors:
            query_membership, global_membership = memberships[row["membership_index"]]; qover, gover = [], []
            for neighbor in row["negative_source_atoms"]:
                other = row_map[(row["source_seed"], neighbor, row["target_seed"], row["rank"])]
                if not other["evaluable"]: continue
                other_query, other_global = memberships[other["membership_index"]]; qover.append(soft_membership_overlap(query_membership, other_query)); gover.append(soft_membership_overlap(global_membership, other_global))
            if not qover: raise ValueError("full-target anchor has no evaluable collision neighbor")
            row["query_collision_mean"] = float(np.mean(qover)); row["global_collision_mean"] = float(np.mean(gover)); row["collision_improvement_over_global"] = row["global_collision_mean"] - row["query_collision_mean"]
        groups = {}
        for row in anchors: groups.setdefault((row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"]), []).append(row)
        decisions = []; found = []
        for key, values in sorted(groups.items()):
            passing = [row for row in sorted(values, key=lambda row: cfg["candidate_ranks"].index(row["rank"])) if row["evaluable"] and row["calibration_positive_bcc"] >= cfg["minimum_calibration_bcc"] and row["calibration_positive_residual"] <= cfg["maximum_calibration_normalized_residual"] and row["calibration_bcc_contrast"] > cfg["minimum_calibration_contrast"] and row["collision_improvement_over_global"] >= cfg["minimum_collision_improvement_over_global"] and row["rank_boundary_relative_gap"] >= cfg["minimum_rank_boundary_relative_gap"]]
            selected = passing[0] if passing else None; decisions.append({"source_seed": key[0], "source_atom": key[1], "target_seed": key[2], "energy_stratum": key[3], "decision": "FOUND_RELATION" if selected else "UNRESOLVED_RELATION", "reason": None if selected else "NO_RANK_PASSED_FULL_TARGET_TRANSFER_GATE", "selected_rank": selected["rank"] if selected else None});
            if selected: found.append(selected)
        coverage = len(found) / cfg["anchor_units"]; directions = {(row["source_seed"], row["target_seed"]) for row in found}; all_directions = {(row["source_seed"], row["target_seed"]) for row in anchors}; strata = {row["energy_stratum"] for row in found}; progression = coverage >= cfg["minimum_progression_coverage"] and len(strata) >= cfg["minimum_covered_strata"] and directions == all_directions
        surface_path = run_dir / "full_target_surface.jsonl"; surface_path.write_text("".join(json.dumps({key: value for key, value in row.items() if key != "_membership"}, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        decision_path = run_dir / "full_target_decisions.jsonl"; decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
        membership_path = run_dir / "target_memberships.npz"; np.savez_compressed(membership_path, query=np.stack([value[0] for value in memberships]), global_control=np.stack([value[1] for value in memberships]))
        loading_path = run_dir / "anchor_maxrank_loadings.npz"; np.savez_compressed(loading_path, source=np.stack([value[0] for value in anchor_loadings]).astype(np.float32), target=np.stack([value[1] for value in anchor_loadings]).astype(np.float32), source_seed=np.asarray([value[2] for value in anchor_loadings]), source_atom=np.asarray([value[3] for value in anchor_loadings]), target_seed=np.asarray([value[4] for value in anchor_loadings]))
        decision = "PROCEED_FULL_TARGET_TO_MATCHED_CAUSAL_GATE" if progression else "STOP_CANDIDATE_TRUNCATION_EXPLANATION"
        checks = {"frozen_inputs_bound": all(bound.values()), "complete_surface_grid": len(output_rows) == cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]), "complete_anchor_decisions": len(decisions) == cfg["anchor_units"], "full_target_exact": all((not row["evaluable"]) or row["target_candidate_count"] == cfg["num_latents"] for row in output_rows), "no_target_square_gram": True, "finite_metrics": all((not row["evaluable"]) or np.isfinite([row["calibration_positive_bcc"], row["calibration_positive_residual"], row["calibration_bcc_contrast"]]).all() for row in output_rows), "all_anchor_collisions": all(np.isfinite(row["collision_improvement_over_global"]) for row in anchors), "no_causal_forward": True, "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"]}
        record = {"checks": {key: bool(value) for key, value in checks.items()}, "screen_decision": decision, "found": len(found), "coverage": coverage, "rank_counts": dict(Counter(row["rank"] for row in found)), "covered_strata": sorted(strata), "covered_directions": len(directions), "progression_pass": bool(progression), "surface_rows": len(output_rows), "decision_rows": len(decisions), "feature_pair_budget": cfg["feature_pair_budget"], "capped_feature_pair_budget": cfg["capped_feature_pair_budget"], "surface_sha256": sha256(surface_path), "decisions_sha256": sha256(decision_path), "memberships_sha256": sha256(membership_path), "loadings_sha256": sha256(loading_path), "wall_seconds": time.perf_counter() - started_compute, "scope_limit": cfg["scope_limit"]}
        status = "PASS" if all(checks.values()) else "FAIL"; write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8"); write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"; raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8"); write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011f3_full_target.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]}); write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error}); (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "screen_decision": record.get("screen_decision") if record else None}) + "\n", encoding="utf-8");
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir); write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)}); print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "screen_decision": record.get("screen_decision") if record else None, "error": error})); return 0 if status == "PASS" and validation.ok else 1

if __name__ == "__main__": raise SystemExit(main())
