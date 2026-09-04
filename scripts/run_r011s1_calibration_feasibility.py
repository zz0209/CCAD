"""Run the frozen, bounded R011-S1 query-conditioned subspace screen."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from ccad.subspace_transport import (  # noqa: E402
    direct_process_transfer_metrics,
    fit_weighted_pca,
    fit_weighted_stitching,
    mean_transfer_metrics,
    projector_subspace_similarity,
    random_orthonormal_basis,
    select_weighted_support,
    stable_seed,
    transfer_metrics,
    weighted_mean,
    weighted_total_energy,
)
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, source: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": "internal", "role": role}


def dense_code(matrix, atom: int) -> np.ndarray:
    return np.asarray(matrix[:, atom].toarray(), dtype=np.float64).reshape(-1)


def reconstruct(matrix, indices: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices] @ dec, dtype=np.float64)


def basis_prefix(basis: np.ndarray, rank: int) -> np.ndarray:
    return basis[:, :rank] if basis.shape[1] >= rank else np.empty((basis.shape[0], 0), dtype=np.float64)


def serial_metrics(value) -> dict:
    return {key: getattr(value, key) for key in value.__dataclass_fields__}


def qualifies(metrics: dict, psc: float | None, mean_metrics: dict, cfg: dict) -> tuple[bool, list[str]]:
    gate = cfg["qualification"]
    reasons = []
    checks = {
        "normalized_residual": metrics["normalized_residual"] is not None and metrics["normalized_residual"] <= gate["maximum_source_normalized_residual"],
        "bcc": metrics["bcc"] is not None and metrics["bcc"] >= gate["minimum_bcc"],
        "psc": psc is not None and psc >= gate["minimum_psc"],
        "source_effect": metrics["source_effect_fraction"] is not None and metrics["source_effect_fraction"] >= gate["minimum_source_effect_fraction"],
        "target_effect": metrics["target_effect_fraction"] is not None and metrics["target_effect_fraction"] >= gate["minimum_target_effect_fraction"],
    }
    source_mean_energy = mean_metrics["source_mean_energy"]
    if source_mean_energy > gate["zero_mean_energy_epsilon"]:
        checks["mean"] = mean_metrics["normalized_mean_residual"] is not None and mean_metrics["normalized_mean_residual"] <= gate["maximum_normalized_mean_residual"]
    else:
        checks["mean"] = mean_metrics["mean_residual"] <= gate["zero_mean_absolute_residual"]
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return all(checks.values()), reasons


def support_record(code: np.ndarray, cfg: dict, split: str) -> dict:
    support = select_weighted_support(code, cfg["max_condition_tokens_per_split"], cfg["condition_weight_power"])
    active_sequences = int(np.unique(support.indices // cfg["context_length"]).size)
    eligible = (
        support.effective_sample_size >= cfg["minimum_effective_sample_size"]
        and active_sequences >= cfg["minimum_active_sequences"][split]
    )
    return {"support": support, "active_sequences": active_sequences, "eligible": bool(eligible)}


def deterministic_sample(total: int, count: int, *seed_parts: object) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(*seed_parts))
    return np.sort(rng.choice(total, size=min(total, count), replace=False)).astype(np.int64)


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
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/subspace_transport.py", ROOT / "scripts/run_r009c_atom_discovery.py"]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    query_path = ROOT / cfg["query_panel_path"]
    native_path = ROOT / cfg["native_calibration_path"]
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    raw_dir = Path(cfg["raw_hook_asset_dir"])
    raw_manifest_path = raw_dir / "raw_hook_manifest.json"
    inputs = [
        file_entry(args.config.resolve(), "CCAD frozen config", "protocol"),
        file_entry(query_path, "R009b", "source_query_panel"),
        file_entry(native_path, "R011c", "native_calibration_control"),
        file_entry(asset_manifest, "R008b", "paired_sparse_code_manifest"),
        file_entry(raw_manifest_path, "R011-S1 raw-hook asset", "raw_hook_manifest"),
    ]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True,
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": "cpu", "seeds": cfg["source_seeds"],
        "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded randomized SVD over pre-audit sparse-code reconstructions and raw-hook memmaps",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {
            "query_panel": sha256(query_path).lower() == cfg["query_panel_sha256"].lower(),
            "native_calibration": sha256(native_path).lower() == cfg["native_calibration_sha256"].lower(),
            "asset_manifest": sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower(),
            "raw_hook_manifest": sha256(raw_manifest_path).lower() == cfg["raw_hook_manifest_sha256"].lower(),
        }
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        if cfg["candidate_ranks"] != [1, 2, 4, 8, 16] or cfg["forbidden_splits"] != ["audit"] or cfg["audit_opened"]:
            raise ValueError("rank/split contract drift")
        query_rows = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line]
        grouped = defaultdict(list)
        for row in query_rows:
            grouped[(row["seed"], row["energy_stratum"])].append(row)
        selected = [min(grouped[(seed, stratum)], key=lambda row: (row["selection_hash"], row["atom"]))
                    for seed in cfg["source_seeds"] for stratum in range(cfg["strata_per_seed"])]
        if len(selected) != len(cfg["source_seeds"]) * cfg["queries_per_seed"]:
            raise ValueError("query subset is incomplete")
        native_rows = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines() if line]
        native_lookup = {(row["source_seed"], row["target_seed"], row["source_atom"]): row for row in native_rows}

        matrices, decoders = {}, {}
        for split, tokens in ((cfg["mean_split"], cfg["mean_tokens"]), (cfg["discovery_split"], cfg["discovery_tokens"]), (cfg["calibration_split"], cfg["calibration_tokens"])):
            matrices[split] = {seed: sparse_codes(asset_dir, split, seed, tokens, cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        raw_meta = {row["split"]: row for row in raw_manifest["splits"]}
        raw = {split: np.memmap(raw_meta[split]["path"], dtype="<f4", mode="r").reshape(raw_meta[split]["shape"])
               for split in (cfg["mean_split"], cfg["discovery_split"], cfg["calibration_split"])}

        global_mean_indices = deterministic_sample(cfg["mean_tokens"], cfg["global_sample_tokens"], cfg["global_sample_salt"], "mean")
        global_discovery_indices = deterministic_sample(cfg["discovery_tokens"], cfg["global_sample_tokens"], cfg["global_sample_salt"], "discovery")
        uniform_mean_weights = np.full(global_mean_indices.size, 1.0 / global_mean_indices.size)
        uniform_discovery_weights = np.full(global_discovery_indices.size, 1.0 / global_discovery_indices.size)
        global_bases = {}
        for seed in cfg["source_seeds"]:
            gmean_samples = reconstruct(matrices[cfg["mean_split"]][seed], global_mean_indices, decoders[seed])
            gmean = weighted_mean(gmean_samples, uniform_mean_weights)
            gsamples = reconstruct(matrices[cfg["discovery_split"]][seed], global_discovery_indices, decoders[seed])
            global_bases[seed], _ = fit_weighted_pca(
                gsamples, uniform_discovery_weights, gmean, max(cfg["candidate_ranks"]),
                random_seed=stable_seed(cfg["global_sample_salt"], seed),
                oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                relative_tolerance=cfg["eigenvalue_relative_tolerance"],
            )

        output_rows, query_ledger, projector_payload = [], [], {}
        primary_decisions = []
        started_compute = time.perf_counter()
        for query in selected:
            source_seed, source_atom = int(query["seed"]), int(query["atom"])
            query_key = f"s{source_seed}_a{source_atom}"
            supports = {}
            for split in (cfg["mean_split"], cfg["discovery_split"], cfg["calibration_split"]):
                supports[split] = support_record(dense_code(matrices[split][source_seed], source_atom), cfg, split)
            query_eligible = all(item["eligible"] for item in supports.values())
            query_ledger.append({
                "source_seed": source_seed, "source_atom": source_atom, "energy_stratum": int(query["energy_stratum"]),
                "selection_hash": query["selection_hash"], "eligible": query_eligible,
                "split_support": {split: {
                    "active_tokens": supports[split]["support"].active_count,
                    "selected_tokens": int(supports[split]["support"].indices.size),
                    "active_sequences": supports[split]["active_sequences"],
                    "effective_sample_size": supports[split]["support"].effective_sample_size,
                    "retained_weight_fraction": supports[split]["support"].retained_weight_fraction,
                    "eligible": supports[split]["eligible"],
                } for split in supports},
            })
            if not query_eligible:
                for target_seed in cfg["source_seeds"]:
                    if target_seed != source_seed:
                        primary_decisions.append({"source_seed": source_seed, "target_seed": target_seed, "source_atom": source_atom,
                                                  "energy_stratum": int(query["energy_stratum"]), "identification": "UNRESOLVED",
                                                  "reason": "SOURCE_CONDITION_INSUFFICIENT", "minimum_rank": None})
                continue
            samples, means, bases = {}, {}, {}
            for seed in cfg["source_seeds"]:
                samples[seed] = {}
                for split in supports:
                    selected_support = supports[split]["support"]
                    samples[seed][split] = reconstruct(matrices[split][seed], selected_support.indices, decoders[seed])
                means[seed] = weighted_mean(samples[seed][cfg["mean_split"]], supports[cfg["mean_split"]]["support"].weights)
                bases[seed], eigenvalues = fit_weighted_pca(
                    samples[seed][cfg["discovery_split"]], supports[cfg["discovery_split"]]["support"].weights,
                    means[seed], max(cfg["candidate_ranks"]), random_seed=stable_seed(cfg["run_id"], query_key, seed),
                    oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                    relative_tolerance=cfg["eigenvalue_relative_tolerance"],
                )
                projector_payload[f"{query_key}_seed{seed}_basis"] = bases[seed].astype(np.float32)
                projector_payload[f"{query_key}_seed{seed}_mean"] = means[seed].astype(np.float32)

            raw_mean_support = supports[cfg["mean_split"]]["support"]
            raw_disc_support = supports[cfg["discovery_split"]]["support"]
            raw_cal_support = supports[cfg["calibration_split"]]["support"]
            raw_mean = weighted_mean(np.asarray(raw[cfg["mean_split"]][raw_mean_support.indices], dtype=np.float64), raw_mean_support.weights)
            raw_basis, _ = fit_weighted_pca(
                np.asarray(raw[cfg["discovery_split"]][raw_disc_support.indices], dtype=np.float64), raw_disc_support.weights,
                raw_mean, max(cfg["candidate_ranks"]), random_seed=stable_seed(cfg["run_id"], query_key, "raw"),
                oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                relative_tolerance=cfg["eigenvalue_relative_tolerance"],
            )
            raw_calibration = np.asarray(raw[cfg["calibration_split"]][raw_cal_support.indices], dtype=np.float64)
            projector_payload[f"{query_key}_raw_basis"] = raw_basis.astype(np.float32)
            projector_payload[f"{query_key}_raw_mean"] = raw_mean.astype(np.float32)

            for target_seed in cfg["source_seeds"]:
                if target_seed == source_seed:
                    continue
                pair_key = (source_seed, target_seed, source_atom)
                native = native_lookup[pair_key]
                mean_metrics = mean_transfer_metrics(means[source_seed], means[target_seed], cfg["qualification"]["zero_mean_energy_epsilon"])
                stitching_left, stitching_right, _ = fit_weighted_stitching(
                    samples[source_seed][cfg["discovery_split"]], samples[target_seed][cfg["discovery_split"]],
                    supports[cfg["discovery_split"]]["support"].weights, means[source_seed], means[target_seed],
                    max(cfg["candidate_ranks"]), random_seed=stable_seed(cfg["run_id"], query_key, target_seed, "stitching"),
                    oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                    relative_tolerance=cfg["eigenvalue_relative_tolerance"],
                )
                pair_primary = []
                for rank in cfg["candidate_ranks"]:
                    method_bases = {
                        "SAE_QUERY_CONDITIONAL_PCA": (basis_prefix(bases[source_seed], rank), basis_prefix(bases[target_seed], rank)),
                        "RAW_HOOK_QUERY_CONDITIONAL_PCA": (basis_prefix(raw_basis, rank), basis_prefix(raw_basis, rank)),
                        "GLOBAL_SAE_PCA": (basis_prefix(global_bases[source_seed], rank), basis_prefix(global_bases[target_seed], rank)),
                        "MATCHED_RANK_RANDOM": (
                            random_orthonormal_basis(cfg["hook_hidden_size"], rank, stable_seed(cfg["random_projector_salt"], query_key, source_seed, rank)),
                            random_orthonormal_basis(cfg["hook_hidden_size"], rank, stable_seed(cfg["random_projector_salt"], query_key, target_seed, rank)),
                        ),
                        "RELAXED_PAIRED_STITCHING": (basis_prefix(stitching_left, rank), basis_prefix(stitching_right, rank)),
                    }
                    for method, (left_basis, right_basis) in method_bases.items():
                        if left_basis.shape[1] != rank or right_basis.shape[1] != rank:
                            metrics = {name: None for name in serial_metrics(direct_process_transfer_metrics(np.ones((1, 1)), np.ones((1, 1)), np.ones(1))).keys()}
                            psc = {"psc": None, "projector_distance_sq": None, "rank_left": left_basis.shape[1], "rank_right": right_basis.shape[1], "principal_cosines": []}
                            passed, failures = False, ["numerical_rank"]
                        elif method == "RAW_HOOK_QUERY_CONDITIONAL_PCA":
                            value = transfer_metrics(raw_calibration, raw_calibration, raw_cal_support.weights, raw_mean, raw_mean, left_basis, right_basis)
                            metrics = serial_metrics(value)
                            psc = projector_subspace_similarity(left_basis, right_basis)
                            raw_means = mean_transfer_metrics(raw_mean, raw_mean)
                            passed, failures = qualifies(metrics, psc["psc"], raw_means, cfg)
                        else:
                            value = transfer_metrics(
                                samples[source_seed][cfg["calibration_split"]], samples[target_seed][cfg["calibration_split"]],
                                supports[cfg["calibration_split"]]["support"].weights, means[source_seed], means[target_seed],
                                left_basis, right_basis,
                            )
                            metrics = serial_metrics(value)
                            psc = projector_subspace_similarity(left_basis, right_basis)
                            passed, failures = qualifies(metrics, psc["psc"], mean_metrics, cfg)
                        row = {
                            "source_seed": source_seed, "target_seed": target_seed, "source_atom": source_atom,
                            "energy_stratum": int(query["energy_stratum"]), "method": method, "rank": rank,
                            **metrics, "psc": psc["psc"], "projector_distance_sq": psc["projector_distance_sq"],
                            "normalized_mean_residual": (0.0 if method == "RAW_HOOK_QUERY_CONDITIONAL_PCA" else mean_metrics["normalized_mean_residual"]),
                            "qualifies": bool(passed), "qualification_failures": failures,
                        }
                        output_rows.append(row)
                        if method == cfg["primary_method"]:
                            pair_primary.append(row)

                target_atom = int(native["method_metrics"]["best_bcc_native_size1"]["support"][0])
                cal_indices = supports[cfg["calibration_split"]]["support"].indices
                cal_weights = supports[cfg["calibration_split"]]["support"].weights
                mean_indices = supports[cfg["mean_split"]]["support"].indices
                mean_weights = supports[cfg["mean_split"]]["support"].weights
                source_code_cal = dense_code(matrices[cfg["calibration_split"]][source_seed], source_atom)[cal_indices]
                target_code_cal = dense_code(matrices[cfg["calibration_split"]][target_seed], target_atom)[cal_indices]
                source_code_mean = dense_code(matrices[cfg["mean_split"]][source_seed], source_atom)[mean_indices]
                target_code_mean = dense_code(matrices[cfg["mean_split"]][target_seed], target_atom)[mean_indices]
                source_process = (source_code_cal - float(mean_weights @ source_code_mean))[:, None] * decoders[source_seed][source_atom][None, :]
                target_process = (target_code_cal - float(mean_weights @ target_code_mean))[:, None] * decoders[target_seed][target_atom][None, :]
                source_total = weighted_total_energy(samples[source_seed][cfg["calibration_split"]], cal_weights, means[source_seed])
                target_total = weighted_total_energy(samples[target_seed][cfg["calibration_split"]], cal_weights, means[target_seed])
                single_value = direct_process_transfer_metrics(source_process, target_process, cal_weights, source_total_energy=source_total, target_total_energy=target_total)
                single_psc = projector_subspace_similarity(decoders[source_seed][source_atom][:, None] / np.linalg.norm(decoders[source_seed][source_atom]), decoders[target_seed][target_atom][:, None] / np.linalg.norm(decoders[target_seed][target_atom]))
                single_mean = mean_transfer_metrics(float(mean_weights @ source_code_mean) * decoders[source_seed][source_atom], float(mean_weights @ target_code_mean) * decoders[target_seed][target_atom])
                single_metrics = serial_metrics(single_value)
                single_pass, single_failures = qualifies(single_metrics, single_psc["psc"], single_mean, cfg)
                output_rows.append({
                    "source_seed": source_seed, "target_seed": target_seed, "source_atom": source_atom,
                    "energy_stratum": int(query["energy_stratum"]), "method": "BEST_FUNCTIONAL_SINGLE_NATIVE", "rank": 1,
                    "target_atom": target_atom, **single_metrics, "psc": single_psc["psc"],
                    "projector_distance_sq": single_psc["projector_distance_sq"],
                    "normalized_mean_residual": single_mean["normalized_mean_residual"], "qualifies": bool(single_pass),
                    "qualification_failures": single_failures,
                })
                output_rows.append({
                    "source_seed": source_seed, "target_seed": target_seed, "source_atom": source_atom,
                    "energy_stratum": int(query["energy_stratum"]), "method": "MSCC_NATIVE_REFUSAL", "rank": None,
                    "native_identification": native["mscc_identification"], "native_reason": native["mscc_reason"],
                    "qualifies": False, "qualification_failures": ["native_refusal"],
                })
                accepted = next((row for row in pair_primary if row["qualifies"]), None)
                primary_decisions.append({
                    "source_seed": source_seed, "target_seed": target_seed, "source_atom": source_atom,
                    "energy_stratum": int(query["energy_stratum"]),
                    "identification": "FOUND_SUBSPACE" if accepted else "UNRESOLVED",
                    "reason": "MINIMUM_RANK_PASSES_CALIBRATION" if accepted else "NO_FIXED_RANK_PASSES_CALIBRATION",
                    "minimum_rank": accepted["rank"] if accepted else None,
                })

        output_path = run_dir / "calibration_subspace_metrics.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        decision_path = run_dir / "primary_decisions.jsonl"
        with decision_path.open("w", encoding="utf-8") as stream:
            for row in primary_decisions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        ledger_path = run_dir / "query_condition_ledger.jsonl"
        with ledger_path.open("w", encoding="utf-8") as stream:
            for row in query_ledger:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        projector_path = run_dir / "projectors_and_means.npz"
        np.savez_compressed(projector_path, **projector_payload)

        eligible_queries = sum(row["eligible"] for row in query_ledger)
        eligible_pairs = eligible_queries * (len(cfg["source_seeds"]) - 1)
        found = [row for row in primary_decisions if row["identification"] == "FOUND_SUBSPACE"]
        primary_coverage = len(found) / eligible_pairs if eligible_pairs else 0.0
        covered_strata = sorted(set(row["energy_stratum"] for row in found))
        method_rank_summary = {}
        for method in cfg["methods"]:
            method_rows = [row for row in output_rows if row["method"] == method and row.get("rank") is not None]
            if not method_rows:
                continue
            method_rank_summary[method] = {}
            for rank in sorted(set(row["rank"] for row in method_rows)):
                rank_rows = [row for row in method_rows if row["rank"] == rank]
                bcc = [row["bcc"] for row in rank_rows if row.get("bcc") is not None]
                residual = [row["normalized_residual"] for row in rank_rows if row.get("normalized_residual") is not None]
                psc = [row["psc"] for row in rank_rows if row.get("psc") is not None]
                method_rank_summary[method][str(rank)] = {
                    "rows": len(rank_rows), "qualified": sum(row["qualifies"] for row in rank_rows),
                    "qualification_coverage": sum(row["qualifies"] for row in rank_rows) / len(rank_rows),
                    "bcc_median": float(np.median(bcc)) if bcc else None,
                    "normalized_residual_median": float(np.median(residual)) if residual else None,
                    "psc_median": float(np.median(psc)) if psc else None,
                }
        raw_any = Counter()
        for row in output_rows:
            if row["method"] == "RAW_HOOK_QUERY_CONDITIONAL_PCA" and row["qualifies"]:
                raw_any[(row["source_seed"], row["target_seed"], row["source_atom"])] += 1
        raw_coverage = len(raw_any) / eligible_pairs if eligible_pairs else 0.0
        rule = cfg["screen_progression_rule"]
        if primary_coverage < rule["minimum_primary_coverage"] or len(covered_strata) < rule["minimum_covered_strata"]:
            screen_decision = "STOP_LOW_SCT_CALIBRATION_COVERAGE"
        elif raw_coverage >= primary_coverage:
            screen_decision = "PROCEED_ONLY_WITH_RAW_HOOK_CRITICAL_CAUSAL_SCREEN"
        else:
            screen_decision = "PROCEED_TO_BOUNDED_CAUSAL_SCREEN"
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "source_only_query_subset": len(selected) == 40 and all(len(grouped[(seed, stratum)]) == 16 for seed in cfg["source_seeds"] for stratum in range(cfg["strata_per_seed"])),
            "independent_mean_split": cfg["mean_constants_source_split"] == "mean",
            "discovery_fit_calibration_evaluation": cfg["threshold_source_split"].startswith("frozen_before_calibration"),
            "complete_primary_decision_grid": len(primary_decisions) == 40 * 4,
            "all_fixed_ranks_reported_for_eligible_pairs": len([row for row in output_rows if row["method"] == cfg["primary_method"]]) == eligible_pairs * len(cfg["candidate_ranks"]),
            "strong_controls_present": set(cfg["methods"]) == {row["method"] for row in output_rows},
            "finite_reported_metrics": all(np.isfinite(value) for row in output_rows for key, value in row.items() if key in {"source_energy", "target_energy", "cross_energy", "residual_energy", "normalized_residual", "bcc", "source_effect_fraction", "target_effect_fraction", "psc", "projector_distance_sq", "normalized_mean_residual"} and value is not None),
            "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"],
            "no_causal_claim": "no causal endpoint" in cfg["scope_limit"],
        }
        record = {
            "checks": checks, "selected_queries": len(selected), "eligible_queries": eligible_queries,
            "eligible_ordered_pairs": eligible_pairs, "primary_found_subspace": len(found),
            "primary_coverage": primary_coverage, "primary_minimum_rank_counts": dict(sorted(Counter(row["minimum_rank"] for row in found).items())),
            "covered_energy_strata": covered_strata, "raw_hook_any_rank_coverage": raw_coverage,
            "screen_decision": screen_decision, "method_rank_summary": method_rank_summary,
            "query_condition_ledger_sha256": sha256(ledger_path), "metrics_output_sha256": sha256(output_path),
            "primary_decisions_sha256": sha256(decision_path), "projectors_sha256": sha256(projector_path),
            "wall_seconds": time.perf_counter() - started_compute,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw_metrics = run_dir / "metrics.raw.jsonl"
    raw_metrics.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_metrics),
        "generator_script_path": "scripts/run_r011s1_calibration_feasibility.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    })
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
