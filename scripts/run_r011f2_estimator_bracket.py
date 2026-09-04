"""Run the bounded C045 FCC estimator bracket on frozen local kernels."""
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
from ccad.fuzzy_correspondence import (  # noqa: E402
    evaluate_fixed_correspondence_from_kernels,
    fit_cross_covariance_relation_from_kernels,
    soft_membership_overlap,
)
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402
from run_r011f1_euclidean_surface import condition_weights, embedded_membership, local_kernels  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> tuple[dict, Path | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    inherited = raw.get("inherits_config")
    if inherited is None:
        return raw, None
    if set(raw) != {"inherits_config", "overrides"}:
        raise ValueError("suffix config may contain only inherits_config and overrides")
    base_path = ROOT / inherited
    base = json.loads(base_path.read_text(encoding="utf-8"))
    merged = {**base, **raw["overrides"]}
    merged["inherited_config_path"] = inherited
    merged["inherited_config_sha256"] = sha256(base_path)
    return merged, base_path


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "source": "CCAD frozen artifact", "license_or_access_boundary": "internal", "role": role}


def fixed(kernels, relation, negative: bool = False):
    return evaluate_fixed_correspondence_from_kernels(
        kernels.negative_source_gram if negative else kernels.source_gram,
        kernels.negative_target_gram if negative else kernels.target_gram,
        kernels.negative_cross_gram if negative else kernels.cross_gram,
        relation.source_loadings, relation.target_loadings,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg, inherited_config_path = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/fuzzy_correspondence.py", ROOT / "scripts/run_r011f1_euclidean_surface.py", ROOT / "scripts/run_r009c_atom_discovery.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    paths = {
        "protocol": ROOT / cfg["protocol_document"], "reference_surface": ROOT / cfg["reference_surface_path"],
        "source_census": ROOT / cfg["source_census_path"], "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json",
    }
    inputs = [file_entry(args.config.resolve(), "run_protocol")]
    if inherited_config_path is not None:
        inputs.append(file_entry(inherited_config_path, "inherited_protocol"))
    inputs.extend(file_entry(path, role) for role, path in paths.items())
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True,
        "mean_constants_source_split": "mean", "threshold_source_split": "calibration_with_thresholds_frozen_in_config",
        "statistics_unit": cfg["statistics_unit"], "device": "cpu", "seeds": cfg["source_seeds"],
        "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded sparse-kernel construction and two-estimator linear algebra over frozen paired assets",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        bound = {
            "reference_surface": sha256(paths["reference_surface"]).lower() == cfg["reference_surface_sha256"],
            "source_census": sha256(paths["source_census"]).lower() == cfg["source_census_sha256"],
            "asset_manifest": sha256(paths["asset_manifest"]).lower() == cfg["asset_manifest_sha256"],
        }
        if not all(bound.values()) or cfg["audit_opened"] or cfg["forbidden_splits"] != ["audit"]:
            raise ValueError(f"frozen input or audit boundary mismatch: {bound}")
        reference = [json.loads(line) for line in paths["reference_surface"].read_text(encoding="utf-8").splitlines() if line]
        reference_map = {(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]): row for row in reference}
        base_rows = [row for row in reference if row["rank"] == 1]
        if len(base_rows) != cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"]:
            raise ValueError("reference condition grid mismatch")
        census = [json.loads(line) for line in paths["source_census"].read_text(encoding="utf-8").splitlines() if line]
        stats = {(int(row["seed"]), int(row["atom"])): row for row in census}
        means = {seed: np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64) for seed in cfg["source_seeds"]}
        asset_dir = Path(cfg["bulk_asset_dir"])
        manifest = json.loads(paths["asset_manifest"].read_text(encoding="utf-8"))
        split_tokens = {row["split"]: int(row["tokens"]) for row in manifest["splits"]}
        matrices = {split: {seed: sparse_codes(asset_dir, split, seed, split_tokens[split], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]} for split in cfg["splits"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}

        output_rows = []
        loading_rows = []
        started_compute = time.perf_counter()
        for base in base_rows:
            source_seed, target_seed, atom = int(base["source_seed"]), int(base["target_seed"]), int(base["source_atom"])
            source_ids = [int(value) for value in base["source_candidate_ids"]]
            target_ids = [int(value) for value in base["target_candidate_ids"]]
            negative_atoms = [int(value) for value in base["negative_source_atoms"]]
            split_kernels = {}
            for split in cfg["splits"]:
                positive_rows, positive_weights = condition_weights(matrices[split][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                negative_rows, negative_weights = condition_weights(matrices[split][source_seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
                split_kernels[split] = local_kernels(
                    matrices[split][source_seed], matrices[split][target_seed], source_ids, target_ids,
                    decoders[source_seed], decoders[target_seed], means[source_seed], means[target_seed],
                    positive_rows, positive_weights, negative_rows, negative_weights,
                )
            for estimator in cfg["estimators"]:
                for rank in cfg["candidate_ranks"]:
                    relation = fit_cross_covariance_relation_from_kernels(
                        split_kernels["discovery"], rank=rank, estimator=estimator,
                        contrast_strength=cfg["contrast_strength"], ridge_fraction=cfg["ridge_fraction"],
                    )
                    positive = fixed(split_kernels["calibration"], relation)
                    negative = fixed(split_kernels["calibration"], relation, negative=True)
                    loading_index = len(loading_rows)
                    loading_rows.append((relation.source_loadings, relation.target_loadings, len(target_ids)))
                    output_rows.append({
                        "estimator": estimator, "source_seed": source_seed, "target_seed": target_seed,
                        "source_atom": atom, "energy_stratum": int(base["energy_stratum"]), "query_role": base["query_role"],
                        "rank": rank, "source_candidate_ids": source_ids, "target_candidate_ids": target_ids,
                        "negative_source_atoms": negative_atoms, "calibration_positive_bcc": positive.bcc,
                        "calibration_positive_residual": positive.normalized_residual,
                        "calibration_negative_bcc": negative.bcc,
                        "calibration_bcc_contrast": positive.bcc - negative.bcc,
                        "rank_boundary_relative_gap": relation.rank_boundary_relative_gap,
                        "source_membership": relation.source_membership.tolist(), "target_membership": relation.target_membership.tolist(),
                        "source_effective_support": relation.source_effective_support, "target_effective_support": relation.target_effective_support,
                        "global_collision_mean": reference_map[(source_seed, atom, target_seed, rank)].get("global_collision_mean"),
                        "loading_index": loading_index,
                    })

        row_map = {(row["estimator"], row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]): row for row in output_rows}
        anchor_rows = [row for row in output_rows if row["query_role"] == "anchor"]
        for row in anchor_rows:
            overlaps = []
            anchor = embedded_membership(row["target_candidate_ids"], np.asarray(row["target_membership"]), cfg["num_latents"])
            for neighbor in row["negative_source_atoms"]:
                other = row_map[(row["estimator"], row["source_seed"], neighbor, row["target_seed"], row["rank"])]
                embedded = embedded_membership(other["target_candidate_ids"], np.asarray(other["target_membership"]), cfg["num_latents"])
                overlaps.append(soft_membership_overlap(anchor, embedded))
            row["query_collision_mean"] = float(np.mean(overlaps))
            row["collision_improvement_over_global"] = row["global_collision_mean"] - row["query_collision_mean"]

        decisions = []
        summaries = {}
        all_directions = {(row["source_seed"], row["target_seed"]) for row in anchor_rows}
        for estimator in cfg["estimators"]:
            estimator_anchors = [row for row in anchor_rows if row["estimator"] == estimator]
            groups = {}
            for row in estimator_anchors:
                groups.setdefault((row["source_seed"], row["source_atom"], row["target_seed"], row["energy_stratum"]), []).append(row)
            found = []
            for key, rows in sorted(groups.items()):
                ordered = sorted(rows, key=lambda row: cfg["candidate_ranks"].index(row["rank"]))
                passing = [row for row in ordered if row["calibration_positive_bcc"] >= cfg["minimum_calibration_bcc"] and row["calibration_positive_residual"] <= cfg["maximum_calibration_normalized_residual"] and row["calibration_bcc_contrast"] > cfg["minimum_calibration_contrast"] and row["collision_improvement_over_global"] >= cfg["minimum_collision_improvement_over_global"] and row["rank_boundary_relative_gap"] is not None and row["rank_boundary_relative_gap"] >= cfg["minimum_rank_boundary_relative_gap"]]
                selected = passing[0] if passing else None
                decision = {"estimator": estimator, "source_seed": key[0], "source_atom": key[1], "target_seed": key[2], "energy_stratum": key[3], "decision": "FOUND_RELATION" if selected else "UNRESOLVED_RELATION", "reason": None if selected else "NO_RANK_PASSED_MEANINGFUL_TRANSFER_GATE", "selected_rank": selected["rank"] if selected else None, "loading_index": selected["loading_index"] if selected else None}
                decisions.append(decision)
                if selected:
                    found.append(selected)
            coverage = len(found) / cfg["anchor_units"]
            directions = {(row["source_seed"], row["target_seed"]) for row in found}
            strata = {row["energy_stratum"] for row in found}
            progression = coverage >= cfg["minimum_progression_coverage"] and len(strata) >= cfg["minimum_covered_strata"] and (directions == all_directions if cfg["require_all_represented_ordered_directions"] else True)
            summaries[estimator] = {"found": len(found), "coverage": coverage, "rank_counts": dict(Counter(row["rank"] for row in found)), "covered_strata": sorted(strata), "covered_directions": len(directions), "progression_pass": bool(progression), "median_found_bcc": float(np.median([row["calibration_positive_bcc"] for row in found])) if found else None, "median_found_residual": float(np.median([row["calibration_positive_residual"] for row in found])) if found else None}

        output = run_dir / "estimator_surface.jsonl"
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        decision_path = run_dir / "estimator_decisions.jsonl"
        decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
        max_rank = max(cfg["candidate_ranks"])
        source_loadings = np.zeros((len(loading_rows), 32, max_rank), dtype=np.float32)
        target_loadings = np.zeros((len(loading_rows), 128, max_rank), dtype=np.float32)
        target_counts = np.zeros(len(loading_rows), dtype=np.int16)
        ranks = np.zeros(len(loading_rows), dtype=np.int8)
        for index, (left, right, count) in enumerate(loading_rows):
            source_loadings[index, :left.shape[0], :left.shape[1]] = left
            target_loadings[index, :right.shape[0], :right.shape[1]] = right
            target_counts[index], ranks[index] = count, left.shape[1]
        loadings_path = run_dir / "estimator_loadings.npz"
        np.savez_compressed(loadings_path, source_loadings=source_loadings, target_loadings=target_loadings, target_counts=target_counts, ranks=ranks)
        decision = "PROCEED_WITH_SINGLE_QUALIFIED_ESTIMATOR" if sum(summary["progression_pass"] for summary in summaries.values()) == 1 else ("STOP_LOCAL_CONTRIBUTION_KERNEL_FAMILY" if not any(summary["progression_pass"] for summary in summaries.values()) else "FAIL_MULTIPLE_ESTIMATORS_QUALIFIED_NO_SELECTION_RULE")
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "estimator_set_exact": cfg["estimators"] == ["ENERGY_BALANCED_PLS", "DIAGONAL_WHITENED_CORRELATION"],
            "complete_surface_grid": len(output_rows) == len(cfg["estimators"]) * cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"]),
            "complete_anchor_decisions": len(decisions) == len(cfg["estimators"]) * cfg["anchor_units"],
            "all_collisions_computed": all(np.isfinite(row["collision_improvement_over_global"]) for row in anchor_rows),
            "candidate_budgets_unchanged": all(len(row["source_candidate_ids"]) == 32 and len(row["target_candidate_ids"]) <= 128 for row in output_rows),
            "finite_metrics": all(np.isfinite([row["calibration_positive_bcc"], row["calibration_positive_residual"], row["calibration_negative_bcc"], row["calibration_bcc_contrast"]]).all() for row in output_rows),
            "no_causal_forward": True,
            "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"],
        }
        record = {"checks": {name: bool(value) for name, value in checks.items()}, "screen_decision": decision, "estimator_summaries": summaries, "surface_rows": len(output_rows), "decision_rows": len(decisions), "surface_sha256": sha256(output), "decisions_sha256": sha256(decision_path), "loadings_sha256": sha256(loadings_path), "wall_seconds": time.perf_counter() - started_compute, "scope_limit": cfg["scope_limit"]}
        status = "PASS" if all(checks.values()) and decision != "FAIL_MULTIPLE_ESTIMATORS_QUALIFIED_NO_SELECTION_RULE" else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011f2_estimator_bracket.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "screen_decision": record.get("screen_decision") if record else None}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "screen_decision": record.get("screen_decision") if record else None, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
