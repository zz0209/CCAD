from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccad.metrics import (  # noqa: E402
    absolute_code_correlation,
    adjusted_rand_index,
    bcc_from_kernels,
    cancellation_diagnostics,
    center_codes,
    contribution_kernel,
    document_bootstrap_bcc,
    explicit_group_contribution,
    projector_subspace_consistency,
    pw_mcc_absolute_cosine,
    occupancy_effective_sample_size,
)
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.matching import (  # noqa: E402
    BalancedCandidate,
    exhaustive_balanced_pairs,
    exhaustive_balanced_search,
    enumerate_maximum_exact_covers,
    forced_partition_projection,
)
from ccad.synthetic import (  # noqa: E402
    cooccurrence_confounding_seeded,
    competing_covers_seeded,
    cancellation_seeded,
    hadamard_gauge_instance,
    hadamard_gauge_seeded,
    local_block_rotations,
    local_block_rotations_seeded,
    non_lipschitz_downstream_cliff_seeded,
    partial_overlap_seeded,
    rare_occupancy_seeded,
    same_span_different_computation,
    same_span_different_computation_seeded,
    same_sum_bloated_span,
    same_sum_bloated_span_seeded,
    unequal_split_merge_seeded,
    whole_dictionary_only_seeded,
)


def stable_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def candidate_json(candidate: BalancedCandidate) -> dict:
    return {
        "left_ids": list(candidate.left_ids),
        "right_ids": list(candidate.right_ids),
        "normalized_residual": candidate.normalized_residual,
    }


def search_json(result, neighborhood_kind: str, left_pool: tuple[int, ...], right_pool: tuple[int, ...]) -> dict:
    return {
        "neighborhood_kind": neighborhood_kind,
        "left_pool": list(left_pool),
        "right_pool": list(right_pool),
        "all_candidates": [candidate_json(item) for item in result.all_candidates],
        "passing_candidates": [candidate_json(item) for item in result.passing_candidates],
        "support_minimal_candidates": [candidate_json(item) for item in result.support_minimal_candidates],
        "best_residual": result.best_residual,
        "second_best_residual": result.second_best_residual,
        "solver_gap": result.solver_gap,
        "tie_set": [candidate_json(item) for item in result.tie_set],
        "evaluated_count": result.evaluated_count,
        "elapsed_seconds": result.elapsed_seconds,
    }


def cover_json(cover: tuple[BalancedCandidate, ...]) -> list[dict]:
    return [candidate_json(item) for item in cover]


def canonical_cover(cover) -> tuple:
    return tuple(sorted((tuple(edge[0]), tuple(edge[1])) for edge in cover))


def _decoder_geometry(block: np.ndarray, rank_rtol: float = 1e-12) -> tuple[np.ndarray, int, float | None]:
    u, singular, _ = np.linalg.svd(block, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return u[:, :0], 0, None
    rank = int(np.sum(singular > rank_rtol * singular[0]))
    condition = float(singular[0] / singular[rank - 1]) if rank else None
    return u[:, :rank], rank, condition


def _solver_surface(record: dict) -> list[dict]:
    found: list[dict] = []

    def visit(value, path: str) -> None:
        if isinstance(value, dict):
            if "evaluated_count" in value and "best_residual" in value:
                found.append({
                    "path": path,
                    "evaluated_count": value.get("evaluated_count"),
                    "best_residual": value.get("best_residual"),
                    "second_best_residual": value.get("second_best_residual"),
                    "solver_gap": value.get("solver_gap"),
                    "tie_count": len(value.get("tie_set", [])),
                    "elapsed_seconds": value.get("elapsed_seconds"),
                })
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(record, "")
    return found


def _reconstruct_pair_and_groups(record: dict, config: dict):
    provenance = record.get("seed_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{record['family_id']} lacks structured seed provenance")
    common = dict(
        structural_seed_a=provenance["structural_seed_a"],
        structural_seed_b=provenance["structural_seed_b"],
        mean_sample_seed=provenance["mean_sample_seed"],
        eval_sample_seed=provenance["eval_sample_seed"],
    )
    family = record["family_id"]
    if family == "F01_hadamard_gauge":
        pair = hadamard_gauge_seeded(record["q"], config["n_mean"], config["n_eval"], **common)
    elif family == "F02_local_block_rotations":
        pair = local_block_rotations_seeded(tuple(config["block_ranks"]), config["n_mean"], config["n_eval"], **common)
    elif family == "F03_unequal_split_merge":
        pair = unequal_split_merge_seeded(config["block_count"], config["n_mean"], config["n_eval"], **common)
    elif family == "F04_partial_overlap":
        pair = partial_overlap_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F05_cancellation":
        pair = cancellation_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F06_cooccurrence_confounding":
        pair = cooccurrence_confounding_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F07_rare_occupancy":
        pair = rare_occupancy_seeded(
            config["n_mean"], config["n_eval"], tokens_per_document=config["tokens_per_document"],
            active_document_count=config["active_document_count"], **common,
        )
    elif family == "F08_competing_covers":
        pair = competing_covers_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F09_whole_dictionary_only":
        pair = whole_dictionary_only_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F10_same_span_different_computation":
        pair = same_span_different_computation_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F11_same_sum_bloated_span":
        pair = same_sum_bloated_span_seeded(config["n_mean"], config["n_eval"], **common)
    elif family == "F12_non_lipschitz_downstream_cliff":
        pair = non_lipschitz_downstream_cliff_seeded(config["n_mean"], config["n_eval"], **common)
    else:
        raise ValueError(f"unknown synthetic family: {family}")
    groups = list(pair.planted_hyperedges)
    if family == "F01_hadamard_gauge":
        groups = [(tuple(range(pair.d_left.shape[1])), tuple(range(pair.d_right.shape[1])))]
    elif family == "F06_cooccurrence_confounding":
        groups = [((0,), (0,))]
    elif family == "F08_competing_covers":
        groups = sorted({edge for cover in pair.expected_covers for edge in cover})
    return pair, groups


def attach_complete_metric_surface(record: dict, config: dict) -> dict:
    pair, groups = _reconstruct_pair_and_groups(record, config)
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    measurements = []
    for index, (left, right) in enumerate(groups):
        left_ids = np.asarray(left, dtype=int)
        right_ids = np.asarray(right, dtype=int)
        bcc = bcc_from_kernels(kll, klr, krr, left_ids, right_ids)
        psc = projector_subspace_consistency(pair.d_left[:, left_ids], pair.d_right[:, right_ids])
        basis_left, rank_left, condition_left = _decoder_geometry(pair.d_left[:, left_ids])
        basis_right, rank_right, condition_right = _decoder_geometry(pair.d_right[:, right_ids])
        cosines = np.linalg.svd(basis_left.T @ basis_right, compute_uv=False) if rank_left and rank_right else np.array([])
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
        cancellation_left = cancellation_diagnostics(pair.d_left, zl, left_ids)
        cancellation_right = cancellation_diagnostics(pair.d_right, zr, right_ids)
        mean_left = np.mean(explicit_group_contribution(pair.d_left, pair.z_left_mean, left_ids), axis=0)
        mean_right = np.mean(explicit_group_contribution(pair.d_right, pair.z_right_mean, right_ids), axis=0)
        measurements.append({
            "label": f"group_{index}", "left_ids": list(left), "right_ids": list(right),
            "bcc": {"status": bcc.status, "value": bcc.value, "normalized_residual": bcc.normalized_residual,
                    "cross_inner": bcc.cross_inner, "energy_left": bcc.energy_left, "energy_right": bcc.energy_right},
            "psc": {"status": psc.status, "value": psc.value, "rank_left": psc.rank_left,
                    "rank_right": psc.rank_right, "projector_distance_sq": psc.projector_distance_sq,
                    "principal_angles_radians": angles.tolist()},
            "mean_contribution": {"norm_left": float(np.linalg.norm(mean_left)),
                                  "norm_right": float(np.linalg.norm(mean_right)),
                                  "error_norm": float(np.linalg.norm(mean_left - mean_right))},
            "geometry": {"group_size_left": len(left), "group_size_right": len(right),
                         "effective_rank_left": rank_left, "effective_rank_right": rank_right,
                         "condition_number_left": condition_left, "condition_number_right": condition_right},
            "cancellation": {
                "left": {"status": cancellation_left.status, "energy_ratio": cancellation_left.cancellation_energy_ratio,
                         "max_leave_one_out_ratio": cancellation_left.max_leave_one_out_energy_ratio,
                         "per_feature_ratios": list(cancellation_left.per_feature_energy_ratios)},
                "right": {"status": cancellation_right.status, "energy_ratio": cancellation_right.cancellation_energy_ratio,
                          "max_leave_one_out_ratio": cancellation_right.max_leave_one_out_energy_ratio,
                          "per_feature_ratios": list(cancellation_right.per_feature_energy_ratios)},
            },
        })
    record["metric_surface"] = {
        "schema_version": 1,
        "status": "OK" if measurements else "NOT_APPLICABLE_NO_EVALUATED_GROUP",
        "group_measurements": measurements,
        "coverage": {"value": record.get("coverage"), "unmatched_energy": record.get("unmatched_energy"),
                     "unmatched_energy_fraction": record.get("unmatched_energy_fraction"),
                     "status": "REPORTED" if "coverage" in record or "unmatched_energy_fraction" in record else "NOT_APPLICABLE_FAMILY_LEVEL"},
        "solver_diagnostics": _solver_surface(record),
        "decision": {"expected": record.get("expected_decision"), "observed": record.get("decision"),
                     "status": "REPORTED" if "decision" in record else "NOT_APPLICABLE_ALGEBRAIC_IDENTITY"},
    }
    return record


def metric_surface_errors(record: dict) -> list[str]:
    errors = []
    surface = record.get("metric_surface")
    if not isinstance(surface, dict):
        return ["missing metric_surface"]
    for field in ("schema_version", "status", "group_measurements", "coverage", "solver_diagnostics", "decision"):
        if field not in surface:
            errors.append(f"metric_surface missing {field}")
    groups = surface.get("group_measurements", [])
    if surface.get("status") == "OK" and not groups:
        errors.append("OK metric_surface has no group measurements")
    for index, group in enumerate(groups):
        for field in ("left_ids", "right_ids", "bcc", "psc", "mean_contribution", "geometry", "cancellation"):
            if field not in group:
                errors.append(f"group[{index}] missing {field}")
        for field in ("status", "value", "normalized_residual", "cross_inner", "energy_left", "energy_right"):
            if field not in group.get("bcc", {}):
                errors.append(f"group[{index}].bcc missing {field}")
        for field in ("status", "value", "rank_left", "rank_right", "projector_distance_sq", "principal_angles_radians"):
            if field not in group.get("psc", {}):
                errors.append(f"group[{index}].psc missing {field}")
        bcc = group.get("bcc", {})
        if bcc.get("status") == "OK":
            denominator = bcc["energy_left"] + bcc["energy_right"]
            expected_value = 2.0 * bcc["cross_inner"] / denominator
            expected_residual = 1.0 - expected_value
            if not np.isclose(bcc["value"], expected_value, atol=1e-12, rtol=1e-12):
                errors.append(f"group[{index}].bcc value identity failed")
            if not np.isclose(bcc["normalized_residual"], expected_residual, atol=1e-12, rtol=1e-12):
                errors.append(f"group[{index}].bcc residual identity failed")
        psc = group.get("psc", {})
        if psc.get("status") == "OK":
            expected_psc = 1.0 - psc["projector_distance_sq"] / (psc["rank_left"] + psc["rank_right"])
            if not np.isclose(psc["value"], expected_psc, atol=1e-12, rtol=1e-12):
                errors.append(f"group[{index}].psc identity failed")
            if len(psc["principal_angles_radians"]) != min(psc["rank_left"], psc["rank_right"]):
                errors.append(f"group[{index}].psc principal-angle count failed")
        for side in ("left", "right"):
            cancellation = group.get("cancellation", {}).get(side, {})
            expected_count = len(group.get(f"{side}_ids", []))
            if cancellation.get("status") == "OK" and len(cancellation.get("per_feature_ratios", [])) != expected_count:
                errors.append(f"group[{index}].cancellation.{side} feature count failed")
    return errors


def metric_record(pair, config: dict, q: int | None = None, generator_seed: int | None = None) -> dict:
    zl, mean_left = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, mean_right = center_codes(pair.z_right_mean, pair.z_right_eval)
    left = np.arange(pair.d_left.shape[1])
    right = np.arange(pair.d_right.shape[1])
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    bcc = bcc_from_kernels(kll, klr, krr, left, right)
    psc = projector_subspace_consistency(pair.d_left, pair.d_right)
    yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, left)
    yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, right)
    explicit_max_error = float(np.max(np.abs(yl - yr)))
    record = {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "q": q,
        "bcc": bcc.value,
        "normalized_residual": bcc.normalized_residual,
        "bcc_status": bcc.status,
        "cross_inner": bcc.cross_inner,
        "energy_left": bcc.energy_left,
        "energy_right": bcc.energy_right,
        "psc": psc.value,
        "psc_status": psc.status,
        "rank_left": psc.rank_left,
        "rank_right": psc.rank_right,
        "projector_distance_sq": psc.projector_distance_sq,
        "explicit_contribution_max_error": explicit_max_error,
        "mean_constant_norm_left": float(np.linalg.norm(mean_left)),
        "mean_constant_norm_right": float(np.linalg.norm(mean_right)),
        "mean_contribution_error_norm": float(np.linalg.norm(
            np.mean(explicit_group_contribution(pair.d_left, pair.z_left_mean, left), axis=0)
            - np.mean(explicit_group_contribution(pair.d_right, pair.z_right_mean, right), axis=0)
        )),
        "seed_provenance": dict(pair.seed_provenance) if pair.seed_provenance else None,
    }
    if q is not None:
        record["pw_mcc"] = pw_mcc_absolute_cosine(pair.d_left, pair.d_right)
        record["pw_mcc_expected"] = q ** -0.5
    return record


def function_mismatch_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    pair = same_span_different_computation_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=provenance["structural_seed_a"],
        structural_seed_b=provenance["structural_seed_b"],
        mean_sample_seed=provenance["mean_sample_seed"],
        eval_sample_seed=provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    risk_left, risk_right = pair.planted_hyperedges[0]
    clean_left, clean_right = pair.planted_hyperedges[1]
    risk_bcc = bcc_from_kernels(kll, klr, krr, risk_left, risk_right)
    clean_bcc = bcc_from_kernels(kll, klr, krr, clean_left, clean_right)
    risk_psc = projector_subspace_consistency(pair.d_left[:, risk_left], pair.d_right[:, risk_right])
    clean_psc = projector_subspace_consistency(pair.d_left[:, clean_left], pair.d_right[:, clean_right])
    risk_search = exhaustive_balanced_search(kll, klr, krr, risk_left, risk_right,
        residual_tolerance=config["residual_tolerance"], tie_tolerance=config["tie_tolerance"], max_group_size=2)
    clean_search = exhaustive_balanced_search(kll, klr, krr, clean_left, clean_right,
        residual_tolerance=config["residual_tolerance"], tie_tolerance=config["tie_tolerance"], max_group_size=1)
    yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, np.asarray(risk_left))
    yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, np.asarray(risk_right))
    decision = ("REFUSE_FUNCTION_MISMATCH_WITH_CLEAN_CONTROL_PASS"
        if risk_bcc.normalized_residual >= config["minimum_f10_residual"]
        and len(clean_search.support_minimal_candidates) == 1 else "INCORRECT")
    return {
        "family_id": pair.family_id, "generator_seed": generator_seed, "seed_provenance": provenance,
        "expected_decision": pair.expected_decision, "decision": decision,
        "risk_group_bcc": risk_bcc.value, "risk_group_residual": risk_bcc.normalized_residual,
        "risk_group_psc": risk_psc.value,
        "risk_contribution_rmse": float(np.sqrt(np.mean((yl - yr) ** 2))),
        "clean_group_bcc": clean_bcc.value, "clean_group_psc": clean_psc.value,
        "risk_oracle": search_json(risk_search, "function_mismatch", risk_left, risk_right),
        "clean_oracle": search_json(clean_search, "clean_control", clean_left, clean_right),
        "false_unique_rate": 0.0 if decision == pair.expected_decision else 1.0,
        "span_only_false_unique_rate": 1.0 if np.isclose(risk_psc.value, 1.0) else 0.0,
    }


def span_bloat_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    provenance = {
        "structural_seed_a": seed_root + 1, "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3, "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    pair = same_sum_bloated_span_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=provenance["structural_seed_a"],
        structural_seed_b=provenance["structural_seed_b"],
        mean_sample_seed=provenance["mean_sample_seed"], eval_sample_seed=provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    risk_left, risk_right = pair.planted_hyperedges[0]
    clean_left, clean_right = pair.planted_hyperedges[1]
    risk_bcc = bcc_from_kernels(kll, klr, krr, risk_left, risk_right)
    clean_bcc = bcc_from_kernels(kll, klr, krr, clean_left, clean_right)
    risk_psc = projector_subspace_consistency(pair.d_left[:, risk_left], pair.d_right[:, risk_right])
    clean_psc = projector_subspace_consistency(pair.d_left[:, clean_left], pair.d_right[:, clean_right])
    risk_search = exhaustive_balanced_search(kll, klr, krr, risk_left, risk_right,
        residual_tolerance=config["residual_tolerance"], tie_tolerance=config["tie_tolerance"], max_group_size=2)
    clean_search = exhaustive_balanced_search(kll, klr, krr, clean_left, clean_right,
        residual_tolerance=config["residual_tolerance"], tie_tolerance=config["tie_tolerance"], max_group_size=1)
    yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, np.asarray(risk_left))
    yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, np.asarray(risk_right))
    decision = ("REFUSE_SPAN_BLOAT_WITH_CLEAN_CONTROL_PASS"
        if risk_psc.value <= config["maximum_f11_psc"] and risk_psc.rank_right > risk_psc.rank_left
        and len(clean_search.support_minimal_candidates) == 1 else "INCORRECT")
    return {
        "family_id": pair.family_id, "generator_seed": generator_seed, "seed_provenance": provenance,
        "expected_decision": pair.expected_decision, "decision": decision,
        "risk_group_bcc": risk_bcc.value, "risk_group_residual": risk_bcc.normalized_residual,
        "risk_group_psc": risk_psc.value, "risk_rank_left": risk_psc.rank_left, "risk_rank_right": risk_psc.rank_right,
        "risk_contribution_max_error": float(np.max(np.abs(yl - yr))),
        "clean_group_bcc": clean_bcc.value, "clean_group_psc": clean_psc.value,
        "span_bloat_alpha": dict(pair.diagnostic_values)["span_bloat_alpha"],
        "risk_oracle": search_json(risk_search, "span_bloat", risk_left, risk_right),
        "clean_oracle": search_json(clean_search, "clean_control", clean_left, clean_right),
        "false_unique_rate": 0.0 if decision == pair.expected_decision else 1.0,
        "no_psc_false_unique_rate": 1.0 if np.isclose(risk_bcc.value, 1.0) else 0.0,
    }


def downstream_cliff_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    provenance = {
        "structural_seed_a": seed_root + 1, "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3, "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    pair = non_lipschitz_downstream_cliff_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=provenance["structural_seed_a"],
        structural_seed_b=provenance["structural_seed_b"],
        mean_sample_seed=provenance["mean_sample_seed"],
        eval_sample_seed=provenance["eval_sample_seed"],
    )
    zl, mean_left = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, mean_right = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    score = bcc_from_kernels(kll, klr, krr, (0,), (0,))
    psc = projector_subspace_consistency(pair.d_left, pair.d_right)
    ya = explicit_group_contribution(pair.d_left, pair.z_left_eval, np.asarray([0]))
    yb = explicit_group_contribution(pair.d_right, pair.z_right_eval, np.asarray([0]))
    u = pair.d_left[:, 0] / np.linalg.norm(pair.d_left[:, 0])
    v = np.array([-u[1], u[0]])
    hook = pair.hook_eval
    dose_curve = []
    states = []
    for dose in config["f12_dose_grid"]:
        state = hook - ((1.0 - dose) * ya + dose * yb)
        states.append(state)
        risk = ((state @ u) * (state @ v) >= 0.0).astype(float)
        dose_curve.append({
            "dose": dose,
            "risk_output_mean": float(np.mean(risk)),
            "smooth_distance_from_left_rmse": float(np.sqrt(np.mean(np.sum((state - states[0]) ** 2, axis=1)))),
        })
    risk_left = ((states[0] @ u) * (states[0] @ v) >= 0.0).astype(float)
    risk_right = ((states[-1] @ u) * (states[-1] @ v) >= 0.0).astype(float)
    contribution_rmse = float(np.sqrt(np.mean(np.sum((ya - yb) ** 2, axis=1))))
    smooth_rmse = float(np.sqrt(np.mean(np.sum((states[0] - states[-1]) ** 2, axis=1))))
    risk_mismatch = float(np.mean(risk_left != risk_right))
    jumps = [abs(dose_curve[i + 1]["risk_output_mean"] - dose_curve[i]["risk_output_mean"])
             for i in range(len(dose_curve) - 1)]
    jump_index = int(np.argmax(jumps))
    jump_hook_step = float(
        np.sqrt(np.mean(np.sum((states[jump_index + 1] - states[jump_index]) ** 2, axis=1)))
    )
    mean_contribution_error = float(np.linalg.norm(pair.d_left @ mean_left - pair.d_right @ mean_right))
    decision = ("NONCAUSAL_UNDER_UNCERTIFIED_READOUT_WITH_SMOOTH_CONTROL_PASS"
        if score.value >= config["minimum_f12_bcc"]
        and risk_mismatch >= config["minimum_f12_risk_mismatch"]
        and max(jumps) >= config["minimum_f12_cliff_jump"]
        and smooth_rmse <= contribution_rmse * (1.0 + config["relative_tolerance"]) + config["absolute_tolerance"]
        else "INCORRECT")
    return {
        "family_id": pair.family_id, "generator_seed": generator_seed, "seed_provenance": provenance,
        "expected_decision": pair.expected_decision, "decision": decision,
        "delta": dict(pair.diagnostic_values)["delta"],
        "bcc": score.value, "normalized_residual": score.normalized_residual,
        "psc": psc.value, "mean_contribution_error_norm": mean_contribution_error,
        "contribution_rmse": contribution_rmse,
        "risk_endpoint_mismatch_rate": risk_mismatch,
        "maximum_adjacent_risk_jump": max(jumps),
        "cliff_hook_step_rmse": jump_hook_step,
        "smooth_endpoint_rmse": smooth_rmse,
        "smooth_transfer_ratio": smooth_rmse / contribution_rmse,
        "dose_curve": dose_curve,
        "false_causal_certificate_rate": 0.0 if decision == pair.expected_decision else 1.0,
        "bcc_only_false_causal_rate": 1.0 if score.value >= config["minimum_f12_bcc"] else 0.0,
    }


def local_rotation_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    pair = local_block_rotations_seeded(
        tuple(config["block_ranks"]),
        config["n_mean"],
        config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    recovered = 0
    passing_candidates = 0
    max_planted_residual = 0.0
    max_contribution_error = 0.0
    max_mean_contribution_error = 0.0
    min_planted_psc = 1.0
    oracle_neighborhoods = []
    planted_solver_gaps = []
    planted_tie_sizes = []
    candidate_pairs_evaluated = 0
    runtime_seconds = 0.0
    predicted_left_labels = np.full(pair.d_left.shape[1], -1, dtype=int)
    predicted_right_labels = np.full(pair.d_right.shape[1], -1, dtype=int)
    true_left_labels = np.full(pair.d_left.shape[1], -1, dtype=int)
    true_right_labels = np.full(pair.d_right.shape[1], -1, dtype=int)
    for index, (left, right) in enumerate(pair.planted_hyperedges):
        search = exhaustive_balanced_search(
            kll,
            klr,
            krr,
            left,
            right,
            residual_tolerance=config["residual_tolerance"],
            tie_tolerance=config.get("tie_tolerance", 0.0),
            max_group_size=config["max_group_size"],
        )
        oracle_neighborhoods.append(search_json(search, "planted", left, right))
        candidate_pairs_evaluated += search.evaluated_count
        runtime_seconds += search.elapsed_seconds
        found = list(search.support_minimal_candidates)
        passing_candidates += len(found)
        planted_tie_sizes.append(len(search.tie_set))
        if search.solver_gap is not None:
            planted_solver_gaps.append(search.solver_gap)
        if len(found) == 1 and found[0].left_ids == left and found[0].right_ids == right:
            recovered += 1
            max_planted_residual = max(max_planted_residual, found[0].normalized_residual)
            predicted_left_labels[np.asarray(found[0].left_ids, dtype=int)] = index
            predicted_right_labels[np.asarray(found[0].right_ids, dtype=int)] = index
        true_left_labels[np.asarray(left, dtype=int)] = index
        true_right_labels[np.asarray(right, dtype=int)] = index
        left_array = np.asarray(left, dtype=int)
        right_array = np.asarray(right, dtype=int)
        yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, left_array)
        yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, right_array)
        max_contribution_error = max(max_contribution_error, float(np.max(np.abs(yl - yr))))
        mean_left = np.mean(explicit_group_contribution(pair.d_left, pair.z_left_mean, left_array), axis=0)
        mean_right = np.mean(explicit_group_contribution(pair.d_right, pair.z_right_mean, right_array), axis=0)
        max_mean_contribution_error = max(max_mean_contribution_error, float(np.linalg.norm(mean_left - mean_right)))
        psc = projector_subspace_consistency(pair.d_left[:, left_array], pair.d_right[:, right_array])
        min_planted_psc = min(min_planted_psc, float(psc.value))
    cross_matches = 0
    for index, (left, _) in enumerate(pair.planted_hyperedges):
        wrong_right = pair.planted_hyperedges[(index + 1) % len(pair.planted_hyperedges)][1]
        cross_search = exhaustive_balanced_search(
            kll,
            klr,
            krr,
            left,
            wrong_right,
            residual_tolerance=config["residual_tolerance"],
            tie_tolerance=config.get("tie_tolerance", 0.0),
            max_group_size=config["max_group_size"],
        )
        oracle_neighborhoods.append(search_json(cross_search, "cross_block_negative", left, wrong_right))
        candidate_pairs_evaluated += cross_search.evaluated_count
        runtime_seconds += cross_search.elapsed_seconds
        cross_matches += len(cross_search.support_minimal_candidates)
    labels_complete = bool(np.all(predicted_left_labels >= 0) and np.all(predicted_right_labels >= 0))
    ari_left = adjusted_rand_index(true_left_labels, predicted_left_labels) if labels_complete else None
    ari_right = adjusted_rand_index(true_right_labels, predicted_right_labels) if labels_complete else None
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "expected_decision": pair.expected_decision,
        "decision": "UNIQUE_PARTITION" if recovered == len(pair.planted_hyperedges) and cross_matches == 0 else "INCORRECT",
        "blocks": len(pair.planted_hyperedges),
        "recovered_blocks": recovered,
        "exact_recovery": recovered == len(pair.planted_hyperedges) and cross_matches == 0,
        "support_minimal_candidates": passing_candidates,
        "spurious_cross_block_candidates": cross_matches,
        "candidate_pairs_evaluated": candidate_pairs_evaluated,
        "max_planted_normalized_residual": max_planted_residual,
        "min_planted_psc": min_planted_psc,
        "max_explicit_contribution_error": max_contribution_error,
        "max_mean_contribution_error_norm": max_mean_contribution_error,
        "planted_hyperedges": [
            {"left_ids": list(left), "right_ids": list(right)} for left, right in pair.planted_hyperedges
        ],
        "oracle_neighborhoods": oracle_neighborhoods,
        "minimum_planted_solver_gap": min(planted_solver_gaps),
        "maximum_planted_tie_set_size": max(planted_tie_sizes),
        "oracle_runtime_seconds": runtime_seconds,
        "diagnostics_complete": len(oracle_neighborhoods) == 2 * len(pair.planted_hyperedges),
        "ari_left": ari_left,
        "ari_right": ari_right,
        "coverage": 1.0 if labels_complete else 0.0,
        "unmatched_energy_fraction": 0.0 if labels_complete else 1.0,
        "group_sizes_left": [len(left) for left, _ in pair.planted_hyperedges],
        "group_sizes_right": [len(right) for _, right in pair.planted_hyperedges],
        "effective_ranks_left": list(config["block_ranks"]),
        "effective_ranks_right": list(config["block_ranks"]),
    }


def unequal_split_merge_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    return _unequal_split_merge_record_impl(config, generator_seed, seed_provenance)


def partial_overlap_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    return _partial_overlap_record_impl(config, generator_seed, seed_provenance)


def cooccurrence_confounding_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    return _cooccurrence_confounding_record_impl(config, generator_seed, seed_provenance)


def competing_covers_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    pair = competing_covers_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    search = exhaustive_balanced_search(
        contribution_kernel(pair.d_left, zl, pair.d_left, zl),
        contribution_kernel(pair.d_left, zl, pair.d_right, zr),
        contribution_kernel(pair.d_right, zr, pair.d_right, zr),
        (0, 1), (0, 1),
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=config["f08_max_group_size"],
    )
    covers = enumerate_maximum_exact_covers(search.support_minimal_candidates, (0, 1), (0, 1))
    observed = {
        canonical_cover(tuple((item.left_ids, item.right_ids) for item in cover))
        for cover in covers.maximum_covers
    }
    expected = {canonical_cover(cover) for cover in pair.expected_covers}
    decision = "AMBIGUOUS" if observed == expected and len(observed) > 1 else "INCORRECT"
    forced_best = covers.maximum_covers[:1]
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "expected_decision": pair.expected_decision,
        "decision": decision,
        "expected_covers": [[
            {"left_ids": list(left), "right_ids": list(right)} for left, right in cover
        ] for cover in pair.expected_covers],
        "maximum_covers": [cover_json(cover) for cover in covers.maximum_covers],
        "maximum_cover_count": len(covers.maximum_covers),
        "maximum_cover_cardinality": covers.maximum_cardinality,
        "exact_cover_count": covers.exact_cover_count,
        "exact_cover_runtime_seconds": covers.elapsed_seconds,
        "ambiguity_accuracy": 1.0 if decision == pair.expected_decision else 0.0,
        "false_unique_rate": 0.0 if decision == "AMBIGUOUS" else 1.0,
        "forced_best_cover": [cover_json(cover) for cover in forced_best],
        "forced_best_false_unique_rate": 1.0 if len(covers.maximum_covers) > 1 and len(forced_best) == 1 else 0.0,
        "oracle": search_json(search, "competing_cover_singletons", (0, 1), (0, 1)),
    }


def whole_dictionary_only_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    return _whole_dictionary_only_record_impl(config, generator_seed, seed_provenance)


def cancellation_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
    }
    return _cancellation_record_impl(config, generator_seed, seed_provenance)


def rare_occupancy_record(config: dict, generator_seed: int) -> dict:
    seed_root = 10 * generator_seed
    seed_provenance = {
        "structural_seed_a": seed_root + 1,
        "structural_seed_b": seed_root + 2,
        "mean_sample_seed": seed_root + 3,
        "eval_sample_seed": seed_root + 4,
        "solver_seed": seed_root + 5,
        "bootstrap_seed": seed_root + 6,
    }
    pair = rare_occupancy_seeded(
        config["n_mean"], config["n_eval"],
        tokens_per_document=config["tokens_per_document"],
        active_document_count=config["active_document_count"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    documents = np.asarray(pair.eval_document_ids, dtype=int)
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    risk_left, risk_right = pair.planted_hyperedges[0]
    clean_left, clean_right = pair.planted_hyperedges[1]
    risk_left_ids = np.asarray(risk_left, dtype=int)
    risk_right_ids = np.asarray(risk_right, dtype=int)
    clean_left_ids = np.asarray(clean_left, dtype=int)
    clean_right_ids = np.asarray(clean_right, dtype=int)
    risk_bcc = bcc_from_kernels(kll, klr, krr, risk_left_ids, risk_right_ids)
    clean_bcc = bcc_from_kernels(kll, klr, krr, clean_left_ids, clean_right_ids)
    risk_search = exhaustive_balanced_search(
        kll, klr, krr, risk_left, risk_right,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=1,
    )
    clean_search = exhaustive_balanced_search(
        kll, klr, krr, clean_left, clean_right,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=1,
    )
    risk_occ_left = occupancy_effective_sample_size(pair.d_left, pair.z_left_eval, risk_left_ids, documents)
    risk_occ_right = occupancy_effective_sample_size(pair.d_right, pair.z_right_eval, risk_right_ids, documents)
    clean_occ_left = occupancy_effective_sample_size(pair.d_left, pair.z_left_eval, clean_left_ids, documents)
    clean_occ_right = occupancy_effective_sample_size(pair.d_right, pair.z_right_eval, clean_right_ids, documents)
    risk_bootstrap = document_bootstrap_bcc(
        pair.d_left, zl, risk_left_ids, pair.d_right, zr, risk_right_ids, documents,
        replicates=config["bootstrap_replicates"], seed=seed_provenance["bootstrap_seed"],
    )
    clean_bootstrap = document_bootstrap_bcc(
        pair.d_left, zl, clean_left_ids, pair.d_right, zr, clean_right_ids, documents,
        replicates=config["bootstrap_replicates"], seed=seed_provenance["bootstrap_seed"] + 1,
    )
    risk_low_support = (
        max(risk_occ_left.document_energy_kish_ess, risk_occ_right.document_energy_kish_ess)
        <= config["maximum_risk_document_ess"]
        and risk_bootstrap.inactive_fraction >= config["minimum_risk_inactive_bootstrap_fraction"]
        and risk_bootstrap.ci_width >= config["minimum_risk_bcc_ci_width"]
    )
    clean_stable = (
        min(clean_occ_left.document_energy_kish_ess, clean_occ_right.document_energy_kish_ess)
        >= config["minimum_clean_document_ess"]
        and clean_bootstrap.inactive_fraction == 0.0
        and clean_bootstrap.ci_width <= config["maximum_clean_bcc_ci_width"]
    )
    risk_passes_point = risk_bcc.value >= config["naive_bcc_acceptance_threshold"] and bool(risk_search.passing_candidates)
    clean_passes_point = clean_bcc.value >= config["naive_bcc_acceptance_threshold"] and bool(clean_search.passing_candidates)
    decision = (
        "REFUSE_LOW_N_EFF_WITH_DENSE_CONTROL_PASS"
        if risk_passes_point and clean_passes_point and risk_low_support and clean_stable
        else "INCORRECT"
    )
    accepted_energy = clean_bcc.energy_left + clean_bcc.energy_right
    refused_energy = risk_bcc.energy_left + risk_bcc.energy_right
    total_energy = accepted_energy + refused_energy
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "diagnostic_values": dict(pair.diagnostic_values),
        "expected_decision": pair.expected_decision,
        "decision": decision,
        "naive_token_count": config["n_eval"],
        "risk_point_bcc": risk_bcc.value,
        "risk_point_residual": risk_bcc.normalized_residual,
        "risk_passes_point_threshold": risk_passes_point,
        "risk_active_tokens_left": risk_occ_left.active_token_count,
        "risk_active_tokens_right": risk_occ_right.active_token_count,
        "risk_active_documents_left": risk_occ_left.active_document_count,
        "risk_active_documents_right": risk_occ_right.active_document_count,
        "risk_token_energy_kish_ess_left": risk_occ_left.token_energy_kish_ess,
        "risk_token_energy_kish_ess_right": risk_occ_right.token_energy_kish_ess,
        "risk_document_energy_kish_ess_left": risk_occ_left.document_energy_kish_ess,
        "risk_document_energy_kish_ess_right": risk_occ_right.document_energy_kish_ess,
        "risk_bootstrap_values": list(risk_bootstrap.values),
        "risk_bootstrap_inactive_replicates": risk_bootstrap.inactive_replicates,
        "risk_bootstrap_inactive_fraction": risk_bootstrap.inactive_fraction,
        "risk_bootstrap_ci_lower": risk_bootstrap.ci_lower,
        "risk_bootstrap_ci_upper": risk_bootstrap.ci_upper,
        "risk_bootstrap_ci_width": risk_bootstrap.ci_width,
        "risk_low_support_flag": risk_low_support,
        "clean_point_bcc": clean_bcc.value,
        "clean_point_residual": clean_bcc.normalized_residual,
        "clean_passes_point_threshold": clean_passes_point,
        "clean_document_energy_kish_ess_left": clean_occ_left.document_energy_kish_ess,
        "clean_document_energy_kish_ess_right": clean_occ_right.document_energy_kish_ess,
        "clean_bootstrap_values": list(clean_bootstrap.values),
        "clean_bootstrap_inactive_fraction": clean_bootstrap.inactive_fraction,
        "clean_bootstrap_ci_lower": clean_bootstrap.ci_lower,
        "clean_bootstrap_ci_upper": clean_bootstrap.ci_upper,
        "clean_bootstrap_ci_width": clean_bootstrap.ci_width,
        "clean_stability_flag": clean_stable,
        "refusal_accuracy": 1.0 if decision == pair.expected_decision else 0.0,
        "clean_specificity": 1.0 if clean_stable else 0.0,
        "false_unique_rate": 0.0 if decision == pair.expected_decision else 1.0,
        "naive_token_count_false_unique_rate": 1.0 if risk_passes_point else 0.0,
        "accepted_energy_coverage": accepted_energy / total_energy,
        "unmatched_energy_fraction": refused_energy / total_energy,
        "risk_oracle": search_json(risk_search, "rare_singleton", risk_left, risk_right),
        "clean_oracle": search_json(clean_search, "dense_singleton_control", clean_left, clean_right),
    }
def _cancellation_record_impl(config: dict, generator_seed: int, seed_provenance: dict) -> dict:
    pair = cancellation_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    risk_left, risk_right = pair.planted_hyperedges[0]
    clean_left, clean_right = pair.planted_hyperedges[1]
    risk_search = exhaustive_balanced_search(
        kll, klr, krr, risk_left, risk_right,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=2,
    )
    clean_search = exhaustive_balanced_search(
        kll, klr, krr, clean_left, clean_right,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=1,
    )
    risk_left_ids = np.asarray(risk_left, dtype=int)
    risk_right_ids = np.asarray(risk_right, dtype=int)
    clean_left_ids = np.asarray(clean_left, dtype=int)
    clean_right_ids = np.asarray(clean_right, dtype=int)
    risk_diag_left = cancellation_diagnostics(pair.d_left, zl, risk_left_ids)
    risk_diag_right = cancellation_diagnostics(pair.d_right, zr, risk_right_ids)
    clean_diag_left = cancellation_diagnostics(pair.d_left, zl, clean_left_ids)
    clean_diag_right = cancellation_diagnostics(pair.d_right, zr, clean_right_ids)
    risk_flag = (
        min(risk_diag_left.cancellation_energy_ratio, risk_diag_right.cancellation_energy_ratio)
        >= config["cancellation_energy_ratio_threshold"]
        and min(risk_diag_left.max_leave_one_out_energy_ratio, risk_diag_right.max_leave_one_out_energy_ratio)
        >= config["leave_one_out_energy_ratio_threshold"]
    )
    clean_flag = (
        max(clean_diag_left.cancellation_energy_ratio, clean_diag_right.cancellation_energy_ratio)
        >= config["cancellation_energy_ratio_threshold"]
        or max(clean_diag_left.max_leave_one_out_energy_ratio, clean_diag_right.max_leave_one_out_energy_ratio)
        >= config["leave_one_out_energy_ratio_threshold"]
    )
    risk_expected = [(risk_left, risk_right)]
    clean_expected = [(clean_left, clean_right)]
    risk_found = [(item.left_ids, item.right_ids) for item in risk_search.support_minimal_candidates]
    clean_found = [(item.left_ids, item.right_ids) for item in clean_search.support_minimal_candidates]
    decision = (
        "REFUSE_CANCELLATION_RISK_WITH_CLEAN_CONTROL_PASS"
        if risk_found == risk_expected and clean_found == clean_expected and risk_flag and not clean_flag
        else "INCORRECT"
    )
    risk_bcc = bcc_from_kernels(kll, klr, krr, risk_left_ids, risk_right_ids)
    clean_bcc = bcc_from_kernels(kll, klr, krr, clean_left_ids, clean_right_ids)
    risk_psc = projector_subspace_consistency(pair.d_left[:, risk_left_ids], pair.d_right[:, risk_right_ids])
    clean_psc = projector_subspace_consistency(pair.d_left[:, clean_left_ids], pair.d_right[:, clean_right_ids])
    risk_y_left = explicit_group_contribution(pair.d_left, pair.z_left_eval, risk_left_ids)
    risk_y_right = explicit_group_contribution(pair.d_right, pair.z_right_eval, risk_right_ids)
    clean_y_left = explicit_group_contribution(pair.d_left, pair.z_left_eval, clean_left_ids)
    clean_y_right = explicit_group_contribution(pair.d_right, pair.z_right_eval, clean_right_ids)
    accepted_energy = clean_bcc.energy_left + clean_bcc.energy_right
    refused_energy = risk_bcc.energy_left + risk_bcc.energy_right
    total_energy = accepted_energy + refused_energy
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "diagnostic_values": dict(pair.diagnostic_values),
        "expected_decision": pair.expected_decision,
        "decision": decision,
        "risk_group_bcc": risk_bcc.value,
        "risk_group_residual": risk_bcc.normalized_residual,
        "risk_group_psc": risk_psc.value,
        "risk_group_max_contribution_error": float(np.max(np.abs(risk_y_left - risk_y_right))),
        "risk_cancellation_energy_ratio_left": risk_diag_left.cancellation_energy_ratio,
        "risk_cancellation_energy_ratio_right": risk_diag_right.cancellation_energy_ratio,
        "risk_max_leave_one_out_ratio_left": risk_diag_left.max_leave_one_out_energy_ratio,
        "risk_max_leave_one_out_ratio_right": risk_diag_right.max_leave_one_out_energy_ratio,
        "risk_per_feature_ratios_left": list(risk_diag_left.per_feature_energy_ratios),
        "risk_per_feature_ratios_right": list(risk_diag_right.per_feature_energy_ratios),
        "risk_flag": risk_flag,
        "clean_group_bcc": clean_bcc.value,
        "clean_group_residual": clean_bcc.normalized_residual,
        "clean_group_psc": clean_psc.value,
        "clean_group_max_contribution_error": float(np.max(np.abs(clean_y_left - clean_y_right))),
        "clean_cancellation_energy_ratio_left": clean_diag_left.cancellation_energy_ratio,
        "clean_cancellation_energy_ratio_right": clean_diag_right.cancellation_energy_ratio,
        "clean_max_leave_one_out_ratio_left": clean_diag_left.max_leave_one_out_energy_ratio,
        "clean_max_leave_one_out_ratio_right": clean_diag_right.max_leave_one_out_energy_ratio,
        "clean_flag": clean_flag,
        "diagnostic_recall": 1.0 if risk_flag else 0.0,
        "clean_specificity": 1.0 if not clean_flag else 0.0,
        "false_unique_rate": 0.0 if decision == pair.expected_decision else 1.0,
        "no_diagnostic_false_unique_rate": 1.0 if risk_found == risk_expected else 0.0,
        "accepted_feature_coverage": 1.0 / 3.0,
        "accepted_energy_coverage": accepted_energy / total_energy,
        "unmatched_energy_fraction": refused_energy / total_energy,
        "risk_oracle": search_json(risk_search, "canceling_group", risk_left, risk_right),
        "clean_oracle": search_json(clean_search, "clean_singleton_control", clean_left, clean_right),
    }
def _whole_dictionary_only_record_impl(config: dict, generator_seed: int, seed_provenance: dict) -> dict:
    pair = whole_dictionary_only_seeded(
        config["n_mean"], config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    pool = (0, 1, 2)
    search = exhaustive_balanced_search(
        kll, klr, krr, pool, pool,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=config["f09_max_group_size"],
    )
    full_edge = pair.planted_hyperedges[0]
    passing_edges = {(item.left_ids, item.right_ids) for item in search.support_minimal_candidates}
    local_passing = passing_edges - {full_edge}
    decision = "REFUSE_GLOBAL_ONLY" if passing_edges == {full_edge} and not local_passing else "INCORRECT"
    ids = np.asarray(pool, dtype=int)
    bcc = bcc_from_kernels(kll, klr, krr, ids, ids)
    y_left = explicit_group_contribution(pair.d_left, pair.z_left_eval, ids)
    y_right = explicit_group_contribution(pair.d_right, pair.z_right_eval, ids)
    mean_left = np.mean(explicit_group_contribution(pair.d_left, pair.z_left_mean, ids), axis=0)
    mean_right = np.mean(explicit_group_contribution(pair.d_right, pair.z_right_mean, ids), axis=0)
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "expected_decision": pair.expected_decision,
        "decision": decision,
        "planted_global_edge": {"left_ids": list(full_edge[0]), "right_ids": list(full_edge[1])},
        "global_balance_bcc": bcc.value,
        "global_balance_residual": bcc.normalized_residual,
        "global_contribution_max_error": float(np.max(np.abs(y_left - y_right))),
        "global_mean_contribution_error_norm": float(np.linalg.norm(mean_left - mean_right)),
        "local_passing_count": len(local_passing),
        "refusal_accuracy": 1.0 if decision == pair.expected_decision else 0.0,
        "false_unique_rate": 0.0 if decision == "REFUSE_GLOBAL_ONLY" else 1.0,
        "global_collapse_false_unique_rate": 1.0 if passing_edges == {full_edge} else 0.0,
        "coverage": 0.0,
        "unmatched_energy_fraction": 1.0,
        "oracle": search_json(search, "whole_dictionary_pool", pool, pool),
    }
def _cooccurrence_confounding_record_impl(config: dict, generator_seed: int, seed_provenance: dict) -> dict:
    pair = cooccurrence_confounding_seeded(
        config["n_mean"],
        config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    code_correlation = absolute_code_correlation(zl[:, 0], zr[:, 0])
    proposal_accepts = code_correlation >= config["proposal_min_abs_code_correlation"]
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    ids = np.asarray([0], dtype=int)
    bcc = bcc_from_kernels(kll, klr, krr, ids, ids)
    psc = projector_subspace_consistency(pair.d_left, pair.d_right)
    search = exhaustive_balanced_search(
        kll, klr, krr, (0,), (0,),
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=1,
    )
    decision = "REFUSE_CONTRIBUTION_MISMATCH" if proposal_accepts and not search.passing_candidates else "INCORRECT"
    y_left = explicit_group_contribution(pair.d_left, pair.z_left_eval, ids)
    y_right = explicit_group_contribution(pair.d_right, pair.z_right_eval, ids)
    contribution_rmse = float(np.sqrt(np.mean(np.sum((y_left - y_right) ** 2, axis=1))))
    mean_left = np.mean(explicit_group_contribution(pair.d_left, pair.z_left_mean, ids), axis=0)
    mean_right = np.mean(explicit_group_contribution(pair.d_right, pair.z_right_mean, ids), axis=0)
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "expected_decision": pair.expected_decision,
        "decision": decision,
        "proposal_score_abs_code_correlation": code_correlation,
        "proposal_threshold": config["proposal_min_abs_code_correlation"],
        "proposal_accepts": proposal_accepts,
        "bcc": bcc.value,
        "normalized_residual": bcc.normalized_residual,
        "energy_left": bcc.energy_left,
        "energy_right": bcc.energy_right,
        "cross_inner": bcc.cross_inner,
        "psc": psc.value,
        "rank_left": psc.rank_left,
        "rank_right": psc.rank_right,
        "contribution_rmse": contribution_rmse,
        "mean_contribution_error_norm": float(np.linalg.norm(mean_left - mean_right)),
        "coverage": 0.0,
        "unmatched_energy": bcc.energy_left + bcc.energy_right,
        "unmatched_energy_fraction": 1.0,
        "left_group_size": 1,
        "right_group_size": 1,
        "cancellation_index": 1.0,
        "condition_number_left": 1.0,
        "condition_number_right": 1.0,
        "oracle": search_json(search, "correlation_proposed_singleton", (0,), (0,)),
    }
def _partial_overlap_record_impl(config: dict, generator_seed: int, seed_provenance: dict) -> dict:
    pair = partial_overlap_seeded(
        config["n_mean"],
        config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    pool = tuple(range(3))
    search = exhaustive_balanced_search(
        kll, klr, krr, pool, pool,
        residual_tolerance=config["residual_tolerance"],
        tie_tolerance=config["tie_tolerance"],
        max_group_size=config["max_group_size"],
    )
    predicted = tuple((item.left_ids, item.right_ids) for item in search.support_minimal_candidates)
    truth = set(pair.planted_hyperedges)
    predicted_set = set(predicted)
    true_positive = len(truth & predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(truth)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    forced = forced_partition_projection(search.support_minimal_candidates)
    forced_set = {(item.left_ids, item.right_ids) for item in forced}
    forced_true_positive = len(truth & forced_set)
    forced_precision = forced_true_positive / len(forced_set) if forced_set else 0.0
    forced_recall = forced_true_positive / len(truth)
    forced_f1 = 2.0 * forced_precision * forced_recall / (forced_precision + forced_recall) if forced_precision + forced_recall else 0.0
    max_contribution_error = 0.0
    min_psc = 1.0
    for left, right in pair.planted_hyperedges:
        left_array = np.asarray(left, dtype=int)
        right_array = np.asarray(right, dtype=int)
        yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, left_array)
        yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, right_array)
        max_contribution_error = max(max_contribution_error, float(np.max(np.abs(yl - yr))))
        psc = projector_subspace_consistency(pair.d_left[:, left_array], pair.d_right[:, right_array])
        min_psc = min(min_psc, float(psc.value))
    shared_left = set(pair.planted_hyperedges[0][0]) & set(pair.planted_hyperedges[1][0])
    shared_right = set(pair.planted_hyperedges[0][1]) & set(pair.planted_hyperedges[1][1])
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "seed_provenance": seed_provenance,
        "expected_decision": pair.expected_decision,
        "decision": "OVERLAPPING_HYPERGRAPH" if predicted_set == truth else "INCORRECT",
        "planted_hyperedges": [
            {"left_ids": list(left), "right_ids": list(right)} for left, right in pair.planted_hyperedges
        ],
        "predicted_hyperedges": [
            {"left_ids": list(left), "right_ids": list(right)} for left, right in predicted
        ],
        "hyperedge_precision": precision,
        "hyperedge_recall": recall,
        "hyperedge_f1": f1,
        "shared_left_atoms": sorted(shared_left),
        "shared_right_atoms": sorted(shared_right),
        "forced_partition_edges": [candidate_json(item) for item in forced],
        "forced_partition_precision": forced_precision,
        "forced_partition_recall": forced_recall,
        "forced_partition_f1": forced_f1,
        "oracle": search_json(search, "global_overlap_pool", pool, pool),
        "max_explicit_contribution_error": max_contribution_error,
        "min_planted_psc": min_psc,
        "ari_status": "not_applicable_overlapping_hypergraph",
    }
def _unequal_split_merge_record_impl(config: dict, generator_seed: int, seed_provenance: dict) -> dict:
    pair = unequal_split_merge_seeded(
        config["block_count"],
        config["n_mean"],
        config["n_eval"],
        structural_seed_a=seed_provenance["structural_seed_a"],
        structural_seed_b=seed_provenance["structural_seed_b"],
        mean_sample_seed=seed_provenance["mean_sample_seed"],
        eval_sample_seed=seed_provenance["eval_sample_seed"],
    )
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
    klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
    krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
    recovered = 0
    passing_candidates = 0
    cross_matches = 0
    max_residual = 0.0
    max_contribution_error = 0.0
    min_psc = 1.0
    candidate_pairs_evaluated = 0
    oracle_neighborhoods = []
    planted_solver_gaps = []
    planted_tie_sizes = []
    runtime_seconds = 0.0
    for index, (left, right) in enumerate(pair.planted_hyperedges):
        search = exhaustive_balanced_search(
            kll, klr, krr, left, right,
            residual_tolerance=config["residual_tolerance"],
            tie_tolerance=config.get("tie_tolerance", 0.0),
            max_group_size=config["max_group_size"],
        )
        oracle_neighborhoods.append(search_json(search, "planted", left, right))
        runtime_seconds += search.elapsed_seconds
        candidate_pairs_evaluated += search.evaluated_count
        found = list(search.support_minimal_candidates)
        passing_candidates += len(found)
        planted_tie_sizes.append(len(search.tie_set))
        if search.solver_gap is not None:
            planted_solver_gaps.append(search.solver_gap)
        if len(found) == 1 and found[0].left_ids == left and found[0].right_ids == right:
            recovered += 1
            max_residual = max(max_residual, found[0].normalized_residual)
        left_array = np.asarray(left, dtype=int)
        right_array = np.asarray(right, dtype=int)
        yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, left_array)
        yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, right_array)
        max_contribution_error = max(max_contribution_error, float(np.max(np.abs(yl - yr))))
        psc = projector_subspace_consistency(pair.d_left[:, left_array], pair.d_right[:, right_array])
        min_psc = min(min_psc, float(psc.value))
        wrong_right = pair.planted_hyperedges[(index + 1) % len(pair.planted_hyperedges)][1]
        cross_search = exhaustive_balanced_search(
            kll, klr, krr, left, wrong_right,
            residual_tolerance=config["residual_tolerance"],
            tie_tolerance=config.get("tie_tolerance", 0.0),
            max_group_size=config["max_group_size"],
        )
        oracle_neighborhoods.append(search_json(cross_search, "cross_block_negative", left, wrong_right))
        runtime_seconds += cross_search.elapsed_seconds
        cross_matches += len(cross_search.support_minimal_candidates)
        candidate_pairs_evaluated += cross_search.evaluated_count
    unequal_groups = sum(len(left) != len(right) for left, right in pair.planted_hyperedges)
    return {
        "family_id": pair.family_id,
        "generator_seed": generator_seed,
        "blocks": len(pair.planted_hyperedges),
        "recovered_blocks": recovered,
        "exact_recovery": recovered == len(pair.planted_hyperedges) and cross_matches == 0,
        "support_minimal_candidates": passing_candidates,
        "spurious_cross_block_candidates": cross_matches,
        "unequal_groups": unequal_groups,
        "candidate_pairs_evaluated": candidate_pairs_evaluated,
        "max_planted_normalized_residual": max_residual,
        "min_planted_psc": min_psc,
        "max_explicit_contribution_error": max_contribution_error,
        "ari_status": "not_applicable_unequal_feature_universes",
        "split_weights": list(pair.split_weights),
        "planted_hyperedges": [
            {"left_ids": list(left), "right_ids": list(right)} for left, right in pair.planted_hyperedges
        ],
        "seed_provenance": seed_provenance,
        "oracle_neighborhoods": oracle_neighborhoods,
        "minimum_planted_solver_gap": min(planted_solver_gaps),
        "maximum_planted_tie_set_size": max(planted_tie_sizes),
        "oracle_runtime_seconds": runtime_seconds,
        "diagnostics_complete": len(oracle_neighborhoods) == 2 * len(pair.planted_hyperedges),
    }


def evaluate(config: dict) -> tuple[list[dict], dict]:
    records = []
    families = set(config["families"])
    if "F01_hadamard_gauge" in families:
        for offset in range(config.get("seed_pair_count", 1)):
            for q in config["q_values"]:
                generator_seed = config["base_seed"] + 100 * offset + q
                seed_root = 10 * generator_seed
                records.append(metric_record(
                    hadamard_gauge_seeded(
                        q, config["n_mean"], config["n_eval"],
                        structural_seed_a=seed_root + 1,
                        structural_seed_b=seed_root + 2,
                        mean_sample_seed=seed_root + 3,
                        eval_sample_seed=seed_root + 4,
                    ),
                    config,
                    q,
                    generator_seed,
                ))
    if "F10_same_span_different_computation" in families:
        for offset in range(config.get("seed_pair_count", 1)):
            generator_seed = config["base_seed"] + 1000 + offset
            if config.get("formal_f10_f11", False):
                records.append(function_mismatch_record(config, generator_seed))
            else:
                f10 = same_span_different_computation(config["n_mean"], 10 * config["n_eval"], generator_seed)
                records.append(metric_record(f10, config, generator_seed=generator_seed))
    if "F11_same_sum_bloated_span" in families:
        for offset in range(config.get("seed_pair_count", 1)):
            generator_seed = config["base_seed"] + 2000 + offset
            if config.get("formal_f10_f11", False):
                records.append(span_bloat_record(config, generator_seed))
            else:
                f11 = same_sum_bloated_span(config["n_mean"], config["n_eval"], generator_seed)
                records.append(metric_record(f11, config, generator_seed=generator_seed))
    if "F02_local_block_rotations" in families:
        records.extend(local_rotation_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F03_unequal_split_merge" in families:
        records.extend(unequal_split_merge_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F04_partial_overlap" in families:
        records.extend(partial_overlap_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F06_cooccurrence_confounding" in families:
        records.extend(cooccurrence_confounding_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F08_competing_covers" in families:
        records.extend(competing_covers_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F09_whole_dictionary_only" in families:
        records.extend(whole_dictionary_only_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F05_cancellation" in families:
        records.extend(cancellation_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F07_rare_occupancy" in families:
        records.extend(rare_occupancy_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if "F12_non_lipschitz_downstream_cliff" in families:
        records.extend(downstream_cliff_record(config, config["base_seed"] + offset) for offset in range(config["seed_pair_count"]))
    if config.get("emit_complete_metric_surface", False):
        records = [attach_complete_metric_surface(record, config) for record in records]
    atol = config["absolute_tolerance"]
    rtol = config["relative_tolerance"]
    checks = []
    metric_surface_error_count = 0
    if config.get("emit_complete_metric_surface", False):
        metric_surface_error_count = sum(len(metric_surface_errors(record)) for record in records)
        checks.append(metric_surface_error_count == 0)
    for record in records:
        if record["family_id"] == "F01_hadamard_gauge":
            checks.extend([
                np.isclose(record["bcc"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["psc"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["pw_mcc"], record["pw_mcc_expected"], atol=atol, rtol=rtol),
                record["explicit_contribution_max_error"] <= atol,
            ])
        elif record["family_id"] == "F10_same_span_different_computation":
            if config.get("formal_f10_f11", False):
                checks.extend([
                    record["decision"] == record["expected_decision"],
                    np.isclose(record["risk_group_psc"], 1.0, atol=atol, rtol=rtol),
                    abs(record["risk_group_bcc"]) <= config["maximum_abs_f10_bcc"],
                    record["risk_group_residual"] >= config["minimum_f10_residual"],
                    record["risk_contribution_rmse"] >= config["minimum_f10_contribution_rmse"],
                    np.isclose(record["clean_group_bcc"], 1.0, atol=atol, rtol=rtol),
                    np.isclose(record["clean_group_psc"], 1.0, atol=atol, rtol=rtol),
                    len(record["risk_oracle"]["passing_candidates"]) == 0,
                    len(record["clean_oracle"]["support_minimal_candidates"]) == 1,
                    np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                    np.isclose(record["span_only_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                ])
            else:
                checks.extend([np.isclose(record["psc"], 1.0, atol=atol, rtol=rtol), abs(record["bcc"]) < 0.05])
        elif record["family_id"] == "F11_same_sum_bloated_span":
            if config.get("formal_f10_f11", False):
                checks.extend([
                    record["decision"] == record["expected_decision"],
                    np.isclose(record["risk_group_bcc"], 1.0, atol=atol, rtol=rtol),
                    record["risk_group_residual"] <= config["residual_tolerance"],
                    np.isclose(record["risk_group_psc"], 2.0 / 3.0, atol=atol, rtol=rtol),
                    record["risk_rank_left"] == 1 and record["risk_rank_right"] == 2,
                    record["risk_contribution_max_error"] <= atol,
                    np.isclose(record["clean_group_bcc"], 1.0, atol=atol, rtol=rtol),
                    np.isclose(record["clean_group_psc"], 1.0, atol=atol, rtol=rtol),
                    len(record["risk_oracle"]["support_minimal_candidates"]) == 1,
                    len(record["clean_oracle"]["support_minimal_candidates"]) == 1,
                    np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                    np.isclose(record["no_psc_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                ])
            else:
                checks.extend([
                    np.isclose(record["bcc"], 1.0, atol=atol, rtol=rtol),
                    np.isclose(record["psc"], 2.0 / 3.0, atol=atol, rtol=rtol),
                ])
        elif record["family_id"] == "F02_local_block_rotations":
            checks.extend([
                record["decision"] == record["expected_decision"],
                record["exact_recovery"],
                record["support_minimal_candidates"] == record["blocks"],
                record["spurious_cross_block_candidates"] == 0,
                record["max_planted_normalized_residual"] <= config["residual_tolerance"],
                np.isclose(record["min_planted_psc"], 1.0, atol=atol, rtol=rtol),
                record["max_explicit_contribution_error"] <= atol,
                record["max_mean_contribution_error_norm"] <= atol,
                record["diagnostics_complete"],
                record["maximum_planted_tie_set_size"] == 1,
                record["minimum_planted_solver_gap"] > config.get("tie_tolerance", 0.0),
                np.isclose(record["ari_left"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["ari_right"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["coverage"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["unmatched_energy_fraction"], 0.0, atol=atol, rtol=rtol),
            ])
        elif record["family_id"] == "F03_unequal_split_merge":
            checks.extend([
                record["exact_recovery"],
                record["support_minimal_candidates"] == record["blocks"],
                record["spurious_cross_block_candidates"] == 0,
                record["unequal_groups"] == record["blocks"],
                record["max_planted_normalized_residual"] <= config["residual_tolerance"],
                np.isclose(record["min_planted_psc"], 1.0, atol=atol, rtol=rtol),
                record["max_explicit_contribution_error"] <= atol,
                record["ari_status"] == "not_applicable_unequal_feature_universes",
                record["diagnostics_complete"],
                record["maximum_planted_tie_set_size"] == 1,
                record["minimum_planted_solver_gap"] > config.get("tie_tolerance", 0.0),
            ])
        elif record["family_id"] == "F04_partial_overlap":
            checks.extend([
                record["decision"] == record["expected_decision"],
                np.isclose(record["hyperedge_precision"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["hyperedge_recall"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["hyperedge_f1"], 1.0, atol=atol, rtol=rtol),
                len(record["shared_left_atoms"]) > 0,
                len(record["shared_right_atoms"]) > 0,
                np.isclose(record["forced_partition_precision"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["forced_partition_recall"], 0.5, atol=atol, rtol=rtol),
                record["oracle"]["evaluated_count"] == 36,
                len(record["oracle"]["support_minimal_candidates"]) == 2,
                len(record["oracle"]["tie_set"]) == 2,
                np.isclose(record["min_planted_psc"], 1.0, atol=atol, rtol=rtol),
                record["max_explicit_contribution_error"] <= atol,
                record["ari_status"] == "not_applicable_overlapping_hypergraph",
            ])
        elif record["family_id"] == "F06_cooccurrence_confounding":
            checks.extend([
                record["decision"] == record["expected_decision"],
                record["proposal_accepts"],
                record["proposal_score_abs_code_correlation"] >= config["proposal_min_abs_code_correlation"],
                abs(record["bcc"]) <= config["maximum_abs_bcc"],
                record["normalized_residual"] >= config["minimum_contribution_residual"],
                record["psc"] <= config["maximum_psc"],
                record["contribution_rmse"] >= config["minimum_contribution_rmse"],
                len(record["oracle"]["all_candidates"]) == 1,
                len(record["oracle"]["passing_candidates"]) == 0,
                len(record["oracle"]["support_minimal_candidates"]) == 0,
                len(record["oracle"]["tie_set"]) == 1,
                np.isclose(record["coverage"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["unmatched_energy_fraction"], 1.0, atol=atol, rtol=rtol),
            ])
        elif record["family_id"] == "F08_competing_covers":
            checks.extend([
                record["decision"] == record["expected_decision"],
                record["maximum_cover_count"] == 2,
                record["maximum_cover_cardinality"] == 2,
                record["exact_cover_count"] == 2,
                np.isclose(record["ambiguity_accuracy"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["forced_best_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                record["oracle"]["evaluated_count"] == 4,
                len(record["oracle"]["support_minimal_candidates"]) == 4,
                len(record["oracle"]["tie_set"]) == 4,
            ])
        elif record["family_id"] == "F09_whole_dictionary_only":
            checks.extend([
                record["decision"] == record["expected_decision"],
                np.isclose(record["global_balance_bcc"], 1.0, atol=atol, rtol=rtol),
                record["global_balance_residual"] <= config["residual_tolerance"],
                record["global_contribution_max_error"] <= atol,
                record["global_mean_contribution_error_norm"] <= atol,
                record["local_passing_count"] == 0,
                np.isclose(record["refusal_accuracy"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["global_collapse_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                record["oracle"]["evaluated_count"] == 49,
                len(record["oracle"]["support_minimal_candidates"]) == 1,
                len(record["oracle"]["tie_set"]) == 1,
            ])
        elif record["family_id"] == "F05_cancellation":
            checks.extend([
                record["decision"] == record["expected_decision"],
                np.isclose(record["risk_group_bcc"], 1.0, atol=atol, rtol=rtol),
                record["risk_group_residual"] <= config["residual_tolerance"],
                np.isclose(record["risk_group_psc"], 1.0, atol=atol, rtol=rtol),
                record["risk_group_max_contribution_error"] <= atol,
                record["risk_flag"],
                min(record["risk_cancellation_energy_ratio_left"], record["risk_cancellation_energy_ratio_right"]) >= config["cancellation_energy_ratio_threshold"],
                min(record["risk_max_leave_one_out_ratio_left"], record["risk_max_leave_one_out_ratio_right"]) >= config["leave_one_out_energy_ratio_threshold"],
                np.isclose(record["clean_group_bcc"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["clean_group_psc"], 1.0, atol=atol, rtol=rtol),
                record["clean_group_max_contribution_error"] <= atol,
                not record["clean_flag"],
                np.isclose(record["clean_cancellation_energy_ratio_left"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["clean_cancellation_energy_ratio_right"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["diagnostic_recall"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["clean_specificity"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["no_diagnostic_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                len(record["risk_oracle"]["support_minimal_candidates"]) == 1,
                len(record["clean_oracle"]["support_minimal_candidates"]) == 1,
            ])
        elif record["family_id"] == "F07_rare_occupancy":
            checks.extend([
                record["decision"] == record["expected_decision"],
                record["risk_passes_point_threshold"],
                record["risk_active_tokens_left"] == 2 * config["active_document_count"],
                record["risk_active_tokens_right"] == 2 * config["active_document_count"],
                record["risk_active_documents_left"] == config["active_document_count"],
                record["risk_active_documents_right"] == config["active_document_count"],
                max(record["risk_document_energy_kish_ess_left"], record["risk_document_energy_kish_ess_right"]) <= config["maximum_risk_document_ess"],
                record["risk_bootstrap_inactive_fraction"] >= config["minimum_risk_inactive_bootstrap_fraction"],
                record["risk_bootstrap_ci_width"] >= config["minimum_risk_bcc_ci_width"],
                record["risk_low_support_flag"],
                record["clean_passes_point_threshold"],
                min(record["clean_document_energy_kish_ess_left"], record["clean_document_energy_kish_ess_right"]) >= config["minimum_clean_document_ess"],
                np.isclose(record["clean_bootstrap_inactive_fraction"], 0.0, atol=atol, rtol=rtol),
                record["clean_bootstrap_ci_width"] <= config["maximum_clean_bcc_ci_width"],
                record["clean_stability_flag"],
                np.isclose(record["refusal_accuracy"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["clean_specificity"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["false_unique_rate"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["naive_token_count_false_unique_rate"], 1.0, atol=atol, rtol=rtol),
                len(record["risk_oracle"]["support_minimal_candidates"]) == 1,
                len(record["clean_oracle"]["support_minimal_candidates"]) == 1,
            ])
        elif record["family_id"] == "F12_non_lipschitz_downstream_cliff":
            checks.extend([
                record["decision"] == record["expected_decision"],
                record["bcc"] >= config["minimum_f12_bcc"],
                record["normalized_residual"] <= config["maximum_f12_residual"],
                np.isclose(record["psc"], 1.0, atol=atol, rtol=rtol),
                record["mean_contribution_error_norm"] <= atol,
                record["contribution_rmse"] <= config["maximum_f12_contribution_rmse"],
                record["risk_endpoint_mismatch_rate"] >= config["minimum_f12_risk_mismatch"],
                record["maximum_adjacent_risk_jump"] >= config["minimum_f12_cliff_jump"],
                record["cliff_hook_step_rmse"] <= config["maximum_f12_cliff_hook_step"],
                np.isclose(record["smooth_transfer_ratio"], 1.0, atol=atol, rtol=rtol),
                np.isclose(record["false_causal_certificate_rate"], 0.0, atol=atol, rtol=rtol),
                np.isclose(record["bcc_only_false_causal_rate"], 1.0, atol=atol, rtol=rtol),
                len(record["dose_curve"]) == len(config["f12_dose_grid"]),
            ])
    summary = {
        "status": "PASS" if all(checks) else "FAIL",
        "checks_passed": int(sum(bool(x) for x in checks)),
        "checks_total": len(checks),
        "families_covered": sorted({r["family_id"] for r in records}),
        "records": len(records),
        "scope_warning": config.get("scope_warning", "Smoke conformance only; not an M1 gate result."),
        "metric_surface_schema": "v1" if config.get("emit_complete_metric_surface", False) else "not_requested",
        "metric_surface_error_count": metric_surface_error_count,
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run_dir = args.output_root.resolve() / args.run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    code_files = [
        Path(__file__).resolve(),
        ROOT / "src" / "ccad" / "metrics.py",
        ROOT / "src" / "ccad" / "synthetic.py",
        ROOT / "src" / "ccad" / "artifacts.py",
        ROOT / "src" / "ccad" / "matching.py",
        args.config.resolve(),
    ]
    code_entries = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_files]
    snapshot_root = run_dir / "source_snapshot"
    for entry, source_path in zip(code_entries, code_files):
        snapshot_path = snapshot_root / entry["path"]
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot_path)
        entry["snapshot_path"] = str(snapshot_path.relative_to(run_dir))
        if sha256(snapshot_path) != entry["sha256"]:
            raise RuntimeError(f"source snapshot hash mismatch: {entry['path']}")
    aggregate = hashlib.sha256("".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(code_entries, key=lambda x: x["path"])).encode()).hexdigest()
    stable_json(run_dir / "config.resolved.json", config)
    stable_json(run_dir / "code_hashes.json", {
        "files": code_entries,
        "aggregate_sha256": aggregate,
        "snapshot_root": "source_snapshot",
    })
    input_files = [ROOT / "SYNTHETIC_SUITE_SPEC.md", ROOT / "goal_aligned_subspace_consistency_complete_proofs.pdf"]
    stable_json(run_dir / "inputs.json", {"inputs": [{
        "path": str(p),
        "sha256": sha256(p),
        "bytes": p.stat().st_size,
        "source": "CCAD project-local artifact",
        "license_or_access_boundary": "internal project material; no redistribution authorized",
        "role": "specification_or_theory",
    } for p in input_files]})
    stable_json(run_dir / "environment.json", {
        "os": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda": "not_applicable",
        "pytorch": "not_applicable",
        "transformers": "not_applicable",
        "sae_framework": "not_applicable",
        "git_available": (ROOT / ".git").exists(),
    })
    stable_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0",
        "run_id": args.run_id,
        "run_parent": config.get("run_parent", "R001"),
        "purpose": config.get("purpose", "Synthetic deterministic conformance"),
        "milestone": "M1",
        "evidence_level": config.get("evidence_level", "synthetic_smoke"),
        "automation_id": "ccad",
        "started_utc": started.isoformat(),
        "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate,
        "source_snapshot_required": True,
        "audit_opened": config["audit_opened"],
        "candidate_family_frozen": config["candidate_family_frozen"],
        "mean_constants_source_split": config["mean_constants_source_split"],
        "threshold_source_split": config["threshold_source_split"],
        "statistics_unit": config["statistics_unit"],
        "device": "cpu",
        "seeds": {"base_seed": config["base_seed"], "derivation": "stored in resolved config and raw records"},
        "resource_lease": config["resource_lease"],
        "resource_lease_reason": config["resource_lease_reason"],
    })
    stable_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    try:
        records, summary = evaluate(config)
        with (run_dir / "metrics.raw.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary["metrics_raw_sha256"] = sha256(run_dir / "metrics.raw.jsonl")
        summary["generator_script_sha256"] = sha256(Path(__file__).resolve())
        stable_json(run_dir / "metrics.summary.json", summary)
        stdout_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        final = summary["status"]
    except Exception:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        final = "FAIL"
    ended = datetime.now(timezone.utc)
    stable_json(run_dir / "status.json", {"status": final, "updated_utc": ended.isoformat(), "ended_utc": ended.isoformat()})
    validation = validate_run_directory(run_dir)
    stable_json(run_dir / "contract.validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    if not validation.ok:
        final = "FAIL"
        stable_json(run_dir / "status.json", {
            "status": final,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "ended_utc": ended.isoformat(),
            "failure_kind": "artifact_contract",
        })
    print(json.dumps({"run_id": args.run_id, "status": final, "run_dir": str(run_dir)}, sort_keys=True))
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
