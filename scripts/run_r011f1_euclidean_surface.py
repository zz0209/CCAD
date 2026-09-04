"""Fit the frozen 40-query Euclidean FCC discovery/calibration surface."""
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
from scipy.sparse import csr_matrix  # noqa: E402

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.causal_metric_probe import select_document_balanced_states  # noqa: E402
from ccad.fuzzy_correspondence import (  # noqa: E402
    evaluate_fixed_correspondence_from_kernels,
    fit_fuzzy_correspondence_from_kernels,
    soft_membership_overlap,
    sparse_contribution_kernels,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, source: str, role: str) -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": "internal", "role": role,
    }


def sparse_codes(asset_dir: Path, split: str, seed: int, tokens: int, k: int, latents: int) -> csr_matrix:
    base = asset_dir / split / f"seed_{seed}"
    indices = np.memmap(base / "top_indices.uint16.bin", dtype="<u2", mode="r", shape=(tokens, k))
    acts = np.memmap(base / "top_acts.float32.bin", dtype="<f4", mode="r", shape=(tokens, k))
    indptr = np.arange(0, (tokens + 1) * k, k, dtype=np.int64)
    matrix = csr_matrix(
        (np.asarray(acts, dtype=np.float32).reshape(-1).copy(), np.asarray(indices, dtype=np.int32).reshape(-1).copy(), indptr),
        shape=(tokens, latents), copy=False,
    )
    matrix.sort_indices()
    return matrix


def decoder(asset_dir: Path, seed: int, latents: int, hidden: int) -> np.ndarray:
    return np.asarray(
        np.memmap(asset_dir / "decoders" / f"seed_{seed}.float32.bin", dtype="<f4", mode="r", shape=(latents, hidden)),
    )


def condition_weights(matrix: csr_matrix, atoms: list[int], power: float, maximum_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    dense = matrix[:, atoms].toarray().astype(np.float64)
    scores = np.sum(np.abs(dense) ** power, axis=1)
    positive = np.flatnonzero(scores > 0)
    if positive.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    order = np.lexsort((positive, -scores[positive]))[:maximum_tokens]
    rows = positive[order]
    weights = scores[rows]
    weights /= weights.sum()
    return rows.astype(np.int64), weights


def sequence_and_document_stats(
    rows: np.ndarray,
    weights: np.ndarray,
    sequence_documents: dict[int, tuple[str, ...]],
    context_length: int,
) -> tuple[int, float]:
    sequences = rows // context_length
    document_weights: dict[str, float] = {}
    for sequence, weight in zip(sequences.tolist(), weights.tolist()):
        documents = sequence_documents[int(sequence)]
        share = weight / len(documents)
        for document in documents:
            document_weights[document] = document_weights.get(document, 0.0) + share
    values = np.asarray(list(document_weights.values()), dtype=np.float64)
    values /= values.sum()
    return len(set(sequences.tolist())), float(1.0 / np.sum(values * values))


def source_family(
    matrix: csr_matrix,
    query_atom: int,
    means: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> list[int]:
    selected = matrix[rows]
    query = selected[:, query_atom].toarray().reshape(-1).astype(np.float64)
    weighted = selected.multiply(weights[:, None])
    raw = np.asarray(query @ weighted, dtype=np.float64).reshape(-1)
    weighted_query = float(query @ weights)
    weighted_all = np.asarray(selected.T @ weights, dtype=np.float64).reshape(-1)
    covariance = raw - weighted_query * means - means[query_atom] * weighted_all + means[query_atom] * means
    ids = np.arange(matrix.shape[1])
    ordered = np.lexsort((ids, -np.abs(covariance)))
    family = [query_atom]
    family.extend(int(atom) for atom in ordered if int(atom) != query_atom and len(family) < count)
    return family


def target_family(candidate: dict, methods: list[str], cap: int) -> list[int]:
    family: list[int] = []
    seen: set[int] = set()
    for rank in range(32):
        for method in methods:
            value = candidate[method]
            if isinstance(value, list):
                if rank >= len(value):
                    continue
                atom = int(value[rank])
            else:
                if rank != 0:
                    continue
                atom = int(value)
            if atom not in seen:
                seen.add(atom)
                family.append(atom)
                if len(family) == cap:
                    return family
    return family


def local_kernels(
    source_matrix: csr_matrix,
    target_matrix: csr_matrix,
    source_ids: list[int],
    target_ids: list[int],
    source_decoder: np.ndarray,
    target_decoder: np.ndarray,
    source_means: np.ndarray,
    target_means: np.ndarray,
    positive_rows: np.ndarray,
    positive_weights: np.ndarray,
    negative_rows: np.ndarray | None = None,
    negative_weights: np.ndarray | None = None,
):
    all_rows = positive_rows if negative_rows is None else np.unique(np.concatenate([positive_rows, negative_rows]))
    position = {int(row): index for index, row in enumerate(all_rows.tolist())}
    positive = np.zeros(len(all_rows), dtype=np.float64)
    for row, weight in zip(positive_rows.tolist(), positive_weights.tolist()):
        positive[position[int(row)]] = weight
    negative = None
    if negative_rows is not None and negative_weights is not None:
        negative = np.zeros(len(all_rows), dtype=np.float64)
        for row, weight in zip(negative_rows.tolist(), negative_weights.tolist()):
            negative[position[int(row)]] = weight
    return sparse_contribution_kernels(
        source_matrix[all_rows][:, source_ids], target_matrix[all_rows][:, target_ids],
        source_decoder[source_ids], target_decoder[target_ids], positive,
        source_mean_codes=source_means[source_ids], target_mean_codes=target_means[target_ids],
        negative_weights=negative,
    )


def fixed_metrics(kernels, source_loadings: np.ndarray, target_loadings: np.ndarray, negative: bool = False):
    if negative:
        return evaluate_fixed_correspondence_from_kernels(
            kernels.negative_source_gram, kernels.negative_target_gram, kernels.negative_cross_gram,
            source_loadings, target_loadings,
        )
    return evaluate_fixed_correspondence_from_kernels(
        kernels.source_gram, kernels.target_gram, kernels.cross_gram,
        source_loadings, target_loadings,
    )


def embedded_membership(ids: list[int], values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float64)
    result[np.asarray(ids, dtype=np.int64)] = values
    return result


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
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/fuzzy_correspondence.py", ROOT / "src/ccad/causal_metric_probe.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})

    paths = {
        "protocol": ROOT / cfg["protocol_config_path"],
        "panel": ROOT / cfg["query_panel_path"],
        "census": ROOT / cfg["source_census_path"],
        "candidates": ROOT / cfg["candidate_path"],
        "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json",
        "sequence_records": ROOT / cfg["sequence_records_path"],
    }
    input_rows = [file_entry(args.config.resolve(), "CCAD frozen run config", "run_protocol")]
    input_rows.extend(file_entry(path, "CCAD frozen upstream", name) for name, path in paths.items())
    write_json(run_dir / "inputs.json", {"inputs": input_rows})
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
        "resource_lease_reason": "sparse discovery/calibration FCC kernels over frozen D: assets",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})

    record, error, status = None, None, "FAIL"
    output_rows: list[dict] = []
    try:
        expected_hashes = {
            "protocol": cfg["protocol_config_sha256"], "panel": cfg["query_panel_sha256"],
            "census": cfg["source_census_sha256"], "candidates": cfg["candidate_sha256"],
            "asset_manifest": cfg["asset_manifest_sha256"], "sequence_records": cfg["sequence_records_sha256"],
        }
        bound = {name: sha256(paths[name]).lower() == value.lower() for name, value in expected_hashes.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        if protocol["audit_opened"] or cfg["audit_opened"] or cfg["forbidden_splits"] != ["audit"]:
            raise ValueError("audit boundary drift")
        panel = [json.loads(line) for line in paths["panel"].read_text(encoding="utf-8").splitlines() if line]
        census = [json.loads(line) for line in paths["census"].read_text(encoding="utf-8").splitlines() if line]
        candidates = [json.loads(line) for line in paths["candidates"].read_text(encoding="utf-8").splitlines() if line]
        stats = {(int(row["seed"]), int(row["atom"])): row for row in census}
        candidate_map = {(int(row["source_seed"]), int(row["source_atom"]), int(row["target_seed"])): row for row in candidates}
        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in panel:
            grouped.setdefault((int(row["seed"]), int(row["energy_stratum"])), []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: row["selection_hash"])
        anchors = [rows[0] for _, rows in sorted(grouped.items())]
        anchor_neighbors = {
            (int(rows[0]["seed"]), int(rows[0]["atom"])): [int(row["atom"]) for row in rows[1:1 + cfg["collision_neighbor_queries"]]]
            for rows in grouped.values()
        }
        condition_rows = [row for _, rows in sorted(grouped.items()) for row in rows[:1 + cfg["collision_neighbor_queries"]]]
        if len(anchors) != cfg["anchor_queries"] or len(condition_rows) != cfg["all_condition_queries"]:
            raise ValueError("anchor/collision query count mismatch")

        asset_dir = Path(cfg["bulk_asset_dir"])
        asset_manifest = json.loads(paths["asset_manifest"].read_text(encoding="utf-8"))
        split_tokens = {row["split"]: int(row["tokens"]) for row in asset_manifest["splits"]}
        matrices = {
            split: {
                seed: sparse_codes(asset_dir, split, seed, split_tokens[split], cfg["k"], cfg["num_latents"])
                for seed in cfg["source_seeds"]
            }
            for split in cfg["splits"]
        }
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]) for seed in cfg["source_seeds"]}
        means = {
            seed: np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64)
            for seed in cfg["source_seeds"]
        }
        sequence_payload = json.loads(paths["sequence_records"].read_text(encoding="utf-8"))["sequences"]
        sequence_documents = {
            split: {
                int(row["sequence_index"]): tuple(sorted(str(value) for value in row["document_ids"]))
                for row in sequence_payload if row["split"] == split
            }
            for split in cfg["splits"]
        }
        global_states = select_document_balanced_states(
            sequence_payload, split="discovery", count=cfg["global_control_tokens"],
            token_positions=tuple(cfg["global_control_state_positions"]), salt=cfg["global_control_state_salt"],
        )
        global_rows = np.asarray(
            [int(row["sequence_index"]) * 128 + int(row["token_position"]) for row in global_states], dtype=np.int64,
        )
        global_weights = np.full(len(global_rows), 1.0 / len(global_rows))

        condition_cache: dict[tuple[str, int, int], dict] = {}
        family_cache: dict[tuple[int, int], list[int]] = {}
        for query in condition_rows:
            seed, atom = int(query["seed"]), int(query["atom"])
            neighbors = anchor_neighbors[(seed, int(grouped[(seed, int(query["energy_stratum"]))][0]["atom"]))]
            negative_atoms = [value for value in neighbors if value != atom][: cfg["collision_neighbor_queries"]]
            if len(negative_atoms) < cfg["collision_neighbor_queries"]:
                ordered_atoms = [int(row["atom"]) for row in grouped[(seed, int(query["energy_stratum"]))] if int(row["atom"]) != atom]
                negative_atoms = ordered_atoms[: cfg["collision_neighbor_queries"]]
            split_payload = {}
            for split in cfg["splits"]:
                pos_rows, pos_weights = condition_weights(
                    matrices[split][seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"],
                )
                neg_rows, neg_weights = condition_weights(
                    matrices[split][seed], negative_atoms, cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"],
                )
                pos_sequences, pos_doc_ess = sequence_and_document_stats(
                    pos_rows, pos_weights, sequence_documents[split], 128,
                ) if len(pos_rows) else (0, 0.0)
                split_payload[split] = {
                    "positive_rows": pos_rows, "positive_weights": pos_weights,
                    "negative_rows": neg_rows, "negative_weights": neg_weights,
                    "active_sequences": pos_sequences, "document_weight_ess": pos_doc_ess,
                }
            evaluable = (
                split_payload["discovery"]["active_sequences"] >= cfg["minimum_discovery_active_sequences"]
                and split_payload["calibration"]["active_sequences"] >= cfg["minimum_calibration_active_sequences"]
                and min(split_payload[split]["document_weight_ess"] for split in cfg["splits"]) >= cfg["minimum_document_weight_ess"]
                and all(len(split_payload[split]["negative_rows"]) > 0 for split in cfg["splits"])
            )
            condition_cache[("query", seed, atom)] = {
                "query": query, "negative_atoms": negative_atoms, "splits": split_payload, "evaluable": bool(evaluable),
            }
            if evaluable:
                family_cache[(seed, atom)] = source_family(
                    matrices["discovery"][seed], atom, means[seed],
                    split_payload["discovery"]["positive_rows"], split_payload["discovery"]["positive_weights"],
                    cfg["source_candidate_count"],
                )

        row_map: dict[tuple[int, int, int, int], dict] = {}
        anchor_loading_rows: list[tuple[np.ndarray, np.ndarray, int]] = []
        started_compute = time.perf_counter()
        for query in condition_rows:
            source_seed, atom, stratum = int(query["seed"]), int(query["atom"]), int(query["energy_stratum"])
            is_anchor = atom == int(grouped[(source_seed, stratum)][0]["atom"])
            condition = condition_cache[("query", source_seed, atom)]
            for target_seed in cfg["source_seeds"]:
                if target_seed == source_seed:
                    continue
                candidate = candidate_map[(source_seed, atom, target_seed)]
                target_ids = target_family(candidate, cfg["target_candidate_method_order"], cfg["target_candidate_cap"])
                if not condition["evaluable"]:
                    for rank in cfg["candidate_ranks"]:
                        row = {
                            "source_seed": source_seed, "source_atom": atom, "energy_stratum": stratum,
                            "target_seed": target_seed, "rank": rank, "query_role": "anchor" if is_anchor else "collision_neighbor",
                            "evaluable": False, "reason": "SOURCE_CONDITION_INSUFFICIENT",
                            "discovery_active_sequences": condition["splits"]["discovery"]["active_sequences"],
                            "calibration_active_sequences": condition["splits"]["calibration"]["active_sequences"],
                            "minimum_document_weight_ess": min(condition["splits"][split]["document_weight_ess"] for split in cfg["splits"]),
                        }
                        output_rows.append(row)
                        row_map[(source_seed, atom, target_seed, rank)] = row
                    continue
                source_ids = family_cache[(source_seed, atom)]
                discovery = local_kernels(
                    matrices["discovery"][source_seed], matrices["discovery"][target_seed], source_ids, target_ids,
                    decoders[source_seed], decoders[target_seed], means[source_seed], means[target_seed],
                    condition["splits"]["discovery"]["positive_rows"], condition["splits"]["discovery"]["positive_weights"],
                    condition["splits"]["discovery"]["negative_rows"], condition["splits"]["discovery"]["negative_weights"],
                )
                calibration = local_kernels(
                    matrices["calibration"][source_seed], matrices["calibration"][target_seed], source_ids, target_ids,
                    decoders[source_seed], decoders[target_seed], means[source_seed], means[target_seed],
                    condition["splits"]["calibration"]["positive_rows"], condition["splits"]["calibration"]["positive_weights"],
                    condition["splits"]["calibration"]["negative_rows"], condition["splits"]["calibration"]["negative_weights"],
                )
                global_kernel = local_kernels(
                    matrices["discovery"][source_seed], matrices["discovery"][target_seed], source_ids, target_ids,
                    decoders[source_seed], decoders[target_seed], means[source_seed], means[target_seed],
                    global_rows, global_weights,
                )
                for rank in cfg["candidate_ranks"]:
                    relation = fit_fuzzy_correspondence_from_kernels(
                        discovery, rank=rank, contrast_strength=cfg["contrast_strength"], ridge_fraction=cfg["ridge_fraction"],
                    )
                    global_relation = fit_fuzzy_correspondence_from_kernels(
                        global_kernel, rank=rank, contrast_strength=0.0, ridge_fraction=cfg["ridge_fraction"],
                    )
                    discovery_positive = fixed_metrics(discovery, relation.source_loadings, relation.target_loadings)
                    discovery_negative = fixed_metrics(discovery, relation.source_loadings, relation.target_loadings, negative=True)
                    calibration_positive = fixed_metrics(calibration, relation.source_loadings, relation.target_loadings)
                    calibration_negative = fixed_metrics(calibration, relation.source_loadings, relation.target_loadings, negative=True)
                    row = {
                        "source_seed": source_seed, "source_atom": atom, "energy_stratum": stratum,
                        "target_seed": target_seed, "rank": rank, "query_role": "anchor" if is_anchor else "collision_neighbor",
                        "evaluable": True, "reason": None, "source_candidate_ids": source_ids, "target_candidate_ids": target_ids,
                        "source_candidate_count": len(source_ids), "target_candidate_count": len(target_ids),
                        "negative_source_atoms": condition["negative_atoms"],
                        "discovery_active_sequences": condition["splits"]["discovery"]["active_sequences"],
                        "calibration_active_sequences": condition["splits"]["calibration"]["active_sequences"],
                        "minimum_document_weight_ess": min(condition["splits"][split]["document_weight_ess"] for split in cfg["splits"]),
                        "canonical_values": relation.canonical_values.tolist(),
                        "rank_boundary_relative_gap": relation.rank_boundary_relative_gap,
                        "source_effective_support": relation.source_effective_support,
                        "target_effective_support": relation.target_effective_support,
                        "source_membership": relation.source_membership.tolist(),
                        "target_membership": relation.target_membership.tolist(),
                        "global_target_membership": global_relation.target_membership.tolist(),
                        "discovery_positive_bcc": discovery_positive.bcc,
                        "discovery_negative_bcc": discovery_negative.bcc,
                        "discovery_bcc_contrast": discovery_positive.bcc - discovery_negative.bcc,
                        "discovery_positive_residual": discovery_positive.normalized_residual,
                        "calibration_positive_bcc": calibration_positive.bcc,
                        "calibration_negative_bcc": calibration_negative.bcc,
                        "calibration_bcc_contrast": calibration_positive.bcc - calibration_negative.bcc,
                        "calibration_positive_residual": calibration_positive.normalized_residual,
                        "loading_index": None,
                    }
                    if is_anchor:
                        row["loading_index"] = len(anchor_loading_rows)
                        anchor_loading_rows.append((relation.source_loadings, relation.target_loadings, len(target_ids)))
                    output_rows.append(row)
                    row_map[(source_seed, atom, target_seed, rank)] = row

        for anchor in anchors:
            source_seed, atom, stratum = int(anchor["seed"]), int(anchor["atom"]), int(anchor["energy_stratum"])
            neighbors = anchor_neighbors[(source_seed, atom)]
            for target_seed in cfg["source_seeds"]:
                if target_seed == source_seed:
                    continue
                for rank in cfg["candidate_ranks"]:
                    row = row_map[(source_seed, atom, target_seed, rank)]
                    if not row["evaluable"]:
                        row["query_collision_mean"] = None
                        row["global_collision_mean"] = None
                        row["collision_improvement_over_global"] = None
                        continue
                    anchor_query = embedded_membership(row["target_candidate_ids"], np.asarray(row["target_membership"]), cfg["num_latents"])
                    anchor_global = embedded_membership(row["target_candidate_ids"], np.asarray(row["global_target_membership"]), cfg["num_latents"])
                    query_overlaps, global_overlaps = [], []
                    for neighbor in neighbors:
                        other = row_map[(source_seed, neighbor, target_seed, rank)]
                        if not other["evaluable"]:
                            continue
                        other_query = embedded_membership(other["target_candidate_ids"], np.asarray(other["target_membership"]), cfg["num_latents"])
                        other_global = embedded_membership(other["target_candidate_ids"], np.asarray(other["global_target_membership"]), cfg["num_latents"])
                        query_overlaps.append(soft_membership_overlap(anchor_query, other_query))
                        global_overlaps.append(soft_membership_overlap(anchor_global, other_global))
                    row["query_collision_mean"] = float(np.mean(query_overlaps)) if query_overlaps else None
                    row["global_collision_mean"] = float(np.mean(global_overlaps)) if global_overlaps else None
                    row["collision_improvement_over_global"] = (
                        row["global_collision_mean"] - row["query_collision_mean"] if query_overlaps else None
                    )

        elapsed = time.perf_counter() - started_compute
        output_path = run_dir / "euclidean_fcc_surface.jsonl"
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        max_rank = max(cfg["candidate_ranks"])
        source_loadings = np.zeros((len(anchor_loading_rows), cfg["source_candidate_count"], max_rank), dtype=np.float32)
        target_loadings = np.zeros((len(anchor_loading_rows), cfg["target_candidate_cap"], max_rank), dtype=np.float32)
        target_counts = np.zeros(len(anchor_loading_rows), dtype=np.int16)
        loading_ranks = np.zeros(len(anchor_loading_rows), dtype=np.int8)
        for index, (left, right, target_count) in enumerate(anchor_loading_rows):
            source_loadings[index, :, : left.shape[1]] = left.astype(np.float32)
            target_loadings[index, :target_count, : right.shape[1]] = right.astype(np.float32)
            target_counts[index] = target_count
            loading_ranks[index] = left.shape[1]
        loadings_path = run_dir / "anchor_loadings.npz"
        np.savez_compressed(
            loadings_path, source_loadings=source_loadings, target_loadings=target_loadings,
            target_counts=target_counts, ranks=loading_ranks,
        )
        anchor_rows = [row for row in output_rows if row["query_role"] == "anchor"]
        evaluable_anchor_rows = [row for row in anchor_rows if row["evaluable"]]
        rank_summaries = []
        for rank in cfg["candidate_ranks"]:
            rows = [row for row in evaluable_anchor_rows if row["rank"] == rank]
            rank_summaries.append({
                "rank": rank, "evaluable_units": len(rows),
                "median_calibration_bcc": float(np.median([row["calibration_positive_bcc"] for row in rows])) if rows else None,
                "median_calibration_bcc_contrast": float(np.median([row["calibration_bcc_contrast"] for row in rows])) if rows else None,
                "positive_calibration_contrast_fraction": float(np.mean([row["calibration_bcc_contrast"] > 0 for row in rows])) if rows else None,
                "median_collision_improvement_over_global": float(np.median([row["collision_improvement_over_global"] for row in rows if row["collision_improvement_over_global"] is not None])) if rows else None,
            })
        expected_rows = cfg["all_condition_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"])
        expected_anchor_loadings = cfg["anchor_queries"] * cfg["ordered_target_seeds_per_query"] * len(cfg["candidate_ranks"])
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "anchor_and_collision_query_counts": len(anchors) == cfg["anchor_queries"] and len(condition_rows) == cfg["all_condition_queries"],
            "complete_query_target_rank_grid": len(output_rows) == expected_rows,
            "unique_surface_rows": len({(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]) for row in output_rows}) == len(output_rows),
            "candidate_budgets_respected": all((not row["evaluable"]) or (row["source_candidate_count"] == cfg["source_candidate_count"] and row["target_candidate_count"] <= cfg["target_candidate_cap"] and row["source_candidate_count"] * row["target_candidate_count"] <= protocol["feature_pair_budget"]) for row in output_rows),
            "all_anchor_loadings_saved": len(anchor_loading_rows) == sum(row["evaluable"] for row in anchor_rows) and len(anchor_loading_rows) <= expected_anchor_loadings,
            "finite_evaluable_metrics": all((not row["evaluable"]) or all(np.isfinite(row[name]) for name in ("discovery_positive_bcc", "discovery_negative_bcc", "calibration_positive_bcc", "calibration_negative_bcc", "calibration_positive_residual")) for row in output_rows),
            "anchor_collisions_complete_when_neighbors_evaluable": all((not row["evaluable"]) or row["collision_improvement_over_global"] is not None for row in anchor_rows),
            "no_found_or_threshold_decision": cfg["threshold_source_split"] == "none_raw_surface_only" and all("decision" not in row for row in output_rows),
            "calibration_but_no_audit": cfg["splits"] == ["discovery", "calibration"] and not cfg["audit_opened"],
        }
        checks = {name: bool(value) for name, value in checks.items()}
        record = {
            "checks": checks, "anchor_queries": len(anchors), "condition_queries": len(condition_rows),
            "surface_rows": len(output_rows), "evaluable_rows": sum(row["evaluable"] for row in output_rows),
            "evaluable_anchor_units": len({(row["source_seed"], row["source_atom"], row["target_seed"]) for row in evaluable_anchor_rows}),
            "rank_summaries": rank_summaries, "surface_sha256": sha256(output_path),
            "anchor_loadings_sha256": sha256(loadings_path), "wall_seconds": elapsed,
            "scope_limit": cfg["scope_limit"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version,
            "platform": platform.platform(), "overlay": cfg["scipy_overlay"],
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})

    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path),
        "generator_script_path": "scripts/run_r011f1_euclidean_surface.py",
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
