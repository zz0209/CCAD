from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ccad.metrics import center_codes, contribution_kernel
from ccad.proposal import (
    absolute_code_correlation_affinity,
    decoder_cosine_affinity,
    degree_matched_random_proposal,
    li15_spectral_proposal,
    proposal_candidate_family,
    singleton_contribution_affinity,
    symmetric_topk_proposal,
    validate_independent_split_seeds,
)
from ccad.synthetic import (
    cooccurrence_confounding_seeded,
    local_block_rotations_seeded,
    partial_overlap_seeded,
    unequal_split_merge_seeded,
)


ROOT = Path(__file__).resolve().parents[1]


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def proposal_hash(proposal) -> str:
    payload = {
        "score_source": proposal.score_source,
        "top_k": proposal.top_k,
        "edges": proposal.edges,
        "neighborhoods": [
            (item.anchor_left, item.anchor_right, item.left_ids, item.right_ids, item.status)
            for item in proposal.neighborhoods
        ],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest().upper()


def make_pair(family: str, settings: dict, config: dict, seeds: dict):
    common = dict(
        n_mean=config["n_mean"], n_eval=config["n_discovery"],
        structural_seed_a=seeds["structural_seed_a"],
        structural_seed_b=seeds["structural_seed_b"],
        mean_sample_seed=seeds["mean_sample_seed"],
        eval_sample_seed=seeds["discovery_sample_seed"],
    )
    if family == "F02_local_block_rotations":
        return local_block_rotations_seeded(tuple(settings["block_ranks"]), **common)
    if family == "F03_unequal_split_merge":
        return unequal_split_merge_seeded(settings["block_count"], **common)
    if family == "F04_partial_overlap":
        return partial_overlap_seeded(**common)
    if family == "F06_cooccurrence_confounding":
        return cooccurrence_confounding_seeded(**common)
    raise ValueError(f"unsupported family {family}")


def one_record(family: str, settings: dict, config: dict, pair_index: int) -> dict:
    root_seed = config["base_seed"] + 100 * pair_index + 1000 * list(config["families"]).index(family)
    seeds = {
        "structural_seed_a": root_seed + 1,
        "structural_seed_b": root_seed + 2,
        "mean_sample_seed": root_seed + 3,
        "discovery_sample_seed": root_seed + 4,
        "eval_sample_seed": root_seed + 5,
    }
    validate_independent_split_seeds(seeds)
    pair = make_pair(family, settings, config, seeds)
    z_left, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    z_right, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    k_ll = contribution_kernel(pair.d_left, z_left, pair.d_left, z_left)
    k_lr = contribution_kernel(pair.d_left, z_left, pair.d_right, z_right)
    k_rr = contribution_kernel(pair.d_right, z_right, pair.d_right, z_right)
    scores = {
        "CONTRIB-KNN": singleton_contribution_affinity(k_ll, k_lr, k_rr),
        "DECODER-KNN": decoder_cosine_affinity(pair.d_left, pair.d_right),
        "CODE-KNN": absolute_code_correlation_affinity(z_left, z_right),
    }
    results = []
    for top_k in config["top_k_grid"]:
        proposals = {
            name: symmetric_topk_proposal(
                score, top_k=top_k, score_source=name,
                max_neighborhood_atoms=config["max_neighborhood_atoms"],
            )
            for name, score in scores.items()
        }
        proposals["RANDOM-MATCHED"] = degree_matched_random_proposal(
            proposals["CONTRIB-KNN"],
            seed=root_seed + 10 + top_k,
            max_neighborhood_atoms=config["max_neighborhood_atoms"],
            swap_attempt_multiplier=config["random_swap_attempt_multiplier"],
        )
        for lane in (name for name in config["lanes"] if name != "LI15-SPECTRAL"):
            proposal = proposals[lane]
            candidate_counts = {
                str(max_group_size): len(proposal_candidate_family(proposal, max_group_size=max_group_size))
                for max_group_size in config["max_group_size_grid"]
            }
            results.append({
                "lane": lane,
                "top_k": top_k,
                "proposal_hash": proposal_hash(proposal),
                "edge_count": len(proposal.edges),
                "left_degrees": proposal.left_degrees,
                "right_degrees": proposal.right_degrees,
                "neighborhood_count": len(proposal.neighborhoods),
                "budget_refusal_count": sum(item.status == "BUDGET_REFUSAL" for item in proposal.neighborhoods),
                "candidate_counts_by_max_group_size": candidate_counts,
            })
    if "LI15-SPECTRAL" in config["lanes"]:
        spectral = li15_spectral_proposal(
            z_left,
            z_right,
            correlation_threshold=config["li15_spectral"]["correlation_threshold"],
            max_clusters=config["li15_spectral"]["max_clusters"],
            kmeans_seed=root_seed + config["li15_spectral"]["kmeans_seed_offset"],
            max_neighborhood_atoms=config["max_neighborhood_atoms"],
        )
        proposal = spectral.proposal
        results.append({
            "lane": "LI15-SPECTRAL",
            "top_k": None,
            "proposal_hash": proposal_hash(proposal),
            "edge_count": len(proposal.edges),
            "left_degrees": proposal.left_degrees,
            "right_degrees": proposal.right_degrees,
            "neighborhood_count": len(proposal.neighborhoods),
            "budget_refusal_count": sum(item.status == "BUDGET_REFUSAL" for item in proposal.neighborhoods),
            "candidate_counts_by_max_group_size": {
                str(max_group_size): len(proposal_candidate_family(proposal, max_group_size=max_group_size))
                for max_group_size in config["max_group_size_grid"]
            },
            "cluster_count": spectral.cluster_count,
            "mixed_cluster_count": spectral.mixed_cluster_count,
            "correlation_threshold": spectral.correlation_threshold,
            "eigenvalues": spectral.eigenvalues,
        })
    return {
        "family_id": family,
        "pair_index": pair_index,
        "seed_provenance": seeds,
        "feature_count_left": pair.d_left.shape[1],
        "feature_count_right": pair.d_right.shape[1],
        "held_out_eval_loaded": False,
        "planted_labels_loaded": False,
        "lanes": results,
    }


def run(config: dict) -> tuple[list[dict], dict]:
    if config["audit_opened"] or config["held_out_eval_loaded"] or config["planted_labels_loaded"]:
        raise ValueError("feasibility config must keep audit, held-out eval, and planted labels closed")
    expected_lanes = {"CONTRIB-KNN", "DECODER-KNN", "CODE-KNN", "RANDOM-MATCHED"}
    if config["protocol_version"] == "m1_unknown_support_feasibility.v2":
        expected_lanes.add("LI15-SPECTRAL")
    if set(config["lanes"]) != expected_lanes:
        raise ValueError("v1 lane set must match the preregistered implemented lanes exactly")
    records = [
        one_record(family, settings, config, pair_index)
        for family, settings in config["families"].items()
        for pair_index in range(config["seed_pair_count"])
    ]
    counts = [
        int(count)
        for record in records for result in record["lanes"]
        for count in result["candidate_counts_by_max_group_size"].values()
    ]
    refusal_count = sum(
        result["budget_refusal_count"] for record in records for result in record["lanes"]
    )
    summary = {
        "status": "PASS",
        "semantic_outcome": "ALL_PREREGISTERED_LANES_FEASIBLE" if "LI15-SPECTRAL" in expected_lanes else "IMPLEMENTED_LANES_FEASIBLE_LI15_PENDING",
        "evidence_level": config["evidence_level"],
        "audit_opened": False,
        "held_out_eval_loaded": False,
        "planted_labels_loaded": False,
        "record_count": len(records),
        "lane_result_count": sum(len(record["lanes"]) for record in records),
        "maximum_observed_candidate_count": max(counts),
        "candidate_common_budget_candidate": max(counts),
        "budget_rule": config["budget_rule"],
        "budget_refusal_count": refusal_count,
        "li15_spectral_status": "IMPLEMENTED_DISCOVERY_ONLY" if "LI15-SPECTRAL" in expected_lanes else "NOT_IMPLEMENTED_BLOCKS_FORMAL_FREEZE",
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stable_json(run_dir / "resolved_config.json", config)
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    sources = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "src/ccad/proposal.py",
        ROOT / "src/ccad/matching.py",
        ROOT / "src/ccad/metrics.py",
        ROOT / "src/ccad/synthetic.py",
    ]
    ledger = []
    for source in sources:
        target = snapshot / source.name
        shutil.copy2(source, target)
        ledger.append({"source": str(source), "snapshot": str(target.relative_to(run_dir)), "sha256": sha256(target)})
    stable_json(run_dir / "code_hashes.json", {"workspace_is_git_repository": False, "files": ledger})
    stable_json(run_dir / "environment.json", {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "device": "cpu",
        "started_utc": started.isoformat(),
    })
    stable_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    try:
        records, summary = run(config)
        raw_path = run_dir / "raw_records.jsonl"
        raw_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        summary["raw_records_sha256"] = sha256(raw_path)
        stable_json(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        final = "PASS"
    except Exception:
        (run_dir / "stdout.log").write_text("", encoding="utf-8")
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        final = "FAIL"
    ended = datetime.now(timezone.utc)
    stable_json(run_dir / "status.json", {"status": final, "updated_utc": ended.isoformat(), "ended_utc": ended.isoformat()})
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
