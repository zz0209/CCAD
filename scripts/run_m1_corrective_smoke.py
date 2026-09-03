from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_r001_smoke import evaluate as evaluate_diagnostics, metric_surface_errors  # noqa: E402
from ccad.matching import (  # noqa: E402
    BalancedCandidate,
    CandidateFamilySearchResult,
    FrozenDiscoveryPrediction,
    evaluate_frozen_hyperedges,
    freeze_discovery_prediction,
    full_universe_balanced_search,
    search_candidate_family,
)
from ccad.metrics import contribution_kernel  # noqa: E402
from ccad.proposal import (  # noqa: E402
    absolute_code_correlation_affinity,
    decoder_cosine_affinity,
    degree_matched_random_proposal,
    li15_spectral_proposal,
    proposal_candidate_family,
    singleton_contribution_affinity,
    symmetric_topk_proposal,
    validate_independent_split_seeds,
)
from ccad.synthetic import (  # noqa: E402
    cooccurrence_confounding_seeded,
    local_block_rotations_seeded,
    partial_overlap_seeded,
    unequal_split_merge_seeded,
)


def stable_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def stable_jsonl(path: Path, values: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest().upper()


def proposal_hash(proposal) -> str:
    payload = {
        "score_source": proposal.score_source,
        "top_k": proposal.top_k,
        "edges": proposal.edges,
        "neighborhoods": [
            (item.anchor_left, item.anchor_right, item.left_ids, item.right_ids, item.status, item.refusal_reason)
            for item in proposal.neighborhoods
        ],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest().upper()


def make_pair(family: str, full: dict, seeds: dict[str, int], sample_seed_field: str, n_samples: int):
    common = dict(
        n_mean=full["family_sample_exceptions"].get(family, {}).get("n_mean", full["n_mean_default"]),
        n_eval=n_samples,
        structural_seed_a=seeds["structural_seed_a"],
        structural_seed_b=seeds["structural_seed_b"],
        mean_sample_seed=seeds["mean_sample_seed"],
        eval_sample_seed=seeds[sample_seed_field],
    )
    if family == "F02_local_block_rotations":
        return local_block_rotations_seeded(tuple(full["block_ranks"]), **common)
    if family == "F03_unequal_split_merge":
        return unequal_split_merge_seeded(full["block_count"], **common)
    if family == "F04_partial_overlap":
        return partial_overlap_seeded(**common)
    if family == "F06_cooccurrence_confounding":
        return cooccurrence_confounding_seeded(**common)
    raise ValueError(f"unknown-support family not implemented: {family}")


def centered_kernels(pair):
    mean_left = np.mean(pair.z_left_mean, axis=0)
    mean_right = np.mean(pair.z_right_mean, axis=0)
    z_left = pair.z_left_eval - mean_left
    z_right = pair.z_right_eval - mean_right
    return (
        contribution_kernel(pair.d_left, z_left, pair.d_left, z_left),
        contribution_kernel(pair.d_left, z_left, pair.d_right, z_right),
        contribution_kernel(pair.d_right, z_right, pair.d_right, z_right),
        z_left,
        z_right,
    )


def all_candidates(left_count: int, right_count: int, max_size: int):
    left = [s for size in range(1, min(left_count, max_size) + 1) for s in combinations(range(left_count), size)]
    right = [s for size in range(1, min(right_count, max_size) + 1) for s in combinations(range(right_count), size)]
    return tuple((a, b) for a in left for b in right)


def frozen_to_json(family: str, lane: str, top_k: int | None, seeds: dict, frozen: FrozenDiscoveryPrediction, extra: dict) -> dict:
    return {
        "family_id": family,
        "lane": lane,
        "top_k": top_k,
        "seed_provenance": seeds,
        "frozen": asdict(frozen),
        **extra,
    }


def frozen_from_json(value: dict) -> FrozenDiscoveryPrediction:
    payload = value["frozen"]
    return FrozenDiscoveryPrediction(
        schema_version=payload["schema_version"],
        proposal_source=payload["proposal_source"],
        proposal_hash=payload["proposal_hash"],
        discovery_fingerprint=payload["discovery_fingerprint"],
        search_status=payload["search_status"],
        candidate_family=tuple((tuple(left), tuple(right)) for left, right in payload["candidate_family"]),
        predictions=tuple(BalancedCandidate(tuple(item["left_ids"]), tuple(item["right_ids"]), item["normalized_residual"]) for item in payload["predictions"]),
        prediction_hash=payload["prediction_hash"],
    )


def diagnostic_records(config: dict) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    summaries: list[dict] = []
    for family_index, family in enumerate(config["families"]):
        source_path = ROOT / config["diagnostic_config_sources"][family]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["families"] = [family]
        source["seed_pair_count"] = config["seed_pair_count"]
        source["base_seed"] = config["base_seed"] + 10000 * family_index
        source["emit_complete_metric_surface"] = True
        family_records, summary = evaluate_diagnostics(source)
        for record in family_records:
            provenance = record["seed_provenance"]
            # Some algebraic legacy paths never invoked a solver.  The smoke still
            # records the frozen six-stream contract without pretending that the
            # unused solver/discovery streams affected those diagnostics.
            highest = max(int(value) for value in provenance.values())
            if "solver_seed" not in provenance:
                provenance["solver_seed"] = highest + 1
            provenance["discovery_sample_seed"] = max(int(value) for value in provenance.values()) + 1
            validate_independent_split_seeds(provenance)
            record["metric_surface_errors"] = metric_surface_errors(record)
        records.extend(family_records)
        summaries.append({"family_id": family, **summary})
    return records, summaries


def discovery_phase(config: dict, full: dict) -> list[dict]:
    frozen_records: list[dict] = []
    for family_index, family in enumerate(config["unknown_support_families"]):
        for pair_index in range(config["seed_pair_count"]):
            root_seed = config["base_seed"] + 200000 + 1000 * family_index + 10 * pair_index
            seeds = {
                "structural_seed_a": root_seed + 1,
                "structural_seed_b": root_seed + 2,
                "mean_sample_seed": root_seed + 3,
                "discovery_sample_seed": root_seed + 4,
                "eval_sample_seed": root_seed + 5,
                "solver_seed": root_seed + 6,
            }
            validate_independent_split_seeds(seeds)
            pair = make_pair(family, full, seeds, "discovery_sample_seed", full["n_discovery"])
            k_ll, k_lr, k_rr, z_left, z_right = centered_kernels(pair)
            fingerprint = array_fingerprint(k_ll, k_lr, k_rr)
            score_lanes = {
                "CONTRIB-KNN": singleton_contribution_affinity(k_ll, k_lr, k_rr),
                "DECODER-KNN": decoder_cosine_affinity(pair.d_left, pair.d_right),
                "CODE-KNN": absolute_code_correlation_affinity(z_left, z_right),
            }
            for top_k in [full["primary_top_k"], *full["top_k_ablations"]]:
                proposals = {
                    lane: symmetric_topk_proposal(score, top_k=top_k, score_source=lane, max_neighborhood_atoms=full["max_neighborhood_atoms"])
                    for lane, score in score_lanes.items()
                }
                proposals["RANDOM-MATCHED"] = degree_matched_random_proposal(
                    proposals["CONTRIB-KNN"], seed=seeds["solver_seed"] + top_k,
                    max_neighborhood_atoms=full["max_neighborhood_atoms"],
                )
                for lane, proposal in proposals.items():
                    candidate_family = proposal_candidate_family(proposal, max_group_size=full["max_group_size"])
                    search = search_candidate_family(
                        k_ll, k_lr, k_rr, candidate_family,
                        residual_tolerance=full["residual_tolerance_exact"], tie_tolerance=full["tie_tolerance"],
                        candidate_budget=full["common_candidate_budget_per_lane_pair"],
                    )
                    frozen = freeze_discovery_prediction(
                        search, proposal_source=lane, proposal_hash=proposal_hash(proposal), discovery_fingerprint=fingerprint,
                    )
                    frozen_records.append(frozen_to_json(family, lane, top_k, seeds, frozen, {
                        "pair_index": pair_index, "edge_count": len(proposal.edges), "neighborhood_count": len(proposal.neighborhoods),
                        "proposal_budget_refusals": sum(x.status == "BUDGET_REFUSAL" for x in proposal.neighborhoods),
                        "evaluated_count": search.evaluated_count, "solver_gap": search.solver_gap,
                    }))
            spectral = li15_spectral_proposal(
                z_left, z_right,
                correlation_threshold=full["li15_spectral"]["correlation_threshold"],
                max_clusters=full["li15_spectral"]["max_clusters"],
                kmeans_seed=seeds["solver_seed"], max_neighborhood_atoms=full["max_neighborhood_atoms"],
            )
            spectral_family = proposal_candidate_family(spectral.proposal, max_group_size=full["max_group_size"])
            spectral_search = search_candidate_family(
                k_ll, k_lr, k_rr, spectral_family,
                residual_tolerance=full["residual_tolerance_exact"], tie_tolerance=full["tie_tolerance"],
                candidate_budget=full["common_candidate_budget_per_lane_pair"],
            )
            spectral_frozen = freeze_discovery_prediction(
                spectral_search, proposal_source="LI15-SPECTRAL", proposal_hash=proposal_hash(spectral.proposal), discovery_fingerprint=fingerprint,
            )
            frozen_records.append(frozen_to_json(family, "LI15-SPECTRAL", None, seeds, spectral_frozen, {
                "pair_index": pair_index, "cluster_count": spectral.cluster_count, "mixed_cluster_count": spectral.mixed_cluster_count,
                "evaluated_count": spectral_search.evaluated_count, "solver_gap": spectral_search.solver_gap,
            }))
            exhaustive_family = all_candidates(k_lr.shape[0], k_lr.shape[1], full["max_group_size"])
            exact = full_universe_balanced_search(
                k_ll, k_lr, k_rr, residual_tolerance=full["residual_tolerance_exact"], tie_tolerance=full["tie_tolerance"],
                max_group_size=full["max_group_size"], candidate_budget=full["full_universe_oracle_candidate_cap"],
            )
            if exact.status == "BUDGET_REFUSAL":
                exact_search = CandidateFamilySearchResult(
                    status=exact.status, candidate_family=exhaustive_family, passing_candidates=(), support_minimal_candidates=(),
                    best_residual=None, second_best_residual=None, solver_gap=None, tie_set=(), evaluated_count=0,
                    candidate_budget=full["full_universe_oracle_candidate_cap"], refusal_reason=exact.refusal_reason, elapsed_seconds=0.0,
                )
            else:
                exact_search = search_candidate_family(
                    k_ll, k_lr, k_rr, exhaustive_family,
                    residual_tolerance=full["residual_tolerance_exact"], tie_tolerance=full["tie_tolerance"],
                    candidate_budget=full["full_universe_oracle_candidate_cap"],
                )
            exact_frozen = freeze_discovery_prediction(
                exact_search, proposal_source="FULL-EXHAUSTIVE",
                proposal_hash=hashlib.sha256(json.dumps(exhaustive_family).encode()).hexdigest().upper(),
                discovery_fingerprint=fingerprint,
            )
            frozen_records.append(frozen_to_json(family, "FULL-EXHAUSTIVE", None, seeds, exact_frozen, {
                "pair_index": pair_index, "planned_candidate_count": exact.planned_candidate_count, "evaluated_count": exact_search.evaluated_count,
                "solver_gap": exact_search.solver_gap,
            }))
    return frozen_records


def held_out_phase(frozen_path: Path, full: dict) -> list[dict]:
    frozen_records = [json.loads(line) for line in frozen_path.read_text(encoding="utf-8").splitlines() if line]
    evaluations = []
    for record in frozen_records:
        family = record["family_id"]
        seeds = record["seed_provenance"]
        pair = make_pair(family, full, seeds, "eval_sample_seed", full["n_eval"])
        k_ll, k_lr, k_rr, _, _ = centered_kernels(pair)
        truth = () if family == "F06_cooccurrence_confounding" else pair.planted_hyperedges
        frozen = frozen_from_json(record)
        result = evaluate_frozen_hyperedges(frozen, k_ll, k_lr, k_rr, truth)
        evaluations.append({
            "family_id": family, "pair_index": record["pair_index"], "lane": record["lane"], "top_k": record["top_k"],
            "prediction_hash": frozen.prediction_hash, "truth_definition": "NO_ACCEPTABLE_MATCH" if not truth else "PLANTED_HYPEREDGES",
            **asdict(result),
        })
    return evaluations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    full_path = ROOT / config["parent_protocol"]
    full = json.loads(full_path.read_text(encoding="utf-8"))
    stable_json(run_dir / "resolved_config.json", config)
    stable_json(run_dir / "parent_protocol.resolved.json", full)
    sources = [config_path, full_path, Path(__file__).resolve(), ROOT / "scripts/validate_m1_corrective_smoke.py", ROOT / "scripts/run_r001_smoke.py", ROOT / "src/ccad/matching.py", ROOT / "src/ccad/proposal.py", ROOT / "src/ccad/metrics.py", ROOT / "src/ccad/synthetic.py"]
    sources.extend(ROOT / path for path in sorted(set(config["diagnostic_config_sources"].values())))
    snapshot = run_dir / "source_snapshot"
    ledger = []
    for source in sources:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        ledger.append({"path": str(relative).replace("\\", "/"), "snapshot": str(target.relative_to(run_dir)).replace("\\", "/"), "sha256": sha256(target), "bytes": target.stat().st_size})
    stable_json(run_dir / "code_hashes.json", {"workspace_is_git_repository": (ROOT / ".git").exists(), "files": ledger})
    stable_json(run_dir / "environment.json", {"python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "platform": platform.platform(), "device": "cpu", "started_utc": started.isoformat()})
    stable_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    try:
        diagnostics, diagnostic_summaries = diagnostic_records(config)
        stable_jsonl(run_dir / "diagnostic_records.jsonl", diagnostics)
        frozen_records = discovery_phase(config, full)
        stable_jsonl(run_dir / "discovery_predictions.jsonl", frozen_records)
        frozen_written = datetime.now(timezone.utc)
        stable_json(run_dir / "phase_ledger.json", {"discovery_predictions_written_utc": frozen_written.isoformat(), "held_out_eval_opened_utc": None, "ordering": "DISCOVERY_FILE_CLOSED_AND_REREAD_BEFORE_EVAL_CONSTRUCTION"})
        held_out_opened = datetime.now(timezone.utc)
        stable_json(run_dir / "phase_ledger.json", {"discovery_predictions_written_utc": frozen_written.isoformat(), "held_out_eval_opened_utc": held_out_opened.isoformat(), "ordering": "DISCOVERY_FILE_CLOSED_AND_REREAD_BEFORE_EVAL_CONSTRUCTION"})
        held_out = held_out_phase(run_dir / "discovery_predictions.jsonl", full)
        stable_jsonl(run_dir / "held_out_evaluations.jsonl", held_out)
        primary = [x for x in held_out if x["lane"] == full["primary_proposal_lane"] and x["top_k"] == full["primary_top_k"]]
        primary_pass = all(
            (x["precision"] == x["recall"] == x["f1"] == 1.0 and x["failure_attribution"] is None)
            for x in primary
        )
        summary = {
            "status": "PASS" if all(x["status"] == "PASS" for x in diagnostic_summaries) and primary_pass else "FAIL",
            "semantic_outcome": "CORRECTIVE_FULL_PRIMARY_PASS" if config["seed_pair_count"] == full["seed_pair_count"] and primary_pass else ("SMOKE_IMPLEMENTATION_PASS" if primary_pass else "PRIMARY_UNKNOWN_SUPPORT_FAIL"),
            "evidence_level": config["evidence_level"], "real_sae_audit_opened": False,
            "diagnostic_record_count": len(diagnostics), "families_covered": sorted({x["family_id"] for x in diagnostics}),
            "metric_surface_error_count": sum(len(x["metric_surface_errors"]) for x in diagnostics),
            "discovery_prediction_count": len(frozen_records), "held_out_evaluation_count": len(held_out),
            "primary_unknown_support_count": len(primary), "primary_unknown_support_pass": primary_pass,
            "lane_set": sorted({x["lane"] for x in frozen_records}),
            "diagnostic_sha256": sha256(run_dir / "diagnostic_records.jsonl"),
            "discovery_predictions_sha256": sha256(run_dir / "discovery_predictions.jsonl"),
            "held_out_evaluations_sha256": sha256(run_dir / "held_out_evaluations.jsonl"),
            "scope_warning": config["scope_warning"],
        }
        stable_json(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        final = summary["status"]
    except Exception:
        (run_dir / "stdout.log").write_text("", encoding="utf-8")
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        final = "FAIL"
    ended = datetime.now(timezone.utc)
    stable_json(run_dir / "status.json", {"status": final, "updated_utc": ended.isoformat(), "ended_utc": ended.isoformat()})
    print(json.dumps({"run_dir": str(run_dir), "status": final}, sort_keys=True))
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
