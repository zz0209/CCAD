"""PC2 P1 truth-closed prediction runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import tracemalloc
import traceback

import numpy as np

from ccad.mscc import minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_baselines import IMPLEMENTED_CONTINUOUS_REFERENCES, IMPLEMENTED_NATIVE_LANES, run_continuous_reference, run_native_baseline
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v2 import generate_cap_identifiable_observed
from ccad.nip_synthetic_v3 import generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/ccad/mscc.py", "src/ccad/nip_baselines.py", "src/ccad/nip_synthetic.py",
    "src/ccad/nip_synthetic_v2.py", "src/ccad/nip_synthetic_v3.py",
    "scripts/run_m1_nip_parent_completion_p1.py",
    "scripts/validate_m1_nip_parent_completion_p1.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_for(protocol_hash: str, code_hash: str, namespace: str, phase: str, family: str, pair: int, stream: str) -> int:
    value = "||".join((protocol_hash, code_hash, namespace, phase, family, str(pair), stream)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def snapshot(source: Path, run_dir: Path, prefix: str, relative: str) -> dict[str, object]:
    target = run_dir / prefix / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": relative, "snapshot": target.relative_to(run_dir).as_posix(), "sha256": sha(target), "bytes": target.stat().st_size}


def _semantic_mscc(result) -> dict:
    payload = asdict(result)
    payload.pop("elapsed_seconds")
    return payload


def _timed(call, semantic=lambda value: value) -> tuple[object, list[float], int]:
    warm = call()
    reference = semantic(warm)
    times = []
    chosen = None
    tracemalloc.start()
    for _ in range(5):
        started = perf_counter()
        result = call()
        times.append(perf_counter() - started)
        if semantic(result) != reference:
            tracemalloc.stop()
            raise RuntimeError("measured repeat changed the scientific output")
        chosen = result
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return chosen, times, int(peak)


def _fairness(config: dict, source_hash: str, target_hash: str, *, full_comparisons: int,
              proposed_count: int, supports: list[tuple[int, ...]], evaluated: int,
              runtime: list[float], peak: int, terminal_reason: str | None) -> dict:
    raw_atoms = [atom for support in supports for atom in support]
    return {
        "source_query_manifest_hash": source_hash,
        "target_universe_hash": target_hash,
        "g_max": config["g_max"],
        "candidate_budget": config["candidate_budget"],
        "full_dictionary_comparisons": full_comparisons,
        "proposed_atom_count": proposed_count,
        "raw_support_count": len(raw_atoms),
        "deduplicated_support_count": len(set(raw_atoms)),
        "evaluated_candidate_count": evaluated,
        "runtime_seconds_descriptive_only": runtime,
        "peak_memory_bytes": peak,
        "terminal_reason": terminal_reason,
    }


def build_records(config: dict, code_hash: str) -> tuple[list[dict], list[dict], list[dict]]:
    proposals: list[dict] = []
    predictions: list[dict] = []
    seed_rows: list[dict] = []
    for family in FAMILIES:
        for pair in range(config["pairs_per_family"]):
            seeds = {stream: seed_for(config["protocol_sha256"], code_hash, config["fresh_namespace"], config["phase"], family, pair, stream) for stream in config["required_seed_streams"]}
            if len(set(seeds.values())) != len(seeds):
                raise RuntimeError("derived seed streams collided")
            seed_rows.append({"family_id": family, "pair_index": pair, "seeds": seeds})
            # The mean stream never constructs or reads an intervention endpoint.
            # Its frozen odd sample size is therefore valid for every family.
            mean = generate_cap_identifiable_observed(family, structural_seed=seeds["structural"], sample_seed=seeds["mean"], n=config["sample_sizes"]["mean"])
            discovery = generate_endpoint_observed(family, structural_seed=seeds["structural"], sample_seed=seeds["discovery"], n=config["sample_sizes"]["discovery"])
            fingerprint = hashlib.sha256(discovery.source_contributions.tobytes() + discovery.target_contributions.tobytes()).hexdigest().upper()
            approximate = family in config["approximate_families"]
            tau_ctr = config["approximate_tau_ctr"] if approximate else config["exact_tau_ctr"]
            tau_mu = config["approximate_tau_mu"] if approximate else config["exact_tau_mu"]
            y = discovery.source_contributions[:, 0, :]
            targets = discovery.target_contributions
            source_hash = hashlib.sha256(y.tobytes()).hexdigest().upper()
            target_hash = hashlib.sha256(targets.tobytes()).hexdigest().upper()
            source_mean = mean.source_mean_contributions[:, 0]
            target_means = mean.target_mean_contributions
            k_ss, k_st, k_tt = observed_kernels(discovery)

            proposal = source_conditioned_topk_proposal(
                k_ss, k_st, k_tt, source_atom_id=0, atom_cap=20, g_max=config["g_max"],
                epsilon=config["epsilon"], candidate_budget=config["candidate_budget"],
                boundary_tie_tolerance=config["tie_tolerance"],
            )
            mscc_call = lambda: minimum_support_contribution_correspondence(
                k_ss, k_st, k_tt, mean.source_mean_contributions, mean.target_mean_contributions,
                source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids,
                g_max=config["g_max"], tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=config["epsilon"],
                candidate_budget=config["candidate_budget"], complete_universe=False,
            )
            mscc, runtime, peak = _timed(mscc_call, _semantic_mscc)
            proposal_row = {
                "family_id": family, "pair_index": pair, "lane": "MSCC",
                "discovery_fingerprint": fingerprint, "status": proposal.status,
                "ranking": sorted(range(20), key=lambda atom: (proposal.singleton_d_ctr[atom], atom)),
                "scores": list(proposal.singleton_d_ctr), "proposed_target_ids": list(proposal.proposed_target_ids),
                "full_dictionary_comparisons": proposal.full_dictionary_comparisons,
                "planned_support_count": proposal.planned_support_count, "proposal_hash": proposal.proposal_hash,
                "terminal_reason": proposal.refusal_reason,
            }
            proposals.append(proposal_row)
            prediction = _semantic_mscc(mscc)
            mscc_supports = [tuple(item["target_ids"]) for item in prediction["supports"]]
            predictions.append({
                "family_id": family, "pair_index": pair, "lane": "MSCC", "kind": "NATIVE",
                "discovery_fingerprint": fingerprint, "prediction": prediction,
                "cost": {"evaluated_candidate_count": mscc.evaluated_count, "runtime_seconds": runtime,
                         "median_runtime_seconds": float(np.median(runtime)), "peak_memory_bytes": peak},
                "fairness": _fairness(config, source_hash, target_hash,
                    full_comparisons=proposal.full_dictionary_comparisons,
                    proposed_count=len(proposal.proposed_target_ids), supports=mscc_supports,
                    evaluated=mscc.evaluated_count, runtime=runtime, peak=peak,
                    terminal_reason=prediction.get("terminal_reason") or prediction.get("unresolved_reason")),
                "truth_opened": False,
            })

            for lane in config["native_lanes"]:
                if lane == "MSCC":
                    continue
                lane_call = lambda lane=lane: run_native_baseline(
                    lane, y, targets, source_mean, target_means, g_max=config["g_max"],
                    tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=config["epsilon"],
                    tie_tolerance=config["tie_tolerance"], solver_seed=seeds["solver"],
                )
                result, runtime, peak = _timed(lane_call)
                diagnostics = []
                diagnostic_cost = 0
                if lane == "RANDOM_MATCHED_GROUP":
                    for replicate in range(1, config["random_diagnostic_replicates"] + 1):
                        other = run_native_baseline(
                            lane, y, targets, source_mean, target_means, g_max=config["g_max"],
                            tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=config["epsilon"],
                            tie_tolerance=config["tie_tolerance"], solver_seed=seeds["solver"] + replicate,
                        )
                        diagnostic_cost += other.evaluated_support_count
                        diagnostics.append({"replicate": replicate, "identification": other.identification, "supports": other.supports, "evaluated": other.evaluated_support_count})
                proposal_row = {
                    "family_id": family, "pair_index": pair, "lane": lane,
                    "discovery_fingerprint": fingerprint, "status": result.status,
                    "ranking": result.ranking, "scores": result.ranking_scores,
                    "proposed_target_ids": sorted(set(result.ranking[:config["g_max"]])),
                    "full_dictionary_comparisons": 20, "planned_support_count": min(config["g_max"], len(result.ranking)),
                    "proposal_hash": digest({"lane": lane, "ranking": result.ranking, "scores": result.ranking_scores, "fingerprint": fingerprint}),
                    "terminal_reason": result.terminal_reason,
                }
                proposals.append(proposal_row)
                predictions.append({
                    "family_id": family, "pair_index": pair, "lane": lane, "kind": "NATIVE",
                    "discovery_fingerprint": fingerprint, "prediction": asdict(result),
                    "random_diagnostics": diagnostics,
                    "cost": {"evaluated_candidate_count": result.evaluated_support_count + diagnostic_cost,
                             "primary_evaluated_candidate_count": result.evaluated_support_count,
                             "diagnostic_evaluated_candidate_count": diagnostic_cost,
                             "runtime_seconds": runtime, "median_runtime_seconds": float(np.median(runtime)),
                             "peak_memory_bytes": peak},
                    "fairness": _fairness(config, source_hash, target_hash,
                        full_comparisons=20, proposed_count=len(set(result.ranking[:config["g_max"]])),
                        supports=[tuple(value) for value in result.supports],
                        evaluated=result.evaluated_support_count + diagnostic_cost,
                        runtime=runtime, peak=peak, terminal_reason=result.terminal_reason),
                    "truth_opened": False,
                })

            for lane in config["continuous_references"]:
                call = lambda lane=lane: run_continuous_reference(lane, y, targets)
                result, runtime, peak = _timed(call)
                proposals.append({
                    "family_id": family, "pair_index": pair, "lane": lane,
                    "discovery_fingerprint": fingerprint, "status": "CONTINUOUS_REFERENCE",
                    "ranking": sorted(range(20), key=lambda atom: (-abs(result.coefficients[atom]), atom)),
                    "scores": list(result.coefficients), "proposed_target_ids": [],
                    "full_dictionary_comparisons": 20, "planned_support_count": 0,
                    "proposal_hash": digest({"lane": lane, "coefficients": result.coefficients, "fingerprint": fingerprint}),
                    "terminal_reason": None,
                })
                predictions.append({
                    "family_id": family, "pair_index": pair, "lane": lane, "kind": "CONTINUOUS_REFERENCE",
                    "discovery_fingerprint": fingerprint, "prediction": asdict(result),
                    "cost": {"evaluated_candidate_count": 0, "runtime_seconds": runtime,
                             "median_runtime_seconds": float(np.median(runtime)), "peak_memory_bytes": peak},
                    "fairness": _fairness(config, source_hash, target_hash,
                        full_comparisons=20, proposed_count=0, supports=[], evaluated=0,
                        runtime=runtime, peak=peak, terminal_reason="CONTINUOUS_REFERENCE"),
                    "truth_opened": False,
                })
    return proposals, predictions, seed_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_config_path"]
    protocol_path = ROOT / config["protocol_path"]
    phase = config["phase"]
    if not config["execution_enabled"] or phase not in {"P1", "P2"} or config["formal_seed_consumed"]:
        raise ValueError("prediction execution contract is closed")
    if any(config[key] for key in ("truth_opened_in_prediction", "evaluation_opened_in_prediction", "intervention_opened_in_prediction", "real_sae_audit_opened")):
        raise ValueError("prediction information boundary is open")
    expected_pairs = 1 if phase == "P1" else 20
    if tuple(config["families"]) != FAMILIES or config["pairs_per_family"] != expected_pairs:
        raise ValueError(f"{phase} grid drift")
    if sha(parent_path) != config["parent_config_sha256"] or sha(protocol_path) != config["protocol_sha256"]:
        raise ValueError("parent config or protocol hash drift")
    if set(config["native_lanes"]) != IMPLEMENTED_NATIVE_LANES | {"MSCC"} or set(config["continuous_references"]) != IMPLEMENTED_CONTINUOUS_REFERENCES:
        raise ValueError("implemented lane registry drift")
    if phase == "P2":
        if not config.get("consume_formal_seeds_on_execution"):
            raise ValueError("P2 must consume a fresh formal seed namespace")
        for binding in config["p1_gate_bindings"]:
            artifact = ROOT / binding["path"]
            if not artifact.is_file() or sha(artifact) != binding["sha256"]:
                raise ValueError(f"P1 gate binding failed: {binding['path']}")
        gate = json.loads((ROOT / config["p1_gate_bindings"][-1]["path"]).read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or gate.get("passed_count") != gate.get("check_count"):
            raise ValueError("P1 score gate is not a complete PASS")

    run_dir.mkdir(parents=True)
    try:
        sources = [snapshot(ROOT / relative, run_dir, "source_snapshot", relative) for relative in SOURCES]
        code_hash = digest(sources)
        inputs = [snapshot(config_path, run_dir, "input_snapshot", "execution_config.json"), snapshot(parent_path, run_dir, "input_snapshot", config["parent_config_path"]), snapshot(protocol_path, run_dir, "input_snapshot", config["protocol_path"])]
        try:
            git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            git_head = None
        write_json(run_dir / "resolved_config.json", config)
        write_json(run_dir / "environment.json", {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "platform": platform.platform(), "device": "cpu", "git_head_at_run": git_head})
        write_json(run_dir / "code_hashes.json", {"aggregate_sha256": code_hash, "files": sources})
        write_json(run_dir / "input_hashes.json", {"files": inputs})
        started = now()
        write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started, "truth_opened": False})
        proposals, predictions, seeds = build_records(config, code_hash)
        if len(predictions) != config["expected_prediction_rows"] or len(proposals) != len(predictions):
            raise RuntimeError("prediction grid is incomplete")
        formal_seed_consumed = phase == "P2"
        write_json(run_dir / "seed_ledger.json", {"schema_version": f"pc2.{phase.lower()}.seed_ledger.v1", "phase": phase, "formal_seed_consumed": formal_seed_consumed, "rows": seeds})
        (run_dir / "proposals.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in proposals), encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
        (run_dir / "stdout.log").write_text(f"sealed {len(predictions)} truth-closed {phase} prediction rows\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        write_json(run_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": now(), "truth_opened": False})
        bound = [name for name in config["required_artifacts"] if name not in {"prediction_closure.json", "prelabel_validation.json"}]
        closure = {"schema_version": f"pc2.{phase.lower()}.prediction_closure.v1", "state": "SEALED", "run_id": run_dir.name, "row_count": len(predictions), "code_snapshot_hash": code_hash, "protocol_sha256": config["protocol_sha256"], "truth_opened": False, "formal_seed_consumed": formal_seed_consumed, "files": {name: sha(run_dir / name) for name in bound}}
        temporary = run_dir / "prediction_closure.json.tmp"
        write_json(temporary, closure)
        os.replace(temporary, run_dir / "prediction_closure.json")
        print(json.dumps({"status": "PASS", "rows": len(predictions), "run_dir": str(run_dir)}))
        return 0
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "FAIL", "ended_utc": now(), "truth_opened": False, "failure_type": type(error).__name__, "failure_message": str(error)})
        (run_dir / "stderr.log").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
