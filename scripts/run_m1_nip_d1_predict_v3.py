"""Truth-blind M1-NIP v3 prediction and atomic closure writer."""
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
from pathlib import Path

import numpy as np

from ccad.mscc import freeze_mscc_prediction, minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v3 import generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/ccad/mscc.py",
    "src/ccad/nip_synthetic.py",
    "src/ccad/nip_synthetic_v2.py",
    "src/ccad/nip_synthetic_v3.py",
    "scripts/run_m1_nip_d1_predict_v3.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_for(protocol: str, code: str, phase: str, family: str, pair: int, stream: str) -> int:
    value = "||".join((protocol, code, phase, family, str(pair), stream)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def snapshot(source: Path, run_dir: Path, prefix: str, relative: str) -> dict[str, object]:
    target = run_dir / prefix / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"snapshot": target.relative_to(run_dir).as_posix(), "sha256": sha(target), "bytes": target.stat().st_size}


def fail(run_dir: Path, error: BaseException) -> None:
    if run_dir.is_dir() and not (run_dir / "prediction_closure.json").exists():
        write_json(run_dir / "status.json", {
            "status": "FAIL", "ended_utc": now(), "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        (run_dir / "stderr.log").write_text("".join(traceback.format_exception(error)), encoding="utf-8")


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
    if not config["execution_enabled"] or config["phase"] not in {"I1", "D1"}:
        raise ValueError("v3 prediction phase is not enabled")
    if config["truth_opened_in_prediction"] or config["held_out_eval_opened"] or config["real_sae_audit_opened"]:
        raise ValueError("prediction information boundary is open")
    if tuple(config["families"]) != FAMILIES or config["atom_caps"] != [4, 8, 12, 16, 20]:
        raise ValueError("grid differs from v3 protocol")
    if config["target_atom_count"] != 20 or config["expected_prediction_rows"] != 12 * config["pairs_per_family"] * 5:
        raise ValueError("v3 prediction dimensions are inconsistent")
    protocol = ROOT / config["protocol_path"]
    diagnostics = ROOT / config["diagnostic_config_path"]
    if sha(protocol) != config["protocol_sha256"] or sha(diagnostics) != config["diagnostic_config_sha256"]:
        raise ValueError("protocol or diagnostic contract hash drift")

    run_dir.mkdir(parents=True)
    source_rows = [snapshot(ROOT / item, run_dir, "source_snapshot", item) for item in SOURCES]
    code_hash = digest(source_rows)
    input_rows = {
        "protocol": snapshot(protocol, run_dir, "input_snapshot", config["protocol_path"]),
        "diagnostic_config": snapshot(diagnostics, run_dir, "input_snapshot", config["diagnostic_config_path"]),
        "execution_config": snapshot(config_path, run_dir, "input_snapshot", "execution_config.json"),
    }
    write_json(run_dir / "config.resolved.json", config)
    write_json(run_dir / "code_hashes.json", {"git_available": False, "aggregate_sha256": code_hash, "files": source_rows})
    write_json(run_dir / "inputs.json", input_rows)
    write_json(run_dir / "environment.json", {"os": platform.platform(), "python": sys.version, "numpy": np.__version__, "device": "cpu"})
    started = now()
    write_json(run_dir / "manifest.json", {
        "artifact_schema_version": "ccad.prediction_run.v1", "run_id": run_dir.name,
        "phase": config["phase"], "evidence_level": config["evidence_level"], "started_utc": started,
        "protocol_sha256": config["protocol_sha256"], "diagnostic_config_sha256": config["diagnostic_config_sha256"],
        "code_snapshot_hash": code_hash, "truth_opened": False,
        "formal_d1_seed_consumed": config["formal_d1_seed_consumed"],
        "statistical_unit": "seed_pair", "caps_paired_within_seed_pair": True,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started})

    records: list[dict[str, object]] = []
    for family in FAMILIES:
        for pair in range(config["pairs_per_family"]):
            seeds = {
                stream: seed_for(config["protocol_sha256"], code_hash, config["phase"], family, pair, stream)
                for stream in ("structural", "sample", "proposal", "solver")
            }
            observed = generate_endpoint_observed(
                family, structural_seed=seeds["structural"], sample_seed=seeds["sample"],
                n=config["observations_per_pair"],
            )
            k_ss, k_st, k_tt = observed_kernels(observed)
            fingerprint = hashlib.sha256(
                observed.source_contributions.tobytes() + observed.target_contributions.tobytes()
            ).hexdigest().upper()
            for cap in config["atom_caps"]:
                proposal = source_conditioned_topk_proposal(
                    k_ss, k_st, k_tt, source_atom_id=0, atom_cap=cap, g_max=config["g_max"],
                    epsilon=config["epsilon"], candidate_budget=config["candidate_budget"],
                    boundary_tie_tolerance=config["boundary_tie_tolerance"],
                )
                approximate = family in config["approximate_families"]
                result = minimum_support_contribution_correspondence(
                    k_ss, k_st, k_tt, observed.source_mean_contributions, observed.target_mean_contributions,
                    source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids, g_max=config["g_max"],
                    tau_ctr=config["approximate_tau_ctr"] if approximate else config["exact_tau_ctr"],
                    tau_mu=config["approximate_tau_mu"] if approximate else config["exact_tau_mu"],
                    epsilon=config["epsilon"], candidate_budget=config["candidate_budget"],
                    complete_universe=cap == 20,
                )
                frozen = freeze_mscc_prediction(
                    result, protocol_hash=config["protocol_sha256"], proposal_hash=proposal.proposal_hash,
                    discovery_fingerprint=fingerprint, source_atom_id=0,
                )
                records.append({
                    "schema_version": "m1_nip_prediction.v3", "run_id": run_dir.name,
                    "family_id": family, "pair_index": pair, "atom_cap": cap, "seeds": seeds,
                    "protocol_hash": frozen.protocol_hash, "diagnostic_config_hash": config["diagnostic_config_sha256"],
                    "discovery_fingerprint": fingerprint, "source_atom_id": 0,
                    "proposal": {
                        "status": proposal.status, "source_atom_id": 0, "atom_cap": cap,
                        "g_max": config["g_max"], "candidate_budget": config["candidate_budget"],
                        "boundary_tie_tolerance": config["boundary_tie_tolerance"],
                        "ranking": sorted(range(20), key=lambda index: (proposal.singleton_d_ctr[index], index)),
                        "singleton_d_ctr": proposal.singleton_d_ctr,
                        "proposed_target_ids": proposal.proposed_target_ids,
                        "boundary_margin": proposal.boundary_margin,
                        "planned_support_count": proposal.planned_support_count,
                        "full_dictionary_comparisons": proposal.full_dictionary_comparisons,
                        "refusal_reason": proposal.refusal_reason,
                        "proposal_hash": proposal.proposal_hash,
                    },
                    "prediction": {
                        "search_status": result.status, "identification": result.identification,
                        "multiplicity": result.multiplicity, "minimum_support_size": result.minimum_support_size,
                        "supports": [asdict(item) for item in result.supports],
                        "best_candidate": None if result.best_candidate is None else asdict(result.best_candidate),
                        "nearest_competitor": None if result.nearest_competitor is None else asdict(result.nearest_competitor),
                        "planned_candidate_count": result.planned_candidate_count,
                        "evaluated_count": result.evaluated_count, "complete_universe": result.complete_universe,
                        "unresolved_reason": result.unresolved_reason,
                        "elapsed_seconds_descriptive_only": result.elapsed_seconds,
                        "prediction_hash": frozen.prediction_hash,
                    },
                    "truth_opened": False,
                })
    raw = run_dir / "predictions.raw.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    write_json(run_dir / "predictions.summary.json", {
        "status": "PASS", "record_count": len(records), "truth_opened": False,
        "raw_sha256": sha(raw), "runtime_used_for_selection": False,
    })
    (run_dir / "stdout.log").write_text("sealed v3 truth-blind predictions\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(run_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": now()})
    bound = [
        "predictions.raw.jsonl", "predictions.summary.json", "config.resolved.json", "code_hashes.json",
        "inputs.json", "environment.json", "manifest.json", "status.json",
    ]
    closure = {
        "schema_version": "ccad.prediction_closure.v1", "state": "SEALED", "run_id": run_dir.name,
        "row_count": len(records), "protocol_sha256": config["protocol_sha256"],
        "diagnostic_config_sha256": config["diagnostic_config_sha256"], "code_snapshot_hash": code_hash,
        "files": {name: sha(run_dir / name) for name in bound},
    }
    temporary = run_dir / "prediction_closure.json.tmp"
    write_json(temporary, closure)
    os.replace(temporary, run_dir / "prediction_closure.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--run-dir" in sys.argv:
            fail(Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve(), error)
        raise
