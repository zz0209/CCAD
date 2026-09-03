from __future__ import annotations

import argparse, hashlib, json, platform, shutil, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ccad.mscc import freeze_mscc_prediction, minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v2 import CAP_PRESSURE, construction_certificate, generate_cap_identifiable_observed


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/ccad/mscc.py", "src/ccad/nip_synthetic.py", "src/ccad/nip_synthetic_v2.py",
    "scripts/run_m1_nip_d0_v2.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now() -> str:
    return datetime.now().astimezone().isoformat()


def snapshot_file(source: Path, run_dir: Path, prefix: str, relative: str) -> dict[str, object]:
    destination = run_dir / prefix / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"path": str(source), "snapshot": str(destination.relative_to(run_dir)), "sha256": sha(destination), "bytes": destination.stat().st_size}


def finalize_failure(run_dir: Path, exc: BaseException) -> None:
    if not run_dir.is_dir():
        return
    status_path = run_dir / "status.json"
    prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    write_json(status_path, {"status": "FAIL", "started_utc": prior.get("started_utc"), "ended_utc": utc_now(), "failure_type": type(exc).__name__, "failure_message": str(exc)})
    with (run_dir / "stderr.log").open("a", encoding="utf-8") as handle:
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def derived_seed(protocol_hash: str, code_hash: str, family: str, pair_index: int, stream: str) -> int:
    payload = "||".join((protocol_hash, code_hash, "D0", family, str(pair_index), stream)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path, run_dir = Path(args.config).resolve(), Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["execution_enabled"] or config["phase"] != "D0" or config["truth_opened"] or config["held_out_eval_opened"] or config["real_sae_audit_opened"]:
        raise ValueError("D0 v2 config violates the closed-information contract")
    if tuple(config["families"]) != FAMILIES or config["pairs_per_family"] != 5 or config["target_atom_count"] != 20:
        raise ValueError("D0 v2 family/target grid must match the frozen protocol")
    protocol_path = ROOT / config["protocol_path"]
    if sha(protocol_path) != config["protocol_sha256"]:
        raise ValueError("protocol hash drift")

    run_dir.mkdir(parents=True)
    source_rows = [snapshot_file(ROOT / name, run_dir, "source_snapshot", name) for name in SOURCES]
    code_hash = hashlib.sha256(json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    protocol_row = snapshot_file(protocol_path, run_dir, "input_snapshot", config["protocol_path"])
    config_row = snapshot_file(config_path, run_dir, "input_snapshot", "configs/m1_nip_d0_v2.json")
    write_json(run_dir / "config.resolved.json", config)
    write_json(run_dir / "code_hashes.json", {"git_available": False, "aggregate_sha256": code_hash, "files": source_rows})
    write_json(run_dir / "inputs.json", {
        "protocol": {**protocol_row, "source": "CCAD internal locked protocol", "access": "local", "license": "internal"},
        "execution_config": {**config_row, "source": "CCAD internal execution adapter", "access": "local", "license": "internal"},
    })
    write_json(run_dir / "environment.json", {"os": platform.platform(), "python": sys.version, "numpy": np.__version__, "cuda": "not_applicable", "pytorch": "not_applicable", "transformers": "not_applicable", "sae_framework": "not_applicable", "solver": "deterministic_enumeration"})
    started, started_local = utc_now(), local_now()
    write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    write_json(run_dir / "manifest.json", {
        "artifact_schema_version": "ccad.run.v1", "run_id": run_dir.name, "tracker_parent": "M1_NIP_protocol_v2",
        "purpose": "D0 v2 cap-identifiable truth-closed engineering", "milestone": "M1", "evidence_level": config["evidence_level"],
        "started_utc": started, "started_local": started_local, "trigger": "automation heartbeat ccad", "operator": "Codex", "project_root": str(ROOT),
        "config_hash": sha(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "protocol_sha256": protocol_row["sha256"], "execution_config_sha256": config_row["sha256"],
        "git_available": False, "model_id": "not_applicable", "model_revision": "not_applicable", "dataset_id": "synthetic_m1_nip_d0_v2", "dataset_revision": "generated_from_snapshotted_code",
        "tokenizer_id": "not_applicable", "sae_framework": "not_applicable", "device": "cpu", "resource_lease": "not_required_lightweight_cpu",
        "seed_fields": ["structural", "sample", "proposal", "solver"], "seed_derivation": "sha256(protocol_hash||code_hash||D0||family||pair_index||stream)[0:8]",
        "phase_seed_manifest_status": "GENERATED_AT_EXECUTION_IN_RAW_LEDGER", "split": "synthetic_discovery_D0", "statistical_unit": "seed_pair",
        "audit_opened": False, "candidate_family_frozen": True, "mean_constants_source_split": "synthetic_declared_mean", "threshold_source_split": "frozen_protocol",
        "artifact_schema": ["manifest", "config", "environment", "inputs", "code_hashes", "status", "logs", "raw_metrics", "summary", "source_snapshot", "input_snapshot"], "protocol_deviations": [],
    })

    records = []
    for family in FAMILIES:
        for pair_index in range(5):
            seeds = {stream: derived_seed(config["protocol_sha256"], code_hash, family, pair_index, stream) for stream in ("structural", "sample", "proposal", "solver")}
            if len(set(seeds.values())) != 4:
                raise RuntimeError("derived seed collision")
            observed = generate_cap_identifiable_observed(family, structural_seed=seeds["structural"], sample_seed=seeds["sample"], n=config["observations_per_pair"])
            certificate = construction_certificate(observed)
            cap_pass = certificate["cap_contract_pass"] in (True, None)
            if certificate["target_atom_count"] != 20 or certificate["minimum_decoy_orthogonal_residual"] < config["minimum_decoy_orthogonal_residual"] or certificate["maximum_decoy_orthogonality_error"] > config["maximum_decoy_orthogonality_error"] or not cap_pass:
                raise RuntimeError(f"construction certificate failed for {family} pair {pair_index}")
            k_ss, k_st, k_tt = observed_kernels(observed)
            proposal = source_conditioned_topk_proposal(k_ss, k_st, k_tt, source_atom_id=0, atom_cap=config["atom_cap"], g_max=config["g_max"], epsilon=config["epsilon"], candidate_budget=config["candidate_budget"], boundary_tie_tolerance=config["boundary_tie_tolerance"])
            if proposal.status != "OK" or proposal.planned_support_count != config["expected_planned_support_count"]:
                raise RuntimeError(f"proposal contract failed for {family} pair {pair_index}")
            tau_ctr = config["approximate_tau_ctr"] if family.startswith("N07") else config["exact_tau_ctr"]
            tau_mu = config["approximate_tau_mu"] if family.startswith("N07") else config["exact_tau_mu"]
            result = minimum_support_contribution_correspondence(k_ss, k_st, k_tt, observed.source_mean_contributions, observed.target_mean_contributions, source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids, g_max=config["g_max"], tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=config["epsilon"], candidate_budget=config["candidate_budget"], complete_universe=True)
            fingerprint = hashlib.sha256(observed.source_contributions.tobytes() + observed.target_contributions.tobytes()).hexdigest().upper()
            frozen = freeze_mscc_prediction(result, protocol_hash=config["protocol_sha256"], proposal_hash=proposal.proposal_hash, discovery_fingerprint=fingerprint, source_atom_id=0)
            certificate_hash = hashlib.sha256(json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
            records.append({
                "run_id": run_dir.name, "family_id": family, "pair_index": pair_index, "statistical_unit": "seed_pair", "metric_version": "m1_nip_d0.v2",
                "seeds": seeds, "source_shape": observed.source_contributions.shape, "target_shape": observed.target_contributions.shape,
                "proposal_status": proposal.status, "proposal_hash": proposal.proposal_hash, "prediction_hash": frozen.prediction_hash,
                "full_dictionary_comparisons": proposal.full_dictionary_comparisons, "planned_support_count": proposal.planned_support_count, "evaluated_count": result.evaluated_count,
                "construction_certificate_hash": certificate_hash, "minimum_decoy_orthogonal_residual": certificate["minimum_decoy_orthogonal_residual"],
                "maximum_decoy_orthogonality_error": certificate["maximum_decoy_orthogonality_error"], "declared_first_cap": certificate["declared_first_cap"],
                "observed_first_cap": certificate["observed_first_cap"], "cap_contract_pass": certificate["cap_contract_pass"],
                "truth_opened": False, "held_out_eval_opened": False,
            })
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    runner_hash = next(row["sha256"] for row in source_rows if Path(str(row["snapshot"])).as_posix().endswith("scripts/run_m1_nip_d0_v2.py"))
    summary = {
        "status": "PASS", "semantic_outcome": "D0_V2_ENGINEERING_ONLY_NO_LABEL_SCORING", "record_count": len(records), "family_count": len(FAMILIES), "pairs_per_family": 5,
        "raw_sha256": sha(raw), "generation_script_sha256": runner_hash, "target_atom_count": 20, "planned_support_count": 6195,
        "minimum_decoy_orthogonal_residual": min(row["minimum_decoy_orthogonal_residual"] for row in records),
        "maximum_decoy_orthogonality_error": max(row["maximum_decoy_orthogonality_error"] for row in records),
        "cap_contract_failure_count": sum(row["cap_contract_pass"] is False for row in records), "proposal_refusal_count": sum(row["proposal_status"] != "OK" for row in records),
        "truth_opened": False, "held_out_eval_opened": False, "real_sae_audit_opened": False,
    }
    write_json(run_dir / "metrics.summary.json", summary)
    (run_dir / "stdout.log").write_text("D0 v2 cap-identifiable truth-closed engineering completed\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(run_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": utc_now()})
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        if "--run-dir" in sys.argv:
            value_index = sys.argv.index("--run-dir") + 1
            if value_index < len(sys.argv):
                finalize_failure(Path(sys.argv[value_index]).resolve(), error)
        raise
    raise SystemExit(exit_code)
