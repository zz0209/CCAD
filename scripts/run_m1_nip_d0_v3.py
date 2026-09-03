from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ccad.mscc import freeze_mscc_prediction, minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v3 import construction_certificate, generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/ccad/mscc.py",
    "src/ccad/nip_synthetic.py",
    "src/ccad/nip_synthetic_v2.py",
    "src/ccad/nip_synthetic_v3.py",
    "scripts/run_m1_nip_d0_v3.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot(source: Path, run_dir: Path, prefix: str, relative: str) -> dict[str, object]:
    destination = run_dir / prefix / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(source),
        "snapshot": str(destination.relative_to(run_dir)),
        "sha256": sha(destination),
        "bytes": destination.stat().st_size,
    }


def derived_seed(protocol_hash: str, code_hash: str, family: str, pair_index: int, stream: str) -> int:
    payload = "||".join((protocol_hash, code_hash, "D0", family, str(pair_index), stream)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def finalize_failure(run_dir: Path, error: BaseException) -> None:
    if not run_dir.is_dir():
        return
    status_path = run_dir / "status.json"
    prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    write_json(status_path, {
        "status": "FAIL",
        "started_utc": prior.get("started_utc"),
        "ended_utc": utc_now(),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
    })
    with (run_dir / "stderr.log").open("a", encoding="utf-8") as handle:
        handle.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["execution_enabled"] or config["phase"] != "D0":
        raise ValueError("D0 v3 execution is not enabled")
    if config["truth_opened"] or config["held_out_eval_opened"] or config["real_sae_audit_opened"]:
        raise ValueError("D0 v3 config violates the closed-information contract")
    if tuple(config["families"]) != FAMILIES or config["pairs_per_family"] != 5:
        raise ValueError("D0 v3 grid differs from the frozen protocol")
    protocol_path = ROOT / config["protocol_path"]
    if sha(protocol_path) != config["protocol_sha256"]:
        raise ValueError("protocol hash drift")

    run_dir.mkdir(parents=True)
    source_rows = [snapshot(ROOT / name, run_dir, "source_snapshot", name) for name in SOURCES]
    code_hash = hashlib.sha256(json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    protocol_row = snapshot(protocol_path, run_dir, "input_snapshot", config["protocol_path"])
    config_row = snapshot(config_path, run_dir, "input_snapshot", "configs/m1_nip_d0_v3.json")
    write_json(run_dir / "config.resolved.json", config)
    write_json(run_dir / "code_hashes.json", {"aggregate_sha256": code_hash, "files": source_rows})
    write_json(run_dir / "inputs.json", {"protocol": protocol_row, "execution_config": config_row})
    write_json(run_dir / "environment.json", {
        "os": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "cuda": "not_applicable", "pytorch": "not_applicable", "device": "cpu",
    })
    started = utc_now()
    write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    write_json(run_dir / "manifest.json", {
        "artifact_schema_version": "ccad.run.v1",
        "run_id": run_dir.name,
        "tracker_parent": "M1_NIP_protocol_v3",
        "purpose": "D0 v3 observable-endpoint truth-closed engineering",
        "milestone": "M1",
        "evidence_level": config["evidence_level"],
        "started_utc": started,
        "config_hash": sha(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash,
        "protocol_sha256": protocol_row["sha256"],
        "execution_config_sha256": config_row["sha256"],
        "dataset_id": "synthetic_m1_nip_d0_v3",
        "dataset_revision": "generated_from_snapshotted_code",
        "device": "cpu",
        "resource_lease": "not_required_lightweight_cpu",
        "seed_fields": ["structural", "sample", "proposal", "solver"],
        "seed_derivation": "sha256(protocol_hash||code_hash||D0||family||pair_index||stream)[0:8]",
        "phase_seed_manifest_status": "GENERATED_AT_EXECUTION_IN_RAW_LEDGER",
        "statistical_unit": "seed_pair",
        "audit_opened": False,
        "truth_opened": False,
        "protocol_deviations": [],
    })

    records: list[dict[str, object]] = []
    for family in FAMILIES:
        for pair_index in range(config["pairs_per_family"]):
            seeds = {
                stream: derived_seed(config["protocol_sha256"], code_hash, family, pair_index, stream)
                for stream in ("structural", "sample", "proposal", "solver")
            }
            if len(set(seeds.values())) != 4:
                raise RuntimeError("derived seed collision")
            observed = generate_endpoint_observed(
                family,
                structural_seed=seeds["structural"],
                sample_seed=seeds["sample"],
                n=config["observations_per_pair"],
            )
            certificate = construction_certificate(observed)
            if certificate["target_atom_count"] != config["target_atom_count"]:
                raise RuntimeError("target atom count drift")
            if certificate["minimum_decoy_orthogonal_residual"] < config["minimum_decoy_orthogonal_residual"]:
                raise RuntimeError("decoy residual gate failed")
            if certificate["maximum_decoy_orthogonality_error"] > config["maximum_decoy_orthogonality_error"]:
                raise RuntimeError("decoy orthogonality gate failed")
            if certificate["cap_contract_pass"] is False:
                raise RuntimeError("cap-pressure contract failed")
            if family == "N11_downstream_cliff":
                endpoint = certificate["n11_endpoint"]
                if abs(certificate["n11_centered_residual"] - config["n11_centered_residual"]) > 1e-12:
                    raise RuntimeError("N11 centered residual drift")
                if config["approximate_tau_ctr"] - certificate["n11_centered_residual"] < config["n11_minimum_threshold_margin"]:
                    raise RuntimeError("N11 feasibility margin failed")
                if endpoint["minimum_normalized_cliff_margin"] + 1e-12 < config["n11_minimum_normalized_cliff_margin"]:
                    raise RuntimeError("N11 cliff boundary margin failed")
                if endpoint["cliff_effect_rmse"] < config["n11_minimum_cliff_effect_rmse"]:
                    raise RuntimeError("N11 cliff effect gate failed")
                if endpoint["smooth_effect_rmse"] > config["n11_maximum_smooth_effect_rmse"]:
                    raise RuntimeError("N11 smooth control gate failed")
            elif certificate["endpoint_present"]:
                raise RuntimeError("non-N11 family unexpectedly has an endpoint")

            k_ss, k_st, k_tt = observed_kernels(observed)
            proposal = source_conditioned_topk_proposal(
                k_ss, k_st, k_tt, source_atom_id=0, atom_cap=config["atom_cap"],
                g_max=config["g_max"], epsilon=config["epsilon"],
                candidate_budget=config["candidate_budget"],
                boundary_tie_tolerance=config["boundary_tie_tolerance"],
            )
            if proposal.status != "OK" or proposal.planned_support_count != config["expected_planned_support_count"]:
                raise RuntimeError("proposal contract failed")
            approximate = family in config["approximate_families"]
            result = minimum_support_contribution_correspondence(
                k_ss, k_st, k_tt,
                observed.source_mean_contributions, observed.target_mean_contributions,
                source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids,
                g_max=config["g_max"],
                tau_ctr=config["approximate_tau_ctr"] if approximate else config["exact_tau_ctr"],
                tau_mu=config["approximate_tau_mu"] if approximate else config["exact_tau_mu"],
                epsilon=config["epsilon"], candidate_budget=config["candidate_budget"],
                complete_universe=config["g_max"] >= k_st.shape[1],
            )
            fingerprint = hashlib.sha256(
                observed.source_contributions.tobytes() + observed.target_contributions.tobytes()
            ).hexdigest().upper()
            frozen = freeze_mscc_prediction(
                result, protocol_hash=config["protocol_sha256"], proposal_hash=proposal.proposal_hash,
                discovery_fingerprint=fingerprint, source_atom_id=0,
            )
            endpoint_metrics = certificate["n11_endpoint"]
            records.append({
                "run_id": run_dir.name,
                "family_id": family,
                "pair_index": pair_index,
                "statistical_unit": "seed_pair",
                "metric_version": "m1_nip_d0.v3",
                "seeds": seeds,
                "source_shape": observed.source_contributions.shape,
                "target_shape": observed.target_contributions.shape,
                "proposal_status": proposal.status,
                "proposal_hash": proposal.proposal_hash,
                "prediction_hash": frozen.prediction_hash,
                "planned_support_count": proposal.planned_support_count,
                "evaluated_count": result.evaluated_count,
                "minimum_decoy_orthogonal_residual": certificate["minimum_decoy_orthogonal_residual"],
                "maximum_decoy_orthogonality_error": certificate["maximum_decoy_orthogonality_error"],
                "cap_contract_pass": certificate["cap_contract_pass"],
                "endpoint_present": certificate["endpoint_present"],
                "n11_centered_residual": certificate["n11_centered_residual"],
                "n11_sample_mean_delta_norm": certificate.get("n11_sample_mean_delta_norm"),
                "n11_endpoint": endpoint_metrics,
                "truth_opened": False,
                "held_out_eval_opened": False,
            })

    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    n11_rows = [row for row in records if row["family_id"] == "N11_downstream_cliff"]
    runner_hash = next(row["sha256"] for row in source_rows if str(row["snapshot"]).endswith("run_m1_nip_d0_v3.py"))
    summary = {
        "status": "PASS",
        "semantic_outcome": "D0_V3_OBSERVABLE_ENDPOINT_ENGINEERING_ONLY_NO_LABEL_SCORING",
        "record_count": len(records),
        "raw_sha256": sha(raw_path),
        "generation_script_sha256": runner_hash,
        "n11_record_count": len(n11_rows),
        "minimum_n11_cliff_effect_rmse": min(row["n11_endpoint"]["cliff_effect_rmse"] for row in n11_rows),
        "minimum_n11_normalized_cliff_margin": min(row["n11_endpoint"]["minimum_normalized_cliff_margin"] for row in n11_rows),
        "maximum_n11_smooth_effect_rmse": max(row["n11_endpoint"]["smooth_effect_rmse"] for row in n11_rows),
        "proposal_refusal_count": sum(row["proposal_status"] != "OK" for row in records),
        "truth_opened": False,
        "held_out_eval_opened": False,
        "real_sae_audit_opened": False,
    }
    write_json(run_dir / "metrics.summary.json", summary)
    (run_dir / "stdout.log").write_text("D0 v3 observable endpoint truth-closed engineering completed\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(run_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": utc_now()})
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        if "--run-dir" in sys.argv:
            index = sys.argv.index("--run-dir") + 1
            if index < len(sys.argv):
                finalize_failure(Path(sys.argv[index]).resolve(), error)
        raise
    raise SystemExit(code)
