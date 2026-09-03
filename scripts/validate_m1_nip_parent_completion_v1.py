from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys


EXPECTED_NATIVE_LANES = {
    "MSCC", "CONTRIBUTION_NEAREST_ATOM", "PW_MCC_HUNGARIAN",
    "GREEDY_DECODER_COSINE", "DUSTBIN_SINKHORN", "BINARY_FORWARD_OMP",
    "OT_MASS_NATIVE_SUPPORT", "SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT",
    "RANDOM_MATCHED_GROUP",
}
EXPECTED_REFERENCES = {"SIGNED_CONTINUOUS_REGRESSION", "NONNEGATIVE_CONTINUOUS_REGRESSION"}
EXPECTED_STREAMS = {"structural", "mean", "discovery", "evaluation", "intervention", "solver"}
CRITICAL_METRICS = {
    "centered_residual_numerator", "centered_source_energy_denominator", "d_ctr",
    "mean_residual_numerator", "mean_source_energy_denominator", "d_mu",
    "bcc_value", "bcc_cross_inner", "bcc_source_energy", "bcc_target_energy",
    "bcc_normalized_residual", "psc_value", "psc_rank_source", "psc_rank_target",
    "psc_projector_distance_sq", "psc_principal_angles_radians", "effective_rank",
    "cancellation_ratio", "leave_one_out_leverage", "document_ess", "solver_gap",
    "proposal_stability", "proposal_recall", "conditional_solver_correctness",
    "end_to_end_recovery", "coverage", "terminal_reason",
}
EXPECTED_CONTROLS = {
    "N06_FULL_BLOCK_BCC_PSC", "N08_SIGNED_CONTINUOUS_REFERENCE",
    "N08_NONNEGATIVE_CONTINUOUS_REFERENCE", "N11_INDEPENDENT_INTERVENTION_STREAM",
    "N11_SMOOTH_CONTROL",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate(root: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = root / config["protocol_document"]
    mscc_source = (root / "src/ccad/mscc.py").read_text(encoding="utf-8")
    existing = sorted(
        path.name for stage in ("P1", "P2", "P3")
        for path in (root / "runs").glob(f"M1_NIP_PC1_V1_{stage}*") if path.is_dir()
    )
    checks = {
        "schema_and_locked_status": config["schema_version"] == "m1_nip_parent_completion.v1" and config["protocol_status"] == "LOCKED_FOR_IMPLEMENTATION",
        "execution_and_labels_closed": not config["execution_enabled"] and config["formal_seed_manifest_status"] == "UNGENERATED" and not config["synthetic_labels_opened"] and not config["real_sae_audit_opened"],
        "protocol_hash": protocol.is_file() and sha256(protocol) == config["protocol_sha256"],
        "fresh_namespace": config["fresh_namespace"] not in set(config["forbidden_reuse_namespaces"]),
        "formal_design": config["family_count"] == 12 and config["formal_pairs_per_family"] == 20 and config["repeat_unit"] == "structural_seed_pair" and config["block"] == "family",
        "independent_streams": set(config["required_seed_streams"]) == EXPECTED_STREAMS and config["require_pairwise_distinct_seed_streams"],
        "fixed_search_contract": config["g_max"] == 4 and config["target_atom_count"] == 20 and config["atom_cap"] == 20 and config["candidate_budget_per_query_per_native_lane"] == 7462,
        "all_native_lanes": set(config["registered_native_lanes"]) == EXPECTED_NATIVE_LANES,
        "all_non_native_references": set(config["registered_non_native_references"]) == EXPECTED_REFERENCES,
        "raw_metric_surface": config["metric_surface_schema"] == "metric_surface.v2-nip" and CRITICAL_METRICS <= set(config["mandatory_metric_fields"]),
        "family_controls": set(config["mandatory_family_controls"]) == EXPECTED_CONTROLS,
        "prelabel_gate": config["prelabel_validation_must_pass_before_truth_import"] and "prelabel_validation.json" in config["prelabel_required_artifacts"],
        "simplicity_rule": config["simplicity_rule"]["challenger"] == "BINARY_FORWARD_OMP" and config["simplicity_rule"]["action"] == "REMOVE_MSCC_FROM_HEADLINE_KEEP_AS_ABLATION",
        "bounded_absence_config": config["complete_universe_requires_gmax_cover_target_count"] and config["scalable_negative_identification"] == "UNRESOLVED",
        "bounded_absence_implementation": 'if complete_universe and g_max < target_count:' in mscc_source,
        "no_existing_completion_run": not existing,
    }
    return {
        "schema_version": "m1_nip_parent_completion_static_validation.v1",
        "config_path": str(config_path.relative_to(root)).replace("\\", "/"),
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(protocol),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "existing_completion_run_directories": existing,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_nip_parent_completion_v1.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, root / args.config)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = root / args.config
    protocol_path = root / json.loads(config_path.read_text(encoding="utf-8"))["protocol_document"]
    source_paths = [
        Path(__file__).resolve(), root / "src/ccad/mscc.py",
        root / "tests/test_m1_nip_parent_completion_v1.py",
    ]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    artifacts = {
        "validation.json": result,
        "resolved_config.json": json.loads(config_path.read_text(encoding="utf-8")),
        "environment.json": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "git_head_at_run": git_head,
        },
        "code_hashes.json": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in source_paths},
        "input_hashes.json": {
            str(config_path.relative_to(root)).replace("\\", "/"): sha256(config_path),
            str(protocol_path.relative_to(root)).replace("\\", "/"): sha256(protocol_path),
        },
        "status.json": {
            "status": result["status"], "finished_utc": datetime.now(timezone.utc).isoformat(),
            "truth_opened": False, "seeds_generated": False,
        },
    }
    for name, payload in artifacts.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stdout.log").write_text(json.dumps({"status": result["status"], "passed": result["passed_count"], "checks": result["check_count"]}) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    manifest = {path.name: sha256(path) for path in sorted(output_dir.iterdir())}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed_count"], "checks": result["check_count"], "output_dir": str(output_dir)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
