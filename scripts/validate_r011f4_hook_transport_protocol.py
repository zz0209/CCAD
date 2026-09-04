"""Static, pre-execution validation of the C047 real-screen protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = [
        ("synthetic_gate_status_path", "synthetic_gate_status_sha256"),
        ("synthetic_gate_metrics_path", "synthetic_gate_metrics_sha256"),
        ("reference_surface_path", "reference_surface_sha256"),
        ("source_census_path", "source_census_sha256"),
        ("sequence_records_path", "sequence_records_sha256"),
    ]
    residual_mode = "nuisance_state_count" in cfg
    if residual_mode:
        pairs.append(("unresidualized_transport_surface_path", "unresidualized_transport_surface_sha256"))
    bound = {path_key: digest(root / cfg[path_key]) == cfg[hash_key] for path_key, hash_key in pairs}
    bound["asset_manifest"] = digest(Path(cfg["bulk_asset_dir"]) / "asset_manifest.json") == cfg["asset_manifest_sha256"]
    bound["raw_hook_manifest"] = digest(Path(cfg["raw_hook_asset_dir"]) / "raw_hook_manifest.json") == cfg["raw_hook_manifest_sha256"]
    synthetic_status = json.loads((root / cfg["synthetic_gate_status_path"]).read_text(encoding="utf-8"))
    checks = {
        "all_inputs_bound": all(bound.values()),
        "synthetic_pass": synthetic_status["status"] == "PASS",
        "execution_enabled": cfg["execution_enabled"] is True,
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"] and cfg["splits"] == ["discovery", "calibration"],
        "source_target_representation": ("source_only_discovery_positive_local_contribution_pca" in cfg["source_process"] and "complete_centered_target_sae_reconstruction" in cfg["target_process"]),
        "rank_and_ridge_frozen": cfg["candidate_ranks"] == [1, 2, 4, 8] and cfg["ridge_fraction"] == .001,
        "strong_controls_frozen": set(cfg["controls"]) == ({"residual_query_conditioned_raw_hook_transport", "residual_query_agnostic_whole_sae_global_transport", "unresidualized_c047_query_transport"} if residual_mode else {"query_conditioned_raw_hook_transport", "query_agnostic_whole_sae_global_transport"}),
        "meaningful_transfer_gate": cfg["minimum_calibration_bcc"] == .8 and cfg["maximum_calibration_normalized_residual"] == .2,
        "control_advantage_gate": cfg["minimum_control_specificity_advantage"] == .05 and cfg["minimum_collision_improvement_over_global"] == .05,
        "replication_structure": cfg["anchor_units"] == 160 and cfg["all_condition_queries"] == 160 and cfg["ordered_target_seeds_per_query"] == 4,
        "progression_gate": cfg["minimum_progression_coverage"] == .1 and cfg["minimum_covered_strata"] == 4 and cfg["require_all_represented_ordered_directions"] is True,
    }
    if residual_mode:
        checks["nuisance_rule_frozen"] = cfg["nuisance_state_count"] == 4096 and cfg["nuisance_explained_variance_threshold"] == .9 and cfg["nuisance_maximum_rank"] == 64
        checks["residual_energy_floor_frozen"] = cfg["minimum_source_residual_energy_fraction"] == .2
    print(json.dumps({"checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "input_bindings": bound}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
