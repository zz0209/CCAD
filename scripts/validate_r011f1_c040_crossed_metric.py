"""Independent artifact/science validation for a crossed C040 metric run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.causal_metric_probe import (  # noqa: E402
    orthonormal_probe_directions,
    select_boundary_safe_document_balanced_states,
)
from ccad.fuzzy_correspondence import fit_crossed_probe_metric  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / cfg["protocol_config_path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    input_rows = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))["inputs"]
    token_manifest_path = ROOT / cfg["token_manifest_path"]
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    token_info = token_manifest["outputs"][cfg["split"]]
    token_path = ROOT / "runs" / cfg["paired_corpus_run"] / token_info["path"]
    sequence_path = ROOT / cfg["sequence_records_path"]
    sequences = json.loads(sequence_path.read_text(encoding="utf-8"))["sequences"]
    tokens = np.memmap(token_path, dtype="<u2", mode="r").reshape(token_info["sequences"], cfg["context_length"])
    expected_states = select_boundary_safe_document_balanced_states(
        sequences, tokens, split=cfg["split"], count=protocol["probe_states"],
        token_positions=tuple(protocol["probe_token_positions"]), salt=protocol["probe_state_salt"],
        eot_token_id=protocol["eot_token_id"],
        minimum_tokens_after_boundary=protocol["minimum_tokens_after_causal_boundary"],
    )
    actual_states = [json.loads(line) for line in (run_dir / "probe_states.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_directions = orthonormal_probe_directions(cfg["hook_hidden_size"], protocol["probe_direction_salt"])
    observations = np.load(run_dir / "probe_observations.npz")
    directions = observations["directions"].astype(np.float64)
    effects = observations["effects"].astype(np.float64)
    recomputed = fit_crossed_probe_metric(
        directions, effects, ridge_fraction=protocol["probe_ridge_fraction"],
        relative_tolerance=protocol["metric_eigenvalue_relative_tolerance"],
    )
    saved = np.load(run_dir / "causal_metric.npz")
    saved_matrix = saved["matrix"].astype(np.float64)
    metric_relative_error = float(np.linalg.norm(recomputed.matrix - saved_matrix) / np.linalg.norm(recomputed.matrix))
    state_energy = np.sum(effects ** 2, axis=(1, 2))
    state_shares = state_energy / np.sum(state_energy)
    effective_state_count = float(1.0 / np.sum(state_shares ** 2))
    expected_science_checks = {
        "state_influence_max_share": float(state_shares.max()) <= protocol["maximum_state_trace_share"],
        "state_influence_effective_count": effective_state_count >= protocol["minimum_effective_state_count"],
    }
    failed_checks = sorted(name for name, value in raw["checks"].items() if not value)
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "protocol_hash": sha256(protocol_path) == cfg["protocol_config_sha256"],
        "all_declared_inputs_bound": all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"] for row in input_rows),
        "input_hashes": sha256(token_manifest_path) == cfg["token_manifest_sha256"] and sha256(sequence_path) == cfg["sequence_records_sha256"],
        "audit_closed": not cfg["audit_opened"] and not protocol["audit_opened"],
        "state_ledger_exact": actual_states == expected_states,
        "boundary_rule_exact": min(row["tokens_since_causal_boundary"] for row in actual_states) >= protocol["minimum_tokens_after_causal_boundary"],
        "shared_basis_exact": np.allclose(directions, expected_directions, atol=1e-7, rtol=0),
        "shared_basis_orthonormal": float(np.max(np.abs(directions @ directions.T - np.eye(len(directions))))) <= 1e-6,
        "effect_tensor_complete": effects.shape == (protocol["probe_states"], protocol["shared_probe_directions"], protocol["output_logit_sketch_dim"]) and np.all(np.isfinite(effects)),
        "metric_recomputed": metric_relative_error <= 1e-6,
        "state_aggregates_recomputed": abs(float(state_shares.max()) - raw["maximum_state_trace_share"]) <= 1e-12 and abs(effective_state_count - raw["effective_state_count"]) <= 1e-12,
        "science_fail_reproduced": status["status"] == "FAIL" and failed_checks == sorted(expected_science_checks) and not any(expected_science_checks.values()),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
        "metric_relative_error_from_saved_float32": metric_relative_error,
        "maximum_state_trace_share": float(state_shares.max()),
        "effective_state_count": effective_state_count,
        "failed_run_checks": failed_checks,
        "interpretation": "INDEPENDENTLY_CONFIRMED_SCIENTIFIC_FAIL_STATE_INFLUENCE",
    }
    output = run_dir / "independent_validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
