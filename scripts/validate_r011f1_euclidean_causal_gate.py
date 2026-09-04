"""Independent integrity/statistical validation of an R011-F1 causal gate run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    pairs = [json.loads(line) for line in (run_dir / "causal_pair_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    units = [json.loads(line) for line in (run_dir / "intervention_units.jsonl").read_text(encoding="utf-8").splitlines() if line]
    selection = json.loads((run_dir / "selected_units_and_sequences.json").read_text(encoding="utf-8"))["units"]
    evaluated = cfg["evaluated_methods"]
    endpoint = cfg["primary_endpoint"]

    pair_keys = [(row["source_seed"], row["source_atom"], row["target_seed"], row["method"]) for row in pairs]
    expected_methods = Counter(cfg["methods"])
    per_relation_methods = {
        (row["source_seed"], row["source_atom"], row["target_seed"]): Counter(
            candidate["method"] for candidate in pairs
            if (candidate["source_seed"], candidate["source_atom"], candidate["target_seed"]) == (row["source_seed"], row["source_atom"], row["target_seed"])
        )
        for row in selection
    }
    coverages = {
        method: float(np.mean([row["qualifies_primary_endpoint"] for row in pairs if row["method"] == method]))
        for method in evaluated
    }
    medians = {}
    for method in evaluated:
        rows = [row for row in pairs if row["method"] == method]
        medians[method] = {
            "normalized_effect_error": float(np.median([row["endpoints"][endpoint]["normalized_effect_error"] for row in rows])),
            "effect_bcc": float(np.median([row["endpoints"][endpoint]["effect_bcc"] for row in rows])),
            "source_off_query_fraction": float(np.median([row["endpoints"][endpoint]["source_off_query_fraction"] for row in rows])),
            "source_effect_rms": float(np.median([row["endpoints"][endpoint]["source_effect_rms"] for row in rows])),
        }
    primary = medians["EUCLIDEAN_FCC_RELATION"]
    minimum_gain = cfg["progression"]["minimum_gain_against_each_raw_and_global_control"]
    gains = {}
    for control in ("RAW_HOOK_QUERY_PCA", "GLOBAL_FCC_RELATION"):
        effect = medians[control]["normalized_effect_error"] - primary["normalized_effect_error"]
        specificity = medians[control]["source_off_query_fraction"] - primary["source_off_query_fraction"]
        gains[control] = {"effect_consistency_gain": effect, "query_specificity_gain": specificity, "passes_either_axis": bool(max(effect, specificity) >= minimum_gain)}
    if coverages["EUCLIDEAN_FCC_RELATION"] < cfg["progression"]["minimum_primary_coverage"]:
        decision = "STOP_FCC_CAUSAL_EFFECT_OR_CONSISTENCY_FLOOR"
    elif not all(value["passes_either_axis"] for value in gains.values()):
        decision = "STOP_FCC_CAUSAL_SPECIFICITY_NOT_IDENTIFIED"
    else:
        decision = "PROCEED_TO_FULL_640_PREAUDIT_FREEZE"

    source_energy_match = []
    target_energy_match = []
    for row in units:
        if "source_unscaled_block_norm" in row:
            source_energy_match.append(abs(row["source_unscaled_block_norm"] * row["source_energy_scale"] - row["reference_hook_norm"]) <= 1e-8 * max(row["reference_hook_norm"], 1.0))
            target_energy_match.append(abs(row["target_unscaled_block_norm"] * row["target_energy_scale"] - row["reference_hook_norm"]) <= 1e-8 * max(row["reference_hook_norm"], 1.0))
        else:
            source_energy_match.append(abs(row["source_unscaled_hook_norm"] * row["source_energy_scale"] - row["reference_hook_norm"]) <= 1e-8 * max(row["reference_hook_norm"], 1.0))
            target_energy_match.append(abs(row["target_unscaled_hook_norm"] * row["target_energy_scale"] - row["reference_hook_norm"]) <= 1e-8 * max(row["reference_hook_norm"], 1.0))

    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "status_pass": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "raw_hash": sha256(run_dir / "metrics.raw.jsonl") == json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))["metrics_raw_sha256"],
        "pair_hash": sha256(run_dir / "causal_pair_metrics.jsonl") == record["causal_pair_metrics_sha256"],
        "unit_hash": sha256(run_dir / "intervention_units.jsonl") == record["intervention_units_sha256"],
        "selection_hash": sha256(run_dir / "selected_units_and_sequences.json") == record["selection_sha256"],
        "eight_unique_strata": sorted(row["energy_stratum"] for row in selection) == list(range(8)),
        "two_sequences_each": all(len(row["sequence_ids"]) == cfg["sequences_per_unit"] for row in selection),
        "unique_pair_method_grid": len(pair_keys) == len(set(pair_keys)) == cfg["selected_units"] * len(cfg["methods"]),
        "method_grid_exact": all(value == expected_methods for value in per_relation_methods.values()),
        "source_energy_matching": all(source_energy_match),
        "target_energy_matching": all(target_energy_match),
        "coverage_recomputed": all(np.isclose(coverages[name], record["coverages"][name]) for name in evaluated),
        "medians_recomputed": all(np.isclose(medians[method][field], record["method_medians"][method][field]) for method in evaluated for field in medians[method]),
        "gains_recomputed": all(np.isclose(gains[control][field], record["primary_gains"][control][field]) for control in gains for field in ("effect_consistency_gain", "query_specificity_gain")),
        "decision_recomputed": decision == record["screen_decision"],
        "model_forward_count": record["model_forwards"] == 2 * cfg["selected_units"] * cfg["sequences_per_unit"] + 2 * len(units),
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
    }
    output = {"run_id": run_dir.name, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "screen_decision": decision, "coverages": coverages, "method_medians": medians}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
