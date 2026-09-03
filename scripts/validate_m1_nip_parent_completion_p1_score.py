"""Independent raw-identity validator for a PC2 P1 post-closure score run."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from ccad.nip_synthetic_v2 import generate_cap_identifiable_observed
from ccad.nip_synthetic_v3 import generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, atol=1e-10, rtol=1e-10))


def validate(prediction_dir: Path, score_dir: Path) -> dict:
    checks = {}
    closure = json.loads((prediction_dir / "prediction_closure.json").read_text(encoding="utf-8"))
    prelabel = json.loads((prediction_dir / "prelabel_validation.json").read_text(encoding="utf-8"))
    config = json.loads((prediction_dir / "resolved_config.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / config["parent_config_path"]).read_text(encoding="utf-8"))
    manifest = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((score_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((score_dir / "status.json").read_text(encoding="utf-8"))
    checks["prediction_and_prelabel_binding"] = closure["state"] == "SEALED" and prelabel["status"] == "PASS" and manifest["prediction_closure_sha256"] == sha(prediction_dir / "prediction_closure.json") and manifest["prelabel_validation_sha256"] == sha(prediction_dir / "prelabel_validation.json")
    checks["scorer_snapshot_binding"] = sha(score_dir / manifest["scorer_snapshot"]) == manifest["scorer_sha256"] and manifest["truth_open_order"] == "AFTER_CLOSURE_AND_PRELABEL_PASS"
    checks["score_hash_and_status"] = status["status"] == "PASS" and summary["status"] == "PASS" and summary["raw_sha256"] == sha(score_dir / "scores.jsonl")
    rows = [json.loads(line) for line in (score_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line]
    predictions = {(row["family_id"], row["pair_index"], row["lane"]): row for row in [json.loads(line) for line in (prediction_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]}
    seeds = {(row["family_id"], row["pair_index"]): row["seeds"] for row in json.loads((prediction_dir / "seed_ledger.json").read_text(encoding="utf-8"))["rows"]}
    expected_rows = config["expected_prediction_rows"]
    checks["row_grid"] = len(rows) == len(predictions) == expected_rows and len({(row["family_id"], row["pair_index"], row["lane"]) for row in rows}) == expected_rows
    truth_module = importlib.import_module("ccad.nip_truth")
    raw_ok = True
    truth_ok = True
    continuous_ok = True
    controls_ok = True
    for row in rows:
        key = row["family_id"], row["pair_index"], row["lane"]
        frozen = predictions[key]
        streams = seeds[key[:2]]
        mean = generate_cap_identifiable_observed(key[0], structural_seed=streams["structural"], sample_seed=streams["mean"], n=config["sample_sizes"]["mean"])
        evaluation = generate_endpoint_observed(key[0], structural_seed=streams["structural"], sample_seed=streams["evaluation"], n=parent["sample_sizes"]["evaluation"])
        truth = truth_module.nip_truth(key[0])
        supports = {tuple(value) for value in row["frozen_supports"]}
        expected_exact = truth.identification == "FOUND" and supports == set(truth.minimum_supports) and frozen["prediction"].get("multiplicity") == truth.multiplicity
        truth_ok &= row["positive_exact"] == expected_exact
        truth_ok &= row["false_native_positive"] == (truth.identification != "FOUND" and bool(supports))
        mandatory = set(parent["mandatory_metric_fields"])
        raw_ok &= all(mandatory <= set(surface) for surface in row["metric_surfaces"])
        for support, surface in zip(row["frozen_supports"], row["metric_surfaces"]):
            if not support:
                continue
            source = evaluation.source_contributions[:, 0, :]
            target = np.sum(evaluation.target_contributions[:, np.asarray(support), :], axis=1)
            ctr_num = float(np.mean(np.sum((source - target) ** 2, axis=1)))
            ctr_den = float(np.mean(np.sum(source * source, axis=1))) + config["epsilon"]
            source_mean = mean.source_mean_contributions[:, 0]
            target_mean = np.sum(mean.target_mean_contributions[:, np.asarray(support)], axis=1)
            mu_num = float(np.sum((source_mean - target_mean) ** 2))
            mu_den = float(np.sum(source_mean * source_mean)) + config["epsilon"]
            raw_ok &= close(surface["centered_residual_numerator"], ctr_num) and close(surface["centered_source_energy_denominator"], ctr_den) and close(surface["d_ctr"], ctr_num / ctr_den)
            raw_ok &= close(surface["mean_residual_numerator"], mu_num) and close(surface["mean_source_energy_denominator"], mu_den) and close(surface["d_mu"], mu_num / mu_den)
        if row["kind"] == "CONTINUOUS_REFERENCE":
            metric = row["continuous_evaluation"]
            coefficients = np.asarray(frozen["prediction"]["coefficients"])
            source = evaluation.source_contributions[:, 0, :]
            target = np.einsum("nat,a->nt", evaluation.target_contributions, coefficients)
            numerator = float(np.sum((source - target) ** 2))
            continuous_ok &= close(metric["residual_numerator"], numerator)
        if key[0] == "N06_exact_dense_orthogonal_rotation" and key[2] == "MSCC":
            control = row["n06_full_block_control"] or {}
            controls_ok &= close(control.get("d_ctr", float("inf")), 0.0)
            controls_ok &= close(control.get("bcc_value", -float("inf")), 1.0)
            controls_ok &= close(control.get("psc_value", -float("inf")), 1.0)
            controls_ok &= control.get("psc_rank_source") == 2 and control.get("psc_rank_target") == 2
        if key[0] == "N11_downstream_cliff" and supports:
            controls_ok &= row["intervention_evaluation"] is not None
    checks["raw_native_identities"] = bool(raw_ok)
    checks["truth_classification_recomputed"] = bool(truth_ok)
    checks["continuous_evaluation_recomputed"] = bool(continuous_ok)
    expected_n08_controls = 2 * config["pairs_per_family"]
    checks["mandatory_controls_present"] = bool(controls_ok) and sum(row["continuous_evaluation"] is not None for row in rows if row["family_id"] == "N08_continuous_only_representation") == expected_n08_controls
    checks["summary_aggregates"] = (
        summary["native_positive_exact"] == {lane: sum(row["positive_exact"] for row in rows if row["lane"] == lane) for lane in config["native_lanes"]}
        and summary["native_false_positive"] == {lane: sum(row["false_native_positive"] for row in rows if row["lane"] == lane) for lane in config["native_lanes"]}
        and summary["native_false_unique"] == {lane: sum(row["false_unique"] for row in rows if row["lane"] == lane) for lane in config["native_lanes"]}
    )
    result = {"schema_version": "pc2.p1.score_validation.v1", "prediction_run": prediction_dir.name, "score_run": score_dir.name, "checks": checks, "check_count": len(checks), "passed_count": sum(checks.values()), "status": "PASS" if all(checks.values()) else "FAIL"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--score-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(Path(args.prediction_run).resolve(), Path(args.score_run).resolve())
    write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
