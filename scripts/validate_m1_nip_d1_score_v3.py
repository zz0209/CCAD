"""Independent rescore validator for a sealed v3 prediction and score run."""
from __future__ import annotations

import argparse, hashlib, importlib, json
from pathlib import Path

from ccad.nip_diagnostics_v3 import evaluate_orthogonal_diagnostics
from ccad.nip_synthetic_v3 import generate_endpoint_observed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def recompute_orthogonal(row: dict, diagnostic: dict, observations: int) -> dict:
    family, prediction = row["family_id"], row["prediction"]
    if family not in {"N09_cancellation", "N10_rare_occupancy", "N11_downstream_cliff", "N12_mean_mismatch"}:
        return {"status": "NOT_TARGETED"}
    if family == "N12_mean_mismatch":
        support, source = tuple(row["diagnostic_candidate"]["target_ids"]), "PRE_MEAN_CENTERED_ONLY_CANDIDATE"
    elif prediction["identification"] == "FOUND" and len(prediction["supports"]) == 1:
        support, source = tuple(prediction["supports"][0]["target_ids"]), "FROZEN_PREDICTED_SUPPORT"
    else:
        return {"status": "NOT_EVALUABLE", "reason": "NO_UNIQUE_FROZEN_PREDICTED_SUPPORT"}
    instance = generate_endpoint_observed(family, structural_seed=row["seeds"]["structural"], sample_seed=row["seeds"]["sample"], n=observations)
    value = evaluate_orthogonal_diagnostics(instance, support)
    result = {"status": "MEASURED", "support_source": source, "target_ids": list(support), "cancellation_energy_ratio": value.cancellation_energy_ratio, "aggregate_target_energy": value.aggregate_target_energy, "source_active_token_count": value.source_active_token_count, "source_active_document_count": value.source_active_document_count, "source_token_energy_kish_ess": value.source_token_energy_kish_ess, "source_document_energy_kish_ess": value.source_document_energy_kish_ess, "d_mu": value.d_mu, "endpoint": value.endpoint}
    tol = 1e-12
    if family == "N09_cancellation":
        result["measured_attribute"] = "OBSERVATIONALLY_UNSAFE" if value.cancellation_energy_ratio is not None and value.cancellation_energy_ratio + tol >= diagnostic["n09_minimum_unsafe_cancellation_energy_ratio"] else "SAFE"
    elif family == "N10_rare_occupancy":
        bad = value.source_active_document_count < diagnostic["n10_minimum_active_documents_for_sufficient_evidence"] or value.source_document_energy_kish_ess <= diagnostic["n10_maximum_insufficient_document_energy_kish_ess"] + tol
        result["measured_attribute"] = "INSUFFICIENT_EVIDENCE" if bad else "SUFFICIENT_EVIDENCE"
    elif family == "N11_downstream_cliff":
        endpoint = value.endpoint or {}
        bad = endpoint.get("cliff_effect_rmse", -1.0) + tol >= diagnostic["n11_minimum_cliff_effect_rmse"] and endpoint.get("smooth_effect_rmse", float("inf")) <= diagnostic["n11_maximum_smooth_effect_rmse"] + tol and endpoint.get("minimum_normalized_cliff_margin", -1.0) + tol >= diagnostic["n11_minimum_normalized_cliff_margin"]
        result["measured_attribute"] = "CAUSAL_FAIL" if bad else "CAUSAL_PASS"
    else:
        result["measured_attribute"] = "MEAN_MISMATCH" if value.d_mu > diagnostic["n12_mean_mismatch_tau"] else "MEAN_MATCH"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--score-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction, score = Path(args.prediction_run).resolve(), Path(args.score_run).resolve()
    checks = []
    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise ValueError(name)
    required = {"manifest.json", "environment.json", "inputs.json", "code_hashes.json", "status.json", "stdout.log", "stderr.log", "scores.raw.jsonl", "scores.summary.json"}
    check("score_files", all((score / name).is_file() for name in required))
    closure = json.loads((prediction / "prediction_closure.json").read_text(encoding="utf-8"))
    check("sealed_prediction", closure["state"] == "SEALED" and all(sha(prediction / name) == value for name, value in closure["files"].items()))
    config = json.loads((prediction / "config.resolved.json").read_text(encoding="utf-8"))
    inputs = json.loads((prediction / "inputs.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((prediction / inputs["diagnostic_config"]["snapshot"]).read_text(encoding="utf-8"))
    manifest = json.loads((score / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((score / "scores.summary.json").read_text(encoding="utf-8"))
    code = json.loads((score / "code_hashes.json").read_text(encoding="utf-8"))
    closure_hash = sha(prediction / "prediction_closure.json")
    check("score_provenance", manifest["prediction_closure_sha256"] == summary["prediction_closure_sha256"] == closure_hash and manifest["phase"] == config["phase"] and manifest["formal_d1_seed_consumed"] == config["formal_d1_seed_consumed"])
    check("score_hashes", summary["raw_sha256"] == sha(score / "scores.raw.jsonl") and code["scorer_sha256"] == summary["generation_script_sha256"] == sha(score / code["scorer_snapshot"]))
    prediction_rows = [json.loads(line) for line in (prediction / "predictions.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    score_rows = [json.loads(line) for line in (score / "scores.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    check("score_row_count", len(score_rows) == len(prediction_rows) == closure["row_count"])
    index = {(row["family_id"], row["pair_index"], row["atom_cap"]): row for row in prediction_rows}
    truth_module = importlib.import_module("ccad.nip_truth")
    negative = {"N04_absent_target", "N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation", "N08_continuous_only_representation", "N12_mean_mismatch"}
    ok = True
    for scored in score_rows:
        row = index[(scored["family_id"], scored["pair_index"], scored["atom_cap"])]
        pred, truth = row["prediction"], truth_module.nip_truth(row["family_id"])
        supports = {tuple(item["target_ids"]) for item in pred["supports"]}
        exact = row["family_id"] not in negative and pred["identification"] == "FOUND" and pred["multiplicity"] == truth.multiplicity and supports == set(truth.minimum_supports)
        ok &= scored["positive_exact"] == exact and scored["false_unique"] == (truth.multiplicity == "AMBIGUOUS" and pred["multiplicity"] == "UNIQUE") and scored["false_native_positive"] == (row["family_id"] in negative and pred["identification"] == "FOUND")
        ok &= scored["orthogonal"] == recompute_orthogonal(row, diagnostic, config["observations_per_pair"])
    check("independent_full_rescore", ok)
    by_cap = {}
    for cap in config["atom_caps"]:
        rows = [row for row in score_rows if row["atom_cap"] == cap]
        by_cap[str(cap)] = {"positive_exact_pairs": sum(row["positive_exact"] for row in rows), "false_unique_count": sum(row["false_unique"] for row in rows), "false_native_positive_count": sum(row["false_native_positive"] for row in rows), "budget_refusal_count": sum(row["budget_refusal"] for row in rows), "total_scored_supports": sum(row["scored_supports"] for row in rows)}
    check("aggregate_recomputed", summary["by_cap"] == by_cap)
    selected = min(config["atom_caps"], key=lambda cap: (-by_cap[str(cap)]["positive_exact_pairs"], by_cap[str(cap)]["false_unique_count"], by_cap[str(cap)]["false_native_positive_count"], by_cap[str(cap)]["budget_refusal_count"], by_cap[str(cap)]["total_scored_supports"], cap))
    check("selection_recomputed", summary["selected_cap"] == selected and summary["runtime_used_for_selection"] is False)
    targeted = [row for row in score_rows if row["family_id"] in {"N09_cancellation", "N10_rare_occupancy", "N11_downstream_cliff", "N12_mean_mismatch"} and row["atom_cap"] == selected]
    expected = {"N09_cancellation": "OBSERVATIONALLY_UNSAFE", "N10_rare_occupancy": "INSUFFICIENT_EVIDENCE", "N11_downstream_cliff": "CAUSAL_FAIL", "N12_mean_mismatch": "MEAN_MISMATCH"}
    check("selected_cap_attribute_suite", len(targeted) == 4 * config["pairs_per_family"] and all(row["orthogonal"].get("measured_attribute") == expected[row["family_id"]] for row in targeted))
    check("terminal_pass", json.loads((score / "status.json").read_text(encoding="utf-8"))["status"] == "PASS")
    result = {"status": "PASS", "prediction_run": prediction.name, "score_run": score.name, "check_count": len(checks), "checks": checks}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
