"""Score sealed v3 predictions and measure orthogonal attributes from observations."""
from __future__ import annotations

import argparse, ast, hashlib, importlib, json, platform, shutil, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

from ccad.nip_diagnostics_v3 import evaluate_orthogonal_diagnostics
from ccad.nip_synthetic_v3 import generate_endpoint_observed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def proposal_payload(row: dict) -> dict:
    p = row["proposal"]
    return {"schema_version": "source_conditioned_proposal.v1", "source_atom_id": p["source_atom_id"], "atom_cap": p["atom_cap"], "g_max": p["g_max"], "candidate_budget": p["candidate_budget"], "boundary_tie_tolerance": p["boundary_tie_tolerance"], "ranking": p["ranking"], "singleton_d_ctr": p["singleton_d_ctr"], "proposed_target_ids": p["proposed_target_ids"], "boundary_margin": p["boundary_margin"], "planned_support_count": p["planned_support_count"], "refusal_reason": p["refusal_reason"]}


def prediction_payload(row: dict) -> dict:
    p = row["prediction"]
    return {"schema_version": "frozen_mscc_prediction.v1", "protocol_hash": row["protocol_hash"], "proposal_hash": row["proposal"]["proposal_hash"], "discovery_fingerprint": row["discovery_fingerprint"], "source_atom_id": row["source_atom_id"], "search_status": p["search_status"], "identification": p["identification"], "multiplicity": p["multiplicity"], "supports": [(x["target_ids"], x["d_ctr"], x["d_mu"]) for x in p["supports"]], "planned_candidate_count": p["planned_candidate_count"], "evaluated_count": p["evaluated_count"], "complete_universe": p["complete_universe"], "unresolved_reason": p["unresolved_reason"]}


def verify_prediction_run(run: Path) -> tuple[list[dict], dict, dict]:
    closure = json.loads((run / "prediction_closure.json").read_text(encoding="utf-8"))
    if closure.get("state") != "SEALED" or closure.get("run_id") != run.name:
        raise ValueError("invalid closure identity/state")
    for name, expected in closure["files"].items():
        if sha(run / name) != expected:
            raise ValueError(f"closure hash mismatch: {name}")
    code = json.loads((run / "code_hashes.json").read_text(encoding="utf-8"))
    if digest(code["files"]) != code["aggregate_sha256"] or code["aggregate_sha256"] != closure["code_snapshot_hash"]:
        raise ValueError("code snapshot aggregate mismatch")
    if not all(sha(run / item["snapshot"]) == item["sha256"] for item in code["files"]):
        raise ValueError("source snapshot mismatch")
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    if not all(sha(run / item["snapshot"]) == item["sha256"] for item in inputs.values()):
        raise ValueError("input snapshot mismatch")
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((run / inputs["diagnostic_config"]["snapshot"]).read_text(encoding="utf-8"))
    if config != json.loads((run / inputs["execution_config"]["snapshot"]).read_text(encoding="utf-8")):
        raise ValueError("resolved config mismatch")
    if config["protocol_sha256"] != closure["protocol_sha256"] or config["diagnostic_config_sha256"] != closure["diagnostic_config_sha256"]:
        raise ValueError("protocol/diagnostic closure mismatch")
    predictor = next(item for item in code["files"] if item["snapshot"].endswith("scripts/run_m1_nip_d1_predict_v3.py"))
    tree = ast.parse((run / predictor["snapshot"]).read_text(encoding="utf-8"))
    if any("nip_truth" in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))):
        raise ValueError("prediction source statically imports truth")
    rows = [json.loads(line) for line in (run / "predictions.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != closure["row_count"]:
        raise ValueError("row count mismatch")
    for row in rows:
        if row["truth_opened"] is not False or digest(proposal_payload(row)) != row["proposal"]["proposal_hash"] or digest(prediction_payload(row)) != row["prediction"]["prediction_hash"]:
            raise ValueError("prediction information boundary/hash mismatch")
        candidate = row["diagnostic_candidate"]
        payload = {"schema_version": "centered_only_candidate.v1", "source_atom_id": row["source_atom_id"], "proposed_target_ids": tuple(row["proposal"]["proposed_target_ids"]), "g_max": config["g_max"], "candidate_budget": config["candidate_budget"], "target_ids": tuple(candidate["target_ids"]), "d_ctr": candidate["d_ctr"], "evaluated_count": candidate["evaluated_count"]}
        if digest(payload) != candidate["candidate_hash"]:
            raise ValueError("diagnostic candidate hash mismatch")
    return rows, config, diagnostic


def orthogonal_measurement(row: dict, diagnostic: dict, observations_per_pair: int) -> dict:
    family, prediction = row["family_id"], row["prediction"]
    if family not in {"N09_cancellation", "N10_rare_occupancy", "N11_downstream_cliff", "N12_mean_mismatch"}:
        return {"status": "NOT_TARGETED"}
    if family == "N12_mean_mismatch":
        support = tuple(row["diagnostic_candidate"]["target_ids"])
        support_source = "PRE_MEAN_CENTERED_ONLY_CANDIDATE"
    elif prediction["identification"] == "FOUND" and len(prediction["supports"]) == 1:
        support = tuple(prediction["supports"][0]["target_ids"])
        support_source = "FROZEN_PREDICTED_SUPPORT"
    else:
        return {"status": "NOT_EVALUABLE", "reason": "NO_UNIQUE_FROZEN_PREDICTED_SUPPORT"}
    observed = generate_endpoint_observed(row["family_id"], structural_seed=row["seeds"]["structural"], sample_seed=row["seeds"]["sample"], n=observations_per_pair)
    measured = evaluate_orthogonal_diagnostics(observed, support)
    result = {"status": "MEASURED", "support_source": support_source, "target_ids": list(support), "cancellation_energy_ratio": measured.cancellation_energy_ratio, "aggregate_target_energy": measured.aggregate_target_energy, "source_active_token_count": measured.source_active_token_count, "source_active_document_count": measured.source_active_document_count, "source_token_energy_kish_ess": measured.source_token_energy_kish_ess, "source_document_energy_kish_ess": measured.source_document_energy_kish_ess, "d_mu": measured.d_mu, "endpoint": measured.endpoint}
    tolerance = 1e-12
    if family == "N09_cancellation":
        result["measured_attribute"] = "OBSERVATIONALLY_UNSAFE" if measured.cancellation_energy_ratio is not None and measured.cancellation_energy_ratio + tolerance >= diagnostic["n09_minimum_unsafe_cancellation_energy_ratio"] else "SAFE"
    elif family == "N10_rare_occupancy":
        insufficient = measured.source_active_document_count < diagnostic["n10_minimum_active_documents_for_sufficient_evidence"] or measured.source_document_energy_kish_ess <= diagnostic["n10_maximum_insufficient_document_energy_kish_ess"] + tolerance
        result["measured_attribute"] = "INSUFFICIENT_EVIDENCE" if insufficient else "SUFFICIENT_EVIDENCE"
    elif family == "N11_downstream_cliff":
        endpoint = measured.endpoint or {}
        failed = endpoint.get("cliff_effect_rmse", -1.0) + tolerance >= diagnostic["n11_minimum_cliff_effect_rmse"] and endpoint.get("smooth_effect_rmse", float("inf")) <= diagnostic["n11_maximum_smooth_effect_rmse"] + tolerance and endpoint.get("minimum_normalized_cliff_margin", -1.0) + tolerance >= diagnostic["n11_minimum_normalized_cliff_margin"]
        result["measured_attribute"] = "CAUSAL_FAIL" if failed else "CAUSAL_PASS"
    else:
        result["measured_attribute"] = "MEAN_MISMATCH" if measured.d_mu > diagnostic["n12_mean_mismatch_tau"] else "MEAN_MATCH"
    return result


def score_prediction_run(prediction_dir: Path, score_dir: Path) -> dict:
    rows, config, diagnostic = verify_prediction_run(prediction_dir)  # Must precede directory creation and truth import.
    if score_dir.exists():
        raise FileExistsError(score_dir)
    score_dir.mkdir(parents=False)
    source = Path(__file__).resolve()
    snapshot = score_dir / "source_snapshot" / "scripts" / source.name
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(source, snapshot)
    closure_hash, started = sha(prediction_dir / "prediction_closure.json"), now()
    write_json(score_dir / "environment.json", {"os": platform.platform(), "python": sys.version, "device": "cpu"})
    write_json(score_dir / "inputs.json", {"prediction_run": str(prediction_dir), "prediction_closure_sha256": closure_hash, "prediction_raw_sha256": sha(prediction_dir / "predictions.raw.jsonl")})
    write_json(score_dir / "code_hashes.json", {"git_available": False, "scorer_snapshot": snapshot.relative_to(score_dir).as_posix(), "scorer_sha256": sha(snapshot)})
    write_json(score_dir / "manifest.json", {"artifact_schema_version": "ccad.score_run.v3", "run_id": score_dir.name, "prediction_run": prediction_dir.name, "prediction_closure_sha256": closure_hash, "information_order": "VERIFY_CLOSURE_THEN_DYNAMIC_TRUTH_IMPORT", "phase": config["phase"], "formal_d1_seed_consumed": config["formal_d1_seed_consumed"], "evidence_level": config["evidence_level"], "started_utc": started})
    write_json(score_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    truth_module = importlib.import_module("ccad.nip_truth")
    negative = {"N04_absent_target", "N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation", "N08_continuous_only_representation", "N12_mean_mismatch"}
    scored = []
    for row in rows:
        truth, pred = truth_module.nip_truth(row["family_id"]), row["prediction"]
        supports = {tuple(item["target_ids"]) for item in pred["supports"]}
        scored.append({"family_id": row["family_id"], "pair_index": row["pair_index"], "atom_cap": row["atom_cap"], "prediction_hash": pred["prediction_hash"], "positive_exact": row["family_id"] not in negative and pred["identification"] == "FOUND" and pred["multiplicity"] == truth.multiplicity and supports == set(truth.minimum_supports), "false_unique": truth.multiplicity == "AMBIGUOUS" and pred["multiplicity"] == "UNIQUE", "false_native_positive": row["family_id"] in negative and pred["identification"] == "FOUND", "budget_refusal": pred["search_status"] == "BUDGET_REFUSAL", "scored_supports": pred["evaluated_count"], "orthogonal": orthogonal_measurement(row, diagnostic, config["observations_per_pair"])})
    by_cap = {}
    for cap in config["atom_caps"]:
        cap_rows = [row for row in scored if row["atom_cap"] == cap]
        by_cap[str(cap)] = {"positive_exact_pairs": sum(row["positive_exact"] for row in cap_rows), "false_unique_count": sum(row["false_unique"] for row in cap_rows), "false_native_positive_count": sum(row["false_native_positive"] for row in cap_rows), "budget_refusal_count": sum(row["budget_refusal"] for row in cap_rows), "total_scored_supports": sum(row["scored_supports"] for row in cap_rows)}
    selected = min(config["atom_caps"], key=lambda cap: (-by_cap[str(cap)]["positive_exact_pairs"], by_cap[str(cap)]["false_unique_count"], by_cap[str(cap)]["false_native_positive_count"], by_cap[str(cap)]["budget_refusal_count"], by_cap[str(cap)]["total_scored_supports"], cap))
    (score_dir / "scores.raw.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in scored), encoding="utf-8")
    summary = {"status": "PASS", "prediction_run": prediction_dir.name, "prediction_closure_sha256": closure_hash, "truth_imported_after_closure_verification": True, "runtime_used_for_selection": False, "selected_cap": selected, "by_cap": by_cap, "raw_sha256": sha(score_dir / "scores.raw.jsonl"), "generation_script_sha256": sha(snapshot)}
    write_json(score_dir / "scores.summary.json", summary)
    (score_dir / "stdout.log").write_text("verified closure before truth import; observational scoring completed\n", encoding="utf-8")
    (score_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(score_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": now()})
    return summary


def finalize_failure(score_dir: Path, exc: BaseException) -> None:
    if score_dir.is_dir():
        status = score_dir / "status.json"
        prior = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {}
        write_json(status, {"status": "FAIL", "started_utc": prior.get("started_utc"), "ended_utc": now(), "failure_type": type(exc).__name__, "failure_message": str(exc)})
        (score_dir / "stderr.log").write_text("".join(traceback.format_exception(exc)), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--score-run", required=True)
    args = parser.parse_args()
    score_prediction_run(Path(args.prediction_run).resolve(), Path(args.score_run).resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--score-run" in sys.argv:
            finalize_failure(Path(sys.argv[sys.argv.index("--score-run") + 1]).resolve(), error)
        raise
