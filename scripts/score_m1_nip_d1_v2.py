"""Verify a sealed prediction artifact before dynamically opening NIP truth."""
from __future__ import annotations

import argparse, ast, hashlib, importlib, json, platform, shutil, sys, traceback
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prediction_payload(row: dict) -> dict:
    prediction = row["prediction"]
    return {"schema_version": "frozen_mscc_prediction.v1", "protocol_hash": row["protocol_hash"], "proposal_hash": row["proposal"]["proposal_hash"], "discovery_fingerprint": row["discovery_fingerprint"], "source_atom_id": row["source_atom_id"], "search_status": prediction["search_status"], "identification": prediction["identification"], "multiplicity": prediction["multiplicity"], "supports": [(item["target_ids"], item["d_ctr"], item["d_mu"]) for item in prediction["supports"]], "planned_candidate_count": prediction["planned_candidate_count"], "evaluated_count": prediction["evaluated_count"], "complete_universe": prediction["complete_universe"], "unresolved_reason": prediction["unresolved_reason"]}


def proposal_payload(row: dict) -> dict:
    p = row["proposal"]
    return {"schema_version": "source_conditioned_proposal.v1", "source_atom_id": p["source_atom_id"], "atom_cap": p["atom_cap"], "g_max": p["g_max"], "candidate_budget": p["candidate_budget"], "boundary_tie_tolerance": p["boundary_tie_tolerance"], "ranking": p["ranking"], "singleton_d_ctr": p["singleton_d_ctr"], "proposed_target_ids": p["proposed_target_ids"], "boundary_margin": p["boundary_margin"], "planned_support_count": p["planned_support_count"], "refusal_reason": p["refusal_reason"]}


def verify_prediction_run(prediction_dir: Path) -> list[dict]:
    closure_path = prediction_dir / "prediction_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if closure.get("state") != "SEALED" or closure.get("run_id") != prediction_dir.name:
        raise ValueError("invalid closure identity/state")
    for name, expected in closure["files"].items():
        if sha(prediction_dir / name) != expected:
            raise ValueError(f"closure hash mismatch: {name}")
    code = json.loads((prediction_dir / "code_hashes.json").read_text(encoding="utf-8"))
    if code["aggregate_sha256"] != closure["code_snapshot_hash"] or digest(code["files"]) != closure["code_snapshot_hash"]:
        raise ValueError("code snapshot aggregate mismatch")
    for item in code["files"]:
        if sha(prediction_dir / item["snapshot"]) != item["sha256"]:
            raise ValueError("source snapshot mismatch")
    inputs = json.loads((prediction_dir / "inputs.json").read_text(encoding="utf-8"))
    for item in inputs.values():
        if sha(prediction_dir / item["snapshot"]) != item["sha256"]:
            raise ValueError("input snapshot mismatch")
    if inputs["protocol"]["sha256"] != closure["protocol_sha256"]:
        raise ValueError("protocol snapshot mismatch")
    config = json.loads((prediction_dir / "config.resolved.json").read_text(encoding="utf-8"))
    snap_config = json.loads((prediction_dir / inputs["execution_config"]["snapshot"]).read_text(encoding="utf-8"))
    if config != snap_config or config["protocol_sha256"] != closure["protocol_sha256"] or config["truth_opened_in_prediction"] is not False:
        raise ValueError("resolved config mismatch")
    predictor = next(item for item in code["files"] if item["snapshot"].endswith("scripts/run_m1_nip_d1_predict_v2.py"))
    tree = ast.parse((prediction_dir / predictor["snapshot"]).read_text(encoding="utf-8"))
    if any("nip_truth" in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))):
        raise ValueError("prediction source statically imports truth")
    records = [json.loads(line) for line in (prediction_dir / "predictions.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != closure["row_count"]:
        raise ValueError("row count mismatch")
    for row in records:
        if row.get("truth_opened") is not False or row["protocol_hash"] != closure["protocol_sha256"]:
            raise ValueError("prediction information boundary mismatch")
        if digest(proposal_payload(row)) != row["proposal"]["proposal_hash"]:
            raise ValueError("proposal hash mismatch")
        if digest(prediction_payload(row)) != row["prediction"]["prediction_hash"]:
            raise ValueError("prediction hash mismatch")
    return records


def score_prediction_run(prediction_dir: Path, score_dir: Path) -> dict:
    records = verify_prediction_run(prediction_dir)  # Mandatory before truth import.
    if score_dir.exists():
        raise FileExistsError(score_dir)
    score_dir.mkdir(parents=False)
    scorer_source = Path(__file__).resolve()
    snapshot = score_dir / "source_snapshot" / "scripts" / scorer_source.name
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(scorer_source, snapshot)
    closure_hash = sha(prediction_dir / "prediction_closure.json")
    write_json(score_dir / "environment.json", {"os": platform.platform(), "python": sys.version, "device": "cpu"})
    write_json(score_dir / "inputs.json", {"prediction_run": str(prediction_dir), "prediction_closure_sha256": closure_hash, "prediction_raw_sha256": sha(prediction_dir / "predictions.raw.jsonl")})
    write_json(score_dir / "code_hashes.json", {"git_available": False, "scorer_snapshot": snapshot.relative_to(score_dir).as_posix(), "scorer_sha256": sha(snapshot)})
    started = now()
    write_json(score_dir / "manifest.json", {"artifact_schema_version": "ccad.score_run.v1", "run_id": score_dir.name, "prediction_run": prediction_dir.name, "prediction_closure_sha256": closure_hash, "information_order": "VERIFY_CLOSURE_THEN_DYNAMIC_TRUTH_IMPORT", "formal_d1_seed_consumed": False, "evidence_level": "I1_two_stage_contract_engineering_only", "started_utc": started})
    write_json(score_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    truth_module = importlib.import_module("ccad.nip_truth")
    by_cap = {}
    scored = []
    negative = {"N04_absent_target", "N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation", "N08_continuous_only_representation", "N12_mean_mismatch"}
    for row in records:
        truth = truth_module.nip_truth(row["family_id"])
        pred, cap = row["prediction"], row["atom_cap"]
        predicted_supports = {tuple(item["target_ids"]) for item in pred["supports"]}
        exact = row["family_id"] not in negative and pred["identification"] == "FOUND" and pred["multiplicity"] == truth.multiplicity and predicted_supports == set(truth.minimum_supports)
        false_unique = truth.multiplicity == "AMBIGUOUS" and pred["multiplicity"] == "UNIQUE"
        false_native = row["family_id"] in negative and pred["identification"] == "FOUND"
        scored.append({"family_id": row["family_id"], "pair_index": row["pair_index"], "atom_cap": cap, "prediction_hash": pred["prediction_hash"], "positive_exact": exact, "false_unique": false_unique, "false_native_positive": false_native, "budget_refusal": pred["search_status"] == "BUDGET_REFUSAL", "scored_supports": pred["evaluated_count"], "absence_lane": "FULL_EXHAUSTIVE" if pred["complete_universe"] else "SCALABLE_UNRESOLVED_ALLOWED"})
    for cap in sorted({row["atom_cap"] for row in scored}):
        rows = [row for row in scored if row["atom_cap"] == cap]
        by_cap[cap] = {"positive_exact_pairs": sum(row["positive_exact"] for row in rows), "false_unique_count": sum(row["false_unique"] for row in rows), "false_native_positive_count": sum(row["false_native_positive"] for row in rows), "budget_refusal_count": sum(row["budget_refusal"] for row in rows), "total_scored_supports": sum(row["scored_supports"] for row in rows)}
    selected = min(by_cap, key=lambda cap: (-by_cap[cap]["positive_exact_pairs"], by_cap[cap]["false_unique_count"], by_cap[cap]["false_native_positive_count"], by_cap[cap]["budget_refusal_count"], by_cap[cap]["total_scored_supports"], cap))
    (score_dir / "scores.raw.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in scored), encoding="utf-8")
    summary = {"status": "PASS", "prediction_run": prediction_dir.name, "prediction_closure_sha256": closure_hash, "truth_imported_after_closure_verification": True, "runtime_used_for_selection": False, "selected_cap": selected, "by_cap": by_cap, "raw_sha256": sha(score_dir / "scores.raw.jsonl"), "generation_script_sha256": sha(snapshot)}
    write_json(score_dir / "scores.summary.json", summary)
    (score_dir / "stdout.log").write_text("verified closure before label import; scoring completed\n", encoding="utf-8")
    (score_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(score_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": now()})
    return summary


def finalize_failure(score_dir: Path, exc: BaseException) -> None:
    if not score_dir.is_dir():
        return
    prior_path = score_dir / "status.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    write_json(prior_path, {"status": "FAIL", "started_utc": prior.get("started_utc"), "ended_utc": now(), "failure_type": type(exc).__name__, "failure_message": str(exc)})
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
