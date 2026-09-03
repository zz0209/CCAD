"""Independent artifact validator for sealed v2 prediction and scored runs."""
from __future__ import annotations

import argparse, ast, hashlib, importlib, json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def proposal_payload(row: dict) -> dict:
    p = row["proposal"]
    return {"schema_version": "source_conditioned_proposal.v1", "source_atom_id": p["source_atom_id"], "atom_cap": p["atom_cap"], "g_max": p["g_max"], "candidate_budget": p["candidate_budget"], "boundary_tie_tolerance": p["boundary_tie_tolerance"], "ranking": p["ranking"], "singleton_d_ctr": p["singleton_d_ctr"], "proposed_target_ids": p["proposed_target_ids"], "boundary_margin": p["boundary_margin"], "planned_support_count": p["planned_support_count"], "refusal_reason": p["refusal_reason"]}


def prediction_payload(row: dict) -> dict:
    p = row["prediction"]
    return {"schema_version": "frozen_mscc_prediction.v1", "protocol_hash": row["protocol_hash"], "proposal_hash": row["proposal"]["proposal_hash"], "discovery_fingerprint": row["discovery_fingerprint"], "source_atom_id": row["source_atom_id"], "search_status": p["search_status"], "identification": p["identification"], "multiplicity": p["multiplicity"], "supports": [(x["target_ids"], x["d_ctr"], x["d_mu"]) for x in p["supports"]], "planned_candidate_count": p["planned_candidate_count"], "evaluated_count": p["evaluated_count"], "complete_universe": p["complete_universe"], "unresolved_reason": p["unresolved_reason"]}


def validate_prediction(run: Path) -> tuple[list[dict], dict, list[dict]]:
    checks = []
    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise ValueError(name)
    closure = json.loads((run / "prediction_closure.json").read_text(encoding="utf-8"))
    check("sealed_identity", closure["state"] == "SEALED" and closure["run_id"] == run.name)
    check("bound_files", all(sha(run / name) == value for name, value in closure["files"].items()))
    code = json.loads((run / "code_hashes.json").read_text(encoding="utf-8"))
    check("code_aggregate", digest(code["files"]) == code["aggregate_sha256"] == closure["code_snapshot_hash"])
    check("source_snapshots", all(sha(run / row["snapshot"]) == row["sha256"] for row in code["files"]))
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    check("input_snapshots", all(sha(run / row["snapshot"]) == row["sha256"] for row in inputs.values()))
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    check("config_binding", config == json.loads((run / inputs["execution_config"]["snapshot"]).read_text(encoding="utf-8")) and config["protocol_sha256"] == closure["protocol_sha256"] == inputs["protocol"]["sha256"])
    predictor = next(row for row in code["files"] if row["snapshot"].endswith("scripts/run_m1_nip_d1_predict_v2.py"))
    tree = ast.parse((run / predictor["snapshot"]).read_text(encoding="utf-8"))
    check("no_static_truth_import", not any("nip_truth" in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))))
    rows = [json.loads(line) for line in (run / "predictions.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = len(config["families"]) * config["pairs_per_family"] * len(config["atom_caps"])
    check("row_count", len(rows) == closure["row_count"] == expected)
    check("paired_grid", len({(x["family_id"], x["pair_index"], x["atom_cap"]) for x in rows}) == expected)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["family_id"], row["pair_index"]), []).append(row)
    check("paired_seeds", all({tuple(sorted(x["seeds"].items())) for x in group}.__len__() == 1 and sorted(x["atom_cap"] for x in group) == config["atom_caps"] for group in grouped.values()))
    check("truth_closed", all(x["truth_opened"] is False for x in rows) and config["truth_opened_in_prediction"] is False)
    check("proposal_hashes", all(digest(proposal_payload(x)) == x["proposal"]["proposal_hash"] for x in rows))
    check("prediction_hashes", all(digest(prediction_payload(x)) == x["prediction"]["prediction_hash"] for x in rows))
    check("terminal_pass", json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "PASS")
    return rows, config, checks


def validate_score(prediction: Path, score: Path, prediction_rows: list[dict], config: dict) -> list[dict]:
    checks = []
    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise ValueError(name)
    required = {"manifest.json", "environment.json", "inputs.json", "code_hashes.json", "status.json", "stdout.log", "stderr.log", "scores.raw.jsonl", "scores.summary.json"}
    check("score_files", all((score / name).is_file() for name in required))
    manifest = json.loads((score / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((score / "scores.summary.json").read_text(encoding="utf-8"))
    code = json.loads((score / "code_hashes.json").read_text(encoding="utf-8"))
    check("score_provenance", manifest["prediction_closure_sha256"] == summary["prediction_closure_sha256"] == sha(prediction / "prediction_closure.json") and manifest["phase"] == config["phase"] and manifest["formal_d1_seed_consumed"] == config["formal_d1_seed_consumed"])
    check("score_hashes", summary["raw_sha256"] == sha(score / "scores.raw.jsonl") and code["scorer_sha256"] == summary["generation_script_sha256"] == sha(score / code["scorer_snapshot"]))
    scored = [json.loads(line) for line in (score / "scores.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    check("score_row_count", len(scored) == len(prediction_rows))
    indexed = {(x["family_id"], x["pair_index"], x["atom_cap"]): x for x in prediction_rows}
    truth_module = importlib.import_module("ccad.nip_truth")
    negative = {"N04_absent_target", "N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation", "N08_continuous_only_representation", "N12_mean_mismatch"}
    outcomes_ok = True
    for row in scored:
        pred = indexed[(row["family_id"], row["pair_index"], row["atom_cap"])]["prediction"]
        truth = truth_module.nip_truth(row["family_id"])
        supports = {tuple(x["target_ids"]) for x in pred["supports"]}
        exact = row["family_id"] not in negative and pred["identification"] == "FOUND" and pred["multiplicity"] == truth.multiplicity and supports == set(truth.minimum_supports)
        outcomes_ok &= row["positive_exact"] == exact and row["false_unique"] == (truth.multiplicity == "AMBIGUOUS" and pred["multiplicity"] == "UNIQUE") and row["false_native_positive"] == (row["family_id"] in negative and pred["identification"] == "FOUND")
    check("rescored_outcomes", outcomes_ok)
    by_cap = {}
    for cap in config["atom_caps"]:
        rows = [x for x in scored if x["atom_cap"] == cap]
        by_cap[str(cap)] = {"positive_exact_pairs": sum(x["positive_exact"] for x in rows), "false_unique_count": sum(x["false_unique"] for x in rows), "false_native_positive_count": sum(x["false_native_positive"] for x in rows), "budget_refusal_count": sum(x["budget_refusal"] for x in rows), "total_scored_supports": sum(x["scored_supports"] for x in rows)}
    check("aggregate_recomputed", summary["by_cap"] == by_cap)
    selected = min(config["atom_caps"], key=lambda cap: (-by_cap[str(cap)]["positive_exact_pairs"], by_cap[str(cap)]["false_unique_count"], by_cap[str(cap)]["false_native_positive_count"], by_cap[str(cap)]["budget_refusal_count"], by_cap[str(cap)]["total_scored_supports"], cap))
    check("selection_recomputed", summary["selected_cap"] == selected and summary["runtime_used_for_selection"] is False)
    check("score_terminal_pass", json.loads((score / "status.json").read_text(encoding="utf-8"))["status"] == "PASS")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--score-run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction = Path(args.prediction_run).resolve()
    rows, config, prediction_checks = validate_prediction(prediction)
    score_checks = validate_score(prediction, Path(args.score_run).resolve(), rows, config) if args.score_run else []
    result = {"status": "PASS", "prediction_run": prediction.name, "prediction_checks": prediction_checks, "score_checks": score_checks, "check_count": len(prediction_checks) + len(score_checks), "prediction_row_count": len(rows)}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
