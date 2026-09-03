"""Independent pre-label validator for sealed v3 prediction artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from ccad.nip_diagnostics_v3 import freeze_centered_only_candidate
from ccad.nip_synthetic import observed_kernels
from ccad.nip_synthetic_v3 import generate_endpoint_observed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def proposal_payload(row: dict) -> dict:
    proposal = row["proposal"]
    return {
        "schema_version": "source_conditioned_proposal.v1", "source_atom_id": proposal["source_atom_id"],
        "atom_cap": proposal["atom_cap"], "g_max": proposal["g_max"],
        "candidate_budget": proposal["candidate_budget"],
        "boundary_tie_tolerance": proposal["boundary_tie_tolerance"], "ranking": proposal["ranking"],
        "singleton_d_ctr": proposal["singleton_d_ctr"], "proposed_target_ids": proposal["proposed_target_ids"],
        "boundary_margin": proposal["boundary_margin"], "planned_support_count": proposal["planned_support_count"],
        "refusal_reason": proposal["refusal_reason"],
    }


def prediction_payload(row: dict) -> dict:
    prediction = row["prediction"]
    return {
        "schema_version": "frozen_mscc_prediction.v1", "protocol_hash": row["protocol_hash"],
        "proposal_hash": row["proposal"]["proposal_hash"], "discovery_fingerprint": row["discovery_fingerprint"],
        "source_atom_id": row["source_atom_id"], "search_status": prediction["search_status"],
        "identification": prediction["identification"], "multiplicity": prediction["multiplicity"],
        "supports": [(item["target_ids"], item["d_ctr"], item["d_mu"]) for item in prediction["supports"]],
        "planned_candidate_count": prediction["planned_candidate_count"],
        "evaluated_count": prediction["evaluated_count"], "complete_universe": prediction["complete_universe"],
        "unresolved_reason": prediction["unresolved_reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.prediction_run).resolve()
    checks: list[dict[str, object]] = []

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
    check("input_keyset", set(inputs) == {"protocol", "diagnostic_config", "execution_config"})
    check("input_snapshots", all(sha(run / row["snapshot"]) == row["sha256"] for row in inputs.values()))
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    snap_config = json.loads((run / inputs["execution_config"]["snapshot"]).read_text(encoding="utf-8"))
    check("config_binding", config == snap_config and config["protocol_sha256"] == closure["protocol_sha256"] == inputs["protocol"]["sha256"])
    check("diagnostic_binding", config["diagnostic_config_sha256"] == closure["diagnostic_config_sha256"] == inputs["diagnostic_config"]["sha256"])
    predictor = next(row for row in code["files"] if row["snapshot"].endswith("scripts/run_m1_nip_d1_predict_v3.py"))
    tree = ast.parse((run / predictor["snapshot"]).read_text(encoding="utf-8"))
    imports = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("no_static_truth_or_outcome_import", not any("nip_truth" in item for item in imports))
    rows = [json.loads(line) for line in (run / "predictions.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = len(config["families"]) * config["pairs_per_family"] * len(config["atom_caps"])
    check("row_count", len(rows) == closure["row_count"] == config["expected_prediction_rows"] == expected)
    check("paired_grid", len({(row["family_id"], row["pair_index"], row["atom_cap"]) for row in rows}) == expected)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["family_id"], row["pair_index"]), []).append(row)
    check("paired_seeds", all(len({tuple(sorted(row["seeds"].items())) for row in group}) == 1 and sorted(row["atom_cap"] for row in group) == config["atom_caps"] for group in grouped.values()))
    check("fresh_seed_values", len({value for group in grouped.values() for value in group[0]["seeds"].values()}) == len(grouped) * 4)
    check("truth_closed", not config["truth_opened_in_prediction"] and all(row["truth_opened"] is False for row in rows))
    check("v3_contract", all(row["schema_version"] == "m1_nip_prediction.v3" and row["protocol_hash"] == config["protocol_sha256"] and row["diagnostic_config_hash"] == config["diagnostic_config_sha256"] for row in rows))
    check("proposal_hashes", all(digest(proposal_payload(row)) == row["proposal"]["proposal_hash"] for row in rows))
    check("prediction_hashes", all(digest(prediction_payload(row)) == row["prediction"]["prediction_hash"] for row in rows))
    candidates_ok = True
    for group in grouped.values():
        seed_row = group[0]
        observed = generate_endpoint_observed(
            seed_row["family_id"], structural_seed=seed_row["seeds"]["structural"],
            sample_seed=seed_row["seeds"]["sample"], n=config["observations_per_pair"],
        )
        k_ss, k_st, k_tt = observed_kernels(observed)
        for row in group:
            expected_candidate = freeze_centered_only_candidate(
                k_ss, k_st, k_tt, source_atom_id=row["source_atom_id"],
                proposed_target_ids=tuple(row["proposal"]["proposed_target_ids"]), g_max=config["g_max"],
                epsilon=config["epsilon"], candidate_budget=config["candidate_budget"],
            )
            observed_candidate = row["diagnostic_candidate"]
            candidates_ok &= (
                observed_candidate["target_ids"] == list(expected_candidate.target_ids)
                and observed_candidate["d_ctr"] == expected_candidate.d_ctr
                and observed_candidate["evaluated_count"] == expected_candidate.evaluated_count
                and observed_candidate["candidate_hash"] == expected_candidate.candidate_hash
            )
    check("centered_candidates_regenerated", candidates_ok)
    check("n11_approximate_observable", all(any(item["target_ids"] == [0] and abs(item["d_ctr"] - 0.01) <= 1e-12 for item in row["prediction"]["supports"]) for row in rows if row["family_id"] == "N11_downstream_cliff" and row["atom_cap"] == 20))
    check("terminal_pass", json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "PASS")
    result = {"status": "PASS", "prediction_run": run.name, "check_count": len(checks), "prediction_row_count": len(rows), "checks": checks}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
