"""Static pre-generation audit for the v3 D2 selection freeze and config."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze_path = ROOT / config["selection_freeze_path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks = []

    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise ValueError(name)

    prediction = ROOT / "runs" / freeze["d1_prediction_run"]
    score = ROOT / "runs" / freeze["d1_score_run"]
    check("freeze_hash", sha(freeze_path) == config["selection_freeze_sha256"])
    check("protocol_diagnostic_bindings", sha(ROOT / config["protocol_path"]) == config["protocol_sha256"] == freeze["protocol_sha256"] and sha(ROOT / config["diagnostic_config_path"]) == config["diagnostic_config_sha256"] == freeze["diagnostic_config_sha256"])
    check("d1_prediction_bindings", sha(prediction / "prediction_closure.json") == freeze["d1_prediction_closure_sha256"] and sha(prediction / "prelabel_validation.json") == freeze["d1_prelabel_validation_sha256"])
    check("d1_score_bindings", sha(score / "scores.raw.jsonl") == freeze["d1_score_raw_sha256"] and sha(score / "scores.summary.json") == freeze["d1_score_summary_sha256"] and sha(score / "independent_validation.json") == freeze["d1_score_validation_sha256"])
    prediction_validation = json.loads((prediction / "prelabel_validation.json").read_text(encoding="utf-8"))
    score_validation = json.loads((score / "independent_validation.json").read_text(encoding="utf-8"))
    summary = json.loads((score / "scores.summary.json").read_text(encoding="utf-8"))
    check("d1_validators_pass", prediction_validation["status"] == score_validation["status"] == "PASS" and prediction_validation["check_count"] == 20 and score_validation["check_count"] == 10)
    check("selection_reproduced", summary["selected_cap"] == freeze["selected_atom_cap"] == config["atom_caps"][0] == 20 and summary["by_cap"]["20"] == {"positive_exact_pairs": 140, "false_unique_count": 0, "false_native_positive_count": 0, "budget_refusal_count": 0, "total_scored_supports": 1486800})
    sources = {"d2_predictor_sha256": "scripts/run_m1_nip_d1_predict_v3.py", "prelabel_validator_sha256": "scripts/validate_m1_nip_d1_prediction_v3.py", "scorer_sha256": "scripts/score_m1_nip_d1_v3.py", "score_validator_sha256": "scripts/validate_m1_nip_d1_score_v3.py"}
    check("source_bindings", all(sha(ROOT / path) == freeze[key] for key, path in sources.items()))
    check("fresh_d2_contract", config["phase"] == freeze["d2_seed_namespace"] == "D2" and config["formal_d2_seed_consumed"] is True and config["pairs_per_family"] == freeze["d2_pairs_per_family"] == 20 and config["expected_prediction_rows"] == 240 and config["atom_caps"] == [20])
    check("information_boundary", freeze["d2_seeds_generated"] is False and not config["truth_opened_in_prediction"] and not config["held_out_eval_opened"] and not config["real_sae_audit_opened"] and not freeze["held_out_eval_opened"] and not freeze["real_sae_audit_opened"])
    check("no_existing_d2_prediction", not any((ROOT / "runs").glob("M1_NIP_D2_predict_v3_formal_*")))
    result = {"status": "PASS", "check_count": len(checks), "config_sha256": sha(config_path), "selection_freeze_sha256": sha(freeze_path), "checks": checks, "d2_seeds_generated": False}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
