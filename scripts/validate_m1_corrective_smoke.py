from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def prediction_hash(payload: dict) -> str:
    canonical = {
        "schema_version": payload["schema_version"],
        "proposal_source": payload["proposal_source"],
        "proposal_hash": payload["proposal_hash"],
        "discovery_fingerprint": payload["discovery_fingerprint"],
        "search_status": payload["search_status"],
        "candidate_family": payload["candidate_family"],
        "predictions": [(x["left_ids"], x["right_ids"], x["normalized_residual"]) for x in payload["predictions"]],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    required = ["resolved_config.json", "parent_protocol.resolved.json", "code_hashes.json", "environment.json", "status.json", "diagnostic_records.jsonl", "discovery_predictions.jsonl", "phase_ledger.json", "held_out_evaluations.jsonl", "summary.json", "stdout.log", "stderr.log"]
    checks = [("required_files", all((run_dir / name).is_file() for name in required))]
    if not checks[-1][1]:
        result = {"status": "FAIL", "checks": checks}
    else:
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        full = json.loads((run_dir / "parent_protocol.resolved.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        ledger = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
        diagnostics = [json.loads(x) for x in (run_dir / "diagnostic_records.jsonl").read_text(encoding="utf-8").splitlines() if x]
        frozen = [json.loads(x) for x in (run_dir / "discovery_predictions.jsonl").read_text(encoding="utf-8").splitlines() if x]
        held_out = [json.loads(x) for x in (run_dir / "held_out_evaluations.jsonl").read_text(encoding="utf-8").splitlines() if x]
        phases = json.loads((run_dir / "phase_ledger.json").read_text(encoding="utf-8"))
        expected_lanes = set(full["lanes"])
        checks.extend([
            ("run_pass", status["status"] == summary["status"] == "PASS"),
            ("information_closed_at_start", not config["synthetic_eval_opened_at_start"] and not config["real_sae_audit_opened"]),
            ("source_snapshots", all((run_dir / x["snapshot"]).is_file() and sha256(run_dir / x["snapshot"]) == x["sha256"] for x in ledger["files"])),
            ("all_families", set(summary["families_covered"]) == set(config["families"]) and len(diagnostics) == summary["diagnostic_record_count"]),
            ("metric_surface", summary["metric_surface_error_count"] == 0 and all(not x["metric_surface_errors"] for x in diagnostics)),
            ("six_distinct_seeds", all(set(full["required_seed_fields"]).issubset(x["seed_provenance"]) and len({x["seed_provenance"][name] for name in full["required_seed_fields"]}) == 6 for x in diagnostics) and all(set(full["required_seed_fields"]).issubset(x["seed_provenance"]) and len({x["seed_provenance"][name] for name in full["required_seed_fields"]}) == 6 for x in frozen)),
            ("freeze_before_eval", datetime.fromisoformat(phases["discovery_predictions_written_utc"]) <= datetime.fromisoformat(phases["held_out_eval_opened_utc"])),
            ("prediction_hashes", all(x["frozen"]["prediction_hash"] == prediction_hash(x["frozen"]) for x in frozen)),
            ("all_lanes_each_family", all({x["lane"] for x in frozen if x["family_id"] == family} == expected_lanes for family in config["unknown_support_families"])),
            ("held_out_binding", len(frozen) == len(held_out) == summary["discovery_prediction_count"] and {(x["family_id"], x["pair_index"], x["lane"], x["top_k"], x["prediction_hash"]) for x in held_out} == {(x["family_id"], x["pair_index"], x["lane"], x["top_k"], x["frozen"]["prediction_hash"]) for x in frozen}),
            ("primary_gate", summary["primary_unknown_support_count"] == len(config["unknown_support_families"]) * config["seed_pair_count"] and summary["primary_unknown_support_pass"] and all(x["precision"] == x["recall"] == x["f1"] == 1.0 and x["failure_attribution"] is None for x in held_out if x["lane"] == full["primary_proposal_lane"] and x["top_k"] == full["primary_top_k"])),
            ("raw_hashes", summary["diagnostic_sha256"] == sha256(run_dir / "diagnostic_records.jsonl") and summary["discovery_predictions_sha256"] == sha256(run_dir / "discovery_predictions.jsonl") and summary["held_out_evaluations_sha256"] == sha256(run_dir / "held_out_evaluations.jsonl")),
        ])
        result = {"status": "PASS" if all(ok for _, ok in checks) else "FAIL", "checks": [{"name": name, "passed": bool(ok)} for name, ok in checks], "checks_passed": sum(bool(ok) for _, ok in checks), "checks_total": len(checks)}
    output = Path(args.output).resolve() if args.output else run_dir / "validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
