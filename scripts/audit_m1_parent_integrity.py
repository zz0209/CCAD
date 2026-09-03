from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_run(project_root: Path, relative: str, required: set[str]) -> dict[str, Any]:
    run_dir = project_root / relative
    present = {path.name for path in run_dir.iterdir() if path.is_file()} if run_dir.is_dir() else set()
    row: dict[str, Any] = {"run": relative, "exists": run_dir.is_dir(), "missing_files": sorted(required - present)}
    if not run_dir.is_dir() or row["missing_files"]:
        row["passes"] = False
        return row

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    contract = json.loads((run_dir / "contract.validation.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    code_hashes = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
    config_path = run_dir / "config.resolved.json"
    raw_path = run_dir / "metrics.raw.jsonl"
    raw_records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    current_code = []
    for item in code_hashes["files"]:
        source_path = project_root / item["path"]
        current_hash = sha256_file(source_path) if source_path.is_file() else None
        current_code.append({"path": item["path"], "recorded_sha256": item["sha256"],
                             "current_sha256": current_hash, "matches_current": current_hash == item["sha256"]})
    row.update({
        "status_pass": status.get("status") == "PASS",
        "contract_ok": contract.get("ok") is True,
        "summary_pass": summary.get("status") == "PASS",
        "checks_complete": summary.get("checks_passed") == summary.get("checks_total"),
        "raw_hash_matches": sha256_file(raw_path) == summary.get("metrics_raw_sha256"),
        "raw_record_count_matches": len(raw_records) == summary.get("records"),
        "config_hash_matches": sha256_file(config_path) == manifest.get("config_hash"),
        "code_aggregate_bound": code_hashes.get("aggregate_sha256") == manifest.get("code_snapshot_hash"),
        "families": summary.get("families_covered", []),
        "records": len(raw_records),
        "checks_passed": summary.get("checks_passed"),
        "checks_total": summary.get("checks_total"),
        "current_code": current_code,
        "current_code_match_count": sum(item["matches_current"] for item in current_code),
        "current_code_file_count": len(current_code),
        "historical_source_snapshot_present": any(path.name.startswith("source_snapshot") for path in run_dir.iterdir()),
    })
    hard = ["status_pass", "contract_ok", "summary_pass", "checks_complete", "raw_hash_matches",
            "raw_record_count_matches", "config_hash_matches", "code_aggregate_bound"]
    row["passes"] = all(row[key] for key in hard)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["audit_opened"]:
        raise ValueError("M1 integrity audit must not open real audit data")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    required = set(config["required_run_files"])
    run_rows = []
    parent_rows = []
    for parent, relative_runs in config["formal_runs"].items():
        inspected = [inspect_run(project_root, relative, required) for relative in relative_runs]
        run_rows.extend(inspected)
        observed = sorted({family for row in inspected for family in row.get("families", [])})
        expected = sorted(config["expected_families"][parent])
        parent_rows.append({"parent": parent, "runs": relative_runs, "observed_families": observed,
                            "expected_families": expected, "family_coverage_exact": observed == expected,
                            "all_runs_pass": all(row["passes"] for row in inspected),
                            "records": sum(row.get("records", 0) for row in inspected),
                            "checks_passed": sum(row.get("checks_passed", 0) for row in inspected),
                            "checks_total": sum(row.get("checks_total", 0) for row in inspected)})
    deterministic_pass = all(row["passes"] for row in run_rows) and all(
        row["family_coverage_exact"] and row["all_runs_pass"] and row["checks_passed"] == row["checks_total"]
        for row in parent_rows
    )
    drifted = sorted({item["path"] for row in run_rows for item in row.get("current_code", []) if not item["matches_current"]})
    snapshot_count = sum(row.get("historical_source_snapshot_present", False) for row in run_rows)
    summary = {"schema_version": 1, "run_status": "PASS" if deterministic_pass else "FAIL",
               "deterministic_parent_gate": "PASS" if deterministic_pass else "FAIL",
               "semantic_review_status": "PENDING_FRESH_SAME_FAMILY_REVIEW",
               "parents": parent_rows, "formal_run_count": len(run_rows),
               "current_source_drift_paths": drifted,
               "historical_source_snapshot_run_count": snapshot_count,
               "reproducibility_warning": "Recorded per-file hashes detect drift, but no run-local historical source snapshot was found." if snapshot_count < len(run_rows) else None,
               "claim_scope": config["claim_scope"], "audit_opened": False}
    finished = dt.datetime.now(dt.timezone.utc)
    resolved = dict(config)
    resolved.update({"run_id": args.output_dir.name, "config_path": str(args.config.resolve()),
                     "output_dir": str(args.output_dir.resolve())})
    environment = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "platform": platform.platform(), "workspace_git_repository": False}
    status = {"status": summary["run_status"], "semantic_review_status": summary["semantic_review_status"],
              "protocol_deviation": False}
    log = {"started_at": started.isoformat(), "finished_at": finished.isoformat(),
           "events": [f"audited {len(run_rows)} formal runs", f"deterministic gate {summary['deterministic_parent_gate']}"]}
    for name, payload in {"resolved_config.json": resolved, "environment.json": environment,
                          "status.json": status, "summary.json": summary,
                          "formal_run_audit.json": run_rows, "run_log.json": log}.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"config_sha256": sha256_file(args.config), "runner_sha256": sha256_file(Path(__file__))}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if not deterministic_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
