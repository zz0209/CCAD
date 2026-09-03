from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    required = {
        "audit_log.json",
        "environment.json",
        "resolved_config.json",
        "status.json",
        "summary.json",
        "task_provenance_manifest.csv",
        "task_provenance_manifest.json",
    }
    present = {path.name for path in args.run_dir.iterdir() if path.is_file()}
    checks: dict[str, bool] = {"required_files": required <= present}

    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    config = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.run_dir / "task_provenance_manifest.json").read_text(encoding="utf-8")
    )
    verified = json.loads(Path(config["verified_config"]).read_text(encoding="utf-8"))
    master_csv = Path(config["source_root"]) / "raw_data" / "probing_datasets_MASTER.csv"
    with master_csv.open(encoding="utf-8-sig", newline="") as handle:
        binary_rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("Data type") or "").strip() == "Binary Classification"
        ]

    task_ids = [entry["task_id"] for entry in manifest]
    verified_ids = set(verified["verified_tasks"])
    checks.update(
        {
            "unique_task_ids": len(task_ids) == len(set(task_ids)) == 113,
            "source_binary_count": len(binary_rows) == 113,
            "all_raw_present": all(entry["raw_present"] for entry in manifest),
            "verified_count": sum(entry["license_status"] == "verified" for entry in manifest)
            == len(verified_ids),
            "verified_subset": verified_ids <= set(task_ids),
            "master_hash": sha256_file(master_csv) == summary["inputs"]["master_csv_sha256"],
            "config_hash": sha256_file(Path(config["verified_config"]))
            == summary["inputs"]["verified_config_sha256"],
            "semantic_fail_closed": status["run_status"] == "PASS"
            and status["b3a_gate"] == "NOT_PASSED"
            and summary["semantic_outcome"] == "INSUFFICIENT_LICENSE_COVERAGE",
            "no_audit_or_probe": not config["audit_opened"]
            and not config["probe_activations_computed"],
        }
    )
    report = {
        "validator_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manifest_sha256": sha256_file(
            args.run_dir / "task_provenance_manifest.json"
        ),
    }
    (args.run_dir / "validator_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if report["validator_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
