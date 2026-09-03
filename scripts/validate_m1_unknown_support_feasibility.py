from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    checks = []
    required = ["resolved_config.json", "environment.json", "code_hashes.json", "raw_records.jsonl", "summary.json", "status.json", "stdout.log", "stderr.log"]
    checks.append(("required_files", all((run_dir / name).is_file() for name in required)))
    if not checks[-1][1]:
        print(json.dumps({"status": "FAIL", "checks": checks}))
        return 1
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "raw_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    checks.extend([
        ("run_pass", status["status"] == "PASS" and summary["status"] == "PASS"),
        ("closed_information", not config["audit_opened"] and not config["held_out_eval_loaded"] and not config["planted_labels_loaded"]),
        ("record_count", len(records) == len(config["families"]) * config["seed_pair_count"] == summary["record_count"]),
        ("no_forbidden_outputs", all("planted_hyperedges" not in record and "recovery" not in record for record in records)),
        ("record_closure", all(not record["held_out_eval_loaded"] and not record["planted_labels_loaded"] for record in records)),
        ("lane_grid", all(len(record["lanes"]) == (len(config["lanes"]) - (1 if "LI15-SPECTRAL" in config["lanes"] else 0)) * len(config["top_k_grid"]) + (1 if "LI15-SPECTRAL" in config["lanes"] else 0) for record in records)),
        ("raw_hash", sha256(run_dir / "raw_records.jsonl") == summary["raw_records_sha256"]),
    ])
    observed_max = max(
        int(count) for record in records for lane in record["lanes"]
        for count in lane["candidate_counts_by_max_group_size"].values()
    )
    checks.append(("budget_recomputed", observed_max == summary["candidate_common_budget_candidate"]))
    ledger = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
    checks.append(("source_snapshots", all((run_dir / item["snapshot"]).is_file() and sha256(run_dir / item["snapshot"]) == item["sha256"] for item in ledger["files"])))
    result = {"status": "PASS" if all(value for _, value in checks) else "FAIL", "checks": [{"name": name, "pass": value} for name, value in checks]}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
