from __future__ import annotations

import argparse
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
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    required = {
        "resolved_config.json", "environment.json", "status.json", "summary.json",
        "formal_run_audit.json", "run_log.json", "provenance.json",
    }
    checks: dict[str, bool] = {
        "required_outputs": run_dir.is_dir() and required <= {p.name for p in run_dir.iterdir()},
    }
    if not checks["required_outputs"]:
        print(json.dumps({"status": "FAIL", "checks": checks}, sort_keys=True))
        raise SystemExit(1)

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "formal_run_audit.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    config_path = Path(resolved["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    observed_by_parent: dict[str, set[str]] = {parent: set() for parent in config["formal_runs"]}
    listed = {row["run"]: row for row in rows}
    all_raw_recomputed = True
    all_configs_recomputed = True
    all_rows_pass = True
    for parent, relatives in config["formal_runs"].items():
        for relative in relatives:
            row = listed.get(relative)
            if row is None:
                all_rows_pass = False
                continue
            observed_by_parent[parent].update(row.get("families", []))
            source = root / relative
            source_summary = json.loads((source / "metrics.summary.json").read_text(encoding="utf-8"))
            source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            all_raw_recomputed &= sha256_file(source / "metrics.raw.jsonl") == source_summary.get("metrics_raw_sha256")
            all_configs_recomputed &= sha256_file(source / "config.resolved.json") == source_manifest.get("config_hash")
            all_rows_pass &= row.get("passes") is True

    exact_families = all(
        sorted(observed_by_parent[parent]) == sorted(config["expected_families"][parent])
        for parent in config["formal_runs"]
    )
    checks.update({
        "no_real_audit_opened": resolved.get("audit_opened") is False and summary.get("audit_opened") is False,
        "formal_run_set_exact": set(listed) == {r for runs in config["formal_runs"].values() for r in runs},
        "raw_hashes_recomputed": all_raw_recomputed,
        "config_hashes_recomputed": all_configs_recomputed,
        "all_rows_pass": all_rows_pass,
        "family_unions_exact": exact_families,
        "parent_gate_consistent": summary.get("deterministic_parent_gate") == "PASS",
        "status_consistent": status.get("status") == "PASS" and summary.get("run_status") == "PASS",
        "semantic_review_not_fabricated": summary.get("semantic_review_status") == "PENDING_FRESH_SAME_FAMILY_REVIEW",
        "config_provenance": sha256_file(config_path) == provenance.get("config_sha256"),
        "runner_provenance": sha256_file(root / "scripts" / "audit_m1_parent_integrity.py") == provenance.get("runner_sha256"),
    })
    passed = sum(checks.values())
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks_passed": passed,
              "checks_total": len(checks), "checks": checks}
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
