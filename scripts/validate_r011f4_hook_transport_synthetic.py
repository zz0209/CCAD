"""Independent artifact and semantic validation for the C047 synthetic gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_family = {row["family"]: row for row in rows}
    checks = {
        "artifact_contract": validate_run_directory(run_dir).ok,
        "status_pass": status["status"] == summary["status"] == "PASS",
        "raw_hash": sha256(run_dir / "metrics.raw.jsonl") == summary["metrics_raw_sha256"],
        "family_order": [row["family"] for row in rows] == cfg["families"],
        "all_fixture_flags": len(rows) == len(cfg["families"]) and all(row["passed"] for row in rows),
        "rotation_recovery": by_family["rotation"]["bcc"] > .999 and by_family["rotation"]["normalized_residual"] < .002,
        "split_merge_recovery": by_family["split_merge"]["bcc"] > .999 and by_family["split_merge"]["normalized_residual"] < .002,
        "query_null_refusal": by_family["query_null_global_nuisance"]["decision"] == "UNRESOLVED_RELATION" and by_family["query_null_global_nuisance"]["reason"] == "HARD_NEGATIVE_CONTRAST",
        "raw_control_refusal": by_family["raw_control_rejection"]["decision"] == "UNRESOLVED_RELATION" and by_family["raw_control_rejection"]["reason"] == "RAW_OR_GLOBAL_CONTROL_NOT_BEATEN",
        "rank_deficiency_refusal": by_family["rank_deficient"]["status"] == "RANK_DEFICIENT" and by_family["rank_deficient"]["effective_rank"] < by_family["rank_deficient"]["requested_rank"],
        "summary_counts": summary["families_passed"] == summary["families_total"] == len(cfg["families"]),
        "audit_closed": cfg["audit_opened"] is False and cfg["real_screen_execution_enabled"] is False,
    }
    print(json.dumps({"run_id": run_dir.name, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks)}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
