"""Independent read-only validator for an R006-B1 run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.sae_quality import budget_stability_checks, ce_recovered  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    contract = validate_run_directory(run_dir)
    milestones = record["milestones"]
    compact = [
        {"fve": item["validation"]["fve"], "ce_recovered": item["validation"]["ce_recovered"], "alive_fraction": item["validation"]["alive_fraction"], "c_dec": item["c_dec"]}
        for item in milestones[-2:]
    ]
    recomputed_stability = budget_stability_checks(compact[0], compact[1], cfg["stability_thresholds"])
    checks = {
        "contract": contract.ok,
        "raw_hash": sha256(run_dir / "metrics.raw.jsonl") == summary["metrics_raw_sha256"],
        "summary_pass": summary["status"] == "PASS" and summary["stability_gate_pass"] is True,
        "all_recorded_hard_checks": all(record["hard_checks"].values()),
        "stability_exact": recomputed_stability == record["stability"],
        "milestone_tokens_exact": [item["train_tokens"] for item in milestones] == cfg["milestone_train_tokens"],
        "trajectory_hashes_exact": record["combined_input_hashes"] == record["expected_input_hashes"],
        "ce_formula_exact": all(
            abs(ce_recovered(item["validation"]["ce"]["clean"], item["validation"]["ce"]["reconstruction"], item["validation"]["ce"]["zero"]) - item["validation"]["ce_recovered"]) < 1e-12
            for item in milestones
        ),
        "checkpoints_present": all((run_dir / "checkpoints" / f"step_{item['global_step']:04d}" / "state.pt").is_file() for item in milestones),
        "safe_saes_present": all((run_dir / "saes" / f"step_{item['global_step']:04d}" / "sae.safetensors").is_file() for item in milestones),
        "audit_closed": cfg["audit_opened"] is False,
    }
    print(json.dumps({"ok": all(checks.values()), "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
