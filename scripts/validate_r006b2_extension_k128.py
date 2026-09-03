"""Independent read-only validator for the R006-B2 k=128 extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.sae_quality import ce_recovered, select_k_extension  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    contract = validate_run_directory(run_dir)
    selector_inputs = {int(k): values for k, values in record["selector_inputs"].items()}
    recomputed = select_k_extension(selector_inputs, cfg["selection_fve_margin"], cfg["selection_ce_recovered_margin"])
    candidate = record["new_candidate"]
    ce = candidate["validation"]["ce"]
    checks = {
        "contract": contract.ok,
        "raw_hash": sha256(run_dir / "metrics.raw.jsonl") == summary["metrics_raw_sha256"],
        "summary_pass": summary["status"] == "PASS",
        "all_suite_checks": all(record["suite_checks"].values()),
        "all_reuse_checks": all(record["reuse_checks"].values()),
        "all_candidate_checks": all(candidate["checks"].values()),
        "selector_exact": recomputed == record["selection"],
        "summary_selection_exact": summary["selection_decision"] == recomputed["decision"] and summary["shortlist_k"] == recomputed["shortlist_k"],
        "candidate_set_exact": sorted(selector_inputs) == cfg["candidate_k"],
        "ce_formula_exact": abs(ce_recovered(ce["clean"], ce["reconstruction"], ce["zero"]) - candidate["validation"]["ce_recovered"]) < 1e-12,
        "input_trace_exact": candidate["train_input_hashes"] == record["expected_input_hashes"],
        "checkpoint_present": (run_dir / "checkpoints/k_128/state.pt").is_file(),
        "safe_sae_present": (run_dir / "saes/k_128/sae.safetensors").is_file(),
        "audit_closed": cfg["audit_opened"] is False,
    }
    print(json.dumps({"ok": all(checks.values()), "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
