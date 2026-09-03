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
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    required = {"resolved_config.json", "environment.json", "status.json", "summary.json",
                "simulation_results.json", "run_log.json", "provenance.json"}
    checks = {"required_files": required <= {p.name for p in args.run_dir.iterdir() if p.is_file()}}
    config = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((args.run_dir / "simulation_results.json").read_text(encoding="utf-8"))
    provenance = json.loads((args.run_dir / "provenance.json").read_text(encoding="utf-8"))
    expected_cells = len(config["family_counts"]) * len(config["tasks_per_family"]) * len(config["icc_values"]) * len(config["utility_profiles"])
    recomputed_gate = {str(f): all(r["passes"] for r in rows if r["families"] == f) for f in config["family_counts"]}
    qualified = [int(k) for k, v in recomputed_gate.items() if v]
    expected_min = min(qualified) if qualified else None
    checks.update({
        "score_free": not config["audit_opened"] and not config["sae_scores_loaded"] and not config["probe_activations_computed"],
        "cell_count": len(rows) == expected_cells == summary["scenario_count"],
        "five_thousand_replicates": config["n_simulations"] >= 5000,
        "gate_recomputed": recomputed_gate == summary["family_gate"] and expected_min == summary["minimum_qualified_families"],
        "semantic_status": status["run_status"] == "PASS" and status["b3a_gate"] == "NOT_PASSED" and summary["b3a_gate"] == "NOT_PASSED",
        "config_hash": sha256_file(Path(config["config_path"])) == provenance["config_sha256"],
        "all_metrics_bounded": all(0.0 <= metric[key] <= 1.0 for row in rows for metric in row["metrics"].values() for key in ("estimate", "mc_ci_low", "mc_ci_high", "mc_half_width")),
        "naive_is_diagnostic_only": summary["naive_diagnostic"]["minimum_coverage"] < 0.93 or summary["naive_diagnostic"]["maximum_null_false_positive"] > 0.07,
    })
    report = {"validator_status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "results_sha256": sha256_file(args.run_dir / "simulation_results.json")}
    (args.run_dir / "validator_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["validator_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
