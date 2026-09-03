from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from ccad.nip_synthetic_v3 import construction_certificate, generate_endpoint_observed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def aggregate(rows: list[dict[str, object]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    runner = Path(args.runner).resolve()
    required = (
        "manifest.json", "config.resolved.json", "environment.json", "inputs.json",
        "code_hashes.json", "status.json", "stdout.log", "stderr.log",
        "metrics.raw.jsonl", "metrics.summary.json",
    )
    checks: list[tuple[str, bool]] = [("required_files", all((run_dir / name).is_file() for name in required))]
    if checks[-1][1]:
        config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
        ledger = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
        code_files = ledger["files"]
        snapshots_valid = all(
            (run_dir / row["snapshot"]).is_file()
            and sha(run_dir / row["snapshot"]) == row["sha256"]
            and (run_dir / row["snapshot"]).stat().st_size == row["bytes"]
            for row in code_files
        )
        runner_rows = [row for row in code_files if str(row["snapshot"]).endswith("run_m1_nip_d0_v3.py")]
        snapshot_runner = run_dir / runner_rows[0]["snapshot"] if len(runner_rows) == 1 else None
        tree = ast.parse(snapshot_runner.read_text(encoding="utf-8")) if snapshot_runner else ast.parse("")
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        forbidden = {"truth", "causal_outcome", "identification", "minimum_supports", "planted_support", "supports", "target_ids"}
        serialized = set(summary) | {key for row in records for key in row}
        n11_rows = [row for row in records if row["family_id"] == "N11_downstream_cliff"]
        regenerated = []
        for row in n11_rows:
            instance = generate_endpoint_observed(
                row["family_id"], structural_seed=row["seeds"]["structural"],
                sample_seed=row["seeds"]["sample"], n=config["observations_per_pair"],
            )
            regenerated.append(construction_certificate(instance))
        endpoint_match = all(
            abs(row["n11_centered_residual"] - cert["n11_centered_residual"]) <= 1e-15
            and row["n11_endpoint"] == cert["n11_endpoint"]
            and row["n11_sample_mean_delta_norm"] == cert["n11_sample_mean_delta_norm"]
            for row, cert in zip(n11_rows, regenerated)
        )
        all_seeds = [seed for row in records for seed in row["seeds"].values()]
        checks.extend([
            ("manifest_closed", manifest["tracker_parent"] == "M1_NIP_protocol_v3" and not manifest["truth_opened"] and not manifest["audit_opened"] and manifest["protocol_deviations"] == []),
            ("source_snapshots_valid", snapshots_valid),
            ("code_aggregate_recomputed", aggregate(code_files) == ledger["aggregate_sha256"] == manifest["code_snapshot_hash"]),
            ("runner_binding", len(runner_rows) == 1 and sha(runner) == runner_rows[0]["sha256"] == summary["generation_script_sha256"]),
            ("runner_truth_free", not any("nip_truth" in name for name in imports)),
            ("forbidden_keys_absent", not (serialized & forbidden)),
            ("record_grid", len(records) == 60 and len(n11_rows) == 5 and all(sum(row["family_id"] == family for row in records) == 5 for family in config["families"])),
            ("fresh_distinct_seeds", len(all_seeds) == len(set(all_seeds)) == 240),
            ("closed_outputs", not summary["truth_opened"] and not summary["held_out_eval_opened"] and not summary["real_sae_audit_opened"]),
            ("endpoint_only_on_n11", all(row["endpoint_present"] == (row["family_id"] == "N11_downstream_cliff") for row in records)),
            ("endpoint_independently_regenerated", endpoint_match),
            ("n11_feasibility_margin", all(abs(row["n11_centered_residual"] - config["n11_centered_residual"]) <= 1e-12 and config["approximate_tau_ctr"] - row["n11_centered_residual"] >= config["n11_minimum_threshold_margin"] for row in n11_rows)),
            ("n11_cliff_gate", all(row["n11_endpoint"]["cliff_disagreement_rate"] == 1.0 and row["n11_endpoint"]["cliff_effect_rmse"] >= config["n11_minimum_cliff_effect_rmse"] and row["n11_endpoint"]["minimum_normalized_cliff_margin"] + 1e-12 >= config["n11_minimum_normalized_cliff_margin"] for row in n11_rows)),
            ("n11_smooth_control", all(row["n11_endpoint"]["smooth_effect_rmse"] <= config["n11_maximum_smooth_effect_rmse"] for row in n11_rows)),
            ("construction_gates", all(row["minimum_decoy_orthogonal_residual"] >= config["minimum_decoy_orthogonal_residual"] and row["maximum_decoy_orthogonality_error"] <= config["maximum_decoy_orthogonality_error"] and row["cap_contract_pass"] is not False for row in records)),
            ("budget_grid", all(row["target_shape"][1] == 20 and row["planned_support_count"] == 6195 and row["evaluated_count"] == 6195 for row in records)),
            ("raw_hash", sha(run_dir / "metrics.raw.jsonl") == summary["raw_sha256"]),
            ("run_pass", status["status"] == summary["status"] == "PASS" and summary["proposal_refusal_count"] == 0),
        ])
    result = {"status": "PASS" if all(value for _, value in checks) else "FAIL", "checks": [{"name": name, "pass": value} for name, value in checks]}
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
