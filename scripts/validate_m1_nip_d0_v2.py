from __future__ import annotations

import argparse, ast, hashlib, json
from pathlib import Path


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
    run_dir, runner = Path(args.run_dir).resolve(), Path(args.runner).resolve()
    required = ("manifest.json", "config.resolved.json", "environment.json", "inputs.json", "code_hashes.json", "status.json", "stdout.log", "stderr.log", "metrics.raw.jsonl", "metrics.summary.json")
    checks = [("required_files", all((run_dir / name).is_file() for name in required))]
    if checks[-1][1]:
        config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        ledger = json.loads((run_dir / "code_hashes.json").read_text(encoding="utf-8"))
        inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (run_dir / "metrics.raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
        code_files = ledger.get("files", [])
        runner_rows = [row for row in code_files if Path(str(row.get("snapshot", ""))).as_posix().endswith("scripts/run_m1_nip_d0_v2.py")]
        snapshot_runner = run_dir / runner_rows[0]["snapshot"] if len(runner_rows) == 1 else None
        tree = ast.parse(snapshot_runner.read_text(encoding="utf-8")) if snapshot_runner and snapshot_runner.is_file() else ast.parse("")
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        imported_names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names]
        forbidden_keys = {"minimum_supports", "identification", "multiplicity", "causal_outcome", "recovery", "planted_support", "target_ids", "supports"}
        serialized_keys = set(summary) | {key for record in records for key in record}
        required_manifest = {"artifact_schema_version", "run_id", "tracker_parent", "purpose", "milestone", "evidence_level", "started_utc", "started_local", "trigger", "operator", "project_root", "config_hash", "code_snapshot_hash", "protocol_sha256", "execution_config_sha256", "git_available", "model_id", "model_revision", "dataset_id", "dataset_revision", "tokenizer_id", "sae_framework", "device", "resource_lease", "seed_fields", "seed_derivation", "phase_seed_manifest_status", "split", "statistical_unit", "artifact_schema", "protocol_deviations"}
        snapshots_valid = bool(code_files) and all((run_dir / row["snapshot"]).is_file() and sha(run_dir / row["snapshot"]) == row["sha256"] and (run_dir / row["snapshot"]).stat().st_size == row["bytes"] for row in code_files)
        inputs_valid = set(inputs) == {"protocol", "execution_config"} and all((run_dir / row["snapshot"]).is_file() and sha(run_dir / row["snapshot"]) == row["sha256"] and (run_dir / row["snapshot"]).stat().st_size == row["bytes"] and all(key in row for key in ("path", "source", "access", "license")) for row in inputs.values())
        runner_binding = len(runner_rows) == 1 and runner.is_file() and sha(runner) == runner_rows[0]["sha256"] == summary.get("generation_script_sha256")
        pressure = config.get("positive_family_first_sufficient_cap", {})
        cap_contract = all(
            (row["declared_first_cap"] == pressure[row["family_id"]] and row["observed_first_cap"] == pressure[row["family_id"]] and row["cap_contract_pass"] is True)
            if row["family_id"] in pressure
            else (row["declared_first_cap"] is None and row["observed_first_cap"] is None and row["cap_contract_pass"] is None)
            for row in records
        )
        all_seed_values = [value for row in records for value in row["seeds"].values()]
        checks += [
            ("manifest_complete", required_manifest <= set(manifest)),
            ("manifest_closed", manifest.get("run_id") == run_dir.name and manifest.get("tracker_parent") == "M1_NIP_protocol_v2" and manifest.get("statistical_unit") == "seed_pair" and manifest.get("audit_opened") is False and manifest.get("protocol_deviations") == []),
            ("source_snapshots_valid", snapshots_valid),
            ("code_aggregate_recomputed", aggregate(code_files) == ledger.get("aggregate_sha256") == manifest.get("code_snapshot_hash")),
            ("input_snapshots_valid", inputs_valid),
            ("config_binding", sha(run_dir / "config.resolved.json") == manifest.get("config_hash") and inputs.get("execution_config", {}).get("sha256") == manifest.get("execution_config_sha256")),
            ("protocol_binding", inputs.get("protocol", {}).get("sha256") == config.get("protocol_sha256") == manifest.get("protocol_sha256")),
            ("runner_binding", runner_binding),
            ("runner_does_not_import_truth_or_private_construction_ids", not any("nip_truth" in name for name in imports) and "_CONSTRUCTION_REQUIRED_IDS" not in imported_names),
            ("closed_config", config["phase"] == "D0" and not config["truth_opened"] and not config["held_out_eval_opened"] and not config["real_sae_audit_opened"]),
            ("closed_outputs", not summary["truth_opened"] and not summary["held_out_eval_opened"] and not summary["real_sae_audit_opened"]),
            ("forbidden_keys_absent", not (serialized_keys & forbidden_keys)),
            ("record_grid", len(records) == 60 == summary["record_count"] and len({row["family_id"] for row in records}) == 12 and all(sum(row["family_id"] == family for row in records) == 5 for family in {row["family_id"] for row in records})),
            ("twenty_atom_complete_budget", all(row["target_shape"][1] == 20 and row["full_dictionary_comparisons"] == 20 and row["planned_support_count"] == 6195 and row["evaluated_count"] == 6195 for row in records)),
            ("construction_residual_gate", min(row["minimum_decoy_orthogonal_residual"] for row in records) >= config["minimum_decoy_orthogonal_residual"] and max(row["maximum_decoy_orthogonality_error"] for row in records) <= config["maximum_decoy_orthogonality_error"]),
            ("cap_contract", cap_contract and summary["cap_contract_failure_count"] == 0),
            ("distinct_seeds_within_and_across_pairs", all(len(set(row["seeds"].values())) == 4 for row in records) and len(set(all_seed_values)) == 240),
            ("prediction_hashes_unique_and_certificate_hashes_well_formed", len({row["prediction_hash"] for row in records}) == 60 and all(len(row["construction_certificate_hash"]) == 64 and set(row["construction_certificate_hash"]) <= set("0123456789ABCDEF") for row in records)),
            ("raw_hash", sha(run_dir / "metrics.raw.jsonl") == summary["raw_sha256"]),
            ("run_pass", status["status"] == "PASS" and summary["status"] == "PASS" and summary["proposal_refusal_count"] == 0),
        ]
    result = {"status": "PASS" if all(value for _, value in checks) else "FAIL", "checks": [{"name": name, "pass": value} for name, value in checks]}
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
