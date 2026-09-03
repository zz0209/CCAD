"""Aggregate the prospectively frozen R006c k=128 two-seed quality gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SEED_DIFFERENCES = {"run_id", "purpose", "init_seeds"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_seed_config(config: dict) -> dict:
    result = deepcopy(config)
    for key in ALLOWED_SEED_DIFFERENCES:
        result.pop(key, None)
    return result


def load_single_record(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"expected exactly one metrics record in {path}, found {len(records)}")
    return records[0]


def evaluate(gate_path: Path) -> dict:
    gate = read_json(gate_path)
    seed_rows = []
    input_hashes = {str(gate_path.relative_to(ROOT)).replace("\\", "/"): sha256(gate_path)}
    seed_configs = []

    for config_name in gate["seed_run_configs"]:
        config_path = ROOT / config_name
        config = read_json(config_path)
        seed_configs.append(config)
        run_dir = ROOT / "runs" / config["run_id"]
        artifacts = {
            name: run_dir / name
            for name in (
                "config.resolved.json",
                "contract_validation.json",
                "manifest.json",
                "metrics.raw.jsonl",
                "metrics.summary.json",
                "status.json",
            )
        }
        for name, path in artifacts.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            input_hashes[f"runs/{config['run_id']}/{name}"] = sha256(path)
        input_hashes[config_name] = sha256(config_path)

        resolved = read_json(artifacts["config.resolved.json"])
        contract = read_json(artifacts["contract_validation.json"])
        manifest = read_json(artifacts["manifest.json"])
        summary = read_json(artifacts["metrics.summary.json"])
        status = read_json(artifacts["status.json"])
        metrics = load_single_record(artifacts["metrics.raw.jsonl"])
        validation = metrics["validation"]
        firing = validation["feature_firing_count_distribution"]
        train_tokens_per_second = 1_000_000.0 / metrics["seconds_per_million_tokens"]
        exact_checkpoint = run_dir / "exact_checkpoint"
        safe_checkpoint = run_dir / "sae"

        checks = {
            "resolved_config_binding": resolved == config,
            "status_pass": status.get("status") == "PASS" and summary.get("status") == "PASS",
            "contract_pass": contract.get("ok") is True and not contract.get("errors"),
            "clean_frozen_git_run": manifest.get("git_head_at_run") == "e71c0f7044a4fa4891d61c5d352661bfa4566680" and manifest.get("git_status_porcelain") == [],
            "required_checks_pass": all(metrics["checks"].get(name) is True for name in gate["required_checks"]),
            "exact_checkpoint_materialized": exact_checkpoint.is_dir() and any(exact_checkpoint.iterdir()),
            "safe_checkpoint_materialized": safe_checkpoint.is_dir() and any(safe_checkpoint.iterdir()),
            "fve": validation["fve"] >= gate["minimum_fve"],
            "ce_recovered": validation["ce_recovered"] >= gate["minimum_ce_recovered"],
            "actual_l0": validation["actual_nonzero_l0"] == gate["required_actual_l0"],
            "alive_fraction": validation["alive_fraction"] >= gate["minimum_alive_fraction"],
            "nonzero_firing": firing["nonzero_min"] >= gate["minimum_nonzero_firing_count"],
            "decoder_norm": metrics["decoder_norm_max_error"] <= gate["maximum_decoder_norm_error"],
            "throughput": train_tokens_per_second >= gate["minimum_train_tokens_per_second"],
            "peak_allocated_vram": metrics["peak_allocated_vram_bytes"] <= gate["maximum_peak_allocated_vram_bytes"],
        }
        seed_rows.append(
            {
                "run_id": config["run_id"],
                "init_seed": config["init_seeds"][0],
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "metrics": {
                    "fve": validation["fve"],
                    "ce_recovered": validation["ce_recovered"],
                    "actual_nonzero_l0": validation["actual_nonzero_l0"],
                    "alive_fraction": validation["alive_fraction"],
                    "nonzero_min_firing_count": firing["nonzero_min"],
                    "decoder_norm_max_error": metrics["decoder_norm_max_error"],
                    "train_tokens_per_second": train_tokens_per_second,
                    "peak_allocated_vram_bytes": metrics["peak_allocated_vram_bytes"],
                    "global_step": metrics["checkpoint_metadata"]["global_step"],
                    "data_cursor_examples": metrics["checkpoint_metadata"]["data_cursor_examples"],
                    "sae_state_sha256": metrics["state_hash"],
                    "exact_checkpoint_state_sha256": metrics["checkpoint_state_sha256"],
                },
            }
        )

    fve_values = [row["metrics"]["fve"] for row in seed_rows]
    ce_values = [row["metrics"]["ce_recovered"] for row in seed_rows]
    alive_values = [row["metrics"]["alive_fraction"] for row in seed_rows]
    pair_metrics = {
        "fve_range": max(fve_values) - min(fve_values),
        "ce_recovered_range": max(ce_values) - min(ce_values),
        "alive_fraction_range": max(alive_values) - min(alive_values),
    }
    pair_checks = {
        "exact_seed_set": sorted(row["init_seed"] for row in seed_rows) == [1, 2],
        "same_config_except_declared_seed_fields": normalized_seed_config(seed_configs[0]) == normalized_seed_config(seed_configs[1]),
        "distinct_sae_states": len({row["metrics"]["sae_state_sha256"] for row in seed_rows}) == len(seed_rows),
        "fve_range": pair_metrics["fve_range"] <= gate["maximum_pair_fve_range"],
        "ce_recovered_range": pair_metrics["ce_recovered_range"] <= gate["maximum_pair_ce_recovered_range"],
        "alive_fraction_range": pair_metrics["alive_fraction_range"] <= gate["maximum_pair_alive_fraction_range"],
    }

    resume_dir = ROOT / "runs" / "R005d_pythia_native_resume_v2_20260902T111500Z"
    resume_summary = read_json(resume_dir / "metrics.summary.json")
    resume_contract = read_json(resume_dir / "contract_validation.json")
    resume_metrics = load_single_record(resume_dir / "metrics.raw.jsonl")
    resume_checks = resume_metrics.get("checks", {})
    resume_qualification = {
        "role": "previously completed framework-level exact trajectory qualification; report-only for R006c selection",
        "run_id": resume_dir.name,
        "summary_sha256": sha256(resume_dir / "metrics.summary.json"),
        "status": "PASS" if resume_summary.get("status") == "PASS" and resume_contract.get("ok") is True and resume_checks and all(resume_checks.values()) else "FAIL",
        "checks": resume_checks,
    }

    all_gate_checks = [value for row in seed_rows for value in row["checks"].values()] + list(pair_checks.values())
    return {
        "schema_version": "r006c.two_seed_quality_gate_result.v1",
        "gate_config": str(gate_path.relative_to(ROOT)).replace("\\", "/"),
        "decision": gate["decision"],
        "claim_limit": gate["claim_limit"],
        "input_hashes": input_hashes,
        "seed_results": seed_rows,
        "pair_metrics": pair_metrics,
        "pair_checks": pair_checks,
        "resume_qualification": resume_qualification,
        "gate_checks_passed": sum(all_gate_checks),
        "gate_checks_total": len(all_gate_checks),
        "status": "PASS" if all(all_gate_checks) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-config", default="configs/r006c_k128_two_seed_gate_v1.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    gate_path = (ROOT / args.gate_config).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = evaluate(gate_path)
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "status.json", {"status": result["status"]})
    write_json(output_dir / "config.resolved.json", read_json(gate_path))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
