"""Static validation of the C049 width-only mechanism screen."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = cfg["matched_baseline_configs"]
    bindings = {seed: digest(ROOT / item["path"]) == item["sha256"] for seed, item in baseline.items()}
    checks = {
        "baseline_bound": all(bindings.values()),
        "single_factor": cfg["treatment_factor"] == "num_latents" and cfg["baseline_num_latents"] == 3072 and cfg["treatment_num_latents"] == 16384,
        "fixed_k_and_seeds": cfg["fixed_k"] == 32 and cfg["formal_seeds"] == [1, 2],
        "fixed_budget": cfg["formal_train_tokens_per_seed"] == 4194304,
        "capacity_gate": cfg["capacity_smoke_steps"] == 256 and cfg["maximum_peak_allocated_vram_bytes"] == 6 * 1024**3 and cfg["maximum_projected_seconds_per_seed"] == 3000 and cfg["maximum_projected_checkpoint_bytes_per_seed"] == 1024**3,
        "quality_gate": cfg["minimum_fve"] == .97 and cfg["minimum_ce_recovered"] == .93 and cfg["minimum_alive_fraction"] == .5 and cfg["required_l0"] == 32 and cfg["maximum_decoder_norm_error"] == 1e-5,
        "structure_gate": cfg["baseline_pw_mcc"] == .4894 and cfg["minimum_pw_mcc_gain"] == .05 and cfg["minimum_native_found_coverage"] == .1 and cfg["minimum_native_found_coverage_per_direction"] == .05,
        "audit_closed": cfg["audit_opened"] is False and cfg["forbidden_splits"] == ["audit"],
        "execution_disabled_until_capacity_implementation": cfg["execution_enabled"] is False,
    }
    print(json.dumps({"checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "bindings": bindings}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
