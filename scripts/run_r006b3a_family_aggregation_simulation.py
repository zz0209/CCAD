from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


T975 = {3: 3.182, 5: 2.571, 7: 2.365, 9: 2.262, 11: 2.201,
        15: 2.131, 19: 2.093, 23: 2.069, 31: 2.040}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - radius, center + radius


def one_se_choice(means: np.ndarray, se_best: float) -> int:
    best = int(np.argmax(means))
    eligible = np.flatnonzero(means >= means[best] - se_best)
    return int(eligible[0])


def simulate_scenario(
    rng: np.random.Generator,
    *,
    n_sims: int,
    families: int,
    tasks: int,
    icc: float,
    profile: np.ndarray,
    total_sd: float,
    cross_corr: float,
    lofo_required_fraction: float,
) -> dict[str, Any]:
    n_configs = profile.size
    family_sd = total_sd * math.sqrt(icc)
    task_sd = total_sd * math.sqrt(1.0 - icc)
    theoretical_se = math.sqrt(family_sd**2 + task_sd**2 / tasks) / math.sqrt(families)
    oracle_choice = one_se_choice(profile, theoretical_se)
    tcrit = T975[families - 1]

    counts = {"cluster_cover": 0, "naive_cover": 0, "selector": 0, "lofo": 0,
              "null_cluster_fp": 0, "null_naive_fp": 0}
    for _ in range(n_sims):
        common_family = rng.normal(size=(families, 1))
        config_family = rng.normal(size=(families, n_configs))
        family_effect = family_sd * (
            math.sqrt(cross_corr) * common_family
            + math.sqrt(1.0 - cross_corr) * config_family
        )
        common_task = rng.normal(size=(families, tasks, 1))
        config_task = rng.normal(size=(families, tasks, n_configs))
        task_effect = task_sd * (
            math.sqrt(cross_corr) * common_task
            + math.sqrt(1.0 - cross_corr) * config_task
        )
        observations = profile + family_effect[:, None, :] + task_effect
        family_means = observations.mean(axis=1)
        means = family_means.mean(axis=0)

        best_index = n_configs - 1
        cluster_se = family_means[:, best_index].std(ddof=1) / math.sqrt(families)
        naive_se = observations[:, :, best_index].reshape(-1).std(ddof=1) / math.sqrt(families * tasks)
        counts["cluster_cover"] += abs(means[best_index] - profile[best_index]) <= tcrit * cluster_se
        counts["naive_cover"] += abs(means[best_index] - profile[best_index]) <= 1.96 * naive_se

        selected = one_se_choice(means, cluster_se)
        counts["selector"] += selected == oracle_choice
        leave_choices = []
        for omitted in range(families):
            kept = np.delete(family_means, omitted, axis=0)
            kept_means = kept.mean(axis=0)
            kept_best = int(np.argmax(kept_means))
            kept_se = kept[:, kept_best].std(ddof=1) / math.sqrt(families - 1)
            leave_choices.append(one_se_choice(kept_means, kept_se))
        counts["lofo"] += (np.mean(np.asarray(leave_choices) == selected) >= lofo_required_fraction)

        null_family = family_means[:, -1] - family_means[:, -2]
        null_family = null_family - null_family.mean()
        null_family = null_family + rng.normal(0.0, family_sd, size=families)
        null_mean = float(null_family.mean())
        null_cluster_se = float(null_family.std(ddof=1) / math.sqrt(families))
        counts["null_cluster_fp"] += abs(null_mean) > tcrit * null_cluster_se
        null_tasks = rng.normal(0.0, total_sd, size=(families, tasks)) + null_family[:, None]
        null_task_mean = float(null_tasks.mean())
        null_naive_se = float(null_tasks.reshape(-1).std(ddof=1) / math.sqrt(families * tasks))
        counts["null_naive_fp"] += abs(null_task_mean) > 1.96 * null_naive_se

    metrics: dict[str, Any] = {}
    for name, successes in counts.items():
        low, high = wilson_interval(int(successes), n_sims)
        metrics[name] = {
            "estimate": successes / n_sims,
            "mc_ci_low": low,
            "mc_ci_high": high,
            "mc_half_width": (high - low) / 2.0,
        }
    return {
        "families": families,
        "tasks_per_family": tasks,
        "icc": icc,
        "profile": profile.tolist(),
        "theoretical_best_se": theoretical_se,
        "oracle_one_se_choice_index": oracle_choice,
        "metrics": metrics,
    }


def scenario_passes(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    m = row["metrics"]
    precision_ok = all(v["mc_half_width"] <= thresholds["maximum_mc_half_width"] for v in m.values())
    return bool(
        m["cluster_cover"]["mc_ci_low"] >= thresholds["cluster_coverage_lower_mc_bound"]
        and m["null_cluster_fp"]["mc_ci_high"] <= thresholds["null_false_positive_upper_mc_bound"]
        and m["selector"]["mc_ci_low"] >= thresholds["oracle_selector_agreement_lower_mc_bound"]
        and m["lofo"]["mc_ci_low"] >= thresholds["lofo_stability_lower_mc_bound"]
        and precision_ok
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["audit_opened"] or config["sae_scores_loaded"] or config["probe_activations_computed"]:
        raise ValueError("T020 must be score-free and pre-activation")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    root_seed = int(config["simulation_seed"])
    seed_sequence = np.random.SeedSequence(root_seed)
    cells = [(f, t, i, name, values) for f in config["family_counts"]
             for t in config["tasks_per_family"] for i in config["icc_values"]
             for name, values in config["utility_profiles"].items()]
    children = seed_sequence.spawn(len(cells))
    rows = []
    for cell, child in zip(cells, children, strict=True):
        families, tasks, icc, profile_name, values = cell
        row = simulate_scenario(
            np.random.default_rng(child), n_sims=int(config["n_simulations"]),
            families=int(families), tasks=int(tasks), icc=float(icc),
            profile=np.asarray(values, dtype=np.float64), total_sd=float(config["total_task_sd"]),
            cross_corr=float(config["cross_configuration_correlation"]),
            lofo_required_fraction=float(config["lofo_required_fraction"]),
        )
        row["profile_name"] = profile_name
        row["passes"] = scenario_passes(row, config["gate_thresholds"])
        rows.append(row)

    family_gate = {}
    for families in config["family_counts"]:
        selected = [row for row in rows if row["families"] == families]
        family_gate[str(families)] = all(row["passes"] for row in selected)
    qualified = [int(key) for key, value in family_gate.items() if value]
    minimum = min(qualified) if qualified else None
    outcome = "QUALIFIED_MINIMUM_FOUND" if minimum is not None else "NO_FAMILY_COUNT_QUALIFIED"
    finished = dt.datetime.now(dt.timezone.utc)

    summary = {
        "schema_version": 1,
        "run_status": "PASS",
        "semantic_outcome": outcome,
        "b3a_gate": "NOT_PASSED",
        "minimum_qualified_families": minimum,
        "family_gate": family_gate,
        "scenario_count": len(rows),
        "passing_scenario_count": sum(row["passes"] for row in rows),
        "naive_diagnostic": {
            "minimum_coverage": min(row["metrics"]["naive_cover"]["estimate"] for row in rows),
            "maximum_null_false_positive": max(row["metrics"]["null_naive_fp"]["estimate"] for row in rows),
        },
        "notes": [
            "PASS means the simulation artifact contract completed, not that T019 or R006-B3-A passed.",
            "No SAE score, activation, or audit datum was loaded.",
        ],
    }
    resolved = dict(config)
    resolved.update({"run_id": args.output_dir.name, "config_path": str(args.config.resolve()),
                     "output_dir": str(args.output_dir.resolve())})
    environment = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "numpy_version": np.__version__, "platform": platform.platform(),
                   "workspace_git_repository": False}
    status = {"run_status": "PASS", "semantic_outcome": outcome,
              "b3a_gate": "NOT_PASSED", "protocol_deviation": False}
    log = {"started_at": started.isoformat(), "finished_at": finished.isoformat(),
           "events": ["validated score-free contract", f"simulated {len(rows)} design cells",
                      f"semantic outcome: {outcome}"]}
    for name, payload in {"resolved_config.json": resolved, "environment.json": environment,
                          "status.json": status, "summary.json": summary,
                          "simulation_results.json": rows, "run_log.json": log}.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"config_sha256": sha256_file(args.config), "runner_sha256": sha256_file(Path(__file__))}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
