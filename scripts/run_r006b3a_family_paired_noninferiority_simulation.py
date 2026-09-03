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

import mpmath as mp
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def student_t_cdf(value: float, df: int) -> float:
    if df < 1:
        raise ValueError("df must be positive")
    if value == 0.0:
        return 0.5
    x = df / (df + value * value)
    tail = 0.5 * mp.betainc(df / 2.0, 0.5, 0.0, x, regularized=True)
    return float(1.0 - tail if value > 0.0 else tail)


def student_t_quantile(probability: float, df: int) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("only upper-half probabilities are supported")
    low, high = 0.0, 2.0
    while student_t_cdf(high, df) < probability:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - radius, center + radius


def select_from_ucb(ucb: np.ndarray, margin: float) -> np.ndarray:
    eligible = ucb <= margin
    any_eligible = eligible.any(axis=-1)
    first = eligible.argmax(axis=-1)
    return np.where(any_eligible, first, ucb.shape[-1]).astype(np.int64)


def simulate_cell(
    rng: np.random.Generator,
    *,
    n_sims: int,
    families: int,
    tasks: int,
    icc: float,
    losses: np.ndarray,
    margin: float,
    total_sd: float,
    cross_corr: float,
    familywise_alpha: float,
    lofo_required_fraction: float,
    chunk_size: int = 250,
) -> dict[str, Any]:
    comparisons = losses.size - 1
    critical = student_t_quantile(1.0 - familywise_alpha / comparisons, families - 1)
    target = int(np.flatnonzero(losses[:comparisons] <= margin)[0])
    counts = {"false_noninferiority": 0, "boundary_false_noninferiority": 0,
              "correct_smallest_safe": 0, "lofo_stability": 0,
              "argmax_correct": 0, "one_se_correct": 0}

    family_sd = total_sd * math.sqrt(icc)
    task_mean_sd = total_sd * math.sqrt(1.0 - icc) / math.sqrt(tasks)
    done = 0
    while done < n_sims:
        batch = min(chunk_size, n_sims - done)
        common_family = rng.normal(size=(batch, families, 1))
        config_family = rng.normal(size=(batch, families, losses.size))
        common_task = rng.normal(size=(batch, families, 1))
        config_task = rng.normal(size=(batch, families, losses.size))
        error = (
            family_sd * (math.sqrt(cross_corr) * common_family + math.sqrt(1.0 - cross_corr) * config_family)
            + task_mean_sd * (math.sqrt(cross_corr) * common_task + math.sqrt(1.0 - cross_corr) * config_task)
        )
        utility = -losses[None, None, :] + error
        family_losses = utility[:, :, -1, None] - utility[:, :, :comparisons]
        sum_loss = family_losses.sum(axis=1)
        sumsq_loss = np.square(family_losses).sum(axis=1)
        mean_loss = sum_loss / families
        variance = (sumsq_loss - families * np.square(mean_loss)) / (families - 1)
        se = np.sqrt(np.maximum(variance, 0.0) / families)
        selected = select_from_ucb(mean_loss + critical * se, margin)
        selected_true_loss = losses[selected]
        counts["false_noninferiority"] += int(np.sum(selected_true_loss > margin))
        counts["correct_smallest_safe"] += int(np.sum(selected == target))

        boundary = margin + (family_losses[:, :, 0] - losses[0])
        boundary_mean = boundary.mean(axis=1)
        boundary_se = boundary.std(axis=1, ddof=1) / math.sqrt(families)
        counts["boundary_false_noninferiority"] += int(np.sum(boundary_mean + critical * boundary_se <= margin))

        kept_n = families - 1
        leave_sum = sum_loss[:, None, :] - family_losses
        leave_sumsq = sumsq_loss[:, None, :] - np.square(family_losses)
        leave_mean = leave_sum / kept_n
        leave_var = (leave_sumsq - kept_n * np.square(leave_mean)) / (kept_n - 1)
        leave_se = np.sqrt(np.maximum(leave_var, 0.0) / kept_n)
        leave_critical = student_t_quantile(1.0 - familywise_alpha / comparisons, kept_n - 1)
        leave_selected = select_from_ucb(leave_mean + leave_critical * leave_se, margin)
        stable_fraction = np.mean(leave_selected == selected[:, None], axis=1)
        counts["lofo_stability"] += int(np.sum(stable_fraction >= lofo_required_fraction))

        means = utility.mean(axis=1)
        counts["argmax_correct"] += int(np.sum(np.argmax(means, axis=1) == target))
        best = np.argmax(means, axis=1)
        best_values = np.take_along_axis(means, best[:, None], axis=1)[:, 0]
        best_family = utility[
            np.arange(batch)[:, None], np.arange(families)[None, :], best[:, None]
        ]
        best_se = best_family.std(axis=1, ddof=1) / math.sqrt(families)
        one_se_eligible = means >= best_values[:, None] - best_se[:, None]
        one_se_selected = one_se_eligible.argmax(axis=1)
        counts["one_se_correct"] += int(np.sum(one_se_selected == target))
        done += batch

    metrics: dict[str, Any] = {}
    for name, successes in counts.items():
        low, high = wilson_interval(successes, n_sims)
        metrics[name] = {"estimate": successes / n_sims, "mc_ci_low": low,
                         "mc_ci_high": high, "mc_half_width": (high - low) / 2.0}
    return {"families": families, "tasks_per_family": tasks, "icc": icc,
            "margin": margin, "losses": losses.tolist(), "target_index": target,
            "critical_value": critical, "metrics": metrics}


def cell_passes(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    m = row["metrics"]
    return bool(
        m["false_noninferiority"]["mc_ci_high"] <= thresholds["false_noninferiority_upper_mc_bound"]
        and m["boundary_false_noninferiority"]["mc_ci_high"] <= thresholds["boundary_false_noninferiority_upper_mc_bound"]
        and m["correct_smallest_safe"]["mc_ci_low"] >= thresholds["correct_smallest_safe_lower_mc_bound"]
        and m["lofo_stability"]["mc_ci_low"] >= thresholds["lofo_stability_lower_mc_bound"]
        and all(metric["mc_half_width"] <= thresholds["maximum_mc_half_width"] for metric in m.values())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["audit_opened"] or config["sae_scores_loaded"] or config["probe_activations_computed"]:
        raise ValueError("T021 must remain score-free and pre-activation")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    margins = [config["primary_sesoi_auc"], *config["sensitivity_sesoi_auc"]]
    cells = [(margin, f, t, icc, name, multiples) for margin in margins
             for f in config["family_counts"] for t in config["tasks_per_family"]
             for icc in config["icc_values"] for name, multiples in config["loss_profiles_as_sesoi_multiples"].items()]
    children = np.random.SeedSequence(int(config["simulation_seed"])).spawn(len(cells))
    rows = []
    for cell, child in zip(cells, children, strict=True):
        margin, families, tasks, icc, profile_name, multiples = cell
        row = simulate_cell(np.random.default_rng(child), n_sims=int(config["n_simulations"]),
                            families=int(families), tasks=int(tasks), icc=float(icc),
                            losses=np.asarray(multiples, dtype=np.float64) * float(margin),
                            margin=float(margin), total_sd=float(config["total_task_sd"]),
                            cross_corr=float(config["cross_configuration_correlation"]),
                            familywise_alpha=float(config["familywise_alpha"]),
                            lofo_required_fraction=float(config["lofo_required_fraction"]))
        row["profile_name"] = profile_name
        row["passes"] = cell_passes(row, config["gate_thresholds"])
        rows.append(row)

    gates: dict[str, dict[str, bool]] = {}
    minimums: dict[str, int | None] = {}
    for margin in margins:
        by_family = {}
        for families in config["family_counts"]:
            subset = [row for row in rows if row["margin"] == margin and row["families"] == families]
            by_family[str(families)] = all(row["passes"] for row in subset)
        gates[str(margin)] = by_family
        passing = [int(f) for f, passed in by_family.items() if passed]
        minimums[str(margin)] = min(passing) if passing else None
    primary_minimum = minimums[str(config["primary_sesoi_auc"])]
    outcome = "PRIMARY_MARGIN_QUALIFIED" if primary_minimum is not None else "PRIMARY_MARGIN_NOT_QUALIFIED"
    summary = {"schema_version": 1, "run_status": "PASS", "semantic_outcome": outcome,
               "b3a_gate": "NOT_PASSED", "scenario_count": len(rows),
               "passing_scenario_count": sum(row["passes"] for row in rows),
               "minimum_qualified_families_by_margin": minimums,
               "family_gates_by_margin": gates,
               "notes": ["PASS is an artifact execution status, not T019/B3-A qualification.",
                         "No SAE score, activation, or audit datum was loaded."]}
    resolved = dict(config)
    resolved.update({"run_id": args.output_dir.name, "config_path": str(args.config.resolve()),
                     "output_dir": str(args.output_dir.resolve())})
    environment = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "numpy_version": np.__version__, "mpmath_version": mp.__version__,
                   "platform": platform.platform(), "workspace_git_repository": False}
    status = {"run_status": "PASS", "semantic_outcome": outcome, "b3a_gate": "NOT_PASSED",
              "protocol_deviation": False}
    finished = dt.datetime.now(dt.timezone.utc)
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
