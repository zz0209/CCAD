"""Quality metrics used by SAE architecture pilots."""

from __future__ import annotations

import math


def ce_recovered(clean_ce: float, reconstruction_ce: float, zero_ablation_ce: float) -> float:
    """Return the fraction of zero-ablation CE damage recovered by reconstruction."""
    values = (clean_ce, reconstruction_ce, zero_ablation_ce)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CE inputs must be finite")
    damage = zero_ablation_ce - clean_ce
    if damage <= 0:
        raise ValueError("zero-ablation CE must exceed clean CE")
    return 1.0 - (reconstruction_ce - clean_ce) / damage


def budget_stability_checks(previous: dict[str, float], current: dict[str, float], thresholds: dict[str, float]) -> dict[str, object]:
    """Evaluate the preregistered adjacent-milestone stability screen."""
    required_metrics = {"fve", "ce_recovered", "alive_fraction", "c_dec"}
    required_thresholds = {"fve_abs", "ce_recovered_abs", "alive_fraction_abs", "c_dec_relative"}
    if set(previous) != required_metrics or set(current) != required_metrics:
        raise ValueError(f"milestone metrics must be exactly {sorted(required_metrics)}")
    if set(thresholds) != required_thresholds:
        raise ValueError(f"thresholds must be exactly {sorted(required_thresholds)}")
    if not all(math.isfinite(float(value)) for value in (*previous.values(), *current.values(), *thresholds.values())):
        raise ValueError("metrics and thresholds must be finite")
    if any(float(value) < 0 for value in thresholds.values()):
        raise ValueError("thresholds must be nonnegative")
    if previous["c_dec"] == 0:
        raise ValueError("previous c_dec must be nonzero for a relative change")
    deltas = {
        "fve_abs": abs(current["fve"] - previous["fve"]),
        "ce_recovered_abs": abs(current["ce_recovered"] - previous["ce_recovered"]),
        "alive_fraction_abs": abs(current["alive_fraction"] - previous["alive_fraction"]),
        "c_dec_relative": abs(current["c_dec"] - previous["c_dec"]) / abs(previous["c_dec"]),
    }
    checks = {
        name: value <= thresholds[name] or math.isclose(value, thresholds[name], rel_tol=1e-12, abs_tol=1e-15)
        for name, value in deltas.items()
    }
    return {"deltas": deltas, "thresholds": dict(thresholds), "checks": checks, "pass": all(checks.values())}


def select_k_bracket(metrics: dict[int, dict[str, float]], fve_margin: float = 0.03, ce_margin: float = 0.03) -> dict[str, object]:
    """Apply the pre-audit B2 reconstruction and geometry selection rule."""
    if sorted(metrics) != [8, 16, 32, 64]:
        raise ValueError("B2 metrics must contain exactly k={8,16,32,64}")
    if not math.isfinite(fve_margin) or not math.isfinite(ce_margin) or fve_margin < 0 or ce_margin < 0:
        raise ValueError("selection margins must be finite and nonnegative")
    required = {"fve", "ce_recovered", "c_dec"}
    if any(set(values) != required for values in metrics.values()):
        raise ValueError(f"each candidate must contain exactly {sorted(required)}")
    if not all(math.isfinite(value) for values in metrics.values() for value in values.values()):
        raise ValueError("candidate metrics must be finite")
    best_fve = max(values["fve"] for values in metrics.values())
    best_ce = max(values["ce_recovered"] for values in metrics.values())
    eligible = [
        k for k, values in sorted(metrics.items())
        if best_fve - values["fve"] <= fve_margin or math.isclose(best_fve - values["fve"], fve_margin, rel_tol=1e-12, abs_tol=1e-15)
        if best_ce - values["ce_recovered"] <= ce_margin or math.isclose(best_ce - values["ce_recovered"], ce_margin, rel_tol=1e-12, abs_tol=1e-15)
    ]
    cdec_by_k = [metrics[k]["c_dec"] for k in sorted(metrics)]
    differences = [right - left for left, right in zip(cdec_by_k, cdec_by_k[1:])]
    strictly_monotonic = all(delta > 0 for delta in differences) or all(delta < 0 for delta in differences)
    expansion_reasons = []
    if not eligible:
        expansion_reasons.append("eligible_set_empty")
    if eligible == [64]:
        expansion_reasons.append("only_upper_boundary_eligible")
    if strictly_monotonic:
        expansion_reasons.append("c_dec_strictly_monotonic_no_internal_elbow")
    provisional = []
    if eligible:
        anchor = min(eligible, key=lambda k: (metrics[k]["c_dec"], k))
        provisional = [anchor]
        higher = [k for k in eligible if k > anchor]
        if higher:
            provisional.append(min(higher))
    return {
        "best_fve": best_fve, "best_ce_recovered": best_ce,
        "fve_margin": fve_margin, "ce_margin": ce_margin,
        "eligible_k": eligible, "c_dec_strictly_monotonic": strictly_monotonic,
        "provisional_shortlist_k": provisional, "expansion_reasons": expansion_reasons,
        "decision": "EXPAND_TO_64_128" if expansion_reasons else "TWO_SEED_PILOT",
        "shortlist_k": [] if expansion_reasons else provisional,
    }


def select_k_extension(metrics: dict[int, dict[str, float]], fve_margin: float = 0.03, ce_margin: float = 0.03) -> dict[str, object]:
    """Apply the frozen five-point B2 extension rule without further auto-expansion."""
    expected_k = [8, 16, 32, 64, 128]
    if sorted(metrics) != expected_k:
        raise ValueError("B2 extension metrics must contain exactly k={8,16,32,64,128}")
    if not math.isfinite(fve_margin) or not math.isfinite(ce_margin) or fve_margin < 0 or ce_margin < 0:
        raise ValueError("selection margins must be finite and nonnegative")
    required = {"fve", "ce_recovered", "c_dec"}
    if any(set(values) != required for values in metrics.values()):
        raise ValueError(f"each candidate must contain exactly {sorted(required)}")
    if not all(math.isfinite(value) for values in metrics.values() for value in values.values()):
        raise ValueError("candidate metrics must be finite")
    best_fve = max(values["fve"] for values in metrics.values())
    best_ce = max(values["ce_recovered"] for values in metrics.values())
    eligible = [
        k for k, values in sorted(metrics.items())
        if best_fve - values["fve"] <= fve_margin or math.isclose(best_fve - values["fve"], fve_margin, rel_tol=1e-12, abs_tol=1e-15)
        if best_ce - values["ce_recovered"] <= ce_margin or math.isclose(best_ce - values["ce_recovered"], ce_margin, rel_tol=1e-12, abs_tol=1e-15)
    ]
    cdec = [metrics[k]["c_dec"] for k in expected_k]
    differences = [right - left for left, right in zip(cdec, cdec[1:])]
    strictly_monotonic = all(delta > 0 for delta in differences) or all(delta < 0 for delta in differences)
    provisional = []
    if eligible:
        anchor = min(eligible, key=lambda k: (metrics[k]["c_dec"], k))
        provisional = [anchor]
        higher = [k for k in eligible if k > anchor]
        if higher:
            provisional.append(min(higher))
    if not eligible:
        decision = "NO_JOINT_ELIGIBLE"
    elif eligible == [128]:
        decision = "UNBOUNDED_HIGH"
    elif strictly_monotonic:
        decision = "UNBOUNDED_GEOMETRY"
    else:
        decision = "TWO_SEED_PILOT"
    return {
        "best_fve": best_fve, "best_ce_recovered": best_ce,
        "fve_margin": fve_margin, "ce_margin": ce_margin,
        "eligible_k": eligible, "c_dec_strictly_monotonic": strictly_monotonic,
        "provisional_shortlist_k": provisional, "decision": decision,
        "shortlist_k": provisional if decision == "TWO_SEED_PILOT" else [],
    }
