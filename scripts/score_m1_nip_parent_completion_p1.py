"""Post-closure P1 scorer; opens truth/evaluation/intervention only after validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback

import numpy as np

from ccad.nip_metric_surface import native_support_metric_surface
from ccad.nip_synthetic_v2 import generate_cap_identifiable_observed
from ccad.nip_synthetic_v3 import evaluate_shared_hook_endpoint, generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_closure(prediction_dir: Path) -> tuple[dict, dict, dict]:
    closure = json.loads((prediction_dir / "prediction_closure.json").read_text(encoding="utf-8"))
    validation = json.loads((prediction_dir / "prelabel_validation.json").read_text(encoding="utf-8"))
    config = json.loads((prediction_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if closure["state"] != "SEALED" or closure["truth_opened"] or not all(sha(prediction_dir / name) == value for name, value in closure["files"].items()):
        raise ValueError("prediction closure verification failed")
    if validation["status"] != "PASS" or validation["passed_count"] != validation["check_count"] or validation["truth_opened"]:
        raise ValueError("prelabel validation is not a closed PASS")
    return closure, validation, config


def _na_surface(fields: list[str], reason: str) -> dict:
    value = {field: {"status": "NOT_APPLICABLE", "reason": reason, "value": None} for field in fields}
    value["schema_version"] = "metric_surface.v2-nip"
    return value


def _supports(row: dict) -> list[tuple[int, ...]]:
    raw = row["prediction"].get("supports", [])
    if row["lane"] == "MSCC":
        return [tuple(item["target_ids"]) for item in raw]
    return [tuple(item) for item in raw]


def score_rows(prediction_dir: Path, config: dict, truth_module) -> list[dict]:
    predictions = _read_jsonl(prediction_dir / "predictions.jsonl")
    proposals = {(row["family_id"], row["pair_index"], row["lane"]): row for row in _read_jsonl(prediction_dir / "proposals.jsonl")}
    seeds = {(row["family_id"], row["pair_index"]): row["seeds"] for row in json.loads((prediction_dir / "seed_ledger.json").read_text(encoding="utf-8"))["rows"]}
    parent = json.loads((ROOT / config["parent_config_path"]).read_text(encoding="utf-8"))
    mandatory = parent["mandatory_metric_fields"]
    scored = []
    for row in predictions:
        family, pair, lane = row["family_id"], row["pair_index"], row["lane"]
        streams = seeds[(family, pair)]
        truth = truth_module.nip_truth(family)
        mean = generate_cap_identifiable_observed(family, structural_seed=streams["structural"], sample_seed=streams["mean"], n=config["sample_sizes"]["mean"])
        evaluation = generate_endpoint_observed(family, structural_seed=streams["structural"], sample_seed=streams["evaluation"], n=parent["sample_sizes"]["evaluation"])
        support_list = _supports(row)
        truth_supports = {tuple(value) for value in truth.minimum_supports}
        predicted_supports = set(support_list)
        positive = truth.identification == "FOUND"
        exact = positive and predicted_supports == truth_supports and row["prediction"].get("multiplicity") == truth.multiplicity
        false_native_positive = not positive and bool(support_list)
        false_unique = truth.multiplicity == "AMBIGUOUS" and row["prediction"].get("multiplicity") == "UNIQUE"
        proposal = proposals[(family, pair, lane)]
        proposed = set(proposal["proposed_target_ids"])
        proposal_recall = None if not positive or row["kind"] != "NATIVE" else any(set(item) <= proposed for item in truth_supports)

        if row["kind"] == "NATIVE" and support_list:
            surfaces = []
            for support in support_list:
                diagnostics = {
                    "multiplicity": {"status": "MEASURED", "value": row["prediction"].get("multiplicity")},
                    "tie_set": {"status": "MEASURED", "value": [list(value) for value in support_list]},
                    "proposal_recall": {"status": "MEASURED", "value": proposal_recall},
                    "conditional_solver_correctness": {"status": "MEASURED", "value": exact if proposal_recall else None},
                    "end_to_end_recovery": {"status": "MEASURED", "value": exact},
                    "coverage": {"status": "MEASURED", "value": 1.0},
                    "terminal_reason": {"status": "MEASURED", "value": row["prediction"].get("terminal_reason") or row["prediction"].get("unresolved_reason")},
                }
                surfaces.append(native_support_metric_surface(
                    evaluation.source_contributions, evaluation.target_contributions,
                    mean.source_mean_contributions, mean.target_mean_contributions,
                    evaluation.document_ids, source_atom_id=0, target_ids=support,
                    epsilon=config["epsilon"], algorithm_diagnostics=diagnostics,
                ))
        else:
            reason = "CONTINUOUS_REFERENCE" if row["kind"] != "NATIVE" else "NO_FROZEN_NATIVE_SUPPORT"
            surfaces = [_na_surface(mandatory, reason)]

        continuous = None
        if row["kind"] == "CONTINUOUS_REFERENCE":
            coefficients = np.asarray(row["prediction"]["coefficients"], dtype=np.float64)
            source = evaluation.source_contributions[:, 0, :]
            target = np.einsum("nat,a->nt", evaluation.target_contributions, coefficients)
            numerator = float(np.sum((source - target) ** 2))
            denominator = float(np.sum(source * source)) + config["epsilon"]
            continuous = {"residual_numerator": numerator, "source_energy_denominator": denominator, "normalized_residual": numerator / denominator, "coefficients": coefficients.tolist(), "converged": row["prediction"]["converged"]}

        intervention = None
        if family == "N11_downstream_cliff" and support_list:
            instance = generate_endpoint_observed(family, structural_seed=streams["structural"], sample_seed=streams["intervention"], n=parent["sample_sizes"]["intervention"])
            intervention = [evaluate_shared_hook_endpoint(instance, support) for support in support_list]

        full_block_control = None
        if family == "N06_exact_dense_orthogonal_rotation" and lane == "MSCC":
            grouped_source = np.sum(evaluation.source_contributions[:, (0, 1), :], axis=1)[:, None, :]
            grouped_source_mean = np.sum(mean.source_mean_contributions[:, (0, 1)], axis=1)[:, None]
            full_block_control = native_support_metric_surface(
                grouped_source, evaluation.target_contributions,
                grouped_source_mean, mean.target_mean_contributions,
                evaluation.document_ids, source_atom_id=0, target_ids=(0, 1), epsilon=config["epsilon"],
            )
        scored.append({
            "family_id": family, "pair_index": pair, "lane": lane, "kind": row["kind"],
            "truth": {"identification": truth.identification, "multiplicity": truth.multiplicity, "minimum_supports": truth.minimum_supports},
            "frozen_supports": support_list, "positive_exact": exact,
            "false_native_positive": false_native_positive, "false_unique": false_unique,
            "proposal_recall": proposal_recall, "metric_surfaces": surfaces,
            "continuous_evaluation": continuous, "intervention_evaluation": intervention,
            "n06_full_block_control": full_block_control,
        })
    return scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--score-run", required=True)
    args = parser.parse_args()
    prediction_dir, score_dir = Path(args.prediction_run).resolve(), Path(args.score_run).resolve()
    if score_dir.exists():
        raise FileExistsError(score_dir)
    closure, validation, config = verify_closure(prediction_dir)
    score_dir.mkdir(parents=True)
    started = now()
    write_json(score_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    try:
        scorer_snapshot = score_dir / "source_snapshot" / "scripts" / Path(__file__).name
        scorer_snapshot.parent.mkdir(parents=True)
        shutil.copy2(Path(__file__), scorer_snapshot)
        # Truth is imported only after closure and validation verification above.
        truth_module = importlib.import_module("ccad.nip_truth")
        rows = score_rows(prediction_dir, config, truth_module)
        raw = score_dir / "scores.jsonl"
        raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        summary = {
            "status": "PASS", "row_count": len(rows), "prediction_run": prediction_dir.name,
            "prediction_closure_sha256": sha(prediction_dir / "prediction_closure.json"),
            "prelabel_validation_sha256": sha(prediction_dir / "prelabel_validation.json"),
            "truth_imported_after_closure_verification": True,
            "native_positive_exact": {lane: sum(row["positive_exact"] for row in rows if row["lane"] == lane) for lane in config["native_lanes"]},
            "native_false_positive": {lane: sum(row["false_native_positive"] for row in rows if row["lane"] == lane) for lane in config["native_lanes"]},
            "raw_sha256": sha(raw),
        }
        write_json(score_dir / "summary.json", summary)
        write_json(score_dir / "environment.json", {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "device": "cpu"})
        write_json(score_dir / "manifest.json", {"prediction_closure_sha256": summary["prediction_closure_sha256"], "prelabel_validation_sha256": summary["prelabel_validation_sha256"], "scorer_snapshot": scorer_snapshot.relative_to(score_dir).as_posix(), "scorer_sha256": sha(scorer_snapshot), "truth_open_order": "AFTER_CLOSURE_AND_PRELABEL_PASS"})
        (score_dir / "stdout.log").write_text(f"scored {len(rows)} post-closure rows\n", encoding="utf-8")
        (score_dir / "stderr.log").write_text("", encoding="utf-8")
        write_json(score_dir / "status.json", {"status": "PASS", "started_utc": started, "ended_utc": now()})
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as error:
        write_json(score_dir / "status.json", {"status": "FAIL", "started_utc": started, "ended_utc": now(), "failure_type": type(error).__name__, "failure_message": str(error)})
        (score_dir / "stderr.log").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
