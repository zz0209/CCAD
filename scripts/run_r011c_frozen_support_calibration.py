"""Evaluate discovery-frozen R011 supports on the independent calibration split."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "r009"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import __version__ as scipy_version  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def metrics(coefficients: np.ndarray, ktt: np.ndarray, kst: np.ndarray, mtt: np.ndarray, mst: np.ndarray, source_energy: float, source_mean_energy: float) -> dict:
    target_energy = float(coefficients @ ktt @ coefficients)
    cross = float(coefficients @ kst)
    residual = max(0.0, source_energy + target_energy - 2.0 * cross)
    target_mean_energy = float(coefficients @ mtt @ coefficients)
    mean_cross = float(coefficients @ mst)
    mean_residual = max(0.0, source_mean_energy + target_mean_energy - 2.0 * mean_cross)
    tiny = np.finfo(np.float64).tiny
    return {
        "d_ctr": residual / max(source_energy, tiny),
        "d_mu": mean_residual / max(source_mean_energy, tiny),
        "bcc": 2.0 * cross / (source_energy + target_energy) if source_energy + target_energy > 0 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "scripts" / "run_r009c_atom_discovery.py"]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    paths = {name: ROOT / cfg[key] for name, key in (("surface", "group_surface_path"), ("separator", "separator_path"), ("panel", "query_panel_path"), ("census", "source_census_path"))}
    asset_dir = Path(cfg["bulk_asset_dir"])
    asset_manifest = asset_dir / "asset_manifest.json"
    inputs = [{"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"}]
    for name, path in paths.items():
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "source": name, "license_or_access_boundary": "internal", "role": name})
    inputs.append({"path": str(asset_manifest.resolve()), "sha256": sha256(asset_manifest), "bytes": asset_manifest.stat().st_size, "source": "R008b", "license_or_access_boundary": "internal", "role": "bulk_asset_hash_ledger"})
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "audit_opened": False,
        "candidate_family_frozen": True, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": "cpu",
        "seeds": cfg["source_seeds"], "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "score 2,560 discovery-frozen support transfers using D: calibration sparse codes",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    output_rows = []
    try:
        expected_hashes = {"surface": cfg["group_surface_sha256"], "separator": cfg["separator_sha256"], "panel": cfg["query_panel_sha256"], "census": cfg["source_census_sha256"]}
        bound = {name: sha256(path).lower() == expected_hashes[name].lower() for name, path in paths.items()}
        bound["asset_manifest"] = sha256(asset_manifest).lower() == cfg["asset_manifest_sha256"].lower()
        if not all(bound.values()):
            raise ValueError(f"frozen input hash mismatch: {bound}")
        surface = [json.loads(line) for line in paths["surface"].read_text(encoding="utf-8").splitlines() if line]
        separator = [json.loads(line) for line in paths["separator"].read_text(encoding="utf-8").splitlines() if line]
        panel = [json.loads(line) for line in paths["panel"].read_text(encoding="utf-8").splitlines() if line]
        census = [json.loads(line) for line in paths["census"].read_text(encoding="utf-8").splitlines() if line]
        surface_map = {(r["source_seed"], r["source_atom"], r["target_seed"]): r for r in surface}
        separator_map = {(r["source_seed"], r["source_atom"], r["target_seed"]): r for r in separator}
        stats = {(r["seed"], r["atom"]): r for r in census}
        queries = {seed: sorted((r for r in panel if r["seed"] == seed), key=lambda r: r["atom"]) for seed in cfg["source_seeds"]}
        matrices = {seed: sparse_codes(asset_dir, cfg["split"], seed, cfg["calibration_tokens"], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        for source_seed in cfg["source_seeds"]:
            query_rows = queries[source_seed]
            query_ids = np.asarray([r["atom"] for r in query_rows], dtype=np.int64)
            zs = matrices[source_seed][:, query_ids]
            ds = decoders[source_seed][query_ids]
            source_means = np.asarray([stats[(source_seed, int(atom))]["mean_code"] for atom in query_ids])
            source_code_energy = np.asarray(zs.multiply(zs).mean(axis=0)).reshape(-1)
            source_energy = np.maximum(0.0, source_code_energy - source_means**2) * np.einsum("ij,ij->i", ds, ds)
            source_mean_vectors = source_means[:, None] * ds
            source_mean_energy = np.einsum("ij,ij->i", source_mean_vectors, source_mean_vectors)
            for target_seed in cfg["source_seeds"]:
                if target_seed == source_seed:
                    continue
                zt, dt = matrices[target_seed], decoders[target_seed]
                target_means = np.asarray([stats[(target_seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])])
                raw_cross = (zs.T @ zt).toarray().astype(np.float64) / cfg["calibration_tokens"]
                cross = (raw_cross - source_means[:, None] * target_means[None, :]) * (ds @ dt.T)
                for qi, query in enumerate(query_rows):
                    key = (source_seed, int(query["atom"]), target_seed)
                    sr, rr = surface_map[key], separator_map[key]
                    proposal = sr["proposal_target_ids"]
                    ids = np.asarray(proposal, dtype=np.int64)
                    local = {atom: index for index, atom in enumerate(proposal)}
                    zsub, dsub = zt[:, ids], dt[ids]
                    code_tt = (zsub.T @ zsub).toarray().astype(np.float64) / cfg["calibration_tokens"] - np.outer(target_means[ids], target_means[ids])
                    ktt = code_tt * (dsub @ dsub.T)
                    ktt = 0.5 * (ktt + ktt.T)
                    kst = cross[qi, ids]
                    mean_vectors = target_means[ids, None] * dsub
                    mtt = mean_vectors @ mean_vectors.T
                    mst = source_mean_vectors[qi] @ mean_vectors.T
                    method_supports = {}
                    for item in sr["best_by_size"]:
                        size = item["size"]
                        method_supports[f"mscc_discovery_min_residual_size{size}"] = item["best_residual_support"]
                        method_supports[f"best_bcc_native_size{size}"] = item["best_bcc_support"]
                    method_supports["signed_omp4_membership_native"] = rr["signed_omp"][3]["support"]
                    decoder_cosine = np.abs(dsub @ ds[qi]) / np.maximum(np.linalg.norm(dsub, axis=1) * np.linalg.norm(ds[qi]), np.finfo(np.float64).tiny)
                    cosine_order = sorted(range(len(proposal)), key=lambda i: (-decoder_cosine[i], proposal[i]))
                    for size in range(1, cfg["g_max"] + 1):
                        method_supports[f"decoder_cosine_prefix_size{size}"] = [proposal[i] for i in cosine_order[:size]]
                    method_metrics = {}
                    for name, support in method_supports.items():
                        coefficients = np.zeros(len(proposal))
                        coefficients[[local[int(atom)] for atom in support]] = 1.0
                        method_metrics[name] = {"support": support, **metrics(coefficients, ktt, kst, mtt, mst, float(source_energy[qi]), float(source_mean_energy[qi]))}
                    primary_candidates = [method_metrics[f"mscc_discovery_min_residual_size{size}"] for size in range(1, cfg["g_max"] + 1)]
                    accepted = [item for item in primary_candidates if item["d_ctr"] <= cfg["primary_tau_ctr"] and item["d_mu"] <= cfg["primary_tau_mu"]]
                    primary = accepted[0] if accepted else None
                    sensitivity = {}
                    for threshold in cfg["report_only_threshold_sensitivity"]:
                        sensitivity[str(threshold)] = any(item["d_ctr"] <= threshold and item["d_mu"] <= threshold for item in primary_candidates)
                    output_rows.append({
                        "source_seed": source_seed, "source_atom": int(query["atom"]), "energy_stratum": int(query["energy_stratum"]), "target_seed": target_seed,
                        "proposal_target_ids": proposal, "method_metrics": method_metrics,
                        "mscc_identification": "FOUND" if primary else "UNRESOLVED",
                        "mscc_support": primary["support"] if primary else [],
                        "mscc_reason": None if primary else "NO_DISCOVERY_FROZEN_SUPPORT_PASSES_CALIBRATION_GATES",
                        "report_only_sensitivity_found": sensitivity,
                    })
        output_path = run_dir / "calibration_transfer.jsonl"
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        expected = len(cfg["source_seeds"]) * (len(cfg["source_seeds"]) - 1) * cfg["queries_per_seed"]
        methods = sorted(output_rows[0]["method_metrics"])
        found = [row["mscc_identification"] == "FOUND" for row in output_rows]
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "complete_ordered_pair_grid": len(output_rows) == expected == 2560,
            "unique_rows": len({(r["source_seed"], r["source_atom"], r["target_seed"]) for r in output_rows}) == len(output_rows),
            "proposal_identity_preserved": all(r["proposal_target_ids"] == surface_map[(r["source_seed"], r["source_atom"], r["target_seed"])]["proposal_target_ids"] for r in output_rows),
            "all_methods_reported": all(sorted(r["method_metrics"]) == methods for r in output_rows),
            "metrics_finite": all(np.isfinite([v[k] for v in r["method_metrics"].values() for k in ("d_ctr", "d_mu", "bcc")]).all() for r in output_rows),
            "calibration_only": cfg["split"] == "calibration" and set(cfg["forbidden_splits"]) == {"audit"},
            "candidate_family_frozen": cfg["candidate_family_frozen"],
            "audit_not_opened": not cfg["audit_opened"],
            "no_certified_absence": all(r["mscc_identification"] in {"FOUND", "UNRESOLVED"} for r in output_rows),
        }
        method_summary = {name: {"median_d_ctr": float(np.median([r["method_metrics"][name]["d_ctr"] for r in output_rows])), "median_d_mu": float(np.median([r["method_metrics"][name]["d_mu"] for r in output_rows])), "median_bcc": float(np.median([r["method_metrics"][name]["bcc"] for r in output_rows]))} for name in methods}
        record = {
            "checks": checks, "row_count": len(output_rows), "methods": methods, "method_summary": method_summary,
            "primary_found_count": int(np.sum(found)), "primary_found_fraction": float(np.mean(found)),
            "sensitivity_found_count": {str(threshold): int(sum(r["report_only_sensitivity_found"][str(threshold)] for r in output_rows)) for threshold in cfg["report_only_threshold_sensitivity"]},
            "output_sha256": sha256(output_path),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version, "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r011c_frozen_support_calibration.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
