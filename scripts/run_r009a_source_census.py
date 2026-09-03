"""Compute full-universe source-only atom census statistics from R008 assets."""
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
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def document_index(documents: list[dict], split: str, salt: str, token_count: int) -> tuple[np.ndarray, list[str]]:
    rows = sorted((row for row in documents if row["split"] == split), key=lambda row: score(salt + "-order", row["document_id"]))
    assignment = np.full(token_count, -1, dtype=np.int32)
    cursor = 1
    ids = []
    for index, row in enumerate(rows):
        end = min(cursor + int(row["included_token_count"]), token_count)
        assignment[cursor:end] = index
        ids.append(row["document_id"])
        cursor = end + 1
        if cursor >= token_count:
            break
    return assignment, ids


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
    code_paths = [Path(__file__).resolve()]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    documents_path = ROOT / cfg["documents_path"]
    asset_manifest_path = Path(cfg["bulk_asset_dir"]) / "asset_manifest.json"
    inputs = [
        {"path": str(args.config.resolve()), "sha256": sha256(args.config.resolve()), "bytes": args.config.stat().st_size, "source": "CCAD config", "license_or_access_boundary": "internal", "role": "protocol"},
        {"path": str(documents_path.resolve()), "sha256": sha256(documents_path), "bytes": documents_path.stat().st_size, "source": cfg["paired_corpus_run"], "license_or_access_boundary": "ODC-By-1.0", "role": "document_ledger"},
        {"path": str(asset_manifest_path.resolve()), "sha256": sha256(asset_manifest_path), "bytes": asset_manifest_path.stat().st_size, "source": cfg["paired_codes_run"], "license_or_access_boundary": "internal", "role": "sparse_code_manifest"},
    ]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {"schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started,
        "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash,
        "audit_opened": False, "candidate_family_frozen": False, "mean_constants_source_split": "mean", "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": "cpu", "seeds": cfg["source_seeds"],
        "resource_lease": "cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run", "resource_lease_reason": "document-level aggregation over source-only sparse codes",
        "git_head_at_run": git_head, "git_status_porcelain": git_status})
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line]
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        split_meta = {row["split"]: row for row in asset_manifest["splits"]}
        mean_tokens = split_meta[cfg["mean_split"]]["tokens"]
        census_tokens = split_meta[cfg["census_split"]]["tokens"]
        doc_assignment, document_ids = document_index(documents, cfg["census_split"], cfg["selection_salt"], census_tokens)
        rows_out, seed_summaries = [], []
        for seed in cfg["source_seeds"]:
            mean_dir = Path(cfg["bulk_asset_dir"]) / cfg["mean_split"] / f"seed_{seed}"
            census_dir = Path(cfg["bulk_asset_dir"]) / cfg["census_split"] / f"seed_{seed}"
            mean_indices = np.memmap(mean_dir / "top_indices.uint16.bin", dtype="<u2", mode="r", shape=(mean_tokens, cfg["k"]))
            mean_acts = np.memmap(mean_dir / "top_acts.float32.bin", dtype="<f4", mode="r", shape=(mean_tokens, cfg["k"]))
            mean_sum = np.bincount(mean_indices.reshape(-1), weights=mean_acts.reshape(-1), minlength=cfg["num_latents"])
            indices = np.memmap(census_dir / "top_indices.uint16.bin", dtype="<u2", mode="r", shape=(census_tokens, cfg["k"]))
            acts = np.memmap(census_dir / "top_acts.float32.bin", dtype="<f4", mode="r", shape=(census_tokens, cfg["k"]))
            firing = np.zeros(cfg["num_latents"], dtype=np.int64)
            energy = np.zeros(cfg["num_latents"], dtype=np.float64)
            signed_sum = np.zeros(cfg["num_latents"], dtype=np.float64)
            doc_energy = np.zeros((len(document_ids), cfg["num_latents"]), dtype=np.float64)
            for begin in range(0, census_tokens, cfg["chunk_tokens"]):
                end = min(begin + cfg["chunk_tokens"], census_tokens)
                idx = np.asarray(indices[begin:end], dtype=np.int64)
                val = np.asarray(acts[begin:end], dtype=np.float64)
                flat_idx, flat_val = idx.reshape(-1), val.reshape(-1)
                firing += np.bincount(flat_idx, minlength=cfg["num_latents"])
                energy += np.bincount(flat_idx, weights=flat_val * flat_val, minlength=cfg["num_latents"])
                signed_sum += np.bincount(flat_idx, weights=flat_val, minlength=cfg["num_latents"])
                docs = np.repeat(doc_assignment[begin:end], cfg["k"])
                valid = docs >= 0
                np.add.at(doc_energy, (docs[valid], flat_idx[valid]), (flat_val[valid] ** 2))
            active_documents = np.count_nonzero(doc_energy > 0, axis=0)
            doc_energy_sum = doc_energy.sum(axis=0)
            doc_energy_square = np.square(doc_energy).sum(axis=0)
            ess = np.divide(np.square(doc_energy_sum), doc_energy_square, out=np.zeros_like(doc_energy_sum), where=doc_energy_square > 0)
            for atom in range(cfg["num_latents"]):
                rows_out.append({"seed": seed, "atom": atom, "mean_code": float(mean_sum[atom] / mean_tokens),
                    "discovery_firing_count": int(firing[atom]), "discovery_firing_rate": float(firing[atom] / census_tokens),
                    "discovery_signed_code_mean": float(signed_sum[atom] / census_tokens), "discovery_code_energy": float(energy[atom]),
                    "active_document_count": int(active_documents[atom]), "document_energy_ess": float(ess[atom])})
            seed_summaries.append({"seed": seed, "alive_atoms": int(np.count_nonzero(firing)), "minimum_firing": int(firing.min()),
                "minimum_active_documents": int(active_documents.min()), "minimum_document_ess": float(ess.min()), "total_code_energy": float(energy.sum())})
        census_path = run_dir / "source_census.jsonl"
        with census_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows_out:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        checks = {"full_five_seed_grid": len(rows_out) == 5 * cfg["num_latents"], "all_atoms_alive": all(row["alive_atoms"] == cfg["num_latents"] for row in seed_summaries),
            "document_assignment_available": len(document_ids) > 0 and np.count_nonzero(doc_assignment >= 0) > 0,
            "source_only_splits": cfg["mean_split"] == "mean" and cfg["census_split"] == "discovery" and set(cfg["forbidden_splits"]) == {"calibration", "audit"},
            "no_query_selection": cfg["query_selection"] == "not_performed_full_source_universe_only", "metrics_finite": bool(all(bool(np.isfinite([row["mean_code"], row["discovery_code_energy"], row["document_energy_ess"]]).all()) for row in rows_out)),
            "audit_not_opened": True}
        record = {"checks": checks, "row_count": len(rows_out), "source_seeds": cfg["source_seeds"], "num_latents": cfg["num_latents"],
            "mean_tokens": mean_tokens, "discovery_tokens": census_tokens, "discovery_documents": len(document_ids), "seed_summaries": seed_summaries,
            "source_census_sha256": sha256(census_path)}
        checks = {name: bool(value) for name, value in checks.items()}
        record["checks"] = checks
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r009a_source_census.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists(): (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
