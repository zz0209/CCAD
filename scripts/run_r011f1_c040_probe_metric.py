"""Fit the discovery-only, query-agnostic C040 causal pullback metric."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.causal_metric_probe import hashed_vocab_sketch, rademacher_direction, select_document_balanced_states  # noqa: E402
from ccad.fuzzy_correspondence import fit_probe_metric  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> tuple[dict, Path | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    inherited = raw.get("inherits_config")
    if inherited is None:
        return raw, None
    if set(raw) != {"inherits_config", "overrides"}:
        raise ValueError("suffix config may contain only inherits_config and overrides")
    base_path = ROOT / inherited
    base = json.loads(base_path.read_text(encoding="utf-8"))
    merged = {**base, **raw["overrides"]}
    merged["inherited_config_path"] = inherited
    merged["inherited_config_sha256"] = sha256(base_path)
    return merged, base_path


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, source: str, role: str, boundary: str = "internal") -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg, inherited_config_path = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [
        Path(__file__).resolve(), ROOT / "src/ccad/causal_metric_probe.py",
        ROOT / "src/ccad/fuzzy_correspondence.py", ROOT / "src/ccad/activation_contract.py",
    ]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    paths = {
        "protocol": ROOT / cfg["protocol_config_path"],
        "token_manifest": ROOT / cfg["token_manifest_path"],
        "sequence_records": ROOT / cfg["sequence_records_path"],
        "raw_hook_manifest": Path(cfg["raw_hook_manifest_path"]),
        "model_config": Path(cfg["model_local_dir"]) / "config.json",
    }
    input_rows = [
        file_entry(args.config.resolve(), "CCAD frozen config", "protocol"),
    ]
    if inherited_config_path is not None:
        input_rows.append(file_entry(inherited_config_path, "CCAD frozen inherited config", "inherited_protocol"))
    input_rows.extend([
        file_entry(paths["protocol"], "R011-F1 pre-audit protocol", "parent_protocol"),
        file_entry(paths["token_manifest"], "R008a paired corpus", "token_manifest"),
        file_entry(paths["sequence_records"], "R008a paired corpus", "sequence_records"),
        file_entry(paths["raw_hook_manifest"], "R011-S1 shared hook asset", "raw_hook_manifest"),
        file_entry(paths["model_config"], cfg["model_id"], "model_config", cfg["model_license"]),
    ])
    write_json(run_dir / "inputs.json", {"inputs": input_rows})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": False, "candidate_family_frozen": False,
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": [],
        "resource_lease": "gpu-0 + cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded central-difference Pythia hook probes and 768-dimensional pullback eigendecomposition",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        expected = {
            "protocol": cfg["protocol_config_sha256"], "token_manifest": cfg["token_manifest_sha256"],
            "sequence_records": cfg["sequence_records_sha256"], "raw_hook_manifest": cfg["raw_hook_manifest_sha256"],
        }
        bound = {name: sha256(paths[name]).lower() == value.lower() for name, value in expected.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        parent = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        if parent["execution_enabled"] is not cfg["parent_execution_enabled_expected"] or parent["audit_opened"]:
            raise ValueError("parent protocol execution/audit boundary drift")
        if cfg["split"] != "discovery" or cfg["forbidden_splits"] != ["mean", "calibration", "audit"] or cfg["audit_opened"]:
            raise ValueError("C040 probe asset must be discovery-only")
        for field in ("probe_states", "probe_directions_per_state", "probe_relative_amplitude", "output_logit_sketch_dim", "probe_ridge_fraction", "metric_eigenvalue_relative_tolerance"):
            if cfg[field] != parent[field]:
                raise ValueError(f"protocol constant drift: {field}")

        sequence_payload = json.loads(paths["sequence_records"].read_text(encoding="utf-8"))
        states = select_document_balanced_states(
            sequence_payload["sequences"], split="discovery", count=cfg["probe_states"],
            token_positions=tuple(cfg["probe_token_positions"]), salt=cfg["probe_state_salt"],
        )
        token_manifest = json.loads(paths["token_manifest"].read_text(encoding="utf-8"))
        token_info = token_manifest["outputs"]["discovery"]
        token_path = ROOT / "runs" / cfg["paired_corpus_run"] / token_info["path"]
        tokens = np.memmap(token_path, dtype="<u2", mode="r").reshape(token_info["sequences"], cfg["context_length"])
        raw_manifest = json.loads(paths["raw_hook_manifest"].read_text(encoding="utf-8"))
        raw_row = next(row for row in raw_manifest["splits"] if row["split"] == "discovery")
        raw_hook = np.memmap(raw_row["path"], dtype="<f4", mode="r").reshape(raw_row["shape"])
        hook_rms = float(np.sqrt(np.mean(np.asarray(raw_hook, dtype=np.float64) ** 2)))
        epsilon = cfg["probe_relative_amplitude"] * hook_rms
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid discovery hook RMS")
        vocab_ids, vocab_signs = hashed_vocab_sketch(cfg["vocab_size"], cfg["output_logit_sketch_dim"], cfg["output_sketch_salt"])
        probes = []
        for state_index, state in enumerate(states):
            for direction_index in range(cfg["probe_directions_per_state"]):
                probes.append({
                    "probe_index": len(probes), "state_index": state_index, "direction_index": direction_index,
                    "direction": rademacher_direction(cfg["hook_hidden_size"], state["state_key"], direction_index, cfg["probe_direction_salt"]),
                })
        directions = np.stack([probe["direction"] for probe in probes])
        plus = np.empty((len(probes), cfg["output_logit_sketch_dim"]), dtype=np.float64)
        minus = np.empty_like(plus)

        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"]})
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_local_dir"], local_files_only=True, dtype=torch.float32,
            attn_implementation=cfg["attn_implementation"],
        ).eval().to(device)
        model.config.use_cache = False
        hook_module = model.get_submodule(cfg["hook_module_path"])
        contract = HookPointContract(cfg["hook_module_path"], 5, "resid_post", cfg["hook_hidden_size"])
        variants = [(probe["probe_index"], sign) for probe in probes for sign in (1, -1)]
        started_compute = time.perf_counter()
        total_forwards = 0
        vocab_tensor = torch.from_numpy(vocab_ids).to(device)
        vocab_sign_tensor = torch.from_numpy(vocab_signs.astype(np.float32)).to(device)
        for begin in range(0, len(variants), cfg["variant_batch_size"]):
            chunk = variants[begin:begin + cfg["variant_batch_size"]]
            batch_np = np.stack([np.asarray(tokens[states[probes[probe_index]["state_index"]]["sequence_index"]], dtype=np.int64) for probe_index, _ in chunk])
            batch = torch.from_numpy(batch_np).to(device)
            positions = torch.tensor([states[probes[probe_index]["state_index"]]["token_position"] for probe_index, _ in chunk], device=device, dtype=torch.long)
            direction_tensor = torch.from_numpy(np.stack([probes[probe_index]["direction"] for probe_index, _ in chunk]).astype(np.float32)).to(device)
            sign_tensor = torch.tensor([sign for _, sign in chunk], device=device, dtype=torch.float32)

            def perturb(_module, _inputs, output):
                primary = extract_primary_hook_tensor(output, contract)
                replacement = primary.clone()
                rows = torch.arange(replacement.shape[0], device=device)
                replacement[rows, positions] += sign_tensor[:, None] * epsilon * direction_tensor
                return replace_primary_hook_tensor(output, replacement, contract)

            handle = hook_module.register_forward_hook(perturb)
            try:
                with torch.no_grad():
                    logits = model(batch, use_cache=False).logits
            finally:
                handle.remove()
            rows = torch.arange(logits.shape[0], device=device)
            sketch = logits[rows, positions][:, vocab_tensor] * vocab_sign_tensor[None, :]
            values = sketch.detach().float().cpu().numpy().astype(np.float64)
            for row_index, (probe_index, sign) in enumerate(chunk):
                (plus if sign == 1 else minus)[probe_index] = values[row_index]
            total_forwards += 1
        effects = (plus - minus) / (2.0 * epsilon)
        fitted = fit_probe_metric(
            directions, effects, ridge_fraction=cfg["probe_ridge_fraction"],
            relative_tolerance=cfg["metric_eigenvalue_relative_tolerance"],
        )
        elapsed = time.perf_counter() - started_compute
        eigenvalues = np.linalg.eigvalsh(fitted.matrix)

        state_path = run_dir / "probe_states.jsonl"
        state_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states), encoding="utf-8")
        sketch_path = run_dir / "output_sketch.json"
        write_json(sketch_path, {"vocab_ids": vocab_ids.tolist(), "signs": vocab_signs.tolist(), "salt": cfg["output_sketch_salt"]})
        observations_path = run_dir / "probe_observations.npz"
        np.savez_compressed(observations_path, directions=directions.astype(np.float32), effects=effects.astype(np.float32))
        metric_path = run_dir / "causal_metric.npz"
        np.savez_compressed(metric_path, matrix=fitted.matrix.astype(np.float32), factor=fitted.factor.astype(np.float32), eigenvalues=eigenvalues.astype(np.float64))
        document_counts = {}
        for state in states:
            document_counts[state["blocking_document_id"]] = document_counts.get(state["blocking_document_id"], 0) + 1
        effect_norms = np.linalg.norm(effects, axis=1)
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "parent_execution_disabled": parent["execution_enabled"] is False,
            "discovery_only": cfg["split"] == "discovery" and cfg["forbidden_splits"] == ["mean", "calibration", "audit"],
            "exact_balanced_state_count": len(states) == cfg["probe_states"] and len({row["state_key"] for row in states}) == cfg["probe_states"],
            "document_blocking_nonconcentrated": max(document_counts.values()) <= 2,
            "complete_central_pairs": len(probes) == cfg["probe_states"] * cfg["probe_directions_per_state"] and np.all(np.isfinite(plus)) and np.all(np.isfinite(minus)),
            "unique_fixed_output_sketch": len(set(vocab_ids.tolist())) == cfg["output_logit_sketch_dim"],
            "finite_nonzero_effects": np.all(np.isfinite(effects)) and float(np.mean(effect_norms > 0)) == 1.0,
            "psd_trace_normalized_metric": eigenvalues[0] >= -1e-8 and abs(float(np.trace(fitted.matrix)) - cfg["hook_hidden_size"]) <= 1e-5 * cfg["hook_hidden_size"],
            "nonzero_numerical_rank": 0 < fitted.rank <= cfg["hook_hidden_size"],
            "audit_not_opened": not cfg["audit_opened"],
        }
        checks = {name: bool(value) for name, value in checks.items()}
        record = {
            "checks": checks, "probe_states": len(states), "blocked_documents": len(document_counts),
            "maximum_states_per_blocking_document": max(document_counts.values()), "probe_directions": len(probes),
            "central_variants": len(variants), "model_forwards": total_forwards, "discovery_hook_rms": hook_rms,
            "absolute_probe_amplitude": epsilon, "effect_norm_min": float(effect_norms.min()),
            "effect_norm_median": float(np.median(effect_norms)), "effect_norm_max": float(effect_norms.max()),
            "metric_rank": fitted.rank, "metric_explained_trace_fraction": fitted.explained_trace_fraction,
            "metric_trace": float(np.trace(fitted.matrix)), "metric_min_eigenvalue": float(eigenvalues[0]),
            "metric_max_eigenvalue": float(eigenvalues[-1]), "metric_effective_rank": float(np.exp(-np.sum((np.maximum(eigenvalues, 0) / np.sum(np.maximum(eigenvalues, 0))) * np.log(np.maximum(np.maximum(eigenvalues, 0) / np.sum(np.maximum(eigenvalues, 0)), 1e-300))))),
            "state_ledger_sha256": sha256(state_path), "output_sketch_sha256": sha256(sketch_path),
            "probe_observations_sha256": sha256(observations_path), "causal_metric_sha256": sha256(metric_path),
            "wall_seconds": elapsed, "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "platform": platform.platform()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path), "generator_script_path": "scripts/run_r011f1_c040_probe_metric.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
