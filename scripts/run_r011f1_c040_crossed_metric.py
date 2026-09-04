"""Fit the discovery-only crossed shared-direction C040 causal metric."""
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
from ccad.causal_metric_probe import (  # noqa: E402
    hashed_vocab_sketch,
    orthonormal_probe_directions,
    select_boundary_safe_document_balanced_states,
)
from ccad.fuzzy_correspondence import fit_crossed_probe_metric  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_entry(path: Path, source: str, role: str, boundary: str = "internal") -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": boundary, "role": role,
    }


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def synthetic_preflight() -> dict:
    rng = np.random.default_rng(9404)
    basis, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    directions = basis.T
    jacobians = rng.normal(size=(5, 6, 8))
    effects = np.stack([directions @ jacobian.T for jacobian in jacobians])
    fitted = fit_crossed_probe_metric(directions, effects, ridge_fraction=0.0, relative_tolerance=1e-12)
    expected = np.mean([jacobian.T @ jacobian for jacobian in jacobians], axis=0)
    expected *= expected.shape[0] / np.trace(expected)
    recovery = float(np.linalg.norm(fitted.matrix - expected) / np.linalg.norm(expected))
    duplicated = fit_crossed_probe_metric(
        directions, np.repeat(effects, 2, axis=0), ridge_fraction=0.0, relative_tolerance=1e-12,
    )
    duplication = float(np.linalg.norm(fitted.matrix - duplicated.matrix) / np.linalg.norm(fitted.matrix))
    return {
        "state_varying_gram_relative_error": recovery,
        "balanced_duplication_relative_error": duplication,
        "pass": bool(recovery <= 1e-10 and duplication <= 1e-10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    protocol_path = ROOT / cfg["protocol_config_path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    resolved = {**cfg, "protocol_config_sha256": sha256(protocol_path)}
    write_json(run_dir / "config.resolved.json", resolved)

    code_paths = [
        Path(__file__).resolve(), ROOT / "src/ccad/causal_metric_probe.py",
        ROOT / "src/ccad/fuzzy_correspondence.py", ROOT / "src/ccad/activation_contract.py",
    ]
    code_rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in code_paths
    ]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})

    token_manifest_path = ROOT / cfg["token_manifest_path"]
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    token_info = token_manifest["outputs"][cfg["split"]]
    token_path = ROOT / "runs" / cfg["paired_corpus_run"] / token_info["path"]
    paths = {
        "protocol": protocol_path,
        "protocol_document": ROOT / protocol["protocol_document"],
        "token_manifest": token_manifest_path,
        "tokens": token_path,
        "sequence_records": ROOT / cfg["sequence_records_path"],
        "raw_hook_manifest": Path(cfg["raw_hook_manifest_path"]),
        "legacy_metric": ROOT / cfg["legacy_metric_path"],
        "model_config": Path(cfg["model_local_dir"]) / "config.json",
    }
    input_rows = [
        file_entry(args.config.resolve(), "CCAD run config", "run_protocol"),
        file_entry(paths["protocol"], "R011-F1 protocol v2", "parent_protocol"),
        file_entry(paths["protocol_document"], "R011-F1 protocol v2", "protocol_document"),
        file_entry(paths["token_manifest"], "R008a paired corpus", "token_manifest"),
        file_entry(paths["tokens"], "R008a paired corpus", "discovery_tokens"),
        file_entry(paths["sequence_records"], "R008a paired corpus", "sequence_records"),
        file_entry(paths["raw_hook_manifest"], "R011-S1 shared hook asset", "raw_hook_manifest"),
        file_entry(paths["legacy_metric"], "C040 v3 rejected scientific metric", "legacy_diagnostic_metric"),
        file_entry(paths["model_config"], cfg["model_id"], "model_config", cfg["model_license"]),
    ]
    write_json(run_dir / "inputs.json", {"inputs": input_rows})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": False, "candidate_family_frozen": False,
        "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"],
        "device": cfg["device"], "seeds": [], "resource_lease": "gpu-0 via SAE Lab resource_manager.run",
        "resource_lease_reason": "crossed full-basis Pythia hook central differences",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})

    record, error, status = None, None, "FAIL"
    try:
        expected_hashes = {
            "token_manifest": cfg["token_manifest_sha256"],
            "sequence_records": cfg["sequence_records_sha256"],
            "raw_hook_manifest": cfg["raw_hook_manifest_sha256"],
            "legacy_metric": cfg["legacy_metric_sha256"],
        }
        bound = {name: sha256(paths[name]).lower() == value.lower() for name, value in expected_hashes.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        if not protocol["execution_enabled"] or protocol["audit_opened"]:
            raise ValueError("protocol is not execution-enabled with audit closed")
        if cfg["split"] != "discovery" or cfg["forbidden_splits"] != ["mean", "calibration", "audit"] or cfg["audit_opened"]:
            raise ValueError("crossed C040 run must be discovery-only")
        if protocol["shared_probe_directions"] != cfg["hook_hidden_size"]:
            raise ValueError("crossed design must use a complete hook basis")
        if protocol["state_weighting"] != "equal" or protocol["metric_estimand"] != "document_balanced_mean_state_jacobian_gram":
            raise ValueError("crossed metric estimand drift")

        preflight = synthetic_preflight()
        if not preflight["pass"]:
            raise ValueError(f"synthetic preflight failed: {preflight}")
        sequences = json.loads(paths["sequence_records"].read_text(encoding="utf-8"))["sequences"]
        tokens = np.memmap(paths["tokens"], dtype="<u2", mode="r").reshape(token_info["sequences"], cfg["context_length"])
        states = select_boundary_safe_document_balanced_states(
            sequences, tokens, split=cfg["split"], count=protocol["probe_states"],
            token_positions=tuple(protocol["probe_token_positions"]), salt=protocol["probe_state_salt"],
            eot_token_id=protocol["eot_token_id"],
            minimum_tokens_after_boundary=protocol["minimum_tokens_after_causal_boundary"],
        )
        directions = orthonormal_probe_directions(cfg["hook_hidden_size"], protocol["probe_direction_salt"])
        orthonormal_error = float(np.max(np.abs(directions @ directions.T - np.eye(cfg["hook_hidden_size"]))))
        if orthonormal_error > 1e-10:
            raise ValueError(f"probe basis is not orthonormal: {orthonormal_error}")

        raw_manifest = json.loads(paths["raw_hook_manifest"].read_text(encoding="utf-8"))
        raw_row = next(row for row in raw_manifest["splits"] if row["split"] == "discovery")
        raw_hook = np.memmap(raw_row["path"], dtype="<f4", mode="r").reshape(raw_row["shape"])
        hook_rms = float(np.sqrt(np.mean(np.asarray(raw_hook, dtype=np.float64) ** 2)))
        epsilon = protocol["probe_relative_amplitude"] * hook_rms
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid discovery hook RMS")
        vocab_ids, vocab_signs = hashed_vocab_sketch(
            cfg["vocab_size"], protocol["output_logit_sketch_dim"], protocol["output_sketch_salt"],
        )
        effects = np.empty(
            (len(states), len(directions), protocol["output_logit_sketch_dim"]), dtype=np.float32,
        )
        variants = [
            (state_index, direction_index, sign)
            for state_index in range(len(states))
            for direction_index in range(len(directions))
            for sign in (1, -1)
        ]
        plus: dict[tuple[int, int], np.ndarray] = {}

        os.environ.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
            "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"],
        })
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_local_dir"], local_files_only=True, dtype=torch.float32,
            attn_implementation=cfg["attn_implementation"],
        ).eval().to(device)
        model.config.use_cache = False
        hook_module = model.get_submodule(cfg["hook_module_path"])
        contract = HookPointContract(cfg["hook_module_path"], 5, "resid_post", cfg["hook_hidden_size"])
        vocab_tensor = torch.from_numpy(vocab_ids).to(device)
        vocab_sign_tensor = torch.from_numpy(vocab_signs.astype(np.float32)).to(device)
        selected_tokens = [np.asarray(tokens[state["sequence_index"]], dtype=np.int64) for state in states]
        started_compute = time.perf_counter()
        total_forwards = 0
        for begin in range(0, len(variants), protocol["variant_batch_size"]):
            chunk = variants[begin:begin + protocol["variant_batch_size"]]
            batch = torch.from_numpy(np.stack([selected_tokens[state_index] for state_index, _, _ in chunk])).to(device)
            positions = torch.tensor(
                [states[state_index]["token_position"] for state_index, _, _ in chunk], device=device, dtype=torch.long,
            )
            direction_tensor = torch.from_numpy(
                np.stack([directions[direction_index] for _, direction_index, _ in chunk]).astype(np.float32),
            ).to(device)
            sign_tensor = torch.tensor([sign for _, _, sign in chunk], device=device, dtype=torch.float32)

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
            values = (
                logits[rows, positions][:, vocab_tensor] * vocab_sign_tensor[None, :]
            ).detach().float().cpu().numpy().astype(np.float32)
            for row_index, (state_index, direction_index, sign) in enumerate(chunk):
                key = (state_index, direction_index)
                if sign == 1:
                    plus[key] = values[row_index]
                else:
                    positive = plus.pop(key)
                    effects[state_index, direction_index] = (positive - values[row_index]) / (2.0 * epsilon)
            total_forwards += 1
        elapsed = time.perf_counter() - started_compute
        if plus:
            raise ValueError("incomplete central-difference pairs")

        fitted = fit_crossed_probe_metric(
            directions, effects, ridge_fraction=protocol["probe_ridge_fraction"],
            relative_tolerance=protocol["metric_eigenvalue_relative_tolerance"],
        )
        eigenvalues = np.linalg.eigvalsh(fitted.matrix)
        positive_eigenvalues = np.maximum(eigenvalues, 0)
        eigen_shares = positive_eigenvalues / np.sum(positive_eigenvalues)
        metric_effective_rank = float(np.exp(-np.sum(eigen_shares * np.log(np.maximum(eigen_shares, 1e-300)))))
        state_energy = np.sum(effects.astype(np.float64) ** 2, axis=(1, 2))
        state_shares = state_energy / np.sum(state_energy)
        effective_state_count = float(1.0 / np.sum(state_shares ** 2))
        legacy = np.load(paths["legacy_metric"])
        legacy_factor = legacy["factor"].astype(np.float64)
        current_q, _ = np.linalg.qr(fitted.factor)
        legacy_q, _ = np.linalg.qr(legacy_factor)
        cross_metric_psc = float(np.linalg.norm(current_q.T @ legacy_q, ord="fro") ** 2 / min(current_q.shape[1], legacy_q.shape[1]))

        state_path = run_dir / "probe_states.jsonl"
        state_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states), encoding="utf-8")
        sketch_path = run_dir / "output_sketch.json"
        write_json(sketch_path, {"vocab_ids": vocab_ids.tolist(), "signs": vocab_signs.tolist(), "salt": protocol["output_sketch_salt"]})
        observations_path = run_dir / "probe_observations.npz"
        np.savez_compressed(observations_path, directions=directions.astype(np.float32), effects=effects)
        metric_path = run_dir / "causal_metric.npz"
        np.savez_compressed(
            metric_path, matrix=fitted.matrix.astype(np.float32), factor=fitted.factor.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float64), state_trace_shares=state_shares.astype(np.float64),
        )
        write_json(run_dir / "synthetic_preflight.json", preflight)
        document_counts: dict[str, int] = {}
        for state in states:
            document_counts[state["blocking_document_id"]] = document_counts.get(state["blocking_document_id"], 0) + 1
        effect_norms = np.linalg.norm(effects.astype(np.float64), axis=2)
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "protocol_execution_enabled_audit_closed": protocol["execution_enabled"] and not protocol["audit_opened"],
            "discovery_only": cfg["split"] == "discovery" and cfg["forbidden_splits"] == ["mean", "calibration", "audit"],
            "synthetic_state_varying_gram_and_duplication": preflight["pass"],
            "complete_orthonormal_shared_basis": len(directions) == cfg["hook_hidden_size"] and orthonormal_error <= 1e-10,
            "exact_boundary_safe_state_count": len(states) == protocol["probe_states"] and min(row["tokens_since_causal_boundary"] for row in states) >= protocol["minimum_tokens_after_causal_boundary"],
            "document_blocking_nonconcentrated": max(document_counts.values()) == 1,
            "complete_central_pairs": effects.shape == (protocol["probe_states"], protocol["shared_probe_directions"], protocol["output_logit_sketch_dim"]),
            "finite_nonzero_effects": np.all(np.isfinite(effects)) and bool(np.all(np.linalg.norm(effects, axis=2) > 0)),
            "psd_trace_normalized_metric": eigenvalues[0] >= -1e-8 and abs(float(np.trace(fitted.matrix)) - cfg["hook_hidden_size"]) <= 1e-5 * cfg["hook_hidden_size"],
            "nonzero_numerical_rank": 0 < fitted.rank <= cfg["hook_hidden_size"],
            "state_influence_max_share": float(state_shares.max()) <= protocol["maximum_state_trace_share"],
            "state_influence_effective_count": effective_state_count >= protocol["minimum_effective_state_count"],
            "audit_not_opened": not cfg["audit_opened"],
        }
        checks = {name: bool(value) for name, value in checks.items()}
        record = {
            "checks": checks, "synthetic_preflight": preflight, "probe_states": len(states),
            "blocked_documents": len(document_counts), "minimum_tokens_since_causal_boundary": min(row["tokens_since_causal_boundary"] for row in states),
            "shared_probe_directions": len(directions), "central_variants": len(variants), "model_forwards": total_forwards,
            "discovery_hook_rms": hook_rms, "absolute_probe_amplitude": epsilon,
            "effect_norm_min": float(effect_norms.min()), "effect_norm_median": float(np.median(effect_norms)),
            "effect_norm_max": float(effect_norms.max()), "maximum_state_trace_share": float(state_shares.max()),
            "effective_state_count": effective_state_count, "metric_rank": fitted.rank,
            "metric_effective_rank": metric_effective_rank, "metric_explained_trace_fraction": fitted.explained_trace_fraction,
            "metric_trace": float(np.trace(fitted.matrix)), "metric_min_eigenvalue": float(eigenvalues[0]),
            "metric_max_eigenvalue": float(eigenvalues[-1]), "legacy_metric_projector_similarity": cross_metric_psc,
            "orthonormal_basis_max_error": orthonormal_error, "state_ledger_sha256": sha256(state_path),
            "output_sketch_sha256": sha256(sketch_path), "probe_observations_sha256": sha256(observations_path),
            "causal_metric_sha256": sha256(metric_path), "wall_seconds": elapsed,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)), "scope_limit": cfg["scope_limit"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "platform": platform.platform(),
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})

    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path),
        "generator_script_path": "scripts/run_r011f1_c040_crossed_metric.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    })
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
