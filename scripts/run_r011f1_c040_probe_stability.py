"""Bounded amplitude-sweep diagnostic for the C040 v3 probe outlier."""
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

    source_dir = ROOT / "runs" / cfg["source_run"]
    token_manifest_path = ROOT / cfg["token_manifest_path"]
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    token_info = token_manifest["outputs"][cfg["split"]]
    token_path = ROOT / "runs" / cfg["paired_corpus_run"] / token_info["path"]
    paths = {
        "states": source_dir / "probe_states.jsonl",
        "observations": source_dir / "probe_observations.npz",
        "sketch": source_dir / "output_sketch.json",
        "source_metrics": source_dir / "metrics.raw.jsonl",
        "token_manifest": token_manifest_path,
        "tokens": token_path,
        "model_config": Path(cfg["model_local_dir"]) / "config.json",
    }
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/activation_contract.py"]
    code_rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in code_paths
    ]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    input_rows = [file_entry(args.config.resolve(), "CCAD diagnostic config", "protocol")]
    input_rows.extend([
        file_entry(paths["states"], cfg["source_run"], "source_state_ledger"),
        file_entry(paths["observations"], cfg["source_run"], "source_probe_observations"),
        file_entry(paths["sketch"], cfg["source_run"], "source_output_sketch"),
        file_entry(paths["source_metrics"], cfg["source_run"], "source_metrics"),
        file_entry(paths["token_manifest"], "R008a paired corpus", "token_manifest"),
        file_entry(paths["tokens"], "R008a paired corpus", "discovery_tokens"),
        file_entry(paths["model_config"], cfg["model_id"], "model_config", cfg["model_license"]),
    ])
    write_json(run_dir / "inputs.json", {"inputs": input_rows})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": False,
        "mean_constants_source_split": "not_applicable_diagnostic", "threshold_source_split": "prewritten_diagnostic_only",
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": [],
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded Pythia central-difference stability diagnostic",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})

    record, rows, error, status = None, [], None, "FAIL"
    try:
        expected = {
            "states": cfg["source_state_ledger_sha256"],
            "observations": cfg["source_probe_observations_sha256"],
            "sketch": cfg["source_output_sketch_sha256"],
            "source_metrics": cfg["source_metrics_raw_sha256"],
            "token_manifest": cfg["token_manifest_sha256"],
        }
        bound = {name: sha256(paths[name]).lower() == value.lower() for name, value in expected.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        if cfg["split"] != "discovery" or cfg["audit_opened"]:
            raise ValueError("diagnostic must remain discovery-only with audit closed")
        amplitudes = [float(value) for value in cfg["relative_amplitudes"]]
        if amplitudes != sorted(set(amplitudes)) or min(amplitudes) <= 0:
            raise ValueError("relative amplitudes must be unique positive sorted values")
        if cfg["reference_relative_amplitude"] not in amplitudes:
            raise ValueError("reference amplitude must be included in sweep")

        states = [json.loads(line) for line in paths["states"].read_text(encoding="utf-8").splitlines()]
        observations = np.load(paths["observations"])
        source_directions = observations["directions"].astype(np.float64)
        source_effects = observations["effects"].astype(np.float64)
        directions_per_state = source_directions.shape[0] // len(states)
        if directions_per_state * len(states) != source_directions.shape[0]:
            raise ValueError("source observation rows do not factor by state")
        selected = [int(value) for value in cfg["state_indices"]]
        if len(selected) != len(set(selected)) or any(value < 0 or value >= len(states) for value in selected):
            raise ValueError("invalid diagnostic state indices")
        sketch = json.loads(paths["sketch"].read_text(encoding="utf-8"))
        source_metrics = json.loads(paths["source_metrics"].read_text(encoding="utf-8"))
        hook_rms = float(source_metrics["discovery_hook_rms"])
        if not np.isfinite(hook_rms) or hook_rms <= 0:
            raise ValueError("invalid source discovery hook RMS")
        vocab_ids = np.asarray(sketch["vocab_ids"], dtype=np.int64)
        vocab_signs = np.asarray(sketch["signs"], dtype=np.float32)
        tokens = np.memmap(paths["tokens"], dtype="<u2", mode="r").reshape(token_info["sequences"], cfg["context_length"])

        source_selected_rows = []
        probes = []
        for state_index in selected:
            for direction_index in range(directions_per_state):
                source_row = state_index * directions_per_state + direction_index
                source_selected_rows.append(source_row)
                probes.append({
                    "state_index": state_index, "direction_index": direction_index,
                    "direction": source_directions[source_row], "source_effect": source_effects[source_row],
                })
        variants = [
            (probe_index, relative_amplitude, sign)
            for relative_amplitude in amplitudes for probe_index in range(len(probes)) for sign in (1, -1)
        ]
        outputs: dict[tuple[int, float, int], np.ndarray] = {}

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
        vocab_sign_tensor = torch.from_numpy(vocab_signs).to(device)
        started_compute = time.perf_counter()
        total_forwards = 0
        for begin in range(0, len(variants), cfg["variant_batch_size"]):
            chunk = variants[begin:begin + cfg["variant_batch_size"]]
            batch_np = np.stack([
                np.asarray(tokens[states[probes[probe_index]["state_index"]]["sequence_index"]], dtype=np.int64)
                for probe_index, _, _ in chunk
            ])
            batch = torch.from_numpy(batch_np).to(device)
            positions = torch.tensor([
                states[probes[probe_index]["state_index"]]["token_position"] for probe_index, _, _ in chunk
            ], device=device, dtype=torch.long)
            direction_tensor = torch.from_numpy(np.stack([
                probes[probe_index]["direction"] for probe_index, _, _ in chunk
            ]).astype(np.float32)).to(device)
            signed_amplitude = torch.tensor([
                sign * relative_amplitude * hook_rms for _, relative_amplitude, sign in chunk
            ], device=device, dtype=torch.float32)

            def perturb(_module, _inputs, output):
                primary = extract_primary_hook_tensor(output, contract)
                replacement = primary.clone()
                rows_tensor = torch.arange(replacement.shape[0], device=device)
                replacement[rows_tensor, positions] += signed_amplitude[:, None] * direction_tensor
                return replace_primary_hook_tensor(output, replacement, contract)

            handle = hook_module.register_forward_hook(perturb)
            try:
                with torch.no_grad():
                    logits = model(batch, use_cache=False).logits
            finally:
                handle.remove()
            batch_rows = torch.arange(logits.shape[0], device=device)
            values = (logits[batch_rows, positions][:, vocab_tensor] * vocab_sign_tensor[None, :]).detach().float().cpu().numpy().astype(np.float64)
            for row_index, key in enumerate(chunk):
                outputs[key] = values[row_index]
            total_forwards += 1
        elapsed = time.perf_counter() - started_compute

        for probe_index, probe in enumerate(probes):
            for relative_amplitude in amplitudes:
                absolute_amplitude = relative_amplitude * hook_rms
                effect = (outputs[(probe_index, relative_amplitude, 1)] - outputs[(probe_index, relative_amplitude, -1)]) / (2.0 * absolute_amplitude)
                source_effect = probe["source_effect"]
                relative_rms = float(np.sqrt(np.mean((effect - source_effect) ** 2)) / max(np.sqrt(np.mean(source_effect ** 2)), np.finfo(float).eps))
                state = states[probe["state_index"]]
                token_row = tokens[state["sequence_index"]]
                position = state["token_position"]
                rows.append({
                    "state_index": probe["state_index"], "state_role": cfg["state_roles"][str(probe["state_index"])],
                    "sequence_index": state["sequence_index"], "token_position": position,
                    "documents_in_sequence": len(state["document_ids"]), "prior_token_id": int(token_row[position - 1]) if position else None,
                    "direction_index": probe["direction_index"], "relative_amplitude": relative_amplitude,
                    "absolute_amplitude": absolute_amplitude,
                    "effect_norm": float(np.linalg.norm(effect)), "source_v3_effect_norm": float(np.linalg.norm(source_effect)),
                    "source_v3_relative_rms": relative_rms,
                })

        reference = cfg["reference_relative_amplitude"]
        repeat_rows = [row for row in rows if row["relative_amplitude"] == reference]
        repeat_max = max(row["source_v3_relative_rms"] for row in repeat_rows)
        state_summary = {}
        for state_index in selected:
            state_rows = [row for row in rows if row["state_index"] == state_index]
            by_direction = []
            for direction_index in range(directions_per_state):
                norms = [row["effect_norm"] for row in state_rows if row["direction_index"] == direction_index]
                by_direction.append(max(norms) / min(norms))
            state_summary[str(state_index)] = {
                "role": cfg["state_roles"][str(state_index)],
                "effect_norm_min": min(row["effect_norm"] for row in state_rows),
                "effect_norm_median": float(np.median([row["effect_norm"] for row in state_rows])),
                "effect_norm_max": max(row["effect_norm"] for row in state_rows),
                "maximum_amplitude_ratio": max(by_direction),
            }
        extreme = state_summary["108"]
        controls = [state_summary[str(index)] for index in selected if index != 108]
        extreme_vs_control = extreme["effect_norm_min"] / max(row["effect_norm_max"] for row in controls)
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "discovery_only_audit_closed": cfg["split"] == "discovery" and not cfg["audit_opened"],
            "all_selected_states_rerun": len(rows) == len(selected) * directions_per_state * len(amplitudes),
            "finite_nonzero_derivatives": all(np.isfinite(row["effect_norm"]) and row["effect_norm"] > 0 for row in rows),
            "reference_repeat_within_tolerance": repeat_max <= cfg["repeat_relative_rms_tolerance"],
        }
        checks = {key: bool(value) for key, value in checks.items()}
        record = {
            "checks": checks, "selected_states": selected, "relative_amplitudes": amplitudes,
            "directions_per_state": directions_per_state, "central_variants": len(variants), "model_forwards": total_forwards,
            "discovery_hook_rms": hook_rms,
            "reference_repeat_max_relative_rms": repeat_max, "state_summary": state_summary,
            "extreme_vs_control_min_to_max_ratio": extreme_vs_control,
            "extreme_is_amplitude_stable": extreme["maximum_amplitude_ratio"] <= cfg["extreme_amplitude_ratio_ceiling"],
            "extreme_is_separated_from_controls": extreme_vs_control >= cfg["extreme_vs_control_floor"],
            "diagnostic_interpretation": (
                "REPRODUCIBLE_STATE_LOCAL_SENSITIVITY"
                if extreme["maximum_amplitude_ratio"] <= cfg["extreme_amplitude_ratio_ceiling"]
                and extreme_vs_control >= cfg["extreme_vs_control_floor"]
                else "FINITE_DIFFERENCE_OR_STATE_LOCALITY_NOT_CONFIRMED"
            ),
            "wall_seconds": elapsed, "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "scope_limit": cfg["scope_limit"],
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
    detail_path = run_dir / "diagnostic_rows.jsonl"
    detail_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path),
        "diagnostic_rows_sha256": sha256(detail_path), "generator_script_path": "scripts/run_r011f1_c040_probe_stability.py",
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
