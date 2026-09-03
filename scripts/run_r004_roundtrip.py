from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccad.activation_contract import (  # noqa: E402
    HookPointContract,
    build_token_alignment_record,
    extract_primary_hook_tensor,
    replace_primary_hook_tensor,
)
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def stable_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def aggregate_code_hash(entries: list[dict]) -> str:
    return hashlib.sha256(
        "".join(f"{item['path']}:{item['sha256']}\n" for item in sorted(entries, key=lambda x: x["path"])).encode()
    ).hexdigest()


def tensor_sha256(tensor) -> str:
    array = tensor.detach().contiguous().float().cpu().numpy().astype("<f4", copy=False)
    header = json.dumps({"dtype": "float32-le", "shape": list(array.shape), "order": "C"}, sort_keys=True).encode()
    digest = hashlib.sha256(header + b"\0" + array.tobytes(order="C"))
    return digest.hexdigest()


def max_abs(left, right) -> float:
    return float((left - right).abs().max().item())


def model_input_entries(model_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        entries.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "source": "Hugging Face EleutherAI/pythia-70m-deduped resolved snapshot",
            "license_or_access_boundary": "Apache-2.0 model card; public research checkpoint",
            "role": "model_or_tokenizer_asset",
        })
    if not entries:
        raise RuntimeError("model directory contains no files")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run_dir = args.output_root.resolve() / args.run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with stdout_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = datetime.now(timezone.utc)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_dir = Path(config["model_local_dir"]).resolve()
    code_files = [
        Path(__file__).resolve(),
        ROOT / "src" / "ccad" / "activation_contract.py",
        ROOT / "src" / "ccad" / "artifacts.py",
        args.config.resolve(),
    ]
    code_entries = [{
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    } for path in code_files]
    code_hash = aggregate_code_hash(code_entries)
    stable_json(run_dir / "config.resolved.json", config)
    stable_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    inputs = model_input_entries(model_dir)
    env_lock = ROOT / "configs" / "r004_environment_lock_candidate_v1.json"
    inputs.append({
        "path": str(env_lock.resolve()), "sha256": sha256(env_lock), "bytes": env_lock.stat().st_size,
        "source": "CCAD project-local environment lock", "license_or_access_boundary": "internal project artifact",
        "role": "environment_lock",
    })
    stable_json(run_dir / "inputs.json", {"inputs": inputs})
    stable_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": args.run_id, "run_parent": "R004",
        "purpose": config["purpose"], "milestone": "M0", "evidence_level": "real_model_debug_fixture",
        "started_utc": started.isoformat(), "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash,
        "audit_opened": config["audit_opened"], "candidate_family_frozen": config["candidate_family_frozen"],
        "mean_constants_source_split": config["mean_constants_source_split"],
        "threshold_source_split": config["threshold_source_split"], "statistics_unit": config["statistics_unit"],
        "device": "cuda:0", "seeds": {"torch": config["seed"], "fixture": config["fixture_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run",
        "resource_lease_reason": "real Pythia inference and hook writeback",
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "tokenizer_id": config["model_id"], "tokenizer_revision": config["model_revision"],
        "sae_framework": "manual_coordinate_fixture_not_trained_sae",
    })
    stable_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    log(f"START {args.run_id}")

    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
        import safetensors
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; R004 has no CPU fallback")
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32,
            attn_implementation=config["attn_implementation"],
        ).eval().to(device)
        model.config.use_cache = False
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        if model.config.hidden_size != config["hidden_size"] or model.config.num_hidden_layers != config["num_hidden_layers"]:
            raise RuntimeError("model architecture differs from frozen R004 contract")
        layer_index = config["layer_index"]
        module = model.gpt_neox.layers[layer_index]
        next_module = model.gpt_neox.layers[layer_index + 1]
        contract = HookPointContract(
            module_path=config["hook_module_path"], layer_index=layer_index,
            tensor_kind="resid_post", hidden_size=config["hidden_size"],
        )
        encoded = tokenizer(config["prompts"], padding=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in encoded.items()}
        token_records = []
        for index, document_id in enumerate(config["document_ids"]):
            token_records.append(build_token_alignment_record(
                dataset_id=config["dataset_id"], document_id=document_id,
                tokenizer_id=config["model_id"], tokenizer_revision=config["model_revision"],
                input_ids=tokens["input_ids"][index].tolist(), attention_mask=tokens["attention_mask"][index].tolist(),
            ))

        def forward_timed():
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
            torch.cuda.synchronize(device)
            return output, time.perf_counter() - start

        with torch.inference_mode():
            baseline, baseline_seconds = forward_timed()
            captured: dict[str, object] = {}

            def capture_hook(_module, _args, output):
                captured["tensor"] = extract_primary_hook_tensor(output, contract).detach().clone()

            handle = module.register_forward_hook(capture_hook)
            capture_output, capture_seconds = forward_timed()
            handle.remove()
            hook_tensor = captured["tensor"]
            hidden_oracle = baseline.hidden_states[layer_index + 1]

            def no_op_hook(_module, _args, output):
                primary = extract_primary_hook_tensor(output, contract)
                return replace_primary_hook_tensor(output, primary.clone(), contract)

            handle = module.register_forward_hook(no_op_hook)
            no_op_output, no_op_seconds = forward_timed()
            handle.remove()

            width = config["fixture_width"]
            decoder = torch.eye(config["hidden_size"], device=device, dtype=torch.float32)[:, :width]
            codes = hook_tensor[..., :width]
            contribution = codes @ decoder.T
            bias = torch.zeros(config["hidden_size"], device=device, dtype=torch.float32)
            reconstruction = bias + codes @ decoder.T
            residual = hook_tensor - reconstruction
            roundtrip_error = max_abs(reconstruction + residual, hook_tensor)

            def self_swap_hook(_module, _args, output):
                primary = extract_primary_hook_tensor(output, contract)
                local_codes = primary[..., :width]
                local_contribution = local_codes @ decoder.T
                replacement = primary - local_contribution + local_contribution
                return replace_primary_hook_tensor(output, replacement, contract)

            handle = module.register_forward_hook(self_swap_hook)
            self_swap_output, self_swap_seconds = forward_timed()
            handle.remove()

            intervention_state: dict[str, object] = {}

            def ablation_hook(_module, _args, output):
                primary = extract_primary_hook_tensor(output, contract)
                local_codes = primary[..., :width]
                local_contribution = local_codes @ decoder.T
                replacement = primary - local_contribution
                intervention_state["replacement"] = replacement.detach().clone()
                intervention_state["contribution"] = local_contribution.detach().clone()
                return replace_primary_hook_tensor(output, replacement, contract)

            def next_pre_hook(_module, args):
                intervention_state["next_input"] = args[0].detach().clone()

            ablation_handle = module.register_forward_hook(ablation_hook)
            next_handle = next_module.register_forward_pre_hook(next_pre_hook)
            intervention_output, intervention_seconds = forward_timed()
            next_handle.remove()
            ablation_handle.remove()

        tolerance = config["absolute_tolerance"]
        metrics = {
            "run_id": args.run_id,
            "metric_version": "r004-roundtrip-v1",
            "statistics_unit": "deterministic_two-document_fixture",
            "model_id": config["model_id"], "model_revision": config["model_revision"],
            "model_type": model.config.model_type, "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers, "hook_module_path": config["hook_module_path"],
            "hook_shape": list(hook_tensor.shape), "hook_dtype": str(hook_tensor.dtype),
            "hidden_oracle_max_error": max_abs(hook_tensor, hidden_oracle),
            "capture_logits_max_error": max_abs(capture_output.logits, baseline.logits),
            "no_op_logits_max_error": max_abs(no_op_output.logits, baseline.logits),
            "self_swap_logits_max_error": max_abs(self_swap_output.logits, baseline.logits),
            "roundtrip_max_error": roundtrip_error,
            "ablation_formula_max_error": max_abs(
                intervention_state["replacement"], hook_tensor - contribution
            ),
            "next_layer_input_max_error": max_abs(
                intervention_state["next_input"], intervention_state["replacement"]
            ),
            "intervention_logits_max_delta": max_abs(intervention_output.logits, baseline.logits),
            "valid_tokens": int(tokens["attention_mask"].sum().item()),
            "sequence_length": int(tokens["input_ids"].shape[1]),
            "token_record_hashes": [record.token_sha256 for record in token_records],
            "load_seconds": load_seconds,
            "baseline_forward_seconds": baseline_seconds,
            "capture_forward_seconds": capture_seconds,
            "no_op_forward_seconds": no_op_seconds,
            "self_swap_forward_seconds": self_swap_seconds,
            "intervention_forward_seconds": intervention_seconds,
            "baseline_tokens_per_second": int(tokens["attention_mask"].sum().item()) / baseline_seconds,
            "intervention_tokens_per_second": int(tokens["attention_mask"].sum().item()) / intervention_seconds,
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "activation_bytes_per_token": hook_tensor.element_size() * hook_tensor.shape[-1],
            "model_asset_bytes": sum(item["bytes"] for item in inputs if item["role"] == "model_or_tokenizer_asset"),
            "hook_tensor_sha256": tensor_sha256(hook_tensor),
            "baseline_logits_sha256": tensor_sha256(baseline.logits),
            "intervention_logits_sha256": tensor_sha256(intervention_output.logits),
        }
        checks = {
            "token_records_unique": len(set(metrics["token_record_hashes"])) == len(token_records),
            "hook_shape": metrics["hook_shape"][-1] == config["hidden_size"] and len(metrics["hook_shape"]) == 3,
            "hidden_oracle": metrics["hidden_oracle_max_error"] <= tolerance,
            "capture_logits": metrics["capture_logits_max_error"] <= tolerance,
            "no_op_logits": metrics["no_op_logits_max_error"] <= tolerance,
            "self_swap_logits": metrics["self_swap_logits_max_error"] <= tolerance,
            "roundtrip": metrics["roundtrip_max_error"] <= tolerance,
            "ablation_formula": metrics["ablation_formula_max_error"] <= tolerance,
            "next_layer_writeback": metrics["next_layer_input_max_error"] <= tolerance,
            "intervention_reaches_logits": metrics["intervention_logits_max_delta"] >= config["minimum_intervention_logit_delta"],
        }
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir()
        np.savez_compressed(
            artifact_dir / "tensors.npz",
            input_ids=tokens["input_ids"].cpu().numpy(), attention_mask=tokens["attention_mask"].cpu().numpy(),
            hook=hook_tensor.cpu().numpy(), baseline_logits=baseline.logits.cpu().numpy(),
            capture_logits=capture_output.logits.cpu().numpy(), no_op_logits=no_op_output.logits.cpu().numpy(),
            self_swap_logits=self_swap_output.logits.cpu().numpy(), intervention_logits=intervention_output.logits.cpu().numpy(),
        )
        stable_json(artifact_dir / "token_records.json", {"records": [record.__dict__ for record in token_records]})
        with (run_dir / "metrics.raw.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        summary = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
            "records": 1, "metrics_raw_sha256": sha256(run_dir / "metrics.raw.jsonl"),
            "generator_script_path": "scripts/run_r004_roundtrip.py",
            "generator_script_sha256": sha256(Path(__file__).resolve()),
            "tensor_artifact_sha256": sha256(artifact_dir / "tensors.npz"),
            "scope_warning": "M0 debug fixture only; not SAE quality or CCAD C1/C2 evidence.",
        }
        stable_json(run_dir / "metrics.summary.json", summary)
        stable_json(run_dir / "environment.json", {
            "os": platform.platform(), "python": sys.version, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "pytorch": torch.__version__, "transformers": transformers.__version__, "safetensors": safetensors.__version__,
            "sae_framework": "manual_coordinate_fixture_not_trained_sae",
        })
        ended = datetime.now(timezone.utc)
        stable_json(run_dir / "status.json", {"status": summary["status"], "updated_utc": ended.isoformat(), "ended_utc": ended.isoformat()})
        validation = validate_run_directory(run_dir)
        stable_json(run_dir / "contract.validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
        log(json.dumps({"run_id": args.run_id, "status": summary["status"], "contract_ok": validation.ok}, sort_keys=True))
        return 0 if summary["status"] == "PASS" and validation.ok else 1
    except Exception:
        error = traceback.format_exc()
        stderr_path.write_text(error, encoding="utf-8")
        ended = datetime.now(timezone.utc)
        if not (run_dir / "metrics.raw.jsonl").exists():
            (run_dir / "metrics.raw.jsonl").write_text("", encoding="utf-8")
        if not (run_dir / "metrics.summary.json").exists():
            stable_json(run_dir / "metrics.summary.json", {
                "status": "FAIL", "checks": {}, "checks_passed": 0, "checks_total": 0, "records": 0,
                "metrics_raw_sha256": sha256(run_dir / "metrics.raw.jsonl"),
                "generator_script_path": "scripts/run_r004_roundtrip.py",
                "generator_script_sha256": sha256(Path(__file__).resolve()),
            })
        stable_json(run_dir / "environment.json", {
            "os": platform.platform(), "python": sys.version, "numpy": np.__version__,
            "cuda": "unresolved_due_failure", "gpu": "unresolved_due_failure", "pytorch": "unresolved_due_failure",
            "transformers": "unresolved_due_failure", "safetensors": "unresolved_due_failure",
            "sae_framework": "manual_coordinate_fixture_not_trained_sae",
        })
        stable_json(run_dir / "status.json", {"status": "FAIL", "updated_utc": ended.isoformat(), "ended_utc": ended.isoformat()})
        log("FAIL; see stderr.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
