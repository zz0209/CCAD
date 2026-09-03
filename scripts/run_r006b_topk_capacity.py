"""R006-B single-config TopK capacity and CE-recovered pipeline benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.checkpointing import canonicalize_sparsify_multiseed_state, save_sparsify_exact_state  # noqa: E402
from ccad.sae_quality import ce_recovered  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_hash(value) -> str:
    value = value.detach().cpu().contiguous()
    payload = str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(tensor_hash(state[name]).encode())
    return digest.hexdigest()


def file_entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


class MemmapTokens:
    def __init__(self, path: Path, context_length: int, max_sequences: int):
        import numpy as np
        self.rows = np.memmap(path, dtype="<u2", mode="r").reshape(-1, context_length)[:max_sequences]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        return {"input_ids": torch.from_numpy(self.rows[index].astype("int64"))}

    def select(self, indices):
        result = MemmapTokens.__new__(MemmapTokens)
        result.rows = self.rows[indices.start:indices.stop]
        return result


def attach_model_trace(model):
    original = model.forward
    model.ccad_input_hashes = []

    def tracked(self, input_ids, *args, **kwargs):
        self.ccad_input_hashes.append(tensor_hash(input_ids))
        return original(input_ids, *args, **kwargs)

    model.forward = types.MethodType(tracked, model)


def attach_loss_trace(trainer):
    traces = {name: [] for name in trainer.saes}
    handles = []
    for name, sae in trainer.saes.items():
        def capture(_module, _inputs, output, sae_name=name):
            traces[sae_name].append({
                "fvu": float(output.fvu.detach().cpu()),
                "auxk_loss": float(output.auxk_loss.detach().cpu()),
            })
        handles.append(sae.register_forward_hook(capture))
    return traces, handles


def evaluate(model, sae, module, contract, oracle_index: int, dataset, batch_size: int, device, torch):
    totals = {"clean": 0.0, "capture": 0.0, "reconstruction": 0.0, "zero": 0.0}
    denominator = 0
    squared_error = 0.0
    activation_sum = torch.zeros(contract.hidden_size, dtype=torch.float64, device=device)
    activation_square_sum = 0.0
    nonzero_count = 0
    selected_count = 0
    activation_rows = 0
    alive = torch.zeros(sae.num_latents, dtype=torch.bool, device=device)
    firing_counts = torch.zeros(sae.num_latents, dtype=torch.int64, device=device)
    hook_oracle_error = 0.0
    capture_logit_error = 0.0

    def run_hook(batch, mode: str):
        observed = {}
        def hook(_module, _inputs, output):
            hidden = extract_primary_hook_tensor(output, contract)
            observed["hidden"] = hidden.detach()
            if mode == "capture":
                return output
            flat = hidden.flatten(0, 1)
            if mode == "reconstruction":
                encoded = sae.encode(flat)
                reconstructed = sae.decode(encoded.top_acts, encoded.top_indices).reshape_as(hidden)
                observed["latent_acts"] = encoded.top_acts.detach()
                observed["latent_indices"] = encoded.top_indices.detach()
                observed["reconstruction"] = reconstructed.detach()
                return replace_primary_hook_tensor(output, reconstructed.type_as(hidden), contract)
            if mode == "zero":
                return replace_primary_hook_tensor(output, torch.zeros_like(hidden), contract)
            raise ValueError(mode)
        handle = module.register_forward_hook(hook)
        try:
            result = model(batch, labels=batch, output_hidden_states=(mode == "capture"), use_cache=False)
        finally:
            handle.remove()
        return result, observed

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = torch.stack([dataset[index]["input_ids"] for index in range(start, min(start + batch_size, len(dataset)))]).to(device)
            weight = batch.shape[0] * (batch.shape[1] - 1)
            clean = model(batch, labels=batch, use_cache=False)
            capture, captured = run_hook(batch, "capture")
            reconstruction, reconstructed = run_hook(batch, "reconstruction")
            zero, _ = run_hook(batch, "zero")
            totals["clean"] += float(clean.loss) * weight
            totals["capture"] += float(capture.loss) * weight
            totals["reconstruction"] += float(reconstruction.loss) * weight
            totals["zero"] += float(zero.loss) * weight
            denominator += weight
            hidden = captured["hidden"].float()
            recon = reconstructed["reconstruction"].float()
            squared_error += float((hidden - recon).square().sum())
            hidden64 = hidden.double().reshape(-1, hidden.shape[-1])
            activation_sum += hidden64.sum(dim=0)
            activation_square_sum += float(hidden64.square().sum())
            acts = reconstructed["latent_acts"]
            indices = reconstructed["latent_indices"]
            nonzero = acts != 0
            nonzero_count += int(nonzero.sum())
            selected_count += int(indices.numel())
            activation_rows += acts.shape[0]
            if nonzero.any():
                alive[indices[nonzero]] = True
                firing_counts += torch.bincount(indices[nonzero].reshape(-1), minlength=sae.num_latents)
            hook_oracle_error = max(hook_oracle_error, float((hidden - capture.hidden_states[oracle_index]).abs().max()))
            capture_logit_error = max(capture_logit_error, float((clean.logits - capture.logits).abs().max()))
    ce = {key: value / denominator for key, value in totals.items()}
    damage = ce["zero"] - ce["clean"]
    recovered = ce_recovered(ce["clean"], ce["reconstruction"], ce["zero"]) if damage > 0 else float("nan")
    total_variance = activation_square_sum - float(activation_sum.square().sum()) / activation_rows
    firing_float = firing_counts.float()
    nonzero_firing = firing_float[firing_float > 0]
    firing_distribution = {
        "dead_count": int((firing_counts == 0).sum()),
        "min": int(firing_counts.min()),
        "q25": float(torch.quantile(firing_float, 0.25)),
        "median": float(torch.quantile(firing_float, 0.50)),
        "q75": float(torch.quantile(firing_float, 0.75)),
        "q90": float(torch.quantile(firing_float, 0.90)),
        "q99": float(torch.quantile(firing_float, 0.99)),
        "max": int(firing_counts.max()),
        "nonzero_min": int(nonzero_firing.min()) if nonzero_firing.numel() else 0,
        "total_firings": int(firing_counts.sum()),
    }
    return {
        "ce": ce, "zero_ablation_damage": damage, "ce_recovered": recovered,
        "fve": 1.0 - squared_error / total_variance,
        "mse_sum": squared_error, "variance_sum": total_variance,
        "actual_nonzero_l0": nonzero_count / activation_rows,
        "selected_l0": selected_count / activation_rows,
        "alive_features": int(alive.sum()), "alive_fraction": float(alive.float().mean()),
        "feature_firing_count_distribution": firing_distribution,
        "activation_rows": activation_rows, "hook_oracle_max_error": hook_oracle_error,
        "capture_logit_max_error": capture_logit_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_text("", encoding="utf-8")
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), args.config.resolve(), ROOT / "src/ccad/checkpointing.py", ROOT / "src/ccad/activation_contract.py", ROOT / "src/ccad/artifacts.py", ROOT / "src/ccad/sae_quality.py"]
    code_entries = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = hashlib.sha256("".join(f"{item['path']}:{item['sha256']}\n" for item in sorted(code_entries, key=lambda x: x["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    model_dir = Path(cfg["model_local_dir"])
    train_path = ROOT / cfg["train_token_path"]
    validation_path = ROOT / cfg["validation_token_path"]
    token_manifest = ROOT / cfg["token_manifest_path"]
    env_lock = ROOT / cfg["environment_lock"]
    inputs = [
        file_entry(train_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_train_tokens"),
        file_entry(validation_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_validation_tokens"),
        file_entry(token_manifest, "CCAD R006a capacity token manifest", "internal provenance artifact", "token_manifest"),
        file_entry(env_lock, "CCAD environment lock", "internal artifact", "environment_lock"),
        file_entry(Path(cfg["sparsify_source_dir"]) / "sparsify/trainer.py", f"sparsify@{cfg['sparsify_commit']}", "MIT", "trainer_source"),
    ]
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.name != "pytorch_model.bin":
            inputs.append(file_entry(path, f"Hugging Face {cfg['model_id']}@{cfg['model_revision']}", cfg["model_license"], "model_or_tokenizer"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started.isoformat(), "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"],
        "seeds": {"init": cfg["init_seeds"], "data_order": cfg["data_order_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "Pythia-160M training and intervention evaluation",
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "tokenizer_revision": cfg["tokenizer_revision"],
        "git_status": "project directory is not a Git repository; exact code/input hashes recorded",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    record = None
    error = None
    status = "FAIL"
    try:
        if sha256(train_path) != cfg["train_token_sha256"] or sha256(validation_path) != cfg["validation_token_sha256"]:
            raise ValueError("capacity token hash mismatch")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DISABLED": "true", "SPARSIFY_DISABLE_TRITON": "1"})
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import torch
        import transformers
        from sparsify import SaeConfig, TrainConfig, Trainer
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("high")
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        train_data = MemmapTokens(train_path, cfg["context_length"], cfg["max_train_sequences"])
        validation_data = MemmapTokens(validation_path, cfg["context_length"], cfg["max_validation_sequences"])
        expected_steps = len(train_data) // cfg["batch_size_sequences"]
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        attach_model_trace(model)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_start
        train_cfg = TrainConfig(
            sae=SaeConfig(activation="topk", num_latents=cfg["num_latents"], k=cfg["k"], normalize_decoder=True),
            batch_size=cfg["batch_size_sequences"], optimizer="adam", lr=cfg["learning_rate"],
            lr_warmup_steps=cfg["lr_warmup_steps"], hookpoints=[cfg["hookpoint"]], init_seeds=cfg["init_seeds"],
            save_every=expected_steps + 1, dead_feature_threshold=cfg["dead_feature_threshold"], auxk_alpha=cfg["auxk_alpha"],
            exclude_tokens=cfg["exclude_token_ids"], save_best=False, log_to_wandb=False,
            run_name=cfg["run_id"], save_dir=str(run_dir / "upstream_unused"),
        )
        trainer = Trainer(train_cfg, train_data, model)
        canonicalize_sparsify_multiseed_state(trainer)
        traces, handles = attach_loss_trace(trainer)
        model.ccad_input_hashes = []
        train_start = time.perf_counter()
        trainer.fit()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        for handle in handles:
            handle.remove()
        training_input_hashes = list(model.ccad_input_hashes)
        sae_name, sae = next(iter(trainer.saes.items()))
        checkpoint_dir = run_dir / "exact_checkpoint"
        checkpoint_meta = save_sparsify_exact_state(trainer, checkpoint_dir, data_cursor_examples=len(train_data))
        safe_dir = run_dir / "sae"
        sae.save_to_disk(safe_dir)
        saved_state_hash = state_hash(sae.state_dict())
        contract = HookPointContract(cfg["hook_module_path"], cfg["layer_index"], "resid_post", model.config.hidden_size)
        evaluation = evaluate(model, sae, model.get_submodule(cfg["hook_module_path"]), contract, cfg["hook_oracle_hidden_state_index"], validation_data, cfg["eval_batch_size_sequences"], device, torch)
        decoder_norm_error = float((sae.W_dec.detach().float().norm(dim=1) - 1).abs().max())
        train_tokens = len(train_data) * cfg["context_length"]
        checks = {
            "train_token_hash": sha256(train_path) == cfg["train_token_sha256"],
            "validation_token_hash": sha256(validation_path) == cfg["validation_token_sha256"],
            "global_step_exact": trainer.global_step == expected_steps,
            "one_base_forward_per_train_step": len(training_input_hashes) == expected_steps,
            "loss_trace_complete": len(traces[sae_name]) == expected_steps,
            "safe_checkpoint_present": (safe_dir / "sae.safetensors").is_file() and (safe_dir / "cfg.json").is_file(),
            "exact_checkpoint_present": (checkpoint_dir / "state.pt").is_file(),
            "hook_oracle_exact": evaluation["hook_oracle_max_error"] == 0.0,
            "capture_logits_exact": evaluation["capture_logit_max_error"] == 0.0,
            "ce_denominator_positive": evaluation["zero_ablation_damage"] > 0,
            "metrics_finite": all(bool(torch.isfinite(torch.tensor(value))) for value in [evaluation["fve"], evaluation["ce_recovered"], evaluation["actual_nonzero_l0"]]),
            "selected_l0_exact": evaluation["selected_l0"] == cfg["k"],
        }
        record = {
            "checks": checks, "sae_name": sae_name, "state_hash": saved_state_hash,
            "checkpoint_metadata": checkpoint_meta, "checkpoint_state_sha256": sha256(checkpoint_dir / "state.pt"),
            "train_loss_trace": traces[sae_name], "training_input_hashes": training_input_hashes,
            "train_sequences": len(train_data), "train_tokens": train_tokens, "train_steps": expected_steps,
            "model_load_seconds": model_load_seconds, "train_seconds": train_seconds,
            "train_tokens_per_second": train_tokens / train_seconds,
            "seconds_per_million_tokens": 1_000_000 * train_seconds / train_tokens,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "safe_checkpoint_bytes": sum(path.stat().st_size for path in safe_dir.rglob("*") if path.is_file()),
            "exact_checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()),
            "decoder_norm_max_error": decoder_norm_error, "validation": evaluation,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "scope_limit": cfg["scope_limit"],
        "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r006b_topk_capacity.py",
        "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
