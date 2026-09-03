"""Dictionary-learning native PyTorch activation-buffer control and cost smoke."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_hash(value) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


def file_entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


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
    code_paths = [Path(__file__).resolve(), args.config.resolve(), ROOT / "src/ccad/artifacts.py"]
    entries = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    aggregate = hashlib.sha256("".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(entries, key=lambda y: y["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": entries, "aggregate_sha256": aggregate})
    model_dir = Path(cfg["model_local_dir"])
    source_manifest = ROOT / cfg["source_manifest"]
    env_lock = ROOT / cfg["environment_lock"]
    buffer_source = Path(cfg["dictionary_source_dir"]) / "dictionary_learning/pytorch_buffer.py"
    training_source = Path(cfg["dictionary_source_dir"]) / "dictionary_learning/training.py"
    inputs = [file_entry(source_manifest, "CCAD debug text", cfg["source_manifest_license"], "debug_text_manifest"),
              file_entry(env_lock, "CCAD environment lock", "internal artifact", "environment_lock"),
              file_entry(buffer_source, f"dictionary_learning@{cfg['dictionary_commit']}", "MIT", "native_buffer"),
              file_entry(training_source, f"dictionary_learning@{cfg['dictionary_commit']}", "MIT", "native_training")]
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.name != "pytorch_model.bin":
            inputs.append(file_entry(path, f"Hugging Face {cfg['model_id']}@{cfg['model_revision']}", cfg["model_license"], "model_or_tokenizer"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started.isoformat(),
        "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": aggregate,
        "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"],
        "seeds": {"init": cfg["init_seeds"], "data_order": cfg["data_order_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "Pythia native activation buffer smoke",
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "tokenizer_revision": cfg["tokenizer_revision"],
        "git_status": "project directory is not a Git repository; exact code/input hashes recorded",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    record = None
    error = None
    status = "FAIL"
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DISABLED": "true"})
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
        sys.path[:0] = [cfg["dictionary_source_dir"], cfg["dictionary_overlay_dir"]]
        import torch
        import transformers
        from dictionary_learning.pytorch_buffer import ActivationBuffer
        from dictionary_learning.training import trainSAE
        from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM

        class IsolatedBuffer(ActivationBuffer):
            def __init__(self, *buffer_args, sampler_seed: int, **buffer_kwargs):
                super().__init__(*buffer_args, **buffer_kwargs)
                self.sampler = torch.Generator(device=self.device)
                self.sampler.manual_seed(sampler_seed)
                self.batch_hashes = []

            def __next__(self):
                with torch.no_grad():
                    if (~self.read).sum() < self.activation_buffer_size // 2:
                        self.refresh()
                    unreads = (~self.read).nonzero().squeeze()
                    idxs = unreads[torch.randperm(len(unreads), device=unreads.device, generator=self.sampler)[:self.out_batch_size]]
                    self.read[idxs] = True
                    batch = self.activations[idxs]
                    self.batch_hashes.append(tensor_hash(batch))
                    return batch

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval().to(device)
        model.config.use_cache = False
        original_forward = model.forward
        model.ccad_trace = []
        def tracked(self, input_ids=None, *forward_args, **forward_kwargs):
            self.ccad_trace.append(tensor_hash(input_ids))
            return original_forward(input_ids=input_ids, *forward_args, **forward_kwargs)
        model.forward = types.MethodType(tracked, model)
        rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
        order = list(range(len(rows)))
        random.Random(cfg["data_order_seed"]).shuffle(order)
        texts = [rows[index]["text"] for index in order]
        text_iter = iter(texts)
        buffer = IsolatedBuffer(
            text_iter, model, model.gpt_neox.layers[cfg["layer_index"]], d_submodule=cfg["hidden_size"], io="out",
            n_ctxs=cfg["n_ctxs"], ctx_len=cfg["ctx_len"], refresh_batch_size=cfg["refresh_batch_size"],
            out_batch_size=cfg["out_batch_size"], device=cfg["device"], sampler_seed=cfg["data_order_seed"],
        )
        expected_trace = []
        for start in range(0, len(texts), cfg["refresh_batch_size"]):
            tokenized = buffer.tokenizer(texts[start:start + cfg["refresh_batch_size"]], return_tensors="pt", max_length=cfg["ctx_len"], padding=True, truncation=True, add_special_tokens=True)
            expected_trace.append(tensor_hash(tokenized["input_ids"]))

        configs = []
        for seed in cfg["init_seeds"]:
            configs.append({"trainer": TopKTrainer, "steps": cfg["training_steps"], "activation_dim": cfg["hidden_size"],
                            "dict_size": cfg["num_latents"], "k": cfg["k"], "layer": cfg["layer_index"],
                            "lm_name": cfg["model_id"], "lr": cfg["learning_rate"], "warmup_steps": 0,
                            "seed": seed, "device": cfg["device"]})
        train_started = time.perf_counter()
        bounded_buffer = itertools.islice(buffer, cfg["training_steps"])
        trainSAE(bounded_buffer, configs, steps=cfg["training_steps"], use_wandb=False, save_dir=str(run_dir / "native_output"),
                 normalize_activations=False, verbose=False, device=cfg["device"], autocast_dtype=torch.float32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - train_started
        safe_hashes = {}
        diagnostics = {}
        eval_batch = buffer.activations[:cfg["out_batch_size"]]
        for index, seed in enumerate(cfg["init_seeds"]):
            native_path = run_dir / "native_output" / f"trainer_{index}" / "ae.pt"
            state = torch.load(native_path, map_location="cpu", weights_only=True)
            safe_path = run_dir / "native_output" / f"trainer_{index}" / "sae.safetensors"
            save_file({name: value.contiguous() for name, value in state.items()}, str(safe_path))
            safe_hashes[str(seed)] = sha256(safe_path)
            ae = AutoEncoderTopK(cfg["hidden_size"], cfg["num_latents"], cfg["k"]).to(device)
            ae.load_state_dict(state)
            with torch.no_grad():
                code = ae.encode(eval_batch)
                reconstruction = ae.decode(code)
                mse = float((eval_batch - reconstruction).square().mean())
                variance = float((eval_batch - eval_batch.mean(0, keepdim=True)).square().mean())
            diagnostics[str(seed)] = {"l0_mean": float((code != 0).sum(-1).float().mean()), "mse": mse,
                                      "fve": 1.0 - mse / variance if variance > 0 else None}

        activation_pool = buffer.activations.detach()
        def sample_after_order(seed_order, isolated):
            holders = [TopKTrainer(steps=1, activation_dim=4, dict_size=8, k=2, layer=0, lm_name="rng-audit", lr=1e-3, warmup_steps=0, seed=seed, device=cfg["device"]) for seed in seed_order]
            unreads = torch.arange(len(activation_pool), device=device)
            if isolated:
                generator = torch.Generator(device=device).manual_seed(cfg["data_order_seed"])
                indices = unreads[torch.randperm(len(unreads), device=device, generator=generator)[:cfg["out_batch_size"]]]
            else:
                indices = unreads[torch.randperm(len(unreads), device=device)[:cfg["out_batch_size"]]]
            result = tensor_hash(activation_pool[indices])
            del holders
            return result

        forward_order = cfg["init_seeds"]
        reverse_order = list(reversed(forward_order))
        native_a = sample_after_order(forward_order, False)
        native_b = sample_after_order(reverse_order, False)
        isolated_a = sample_after_order(forward_order, True)
        isolated_b = sample_after_order(reverse_order, True)
        checks = {
            "fixed_model_input_trace": model.ccad_trace == expected_trace,
            "expected_base_forwards": len(model.ccad_trace) == len(texts) // cfg["refresh_batch_size"] == 8,
            "expected_activation_pool": len(buffer.activations) == cfg["n_ctxs"] * cfg["ctx_len"] == 208,
            "expected_training_batches": len(buffer.batch_hashes) == cfg["training_steps"],
            "native_seed_order_coupling_detected": native_a != native_b,
            "isolated_sampler_seed_order_independent": isolated_a == isolated_b,
            "two_safe_exports": len(safe_hashes) == 2 and len(set(safe_hashes.values())) == 2,
            "topk_l0_exact": all(value["l0_mean"] == cfg["k"] for value in diagnostics.values()),
        }
        record = {"checks": checks, "model_input_trace": model.ccad_trace, "activation_batch_hashes": buffer.batch_hashes,
                  "native_order_hashes": [native_a, native_b], "isolated_order_hashes": [isolated_a, isolated_b],
                  "safe_export_sha256": safe_hashes, "diagnostics": diagnostics, "base_forwards": len(model.ccad_trace),
                  "activation_pool_rows": len(buffer.activations), "activation_pool_bytes": buffer.activations.numel() * buffer.activations.element_size(),
                  "wall_seconds": elapsed, "valid_activation_rows_per_second": len(buffer.activations) / elapsed,
                  "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))}
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error,
        "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0,
        "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r005e_dictionary_native_buffer.py", "generator_script_sha256": entries[0]["sha256"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
