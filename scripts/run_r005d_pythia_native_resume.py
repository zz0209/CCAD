"""Native Pythia sparsify multi-seed exact-resume and cost smoke."""

from __future__ import annotations

import argparse
import hashlib
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
from ccad.checkpointing import (  # noqa: E402
    canonicalize_sparsify_multiseed_state,
    load_sparsify_exact_state,
    save_sparsify_exact_state,
)


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


def nested_equal(left, right) -> bool:
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def file_entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


class TokenDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"input_ids": self.rows[index]}

    def select(self, indices):
        return TokenDataset([self.rows[index] for index in indices])


def attach_model_trace(model):
    original = model.forward
    model.ccad_calls = 0
    model.ccad_fail_on_call = None
    model.ccad_trace = []

    def tracked(self, input_ids, *args, **kwargs):
        self.ccad_calls += 1
        if self.ccad_calls == self.ccad_fail_on_call:
            raise RuntimeError("expected R005d interruption")
        self.ccad_trace.append(tensor_hash(input_ids))
        return original(input_ids, *args, **kwargs)

    model.forward = types.MethodType(tracked, model)


def reset_model_trace(model, fail_on_call=None):
    model.ccad_calls = 0
    model.ccad_fail_on_call = fail_on_call
    model.ccad_trace = []


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


def remove_handles(handles):
    for handle in handles:
        handle.remove()


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
    code_paths = [Path(__file__).resolve(), args.config.resolve(), ROOT / "src/ccad/checkpointing.py", ROOT / "src/ccad/artifacts.py"]
    code_entries = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    aggregate = hashlib.sha256("".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(code_entries, key=lambda y: y["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": aggregate})
    model_dir = Path(cfg["model_local_dir"])
    source_manifest = ROOT / cfg["source_manifest"]
    env_lock = ROOT / cfg["environment_lock"]
    trainer_source = Path(cfg["sparsify_source_dir"]) / "sparsify/trainer.py"
    inputs = [
        file_entry(source_manifest, "CCAD debug text", cfg["source_manifest_license"], "debug_text_manifest"),
        file_entry(env_lock, "CCAD environment lock", "internal artifact", "environment_lock"),
        file_entry(trainer_source, f"sparsify@{cfg['sparsify_commit']}", "MIT", "wrapped_trainer"),
    ]
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.name != "pytorch_model.bin":
            inputs.append(file_entry(path, f"Hugging Face {cfg['model_id']}@{cfg['model_revision']}", cfg["model_license"], "model_or_tokenizer"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started.isoformat(), "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate, "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"],
        "seeds": {"init": cfg["init_seeds"], "data_order": cfg["data_order_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "native Pythia training/resume smoke",
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "tokenizer_revision": cfg["tokenizer_revision"],
        "git_status": "project directory is not a Git repository; exact code/input hashes recorded",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})

    record = None
    error = None
    status = "FAIL"
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DISABLED": "true", "SPARSIFY_DISABLE_TRITON": "1"})
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import torch
        import transformers
        from sparsify import SaeConfig, TrainConfig, Trainer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
        order = list(range(len(rows)))
        random.Random(cfg["data_order_seed"]).shuffle(order)
        texts = [rows[index]["text"] for index in order]
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        dataset = TokenDataset([row.clone() for row in encoded["input_ids"]])
        expected_steps = len(dataset) // cfg["batch_size_sequences"]
        if expected_steps != 8:
            raise RuntimeError(f"unexpected debug step count {expected_steps}")
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        attach_model_trace(model)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started

        def make_config(name):
            return TrainConfig(
                sae=SaeConfig(activation="topk", num_latents=cfg["num_latents"], k=cfg["k"], normalize_decoder=True),
                batch_size=cfg["batch_size_sequences"], optimizer="adam", lr=cfg["learning_rate"], lr_warmup_steps=0,
                hookpoints=[cfg["hookpoint"]], init_seeds=cfg["init_seeds"], save_every=1000,
                dead_feature_threshold=cfg["dead_feature_threshold"], auxk_alpha=cfg["auxk_alpha"],
                save_best=False, log_to_wandb=False, run_name=name, save_dir=str(run_dir / "upstream_unused"),
            )

        full = Trainer(make_config("full"), dataset, model)
        canonicalize_sparsify_multiseed_state(full)
        full_losses, handles = attach_loss_trace(full)
        reset_model_trace(model)
        full_started = time.perf_counter()
        full.fit()
        torch.cuda.synchronize()
        full_seconds = time.perf_counter() - full_started
        remove_handles(handles)
        full_input_trace = list(model.ccad_trace)
        full_rng = torch.get_rng_state().clone()
        full_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]

        interrupted = Trainer(make_config("interrupted"), dataset, model)
        canonicalize_sparsify_multiseed_state(interrupted)
        interrupted_losses, handles = attach_loss_trace(interrupted)
        reset_model_trace(model, cfg["interrupt_after_steps"] + 1)
        saw_interrupt = False
        interrupted_started = time.perf_counter()
        try:
            interrupted.fit()
        except RuntimeError as exc:
            if "expected R005d interruption" not in str(exc):
                raise
            saw_interrupt = True
        torch.cuda.synchronize()
        interrupted_seconds = time.perf_counter() - interrupted_started
        remove_handles(handles)
        interrupted_input_trace = list(model.ccad_trace)
        cursor = interrupted.global_step * cfg["batch_size_sequences"]
        checkpoint = run_dir / "exact_checkpoint"
        metadata = save_sparsify_exact_state(interrupted, checkpoint, cursor)

        resumed = Trainer(make_config("resumed"), dataset, model)
        canonicalize_sparsify_multiseed_state(resumed)
        resumed_losses, handles = attach_loss_trace(resumed)
        loaded = load_sparsify_exact_state(resumed, checkpoint, expected_data_cursor_examples=cursor)
        reset_model_trace(model)
        resumed_started = time.perf_counter()
        resumed.fit()
        torch.cuda.synchronize()
        resumed_seconds = time.perf_counter() - resumed_started
        remove_handles(handles)
        resumed_input_trace = list(model.ccad_trace)

        full_weights = {name: state_hash(sae.state_dict()) for name, sae in full.saes.items()}
        resumed_weights = {name: state_hash(sae.state_dict()) for name, sae in resumed.saes.items()}
        joined_losses = {name: interrupted_losses[name] + resumed_losses[name] for name in full.saes}
        resumed_cuda_rng = torch.cuda.get_rng_state_all()
        checks = {
            "interruption_after_expected_steps": saw_interrupt and interrupted.global_step == cfg["interrupt_after_steps"],
            "metadata_covers_seeded_saes": metadata["sae_names"] == sorted(full.saes) and metadata["counter_names"] == sorted(full.saes) and metadata["best_loss_names"] == sorted(full.saes),
            "data_cursor_exact": loaded["data_cursor_examples"] == cursor == cfg["interrupt_after_steps"] * cfg["batch_size_sequences"],
            "input_trace_exact": full_input_trace == interrupted_input_trace + resumed_input_trace,
            "loss_trace_exact": nested_equal(full_losses, joined_losses),
            "weights_exact": full_weights == resumed_weights,
            "optimizer_exact": nested_equal([x.state_dict() for x in full.optimizers], [x.state_dict() for x in resumed.optimizers]),
            "scheduler_exact": nested_equal([x.state_dict() for x in full.lr_schedulers], [x.state_dict() for x in resumed.lr_schedulers]),
            "counters_exact": nested_equal(full.num_tokens_since_fired, resumed.num_tokens_since_fired),
            "best_loss_exact": nested_equal(full.best_loss, resumed.best_loss),
            "cpu_rng_exact": bool(torch.equal(full_rng, torch.get_rng_state())),
            "cuda_rng_exact": nested_equal(full_cuda_rng, resumed_cuda_rng),
            "global_step_exact": full.global_step == resumed.global_step == expected_steps,
        }
        final_dead_fraction = {name: float((value > cfg["dead_feature_threshold"]).float().mean().cpu()) for name, value in full.num_tokens_since_fired.items()}
        record = {
            "checks": checks, "checkpoint_metadata": metadata, "checkpoint_state_sha256": sha256(checkpoint / "state.pt"),
            "full_weights": full_weights, "resumed_weights": resumed_weights, "full_loss_trace": full_losses,
            "full_input_trace": full_input_trace, "interrupted_input_trace": interrupted_input_trace, "resumed_input_trace": resumed_input_trace,
            "debug_valid_tokens": int((encoded["input_ids"] != tokenizer.pad_token_id).sum()), "sequence_count": len(dataset),
            "model_load_seconds": model_load_seconds, "full_train_seconds": full_seconds,
            "interrupted_seconds": interrupted_seconds, "resumed_seconds": resumed_seconds,
            "full_base_forwards": len(full_input_trace), "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)), "final_dead_fraction": final_dead_fraction,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)})
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
        "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r005d_pythia_native_resume.py",
        "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
