"""R006-B1 nested-milestone budget-stability trajectory."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ccad.activation_contract import HookPointContract  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.checkpointing import canonicalize_sparsify_multiseed_state, load_sparsify_exact_state, save_sparsify_exact_state  # noqa: E402
from ccad.decoder_diagnostics import pairwise_decoder_cosine_similarity  # noqa: E402
from ccad.sae_quality import budget_stability_checks  # noqa: E402
from run_r006b_topk_capacity import MemmapTokens, attach_loss_trace, evaluate, file_entry, state_hash, tensor_hash, write_json  # noqa: E402


INTERRUPT_MESSAGE = "expected R006-B1 milestone interruption"


def attach_interruptible_trace(model) -> None:
    original = model.forward
    model.ccad_calls = 0
    model.ccad_fail_on_call = None
    model.ccad_trace = []

    def tracked(self, input_ids, *args, **kwargs):
        self.ccad_calls += 1
        if self.ccad_calls == self.ccad_fail_on_call:
            raise RuntimeError(INTERRUPT_MESSAGE)
        self.ccad_trace.append(tensor_hash(input_ids))
        return original(input_ids, *args, **kwargs)

    model.forward = types.MethodType(tracked, model)


def reset_trace(model, fail_on_call: int | None) -> None:
    model.ccad_calls = 0
    model.ccad_fail_on_call = fail_on_call
    model.ccad_trace = []


def remove_handles(handles) -> None:
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

    code_paths = [
        Path(__file__).resolve(), args.config.resolve(), ROOT / "scripts/run_r006b_topk_capacity.py",
        ROOT / "src/ccad/checkpointing.py", ROOT / "src/ccad/activation_contract.py",
        ROOT / "src/ccad/artifacts.py", ROOT / "src/ccad/sae_quality.py",
        ROOT / "src/ccad/decoder_diagnostics.py",
    ]
    code_entries = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = hashlib.sha256("".join(f"{item['path']}:{item['sha256']}\n" for item in sorted(code_entries, key=lambda x: x["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})

    model_dir = Path(cfg["model_local_dir"])
    train_path = ROOT / cfg["train_token_path"]
    validation_path = ROOT / cfg["validation_token_path"]
    input_specs = [
        (train_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_train_tokens"),
        (validation_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_validation_tokens"),
        (ROOT / cfg["token_manifest_path"], "CCAD R006a capacity token manifest", "internal provenance artifact", "token_manifest"),
        (ROOT / cfg["environment_lock"], "CCAD environment lock", "internal artifact", "environment_lock"),
        (ROOT / cfg["compute_environment_spec"], "ARIS local compute environment spec", "internal artifact", "compute_environment_spec"),
        (ROOT / cfg["compute_environment_ledger"], "ARIS local compute environment ledger", "internal artifact", "compute_environment_ledger"),
        (Path(cfg["sparsify_source_dir"]) / "sparsify/trainer.py", f"sparsify@{cfg['sparsify_commit']}", "MIT", "trainer_source"),
    ]
    inputs = [file_entry(*spec) for spec in input_specs]
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
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run",
        "resource_lease_reason": "Pythia-160M nested budget-stability training and intervention evaluation",
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
        if sha256(ROOT / cfg["compute_environment_spec"]) != cfg["compute_environment_spec_sha256"]:
            raise ValueError("compute environment spec hash mismatch")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DISABLED": "true", "SPARSIFY_DISABLE_TRITON": "1"})
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import numpy as np
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
        milestone_steps = [tokens // (cfg["context_length"] * cfg["batch_size_sequences"]) for tokens in cfg["milestone_train_tokens"]]
        if milestone_steps[-1] != expected_steps or milestone_steps != sorted(set(milestone_steps)):
            raise RuntimeError(f"invalid milestone steps {milestone_steps} for {expected_steps} total steps")
        expected_input_hashes = [
            tensor_hash(torch.stack([train_data[index]["input_ids"] for index in range(step * cfg["batch_size_sequences"], (step + 1) * cfg["batch_size_sequences"])]))
            for step in range(expected_steps)
        ]

        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        attach_interruptible_trace(model)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_start

        def make_trainer() -> object:
            train_cfg = TrainConfig(
                sae=SaeConfig(activation="topk", num_latents=cfg["num_latents"], k=cfg["k"], normalize_decoder=True),
                batch_size=cfg["batch_size_sequences"], optimizer="adam", lr=cfg["learning_rate"],
                lr_warmup_steps=cfg["lr_warmup_steps"], hookpoints=[cfg["hookpoint"]], init_seeds=cfg["init_seeds"],
                save_every=expected_steps + 1, dead_feature_threshold=cfg["dead_feature_threshold"], auxk_alpha=cfg["auxk_alpha"],
                exclude_tokens=cfg["exclude_token_ids"], save_best=False, log_to_wandb=False,
                run_name=cfg["run_id"], save_dir=str(run_dir / "upstream_unused"),
            )
            result = Trainer(train_cfg, train_data, model)
            canonicalize_sparsify_multiseed_state(result)
            return result

        milestones = []
        combined_input_trace = []
        combined_loss_trace = []
        previous_checkpoint = None
        previous_step = 0
        total_train_seconds = 0.0
        for milestone_index, target_step in enumerate(milestone_steps):
            trainer = make_trainer()
            if previous_checkpoint is not None:
                load_sparsify_exact_state(trainer, previous_checkpoint, expected_data_cursor_examples=previous_step * cfg["batch_size_sequences"])
            traces, handles = attach_loss_trace(trainer)
            segment_steps = target_step - previous_step
            fail_on_call = segment_steps + 1 if target_step < expected_steps else None
            reset_trace(model, fail_on_call)
            saw_interrupt = False
            train_start = time.perf_counter()
            try:
                trainer.fit()
            except RuntimeError as exc:
                if INTERRUPT_MESSAGE not in str(exc):
                    raise
                saw_interrupt = True
            torch.cuda.synchronize()
            segment_seconds = time.perf_counter() - train_start
            total_train_seconds += segment_seconds
            remove_handles(handles)
            segment_input_trace = list(model.ccad_trace)
            sae_name, sae = next(iter(trainer.saes.items()))
            segment_loss_trace = traces[sae_name]
            combined_input_trace.extend(segment_input_trace)
            combined_loss_trace.extend(segment_loss_trace)
            checkpoint_dir = run_dir / "checkpoints" / f"step_{target_step:04d}"
            checkpoint_meta = save_sparsify_exact_state(trainer, checkpoint_dir, target_step * cfg["batch_size_sequences"])
            safe_dir = run_dir / "saes" / f"step_{target_step:04d}"
            sae.save_to_disk(safe_dir)
            contract = HookPointContract(cfg["hook_module_path"], cfg["layer_index"], "resid_post", model.config.hidden_size)
            evaluation = evaluate(model, sae, model.get_submodule(cfg["hook_module_path"]), contract, cfg["hook_oracle_hidden_state_index"], validation_data, cfg["eval_batch_size_sequences"], device, torch)
            decoder = sae.W_dec.detach().float().cpu().numpy()
            c_dec = pairwise_decoder_cosine_similarity(decoder, block_size=cfg["decoder_cosine_block_size"])
            milestone = {
                "milestone_index": milestone_index, "train_tokens": target_step * cfg["batch_size_sequences"] * cfg["context_length"],
                "global_step": trainer.global_step, "segment_steps": segment_steps, "segment_train_seconds": segment_seconds,
                "expected_interrupt": target_step < expected_steps, "saw_expected_interrupt": saw_interrupt,
                "segment_input_hashes": segment_input_trace, "segment_loss_trace": segment_loss_trace,
                "checkpoint_metadata": checkpoint_meta, "checkpoint_state_sha256": sha256(checkpoint_dir / "state.pt"),
                "state_hash": state_hash(sae.state_dict()), "decoder_norm_max_error": float((sae.W_dec.detach().float().norm(dim=1) - 1).abs().max()),
                "c_dec": c_dec, "validation": evaluation,
            }
            milestones.append(milestone)
            previous_checkpoint = checkpoint_dir
            previous_step = target_step

        stability_inputs = [
            {"fve": item["validation"]["fve"], "ce_recovered": item["validation"]["ce_recovered"], "alive_fraction": item["validation"]["alive_fraction"], "c_dec": item["c_dec"]}
            for item in milestones[-2:]
        ]
        stability = budget_stability_checks(stability_inputs[0], stability_inputs[1], cfg["stability_thresholds"])
        hard_checks = {
            "train_token_hash": sha256(train_path) == cfg["train_token_sha256"],
            "validation_token_hash": sha256(validation_path) == cfg["validation_token_sha256"],
            "environment_spec_hash": sha256(ROOT / cfg["compute_environment_spec"]) == cfg["compute_environment_spec_sha256"],
            "milestone_steps_exact": [item["global_step"] for item in milestones] == milestone_steps,
            "interruptions_exact": all(item["saw_expected_interrupt"] == item["expected_interrupt"] for item in milestones),
            "input_trace_exact": combined_input_trace == expected_input_hashes,
            "loss_trace_complete": len(combined_loss_trace) == expected_steps,
            "checkpoints_complete": all((run_dir / "checkpoints" / f"step_{step:04d}" / "state.pt").is_file() for step in milestone_steps),
            "safe_saes_complete": all((run_dir / "saes" / f"step_{step:04d}" / "sae.safetensors").is_file() for step in milestone_steps),
            "hook_oracle_exact": all(item["validation"]["hook_oracle_max_error"] == 0.0 for item in milestones),
            "capture_logits_exact": all(item["validation"]["capture_logit_max_error"] == 0.0 for item in milestones),
            "selected_l0_exact": all(item["validation"]["selected_l0"] == cfg["k"] for item in milestones),
            "ce_denominators_positive": all(item["validation"]["zero_ablation_damage"] > 0 for item in milestones),
            "metrics_finite": all(np.isfinite(value) for item in milestones for value in (item["validation"]["fve"], item["validation"]["ce_recovered"], item["validation"]["alive_fraction"], item["c_dec"])),
        }
        record = {
            "hard_checks": hard_checks, "stability": stability, "milestones": milestones,
            "milestone_steps": milestone_steps, "combined_input_hashes": combined_input_trace,
            "combined_loss_trace": combined_loss_trace, "expected_input_hashes": expected_input_hashes,
            "model_load_seconds": model_load_seconds, "total_train_seconds": total_train_seconds,
            "train_tokens_per_second": cfg["milestone_train_tokens"][-1] / total_train_seconds,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        hard_pass = all(hard_checks.values())
        status = "PASS" if hard_pass and stability["pass"] else "FAIL"
        if hard_pass and not stability["pass"]:
            error = "scientific_gate_failure: preregistered budget-stability thresholds not met"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "compute_environment_spec_sha256": cfg["compute_environment_spec_sha256"],
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})

    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    hard_checks = record["hard_checks"] if record else {}
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "hard_checks_passed": sum(hard_checks.values()), "hard_checks_total": len(hard_checks),
        "stability_gate_pass": record["stability"]["pass"] if record else False,
        "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r006b1_budget_stability.py", "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "stability_gate_pass": record["stability"]["pass"] if record else False, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
