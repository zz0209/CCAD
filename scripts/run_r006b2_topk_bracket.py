"""R006-B2 pre-audit TopK k bracket with fail-closed k=16 reuse."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ccad.activation_contract import HookPointContract  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.checkpointing import canonicalize_sparsify_multiseed_state, save_sparsify_exact_state  # noqa: E402
from ccad.decoder_diagnostics import pairwise_decoder_cosine_similarity  # noqa: E402
from ccad.sae_quality import select_k_bracket  # noqa: E402
from run_r006b_topk_capacity import (  # noqa: E402
    MemmapTokens, attach_loss_trace, attach_model_trace, evaluate, file_entry,
    state_hash, tensor_hash, write_json,
)


REUSE_FIELDS = [
    "model_id", "model_revision", "tokenizer_revision", "model_local_dir", "sparsify_commit",
    "sparsify_source_dir", "sparsify_overlay_dir", "environment_lock", "train_token_path",
    "train_token_sha256", "validation_token_path", "validation_token_sha256", "token_manifest_path",
    "dataset_commit", "context_length", "max_train_sequences", "max_validation_sequences",
    "batch_size_sequences", "eval_batch_size_sequences", "layer_index", "hookpoint",
    "hook_module_path", "hook_oracle_hidden_state_index", "num_latents", "learning_rate",
    "lr_warmup_steps", "dead_feature_threshold", "auxk_alpha", "exclude_token_ids", "init_seeds",
    "data_order_seed", "precision", "attn_implementation", "device", "cublas_workspace_config",
    "decoder_cosine_block_size",
]


def canonical_json_hash(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        ROOT / "src/ccad/artifacts.py", ROOT / "src/ccad/sae_quality.py", ROOT / "src/ccad/decoder_diagnostics.py",
    ]
    code_entries = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = hashlib.sha256("".join(f"{item['path']}:{item['sha256']}\n" for item in sorted(code_entries, key=lambda x: x["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})

    model_dir = Path(cfg["model_local_dir"])
    train_path = ROOT / cfg["train_token_path"]
    validation_path = ROOT / cfg["validation_token_path"]
    spec_path = ROOT / cfg["compute_environment_spec"]
    reuse_dir = ROOT / "runs" / cfg["reuse_run"]
    input_specs = [
        (train_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_train_tokens"),
        (validation_path, f"FineWeb@{cfg['dataset_commit']}", cfg["dataset_license"], "capacity_validation_tokens"),
        (ROOT / cfg["token_manifest_path"], "CCAD R006a capacity token manifest", "internal provenance artifact", "token_manifest"),
        (ROOT / cfg["environment_lock"], "CCAD environment lock", "internal artifact", "environment_lock"),
        (spec_path, "ARIS local compute environment spec", "internal artifact", "compute_environment_spec"),
        (ROOT / cfg["compute_environment_ledger"], "ARIS local compute environment ledger", "internal artifact", "compute_environment_ledger"),
        (Path(cfg["sparsify_source_dir"]) / "sparsify/trainer.py", f"sparsify@{cfg['sparsify_commit']}", "MIT", "trainer_source"),
        (reuse_dir / "config.resolved.json", cfg["reuse_run"], "internal immutable run artifact", "reuse_config"),
        (reuse_dir / "code_hashes.json", cfg["reuse_run"], "internal immutable run artifact", "reuse_code_hashes"),
        (reuse_dir / "metrics.raw.jsonl", cfg["reuse_run"], "internal immutable run artifact", "reuse_metrics"),
        (reuse_dir / "metrics.summary.json", cfg["reuse_run"], "internal immutable run artifact", "reuse_summary"),
        (reuse_dir / "contract_validation.json", cfg["reuse_run"], "internal immutable run artifact", "reuse_contract_validation"),
        (reuse_dir / "checkpoints" / f"step_{cfg['reuse_step']:04d}" / "state.pt", cfg["reuse_run"], "internal immutable run artifact", "reuse_exact_checkpoint"),
        (reuse_dir / "saes" / f"step_{cfg['reuse_step']:04d}" / "sae.safetensors", cfg["reuse_run"], "internal immutable run artifact", "reuse_safe_sae"),
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
        "resource_lease_reason": "Pythia-160M TopK k bracket training and full intervention evaluation",
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
        if sha256(spec_path) != cfg["compute_environment_source_sha256"]:
            raise ValueError("compute environment source-file hash mismatch")
        if canonical_json_hash(spec_path) != cfg["compute_environment_canonical_sha256"]:
            raise ValueError("compute environment canonical hash mismatch")
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
        if len(train_data) * cfg["context_length"] != cfg["train_tokens"]:
            raise RuntimeError("configured train-token budget differs from loaded dataset")
        expected_input_hashes = [
            tensor_hash(torch.stack([train_data[index]["input_ids"] for index in range(step * cfg["batch_size_sequences"], (step + 1) * cfg["batch_size_sequences"])]))
            for step in range(expected_steps)
        ]
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        attach_model_trace(model)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_start

        def make_trainer(k: int):
            train_cfg = TrainConfig(
                sae=SaeConfig(activation="topk", num_latents=cfg["num_latents"], k=k, normalize_decoder=True),
                batch_size=cfg["batch_size_sequences"], optimizer="adam", lr=cfg["learning_rate"],
                lr_warmup_steps=cfg["lr_warmup_steps"], hookpoints=[cfg["hookpoint"]], init_seeds=cfg["init_seeds"],
                save_every=expected_steps + 1, dead_feature_threshold=cfg["dead_feature_threshold"], auxk_alpha=cfg["auxk_alpha"],
                exclude_tokens=cfg["exclude_token_ids"], save_best=False, log_to_wandb=False,
                run_name=f"{cfg['run_id']}_k{k}", save_dir=str(run_dir / "upstream_unused"),
            )
            trainer = Trainer(train_cfg, train_data, model)
            canonicalize_sparsify_multiseed_state(trainer)
            return trainer

        candidates = {}
        initial_state_hashes = {}
        for k in cfg["train_k"]:
            trainer = make_trainer(k)
            sae_name, sae = next(iter(trainer.saes.items()))
            initial_state_hashes[str(k)] = state_hash(sae.state_dict())
            traces, handles = attach_loss_trace(trainer)
            model.ccad_input_hashes = []
            train_start = time.perf_counter()
            trainer.fit()
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - train_start
            remove_handles(handles)
            input_trace = list(model.ccad_input_hashes)
            checkpoint_dir = run_dir / "checkpoints" / f"k_{k}"
            checkpoint_meta = save_sparsify_exact_state(trainer, checkpoint_dir, len(train_data))
            safe_dir = run_dir / "saes" / f"k_{k}"
            sae.save_to_disk(safe_dir)
            contract = HookPointContract(cfg["hook_module_path"], cfg["layer_index"], "resid_post", model.config.hidden_size)
            evaluation = evaluate(model, sae, model.get_submodule(cfg["hook_module_path"]), contract, cfg["hook_oracle_hidden_state_index"], validation_data, cfg["eval_batch_size_sequences"], device, torch)
            c_dec = pairwise_decoder_cosine_similarity(sae.W_dec.detach().float().cpu().numpy(), block_size=cfg["decoder_cosine_block_size"])
            checks = {
                "global_step_exact": trainer.global_step == expected_steps,
                "input_trace_exact": input_trace == expected_input_hashes,
                "loss_trace_complete": len(traces[sae_name]) == expected_steps,
                "exact_checkpoint_present": (checkpoint_dir / "state.pt").is_file(),
                "safe_checkpoint_present": (safe_dir / "sae.safetensors").is_file() and (safe_dir / "cfg.json").is_file(),
                "hook_oracle_exact": evaluation["hook_oracle_max_error"] == 0.0,
                "capture_logits_exact": evaluation["capture_logit_max_error"] == 0.0,
                "ce_denominator_positive": evaluation["zero_ablation_damage"] > 0,
                "selected_l0_exact": evaluation["selected_l0"] == k and evaluation["actual_nonzero_l0"] == k,
                "metrics_finite": all(np.isfinite(value) for value in (evaluation["fve"], evaluation["ce_recovered"], evaluation["alive_fraction"], c_dec)),
            }
            candidates[str(k)] = {
                "source": "trained_in_suite", "checks": checks, "sae_name": sae_name,
                "initial_state_hash": initial_state_hashes[str(k)], "final_state_hash": state_hash(sae.state_dict()),
                "checkpoint_metadata": checkpoint_meta, "checkpoint_state_sha256": sha256(checkpoint_dir / "state.pt"),
                "train_input_hashes": input_trace, "train_loss_trace": traces[sae_name],
                "train_seconds": train_seconds, "train_tokens_per_second": cfg["train_tokens"] / train_seconds,
                "decoder_norm_max_error": float((sae.W_dec.detach().float().norm(dim=1) - 1).abs().max()),
                "c_dec": c_dec, "validation": evaluation,
            }

        reuse_cfg = json.loads((reuse_dir / "config.resolved.json").read_text(encoding="utf-8"))
        reuse_code = json.loads((reuse_dir / "code_hashes.json").read_text(encoding="utf-8"))
        reuse_raw = json.loads((reuse_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
        reuse_summary = json.loads((reuse_dir / "metrics.summary.json").read_text(encoding="utf-8"))
        reuse_contract = json.loads((reuse_dir / "contract_validation.json").read_text(encoding="utf-8"))
        reuse_final = reuse_raw["milestones"][-1]
        historical_hashes = {item["path"]: item["sha256"] for item in reuse_code["files"]}
        current_reuse_critical = {
            path: sha256(ROOT / path) for path in (
                "scripts/run_r006b1_budget_stability.py", "scripts/run_r006b_topk_capacity.py",
                "src/ccad/checkpointing.py", "src/ccad/activation_contract.py", "src/ccad/decoder_diagnostics.py",
            )
        }
        reuse_checks = {
            "source_status_pass": reuse_summary["status"] == "PASS" and reuse_summary["stability_gate_pass"] is True,
            "source_contract_pass": reuse_contract["ok"] is True,
            "source_raw_hash": sha256(reuse_dir / "metrics.raw.jsonl") == reuse_summary["metrics_raw_sha256"],
            "critical_code_unchanged": all(historical_hashes[path] == digest for path, digest in current_reuse_critical.items()),
            "fixed_config_compatible": all(reuse_cfg[field] == cfg[field] for field in REUSE_FIELDS),
            "k_exact": reuse_cfg["k"] == cfg["reuse_k"] == 16,
            "budget_exact": reuse_final["global_step"] == cfg["reuse_step"] == expected_steps and reuse_final["train_tokens"] == cfg["train_tokens"],
            "source_hard_checks_pass": all(reuse_raw["hard_checks"].values()),
            "source_input_trace_exact": reuse_raw["combined_input_hashes"] == expected_input_hashes,
            "source_checkpoint_hash": sha256(reuse_dir / "checkpoints" / f"step_{cfg['reuse_step']:04d}" / "state.pt") == reuse_final["checkpoint_state_sha256"],
        }
        candidates[str(cfg["reuse_k"])] = {
            "source": "reused_from_budget_stability_run", "source_run": cfg["reuse_run"],
            "reuse_checks": reuse_checks, "final_state_hash": reuse_final["state_hash"],
            "checkpoint_state_sha256": reuse_final["checkpoint_state_sha256"],
            "decoder_norm_max_error": reuse_final["decoder_norm_max_error"],
            "c_dec": reuse_final["c_dec"], "validation": reuse_final["validation"],
        }

        selector_inputs = {
            int(k): {"fve": item["validation"]["fve"], "ce_recovered": item["validation"]["ce_recovered"], "c_dec": item["c_dec"]}
            for k, item in candidates.items()
        }
        selection = select_k_bracket(selector_inputs, cfg["selection_fve_margin"], cfg["selection_ce_recovered_margin"])
        suite_checks = {
            "train_token_hash": sha256(train_path) == cfg["train_token_sha256"],
            "validation_token_hash": sha256(validation_path) == cfg["validation_token_sha256"],
            "environment_source_hash": sha256(spec_path) == cfg["compute_environment_source_sha256"],
            "environment_canonical_hash": canonical_json_hash(spec_path) == cfg["compute_environment_canonical_sha256"],
            "candidate_set_exact": sorted(map(int, candidates)) == cfg["candidate_k"],
            "all_new_candidate_checks": all(all(candidates[str(k)]["checks"].values()) for k in cfg["train_k"]),
            "reused_candidate_checks": all(reuse_checks.values()),
            "new_initial_states_equal": len(set(initial_state_hashes.values())) == 1,
            "selector_has_valid_decision": selection["decision"] in {"TWO_SEED_PILOT", "EXPAND_TO_64_128"},
        }
        record = {
            "suite_checks": suite_checks, "candidates": candidates, "selector_inputs": selector_inputs,
            "selection": selection, "expected_input_hashes": expected_input_hashes,
            "reuse_critical_code_hashes": {"historical": {k: historical_hashes[k] for k in current_reuse_critical}, "current": current_reuse_critical},
            "model_load_seconds": model_load_seconds,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        status = "PASS" if all(suite_checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "compute_environment_canonical_sha256": cfg["compute_environment_canonical_sha256"],
            "compute_environment_source_sha256": cfg["compute_environment_source_sha256"],
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})

    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    suite_checks = record["suite_checks"] if record else {}
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "suite_checks_passed": sum(suite_checks.values()), "suite_checks_total": len(suite_checks),
        "selection_decision": record["selection"]["decision"] if record else None,
        "shortlist_k": record["selection"]["shortlist_k"] if record else [],
        "provisional_shortlist_k": record["selection"]["provisional_shortlist_k"] if record else [],
        "expansion_reasons": record["selection"]["expansion_reasons"] if record else [],
        "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r006b2_topk_bracket.py", "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({
        "run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok,
        "selection_decision": record["selection"]["decision"] if record else None,
        "shortlist_k": record["selection"]["shortlist_k"] if record else [],
    }))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
