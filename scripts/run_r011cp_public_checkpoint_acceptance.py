"""Evaluate one pinned public Pythia SAE as an external quality anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ccad.activation_contract import HookPointContract  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from run_r006b_topk_capacity import MemmapTokens, evaluate  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        raise FileExistsError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_text("", encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_r006b_topk_capacity.py",
                  ROOT / "src/ccad/activation_contract.py", ROOT / "src/ccad/artifacts.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
                 for path in code_paths]
    code_hash = hashlib.sha256("".join(
        f"{row['path']}:{row['sha256']}\n" for row in sorted(code_rows, key=lambda item: item["path"])
    ).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    checkpoint_dir = Path(cfg["checkpoint_local_dir"])
    validation_path = ROOT / cfg["validation_token_path"]
    inputs = [file_entry(args.config.resolve(), "CCAD frozen acceptance config", "internal", "protocol"),
              file_entry(validation_path, "CCAD R006a held-out quality validation", "ODC-By-1.0", "validation_tokens"),
              file_entry(checkpoint_dir / "cfg.json", f"{cfg['checkpoint_id']}@{cfg['checkpoint_revision']}", cfg["checkpoint_license"], "public_sae_config"),
              file_entry(checkpoint_dir / "sae.safetensors", f"{cfg['checkpoint_id']}@{cfg['checkpoint_revision']}", cfg["checkpoint_license"], "public_sae_weights")]
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"],
        "candidate_family_frozen": cfg["candidate_family_frozen"], "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"],
        "device": cfg["device"], "seeds": {"public_checkpoint": "unknown/single"},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "public SAE hook/quality acceptance",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
        "checkpoint_id": cfg["checkpoint_id"], "checkpoint_revision": cfg["checkpoint_revision"]
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        if sha256(validation_path) != cfg["validation_token_sha256"]:
            raise ValueError("validation token hash mismatch")
        public_cfg = json.loads((checkpoint_dir / "cfg.json").read_text(encoding="utf-8"))
        expected = {"d_in": 768, "num_latents": cfg["num_latents"], "k": cfg["k"], "signed": False}
        if any(public_cfg.get(key) != value for key, value in expected.items()):
            raise ValueError(f"public checkpoint config mismatch: {public_cfg}")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
                           "SPARSIFY_DISABLE_TRITON": "1", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"]})
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(cfg["model_local_dir"], local_files_only=True, dtype=torch.float32,
                                                     attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        sae = SparseCoder.load_from_disk(checkpoint_dir, device=device).eval()
        validation = MemmapTokens(validation_path, cfg["context_length"], cfg["max_validation_sequences"])
        contract = HookPointContract(cfg["hook_module_path"], cfg["layer_index"], "resid_post", model.config.hidden_size)
        metrics = evaluate(model, sae, model.get_submodule(cfg["hook_module_path"]), contract,
                           cfg["hook_oracle_hidden_state_index"], validation, cfg["eval_batch_size_sequences"], device, torch)
        decoder_norm_error = float((sae.W_dec.detach().float().norm(dim=1) - 1).abs().max())
        checks = {"public_config_exact": True, "hook_oracle_exact": metrics["hook_oracle_max_error"] == 0.0,
                  "capture_logits_exact": metrics["capture_logit_max_error"] == 0.0,
                  "selected_l0_exact": metrics["selected_l0"] == cfg["k"],
                  "actual_l0_positive": metrics["actual_nonzero_l0"] > 0,
                  "ce_denominator_positive": metrics["zero_ablation_damage"] > 0,
                  "metrics_finite": all(bool(torch.isfinite(torch.tensor(metrics[key]))) for key in ("fve", "ce_recovered", "actual_nonzero_l0"))}
        record = {"checks": checks, "public_config": public_cfg, "validation": metrics,
                  "decoder_norm_max_error": decoder_norm_error,
                  "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "checkpoint_bytes": (checkpoint_dir / "sae.safetensors").stat().st_size}
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                                                   "transformers": transformers.__version__, "cuda": torch.version.cuda,
                                                   "gpu": torch.cuda.get_device_name(device)})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error,
                                                   "checks_passed": sum(record["checks"].values()) if record else 0,
                                                   "checks_total": len(record["checks"]) if record else 0,
                                                   "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw)})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
