"""Execute the fixed sparsify CLI with only local assets and audit CLI surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


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
        raise SystemExit(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    write_json(run_dir / "config.resolved.json", cfg)
    guard = ROOT / "scripts/offline_guard/sitecustomize.py"
    code_paths = [Path(__file__).resolve(), args.config.resolve(), guard, ROOT / "src/ccad/artifacts.py"]
    entries = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    aggregate = hashlib.sha256("".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(entries, key=lambda y: y["path"])).encode()).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": entries, "aggregate_sha256": aggregate})
    model_dir = Path(cfg["model_local_dir"])
    source_manifest = ROOT / cfg["source_manifest"]
    env_lock = ROOT / cfg["environment_lock"]
    sparsify_pyproject = Path(cfg["sparsify_source_dir"]) / "pyproject.toml"
    dictionary_pyproject = Path(cfg["dictionary_source_dir"]) / "pyproject.toml"
    inputs = [file_entry(source_manifest, "CCAD debug text", cfg["source_manifest_license"], "debug_text_manifest"),
              file_entry(env_lock, "CCAD environment lock", "internal artifact", "environment_lock"),
              file_entry(sparsify_pyproject, f"sparsify@{cfg['sparsify_commit']}", "MIT", "cli_declaration"),
              file_entry(dictionary_pyproject, f"dictionary_learning@{cfg['dictionary_commit']}", "MIT", "cli_surface_audit")]
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
        "seeds": {"init": cfg["init_seeds"], "shuffle": cfg["shuffle_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "real sparsify CLI Pythia smoke",
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "tokenizer_revision": cfg["tokenizer_revision"],
        "git_status": "project directory is not a Git repository; exact code/input hashes recorded",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    record = None
    error = None
    status = "FAIL"
    try:
        import numpy as np
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
        encoded = tokenizer([row["text"] for row in rows[:cfg["max_examples"]]], max_length=cfg["ctx_len"], truncation=True,
                            padding="max_length", return_tensors="np", add_special_tokens=True)["input_ids"]
        if encoded.max() > np.iinfo(np.uint16).max:
            raise RuntimeError("token id exceeds uint16 memmap contract")
        memmap_path = run_dir / "cli_tokens.uint16.bin"
        encoded.astype(np.uint16).tofile(memmap_path)
        socket_log = run_dir / "socket_attempts.jsonl"
        socket_log.write_text("", encoding="utf-8")
        cli_output = run_dir / "cli_output"
        command = [cfg["python_executable"], "-m", "sparsify", str(model_dir), str(memmap_path),
                   "--ctx_len", str(cfg["ctx_len"]), "--max_examples", str(cfg["max_examples"]),
                   "--batch_size", str(cfg["batch_size"]), "--hookpoints", cfg["hookpoint"],
                   "--init_seeds", *[str(seed) for seed in cfg["init_seeds"]], "--num_latents", str(cfg["num_latents"]),
                   "-k", str(cfg["k"]), "--lr", str(cfg["learning_rate"]), "--lr_warmup_steps", "0",
                   "--save_every", "1000", "--nolog_to_wandb", "--run_name", "cli_smoke", "--save_dir", str(cli_output)]
        env = os.environ.copy()
        env.update({"PYTHONPATH": os.pathsep.join([str(guard.parent), cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]),
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DISABLED": "true",
                    "SPARSIFY_DISABLE_TRITON": "1", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"],
                    "CCAD_SOCKET_GUARD_LOG": str(socket_log)})
        cli_started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        elapsed = time.perf_counter() - cli_started
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        safe_weights = sorted(cli_output.glob("**/sae.safetensors"))
        native_configs = sorted(cli_output.glob("**/config.json"))
        optimizer_states = sorted(cli_output.glob("**/optimizer_*.pt"))
        scheduler_states = sorted(cli_output.glob("**/lr_scheduler_*.pt"))
        dictionary_toml = tomllib.loads(dictionary_pyproject.read_text(encoding="utf-8"))
        sparsify_toml = tomllib.loads(sparsify_pyproject.read_text(encoding="utf-8"))
        declared_script = sparsify_toml.get("project", {}).get("scripts", {}).get("sparsify")
        dictionary_scripts = dictionary_toml.get("tool", {}).get("poetry", {}).get("scripts", {})
        checks = {
            "cli_exit_zero": completed.returncode == 0,
            "sparsify_entrypoint_declared": declared_script == "sparsify.__main__:run",
            "dictionary_has_no_declared_cli": dictionary_scripts == {},
            "socket_guard_clean": socket_log.stat().st_size == 0,
            "two_seed_safe_weights": len(safe_weights) == len(cfg["init_seeds"]),
            "resolved_cli_config_present": len(native_configs) >= 1,
            "optimizer_scheduler_present": len(optimizer_states) == 1 and len(scheduler_states) == 1,
            "memmap_shape_exact": memmap_path.stat().st_size == cfg["max_examples"] * cfg["ctx_len"] * 2,
        }
        record = {"checks": checks, "command": command, "returncode": completed.returncode, "wall_seconds": elapsed,
                  "memmap_sha256": sha256(memmap_path), "memmap_bytes": memmap_path.stat().st_size,
                  "socket_attempt_count": len(socket_log.read_text(encoding="utf-8").splitlines()),
                  "safe_weights": [{"path": str(path.relative_to(run_dir)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in safe_weights],
                  "native_config_paths": [str(path.relative_to(run_dir)).replace("\\", "/") for path in native_configs],
                  "optimizer_paths": [str(path.relative_to(run_dir)).replace("\\", "/") for path in optimizer_states],
                  "scheduler_paths": [str(path.relative_to(run_dir)).replace("\\", "/") for path in scheduler_states],
                  "sparsify_declared_entrypoint": declared_script, "dictionary_declared_scripts": dictionary_scripts}
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "cli_python": cfg["python_executable"],
                   "subprocess_returncode": completed.returncode, "offline_guard": str(guard.relative_to(ROOT))})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with stderr_path.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error,
        "checks_passed": sum(record["checks"].values()) if record else 0, "checks_total": len(record["checks"]) if record else 0,
        "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r005f_cli_boundary.py", "generator_script_sha256": entries[0]["sha256"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
