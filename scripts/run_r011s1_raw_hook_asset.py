"""Cache shared-hook activations for the bounded R011-S1 pre-audit screen."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def entry(path: Path, source: str, role: str, boundary: str = "internal") -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": boundary, "role": role,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    bulk_dir = Path(cfg["bulk_output_dir"])
    if run_dir.exists() or bulk_dir.exists():
        raise FileExistsError(f"refusing overwrite: {run_dir} or {bulk_dir}")
    run_dir.mkdir(parents=True)
    bulk_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/artifacts.py"]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    token_manifest_path = ROOT / cfg["token_manifest_path"]
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    model_config_path = Path(cfg["model_local_dir"]) / "config.json"
    inputs = [
        entry(args.config.resolve(), "CCAD frozen config", "protocol"),
        entry(token_manifest_path, "R008a paired corpus", "paired_token_manifest"),
        entry(model_config_path, cfg["model_id"], "model_config", cfg["model_license"]),
    ]
    for split in cfg["splits"]:
        info = token_manifest["outputs"][split]
        inputs.append(entry(ROOT / "runs" / cfg["paired_corpus_run"] / info["path"], "R008a paired corpus", f"{split}_tokens", "ODC-By-1.0"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": False,
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": [],
        "resource_lease": "gpu-0 + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "shared Pythia layer-5 inference and bounded mean/discovery/calibration activation writes",
        "git_head_at_run": git_head, "git_status_porcelain": git_status, "bulk_output_dir": str(bulk_dir),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        if sha256(token_manifest_path).lower() != cfg["token_manifest_sha256"].lower():
            raise ValueError("token manifest hash mismatch")
        if cfg["splits"] != ["mean", "discovery", "calibration"] or cfg["forbidden_splits"] != ["audit"]:
            raise ValueError("R011-S1 raw asset must exclude audit")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"]})
        import numpy as np
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
        module = model.get_submodule(cfg["hook_module_path"])
        rows, total_forwards, total_tokens = [], 0, 0
        started_compute = time.perf_counter()
        for split in cfg["splits"]:
            info = token_manifest["outputs"][split]
            token_path = ROOT / "runs" / cfg["paired_corpus_run"] / info["path"]
            tokens = np.memmap(token_path, dtype="<u2", mode="r").reshape(info["sequences"], cfg["context_length"])
            output_path = bulk_dir / f"{split}.float32.bin"
            writer = np.memmap(output_path, dtype="<f4", mode="w+", shape=(info["tokens"], cfg["hook_hidden_size"]))
            observed = 0
            for begin in range(0, len(tokens), cfg["batch_size_sequences"]):
                batch_np = np.asarray(tokens[begin:begin + cfg["batch_size_sequences"]], dtype=np.int64)
                batch = torch.from_numpy(batch_np).to(device)
                captured = {}

                def hook(_module, _inputs, output):
                    captured["hidden"] = output[0] if isinstance(output, tuple) else output

                handle = module.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        model(batch, use_cache=False)
                finally:
                    handle.remove()
                hidden = captured["hidden"].reshape(-1, cfg["hook_hidden_size"])
                offset = begin * cfg["context_length"]
                end = offset + hidden.shape[0]
                writer[offset:end] = hidden.detach().float().cpu().numpy().astype("<f4", copy=False)
                observed += hidden.shape[0]
                total_forwards += 1
            writer.flush()
            del writer
            rows.append({
                "split": split, "tokens": info["tokens"], "documents": info["documents"], "observed_rows": observed,
                "path": str(output_path), "sha256": sha256(output_path), "bytes": output_path.stat().st_size,
                "dtype": "float32", "shape": [info["tokens"], cfg["hook_hidden_size"]],
            })
            total_tokens += info["tokens"]
        elapsed = time.perf_counter() - started_compute
        checks = {
            "frozen_token_manifest": sha256(token_manifest_path).lower() == cfg["token_manifest_sha256"].lower(),
            "three_pre_audit_splits": [row["split"] for row in rows] == cfg["splits"],
            "all_rows_written": all(row["tokens"] == row["observed_rows"] for row in rows),
            "shared_forward_count": total_forwards == sum((token_manifest["outputs"][split]["sequences"] + cfg["batch_size_sequences"] - 1) // cfg["batch_size_sequences"] for split in cfg["splits"]),
            "shape_contract": all(row["shape"] == [row["tokens"], cfg["hook_hidden_size"]] for row in rows),
            "all_files_present": all(Path(row["path"]).is_file() for row in rows),
            "audit_excluded": "audit" not in {row["split"] for row in rows} and not cfg["audit_opened"],
        }
        raw_manifest = {"schema_version": "r011s1.shared_hook.v1", "run_id": cfg["run_id"], "splits": rows}
        write_json(bulk_dir / "raw_hook_manifest.json", raw_manifest)
        record = {
            "checks": checks, "splits": rows, "total_tokens": total_tokens, "shared_base_forwards": total_forwards,
            "wall_seconds": elapsed, "tokens_per_second": total_tokens / elapsed,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "raw_hook_manifest_sha256": sha256(bulk_dir / "raw_hook_manifest.json"),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r011s1_raw_hook_asset.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
