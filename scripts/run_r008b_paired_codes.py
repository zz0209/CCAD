"""Encode one shared-hook pass into a frozen same-configuration SAE set."""
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


def file_entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


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
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    token_manifest_path = ROOT / cfg["token_manifest_path"]
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    inputs = [file_entry(args.config.resolve(), "CCAD frozen config", "internal", "protocol"),
              file_entry(token_manifest_path, "R008a paired token manifest", "internal", "paired_tokens")]
    for split, info in token_manifest["outputs"].items():
        inputs.append(file_entry(ROOT / "runs" / cfg["paired_corpus_run"] / info["path"], "R008a paired corpus", "ODC-By-1.0", f"{split}_tokens"))
    for item in cfg["saes"]:
        inputs.append(file_entry(ROOT / item["path"] / "sae.safetensors", f"CCAD {item['run_id']}", "internal", f"seed_{item['seed']}_sae"))
        inputs.append(file_entry(ROOT / item["path"] / "cfg.json", f"CCAD {item['run_id']}", "internal", f"seed_{item['seed']}_sae_config"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "audit_opened": False,
        "candidate_family_frozen": False, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"], "device": cfg["device"],
        "seeds": [item["seed"] for item in cfg["saes"]], "resource_lease": "gpu-0 + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "shared-hook inference and paired sparse-code writes", "git_head_at_run": git_head, "git_status_porcelain": git_status,
        "bulk_output_dir": str(bulk_dir),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        if sha256(token_manifest_path) != cfg["token_manifest_sha256"]:
            raise ValueError("token manifest hash mismatch")
        for item in cfg["saes"]:
            if sha256(ROOT / item["path"] / "sae.safetensors") != item["sha256"]:
                raise ValueError(f"SAE hash mismatch for seed {item['seed']}")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "SPARSIFY_DISABLE_TRITON": "1", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"]})
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import numpy as np
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(cfg["model_local_dir"], local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        module = model.get_submodule(cfg["hook_module_path"])
        saes = {item["seed"]: SparseCoder.load_from_disk(ROOT / item["path"], device=device).eval() for item in cfg["saes"]}
        output_rows, total_forwards, total_tokens = [], 0, 0
        start_time = time.perf_counter()
        for split in ("mean", "discovery", "calibration", "audit"):
            info = token_manifest["outputs"][split]
            token_path = ROOT / "runs" / cfg["paired_corpus_run"] / info["path"]
            tokens = np.memmap(token_path, dtype="<u2", mode="r").reshape(info["sequences"], cfg["context_length"])
            split_dir = bulk_dir / split
            split_dir.mkdir()
            writers = {}
            rows = info["tokens"]
            for seed in saes:
                seed_dir = split_dir / f"seed_{seed}"
                seed_dir.mkdir()
                writers[seed] = (
                    np.memmap(seed_dir / "top_indices.uint16.bin", dtype="<u2", mode="w+", shape=(rows, cfg["k"])),
                    np.memmap(seed_dir / "top_acts.float32.bin", dtype="<f4", mode="w+", shape=(rows, cfg["k"])),
                )
            observed_rows = selected_slots = nonzero_slots = 0
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
                for seed, sae in saes.items():
                    with torch.no_grad():
                        encoded = sae.encode(hidden)
                    indices = encoded.top_indices.detach().cpu().numpy().astype("<u2", copy=False)
                    acts = encoded.top_acts.detach().float().cpu().numpy().astype("<f4", copy=False)
                    writers[seed][0][offset:end] = indices
                    writers[seed][1][offset:end] = acts
                    selected_slots += indices.size
                    nonzero_slots += int(np.count_nonzero(acts))
                observed_rows += hidden.shape[0]
                total_forwards += 1
            files = []
            for seed, (index_writer, act_writer) in writers.items():
                index_writer.flush(); act_writer.flush()
                del index_writer, act_writer
                seed_dir = split_dir / f"seed_{seed}"
                for name, dtype, shape in (("top_indices.uint16.bin", "uint16", [rows, cfg["k"]]), ("top_acts.float32.bin", "float32", [rows, cfg["k"]])):
                    path = seed_dir / name
                    files.append({"seed": seed, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "dtype": dtype, "shape": shape})
            output_rows.append({"split": split, "tokens": rows, "documents": info["documents"], "observed_rows": observed_rows,
                                "selected_l0": selected_slots / (observed_rows * len(saes)), "nonzero_l0": nonzero_slots / (observed_rows * len(saes)), "files": files})
            total_tokens += rows
        decoder_files = []
        decoder_dir = bulk_dir / "decoders"
        decoder_dir.mkdir()
        for seed, sae in saes.items():
            path = decoder_dir / f"seed_{seed}.float32.bin"
            sae.W_dec.detach().float().cpu().numpy().astype("<f4", copy=False).tofile(path)
            decoder_files.append({"seed": seed, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "dtype": "float32", "shape": [cfg["num_latents"], cfg["hook_hidden_size"]]})
        elapsed = time.perf_counter() - start_time
        expected_seed_ids = cfg.get("expected_seed_ids", [1, 2, 3, 4, 5])
        expected_sae_count = cfg.get("expected_sae_count", 5)
        checks = {
            "frozen_sae_set": len(saes) == expected_sae_count and sorted(saes) == expected_seed_ids,
            "four_splits_complete": len(output_rows) == 4,
            "shared_forward_count": total_forwards == sum((info["sequences"] + cfg["batch_size_sequences"] - 1) // cfg["batch_size_sequences"] for info in token_manifest["outputs"].values()),
            "all_rows_encoded": all(row["observed_rows"] == row["tokens"] for row in output_rows),
            "selected_l0_exact": all(row["selected_l0"] == cfg["k"] for row in output_rows),
            "indices_fit_uint16": cfg["num_latents"] <= 65536,
            "all_output_files_present": all(Path(item["path"]).is_file() for row in output_rows for item in row["files"]) and all(Path(item["path"]).is_file() for item in decoder_files),
            "audit_metrics_not_computed": True,
        }
        record = {"checks": checks, "splits": output_rows, "decoders": decoder_files, "total_tokens": total_tokens,
                  "shared_base_forwards": total_forwards, "wall_seconds": elapsed, "tokens_per_second": total_tokens / elapsed,
                  "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))}
        write_json(bulk_dir / "asset_manifest.json", {"schema_version": "r008.paired_sparse_codes.v1", "run_id": cfg["run_id"], "splits": output_rows, "decoders": decoder_files})
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
               "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r008b_paired_codes.py",
               "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
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
