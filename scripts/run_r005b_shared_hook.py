"""R005-B shared-token/shared-hook Pythia-160M framework conformance smoke."""

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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(entries: list[dict]) -> str:
    text = "".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(entries, key=lambda y: y["path"]))
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_hash(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": boundary, "role": role,
    }


def token_digest(document_ids: list[str], input_ids, attention_mask) -> str:
    payload = {
        "document_ids": document_ids,
        "input_ids": input_ids.detach().cpu().tolist(),
        "attention_mask": attention_mask.detach().cpu().tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diagnostic(sae, x, kind: str, torch) -> dict:
    with torch.no_grad():
        if kind == "dictionary_learning":
            features = sae.encode(x)
            reconstruction = sae.decode(features)
            l0 = (features != 0).sum(dim=-1)
            alive = int((features != 0).any(dim=0).sum())
            norms = sae.decoder.weight.norm(dim=0)
        else:
            output = sae(x)
            reconstruction = output.sae_out
            l0 = (output.latent_acts != 0).sum(dim=-1)
            alive = int(torch.unique(output.latent_indices).numel())
            norms = sae.W_dec.norm(dim=1)
        residual = x - reconstruction
        mse = float(residual.square().mean())
        denominator = float((x - x.mean(dim=0, keepdim=True)).square().mean())
        fve = 1.0 - mse / denominator if denominator > 0 else float("nan")
        return {
            "mse": mse, "fve": fve, "l0_mean": float(l0.float().mean()),
            "l0_min": int(l0.min()), "l0_max": int(l0.max()), "alive_in_batch": alive,
            "decoder_norm_max_error": float((norms - 1).abs().max()),
        }


def train_dictionary(cfg, batches, seed, torch):
    from dictionary_learning.trainers.top_k import TopKTrainer

    trainer = TopKTrainer(
        steps=len(batches), activation_dim=cfg["hidden_size"], dict_size=cfg["num_latents"],
        k=cfg["k"], layer=cfg["layer_index"], lm_name=cfg["model_id"],
        lr=cfg["learning_rate"], warmup_steps=0, seed=seed, device="cuda:0",
    )
    initial = state_hash(trainer.ae.state_dict())
    trace = []
    for step, batch in enumerate(batches):
        loss = float(trainer.update(step, batch))
        trace.append({"step": step, "loss": loss, **diagnostic(trainer.ae, batch, "dictionary_learning", torch)})
    return trainer.ae, initial, state_hash(trainer.ae.state_dict()), trace


def train_sparsify(cfg, batches, seed, torch):
    from sparsify import SaeConfig
    from sparsify.sparse_coder import SparseCoder

    torch.manual_seed(seed)
    sae_cfg = SaeConfig(activation="topk", num_latents=cfg["num_latents"], k=cfg["k"], normalize_decoder=True)
    sae = SparseCoder(cfg["hidden_size"], sae_cfg, device="cuda:0", dtype=torch.float32)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["learning_rate"])
    initial = state_hash(sae.state_dict())
    trace = []
    for step, batch in enumerate(batches):
        output = sae(batch)
        loss = output.fvu
        loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sae.set_decoder_norm_to_unit_norm()
        trace.append({"step": step, "loss": float(loss.detach()), **diagnostic(sae, batch, "sparsify", torch)})
    return sae, initial, state_hash(sae.state_dict()), trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
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
    code_paths = [Path(__file__).resolve(), args.config.resolve(), ROOT / "src/ccad/artifacts.py", ROOT / "src/ccad/activation_contract.py"]
    code_entries = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    code_hash = aggregate(code_entries)
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    model_dir = Path(cfg["model_local_dir"])
    source_manifest = ROOT / cfg["source_manifest"]
    env_lock = ROOT / cfg["environment_lock"]
    inputs = [file_entry(source_manifest, "CCAD project-authored debug text", "CC0-1.0", "debug_text_manifest"), file_entry(env_lock, "CCAD environment lock", "internal artifact", "environment_lock")]
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.name != "pytorch_model.bin":
            inputs.append(file_entry(path, f"Hugging Face {cfg['model_id']}@{cfg['model_revision']}", cfg["model_license"], "model_or_tokenizer"))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started.isoformat(), "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash,
        "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": "cuda:0",
        "seeds": {"init": cfg["init_seeds"], "replay": cfg["replay_seed"], "data_order": cfg["data_order_seed"]},
        "resource_lease": "gpu-0 via SAE Lab resource_manager.run", "resource_lease_reason": "Pythia-160M activation extraction and SAE smoke",
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "tokenizer_revision": cfg["tokenizer_revision"],
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    records = []
    status = "FAIL"
    error = None
    try:
        for key, value in cfg["offline_environment"].items():
            os.environ[key] = value
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
        sys.path[:0] = [cfg["dictionary_learning"]["source_dir"], cfg["dictionary_learning"]["overlay_dir"], cfg["sparsify"]["source_dir"], cfg["sparsify"]["overlay_dir"]]
        import torch
        import transformers
        import dictionary_learning
        import sparsify
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["data_order_seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        if model.config.hidden_size != cfg["hidden_size"] or model.config.num_hidden_layers != cfg["num_hidden_layers"]:
            raise RuntimeError("model architecture differs from config")
        rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
        order = list(range(len(rows)))
        random.Random(cfg["data_order_seed"]).shuffle(order)
        ordered = [rows[i] for i in order]
        module = model.gpt_neox.layers[cfg["layer_index"]]
        hook_contract = HookPointContract(
            module_path=cfg["hook_module_path"], layer_index=cfg["layer_index"],
            tensor_kind="debug_layer_output", hidden_size=cfg["hidden_size"],
        )
        cached_batches = []
        trace = []
        hook_errors = []
        capture_logits_errors = []
        base_forwards = 0
        extraction_started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(ordered), cfg["batch_size_sequences"]):
                group = ordered[start:start + cfg["batch_size_sequences"]]
                encoded = tokenizer([x["text"] for x in group], padding=True, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in encoded.items()}
                clean = model(**tokens, output_hidden_states=True, return_dict=True)
                base_forwards += 1
                captured = []
                handle = module.register_forward_hook(
                    lambda _m, _a, out: captured.append(extract_primary_hook_tensor(out, hook_contract).detach().clone())
                )
                hooked = model(**tokens, output_hidden_states=True, return_dict=True)
                handle.remove()
                base_forwards += 1
                activation = captured[0]
                hook_errors.append(float((activation - hooked.hidden_states[cfg["hook_oracle_hidden_state_index"]]).abs().max()))
                capture_logits_errors.append(float((clean.logits - hooked.logits).abs().max()))
                valid = tokens["attention_mask"].bool()
                flat = activation[valid].detach()
                cached_batches.append(flat)
                trace.append({
                    "step": len(trace), "document_ids": [x["document_id"] for x in group],
                    "token_hash": token_digest([x["document_id"] for x in group], tokens["input_ids"], tokens["attention_mask"]),
                    "activation_hash": tensor_hash(flat), "valid_tokens": int(valid.sum()), "padded_shape": list(activation.shape),
                })
        torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - extraction_started
        write_json(run_dir / "token_activation_trace.json", {"steps": trace})
        batch_hashes_before = [tensor_hash(x) for x in cached_batches]
        frameworks = {}
        trained = {}
        for name, trainer in (("dictionary_learning", train_dictionary), ("sparsify", train_sparsify)):
            run_started = time.perf_counter()
            seed0 = trainer(cfg, cached_batches, cfg["replay_seed"], torch)
            replay = trainer(cfg, cached_batches, cfg["replay_seed"], torch)
            seed1 = trainer(cfg, cached_batches, cfg["init_seeds"][1], torch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - run_started
            safe_dir = run_dir / name
            safe_dir.mkdir()
            safe_path = safe_dir / "sae_seed0.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in seed0[0].state_dict().items()}, str(safe_path))
            loaded_hash = state_hash(load_file(str(safe_path), device="cpu"))
            checks = {
                "same_seed_replay_initial": seed0[1] == replay[1],
                "same_seed_replay_final": seed0[2] == replay[2],
                "different_seed_initial": seed0[1] != seed1[1],
                "input_activation_trace_immutable": batch_hashes_before == [tensor_hash(x) for x in cached_batches],
                "safe_export_roundtrip": loaded_hash == seed0[2],
            }
            frameworks[name] = {
                "commit": cfg[name]["commit"], "initial_hash_seed0": seed0[1], "initial_hash_seed1": seed1[1],
                "final_hash_seed0": seed0[2], "replay_final_hash": replay[2], "trace_seed0": seed0[3],
                "trace_seed1": seed1[3], "training_wall_seconds_three_runs": elapsed,
                "safe_export_sha256": sha256(safe_path), "checks": checks,
            }
            trained[name] = seed0[0]
        global_checks = {
            "source_manifest_hash": sha256(source_manifest) == cfg["source_manifest_sha256"],
            "training_steps_match": len(cached_batches) == cfg["training_steps"],
            "hook_matches_hidden_state_oracle": max(hook_errors) <= cfg["absolute_tolerance"],
            "capture_only_logits_equal": max(capture_logits_errors) <= cfg["absolute_tolerance"],
            "shared_token_activation_trace": all(x["valid_tokens"] > 0 for x in trace),
            "all_framework_checks": all(all(x["checks"].values()) for x in frameworks.values()),
        }
        records = [{"kind": "global", "checks": global_checks, "max_hook_error": max(hook_errors), "max_capture_logits_error": max(capture_logits_errors), "total_valid_tokens": sum(x["valid_tokens"] for x in trace), "base_forwards": base_forwards, "model_load_seconds": model_load_seconds, "extraction_seconds": extraction_seconds, "peak_vram_bytes": torch.cuda.max_memory_allocated()}, {"kind": "frameworks", "frameworks": frameworks}]
        status = "PASS" if all(global_checks.values()) else "FAIL"
        environment = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "dictionary_learning": dictionary_learning.__version__, "sparsify": sparsify.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        write_json(run_dir / "environment.json", environment)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw_path = run_dir / "metrics.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "status": status, "error": error, "scope_limit": cfg["scope_limit"],
        "metrics_raw_sha256": sha256(raw_path), "generator_script_path": "scripts/run_r005b_shared_hook.py",
        "generator_script_sha256": code_entries[0]["sha256"], "checks_passed": sum(records[0]["checks"].values()) if records else 0,
        "checks_total": len(records[0]["checks"]) if records else 0,
    }
    write_json(run_dir / "metrics.summary.json", summary)
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
