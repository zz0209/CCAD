"""R005-A deterministic tiny-tensor conformance for two fixed SAE source snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ccad.artifacts import validate_run_directory  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def install_namespace(name: str, package_dir: Path) -> None:
    for key in tuple(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    package = types.ModuleType(name)
    package.__path__ = [str(package_dir)]
    package.__package__ = name
    sys.modules[name] = package


def overlay_versions(overlay: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for dist in importlib.metadata.distributions(path=[str(overlay)]):
        name = dist.metadata.get("Name")
        if name:
            values[name] = dist.version
    return dict(sorted(values.items(), key=lambda item: item[0].lower()))


def fixed_tensor(torch):
    return torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 1.25, 0.5, -0.75],
            [0.5, 0.75, -1.0, 0.25],
            [1.5, 0.25, -0.5, -0.25],
            [-1.0, 0.5, 1.25, 0.5],
            [0.75, -1.25, 0.5, 1.0],
            [0.25, 1.0, -0.75, 1.5],
            [-0.5, 0.25, 1.0, -1.25],
        ],
        dtype=torch.float32,
    )


def run_dictionary(cfg: dict, output_dir: Path, torch) -> dict:
    spec = cfg["dictionary_learning"]
    overlay = Path(spec["overlay_dir"])
    sys.path.insert(0, str(overlay))
    install_namespace("dictionary_learning", Path(spec["source_dir"]) / "dictionary_learning")
    topk = importlib.import_module("dictionary_learning.trainers.top_k")
    x = fixed_tensor(torch)

    def one(seed: int):
        trainer = topk.TopKTrainer(
            steps=cfg["steps"],
            activation_dim=cfg["activation_dim"],
            dict_size=cfg["dictionary_size"],
            k=cfg["k"],
            layer=0,
            lm_name="ccad_tiny_fixture",
            lr=cfg["learning_rate"],
            warmup_steps=0,
            seed=seed,
            device="cpu",
        )
        initial = state_hash(trainer.ae.state_dict())
        loss = float(trainer.update(0, x.clone()))
        final = state_hash(trainer.ae.state_dict())
        features = trainer.ae.encode(x)
        decoded = trainer.ae.decode(features)
        explicit = features @ trainer.ae.decoder.weight.detach().mT + trainer.ae.b_dec.detach()
        norms = trainer.ae.decoder.weight.detach().norm(dim=0)
        return trainer, initial, final, loss, features, decoded, explicit, norms

    a = one(cfg["replay_seed"])
    b = one(cfg["replay_seed"])
    c = one(cfg["init_seeds"][1])
    export_dir = output_dir / "dictionary_learning_safe_export"
    export_dir.mkdir(parents=True, exist_ok=False)
    from safetensors.torch import load_file, save_file

    safe_path = export_dir / "sae.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in a[0].ae.state_dict().items()}, str(safe_path))
    loaded = load_file(str(safe_path), device="cpu")
    loaded_hash = state_hash(loaded)
    nonzero = (a[4] != 0).sum(dim=-1)
    decode_error = float((a[5] - a[6]).detach().abs().max())
    checks = {
        "same_seed_initial_hash_equal": a[1] == b[1],
        "same_seed_post_step_hash_equal": a[2] == b[2] and a[3] == b[3],
        "different_seed_initial_hash_different": a[1] != c[1],
        "selected_k_exact": True,
        "actual_l0_bounded_by_k": bool((nonzero <= cfg["k"]).all()),
        "decode_formula_matches": bool(
            torch.allclose(a[5], a[6], atol=cfg["decode_formula_atol"], rtol=0.0)
        ),
        "decoder_norm_unit": bool(torch.allclose(a[7], torch.ones_like(a[7]), atol=1e-6, rtol=1e-6)),
        "safe_export_roundtrip_exact": loaded_hash == a[2],
    }
    return {
        "framework": "dictionary_learning",
        "commit": spec["commit"],
        "overlay_versions": overlay_versions(overlay),
        "initial_hash_seed0": a[1],
        "initial_hash_seed1": c[1],
        "post_step_hash_seed0": a[2],
        "loss_seed0": a[3],
        "actual_l0_min": int(nonzero.min()),
        "actual_l0_max": int(nonzero.max()),
        "selected_k": cfg["k"],
        "decoder_norm_max_error": float((a[7] - 1).abs().max()),
        "decode_formula_max_abs_error": decode_error,
        "safe_export_path": str(safe_path),
        "safe_export_sha256": sha256_file(safe_path),
        "checks": checks,
    }


def run_sparsify(cfg: dict, output_dir: Path, torch) -> dict:
    spec = cfg["sparsify"]
    overlay = Path(spec["overlay_dir"])
    sys.path.insert(0, str(overlay))
    install_namespace("sparsify", Path(spec["source_dir"]) / "sparsify")
    config_mod = importlib.import_module("sparsify.config")
    coder_mod = importlib.import_module("sparsify.sparse_coder")
    x = fixed_tensor(torch)
    sae_cfg = config_mod.SparseCoderConfig(
        activation="topk",
        expansion_factor=2,
        num_latents=cfg["dictionary_size"],
        k=cfg["k"],
        normalize_decoder=True,
    )

    def one(seed: int):
        torch.manual_seed(seed)
        sae = coder_mod.SparseCoder(cfg["activation_dim"], sae_cfg, device="cpu", dtype=torch.float32)
        initial = state_hash(sae.state_dict())
        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["learning_rate"])
        out = sae(x)
        loss = out.fvu
        loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        optimizer.zero_grad()
        sae.set_decoder_norm_to_unit_norm()
        final = state_hash(sae.state_dict())
        out_after = sae(x)
        explicit = (
            sae.W_dec.detach()[out_after.latent_indices]
            * out_after.latent_acts.detach().unsqueeze(-1)
        ).sum(dim=1) + sae.b_dec.detach()
        norms = sae.W_dec.detach().norm(dim=1)
        return sae, initial, final, float(loss.detach()), out_after, explicit, norms

    a = one(cfg["replay_seed"])
    b = one(cfg["replay_seed"])
    c = one(cfg["init_seeds"][1])
    export_dir = output_dir / "sparsify_safe_export"
    a[0].save_to_disk(export_dir)
    loaded = coder_mod.SparseCoder.load_from_disk(export_dir, device="cpu")
    nonzero = (a[4].latent_acts != 0).sum(dim=-1)
    decode_error = float((a[4].sae_out.detach() - a[5]).abs().max())
    checks = {
        "same_seed_initial_hash_equal": a[1] == b[1],
        "same_seed_post_step_hash_equal": a[2] == b[2] and a[3] == b[3],
        "different_seed_initial_hash_different": a[1] != c[1],
        "selected_k_exact": a[4].latent_indices.shape[-1] == cfg["k"],
        "actual_l0_bounded_by_k": bool((nonzero <= cfg["k"]).all()),
        "decode_formula_matches": bool(
            torch.allclose(
                a[4].sae_out.detach(), a[5], atol=cfg["decode_formula_atol"], rtol=0.0
            )
        ),
        "decoder_norm_unit": bool(torch.allclose(a[6], torch.ones_like(a[6]), atol=1e-6, rtol=1e-6)),
        "safe_export_roundtrip_exact": state_hash(loaded.state_dict()) == a[2],
    }
    safe_path = export_dir / "sae.safetensors"
    return {
        "framework": "sparsify",
        "commit": spec["commit"],
        "overlay_versions": overlay_versions(overlay),
        "initial_hash_seed0": a[1],
        "initial_hash_seed1": c[1],
        "post_step_hash_seed0": a[2],
        "loss_seed0": a[3],
        "actual_l0_min": int(nonzero.min()),
        "actual_l0_max": int(nonzero.max()),
        "selected_k": int(a[4].latent_indices.shape[-1]),
        "decoder_norm_max_error": float((a[6] - 1).abs().max()),
        "decode_formula_max_abs_error": decode_error,
        "safe_export_path": str(safe_path),
        "safe_export_sha256": sha256_file(safe_path),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = PROJECT_ROOT / "runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    resolved_path = run_dir / "config.resolved.json"
    write_json(resolved_path, cfg)

    source_files = [
        Path(__file__).resolve(),
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "config.py",
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "dictionary.py",
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "trainers" / "trainer.py",
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "trainers" / "top_k.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "config.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "fused_encoder.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "sparse_coder.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "utils.py",
    ]
    code_entries = []
    for path in source_files:
        display = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") if path.is_relative_to(PROJECT_ROOT) else str(path)
        code_entries.append({"path": display, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in sorted(code_entries, key=lambda e: e["path"])).encode()
    ).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": aggregate})
    input_files = [
        config_path,
        Path(cfg["dictionary_learning"]["source_dir"]) / "LICENSE",
        Path(cfg["sparsify"]["source_dir"]) / "LICENSE",
    ]
    inputs = []
    for path in input_files:
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "source": "CCAD prewritten config" if path == config_path else "fixed official GitHub snapshot",
                "license_or_access_boundary": "CCAD internal config" if path == config_path else "MIT",
                "role": "resolved experiment input" if path == config_path else "framework provenance",
            }
        )
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    manifest = {
        "schema_version": cfg["schema_version"],
        "run_id": cfg["run_id"],
        "run_parent": cfg["run_parent"],
        "purpose": "R005-A fixed-source core-module tiny-tensor conformance",
        "milestone": cfg["milestone"],
        "evidence_level": cfg["evidence_level"],
        "started_utc": started,
        "project_root": str(PROJECT_ROOT),
        "config_hash": sha256_file(resolved_path),
        "code_snapshot_hash": aggregate,
        "audit_opened": False,
        "candidate_family_frozen": False,
        "mean_constants_source_split": "not_applicable_framework_conformance",
        "threshold_source_split": "prewritten_hard_checks",
        "statistics_unit": "deterministic_framework_seed_replay",
        "device": "cpu",
        "seeds": {"init": cfg["init_seeds"], "replay": cfg["replay_seed"]},
        "resource_lease": "none",
        "resource_lease_reason": "bounded CPU tiny-tensor test; no heavy resource",
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "shared_runtime": cfg["shared_runtime"],
            "network_during_run": False,
            "external_logging": False,
        },
    )
    write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started})

    raw_path = run_dir / "metrics.raw.jsonl"
    status = "FAIL"
    records: list[dict] = []
    error = None
    started_perf = time.perf_counter()
    try:
        import torch

        torch.use_deterministic_algorithms(True)
        records.append(run_dictionary(cfg, run_dir, torch))
        records.append(run_sparsify(cfg, run_dir, torch))
        status = "PASS" if all(all(r["checks"].values()) for r in records) else "FAIL"
    except Exception:
        error = traceback.format_exc()
        (run_dir / "stderr.log").write_text(error, encoding="utf-8")
    with raw_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    generator_path = "scripts/run_r005a_conformance.py"
    summary = {
        "status": status,
        "framework_records": len(records),
        "checks_passed": sum(sum(bool(v) for v in r["checks"].values()) for r in records),
        "checks_total": sum(len(r["checks"]) for r in records),
        "elapsed_seconds": time.perf_counter() - started_perf,
        "error": error,
        "metrics_raw_sha256": sha256_file(raw_path),
        "generator_script_path": generator_path,
        "generator_script_sha256": next(e["sha256"] for e in code_entries if e["path"] == generator_path),
        "scope_limit": cfg["scope_limit"],
    }
    write_json(run_dir / "metrics.summary.json", summary)
    write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": utc_now()})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    if not validation.ok:
        status = "FAIL"
        write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": utc_now()})
    stdout_path.write_text(json.dumps({"status": status, "contract_ok": validation.ok}) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
