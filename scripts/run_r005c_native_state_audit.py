"""R005-C audit of upstream-native checkpoint completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(entries: list[dict]) -> str:
    material = "".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(entries, key=lambda y: y["path"]))
    return hashlib.sha256(material.encode()).hexdigest()


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dictionary_audit(cfg: dict, output: Path) -> dict:
    sys.path[:0] = [cfg["dictionary_learning"]["source_dir"], cfg["dictionary_learning"]["overlay_dir"]]
    import torch
    import dictionary_learning.training as training
    from dictionary_learning.trainers.top_k import TopKTrainer

    save_dir = output / "dictionary_native"
    trainer_cfg = {
        "trainer": TopKTrainer, "steps": 1, "activation_dim": cfg["activation_dim"],
        "dict_size": cfg["dictionary_size"], "k": cfg["k"], "layer": 0,
        "lm_name": "ccad_checkpoint_fixture", "lr": cfg["learning_rate"],
        "warmup_steps": 0, "seed": cfg["init_seeds"][0], "device": "cpu",
    }
    batch = torch.tensor([[1.0, -0.5, 0.25, 0.75], [-0.25, 1.25, 0.5, -0.75]])

    class ReiterableBatches:
        def __iter__(self):
            for _ in range(102):
                yield batch.clone()

    training.trainSAE(
        ReiterableBatches(), [trainer_cfg], steps=1, use_wandb=False,
        save_dir=str(save_dir), backup_steps=1, normalize_activations=True, device="cpu",
    )
    path = save_dir / "trainer_0" / "ae.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    parameter_keys = sorted(payload)
    native_fields = {"step", "ae", "optimizer", "config", "norm_factor"}
    return {
        "framework": "dictionary_learning", "checkpoint_sha256": sha256(path),
        "final_payload_keys": parameter_keys, "native_resume_fields_present": sorted(native_fields & set(payload)),
        "native_resume_complete": native_fields.issubset(payload),
        "expected_limitation_detected": not native_fields.issubset(payload),
        "additional_boundary": "backup_steps crashes with unbound norm_factor when normalize_activations=False (observed in v1)",
        "interpretation": "with normalization enabled, final state_dict overwrote the backup payload at the same ae.pt path",
    }


def sparsify_audit(cfg: dict, output: Path) -> dict:
    sys.path[:0] = [cfg["sparsify"]["source_dir"], cfg["sparsify"]["overlay_dir"]]
    import torch
    from transformers import PreTrainedModel, PretrainedConfig
    from sparsify import SaeConfig, TrainConfig, Trainer

    class TinyConfig(PretrainedConfig):
        model_type = "ccad-state-audit"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.num_hidden_layers = 1
            self.hidden_size = cfg["activation_dim"]
            self.vocab_size = 16

    class TinyModel(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self):
            torch.manual_seed(1234)
            super().__init__(TinyConfig())
            self.embed = torch.nn.Embedding(16, cfg["activation_dim"])
            self.layers = torch.nn.ModuleList([torch.nn.Linear(cfg["activation_dim"], cfg["activation_dim"], bias=False)])

        @property
        def dummy_inputs(self):
            return {"input_ids": torch.tensor([[1, 2]])}

        def forward(self, input_ids, **_kwargs):
            value = self.embed(input_ids)
            for layer in self.layers:
                value = layer(value)
            return {"last_hidden_state": value}

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"input_ids": torch.tensor([1 + 2 * index, 2 + 2 * index])}

        def select(self, indices):
            return self

    save_root = output / "sparsify_native"
    train_cfg = TrainConfig(
        sae=SaeConfig(activation="topk", num_latents=cfg["dictionary_size"], k=cfg["k"], normalize_decoder=True),
        batch_size=1, optimizer="adam", lr=cfg["learning_rate"], lr_warmup_steps=0,
        hookpoints=["layers.0"], init_seeds=cfg["init_seeds"], save_every=1000,
        save_best=False, log_to_wandb=False, run_name="multi_seed", save_dir=str(save_root),
    )
    original = Trainer(train_cfg, TinyDataset(), TinyModel())
    expected = {}
    for offset, (name, counts) in enumerate(original.num_tokens_since_fired.items()):
        counts.copy_(torch.arange(counts.numel()) + 10 + offset * 100)
        expected[name] = counts.detach().clone()
    sae_hashes = {name: state_hash(sae.state_dict()) for name, sae in original.saes.items()}
    original.global_step = 1
    original.save()
    checkpoint = save_root / "multi_seed"
    restored = Trainer(train_cfg, TinyDataset(), TinyModel())
    restored.load_state(str(checkpoint))
    restored_counts = restored.num_tokens_since_fired
    counter_equal = {name: bool(torch.equal(expected[name], restored_counts[name])) for name in expected}
    restored_hashes = {name: state_hash(sae.state_dict()) for name, sae in restored.saes.items()}
    sae_names = sorted(original.saes)
    best_keys = sorted(original.best_loss) if isinstance(original.best_loss, dict) else []
    return {
        "framework": "sparsify", "sae_names": sae_names, "best_loss_keys": best_keys,
        "best_loss_keys_cover_saes": set(sae_names).issubset(best_keys),
        "weights_restore_exact": sae_hashes == restored_hashes,
        "counter_restore_by_sae": counter_equal,
        "native_multiseed_state_complete": all(counter_equal.values()) and set(sae_names).issubset(best_keys),
        "expected_limitation_detected": (not all(counter_equal.values())) and not set(sae_names).issubset(best_keys),
        "interpretation": "save keys use hook/seedN names while load_state iterates unsuffixed local hookpoints",
    }


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
    code_entries = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    code_hash = aggregate(code_entries)
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    inputs = []
    for key in ("dictionary_learning", "sparsify"):
        source = Path(cfg[key]["source_dir"])
        for relative in (("dictionary_learning/training.py",) if key == "dictionary_learning" else ("sparsify/trainer.py",)):
            path = source / relative
            inputs.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "source": f"fixed upstream {key} commit {cfg[key]['commit']}", "license_or_access_boundary": "MIT", "role": "checkpoint_implementation_under_audit"})
    env_lock = ROOT / cfg["environment_lock"]
    inputs.append({"path": str(env_lock.resolve()), "sha256": sha256(env_lock), "bytes": env_lock.stat().st_size, "source": "CCAD environment lock", "license_or_access_boundary": "internal artifact", "role": "environment_lock"})
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started.isoformat(), "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": {"init": cfg["init_seeds"]},
        "resource_lease": "none", "resource_lease_reason": "bounded CPU checkpoint audit",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    records = []
    error = None
    status = "FAIL"
    try:
        records = [dictionary_audit(cfg, run_dir), sparsify_audit(cfg, run_dir)]
        status = "PASS" if all(x["expected_limitation_detected"] for x in records) else "FAIL"
        import torch
        import transformers
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw = run_dir / "metrics.raw.jsonl"
    with raw.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "limitations_detected": sum(bool(x.get("expected_limitation_detected")) for x in records),
        "framework_records": len(records), "scope_limit": cfg["scope_limit"], "metrics_raw_sha256": sha256(raw),
        "generator_script_path": "scripts/run_r005c_native_state_audit.py", "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
