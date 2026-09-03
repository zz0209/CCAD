"""T011 exact multi-seed checkpoint/resume conformance on a tiny local model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
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


def aggregate(entries: list[dict]) -> str:
    text = "".join(f"{x['path']}:{x['sha256']}\n" for x in sorted(entries, key=lambda y: y["path"]))
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_hash(value) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


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
        return set(left) == set(right) and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


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
    code_entries = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    code_hash = aggregate(code_entries)
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    upstream = Path(cfg["sparsify_source_dir"]) / "sparsify/trainer.py"
    env_lock = ROOT / cfg["environment_lock"]
    write_json(run_dir / "inputs.json", {"inputs": [
        {"path": str(upstream), "sha256": sha256(upstream), "bytes": upstream.stat().st_size, "source": f"fixed sparsify commit {cfg['sparsify_commit']}", "license_or_access_boundary": "MIT", "role": "wrapped_trainer"},
        {"path": str(env_lock.resolve()), "sha256": sha256(env_lock), "bytes": env_lock.stat().st_size, "source": "CCAD environment lock", "license_or_access_boundary": "internal artifact", "role": "environment_lock"},
    ]})
    write_json(run_dir / "manifest.json", {
        "schema_version": "0.1.0", "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started.isoformat(), "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"],
        "seeds": {"init": cfg["init_seeds"], "model": cfg["model_seed"]},
        "resource_lease": "none", "resource_lease_reason": "bounded CPU deterministic wrapper test",
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started.isoformat()})
    record = None
    error = None
    status = "FAIL"
    try:
        sys.path[:0] = [cfg["sparsify_source_dir"], cfg["sparsify_overlay_dir"]]
        import torch
        import transformers
        from transformers import PreTrainedModel, PretrainedConfig
        from sparsify import SaeConfig, TrainConfig, Trainer

        torch.use_deterministic_algorithms(True)

        class TinyConfig(PretrainedConfig):
            model_type = "ccad-wrapper-test"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.num_hidden_layers = 1
                self.hidden_size = cfg["activation_dim"]
                self.vocab_size = 32

        class TinyModel(PreTrainedModel):
            config_class = TinyConfig
            def __init__(self):
                torch.manual_seed(cfg["model_seed"])
                super().__init__(TinyConfig())
                self.embed = torch.nn.Embedding(32, cfg["activation_dim"])
                self.layers = torch.nn.ModuleList([torch.nn.Linear(cfg["activation_dim"], cfg["activation_dim"], bias=False)])
                self.calls = 0
                self.fail_on_call = None
                self.input_trace = []
            @property
            def dummy_inputs(self):
                return {"input_ids": torch.tensor([[1, 2]])}
            def forward(self, input_ids, **_kwargs):
                self.calls += 1
                if self.fail_on_call == self.calls:
                    raise RuntimeError("expected T011 interruption")
                self.input_trace.append(tensor_hash(input_ids))
                value = self.embed(input_ids)
                for layer in self.layers:
                    value = layer(value)
                return {"last_hidden_state": value}

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, rows=None):
                self.rows = rows or [[1, 2], [3, 4], [5, 6], [7, 8]]
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                return {"input_ids": torch.tensor(self.rows[index])}
            def select(self, indices):
                return TinyDataset([self.rows[index] for index in indices])

        def make_cfg(run_name):
            return TrainConfig(
                sae=SaeConfig(activation="topk", num_latents=cfg["dictionary_size"], k=cfg["k"], normalize_decoder=True),
                batch_size=cfg["batch_size"], optimizer="adam", lr=cfg["learning_rate"], lr_warmup_steps=0,
                hookpoints=["layers.0"], init_seeds=cfg["init_seeds"], save_every=1000,
                dead_feature_threshold=cfg["dead_feature_threshold"], auxk_alpha=cfg["auxk_alpha"],
                save_best=False, log_to_wandb=False, run_name=run_name, save_dir=str(run_dir / "upstream_unused"),
            )

        dataset = TinyDataset()
        full_model = TinyModel()
        full = Trainer(make_cfg("full"), dataset, full_model)
        canonicalize_sparsify_multiseed_state(full)
        full_model.calls = 0
        full_model.input_trace = []
        full.fit()
        full_rng = torch.get_rng_state().clone()

        interrupted_model = TinyModel()
        interrupted = Trainer(make_cfg("interrupted"), dataset, interrupted_model)
        canonicalize_sparsify_multiseed_state(interrupted)
        interrupted_model.calls = 0
        interrupted_model.input_trace = []
        interrupted_model.fail_on_call = cfg["interrupt_after_steps"] + 1
        saw_interrupt = False
        try:
            interrupted.fit()
        except RuntimeError as exc:
            if "expected T011 interruption" not in str(exc):
                raise
            saw_interrupt = True
        cursor = interrupted.global_step * cfg["batch_size"]
        checkpoint = run_dir / "exact_checkpoint"
        metadata = save_sparsify_exact_state(interrupted, checkpoint, cursor)

        resumed_model = TinyModel()
        resumed = Trainer(make_cfg("resumed"), dataset, resumed_model)
        canonicalize_sparsify_multiseed_state(resumed)
        loaded = load_sparsify_exact_state(resumed, checkpoint, expected_data_cursor_examples=cursor)
        resumed_model.calls = 0
        resumed_model.input_trace = []
        resumed.fit()
        resumed_rng = torch.get_rng_state().clone()

        full_weights = {name: state_hash(sae.state_dict()) for name, sae in full.saes.items()}
        resumed_weights = {name: state_hash(sae.state_dict()) for name, sae in resumed.saes.items()}
        counters_equal = nested_equal(full.num_tokens_since_fired, resumed.num_tokens_since_fired)
        optimizer_equal = nested_equal([x.state_dict() for x in full.optimizers], [x.state_dict() for x in resumed.optimizers])
        scheduler_equal = nested_equal([x.state_dict() for x in full.lr_schedulers], [x.state_dict() for x in resumed.lr_schedulers])
        joined_trace = interrupted_model.input_trace + resumed_model.input_trace
        checks = {
            "interruption_after_expected_steps": saw_interrupt and interrupted.global_step == cfg["interrupt_after_steps"],
            "metadata_covers_seeded_saes": metadata["sae_names"] == sorted(full.saes) and metadata["counter_names"] == sorted(full.saes) and metadata["best_loss_names"] == sorted(full.saes),
            "data_cursor_exact": loaded["data_cursor_examples"] == cursor == cfg["interrupt_after_steps"] * cfg["batch_size"],
            "input_trace_exact": full_model.input_trace == joined_trace,
            "weights_exact": full_weights == resumed_weights,
            "optimizer_exact": optimizer_equal,
            "scheduler_exact": scheduler_equal,
            "counters_exact": counters_equal,
            "best_loss_exact": nested_equal(full.best_loss, resumed.best_loss),
            "rng_exact": bool(torch.equal(full_rng, resumed_rng)),
            "global_step_exact": full.global_step == resumed.global_step == cfg["examples"],
        }
        record = {
            "checks": checks, "full_weights": full_weights, "resumed_weights": resumed_weights,
            "full_trace": full_model.input_trace, "interrupted_trace": interrupted_model.input_trace,
            "resumed_trace": resumed_model.input_trace, "checkpoint_metadata": metadata,
            "checkpoint_state_sha256": sha256(checkpoint / "state.pt"),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__})
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
        "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r005c_multiseed_wrapper.py",
        "generator_script_sha256": code_entries[0]["sha256"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
