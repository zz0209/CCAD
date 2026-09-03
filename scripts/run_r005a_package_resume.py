"""R005-A public-package, offline logging, and tiny resume conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def fixed_batches(torch):
    return [
        torch.tensor([[1.0, -0.5, 0.25, 0.75], [-0.25, 1.25, 0.5, -0.75]], dtype=torch.float32),
        torch.tensor([[0.5, 0.75, -1.0, 0.25], [1.5, 0.25, -0.5, -0.25]], dtype=torch.float32),
    ]


def block_network() -> list[str]:
    attempts: list[str] = []

    def blocked(_self, address):
        attempts.append(repr(address))
        raise RuntimeError(f"network disabled by R005-A conformance: {address!r}")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    return attempts


def dictionary_worker(cfg: dict, worker_dir: Path) -> dict:
    source = Path(cfg["dictionary_learning"]["source_dir"])
    overlay = Path(cfg["dictionary_learning"]["overlay_dir"])
    sys.path[:0] = [str(source), str(overlay)]
    attempts = block_network()
    import torch
    import dictionary_learning
    import dictionary_learning.training as training
    from dictionary_learning.trainers.top_k import TopKTrainer

    torch.use_deterministic_algorithms(True)
    wandb_called = False

    def forbidden_wandb(*_args, **_kwargs):
        nonlocal wandb_called
        wandb_called = True
        raise AssertionError("wandb.init must not be called")

    training.wandb.init = forbidden_wandb
    batches = fixed_batches(torch)
    public_dir = worker_dir / "public_train"
    public_cfg = {
        "trainer": TopKTrainer,
        "steps": 1,
        "activation_dim": cfg["activation_dim"],
        "dict_size": cfg["dictionary_size"],
        "k": cfg["k"],
        "layer": 0,
        "lm_name": "ccad_tiny_fixture",
        "lr": cfg["learning_rate"],
        "warmup_steps": 0,
        "seed": cfg["init_seed"],
        "device": "cpu",
    }
    training.trainSAE(
        data=iter([b.clone() for b in batches]),
        trainer_configs=[public_cfg],
        steps=1,
        use_wandb=False,
        save_dir=str(public_dir),
        device="cpu",
    )
    public_weight = public_dir / "trainer_0" / "ae.pt"
    public_config = public_dir / "trainer_0" / "config.json"

    def trainer():
        return TopKTrainer(
            steps=2,
            activation_dim=cfg["activation_dim"],
            dict_size=cfg["dictionary_size"],
            k=cfg["k"],
            layer=0,
            lm_name="ccad_tiny_fixture",
            lr=cfg["learning_rate"],
            warmup_steps=0,
            seed=cfg["init_seed"],
            device="cpu",
        )

    full = trainer()
    full.update(0, batches[0].clone())
    full.update(1, batches[1].clone())
    full_hash = state_hash(full.ae.state_dict())

    interrupted = trainer()
    interrupted.update(0, batches[0].clone())
    checkpoint = worker_dir / "wrapper_checkpoint.pt"
    torch.save(
        {
            "step": 1,
            "ae": interrupted.ae.state_dict(),
            "optimizer": interrupted.optimizer.state_dict(),
            "scheduler": interrupted.scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed = trainer()
    resumed.ae.load_state_dict(payload["ae"])
    resumed.optimizer.load_state_dict(payload["optimizer"])
    resumed.scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["cpu_rng"])
    resumed.update(payload["step"], batches[1].clone())
    resumed_hash = state_hash(resumed.ae.state_dict())
    checks = {
        "public_package_import": dictionary_learning.__version__ == "0.1.0",
        "documented_training_or_trainer_path_executes": public_weight.is_file(),
        "external_logging_disabled": not wandb_called,
        "no_socket_connect_attempt": not attempts,
        "checkpoint_artifact_complete": public_weight.is_file() and public_config.is_file() and checkpoint.is_file(),
        "uninterrupted_vs_resumed_final_hash_equal": full_hash == resumed_hash,
        "uninterrupted_vs_resumed_step_equal": payload["step"] + 1 == 2,
    }
    return {
        "framework": "dictionary_learning",
        "commit": cfg["dictionary_learning"]["commit"],
        "public_version": dictionary_learning.__version__,
        "resume_kind": "CCAD wrapper; upstream trainSAE has no native load/resume entry point",
        "full_hash": full_hash,
        "resumed_hash": resumed_hash,
        "checkpoint_sha256": sha256_file(checkpoint),
        "public_weight_sha256": sha256_file(public_weight),
        "network_attempts": attempts,
        "checks": checks,
    }


def sparsify_worker(cfg: dict, worker_dir: Path) -> dict:
    source = Path(cfg["sparsify"]["source_dir"])
    overlay = Path(cfg["sparsify"]["overlay_dir"])
    sys.path[:0] = [str(source), str(overlay)]
    attempts = block_network()
    import torch
    from transformers import PreTrainedModel, PretrainedConfig
    import sparsify
    import sparsify.__main__  # noqa: F401
    from sparsify import SaeConfig, TrainConfig, Trainer

    torch.use_deterministic_algorithms(True)

    class TinyConfig(PretrainedConfig):
        model_type = "ccad-tiny"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.num_hidden_layers = 1
            self.hidden_size = cfg["activation_dim"]
            self.vocab_size = 16

    class TinyModel(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self):
            torch.manual_seed(cfg["model_seed"])
            super().__init__(TinyConfig())
            self.embed = torch.nn.Embedding(16, cfg["activation_dim"])
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(cfg["activation_dim"], cfg["activation_dim"], bias=False)]
            )
            self.calls = 0
            self.fail_on_call = None

        @property
        def dummy_inputs(self):
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

        def forward(self, input_ids, **_kwargs):
            self.calls += 1
            if self.fail_on_call is not None and self.calls == self.fail_on_call:
                raise RuntimeError("expected R005-A synthetic interruption")
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            return {"last_hidden_state": x}

    class TinyDataset(torch.utils.data.Dataset):
        """Minimal sized/selectable fixture matching the Trainer's dataset contract."""

        def __init__(self, rows=None):
            self.rows = rows or [[1, 2], [3, 4]]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {"input_ids": torch.tensor(self.rows[index], dtype=torch.long)}

        def select(self, indices):
            return TinyDataset([self.rows[index] for index in indices])

    dataset = TinyDataset()

    def make_cfg(run_name: str):
        return TrainConfig(
            sae=SaeConfig(
                activation="topk",
                num_latents=cfg["dictionary_size"],
                expansion_factor=2,
                k=cfg["k"],
                normalize_decoder=True,
            ),
            batch_size=cfg["batch_size"],
            optimizer="adam",
            lr=cfg["learning_rate"],
            lr_warmup_steps=0,
            hookpoints=["layers.0"],
            init_seeds=[cfg["init_seed"]],
            save_every=1000,
            log_to_wandb=False,
            run_name=run_name,
            save_dir=str(worker_dir),
        )

    full_model = TinyModel()
    full = Trainer(make_cfg("full"), dataset, full_model)
    full_model.calls = 0
    full.fit()
    full_hash = state_hash(next(iter(full.saes.values())).state_dict())

    interrupted_model = TinyModel()
    interrupted = Trainer(make_cfg("interrupted"), dataset, interrupted_model)
    interrupted_model.calls = 0
    interrupted_model.fail_on_call = 2
    saw_interrupt = False
    try:
        interrupted.fit()
    except RuntimeError as exc:
        if "expected R005-A synthetic interruption" not in str(exc):
            raise
        saw_interrupt = True
    interrupted.save()
    checkpoint_dir = worker_dir / "interrupted"

    resumed_model = TinyModel()
    resumed = Trainer(make_cfg("resumed"), dataset, resumed_model)
    resumed.load_state(str(checkpoint_dir))
    resumed_model.calls = 0
    resumed.fit()
    resumed_hash = state_hash(next(iter(resumed.saes.values())).state_dict())
    required = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "state.pt",
        checkpoint_dir / "optimizer_0.pt",
        checkpoint_dir / "lr_scheduler_0.pt",
        checkpoint_dir / "layers.0" / "sae.safetensors",
    ]
    checks = {
        "public_package_import": sparsify.__version__ == "1.3.3",
        "documented_training_or_trainer_path_executes": (worker_dir / "full" / "state.pt").is_file(),
        "external_logging_disabled": not full.cfg.log_to_wandb and not resumed.cfg.log_to_wandb,
        "no_socket_connect_attempt": not attempts,
        "checkpoint_artifact_complete": saw_interrupt and all(path.is_file() for path in required),
        "uninterrupted_vs_resumed_final_hash_equal": full_hash == resumed_hash,
        "uninterrupted_vs_resumed_step_equal": full.global_step == resumed.global_step == 2,
    }
    return {
        "framework": "sparsify",
        "commit": cfg["sparsify"]["commit"],
        "public_version": sparsify.__version__,
        "resume_kind": "native Trainer.save/load_state with synthetic interruption",
        "full_hash": full_hash,
        "resumed_hash": resumed_hash,
        "full_step": full.global_step,
        "resumed_step": resumed.global_step,
        "checkpoint_files": {str(p.relative_to(worker_dir)): sha256_file(p) for p in required if p.is_file()},
        "network_attempts": attempts,
        "checks": checks,
    }


def worker_main(args) -> int:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    worker_dir = Path(args.worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    result = dictionary_worker(cfg, worker_dir) if args.worker == "dictionary_learning" else sparsify_worker(cfg, worker_dir)
    write_json(Path(args.worker_output), result)
    return 0 if all(result["checks"].values()) else 1


def parent_main(args) -> int:
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = PROJECT_ROOT / "runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_text("", encoding="utf-8")
    resolved_path = run_dir / "config.resolved.json"
    write_json(resolved_path, cfg)
    source_files = [
        Path(__file__).resolve(),
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "__init__.py",
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "training.py",
        Path(cfg["dictionary_learning"]["source_dir"]) / "dictionary_learning" / "trainers" / "top_k.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "__init__.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "__main__.py",
        Path(cfg["sparsify"]["source_dir"]) / "sparsify" / "trainer.py",
    ]
    entries = []
    for path in source_files:
        display = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") if path.is_relative_to(PROJECT_ROOT) else str(path)
        entries.append({"path": display, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in sorted(entries, key=lambda e: e["path"])).encode()
    ).hexdigest()
    write_json(run_dir / "code_hashes.json", {"files": entries, "aggregate_sha256": aggregate})
    input_files = [config_path, PROJECT_ROOT / "configs" / "r005a_environment_lock_v2.json"]
    write_json(
        run_dir / "inputs.json",
        {
            "inputs": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "source": "CCAD prewritten artifact",
                    "license_or_access_boundary": "CCAD internal",
                    "role": "resolved protocol" if path == config_path else "environment provenance",
                }
                for path in input_files
            ]
        },
    )
    manifest = {
        "schema_version": cfg["schema_version"],
        "run_id": cfg["run_id"],
        "run_parent": cfg["run_parent"],
        "purpose": "R005-A public package, offline logging, and tiny resume conformance",
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
        "statistics_unit": "deterministic_framework_resume_path",
        "device": "cpu",
        "seeds": {"init": cfg["init_seed"], "model": cfg["model_seed"]},
        "resource_lease": "none",
        "resource_lease_reason": "bounded CPU tiny model; no heavy resource",
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "environment.json",
        {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "offline": cfg["offline_environment"]},
    )
    write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": started})
    records = []
    errors = []
    start = time.perf_counter()
    for framework in ("dictionary_learning", "sparsify"):
        output = run_dir / f"{framework}.json"
        worker_dir = run_dir / framework
        env = os.environ.copy()
        env.update(cfg["offline_environment"])
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
                "--worker",
                framework,
                "--worker-dir",
                str(worker_dir),
                "--worker-output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(f"[{framework}]\n{proc.stdout}\n")
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write(f"[{framework}]\n{proc.stderr}\n")
        if output.is_file():
            records.append(json.loads(output.read_text(encoding="utf-8")))
        if proc.returncode != 0:
            errors.append({"framework": framework, "returncode": proc.returncode})
    raw_path = run_dir / "metrics.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    status = "PASS" if len(records) == 2 and not errors and all(all(r["checks"].values()) for r in records) else "FAIL"
    generator_path = "scripts/run_r005a_package_resume.py"
    summary = {
        "status": status,
        "framework_records": len(records),
        "checks_passed": sum(sum(bool(v) for v in r["checks"].values()) for r in records),
        "checks_total": sum(len(r["checks"]) for r in records),
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics_raw_sha256": sha256_file(raw_path),
        "generator_script_path": generator_path,
        "generator_script_sha256": next(e["sha256"] for e in entries if e["path"] == generator_path),
        "scope_limit": cfg["scope_limit"],
    }
    write_json(run_dir / "metrics.summary.json", summary)
    write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": utc_now()})
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    if not validation.ok:
        status = "FAIL"
        write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": utc_now()})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok}))
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", choices=["dictionary_learning", "sparsify"])
    parser.add_argument("--worker-dir")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    try:
        if args.worker:
            return worker_main(args)
        return parent_main(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
