"""Fail-closed checkpoint helpers for CCAD-controlled SAE training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ccad-sparsify-exact-state-v1"


class _RestoreRngOnFirstAccess:
    """Neutralize the extra DataLoader iterator seed draw on a resumed fit."""

    def __init__(self, dataset: Any, cpu_state: Any, cuda_states: list[Any]):
        self._dataset = dataset
        self._cpu_state = cpu_state
        self._cuda_states = cuda_states
        self._armed = True

    def __len__(self) -> int:
        return len(self._dataset)

    def _restore_once(self) -> None:
        if not self._armed:
            return
        import torch

        torch.set_rng_state(self._cpu_state.cpu())
        if torch.cuda.is_available() and self._cuda_states:
            torch.cuda.set_rng_state_all([state.cpu() for state in self._cuda_states])
        self._armed = False

    def __getitem__(self, index: Any) -> Any:
        self._restore_once()
        return self._dataset[index]

    def select(self, indices: Any) -> "_RestoreRngOnFirstAccess":
        if not hasattr(self._dataset, "select"):
            raise AttributeError("wrapped dataset does not implement select")
        selected = self._dataset.select(indices)
        return _RestoreRngOnFirstAccess(selected, self._cpu_state, self._cuda_states)


def canonicalize_sparsify_multiseed_state(trainer: Any) -> None:
    """Make per-SAE bookkeeping use the same seed-suffixed key space as `trainer.saes`."""
    sae_names = set(trainer.saes)
    counter_names = set(trainer.num_tokens_since_fired)
    if counter_names != sae_names:
        raise ValueError(f"counter/SAE key mismatch: counters={sorted(counter_names)}, saes={sorted(sae_names)}")
    if isinstance(trainer.best_loss, dict) and set(trainer.best_loss) != sae_names:
        old = dict(trainer.best_loss)
        migrated = {}
        for sae_name in sorted(sae_names):
            hook_name = sae_name.rsplit("/seed", 1)[0]
            if sae_name in old:
                migrated[sae_name] = old[sae_name]
            elif hook_name in old:
                migrated[sae_name] = old[hook_name]
            else:
                raise ValueError(f"cannot map best_loss key for {sae_name!r}")
        trainer.best_loss = migrated


def save_sparsify_exact_state(trainer: Any, directory: Path, data_cursor_examples: int) -> dict:
    """Save all state needed for exact local multi-seed continuation."""
    import torch
    from safetensors.torch import save_file

    canonicalize_sparsify_multiseed_state(trainer)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sae_names = sorted(trainer.saes)
    for name in sae_names:
        target = directory / "saes" / name / "sae.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in trainer.saes[name].state_dict().items()},
            str(target),
        )
    state = {
        "schema_version": SCHEMA_VERSION,
        "global_step": int(trainer.global_step),
        "data_cursor_examples": int(data_cursor_examples),
        "sae_names": sae_names,
        "optimizers": [optimizer.state_dict() for optimizer in trainer.optimizers],
        "lr_schedulers": [scheduler.state_dict() for scheduler in trainer.lr_schedulers],
        "num_tokens_since_fired": {name: value.detach().cpu() for name, value in trainer.num_tokens_since_fired.items()},
        "best_loss": trainer.best_loss,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temp_state = directory / "state.pt.tmp"
    torch.save(state, temp_state)
    os.replace(temp_state, directory / "state.pt")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "global_step": state["global_step"],
        "data_cursor_examples": state["data_cursor_examples"],
        "sae_names": sae_names,
        "optimizer_count": len(state["optimizers"]),
        "scheduler_count": len(state["lr_schedulers"]),
        "counter_names": sorted(state["num_tokens_since_fired"]),
        "best_loss_names": sorted(state["best_loss"]) if isinstance(state["best_loss"], dict) else None,
    }
    temp_metadata = directory / "metadata.json.tmp"
    temp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_metadata, directory / "metadata.json")
    return metadata


def load_sparsify_exact_state(trainer: Any, directory: Path, expected_data_cursor_examples: int | None = None) -> dict:
    """Restore exact multi-seed state and reject key/count/cursor drift."""
    import torch
    from safetensors.torch import load_model

    directory = Path(directory)
    state = torch.load(directory / "state.pt", map_location=trainer.model.device, weights_only=True)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected checkpoint schema: {state.get('schema_version')!r}")
    sae_names = sorted(trainer.saes)
    if state.get("sae_names") != sae_names:
        raise ValueError("checkpoint SAE names differ from initialized trainer")
    if set(state["num_tokens_since_fired"]) != set(sae_names):
        raise ValueError("checkpoint counters do not cover exact SAE key space")
    if isinstance(state["best_loss"], dict) and set(state["best_loss"]) != set(sae_names):
        raise ValueError("checkpoint best_loss does not cover exact SAE key space")
    if len(state["optimizers"]) != len(trainer.optimizers) or len(state["lr_schedulers"]) != len(trainer.lr_schedulers):
        raise ValueError("optimizer or scheduler count differs")
    if expected_data_cursor_examples is not None and state["data_cursor_examples"] != expected_data_cursor_examples:
        raise ValueError("checkpoint data cursor differs from expected cursor")
    for name in sae_names:
        load_model(trainer.saes[name], directory / "saes" / name / "sae.safetensors", device=str(trainer.model.device))
    for optimizer, optimizer_state in zip(trainer.optimizers, state["optimizers"]):
        optimizer.load_state_dict(optimizer_state)
    for scheduler, scheduler_state in zip(trainer.lr_schedulers, state["lr_schedulers"]):
        scheduler.load_state_dict(scheduler_state)
    trainer.global_step = int(state["global_step"])
    trainer.num_tokens_since_fired = {
        name: value.to(trainer.model.device) for name, value in state["num_tokens_since_fired"].items()
    }
    trainer.best_loss = state["best_loss"]
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    if torch.cuda.is_available() and state["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng_state_all"]])
    # A fresh DataLoader iterator consumes global RNG to create its base seed,
    # even with shuffle=False.  The uninterrupted trajectory has no such extra
    # iterator at the checkpoint boundary, so restore immediately before the
    # first resumed sample is materialized.  This also precedes any stochastic
    # dataset transform.
    trainer.dataset = _RestoreRngOnFirstAccess(
        trainer.dataset,
        state["cpu_rng_state"],
        state["cuda_rng_state_all"],
    )
    return {
        "schema_version": state["schema_version"],
        "global_step": trainer.global_step,
        "data_cursor_examples": int(state["data_cursor_examples"]),
        "sae_names": sae_names,
    }
