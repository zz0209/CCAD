"""Run the bounded matched causal specificity gate for frozen Euclidean FCC."""
from __future__ import annotations

import argparse
from collections import defaultdict
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "r009"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import __version__ as scipy_version  # noqa: E402
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor  # noqa: E402
from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.causal_metric_probe import select_document_balanced_states  # noqa: E402
from ccad.fuzzy_correspondence import (  # noqa: E402
    fit_fuzzy_correspondence_from_kernels,
    loading_component_contributions,
    membership_weighted_contribution,
)
from ccad.subspace_transport import (  # noqa: E402
    fit_weighted_pca,
    fit_weighted_stitching,
    stable_seed,
    weighted_mean,
)
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402
from run_r011f1_euclidean_surface import condition_weights, local_kernels  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> tuple[dict, Path | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    inherited = raw.get("inherits_config")
    if inherited is None:
        return raw, None
    if set(raw) != {"inherits_config", "overrides"}:
        raise ValueError("suffix config may contain only inherits_config and overrides")
    base_path = ROOT / inherited
    base = json.loads(base_path.read_text(encoding="utf-8"))
    merged = {**base, **raw["overrides"]}
    merged["inherited_config_path"] = inherited
    merged["inherited_config_sha256"] = sha256(base_path)
    return merged, base_path


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, source: str, role: str, boundary: str = "internal") -> dict:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": boundary, "role": role,
    }


def reconstruct(matrix, indices: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices] @ dec, dtype=np.float64)


def projected(samples: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(samples, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return (centered @ basis) @ basis.T


def rescale_to_norm(value: np.ndarray, reference_norm: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or reference_norm <= 1e-12:
        return np.zeros_like(value), 0.0
    scale = reference_norm / norm
    return value * scale, scale


def random_memberships(source_count: int, target_count: int, rank: int, *parts: object) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(*parts))
    left = rng.normal(size=(source_count, rank))
    right = rng.normal(size=(target_count, rank))
    left, _ = np.linalg.qr(left)
    right, _ = np.linalg.qr(right)
    singular = np.linspace(1.0, 0.5, rank)
    magnitude = np.abs(left[:, :rank] @ np.diag(singular) @ right[:, :rank].T)
    coupling = magnitude / np.sum(magnitude)
    return np.sum(coupling, axis=1), np.sum(coupling, axis=0)


def random_loadings(source_count: int, target_count: int, rank: int, *parts: object) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(*parts))
    left, _ = np.linalg.qr(rng.normal(size=(source_count, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(target_count, rank)))
    return left[:, :rank], right[:, :rank]


def projector_components(samples: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(samples, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    scores = centered @ basis
    return np.einsum("tr,dr->rtd", scores, basis, optimize=True)


def endpoint_metrics(accum: dict, key: str) -> dict:
    source = accum[f"{key}_source_energy"]
    target = accum[f"{key}_target_energy"]
    cross = accum[f"{key}_cross_energy"]
    residual = max(0.0, source + target - 2.0 * cross)
    return {
        "source_energy": source,
        "target_energy": target,
        "cross_energy": cross,
        "residual_energy": residual,
        "normalized_effect_error": residual / source if source > 0 else None,
        "effect_bcc": 2.0 * cross / (source + target) if source + target > 0 else None,
        "source_effect_rms": np.sqrt(source / accum[f"{key}_elements"]) if accum[f"{key}_elements"] else None,
        "target_effect_rms": np.sqrt(target / accum[f"{key}_elements"]) if accum[f"{key}_elements"] else None,
        "source_off_query_fraction": accum[f"{key}_source_off_energy"] / source if source > 0 else None,
        "target_off_query_fraction": accum[f"{key}_target_off_energy"] / target if target > 0 else None,
    }


def qualifies(value: dict, gate: dict) -> tuple[bool, list[str]]:
    checks = {
        "effect_error": value["normalized_effect_error"] is not None and value["normalized_effect_error"] <= gate["maximum_effect_normalized_error"],
        "effect_bcc": value["effect_bcc"] is not None and value["effect_bcc"] >= gate["minimum_effect_bcc"],
        "effect_floor": value["source_effect_rms"] is not None and value["source_effect_rms"] >= gate["minimum_source_effect_rms"],
    }
    return all(checks.values()), [name for name, passed in checks.items() if not passed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg, inherited_config_path = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [
        Path(__file__).resolve(), ROOT / "src/ccad/fuzzy_correspondence.py",
        ROOT / "src/ccad/subspace_transport.py", ROOT / "src/ccad/activation_contract.py",
        ROOT / "scripts/run_r011f1_euclidean_surface.py", ROOT / "scripts/run_r009c_atom_discovery.py",
    ]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    paths = {
        "protocol": ROOT / cfg["protocol_document"],
        "decisions": ROOT / cfg["decisions_path"],
        "surface": ROOT / cfg["surface_path"],
        "query_panel": ROOT / cfg["query_panel_path"],
        "source_census": ROOT / cfg["source_census_path"],
        "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json",
        "raw_hook_manifest": Path(cfg["raw_hook_asset_dir"]) / "raw_hook_manifest.json",
        "token_manifest": ROOT / cfg["token_manifest_path"],
        "sequence_records": ROOT / cfg["sequence_records_path"],
    }
    if "loadings_path" in cfg:
        paths["loadings"] = ROOT / cfg["loadings_path"]
    if "frozen_selection_path" in cfg:
        paths["frozen_selection"] = ROOT / cfg["frozen_selection_path"]
    inputs = [file_entry(args.config.resolve(), "CCAD frozen config", "run_protocol")]
    if inherited_config_path is not None:
        inputs.append(file_entry(inherited_config_path, "CCAD inherited frozen config", "inherited_protocol"))
    inputs.extend(file_entry(path, "CCAD frozen upstream artifact", role) for role, path in paths.items())
    model_config = Path(cfg["model_local_dir"]) / "config.json"
    inputs.append(file_entry(model_config, cfg["model_id"], "model_config", cfg["model_license"]))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True,
        "mean_constants_source_split": "mean", "threshold_source_split": "calibration_frozen_before_endpoint_forwards",
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": cfg["source_seeds"],
        "resource_lease": "gpu-0 + cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded FCC/projector construction and downstream Pythia intervention forwards on calibration sequences",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines(),
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        expected = {
            "decisions": cfg["decisions_sha256"], "surface": cfg["surface_sha256"],
            "query_panel": cfg["query_panel_sha256"], "source_census": cfg["source_census_sha256"],
            "asset_manifest": cfg["asset_manifest_sha256"], "raw_hook_manifest": cfg["raw_hook_manifest_sha256"],
            "token_manifest": cfg["token_manifest_sha256"], "sequence_records": cfg["sequence_records_sha256"],
        }
        if "loadings" in paths:
            expected["loadings"] = cfg["loadings_sha256"]
        if "frozen_selection" in paths:
            expected["frozen_selection"] = cfg["frozen_selection_sha256"]
        bound = {name: sha256(paths[name]).lower() == value.lower() for name, value in expected.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        if cfg["audit_opened"] or cfg["forbidden_splits"] != ["audit"] or cfg["primary_endpoint"] != "next_state":
            raise ValueError("causal protocol boundary drift")

        decisions = [json.loads(line) for line in paths["decisions"].read_text(encoding="utf-8").splitlines() if line]
        surface_rows = [json.loads(line) for line in paths["surface"].read_text(encoding="utf-8").splitlines() if line]
        surface = {(row["source_seed"], row["source_atom"], row["target_seed"], row["rank"]): row for row in surface_rows if row["query_role"] == "anchor"}
        query_rows = [json.loads(line) for line in paths["query_panel"].read_text(encoding="utf-8").splitlines() if line]
        query_lookup = {(row["seed"], row["atom"]): row for row in query_rows}
        selected = []
        for stratum in range(8):
            eligible = [row for row in decisions if row["decision"] == "FOUND_RELATION" and row["energy_stratum"] == stratum]
            if not eligible:
                raise ValueError(f"no frozen found relation in stratum {stratum}")
            selected.append(min(eligible, key=lambda row: (query_lookup[(row["source_seed"], row["source_atom"])]["selection_hash"], row["target_seed"])))
        if len(selected) != cfg["selected_units"]:
            raise ValueError("bounded unit count drift")
        if "frozen_selection" in paths:
            frozen_units = json.loads(paths["frozen_selection"].read_text(encoding="utf-8"))["units"]
            selected_keys = [(row["source_seed"], row["target_seed"], row["source_atom"], row["energy_stratum"], row["selected_rank"]) for row in selected]
            frozen_keys = [(row["source_seed"], row["target_seed"], row["source_atom"], row["energy_stratum"], row["rank"]) for row in frozen_units]
            if selected_keys != frozen_keys:
                raise ValueError("fresh causal suffix changed the endpoint-blind unit selection")

        census = [json.loads(line) for line in paths["source_census"].read_text(encoding="utf-8").splitlines() if line]
        stats = {(int(row["seed"]), int(row["atom"])): row for row in census}
        means = {seed: np.asarray([stats[(seed, atom)]["mean_code"] for atom in range(cfg["num_latents"])], dtype=np.float64) for seed in cfg["source_seeds"]}
        asset_dir = Path(cfg["bulk_asset_dir"])
        asset_manifest = json.loads(paths["asset_manifest"].read_text(encoding="utf-8"))
        split_tokens = {row["split"]: int(row["tokens"]) for row in asset_manifest["splits"]}
        matrices = {split: {seed: sparse_codes(asset_dir, split, seed, split_tokens[split], cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]} for split in ("mean", "discovery", "calibration")}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}

        raw_manifest = json.loads(paths["raw_hook_manifest"].read_text(encoding="utf-8"))
        raw_meta = {row["split"]: row for row in raw_manifest["splits"]}
        raw = {split: np.memmap(raw_meta[split]["path"], dtype="<f4", mode="r").reshape(raw_meta[split]["shape"]) for split in ("mean", "discovery", "calibration")}
        loading_data = np.load(paths["loadings"], allow_pickle=False) if "loadings" in paths else None
        sequences = json.loads(paths["sequence_records"].read_text(encoding="utf-8"))["sequences"]
        global_states = select_document_balanced_states(
            sequences, split="discovery", count=cfg["global_control_tokens"],
            token_positions=tuple(cfg["global_control_state_positions"]), salt=cfg["global_control_state_salt"],
        )
        global_rows = np.asarray([int(row["sequence_index"]) * cfg["context_length"] + int(row["token_position"]) for row in global_states], dtype=np.int64)
        global_weights = np.full(len(global_rows), 1.0 / len(global_rows))

        payload = []
        for decision in selected:
            source_seed, target_seed, atom, rank = (int(decision[name]) for name in ("source_seed", "target_seed", "source_atom", "selected_rank"))
            row = surface[(source_seed, atom, target_seed, rank)]
            source_ids = [int(value) for value in row["source_candidate_ids"]]
            target_ids = [int(value) for value in row["target_candidate_ids"]]
            primary_source_membership = np.asarray(row["source_membership"], dtype=np.float64)
            primary_target_membership = np.asarray(row["target_membership"], dtype=np.float64)

            discovery_rows, discovery_weights = condition_weights(matrices["discovery"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
            mean_rows, mean_weights = condition_weights(matrices["mean"][source_seed], [atom], cfg["condition_weight_power"], cfg["max_condition_tokens_per_split"])
            source_discovery = reconstruct(matrices["discovery"][source_seed], discovery_rows, decoders[source_seed])
            target_discovery = reconstruct(matrices["discovery"][target_seed], discovery_rows, decoders[target_seed])
            source_mean = weighted_mean(reconstruct(matrices["mean"][source_seed], mean_rows, decoders[source_seed]), mean_weights)
            target_mean = weighted_mean(reconstruct(matrices["mean"][target_seed], mean_rows, decoders[target_seed]), mean_weights)
            source_basis, _ = fit_weighted_pca(source_discovery, discovery_weights, source_mean, rank, random_seed=stable_seed(cfg["run_id"], source_seed, atom, target_seed, "source_pca"), oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"], relative_tolerance=cfg["eigenvalue_relative_tolerance"])
            target_basis, _ = fit_weighted_pca(target_discovery, discovery_weights, target_mean, rank, random_seed=stable_seed(cfg["run_id"], source_seed, atom, target_seed, "target_pca"), oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"], relative_tolerance=cfg["eigenvalue_relative_tolerance"])
            stitch_source, stitch_target, _ = fit_weighted_stitching(source_discovery, target_discovery, discovery_weights, source_mean, target_mean, rank, random_seed=stable_seed(cfg["run_id"], source_seed, atom, target_seed, "stitching"), oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"], relative_tolerance=cfg["eigenvalue_relative_tolerance"])
            raw_mean = weighted_mean(np.asarray(raw["mean"][mean_rows], dtype=np.float64), mean_weights)
            raw_basis, _ = fit_weighted_pca(np.asarray(raw["discovery"][discovery_rows], dtype=np.float64), discovery_weights, raw_mean, rank, random_seed=stable_seed(cfg["run_id"], source_seed, atom, target_seed, "raw_pca"), oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"], relative_tolerance=cfg["eigenvalue_relative_tolerance"])

            global_kernel = local_kernels(
                matrices["discovery"][source_seed], matrices["discovery"][target_seed], source_ids, target_ids,
                decoders[source_seed], decoders[target_seed], means[source_seed], means[target_seed], global_rows, global_weights,
            )
            global_relation = fit_fuzzy_correspondence_from_kernels(global_kernel, rank=rank, contrast_strength=0.0, ridge_fraction=1e-6)
            random_source_membership, random_target_membership = random_memberships(len(source_ids), len(target_ids), rank, cfg["random_relation_salt"], source_seed, atom, target_seed, rank)
            random_source_loadings, random_target_loadings = random_loadings(len(source_ids), len(target_ids), rank, cfg["random_relation_salt"], source_seed, atom, target_seed, rank, "loadings")
            if loading_data is not None:
                loading_index = int(row["loading_index"])
                primary_source_loadings = np.asarray(loading_data["source_loadings"][loading_index, :len(source_ids), :rank], dtype=np.float64)
                primary_target_loadings = np.asarray(loading_data["target_loadings"][loading_index, :len(target_ids), :rank], dtype=np.float64)
            else:
                primary_source_loadings = None
                primary_target_loadings = None

            calibration_code = np.asarray(matrices["calibration"][source_seed][:, atom].toarray(), dtype=np.float64).reshape(-1)
            sequence_energy = np.sum(calibration_code.reshape(-1, cfg["context_length"]) ** 2, axis=1)
            order = np.lexsort((np.arange(sequence_energy.size), -sequence_energy))
            sequence_ids = [int(value) for value in order[:cfg["sequences_per_unit"]] if sequence_energy[value] > 0]
            if len(sequence_ids) != cfg["sequences_per_unit"]:
                raise ValueError(f"insufficient active calibration sequences for seed={source_seed} atom={atom}")
            payload.append({
                "decision": decision, "surface": row, "source_ids": source_ids, "target_ids": target_ids,
                "primary_source_membership": primary_source_membership, "primary_target_membership": primary_target_membership,
                "global_source_membership": global_relation.source_membership, "global_target_membership": global_relation.target_membership,
                "random_source_membership": random_source_membership, "random_target_membership": random_target_membership,
                "primary_source_loadings": primary_source_loadings, "primary_target_loadings": primary_target_loadings,
                "global_source_loadings": global_relation.source_loadings, "global_target_loadings": global_relation.target_loadings,
                "random_source_loadings": random_source_loadings, "random_target_loadings": random_target_loadings,
                "source_mean": source_mean, "target_mean": target_mean, "source_basis": source_basis[:, :rank], "target_basis": target_basis[:, :rank],
                "stitch_source": stitch_source[:, :rank], "stitch_target": stitch_target[:, :rank], "raw_mean": raw_mean, "raw_basis": raw_basis[:, :rank],
                "sequence_ids": sequence_ids, "sequence_energy": [float(sequence_energy[value]) for value in sequence_ids],
            })

        token_manifest = json.loads(paths["token_manifest"].read_text(encoding="utf-8"))
        calibration_info = token_manifest["outputs"]["calibration"]
        token_path = ROOT / "runs" / cfg["paired_corpus_run"] / calibration_info["path"]
        tokens = np.memmap(token_path, dtype="<u2", mode="r").reshape(calibration_info["sequences"], cfg["context_length"])
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "CUBLAS_WORKSPACE_CONFIG": cfg["cublas_workspace_config"]})
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(cfg["model_local_dir"], local_files_only=True, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]).eval().to(device)
        model.config.use_cache = False
        hook_module = model.get_submodule(cfg["hook_module_path"])
        next_module = model.get_submodule(cfg["next_module_path"])
        contract = HookPointContract(cfg["hook_module_path"], 5, "resid_post", cfg["hook_hidden_size"])

        def forward(batch, contribution=None, no_op=False):
            captured = {}
            def hook(_module, _inputs, output):
                primary = extract_primary_hook_tensor(output, contract)
                captured["hook"] = primary.detach().clone()
                if no_op:
                    return replace_primary_hook_tensor(output, primary.clone(), contract)
                if contribution is not None:
                    return replace_primary_hook_tensor(output, primary - contribution, contract)
                return None
            def next_hook(_module, _inputs, output):
                captured["next"] = (output[0] if isinstance(output, tuple) else output).detach().clone()
            first = hook_module.register_forward_hook(hook)
            second = next_module.register_forward_hook(next_hook)
            try:
                with torch.no_grad():
                    result = model(batch, use_cache=False)
            finally:
                second.remove(); first.remove()
            return captured["hook"], captured["next"], result.logits.detach()

        pair_accum = {(item["decision"]["source_seed"], item["decision"]["source_atom"], item["decision"]["target_seed"], method): defaultdict(float) for item in payload for method in cfg["evaluated_methods"]}
        unit_rows, noop_errors, replay_abs, replay_relative = [], [], [], []
        total_forwards = 0
        expected_forwards = 0
        compute_started = time.perf_counter()
        for item in payload:
            decision = item["decision"]
            source_seed, target_seed, atom, rank = (int(decision[name]) for name in ("source_seed", "target_seed", "source_atom", "selected_rank"))
            for sequence_id in item["sequence_ids"]:
                begin, stop = sequence_id * cfg["context_length"], (sequence_id + 1) * cfg["context_length"]
                batch = torch.from_numpy(np.asarray(tokens[sequence_id:sequence_id + 1], dtype=np.int64)).to(device)
                hook, baseline_next, baseline_logits = forward(batch)
                _, noop_next, noop_logits = forward(batch, no_op=True)
                total_forwards += 2
                noop_errors.append(max(float((noop_next - baseline_next).abs().max().item()), float((noop_logits - baseline_logits).abs().max().item())))
                replay = np.asarray(raw["calibration"][begin:stop], dtype=np.float64)
                live = hook[0].detach().cpu().numpy().astype(np.float64)
                difference = replay - live
                replay_abs.append(float(np.max(np.abs(difference))))
                replay_relative.append(float(np.sqrt(np.mean(difference * difference)) / max(np.sqrt(np.mean(live * live)), 1e-12)))

                source_codes = matrices["calibration"][source_seed][begin:stop][:, item["source_ids"]]
                target_codes = matrices["calibration"][target_seed][begin:stop][:, item["target_ids"]]
                primary_source = membership_weighted_contribution(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["primary_source_membership"])
                primary_target = membership_weighted_contribution(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["primary_target_membership"])
                global_source = membership_weighted_contribution(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["global_source_membership"])
                global_target = membership_weighted_contribution(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["global_target_membership"])
                random_source = membership_weighted_contribution(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["random_source_membership"])
                random_target = membership_weighted_contribution(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["random_target_membership"])
                source_reconstruction = reconstruct(matrices["calibration"][source_seed], np.arange(begin, stop), decoders[source_seed])
                target_reconstruction = reconstruct(matrices["calibration"][target_seed], np.arange(begin, stop), decoders[target_seed])
                source_query_code = np.asarray(matrices["calibration"][source_seed][begin:stop, atom].toarray(), dtype=np.float64).reshape(-1)
                target_single_atom = int(item["target_ids"][0])
                target_single_code = np.asarray(matrices["calibration"][target_seed][begin:stop, target_single_atom].toarray(), dtype=np.float64).reshape(-1)
                if cfg["relation_intervention"] == "signed_paired_loading_components_with_quadratic_unit_aggregation":
                    source_rank_ids = np.argsort(-item["primary_source_membership"])[:rank]
                    target_rank_ids = np.argsort(-item["primary_target_membership"])[:rank]
                    native_source_components = np.stack([
                        (np.asarray(source_codes[:, int(index)].toarray(), dtype=np.float64).reshape(-1) - means[source_seed][item["source_ids"][int(index)]])[:, None] * decoders[source_seed][item["source_ids"][int(index)]][None, :]
                        for index in source_rank_ids
                    ])
                    native_target_components = np.stack([
                        (np.asarray(target_codes[:, int(index)].toarray(), dtype=np.float64).reshape(-1) - means[target_seed][item["target_ids"][int(index)]])[:, None] * decoders[target_seed][item["target_ids"][int(index)]][None, :]
                        for index in target_rank_ids
                    ])
                    methods = {
                        "EUCLIDEAN_FCC_RELATION": (
                            loading_component_contributions(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["primary_source_loadings"]),
                            loading_component_contributions(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["primary_target_loadings"]),
                        ),
                        "GLOBAL_FCC_RELATION": (
                            loading_component_contributions(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["global_source_loadings"]),
                            loading_component_contributions(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["global_target_loadings"]),
                        ),
                        "RAW_HOOK_QUERY_PCA": (projector_components(replay, item["raw_mean"], item["raw_basis"]), projector_components(replay, item["raw_mean"], item["raw_basis"])),
                        "SAE_QUERY_MARGINAL_PCA": (projector_components(source_reconstruction, item["source_mean"], item["source_basis"]), projector_components(target_reconstruction, item["target_mean"], item["target_basis"])),
                        "RELAXED_PAIRED_STITCHING": (projector_components(source_reconstruction, item["source_mean"], item["stitch_source"]), projector_components(target_reconstruction, item["target_mean"], item["stitch_target"])),
                        "BEST_FUNCTIONAL_SINGLE_NATIVE": (native_source_components, native_target_components),
                        "MATCHED_RANDOM_RELATION": (
                            loading_component_contributions(source_codes, decoders[source_seed][item["source_ids"]], means[source_seed][item["source_ids"]], item["random_source_loadings"]),
                            loading_component_contributions(target_codes, decoders[target_seed][item["target_ids"]], means[target_seed][item["target_ids"]], item["random_target_loadings"]),
                        ),
                    }
                else:
                    methods = {
                        "EUCLIDEAN_FCC_RELATION": (primary_source[None], primary_target[None]),
                        "GLOBAL_FCC_RELATION": (global_source[None], global_target[None]),
                        "RAW_HOOK_QUERY_PCA": (projected(replay, item["raw_mean"], item["raw_basis"])[None], projected(replay, item["raw_mean"], item["raw_basis"])[None]),
                        "SAE_QUERY_MARGINAL_PCA": (projected(source_reconstruction, item["source_mean"], item["source_basis"])[None], projected(target_reconstruction, item["target_mean"], item["target_basis"])[None]),
                        "RELAXED_PAIRED_STITCHING": (projected(source_reconstruction, item["source_mean"], item["stitch_source"])[None], projected(target_reconstruction, item["target_mean"], item["stitch_target"])[None]),
                        "BEST_FUNCTIONAL_SINGLE_NATIVE": (((source_query_code - means[source_seed][atom])[:, None] * decoders[source_seed][atom][None, :])[None], ((target_single_code - means[target_seed][target_single_atom])[:, None] * decoders[target_seed][target_single_atom][None, :])[None]),
                        "MATCHED_RANDOM_RELATION": (random_source[None], random_target[None]),
                    }
                reference_norm = float(np.linalg.norm(methods["EUCLIDEAN_FCC_RELATION"][0]))
                active_mask = torch.from_numpy(source_query_code != 0).to(device)
                off_mask = ~active_mask
                expected_forwards += 2 + 2 * sum(values[0].shape[0] for values in methods.values())
                for method, (source_block, target_block) in methods.items():
                    source_scale = reference_norm / max(float(np.linalg.norm(source_block)), 1e-12)
                    target_scale = reference_norm / max(float(np.linalg.norm(target_block)), 1e-12)
                    source_matched_block = source_block * source_scale
                    target_matched_block = target_block * target_scale
                    key = (source_seed, atom, target_seed, method)
                    acc = pair_accum[key]
                    for component_index, (source_matched, target_matched) in enumerate(zip(source_matched_block, target_matched_block)):
                        source_tensor = torch.from_numpy(source_matched.astype(np.float32))[None].to(device)
                        target_tensor = torch.from_numpy(target_matched.astype(np.float32))[None].to(device)
                        _, source_next, source_logits = forward(batch, contribution=source_tensor)
                        _, target_next, target_logits = forward(batch, contribution=target_tensor)
                        total_forwards += 2
                        for endpoint, base, source_output, target_output in (("next_state", baseline_next, source_next, target_next), ("next_logits", baseline_logits, source_logits, target_logits)):
                            source_effect = base - source_output
                            target_effect = base - target_output
                            acc[f"{endpoint}_source_energy"] += float(torch.sum(source_effect * source_effect).item())
                            acc[f"{endpoint}_target_energy"] += float(torch.sum(target_effect * target_effect).item())
                            acc[f"{endpoint}_cross_energy"] += float(torch.sum(source_effect * target_effect).item())
                            acc[f"{endpoint}_elements"] += source_effect.numel()
                            acc[f"{endpoint}_source_off_energy"] += float(torch.sum(source_effect[0, off_mask] ** 2).item())
                            acc[f"{endpoint}_target_off_energy"] += float(torch.sum(target_effect[0, off_mask] ** 2).item())
                        acc["hook_source_energy"] += float(np.sum(source_matched * source_matched))
                        acc["hook_target_energy"] += float(np.sum(target_matched * target_matched))
                        unit_rows.append({
                            "source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": decision["energy_stratum"],
                            "sequence_id": sequence_id, "method": method, "rank": rank, "component_index": component_index,
                            "active_query_tokens": int(np.sum(source_query_code != 0)), "reference_hook_norm": reference_norm,
                            "source_unscaled_block_norm": float(np.linalg.norm(source_block)), "target_unscaled_block_norm": float(np.linalg.norm(target_block)),
                            "source_energy_scale": source_scale, "target_energy_scale": target_scale,
                        })

        pair_rows = []
        for item in payload:
            decision = item["decision"]
            source_seed, target_seed, atom, rank = (int(decision[name]) for name in ("source_seed", "target_seed", "source_atom", "selected_rank"))
            for method in cfg["evaluated_methods"]:
                acc = pair_accum[(source_seed, atom, target_seed, method)]
                endpoints = {name: endpoint_metrics(acc, name) for name in ("next_state", "next_logits")}
                qualified, failures = qualifies(endpoints[cfg["primary_endpoint"]], cfg["causal_qualification"])
                pair_rows.append({
                    "source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": decision["energy_stratum"],
                    "method": method, "rank": rank, "sequence_ids": item["sequence_ids"], "endpoints": endpoints,
                    "qualifies_primary_endpoint": bool(qualified), "qualification_failures": failures,
                    "hook_source_energy": acc["hook_source_energy"], "hook_target_energy": acc["hook_target_energy"],
                })
            pair_rows.append({
                "source_seed": source_seed, "target_seed": target_seed, "source_atom": atom, "energy_stratum": decision["energy_stratum"],
                "method": "MSCC_REFUSAL", "rank": rank, "sequence_ids": item["sequence_ids"], "endpoints": None,
                "qualifies_primary_endpoint": False, "qualification_failures": ["frozen_native_support_unresolved"],
                "availability": "UNAVAILABLE_REFUSAL_NOT_ZERO_INTERVENTION",
            })

        endpoint = cfg["primary_endpoint"]
        evaluated = cfg["evaluated_methods"]
        coverages = {method: float(np.mean([row["qualifies_primary_endpoint"] for row in pair_rows if row["method"] == method])) for method in evaluated}
        medians = {}
        for method in evaluated:
            rows = [row for row in pair_rows if row["method"] == method]
            medians[method] = {
                "normalized_effect_error": float(np.median([row["endpoints"][endpoint]["normalized_effect_error"] for row in rows])),
                "effect_bcc": float(np.median([row["endpoints"][endpoint]["effect_bcc"] for row in rows])),
                "source_off_query_fraction": float(np.median([row["endpoints"][endpoint]["source_off_query_fraction"] for row in rows])),
                "source_effect_rms": float(np.median([row["endpoints"][endpoint]["source_effect_rms"] for row in rows])),
            }
        primary = medians["EUCLIDEAN_FCC_RELATION"]
        gains = {}
        minimum_gain = cfg["progression"]["minimum_gain_against_each_raw_and_global_control"]
        for control in ("RAW_HOOK_QUERY_PCA", "GLOBAL_FCC_RELATION"):
            gains[control] = {
                "effect_consistency_gain": medians[control]["normalized_effect_error"] - primary["normalized_effect_error"],
                "query_specificity_gain": medians[control]["source_off_query_fraction"] - primary["source_off_query_fraction"],
            }
            gains[control]["passes_either_axis"] = bool(max(gains[control]["effect_consistency_gain"], gains[control]["query_specificity_gain"]) >= minimum_gain)
        conformance = {
            "noop": max(noop_errors) <= cfg["causal_qualification"]["maximum_noop_absolute_error"],
            "raw_replay_absolute": max(replay_abs) <= cfg["raw_hook_replay_tolerance"]["maximum_absolute_error"],
            "raw_replay_relative": max(replay_relative) <= cfg["raw_hook_replay_tolerance"]["maximum_relative_rms_error"],
        }
        primary_coverage = coverages["EUCLIDEAN_FCC_RELATION"]
        if not all(conformance.values()):
            screen_decision = "FAIL_NUMERICAL_CONFORMANCE"
        elif primary_coverage < cfg["progression"]["minimum_primary_coverage"]:
            screen_decision = "STOP_FCC_CAUSAL_EFFECT_OR_CONSISTENCY_FLOOR"
        elif not all(value["passes_either_axis"] for value in gains.values()):
            screen_decision = "STOP_FCC_CAUSAL_SPECIFICITY_NOT_IDENTIFIED"
        else:
            screen_decision = "PROCEED_TO_FULL_640_PREAUDIT_FREEZE"

        unit_path = run_dir / "intervention_units.jsonl"
        unit_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in unit_rows), encoding="utf-8")
        pair_path = run_dir / "causal_pair_metrics.jsonl"
        pair_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows), encoding="utf-8")
        selection_path = run_dir / "selected_units_and_sequences.json"
        write_json(selection_path, {"units": [{
            "source_seed": item["decision"]["source_seed"], "target_seed": item["decision"]["target_seed"],
            "source_atom": item["decision"]["source_atom"], "energy_stratum": item["decision"]["energy_stratum"],
            "rank": item["decision"]["selected_rank"], "sequence_ids": item["sequence_ids"], "sequence_energy": item["sequence_energy"],
        } for item in payload]})
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "eight_strata_once": sorted(int(item["decision"]["energy_stratum"]) for item in payload) == list(range(8)),
            "found_relations_only": all(item["decision"]["decision"] == "FOUND_RELATION" for item in payload),
            "endpoint_blind_two_sequences_per_unit": all(len(item["sequence_ids"]) == cfg["sequences_per_unit"] for item in payload),
            "complete_method_grid": len(pair_rows) == cfg["selected_units"] * len(cfg["methods"]),
            "matched_rank": all(row["rank"] == next(item["decision"]["selected_rank"] for item in payload if item["decision"]["source_seed"] == row["source_seed"] and item["decision"]["source_atom"] == row["source_atom"] and item["decision"]["target_seed"] == row["target_seed"]) for row in pair_rows),
            "matched_energy_finite": all(np.isfinite(row["reference_hook_norm"]) and row["reference_hook_norm"] > 0 for row in unit_rows),
            "noop_conformance": conformance["noop"],
            "raw_hook_replay_conformance": conformance["raw_replay_absolute"] and conformance["raw_replay_relative"],
            "primary_endpoint_frozen": cfg["primary_endpoint"] == "next_state",
            "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"],
            "model_forward_count_exact": total_forwards == expected_forwards,
        }
        record = {
            "checks": {name: bool(value) for name, value in checks.items()}, "screen_decision": screen_decision,
            "selected_units": len(payload), "pair_rows": len(pair_rows), "unit_rows": len(unit_rows),
            "primary_endpoint": endpoint, "coverages": coverages, "method_medians": medians, "primary_gains": gains,
            "maximum_noop_absolute_error": max(noop_errors), "maximum_raw_hook_replay_absolute_error": max(replay_abs),
            "maximum_raw_hook_replay_relative_rms_error": max(replay_relative), "model_forwards": total_forwards,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "wall_seconds": time.perf_counter() - compute_started, "intervention_units_sha256": sha256(unit_path),
            "causal_pair_metrics_sha256": sha256(pair_path), "selection_sha256": sha256(selection_path),
            "scope_limit": cfg["scope_limit"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version,
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device), "platform": platform.platform(),
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})

    raw_path = run_dir / "metrics.raw.jsonl"
    raw_path.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_path),
        "generator_script_path": "scripts/run_r011f1_euclidean_causal_gate.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "screen_decision": record.get("screen_decision") if record else None}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "screen_decision": record.get("screen_decision") if record else None, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
