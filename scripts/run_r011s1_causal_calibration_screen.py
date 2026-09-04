"""Run the bounded R011-S1 matched-rank/matched-energy causal screen."""
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
from ccad.subspace_transport import (  # noqa: E402
    fit_weighted_pca,
    fit_weighted_stitching,
    random_orthonormal_basis,
    select_weighted_support,
    stable_seed,
    weighted_mean,
)
from run_r009c_atom_discovery import decoder, sparse_codes  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_entry(path: Path, source: str, role: str, boundary: str = "internal") -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


def dense_code(matrix, atom: int) -> np.ndarray:
    return np.asarray(matrix[:, atom].toarray(), dtype=np.float64).reshape(-1)


def reconstruct(matrix, indices: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices] @ dec, dtype=np.float64)


def deterministic_sample(total: int, count: int, *parts: object) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(*parts))
    return np.sort(rng.choice(total, size=min(total, count), replace=False)).astype(np.int64)


def projected(samples: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(samples, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return (centered @ basis) @ basis.T


def rescale_to_norm(value: np.ndarray, reference_norm: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or reference_norm <= 1e-12:
        return np.zeros_like(value), 0.0
    scale = reference_norm / norm
    return value * scale, scale


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
        "source_off_target_fraction": accum[f"{key}_source_off_energy"] / source if source > 0 else None,
        "target_off_target_fraction": accum[f"{key}_target_off_energy"] / target if target > 0 else None,
    }


def qualifies(value: dict, cfg: dict) -> tuple[bool, list[str]]:
    gate = cfg["causal_qualification"]
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
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/subspace_transport.py", ROOT / "src/ccad/activation_contract.py", ROOT / "scripts/run_r009c_atom_discovery.py"]
    code_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths]
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": aggregate(code_rows)})
    paths = {
        "feasibility_metrics": ROOT / cfg["feasibility_metrics_path"],
        "primary_decisions": ROOT / cfg["primary_decisions_path"],
        "projectors": ROOT / cfg["projectors_path"],
        "query_panel": ROOT / cfg["query_panel_path"],
        "native_calibration": ROOT / cfg["native_calibration_path"],
        "asset_manifest": Path(cfg["bulk_asset_dir"]) / "asset_manifest.json",
        "raw_hook_manifest": Path(cfg["raw_hook_asset_dir"]) / "raw_hook_manifest.json",
        "token_manifest": ROOT / cfg["token_manifest_path"],
    }
    inputs = [file_entry(args.config.resolve(), "CCAD frozen config", "protocol")]
    for role, path in paths.items():
        inputs.append(file_entry(path, "CCAD frozen upstream artifact", role))
    model_config = Path(cfg["model_local_dir"]) / "config.json"
    inputs.append(file_entry(model_config, cfg["model_id"], "model_config", cfg["model_license"]))
    write_json(run_dir / "inputs.json", {"inputs": inputs})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": aggregate(code_rows), "audit_opened": False, "candidate_family_frozen": True,
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": cfg["device"], "seeds": cfg["source_seeds"],
        "resource_lease": "gpu-0 + cpu-heavy + disk-d-io via nested SAE Lab resource_manager.run",
        "resource_lease_reason": "bounded projector recomputation and downstream Pythia intervention forwards on calibration sequences",
        "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    record, error, status = None, None, "FAIL"
    try:
        expected_hashes = {
            "feasibility_metrics": cfg["feasibility_metrics_sha256"], "primary_decisions": cfg["primary_decisions_sha256"],
            "projectors": cfg["projectors_sha256"], "query_panel": cfg["query_panel_sha256"],
            "native_calibration": cfg["native_calibration_sha256"], "asset_manifest": cfg["asset_manifest_sha256"],
            "raw_hook_manifest": cfg["raw_hook_manifest_sha256"], "token_manifest": cfg["token_manifest_sha256"],
        }
        bound = {name: sha256(paths[name]).lower() == expected.lower() for name, expected in expected_hashes.items()}
        if not all(bound.values()):
            raise ValueError(f"frozen input mismatch: {bound}")
        if cfg["rank"] != 1 or cfg["selected_pairs"] != 8 or cfg["forbidden_splits"] != ["audit"] or cfg["audit_opened"]:
            raise ValueError("bounded causal protocol drift")
        feasibility = json.loads(paths["feasibility_metrics"].read_text(encoding="utf-8"))
        if feasibility["screen_decision"] != "PROCEED_ONLY_WITH_RAW_HOOK_CRITICAL_CAUSAL_SCREEN":
            raise ValueError("upstream screen does not authorize this causal screen")
        decisions = [json.loads(line) for line in paths["primary_decisions"].read_text(encoding="utf-8").splitlines() if line]
        query_rows = [json.loads(line) for line in paths["query_panel"].read_text(encoding="utf-8").splitlines() if line]
        query_lookup = {(row["seed"], row["atom"]): row for row in query_rows}
        selected_pairs = []
        for stratum in range(8):
            candidates = [row for row in decisions if row["energy_stratum"] == stratum and row["identification"] == "FOUND_SUBSPACE" and row["minimum_rank"] == 1]
            chosen = min(candidates, key=lambda row: (query_lookup[(row["source_seed"], row["source_atom"])]["selection_hash"], row["target_seed"]))
            selected_pairs.append(chosen)
        native_rows = [json.loads(line) for line in paths["native_calibration"].read_text(encoding="utf-8").splitlines() if line]
        native_lookup = {(row["source_seed"], row["target_seed"], row["source_atom"]): row for row in native_rows}
        projector_data = np.load(paths["projectors"], allow_pickle=False)

        asset_dir = Path(cfg["bulk_asset_dir"])
        matrices = {}
        for split, tokens in (("mean", 32768), ("discovery", 131072), ("calibration", cfg["calibration_tokens"])):
            matrices[split] = {seed: sparse_codes(asset_dir, split, seed, tokens, cfg["k"], cfg["num_latents"]) for seed in cfg["source_seeds"]}
        decoders = {seed: decoder(asset_dir, seed, cfg["num_latents"], cfg["hook_hidden_size"]).astype(np.float64, copy=False) for seed in cfg["source_seeds"]}
        raw_manifest = json.loads(paths["raw_hook_manifest"].read_text(encoding="utf-8"))
        raw_meta = {row["split"]: row for row in raw_manifest["splits"]}
        raw_calibration = np.memmap(raw_meta["calibration"]["path"], dtype="<f4", mode="r").reshape(raw_meta["calibration"]["shape"])

        global_mean_indices = deterministic_sample(32768, cfg["global_sample_tokens"], cfg["global_sample_salt"], "mean")
        global_discovery_indices = deterministic_sample(131072, cfg["global_sample_tokens"], cfg["global_sample_salt"], "discovery")
        mean_weights = np.full(global_mean_indices.size, 1.0 / global_mean_indices.size)
        discovery_weights = np.full(global_discovery_indices.size, 1.0 / global_discovery_indices.size)
        global_bases = {}
        for seed in cfg["source_seeds"]:
            global_mean = weighted_mean(reconstruct(matrices["mean"][seed], global_mean_indices, decoders[seed]), mean_weights)
            global_full_basis, _ = fit_weighted_pca(
                reconstruct(matrices["discovery"][seed], global_discovery_indices, decoders[seed]), discovery_weights,
                global_mean, max(cfg["candidate_ranks"]), random_seed=stable_seed(cfg["global_sample_salt"], seed),
                oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                relative_tolerance=cfg["eigenvalue_relative_tolerance"],
            )
            global_bases[seed] = global_full_basis[:, :cfg["rank"]]

        pair_payload = []
        for pair in selected_pairs:
            source_seed, target_seed, atom = pair["source_seed"], pair["target_seed"], pair["source_atom"]
            query_key = f"s{source_seed}_a{atom}"
            source_discovery_code = dense_code(matrices["discovery"][source_seed], atom)
            discovery_support = select_weighted_support(source_discovery_code, cfg["max_condition_tokens_per_split"], cfg["condition_weight_power"])
            source_mean_code = dense_code(matrices["mean"][source_seed], atom)
            mean_support = select_weighted_support(source_mean_code, cfg["max_condition_tokens_per_split"], cfg["condition_weight_power"])
            source_discovery = reconstruct(matrices["discovery"][source_seed], discovery_support.indices, decoders[source_seed])
            target_discovery = reconstruct(matrices["discovery"][target_seed], discovery_support.indices, decoders[target_seed])
            source_mean = np.asarray(projector_data[f"{query_key}_seed{source_seed}_mean"], dtype=np.float64)
            target_mean = np.asarray(projector_data[f"{query_key}_seed{target_seed}_mean"], dtype=np.float64)
            stitch_source_full, stitch_target_full, _ = fit_weighted_stitching(
                source_discovery, target_discovery, discovery_support.weights, source_mean, target_mean, max(cfg["candidate_ranks"]),
                random_seed=stable_seed(cfg["feasibility_run"], query_key, target_seed, "stitching"),
                oversample=cfg["randomized_svd_oversample"], power_iterations=cfg["randomized_svd_power_iterations"],
                relative_tolerance=cfg["eigenvalue_relative_tolerance"],
            )
            stitch_source = stitch_source_full[:, :cfg["rank"]]
            stitch_target = stitch_target_full[:, :cfg["rank"]]
            raw_basis = np.asarray(projector_data[f"{query_key}_raw_basis"], dtype=np.float64)[:, :cfg["rank"]]
            raw_mean = np.asarray(projector_data[f"{query_key}_raw_mean"], dtype=np.float64)
            main_source_basis = np.asarray(projector_data[f"{query_key}_seed{source_seed}_basis"], dtype=np.float64)[:, :cfg["rank"]]
            main_target_basis = np.asarray(projector_data[f"{query_key}_seed{target_seed}_basis"], dtype=np.float64)[:, :cfg["rank"]]
            random_source = random_orthonormal_basis(cfg["hook_hidden_size"], cfg["rank"], stable_seed(cfg["random_projector_salt"], query_key, source_seed, cfg["rank"]))
            random_target = random_orthonormal_basis(cfg["hook_hidden_size"], cfg["rank"], stable_seed(cfg["random_projector_salt"], query_key, target_seed, cfg["rank"]))
            native = native_lookup[(source_seed, target_seed, atom)]
            target_atom = int(native["method_metrics"]["best_bcc_native_size1"]["support"][0])
            source_atom_mean = float(mean_support.weights @ source_mean_code[mean_support.indices])
            target_atom_mean = float(mean_support.weights @ dense_code(matrices["mean"][target_seed], target_atom)[mean_support.indices])
            calibration_code = dense_code(matrices["calibration"][source_seed], atom)
            sequence_energy = np.sum(calibration_code.reshape(-1, cfg["context_length"]) ** 2, axis=1)
            sequence_order = np.lexsort((np.arange(sequence_energy.size), -sequence_energy))
            sequence_ids = [int(value) for value in sequence_order[:cfg["sequences_per_pair"]] if sequence_energy[value] > 0]
            if len(sequence_ids) != cfg["sequences_per_pair"]:
                raise ValueError(f"insufficient active calibration sequences for {query_key}")
            pair_payload.append({
                "pair": pair, "query_key": query_key, "source_mean": source_mean, "target_mean": target_mean,
                "raw_mean": raw_mean, "raw_basis": raw_basis,
                "main_source_basis": main_source_basis, "main_target_basis": main_target_basis,
                "global_source_basis": global_bases[source_seed], "global_target_basis": global_bases[target_seed],
                "stitch_source_basis": stitch_source, "stitch_target_basis": stitch_target,
                "random_source_basis": random_source, "random_target_basis": random_target,
                "source_atom_mean": source_atom_mean, "target_atom_mean": target_atom_mean, "target_atom": target_atom,
                "sequence_ids": sequence_ids, "sequence_energy": [float(sequence_energy[index]) for index in sequence_ids],
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
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_local_dir"], local_files_only=True, dtype=torch.float32,
            attn_implementation=cfg["attn_implementation"],
        ).eval().to(device)
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
                    replacement = primary - contribution
                    return replace_primary_hook_tensor(output, replacement, contract)
                return None

            def next_hook(_module, _inputs, output):
                captured["next"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

            hook_handle = hook_module.register_forward_hook(hook)
            next_handle = next_module.register_forward_hook(next_hook)
            try:
                with torch.no_grad():
                    result = model(batch, use_cache=False)
            finally:
                next_handle.remove(); hook_handle.remove()
            return captured["hook"], captured["next"], result.logits.detach()

        pair_accum = {(item["query_key"], item["pair"]["target_seed"], method): defaultdict(float) for item in pair_payload for method in cfg["methods"]}
        unit_rows, noop_errors, raw_replay_errors = [], [], []
        total_forwards = 0
        started_compute = time.perf_counter()
        for item in pair_payload:
            pair = item["pair"]
            source_seed, target_seed, atom = pair["source_seed"], pair["target_seed"], pair["source_atom"]
            for sequence_id in item["sequence_ids"]:
                begin, stop = sequence_id * cfg["context_length"], (sequence_id + 1) * cfg["context_length"]
                batch = torch.from_numpy(np.asarray(tokens[sequence_id:sequence_id + 1], dtype=np.int64)).to(device)
                hook, baseline_next, baseline_logits = forward(batch)
                _, noop_next, noop_logits = forward(batch, no_op=True)
                total_forwards += 2
                noop_next_error = float((noop_next - baseline_next).abs().max().item())
                noop_logit_error = float((noop_logits - baseline_logits).abs().max().item())
                noop_errors.append(max(noop_next_error, noop_logit_error))
                replay = np.asarray(raw_calibration[begin:stop], dtype=np.float32)
                raw_replay_errors.append(float(np.max(np.abs(replay - hook[0].detach().cpu().numpy()))))
                source_reconstruction = reconstruct(matrices["calibration"][source_seed], np.arange(begin, stop), decoders[source_seed])
                target_reconstruction = reconstruct(matrices["calibration"][target_seed], np.arange(begin, stop), decoders[target_seed])
                source_query_code = dense_code(matrices["calibration"][source_seed], atom)[begin:stop]
                target_single_code = dense_code(matrices["calibration"][target_seed], item["target_atom"])[begin:stop]
                method_contributions = {
                    "SAE_QUERY_CONDITIONAL_PCA": (
                        projected(source_reconstruction, item["source_mean"], item["main_source_basis"]),
                        projected(target_reconstruction, item["target_mean"], item["main_target_basis"]),
                    ),
                    "RAW_HOOK_QUERY_CONDITIONAL_PCA": (
                        projected(replay, item["raw_mean"], item["raw_basis"]),
                        projected(replay, item["raw_mean"], item["raw_basis"]),
                    ),
                    "GLOBAL_SAE_PCA": (
                        projected(source_reconstruction, item["source_mean"], item["global_source_basis"]),
                        projected(target_reconstruction, item["target_mean"], item["global_target_basis"]),
                    ),
                    "RELAXED_PAIRED_STITCHING": (
                        projected(source_reconstruction, item["source_mean"], item["stitch_source_basis"]),
                        projected(target_reconstruction, item["target_mean"], item["stitch_target_basis"]),
                    ),
                    "MATCHED_RANK_RANDOM": (
                        projected(source_reconstruction, item["source_mean"], item["random_source_basis"]),
                        projected(target_reconstruction, item["target_mean"], item["random_target_basis"]),
                    ),
                    "BEST_FUNCTIONAL_SINGLE_NATIVE": (
                        (source_query_code - item["source_atom_mean"])[:, None] * decoders[source_seed][atom][None, :],
                        (target_single_code - item["target_atom_mean"])[:, None] * decoders[target_seed][item["target_atom"]][None, :],
                    ),
                }
                reference_norm = float(np.linalg.norm(method_contributions["SAE_QUERY_CONDITIONAL_PCA"][0]))
                active_mask = torch.from_numpy(source_query_code != 0).to(device)
                off_mask = ~active_mask
                for method, (source_value, target_value) in method_contributions.items():
                    source_matched, source_scale = rescale_to_norm(source_value, reference_norm)
                    target_matched, target_scale = rescale_to_norm(target_value, reference_norm)
                    source_tensor = torch.from_numpy(source_matched.astype(np.float32))[None].to(device)
                    target_tensor = torch.from_numpy(target_matched.astype(np.float32))[None].to(device)
                    _, source_next, source_logits = forward(batch, contribution=source_tensor)
                    _, target_next, target_logits = forward(batch, contribution=target_tensor)
                    total_forwards += 2
                    key = (item["query_key"], target_seed, method)
                    acc = pair_accum[key]
                    for endpoint, base, source_output, target_output in (
                        ("next_state", baseline_next, source_next, target_next),
                        ("next_logits", baseline_logits, source_logits, target_logits),
                    ):
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
                        "source_seed": source_seed, "target_seed": target_seed, "source_atom": atom,
                        "energy_stratum": pair["energy_stratum"], "sequence_id": sequence_id, "method": method,
                        "rank": cfg["rank"], "active_query_tokens": int(np.sum(source_query_code != 0)),
                        "reference_hook_norm": reference_norm, "source_native_hook_norm": float(np.linalg.norm(source_value)),
                        "target_native_hook_norm": float(np.linalg.norm(target_value)),
                        "source_energy_scale": source_scale, "target_energy_scale": target_scale,
                    })

        pair_rows = []
        for item in pair_payload:
            pair = item["pair"]
            for method in cfg["methods"]:
                acc = pair_accum[(item["query_key"], pair["target_seed"], method)]
                endpoints = {name: endpoint_metrics(acc, name) for name in ("next_state", "next_logits")}
                pair_rows.append({
                    "source_seed": pair["source_seed"], "target_seed": pair["target_seed"], "source_atom": pair["source_atom"],
                    "energy_stratum": pair["energy_stratum"], "method": method, "rank": cfg["rank"],
                    "sequence_ids": item["sequence_ids"], "endpoints": endpoints,
                    "hook_source_energy": acc["hook_source_energy"], "hook_target_energy": acc["hook_target_energy"],
                })
        noop_pass = max(noop_errors) <= cfg["causal_qualification"]["maximum_noop_absolute_error"]
        primary_rows = [row for row in pair_rows if row["method"] == "SAE_QUERY_CONDITIONAL_PCA"]
        next_state_floor = sum(row["endpoints"]["next_state"]["source_effect_rms"] >= cfg["causal_qualification"]["minimum_source_effect_rms"] for row in primary_rows) / len(primary_rows)
        next_logit_floor = sum(row["endpoints"]["next_logits"]["source_effect_rms"] >= cfg["causal_qualification"]["minimum_source_effect_rms"] for row in primary_rows) / len(primary_rows)
        if noop_pass and next_state_floor >= 0.75:
            frozen_endpoint = "next_state"
        elif noop_pass and next_logit_floor >= 0.75:
            frozen_endpoint = "next_logits"
        else:
            frozen_endpoint = "NO_ENDPOINT"
        for row in pair_rows:
            if frozen_endpoint == "NO_ENDPOINT":
                row["qualifies_frozen_endpoint"], row["qualification_failures"] = False, ["no_endpoint"]
            else:
                row["qualifies_frozen_endpoint"], row["qualification_failures"] = qualifies(row["endpoints"][frozen_endpoint], cfg)
        coverages = {method: sum(row["qualifies_frozen_endpoint"] for row in pair_rows if row["method"] == method) / cfg["selected_pairs"] for method in cfg["methods"]}
        off_target_medians = {
            method: float(np.median([row["endpoints"][frozen_endpoint]["source_off_target_fraction"] for row in pair_rows if row["method"] == method])) if frozen_endpoint != "NO_ENDPOINT" else None
            for method in cfg["methods"]
        }
        progression = cfg["sae_specific_progression"]
        primary_coverage = coverages["SAE_QUERY_CONDITIONAL_PCA"]
        global_advantage = primary_coverage - coverages["GLOBAL_SAE_PCA"]
        specificity_advantage = min(
            off_target_medians["RAW_HOOK_QUERY_CONDITIONAL_PCA"] - off_target_medians["SAE_QUERY_CONDITIONAL_PCA"],
            off_target_medians["GLOBAL_SAE_PCA"] - off_target_medians["SAE_QUERY_CONDITIONAL_PCA"],
        ) if frozen_endpoint != "NO_ENDPOINT" else float("-inf")
        if frozen_endpoint == "NO_ENDPOINT" or primary_coverage < progression["minimum_primary_pair_coverage"]:
            screen_decision = "STOP_CAUSAL_EFFECT_OR_COVERAGE_FLOOR"
        elif global_advantage < progression["minimum_coverage_advantage_over_global"] or specificity_advantage < progression["minimum_median_off_target_fraction_advantage_over_raw_or_global"]:
            screen_decision = "STOP_SAE_SPECIFIC_SCT_NOT_IDENTIFIED"
        else:
            screen_decision = "PROCEED_TO_FULL_CALIBRATION_FREEZE_BEFORE_AUDIT"

        unit_path = run_dir / "intervention_units.jsonl"
        with unit_path.open("w", encoding="utf-8") as stream:
            for row in unit_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        pair_path = run_dir / "causal_pair_metrics.jsonl"
        with pair_path.open("w", encoding="utf-8") as stream:
            for row in pair_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        selection_path = run_dir / "selected_pair_sequences.json"
        write_json(selection_path, {"pairs": [{
            "source_seed": item["pair"]["source_seed"], "target_seed": item["pair"]["target_seed"],
            "source_atom": item["pair"]["source_atom"], "energy_stratum": item["pair"]["energy_stratum"],
            "sequence_ids": item["sequence_ids"], "sequence_energy": item["sequence_energy"],
        } for item in pair_payload]})
        checks = {
            "frozen_inputs_bound": all(bound.values()),
            "eight_strata_once": sorted(item["pair"]["energy_stratum"] for item in pair_payload) == list(range(8)),
            "rank1_found_pairs_only": all(item["pair"]["identification"] == "FOUND_SUBSPACE" and item["pair"]["minimum_rank"] == 1 for item in pair_payload),
            "two_sequences_per_pair": all(len(item["sequence_ids"]) == 2 for item in pair_payload),
            "complete_method_grid": len(pair_rows) == cfg["selected_pairs"] * len(cfg["methods"]),
            "matched_rank": all(row["rank"] == 1 for row in pair_rows),
            "matched_energy_finite": all(np.isfinite(row["reference_hook_norm"]) and row["reference_hook_norm"] > 0 for row in unit_rows),
            "noop_control": noop_pass,
            "raw_hook_replay": max(raw_replay_errors) <= 1e-5,
            "endpoint_frozen": frozen_endpoint in {"next_state", "next_logits"},
            "finite_pair_metrics": all(np.isfinite(value) for row in pair_rows for endpoint in row["endpoints"].values() for value in endpoint.values() if value is not None),
            "audit_not_opened": not cfg["audit_opened"] and cfg["forbidden_splits"] == ["audit"],
        }
        record = {
            "checks": checks, "selected_pairs": len(pair_payload), "intervention_units": len(unit_rows),
            "total_model_forwards": total_forwards, "noop_max_absolute_error": max(noop_errors),
            "raw_hook_replay_max_absolute_error": max(raw_replay_errors), "frozen_primary_endpoint": frozen_endpoint,
            "next_state_primary_effect_floor_coverage": next_state_floor, "next_logits_primary_effect_floor_coverage": next_logit_floor,
            "method_pair_coverage": coverages, "method_median_source_off_target_fraction": off_target_medians,
            "primary_minus_global_coverage": global_advantage, "primary_min_specificity_advantage": specificity_advantage,
            "screen_decision": screen_decision, "selection_sha256": sha256(selection_path),
            "units_sha256": sha256(unit_path), "pair_metrics_sha256": sha256(pair_path),
            "wall_seconds": time.perf_counter() - started_compute,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version,
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "platform": platform.platform(),
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "error": error})
    raw_metrics = run_dir / "metrics.raw.jsonl"
    raw_metrics.write_text((json.dumps(record, sort_keys=True) + "\n") if record else "", encoding="utf-8")
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "error": error, "checks_passed": sum(record["checks"].values()) if record else 0,
        "checks_total": len(record["checks"]) if record else 0, "metrics_raw_sha256": sha256(raw_metrics),
        "generator_script_path": "scripts/run_r011s1_causal_calibration_screen.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "error": error}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
