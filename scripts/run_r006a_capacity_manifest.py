"""Build a small cross-shard, revision-pinned FineWeb token manifest for R006 capacity runs."""

from __future__ import annotations

import argparse
import array
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
from ccad.data_manifest import (  # noqa: E402
    FINEWEB_FIELDS,
    canonical_sha256,
    fineweb_document_record,
    validate_document_records,
)
from ccad.http_range import RequestsRangeReader  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def aggregate(entries: list[dict[str, object]]) -> str:
    payload = "".join(f"{item['path']}:{item['sha256']}\n" for item in sorted(entries, key=lambda x: str(x["path"])))
    return hashlib.sha256(payload.encode()).hexdigest()


def local_entry(path: Path, source: str, boundary: str, role: str) -> dict[str, object]:
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
        "source": source, "license_or_access_boundary": boundary, "role": role,
    }


def score(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()


def pinned_url(cfg: dict[str, object], source_path: str, requests) -> str:
    quoted = "/".join(requests.utils.quote(piece, safe="") for piece in source_path.split("/"))
    return f"https://huggingface.co/datasets/{cfg['dataset_id']}/resolve/{cfg['dataset_commit']}/{quoted}"


def write_uint16(path: Path, tokens: list[int]) -> None:
    values = array.array("H", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def pack_split(rows: list[dict[str, object]], tokenizer, cfg: dict[str, object], split: str) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]]]:
    target = int(cfg[f"{split}_sequences"]) * int(cfg["context_length"])
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")
    tokens = [int(eos)]
    used_documents = []
    spans = []
    ordered = sorted(rows, key=lambda item: score(str(cfg["selection_salt"]) + "-order", str(item["document_id"])))
    for item in ordered:
        encoded = tokenizer(str(item["text"]), add_special_tokens=False, return_attention_mask=False)["input_ids"]
        encoded = [int(value) for value in encoded[: int(cfg["max_tokens_per_document"])]]
        if not encoded:
            continue
        start = len(tokens)
        tokens.extend(encoded)
        end = len(tokens)
        tokens.append(int(eos))
        used_documents.append({key: value for key, value in item.items() if key != "text"} | {
            "original_token_count": len(tokenizer(str(item["text"]), add_special_tokens=False, return_attention_mask=False)["input_ids"]),
            "included_token_count": len(encoded),
            "included_token_sha256": canonical_sha256(encoded),
        })
        spans.append({"document_id": item["document_id"], "start": start, "end": end})
        if len(tokens) >= target:
            break
    if len(tokens) < target:
        raise ValueError(f"not enough {split} tokens: {len(tokens)} < {target}")
    tokens = tokens[:target]
    sequence_records = []
    width = int(cfg["context_length"])
    for sequence_index in range(target // width):
        start, end = sequence_index * width, (sequence_index + 1) * width
        document_ids = [span["document_id"] for span in spans if span["start"] < end and span["end"] > start]
        sequence = tokens[start:end]
        sequence_records.append({
            "split": split, "sequence_index": sequence_index, "document_ids": document_ids,
            "token_sha256": canonical_sha256(sequence),
        })
    return tokens, used_documents, sequence_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    stderr_lines: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src" / "ccad" / "data_manifest.py", ROOT / "src" / "ccad" / "http_range.py"]
    code_entries = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_entries)
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"], "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"], "threshold_source_split": cfg["threshold_source_split"],
        "statistics_unit": cfg["statistics_unit"], "device": "cpu_network_range_and_tokenizer",
        "seeds": [], "resource_lease": None,
        "resource_lease_reason": "Bounded metadata plus five row-group range reads and sub-MB token outputs",
    })
    write_json(run_dir / "environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform()})
    catalog_path = artifacts / "source_catalog.json"
    documents_path = artifacts / "sampled_documents.jsonl"
    input_entries = [local_entry(config_path, "CCAD prewritten config", "internal artifact", "capacity_manifest_protocol")]
    try:
        import pyarrow
        import pyarrow.parquet as pq
        import requests
        import transformers
        from transformers import AutoTokenizer

        write_json(run_dir / "environment.json", {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "pyarrow": pyarrow.__version__, "requests": requests.__version__, "transformers": transformers.__version__,
        })
        tree_url = (
            f"https://huggingface.co/api/datasets/{cfg['dataset_id']}/tree/{cfg['dataset_revision_name']}/"
            f"{cfg['source_directory']}?recursive=true&expand=false"
        )
        response = requests.get(tree_url, timeout=30)
        response.raise_for_status()
        files = [entry for entry in response.json() if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".parquet")]
        checks["expected_shard_count"] = len(files) == cfg["expected_shard_count"]
        checks["all_catalog_files_have_lfs"] = all(entry.get("lfs", {}).get("oid") and entry.get("lfs", {}).get("size") == entry.get("size") for entry in files)
        selected = sorted(files, key=lambda item: score(str(cfg["selection_salt"]) + "-shard", str(item["path"])))[: int(cfg["shards_to_sample"])]
        checks["selected_shard_count"] = len(selected) == cfg["shards_to_sample"]
        sampled_rows = []
        source_records = []
        total_range_requests = 0
        total_range_bytes = 0
        session = requests.Session()
        for entry in selected:
            source_path = str(entry["path"])
            url = pinned_url(cfg, source_path, requests)
            head = session.get(url, allow_redirects=False, stream=True, timeout=30)
            headers = {key.lower(): value for key, value in head.headers.items()}
            source_checks = {
                "repo_commit": headers.get("x-repo-commit") == cfg["dataset_commit"],
                "lfs_sha256": headers.get("x-linked-etag", "").strip('"') == entry["lfs"]["oid"],
                "linked_size": int(headers.get("x-linked-size", -1)) == entry["size"],
                "range": headers.get("accept-ranges", "").lower() == "bytes",
                "redirect": head.status_code in {302, 303, 307, 308} and bool(headers.get("location")),
            }
            final_url = headers.get("location")
            head.close()
            if not all(source_checks.values()) or not final_url:
                raise ValueError(f"source header check failed for {source_path}: {source_checks}")
            reader = RequestsRangeReader(session, final_url, int(entry["size"]), int(cfg["range_block_bytes"]))
            with reader:
                parquet = pq.ParquetFile(reader)
                metadata = parquet.metadata
                row_group_index = int(score(str(cfg["selection_salt"]) + "-row-group", source_path), 16) % metadata.num_row_groups
                row_offset = sum(metadata.row_group(index).num_rows for index in range(row_group_index))
                table = parquet.read_row_group(row_group_index, columns=list(FINEWEB_FIELDS))
                rows = table.to_pylist()
                for local_index, row in enumerate(rows):
                    record = fineweb_document_record(
                        row, row_index=row_offset + local_index, dataset_id=str(cfg["dataset_id"]),
                        dataset_config=str(cfg["dataset_config"]), dataset_commit=str(cfg["dataset_commit"]),
                        source_path=source_path, split_salt=str(cfg["split_salt"]),
                        validation_basis_points=int(cfg["validation_basis_points"]),
                    )
                    sampled_rows.append({**record, "text": row["text"]})
                source_records.append({
                    "path": source_path, "size": entry["size"], "lfs_sha256": entry["lfs"]["oid"],
                    "xet_hash": entry.get("xetHash"), "header_checks": source_checks,
                    "num_rows": metadata.num_rows, "num_row_groups": metadata.num_row_groups,
                    "selected_row_group": row_group_index, "selected_row_offset": row_offset,
                    "selected_row_count": len(rows), "range_requests": reader.range_requests,
                    "range_bytes_received": reader.bytes_received,
                })
                total_range_requests += reader.range_requests
                total_range_bytes += reader.bytes_received
        excluded_ids: set[str] = set()
        excluded_text_hashes: set[str] = set()
        excluded_paths = [Path(str(value)) for value in cfg.get("excluded_document_paths", [])]
        for excluded_path in excluded_paths:
            resolved_excluded = excluded_path if excluded_path.is_absolute() else ROOT / excluded_path
            input_entries.append(local_entry(
                resolved_excluded,
                "CCAD frozen document exclusion ledger",
                "internal artifact",
                "prevent SAE-training overlap with prior training and paired corpora",
            ))
            for line in resolved_excluded.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                if row.get("document_id") is not None:
                    excluded_ids.add(str(row["document_id"]))
                if row.get("text_sha256") is not None:
                    excluded_text_hashes.add(str(row["text_sha256"]))
        pre_exclusion_count = len(sampled_rows)
        sampled_rows = [
            row for row in sampled_rows
            if str(row["document_id"]) not in excluded_ids
            and str(row["text_sha256"]) not in excluded_text_hashes
        ]
        report = validate_document_records(sampled_rows)
        checks["sample_document_ids_unique"] = bool(report["unique_document_ids"])
        checks["sample_source_rows_unique"] = bool(report["unique_source_rows"])
        checks["sample_text_hashes_unique"] = bool(report["unique_text_hashes"])
        checks["sample_has_both_splits"] = report["train_documents"] > 0 and report["validation_documents"] > 0
        checks["excluded_document_ids_absent"] = not any(str(row["document_id"]) in excluded_ids for row in sampled_rows)
        checks["excluded_text_hashes_absent"] = not any(str(row["text_sha256"]) in excluded_text_hashes for row in sampled_rows)
        tokenizer_dir = Path(str(cfg["tokenizer_local_dir"]))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        tokenizer_files = [path for path in tokenizer_dir.iterdir() if path.is_file() and "tokenizer" in path.name or path.name == "special_tokens_map.json"]
        tokenizer_hashes = {path.name: sha256(path) for path in sorted(tokenizer_files)}
        checks["tokenizer_name_or_path_local"] = Path(str(tokenizer.name_or_path)).resolve() == tokenizer_dir.resolve()
        split_rows = {
            split: [row for row in sampled_rows if row["split"] == split]
            for split in ("train", "validation")
        }
        all_used_documents = []
        all_sequences = []
        token_outputs = {}
        for split in ("train", "validation"):
            tokens, used_documents, sequences = pack_split(split_rows[split], tokenizer, cfg, split)
            checks[f"{split}_token_count_exact"] = len(tokens) == int(cfg[f"{split}_sequences"]) * int(cfg["context_length"])
            checks[f"{split}_token_ids_fit_uint16"] = min(tokens) >= 0 and max(tokens) <= 65535
            token_path = artifacts / f"{split}.uint16.bin"
            write_uint16(token_path, tokens)
            checks[f"{split}_binary_size_exact"] = token_path.stat().st_size == 2 * len(tokens)
            token_outputs[split] = {
                "path": str(token_path.relative_to(run_dir)), "sha256": sha256(token_path),
                "tokens": len(tokens), "sequences": len(sequences), "documents": len(used_documents),
            }
            all_used_documents.extend(used_documents)
            all_sequences.extend(sequences)
        used_ids = {str(row["document_id"]) for row in all_used_documents}
        included_rows = [row for row in sampled_rows if str(row["document_id"]) in used_ids]
        with documents_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(included_rows, key=lambda item: (str(item["split"]), str(item["document_id"]))):
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        write_json(artifacts / "document_token_records.json", {"documents": all_used_documents})
        write_json(artifacts / "sequence_records.json", {"sequences": all_sequences})
        write_json(artifacts / "token_manifest.json", {
            "dataset": {"id": cfg["dataset_id"], "config": cfg["dataset_config"], "commit": cfg["dataset_commit"], "license": cfg["dataset_license"]},
            "tokenizer": {"id": cfg["tokenizer_id"], "revision": cfg["tokenizer_revision"], "local_dir": str(tokenizer_dir), "files": tokenizer_hashes, "eos_token_id": tokenizer.eos_token_id, "vocab_size": len(tokenizer)},
            "packing": {"context_length": cfg["context_length"], "max_tokens_per_document": cfg["max_tokens_per_document"], "leading_eos": True, "document_separator": "eos", "trailing_partial_sequence": "dropped"},
            "outputs": token_outputs,
        })
        write_json(catalog_path, {
            "tree_url": tree_url, "dataset_commit": cfg["dataset_commit"],
            "catalog_files": [{"path": item["path"], "size": item["size"], "lfs_sha256": item["lfs"]["oid"], "xet_hash": item.get("xetHash")} for item in files],
            "selection_rule": "lowest SHA256(selection_salt + NUL + path), then SHA256-selected row group modulo count",
            "selected_sources": source_records,
        })
        details = {
            "catalog_shards": len(files), "selected_shards": len(selected),
            "sampled_documents_before_exclusion": pre_exclusion_count,
            "sampled_documents": len(sampled_rows), "excluded_document_ids": len(excluded_ids),
            "excluded_text_hashes": len(excluded_text_hashes),
            "sample_report": report, "used_documents": len(all_used_documents),
            "range_requests": total_range_requests, "range_bytes_received": total_range_bytes,
            "token_outputs": token_outputs,
        }
    except Exception:
        stderr_lines.append(traceback.format_exc())
        checks["capacity_manifest_completed"] = False
        if not catalog_path.exists():
            write_json(catalog_path, {"error": "capacity manifest failed before catalog completion", "config_source": str(config_path)})
    input_entries.append(local_entry(catalog_path, "Hugging Face revision-pinned source catalog", cfg["dataset_license"], "source_catalog"))
    if documents_path.exists():
        input_entries.append(local_entry(documents_path, "FineWeb revision-pinned selected documents", cfg["dataset_license"], "selected_source_documents"))
    write_json(run_dir / "inputs.json", {"inputs": input_entries})
    with (run_dir / "metrics.raw.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for name, passed in sorted(checks.items()):
            stream.write(json.dumps({"check": name, "passed": passed}, sort_keys=True) + "\n")
    status = "PASS" if checks and all(checks.values()) and not stderr_lines else "FAIL"
    write_json(run_dir / "metrics.summary.json", {
        "status": status, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed), "details": details,
        "metrics_raw_sha256": sha256(run_dir / "metrics.raw.jsonl"),
        "generator_script_path": "scripts/run_r006a_capacity_manifest.py", "generator_script_sha256": sha256(Path(__file__).resolve()),
        "scope_limit": cfg["scope_limit"],
    })
    write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat()})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status, "checks": f"{sum(checks.values())}/{len(checks)}"}, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("\n".join(stderr_lines), encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "checks": checks}, sort_keys=True))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
