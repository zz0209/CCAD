"""Build the R008 paired corpus with a locked document-level four-way split."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.artifacts import sha256, validate_run_directory  # noqa: E402
from ccad.data_manifest import FINEWEB_FIELDS, canonical_sha256, paired_document_split  # noqa: E402
from ccad.http_range import RequestsRangeReader  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def score(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()


def aggregate(rows: list[dict]) -> str:
    payload = "".join(f"{row['path']}:{row['sha256']}\n" for row in sorted(rows, key=lambda item: item["path"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_uint16(path: Path, tokens: list[int]) -> None:
    values = array.array("H", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def entry(path: Path, source: str, boundary: str, role: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
            "source": source, "license_or_access_boundary": boundary, "role": role}


def pack(rows: list[dict], tokenizer, target_sequences: int, context_length: int, max_tokens: int) -> tuple[list[int], list[dict], list[dict]]:
    target = target_sequences * context_length
    eos = int(tokenizer.eos_token_id)
    tokens = [eos]
    documents, spans = [], []
    for row in rows:
        encoded = [int(value) for value in tokenizer(row["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"][:max_tokens]]
        if not encoded:
            continue
        start = len(tokens)
        tokens.extend(encoded)
        end = len(tokens)
        tokens.append(eos)
        documents.append({key: value for key, value in row.items() if key != "text"} | {
            "included_token_count": len(encoded), "included_token_sha256": canonical_sha256(encoded)})
        spans.append({"document_id": row["document_id"], "start": start, "end": end})
        if len(tokens) >= target:
            break
    if len(tokens) < target:
        raise ValueError(f"insufficient tokens: {len(tokens)} < {target}")
    tokens = tokens[:target]
    sequences = []
    for index in range(target_sequences):
        start, end = index * context_length, (index + 1) * context_length
        sequence = tokens[start:end]
        sequences.append({"sequence_index": index, "document_ids": [span["document_id"] for span in spans if span["start"] < end and span["end"] > start], "token_sha256": canonical_sha256(sequence)})
    return tokens, documents, sequences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.resolved.json", cfg)
    code_paths = [Path(__file__).resolve(), ROOT / "src/ccad/data_manifest.py", ROOT / "src/ccad/http_range.py"]
    code_rows = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in code_paths]
    code_hash = aggregate(code_rows)
    write_json(run_dir / "code_hashes.json", {"files": code_rows, "aggregate_sha256": code_hash})
    excluded_path = ROOT / cfg["excluded_document_path"]
    catalog_path = ROOT / cfg["source_catalog_path"]
    write_json(run_dir / "inputs.json", {"inputs": [
        entry(config_path, "CCAD frozen config", "internal", "protocol"),
        entry(excluded_path, "R006 SAE corpus", "internal", "exclusion_ledger"),
        entry(catalog_path, "FineWeb source catalog", cfg["dataset_license"], "source_catalog"),
    ]})
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"], "purpose": cfg["purpose"],
        "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"], "started_utc": started, "project_root": str(ROOT),
        "config_hash": sha256(run_dir / "config.resolved.json"), "code_snapshot_hash": code_hash, "audit_opened": False,
        "candidate_family_frozen": False, "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": "document", "device": "cpu_network_range_and_tokenizer",
        "seeds": {"selection_salt": cfg["selection_salt"], "split_salt": cfg["split_salt"]}, "resource_lease": "disk-e-io via SAE Lab resource_manager.run",
        "resource_lease_reason": "revision-pinned range reads and paired token asset writes", "git_head_at_run": git_head, "git_status_porcelain": git_status,
    })
    write_json(run_dir / "status.json", {"status": "RUNNING", "updated_utc": started})
    checks, details, error = {}, {}, None
    try:
        sys.path[:0] = [cfg["data_overlay_dir"], cfg["dependency_overlay_dir"]]
        import pyarrow.parquet as pq
        import requests
        import transformers
        from transformers import AutoTokenizer

        excluded = [json.loads(line) for line in excluded_path.read_text(encoding="utf-8").splitlines() if line]
        excluded_ids = {row["document_id"] for row in excluded}
        excluded_text = {row["text_sha256"] for row in excluded}
        old_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        old_groups = {row["path"]: row["selected_row_group"] for row in old_catalog["selected_sources"]}
        files = old_catalog["catalog_files"]
        checks["catalog_commit_bound"] = old_catalog["dataset_commit"] == cfg["dataset_commit"]
        checks["expected_shard_count"] = len(files) == cfg["expected_shard_count"]
        selected = sorted(files, key=lambda item: score(cfg["selection_salt"] + "-shard", item["path"]))[:cfg["shards_to_sample"]]
        session, sampled, sources = requests.Session(), [], []
        total_requests = total_bytes = 0
        for source in selected:
            path = source["path"]
            quoted = "/".join(requests.utils.quote(piece, safe="") for piece in path.split("/"))
            url = f"https://huggingface.co/datasets/{cfg['dataset_id']}/resolve/{cfg['dataset_commit']}/{quoted}"
            response = session.get(url, allow_redirects=False, stream=True, timeout=30)
            headers = {key.lower(): value for key, value in response.headers.items()}
            final_url = headers.get("location")
            header_ok = headers.get("x-repo-commit") == cfg["dataset_commit"] and headers.get("x-linked-etag", "").strip('"') == source["lfs_sha256"] and bool(final_url)
            response.close()
            if not header_ok:
                raise ValueError(f"source binding failed: {path}")
            reader = RequestsRangeReader(session, final_url, int(source["size"]), cfg["range_block_bytes"])
            with reader:
                parquet = pq.ParquetFile(reader)
                group = int(score(cfg["selection_salt"] + "-row-group", path), 16) % parquet.metadata.num_row_groups
                if old_groups.get(path) == group:
                    group = (group + 1) % parquet.metadata.num_row_groups
                offset = sum(parquet.metadata.row_group(index).num_rows for index in range(group))
                rows = parquet.read_row_group(group, columns=list(FINEWEB_FIELDS)).to_pylist()
                for local_index, row in enumerate(rows):
                    document_id, text = row["id"], row["text"]
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    if document_id in excluded_ids or text_hash in excluded_text:
                        continue
                    split = paired_document_split(cfg["dataset_commit"], document_id, salt=cfg["split_salt"])
                    sampled.append({"dataset_id": cfg["dataset_id"], "dataset_commit": cfg["dataset_commit"], "source_parquet_path": path,
                                    "source_row_index": offset + local_index, "document_id": document_id, "text_sha256": text_hash, "split": split, "text": text})
                sources.append({"path": path, "lfs_sha256": source["lfs_sha256"], "selected_row_group": group, "rows": len(rows), "range_requests": reader.range_requests, "range_bytes": reader.bytes_received})
                total_requests += reader.range_requests
                total_bytes += reader.bytes_received
        tokenizer_dir = Path(cfg["tokenizer_local_dir"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        checks["tokenizer_local"] = Path(tokenizer.name_or_path).resolve() == tokenizer_dir.resolve()
        split_outputs, all_documents, all_sequences, split_id_sets = {}, [], [], {}
        for split, target_sequences in cfg["target_sequences"].items():
            rows = sorted((row for row in sampled if row["split"] == split), key=lambda row: score(cfg["selection_salt"] + "-order", row["document_id"]))
            tokens, documents, sequences = pack(rows, tokenizer, target_sequences, cfg["context_length"], cfg["max_tokens_per_document"])
            token_path = artifacts / f"{split}.uint16.bin"
            write_uint16(token_path, tokens)
            for sequence in sequences:
                sequence["split"] = split
            all_documents.extend(documents)
            all_sequences.extend(sequences)
            split_id_sets[split] = {row["document_id"] for row in documents}
            checks[f"{split}_sequence_count"] = len(sequences) == target_sequences
            checks[f"{split}_minimum_documents"] = len(documents) >= cfg["minimum_documents"][split]
            split_outputs[split] = {"path": token_path.relative_to(run_dir).as_posix(), "sha256": sha256(token_path), "tokens": len(tokens), "sequences": len(sequences), "documents": len(documents)}
        checks["all_splits_disjoint"] = sum(len(ids) for ids in split_id_sets.values()) == len(set().union(*split_id_sets.values()))
        checks["sae_corpus_disjoint"] = not (set().union(*split_id_sets.values()) & excluded_ids) and not ({row["text_sha256"] for row in all_documents} & excluded_text)
        with (artifacts / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(all_documents, key=lambda item: (item["split"], item["document_id"])):
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        write_json(artifacts / "sequence_records.json", {"sequences": all_sequences})
        write_json(artifacts / "source_records.json", {"sources": sources})
        write_json(artifacts / "token_manifest.json", {"dataset": {"id": cfg["dataset_id"], "commit": cfg["dataset_commit"], "license": cfg["dataset_license"]},
                   "tokenizer": {"id": cfg["tokenizer_id"], "revision": cfg["tokenizer_revision"]}, "split_rule": "document SHA256 buckets: 10% mean, 40% discovery, 20% calibration, 30% audit", "outputs": split_outputs})
        details = {"sampled_documents": len(sampled), "used_documents": len(all_documents), "range_requests": total_requests, "range_bytes": total_bytes, "outputs": split_outputs}
        checks["audit_not_opened"] = True
        import pyarrow
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "pyarrow": pyarrow.__version__, "transformers": transformers.__version__, "platform": platform.platform(), "data_overlay_dir": cfg["data_overlay_dir"], "dependency_overlay_dir": cfg["dependency_overlay_dir"]})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "error": error, "data_overlay_dir": cfg.get("data_overlay_dir"), "dependency_overlay_dir": cfg.get("dependency_overlay_dir")})
    raw = run_dir / "metrics.raw.jsonl"
    raw.write_text("".join(json.dumps({"check": name, "passed": passed}, sort_keys=True) + "\n" for name, passed in sorted(checks.items())), encoding="utf-8")
    status = "PASS" if checks and all(checks.values()) and error is None else "FAIL"
    write_json(run_dir / "metrics.summary.json", {"status": status, "error": error, "checks_passed": sum(checks.values()), "checks_total": len(checks), "details": details,
               "metrics_raw_sha256": sha256(raw), "generator_script_path": "scripts/run_r008a_paired_corpus.py", "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"]})
    write_json(run_dir / "status.json", {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "error": error})
    (run_dir / "stdout.log").write_text(json.dumps({"run_id": cfg["run_id"], "status": status}) + "\n", encoding="utf-8")
    if not (run_dir / "stderr.log").exists():
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "checks": checks}))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
