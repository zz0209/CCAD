"""R006-A revision-pinned FineWeb manifest/access-path probe."""

from __future__ import annotations

import argparse
import hashlib
import io
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
from ccad.data_manifest import (  # noqa: E402
    FINEWEB_FIELDS,
    canonical_sha256,
    fineweb_document_record,
    validate_document_records,
)


class RequestsRangeReader(io.RawIOBase):
    """Small seekable HTTP reader with an aligned in-memory range cache."""

    def __init__(self, session, url: str, size: int, block_size: int):
        super().__init__()
        self.session = session
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        self.range_requests = 0
        self.bytes_received = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = min(target, self.size)
        return self.position

    def _fetch(self, position: int) -> None:
        start = (position // self.block_size) * self.block_size
        end = min(start + self.block_size, self.size) - 1
        response = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60
        )
        if response.status_code != 206:
            raise OSError(f"range request returned {response.status_code}, expected 206")
        expected_range = f"bytes {start}-{end}/{self.size}"
        if response.headers.get("Content-Range") != expected_range:
            raise OSError(
                f"unexpected Content-Range {response.headers.get('Content-Range')!r}; expected {expected_range!r}"
            )
        payload = response.content
        if len(payload) != end - start + 1:
            raise OSError("range response length mismatch")
        self.cache_start = start
        self.cache = payload
        self.range_requests += 1
        self.bytes_received += len(payload)

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(self.position)
                cache_end = self.cache_start + len(self.cache)
            take = min(remaining, cache_end - self.position)
            offset = self.position - self.cache_start
            chunks.append(self.cache[offset:offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


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


def normalized_row(row: dict[str, object]) -> dict[str, object]:
    return {field: row[field] for field in FINEWEB_FIELDS}


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
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    stdout_lines = [f"START {cfg['run_id']}"]
    stderr_lines: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    write_json(run_dir / "config.resolved.json", cfg)
    code_entries = []
    for path in (Path(__file__).resolve(), ROOT / "src" / "ccad" / "data_manifest.py"):
        code_entries.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    code_hash = aggregate(code_entries)
    write_json(run_dir / "code_hashes.json", {"files": code_entries, "aggregate_sha256": code_hash})
    write_json(run_dir / "environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "pyarrow": None, "fsspec": None, "requests": None,
    })
    write_json(run_dir / "manifest.json", {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "run_parent": cfg["run_parent"],
        "purpose": cfg["purpose"], "milestone": cfg["milestone"], "evidence_level": cfg["evidence_level"],
        "started_utc": started, "project_root": str(ROOT), "config_hash": sha256(run_dir / "config.resolved.json"),
        "code_snapshot_hash": code_hash, "audit_opened": cfg["audit_opened"],
        "candidate_family_frozen": cfg["candidate_family_frozen"],
        "mean_constants_source_split": cfg["mean_constants_source_split"],
        "threshold_source_split": cfg["threshold_source_split"], "statistics_unit": cfg["statistics_unit"],
        "device": "cpu_network_range_probe", "seeds": [], "resource_lease": None,
        "resource_lease_reason": "Small HTTP metadata/range reads; no GPU, CPU-heavy, or multi-GB disk workload",
    })

    source_snapshot = artifacts_dir / "source_snapshot.json"
    records_path = artifacts_dir / "document_records.jsonl"
    try:
        import pyarrow
        import pyarrow.parquet as pq
        import requests

        write_json(run_dir / "environment.json", {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "pyarrow": pyarrow.__version__, "fsspec": "not_used_v2", "requests": requests.__version__,
        })
        quoted_path = "/".join(requests.utils.quote(piece, safe="") for piece in cfg["source_path"].split("/"))
        pinned_url = f"https://huggingface.co/datasets/{cfg['dataset_id']}/resolve/{cfg['dataset_commit']}/{quoted_path}"
        head = requests.get(pinned_url, allow_redirects=False, stream=True, timeout=30)
        details["head_status"] = head.status_code
        headers = {key.lower(): value for key, value in head.headers.items()}
        checks["pinned_url_redirect"] = head.status_code in {302, 303, 307, 308}
        checks["repo_commit_exact"] = headers.get("x-repo-commit") == cfg["dataset_commit"]
        checks["linked_etag_exact"] = headers.get("x-linked-etag", "").strip('"') == cfg["source_lfs_sha256"]
        checks["linked_size_exact"] = int(headers.get("x-linked-size", -1)) == cfg["source_bytes"]
        checks["range_advertised"] = headers.get("accept-ranges", "").lower() == "bytes"
        head.close()

        final_url = headers.get("location")
        if not final_url:
            raise OSError("pinned URL response omitted redirect location")
        session = requests.Session()
        with RequestsRangeReader(session, final_url, cfg["source_bytes"], cfg["range_block_bytes"]) as stream:
            parquet = pq.ParquetFile(stream)
            metadata = parquet.metadata
            checks["parquet_has_rows"] = metadata.num_rows > 0 and metadata.num_row_groups > 0
            first_group = metadata.row_group(0)
            table = parquet.read_row_group(0, columns=list(FINEWEB_FIELDS)).slice(0, cfg["rows_to_probe"])
            pinned_rows = [normalized_row(row) for row in table.to_pylist()]
            details["parquet"] = {
                "num_rows": metadata.num_rows, "num_row_groups": metadata.num_row_groups,
                "first_row_group_rows": first_group.num_rows,
                "first_row_group_uncompressed_bytes": first_group.total_byte_size,
                "rows_read": len(pinned_rows),
                "range_requests": stream.range_requests,
                "range_bytes_received": stream.bytes_received,
            }
        checks["requested_rows_read"] = len(pinned_rows) == cfg["rows_to_probe"]

        viewer_payload = None
        if cfg["viewer_api_cross_check"]:
            viewer_url = (
                "https://datasets-server.huggingface.co/rows?"
                f"dataset={requests.utils.quote(cfg['dataset_id'], safe='')}&"
                f"config={requests.utils.quote(cfg['dataset_config'], safe='')}&split={cfg['dataset_split']}&"
                f"offset=0&length={cfg['rows_to_probe']}"
            )
            viewer_response = requests.get(viewer_url, timeout=30)
            viewer_response.raise_for_status()
            viewer_payload = viewer_response.json()
            viewer_rows = []
            no_truncation = True
            exact_indices = True
            for expected_index, item in enumerate(viewer_payload.get("rows", [])):
                exact_indices &= item.get("row_idx") == expected_index
                no_truncation &= not item.get("truncated_cells")
                viewer_rows.append(normalized_row(item["row"]))
            checks["viewer_row_indices_exact"] = exact_indices and len(viewer_rows) == len(pinned_rows)
            checks["viewer_cells_untruncated"] = no_truncation
            checks["viewer_matches_pinned_parquet"] = canonical_sha256(viewer_rows) == canonical_sha256(pinned_rows)
            details["viewer"] = {
                "rows": len(viewer_rows), "num_rows_total": viewer_payload.get("num_rows_total"),
                "partial": viewer_payload.get("partial"),
                "revision_boundary": "viewer API has no revision parameter; used only as sampled parity cross-check",
            }

        records = [
            fineweb_document_record(
                row, row_index=index, dataset_id=cfg["dataset_id"], dataset_config=cfg["dataset_config"],
                dataset_commit=cfg["dataset_commit"], source_path=cfg["source_path"],
                split_salt=cfg["split_salt"], validation_basis_points=cfg["validation_basis_points"],
            )
            for index, row in enumerate(pinned_rows)
        ]
        record_report = validate_document_records(records)
        checks["document_ids_unique"] = bool(record_report["unique_document_ids"])
        checks["source_rows_unique"] = bool(record_report["unique_source_rows"])
        checks["text_hashes_unique_in_probe"] = bool(record_report["unique_text_hashes"])
        checks["both_document_splits_present"] = record_report["train_documents"] > 0 and record_report["validation_documents"] > 0
        with records_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row, record in zip(pinned_rows, records, strict=True):
                stream.write(json.dumps({**record, "text": row["text"]}, sort_keys=True, ensure_ascii=False) + "\n")
        details["record_report"] = record_report
        write_json(source_snapshot, {
            "pinned_url": pinned_url, "response_headers": headers,
            "expected": {"commit": cfg["dataset_commit"], "lfs_sha256": cfg["source_lfs_sha256"], "bytes": cfg["source_bytes"]},
            "parquet": details["parquet"], "viewer": details.get("viewer"),
        })
    except Exception:
        stderr_lines.append(traceback.format_exc())
        if not source_snapshot.exists():
            write_json(source_snapshot, {"error": "probe failed before source snapshot completed", "config_source": str(config_path)})

    input_entries = [local_entry(config_path, "CCAD prewritten config", "internal artifact", "resolved_probe_protocol")]
    input_entries.append(local_entry(source_snapshot, "Hugging Face pinned dataset response", cfg["dataset_license"], "source_response_snapshot"))
    if records_path.exists():
        input_entries.append(local_entry(records_path, "FineWeb pinned Parquet range read", cfg["dataset_license"], "document_probe_records"))
    write_json(run_dir / "inputs.json", {"inputs": input_entries})
    with (run_dir / "metrics.raw.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for name, passed in sorted(checks.items()):
            stream.write(json.dumps({"check": name, "passed": passed}, sort_keys=True) + "\n")
    status = "PASS" if checks and all(checks.values()) and not stderr_lines else "FAIL"
    summary = {
        "status": status, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed), "details": details,
        "metrics_raw_sha256": sha256(run_dir / "metrics.raw.jsonl"),
        "generator_script_path": "scripts/run_r006a_fineweb_manifest_probe.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()), "scope_limit": cfg["scope_limit"],
    }
    write_json(run_dir / "metrics.summary.json", summary)
    write_json(run_dir / "status.json", {"status": status, "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat()})
    stdout_lines.append(json.dumps({"run_id": cfg["run_id"], "status": status, "checks": f"{sum(checks.values())}/{len(checks)}"}, sort_keys=True))
    (run_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("\n".join(stderr_lines), encoding="utf-8")
    validation = validate_run_directory(run_dir)
    write_json(run_dir / "contract_validation.json", {"ok": validation.ok, "errors": list(validation.errors)})
    print(json.dumps({"run_id": cfg["run_id"], "status": status, "contract_ok": validation.ok, "checks": checks}, sort_keys=True))
    return 0 if status == "PASS" and validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
