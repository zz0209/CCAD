"""Deterministic document records for CCAD corpus manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping


FINEWEB_FIELDS = (
    "text", "id", "dump", "url", "date", "file_path",
    "language", "language_score", "token_count",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def document_split(
    dataset_commit: str,
    document_id: str,
    *,
    salt: str,
    validation_basis_points: int,
) -> str:
    """Assign a document to train/validation without depending on source order."""
    if not 0 < validation_basis_points < 10_000:
        raise ValueError("validation_basis_points must be in (0, 10000)")
    key = f"{salt}\0{dataset_commit}\0{document_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10_000
    return "validation" if bucket < validation_basis_points else "train"


def paired_document_split(dataset_commit: str, document_id: str, *, salt: str) -> str:
    """Assign an immutable document-level 10/40/20/30 paired-data split."""
    key = f"{salt}\0{dataset_commit}\0{document_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10_000
    if bucket < 1_000:
        return "mean"
    if bucket < 5_000:
        return "discovery"
    if bucket < 7_000:
        return "calibration"
    return "audit"


def fineweb_document_record(
    row: Mapping[str, object],
    *,
    row_index: int,
    dataset_id: str,
    dataset_config: str,
    dataset_commit: str,
    source_path: str,
    split_salt: str,
    validation_basis_points: int,
) -> dict[str, object]:
    missing = [field for field in FINEWEB_FIELDS if field not in row]
    if missing:
        raise ValueError(f"FineWeb row missing fields: {missing}")
    document_id = row["id"]
    text = row["text"]
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("FineWeb id must be a non-empty string")
    if not isinstance(text, str) or not text:
        raise ValueError("FineWeb text must be a non-empty string")
    source_file_path = row["file_path"]
    if not isinstance(source_file_path, str) or not source_file_path:
        raise ValueError("FineWeb file_path must be a non-empty string")
    return {
        "schema_version": "0.1.0",
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_commit": dataset_commit,
        "source_parquet_path": source_path,
        "source_row_index": int(row_index),
        "document_id": document_id,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "split": document_split(
            dataset_commit,
            document_id,
            salt=split_salt,
            validation_basis_points=validation_basis_points,
        ),
        "source_metadata": {
            "dump": row["dump"],
            "url": row["url"],
            "date": row["date"],
            "file_path": source_file_path,
            "language": row["language"],
            "language_score": row["language_score"],
            "reported_token_count": row["token_count"],
        },
    }


def validate_document_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    ids = [str(record.get("document_id")) for record in rows]
    row_keys = [
        (str(record.get("source_parquet_path")), int(record.get("source_row_index", -1)))
        for record in rows
    ]
    text_hashes = [str(record.get("text_sha256")) for record in rows]
    splits = [str(record.get("split")) for record in rows]
    return {
        "documents": len(rows),
        "unique_document_ids": len(set(ids)) == len(ids),
        "unique_source_rows": len(set(row_keys)) == len(row_keys),
        "unique_text_hashes": len(set(text_hashes)) == len(text_hashes),
        "train_documents": splits.count("train"),
        "validation_documents": splits.count("validation"),
        "records_sha256": canonical_sha256(rows),
    }
