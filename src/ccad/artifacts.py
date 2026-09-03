"""Deterministic validation for the CCAD run artifact contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FILES = {
    "manifest.json",
    "config.resolved.json",
    "environment.json",
    "inputs.json",
    "code_hashes.json",
    "status.json",
    "stdout.log",
    "stderr.log",
    "metrics.raw.jsonl",
    "metrics.summary.json",
}

REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "run_parent",
    "purpose",
    "milestone",
    "evidence_level",
    "started_utc",
    "project_root",
    "config_hash",
    "code_snapshot_hash",
    "audit_opened",
    "candidate_family_frozen",
    "mean_constants_source_split",
    "threshold_source_split",
    "statistics_unit",
    "device",
    "seeds",
    "resource_lease",
    "resource_lease_reason",
}


@dataclass(frozen=True)
class ContractValidation:
    ok: bool
    errors: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def validate_run_directory(run_dir: Path) -> ContractValidation:
    errors: list[str] = []
    present = {p.name for p in run_dir.iterdir()} if run_dir.is_dir() else set()
    missing = sorted(REQUIRED_FILES - present)
    if missing:
        errors.append(f"missing required files: {missing}")
        return ContractValidation(False, tuple(errors))
    try:
        manifest = _json(run_dir / "manifest.json")
        config = _json(run_dir / "config.resolved.json")
        inputs = _json(run_dir / "inputs.json")
        code_hashes = _json(run_dir / "code_hashes.json")
        summary = _json(run_dir / "metrics.summary.json")
        status = _json(run_dir / "status.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ContractValidation(False, (f"invalid JSON artifact: {exc}",))
    absent_fields = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if absent_fields:
        errors.append(f"manifest missing fields: {absent_fields}")
    if manifest.get("audit_opened") is True and manifest.get("candidate_family_frozen") is not True:
        errors.append("audit opened without frozen candidate family")
    if status.get("status") not in {"RUNNING", "PASS", "FAIL", "BLOCKED", "CUT"}:
        errors.append(f"invalid status: {status.get('status')!r}")
    if manifest.get("config_hash") != sha256(run_dir / "config.resolved.json"):
        errors.append("config_hash does not match resolved config artifact")
    raw_hash = sha256(run_dir / "metrics.raw.jsonl")
    if summary.get("metrics_raw_sha256") != raw_hash:
        errors.append("summary does not identify exact raw metrics hash")
    entries = inputs.get("inputs")
    if not isinstance(entries, list) or not entries:
        errors.append("inputs.json must contain a non-empty inputs list")
    else:
        for index, entry in enumerate(entries):
            for field in ("path", "sha256", "bytes", "source", "license_or_access_boundary", "role"):
                if field not in entry:
                    errors.append(f"inputs[{index}] missing {field}")
    code_entries = code_hashes.get("files")
    if not isinstance(code_entries, list) or not code_entries:
        errors.append("code_hashes.json must contain imported source files")
    elif not code_hashes.get("aggregate_sha256"):
        errors.append("code_hashes.json missing aggregate_sha256")
    else:
        recorded = {entry.get("path"): entry.get("sha256") for entry in code_entries}
        generator_path = summary.get("generator_script_path", "scripts/run_r001_smoke.py")
        generator_hash = recorded.get(generator_path) or recorded.get(generator_path.replace("/", "\\"))
        if summary.get("generator_script_sha256") != generator_hash:
            errors.append("summary generator script hash does not match recorded run snapshot")
        aggregate = hashlib.sha256(
            "".join(
                f"{entry.get('path')}:{entry.get('sha256')}\n"
                for entry in sorted(code_entries, key=lambda item: str(item.get("path")))
            ).encode()
        ).hexdigest()
        if code_hashes.get("aggregate_sha256") != aggregate:
            errors.append("code aggregate hash mismatch")
        if manifest.get("code_snapshot_hash") != aggregate:
            errors.append("manifest code_snapshot_hash mismatch")
        if manifest.get("source_snapshot_required") is True:
            snapshot_root = code_hashes.get("snapshot_root")
            if snapshot_root != "source_snapshot":
                errors.append("required source snapshot root is missing or invalid")
            for index, entry in enumerate(code_entries):
                snapshot_relative = entry.get("snapshot_path")
                if not isinstance(snapshot_relative, str):
                    errors.append(f"code_hashes.files[{index}] missing snapshot_path")
                    continue
                snapshot_path = run_dir / snapshot_relative
                try:
                    inside_run = snapshot_path.resolve().is_relative_to(run_dir.resolve())
                except OSError:
                    inside_run = False
                if not inside_run or not snapshot_path.is_file():
                    errors.append(f"source snapshot missing or outside run: {snapshot_relative}")
                elif sha256(snapshot_path) != entry.get("sha256"):
                    errors.append(f"source snapshot hash mismatch: {snapshot_relative}")
    if config.get("audit_opened") != manifest.get("audit_opened"):
        errors.append("audit_opened differs between config and manifest")
    return ContractValidation(not errors, tuple(errors))
