from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


EXPECTED_BINARY_TASKS = 113


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id_from_save_name(value: str) -> str:
    return Path(value).stem


def load_rows(master_csv: Path) -> list[dict[str, str]]:
    with master_csv.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_evidence(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    parents: list[tuple[str, str, str]] = []
    for row in rows:
        tag = (row.get("Dataset Tag") or "").strip()
        url = (row.get("Link") or "").strip()
        source = (row.get("Source") or "").strip()
        if tag and url.startswith("http") and row.get("Data type") != "Binary Classification":
            parents.append((tag, url, source))

    result: dict[str, dict[str, str]] = {}
    last_explicit_url = ""
    for row in rows:
        url = (row.get("Link") or "").strip()
        if url.startswith("http"):
            last_explicit_url = url
        if (row.get("Data type") or "").strip() != "Binary Classification":
            continue

        save_name = (row.get("Dataset save name") or "").strip()
        task_id = task_id_from_save_name(save_name)
        tag = (row.get("Dataset Tag") or "").strip()
        evidence_kind = "unresolved"
        inferred_url = ""
        inferred_parent = ""

        if url.startswith("http"):
            evidence_kind = "explicit_url"
            inferred_url = url
        elif url == "^" and last_explicit_url:
            evidence_kind = "caret_reference"
            inferred_url = last_explicit_url
        else:
            candidates = [item for item in parents if tag.startswith(item[0] + "_")]
            if candidates:
                parent_tag, parent_url, _ = max(candidates, key=lambda item: len(item[0]))
                evidence_kind = "derived_parent_url"
                inferred_url = parent_url
                inferred_parent = parent_tag

        result[task_id] = {
            "task_id": task_id,
            "dataset_tag": tag,
            "save_name": save_name,
            "catalog_source": (row.get("Source") or "").strip(),
            "catalog_url": url,
            "source_evidence_kind": evidence_kind,
            "inferred_source_url": inferred_url,
            "inferred_parent_tag": inferred_parent,
        }
    return result


def build_manifest(
    source_root: Path, verified_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    master_csv = source_root / "raw_data" / "probing_datasets_MASTER.csv"
    rows = load_rows(master_csv)
    tasks = source_evidence(rows)
    if len(tasks) != EXPECTED_BINARY_TASKS:
        raise ValueError(f"expected {EXPECTED_BINARY_TASKS} binary tasks, found {len(tasks)}")

    verified = verified_config["verified_tasks"]
    unknown_verified = sorted(set(verified) - set(tasks))
    if unknown_verified:
        raise ValueError(f"verified config contains unknown tasks: {unknown_verified}")

    manifest: list[dict[str, Any]] = []
    for task_id in sorted(tasks, key=lambda value: int(value.split("_", 1)[0])):
        entry: dict[str, Any] = dict(tasks[task_id])
        raw_path = source_root / "raw_data" / entry["save_name"]
        entry["raw_present"] = raw_path.is_file()
        entry["raw_size_bytes"] = raw_path.stat().st_size if raw_path.is_file() else None
        license_entry = verified.get(task_id)
        entry["license_status"] = "verified" if license_entry else "unresolved"
        entry["license_evidence"] = license_entry
        entry["eligible_for_internal_probe"] = bool(license_entry)
        entry["eligible_for_redistribution"] = False
        manifest.append(entry)

    evidence_counts: dict[str, int] = {}
    for entry in manifest:
        key = entry["source_evidence_kind"]
        evidence_counts[key] = evidence_counts.get(key, 0) + 1

    verified_count = sum(entry["license_status"] == "verified" for entry in manifest)
    use_class_counts: dict[str, int] = {}
    for entry in manifest:
        if entry["license_evidence"]:
            key = entry["license_evidence"]["use_class"]
            use_class_counts[key] = use_class_counts.get(key, 0) + 1
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "semantic_outcome": "INSUFFICIENT_LICENSE_COVERAGE",
        "b3a_gate": "NOT_PASSED",
        "binary_task_count": len(manifest),
        "raw_present_count": sum(entry["raw_present"] for entry in manifest),
        "source_evidence_counts": evidence_counts,
        "license_verified_count": verified_count,
        "license_unresolved_count": len(manifest) - verified_count,
        "verified_use_class_counts": use_class_counts,
        "redistribution_eligible_count": 0,
        "notes": [
            "PASS means the provenance inventory executed and validated its inputs; it is not an evaluator qualification PASS.",
            "No raw task data were copied and no probe activation was computed.",
            "License evidence is task-specific and does not inherit from the sae-probes MIT software license.",
        ],
    }
    return manifest, summary


def main() -> None:
    started_at = dt.datetime.now(dt.timezone.utc)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--verified-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    verified_config = json.loads(args.verified_config.read_text(encoding="utf-8"))
    manifest, summary = build_manifest(args.source_root, verified_config)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    master_csv = args.source_root / "raw_data" / "probing_datasets_MASTER.csv"
    summary["inputs"] = {
        "source_root": str(args.source_root.resolve()),
        "source_commit": "d71c1661fb4017195c147f58547393b4009aacac",
        "source_tag": "v0.4.0",
        "master_csv_sha256": sha256_file(master_csv),
        "verified_config": str(args.verified_config.resolve()),
        "verified_config_sha256": sha256_file(args.verified_config),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    resolved_config = {
        "run_id": args.output_dir.name,
        "source_root": str(args.source_root.resolve()),
        "verified_config": str(args.verified_config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "expected_binary_tasks": EXPECTED_BINARY_TASKS,
        "audit_opened": False,
        "probe_activations_computed": False,
    }
    environment = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "workspace_git_repository": False,
        "source_repository_commit": "d71c1661fb4017195c147f58547393b4009aacac",
    }

    fieldnames = list(manifest[0].keys())
    with (args.output_dir / "task_provenance_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest:
            row = dict(entry)
            row["license_evidence"] = json.dumps(
                row["license_evidence"], sort_keys=True, ensure_ascii=False
            )
            writer.writerow(row)

    (args.output_dir / "task_provenance_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "run_status": "PASS",
                "b3a_gate": "NOT_PASSED",
                "semantic_outcome": summary["semantic_outcome"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ended_at = dt.datetime.now(dt.timezone.utc)
    (args.output_dir / "audit_log.json").write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "exit_status": 0,
                "actions": [
                    "parsed source catalog",
                    "validated 113 binary task identities",
                    "joined pre-verified task license evidence",
                    "wrote fail-closed provenance manifest",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
