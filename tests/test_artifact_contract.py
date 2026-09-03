from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ccad.artifacts import validate_run_directory


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = ROOT / "runs" / "R001_smoke_20260902T020300Z"


class ArtifactContractTests(unittest.TestCase):
    def copied_run(self, parent: Path) -> Path:
        target = parent / "run"
        shutil.copytree(REFERENCE_RUN, target)
        return target

    def test_recorded_hash_ledger_survives_workspace_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            self.assertTrue(validate_run_directory(run).ok)

    def test_required_source_snapshot_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            manifest_path = run / "manifest.json"
            hashes_path = run / "code_hashes.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            manifest["source_snapshot_required"] = True
            hashes["snapshot_root"] = "source_snapshot"
            for entry in hashes["files"]:
                snapshot = run / "source_snapshot" / entry["path"]
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                source = ROOT / entry["path"]
                shutil.copy2(source, snapshot)
                entry["snapshot_path"] = str(snapshot.relative_to(run))
                entry["sha256"] = __import__("hashlib").sha256(snapshot.read_bytes()).hexdigest()
            aggregate = __import__("hashlib").sha256(
                "".join(
                    f"{entry['path']}:{entry['sha256']}\n"
                    for entry in sorted(hashes["files"], key=lambda item: item["path"])
                ).encode()
            ).hexdigest()
            hashes["aggregate_sha256"] = aggregate
            manifest["code_snapshot_hash"] = aggregate
            summary_path = run / "metrics.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            generator = next(item for item in hashes["files"] if item["path"].replace("\\", "/") == "scripts/run_r001_smoke.py")
            summary["generator_script_sha256"] = generator["sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertTrue(validate_run_directory(run).ok)
            (run / generator["snapshot_path"]).write_text("tampered", encoding="utf-8")
            result = validate_run_directory(run)
            self.assertFalse(result.ok)
            self.assertTrue(any("source snapshot hash mismatch" in error for error in result.errors))

    def test_raw_metric_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            with (run / "metrics.raw.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            result = validate_run_directory(run)
            self.assertFalse(result.ok)
            self.assertTrue(any("raw metrics hash" in error for error in result.errors))

    def test_audit_without_candidate_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["audit_opened"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            result = validate_run_directory(run)
            self.assertFalse(result.ok)
            self.assertTrue(any("audit opened" in error for error in result.errors))

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            (run / "inputs.json").unlink()
            result = validate_run_directory(run)
            self.assertFalse(result.ok)
            self.assertTrue(any("missing required files" in error for error in result.errors))

    def test_explicit_generator_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = self.copied_run(Path(temp))
            summary_path = run / "metrics.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["generator_script_path"] = "scripts/run_r001_smoke.py"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.assertTrue(validate_run_directory(run).ok)
