from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_m1_parent_integrity import inspect_run


class ParentIntegrityAuditTests(unittest.TestCase):
    def test_missing_run_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            row = inspect_run(Path(temporary), "runs/absent", {"status.json"})
        self.assertFalse(row["exists"])
        self.assertFalse(row["passes"])
        self.assertEqual(row["missing_files"], ["status.json"])

    def test_incomplete_run_fails_closed_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs" / "partial"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            row = inspect_run(Path(temporary), "runs/partial", {"status.json", "summary.json"})
        self.assertFalse(row["passes"])
        self.assertEqual(row["missing_files"], ["summary.json"])


if __name__ == "__main__":
    unittest.main()
