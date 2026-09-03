from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_m1_nip_d0", ROOT / "scripts/run_m1_nip_d0.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)

SPEC_V2 = importlib.util.spec_from_file_location("run_m1_nip_d0_v2", ROOT / "scripts/run_m1_nip_d0_v2.py")
RUNNER_V2 = importlib.util.module_from_spec(SPEC_V2)
assert SPEC_V2 and SPEC_V2.loader
SPEC_V2.loader.exec_module(RUNNER_V2)


class M1NIPD0ContractTests(unittest.TestCase):
    def test_exception_finalizer_persists_terminal_failure_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            RUNNER.write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": "fixed"})
            try:
                raise RuntimeError("deliberate-contract-test")
            except RuntimeError as error:
                RUNNER.finalize_failure(run_dir, error)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            stderr = (run_dir / "stderr.log").read_text(encoding="utf-8")
            self.assertEqual(status["status"], "FAIL")
            self.assertEqual(status["started_utc"], "fixed")
            self.assertEqual(status["failure_type"], "RuntimeError")
            self.assertIn("deliberate-contract-test", status["failure_message"])
            self.assertIn("RuntimeError: deliberate-contract-test", stderr)

    def test_v2_exception_finalizer_persists_terminal_failure_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            RUNNER_V2.write_json(run_dir / "status.json", {"status": "RUNNING", "started_utc": "fixed-v2"})
            try:
                raise ValueError("deliberate-v2-contract-test")
            except ValueError as error:
                RUNNER_V2.finalize_failure(run_dir, error)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            stderr = (run_dir / "stderr.log").read_text(encoding="utf-8")
            self.assertEqual(status["status"], "FAIL")
            self.assertEqual(status["started_utc"], "fixed-v2")
            self.assertEqual(status["failure_type"], "ValueError")
            self.assertIn("ValueError: deliberate-v2-contract-test", stderr)


if __name__ == "__main__":
    unittest.main()
