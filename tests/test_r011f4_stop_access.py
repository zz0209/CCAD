"""Exercise the real runner's early stop using small files and forbidden loaders."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transport_stop_runner", PROJECT / "scripts/run_r011f4_hook_transport_real.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerStopAccessTests(unittest.TestCase):
    def test_nuisance_stop_finalizes_without_opening_calibration(self):
        cfg = json.loads((PROJECT / "configs/r011f5_residual_transport_real_v3.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
                return path
            for name in ("scripts/runner.py", "src/ccad/hook_transport.py", "src/ccad/split_access.py", "scripts/run_r009c_atom_discovery.py", "scripts/run_r011f1_euclidean_surface.py"):
                save(name, {})
            cfg.update(run_id="early_stop_fixture", source_seeds=[1], num_latents=2, hook_hidden_size=2,
                       bulk_asset_dir=str(root / "assets"), raw_hook_asset_dir=str(root / "raw"), execution_enabled=True)
            bindings = {"protocol_document": "protocol.json", "synthetic_gate_status_path": "synthetic.json", "synthetic_gate_metrics_path": "metrics.json", "reference_surface_path": "reference.jsonl", "source_census_path": "census.jsonl", "sequence_records_path": "sequences.json", "unresidualized_transport_surface_path": "old.jsonl"}
            for key, name in bindings.items():
                cfg[key] = name
                save(name, {})
            save("synthetic.json", {"status": "PASS"})
            (root / "reference.jsonl").write_text("")
            (root / "census.jsonl").write_text("\n".join(json.dumps({"seed": 1, "atom": atom, "mean_code": 0}) for atom in (0, 1)))
            save("sequences.json", {"sequences": []})
            save("assets/asset_manifest.json", {"splits": [{"split": s, "tokens": 2} for s in ("discovery", "calibration")]})
            raw_rows = []
            for split in ("mean", "discovery", "calibration"):
                path = root / f"{split}.bin"
                if split != "calibration":
                    np.ones((2, 2), dtype="<f4").tofile(path)
                raw_rows.append({"split": split, "path": str(path), "shape": [2, 2]})
            save("raw/raw_hook_manifest.json", {"splits": raw_rows})
            for key in cfg:
                if key.endswith("_sha256"):
                    cfg[key] = "0" * 64
            config_path = save("config.json", cfg)
            failed_nuisance = SimpleNamespace(status="THRESHOLD_NOT_REACHED", rank=64, explained_variance_fraction=0.8873)
            with patch.object(runner, "ROOT", root), patch.object(runner, "__file__", str(root / "scripts/runner.py")), \
                 patch.object(runner, "sha256", return_value="0" * 64), \
                 patch.object(runner.subprocess, "check_output", return_value="fixture"), \
                 patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="")), \
                 patch.object(runner, "decoder", return_value=np.eye(2)), \
                 patch.object(runner, "sparse_codes", side_effect=AssertionError("stop must precede code loading")) as codes, \
                 patch.object(runner, "select_document_balanced_states", return_value=[{"sequence_index": 0, "token_position": 0}]), \
                 patch.object(runner, "fit_nuisance_projector", return_value=failed_nuisance), \
                 patch.object(runner, "validate_run_directory", return_value=SimpleNamespace(ok=True, errors=[])), \
                 patch.object(sys, "argv", ["runner", "--config", str(config_path)]):
                self.assertEqual(runner.main(), 0)
            codes.assert_not_called()
            run = root / "runs" / cfg["run_id"]
            access = json.loads((run / "split_access.json").read_text())
            self.assertEqual([e["split"] for e in access["raw_hook"]], ["mean", "discovery"])
            self.assertEqual(access["codes"], [])
            record = json.loads((run / "metrics.raw.jsonl").read_text())
            self.assertTrue(record["checks"]["no_calibration_read"])
            self.assertFalse(record["calibration_transport_evaluated"])
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "PASS")
