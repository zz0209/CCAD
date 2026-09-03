import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "audit_r006b3a_task_provenance.py"
SPEC = importlib.util.spec_from_file_location("r006b3a_provenance", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProvenanceAuditTests(unittest.TestCase):
    def test_task_id_from_save_name_is_stable(self) -> None:
        self.assertEqual(
            MODULE.task_id_from_save_name("cleaned_data/54_cs_tf.csv"), "54_cs_tf"
        )

    def test_source_evidence_is_fail_closed(self) -> None:
        rows = [
            {
                "Dataset Tag": "base",
                "Dataset save name": "cleaned_data/1_base.csv",
                "Data type": "Multiclass Classification",
                "Link": "https://example.test/base",
                "Source": "upstream",
            },
            {
                "Dataset Tag": "base_A",
                "Dataset save name": "cleaned_data/2_base_A.csv",
                "Data type": "Binary Classification",
                "Link": "",
                "Source": "",
            },
            {
                "Dataset Tag": "orphan",
                "Dataset save name": "cleaned_data/3_orphan.csv",
                "Data type": "Binary Classification",
                "Link": "",
                "Source": "",
            },
        ]
        evidence = MODULE.source_evidence(rows)
        self.assertEqual(
            evidence["2_base_A"]["source_evidence_kind"], "derived_parent_url"
        )
        self.assertEqual(evidence["3_orphan"]["source_evidence_kind"], "unresolved")


if __name__ == "__main__":
    unittest.main()
