import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.data_manifest import (  # noqa: E402
    document_split,
    fineweb_document_record,
    paired_document_split,
    validate_document_records,
)


class DataManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "text": "deterministic public text",
            "id": "<urn:uuid:00000000-0000-0000-0000-000000000001>",
            "dump": "CC-MAIN-TEST",
            "url": "https://example.test/document",
            "date": "2026-01-01T00:00:00Z",
            "file_path": "s3://example/test.warc.gz",
            "language": "en",
            "language_score": 0.99,
            "token_count": 3,
        }

    def test_split_is_order_independent_and_commit_bound(self) -> None:
        first = document_split("commit-a", "doc-1", salt="r006", validation_basis_points=1000)
        self.assertEqual(first, document_split("commit-a", "doc-1", salt="r006", validation_basis_points=1000))
        assignments = {
            document_split("commit-a", f"doc-{i}", salt="r006", validation_basis_points=1000)
            for i in range(100)
        }
        self.assertEqual(assignments, {"train", "validation"})

    def test_paired_split_is_document_stable_and_has_locked_four_way_support(self) -> None:
        first = paired_document_split("commit-a", "doc-1", salt="r008")
        self.assertEqual(first, paired_document_split("commit-a", "doc-1", salt="r008"))
        labels = {paired_document_split("commit-a", f"doc-{index}", salt="r008") for index in range(1000)}
        self.assertEqual(labels, {"mean", "discovery", "calibration", "audit"})

    def test_record_excludes_text_but_binds_its_hash(self) -> None:
        record = fineweb_document_record(
            self.row,
            row_index=7,
            dataset_id="HuggingFaceFW/fineweb",
            dataset_config="sample-10BT",
            dataset_commit="abc123",
            source_path="sample/10BT/000_00000.parquet",
            split_salt="r006",
            validation_basis_points=1000,
        )
        self.assertNotIn("text", record)
        self.assertEqual(record["source_row_index"], 7)
        self.assertEqual(len(record["text_sha256"]), 64)

    def test_missing_required_field_fails_closed(self) -> None:
        del self.row["id"]
        with self.assertRaises(ValueError):
            fineweb_document_record(
                self.row, row_index=0, dataset_id="fineweb", dataset_config="sample",
                dataset_commit="abc", source_path="a.parquet", split_salt="r006",
                validation_basis_points=1000,
            )

    def test_duplicate_text_is_reported(self) -> None:
        base = fineweb_document_record(
            self.row, row_index=0, dataset_id="fineweb", dataset_config="sample",
            dataset_commit="abc", source_path="a.parquet", split_salt="r006",
            validation_basis_points=1000,
        )
        other = dict(base)
        other["document_id"] = "different-id"
        other["source_row_index"] = 1
        report = validate_document_records([base, other])
        self.assertTrue(report["unique_document_ids"])
        self.assertFalse(report["unique_text_hashes"])


if __name__ == "__main__":
    unittest.main()
