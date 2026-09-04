from __future__ import annotations

import unittest

import numpy as np

from ccad.causal_metric_probe import (
    hashed_vocab_sketch,
    rademacher_direction,
    select_document_balanced_states,
)


class CausalMetricProbeTests(unittest.TestCase):
    def test_document_balanced_selection_is_deterministic_unique_and_split_bound(self) -> None:
        records = [
            {"split": "discovery", "sequence_index": index, "token_sha256": f"h{index}", "document_ids": [f"d{index % 5}"]}
            for index in range(20)
        ] + [{"split": "audit", "sequence_index": 0, "token_sha256": "audit", "document_ids": ["a"]}]
        first = select_document_balanced_states(records, split="discovery", count=12, token_positions=(3, 7), salt="x")
        second = select_document_balanced_states(records, split="discovery", count=12, token_positions=(3, 7), salt="x")
        self.assertEqual(first, second)
        self.assertEqual(len({row["state_key"] for row in first}), 12)
        self.assertTrue(all(row["split"] == "discovery" for row in first))
        counts = {document: sum(row["blocking_document_id"] == document for row in first) for document in {f"d{i}" for i in range(5)}}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_vocab_sketch_and_directions_are_reproducible(self) -> None:
        ids, signs = hashed_vocab_sketch(100, 16, "salt")
        ids_again, signs_again = hashed_vocab_sketch(100, 16, "salt")
        np.testing.assert_array_equal(ids, ids_again)
        np.testing.assert_array_equal(signs, signs_again)
        self.assertEqual(len(set(ids.tolist())), 16)
        direction = rademacher_direction(64, "state", 2, "salt")
        np.testing.assert_allclose(direction, rademacher_direction(64, "state", 2, "salt"))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertTrue(set(np.unique(direction)).issubset({-0.125, 0.125}))


if __name__ == "__main__":
    unittest.main()
