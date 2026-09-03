from __future__ import annotations

import unittest

import numpy as np

from ccad.decoder_diagnostics import pairwise_decoder_cosine_similarity


class DecoderDiagnosticsTests(unittest.TestCase):
    def test_orthogonal_decoder_has_zero_cdec(self) -> None:
        self.assertEqual(pairwise_decoder_cosine_similarity(np.eye(4)), 0.0)

    def test_absolute_cosine_and_block_accounting(self) -> None:
        decoder = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0]])
        expected = (1.0 + 2.0 / np.sqrt(2.0)) / 3.0
        self.assertAlmostEqual(pairwise_decoder_cosine_similarity(decoder, block_size=2), expected)
        self.assertAlmostEqual(pairwise_decoder_cosine_similarity(decoder, block_size=99), expected)

    def test_invalid_decoder_fails_closed(self) -> None:
        invalid = [np.ones((1, 3)), np.zeros((2, 3)), np.array([[1.0], [np.nan]])]
        for decoder in invalid:
            with self.subTest(shape=decoder.shape), self.assertRaises(ValueError):
                pairwise_decoder_cosine_similarity(decoder)

        with self.assertRaises(ValueError):
            pairwise_decoder_cosine_similarity(np.eye(2), block_size=0)


if __name__ == "__main__":
    unittest.main()
