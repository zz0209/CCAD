"""A gate counterexample, not a claim about the outcome of real F4 data."""
import unittest
import numpy as np
from ccad.hook_transport import transport_metrics


class SpecificityCounterexample(unittest.TestCase):
    def test_equal_accuracy_can_coexist_with_very_selective_effect(self):
        positive = np.array([[-1.0], [1.0]])
        negative = 0.01 * positive
        pos = transport_metrics(positive, positive, np.ones(2))
        neg = transport_metrics(negative, negative, np.ones(2))
        self.assertAlmostEqual(pos.bcc - neg.bcc, 0.0)
        # Linear downstream endpoint: effect = removed contribution.
        selective_effect = (np.sum(positive**2) - np.sum(negative**2)) / (np.sum(positive**2) + np.sum(negative**2))
        self.assertGreater(selective_effect, 0.999)
