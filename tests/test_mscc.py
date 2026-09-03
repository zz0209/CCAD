from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import numpy as np

from ccad.mscc import (
    freeze_mscc_prediction,
    minimum_support_contribution_correspondence,
    source_conditioned_topk_proposal,
    verify_frozen_mscc_prediction,
)


def kernels(source: np.ndarray, targets: np.ndarray):
    """Build empirical contribution kernels from nxd and nxtxd processes."""
    n = source.shape[0]
    flat_targets = targets.transpose(1, 0, 2).reshape(targets.shape[1], -1)
    flat_source = source.reshape(1, -1)
    return flat_source @ flat_source.T / n, flat_source @ flat_targets.T / n, flat_targets @ flat_targets.T / n


def run_mscc(source, targets, *, means=None, proposed=None, complete=True, budget=7462, g_max=4):
    k_ss, k_st, k_tt = kernels(source, targets)
    d = source.shape[1]
    if means is None:
        means = (np.zeros((d, 1)), np.zeros((d, targets.shape[1])))
    if proposed is None:
        proposed = tuple(range(targets.shape[1]))
    return minimum_support_contribution_correspondence(
        k_ss, k_st, k_tt, means[0], means[1], source_atom_id=0,
        proposed_target_ids=proposed, g_max=g_max, tau_ctr=1e-10,
        tau_mu=1e-10, epsilon=1e-12, candidate_budget=budget,
        complete_universe=complete,
    )


class MSCCTests(unittest.TestCase):
    def test_public_api_is_truth_and_eval_blind(self):
        names = set(inspect.signature(minimum_support_contribution_correspondence).parameters)
        self.assertTrue(names.isdisjoint({"truth", "labels", "planted_support", "eval", "audit"}))

    def test_unique_minimum_support_beats_bloated_superset(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4096, 1))
        noise = rng.standard_normal((4096, 1))
        targets = np.stack([0.4 * x, 0.6 * x, noise], axis=1)
        result = run_mscc(x, targets)
        self.assertEqual((result.identification, result.multiplicity, result.minimum_support_size), ("FOUND", "UNIQUE", 2))
        self.assertEqual(result.supports[0].target_ids, (0, 1))
        self.assertEqual(result.evaluated_count, 7)

    def test_tied_minimum_supports_are_all_returned(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((2048, 1))
        targets = np.stack([0.25 * x, 0.75 * x, 0.4 * x, 0.6 * x], axis=1)
        result = run_mscc(x, targets)
        self.assertEqual((result.identification, result.multiplicity), ("FOUND", "AMBIGUOUS"))
        self.assertEqual({item.target_ids for item in result.supports}, {(0, 1), (2, 3)})

    def test_absence_wording_requires_complete_universe(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((2048, 1))
        targets = np.stack([2.0 * x, 3.0 * x], axis=1)
        complete = run_mscc(x, targets, complete=True)
        partial = run_mscc(x, targets, proposed=(0,), complete=False)
        self.assertEqual(complete.identification, "CERTIFIED_ABSENT")
        self.assertEqual(partial.identification, "UNRESOLVED")
        self.assertEqual(partial.unresolved_reason, "NO_ACCEPTED_IN_FROZEN_FAMILY")
        with self.assertRaises(ValueError):
            run_mscc(x, targets, proposed=(0,), complete=True)

    def test_budget_refusal_happens_before_scoring(self):
        rng = np.random.default_rng(4)
        source = rng.standard_normal((32, 1))
        targets = rng.standard_normal((32, 21, 1))
        result = run_mscc(source, targets, complete=True, budget=7462)
        self.assertEqual(result.status, "BUDGET_REFUSAL")
        self.assertEqual(result.identification, "UNRESOLVED")
        self.assertEqual(result.evaluated_count, 0)
        self.assertEqual(result.planned_candidate_count, 7546)

    def test_dense_rotation_atom_absent_but_full_block_equal(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((8192, 2))
        c = 1.0 / np.sqrt(2.0)
        q = np.asarray([[c, -c], [c, c]])
        source_atoms = np.stack([x[:, [i]] * np.eye(2)[i] for i in range(2)], axis=1)
        rotated_coordinates = x @ q
        target_atoms = np.stack([rotated_coordinates[:, [j]] * q[:, j] for j in range(2)], axis=1)
        result = run_mscc(source_atoms[:, 0, :], target_atoms, complete=True, g_max=2)
        self.assertEqual(result.identification, "CERTIFIED_ABSENT")
        np.testing.assert_allclose(np.sum(source_atoms, axis=1), np.sum(target_atoms, axis=1), atol=1e-12, rtol=1e-12)

    def test_mean_mismatch_rejects_centered_match(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((2048, 1))
        targets = np.stack([x], axis=1)
        source_means = np.asarray([[1.0]])
        target_means = np.asarray([[2.0]])
        result = run_mscc(x, targets, means=(source_means, target_means))
        self.assertEqual(result.identification, "CERTIFIED_ABSENT")
        self.assertLessEqual(result.best_candidate.d_ctr, 1e-10)
        self.assertGreater(result.best_candidate.d_mu, 1e-10)

    def test_materially_invalid_kernel_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "negative squared residual"):
            minimum_support_contribution_correspondence(
                np.asarray([[1.0]]), np.asarray([[2.0]]), np.asarray([[1.0]]),
                np.zeros((1, 1)), np.zeros((1, 1)), source_atom_id=0,
                proposed_target_ids=(0,), g_max=1, tau_ctr=1e-10, tau_mu=1e-10,
                epsilon=1e-12, candidate_budget=1, complete_universe=True,
            )

    def test_source_conditioned_proposal_selects_lowest_residual_atoms(self):
        k_ss = np.asarray([[1.0]])
        k_st = np.asarray([[0.45, 0.1, 0.4]])
        k_tt = np.diag([0.25, 1.0, 0.16])
        proposal = source_conditioned_topk_proposal(
            k_ss, k_st, k_tt, source_atom_id=0, atom_cap=2, g_max=2,
            epsilon=1e-12, candidate_budget=3, boundary_tie_tolerance=1e-12,
        )
        self.assertEqual(proposal.status, "OK")
        self.assertEqual(proposal.proposed_target_ids, (0, 2))
        self.assertEqual(proposal.full_dictionary_comparisons, 3)
        self.assertEqual(proposal.planned_support_count, 3)

    def test_source_conditioned_proposal_refuses_boundary_tie(self):
        proposal = source_conditioned_topk_proposal(
            np.asarray([[1.0]]), np.asarray([[0.5, 0.5, 0.0]]), np.eye(3),
            source_atom_id=0, atom_cap=1, g_max=1, epsilon=1e-12,
            candidate_budget=1, boundary_tie_tolerance=1e-12,
        )
        self.assertEqual(proposal.status, "BUDGET_REFUSAL")
        self.assertEqual(proposal.refusal_reason, "BOUNDARY_TIE")
        self.assertEqual(proposal.proposed_target_ids, ())

    def test_prediction_freeze_is_deterministic_and_tamper_evident(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((1024, 1))
        targets = np.stack([0.4 * x, 0.6 * x], axis=1)
        result = run_mscc(x, targets)
        frozen = freeze_mscc_prediction(
            result, protocol_hash="A" * 64, proposal_hash="B" * 64,
            discovery_fingerprint="C" * 64, source_atom_id=0,
        )
        repeated = freeze_mscc_prediction(
            result, protocol_hash="A" * 64, proposal_hash="B" * 64,
            discovery_fingerprint="C" * 64, source_atom_id=0,
        )
        self.assertEqual(frozen, repeated)
        self.assertTrue(verify_frozen_mscc_prediction(frozen, result))
        self.assertFalse(verify_frozen_mscc_prediction(replace(frozen, proposal_hash="D" * 64), result))


if __name__ == "__main__":
    unittest.main()
