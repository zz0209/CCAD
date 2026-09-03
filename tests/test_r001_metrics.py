from __future__ import annotations

import unittest

import numpy as np

from ccad.metrics import (
    absolute_code_correlation,
    adjusted_rand_index,
    bcc_from_kernels,
    cancellation_diagnostics,
    center_codes,
    contribution_kernel,
    document_bootstrap_bcc,
    explicit_group_contribution,
    projector_subspace_consistency,
    pw_mcc_absolute_cosine,
    occupancy_effective_sample_size,
)
from ccad.matching import enumerate_maximum_exact_covers, exhaustive_balanced_pairs, exhaustive_balanced_search, forced_partition_projection
from ccad.synthetic import cancellation_seeded, competing_covers_seeded, cooccurrence_confounding_seeded, hadamard_gauge_instance, hadamard_gauge_seeded, local_block_rotations, non_lipschitz_downstream_cliff_seeded, partial_overlap_seeded, rare_occupancy_seeded, same_span_different_computation, same_span_different_computation_seeded, same_sum_bloated_span, same_sum_bloated_span_seeded, unequal_split_merge, whole_dictionary_only_seeded


class R001MetricTests(unittest.TestCase):
    def test_seeded_hadamard_separates_structure_and_samples(self) -> None:
        kwargs = dict(q=4, n_mean=17, n_eval=31, mean_sample_seed=3, eval_sample_seed=4)
        first = hadamard_gauge_seeded(structural_seed_a=1, structural_seed_b=2, **kwargs)
        replay = hadamard_gauge_seeded(structural_seed_a=1, structural_seed_b=2, **kwargs)
        changed = hadamard_gauge_seeded(structural_seed_a=11, structural_seed_b=12, **kwargs)
        self.assertTrue(np.array_equal(first.d_left, replay.d_left))
        self.assertTrue(np.array_equal(first.z_left_eval, replay.z_left_eval))
        self.assertFalse(np.array_equal(first.d_left, changed.d_left))
        self.assertFalse(np.array_equal(first.d_right, changed.d_right))
        left = first.z_left_eval @ first.d_left.T
        right = first.z_right_eval @ first.d_right.T
        self.assertTrue(np.allclose(left, right, atol=1e-12, rtol=1e-12))
        self.assertEqual(dict(first.seed_provenance)["structural_seed_a"], 1)

    def test_hadamard_conformance(self) -> None:
        for q in (2, 4, 8):
            pair = hadamard_gauge_instance(q, n_mean=257, n_eval=1024, seed=20260902 + q)
            zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
            zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
            kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
            klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
            krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
            left = np.arange(pair.d_left.shape[1])
            right = np.arange(pair.d_right.shape[1])
            bcc = bcc_from_kernels(kll, klr, krr, left, right)
            self.assertEqual(bcc.status, "OK")
            self.assertAlmostEqual(bcc.value, 1.0, places=12)
            self.assertAlmostEqual(bcc.normalized_residual, 0.0, places=12)
            psc = projector_subspace_consistency(pair.d_left, pair.d_right)
            self.assertAlmostEqual(psc.value, 1.0, places=12)
            self.assertAlmostEqual(pw_mcc_absolute_cosine(pair.d_left, pair.d_right), q ** -0.5, places=12)
            yl = explicit_group_contribution(pair.d_left, pair.z_left_eval, left)
            yr = explicit_group_contribution(pair.d_right, pair.z_right_eval, right)
            np.testing.assert_allclose(yl, yr, rtol=1e-12, atol=1e-12)

    def test_same_span_different_computation(self) -> None:
        pair = same_span_different_computation(2001, 10000, 17)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        ids = np.arange(2)
        bcc = bcc_from_kernels(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            ids,
            ids,
        )
        psc = projector_subspace_consistency(pair.d_left, pair.d_right)
        self.assertAlmostEqual(psc.value, 1.0, places=12)
        self.assertLess(abs(bcc.value), 0.05)

    def test_same_sum_bloated_span(self) -> None:
        pair = same_sum_bloated_span(257, 1024, 23)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        left = np.arange(1)
        right = np.arange(2)
        bcc = bcc_from_kernels(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            left,
            right,
        )
        psc = projector_subspace_consistency(pair.d_left, pair.d_right)
        self.assertAlmostEqual(bcc.value, 1.0, places=12)
        self.assertAlmostEqual(psc.value, 2.0 / 3.0, places=12)

    def test_seeded_function_mismatch_has_clean_control(self) -> None:
        pair = same_span_different_computation_seeded(
            257, 2048, structural_seed_a=101, structural_seed_b=102,
            mean_sample_seed=103, eval_sample_seed=104,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
        klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
        krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
        risk = bcc_from_kernels(kll, klr, krr, (0, 1), (0, 1))
        clean = bcc_from_kernels(kll, klr, krr, (2,), (2,))
        self.assertAlmostEqual(risk.value, 0.0, places=12)
        self.assertAlmostEqual(projector_subspace_consistency(pair.d_left[:, :2], pair.d_right[:, :2]).value, 1.0, places=12)
        self.assertAlmostEqual(clean.value, 1.0, places=12)

    def test_seeded_span_bloat_preserves_sum_and_clean_control(self) -> None:
        pair = same_sum_bloated_span_seeded(
            257, 2048, structural_seed_a=111, structural_seed_b=112,
            mean_sample_seed=113, eval_sample_seed=114,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        risk = bcc_from_kernels(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0,), (0, 1),
        )
        psc = projector_subspace_consistency(pair.d_left[:, (0,)], pair.d_right[:, (0, 1)])
        self.assertAlmostEqual(risk.value, 1.0, places=12)
        self.assertAlmostEqual(psc.value, 2.0 / 3.0, places=12)
        self.assertEqual((psc.rank_left, psc.rank_right), (1, 2))
        np.testing.assert_allclose(
            explicit_group_contribution(pair.d_left, pair.z_left_eval, np.asarray([0])),
            explicit_group_contribution(pair.d_right, pair.z_right_eval, np.asarray([0, 1])),
            rtol=1e-12, atol=1e-12,
        )

    def test_non_lipschitz_cliff_and_lipschitz_control(self) -> None:
        pair = non_lipschitz_downstream_cliff_seeded(
            256, 2048, structural_seed_a=121, structural_seed_b=122,
            mean_sample_seed=123, eval_sample_seed=124,
        )
        zl, mean_left = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, mean_right = center_codes(pair.z_right_mean, pair.z_right_eval)
        bcc = bcc_from_kernels(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0,), (0,),
        )
        ya = explicit_group_contribution(pair.d_left, pair.z_left_eval, np.asarray([0]))
        yb = explicit_group_contribution(pair.d_right, pair.z_right_eval, np.asarray([0]))
        u = pair.d_left[:, 0] / np.linalg.norm(pair.d_left[:, 0])
        v = np.array([-u[1], u[0]])
        state_a = pair.hook_eval - ya
        state_b = pair.hook_eval - yb
        risk_a = (state_a @ u) * (state_a @ v) >= 0.0
        risk_b = (state_b @ u) * (state_b @ v) >= 0.0
        contribution_rmse = np.sqrt(np.mean(np.sum((ya - yb) ** 2, axis=1)))
        smooth_rmse = np.sqrt(np.mean(np.sum((state_a - state_b) ** 2, axis=1)))
        self.assertGreater(bcc.value, 0.9998)
        self.assertAlmostEqual(np.linalg.norm(pair.d_left @ mean_left - pair.d_right @ mean_right), 0.0, places=12)
        self.assertAlmostEqual(np.mean(risk_a != risk_b), 1.0, places=12)
        self.assertAlmostEqual(smooth_rmse / contribution_rmse, 1.0, places=12)

    def test_inactive_and_degenerate_are_not_perfect(self) -> None:
        zero_kernel = np.zeros((1, 1))
        ids = np.arange(1)
        bcc = bcc_from_kernels(zero_kernel, zero_kernel, zero_kernel, ids, ids)
        self.assertEqual(bcc.status, "INACTIVE")
        self.assertIsNone(bcc.value)
        psc = projector_subspace_consistency(np.zeros((2, 1)), np.eye(2))
        self.assertEqual(psc.status, "DEGENERATE_PSC")
        self.assertIsNone(psc.value)

    def test_local_block_rotations_recover_support_minimal_blocks(self) -> None:
        pair = local_block_rotations((2, 2, 2), n_mean=257, n_eval=2048, seed=31)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
        klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
        krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
        for left, right in pair.planted_hyperedges:
            found = exhaustive_balanced_pairs(
                kll,
                klr,
                krr,
                left,
                right,
                residual_tolerance=1e-10,
                max_group_size=4,
            )
            self.assertEqual([(item.left_ids, item.right_ids) for item in found], [(left, right)])
        wrong = exhaustive_balanced_pairs(
            kll,
            klr,
            krr,
            pair.planted_hyperedges[0][0],
            pair.planted_hyperedges[1][1],
            residual_tolerance=1e-10,
            max_group_size=4,
        )
        self.assertEqual(wrong, [])

    def test_exact_tolerance_rejects_visible_code_perturbation(self) -> None:
        pair = local_block_rotations((2, 2, 2), n_mean=257, n_eval=2048, seed=41)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        rng = np.random.default_rng(42)
        _, right = pair.planted_hyperedges[0]
        zr[:, np.asarray(right)] += 1e-3 * rng.standard_normal((zr.shape[0], len(right)))
        kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
        klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
        krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
        left = pair.planted_hyperedges[0][0]
        found = exhaustive_balanced_pairs(
            kll,
            klr,
            krr,
            left,
            right,
            residual_tolerance=1e-10,
            max_group_size=4,
        )
        self.assertEqual(found, [])

    def test_unequal_split_merge_recovers_only_full_hyperedges(self) -> None:
        pair = unequal_split_merge(4, n_mean=257, n_eval=2048, seed=53)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
        klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
        krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
        for left, right in pair.planted_hyperedges:
            self.assertNotEqual(len(left), len(right))
            found = exhaustive_balanced_pairs(
                kll, klr, krr, left, right,
                residual_tolerance=1e-10,
                max_group_size=2,
            )
            self.assertEqual([(item.left_ids, item.right_ids) for item in found], [(left, right)])
        wrong = exhaustive_balanced_pairs(
            kll, klr, krr,
            pair.planted_hyperedges[0][0],
            pair.planted_hyperedges[1][1],
            residual_tolerance=1e-10,
            max_group_size=2,
        )
        self.assertEqual(wrong, [])

    def test_exhaustive_search_exposes_complete_diagnostics(self) -> None:
        pair = unequal_split_merge(4, n_mean=257, n_eval=2048, seed=59)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        kll = contribution_kernel(pair.d_left, zl, pair.d_left, zl)
        klr = contribution_kernel(pair.d_left, zl, pair.d_right, zr)
        krr = contribution_kernel(pair.d_right, zr, pair.d_right, zr)
        left, right = pair.planted_hyperedges[0]
        result = exhaustive_balanced_search(
            kll, klr, krr, left, right,
            residual_tolerance=1e-10,
            tie_tolerance=1e-12,
            max_group_size=2,
        )
        self.assertEqual(result.evaluated_count, 3)
        self.assertEqual(len(result.all_candidates), 3)
        self.assertEqual(len(result.passing_candidates), 1)
        self.assertEqual(len(result.support_minimal_candidates), 1)
        self.assertEqual(len(result.tie_set), 1)
        self.assertGreater(result.solver_gap, 1e-12)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_partial_overlap_requires_hypergraph_output(self) -> None:
        pair = partial_overlap_seeded(
            n_mean=257,
            n_eval=2048,
            structural_seed_a=61,
            structural_seed_b=62,
            mean_sample_seed=63,
            eval_sample_seed=64,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        result = exhaustive_balanced_search(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0, 1, 2),
            (0, 1, 2),
            residual_tolerance=1e-10,
            tie_tolerance=1e-12,
            max_group_size=2,
        )
        predicted = {(item.left_ids, item.right_ids) for item in result.support_minimal_candidates}
        self.assertEqual(predicted, set(pair.planted_hyperedges))
        self.assertEqual(result.evaluated_count, 36)
        self.assertEqual(len(result.tie_set), 2)
        forced = forced_partition_projection(result.support_minimal_candidates)
        self.assertEqual(len(forced), 1)

    def test_code_correlation_proposal_is_rejected_by_contribution(self) -> None:
        pair = cooccurrence_confounding_seeded(
            n_mean=257,
            n_eval=2048,
            structural_seed_a=71,
            structural_seed_b=72,
            mean_sample_seed=73,
            eval_sample_seed=74,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        self.assertAlmostEqual(absolute_code_correlation(zl[:, 0], zr[:, 0]), 1.0, places=12)
        result = exhaustive_balanced_search(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0,),
            (0,),
            residual_tolerance=0.1,
            tie_tolerance=1e-12,
            max_group_size=1,
        )
        self.assertEqual(result.evaluated_count, 1)
        self.assertEqual(result.passing_candidates, ())
        self.assertGreaterEqual(result.best_residual, 0.8)

    def test_adjusted_rand_index_partition_metric(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(adjusted_rand_index(truth, np.asarray([4, 4, 7, 7, 9, 9])), 1.0)
        self.assertLess(adjusted_rand_index(truth, np.asarray([0, 1, 0, 1, 0, 1])), 0.0)

    def test_competing_covers_are_preserved_as_ambiguity(self) -> None:
        pair = competing_covers_seeded(
            257, 2048,
            structural_seed_a=81,
            structural_seed_b=82,
            mean_sample_seed=83,
            eval_sample_seed=84,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        search = exhaustive_balanced_search(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0, 1), (0, 1),
            residual_tolerance=1e-10,
            tie_tolerance=1e-12,
            max_group_size=1,
        )
        covers = enumerate_maximum_exact_covers(search.support_minimal_candidates, (0, 1), (0, 1))
        self.assertEqual(len(search.support_minimal_candidates), 4)
        self.assertEqual(covers.maximum_cardinality, 2)
        self.assertEqual(len(covers.maximum_covers), 2)

    def test_whole_dictionary_balance_is_not_a_local_match(self) -> None:
        pair = whole_dictionary_only_seeded(
            257, 2048,
            structural_seed_a=91,
            structural_seed_b=92,
            mean_sample_seed=93,
            eval_sample_seed=94,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        search = exhaustive_balanced_search(
            contribution_kernel(pair.d_left, zl, pair.d_left, zl),
            contribution_kernel(pair.d_left, zl, pair.d_right, zr),
            contribution_kernel(pair.d_right, zr, pair.d_right, zr),
            (0, 1, 2), (0, 1, 2),
            residual_tolerance=1e-10,
            tie_tolerance=1e-12,
            max_group_size=3,
        )
        observed = {(item.left_ids, item.right_ids) for item in search.support_minimal_candidates}
        self.assertEqual(observed, {pair.planted_hyperedges[0]})

    def test_cancellation_diagnostic_flags_risk_not_clean_control(self) -> None:
        pair = cancellation_seeded(
            257, 2048,
            structural_seed_a=101,
            structural_seed_b=102,
            mean_sample_seed=103,
            eval_sample_seed=104,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        risk_left, risk_right = pair.planted_hyperedges[0]
        clean_left, clean_right = pair.planted_hyperedges[1]
        risk_a = cancellation_diagnostics(pair.d_left, zl, np.asarray(risk_left))
        risk_b = cancellation_diagnostics(pair.d_right, zr, np.asarray(risk_right))
        clean_a = cancellation_diagnostics(pair.d_left, zl, np.asarray(clean_left))
        clean_b = cancellation_diagnostics(pair.d_right, zr, np.asarray(clean_right))
        self.assertGreater(min(risk_a.cancellation_energy_ratio, risk_b.cancellation_energy_ratio), 100.0)
        self.assertGreater(min(risk_a.max_leave_one_out_energy_ratio, risk_b.max_leave_one_out_energy_ratio), 50.0)
        self.assertAlmostEqual(clean_a.cancellation_energy_ratio, 1.0, places=12)
        self.assertAlmostEqual(clean_b.cancellation_energy_ratio, 1.0, places=12)

    def test_rare_occupancy_uses_document_ess_and_cluster_bootstrap(self) -> None:
        pair = rare_occupancy_seeded(
            1024, 2048,
            tokens_per_document=64,
            active_document_count=2,
            structural_seed_a=111,
            structural_seed_b=112,
            mean_sample_seed=113,
            eval_sample_seed=114,
        )
        documents = np.asarray(pair.eval_document_ids)
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        risk_left, risk_right = pair.planted_hyperedges[0]
        clean_left, _ = pair.planted_hyperedges[1]
        risk_occ = occupancy_effective_sample_size(pair.d_left, pair.z_left_eval, np.asarray(risk_left), documents)
        clean_occ = occupancy_effective_sample_size(pair.d_left, pair.z_left_eval, np.asarray(clean_left), documents)
        bootstrap = document_bootstrap_bcc(
            pair.d_left, zl, np.asarray(risk_left),
            pair.d_right, zr, np.asarray(risk_right),
            documents, replicates=500, seed=115,
        )
        self.assertEqual(risk_occ.active_token_count, 4)
        self.assertEqual(risk_occ.active_document_count, 2)
        self.assertLessEqual(risk_occ.document_energy_kish_ess, 2.0)
        self.assertGreater(clean_occ.document_energy_kish_ess, 20.0)
        self.assertGreater(bootstrap.inactive_fraction, 0.05)
        self.assertGreater(bootstrap.ci_width, 0.05)


if __name__ == "__main__":
    unittest.main()
