from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import numpy as np

from ccad.matching import (
    freeze_discovery_prediction,
    evaluate_frozen_hyperedges,
    full_universe_balanced_search,
    search_candidate_family,
    verify_frozen_discovery_prediction,
)
from ccad.metrics import center_codes, contribution_kernel
from ccad.proposal import (
    absolute_code_correlation_affinity,
    decoder_cosine_affinity,
    degree_matched_random_proposal,
    li15_spectral_proposal,
    proposal_candidate_family,
    singleton_contribution_affinity,
    symmetric_topk_proposal,
    validate_independent_split_seeds,
)
from ccad.synthetic import (
    cooccurrence_confounding_seeded,
    local_block_rotations_seeded,
    partial_overlap_seeded,
    unequal_split_merge_seeded,
)


def kernels(pair):
    zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
    zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
    return (
        contribution_kernel(pair.d_left, zl, pair.d_left, zl),
        contribution_kernel(pair.d_left, zl, pair.d_right, zr),
        contribution_kernel(pair.d_right, zr, pair.d_right, zr),
    )


class UnknownSupportProposalTests(unittest.TestCase):
    def test_public_apis_have_no_truth_or_planted_support_parameter(self) -> None:
        for function in (full_universe_balanced_search, symmetric_topk_proposal):
            names = set(inspect.signature(function).parameters)
            self.assertTrue(names.isdisjoint({"truth", "labels", "planted_support", "planted_hyperedges"}))

    def test_split_seed_contract_fails_closed_on_missing_or_reused_stream(self) -> None:
        valid = {
            "structural_seed_a": 1,
            "structural_seed_b": 2,
            "mean_sample_seed": 3,
            "discovery_sample_seed": 4,
            "eval_sample_seed": 5,
        }
        validate_independent_split_seeds(valid)
        with self.assertRaises(ValueError):
            validate_independent_split_seeds({key: value for key, value in valid.items() if key != "discovery_sample_seed"})
        reused = dict(valid, eval_sample_seed=valid["discovery_sample_seed"])
        with self.assertRaises(ValueError):
            validate_independent_split_seeds(reused)

    def test_full_universe_search_budget_refuses_before_enumeration(self) -> None:
        pair = unequal_split_merge_seeded(
            4, 257, 1024,
            structural_seed_a=11, structural_seed_b=12,
            mean_sample_seed=13, eval_sample_seed=14,
        )
        result = full_universe_balanced_search(
            *kernels(pair), residual_tolerance=1e-10, tie_tolerance=1e-12,
            max_group_size=2, candidate_budget=440,
        )
        self.assertEqual(result.status, "BUDGET_REFUSAL")
        self.assertEqual(result.planned_candidate_count, 441)
        self.assertIsNone(result.search)

    def test_full_universe_ledger_counts_inactive_attempts(self) -> None:
        zeros = np.zeros((1, 1))
        result = full_universe_balanced_search(
            zeros, zeros, zeros, residual_tolerance=0.0, tie_tolerance=0.0,
            max_group_size=1, candidate_budget=1,
        )
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.planned_candidate_count, 1)
        self.assertEqual(result.search.evaluated_count, 1)
        self.assertEqual(result.search.all_candidates, ())

    def test_full_universe_f02_f03_f04_recovery_and_f06_refusal(self) -> None:
        cases = [
            (local_block_rotations_seeded(
                (2, 2), 257, 2048,
                structural_seed_a=21, structural_seed_b=22,
                mean_sample_seed=23, eval_sample_seed=24,
            ), 4, 1_000_000),
            (unequal_split_merge_seeded(
                4, 257, 2048,
                structural_seed_a=31, structural_seed_b=32,
                mean_sample_seed=33, eval_sample_seed=34,
            ), 2, 10_000),
            (partial_overlap_seeded(
                257, 2048,
                structural_seed_a=41, structural_seed_b=42,
                mean_sample_seed=43, eval_sample_seed=44,
            ), 2, 100),
        ]
        for pair, max_group_size, budget in cases:
            with self.subTest(family=pair.family_id):
                result = full_universe_balanced_search(
                    *kernels(pair), residual_tolerance=1e-10, tie_tolerance=1e-12,
                    max_group_size=max_group_size, candidate_budget=budget,
                )
                self.assertEqual(result.status, "OK")
                observed = {(item.left_ids, item.right_ids) for item in result.search.support_minimal_candidates}
                self.assertEqual(observed, set(pair.planted_hyperedges))
        negative = cooccurrence_confounding_seeded(
            257, 2048,
            structural_seed_a=51, structural_seed_b=52,
            mean_sample_seed=53, eval_sample_seed=54,
        )
        refused = full_universe_balanced_search(
            *kernels(negative), residual_tolerance=0.1, tie_tolerance=1e-12,
            max_group_size=1, candidate_budget=1,
        )
        self.assertEqual(refused.status, "OK")
        self.assertEqual(refused.search.support_minimal_candidates, ())

    def test_proposal_is_deterministic_and_independent_of_held_out_eval_seed(self) -> None:
        common = dict(
            block_count=4, n_mean=257, n_eval=1024,
            structural_seed_a=61, structural_seed_b=62, mean_sample_seed=63,
        )
        discovery_pair = unequal_split_merge_seeded(**common, eval_sample_seed=64)
        eval_pair_a = unequal_split_merge_seeded(**common, eval_sample_seed=65)
        eval_pair_b = unequal_split_merge_seeded(**common, eval_sample_seed=66)
        discovery_scores = singleton_contribution_affinity(*kernels(discovery_pair))
        proposal_a = symmetric_topk_proposal(
            discovery_scores, top_k=2, score_source="CONTRIB-KNN", max_neighborhood_atoms=6,
        )
        proposal_b = symmetric_topk_proposal(
            discovery_scores.copy(), top_k=2, score_source="CONTRIB-KNN", max_neighborhood_atoms=6,
        )
        self.assertEqual(proposal_a, proposal_b)
        self.assertFalse(np.array_equal(eval_pair_a.z_left_eval, eval_pair_b.z_left_eval))
        self.assertTrue(proposal_a.edges)

    def test_neighborhood_overflow_is_explicit_not_truncated(self) -> None:
        proposal = symmetric_topk_proposal(
            np.ones((3, 3)), top_k=3, score_source="TIE_FIXTURE", max_neighborhood_atoms=5,
        )
        self.assertEqual(proposal.edges, tuple((left, right) for left in range(3) for right in range(3)))
        self.assertTrue(all(item.status == "BUDGET_REFUSAL" for item in proposal.neighborhoods))
        self.assertTrue(all(item.left_ids == (0, 1, 2) and item.right_ids == (0, 1, 2) for item in proposal.neighborhoods))

    def test_baseline_affinities_and_random_lane_preserve_budget_surface(self) -> None:
        pair = unequal_split_merge_seeded(
            4, 257, 1024,
            structural_seed_a=71, structural_seed_b=72,
            mean_sample_seed=73, eval_sample_seed=74,
        )
        zl, _ = center_codes(pair.z_left_mean, pair.z_left_eval)
        zr, _ = center_codes(pair.z_right_mean, pair.z_right_eval)
        score_lanes = {
            "DECODER-KNN": decoder_cosine_affinity(pair.d_left, pair.d_right),
            "CODE-KNN": absolute_code_correlation_affinity(zl, zr),
        }
        for source, scores in score_lanes.items():
            proposal = symmetric_topk_proposal(scores, top_k=2, score_source=source, max_neighborhood_atoms=8)
            self.assertTrue(proposal.edges)
            self.assertTrue(proposal_candidate_family(proposal, max_group_size=2))
        reference = symmetric_topk_proposal(
            score_lanes["CODE-KNN"], top_k=2, score_source="CODE-KNN", max_neighborhood_atoms=8,
        )
        randomized = degree_matched_random_proposal(reference, seed=75, max_neighborhood_atoms=8)
        self.assertEqual(randomized.left_degrees, reference.left_degrees)
        self.assertEqual(randomized.right_degrees, reference.right_degrees)
        self.assertEqual(len(randomized.edges), len(reference.edges))

    def test_li15_spectral_recovers_two_mixed_correlation_components(self) -> None:
        rng = np.random.default_rng(81)
        signals = rng.standard_normal((2048, 2))
        result = li15_spectral_proposal(
            signals, signals,
            correlation_threshold=0.2,
            max_clusters=3,
            kmeans_seed=82,
            max_neighborhood_atoms=4,
        )
        self.assertEqual(result.cluster_count, 2)
        self.assertEqual(result.mixed_cluster_count, 2)
        self.assertEqual(set(result.proposal.edges), {(0, 0), (1, 1)})
        self.assertTrue(all(item.status == "OK" for item in result.proposal.neighborhoods))

    def test_arbitrary_candidate_solver_is_truth_blind_and_recovers_f03(self) -> None:
        names = set(inspect.signature(search_candidate_family).parameters)
        self.assertTrue(names.isdisjoint({"truth", "labels", "eval", "planted_hyperedges"}))
        pair = unequal_split_merge_seeded(
            4, 257, 1024,
            structural_seed_a=91, structural_seed_b=92,
            mean_sample_seed=93, eval_sample_seed=94,
        )
        k_ll, k_lr, k_rr = kernels(pair)
        proposal = symmetric_topk_proposal(
            singleton_contribution_affinity(k_ll, k_lr, k_rr),
            top_k=4, score_source="CONTRIB-KNN", max_neighborhood_atoms=12,
        )
        family = proposal_candidate_family(proposal, max_group_size=4)
        result = search_candidate_family(
            k_ll, k_lr, k_rr, family,
            residual_tolerance=1e-10, tie_tolerance=1e-12, candidate_budget=7462,
        )
        self.assertEqual(result.status, "OK")
        observed = {(item.left_ids, item.right_ids) for item in result.support_minimal_candidates}
        self.assertEqual(observed, set(pair.planted_hyperedges))

    def test_candidate_solver_refuses_over_budget_before_scoring(self) -> None:
        family = (((0,), (0,)), ((0,), (1,)))
        result = search_candidate_family(
            np.eye(1), np.zeros((1, 2)), np.eye(2), family,
            residual_tolerance=0.0, tie_tolerance=0.0, candidate_budget=1,
        )
        self.assertEqual(result.status, "BUDGET_REFUSAL")
        self.assertEqual(result.evaluated_count, 0)
        self.assertEqual(result.refusal_reason, "CANDIDATE_FAMILY_EXCEEDS_BUDGET")

    def test_prediction_freeze_is_deterministic_and_tamper_evident(self) -> None:
        search = search_candidate_family(
            np.eye(1), np.eye(1), np.eye(1), (((0,), (0,)),),
            residual_tolerance=1e-10, tie_tolerance=1e-12, candidate_budget=1,
        )
        first = freeze_discovery_prediction(
            search, proposal_source="CONTRIB-KNN", proposal_hash="A" * 64,
            discovery_fingerprint="B" * 64,
        )
        second = freeze_discovery_prediction(
            search, proposal_source="CONTRIB-KNN", proposal_hash="A" * 64,
            discovery_fingerprint="B" * 64,
        )
        self.assertEqual(first, second)
        self.assertTrue(verify_frozen_discovery_prediction(first))
        self.assertFalse(verify_frozen_discovery_prediction(replace(first, discovery_fingerprint="C" * 64)))
        with self.assertRaises(ValueError):
            evaluate_frozen_hyperedges(
                replace(first, discovery_fingerprint="C" * 64),
                np.eye(1), np.eye(1), np.eye(1), (((0,), (0,)),),
            )
        held_out = evaluate_frozen_hyperedges(
            first, np.eye(1), np.eye(1), np.eye(1), (((0,), (0,)),),
        )
        self.assertEqual((held_out.proposal_recall, held_out.precision, held_out.recall, held_out.f1), (1.0, 1.0, 1.0, 1.0))
        self.assertIsNone(held_out.failure_attribution)


if __name__ == "__main__":
    unittest.main()
