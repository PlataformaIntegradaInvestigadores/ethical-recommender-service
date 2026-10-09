"""Group scoring, constrained re-ranking, and ranking metrics on small synthetic fixtures."""

import math
import unittest

import numpy as np

from application.services import ranking_metrics as rm
from application.services.fair_reranker import (
    RerankConfig, RerankConfigError, institution_novelty, normalize_relevance, required_protected_counts, rerank,
)
from application.services.group_scoring import (
    GroupScoringError, aggregate_scores, retrieve_top_n, score_all_candidates,
)
from domain.entities.ranking import Candidate
from domain.entities.research_group import UNKNOWN_INSTITUTION


def unit(rows):
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class GroupScoringTests(unittest.TestCase):
    def test_hybrid_aggregation_combines_mean_and_max(self):
        sims = np.array([[0.9, 0.2], [0.1, 0.4]])
        np.testing.assert_allclose(aggregate_scores(sims, alpha=1.0), [0.5, 0.3])
        np.testing.assert_allclose(aggregate_scores(sims, alpha=0.0), [0.9, 0.4])
        np.testing.assert_allclose(aggregate_scores(sims, alpha=0.5), [0.7, 0.35])
        np.testing.assert_allclose(aggregate_scores(sims, alpha=1.0, weights=[3, 1]), [0.7, 0.25])

    def test_scores_use_individual_member_vectors(self):
        emb = unit([[1, 0], [0, 1], [1, 1], [1, 0.01], [-1, 0]])
        scores = score_all_candidates(emb, [0, 1], alpha=0.5)
        self.assertGreater(scores[3], scores[2])  # max term rewards a close match to one member
        top = retrieve_top_n(scores, excluded_rows=[0, 1], top_n=2)
        self.assertEqual([row for row, _ in top], [3, 2])

    def test_invalid_inputs(self):
        with self.assertRaises(GroupScoringError):
            aggregate_scores(np.ones((2, 3)), alpha=1.5)
        with self.assertRaises(GroupScoringError):
            aggregate_scores(np.ones((2, 3)), weights=[1, -1])
        with self.assertRaises(GroupScoringError):
            score_all_candidates(np.ones((3, 2), dtype=np.float32), [0])
        self.assertEqual(retrieve_top_n(np.array([1.0, 2.0]), [0, 1]), [])


def pool():
    # relevance descending; institution U dominates the head; p* are low-visibility
    spec = [("c1", 1.00, "U", False), ("c2", 0.95, "U", False), ("c3", 0.90, "U", False),
            ("c4", 0.85, "U", False), ("p1", 0.80, "V", True), ("c5", 0.75, "W", False),
            ("p2", 0.70, UNKNOWN_INSTITUTION, True), ("c6", 0.60, "U", False), ("p3", 0.05, "X", True),
            ("c7", 0.00, "Y", False)]
    return [Candidate(*row) for row in spec]


class RerankTests(unittest.TestCase):
    def test_pure_relevance_without_constraints_keeps_order(self):
        config = RerankConfig(lambda_=1.0, max_per_institution=None, min_protected_share=0.0, fairness_mode="none")
        self.assertEqual(rerank(pool(), 10, config).author_ids, [c.author_id for c in pool()])

    def test_institution_cap_and_query_penalty(self):
        config = RerankConfig(lambda_=1.0, max_per_institution=2, min_protected_share=0.0)
        ids = rerank(pool(), 6, config).author_ids
        self.assertEqual(ids, ["c1", "c2", "p1", "c5", "p2", "p3"])
        novelty = institution_novelty("U", __import__("collections").Counter({"U": 1}), frozenset({"U"}))
        self.assertAlmostEqual(novelty, 1 / 3)
        self.assertEqual(institution_novelty(UNKNOWN_INSTITUTION, __import__("collections").Counter(), frozenset()), 0.0)

    def test_prefix_quota_respects_relevance_floor(self):
        config = RerankConfig(lambda_=1.0, max_per_institution=None, min_protected_share=0.5, relevance_floor=0.5)
        result = rerank(pool(), 6, config)
        ids = result.author_ids
        self.assertEqual(ids[:2], ["c1", "p1"])
        self.assertNotIn("p3", ids)  # below the relevance floor: never promoted
        protected_prefix = [sum(c in {"p1", "p2", "p3"} for c in ids[:i]) for i in range(1, 7)]
        required = result.required_protected
        self.assertTrue(all(have >= need for have, need in zip(protected_prefix[:4], required[:4])))
        self.assertGreater(result.quota_shortfalls, 0)  # only two eligible protected candidates exist

    def test_cap_relaxation_when_only_capped_remain(self):
        candidates = [Candidate(f"u{i}", 1 - i / 10, "U", False) for i in range(4)]
        result = rerank(candidates, 4, RerankConfig(max_per_institution=2, min_protected_share=0.0))
        self.assertEqual(result.cap_relaxations, 2)
        self.assertEqual(len(result.author_ids), 4)

    def test_probabilistic_mode_is_seed_deterministic(self):
        config = RerankConfig(temperature=0.2, seed=11)
        first = rerank(pool(), 8, config).author_ids
        self.assertEqual(first, rerank(pool(), 8, config).author_ids)
        self.assertEqual(len(set(first)), 8)

    def test_required_counts(self):
        self.assertEqual(required_protected_counts(5, 0.4, "prefix"), [0, 0, 1, 1, 2])
        fair = required_protected_counts(20, 0.3, "fair", 0.1)
        self.assertEqual(fair[0], 0)
        self.assertTrue(all(b >= a for a, b in zip(fair, fair[1:])))
        self.assertLessEqual(fair[-1], 6)
        self.assertEqual(required_protected_counts(3, 0.5, "none"), [0, 0, 0])
        with self.assertRaises(RerankConfigError):
            RerankConfig(lambda_=2)
        self.assertEqual(normalize_relevance([2.0, 2.0]), [1.0, 1.0])


class MetricTests(unittest.TestCase):
    def test_accuracy_metrics(self):
        ranked = ["a", "b", "t", "c"]
        self.assertAlmostEqual(rm.precision_at_k(ranked, {"t"}, 4), 0.25)
        self.assertEqual(rm.recall_at_k(ranked, {"t"}, 2), 0.0)
        self.assertEqual(rm.hit_rate_at_k(ranked, {"t"}, 3), 1.0)
        self.assertAlmostEqual(rm.ndcg_at_k(ranked, {"t"}, 4), 1 / math.log2(4))
        self.assertEqual(rm.ndcg_at_k(["t"], {"t"}, 10), 1.0)

    def test_diversity_metrics_ignore_unknown_and_tokens(self):
        insts = ["A", "A", "B", UNKNOWN_INSTITUTION]
        self.assertAlmostEqual(rm.gini_simpson(insts), 1 - (4 / 9 + 1 / 9))
        self.assertAlmostEqual(rm.anti_tokenism_gini_simpson(insts, m=2), 1 - (4 / 9 + 1 / 9))
        self.assertAlmostEqual(rm.anti_tokenism_gini_simpson(["A", "B", "C"], m=2), 0.0)
        self.assertEqual(rm.unique_institutions(insts), 2)
        self.assertAlmostEqual(rm.institution_coverage(insts, ["A", "B", "C", "D"]), 0.5)

    def test_fairness_metrics(self):
        protected = {"p": True, "q": True}
        ranked = ["p", "a", "q", "b"]
        self.assertEqual(rm.protected_share(ranked, protected, 4), 0.5)
        exp_p, exp_np = rm.group_exposure(ranked, protected, 4)
        self.assertGreater(exp_p, exp_np)
        # prefix 1: KL(1 || 0.5) = ln 2 with weight 1; prefix 2: KL(0.5 || 0.5) = 0
        self.assertAlmostEqual(rm.ndkl(["p", "a"], protected, 2, 0.5), math.log(2) / (1 + 1 / math.log2(3)), places=4)
        self.assertGreater(rm.ndkl(["a", "b"], protected, 2, 0.5), rm.ndkl(["p", "a"], protected, 2, 0.5))
        self.assertAlmostEqual(rm.relevance_retained(["b"], ["a"], {"a": 2.0, "b": 1.0}, 1), 0.5)
        self.assertGreater(rm.exposure_share(ranked, protected, 4), 0.5)


if __name__ == "__main__":
    unittest.main()
