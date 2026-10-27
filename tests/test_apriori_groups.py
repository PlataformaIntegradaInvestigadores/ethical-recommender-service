"""Apriori group mining: correctness against brute force, pruning, flags, and guards."""

import random
import unittest
from itertools import combinations

from application.services.apriori_groups import (
    AprioriConfig,
    AprioriConfigError,
    build_tidsets,
    generate_candidates,
    mine_frequent_itemsets,
    select_groups,
)
from application.services.evaluation_split import EvaluationSplitError, leave_one_out_cases, sample_cases

TRANSACTIONS = {
    "p1": ["a", "b", "c"],
    "p2": ["a", "b", "c"],
    "p3": ["a", "b", "c", "d"],
    "p4": ["a", "b"],
    "p5": ["d", "e"],
    "p6": ["d", "e"],
    "p7": ["d", "e"],
    "big": ["a", "b", "c", "d", "e", "x1", "x2"],
}


def brute_force(transactions, min_support, max_k, max_size):
    kept = [set(items) for items in transactions.values() if len(set(items)) <= max_size]
    items = sorted({i for t in kept for i in t})
    found = {}
    for k in range(1, max_k + 1):
        for combo in combinations(items, k):
            support = sum(1 for t in kept if set(combo) <= t)
            if support >= min_support:
                found[combo] = support
    return found


class AprioriTests(unittest.TestCase):
    def test_matches_brute_force_on_random_data(self):
        rng = random.Random(7)
        authors = [f"a{i}" for i in range(9)]
        transactions = {f"t{i}": rng.sample(authors, rng.randint(1, 5)) for i in range(60)}
        result = mine_frequent_itemsets(transactions, AprioriConfig(min_support=3, max_k=4, max_authors_per_article=4))
        mined = {itemset: len(tids) for level in result.levels.values() for itemset, tids in level.items()}
        self.assertEqual(mined, brute_force(transactions, 3, 4, 4))

    def test_hyperauthored_articles_are_dropped_and_counted(self):
        result = mine_frequent_itemsets(TRANSACTIONS, AprioriConfig(min_support=3, max_k=5, max_authors_per_article=5))
        self.assertEqual(result.transactions_dropped_hyperauthored, 1)
        self.assertEqual(result.levels[3][("a", "b", "c")], frozenset({"p1", "p2", "p3"}))
        self.assertNotIn(("x1",), result.levels[1])
        relaxed = mine_frequent_itemsets(TRANSACTIONS, AprioriConfig(min_support=3, max_authors_per_article=None))
        self.assertEqual(relaxed.transactions_dropped_hyperauthored, 0)
        self.assertEqual(len(relaxed.levels[3][("a", "b", "c")]), 4)

    def test_allowed_items_restrict_population(self):
        result = mine_frequent_itemsets(TRANSACTIONS, AprioriConfig(min_support=3), allowed_items=["a", "b"])
        self.assertEqual(set(result.levels[1]), {("a",), ("b",)})
        self.assertEqual(set(result.levels[2]), {("a", "b")})

    def test_candidate_generation_joins_prefix_and_prunes_infrequent_subsets(self):
        candidates, pruned = generate_candidates([("a", "b"), ("a", "c"), ("b", "c"), ("a", "d")])
        self.assertEqual(candidates, [("a", "b", "c")])
        self.assertEqual(pruned, 2)  # (a,b,d) lacks (b,d); (a,c,d) lacks (c,d)

    def test_maximal_and_closed_flags(self):
        result = mine_frequent_itemsets(TRANSACTIONS, AprioriConfig(min_support=3, max_authors_per_article=5))
        flags = {g.members: (g.maximal, g.closed, g.support) for g in result.groups}
        self.assertEqual(flags[("a", "b", "c")], (True, True, 3))
        self.assertEqual(flags[("a", "b")], (False, True, 4))
        self.assertEqual(flags[("a", "c")], (False, False, 3))
        self.assertEqual(flags[("d", "e")], (True, True, 3))
        self.assertEqual({g.members for g in select_groups(result.groups, "maximal")}, {("a", "b", "c"), ("d", "e")})
        self.assertTrue(all(g.size >= 2 for g in result.groups))

    def test_max_k_truncation_is_reported(self):
        capped = mine_frequent_itemsets(TRANSACTIONS, AprioriConfig(min_support=3, max_k=2, max_authors_per_article=5))
        self.assertEqual(max(capped.levels), 2)
        self.assertEqual(capped.truncated_at_k, 2)
        unbounded = mine_frequent_itemsets(
            TRANSACTIONS, AprioriConfig(min_support=3, max_k=None, max_authors_per_article=5)
        )
        self.assertIsNone(unbounded.truncated_at_k)

    def test_invalid_configuration(self):
        with self.assertRaises(AprioriConfigError):
            AprioriConfig(min_support=0)
        with self.assertRaises(AprioriConfigError):
            AprioriConfig(max_k=2, min_group_size=3)
        with self.assertRaises(AprioriConfigError):
            select_groups([], "everything")

    def test_build_tidsets_inverts_transactions(self):
        tidsets, dropped = build_tidsets({"t1": ["a", "a", "b"], "t2": ["b"]}, max_items_per_transaction=None)
        self.assertEqual(tidsets, {"a": frozenset({"t1"}), "b": frozenset({"t1", "t2"})})
        self.assertEqual(dropped, 0)


class LeaveOneOutTests(unittest.TestCase):
    GROUPS = [("G1", ("c", "a", "b")), ("G2", ("d", "e")), ("G3", ("f", "g", "h", "i"))]

    def test_one_target_per_group_is_seeded_and_disjoint(self):
        cases = leave_one_out_cases(self.GROUPS, seed=3)
        self.assertEqual([c.group_id for c in cases], ["G1", "G3"])
        for case in cases:
            self.assertNotIn(case.target, case.query_members)
        self.assertEqual(cases, leave_one_out_cases(self.GROUPS, seed=3))

    def test_all_members_mode_and_sampling(self):
        cases = leave_one_out_cases(self.GROUPS, all_members=True)
        self.assertEqual(len(cases), 7)
        self.assertEqual(sample_cases(cases, 3, seed=1), sample_cases(cases, 3, seed=1))
        self.assertEqual(len(sample_cases(cases, 99)), 7)
        with self.assertRaises(EvaluationSplitError):
            leave_one_out_cases(self.GROUPS, min_group_size=1)


if __name__ == "__main__":
    unittest.main()
