"""Level-wise Apriori mining of frequent co-author groups with vertical (tidset) counting.

Transactions are articles and items are author identifiers. An itemset X is a
frequent group when the authors in X co-signed at least ``min_support`` articles.

Search-space control
--------------------
A naive search enumerates all 2^n author subsets (n = 7,621 gives ~10^2294).
Apriori instead grows itemsets one level at a time and relies on
anti-monotonicity: supp(X ∪ {a}) <= supp(X), so a k-itemset can only be
frequent if *every* (k-1)-subset is frequent. Candidates C_k are built only by
joining frequent (k-1)-itemsets that share their first k-2 items, and any
candidate with an infrequent (k-1)-subset is pruned before counting.

Support is counted vertically (Eclat-style): each itemset keeps its tidset, the
set of article ids it occurs in, and tid(P ∪ {a, b}) = tid(P ∪ {a}) ∩ tid(P ∪ {b}).
No pass over the transactions is needed after level 1.

Level 2 is the one place where the prefix join is still quadratic (every pair
of frequent authors shares the empty prefix). Pairs that never co-occur have
support 0, so C_2 is restricted to co-occurring pairs; this is exact, not a
heuristic.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final

from domain.entities.research_group import FrequentGroup

DEFAULT_MIN_SUPPORT: Final[int] = 3
DEFAULT_MAX_K: Final[int] = 5
DEFAULT_MAX_AUTHORS_PER_ARTICLE: Final[int] = 25

Itemset = tuple[str, ...]
TidSet = frozenset[str]


class AprioriConfigError(ValueError):
    """The mining parameters are inconsistent."""


@dataclass(frozen=True, slots=True)
class AprioriConfig:
    """Mining parameters.

    ``max_authors_per_article`` guards against hyper-authored articles, whose
    author lists would otherwise create combinatorially many spurious groups.
    ``None`` disables the guard. ``max_k=None`` mines until a level is empty
    (Apriori always terminates because L_k shrinks to the empty set).
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    max_k: int | None = DEFAULT_MAX_K
    max_authors_per_article: int | None = DEFAULT_MAX_AUTHORS_PER_ARTICLE
    min_group_size: int = 2

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise AprioriConfigError("min_support must be >= 1")
        if self.max_k is not None and self.max_k < 1:
            raise AprioriConfigError("max_k must be >= 1 or None")
        if self.min_group_size < 1 or (self.max_k is not None and self.min_group_size > self.max_k):
            raise AprioriConfigError("min_group_size must be in [1, max_k]")

    @property
    def level_limit(self) -> int:
        return self.max_k if self.max_k is not None else sys.maxsize
        if self.max_authors_per_article is not None and self.max_authors_per_article < 1:
            raise AprioriConfigError("max_authors_per_article must be >= 1 or None")


@dataclass(slots=True)
class LevelStats:
    """Counters for one Apriori level."""

    k: int
    candidates: int = 0
    pruned_by_subset: int = 0
    frequent: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class AprioriResult:
    """Frequent itemsets per level plus diagnostics."""

    levels: dict[int, dict[Itemset, TidSet]]
    stats: list[LevelStats]
    transactions_total: int
    transactions_dropped_hyperauthored: int
    transactions_used: int
    runtime_seconds: float
    truncated_at_k: int | None = None
    groups: list[FrequentGroup] = field(default_factory=list)


def build_tidsets(
    transactions: Mapping[str, Iterable[str]],
    allowed_items: Iterable[str] | None = None,
    max_items_per_transaction: int | None = DEFAULT_MAX_AUTHORS_PER_ARTICLE,
) -> tuple[dict[str, TidSet], int]:
    """Invert transactions (article -> authors) into tidsets (author -> articles).

    The hyper-authorship guard uses the article's *full* author count, before
    restricting to ``allowed_items``, because that is what makes an article
    hyper-authored. Returns the tidsets and the number of dropped transactions.
    """
    allowed = None if allowed_items is None else set(allowed_items)
    tidsets: dict[str, set[str]] = defaultdict(set)
    dropped = 0
    for tid, items in transactions.items():
        unique_items = set(items)
        if max_items_per_transaction is not None and len(unique_items) > max_items_per_transaction:
            dropped += 1
            continue
        for item in unique_items:
            if allowed is None or item in allowed:
                tidsets[item].add(tid)
    return {item: frozenset(tids) for item, tids in tidsets.items()}, dropped


def generate_candidates(frequent_prev: Iterable[Itemset]) -> tuple[list[Itemset], int]:
    """Apriori-gen: prefix join of sorted (k-1)-itemsets followed by subset pruning.

    Two itemsets join when they share their first k-2 items; the joined
    candidate is discarded if any of its (k-1)-subsets is not frequent.
    Returns the surviving candidates and the number pruned.
    """
    previous = sorted(set(frequent_prev))
    previous_set = set(previous)
    by_prefix: dict[Itemset, list[str]] = defaultdict(list)
    for itemset in previous:
        by_prefix[itemset[:-1]].append(itemset[-1])
    candidates: list[Itemset] = []
    pruned = 0
    for prefix, tails in by_prefix.items():
        for first, second in combinations(tails, 2):
            candidate = (*prefix, first, second)
            if all(subset in previous_set for subset in combinations(candidate, len(candidate) - 1)):
                candidates.append(candidate)
            else:
                pruned += 1
    return candidates, pruned


def _level_two(
    level_one: dict[Itemset, TidSet],
    transactions_items: Iterable[set[str]],
    min_support: int,
) -> tuple[dict[Itemset, TidSet], int]:
    """Exact C_2 restricted to co-occurring frequent pairs, counted by intersection."""
    frequent_items = {itemset[0] for itemset in level_one}
    neighbours: dict[str, set[str]] = defaultdict(set)
    for items in transactions_items:
        present = sorted(items & frequent_items)
        for first, second in combinations(present, 2):
            neighbours[first].add(second)
    level: dict[Itemset, TidSet] = {}
    candidates = 0
    for first in sorted(neighbours):
        first_tids = level_one[(first,)]
        for second in sorted(neighbours[first]):
            candidates += 1
            tids = first_tids & level_one[(second,)]
            if len(tids) >= min_support:
                level[(first, second)] = tids
    return level, candidates


def mine_frequent_itemsets(
    transactions: Mapping[str, Iterable[str]],
    config: AprioriConfig = AprioriConfig(),
    allowed_items: Iterable[str] | None = None,
) -> AprioriResult:
    """Run level-wise Apriori with tidset intersection up to ``config.max_k``."""
    started = time.perf_counter()
    tidsets, dropped = build_tidsets(transactions, allowed_items, config.max_authors_per_article)
    stats: list[LevelStats] = []

    level_started = time.perf_counter()
    level_one = {(item,): tids for item, tids in tidsets.items() if len(tids) >= config.min_support}
    stats.append(LevelStats(1, len(tidsets), 0, len(level_one), time.perf_counter() - level_started))
    levels: dict[int, dict[Itemset, TidSet]] = {1: level_one}

    if config.level_limit >= 2 and level_one:
        level_started = time.perf_counter()
        used_transactions = (
            set(items)
            for items in transactions.values()
            if config.max_authors_per_article is None or len(set(items)) <= config.max_authors_per_article
        )
        level_two, candidate_count = _level_two(level_one, used_transactions, config.min_support)
        stats.append(LevelStats(2, candidate_count, 0, len(level_two), time.perf_counter() - level_started))
        levels[2] = level_two

    k = 3
    while k <= config.level_limit and levels.get(k - 1):
        level_started = time.perf_counter()
        previous = levels[k - 1]
        candidates, pruned = generate_candidates(previous)
        current: dict[Itemset, TidSet] = {}
        for candidate in candidates:
            tids = previous[candidate[:-1]] & previous[(*candidate[:-2], candidate[-1])]
            if len(tids) >= config.min_support:
                current[candidate] = tids
        stats.append(LevelStats(k, len(candidates), pruned, len(current), time.perf_counter() - level_started))
        levels[k] = current
        k += 1

    result = AprioriResult(
        levels=levels,
        stats=stats,
        transactions_total=len(transactions),
        transactions_dropped_hyperauthored=dropped,
        transactions_used=len(transactions) - dropped,
        runtime_seconds=0.0,
    )
    deepest = max(levels)
    if config.max_k is not None and deepest == config.max_k and levels[deepest]:
        result.truncated_at_k = deepest
    result.groups = flag_maximal_and_closed(levels, config.min_group_size)
    result.runtime_seconds = time.perf_counter() - started
    return result


def flag_maximal_and_closed(
    levels: Mapping[int, Mapping[Itemset, TidSet]],
    min_group_size: int = 2,
) -> list[FrequentGroup]:
    """Mark each frequent itemset as maximal and/or closed using its (k+1)-supersets.

    Maximal: no frequent (k+1)-superset. Closed: no (k+1)-superset with equal
    support. Checking only the next level suffices for both properties. At the
    last mined level (``max_k``), both flags mean "within max_k".
    """
    non_maximal: set[Itemset] = set()
    non_closed: set[Itemset] = set()
    for k, itemsets in levels.items():
        lower = levels.get(k - 1, {})
        for superset, tids in itemsets.items():
            for subset in combinations(superset, k - 1):
                if subset in lower:
                    non_maximal.add(subset)
                    if len(lower[subset]) == len(tids):
                        non_closed.add(subset)
    groups = [
        FrequentGroup(itemset, tids, itemset not in non_maximal, itemset not in non_closed)
        for k in sorted(levels)
        if k >= min_group_size
        for itemset, tids in sorted(levels[k].items())
    ]
    return groups


def select_groups(groups: Iterable[FrequentGroup], mode: str = "maximal") -> list[FrequentGroup]:
    """Keep ``"all"``, ``"maximal"`` or ``"closed"`` groups."""
    if mode == "all":
        return list(groups)
    if mode == "maximal":
        return [group for group in groups if group.maximal]
    if mode == "closed":
        return [group for group in groups if group.closed]
    raise AprioriConfigError(f"Unknown group selection mode: {mode!r}")
