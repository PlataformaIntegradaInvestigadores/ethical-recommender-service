"""Accuracy, institutional-diversity and exposure-fairness metrics for Top-K lists.

All functions are pure and operate on ordered author-id lists plus lookups.
Accuracy metrics use binary relevance against the leave-one-out ground truth.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Final

from domain.entities.research_group import UNKNOWN_INSTITUTION

DEFAULT_ANTI_TOKENISM_M: Final[int] = 2
KL_EPSILON: Final[float] = 1e-6


class MetricError(ValueError):
    """Invalid metric arguments."""


def _check_k(k: int) -> None:
    if k < 1:
        raise MetricError("k must be >= 1")


def position_discount(position: int) -> float:
    """Logarithmic discount 1 / log2(1 + position) for 1-based positions."""
    return 1.0 / math.log2(1.0 + position)


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check_k(k)
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check_k(k)
    return float(any(item in relevant for item in ranked[:k]))


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    _check_k(k)
    dcg = sum(position_discount(i) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(position_discount(i) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def gini_simpson(institutions: Sequence[str]) -> float:
    """1 - sum p_j^2 over known institutions; UNKNOWN is ignored (no diversity credit)."""
    known = [inst for inst in institutions if inst != UNKNOWN_INSTITUTION]
    if not known:
        return 0.0
    total = len(known)
    return 1.0 - sum((count / total) ** 2 for count in Counter(known).values())


def anti_tokenism_gini_simpson(institutions: Sequence[str], m: int = DEFAULT_ANTI_TOKENISM_M) -> float:
    """Gini-Simpson where an institution counts as represented only with >= m members.

    Institutions below the threshold are pooled into one residual category, so
    isolated "tokens" do not inflate diversity.
    """
    known = [inst for inst in institutions if inst != UNKNOWN_INSTITUTION]
    if not known:
        return 0.0
    counts = Counter(known)
    effective = Counter({inst: c for inst, c in counts.items() if c >= m})
    residual = sum(c for c in counts.values() if c < m)
    values = list(effective.values()) + ([residual] if residual else [])
    total = len(known)
    return 1.0 - sum((count / total) ** 2 for count in values)


def unique_institutions(institutions: Sequence[str]) -> int:
    return len({inst for inst in institutions if inst != UNKNOWN_INSTITUTION})


def institution_coverage(top_k_institutions: Sequence[str], pool_institutions: Sequence[str]) -> float:
    """Share of known institutions in the candidate pool that appear in the Top-K."""
    pool = {inst for inst in pool_institutions if inst != UNKNOWN_INSTITUTION}
    if not pool:
        return 0.0
    return unique_institutions(top_k_institutions) / len(pool)


def protected_share(ranked: Sequence[str], protected: Mapping[str, bool], k: int) -> float:
    _check_k(k)
    top = ranked[:k]
    return sum(1 for item in top if protected.get(item, False)) / len(top) if top else 0.0


def group_exposure(ranked: Sequence[str], protected: Mapping[str, bool], k: int) -> tuple[float, float]:
    """Mean log-discounted exposure per protected and per non-protected item in the Top-K."""
    _check_k(k)
    exposure = {True: [], False: []}
    for position, item in enumerate(ranked[:k], start=1):
        exposure[bool(protected.get(item, False))].append(position_discount(position))
    mean = lambda values: sum(values) / len(values) if values else 0.0  # noqa: E731
    return mean(exposure[True]), mean(exposure[False])


def exposure_share(ranked: Sequence[str], protected: Mapping[str, bool], k: int) -> float:
    """Fraction of total Top-K exposure that goes to protected items."""
    _check_k(k)
    total = sum(position_discount(i) for i in range(1, min(k, len(ranked)) + 1))
    gained = sum(position_discount(i) for i, item in enumerate(ranked[:k], start=1) if protected.get(item, False))
    return gained / total if total else 0.0


def _kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, KL_EPSILON), 1 - KL_EPSILON)
    q = min(max(q, KL_EPSILON), 1 - KL_EPSILON)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def ndkl(ranked: Sequence[str], protected: Mapping[str, bool], k: int, target_share: float) -> float:
    """Normalized Discounted KL divergence (Geyik et al. 2019) for a binary attribute.

    NDKL = (1/Z) * sum_{i=1..k} d(i) * KL(D_i || D*), with D_i the protected share
    of the top-i prefix, D* = ``target_share`` and Z = sum d(i). 0 is ideal.
    """
    _check_k(k)
    top = ranked[:k]
    if not top:
        return 0.0
    total, z, protected_seen = 0.0, 0.0, 0
    for i, item in enumerate(top, start=1):
        protected_seen += int(protected.get(item, False))
        weight = position_discount(i)
        total += weight * _kl_bernoulli(protected_seen / i, target_share)
        z += weight
    return total / z


def relevance_retained(reranked: Sequence[str], baseline: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """sum relevance(reranked Top-K) / sum relevance(baseline Top-K); 1 - value is the relevance loss."""
    _check_k(k)
    base = sum(relevance[item] for item in baseline[:k])
    if base == 0:
        return 1.0
    return sum(relevance[item] for item in reranked[:k]) / base
