"""Stage 2: constrained greedy re-ranking for institutional diversity and exposure fairness.

At each position the re-ranker picks the candidate maximizing

    final(c) = lambda * relevance_norm(c) + (1 - lambda) * institution_novelty(c)

(MMR-style marginal gain) subject to:

* an institution cap: at most ``max_per_institution`` items per known institution;
* a ranked-group-fairness constraint for low-visibility (protected) researchers:
  every prefix of length i must contain at least m(i) protected items, where
  m(i) = floor(p * i) (``"prefix"`` mode) or the fa*ir binomial lower bound
  (``"fair"`` mode, Zehlike et al. 2017). The constraint is only satisfied with
  candidates whose normalized relevance is >= ``relevance_floor``; when none is
  available the shortfall is recorded instead of promoting irrelevant people.

``UNKNOWN`` affiliations get zero novelty (they never count as diversity) and
are not subject to the cap. With ``temperature > 0`` the pick at each step is
sampled from a softmax over ``final`` using a seeded RNG (COIN-style
probabilistic exposure); with ``temperature == 0`` the function is deterministic.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

import numpy as np

from domain.entities.ranking import Candidate, RankedItem
from domain.entities.research_group import UNKNOWN_INSTITUTION

DEFAULT_LAMBDA: Final[float] = 0.7
DEFAULT_MAX_PER_INSTITUTION: Final[int] = 3
DEFAULT_MIN_PROTECTED_SHARE: Final[float] = 0.25
DEFAULT_RELEVANCE_FLOOR: Final[float] = 0.5
DEFAULT_FAIR_ALPHA: Final[float] = 0.1
DEFAULT_QUERY_INSTITUTION_PENALTY: Final[float] = 1.0
FAIRNESS_MODES: Final[frozenset[str]] = frozenset({"prefix", "fair", "none"})


class RerankConfigError(ValueError):
    """Invalid re-ranking parameters."""


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Tunable knobs of the stage-2 re-ranker."""

    lambda_: float = DEFAULT_LAMBDA
    max_per_institution: int | None = DEFAULT_MAX_PER_INSTITUTION
    min_protected_share: float = DEFAULT_MIN_PROTECTED_SHARE
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR
    fairness_mode: str = "prefix"
    fair_alpha: float = DEFAULT_FAIR_ALPHA
    query_institution_penalty: float = DEFAULT_QUERY_INSTITUTION_PENALTY
    temperature: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_ <= 1.0:
            raise RerankConfigError("lambda_ must be in [0, 1]")
        if self.max_per_institution is not None and self.max_per_institution < 1:
            raise RerankConfigError("max_per_institution must be >= 1 or None")
        if not 0.0 <= self.min_protected_share <= 1.0:
            raise RerankConfigError("min_protected_share must be in [0, 1]")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise RerankConfigError("relevance_floor is on the normalized [0, 1] scale")
        if self.fairness_mode not in FAIRNESS_MODES:
            raise RerankConfigError(f"fairness_mode must be one of {sorted(FAIRNESS_MODES)}")
        if not 0.0 < self.fair_alpha < 1.0:
            raise RerankConfigError("fair_alpha must be in (0, 1)")
        if self.temperature < 0.0:
            raise RerankConfigError("temperature must be >= 0")
        if self.query_institution_penalty < 0.0:
            raise RerankConfigError("query_institution_penalty must be >= 0")


@dataclass(slots=True)
class RerankResult:
    """Re-ranked list plus constraint diagnostics for the audit layer."""

    items: list[RankedItem]
    quota_shortfalls: int = 0
    cap_relaxations: int = 0
    required_protected: list[int] = field(default_factory=list)

    @property
    def author_ids(self) -> list[str]:
        return [item.candidate.author_id for item in self.items]


def _binomial_cdf(m: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(m + 1))


@lru_cache(maxsize=256)
def _fair_table(k: int, p: float, alpha: float) -> tuple[int, ...]:
    table = []
    for i in range(1, k + 1):
        m = 0
        while _binomial_cdf(m, i, p) <= alpha:
            m += 1
        table.append(m)
    return tuple(table)


def required_protected_counts(k: int, p: float, mode: str = "prefix", alpha: float = DEFAULT_FAIR_ALPHA) -> list[int]:
    """Minimum protected items required in each prefix 1..k.

    ``prefix``: floor(p * i). ``fair``: smallest m with BinomCDF(m; i, p) > alpha,
    i.e. the fa*ir mtable without multiple-testing adjustment. ``none``: zeros.
    """
    if mode == "none" or p <= 0.0:
        return [0] * k
    if mode == "prefix":
        return [math.floor(p * i + 1e-9) for i in range(1, k + 1)]
    if mode == "fair":
        return list(_fair_table(k, round(p, 6), alpha))
    raise RerankConfigError(f"Unknown fairness mode {mode!r}")


def normalize_relevance(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1]; a constant list maps to 1.0."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def institution_novelty(
    institution: str,
    selected_counts: Counter[str],
    query_institutions: frozenset[str],
    query_penalty: float = DEFAULT_QUERY_INSTITUTION_PENALTY,
) -> float:
    """1 / (1 + times already selected + penalty if shared with the query group); UNKNOWN -> 0."""
    if institution == UNKNOWN_INSTITUTION:
        return 0.0
    penalty = selected_counts[institution] + (query_penalty if institution in query_institutions else 0.0)
    return 1.0 / (1.0 + penalty)


def rerank(
    candidates: Sequence[Candidate],
    k: int,
    config: RerankConfig = RerankConfig(),
    query_institutions: Iterable[str] = (),
) -> RerankResult:
    """Greedy constrained re-ranking of a stage-1 candidate list into a Top-k."""
    if k < 1:
        raise RerankConfigError("k must be >= 1")
    k = min(k, len(candidates))
    query_insts = frozenset(i for i in query_institutions if i != UNKNOWN_INSTITUTION)
    rel_norm = normalize_relevance([c.relevance for c in candidates])
    required = required_protected_counts(k, config.min_protected_share, config.fairness_mode, config.fair_alpha)
    rng = np.random.default_rng(config.seed)
    remaining = list(range(len(candidates)))
    counts: Counter[str] = Counter()
    protected_selected = 0
    result = RerankResult(items=[], required_protected=required)

    def capped(index: int) -> bool:
        inst = candidates[index].institution
        cap = config.max_per_institution
        return cap is not None and inst != UNKNOWN_INSTITUTION and counts[inst] >= cap

    def scored(pool: list[int]) -> list[tuple[float, float, int]]:
        out = []
        for index in pool:
            novelty = institution_novelty(
                candidates[index].institution,
                counts,
                query_insts,
                config.query_institution_penalty,
            )
            out.append((config.lambda_ * rel_norm[index] + (1.0 - config.lambda_) * novelty, novelty, index))
        return out

    def pick(pool: list[int]) -> tuple[float, float, int]:
        options = scored(pool)
        if config.temperature > 0.0 and len(options) > 1:
            finals = np.array([option[0] for option in options])
            weights = np.exp((finals - finals.max()) / config.temperature)
            return options[int(rng.choice(len(options), p=weights / weights.sum()))]
        return max(options, key=lambda option: (option[0], rel_norm[option[2]], -option[2]))

    for position in range(1, k + 1):
        open_pool = [i for i in remaining if not capped(i)]
        reason = "score"
        if not open_pool:
            open_pool, reason = list(remaining), "cap_relaxed"
            result.cap_relaxations += 1
        if protected_selected < required[position - 1]:
            eligible = [i for i in open_pool if candidates[i].protected and rel_norm[i] >= config.relevance_floor]
            if eligible:
                open_pool, reason = eligible, "quota"
            else:
                result.quota_shortfalls += 1
        final, novelty, index = pick(open_pool)
        candidate = candidates[index]
        result.items.append(RankedItem(candidate, position, rel_norm[index], novelty, final, reason))
        remaining.remove(index)
        counts[candidate.institution] += 1
        protected_selected += int(candidate.protected)
    return result
