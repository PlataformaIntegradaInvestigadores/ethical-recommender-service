"""Leave-one-out query/target split over ground-truth co-author groups."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Final

from domain.entities.research_group import LeaveOneOutCase

DEFAULT_MIN_GROUP_SIZE: Final[int] = 3
DEFAULT_SEED: Final[int] = 42


class EvaluationSplitError(ValueError):
    """The split parameters cannot produce valid cases."""


def leave_one_out_cases(
    groups: Iterable[tuple[str, tuple[str, ...]]],
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    all_members: bool = False,
    seed: int = DEFAULT_SEED,
) -> list[LeaveOneOutCase]:
    """Hide one member of every group with ``size >= min_group_size``.

    ``groups`` yields ``(group_id, members)``. With ``all_members=False`` one
    member is hidden per group, chosen with a seeded RNG (deterministic); with
    ``True`` every member is hidden in turn, giving ``size`` cases per group.
    The remaining members form the query; the hidden one is the single relevant
    item for P@k, Recall@k, nDCG@k and hit-rate.
    """
    if min_group_size < 2:
        raise EvaluationSplitError("min_group_size must be >= 2 so the query is non-empty")
    rng = random.Random(seed)
    cases: list[LeaveOneOutCase] = []
    for group_id, members in groups:
        ordered = tuple(sorted(set(members)))
        if len(ordered) < min_group_size:
            continue
        targets = ordered if all_members else (rng.choice(ordered),)
        for target in targets:
            query = tuple(member for member in ordered if member != target)
            cases.append(LeaveOneOutCase(group_id, query, target))
    return cases


def sample_cases(cases: list[LeaveOneOutCase], n: int, seed: int = DEFAULT_SEED) -> list[LeaveOneOutCase]:
    """Seeded sample without replacement; returns all cases when ``n >= len(cases)``."""
    if n >= len(cases):
        return list(cases)
    return random.Random(seed).sample(cases, n)
