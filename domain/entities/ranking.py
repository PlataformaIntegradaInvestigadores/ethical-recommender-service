"""Domain entities for ranked researcher recommendations and their audit trail."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    """A researcher retrieved in stage 1 with the attributes stage 2 needs."""

    author_id: str
    relevance: float
    institution: str
    protected: bool


@dataclass(frozen=True, slots=True)
class RankedItem:
    """A re-ranked candidate plus the explanation of why it was placed there.

    ``reason`` is one of ``"score"`` (best combined score), ``"quota"`` (placed to
    satisfy the low-visibility prefix constraint) or ``"cap_relaxed"`` (chosen
    after every remaining candidate violated the institution cap).
    """

    candidate: Candidate
    position: int
    relevance_norm: float
    institution_novelty: float
    final_score: float
    reason: str
