"""Stage 1: group-to-candidate affinity with hybrid preference aggregation.

Each query-group member keeps an individual embedding; no merged text vector
is built. For candidate c and members m_1..m_k with unit vectors:

    s_i(c)   = cos(e_c, e_{m_i})
    score(c) = alpha * sum_i w_i s_i(c)   (Additive Utilitarian, weighted mean)
             + (1 - alpha) * max_i s_i(c) (Most Pleasure)

following the hybrid group-aggregation strategies discussed by Boratto et al.
Weights default to uniform (1/k) and are re-normalized to sum to 1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

DEFAULT_ALPHA: Final[float] = 0.5
DEFAULT_TOP_N: Final[int] = 100
NORM_TOLERANCE: Final[float] = 1e-3

FloatMatrix = npt.NDArray[np.float32]
FloatVector = npt.NDArray[np.float64]


class GroupScoringError(ValueError):
    """Invalid query group, weights, or embedding matrix."""


def aggregate_scores(
    member_similarities: npt.NDArray[np.floating],
    alpha: float = DEFAULT_ALPHA,
    weights: Sequence[float] | None = None,
) -> FloatVector:
    """Combine a (k, n) member-by-candidate similarity matrix into n group scores."""
    if not 0.0 <= alpha <= 1.0:
        raise GroupScoringError("alpha must be in [0, 1]")
    similarities = np.atleast_2d(np.asarray(member_similarities, dtype=np.float64))
    k = similarities.shape[0]
    if k == 0:
        raise GroupScoringError("The query group is empty")
    if weights is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (k,) or np.any(w < 0) or w.sum() <= 0:
            raise GroupScoringError("weights must be k non-negative values with a positive sum")
        w = w / w.sum()
    utilitarian = w @ similarities
    most_pleasure = similarities.max(axis=0)
    return alpha * utilitarian + (1.0 - alpha) * most_pleasure


def score_all_candidates(
    embeddings: FloatMatrix,
    member_rows: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
    weights: Sequence[float] | None = None,
) -> FloatVector:
    """Score every row of an L2-normalized embedding matrix against the query group."""
    if len(member_rows) == 0:
        raise GroupScoringError("The query group is empty")
    members = embeddings[np.asarray(member_rows, dtype=np.int64)]
    if not np.allclose(np.linalg.norm(members, axis=1), 1.0, atol=NORM_TOLERANCE):
        raise GroupScoringError("Embeddings must be L2-normalized so the dot product is the cosine")
    similarities = members @ embeddings.T
    return aggregate_scores(similarities, alpha, weights)


def retrieve_top_n(
    scores: npt.NDArray[np.floating],
    excluded_rows: Iterable[int] = (),
    top_n: int = DEFAULT_TOP_N,
) -> list[tuple[int, float]]:
    """Return ``(row, score)`` pairs of the best ``top_n`` non-excluded rows.

    Ties are broken by row index so the result is deterministic.
    """
    if top_n < 1:
        raise GroupScoringError("top_n must be >= 1")
    masked = np.asarray(scores, dtype=np.float64).copy()
    excluded = np.fromiter(excluded_rows, dtype=np.int64)
    if excluded.size:
        masked[excluded] = -np.inf
    available = int(np.isfinite(masked).sum())
    n = min(top_n, available)
    if n == 0:
        return []
    order = np.lexsort((np.arange(masked.size), -masked))[:n]
    return [(int(row), float(masked[row])) for row in order]
