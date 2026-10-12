"""Domain entities for co-authorship groups and leave-one-out evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

UNKNOWN_INSTITUTION: Final[str] = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FrequentGroup:
    """A frequent co-author itemset mined from the article-author relation.

    ``members`` is sorted lexicographically so the same group always has the
    same representation. ``article_ids`` is the tidset: the articles shared by
    every member, so ``support == len(article_ids)``.
    """

    members: tuple[str, ...]
    article_ids: frozenset[str]
    maximal: bool = False
    closed: bool = False

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def support(self) -> int:
        return len(self.article_ids)


@dataclass(frozen=True, slots=True)
class LeaveOneOutCase:
    """One evaluation query: a known group with one member hidden as the target."""

    group_id: str
    query_members: tuple[str, ...]
    target: str
