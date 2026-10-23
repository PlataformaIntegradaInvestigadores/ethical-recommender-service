"""Researcher profile as exposed to API consumers."""

from __future__ import annotations

from dataclasses import dataclass

from domain.entities.research_group import UNKNOWN_INSTITUTION


@dataclass(frozen=True, slots=True)
class ResearcherProfile:
    """Static metadata of one researcher; ``low_visibility`` is the protected-group flag."""

    author_id: str
    name: str | None = None
    institution: str = UNKNOWN_INSTITUTION
    low_visibility: bool = False


@dataclass(frozen=True, slots=True)
class AuthorSummary:
    """Searchable author entry (profile plus publication count) used by the team builder UI."""

    author_id: str
    name: str | None = None
    institution: str = UNKNOWN_INSTITUTION
    low_visibility: bool = False
    article_count: int | None = None
