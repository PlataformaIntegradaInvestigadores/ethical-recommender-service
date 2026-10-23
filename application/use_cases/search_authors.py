"""Use case: search the author catalogue for the team-builder autocomplete.

Matching is case- and accent-insensitive over the author name (``"carrion"``
finds ``"Carrión V."``); every query token must appear in the name. A purely
numeric query is treated as a Scopus author id and matched exactly. Only
authors that have a row in the embedding matrix are returned, so every hit can
be sent to ``POST /api/v1/recommend`` without a 404.

Ranking: exact name > name prefix > a name word starting with the query >
substring, then more publications first, then alphabetical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from domain.entities.researcher import AuthorSummary
from domain.repositories.author_repositories import AuthorEmbeddingRepository, AuthorMetadataRepository

MIN_QUERY_CHARS = 2
MAX_LIMIT = 50
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


class AuthorSearchError(ValueError):
    """Invalid search request."""


def normalize_text(text: str | None) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return _NON_ALNUM.sub(" ", ascii_text).strip()


@dataclass(frozen=True, slots=True)
class AuthorSearchQuery:
    text: str
    limit: int = 10


@dataclass(frozen=True, slots=True)
class _Entry:
    summary: AuthorSummary
    name: str
    words: tuple[str, ...]


class SearchAuthors:
    def __init__(self, embeddings: AuthorEmbeddingRepository, metadata: AuthorMetadataRepository):
        self._embeddings = embeddings
        self._metadata = metadata
        self._by_id: dict[str, AuthorSummary] = {}
        self._entries: list[_Entry] = []
        for summary in metadata.summaries():
            if embeddings.row_of(summary.author_id) is None:
                continue
            self._by_id[summary.author_id] = summary
            name = normalize_text(summary.name)
            if name:
                self._entries.append(_Entry(summary, name, tuple(name.split())))

    def __len__(self) -> int:
        return len(self._by_id)

    def execute(self, query: AuthorSearchQuery) -> list[AuthorSummary]:
        if not 1 <= query.limit <= MAX_LIMIT:
            raise AuthorSearchError(f"limit must be between 1 and {MAX_LIMIT}")
        raw = (query.text or "").strip()
        if raw.isdigit():
            return self._by_id_lookup(raw)
        text = normalize_text(raw)
        if len(text.replace(" ", "")) < MIN_QUERY_CHARS:
            raise AuthorSearchError(f"q must contain at least {MIN_QUERY_CHARS} letters or digits")
        tokens = text.split()
        ranked: list[tuple[int, int, str, AuthorSummary]] = []
        for entry in self._entries:
            if not all(t in entry.name for t in tokens):
                continue
            ranked.append((self._tier(entry, text, tokens[0]), -(entry.summary.article_count or 0),
                           entry.name, entry.summary))
        ranked.sort(key=lambda r: r[:3])
        return [r[3] for r in ranked[:query.limit]]

    def _by_id_lookup(self, author_id: str) -> list[AuthorSummary]:
        if author_id in self._by_id:
            return [self._by_id[author_id]]
        if self._embeddings.row_of(author_id) is not None:  # in the matrix but without metadata
            profile = self._metadata.get(author_id)
            return [AuthorSummary(author_id, profile.name, profile.institution, profile.low_visibility)]
        return []

    @staticmethod
    def _tier(entry: _Entry, text: str, first_token: str) -> int:
        if entry.name == text:
            return 0
        if entry.name.startswith(text):
            return 1
        if any(w.startswith(first_token) for w in entry.words):
            return 2
        return 3
