"""Use case: search the author catalogue for the team-builder autocomplete.

Flexible (fuzzy) matching over the Scopus name (``"Piedrahita P."``) and, when
available, the full name (``"Piedrahita, Paolo"``). Text is normalized
(lower-case, no accents or punctuation) and split into words; every query token
must match a different word of the author, in any order, by one of:

* exact word (``"piedrahita"``), score 3
* word prefix (``"piedra"``), score 2
* one-edit typo for tokens of 4+ letters (``"pablo"`` ~ ``"paolo"``), score 1.5
* initial (``"pablo"`` ~ ``"p"`` in ``"Piedrahita P."``), score 1; only when
  another token matched a word by exact/prefix/typo, so a lone first name does
  not flood the list with every author sharing that initial
* substring (``"arrion"`` in ``"carrion"``), score 0.5

With two or more tokens, at least one token of 2+ letters must match a surname word
(exact, prefix or typo), so ``"Pablo Piedrahita"`` does not return "Quezada, Pablo A." through the
initial "P.". So ``"Pablo Piedrahita"``, ``"piedrahita pablo"`` and the strict
Scopus form ``"Piedrahita P."`` all find the same author. A purely numeric query is a Scopus
author id and is matched exactly. Only authors with a row in the embedding matrix
are returned, so every hit can be sent to ``POST /api/v1/recommend``.

Ranking: higher total match score, then surname matched, then exact name > name
prefix > other, then more publications, then alphabetical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from domain.entities.researcher import AuthorSummary
from domain.repositories.author_repositories import AuthorEmbeddingRepository, AuthorMetadataRepository

MIN_QUERY_CHARS = 2
MAX_LIMIT = 50
MIN_TYPO_CHARS = 4
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

EXACT, PREFIX, TYPO, INITIAL, SUBSTRING = 3.0, 2.0, 1.5, 1.0, 0.5
_ANCHORS = (EXACT, PREFIX, TYPO)


class AuthorSearchError(ValueError):
    """Invalid search request."""


def normalize_text(text: str | None) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return _NON_ALNUM.sub(" ", ascii_text).strip()


def within_one_edit(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by at most one insertion, deletion, substitution or swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def word_score(token: str, word: str) -> float:
    """How well one query token matches one author-name word (0 = no match)."""
    if word == token:
        return EXACT
    if word.startswith(token):
        return PREFIX
    if (
        len(token) >= MIN_TYPO_CHARS
        and len(word) >= MIN_TYPO_CHARS
        and word[0] == token[0]
        and within_one_edit(token, word)
    ):
        return TYPO
    if len(word) == 1 and len(token) > 1 and token[0] == word:
        return INITIAL
    if len(token) >= MIN_QUERY_CHARS and token in word:
        return SUBSTRING
    return 0.0


@dataclass(frozen=True, slots=True)
class AuthorSearchQuery:
    text: str
    limit: int = 10


@dataclass(frozen=True, slots=True)
class _Entry:
    summary: AuthorSummary
    names: tuple[str, ...]  # normalized Scopus name and full name
    words: tuple[str, ...]  # distinct words of both
    surname: frozenset[str]  # words of the last name(s)


def surname_words(scopus_name: str | None, full_name: str | None) -> frozenset[str]:
    """Last-name words: Scopus name without its trailing initials, and the part before the comma of
    ``"Apellido, Nombre"``."""
    words: set[str] = set()
    scopus = normalize_text(scopus_name).split()
    while len(scopus) > 1 and len(scopus[-1]) == 1:
        scopus.pop()
    words.update(scopus)
    if full_name and "," in full_name:
        words.update(normalize_text(full_name.split(",", 1)[0]).split())
    return frozenset(words)


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
            names = tuple(n for n in (normalize_text(summary.name), normalize_text(summary.full_name)) if n)
            if names:
                words = tuple(dict.fromkeys(w for n in names for w in n.split()))
                self._entries.append(_Entry(summary, names, words, surname_words(summary.name, summary.full_name)))
        self._vocabulary = {w for e in self._entries for w in e.words}

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
        tokens = sorted(text.split(), key=len, reverse=True)  # longest (most selective) first
        # Score each token against the vocabulary once; entries then only do dict lookups.
        token_scores = [(t, self._score_vocabulary(t)) for t in tokens]
        ranked: list[tuple[float, bool, int, int, str, AuthorSummary]] = []
        for entry in self._entries:
            match = _match(entry, token_scores)
            if match is None:
                continue
            score, surname_hit = match
            if len(tokens) > 1 and not surname_hit:
                continue
            ranked.append(
                (
                    -score,
                    not surname_hit,
                    _tier(entry.names, text),
                    -(entry.summary.article_count or 0),
                    entry.names[0],
                    entry.summary,
                )
            )
        ranked.sort(key=lambda r: r[:5])
        return [r[5] for r in ranked[: query.limit]]

    def _score_vocabulary(self, token: str) -> dict[str, float]:
        return {w: s for w in self._vocabulary if (s := word_score(token, w)) > 0}

    def _by_id_lookup(self, author_id: str) -> list[AuthorSummary]:
        if author_id in self._by_id:
            return [self._by_id[author_id]]
        if self._embeddings.row_of(author_id) is not None:  # in the matrix but without metadata
            profile = self._metadata.get(author_id)
            return [AuthorSummary(author_id, profile.name, profile.institution, profile.low_visibility)]
        return []


def _match(entry: _Entry, token_scores: list[tuple[str, dict[str, float]]]) -> tuple[float, bool] | None:
    """Greedy one-to-one token->word assignment: ``(score, surname matched)``, or ``None`` when some
    token has no word left."""
    used: set[str] = set()
    total, anchored, initials, surname_hit = 0.0, False, False, False
    for token, scores in token_scores:
        best_word, best = None, 0.0
        for w in entry.words:
            s = scores.get(w, 0.0)
            if s > best and w not in used:
                best_word, best = w, s
        if best_word is None:
            return None
        used.add(best_word)
        total += best
        anchored = anchored or best in _ANCHORS
        initials = initials or best == INITIAL
        # a one-letter token is an initial ("P."), never evidence of the surname
        surname_hit = surname_hit or (best in _ANCHORS and best_word in entry.surname and len(token) > 1)
    if initials and not anchored:
        return None
    return total, surname_hit


def _tier(names: tuple[str, ...], text: str) -> int:
    if text in names:
        return 0
    if any(n.startswith(text) for n in names):
        return 1
    return 2
