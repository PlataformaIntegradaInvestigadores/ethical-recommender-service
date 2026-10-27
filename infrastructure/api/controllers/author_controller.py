"""Controller: maps the author-search HTTP query <-> use-case objects."""

from __future__ import annotations

from dataclasses import asdict

from application.use_cases.search_authors import AuthorSearchQuery, SearchAuthors
from infrastructure.schemas.authors import AuthorOut, AuthorSearchResponse


class AuthorController:
    def __init__(self, use_case: SearchAuthors):
        self._use_case = use_case

    def search(self, q: str, limit: int) -> AuthorSearchResponse:
        hits = self._use_case.execute(AuthorSearchQuery(text=q, limit=limit))
        return AuthorSearchResponse(query=q, count=len(hits), results=[AuthorOut(**asdict(h)) for h in hits])
