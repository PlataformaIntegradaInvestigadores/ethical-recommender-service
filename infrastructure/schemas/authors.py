"""Pydantic schemas of the author-search API (team-builder autocomplete)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorOut(BaseModel):
    author_id: str = Field(..., description="ID Scopus del autor")
    name: str | None
    institution: str
    low_visibility: bool
    article_count: int | None = None


class AuthorSearchResponse(BaseModel):
    query: str
    count: int
    results: list[AuthorOut]

    model_config = {"json_schema_extra": {"examples": [{
        "query": "carrion", "count": 1,
        "results": [{"author_id": "10040712400", "name": "Carrion V.", "institution": "parque nacional galápagos",
                     "low_visibility": False, "article_count": 13}]}]}}
