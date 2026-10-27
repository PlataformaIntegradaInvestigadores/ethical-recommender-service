"""Pydantic schemas of the author-search API (team-builder autocomplete)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorOut(BaseModel):
    author_id: str = Field(..., description="ID Scopus del autor")
    name: str | None
    institution: str
    low_visibility: bool
    article_count: int | None = None
    full_name: str | None = Field(None, description="Nombre completo 'Apellido, Nombre' (si se conoce)")


class AuthorSearchResponse(BaseModel):
    query: str
    count: int
    results: list[AuthorOut]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "pablo piedrahita",
                    "count": 1,
                    "results": [
                        {
                            "author_id": "37026882700",
                            "name": "Piedrahita P.",
                            "institution": "escuela superior politecnica del litoral ecuador",
                            "low_visibility": False,
                            "article_count": 16,
                            "full_name": "Piedrahita, Paolo",
                        }
                    ],
                }
            ]
        }
    }
