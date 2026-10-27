"""HTTP routes of API v1: author search for the team builder."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from application.use_cases.search_authors import MAX_LIMIT, AuthorSearchError
from infrastructure.api.controllers.author_controller import AuthorController
from infrastructure.api.dependencies import get_author_controller
from infrastructure.schemas.authors import AuthorSearchResponse

router = APIRouter(prefix="/api/v1", tags=["authors"])


@router.get("/authors", response_model=AuthorSearchResponse,
            responses={422: {"description": "consulta demasiado corta o parámetros inválidos"}})
def search_authors(
    q: str = Query(..., min_length=1, max_length=100,
                   description="Nombre (sin distinguir mayúsculas ni tildes) o ID Scopus exacto"),
    limit: int = Query(10, ge=1, le=MAX_LIMIT),
    controller: AuthorController = Depends(get_author_controller),
) -> AuthorSearchResponse:
    """Autores del corpus con embedding disponible, para armar el equipo a-priori."""
    try:
        return controller.search(q, limit)
    except AuthorSearchError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
