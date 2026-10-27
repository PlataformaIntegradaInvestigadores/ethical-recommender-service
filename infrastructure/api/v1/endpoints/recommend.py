"""HTTP routes of API v1."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from application.services.fair_reranker import RerankConfigError
from application.services.group_scoring import GroupScoringError
from application.use_cases.recommend_collaborators import RecommendationError, UnknownAuthorsError
from infrastructure.api.controllers.recommend_controller import RecommendController
from infrastructure.api.dependencies import get_controller
from infrastructure.schemas.recommendation import RecommendRequest, RecommendResponse

router = APIRouter(prefix="/api/v1", tags=["recommendation"])


@router.post("/recommend", response_model=RecommendResponse,
             responses={404: {"description": "author_ids desconocidos"}, 422: {"description": "payload inválido"}})
def recommend(request: RecommendRequest, controller: RecommendController = Depends(get_controller)) -> RecommendResponse:
    """Centroide (Average) + K-NN y re-ranking R1 (Boratto / equidad) para completar el equipo."""
    try:
        return controller.recommend(request)
    except UnknownAuthorsError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            {"message": "Unknown author_ids", "unknown_author_ids": exc.unknown}) from exc
    except (RecommendationError, RerankConfigError, GroupScoringError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
