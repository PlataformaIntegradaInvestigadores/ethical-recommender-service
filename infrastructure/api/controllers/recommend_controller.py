"""Controller: maps HTTP schemas <-> use-case objects."""

from __future__ import annotations

from dataclasses import asdict

from application.services.fair_reranker import RerankConfig
from application.use_cases.recommend_collaborators import RecommendationQuery, RecommendCollaborators
from infrastructure.schemas.recommendation import RecommendedProfile, RecommendRequest, RecommendResponse, ResearcherOut


class RecommendController:
    def __init__(self, use_case: RecommendCollaborators):
        self._use_case = use_case

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        f = request.fairness
        config = RerankConfig(lambda_=f.lambda_, max_per_institution=f.max_per_institution,
                              min_protected_share=f.min_low_visibility_share,
                              relevance_floor=f.relevance_floor, fairness_mode=f.mode)
        result = self._use_case.execute(RecommendationQuery(
            author_ids=request.author_ids, top_k=request.top_k, k_candidates=request.k_candidates,
            alpha=request.alpha, apply_fairness=f.enabled, rerank_config=config))
        return RecommendResponse(
            model=result.model_name,
            team=[ResearcherOut(**asdict(p)) for p in result.team],
            recommendations=[RecommendedProfile(
                **asdict(i.profile), rank=i.rank, baseline_rank=i.baseline_rank,
                relevance_score=round(i.relevance, 6), relevance_norm=round(i.relevance_norm, 6),
                institution_novelty=round(i.institution_novelty, 6), final_score=round(i.final_score, 6),
                selection_reason=i.reason) for i in result.items],
            fairness_metrics=result.metrics,
            diagnostics=result.diagnostics,
            parameters={"top_k": request.top_k, "k_candidates": request.k_candidates, "alpha": request.alpha,
                        "rerank": asdict(result.config) if result.config else None},
        )
