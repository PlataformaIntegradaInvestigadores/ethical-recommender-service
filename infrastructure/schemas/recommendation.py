"""Pydantic request/response schemas of the recommendation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FairnessOptions(BaseModel):
    """Overrides of the R1 re-ranking setting (defaults = R1_suave)."""

    enabled: bool = Field(True, description="False returns the stage-1 relevance order (baseline)")
    lambda_: float = Field(0.8, ge=0.0, le=1.0, alias="lambda", description="relevance weight in the MMR score")
    max_per_institution: int | None = Field(3, ge=1, description="institution cap; null disables it")
    min_low_visibility_share: float = Field(0.2, ge=0.0, le=1.0, description="prefix quota of low-visibility authors")
    relevance_floor: float = Field(0.5, ge=0.0, le=1.0)
    mode: Literal["prefix", "fair", "none"] = "prefix"

    model_config = {"populate_by_name": True}


class RecommendRequest(BaseModel):
    author_ids: list[str] = Field(..., min_length=1, description="IDs Scopus del equipo a-priori")
    top_k: int = Field(20, ge=1, le=200)
    k_candidates: int = Field(100, ge=1, le=2000, description="tamaño del pool K-NN de la etapa 1")
    alpha: float = Field(1.0, ge=0.0, le=1.0, description="1 = Average (centroide); <1 mezcla Most Pleasure")
    fairness: FairnessOptions = Field(default_factory=FairnessOptions)

    model_config = {"json_schema_extra": {"examples": [
        {"author_ids": ["10040712400", "15842091200"], "top_k": 20}]}}


class ResearcherOut(BaseModel):
    author_id: str
    name: str | None
    institution: str
    low_visibility: bool


class RecommendedProfile(ResearcherOut):
    rank: int
    baseline_rank: int
    relevance_score: float
    relevance_norm: float
    institution_novelty: float
    final_score: float
    selection_reason: Literal["score", "quota", "cap_relaxed"]


class RecommendResponse(BaseModel):
    model: str
    team: list[ResearcherOut]
    recommendations: list[RecommendedProfile]
    fairness_metrics: dict[str, float]
    diagnostics: dict[str, int]
    parameters: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    authors: int
    embedding_dim: int
    metadata_source: str
