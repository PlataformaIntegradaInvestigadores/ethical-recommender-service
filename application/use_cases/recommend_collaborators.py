"""Use case: recommend researchers to complete an a-priori team.

Pipeline (reuses the offline services, no logic is duplicated):

1. Stage 1 – ``group_scoring.score_all_candidates``: hybrid Boratto aggregation
   ``alpha * mean_i cos(c, m_i) + (1 - alpha) * max_i cos(c, m_i)``. With
   ``alpha = 1`` (default) it is the Average strategy, i.e. the cosine to the
   team centroid up to a constant factor, so the K-NN order is the centroid one.
   ``group_scoring.retrieve_top_n`` returns the ``k_candidates`` nearest
   neighbours, excluding the team members.
2. Stage 2 – ``fair_reranker.rerank`` with the R1 setting of the experiments
   (lambda 0.8, institution cap 3, 20 % low-visibility prefix quota, relevance
   floor 0.5).
3. Fairness metrics of the Top-k – ``ranking_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from application.services import ranking_metrics as rm
from application.services.fair_reranker import RerankConfig, rerank
from application.services.group_scoring import retrieve_top_n, score_all_candidates
from domain.entities.ranking import Candidate
from domain.entities.research_group import UNKNOWN_INSTITUTION
from domain.entities.researcher import ResearcherProfile
from domain.repositories.author_repositories import AuthorEmbeddingRepository, AuthorMetadataRepository

# Same values as SETTINGS["R1_suave"] in infrastructure/scripts/04_run_rerank_experiment.py.
R1_CONFIG = RerankConfig(lambda_=0.8, max_per_institution=3, min_protected_share=0.2,
                         relevance_floor=0.5, fairness_mode="prefix")


class RecommendationError(ValueError):
    """Invalid recommendation request."""


class UnknownAuthorsError(RecommendationError):
    def __init__(self, unknown: list[str]):
        super().__init__(f"Unknown author_ids: {', '.join(unknown)}")
        self.unknown = unknown


@dataclass(frozen=True, slots=True)
class RecommendationQuery:
    author_ids: list[str]
    top_k: int = 20
    k_candidates: int = 100
    alpha: float = 1.0
    apply_fairness: bool = True
    rerank_config: RerankConfig = R1_CONFIG


@dataclass(frozen=True, slots=True)
class RecommendedResearcher:
    profile: ResearcherProfile
    rank: int
    relevance: float
    relevance_norm: float
    institution_novelty: float
    final_score: float
    reason: str
    baseline_rank: int


@dataclass(slots=True)
class RecommendationResult:
    model_name: str
    team: list[ResearcherProfile]
    items: list[RecommendedResearcher]
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)
    config: RerankConfig | None = None


class RecommendCollaborators:
    def __init__(self, embeddings: AuthorEmbeddingRepository, metadata: AuthorMetadataRepository):
        self._embeddings = embeddings
        self._metadata = metadata

    def execute(self, query: RecommendationQuery) -> RecommendationResult:
        team_ids = list(dict.fromkeys(a.strip() for a in query.author_ids if a and a.strip()))
        if not team_ids:
            raise RecommendationError("author_ids must contain at least one id")
        if query.top_k < 1:
            raise RecommendationError("top_k must be >= 1")
        if query.k_candidates < query.top_k:
            raise RecommendationError("k_candidates must be >= top_k")
        rows = {a: self._embeddings.row_of(a) for a in team_ids}
        unknown = [a for a, r in rows.items() if r is None]
        if unknown:
            raise UnknownAuthorsError(unknown)

        member_rows = [rows[a] for a in team_ids]
        scores = score_all_candidates(self._embeddings.matrix, member_rows, alpha=query.alpha)
        top = retrieve_top_n(scores, member_rows, query.k_candidates)
        ids = self._embeddings.author_ids
        profiles = {ids[r]: self._metadata.get(ids[r]) for r, _ in top}
        candidates = [Candidate(ids[r], s, profiles[ids[r]].institution, profiles[ids[r]].low_visibility)
                      for r, s in top]
        baseline_rank = {c.author_id: i + 1 for i, c in enumerate(candidates)}
        team = [self._metadata.get(a) for a in team_ids]
        config = query.rerank_config if query.apply_fairness else replace(
            query.rerank_config, lambda_=1.0, max_per_institution=None, fairness_mode="none")
        result = rerank(candidates, query.top_k, config, [p.institution for p in team]) if candidates else None

        items = [] if result is None else [
            RecommendedResearcher(profiles[it.candidate.author_id], it.position, it.candidate.relevance,
                                  it.relevance_norm, it.institution_novelty, it.final_score, it.reason,
                                  baseline_rank[it.candidate.author_id])
            for it in result.items
        ]
        ranked = [i.profile.author_id for i in items]
        k = len(ranked)
        metrics: dict[str, float] = {}
        if k:
            protected = {c.author_id: c.protected for c in candidates}
            insts = [i.profile.institution for i in items]
            pool_share = sum(protected.values()) / len(protected)
            metrics = {
                "gini_simpson": rm.gini_simpson(insts),
                "gini_simpson_m2": rm.anti_tokenism_gini_simpson(insts),
                "unique_institutions": float(rm.unique_institutions(insts)),
                "unknown_institution_share": sum(i == UNKNOWN_INSTITUTION for i in insts) / k,
                "low_visibility_share": rm.protected_share(ranked, protected, k),
                "low_visibility_exposure_share": rm.exposure_share(ranked, protected, k),
                "pool_low_visibility_share": pool_share,
                "ndkl": rm.ndkl(ranked, protected, k, pool_share),
                "relevance_retained": rm.relevance_retained(
                    ranked, [c.author_id for c in candidates[:k]], {c.author_id: c.relevance for c in candidates}, k),
            }
            metrics = {key: round(float(v), 6) for key, v in metrics.items()}
        diagnostics = {"candidates_retrieved": len(candidates),
                       "quota_picks": sum(i.reason == "quota" for i in items),
                       "quota_shortfalls": result.quota_shortfalls if result else 0,
                       "cap_relaxations": result.cap_relaxations if result else 0}
        return RecommendationResult(self._embeddings.model_name, team, items, metrics, diagnostics, config)
