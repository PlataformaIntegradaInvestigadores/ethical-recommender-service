"""Composition root: builds repositories once and injects them via app.state."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from application.use_cases.recommend_collaborators import RecommendCollaborators
from application.use_cases.search_authors import SearchAuthors
from infrastructure.api.controllers.author_controller import AuthorController
from infrastructure.api.controllers.recommend_controller import RecommendController
from infrastructure.api.settings import Settings
from infrastructure.db.artifact_loaders import CsvMetadataRepository, NpyEmbeddingRepository


@dataclass(slots=True)
class Container:
    embeddings: NpyEmbeddingRepository
    metadata: CsvMetadataRepository
    controller: RecommendController
    author_controller: AuthorController


def build_container(settings: Settings) -> Container:
    embeddings = NpyEmbeddingRepository(settings.embeddings_path, settings.mapping_path)
    metadata = CsvMetadataRepository(settings.metadata_path, settings.profiles_path, settings.authors_path)
    return Container(
        embeddings,
        metadata,
        RecommendController(RecommendCollaborators(embeddings, metadata)),
        AuthorController(SearchAuthors(embeddings, metadata)),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_controller(request: Request) -> RecommendController:
    return request.app.state.container.controller


def get_author_controller(request: Request) -> AuthorController:
    return request.app.state.container.author_controller
