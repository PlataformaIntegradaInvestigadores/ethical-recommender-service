"""Ports the recommendation use case depends on; infrastructure provides the adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from domain.entities.researcher import ResearcherProfile


class AuthorEmbeddingRepository(Protocol):
    """Read-only access to the L2-normalized author embedding matrix."""

    @property
    def model_name(self) -> str: ...

    @property
    def matrix(self) -> np.ndarray: ...

    @property
    def author_ids(self) -> list[str]: ...

    def row_of(self, author_id: str) -> int | None: ...


class AuthorMetadataRepository(Protocol):
    """Lookup of researcher metadata (name, primary institution, low-visibility flag)."""

    def get(self, author_id: str) -> ResearcherProfile: ...
