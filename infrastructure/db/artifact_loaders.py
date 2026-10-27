"""Adapters that load the offline artifacts once (embeddings, id mapping, author metadata)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from domain.entities.research_group import UNKNOWN_INSTITUTION
from domain.entities.researcher import AuthorSummary, ResearcherProfile

METADATA_COLUMNS = ("author_id", "author_name", "institution", "low_visibility")
RAW_AUTHOR_COLUMNS = ("identifier", "first_name", "last_name")  # Scopus authors.csv


class ArtifactLoadError(RuntimeError):
    """Missing or inconsistent artifact."""


class NpyEmbeddingRepository:
    """``author_embeddings.npy`` + ``author_index_mapping.json`` written by 02_generate_embeddings.py."""

    def __init__(self, embeddings_path: Path, mapping_path: Path):
        if not Path(embeddings_path).is_file() or not Path(mapping_path).is_file():
            raise ArtifactLoadError(f"Missing {embeddings_path} or {mapping_path}")
        matrix = np.load(embeddings_path).astype(np.float32, copy=False)
        try:
            mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            ids = [str(a) for a in mapping["index_to_author_id"]]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactLoadError(f"Invalid mapping {mapping_path}: {exc}") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ArtifactLoadError(f"Embeddings {matrix.shape} do not match {len(ids)} mapped author ids")
        if len(set(ids)) != len(ids):
            raise ArtifactLoadError("Mapping contains duplicated author ids")
        self._matrix, self._ids = matrix, ids
        self._rows = {a: i for i, a in enumerate(ids)}
        self._model = str(mapping.get("model_name", "unknown"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def author_ids(self) -> list[str]:
        return self._ids

    def row_of(self, author_id: str) -> int | None:
        return self._rows.get(author_id)


def compose_full_name(first_name: object, last_name: object) -> str | None:
    """``"Apellido, Nombre"`` from Scopus ``last_name``/``first_name``; ``None`` without a last name."""
    last = last_name.strip() if isinstance(last_name, str) else ""
    first = first_name.strip() if isinstance(first_name, str) else ""
    if not last:
        return None
    return f"{last}, {first}" if first else last


class CsvMetadataRepository:
    """``author_metadata.csv`` (05_export_author_metadata.py); falls back to names from the ETL profiles.

    Full names come from the optional ``full_name`` column of the metadata; if it is absent (metadata
    exported before that column existed) they are read from the raw Scopus ``authors.csv`` when given.
    """

    def __init__(
        self, metadata_path: Path | None = None, profiles_path: Path | None = None, authors_path: Path | None = None
    ):
        self._profiles: dict[str, ResearcherProfile] = {}
        self._article_counts: dict[str, int] = {}
        self._full_names: dict[str, str] = {}
        self.source = "none"
        if metadata_path and Path(metadata_path).is_file():
            frame = pd.read_csv(metadata_path, dtype={"author_id": str, "author_name": str, "institution": str})
            missing = set(METADATA_COLUMNS) - set(frame.columns)
            if missing:
                raise ArtifactLoadError(f"{metadata_path} lacks columns {sorted(missing)}")
            for row in frame.itertuples(index=False):
                inst = row.institution if isinstance(row.institution, str) and row.institution else UNKNOWN_INSTITUTION
                name = row.author_name if isinstance(row.author_name, str) else None
                self._profiles[row.author_id] = ResearcherProfile(
                    row.author_id, name, inst, str(row.low_visibility).strip().lower() in {"true", "1"}
                )
            self._article_counts = _article_counts(frame)
            if "full_name" in frame.columns:
                self._full_names = {
                    a: n.strip()
                    for a, n in zip(frame["author_id"], frame["full_name"])
                    if isinstance(n, str) and n.strip()
                }
            self.source = str(metadata_path)
        elif profiles_path and Path(profiles_path).is_file():
            header = pd.read_csv(profiles_path, nrows=0).columns
            cols = [c for c in ("author_id", "author_name", "article_count") if c in header]
            frame = pd.read_csv(profiles_path, dtype=str, usecols=cols)
            self._profiles = {
                a: ResearcherProfile(a, n if isinstance(n, str) else None)
                for a, n in zip(frame["author_id"], frame["author_name"])
            }
            self._article_counts = _article_counts(frame)
            self.source = str(profiles_path)
        if not self._full_names and authors_path and Path(authors_path).is_file():
            self._full_names = read_full_names(Path(authors_path), set(self._profiles))

    def get(self, author_id: str) -> ResearcherProfile:
        return self._profiles.get(author_id) or ResearcherProfile(author_id)

    def summaries(self) -> list[AuthorSummary]:
        return [
            AuthorSummary(
                p.author_id,
                p.name,
                p.institution,
                p.low_visibility,
                self._article_counts.get(p.author_id),
                self._full_names.get(p.author_id),
            )
            for p in self._profiles.values()
        ]

    def full_name(self, author_id: str) -> str | None:
        return self._full_names.get(author_id)

    def __len__(self) -> int:
        return len(self._profiles)


def _article_counts(frame: pd.DataFrame) -> dict[str, int]:
    """``article_count`` column as ints (missing column or non-numeric values are skipped)."""
    if "article_count" not in frame.columns:
        return {}
    counts = pd.to_numeric(frame["article_count"], errors="coerce")
    return {a: int(c) for a, c in zip(frame["author_id"], counts) if pd.notna(c)}


def read_full_names(authors_path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    """``"Apellido, Nombre"`` per id from Scopus ``authors.csv`` (only ``wanted`` ids when given);
    empty if the first/last name columns are missing."""
    header = pd.read_csv(authors_path, nrows=0).columns
    if not set(RAW_AUTHOR_COLUMNS) <= set(header):
        return {}
    frame = pd.read_csv(authors_path, dtype=str, usecols=list(RAW_AUTHOR_COLUMNS))
    if wanted is not None:
        frame = frame[frame["identifier"].isin(wanted)]
    frame = frame.drop_duplicates("identifier")
    names = {
        a: compose_full_name(first, last)
        for a, first, last in zip(frame["identifier"], frame["first_name"], frame["last_name"])
    }
    return {a: n for a, n in names.items() if n}
