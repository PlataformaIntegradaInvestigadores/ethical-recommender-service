"""API settings from environment variables (defaults point to data/processed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
# Angular dev server (centinela-front, ``ng serve``). Override with RECO_CORS_ORIGINS="a,b"; "" disables CORS.
DEFAULT_CORS_ORIGINS = ("http://localhost:4200", "http://127.0.0.1:4200")


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True, slots=True)
class Settings:
    embeddings_path: Path
    mapping_path: Path
    metadata_path: Path
    profiles_path: Path
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embeddings_path=_path("RECO_EMBEDDINGS_PATH", PROCESSED / "author_embeddings.npy"),
            mapping_path=_path("RECO_MAPPING_PATH", PROCESSED / "author_index_mapping.json"),
            metadata_path=_path("RECO_METADATA_PATH", PROCESSED / "author_metadata.csv"),
            profiles_path=_path("RECO_PROFILES_PATH", PROCESSED / "filtered_authors_profiles.csv"),
            cors_origins=_origins(os.environ.get("RECO_CORS_ORIGINS")),
        )


def _origins(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_CORS_ORIGINS
    return tuple(o.strip().rstrip("/") for o in value.split(",") if o.strip())
