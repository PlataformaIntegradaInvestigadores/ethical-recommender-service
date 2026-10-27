"""Filter authors by publication count and build their semantic text profiles."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Final

import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH: Final[Path] = PROCESSED_DIR / "filtered_authors_profiles.csv"
MIN_ARTICLES: Final[int] = 3
ARTICLE_SEPARATOR: Final[str] = "\n\n"
AUTHORS_FILE: Final[str] = "authors.csv"
ARTICLES_FILE: Final[str] = "articles.csv"
EDGES_FILE: Final[str] = "articles_authors.csv"
AUTHOR_ID_COL: Final[str] = "identifier"
AUTHOR_NAME_COL: Final[str] = "auth_name"
ARTICLE_ID_COL: Final[str] = "identifier"
ARTICLE_TITLE_COL: Final[str] = "title"
ARTICLE_ABSTRACT_COL: Final[str] = "abstract"
EDGE_AUTHOR_COL: Final[str] = "author_id"
EDGE_ARTICLE_COL: Final[str] = "article_id"
OUTPUT_COLUMNS: Final[list[str]] = [
    "author_id",
    "author_name",
    "article_count",
    "concatenated_profile_text",
]
logger = logging.getLogger("etl.filter_and_merge_authors")


class DataLoadError(RuntimeError):
    """A source file is missing, unreadable, or has an unexpected schema."""


def read_csv(path: Path, required_columns: list[str], id_columns: list[str]) -> pd.DataFrame:
    """Read standard CSV and exports with an extra quoting layer around a row.

    The repository's articles export mixes both formats. Decode only single-field
    records, then validate their width to avoid silently shifting article fields.
    All fields remain strings, including identifiers such as '001' and 'NA'.
    """
    if not path.is_file():
        raise DataLoadError(f"Input file not found: {path}")
    try:
        csv.field_size_limit(10_000_000)
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            headers = next(reader, None)
            if not headers:
                raise DataLoadError(f"Empty CSV file: {path}")
            missing = [column for column in required_columns if column not in headers]
            if missing:
                raise DataLoadError(f"{path.name} is missing columns {missing}. Available columns: {headers}")
            if len(set(headers)) != len(headers):
                raise DataLoadError(f"{path.name} has duplicate column names")
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) == 1 and len(headers) > 1:
                    row = next(csv.reader([row[0]], strict=True))
                if len(row) != len(headers):
                    raise DataLoadError(
                        f"{path.name}, line {reader.line_num}: expected {len(headers)} " f"fields, found {len(row)}"
                    )
                rows.append(row)
        frame = pd.DataFrame(rows, columns=headers, dtype=str)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DataLoadError(f"Could not read {path}: {exc}") from exc
    for column in id_columns:
        frame[column] = frame[column].replace("", pd.NA)
    return frame


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load authors, articles, and article-author edges from data/raw."""
    authors = read_csv(
        RAW_DIR / AUTHORS_FILE,
        [AUTHOR_ID_COL, AUTHOR_NAME_COL],
        [AUTHOR_ID_COL],
    )
    articles = read_csv(
        RAW_DIR / ARTICLES_FILE,
        [ARTICLE_ID_COL, ARTICLE_TITLE_COL, ARTICLE_ABSTRACT_COL],
        [ARTICLE_ID_COL],
    )
    edges = read_csv(
        RAW_DIR / EDGES_FILE,
        [EDGE_AUTHOR_COL, EDGE_ARTICLE_COL],
        [EDGE_AUTHOR_COL, EDGE_ARTICLE_COL],
    )
    return authors, articles, edges


def build_article_texts(articles: pd.DataFrame) -> pd.DataFrame:
    """Return one title-plus-abstract text per article."""
    title = articles[ARTICLE_TITLE_COL].fillna("").astype(str).str.strip()
    abstract = articles[ARTICLE_ABSTRACT_COL].fillna("").astype(str).str.strip()
    texts = pd.DataFrame(
        {
            EDGE_ARTICLE_COL: articles[ARTICLE_ID_COL],
            "article_text": (title + "\n" + abstract).str.strip(),
        }
    )
    return texts.drop_duplicates(subset=EDGE_ARTICLE_COL)


def clean_edges(edges: pd.DataFrame, authors: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Drop null, duplicate, and orphan edges before counting."""
    cleaned = edges[[EDGE_AUTHOR_COL, EDGE_ARTICLE_COL]].dropna().drop_duplicates()
    known_authors = cleaned[EDGE_AUTHOR_COL].isin(authors[AUTHOR_ID_COL].dropna())
    known_articles = cleaned[EDGE_ARTICLE_COL].isin(articles[ARTICLE_ID_COL].dropna())
    return cleaned[known_authors & known_articles]


def count_articles_per_author(edges: pd.DataFrame) -> pd.Series:
    """Count distinct valid articles per author."""
    return edges.groupby(EDGE_AUTHOR_COL)[EDGE_ARTICLE_COL].nunique().rename("article_count")


def apply_quality_filter(counts: pd.Series, min_articles: int = MIN_ARTICLES) -> pd.Series:
    """Keep authors with at least min_articles publications."""
    return counts[counts >= min_articles]


def build_profiles(
    surviving_counts: pd.Series,
    edges: pd.DataFrame,
    article_texts: pd.DataFrame,
    authors: pd.DataFrame,
) -> pd.DataFrame:
    """Attach counts and names to deterministically concatenated article texts."""
    surviving_edges = edges[edges[EDGE_AUTHOR_COL].isin(surviving_counts.index)]
    merged = surviving_edges.merge(article_texts, on=EDGE_ARTICLE_COL, how="inner")
    merged = merged[merged["article_text"] != ""].sort_values([EDGE_AUTHOR_COL, EDGE_ARTICLE_COL])
    profiles = (
        merged.groupby(EDGE_AUTHOR_COL)["article_text"]
        .agg(ARTICLE_SEPARATOR.join)
        .rename("concatenated_profile_text")
        .reset_index()
        .rename(columns={EDGE_AUTHOR_COL: "author_id"})
    )
    names = (
        authors[[AUTHOR_ID_COL, AUTHOR_NAME_COL]]
        .drop_duplicates(subset=AUTHOR_ID_COL)
        .rename(columns={AUTHOR_ID_COL: "author_id", AUTHOR_NAME_COL: "author_name"})
    )
    counts = surviving_counts.rename_axis("author_id").reset_index()
    result = counts.merge(names, on="author_id", how="left").merge(profiles, on="author_id", how="left")
    result["concatenated_profile_text"] = result["concatenated_profile_text"].fillna("")
    return result[OUTPUT_COLUMNS].sort_values("author_id").reset_index(drop=True)


def export_profiles(profiles: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Export UTF-8 CSV, creating the destination directory when needed."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiles.to_csv(output_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise DataLoadError(f"Could not write output file {output_path}: {exc}") from exc


def print_statistics(initial_authors: int, surviving_authors: int) -> None:
    """Print before/after author counts and retention."""
    retention = surviving_authors / initial_authors * 100 if initial_authors else 0.0
    print(f"Initial authors count:   {initial_authors:,}")
    print(f"Surviving authors count: {surviving_authors:,} (>= {MIN_ARTICLES} articles)")
    print(f"Dropped authors:         {initial_authors - surviving_authors:,}")
    print(f"Retention rate:          {retention:.2f}%")


def run() -> int:
    """Execute the pipeline and return its process exit code."""
    authors, articles, edges = load_sources()
    edges = clean_edges(edges, authors, articles)
    surviving_counts = apply_quality_filter(count_articles_per_author(edges))
    surviving_edges = edges[edges[EDGE_AUTHOR_COL].isin(surviving_counts.index)]
    surviving_articles = articles[articles[ARTICLE_ID_COL].isin(surviving_edges[EDGE_ARTICLE_COL])]
    profiles = build_profiles(
        surviving_counts,
        surviving_edges,
        build_article_texts(surviving_articles),
        authors,
    )
    export_profiles(profiles)
    print_statistics(authors[AUTHOR_ID_COL].nunique(), len(profiles))
    logger.info("Profiles written to %s", OUTPUT_PATH)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run()
    except DataLoadError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
