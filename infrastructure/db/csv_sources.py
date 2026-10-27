"""CSV adapters shared by the offline scripts: filtered edges, institutions, and visibility.

Reuses ``read_csv`` and the cleaning/filter steps of
``01_filter_and_merge_authors.py`` so the author population (>= 3 articles) is
exactly the one used for the embeddings.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from domain.entities.research_group import UNKNOWN_INSTITUTION

etl = importlib.import_module("infrastructure.scripts.01_filter_and_merge_authors")

AFFILIATIONS_FILE: Final[str] = "affiliations.csv"
AUTHORS_AFFILIATIONS_FILE: Final[str] = "authors_affiliations.csv"
ARTICLES_AFFILIATIONS_FILE: Final[str] = "articles_affiliations.csv"
AFFILIATION_ID_COL: Final[str] = "identifier"
AFFILIATION_NAME_COL: Final[str] = "affiliation_name"
LINK_AFFILIATION_COL: Final[str] = "affiliation_id"
LINK_AUTHOR_COL: Final[str] = "author_id"
LINK_ARTICLE_COL: Final[str] = "article_id"
VISIBILITY_QUANTILE: Final[float] = 0.25
DataLoadError = etl.DataLoadError


@dataclass(frozen=True, slots=True)
class InstitutionStats:
    """Data-quality counters for the author -> primary institution mapping."""

    authors: int
    missing: int
    multiple: int
    ties_broken: int
    distinct_institutions: int


def load_clean_edges(raw_dir: Path = etl.RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return cleaned article-author edges (all authors) and counts of the >= MIN_ARTICLES survivors."""
    authors = etl.read_csv(raw_dir / etl.AUTHORS_FILE, [etl.AUTHOR_ID_COL, etl.AUTHOR_NAME_COL], [etl.AUTHOR_ID_COL])
    articles = etl.read_csv(
        raw_dir / etl.ARTICLES_FILE,
        [etl.ARTICLE_ID_COL, etl.ARTICLE_TITLE_COL, etl.ARTICLE_ABSTRACT_COL], [etl.ARTICLE_ID_COL],
    )
    edges = etl.read_csv(
        raw_dir / etl.EDGES_FILE, [etl.EDGE_AUTHOR_COL, etl.EDGE_ARTICLE_COL],
        [etl.EDGE_AUTHOR_COL, etl.EDGE_ARTICLE_COL],
    )
    edges = etl.clean_edges(edges, authors, articles)
    survivors = etl.apply_quality_filter(etl.count_articles_per_author(edges))
    return edges.reset_index(drop=True), survivors


def normalize_institution_name(name: str) -> str:
    """Casefold and collapse whitespace so duplicated Scopus ids with one name merge."""
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def primary_institutions(
    author_ids: list[str],
    edges: pd.DataFrame,
    raw_dir: Path = etl.RAW_DIR,
    merge_by_name: bool = True,
) -> tuple[dict[str, str], InstitutionStats]:
    """Map each author to the institution most frequent across their articles.

    For an author with several affiliations, each candidate institution is
    scored by the number of the author's articles that list it in
    ``articles_affiliations.csv``; ties fall back to the lexicographically
    smallest key. Authors without any affiliation get ``UNKNOWN``.
    """
    affiliations = etl.read_csv(raw_dir / AFFILIATIONS_FILE, [AFFILIATION_ID_COL, AFFILIATION_NAME_COL], [AFFILIATION_ID_COL])
    author_links = etl.read_csv(raw_dir / AUTHORS_AFFILIATIONS_FILE, [LINK_AUTHOR_COL, LINK_AFFILIATION_COL],
                                [LINK_AUTHOR_COL, LINK_AFFILIATION_COL])
    article_links = etl.read_csv(raw_dir / ARTICLES_AFFILIATIONS_FILE, [LINK_ARTICLE_COL, LINK_AFFILIATION_COL],
                                 [LINK_ARTICLE_COL, LINK_AFFILIATION_COL])
    names = dict(affiliations.drop_duplicates(AFFILIATION_ID_COL)[[AFFILIATION_ID_COL, AFFILIATION_NAME_COL]]
                 .itertuples(index=False, name=None))

    def key(affiliation_id: str) -> str:
        if not merge_by_name:
            return str(affiliation_id)
        name = names.get(affiliation_id)
        return normalize_institution_name(name) if name else f"id:{affiliation_id}"

    wanted = set(author_ids)
    links = author_links[[LINK_AUTHOR_COL, LINK_AFFILIATION_COL]].dropna().drop_duplicates()
    links = links[links[LINK_AUTHOR_COL].isin(wanted)].copy()
    links["institution"] = links[LINK_AFFILIATION_COL].map(key)
    links = links.drop_duplicates([LINK_AUTHOR_COL, "institution"])

    author_articles = edges[edges[etl.EDGE_AUTHOR_COL].isin(wanted)][[etl.EDGE_AUTHOR_COL, etl.EDGE_ARTICLE_COL]]
    art_inst = article_links[[LINK_ARTICLE_COL, LINK_AFFILIATION_COL]].dropna().copy()
    art_inst["institution"] = art_inst[LINK_AFFILIATION_COL].map(key)
    art_inst = art_inst[[LINK_ARTICLE_COL, "institution"]].drop_duplicates()
    evidence = (
        author_articles.rename(columns={etl.EDGE_AUTHOR_COL: LINK_AUTHOR_COL, etl.EDGE_ARTICLE_COL: LINK_ARTICLE_COL})
        .merge(art_inst, on=LINK_ARTICLE_COL)
        .groupby([LINK_AUTHOR_COL, "institution"]).size().rename("support").reset_index()
    )
    scored = links.merge(evidence, on=[LINK_AUTHOR_COL, "institution"], how="left").fillna({"support": 0})
    scored = scored.sort_values([LINK_AUTHOR_COL, "support", "institution"], ascending=[True, False, True])
    top_support = scored.groupby(LINK_AUTHOR_COL)["support"].transform("max")
    ties = scored[scored["support"] == top_support].groupby(LINK_AUTHOR_COL).size()
    primary = scored.drop_duplicates(LINK_AUTHOR_COL).set_index(LINK_AUTHOR_COL)["institution"].to_dict()

    mapping = {author: primary.get(author, UNKNOWN_INSTITUTION) for author in author_ids}
    per_author = links.groupby(LINK_AUTHOR_COL).size()
    stats = InstitutionStats(
        authors=len(author_ids),
        missing=sum(1 for value in mapping.values() if value == UNKNOWN_INSTITUTION),
        multiple=int((per_author > 1).sum()),
        ties_broken=int((ties > 1).sum()),
        distinct_institutions=len({v for v in mapping.values() if v != UNKNOWN_INSTITUTION}),
    )
    return mapping, stats


def visibility_table(
    edges: pd.DataFrame,
    survivor_counts: pd.Series,
    max_authors_per_article: int | None = 25,
    quantile: float = VISIBILITY_QUANTILE,
) -> pd.DataFrame:
    """Article count, co-authorship degree and the low-visibility (protected) flag.

    Degree is the number of distinct co-authors in the full static co-authorship
    graph (all authors, hyper-authored articles excluded); degree centrality is
    degree / (|V| - 1), equal to ``networkx.degree_centrality``. An author is
    protected when both article count and degree are <= their ``quantile``
    within the filtered population.
    """
    pairs = edges[[etl.EDGE_ARTICLE_COL, etl.EDGE_AUTHOR_COL]].drop_duplicates()
    if max_authors_per_article is not None:
        sizes = pairs.groupby(etl.EDGE_ARTICLE_COL)[etl.EDGE_AUTHOR_COL].transform("size")
        pairs = pairs[sizes <= max_authors_per_article]
    joined = pairs.merge(pairs, on=etl.EDGE_ARTICLE_COL, suffixes=("", "_co"))
    joined = joined[joined[etl.EDGE_AUTHOR_COL] != joined[f"{etl.EDGE_AUTHOR_COL}_co"]]
    degree = joined.drop_duplicates([etl.EDGE_AUTHOR_COL, f"{etl.EDGE_AUTHOR_COL}_co"]) \
        .groupby(etl.EDGE_AUTHOR_COL).size()
    vertex_count = edges[etl.EDGE_AUTHOR_COL].nunique()
    table = survivor_counts.rename("article_count").rename_axis("author_id").reset_index()
    table["coauthor_degree"] = table["author_id"].map(degree).fillna(0).astype(int)
    table["degree_centrality"] = table["coauthor_degree"] / max(vertex_count - 1, 1)
    q_articles = table["article_count"].quantile(quantile)
    q_degree = table["coauthor_degree"].quantile(quantile)
    table["protected"] = (table["article_count"] <= q_articles) & (table["coauthor_degree"] <= q_degree)
    table.attrs["q_articles"] = float(q_articles)
    table.attrs["q_degree"] = float(q_degree)
    return table
