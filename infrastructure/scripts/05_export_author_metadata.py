"""Export author_metadata.csv (name, primary institution, low-visibility flag) for the API.

Uses the same institution mapping and visibility rule as 04_run_rerank_experiment.py.
Run: python -m infrastructure.scripts.05_export_author_metadata
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from domain.entities.research_group import UNKNOWN_INSTITUTION
from infrastructure.db.csv_sources import DataLoadError, etl, load_clean_edges, primary_institutions, visibility_table

DEFAULT_OUTPUT = etl.PROCESSED_DIR / "author_metadata.csv"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        edges, survivors = load_clean_edges()
        authors = etl.read_csv(etl.RAW_DIR / etl.AUTHORS_FILE, [etl.AUTHOR_ID_COL, etl.AUTHOR_NAME_COL],
                               [etl.AUTHOR_ID_COL])
    except DataLoadError as exc:
        logging.error("%s", exc)
        return 1
    ids = list(survivors.index)
    institutions, stats = primary_institutions(ids, edges)
    vis = visibility_table(edges, survivors).set_index("author_id")
    names = authors.drop_duplicates(etl.AUTHOR_ID_COL).set_index(etl.AUTHOR_ID_COL)[etl.AUTHOR_NAME_COL]
    out = pd.DataFrame({
        "author_id": ids,
        "author_name": [names.get(a) for a in ids],
        "institution": [institutions.get(a, UNKNOWN_INSTITUTION) for a in ids],
        "low_visibility": [bool(vis.at[a, "protected"]) for a in ids],
        "article_count": [int(vis.at[a, "article_count"]) for a in ids],
        "coauthor_degree": [int(vis.at[a, "coauthor_degree"]) for a in ids],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logging.info("Wrote %d authors to %s (%s)", len(out), args.output, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
