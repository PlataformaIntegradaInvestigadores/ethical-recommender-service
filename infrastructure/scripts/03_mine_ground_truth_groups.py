"""Mine frequent co-author groups (Apriori + tidsets) as ground truth and build the LOO split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Final

import pandas as pd

from application.services.apriori_groups import (
    DEFAULT_MAX_AUTHORS_PER_ARTICLE,
    DEFAULT_MAX_K,
    DEFAULT_MIN_SUPPORT,
    AprioriConfig,
    mine_frequent_itemsets,
)
from application.services.evaluation_split import leave_one_out_cases
from application.services.ranking_metrics import anti_tokenism_gini_simpson, gini_simpson, unique_institutions
from infrastructure.db.csv_sources import DataLoadError, etl, load_clean_edges, primary_institutions

PROCESSED_DIR: Final[Path] = etl.PROCESSED_DIR
GROUPS_PATH: Final[Path] = PROCESSED_DIR / "ground_truth_groups.csv"
SPLIT_PATH: Final[Path] = PROCESSED_DIR / "loo_eval_split.csv"
STATS_PATH: Final[Path] = PROCESSED_DIR / "ground_truth_groups_stats.json"
ID_SEPARATOR: Final[str] = "|"
OUTPUT_COLUMNS: Final[list[str]] = [
    "group_id",
    "size",
    "author_ids",
    "support",
    "article_ids",
    "maximal",
    "closed",
    "truncated",
    "n_institutions",
    "gini_simpson",
    "gini_simpson_m2",
]
logger = logging.getLogger("etl.mine_ground_truth_groups")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
        help="largest group size to mine; 0 = unbounded (until a level is empty)",
    )
    parser.add_argument(
        "--max-authors-per-article",
        type=int,
        default=DEFAULT_MAX_AUTHORS_PER_ARTICLE,
        help="ignore hyper-authored articles above this size; 0 disables the guard",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-suffix", default="", help="suffix for output file names, e.g. _k5")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    suffix = args.output_suffix
    groups_path = GROUPS_PATH.with_name(f"{GROUPS_PATH.stem}{suffix}.csv")
    split_path = SPLIT_PATH.with_name(f"{SPLIT_PATH.stem}{suffix}.csv")
    stats_path = STATS_PATH.with_name(f"{STATS_PATH.stem}{suffix}.json")
    load_started = time.perf_counter()
    edges, survivors = load_clean_edges()
    transactions = edges.groupby(etl.EDGE_ARTICLE_COL)[etl.EDGE_AUTHOR_COL].agg(list).to_dict()
    load_seconds = time.perf_counter() - load_started
    config = AprioriConfig(
        min_support=args.min_support,
        max_k=args.max_k or None,
        max_authors_per_article=args.max_authors_per_article or None,
    )
    result = mine_frequent_itemsets(transactions, config, allowed_items=survivors.index)
    institutions, inst_stats = primary_institutions(list(survivors.index), edges)

    rows = []
    for number, group in enumerate(result.groups, start=1):
        insts = [institutions.get(member, "UNKNOWN") for member in group.members]
        rows.append(
            {
                "group_id": f"G{number:06d}",
                "size": group.size,
                "author_ids": ID_SEPARATOR.join(group.members),
                "support": group.support,
                "article_ids": ID_SEPARATOR.join(sorted(group.article_ids)),
                "maximal": group.maximal,
                "closed": group.closed,
                "truncated": result.truncated_at_k is not None and group.size == result.truncated_at_k,
                "n_institutions": unique_institutions(insts),
                "gini_simpson": round(gini_simpson(insts), 6),
                "gini_simpson_m2": round(anti_tokenism_gini_simpson(insts), 6),
            }
        )
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(groups_path, index=False, encoding="utf-8")

    maximal = frame[frame["maximal"]]
    cases = leave_one_out_cases(
        ((row.group_id, tuple(row.author_ids.split(ID_SEPARATOR))) for row in maximal.itertuples()),
        min_group_size=3,
        seed=args.seed,
    )
    pd.DataFrame(
        [
            {
                "group_id": c.group_id,
                "query_author_ids": ID_SEPARATOR.join(c.query_members),
                "target_author_id": c.target,
            }
            for c in cases
        ]
    ).to_csv(split_path, index=False, encoding="utf-8")

    n_items = len(result.levels.get(1, {}))
    stats = {
        "config": {
            "min_support": config.min_support,
            "max_k": config.max_k,
            "max_authors_per_article": config.max_authors_per_article,
        },
        "items_allowed": int(len(survivors)),
        "transactions_total": result.transactions_total,
        "transactions_dropped_hyperauthored": result.transactions_dropped_hyperauthored,
        "load_seconds": round(load_seconds, 3),
        "mining_seconds": round(result.runtime_seconds, 3),
        "naive_pairs_c2": n_items * (n_items - 1) // 2,
        "levels": [
            {
                "k": s.k,
                "candidates": s.candidates,
                "pruned_by_subset": s.pruned_by_subset,
                "frequent": s.frequent,
                "seconds": round(s.seconds, 4),
            }
            for s in result.stats
        ],
        "truncated_at_k": result.truncated_at_k,
        "groups_total": len(frame),
        "maximal_by_size": {int(k): int(v) for k, v in maximal["size"].value_counts().sort_index().items()},
        "closed_by_size": {
            int(k): int(v) for k, v in frame[frame["closed"]]["size"].value_counts().sort_index().items()
        },
        "maximal_mean_support": round(float(maximal["support"].mean()), 3) if len(maximal) else 0.0,
        "maximal_mean_gini_simpson": round(float(maximal["gini_simpson"].mean()), 4) if len(maximal) else 0.0,
        "maximal_mean_gini_simpson_m2": round(float(maximal["gini_simpson_m2"].mean()), 4) if len(maximal) else 0.0,
        "maximal_single_institution_share": (
            round(float((maximal["n_institutions"] <= 1).mean()), 4) if len(maximal) else 0.0
        ),
        "maximal_authors_covered": len({a for ids in maximal["author_ids"] for a in ids.split(ID_SEPARATOR)}),
        "loo_cases": len(cases),
        "institutions": {
            "authors": inst_stats.authors,
            "missing": inst_stats.missing,
            "missing_pct": round(100 * inst_stats.missing / max(inst_stats.authors, 1), 3),
            "multiple_affiliations": inst_stats.multiple,
            "ties_broken": inst_stats.ties_broken,
            "distinct_institutions": inst_stats.distinct_institutions,
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Articles (transactions): {result.transactions_total:,}; dropped hyper-authored "
        f"(> {config.max_authors_per_article} authors): {result.transactions_dropped_hyperauthored:,}"
    )
    print(
        f"Items allowed (authors >= {etl.MIN_ARTICLES} articles): {len(survivors):,}; "
        f"naive |C2| would be {stats['naive_pairs_c2']:,}"
    )
    for s in result.stats:
        print(
            f"  L{s.k}: candidates={s.candidates:,} pruned={s.pruned_by_subset:,} "
            f"frequent={s.frequent:,} ({s.seconds:.3f}s)"
        )
    print(f"Mining runtime: {result.runtime_seconds:.2f}s (load {load_seconds:.2f}s)")
    if result.truncated_at_k is not None:
        print(
            f"WARNING: level {result.truncated_at_k} is non-empty; its 'maximal' flags only hold within max_k "
            "(rerun with --max-k 0 for true maximal groups)"
        )
    print(f"Groups (k>=2): {len(frame):,}; maximal by size: {stats['maximal_by_size']}")
    print(
        f"Maximal groups mean Gini-Simpson: {stats['maximal_mean_gini_simpson']} "
        f"(m>=2: {stats['maximal_mean_gini_simpson_m2']}); single-institution share "
        f"{stats['maximal_single_institution_share']:.1%}"
    )
    print(f"Leave-one-out cases (maximal, size>=3): {len(cases):,}")
    print(f"Outputs: {groups_path}, {split_path}, {stats_path}")
    sizes = Counter(len(c.query_members) + 1 for c in cases)
    logger.info("LOO cases by group size: %s", dict(sorted(sizes.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(parse_args(argv))
    except DataLoadError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
