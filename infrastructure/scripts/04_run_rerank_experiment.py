"""Offline experiment: baseline group affinity vs. fair/diverse re-ranking on leave-one-out groups."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from application.services.evaluation_split import sample_cases
from application.services.fair_reranker import RerankConfig, rerank
from application.services.group_scoring import retrieve_top_n, score_all_candidates
from application.services import ranking_metrics as rm
from domain.entities.ranking import Candidate
from domain.entities.research_group import UNKNOWN_INSTITUTION, LeaveOneOutCase
from infrastructure.db.csv_sources import DataLoadError, etl, load_clean_edges, primary_institutions, visibility_table

PROCESSED_DIR: Final[Path] = etl.PROCESSED_DIR
EMBEDDINGS_PATH: Final[Path] = PROCESSED_DIR / "author_embeddings.npy"
MAPPING_PATH: Final[Path] = PROCESSED_DIR / "author_index_mapping.json"
GROUPS_PATH: Final[Path] = PROCESSED_DIR / "ground_truth_groups.csv"
SPLIT_PATH: Final[Path] = PROCESSED_DIR / "loo_eval_split.csv"
RESULTS_DIR: Final[Path] = PROCESSED_DIR / "experiments"
ID_SEPARATOR: Final[str] = "|"
CUTOFFS: Final[tuple[int, ...]] = (10, 20)
SETTINGS: Final[dict[str, RerankConfig]] = {
    "R0_solo_equidad": RerankConfig(lambda_=1.0, max_per_institution=None, min_protected_share=0.2,
                                    relevance_floor=0.5, fairness_mode="prefix"),
    "R1_suave": RerankConfig(lambda_=0.8, max_per_institution=3, min_protected_share=0.2,
                             relevance_floor=0.5, fairness_mode="prefix"),
    "R2_moderado": RerankConfig(lambda_=0.6, max_per_institution=2, min_protected_share=0.3,
                                relevance_floor=0.3, fairness_mode="prefix"),
    "R3_fair_binomial": RerankConfig(lambda_=0.7, max_per_institution=2, min_protected_share=0.3,
                                     relevance_floor=0.3, fairness_mode="fair", fair_alpha=0.1),
}
logger = logging.getLogger("experiments.rerank")


class ExperimentError(RuntimeError):
    """Inputs for the experiment are missing or inconsistent."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the utilitarian mean vs. max")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--exclude-coauthors", action="store_true",
                        help="also exclude frequent co-authors of the query (removes the LOO target by design)")
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--tag", default="", help="suffix for result files")
    parser.add_argument("--leakage-free", action="store_true",
                        help="re-embed query members and target without the articles the target co-signed "
                             "with any query member (needs sentence-transformers)")
    parser.add_argument("--model", default=None, help="model for --leakage-free; defaults to the mapping's model")
    return parser.parse_args(argv)


def load_embeddings(embeddings_path: Path, mapping_path: Path) -> tuple[np.ndarray, list[str], dict]:
    if not embeddings_path.is_file() or not mapping_path.is_file():
        raise ExperimentError(f"Missing {embeddings_path} or {mapping_path}; run 02_generate_embeddings.py")
    matrix = np.load(embeddings_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    ids = mapping["index_to_author_id"]
    if matrix.shape[0] != len(ids):
        raise ExperimentError("Embedding rows and mapping are misaligned")
    return matrix, ids, mapping


def load_cases() -> tuple[list[LeaveOneOutCase], pd.DataFrame]:
    for path in (GROUPS_PATH, SPLIT_PATH):
        if not path.is_file():
            raise ExperimentError(f"Missing {path}; run 03_mine_ground_truth_groups.py")
    groups = pd.read_csv(GROUPS_PATH, dtype={"author_ids": str, "group_id": str})
    split = pd.read_csv(SPLIT_PATH, dtype=str)
    cases = [LeaveOneOutCase(r.group_id, tuple(r.query_author_ids.split(ID_SEPARATOR)), r.target_author_id)
             for r in split.itertuples()]
    return cases, groups


def plan_leakage_free(
    cases: list[LeaveOneOutCase], edges: pd.DataFrame, texts: dict[str, str],
) -> dict[tuple[str, str], list[str]]:
    """Per case, the articles each member and the target keep after removing leakage.

    Removed: every article the target co-signed with any query member (the very
    evidence that defines the ground-truth group). A case is ineligible when the
    target or a member has no remaining article with text.
    """
    by_author = edges.groupby(etl.EDGE_AUTHOR_COL)[etl.EDGE_ARTICLE_COL].agg(set).to_dict()
    plans: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        target_articles = by_author.get(case.target, set())
        shared = {a for m in case.query_members for a in by_author.get(m, set()) & target_articles}
        case_plan = {
            (case.group_id, person): sorted(a for a in by_author.get(person, set()) - shared if texts.get(a))
            for person in (*case.query_members, case.target)
        }
        if all(case_plan.values()):
            plans |= case_plan
    return plans


def encode_plans(
    plans: dict[tuple[str, str], list[str]], texts: dict[str, str], model_name: str,
) -> dict[tuple[str, str], np.ndarray]:
    """Mirror 02_generate_embeddings.py: L2-normalized article vectors, mean pooling, re-normalization."""
    from sentence_transformers import SentenceTransformer

    unique_articles = sorted({a for docs in plans.values() for a in docs})
    logger.info("Leakage-free: encoding %d unique articles", len(unique_articles))
    model = SentenceTransformer(model_name, device="cpu")
    encoded = model.encode([texts[a] for a in unique_articles], batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    position = {a: i for i, a in enumerate(unique_articles)}
    vectors = {}
    for key, docs in plans.items():
        mean = encoded[[position[a] for a in docs]].mean(axis=0)
        vectors[key] = (mean / np.linalg.norm(mean)).astype(np.float32)
    return vectors


def evaluate_list(
    name: str, ranked: list[str], baseline: list[str], target: str, pool: list[str],
    relevance: dict[str, float], institutions: dict[str, str], protected: dict[str, bool], target_share: float,
    population_share: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"method": name}
    relevant = {target}
    for k in CUTOFFS:
        insts = [institutions[a] for a in ranked[:k]]
        exp_p, exp_np = rm.group_exposure(ranked, protected, k)
        row |= {
            f"P@{k}": rm.precision_at_k(ranked, relevant, k),
            f"Recall@{k}": rm.recall_at_k(ranked, relevant, k),
            f"nDCG@{k}": rm.ndcg_at_k(ranked, relevant, k),
            f"HR@{k}": rm.hit_rate_at_k(ranked, relevant, k),
            f"GiniSimpson@{k}": rm.gini_simpson(insts),
            f"GiniSimpson_m2@{k}": rm.anti_tokenism_gini_simpson(insts),
            f"UniqueInst@{k}": rm.unique_institutions(insts),
            f"InstCoverage@{k}": rm.institution_coverage(insts, [institutions[a] for a in pool]),
            f"ProtectedShare@{k}": rm.protected_share(ranked, protected, k),
            f"ExposureShareProt@{k}": rm.exposure_share(ranked, protected, k),
            f"MeanExpProt@{k}": exp_p,
            f"MeanExpNonProt@{k}": exp_np,
            f"NDKL@{k}": rm.ndkl(ranked, protected, k, target_share),
            f"NDKLpop@{k}": rm.ndkl(ranked, protected, k, population_share),
            f"RelRetained@{k}": rm.relevance_retained(ranked, baseline, relevance, k),
        }
    return row


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    matrix, ids, mapping_meta = load_embeddings(args.embeddings, args.mapping)
    row_of = {author: row for row, author in enumerate(ids)}
    cases, groups = load_cases()
    edges, survivors = load_clean_edges()
    institutions, inst_stats = primary_institutions(list(survivors.index), edges)
    vis = visibility_table(edges, survivors)
    protected = dict(zip(vis["author_id"], vis["protected"].astype(bool)))
    population_share = float(vis["protected"].mean())

    coauthors: dict[str, set[str]] = {}
    for members in groups["author_ids"]:
        people = members.split(ID_SEPARATOR)
        for person in people:
            coauthors.setdefault(person, set()).update(people)

    usable = [c for c in cases if c.target in row_of and all(m in row_of for m in c.query_members)]
    replacements: dict[tuple[str, str], np.ndarray] = {}
    leakage_ineligible = 0
    if args.leakage_free:
        articles = etl.read_csv(etl.RAW_DIR / etl.ARTICLES_FILE,
                                [etl.ARTICLE_ID_COL, etl.ARTICLE_TITLE_COL, etl.ARTICLE_ABSTRACT_COL],
                                [etl.ARTICLE_ID_COL])
        texts = etl.build_article_texts(articles).set_index(etl.EDGE_ARTICLE_COL)["article_text"].to_dict()
        plans = plan_leakage_free(usable, edges, texts)
        eligible = [c for c in usable if (c.group_id, c.target) in plans]
        leakage_ineligible = len(usable) - len(eligible)
        usable = eligible
    sample = sample_cases(usable, args.n_cases, args.seed)
    logger.info("Cases: %d total, %d usable, %d sampled", len(cases), len(usable), len(sample))
    if args.leakage_free:
        keys = {(c.group_id, p) for c in sample for p in (*c.query_members, c.target)}
        replacements = encode_plans({k: plans[k] for k in keys}, texts, args.model or mapping_meta.get("model_name"))
        matrix = matrix.copy()

    rows: list[dict] = []
    diagnostics: list[dict] = []
    k_max = max(CUTOFFS)
    for case in sample:
        member_rows = [row_of[m] for m in case.query_members]
        originals = {}
        if args.leakage_free:
            for person in (*case.query_members, case.target):
                originals[row_of[person]] = matrix[row_of[person]].copy()
                matrix[row_of[person]] = replacements[(case.group_id, person)]
        scores = score_all_candidates(matrix, member_rows, alpha=args.alpha)
        for row, vector in originals.items():
            matrix[row] = vector
        excluded = set(member_rows)
        if args.exclude_coauthors:
            excluded |= {row_of[a] for m in case.query_members for a in coauthors.get(m, ()) if a in row_of}
        top = retrieve_top_n(scores, excluded, args.top_n)
        pool = [ids[r] for r, _ in top]
        relevance = {ids[r]: s for r, s in top}
        candidates = [Candidate(a, relevance[a], institutions.get(a, UNKNOWN_INSTITUTION), protected.get(a, False))
                      for a in pool]
        target_share = sum(c.protected for c in candidates) / len(candidates)
        query_insts = [institutions.get(m, UNKNOWN_INSTITUTION) for m in case.query_members]
        baseline = pool[:k_max]
        common = dict(baseline=baseline, target=case.target, pool=pool, relevance=relevance,
                      institutions=institutions, protected=protected, target_share=target_share,
                      population_share=population_share)
        base_row = evaluate_list("baseline", baseline, **common)
        base_row |= {"group_id": case.group_id, "group_size": len(case.query_members) + 1,
                     "target_in_pool": case.target in relevance, "pool_protected_share": target_share}
        rows.append(base_row)
        for name, config in SETTINGS.items():
            result = rerank(candidates, k_max, config, query_insts)
            row = evaluate_list(name, result.author_ids, **common)
            row |= {"group_id": case.group_id, "group_size": len(case.query_members) + 1,
                    "target_in_pool": case.target in relevance, "pool_protected_share": target_share,
                    "quota_shortfalls": result.quota_shortfalls, "cap_relaxations": result.cap_relaxations,
                    "quota_picks": sum(1 for item in result.items if item.reason == "quota")}
            rows.append(row)
        diagnostics.append({"group_id": case.group_id, "target": case.target,
                            "target_rank_in_pool": pool.index(case.target) + 1 if case.target in relevance else None})

    per_case = pd.DataFrame(rows)
    metric_cols = [c for c in per_case.columns if "@" in c] + ["quota_shortfalls", "cap_relaxations", "quota_picks"]
    summary = per_case.groupby("method", sort=False)[metric_cols].mean().round(4)
    summary.insert(0, "n_cases", per_case.groupby("method", sort=False).size())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    per_case.to_csv(RESULTS_DIR / f"rerank_per_case{tag}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"rerank_summary{tag}.csv")
    target_ranks = pd.Series([d["target_rank_in_pool"] for d in diagnostics], dtype="float")
    meta = {
        "embedding_model": mapping_meta.get("model_name"), "pooling": mapping_meta.get("pooling"),
        "alpha": args.alpha, "top_n": args.top_n, "seed": args.seed, "n_cases": len(sample),
        "cases_total": len(cases), "cases_usable": len(usable), "exclude_coauthors": args.exclude_coauthors,
        "leakage_free": args.leakage_free, "leakage_free_ineligible_cases": leakage_ineligible,
        "protected_population_share": round(population_share, 4),
        "protected_thresholds": {"article_count_le": vis.attrs["q_articles"], "coauthor_degree_le": vis.attrs["q_degree"]},
        "target_in_top_n_share": round(float(target_ranks.notna().mean()), 4),
        "target_rank_median_when_found": float(target_ranks.median()) if target_ranks.notna().any() else None,
        "institutions": asdict(inst_stats),
        "settings": {name: asdict(cfg) for name, cfg in SETTINGS.items()},
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (RESULTS_DIR / f"rerank_meta{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_markdown(summary, meta, RESULTS_DIR / f"rerank_results{tag}.md")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.T)
    print(json.dumps({k: v for k, v in meta.items() if k != "settings"}, indent=2))
    return 0


def write_markdown(summary: pd.DataFrame, meta: dict, path: Path) -> None:
    keys = ["nDCG", "P", "Recall", "HR", "GiniSimpson", "GiniSimpson_m2", "UniqueInst", "InstCoverage",
            "ProtectedShare", "ExposureShareProt", "MeanExpProt", "MeanExpNonProt", "NDKL", "NDKLpop", "RelRetained"]
    lines = [f"# Re-ranking experiment ({meta['n_cases']} LOO cases, seed {meta['seed']}, "
             f"alpha={meta['alpha']}, Top-N={meta['top_n']}, model {meta['embedding_model']}, "
             f"leakage_free={meta['leakage_free']})", "",
             "NDKL uses the protected share of each query's Top-N pool as target; NDKLpop uses the "
             f"population share ({meta['protected_population_share']}).", ""]
    for k in CUTOFFS:
        cols = [f"{key}@{k}" for key in keys]
        lines.append(f"## @{k}")
        lines.append("")
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for method, row in summary.iterrows():
            lines.append(f"| {method} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")
        lines.append("")
    lines.append("Constraint diagnostics (mean per query): quota_picks, quota_shortfalls, cap_relaxations")
    lines.append("")
    for method, row in summary.iterrows():
        lines.append(f"- {method}: quota_picks={row['quota_picks']:.2f}, quota_shortfalls={row['quota_shortfalls']:.2f}, "
                     f"cap_relaxations={row['cap_relaxations']:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(parse_args(argv))
    except (DataLoadError, ExperimentError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
