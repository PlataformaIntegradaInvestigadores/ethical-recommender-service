"""Offline experiment: baseline group affinity vs. fair/diverse re-ranking on leave-one-out groups."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

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
LEGACY_EMBEDDINGS_PATH: Final[Path] = PROCESSED_DIR / "author_embeddings.npy"
LEGACY_MAPPING_PATH: Final[Path] = PROCESSED_DIR / "author_index_mapping.json"
# Aliases and naming follow 02_generate_embeddings.py (feat/models-comparative-study):
# author_embeddings_{alias}.npy + author_index_mapping_{alias}.json.
MODEL_IDS: Final[dict[str, str]] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "specter2": "allenai/specter2_base",
    "scincl": "malteos/scincl",
}
DEFAULT_MODEL_ALIAS: Final[str] = "minilm"
# Legacy mappings (pre comparative study) only store model_name/pooling: assume the MiniLM run.
LEGACY_MINILM_SPEC: Final[dict[str, Any]] = {
    "token_pooling": "mean", "max_seq_length": 256, "input_format": "raw",
}
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


@dataclass(frozen=True)
class EncoderSpec:
    """How the author embeddings were produced, so leakage-free re-encoding uses the same encoder."""

    alias: str
    model_id: str
    token_pooling: str
    max_seq_length: int
    input_format: str
    dimension: int


def embedding_paths(alias: str, processed_dir: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    """Model-specific artifact names written by 02_generate_embeddings.py --model {alias}."""
    return (processed_dir / f"author_embeddings_{alias}.npy",
            processed_dir / f"author_index_mapping_{alias}.json")


def resolve_embedding_paths(
    alias: str, embeddings: Path | None = None, mapping: Path | None = None, processed_dir: Path = PROCESSED_DIR,
) -> tuple[Path, Path]:
    """Explicit overrides win; otherwise suffixed files, with legacy names as MiniLM fallback."""
    suffixed_embeddings, suffixed_mapping = embedding_paths(alias, processed_dir)
    chosen_embeddings = embeddings or suffixed_embeddings
    chosen_mapping = mapping or suffixed_mapping
    legacy_embeddings = processed_dir / LEGACY_EMBEDDINGS_PATH.name
    legacy_mapping = processed_dir / LEGACY_MAPPING_PATH.name
    if alias == DEFAULT_MODEL_ALIAS:
        if embeddings is None and not chosen_embeddings.is_file() and legacy_embeddings.is_file():
            logger.info("%s not found; using legacy %s (MiniLM run).", chosen_embeddings.name, legacy_embeddings)
            chosen_embeddings = legacy_embeddings
        if mapping is None and not chosen_mapping.is_file() and legacy_mapping.is_file():
            logger.info("%s not found; using legacy %s (MiniLM run).", chosen_mapping.name, legacy_mapping)
            chosen_mapping = legacy_mapping
    missing = [str(path) for path in (chosen_embeddings, chosen_mapping) if not path.is_file()]
    if missing:
        raise ExperimentError(
            f"Missing embedding artifacts for model '{alias}': {', '.join(missing)}. Generate them with: "
            f"python infrastructure/scripts/02_generate_embeddings.py --model {alias}"
        )
    return chosen_embeddings, chosen_mapping


def result_paths(results_dir: Path, alias: str, tag: str = "") -> dict[str, Path]:
    """Experiment outputs carry the model alias so runs per model never overwrite each other."""
    stem = f"{alias}{tag}"
    return {
        "per_case": results_dir / f"rerank_per_case_{stem}.csv",
        "summary": results_dir / f"rerank_summary_{stem}.csv",
        "meta": results_dir / f"rerank_meta_{stem}.json",
        "markdown": results_dir / f"rerank_results_{stem}.md",
    }


def encoder_spec(alias: str, mapping: dict, dimension: int) -> EncoderSpec:
    """Read model id, pooling, max length and input format from the mapping (legacy -> MiniLM defaults)."""
    model_id = mapping.get("model_name")
    stored_alias = mapping.get("model_alias")
    if stored_alias is not None and stored_alias != alias:
        raise ExperimentError(f"Mapping was produced with model '{stored_alias}', but --model {alias} was given.")
    if model_id != MODEL_IDS[alias]:
        raise ExperimentError(
            f"Mapping model '{model_id}' does not match --model {alias} ({MODEL_IDS[alias]})."
        )
    if stored_alias is None:
        if alias != DEFAULT_MODEL_ALIAS:
            raise ExperimentError(f"Mapping has no model metadata; regenerate it with 02 --model {alias}.")
        defaults = LEGACY_MINILM_SPEC
    else:
        defaults = {}
    try:
        return EncoderSpec(
            alias=alias, model_id=model_id,
            token_pooling=str(mapping.get("token_pooling", defaults.get("token_pooling"))),
            max_seq_length=int(mapping.get("max_seq_length", defaults.get("max_seq_length"))),
            input_format=str(mapping.get("input_format", defaults.get("input_format"))),
            dimension=dimension,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"Mapping lacks encoder metadata (token_pooling/max_seq_length/input_format): {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the utilitarian mean vs. max")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--exclude-coauthors", action="store_true",
                        help="also exclude frequent co-authors of the query (removes the LOO target by design)")
    parser.add_argument("--model", choices=sorted(MODEL_IDS), default=DEFAULT_MODEL_ALIAS,
                        help="embedding model alias from 02_generate_embeddings.py (default: %(default)s)")
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="override: embeddings .npy (default: author_embeddings_{model}.npy)")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="override: mapping .json (default: author_index_mapping_{model}.json)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tag", default="", help="suffix for result files, after the model alias")
    parser.add_argument("--leakage-free", action="store_true",
                        help="re-embed query members and target without the articles the target co-signed "
                             "with any query member, with the same encoder as --model (needs sentence-transformers)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="device for --leakage-free re-encoding (default: %(default)s)")
    return parser.parse_args(argv)


def load_embeddings(embeddings_path: Path, mapping_path: Path) -> tuple[np.ndarray, list[str], dict]:
    if not embeddings_path.is_file() or not mapping_path.is_file():
        raise ExperimentError(f"Missing {embeddings_path} or {mapping_path}; run 02_generate_embeddings.py")
    matrix = np.load(embeddings_path)
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        ids = mapping["index_to_author_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExperimentError(f"Invalid mapping {mapping_path}: {exc}") from exc
    if matrix.ndim != 2:
        raise ExperimentError(f"{embeddings_path.name} must be a 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(ids):
        raise ExperimentError(
            f"Embedding rows and mapping are misaligned: {matrix.shape[0]} rows vs {len(ids)} author ids"
        )
    if len(set(ids)) != len(ids):
        raise ExperimentError(f"{mapping_path.name} contains duplicated author ids")
    declared = mapping.get("embedding_dim")
    if declared is not None and int(declared) != matrix.shape[1]:
        raise ExperimentError(
            f"Dimension mismatch: {embeddings_path.name} has {matrix.shape[1]} columns, mapping declares {declared}"
        )
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


def format_article(text: str, input_format: str, sep_token: str | None) -> str:
    """Same model input as 02_generate_embeddings.py: raw 'title\nabstract' or title + [SEP] + abstract."""
    if input_format == "raw":
        return text
    if input_format != "title_sep_abstract":
        raise ExperimentError(f"Unknown input format '{input_format}'")
    if not sep_token:
        raise ExperimentError("Tokenizer has no sep_token for title/abstract concatenation.")
    title, _, abstract = text.partition("\n")
    return title.strip() + sep_token + abstract.strip()


def resolve_device(requested: str) -> str:
    import torch

    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise ExperimentError("--device cuda was requested but CUDA is not available")
    return "cuda" if requested != "cpu" and available else "cpu"


def load_encoder(spec: EncoderSpec, device: str) -> Any:
    """Build the encoder exactly as 02 does: hub config for mean pooling, Transformer + Pooling(cls) otherwise."""
    from sentence_transformers import SentenceTransformer

    if spec.token_pooling == "cls":
        try:
            from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
        except ImportError:  # sentence-transformers < 6
            from sentence_transformers.models import Pooling, Transformer
        transformer = Transformer(spec.model_id, max_seq_length=spec.max_seq_length)
        model = SentenceTransformer(modules=[transformer, Pooling(spec.dimension, pooling_mode="cls")], device=device)
    elif spec.token_pooling == "mean":
        model = SentenceTransformer(spec.model_id, device=device)
    else:
        raise ExperimentError(f"Unsupported token pooling '{spec.token_pooling}'")
    model.max_seq_length = spec.max_seq_length
    model.to(device)
    model.eval()
    return model


def encode_plans(
    plans: dict[tuple[str, str], list[str]], texts: dict[str, str], spec: EncoderSpec, device: str = "cpu",
    model: Any | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Mirror 02_generate_embeddings.py: L2-normalized article vectors, mean pooling, re-normalization."""
    unique_articles = sorted({a for docs in plans.values() for a in docs})
    logger.info("Leakage-free: encoding %d unique articles with %s on %s", len(unique_articles), spec.model_id, device)
    model = model or load_encoder(spec, device)
    sep_token = getattr(getattr(model, "tokenizer", None), "sep_token", None)
    inputs = [format_article(texts[a], spec.input_format, sep_token) for a in unique_articles]
    batch_size = 64 if spec.max_seq_length <= 256 else 16
    encoded = np.asarray(model.encode(inputs, batch_size=batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    if encoded.shape[1] != spec.dimension:
        raise ExperimentError(f"Encoder returned dim {encoded.shape[1]}, embeddings have {spec.dimension}")
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
    embeddings_path, mapping_path = resolve_embedding_paths(args.model, args.embeddings, args.mapping)
    matrix, ids, mapping_meta = load_embeddings(embeddings_path, mapping_path)
    spec = encoder_spec(args.model, mapping_meta, matrix.shape[1])
    logger.info("Model %s (%s), dim %d, from %s", spec.alias, spec.model_id, spec.dimension, embeddings_path)
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
        replacements = encode_plans({k: plans[k] for k in keys}, texts, spec, resolve_device(args.device))
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

    outputs = result_paths(args.results_dir, spec.alias, args.tag)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(outputs["per_case"], index=False)
    summary.to_csv(outputs["summary"])
    target_ranks = pd.Series([d["target_rank_in_pool"] for d in diagnostics], dtype="float")
    meta = {
        "embedding_model": spec.model_id, "model_alias": spec.alias, "embedding_dim": spec.dimension,
        "pooling": mapping_meta.get("pooling"), "token_pooling": spec.token_pooling,
        "max_seq_length": spec.max_seq_length, "input_format": spec.input_format,
        "embeddings_path": str(embeddings_path), "mapping_path": str(mapping_path),
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
    outputs["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_markdown(summary, meta, outputs["markdown"])
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.T)
    print(json.dumps({k: v for k, v in meta.items() if k != "settings"}, indent=2))
    return 0


def write_markdown(summary: pd.DataFrame, meta: dict, path: Path) -> None:
    keys = ["nDCG", "P", "Recall", "HR", "GiniSimpson", "GiniSimpson_m2", "UniqueInst", "InstCoverage",
            "ProtectedShare", "ExposureShareProt", "MeanExpProt", "MeanExpNonProt", "NDKL", "NDKLpop", "RelRetained"]
    lines = [f"# Re-ranking experiment ({meta['n_cases']} LOO cases, seed {meta['seed']}, "
             f"alpha={meta['alpha']}, Top-N={meta['top_n']}, model {meta['model_alias']} ({meta['embedding_model']}), "
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
