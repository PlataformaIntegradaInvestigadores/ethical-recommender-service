"""Intrinsic embedding statistics per model: anisotropy and co-author separation (AUC)."""
import importlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
exp = importlib.import_module("infrastructure.scripts.04_run_rerank_experiment")
from infrastructure.db.csv_sources import load_clean_edges, primary_institutions

P = exp.PROCESSED_DIR
rng = np.random.default_rng(42)
groups = pd.read_csv(P / "ground_truth_groups.csv", dtype=str)
pairs = groups[groups["size"] == "2"]["author_ids"].str.split("|")
edges, survivors = load_clean_edges()
inst, _ = primary_institutions(list(survivors.index), edges)

def auc(pos, neg):
    allv = np.concatenate([pos, neg]); ranks = pd.Series(allv).rank().to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

rows = []
for alias in ("minilm", "specter2", "scincl"):
    e, m = exp.resolve_embedding_paths(alias)
    X, ids, _ = exp.load_embeddings(e, m)
    row = {a: i for i, a in enumerate(ids)}
    n = len(ids)
    a = rng.integers(0, n, 300_000); b = rng.integers(0, n, 300_000); keep = a != b
    rnd = np.einsum("ij,ij->i", X[a[keep]], X[b[keep]])
    pp = np.array([(row[x], row[y]) for x, y in pairs if x in row and y in row])
    co = np.einsum("ij,ij->i", X[pp[:, 0]], X[pp[:, 1]])
    # random pairs from the same primary institution (harder negatives)
    by_inst = pd.Series({i: inst.get(aid) for i, aid in enumerate(ids)}).dropna()
    by_inst = by_inst[by_inst != exp.UNKNOWN_INSTITUTION]
    groups_inst = [np.array(v) for v in by_inst.groupby(by_inst).groups.values() if len(v) > 1]
    weights = np.array([len(g) for g in groups_inst], float); weights /= weights.sum()
    picks = rng.choice(len(groups_inst), 100_000, p=weights)
    si = np.array([rng.choice(groups_inst[g], 2, replace=False) for g in picks])
    same = np.einsum("ij,ij->i", X[si[:, 0]], X[si[:, 1]])
    centered = X - X.mean(0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    rnd_c = np.einsum("ij,ij->i", centered[a[keep]], centered[b[keep]])
    co_c = np.einsum("ij,ij->i", centered[pp[:, 0]], centered[pp[:, 1]])
    ev = np.linalg.svd(X - X.mean(0), full_matrices=False, compute_uv=False) ** 2
    rows.append({
        "model": alias, "dim": X.shape[1],
        "mean_cos_random": rnd.mean(), "std_cos_random": rnd.std(),
        "mean_cos_coauthors": co.mean(), "mean_cos_same_inst_random": same.mean(),
        "gap_coauthor_minus_random": co.mean() - rnd.mean(),
        "gap_in_random_std_units": (co.mean() - rnd.mean()) / rnd.std(),
        "AUC_coauthor_vs_random": auc(co, rnd), "AUC_coauthor_vs_same_inst": auc(co, same),
        "mean_cos_random_centered": rnd_c.mean(), "AUC_coauthor_vs_random_centered": auc(co_c, rnd_c),
        "var_share_top1_pc": ev[0] / ev.sum(), "var_share_top10_pc": ev[:10].sum() / ev.sum(),
        "participation_ratio": ev.sum() ** 2 / (ev ** 2).sum(),
        "n_coauthor_pairs": len(co), "n_random_pairs": int(keep.sum()), "n_same_inst_pairs": len(same),
    })
out = pd.DataFrame(rows).set_index("model").round(4)
out.to_csv(P / "experiments" / "embedding_intrinsic_stats.csv")
print(out.T.to_string())
