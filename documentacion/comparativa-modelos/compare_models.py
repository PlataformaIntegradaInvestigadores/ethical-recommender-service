"""Requiere pandas, scipy y tabulate. Cross-model comparison table + paired significance on the same 200 LOO cases."""
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
E = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "experiments"
MODELS = ["minilm", "specter2", "scincl"]
rng = np.random.default_rng(42)

def load(alias, tag=""):
    pc = pd.read_csv(E / f"rerank_per_case_{alias}{tag}.csv", dtype={"group_id": str})
    sm = pd.read_csv(E / f"rerank_summary_{alias}{tag}.csv", index_col=0)
    meta = json.loads((E / f"rerank_meta_{alias}{tag}.json").read_text())
    return pc, sm, meta

rows, rer_rows, sweep = [], [], []
per = {}
for mode, tag in (("estándar", ""), ("sin fuga", "_leakage_free")):
    for alias in MODELS:
        pc, sm, meta = load(alias, tag)
        base = pc[pc.method == "baseline"].set_index("group_id")
        per[(mode, alias)] = pc
        r = base["target_rank_full"]
        rows.append({"modo": mode, "modelo": alias, "n": len(base),
                     "nDCG@10": sm.loc["baseline", "nDCG@10"], "nDCG@20": sm.loc["baseline", "nDCG@20"],
                     "HR@10": sm.loc["baseline", "HR@10"], "HR@20": sm.loc["baseline", "HR@20"],
                     "HR@100 (en pool)": meta["target_in_top_n_share"],
                     "rango_mediano": r.median(), "rango_medio_recíproco": (1 / r).mean(),
                     "GiniSimpson@20": sm.loc["baseline", "GiniSimpson@20"],
                     "UniqueInst@20": sm.loc["baseline", "UniqueInst@20"],
                     "ProtectedShare@20": sm.loc["baseline", "ProtectedShare@20"]})
        for method in ("R1_suave", "R2_moderado", "R3_fair_binomial"):
            rer_rows.append({"modo": mode, "modelo": alias, "ajuste": method,
                             **{c: sm.loc[method, c] for c in ("HR@20", "nDCG@20", "GiniSimpson@20", "GiniSimpson_m2@20",
                                                              "UniqueInst@20", "ProtectedShare@20", "RelRetained@20",
                                                              "RelRetainedNorm@20", "quota_shortfalls")},
                             "ΔHR@20": sm.loc[method, "HR@20"] - sm.loc["baseline", "HR@20"],
                             "ΔGiniSimpson@20": sm.loc[method, "GiniSimpson@20"] - sm.loc["baseline", "GiniSimpson@20"]})
        for alpha, suffix in ((0.0, "_alpha0.0"), (0.5, ""), (1.0, "_alpha1.0")):
            _, s2, _ = load(alias, tag + suffix)
            sweep.append({"modo": mode, "modelo": alias, "alpha": alpha, "nDCG@20": s2.loc["baseline", "nDCG@20"],
                          "HR@20": s2.loc["baseline", "HR@20"]})

base_tab = pd.DataFrame(rows).round(4); rer_tab = pd.DataFrame(rer_rows).round(4)
sweep_tab = pd.DataFrame(sweep).pivot_table(index=["modo", "modelo"], columns="alpha", values="nDCG@20").round(4)

sig = []
for mode in ("estándar", "sin fuga"):
    b = {m: per[(mode, m)].query("method == 'baseline'").set_index("group_id").sort_index() for m in MODELS}
    assert all((b[m].index == b["minilm"].index).all() for m in MODELS)
    for x, y in itertools.combinations(MODELS, 2):
        for metric in ("HR@20", "HR@10", "nDCG@20"):
            dx, dy = b[x][metric].to_numpy(), b[y][metric].to_numpy()
            diff = dy - dx
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry = {"modo": mode, "comparación": f"{y} − {x}", "métrica": metric, "Δ": diff.mean(),
                     "IC95_lo": lo, "IC95_hi": hi}
            if metric.startswith("HR"):
                n01 = int(((dx == 0) & (dy == 1)).sum()); n10 = int(((dx == 1) & (dy == 0)).sum())
                entry |= {"discordantes (solo y / solo x)": f"{n01}/{n10}",
                          "p_McNemar_exacto": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0}
            sig.append(entry)
        rx, ry = b[x]["target_rank_full"].to_numpy(), b[y]["target_rank_full"].to_numpy()
        sig.append({"modo": mode, "comparación": f"{y} − {x}", "métrica": "rango completo (Wilcoxon)",
                    "Δ": float(np.median(ry - rx)), "p_McNemar_exacto": wilcoxon(rx, ry).pvalue})
sig_tab = pd.DataFrame(sig).round(4)

base_tab.to_csv(E / "model_comparison_baseline.csv", index=False)
rer_tab.to_csv(E / "model_comparison_rerank.csv", index=False)
sweep_tab.to_csv(E / "model_comparison_alpha_sweep.csv")
sig_tab.to_csv(E / "model_comparison_significance.csv", index=False)
intr = pd.read_csv(E / "embedding_intrinsic_stats.csv", index_col=0)
md = ["# Comparativa de modelos de embeddings (200 casos LOO, semilla 42, α=0,5, Top-N=100)", "",
      "## Línea base", "", base_tab.to_markdown(index=False), "",
      "## Re-ranking", "", rer_tab.to_markdown(index=False), "",
      "## Barrido de α (nDCG@20 de la línea base)", "", sweep_tab.to_markdown(), "",
      "## Significancia (pareada, mismos 200 casos)", "", sig_tab.to_markdown(index=False), "",
      "## Estadísticas intrínsecas", "", intr.T.to_markdown(), ""]
(E / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

# Protected (low-visibility) share in baseline Top-20, paired bootstrap
extra = []
for mode in ("estándar", "sin fuga"):
    b = {m: per[(mode, m)].query("method == 'baseline'").set_index("group_id").sort_index() for m in MODELS}
    for y in ("specter2", "scincl"):
        diff = (b[y]["ProtectedShare@20"] - b["minilm"]["ProtectedShare@20"]).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
        extra.append({"modo": mode, "comparación": f"{y} − minilm", "métrica": "ProtectedShare@20 (línea base)",
                      "Δ": diff.mean(), "IC95_lo": np.percentile(boots, 2.5), "IC95_hi": np.percentile(boots, 97.5),
                      "p_Wilcoxon": wilcoxon(diff[diff != 0]).pvalue if (diff != 0).any() else 1.0})
extra_tab = pd.DataFrame(extra).round(4)
extra_tab.to_csv(E / "model_comparison_protected_share.csv", index=False)
with open(E / "model_comparison.md", "a", encoding="utf-8") as fh:
    fh.write("\n## Cuota de perfiles de baja visibilidad en el Top-20 de la línea base\n\n" + extra_tab.to_markdown(index=False) + "\n")
print(extra_tab.to_markdown(index=False))
