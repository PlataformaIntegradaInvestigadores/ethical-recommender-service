"""Uso: python documentacion/comparativa-modelos/run_model_experiments.py [alias ...]  (por defecto, los 5 modelos)

Run 04_run_rerank_experiment for each model (standard + leakage-free, alpha sweep).

Leakage-free re-encoding is cached per model (same sample for every alpha) by wrapping
encode_plans; the experiment code path itself is unchanged.
"""
import importlib, pickle, sys, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
exp = importlib.import_module("infrastructure.scripts.04_run_rerank_experiment")
CACHE = ROOT / "data" / "processed" / "experiments" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
original = exp.encode_plans
timings = {}

def cached(plans, texts, spec, device="cpu", model=None):
    path = CACHE / f"leakage_free_vectors_{spec.alias}.pkl"
    if path.is_file():
        stored = pickle.loads(path.read_bytes())
        if set(stored) == set(plans):
            return stored
    started = time.perf_counter()
    vectors = original(plans, texts, spec, device, model)
    timings[spec.alias] = time.perf_counter() - started
    print(f"ENCODE_TIME {spec.alias} {timings[spec.alias]:.1f}s for {len({a for d in plans.values() for a in d})} articles", flush=True)
    path.write_bytes(pickle.dumps(vectors))
    return vectors

exp.encode_plans = cached

# --- extra per-case diagnostics (analysis only; 04's own metrics are untouched) ---
import numpy as np
state = {}
_score, _top, _evaluate = exp.score_all_candidates, exp.retrieve_top_n, exp.evaluate_list

def score_spy(matrix, member_rows, **kw):
    scores = _score(matrix, member_rows, **kw); state["scores"] = np.asarray(scores); return scores

def top_spy(scores, excluded, n):
    state["excluded"] = set(excluded); return _top(scores, excluded, n)

def evaluate_spy(name, ranked, baseline, target, pool, relevance, *args, **kw):
    row = _evaluate(name, ranked, baseline, target, pool, relevance, *args, **kw)
    values = np.array([relevance[a] for a in pool]); lo, hi = values.min(), values.max()
    norm = {a: (relevance[a] - lo) / (hi - lo) if hi > lo else 1.0 for a in pool}
    for k in exp.CUTOFFS:
        base = sum(norm[a] for a in baseline[:k])
        row[f"RelRetainedNorm@{k}"] = sum(norm[a] for a in ranked[:k]) / base if base else 1.0
    row["target_rank_in_list"] = ranked.index(target) + 1 if target in ranked else np.nan
    if name == "baseline":
        scores, excluded = state["scores"], state["excluded"]
        t = state["row_of"][target]
        mask = np.ones(len(scores), bool); mask[list(excluded)] = False
        row["target_rank_full"] = int((scores[mask] > scores[t]).sum()) + 1
        row["target_score"] = float(scores[t])
    return row

exp.score_all_candidates, exp.retrieve_top_n, exp.evaluate_list = score_spy, top_spy, evaluate_spy
_load = exp.load_embeddings
def load_spy(e, m):
    matrix, ids, meta = _load(e, m); state["row_of"] = {a: i for i, a in enumerate(ids)}; return matrix, ids, meta
exp.load_embeddings = load_spy
models = sys.argv[1:] or list(exp.MODEL_IDS)
unknown = [m for m in models if m not in exp.MODEL_IDS]
if unknown:
    sys.exit(f"Unknown model(s) {unknown}; choose from {list(exp.MODEL_IDS)}")
for alias in models:
    for leakage in (False, True):
        for alpha in (0.5, 0.0, 1.0):
            tag = ("_leakage_free" if leakage else "") + ("" if alpha == 0.5 else f"_alpha{alpha}")
            argv = ["--model", alias, "--alpha", str(alpha), "--n-cases", "200", "--seed", "42", "--top-n", "100",
                    "--device", "cpu"] + (["--leakage-free"] if leakage else []) + (["--tag", tag] if tag else [])
            started = time.perf_counter()
            code = exp.main(argv)
            print(f"RUN_DONE {alias} {tag or '(standard)'} code={code} {time.perf_counter()-started:.1f}s", flush=True)
            if code:
                sys.exit(code)
