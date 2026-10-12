"""Comparativa completa de modelos de embeddings en un solo comando.

Uso (desde la raíz del repositorio):
    python documentacion/comparativa-modelos/run_full_comparison.py [--models ...] [--force]

1. Verifica que existan author_embeddings_<alias>.npy + author_index_mapping_<alias>.json.
2. Para cada modelo sin resultados completos ejecuta run_model_experiments.py: 04 estándar y
   --leakage-free, 200 casos, semilla 42, Top-N 100, R0–R3, α = 0, 0,5 y 1. Los modelos que ya
   tienen sus 6 resúmenes se omiten, salvo --force.
3. Recalcula las estadísticas intrínsecas (intrinsic_stats.py) y las tablas comparativas con
   significancia frente a minilm (compare_models.py).
"""
import argparse, importlib, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
exp = importlib.import_module("infrastructure.scripts.04_run_rerank_experiment")
TAGS = [lf + al for lf in ("", "_leakage_free") for al in ("", "_alpha0.0", "_alpha1.0")]


def complete(alias: str) -> bool:
    return all((exp.RESULTS_DIR / f"rerank_summary_{alias}{tag}.csv").is_file() for tag in TAGS)


def step(name: str, args: list[str]) -> None:
    started = time.perf_counter()
    print(f"==> {name}: {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"<== {name}: {time.perf_counter() - started:.0f} s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(exp.MODEL_IDS), default=list(exp.MODEL_IDS))
    parser.add_argument("--force", action="store_true", help="re-run 04 even when results exist")
    args = parser.parse_args()
    try:
        for alias in args.models:
            exp.resolve_embedding_paths(alias)
    except exp.ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    pending = [a for a in args.models if args.force or not complete(a)]
    print(f"Modelos: {args.models}; por ejecutar en 04: {pending or 'ninguno'}", flush=True)
    for alias in pending:
        step(f"04 {alias}", [str(HERE / "run_model_experiments.py"), alias])
    step("intrínsecas", [str(HERE / "intrinsic_stats.py"), "--models", *args.models])
    step("comparación", [str(HERE / "compare_models.py"), "--models", *args.models])
    print(f"Listo: {exp.RESULTS_DIR / 'model_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
