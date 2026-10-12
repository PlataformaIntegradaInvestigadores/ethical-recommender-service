# Comparativa de modelos de embeddings: MiniLM vs. SPECTER2 vs. SciNCL

Estudio comparativo de la Fase 1 (representación semántica de investigadores). Se evalúa si un
codificador científico mejora la recuperación del integrante oculto y cómo cambia el re-ranking
ético con cada modelo. Fecha de las corridas: 2026-10-10.

## 1. Configuración

| Alias | Modelo | Dim. | Pooling por artículo | Long. máx. | Entrada | Generación (02) |
|---|---|---|---|---|---|---|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | media de tokens | 256 | `título\nresumen` | CPU, ~16 min |
| `specter2` | `allenai/specter2_base` (sin adaptador) | 768 | `[CLS]` | 512 | `título[SEP]resumen` | RTX 3060 Laptop, fp16, 319 s |
| `scincl` | `malteos/scincl` | 768 | `[CLS]` | 512 | `título[SEP]resumen` | RTX 3060 Laptop, fp16, 242 s |

En los tres casos el vector del autor es la media renormalizada de sus artículos (7 621 autores,
33 477 textos únicos). Detalle de ejecución en `docs/execution_history.md` (rama
`feat/models-comparative-study` / `docs/embeddings-gpu-runs`).

Evaluación (`04_run_rerank_experiment.py --model <alias>`): los mismos 200 casos leave-one-out
(semilla 42) de los 1 415 grupos maximales con ≥ 3 integrantes, agregación híbrida con α = 0,5,
Top-N = 100, K = 20, ajustes R0–R3 sin cambios. Dos modos:

- **Estándar**: embeddings completos. Incluye la fuga de datos descrita en
  `resultados-apriori-reranking.md` (el vector del integrante oculto contiene los artículos que
  firmó con el grupo).
- **Sin fuga** (`--leakage-free`): los integrantes y el objetivo se re-codifican sin los artículos
  que el objetivo co-firmó con algún integrante, con el **mismo** modelo, pooling y formato de
  entrada del alias. Es el número que debe citarse en la tesis.

Archivos: `data/processed/experiments/rerank_*_{alias}{tag}.*` y `model_comparison*.{csv,md}`,
`embedding_intrinsic_stats.csv`. Scripts reproducibles en `documentacion/comparativa-modelos/`; `run_full_comparison.py` ejecuta todo
el flujo para los modelos disponibles (incluidos `minilm-l12` y `mpnet` cuando existan sus embeddings).

## 2. Línea base (sin re-ranking)

| Modo | Modelo | nDCG@10 | nDCG@20 | HR@10 | HR@20 | En Top-100 | Rango mediano | MRR |
|---|---|---|---|---|---|---|---|---|
| estándar | minilm | 0,764 | 0,776 | 0,915 | 0,960 | 0,990 | 1 | 0,720 |
| estándar | specter2 | 0,750 | 0,759 | 0,900 | 0,935 | 0,985 | 1 | 0,705 |
| estándar | scincl | 0,741 | 0,753 | 0,880 | 0,925 | 0,980 | 1 | 0,701 |
| **sin fuga** | minilm | 0,062 | 0,079 | 0,140 | 0,205 | 0,470 | 133 | 0,052 |
| **sin fuga** | specter2 | 0,063 | 0,082 | 0,135 | 0,210 | 0,445 | 138,5 | 0,054 |
| **sin fuga** | scincl | 0,065 | 0,086 | 0,140 | 0,225 | 0,460 | 133,5 | 0,054 |

El rango mediano es la posición del integrante oculto entre los ~7 600 candidatos restantes (sin truncar a 100).

Barrido de α (nDCG@20 de la línea base, sin fuga): MiniLM 0,063 / 0,079 / 0,092; SPECTER2
0,058 / 0,082 / 0,093; SciNCL 0,068 / 0,086 / 0,094 para α = 0 / 0,5 / 1. Con los tres modelos el
promedio utilitarista puro (α = 1) supera al híbrido; el orden entre modelos no cambia la conclusión.

## 3. Re-ranking (sin fuga)

| Modelo | Ajuste | HR@20 | ΔHR@20 | Gini-Simpson@20 | Instituciones únicas@20 | Cuota protegidos@20 | Relevancia (coseno) | Relevancia normalizada |
|---|---|---|---|---|---|---|---|---|
| minilm | línea base | 0,205 | — | 0,696 | 7,80 | 0,027 | 1,000 | 1,000 |
| minilm | R1_suave | 0,150 | −0,055 | 0,906 | 13,38 | 0,037 | 0,970 | 0,866 |
| minilm | R2_moderado | 0,110 | −0,095 | 0,934 | 16,90 | 0,071 | 0,947 | 0,755 |
| minilm | R3_fair_binomial | 0,120 | −0,085 | 0,929 | 15,84 | 0,052 | 0,954 | 0,788 |
| specter2 | línea base | 0,210 | — | 0,734 | 8,48 | 0,011 | 1,000 | 1,000 |
| specter2 | R1_suave | 0,135 | −0,075 | 0,907 | 13,51 | 0,018 | 0,998 | 0,893 |
| specter2 | R2_moderado | 0,100 | −0,110 | 0,934 | 16,94 | 0,040 | 0,996 | 0,794 |
| specter2 | R3_fair_binomial | 0,120 | −0,090 | 0,929 | 15,86 | 0,030 | 0,997 | 0,823 |
| scincl | línea base | 0,225 | — | 0,730 | 8,50 | 0,016 | 1,000 | 1,000 |
| scincl | R1_suave | 0,130 | −0,095 | 0,906 | 13,39 | 0,027 | 0,996 | 0,898 |
| scincl | R2_moderado | 0,090 | −0,135 | 0,934 | 16,87 | 0,056 | 0,993 | 0,799 |
| scincl | R3_fair_binomial | 0,105 | −0,120 | 0,929 | 15,82 | 0,041 | 0,994 | 0,828 |

Tablas completas (también en modo estándar): `model_comparison_baseline.csv` y
`model_comparison_rerank.csv`.

**Advertencia sobre la métrica de relevancia retenida.** `RelRetained@20` es el cociente de la suma
de cosenos crudos. Su valor depende de la escala del modelo: en SPECTER2 dos autores cualesquiera ya
tienen coseno ≈ 0,89, por lo que el cociente queda cerca de 1 aunque la lista cambie mucho. Para
comparar modelos se añadió `RelRetainedNorm@20`, el mismo cociente con la relevancia normalizada
min–max dentro del pool de cada consulta (la escala que usa el propio re-ranker). Con esta medida,
R1 conserva el 86,6 % de la relevancia con MiniLM, y no el 97 % que se reportó con la métrica cruda.
Se recomienda citar ambas en el Capítulo 3.

## 4. Significancia estadística (pareada, mismos 200 casos sin fuga)

| Comparación | ΔHR@20 [IC 95 % bootstrap] | McNemar exacto (discordantes) | ΔnDCG@20 [IC 95 %] | Rango completo (Wilcoxon) |
|---|---|---|---|---|
| SPECTER2 − MiniLM | +0,005 [−0,035; 0,045] | p = 1,00 (9/8) | +0,003 [−0,009; 0,016] | mediana Δ = +4 posiciones, p = 0,009 |
| SciNCL − MiniLM | +0,020 [−0,020; 0,060] | p = 0,48 (11/7) | +0,007 [−0,005; 0,020] | mediana Δ = +4 posiciones, p = 0,024 |
| SciNCL − SPECTER2 | +0,015 [−0,015; 0,045] | p = 0,51 (6/3) | +0,004 [−0,006; 0,014] | p = 0,72 |

Bootstrap de 10 000 remuestras sobre los casos. En el rango completo, una Δ positiva significa que
el integrante oculto queda **más abajo** con el modelo científico que con MiniLM.

Cuota de perfiles de baja visibilidad en el Top-20 de la línea base (sin fuga): MiniLM 2,7 %,
SPECTER2 1,1 % (Δ −1,6 pp, IC 95 % [−2,1; −1,2], Wilcoxon p < 0,001), SciNCL 1,6 % (Δ −1,1 pp,
IC 95 % [−1,6; −0,7], p < 0,001).

## 5. Estadísticas intrínsecas

| Estadística | MiniLM | SPECTER2 | SciNCL |
|---|---|---|---|
| Coseno medio entre autores al azar (anisotropía) | 0,143 | 0,889 | 0,793 |
| Desv. estándar del coseno al azar | 0,159 | 0,034 | 0,058 |
| Coseno medio entre co-autores (pares frecuentes, soporte ≥ 3) | 0,814 | 0,982 | 0,966 |
| Coseno medio, pares al azar de la misma institución | 0,266 | 0,909 | 0,829 |
| Brecha co-autores − azar, en desviaciones estándar | 4,23 | 2,73 | 2,99 |
| AUC co-autores vs. azar | 0,995 | 0,995 | 0,994 |
| AUC co-autores vs. misma institución | 0,943 | 0,947 | 0,941 |
| Varianza en la 1.ª componente principal | 8,9 % | 13,2 % | 11,7 % |
| Dimensión efectiva (participation ratio) | 42,0 | 24,3 | 26,3 |

7 403 pares de co-autores, 300 000 pares al azar y 100 000 pares de la misma institución; los pares
muestreados son los mismos para todos los modelos (generador con semilla 42 por modelo).

Limitación: los co-autores comparten artículos, así que su coseno también tiene fuga. Las AUC
cercanas a 0,995 miden sobre todo texto compartido y no discriminan entre modelos. La evidencia
válida sobre capacidad de recomendación es la del modo sin fuga (secciones 2–4).

## 6. Interpretación

1. **No hay un ganador estadísticamente significativo en precisión.** Sin fuga, SciNCL tiene el
   mejor valor puntual (HR@20 0,225 frente a 0,205; nDCG@20 0,086 frente a 0,079), pero todos los
   intervalos de confianza incluyen el cero y McNemar no rechaza la igualdad. Con 200 casos, una
   diferencia de 2 puntos en HR@20 no se distingue del azar.
2. **En el ranking completo los modelos científicos son levemente peores.** El integrante oculto
   queda en promedio unas 4 posiciones más abajo (de ~7 600) con SPECTER2 y SciNCL (p < 0,05). Es un
   efecto pequeño, pero va en contra de la hipótesis de que el vocabulario científico mejora la
   afinidad en este corpus.
3. **Anisotropía.** SPECTER2 y SciNCL concentran a todos los autores en un cono estrecho (coseno al
   azar 0,89 y 0,79) y usan menos dimensiones efectivas. El ranking no se ve afectado por ser
   relativo, pero cualquier umbral o métrica sobre cosenos crudos (como `RelRetained`) deja de ser
   comparable entre modelos. El re-ranker usa relevancia normalizada, así que su comportamiento es
   el mismo con los tres.
4. **Equidad.** Con los modelos científicos la línea base expone menos a los perfiles de baja
   visibilidad (2,7 % → 1,1–1,6 %, diferencia significativa). Una hipótesis plausible es que los
   autores prolíficos tienen perfiles más "centrales" en el espacio científico (efecto Mateo). La
   cuota del re-ranking compensa parcialmente, pero parte de más abajo.
5. **El re-ranking se comporta igual con los tres modelos.** R1 eleva el Gini-Simpson@20 a ≈ 0,91 y
   las instituciones únicas a ≈ 13,4 en todos los casos. El costo en HR@20 es algo mayor con los
   modelos científicos (−0,075 y −0,095 frente a −0,055 con MiniLM).
6. **Posibles causas** (hipótesis, no verificadas). El idioma no lo explica: según un conteo
   heurístico de palabras vacías, solo 87 de 33 477 textos (0,26 %) están predominantemente en
   español. Una posibilidad es que la verdad de terreno sea social e institucional (coautoría real)
   más que temática, y que la cercanía temática fina que aportan los modelos entrenados con citas
   no ayude a predecirla. Además se usó SPECTER2 sin su adaptador de proximidad.

## 7. Recomendación

Mantener **all-MiniLM-L6-v2 como modelo principal** del recomendador y del Capítulo 3, y reportar
SciNCL y SPECTER2 como alternativas evaluadas. Razones:

- precisión equivalente (sin diferencias significativas en HR@10, HR@20 ni nDCG@20 sin fuga);
- mejor exposición de perfiles de baja visibilidad en la línea base, coherente con el objetivo ético;
- vectores de la mitad del tamaño (384 frente a 768) y generación viable en CPU;
- un espacio menos anisotrópico, donde las similitudes crudas son interpretables.

Si se quiere un modelo científico, **SciNCL** es preferible a SPECTER2: mejor valor puntual, menor
anisotropía, menor caída de equidad y generación más rápida en GPU (242 s frente a 319 s). Para
afirmar una mejora habría que evaluar con más casos (p. ej. los 1 415) y probar el adaptador
`allenai/specter2`.

Con los tres modelos, α = 1 (promedio puro) dio mejor nDCG@20 que α = 0,5. Conviene revisar ese
parámetro antes de fijar la configuración final.

## 8. Tiempos de cómputo

| Paso | MiniLM | SPECTER2 | SciNCL |
|---|---|---|---|
| Generación de embeddings (02) | ~16 min en CPU (ejecución #1) | 319 s en GPU, fp16 | 242 s en GPU, fp16 |
| Re-codificación sin fuga (04, 7 376 artículos, CPU de 8 núcleos) | 154 s | 1 084 s | 652 s |
| Evaluación estándar (04, 200 casos) | ~3 s | ~4 s | ~3 s |
