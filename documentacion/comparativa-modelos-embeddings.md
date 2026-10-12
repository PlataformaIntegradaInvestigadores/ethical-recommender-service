# Comparativa de modelos de embeddings: tres generalistas y dos científicos

Estudio comparativo de la Fase 1 (representación semántica de investigadores). Se evalúa si otro
codificador mejora la recuperación del integrante oculto respecto a la línea base
`all-MiniLM-L6-v2`, y cómo cambia el re-ranking ético con cada modelo. Corridas: 2026-10-10
(SPECTER2, SciNCL) y 2026-10-11 (MiniLM-L12, MPNet).

## 1. Configuración

| Alias | Modelo | Tipo | Dim. | Pooling por artículo | Long. máx. | Entrada | Generación (02) |
|---|---|---|---|---|---|---|---|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | generalista (línea base) | 384 | media de tokens | 256 | `título\nresumen` | CPU, ~16 min |
| `minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | generalista | 384 | media de tokens | 256 | `título\nresumen` | Colab T4, fp16, 92,5 s |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | generalista | 768 | media de tokens | 384 | `título\nresumen` | Colab T4, fp16, 241,7 s |
| `specter2` | `allenai/specter2_base` (sin adaptador) | científico (citas) | 768 | `[CLS]` | 512 | `título[SEP]resumen` | RTX 3060 Laptop, fp16, 319 s |
| `scincl` | `malteos/scincl` | científico (grafo de citas) | 768 | `[CLS]` | 512 | `título[SEP]resumen` | RTX 3060 Laptop, fp16, 242 s |

En todos los casos el vector del autor es la media renormalizada de sus artículos (7 621 autores,
33 477 textos únicos, mismo orden de autores en los cinco archivos). Detalle de ejecución en
`docs/execution_history.md`. MiniLM-L12 y MPNet se generaron en Google Colab porque Windows
Application Control bloqueó una DLL de `scipy` en el entorno local.

Evaluación (`04_run_rerank_experiment.py --model <alias>`): los mismos 200 casos leave-one-out
(semilla 42) de los 1 415 grupos maximales con ≥ 3 integrantes, agregación híbrida con α = 0,5 (y
barrido α = 0 / 0,5 / 1), Top-N = 100, K = 20, ajustes R0–R3 sin cambios. Dos modos:

- **Estándar**: embeddings completos. Incluye la fuga de datos descrita en
  `resultados-apriori-reranking.md` (el vector del integrante oculto contiene los artículos que
  firmó con el grupo).
- **Sin fuga** (`--leakage-free`): los integrantes y el objetivo se re-codifican sin los artículos
  que el objetivo co-firmó con algún integrante, con el **mismo** modelo, pooling y formato de
  entrada del alias. Es el número que debe citarse en la tesis.

Archivos: `data/processed/experiments/rerank_*_{alias}{tag}.*`, `model_comparison*.{csv,md}` y
`embedding_intrinsic_stats.csv`. Todo el flujo se reproduce con
`python documentacion/comparativa-modelos/run_full_comparison.py`.

## 2. Línea base (sin re-ranking, α = 0,5)

| Modo | Modelo | nDCG@10 | nDCG@20 | HR@10 | HR@20 | En Top-100 | Rango mediano | MRR |
|---|---|---|---|---|---|---|---|---|
| estándar | minilm | 0,764 | 0,776 | 0,915 | 0,960 | 0,990 | 1 | 0,720 |
| estándar | minilm-l12 | 0,759 | 0,772 | 0,920 | 0,970 | 0,985 | 1 | 0,711 |
| estándar | mpnet | 0,753 | 0,763 | 0,920 | 0,960 | 0,990 | 1 | 0,704 |
| estándar | specter2 | 0,750 | 0,759 | 0,900 | 0,935 | 0,985 | 1 | 0,705 |
| estándar | scincl | 0,741 | 0,753 | 0,880 | 0,925 | 0,980 | 1 | 0,701 |
| **sin fuga** | minilm | 0,062 | 0,079 | 0,140 | 0,205 | 0,470 | 133 | 0,052 |
| **sin fuga** | minilm-l12 | 0,064 | 0,086 | 0,145 | 0,235 | 0,470 | 117 | 0,053 |
| **sin fuga** | mpnet | 0,061 | 0,087 | 0,130 | 0,235 | 0,490 | 105 | 0,055 |
| **sin fuga** | specter2 | 0,063 | 0,082 | 0,135 | 0,210 | 0,445 | 138,5 | 0,054 |
| **sin fuga** | scincl | 0,065 | 0,086 | 0,140 | 0,225 | 0,460 | 133,5 | 0,054 |

El rango mediano es la posición del integrante oculto entre los ~7 600 candidatos restantes (sin
truncar a 100). MRR = media del rango recíproco.

### Barrido de α (nDCG@20 de la línea base, sin fuga)

| Modelo | α = 0 | α = 0,5 | α = 1 |
|---|---|---|---|
| minilm | 0,063 | 0,079 | 0,092 |
| minilm-l12 | 0,069 | 0,086 | 0,093 |
| mpnet | 0,071 | 0,087 | 0,093 |
| specter2 | 0,058 | 0,082 | 0,093 |
| scincl | 0,068 | 0,086 | 0,094 |

Con los cinco modelos el promedio utilitarista puro (α = 1) es el mejor, y con α = 1 las
diferencias entre modelos desaparecen (0,092–0,094).

## 3. Re-ranking R1 (sin fuga, α = 0,5)

| Modelo | HR@20 | ΔHR@20 vs. base | Gini-Simpson@20 | Instituciones únicas@20 | Cuota protegidos@20 (base → R1) | Relevancia (coseno) | Relevancia normalizada |
|---|---|---|---|---|---|---|---|
| minilm | 0,150 | −0,055 | 0,906 | 13,38 | 0,027 → 0,037 | 0,970 | 0,866 |
| minilm-l12 | 0,155 | −0,080 | 0,906 | 13,40 | 0,029 → 0,040 | 0,970 | 0,864 |
| mpnet | 0,135 | −0,100 | 0,905 | 13,27 | 0,031 → 0,045 | 0,973 | 0,870 |
| specter2 | 0,135 | −0,075 | 0,907 | 13,51 | 0,011 → 0,018 | 0,998 | 0,893 |
| scincl | 0,130 | −0,095 | 0,906 | 13,39 | 0,016 → 0,027 | 0,996 | 0,898 |

R2 y R3 se comportan igual con los cinco modelos: Gini-Simpson@20 ≈ 0,93 y 15,7–16,9 instituciones
únicas. Tablas completas, también en modo estándar: `model_comparison_rerank.csv`.

**Dos métricas de relevancia retenida.** `RelRetained@20` divide la suma de cosenos crudos del
Top@20 re-rankeado entre la de la línea base. Su valor depende de la escala del modelo: en SPECTER2
dos autores cualesquiera ya tienen coseno ≈ 0,89, así que el cociente queda cerca de 1 aunque la lista
cambie mucho. `RelRetainedNorm@20` aplica el mismo cociente sobre la relevancia normalizada min–max
dentro del pool de cada consulta, que es la escala que usa el propio re-ranker. Con MiniLM-L6, R1
conserva el 97,0 % con la métrica cruda y el 86,6 % con la normalizada. Para comparar modelos debe
usarse la normalizada; se recomienda citar ambas en el Capítulo 3.

## 4. Significancia estadística frente a MiniLM-L6 (pareada, mismos 200 casos, sin fuga)

Bootstrap de 10 000 remuestras sobre los casos (IC 95 %), McNemar exacto sobre los casos
discordantes y Wilcoxon sobre el rango completo del integrante oculto.

| Comparación (α = 0,5) | ΔHR@20 [IC 95 %] | McNemar (discordantes) | ΔnDCG@20 [IC 95 %] | Rango (Wilcoxon) | ΔCuota protegidos@20 base [IC 95 %] |
|---|---|---|---|---|---|
| MiniLM-L12 − L6 | +0,030 [+0,010; +0,055] | p = 0,031 (6/0) | +0,0074 [+0,0005; +0,0151] | mediana 117 vs. 133, p = 0,13 | +0,2 pp [−0,2; +0,6], p = 0,50 |
| MPNet − L6 | +0,030 [0,000; +0,060] | p = 0,11 (8/2) | +0,0083 [−0,0016; +0,0185] | mediana 105 vs. 133, p = 0,046 | +0,4 pp [0,0; +0,8], p = 0,020 |
| SPECTER2 − L6 | +0,005 [−0,035; +0,045] | p = 1,00 (9/8) | +0,0033 [−0,009; +0,016] | peor, p = 0,009 | −1,6 pp [−2,1; −1,2], p < 0,001 |
| SciNCL − L6 | +0,020 [−0,020; +0,060] | p = 0,48 (11/7) | +0,0071 [−0,005; +0,020] | peor, p = 0,024 | −1,1 pp [−1,6; −0,7], p < 0,001 |

**Robustez de la ganancia de los generalistas** (`model_comparison_alpha_significance.csv`):

| Comparación | Lista | α = 0 | α = 0,5 | α = 1 |
|---|---|---|---|---|
| MiniLM-L12 − L6, ΔHR@20 | línea base | +0,025 (p = 0,13) | **+0,030 (p = 0,031)** | −0,005 (p = 1,00) |
| MiniLM-L12 − L6, ΔHR@20 | R1 | +0,005 (p = 1,00) | +0,005 (p = 1,00) | +0,010 (p = 0,73) |
| MPNet − L6, ΔHR@20 | línea base | +0,020 (p = 0,29) | +0,030 (p = 0,11) | +0,010 (p = 0,77) |
| MPNet − L6, ΔHR@20 | R1 | −0,010 (p = 0,75) | −0,015 (p = 0,45) | −0,010 (p = 0,73) |

Ninguna diferencia de nDCG@20 frente a L6 con α = 0 o α = 1, ni tras R1, excluye el cero. El único
resultado con p < 0,05 en precisión (L12, HR@20, α = 0,5) no sobrevive a una corrección de
Bonferroni por las cuatro comparaciones contra L6 (umbral 0,0125).

## 5. Estadísticas intrínsecas

| Estadística | MiniLM-L6 | MiniLM-L12 | MPNet | SPECTER2 | SciNCL |
|---|---|---|---|---|---|
| Coseno medio entre autores al azar (anisotropía) | 0,143 | 0,155 | 0,171 | 0,889 | 0,793 |
| Coseno medio entre co-autores (pares frecuentes, soporte ≥ 3) | 0,814 | 0,813 | 0,834 | 0,982 | 0,966 |
| Brecha co-autores − azar, en desviaciones estándar | 4,23 | 4,15 | 4,08 | 2,73 | 2,99 |
| AUC co-autores vs. misma institución | 0,943 | 0,941 | 0,941 | 0,947 | 0,941 |
| Varianza en la 1.ª componente principal | 8,9 % | 8,3 % | 9,9 % | 13,2 % | 11,7 % |
| Dimensión efectiva (participation ratio) | 42,0 | 44,5 | 39,2 | 24,3 | 26,3 |

Muestras: 7 403 pares de co-autores, 300 000 pares al azar y 100 000 pares de la misma institución,
los mismos para todos los modelos (semilla 42). Los co-autores comparten artículos, así que su coseno
también tiene fuga; las AUC (≈ 0,94–0,995) no discriminan entre modelos. La evidencia válida sobre
capacidad de recomendación es la del modo sin fuga.

## 6. Interpretación

1. **Ningún modelo mejora a MiniLM-L6 de forma significativa y consistente.** Con α = 0,5,
   MiniLM-L12 y MPNet suben HR@20 de 0,205 a 0,235 sin fuga. Para L12 la diferencia es significativa
   sin corrección (p = 0,031), pero no tras Bonferroni. Además no se sostiene: con α = 1, la mejor
   configuración de agregación para todos los modelos, la ventaja desaparece (−0,005 y +0,010), y
   tras el re-ranking R1 tampoco hay diferencias. HR@10 no mejora con ninguno de los dos.
2. **Los generalistas más grandes mejoran el rango profundo.** El rango mediano del integrante
   oculto baja de 133 (L6) a 117 (L12) y 105 (MPNet). Para MPNet es significativo con α = 0,5
   (p = 0,046); para L12, con α = 1 (p = 0,026). La mejora ocurre lejos del Top@20 y no cambia las
   listas que se recomiendan.
3. **Geometría y equidad de los generalistas.** L12 y MPNet conservan la geometría de L6: anisotropía
   baja (0,15–0,17), brecha co-autores–azar similar y dimensión efectiva comparable (44,5 y 39,2
   frente a 42,0). Tampoco reducen la exposición de perfiles de baja visibilidad; MPNet la aumenta
   levemente (+0,4 pp, p = 0,02).
4. **Los modelos basados en citas exponen menos a los perfiles de baja visibilidad y son mucho más
   anisotrópicos.** Medimos una cuota de protegidos en la línea base de 1,1 % (SPECTER2) y 1,6 %
   (SciNCL) frente a 2,7 % con L6, una diferencia significativa. También medimos cosenos al azar de
   0,89 y 0,79 y una dimensión efectiva de 24–26. Una hipótesis plausible es un sesgo hacia la
   popularidad: estos modelos se entrenaron con señales de citas, que favorecen la producción
   abundante y muy citada. No lo hemos demostrado; no medimos las citas de los autores del corpus.
5. **La anisotropía afecta a las métricas de escala, no al ranking.** El ranking es relativo y el
   re-ranker usa relevancia normalizada, así que su comportamiento es el mismo con los cinco modelos.
   En cambio, cualquier umbral o métrica sobre cosenos crudos, como `RelRetained`, deja de ser
   comparable entre modelos.
6. **El idioma no explica los resultados.** Según un conteo heurístico de palabras vacías, solo 87
   de 33 477 textos (0,26 %) están predominantemente en español. Una posibilidad es que la verdad de
   terreno sea social e institucional (coautoría real) más que temática. Además, SPECTER2 se usó sin
   su adaptador de proximidad.

## 7. Recomendación final

Se mantiene **all-MiniLM-L6-v2** como modelo del recomendador y del Capítulo 3. Según el criterio
fijado (adoptar L12 o MPNet solo si mejoran nDCG/HR de forma significativa o consistente sin
deteriorar la equidad ni la geometría), ninguno califica:

- **MiniLM-L12**: mejor valor puntual a α = 0,5 y misma geometría y equidad, pero la ganancia no
  sobrevive a la corrección por comparaciones múltiples, desaparece a α = 1 y tras R1, y duplica el
  costo de codificación (12 capas frente a 6; 61 frente a 120 artículos/s en CPU).
- **MPNet**: rango profundo algo mejor y exposición de protegidos levemente mayor, pero sin ganancia
  significativa en HR@20 ni nDCG@20, HR@10 menor, vectores del doble de tamaño (768) y entre 4 y 10 veces
  más tiempo de codificación en CPU (632 s frente a 154 s en la re-codificación sin fuga; 12 frente a
  120 artículos/s en una medición aislada).
- **SPECTER2 y SciNCL**: sin ganancia de precisión, y con menor exposición de protegidos y alta
  anisotropía.

MiniLM-L12 queda como la alternativa más cercana. Si se amplía la evaluación (p. ej. a los 685 casos
elegibles sin fuga), es el primer candidato a reevaluar. Las tablas del Capítulo 3, basadas en
MiniLM-L6, no cambian.

## 8. Tiempos de cómputo

| Paso | MiniLM-L6 | MiniLM-L12 | MPNet | SPECTER2 | SciNCL |
|---|---|---|---|---|---|
| Generación de embeddings (02) | ~16 min CPU | 92,5 s T4 | 241,7 s T4 | 319 s RTX 3060 | 242 s RTX 3060 |
| Re-codificación sin fuga (04, 7 376 artículos, CPU de 8 núcleos) | 154 s | 118 s | 632 s | 1 084 s | 652 s |
| Evaluación estándar (04, 200 casos) | ~3 s | ~3 s | ~2 s | ~4 s | ~3 s |

Los tiempos de re-codificación en CPU varían con la carga del servidor; la corrida de L6 coincidió con
otros procesos.
