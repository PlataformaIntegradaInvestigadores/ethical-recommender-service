# Resultados experimentales: grupos a priori y re-ranking (datos reales, commit 5465da2)

## Minería de grupos (`03_mine_ground_truth_groups.py`)

Parámetros: min_support = 3; artículos con más de 25 autores descartados (8 de 39 951 transacciones); 7 621 autores admitidos (7 607 frecuentes en L1 tras el descarte). Una enumeración ingenua tendría |C2| = 28 929 421 pares.

| k | candidatos | podados (subconjunto infrecuente) | frecuentes |
|---|---|---|---|
| 1 | 7 621 | 0 | 7 607 |
| 2 | 26 545 (solo pares que coocurren) | 0 | 7 403 |
| 3 | 5 972 | 12 715 | 5 334 |
| 4 | 4 024 | 2 000 | 4 011 |
| 5 | 3 212 | 467 | 3 212 |
| 6 | 2 407 | 129 | 2 407 |
| 7 | 1 535 | 28 | 1 535 |
| 8 | 779 | 3 | 779 |
| 9 | 297 | 0 | 297 |
| 10 | 79 | 0 | 79 |
| 11 | 13 | 0 | 13 |
| 12 | 1 | 0 | 1 |
| 13 | 0 | 0 | 0 |

- Tiempo de minería: 0,22 s; la carga de los CSV toma 1,3 s (CPU de 8 núcleos).
- Hay 25 071 grupos frecuentes con k ≥ 2. Los grupos maximales por tamaño son: {2: 2 025, 3: 930, 4: 297, 5: 119, 6: 37, 7: 15, 8: 11, 9: 2, 10: 2, 11: 1, 12: 1}.
- Con `max_k = 5` (valor por defecto) el nivel 5 no queda vacío. En ese caso aparecen 3 212 conjuntos de tamaño 5 marcados como "maximales", pero en realidad son subconjuntos truncados de unos pocos equipos grandes (668 autores y 275 tidsets distintos). Para la verdad de terreno se usa por eso `--max-k 0` (sin límite), que termina en k = 13 en 0,22 s. La corrida con k = 5 se conserva en `*_k5.*`.
- Diversidad institucional de los grupos maximales: Gini-Simpson medio de 0,149 (0,059 con anti-tokenismo m ≥ 2). El 69,8 % de los grupos tiene una sola institución.
- Casos leave-one-out (maximales con tamaño ≥ 3): 1 415.

## Re-ranking (`04_run_rerank_experiment.py`)

Configuración común: 200 casos con semilla 42, α = 0,5, Top-N = 100, K = 20 (también se informa @10).

| Ajuste | λ | tope/inst. | p protegidos | piso de relevancia | modo |
|---|---|---|---|---|---|
| R0_solo_equidad | 1,0 | — | 0,2 | 0,5 | prefijo ⌊p·i⌋ |
| R1_suave | 0,8 | 3 | 0,2 | 0,5 | prefijo |
| R2_moderado | 0,6 | 2 | 0,3 | 0,3 | prefijo |
| R3_fair_binomial | 0,7 | 2 | 0,3 | 0,3 | fa*ir binomial, α_f = 0,1 |

### A. Evaluación estándar (embeddings completos; incluye fuga, ver advertencia 1)
| Métrica | baseline | R0_solo_equidad | R1_suave | R2_moderado | R3_fair_binomial |
|---|---|---|---|---|---|
| nDCG@10 | 0.7641 | 0.7640 | 0.7035 | 0.5463 | 0.6424 |
| P@10 | 0.0915 | 0.0915 | 0.0860 | 0.0740 | 0.0800 |
| HR@10 | 0.9150 | 0.9150 | 0.8600 | 0.7400 | 0.8000 |
| nDCG@20 | 0.7756 | 0.7754 | 0.7087 | 0.5595 | 0.6464 |
| P@20 | 0.0480 | 0.0480 | 0.0440 | 0.0395 | 0.0408 |
| HR@20 | 0.9600 | 0.9600 | 0.8800 | 0.7900 | 0.8150 |
| GiniSimpson@20 | 0.7229 | 0.7236 | 0.9095 | 0.9368 | 0.9318 |
| GiniSimpson_m2@20 | 0.6721 | 0.6726 | 0.6714 | 0.4049 | 0.5273 |
| UniqueInst@20 | 8.0450 | 8.0550 | 13.8400 | 17.3550 | 16.3600 |
| InstCoverage@20 | 0.3083 | 0.3087 | 0.5603 | 0.7052 | 0.6649 |
| ProtectedShare@20 | 0.0318 | 0.0352 | 0.0388 | 0.0615 | 0.0508 |
| ExposureShareProt@20 | 0.0292 | 0.0339 | 0.0358 | 0.0607 | 0.0452 |
| MeanExpProt@20 | 0.1081 | 0.1279 | 0.1407 | 0.2065 | 0.1657 |
| MeanExpNonProt@20 | 0.3531 | 0.3526 | 0.3532 | 0.3524 | 0.3542 |
| NDKL@20 | 0.0544 | 0.0519 | 0.0482 | 0.0594 | 0.0484 |
| NDKLpop@20 | 0.1276 | 0.1215 | 0.1191 | 0.1078 | 0.1147 |
| RelRetained@20 | 1.0000 | 0.9999 | 0.9672 | 0.9406 | 0.9474 |
| quota_picks | — | 0.2200 | 0.1550 | 0.6450 | 0.3650 |
| quota_shortfalls | — | 14.0400 | 14.1000 | 14.5100 | 10.9200 |

### B. Evaluación sin fuga (se re-embeben la consulta y el objetivo excluyendo los artículos que el objetivo comparte con la consulta; 685 de 1 415 casos son elegibles, de los cuales se toma una muestra de 200)
| Métrica | baseline | R0_solo_equidad | R1_suave | R2_moderado | R3_fair_binomial |
|---|---|---|---|---|---|
| nDCG@10 | 0.0624 | 0.0593 | 0.0497 | 0.0219 | 0.0334 |
| P@10 | 0.0140 | 0.0130 | 0.0115 | 0.0055 | 0.0080 |
| HR@10 | 0.1400 | 0.1300 | 0.1150 | 0.0550 | 0.0800 |
| nDCG@20 | 0.0789 | 0.0784 | 0.0586 | 0.0356 | 0.0434 |
| P@20 | 0.0103 | 0.0103 | 0.0075 | 0.0055 | 0.0060 |
| HR@20 | 0.2050 | 0.2050 | 0.1500 | 0.1100 | 0.1200 |
| GiniSimpson@20 | 0.6964 | 0.6963 | 0.9059 | 0.9344 | 0.9291 |
| GiniSimpson_m2@20 | 0.6486 | 0.6489 | 0.7102 | 0.4689 | 0.5960 |
| UniqueInst@20 | 7.8000 | 7.7950 | 13.3800 | 16.9000 | 15.8350 |
| InstCoverage@20 | 0.3082 | 0.3081 | 0.5653 | 0.7157 | 0.6715 |
| ProtectedShare@20 | 0.0270 | 0.0342 | 0.0372 | 0.0712 | 0.0515 |
| ExposureShareProt@20 | 0.0251 | 0.0344 | 0.0349 | 0.0718 | 0.0466 |
| MeanExpProt@20 | 0.1014 | 0.1417 | 0.1449 | 0.2465 | 0.1800 |
| MeanExpNonProt@20 | 0.3527 | 0.3520 | 0.3529 | 0.3519 | 0.3539 |
| NDKL@20 | 0.0598 | 0.0565 | 0.0536 | 0.0667 | 0.0503 |
| NDKLpop@20 | 0.1269 | 0.1149 | 0.1179 | 0.0956 | 0.1079 |
| RelRetained@20 | 1.0000 | 0.9998 | 0.9704 | 0.9469 | 0.9537 |
| quota_picks | — | 0.3800 | 0.2400 | 1.0550 | 0.5650 |
| quota_shortfalls | — | 13.3900 | 13.9200 | 13.3750 | 10.1100 |

### C. Sensibilidad de α en el baseline (sin fuga, 200 casos)
| alpha | nDCG@10 | HR@10 | nDCG@20 | HR@20 |
|---|---|---|---|---|
| 1.0 (media) | 0.0697 | 0.155 | 0.0924 | 0.245 |
| 0.5 (híbrido) | 0.0624 | 0.14 | 0.0789 | 0.205 |
| 0.0 (máximo) | 0.0428 | 0.1 | 0.0629 | 0.18 |

## Lectura honesta

1. **Fuga de información en la evaluación estándar.** El embedding del objetivo se construye con los mismos artículos que comparte con la consulta, porque el grupo se define precisamente por esos artículos conjuntos. Por eso el baseline alcanza HR@10 = 0,915 con un rango mediano de 1. Al eliminar esos artículos, HR@10 cae a 0,14 y HR@20 a 0,205, con un rango mediano de 24 cuando el objetivo sigue apareciendo en el Top-100 (47 % de los casos). La tabla B es la cifra defendible. Además, la evaluación sin fuga tiene sesgo de selección, porque solo incluye los casos donde el objetivo y los integrantes tienen otros artículos.
2. **La diversidad institucional y el leave-one-out están en tensión por construcción.** El 69,8 % de los grupos reales pertenece a una sola institución, de modo que penalizar las instituciones de la consulta aleja al objetivo de las primeras posiciones. Con R1 se obtiene Gini-Simpson@20 de 0,696 → 0,906 y 7,8 → 13,4 instituciones únicas, a cambio de HR@20 0,205 → 0,150 y una relevancia conservada del 97,0 %. R2 y R3 cuestan más precisión. Esa caída no implica peores grupos para formar equipos nuevos; muestra que el criterio de verdad de terreno premia la endogamia institucional.
3. **El anti-tokenismo baja con topes estrictos.** Un tope de 2 por institución dispersa la lista en muchas instituciones con un solo integrante, y por eso GiniSimpson_m2 baja (0,649 → 0,469 en R2). Con tope 3 (R1) el índice sube (0,710).
4. **La equidad de exposición está limitada por la oferta.** Solo el 4,9 % del Top-100 (media por consulta) son protegidos, frente al 12,8 % de la población. Como el piso de relevancia impide promover candidatos irrelevantes, la cuota queda incumplida en unas 13 a 14 de las 20 posiciones (en 10 con el modo binomial). La proporción de protegidos en el Top-20 sube de 2,7 % a 7,1 % como máximo (R2), y su exposición media por ítem pasa de 0,10 a 0,25. NDKL frente a la proporción poblacional mejora (0,127 → 0,096). NDKL frente al pool empeora en R2 (0,060 → 0,067) porque sobrerrepresenta a los protegidos respecto de un pool que ya los subrepresenta.
5. **R0 (solo equidad) casi no cuesta precisión** (nDCG@20 0,0789 → 0,0784), pero su efecto es pequeño (2,7 % → 3,4 % de protegidos) por la misma escasez de candidatos.
6. **α:** la media utilitaria pura (α = 1) supera al híbrido y al máximo en esta tarea (nDCG@20 de 0,092 frente a 0,079 y 0,063).
7. `--exclude-coauthors`, que excluye a los coautores frecuentes de la consulta, elimina al objetivo por definición (HR = 0 en todos los métodos). Es útil en producción, pero no se puede evaluar con este leave-one-out.
