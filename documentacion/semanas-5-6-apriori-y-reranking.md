# Grupos de verdad de terreno (Apriori) y re-ranking ético del Top@20

## 1. Minería a priori de grupos de coautoría

**Modelo transaccional.** Cada artículo es una transacción y sus autores son los ítems. Solo se admiten como ítems los 7 621 autores con al menos tres artículos (el mismo filtro de `01_filter_and_merge_authors.py`). Un conjunto de autores *X* es un **grupo frecuente** si sus integrantes firmaron juntos al menos `min_support` artículos (por defecto 3). El soporte de *X* es |tid(*X*)|, donde tid(*X*) es el conjunto de identificadores de los artículos firmados por todos los integrantes de *X*.

**Control del espacio de búsqueda.** Enumerar todos los subconjuntos posibles de autores implica 2^7621 candidatos, y solo los pares ya suman 28 929 421. Apriori recorre el espacio por niveles y aplica la **anti-monotonía** del soporte: supp(*X* ∪ {a}) ≤ supp(*X*). Por eso un *k*-conjunto solo puede ser frecuente si todos sus (*k*−1)-subconjuntos lo son. Los candidatos C_k se generan uniendo únicamente conjuntos de L_{k−1} que comparten el mismo prefijo de *k*−2 autores. Se descarta antes de contar todo candidato que tenga algún subconjunto infrecuente. El soporte se cuenta de forma vertical, al estilo Eclat, con la operación tid(P∪{a,b}) = tid(P∪{a}) ∩ tid(P∪{b}). Así se evita volver a recorrer los artículos después del primer nivel. En el nivel 2 todos los pares comparten el prefijo vacío, por lo que C_2 se restringe a los pares que coocurren en al menos un artículo. Esta restricción es exacta, porque un par sin coocurrencias tiene soporte 0.

**Guardas.** Los artículos con más de `max_authors_per_article` autores (25 por defecto) se ignoran, porque generarían combinatoriamente muchos grupos espurios. La profundidad `max_k` es configurable (5 por defecto; con 0 no tiene límite). Si el último nivel minado no queda vacío, el resultado se marca como `truncated`. Se conservan los indicadores `maximal` (no existe un superconjunto frecuente) y `closed` (no existe un superconjunto con el mismo soporte).

```text
Algoritmo 1. GruposFrecuentesApriori(T, A, s_min, k_max, n_max)
Entrada: T = {artículo → autores}, A = autores admitidos (≥ 3 artículos),
         s_min = soporte mínimo, k_max = tamaño máximo, n_max = autores máx. por artículo
Salida:  grupos frecuentes con soporte, tidset y marcas maximal/closed
 1: T' ← {t ∈ T : |autores(t)| ≤ n_max};  descartados ← |T| − |T'|
 2: para cada autor a ∈ A: tid(a) ← {t ∈ T' : a ∈ autores(t)}            ▷ formato vertical
 3: L1 ← {{a} : |tid(a)| ≥ s_min}
 4: C2 ← {{a,b} ⊆ L1 : a y b coocurren en algún t ∈ T'}                ▷ exacto: el resto tiene soporte 0
 5: L2 ← {X ∈ C2 : |tid(a) ∩ tid(b)| ≥ s_min}
 6: k ← 3
 7: mientras k ≤ k_max y L_{k−1} ≠ ∅ hacer
 8:     C_k ← ∅
 9:     para cada par P∪{a}, P∪{b} ∈ L_{k−1} con prefijo común P (|P| = k−2), a < b hacer
10:         X ← P ∪ {a, b}
11:         si todo (k−1)-subconjunto de X ∈ L_{k−1} entonces C_k ← C_k ∪ {X}   ▷ poda anti-monótona
12:     L_k ← ∅
13:     para cada X ∈ C_k hacer
14:         tid(X) ← tid(P∪{a}) ∩ tid(P∪{b})                             ▷ conteo por intersección
15:         si |tid(X)| ≥ s_min entonces L_k ← L_k ∪ {X}
16:     k ← k + 1
17: para cada X ∈ ⋃ L_k (k ≥ 2):
18:     maximal(X) ← ∄ Y ∈ L_{|X|+1} con X ⊂ Y
19:     closed(X)  ← ∄ Y ∈ L_{|X|+1} con X ⊂ Y y |tid(Y)| = |tid(X)|
20: devolver grupos, estadísticas por nivel, descartados
```

**Partición de evaluación leave-one-out.** En cada grupo maximal con tamaño ≥ 3 se oculta un integrante, elegido con un generador aleatorio con semilla. Los demás integrantes forman la consulta y el integrante oculto es el único ítem relevante para calcular P@k, Recall@k, nDCG@k y la tasa de acierto (HR@k).

## 2. Puntuación grupal y re-ranking ético

**Etapa 1: afinidad grupal.** No se fusionan los textos de los integrantes. Cada integrante m_i conserva su embedding e_{m_i} (all-MiniLM-L6-v2, 384 dimensiones, normalizado L2). Para cada candidato c se calcula s_i = cos(e_c, e_{m_i}) y se aplica la agregación híbrida de Boratto et al.:

score(c) = α · Σ_i w_i s_i (*Additive Utilitarian*) + (1−α) · max_i s_i (*Most Pleasure*),

con pesos w_i ≥ 0 normalizados, uniformes por defecto. Se excluye a los integrantes de la consulta y, si se configura, también a sus coautores frecuentes. Después se recuperan los Top-N = 100 candidatos.

**Etapa 2: re-ranking voraz con restricciones.** La relevancia se normaliza con min-max dentro del Top-N y se elige una posición a la vez:

final(c) = λ · rel_norm(c) + (1−λ) · nov_inst(c),  con nov_inst(c) = 1 / (1 + n_sel(inst_c) + β·[inst_c ∈ inst(G)]).

La regla es de tipo MMR: penaliza las instituciones ya presentes en la lista y las del propio grupo consultado. UNKNOWN cuenta como una categoría propia, pero con novedad 0, y no se toma en cuenta como diversidad. El re-ranking cumple dos restricciones:

1. Tope institucional: como máximo `max_per_institution` integrantes por institución conocida.
2. Equidad por prefijos para investigadores de baja visibilidad (estilo fa*ir): cada prefijo de longitud i debe contener al menos m(i) investigadores protegidos. Con la cuota simple, m(i) = ⌊p·i⌋. Con la variante binomial, m(i) es el menor m que cumple BinomCDF(m; i, p) > α_f. La cuota solo se cubre con candidatos que superan el piso de relevancia (`relevance_floor`). Si no hay candidatos elegibles, se registra el déficit y no se promueve a nadie irrelevante.

Con temperatura τ > 0, cada elección se muestrea mediante softmax(final/τ) con semilla fija (exposición probabilística, estilo COIN). Con τ = 0 el algoritmo es determinista.

```text
Algoritmo 2. ReRankingÉtico(C, K, λ, cap, p, piso, inst(G))
Entrada: C = Top-N candidatos (relevancia, institución, protegido), K = tamaño de la lista final
Salida:  lista R de longitud K y diagnóstico (picks por cuota, déficits, relajaciones de tope)
 1: rel_norm ← minmax(relevancia(C));  m(1..K) ← cuota_mínima(K, p)
 2: R ← [];  cuenta_inst ← 0;  prot ← 0
 3: para i = 1..K hacer
 4:     D ← {c ∈ C∖R : inst(c) = UNKNOWN o cuenta_inst[inst(c)] < cap}
 5:     si D = ∅ entonces D ← C∖R                                 ▷ relajación registrada
 6:     si prot < m(i) entonces
 7:         E ← {c ∈ D : protegido(c) y rel_norm(c) ≥ piso}
 8:         si E ≠ ∅ entonces D ← E  si no  déficit ← déficit + 1    ▷ nunca promueve irrelevantes
 9:     para cada c ∈ D: final(c) ← λ·rel_norm(c) + (1−λ)·nov_inst(c | cuenta_inst, inst(G))
10:     c* ← argmax_{c∈D} final(c)   (o muestreo softmax(final/τ) con semilla)
11:     R ← R ∥ c*;  cuenta_inst[inst(c*)] += 1;  prot ← prot + protegido(c*)
12: devolver R
```

**Visibilidad (atributo protegido).** Un investigador es de baja visibilidad si su número de artículos es ≤ Q1 (3) y su grado en la red de coautoría es ≤ Q1 (5 coautores distintos). Los cuartiles se calculan sobre los 7 621 autores filtrados, y el grafo excluye los artículos con más de 25 autores. Con esta definición, 974 autores (12,8 %) quedan como protegidos.

**Métricas.** Se calculan nDCG@k, P@k, Recall@k y HR@k frente al integrante oculto. La diversidad institucional se mide con el índice de Gini-Simpson (sin contar UNKNOWN), con la versión anti-tokenismo m ≥ 2 (las instituciones con menos de dos integrantes se agrupan en una categoría residual), con el número de instituciones únicas y con la cobertura institucional del Top-N. Para la equidad se mide la proporción de protegidos, la exposición con descuento logarítmico de protegidos frente a no protegidos y NDKL, con dos referencias: la proporción de protegidos en el Top-N y la proporción en la población. La relevancia conservada es Σ rel(re-rank) / Σ rel(baseline).
