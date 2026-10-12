# Ethical recommender service

Servicio de recomendación de equipos de investigación con arquitectura limpia:

- `domain/entities/` y `domain/repositories/`: entidades y contratos del dominio.
- `application/use_cases/` y `application/services/`: casos de uso y servicios.
- `infrastructure/api/v1/endpoints/`, `infrastructure/schemas/` e
  `infrastructure/db/`: adaptadores de API, esquemas y persistencia.
- `infrastructure/scripts/`: procesamiento offline de datos.
- `tests/`: pruebas automatizadas.
- `data/raw/` y `data/processed/`: datos locales excluidos de Git.

## Preparación y ejecución (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path data/raw,data/processed
.\.venv\Scripts\python.exe infrastructure/scripts/01_filter_and_merge_authors.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

En una copia nueva del repositorio, colocar los CSV originales en `data/raw/`.
El ETL requiere `authors.csv` (`identifier`, `auth_name`), `articles.csv`
(`identifier`, `title`, `abstract`) y `articles_authors.csv` (`author_id`,
`article_id`). Los nombres de archivos y columnas están definidos como constantes
al inicio del script. Se admiten CSV estándar y registros con una capa adicional
de comillas alrededor de toda la fila, presentes en el archivo de artículos.
Una fila con un número incorrecto de campos detiene la ejecución con un diagnóstico.

Se conservan autores con **al menos tres artículos distintos y válidos**.
Las relaciones duplicadas, sin ID o sin autor/artículo existente se descartan
antes de contar. Los IDs se mantienen como texto, incluidos los ceros iniciales.
Solo se construye texto para artículos de los autores que superan el filtro.
Cada artículo aporta `título\nresumen`; los artículos se ordenan por ID textual
y se separan por una línea en blanco. Un artículo sin texto cuenta como publicación
pero no añade contenido al perfil. Ante IDs repetidos en las tablas de entidades,
se conserva el primer nombre o texto encontrado.

La salida es `data/processed/filtered_authors_profiles.csv`, con columnas
`author_id`, `author_name`, `article_count` y `concatenated_profile_text`.
El proceso imprime conteos iniciales, supervivientes y porcentaje de retención.
Los errores de carga o escritura devuelven código de salida 1.

En este equipo, Windows bloquea una DLL de pandas instalada en `.venv`.
La ejecución sobre los CSV reales y las cinco pruebas se verificaron con el
Python incluido en Codex (pandas 3.0.1). Para reproducir esa ejecución aquí:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" infrastructure/scripts/01_filter_and_merge_authors.py
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m unittest discover -s tests -v
```

## Embeddings semánticos (Etapa 1) y estudio comparativo de modelos

```powershell
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model minilm
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model specter2
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model scincl
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model minilm-l12
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model mpnet
# prueba rápida con 20 autores (añade _sample20 al nombre de los archivos)
.\.venv\Scripts\python.exe infrastructure/scripts/02_generate_embeddings.py --model scincl --limit-authors 20
```

| `--model` | Modelo Hugging Face | Pooling por artículo | Longitud máx. | Entrada |
|---|---|---|---|---|
| `minilm` (defecto) | `sentence-transformers/all-MiniLM-L6-v2` | media de tokens | 256 | `título\nresumen` |
| `specter2` | `allenai/specter2_base` | `[CLS]` | 512 | `título[SEP]resumen` |
| `scincl` | `malteos/scincl` | `[CLS]` | 512 | `título[SEP]resumen` |
| `minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | media de tokens | 256 | `título\nresumen` |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | media de tokens | 384 | `título\nresumen` |

Los modelos generalistas (`minilm`, `minilm-l12`, `mpnet`) usan el pooling de su configuración en
Hugging Face. `all-MiniLM-L12-v2` trae `max_seq_length = 128` en su `sentence_bert_config.json`, pero
su tarjeta indica 256; aquí se usa 256 para truncar igual que la línea base L6.

Lee `data/processed/filtered_authors_profiles.csv` y genera, por modelo:

- `data/processed/author_embeddings_<modelo>.npy`: matriz `float32` de forma
  `(n_autores, dim)` con filas normalizadas L2 (similitud coseno = producto punto).
  `dim` es 384 para `minilm` y `minilm-l12`, y 768 para `specter2`, `scincl` y `mpnet`.
- `data/processed/author_index_mapping_<modelo>.json`: `index_to_author_id[i]` es el
  `author_id` de la fila `i`, junto con modelo, dimensión, pooling, longitud máxima,
  dispositivo, GPU, versiones de torch/CUDA, fp16, batch size y tiempo transcurrido.

Cada perfil se divide en sus artículos (separados por una línea en blanco). SPECTER2 y SciNCL
siguen sus tarjetas de modelo: `título + tokenizer.sep_token + resumen`, truncado a 512 tokens y
embedding del token `[CLS]`, construido con un `Transformer` + `Pooling(cls)` explícito. MiniLM
conserva la entrada y el pooling originales. Los textos repetidos (artículos en coautoría) se
codifican una sola vez y el vector del autor es el promedio renormalizado de sus artículos.
Al terminar, el script imprime el tiempo y una fila lista para `docs/execution_history.md`.

Compatibilidad: `author_embeddings.npy` / `author_index_mapping.json` (ejecución #1) equivalen a la
salida de `--model minilm`. Para seguir escribiendo esos nombres usar `--legacy-names`.

Opciones: `--device {auto,cuda,cpu}` (defecto `auto`: usa CUDA si `torch.cuda.is_available()`,
si no emite un WARNING y continúa en CPU; `--device cuda` sin GPU termina con error),
`--batch-size` (defecto 128 para MiniLM y 32 para SPECTER2/SciNCL en GPU; ante un error de memoria
CUDA se reduce a la mitad y se reintenta), `--fp16` / `--no-fp16` (fp16 activo por defecto en
CUDA), `--input`, `--output-dir`, `--limit-authors N`. La primera ejecución descarga el modelo.

SPECTER2: la tarjeta oficial recomienda cargar el adaptador de proximidad `allenai/specter2` sobre
`allenai/specter2_base` con la librería `adapters`. Aquí se usa el modelo base, porque `adapters`
exige `transformers` 4.x y es incompatible con la versión de `sentence-transformers` del proyecto.
Evaluar el adaptador queda como trabajo futuro.

### GPU / CUDA (Windows, NVIDIA RTX 3060)

PyTorch se instala aparte para obtener la build con CUDA 12.6 (PyTorch 2.14.1 estable):

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

La verificación debe mostrar algo como `2.14.1+cu126 12.6 True NVIDIA GeForce RTX 3060`.
Requiere un controlador NVIDIA compatible con CUDA 12 (≥ 528.33 en Windows; se recomienda el
controlador Game Ready/Studio más reciente). `nvidia-smi` muestra la versión instalada.
La bitácora de ejecuciones está en `docs/execution_history.md`.

Las ramas de trabajo iniciales son `main`, `develop` y `feat/setup-etl-pipeline`.
Los CSV que estaban versionados se retiraron del índice, conservando los archivos
locales en `data/raw/`; las versiones anteriores permanecen en el historial de Git.

## Grupos de verdad de terreno y re-ranking ético

```bash
python infrastructure/scripts/01_filter_and_merge_authors.py
python infrastructure/scripts/02_generate_embeddings.py --device cpu
python -m infrastructure.scripts.03_mine_ground_truth_groups --max-k 0      # 0 = sin límite de k
python -m infrastructure.scripts.04_run_rerank_experiment                  # evaluación estándar (minilm)
python -m infrastructure.scripts.04_run_rerank_experiment --leakage-free --tag _leakage_free
python -m infrastructure.scripts.04_run_rerank_experiment --model scincl --leakage-free --tag _leakage_free
python -m pytest -q tests
```

`04_run_rerank_experiment.py --model {minilm,specter2,scincl}` (defecto `minilm`) lee
`data/processed/author_embeddings_<modelo>.npy` y `author_index_mapping_<modelo>.json`, generados con
`02_generate_embeddings.py --model <modelo>`. Con `minilm`, si no existen, usa los nombres anteriores
`author_embeddings.npy` / `author_index_mapping.json` (lo informa en el log). `--embeddings` y
`--mapping` permiten indicar rutas explícitas. Se valida que filas e IDs coincidan, que la dimensión
corresponda a la declarada y que el modelo del mapping sea el pedido. Las salidas incluyen el alias:
`experiments/rerank_{summary,per_case,meta,results}_<modelo><tag>.*` (`--results-dir` cambia la carpeta).
En `--leakage-free` los integrantes y el objetivo se re-codifican con el mismo modelo, pooling,
longitud máxima y formato de entrada del mapping (`título[SEP]resumen` y `[CLS]` para SPECTER2/SciNCL;
`--device auto|cuda|cpu`).

- `application/services/apriori_groups.py`: Apriori por niveles con tidsets (Eclat), poda anti-monótona,
  guarda de hiper-autoría, marcas maximal/closed.
- `application/services/evaluation_split.py`: partición leave-one-out con semilla.
- `application/services/group_scoring.py`: agregación híbrida (Additive Utilitarian + Most Pleasure).
- `application/services/fair_reranker.py`: re-ranking voraz MMR con tope institucional y cuota de baja visibilidad.
- `application/services/ranking_metrics.py`: nDCG/P/Recall/HR, Gini-Simpson (m ≥ 2), exposición, NDKL.
- `infrastructure/db/csv_sources.py`: afiliación principal, visibilidad (artículos y grado de coautoría).
- Salidas en `data/processed/`: `ground_truth_groups.csv`, `loo_eval_split.csv`, `experiments/`.
- Documentación: `documentacion/semanas-5-6-apriori-y-reranking.md`, `documentacion/resultados-apriori-reranking.md`.
