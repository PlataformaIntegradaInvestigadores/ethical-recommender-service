# Bitácora de ejecución — Generación de embeddings (Fase 1)

Registro de cada ejecución de `infrastructure/scripts/02_generate_embeddings.py` para el estudio
comparativo de modelos de embeddings del TIC. Cada fila documenta qué se ejecutó, con qué
configuración y en qué hardware, de modo que los resultados de la tesis sean reproducibles.

Modelos del estudio:

| Alias | Modelo (Hugging Face) | Tipo | Pooling por artículo | Longitud máx. | Formato de entrada |
|---|---|---|---|---|---|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | Generalista (línea base) | media de tokens | 256 | `título\nresumen` |
| `specter2` | `allenai/specter2_base` | Científico, basado en citas | token `[CLS]` | 512 | `título[SEP]resumen` |
| `scincl` | `malteos/scincl` | Científico, grafo de citas contrastivo | token `[CLS]` | 512 | `título[SEP]resumen` |
| `minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | Generalista, 12 capas | media de tokens | 256 | `título\nresumen` |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | Generalista, base | media de tokens | 384 | `título\nresumen` |

En los tres casos el vector del autor es el promedio de los embeddings L2-normalizados de sus
artículos, renormalizado a norma 1 (similitud coseno = producto punto).

## Resumen de ejecuciones

| # | Fecha | Script | Modelo | Input | Output | Hardware | Tiempo |
|---|---|---|---|---|---|---|---|
| 1 | Octubre 2026 | `02_generate_embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×384 (`author_embeddings.npy`, ~11.7 MB) | CPU | ~16 min |
| 2 | 2026-10-10 | `02_generate_embeddings.py --model specter2` | `allenai/specter2_base` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×768 (`author_embeddings_specter2.npy`, ~23.4 MB) | GPU NVIDIA GeForce RTX 3060 Laptop (6 GB), fp16 | 5.3 min (319 s) |
| 3 | 2026-10-10 | `02_generate_embeddings.py --model scincl` | `malteos/scincl` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×768 (`author_embeddings_scincl.npy`, ~23.4 MB) | GPU NVIDIA GeForce RTX 3060 Laptop (6 GB), fp16 | 4.0 min (242 s) |
| 4 | 2026-10-11 | `02_generate_embeddings.py --model minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×384 (`author_embeddings_minilm-l12.npy`, ~11.7 MB) | GPU Tesla T4 (Google Colab), fp16 | 1.5 min (92.5 s) |
| 5 | 2026-10-11 | `02_generate_embeddings.py --model mpnet` | `sentence-transformers/all-mpnet-base-v2` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×768 (`author_embeddings_mpnet.npy`, ~23.4 MB) | GPU Tesla T4 (Google Colab), fp16 | 4.0 min (241.7 s) |

Al terminar, el script imprime una fila Markdown lista para pegar en la sección de detalle.

Filas impresas por el script en las ejecuciones en GPU (equipo de René):

| Fecha | Comando | Modelo | Pooling | Long. máx. | Hardware | torch / CUDA | Batch | fp16 | Salida | Tiempo |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | 02_generate_embeddings.py --model specter2 | allenai/specter2_base | cls | 512 | GPU NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB) | torch 2.14.1+cu126 / CUDA 12.6 | 32 | sí | 7621×768 | 5.3 min (319 s) |
| 2026-10-10 | 02_generate_embeddings.py --model scincl | malteos/scincl | cls | 512 | GPU NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB) | torch 2.14.1+cu126 / CUDA 12.6 | 32 | sí | 7621×768 | 4.0 min (242 s) |

> **Ejecuciones #4 y #5 en Google Colab.** En el equipo de René, Windows Application Control
> (control de aplicaciones de Windows) bloqueó la carga de una DLL de `scipy` dentro de `.venv`, por lo
> que ambas ejecuciones se hicieron en Google Colab con una GPU Tesla T4 (torch 2.11.0+cu130), usando el
> script de la rama `feat/generalist-models`. Colab trabaja en UTC y sus filas impresas muestran la fecha
> 2026-10-12; la fecha local (UTC-5) es 2026-10-11. Por eso los tiempos no son comparables con los de la
> RTX 3060 de las ejecuciones #2 y #3.

Frente a la ejecución #1 en CPU (~16 min con un modelo 6 capas y 256 tokens), la GPU procesó dos
modelos de 12 capas y 512 tokens en 4–5 min cada uno (~151 textos/s).

---

## Ejecución #1 — all-MiniLM-L6-v2 (línea base)

- **Fecha:** Octubre 2026
- **Script:** `infrastructure/scripts/02_generate_embeddings.py`
- **Modelo:** `sentence-transformers/all-MiniLM-L6-v2`
- **Input:** 33,477 artículos únicos / 7,621 autores (`data/processed/filtered_authors_profiles.csv`)
- **Output:** matriz 7621×384 `float32` (`author_embeddings.npy`, ~11.7 MB) y `author_index_mapping.json`
- **Hardware:** CPU
- **Tiempo:** ~16 min

> **Nota (re-ejecución en el servidor de análisis, no es la ejecución oficial):** al reproducir esta
> misma ejecución en el servidor de análisis (8 núcleos de CPU, sin GPU) tomó ~5 min.

> **Re-ejecución con el script refactorizado (2026-10-10):** René ejecutó `--model minilm` con la
> nueva versión del script y confirmó que la matriz es idéntica a `author_embeddings.npy`; por eso
> la ejecución #1 sigue siendo la referencia de MiniLM. Tiempo y hardware de esa re-ejecución: no registrados.

> **Compatibilidad:** desde la refactorización del estudio comparativo, el script escribe
> `author_embeddings_minilm.npy` / `author_index_mapping_minilm.json`. Ese par es equivalente al
> `author_embeddings.npy` / `author_index_mapping.json` de esta ejecución (mismo modelo, misma entrada
> y mismo pooling). Para regenerar los nombres antiguos: `--model minilm --legacy-names`.

---

## Ejecución #2 — SPECTER2 (`allenai/specter2_base`)

| Campo | Valor |
|---|---|
| Fecha | 2026-10-10 |
| Commit (`git rev-parse --short HEAD`) | no registrado (rama `feat/models-comparative-study`) |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model specter2` (dispositivo `auto` → CUDA) |
| Modelo / HF id | `allenai/specter2_base` |
| Revisión del modelo (hash HF) | no registrado |
| Pooling | `[CLS]` por artículo → media L2 por autor |
| `max_seq_length` | 512 |
| Formato de entrada | `título + [SEP] + resumen` |
| Hardware / GPU / VRAM | NVIDIA GeForce RTX 3060 Laptop GPU, 6.0 GB |
| Versión torch / CUDA | torch 2.14.1+cu126 / CUDA 12.6 |
| Batch size (final, tras reintentos OOM) | 32 (sin reintentos registrados) |
| fp16 | sí |
| Tiempo | 5.3 min (319.0 s), incluye la primera descarga del modelo (~440 MB) |
| VRAM pico | no registrado |
| Textos codificados | 33,477 textos únicos de 65,607 documentos artículo-autor (~151 textos/s) |
| Forma y tamaño de salida | 7621×768 `float32` (~23.4 MB) |
| Archivos generados | `author_embeddings_specter2.npy`, `author_index_mapping_specter2.json` |
| Observaciones | Modelo base sin adaptador de proximidad (`allenai/specter2`); ver README. Verificado contra la prueba de humo de 20 autores en CPU: diferencia máx. ~2e-4 (efecto de fp16). |

## Ejecución #3 — SciNCL (`malteos/scincl`)

| Campo | Valor |
|---|---|
| Fecha | 2026-10-10 |
| Commit (`git rev-parse --short HEAD`) | no registrado (rama `feat/models-comparative-study`) |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model scincl` (dispositivo `auto` → CUDA) |
| Modelo / HF id | `malteos/scincl` |
| Revisión del modelo (hash HF) | no registrado |
| Pooling | `[CLS]` por artículo → media L2 por autor |
| `max_seq_length` | 512 |
| Formato de entrada | `título + [SEP] + resumen` |
| Hardware / GPU / VRAM | NVIDIA GeForce RTX 3060 Laptop GPU, 6.0 GB |
| Versión torch / CUDA | torch 2.14.1+cu126 / CUDA 12.6 |
| Batch size (final, tras reintentos OOM) | 32 (sin reintentos registrados) |
| fp16 | sí |
| Tiempo | 4.0 min (242.2 s), incluye la primera descarga del modelo (~440 MB) |
| VRAM pico | no registrado |
| Textos codificados | 33,477 textos únicos de 65,607 documentos artículo-autor |
| Forma y tamaño de salida | 7621×768 `float32` (~23.4 MB) |
| Archivos generados | `author_embeddings_scincl.npy`, `author_index_mapping_scincl.json` |
| Observaciones | Verificado contra la prueba de humo de 20 autores en CPU: diferencia máx. ~2e-4 (efecto de fp16). |

## Ejecución #4 — all-MiniLM-L12-v2 (`sentence-transformers/all-MiniLM-L12-v2`)

| Campo | Valor |
|---|---|
| Fecha | 2026-10-11 (UTC-5; Colab registró 2026-10-12 en UTC) |
| Commit (`git rev-parse --short HEAD`) | no registrado (rama `feat/generalist-models`, `7e95463` o posterior) |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model minilm-l12` (Colab, dispositivo `auto` → CUDA) |
| Modelo / HF id | `sentence-transformers/all-MiniLM-L12-v2` |
| Revisión del modelo (hash HF) | no registrado |
| Pooling | media de tokens (configuración del hub) por artículo → media L2 por autor |
| `max_seq_length` | 256 (el hub trae 128; se usa 256 como la tarjeta y la línea base) |
| Formato de entrada | `título\nresumen` (igual que la línea base) |
| Hardware / GPU / VRAM | Google Colab, NVIDIA Tesla T4 (16 GB); motivo: bloqueo de Windows Application Control en el equipo local |
| Versión torch / CUDA | torch 2.11.0+cu130 / CUDA 13.0 |
| Batch size (final, tras reintentos OOM) | 128 (sin reintentos registrados) |
| fp16 | sí |
| Tiempo | 1.5 min (92.5 s), incluye la descarga del modelo |
| VRAM pico | no registrado |
| Textos codificados | 33,477 textos únicos de 65,607 documentos artículo-autor |
| Forma y tamaño de salida | 7621×384 `float32` (~11.7 MB) |
| Archivos generados | `author_embeddings_minilm-l12.npy`, `author_index_mapping_minilm-l12.json` |
| Observaciones | Verificado contra la prueba de humo de 20 autores en CPU: diferencia máx. 3.2e-4 (fp16), coseno mínimo 0.999999; mismo orden de autores que los demás modelos. |

## Ejecución #5 — all-mpnet-base-v2 (`sentence-transformers/all-mpnet-base-v2`)

| Campo | Valor |
|---|---|
| Fecha | 2026-10-11 (UTC-5; Colab registró 2026-10-12 en UTC) |
| Commit (`git rev-parse --short HEAD`) | no registrado (rama `feat/generalist-models`, `7e95463` o posterior) |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model mpnet` (Colab, dispositivo `auto` → CUDA) |
| Modelo / HF id | `sentence-transformers/all-mpnet-base-v2` |
| Revisión del modelo (hash HF) | no registrado |
| Pooling | media de tokens (configuración del hub) por artículo → media L2 por autor |
| `max_seq_length` | 384 |
| Formato de entrada | `título\nresumen` (igual que la línea base) |
| Hardware / GPU / VRAM | Google Colab, NVIDIA Tesla T4 (16 GB); motivo: bloqueo de Windows Application Control en el equipo local |
| Versión torch / CUDA | torch 2.11.0+cu130 / CUDA 13.0 |
| Batch size (final, tras reintentos OOM) | 64 (sin reintentos registrados) |
| fp16 | sí |
| Tiempo | 4.0 min (241.7 s), incluye la descarga del modelo |
| VRAM pico | no registrado |
| Textos codificados | 33,477 textos únicos de 65,607 documentos artículo-autor |
| Forma y tamaño de salida | 7621×768 `float32` (~23.4 MB) |
| Archivos generados | `author_embeddings_mpnet.npy`, `author_index_mapping_mpnet.json` |
| Observaciones | Verificado contra la prueba de humo de 20 autores en CPU: diferencia máx. 2.2e-4 (fp16), coseno mínimo 0.999999; mismo orden de autores que los demás modelos. |

---

## Plantilla para nuevas ejecuciones

```markdown
## Ejecución #N — <modelo>

| Campo | Valor |
|---|---|
| Fecha | |
| Commit | |
| Comando exacto | |
| Modelo / HF id | |
| Revisión del modelo | |
| Pooling | |
| `max_seq_length` | |
| Formato de entrada | |
| Hardware / GPU / VRAM | |
| Versión torch / CUDA | |
| Batch size | |
| fp16 | |
| Tiempo | |
| VRAM pico | |
| Forma y tamaño de salida | |
| Archivos generados | |
| Observaciones | |
```

Comandos útiles para completar los campos:

```powershell
git rev-parse --short HEAD
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv -l 5   # VRAM pico durante la ejecución
```

La revisión del modelo es la carpeta de `snapshots` en
`%USERPROFILE%\.cache\huggingface\hub\models--<org>--<modelo>\snapshots\`.

## Pruebas de humo (no forman parte del estudio)

| Fecha | Comando | Hardware | Resultado |
|---|---|---|---|
| 2026-10-10 | `--model scincl --limit-authors 20` | CPU servidor de análisis (8 núcleos) | 20×768, 239 artículos (236 únicos), 23.8 s, ~16.5 art./s |
| 2026-10-10 | `--model specter2 --limit-authors 20` | CPU servidor de análisis (8 núcleos) | 20×768, 239 artículos (236 únicos), 22.4 s, ~17.3 art./s |
| 2026-10-11 | `--model minilm-l12 --limit-authors 20` | CPU servidor de análisis (8 núcleos) | 20×384, 17.6 s (incluye carga); diferencia máx. 3.7e-8 frente a `SentenceTransformer.encode` oficial con 256 tokens |
| 2026-10-11 | `--model mpnet --limit-authors 20` | CPU servidor de análisis (8 núcleos) | 20×768, 34.6 s (incluye carga); idéntico (diferencia 0.0) a `SentenceTransformer.encode` oficial |

Ambas salidas coinciden (diferencia máx. < 1e-7) con la referencia de las tarjetas de los modelos
(`transformers` + `last_hidden_state[:, 0, :]` + normalización L2). Al ritmo observado en CPU, una
ejecución completa (33,477 artículos) tomaría ~34 min por modelo, de ahí el uso de la GPU.
