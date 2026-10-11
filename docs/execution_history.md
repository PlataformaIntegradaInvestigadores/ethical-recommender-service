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

En los tres casos el vector del autor es el promedio de los embeddings L2-normalizados de sus
artículos, renormalizado a norma 1 (similitud coseno = producto punto).

## Resumen de ejecuciones

| # | Fecha | Script | Modelo | Input | Output | Hardware | Tiempo |
|---|---|---|---|---|---|---|---|
| 1 | Octubre 2026 | `02_generate_embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×384 (`author_embeddings.npy`, ~11.7 MB) | CPU | ~16 min |
| 2 | _pendiente_ | `02_generate_embeddings.py --model specter2` | `allenai/specter2_base` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×768 (esperada) | GPU NVIDIA RTX 3060 (6 GB) | _pendiente_ |
| 3 | _pendiente_ | `02_generate_embeddings.py --model scincl` | `malteos/scincl` | 33,477 artículos únicos / 7,621 autores | Matriz 7621×768 (esperada) | GPU NVIDIA RTX 3060 (6 GB) | _pendiente_ |

Al terminar, el script imprime una fila Markdown lista para pegar en la sección de detalle.

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

> **Compatibilidad:** desde la refactorización del estudio comparativo, el script escribe
> `author_embeddings_minilm.npy` / `author_index_mapping_minilm.json`. Ese par es equivalente al
> `author_embeddings.npy` / `author_index_mapping.json` de esta ejecución (mismo modelo, misma entrada
> y mismo pooling). Para regenerar los nombres antiguos: `--model minilm --legacy-names`.

---

## Ejecución #2 — SPECTER2 (`allenai/specter2_base`)

| Campo | Valor |
|---|---|
| Fecha | |
| Commit (`git rev-parse --short HEAD`) | |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model specter2 --device cuda` |
| Modelo / HF id | `allenai/specter2_base` |
| Revisión del modelo (hash HF) | |
| Pooling | `[CLS]` por artículo → media L2 por autor |
| `max_seq_length` | 512 |
| Formato de entrada | `título + [SEP] + resumen` |
| Hardware / GPU / VRAM | NVIDIA GeForce RTX 3060, 6 GB |
| Versión torch / CUDA | |
| Batch size (final, tras reintentos OOM) | 32 (por defecto) |
| fp16 | sí (por defecto en CUDA) |
| Tiempo | |
| VRAM pico | |
| Forma y tamaño de salida | 7621×768 `float32` (~23.4 MB esperado) |
| Archivos generados | `author_embeddings_specter2.npy`, `author_index_mapping_specter2.json` |
| Observaciones | Modelo base sin adaptador de proximidad (`allenai/specter2`); ver README. |

## Ejecución #3 — SciNCL (`malteos/scincl`)

| Campo | Valor |
|---|---|
| Fecha | |
| Commit (`git rev-parse --short HEAD`) | |
| Comando exacto | `python infrastructure/scripts/02_generate_embeddings.py --model scincl --device cuda` |
| Modelo / HF id | `malteos/scincl` |
| Revisión del modelo (hash HF) | |
| Pooling | `[CLS]` por artículo → media L2 por autor |
| `max_seq_length` | 512 |
| Formato de entrada | `título + [SEP] + resumen` |
| Hardware / GPU / VRAM | NVIDIA GeForce RTX 3060, 6 GB |
| Versión torch / CUDA | |
| Batch size (final, tras reintentos OOM) | 32 (por defecto) |
| fp16 | sí (por defecto en CUDA) |
| Tiempo | |
| VRAM pico | |
| Forma y tamaño de salida | 7621×768 `float32` (~23.4 MB esperado) |
| Archivos generados | `author_embeddings_scincl.npy`, `author_index_mapping_scincl.json` |
| Observaciones | |

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

Ambas salidas coinciden (diferencia máx. < 1e-7) con la referencia de las tarjetas de los modelos
(`transformers` + `last_hidden_state[:, 0, :]` + normalización L2). Al ritmo observado en CPU, una
ejecución completa (33,477 artículos) tomaría ~34 min por modelo, de ahí el uso de la GPU.
