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

Las ramas de trabajo iniciales son `main`, `develop` y `feat/setup-etl-pipeline`.
Los CSV que estaban versionados se retiraron del índice, conservando los archivos
locales en `data/raw/`; las versiones anteriores permanecen en el historial de Git.
