"""Generate L2-normalized author embeddings by mean-pooling per-article sentence embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
INPUT_PATH: Final[Path] = PROCESSED_DIR / "filtered_authors_profiles.csv"
EMBEDDINGS_PATH: Final[Path] = PROCESSED_DIR / "author_embeddings.npy"
MAPPING_PATH: Final[Path] = PROCESSED_DIR / "author_index_mapping.json"
DEFAULT_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = 64
ENCODE_CHUNK_SIZE: Final[int] = 2048
ARTICLE_SEPARATOR: Final[str] = "\n\n"
POOLING_STRATEGY: Final[str] = "mean_of_l2_normalized_article_embeddings"
AUTHOR_ID_COL: Final[str] = "author_id"
PROFILE_TEXT_COL: Final[str] = "concatenated_profile_text"

FloatMatrix = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]
logger = logging.getLogger("etl.generate_embeddings")


class EmbeddingPipelineError(RuntimeError):
    """The embedding pipeline cannot complete."""


def load_profiles(path: Path) -> pd.DataFrame:
    """Load author profiles, validate the schema, and drop authors without usable text."""
    if not path.is_file():
        raise EmbeddingPipelineError(
            f"Input file not found: {path}. Run 01_filter_and_merge_authors.py first."
        )
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise EmbeddingPipelineError(f"Could not read {path}: {exc}") from exc
    missing = [column for column in (AUTHOR_ID_COL, PROFILE_TEXT_COL) if column not in frame.columns]
    if missing:
        raise EmbeddingPipelineError(f"{path.name} is missing columns {missing}. Found: {list(frame.columns)}")
    duplicated = frame[AUTHOR_ID_COL].duplicated()
    if duplicated.any():
        raise EmbeddingPipelineError(f"{path.name} has {int(duplicated.sum())} duplicated author_id values")
    frame[PROFILE_TEXT_COL] = frame[PROFILE_TEXT_COL].str.strip()
    empty = frame[PROFILE_TEXT_COL] == ""
    if empty.any():
        logger.warning("Dropping %d authors with empty profile text.", int(empty.sum()))
    return frame.loc[~empty, [AUTHOR_ID_COL, PROFILE_TEXT_COL]].reset_index(drop=True)


def split_into_documents(profile_texts: pd.Series) -> tuple[list[str], IndexArray]:
    """Split each profile into per-article documents and record the owning author row."""
    documents: list[str] = []
    owners: list[int] = []
    for row_index, text in enumerate(profile_texts):
        for chunk in text.split(ARTICLE_SEPARATOR):
            chunk = chunk.strip()
            if chunk:
                documents.append(chunk)
                owners.append(row_index)
    return documents, np.asarray(owners, dtype=np.int64)


def load_model(model_name: str, device: str | None) -> SentenceTransformer:
    """Load a sentence-transformers model, downloading it on first use."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingPipelineError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise EmbeddingPipelineError(f"Could not load model '{model_name}': {exc}") from exc
    logger.info(
        "Loaded model %s on %s (dim=%s, max_seq_length=%s)", model_name, model.device,
        model.get_sentence_embedding_dimension(), model.max_seq_length,
    )
    return model


def encode_documents(model: SentenceTransformer, documents: list[str], batch_size: int) -> FloatMatrix:
    """Encode documents into L2-normalized float32 vectors with a progress bar."""
    dimension = model.get_sentence_embedding_dimension()
    if not dimension:
        raise EmbeddingPipelineError("Model does not report a fixed embedding dimension.")
    embeddings = np.empty((len(documents), dimension), dtype=np.float32)
    with tqdm(total=len(documents), desc="Encoding articles", unit="doc") as progress:
        for start in range(0, len(documents), ENCODE_CHUNK_SIZE):
            batch = documents[start:start + ENCODE_CHUNK_SIZE]
            embeddings[start:start + len(batch)] = model.encode(
                batch, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            progress.update(len(batch))
    return embeddings


def encode_unique_documents(model: SentenceTransformer, documents: list[str], batch_size: int) -> FloatMatrix:
    """Encode each distinct text once, since co-authored articles repeat across profiles."""
    positions: dict[str, int] = {}
    inverse = np.fromiter(
        (positions.setdefault(document, len(positions)) for document in documents),
        dtype=np.int64, count=len(documents),
    )
    logger.info("Encoding %d unique texts out of %d article documents.", len(positions), len(documents))
    return encode_documents(model, list(positions), batch_size)[inverse]


def pool_by_author(document_embeddings: FloatMatrix, owners: IndexArray, author_count: int) -> FloatMatrix:
    """Mean-pool contiguous document embeddings per author and re-normalize to unit length."""
    if len(owners) == 0:
        return np.empty((0, document_embeddings.shape[1]), dtype=np.float32)
    boundaries = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    if len(boundaries) != author_count or not np.array_equal(owners[boundaries], np.arange(author_count)):
        raise EmbeddingPipelineError(f"Pooling mismatch: {len(boundaries)} groups for {author_count} authors")
    sums = np.add.reduceat(document_embeddings, boundaries, axis=0)
    counts = np.diff(np.r_[boundaries, len(owners)]).astype(np.float32)
    means = sums / counts[:, np.newaxis]
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise EmbeddingPipelineError("Encountered a zero-norm author embedding.")
    return (means / norms).astype(np.float32)


def _atomic_write(path: Path, write: Any) -> None:
    """Write through a temporary sibling file and atomically replace the target."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            write(handle)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EmbeddingPipelineError(f"Could not write {path}: {exc}") from exc


def save_embeddings(embeddings: FloatMatrix, path: Path = EMBEDDINGS_PATH) -> None:
    """Persist the embedding matrix as a .npy file."""
    _atomic_write(path, lambda handle: np.save(handle, embeddings, allow_pickle=False))


def save_mapping(author_ids: list[str], model_name: str, dimension: int, path: Path = MAPPING_PATH) -> None:
    """Persist the row-index to author_id mapping with model metadata."""
    payload: dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": dimension,
        "dtype": "float32",
        "normalized": True,
        "pooling": POOLING_STRATEGY,
        "num_authors": len(author_ids),
        "index_to_author_id": author_ids,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(path, lambda handle: handle.write(content))


def verify_outputs(embeddings_path: Path = EMBEDDINGS_PATH, mapping_path: Path = MAPPING_PATH) -> None:
    """Reload both artifacts and check they are aligned and unit-normalized."""
    matrix = np.load(embeddings_path, mmap_mode="r")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if matrix.shape[0] != len(mapping["index_to_author_id"]):
        raise EmbeddingPipelineError("Embedding rows and mapping entries are misaligned.")
    if matrix.shape[0] and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise EmbeddingPipelineError("Stored embeddings are not unit-normalized.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="sentence-transformers model name or path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="encoding batch size")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; auto-detected when omitted")
    return parser.parse_args(argv)


def run(
    model_name: str,
    batch_size: int,
    device: str | None,
    input_path: Path = INPUT_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    mapping_path: Path = MAPPING_PATH,
) -> int:
    """Execute the embedding pipeline and return its process exit code."""
    profiles = load_profiles(input_path)
    if profiles.empty:
        raise EmbeddingPipelineError(f"No author profiles with text found in {input_path}")
    documents, owners = split_into_documents(profiles[PROFILE_TEXT_COL])
    logger.info("Loaded %d authors split into %d article documents.", len(profiles), len(documents))
    model = load_model(model_name, device)
    author_embeddings = pool_by_author(
        encode_unique_documents(model, documents, batch_size), owners, len(profiles),
    )
    save_embeddings(author_embeddings, embeddings_path)
    save_mapping(profiles[AUTHOR_ID_COL].tolist(), model_name, author_embeddings.shape[1], mapping_path)
    verify_outputs(embeddings_path, mapping_path)
    print(f"Authors embedded:  {author_embeddings.shape[0]:,}")
    print(f"Articles encoded:  {len(documents):,}")
    print(f"Embedding shape:   {author_embeddings.shape} ({author_embeddings.dtype})")
    print(f"Embeddings file:   {embeddings_path}")
    print(f"Mapping file:      {mapping_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.batch_size <= 0:
        logger.error("--batch-size must be a positive integer.")
        return 2
    try:
        return run(args.model, args.batch_size, args.device)
    except EmbeddingPipelineError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
