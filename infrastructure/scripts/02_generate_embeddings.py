"""Generate L2-normalized author embeddings by mean-pooling per-article document embeddings.

Supports a comparative study of three encoders (see docs/execution_history.md):

* ``minilm``   -> sentence-transformers/all-MiniLM-L6-v2 (general-purpose baseline, mean pooling, 256 tokens)
* ``specter2`` -> allenai/specter2_base (scientific, citation-informed; CLS pooling, 512 tokens)
* ``scincl``   -> malteos/scincl (scientific, citation-graph contrastive; CLS pooling, 512 tokens)
* ``minilm-l12`` -> sentence-transformers/all-MiniLM-L12-v2 (general-purpose, 12 layers; mean pooling, 256 tokens)
* ``mpnet``    -> sentence-transformers/all-mpnet-base-v2 (general-purpose; mean pooling, 384 tokens)

The general-purpose models keep the ETL input 'title\nabstract' and the pooling of their hub config.
all-MiniLM-L12-v2 ships ``max_seq_length=128`` in sentence_bert_config.json while its model card states
256 word pieces; 256 is used here so it truncates like the L6 baseline.

SPECTER2 and SciNCL follow their model cards: input ``title + tokenizer.sep_token + abstract``,
truncation at 512 tokens and the first ([CLS]) token as document embedding. The official SPECTER2
usage additionally loads the proximity adapter ``allenai/specter2`` through the ``adapters`` library;
this script uses the base encoder only (the adapter is documented as future work in the README).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
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
LEGACY_EMBEDDINGS_NAME: Final[str] = "author_embeddings.npy"
LEGACY_MAPPING_NAME: Final[str] = "author_index_mapping.json"
ENCODE_CHUNK_SIZE: Final[int] = 2048
ARTICLE_SEPARATOR: Final[str] = "\n\n"
TITLE_SEPARATOR: Final[str] = "\n"
POOLING_STRATEGY: Final[str] = "mean_of_l2_normalized_article_embeddings"
AUTHOR_ID_COL: Final[str] = "author_id"
PROFILE_TEXT_COL: Final[str] = "concatenated_profile_text"
CUDA_INSTALL_HINT: Final[str] = "pip install torch --index-url https://download.pytorch.org/whl/cu126 (see README)"

FloatMatrix = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]
logger = logging.getLogger("etl.generate_embeddings")


class EmbeddingPipelineError(RuntimeError):
    """The embedding pipeline cannot complete."""


@dataclass(frozen=True)
class ModelSpec:
    """Per-model configuration for the comparative study."""

    alias: str
    hf_id: str
    token_pooling: str  # "mean" or "cls"
    max_seq_length: int
    input_format: str  # "raw" keeps 'title\nabstract'; "title_sep_abstract" uses tokenizer.sep_token
    cuda_batch_size: int
    cpu_batch_size: int
    explicit_stack: bool  # build Transformer + Pooling explicitly instead of trusting the hub config


MODEL_REGISTRY: Final[dict[str, ModelSpec]] = {
    "minilm": ModelSpec(
        alias="minilm",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        token_pooling="mean",
        max_seq_length=256,
        input_format="raw",
        cuda_batch_size=128,
        cpu_batch_size=64,
        explicit_stack=False,
    ),
    "specter2": ModelSpec(
        alias="specter2",
        hf_id="allenai/specter2_base",
        token_pooling="cls",
        max_seq_length=512,
        input_format="title_sep_abstract",
        cuda_batch_size=32,
        cpu_batch_size=16,
        explicit_stack=True,
    ),
    "scincl": ModelSpec(
        alias="scincl",
        hf_id="malteos/scincl",
        token_pooling="cls",
        max_seq_length=512,
        input_format="title_sep_abstract",
        cuda_batch_size=32,
        cpu_batch_size=16,
        explicit_stack=True,
    ),
    "minilm-l12": ModelSpec(
        alias="minilm-l12",
        hf_id="sentence-transformers/all-MiniLM-L12-v2",
        token_pooling="mean",
        max_seq_length=256,
        input_format="raw",
        cuda_batch_size=128,
        cpu_batch_size=64,
        explicit_stack=False,
    ),
    "mpnet": ModelSpec(
        alias="mpnet",
        hf_id="sentence-transformers/all-mpnet-base-v2",
        token_pooling="mean",
        max_seq_length=384,
        input_format="raw",
        cuda_batch_size=64,
        cpu_batch_size=32,
        explicit_stack=False,
    ),
}
DEFAULT_MODEL_ALIAS: Final[str] = "minilm"


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device and the environment details worth logging."""

    device: str
    torch_version: str
    cuda_version: str | None = None
    gpu_name: str | None = None
    vram_gb: float | None = None


def output_paths(
    alias: str,
    output_dir: Path = PROCESSED_DIR,
    limit_authors: int | None = None,
    legacy_names: bool = False,
) -> tuple[Path, Path]:
    """Return (embeddings, mapping) paths, named after the model so runs never overwrite each other."""
    if legacy_names:
        return output_dir / LEGACY_EMBEDDINGS_NAME, output_dir / LEGACY_MAPPING_NAME
    suffix = alias if limit_authors is None else f"{alias}_sample{limit_authors}"
    return output_dir / f"author_embeddings_{suffix}.npy", output_dir / f"author_index_mapping_{suffix}.json"


def load_profiles(path: Path) -> pd.DataFrame:
    """Load author profiles, validate the schema, and drop authors without usable text."""
    if not path.is_file():
        raise EmbeddingPipelineError(f"Input file not found: {path}. Run 01_filter_and_merge_authors.py first.")
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


def format_document(document: str, input_format: str, sep_token: str | None) -> str:
    """Turn an ETL article chunk ('title\\nabstract') into the input the model was trained on."""
    if input_format == "raw":
        return document
    if input_format != "title_sep_abstract":
        raise EmbeddingPipelineError(f"Unknown input format '{input_format}'")
    if not sep_token:
        raise EmbeddingPipelineError("Model tokenizer has no sep_token for title/abstract concatenation.")
    title, _, abstract = document.partition(TITLE_SEPARATOR)
    return title.strip() + sep_token + abstract.strip()


def resolve_device(requested: str) -> DeviceInfo:
    """Pick the compute device; warn and fall back to CPU when CUDA is missing under 'auto'."""
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingPipelineError(f"PyTorch is not installed. Install it with: {CUDA_INSTALL_HINT}") from exc
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise EmbeddingPipelineError(
            f"--device cuda was requested but CUDA is not available (torch {torch.__version__}, "
            f"built with CUDA {torch.version.cuda}). Install a CUDA build: {CUDA_INSTALL_HINT}"
        )
    if requested == "cpu" or not cuda_available:
        if requested == "auto":
            logger.warning(
                "CUDA was not detected (torch %s, built with CUDA %s); falling back to CPU, which is much "
                "slower. For an NVIDIA GPU install a CUDA build: %s",
                torch.__version__,
                torch.version.cuda,
                CUDA_INSTALL_HINT,
            )
        return DeviceInfo(device="cpu", torch_version=str(torch.__version__))
    properties = torch.cuda.get_device_properties(0)
    info = DeviceInfo(
        device="cuda",
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        gpu_name=str(torch.cuda.get_device_name(0)),
        vram_gb=round(properties.total_memory / 1024**3, 2),
    )
    logger.info(
        "Using CUDA GPU %s (%.2f GB VRAM), torch %s, CUDA %s",
        info.gpu_name,
        info.vram_gb,
        info.torch_version,
        info.cuda_version,
    )
    return info


def _sentence_transformer_modules() -> tuple[Any, Any, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingPipelineError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    except ImportError:  # sentence-transformers < 6
        from sentence_transformers.models import Pooling, Transformer
    return SentenceTransformer, Transformer, Pooling


def load_model(spec: ModelSpec, device: str, fp16: bool = False) -> SentenceTransformer:
    """Load the encoder for ``spec``, move it to ``device`` and optionally cast it to fp16."""
    SentenceTransformer, Transformer, Pooling = _sentence_transformer_modules()
    try:
        if spec.explicit_stack:
            transformer = Transformer(spec.hf_id, max_seq_length=spec.max_seq_length)
            pooling = Pooling(embedding_dimension(transformer), pooling_mode=spec.token_pooling)
            model = SentenceTransformer(modules=[transformer, pooling], device=device)
        else:
            model = SentenceTransformer(spec.hf_id, device=device)
        model.max_seq_length = spec.max_seq_length
        model.to(device)
        model.eval()
        if fp16 and device == "cuda":
            model.half()
    except Exception as exc:
        raise EmbeddingPipelineError(f"Could not load model '{spec.hf_id}': {exc}") from exc
    logger.info(
        "Loaded model %s on %s (dim=%s, max_seq_length=%s, pooling=%s, fp16=%s)",
        spec.hf_id,
        model.device,
        embedding_dimension(model),
        model.max_seq_length,
        spec.token_pooling,
        fp16 and device == "cuda",
    )
    return model


def embedding_dimension(module: Any) -> int | None:
    """Embedding size across sentence-transformers versions (get_embedding_dimension in v6+)."""
    getter = getattr(module, "get_embedding_dimension", None)
    if getter is None:
        getter = getattr(module, "get_sentence_embedding_dimension", None) or module.get_word_embedding_dimension
    return getter()


def model_sep_token(model: Any) -> str | None:
    """Return the tokenizer separator token ([SEP] for BERT/SciBERT), if any."""
    tokenizer = getattr(model, "tokenizer", None)
    return getattr(tokenizer, "sep_token", None)


def is_out_of_memory(exc: BaseException) -> bool:
    """Detect CUDA out-of-memory errors without requiring a GPU to test it."""
    oom_type = None
    try:
        import torch

        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    except ImportError:
        pass
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def encode_with_oom_retry(
    encode: Callable[[list[str], int], FloatMatrix],
    batch: list[str],
    batch_size: int,
) -> tuple[FloatMatrix, int]:
    """Encode ``batch``; on CUDA OOM halve the batch size and retry. Returns vectors and the size that worked."""
    while True:
        try:
            return encode(batch, batch_size), batch_size
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            _empty_cuda_cache()
            if batch_size <= 1:
                raise EmbeddingPipelineError("CUDA out of memory even with batch size 1.") from exc
            batch_size = max(1, batch_size // 2)
            logger.warning("CUDA out of memory; retrying with batch size %d.", batch_size)


def encode_documents(model: SentenceTransformer, documents: list[str], batch_size: int) -> FloatMatrix:
    """Encode documents into L2-normalized float32 vectors with a progress bar."""
    dimension = embedding_dimension(model)
    if not dimension:
        raise EmbeddingPipelineError("Model does not report a fixed embedding dimension.")

    def encode(batch: list[str], size: int) -> FloatMatrix:
        return model.encode(
            batch,
            batch_size=size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    embeddings = np.empty((len(documents), dimension), dtype=np.float32)
    with tqdm(total=len(documents), desc="Encoding articles", unit="doc") as progress:
        for start in range(0, len(documents), ENCODE_CHUNK_SIZE):
            batch = documents[start : start + ENCODE_CHUNK_SIZE]
            vectors, batch_size = encode_with_oom_retry(encode, batch, batch_size)
            embeddings[start : start + len(batch)] = np.asarray(vectors, dtype=np.float32)
            progress.update(len(batch))
    return embeddings


def encode_unique_documents(model: SentenceTransformer, documents: list[str], batch_size: int) -> FloatMatrix:
    """Encode each distinct text once, since co-authored articles repeat across profiles."""
    positions: dict[str, int] = {}
    inverse = np.fromiter(
        (positions.setdefault(document, len(positions)) for document in documents),
        dtype=np.int64,
        count=len(documents),
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
    sums = np.add.reduceat(document_embeddings.astype(np.float32), boundaries, axis=0)
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


def save_embeddings(embeddings: FloatMatrix, path: Path) -> None:
    """Persist the embedding matrix as a .npy file."""
    _atomic_write(path, lambda handle: np.save(handle, embeddings, allow_pickle=False))


def save_mapping(
    author_ids: list[str],
    model_name: str,
    dimension: int,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the row-index to author_id mapping with model and run metadata."""
    payload: dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": dimension,
        "dtype": "float32",
        "normalized": True,
        "pooling": POOLING_STRATEGY,
        **(metadata or {}),
        "num_authors": len(author_ids),
        "index_to_author_id": author_ids,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(path, lambda handle: handle.write(content))


def verify_outputs(embeddings_path: Path, mapping_path: Path) -> None:
    """Reload both artifacts and check they are aligned and unit-normalized."""
    matrix = np.load(embeddings_path, mmap_mode="r")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if matrix.shape[0] != len(mapping["index_to_author_id"]):
        raise EmbeddingPipelineError("Embedding rows and mapping entries are misaligned.")
    if matrix.shape[0] and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise EmbeddingPipelineError("Stored embeddings are not unit-normalized.")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got '{value}'") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default=DEFAULT_MODEL_ALIAS,
        help="encoder alias (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when available and warns otherwise (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="encoding batch size; default depends on model and device (halved on CUDA OOM)",
    )
    fp16 = parser.add_mutually_exclusive_group()
    fp16.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="half precision on CUDA (default on CUDA, ignored on CPU)",
    )
    fp16.add_argument("--no-fp16", dest="fp16", action="store_false", help="keep float32 weights on CUDA")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="profiles CSV (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="output directory")
    parser.add_argument(
        "--limit-authors",
        type=_positive_int,
        default=None,
        help="embed only the first N authors (smoke tests); adds _sampleN to file names",
    )
    parser.add_argument(
        "--legacy-names",
        action="store_true",
        help="write author_embeddings.npy / author_index_mapping.json (pre-study names)",
    )
    return parser.parse_args(argv)


def history_line(
    spec: ModelSpec, device: DeviceInfo, batch_size: int, fp16: bool, shape: tuple[int, ...], elapsed: float
) -> str:
    """Markdown table row ready to paste into docs/execution_history.md."""
    hardware = f"GPU {device.gpu_name} ({device.vram_gb} GB)" if device.device == "cuda" else "CPU"
    cuda = device.cuda_version or "-"
    return (
        f"| {datetime.now():%Y-%m-%d} | 02_generate_embeddings.py --model {spec.alias} | {spec.hf_id} | "
        f"{spec.token_pooling} | {spec.max_seq_length} | {hardware} | torch {device.torch_version} / CUDA {cuda} | "
        f"{batch_size} | {'sí' if fp16 else 'no'} | {shape[0]}×{shape[1]} | {elapsed / 60:.1f} min ({elapsed:.0f} s) |"
    )


def run(
    model_alias: str = DEFAULT_MODEL_ALIAS,
    batch_size: int | None = None,
    device: str = "auto",
    input_path: Path = INPUT_PATH,
    embeddings_path: Path | None = None,
    mapping_path: Path | None = None,
    fp16: bool | None = None,
    limit_authors: int | None = None,
) -> int:
    """Execute the embedding pipeline and return its process exit code."""
    if model_alias not in MODEL_REGISTRY:
        raise EmbeddingPipelineError(f"Unknown model '{model_alias}'. Choose from {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_alias]
    default_embeddings, default_mapping = output_paths(spec.alias, limit_authors=limit_authors)
    embeddings_path = embeddings_path or default_embeddings
    mapping_path = mapping_path or default_mapping
    started = time.perf_counter()
    profiles = load_profiles(input_path)
    if limit_authors is not None:
        profiles = profiles.head(limit_authors).reset_index(drop=True)
    if profiles.empty:
        raise EmbeddingPipelineError(f"No author profiles with text found in {input_path}")
    documents, owners = split_into_documents(profiles[PROFILE_TEXT_COL])
    logger.info("Loaded %d authors split into %d article documents.", len(profiles), len(documents))
    device_info = resolve_device(device)
    use_fp16 = device_info.device == "cuda" and (fp16 is None or fp16)
    if batch_size is None:
        batch_size = spec.cuda_batch_size if device_info.device == "cuda" else spec.cpu_batch_size
    model = load_model(spec, device_info.device, use_fp16)
    sep_token = model_sep_token(model) if spec.input_format != "raw" else None
    model_inputs = [format_document(document, spec.input_format, sep_token) for document in documents]
    author_embeddings = pool_by_author(
        encode_unique_documents(model, model_inputs, batch_size),
        owners,
        len(profiles),
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "model_alias": spec.alias,
        "token_pooling": spec.token_pooling,
        "max_seq_length": spec.max_seq_length,
        "input_format": spec.input_format,
        "device": device_info.device,
        "gpu_name": device_info.gpu_name,
        "torch_version": device_info.torch_version,
        "cuda_version": device_info.cuda_version,
        "fp16": use_fp16,
        "batch_size": batch_size,
        "limit_authors": limit_authors,
        "elapsed_seconds": round(elapsed, 2),
    }
    save_embeddings(author_embeddings, embeddings_path)
    save_mapping(profiles[AUTHOR_ID_COL].tolist(), spec.hf_id, author_embeddings.shape[1], mapping_path, metadata)
    verify_outputs(embeddings_path, mapping_path)
    print(
        f"Model:             {spec.alias} ({spec.hf_id}, pooling={spec.token_pooling}, max_len={spec.max_seq_length})"
    )
    print(f"Device:            {device_info.device}" + (f" ({device_info.gpu_name})" if device_info.gpu_name else ""))
    print(f"Authors embedded:  {author_embeddings.shape[0]:,}")
    print(f"Articles encoded:  {len(documents):,}")
    print(f"Embedding shape:   {author_embeddings.shape} ({author_embeddings.dtype})")
    print(f"Embeddings file:   {embeddings_path}")
    print(f"Mapping file:      {mapping_path}")
    print(f"Elapsed:           {elapsed / 60:.1f} min ({elapsed:.1f} s)")
    print("Execution log row (docs/execution_history.md):")
    print(history_line(spec, device_info, batch_size, use_fp16, author_embeddings.shape, elapsed))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    embeddings_path, mapping_path = output_paths(args.model, args.output_dir, args.limit_authors, args.legacy_names)
    try:
        return run(
            args.model,
            args.batch_size,
            args.device,
            args.input,
            embeddings_path,
            mapping_path,
            fp16=args.fp16,
            limit_authors=args.limit_authors,
        )
    except EmbeddingPipelineError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
