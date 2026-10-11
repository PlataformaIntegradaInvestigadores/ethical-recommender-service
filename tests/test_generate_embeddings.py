"""Checks for profile splitting, pooling, persistence, model registry, device selection and the embedding run.

No test downloads a model: encoders are replaced by deterministic fakes and torch.cuda is mocked.
"""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

emb = importlib.import_module("infrastructure.scripts.02_generate_embeddings")


class FakeModel:
    """Deterministic stand-in for SentenceTransformer that embeds text by length and vowels."""

    tokenizer = SimpleNamespace(sep_token="[SEP]")

    def get_embedding_dimension(self):
        return 3

    def encode(self, sentences, **kwargs):
        vectors = np.array(
            [[len(s), sum(c in "aeiou" for c in s), 1.0] for s in sentences], dtype=np.float32,
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def write_profiles(path, rows):
    pd.DataFrame(rows, columns=["author_id", "author_name", "article_count", "concatenated_profile_text"]).to_csv(
        path, index=False, encoding="utf-8",
    )


class LoadProfilesTests(unittest.TestCase):
    def test_keeps_string_ids_and_drops_empty_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.csv"
            write_profiles(path, [["001", "Ana", 3, "A\nB\n\nC"], ["NA", "Luis", 3, "  "], ["003", "Eva", 4, "D"]])
            frame = emb.load_profiles(path)
            self.assertEqual(frame["author_id"].tolist(), ["001", "003"])
            self.assertEqual(frame.loc[0, "concatenated_profile_text"], "A\nB\n\nC")

    def test_schema_duplicates_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.csv"
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "Input file not found"):
                emb.load_profiles(path)
            path.write_text("author_id,text\n1,x\n", encoding="utf-8")
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "missing columns"):
                emb.load_profiles(path)
            write_profiles(path, [["1", "A", 3, "x"], ["1", "B", 3, "y"]])
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "duplicated"):
                emb.load_profiles(path)


class PoolingTests(unittest.TestCase):
    def test_split_assigns_articles_to_owner_rows(self):
        documents, owners = emb.split_into_documents(pd.Series(["T1\nA1\n\nT2\n\n", "T3"]))
        self.assertEqual(documents, ["T1\nA1", "T2", "T3"])
        self.assertEqual(owners.tolist(), [0, 0, 1])

    def test_pool_averages_and_normalizes_per_author(self):
        vectors = np.array([[1, 0], [0, 1], [3, 4]], dtype=np.float32)
        pooled = emb.pool_by_author(vectors, np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(pooled[0], [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-6)
        np.testing.assert_allclose(pooled[1], [0.6, 0.8], rtol=1e-6)
        self.assertEqual(pooled.dtype, np.float32)

    def test_pool_rejects_misaligned_or_zero_vectors(self):
        with self.assertRaisesRegex(emb.EmbeddingPipelineError, "Pooling mismatch"):
            emb.pool_by_author(np.ones((2, 2), dtype=np.float32), np.array([0, 0]), 2)
        with self.assertRaisesRegex(emb.EmbeddingPipelineError, "zero-norm"):
            emb.pool_by_author(np.array([[1, 0], [-1, 0]], dtype=np.float32), np.array([0, 0]), 1)

    def test_encode_fills_matrix_across_chunks(self):
        documents = [f"doc {i}" for i in range(5)]
        with mock.patch.object(emb, "ENCODE_CHUNK_SIZE", 2):
            matrix = emb.encode_documents(FakeModel(), documents, batch_size=2)
        np.testing.assert_allclose(matrix, FakeModel().encode(documents))

    def test_encode_unique_encodes_repeated_texts_once(self):
        model = FakeModel()
        documents = ["shared", "solo", "shared"]
        with mock.patch.object(model, "encode", wraps=model.encode) as spy:
            matrix = emb.encode_unique_documents(model, documents, batch_size=4)
        self.assertEqual(spy.call_args.args[0], ["shared", "solo"])
        np.testing.assert_allclose(matrix, FakeModel().encode(documents))


class RunTests(unittest.TestCase):
    def test_run_writes_aligned_normalized_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profiles.csv"
            write_profiles(source, [["001", "Ana", 3, "Alpha\nabs\n\nBeta"], ["002", "Eva", 3, ""], ["003", "Io", 3, "Gamma"]])
            embeddings_path = root / "out" / "emb.npy"
            mapping_path = root / "out" / "map.json"
            with mock.patch.object(emb, "load_model", return_value=FakeModel()), \
                    mock.patch.object(emb, "resolve_device", return_value=CPU), \
                    mock.patch("sys.stdout"):
                code = emb.run("minilm", 8, "cpu", source, embeddings_path, mapping_path)
            self.assertEqual(code, 0)
            matrix = np.load(embeddings_path)
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.assertEqual(matrix.shape, (2, 3))
            self.assertEqual(mapping["index_to_author_id"], ["001", "003"])
            self.assertEqual(mapping["model_name"], "sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(mapping["embedding_dim"], 3)
            self.assertEqual(mapping["model_alias"], "minilm")
            self.assertEqual(mapping["device"], "cpu")
            self.assertIn("elapsed_seconds", mapping)
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
            self.assertFalse(any(path.suffix == ".tmp" for path in (root / "out").iterdir()))

    def test_run_formats_scientific_inputs_and_limits_authors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profiles.csv"
            write_profiles(source, [["001", "Ana", 3, "T1\nA1\n\nT2"], ["002", "Eva", 3, "T3\nA3"]])
            model = FakeModel()
            with mock.patch.object(emb, "load_model", return_value=model) as loader, \
                    mock.patch.object(emb, "resolve_device", return_value=CPU), \
                    mock.patch.object(model, "encode", wraps=model.encode) as spy, mock.patch("sys.stdout"):
                emb.run("scincl", None, "auto", source, root / "e.npy", root / "m.json", limit_authors=1)
            self.assertEqual(spy.call_args.args[0], ["T1[SEP]A1", "T2[SEP]"])
            self.assertEqual(spy.call_args.kwargs["batch_size"], emb.MODEL_REGISTRY["scincl"].cpu_batch_size)
            self.assertEqual(loader.call_args.args, (emb.MODEL_REGISTRY["scincl"], "cpu", False))
            mapping = json.loads((root / "m.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["index_to_author_id"], ["001"])
            self.assertEqual((mapping["token_pooling"], mapping["max_seq_length"]), ("cls", 512))

    def test_run_enables_fp16_and_gpu_batch_size_on_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profiles.csv"
            write_profiles(source, [["001", "Ana", 3, "T1\nA1"]])
            with mock.patch.object(emb, "load_model", return_value=FakeModel()) as loader, \
                    mock.patch.object(emb, "resolve_device", return_value=GPU), mock.patch("sys.stdout"):
                emb.run("specter2", None, "auto", source, root / "e.npy", root / "m.json")
            self.assertEqual(loader.call_args.args, (emb.MODEL_REGISTRY["specter2"], "cuda", True))
            mapping = json.loads((root / "m.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["batch_size"], emb.MODEL_REGISTRY["specter2"].cuda_batch_size)
            self.assertTrue(mapping["fp16"])
            self.assertEqual(mapping["gpu_name"], "NVIDIA GeForce RTX 3060")

    def test_main_reports_failures_with_exit_codes(self):
        with mock.patch("sys.stderr"):
            self.assertEqual(emb.main(["--batch-size", "0"]), 2)
            self.assertEqual(emb.main(["--model", "bert"]), 2)
        with mock.patch.object(emb, "run", side_effect=emb.EmbeddingPipelineError("boom")):
            self.assertEqual(emb.main([]), 1)

    def test_model_load_network_failure_becomes_pipeline_error(self):
        fake_module = mock.MagicMock()
        fake_module.SentenceTransformer.side_effect = ConnectionError("403 Forbidden")
        fake_modules = {"sentence_transformers": fake_module,
                        "sentence_transformers.sentence_transformer.modules": mock.MagicMock()}
        with mock.patch.dict("sys.modules", fake_modules):
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "Could not load model 'malteos/scincl': 403"):
                emb.load_model(emb.MODEL_REGISTRY["scincl"], "cpu")

    def test_verify_detects_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emb.save_embeddings(np.eye(2, dtype=np.float32), root / "e.npy")
            emb.save_mapping(["only-one"], "m", 2, root / "m.json")
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "misaligned"):
                emb.verify_outputs(root / "e.npy", root / "m.json")


CPU = emb.DeviceInfo(device="cpu", torch_version="2.x")
GPU = emb.DeviceInfo(device="cuda", torch_version="2.x+cu126", cuda_version="12.6",
                     gpu_name="NVIDIA GeForce RTX 3060", vram_gb=6.0)


def fake_torch(cuda_available):
    """Minimal torch stand-in so device selection can be tested without a GPU."""
    torch = mock.MagicMock()
    torch.__version__ = "2.14.1+cu126" if cuda_available else "2.14.1+cpu"
    torch.version.cuda = "12.6" if cuda_available else None
    torch.cuda.is_available.return_value = cuda_available
    torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
    torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=6 * 1024**3)
    torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    return torch


class CliAndRegistryTests(unittest.TestCase):
    def test_defaults(self):
        args = emb.parse_args([])
        self.assertEqual((args.model, args.device, args.batch_size, args.fp16), ("minilm", "auto", None, None))
        self.assertFalse(args.legacy_names)
        self.assertIsNone(args.limit_authors)

    def test_parses_model_device_and_flags(self):
        args = emb.parse_args(["--model", "specter2", "--device", "cuda", "--batch-size", "16", "--no-fp16",
                               "--limit-authors", "50", "--output-dir", "out"])
        self.assertEqual((args.model, args.device, args.batch_size, args.fp16), ("specter2", "cuda", 16, False))
        self.assertEqual((args.limit_authors, args.output_dir), (50, Path("out")))
        self.assertTrue(emb.parse_args(["--fp16"]).fp16)

    def test_rejects_unknown_model_and_bad_values(self):
        with mock.patch("sys.stderr"):
            for argv in (["--model", "bert"], ["--device", "tpu"], ["--batch-size", "-1"], ["--fp16", "--no-fp16"]):
                with self.assertRaises(SystemExit, msg=argv):
                    emb.parse_args(argv)

    def test_registry_matches_model_cards(self):
        self.assertEqual(
            {alias: spec.hf_id for alias, spec in emb.MODEL_REGISTRY.items()},
            {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "specter2": "allenai/specter2_base",
             "scincl": "malteos/scincl"},
        )
        for alias in ("specter2", "scincl"):
            spec = emb.MODEL_REGISTRY[alias]
            self.assertEqual((spec.token_pooling, spec.max_seq_length, spec.input_format, spec.explicit_stack),
                             ("cls", 512, "title_sep_abstract", True))
        minilm = emb.MODEL_REGISTRY["minilm"]
        self.assertEqual((minilm.token_pooling, minilm.max_seq_length, minilm.input_format), ("mean", 256, "raw"))
        self.assertEqual(minilm.cuda_batch_size, 128)

    def test_output_paths_are_model_specific(self):
        root = Path("out")
        self.assertEqual(emb.output_paths("specter2", root),
                         (root / "author_embeddings_specter2.npy", root / "author_index_mapping_specter2.json"))
        self.assertEqual(emb.output_paths("scincl", root, limit_authors=20)[0],
                         root / "author_embeddings_scincl_sample20.npy")
        self.assertEqual(emb.output_paths("minilm", root, legacy_names=True),
                         (root / "author_embeddings.npy", root / "author_index_mapping.json"))
        self.assertEqual(len({emb.output_paths(alias)[0] for alias in emb.MODEL_REGISTRY}), 3)

    def test_main_passes_dynamic_paths_to_run(self):
        with mock.patch.object(emb, "run", return_value=0) as run:
            self.assertEqual(emb.main(["--model", "scincl", "--output-dir", "o"]), 0)
        self.assertEqual(run.call_args.args[:5], ("scincl", None, "auto", emb.INPUT_PATH,
                                                  Path("o") / "author_embeddings_scincl.npy"))

    def test_history_line_is_a_markdown_row(self):
        line = emb.history_line(emb.MODEL_REGISTRY["scincl"], GPU, 32, True, (7621, 768), 300.0)
        self.assertTrue(line.startswith("| ") and line.endswith(" |"))
        for fragment in ("malteos/scincl", "RTX 3060", "7621×768", "5.0 min", "CUDA 12.6"):
            self.assertIn(fragment, line)


class InputFormattingTests(unittest.TestCase):
    def test_title_sep_abstract(self):
        self.assertEqual(emb.format_document("Title\nAbstract text\nmore", "title_sep_abstract", "[SEP]"),
                         "Title[SEP]Abstract text\nmore")

    def test_title_only_keeps_trailing_sep_like_model_card(self):
        self.assertEqual(emb.format_document("Only title", "title_sep_abstract", "[SEP]"), "Only title[SEP]")

    def test_raw_keeps_baseline_input_unchanged(self):
        self.assertEqual(emb.format_document("T\nA", "raw", None), "T\nA")

    def test_missing_sep_token_or_unknown_format_fail(self):
        with self.assertRaisesRegex(emb.EmbeddingPipelineError, "sep_token"):
            emb.format_document("T\nA", "title_sep_abstract", None)
        with self.assertRaisesRegex(emb.EmbeddingPipelineError, "Unknown input format"):
            emb.format_document("T\nA", "other", "[SEP]")

    def test_sep_token_read_from_tokenizer(self):
        self.assertEqual(emb.model_sep_token(FakeModel()), "[SEP]")
        self.assertIsNone(emb.model_sep_token(object()))


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_uses_cuda_and_logs_gpu(self):
        with mock.patch.dict("sys.modules", {"torch": fake_torch(True)}):
            with self.assertLogs(emb.logger, level="INFO") as logs:
                info = emb.resolve_device("auto")
        self.assertEqual((info.device, info.gpu_name, info.vram_gb, info.cuda_version),
                         ("cuda", "NVIDIA GeForce RTX 3060", 6.0, "12.6"))
        self.assertIn("RTX 3060", "\n".join(logs.output))

    def test_auto_warns_and_falls_back_to_cpu(self):
        with mock.patch.dict("sys.modules", {"torch": fake_torch(False)}):
            with self.assertLogs(emb.logger, level="WARNING") as logs:
                info = emb.resolve_device("auto")
        self.assertEqual(info.device, "cpu")
        self.assertIn("CUDA was not detected", logs.output[0])
        self.assertIn("download.pytorch.org/whl/cu126", logs.output[0])

    def test_explicit_cuda_without_gpu_fails(self):
        with mock.patch.dict("sys.modules", {"torch": fake_torch(False)}):
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "CUDA is not available"):
                emb.resolve_device("cuda")

    def test_explicit_cpu_ignores_gpu_silently(self):
        with mock.patch.dict("sys.modules", {"torch": fake_torch(True)}):
            with self.assertNoLogs(emb.logger, level="WARNING"):
                self.assertEqual(emb.resolve_device("cpu").device, "cpu")

    def test_load_model_moves_to_device_and_casts_fp16(self):
        transformer = mock.MagicMock()
        transformer.get_embedding_dimension.return_value = 768
        st_cls, transformer_cls, pooling_cls = mock.MagicMock(), mock.MagicMock(return_value=transformer), mock.MagicMock()
        model = st_cls.return_value
        model.get_embedding_dimension.return_value = 768
        with mock.patch.object(emb, "_sentence_transformer_modules", return_value=(st_cls, transformer_cls, pooling_cls)):
            emb.load_model(emb.MODEL_REGISTRY["scincl"], "cuda", fp16=True)
        transformer_cls.assert_called_once_with("malteos/scincl", max_seq_length=512)
        pooling_cls.assert_called_once_with(768, pooling_mode="cls")
        self.assertEqual(st_cls.call_args.kwargs["device"], "cuda")
        model.to.assert_called_once_with("cuda")
        model.half.assert_called_once_with()
        self.assertEqual(model.max_seq_length, 512)

    def test_load_minilm_uses_hub_config_and_no_fp16_on_cpu(self):
        st_cls = mock.MagicMock()
        st_cls.return_value.get_embedding_dimension.return_value = 384
        with mock.patch.object(emb, "_sentence_transformer_modules", return_value=(st_cls, mock.MagicMock(), mock.MagicMock())):
            emb.load_model(emb.MODEL_REGISTRY["minilm"], "cpu", fp16=True)
        st_cls.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        st_cls.return_value.half.assert_not_called()


class OomRetryTests(unittest.TestCase):
    def test_halves_batch_size_until_it_fits(self):
        torch = fake_torch(True)
        calls = []

        def encode(batch, size):
            calls.append(size)
            if size > 8:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.5 GiB")
            return np.ones((len(batch), 2), dtype=np.float32)

        with mock.patch.dict("sys.modules", {"torch": torch}):
            with self.assertLogs(emb.logger, level="WARNING") as logs:
                vectors, size = emb.encode_with_oom_retry(encode, ["a", "b"], 32)
        self.assertEqual((calls, size, vectors.shape), ([32, 16, 8], 8, (2, 2)))
        self.assertEqual(len(logs.output), 2)
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)

    def test_gives_up_at_batch_size_one(self):
        def encode(batch, size):
            raise RuntimeError("CUDA error: out of memory")

        with mock.patch.dict("sys.modules", {"torch": fake_torch(True)}), self.assertLogs(emb.logger, "WARNING"):
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "batch size 1"):
                emb.encode_with_oom_retry(encode, ["a"], 4)

    def test_other_errors_propagate(self):
        def encode(batch, size):
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            emb.encode_with_oom_retry(encode, ["a"], 4)

    def test_encode_documents_keeps_reduced_batch_size_across_chunks(self):
        model = FakeModel()
        sizes = []
        original = model.encode

        def flaky(sentences, batch_size, **kwargs):
            sizes.append(batch_size)
            if batch_size > 2:
                raise RuntimeError("CUDA out of memory")
            return original(sentences)

        documents = [f"doc {i}" for i in range(4)]
        with mock.patch.object(model, "encode", side_effect=flaky), mock.patch.object(emb, "ENCODE_CHUNK_SIZE", 2), \
                mock.patch.dict("sys.modules", {"torch": fake_torch(True)}), self.assertLogs(emb.logger, "WARNING"):
            matrix = emb.encode_documents(model, documents, batch_size=4)
        self.assertEqual(sizes, [4, 2, 2])
        np.testing.assert_allclose(matrix, FakeModel().encode(documents))


if __name__ == "__main__":
    unittest.main()
