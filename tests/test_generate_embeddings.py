"""Checks for profile splitting, pooling, persistence, and the end-to-end embedding run."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

emb = importlib.import_module("infrastructure.scripts.02_generate_embeddings")


class FakeModel:
    """Deterministic stand-in for SentenceTransformer that embeds text by length and vowels."""

    def get_sentence_embedding_dimension(self):
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
            with mock.patch.object(emb, "load_model", return_value=FakeModel()):
                code = emb.run("fake-model", 8, None, source, embeddings_path, mapping_path)
            self.assertEqual(code, 0)
            matrix = np.load(embeddings_path)
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.assertEqual(matrix.shape, (2, 3))
            self.assertEqual(mapping["index_to_author_id"], ["001", "003"])
            self.assertEqual(mapping["model_name"], "fake-model")
            self.assertEqual(mapping["embedding_dim"], 3)
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
            self.assertFalse(any(path.suffix == ".tmp" for path in (root / "out").iterdir()))

    def test_main_reports_failures_with_exit_codes(self):
        self.assertEqual(emb.main(["--batch-size", "0"]), 2)
        with mock.patch.object(emb, "run", side_effect=emb.EmbeddingPipelineError("boom")):
            self.assertEqual(emb.main([]), 1)

    def test_model_load_network_failure_becomes_pipeline_error(self):
        fake_module = mock.MagicMock()
        fake_module.SentenceTransformer.side_effect = ConnectionError("403 Forbidden")
        with mock.patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "Could not load model 'm': 403"):
                emb.load_model("m", None)

    def test_verify_detects_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emb.save_embeddings(np.eye(2, dtype=np.float32), root / "e.npy")
            emb.save_mapping(["only-one"], "m", 2, root / "m.json")
            with self.assertRaisesRegex(emb.EmbeddingPipelineError, "misaligned"):
                emb.verify_outputs(root / "e.npy", root / "m.json")


if __name__ == "__main__":
    unittest.main()
