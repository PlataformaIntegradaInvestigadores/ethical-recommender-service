"""CLI, embedding path resolution, validation and output naming of 04_run_rerank_experiment.py (no data, no models)."""

import importlib
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

exp = importlib.import_module("infrastructure.scripts.04_run_rerank_experiment")


def write_artifacts(directory, stem, rows=2, dim=3, ids=None, **mapping_fields):
    embeddings = Path(directory) / f"author_embeddings{stem}.npy"
    mapping = Path(directory) / f"author_index_mapping{stem}.json"
    np.save(embeddings, np.eye(rows, dim, dtype=np.float32))
    payload = {
        "model_name": exp.MODEL_IDS["minilm"],
        "embedding_dim": dim,
        "index_to_author_id": ids if ids is not None else [f"a{i}" for i in range(rows)],
        **mapping_fields,
    }
    mapping.write_text(json.dumps(payload), encoding="utf-8")
    return embeddings, mapping


class ArgParsingTests(unittest.TestCase):
    def test_defaults(self):
        args = exp.parse_args([])
        self.assertEqual(
            (args.model, args.embeddings, args.mapping, args.tag, args.device), ("minilm", None, None, "", "auto")
        )
        self.assertFalse(args.leakage_free)

    def test_model_and_overrides(self):
        args = exp.parse_args(
            [
                "--model",
                "scincl",
                "--embeddings",
                "e.npy",
                "--mapping",
                "m.json",
                "--leakage-free",
                "--tag",
                "_lf",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(
            (args.model, args.embeddings, args.mapping, args.tag), ("scincl", Path("e.npy"), Path("m.json"), "_lf")
        )
        self.assertTrue(args.leakage_free)

    def test_rejects_unknown_model(self):
        with unittest.mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            exp.parse_args(["--model", "bert"])


class GeneralistAliasTests(unittest.TestCase):
    def test_aliases_match_02_registry_ids(self):
        self.assertEqual(exp.MODEL_IDS["minilm-l12"], "sentence-transformers/all-MiniLM-L12-v2")
        self.assertEqual(exp.MODEL_IDS["mpnet"], "sentence-transformers/all-mpnet-base-v2")
        for alias in ("minilm-l12", "mpnet"):
            self.assertEqual(exp.parse_args(["--model", alias, "--leakage-free"]).model, alias)

    def test_hyphenated_alias_paths_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifacts(directory, "_minilm-l12")
            self.assertEqual(exp.resolve_embedding_paths("minilm-l12", processed_dir=Path(directory)), paths)
            self.assertEqual(paths[0].name, "author_embeddings_minilm-l12.npy")
        self.assertEqual(
            exp.result_paths(Path("x"), "mpnet", "_leakage_free")["summary"].name,
            "rerank_summary_mpnet_leakage_free.csv",
        )

    def test_generalist_models_never_use_legacy_minilm_files(self):
        with tempfile.TemporaryDirectory() as directory:
            write_artifacts(directory, "")
            with self.assertRaisesRegex(exp.ExperimentError, "--model minilm-l12"):
                exp.resolve_embedding_paths("minilm-l12", processed_dir=Path(directory))

    def test_encoder_spec_from_02_mapping_for_mpnet(self):
        mapping = {
            "model_name": exp.MODEL_IDS["mpnet"],
            "model_alias": "mpnet",
            "token_pooling": "mean",
            "max_seq_length": 384,
            "input_format": "raw",
        }
        spec = exp.encoder_spec("mpnet", mapping, 768)
        self.assertEqual((spec.token_pooling, spec.max_seq_length, spec.input_format), ("mean", 384, "raw"))
        with self.assertRaisesRegex(exp.ExperimentError, "no model metadata"):
            exp.encoder_spec("minilm-l12", {"model_name": exp.MODEL_IDS["minilm-l12"]}, 384)


class PathResolutionTests(unittest.TestCase):
    def test_suffixed_files_are_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            suffixed = write_artifacts(directory, "_minilm")
            write_artifacts(directory, "")
            self.assertEqual(exp.resolve_embedding_paths("minilm", processed_dir=Path(directory)), suffixed)

    def test_minilm_falls_back_to_legacy_names_with_info_log(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = write_artifacts(directory, "")
            with self.assertLogs(exp.logger, level="INFO") as logs:
                self.assertEqual(exp.resolve_embedding_paths("minilm", processed_dir=Path(directory)), legacy)
            self.assertIn("legacy", "\n".join(logs.output))

    def test_other_models_never_use_legacy_names(self):
        with tempfile.TemporaryDirectory() as directory:
            write_artifacts(directory, "")
            with self.assertRaisesRegex(exp.ExperimentError, r"02_generate_embeddings.py --model scincl"):
                exp.resolve_embedding_paths("scincl", processed_dir=Path(directory))

    def test_specter2_suffixed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifacts(directory, "_specter2")
            self.assertEqual(exp.resolve_embedding_paths("specter2", processed_dir=Path(directory)), paths)

    def test_explicit_overrides_win(self):
        with tempfile.TemporaryDirectory() as directory:
            write_artifacts(directory, "_minilm")
            custom = write_artifacts(directory, "_custom")
            self.assertEqual(exp.resolve_embedding_paths("minilm", *custom, processed_dir=Path(directory)), custom)

    def test_missing_override_is_an_error_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            write_artifacts(directory, "")
            with self.assertRaisesRegex(exp.ExperimentError, "nope.npy"):
                exp.resolve_embedding_paths("minilm", Path(directory) / "nope.npy", processed_dir=Path(directory))

    def test_missing_everything_suggests_generation_command(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(exp.ExperimentError, r"--model minilm"):
                exp.resolve_embedding_paths("minilm", processed_dir=Path(directory))


class ValidationTests(unittest.TestCase):
    def test_load_checks_alignment_dimension_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix, ids, _ = exp.load_embeddings(*write_artifacts(directory, "_ok"))
            self.assertEqual((matrix.shape, ids), ((2, 3), ["a0", "a1"]))
            with self.assertRaisesRegex(exp.ExperimentError, "misaligned: 2 rows vs 1"):
                exp.load_embeddings(*write_artifacts(directory, "_rows", ids=["a0"]))
            emb_path, map_path = write_artifacts(directory, "_dim")
            payload = json.loads(map_path.read_text(encoding="utf-8")) | {"embedding_dim": 768}
            map_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(exp.ExperimentError, "Dimension mismatch"):
                exp.load_embeddings(emb_path, map_path)
            with self.assertRaisesRegex(exp.ExperimentError, "duplicated"):
                exp.load_embeddings(*write_artifacts(directory, "_dup", ids=["a0", "a0"]))

    def test_encoder_spec_from_new_mapping(self):
        mapping = {
            "model_name": "malteos/scincl",
            "model_alias": "scincl",
            "token_pooling": "cls",
            "max_seq_length": 512,
            "input_format": "title_sep_abstract",
        }
        spec = exp.encoder_spec("scincl", mapping, 768)
        self.assertEqual(
            (spec.model_id, spec.token_pooling, spec.max_seq_length, spec.input_format, spec.dimension),
            ("malteos/scincl", "cls", 512, "title_sep_abstract", 768),
        )

    def test_encoder_spec_legacy_minilm_defaults(self):
        spec = exp.encoder_spec("minilm", {"model_name": exp.MODEL_IDS["minilm"]}, 384)
        self.assertEqual((spec.token_pooling, spec.max_seq_length, spec.input_format), ("mean", 256, "raw"))

    def test_encoder_spec_rejects_model_mismatch(self):
        with self.assertRaisesRegex(exp.ExperimentError, "does not match --model specter2"):
            exp.encoder_spec("specter2", {"model_name": exp.MODEL_IDS["minilm"]}, 384)
        with self.assertRaisesRegex(exp.ExperimentError, "produced with model 'scincl'"):
            exp.encoder_spec("specter2", {"model_name": "malteos/scincl", "model_alias": "scincl"}, 768)


class OutputNamingTests(unittest.TestCase):
    def test_result_files_include_alias_and_tag(self):
        outputs = exp.result_paths(Path("x"), "specter2", "_leakage_free")
        self.assertEqual(outputs["summary"], Path("x") / "rerank_summary_specter2_leakage_free.csv")
        self.assertEqual(outputs["per_case"], Path("x") / "rerank_per_case_specter2_leakage_free.csv")
        self.assertEqual(outputs["meta"], Path("x") / "rerank_meta_specter2_leakage_free.json")
        self.assertEqual(outputs["markdown"], Path("x") / "rerank_results_specter2_leakage_free.md")
        self.assertEqual(exp.result_paths(Path("x"), "minilm")["summary"], Path("x") / "rerank_summary_minilm.csv")

    def test_models_do_not_collide(self):
        names = {exp.result_paths(Path("x"), alias, "_t")["summary"] for alias in exp.MODEL_IDS}
        self.assertEqual(len(names), len(exp.MODEL_IDS))


class LeakageFreeEncodingTests(unittest.TestCase):
    def test_format_article_matches_02(self):
        self.assertEqual(exp.format_article("T\nA b", "title_sep_abstract", "[SEP]"), "T[SEP]A b")
        self.assertEqual(exp.format_article("T\nA", "raw", None), "T\nA")
        with self.assertRaises(exp.ExperimentError):
            exp.format_article("T\nA", "title_sep_abstract", None)

    def test_encode_plans_uses_model_input_format_and_pools(self):
        seen = []

        class Fake:
            tokenizer = SimpleNamespace(sep_token="[SEP]")

            def encode(self, inputs, **kwargs):
                seen.extend(inputs)
                vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
                return vectors[: len(inputs)]

        spec = exp.EncoderSpec("scincl", "malteos/scincl", "cls", 512, "title_sep_abstract", 2)
        texts = {"p1": "T1\nA1", "p2": "T2\nA2"}
        vectors = exp.encode_plans({("g", "x"): ["p1", "p2"], ("g", "y"): ["p2"]}, texts, spec, model=Fake())
        self.assertEqual(seen, ["T1[SEP]A1", "T2[SEP]A2"])
        np.testing.assert_allclose(vectors[("g", "x")], [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-6)
        np.testing.assert_allclose(vectors[("g", "y")], [0.0, 1.0])

    def test_encode_plans_rejects_dimension_mismatch(self):
        class Fake:
            def encode(self, inputs, **kwargs):
                return np.ones((len(inputs), 4), dtype=np.float32)

        spec = exp.EncoderSpec("minilm", exp.MODEL_IDS["minilm"], "mean", 256, "raw", 384)
        with self.assertRaisesRegex(exp.ExperimentError, "dim 4"):
            exp.encode_plans({("g", "x"): ["p"]}, {"p": "t"}, spec, model=Fake())


if __name__ == "__main__":
    unittest.main()
