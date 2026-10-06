"""Regression checks for CSV decoding and publication/profile semantics."""

import csv
import importlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

etl = importlib.import_module("infrastructure.scripts.01_filter_and_merge_authors")


class CsvTests(unittest.TestCase):
    def test_mixed_quoting_preserves_identifiers_and_multiline_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "articles.csv"
            with path.open("w", encoding="utf-8", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(["identifier", "title", "abstract"])
                writer.writerow(["001", "Standard", "Two\nlines"])
                writer.writerow(['NA,"Title, with comma","Quoted ""word"""'])
                writer.writerow(["", "No ID", ""])
            frame = etl.read_csv(path, ["identifier", "title", "abstract"], ["identifier"])
            self.assertEqual(frame.loc[0, "identifier"], "001")
            self.assertEqual(frame.loc[0, "abstract"], "Two\nlines")
            self.assertEqual(frame.loc[1, "identifier"], "NA")
            self.assertEqual(frame.loc[1, "title"], "Title, with comma")
            self.assertEqual(frame.loc[1, "abstract"], 'Quoted "word"')
            self.assertTrue(pd.isna(frame.loc[2, "identifier"]))

    def test_bad_schema_and_bad_record_fail_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "articles.csv"
            path.write_text("identifier,title\n001,Title\n", encoding="utf-8")
            with self.assertRaisesRegex(etl.DataLoadError, "Available columns"):
                etl.read_csv(path, ["abstract"], [])
            path.write_text("identifier,title\n001,Title,Extra\n", encoding="utf-8")
            with self.assertRaisesRegex(etl.DataLoadError, "expected 2 fields, found 3"):
                etl.read_csv(path, ["identifier"], ["identifier"])
            with self.assertRaisesRegex(etl.DataLoadError, "Input file not found"):
                etl.read_csv(Path(directory) / "missing.csv", [], [])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.authors = pd.DataFrame({"identifier": ["001", "002"], "auth_name": ["Ana", "Luis"]})
        self.articles = pd.DataFrame({
            "identifier": ["03", "01", "02", "02"],
            "title": ["", " First ", "Second", "Duplicate"],
            "abstract": ["", " Abstract ", "", "Ignored"],
        })
        self.edges = pd.DataFrame({
            "author_id": ["001", "001", "001", "001", "001", "999", None, "002"],
            "article_id": ["03", "02", "01", "01", "missing", "01", "01", "01"],
        })

    def profiles(self, min_articles=3):
        edges = etl.clean_edges(self.edges, self.authors, self.articles)
        counts = etl.apply_quality_filter(etl.count_articles_per_author(edges), min_articles)
        return etl.build_profiles(counts, edges, etl.build_article_texts(self.articles), self.authors)

    def test_counts_valid_distinct_articles_and_orders_text(self):
        result = self.profiles()
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "author_id"], "001")
        self.assertEqual(result.loc[0, "author_name"], "Ana")
        self.assertEqual(result.loc[0, "article_count"], 3)
        self.assertEqual(result.loc[0, "concatenated_profile_text"], "First\nAbstract\n\nSecond")

    def test_empty_text_articles_count_without_dropping_the_author(self):
        self.articles[["title", "abstract"]] = ""
        result = self.profiles()
        self.assertEqual(result.loc[0, "article_count"], 3)
        self.assertEqual(result.loc[0, "concatenated_profile_text"], "")

    def test_no_survivors_still_exports_output_headers(self):
        result = self.profiles(min_articles=4)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), etl.OUTPUT_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed" / "profiles.csv"
            etl.export_profiles(result, path)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), ",".join(etl.OUTPUT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
