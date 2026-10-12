"""Primary-institution resolution and low-visibility flags on a tiny synthetic CSV set."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from domain.entities.research_group import UNKNOWN_INSTITUTION
from infrastructure.db.csv_sources import normalize_institution_name, primary_institutions, visibility_table


def write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


class CsvSourcesTests(unittest.TestCase):
    def setUp(self):
        self.edges = pd.DataFrame({
            "author_id": ["a", "a", "a", "b", "b", "c", "d"],
            "article_id": ["p1", "p2", "p3", "p1", "p2", "p3", "p4"],
        })

    def test_most_frequent_affiliation_wins_and_names_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            write(raw / "affiliations.csv", "identifier,affiliation_name,city,country",
                  ["1,Escuela Politécnica Nacional,Quito,Ecuador", "2,ESCUELA  politécnica nacional,,Ecuador",
                   "3,Universidad X,Quito,Ecuador"])
            write(raw / "authors_affiliations.csv", ",author_id,affiliation_id", ["0,a,1", "1,a,3", "2,b,2"])
            write(raw / "articles_affiliations.csv", ",article_id,affiliation_id",
                  ["0,p1,3", "1,p2,3", "2,p3,1"])
            mapping, stats = primary_institutions(["a", "b", "c"], self.edges, raw)
        self.assertEqual(mapping["a"], "universidad x")
        self.assertEqual(mapping["b"], normalize_institution_name("Escuela Politécnica Nacional"))
        self.assertEqual(mapping["c"], UNKNOWN_INSTITUTION)
        self.assertEqual((stats.missing, stats.multiple, stats.distinct_institutions), (1, 1, 2))

    def test_visibility_requires_both_low_count_and_low_degree(self):
        counts = pd.Series({"a": 3, "b": 2, "c": 1, "d": 1})
        table = visibility_table(self.edges, counts, max_authors_per_article=None).set_index("author_id")
        self.assertEqual(table.loc["a", "coauthor_degree"], 2)
        self.assertEqual(table.loc["d", "coauthor_degree"], 0)
        self.assertTrue(table.loc["d", "protected"])
        self.assertFalse(table.loc["a", "protected"])


if __name__ == "__main__":
    unittest.main()
