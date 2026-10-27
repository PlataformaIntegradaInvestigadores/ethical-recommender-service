"""Author search endpoint, search use case and CORS (synthetic artifacts)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from application.use_cases.search_authors import (
    AuthorSearchError, AuthorSearchQuery, SearchAuthors, normalize_text, within_one_edit, word_score)
from domain.entities.researcher import AuthorSummary, ResearcherProfile
from infrastructure.api.main import create_app
from infrastructure.api.settings import Settings, _origins
from infrastructure.db.artifact_loaders import CsvMetadataRepository, compose_full_name, read_full_names

AUTHORS = [  # (id, name, institution, low_visibility, article_count)
    ("10040712400", "Carrión V.", "parque nacional galápagos", False, 13),
    ("15842091200", "Lavoie C.", "university of galapagos", True, 4),
    ("20000000001", "Carrion-Pérez M.", "escuela politécnica nacional", False, 30),
    ("20000000002", "Pérez Ortiz J.", "universidad central del ecuador", True, 3),
    ("20000000003", "Ortiz Á.", "UNKNOWN", True, 5),
    ("20000000004", None, "UNKNOWN", False, 3),
    ("37026882700", "Piedrahita P.", "escuela superior politecnica del litoral ecuador", False, 16),
    ("57222268863", "Cadena-Piedrahita D.", "universidad tecnica de babahoyo", False, 6),
    ("24290235400", "Luis Piedrahita C.", "UNKNOWN", False, 2),
    ("20000000005", "Quezada P.A.", "universidad tecnica particular de loja", False, 40),
    ("20000000006", "Paz C.P.", "UNKNOWN", False, 7),
]
NOT_EMBEDDED = ("99999999999", "Carrion Fantasma X.", "escuela politécnica nacional", False, 50)
# Scopus authors.csv first/last names (raw source of full_name); Lavoie and Ortiz have none.
RAW_NAMES = {"10040712400": ("Víctor", "Carrión"), "20000000001": ("María", "Carrion-Pérez"),
             "20000000002": ("Juan", "Pérez Ortiz"), "37026882700": ("Paolo", "Piedrahita"),
             "57222268863": ("Dalton", "Cadena-Piedrahita"), "24290235400": ("C.", "Luis Piedrahita"),
             "20000000005": ("Pablo A.", "Quezada"), "20000000006": ("Clara P.", "Paz"),
             "99999999999": ("Xavier", "Carrion Fantasma")}


def write_raw_authors(path: Path) -> None:
    rows = [(a, f, l, "x", "x") for a, (f, l) in RAW_NAMES.items()] + [("15842091200", None, "Lavoie", "x", "x")]
    pd.DataFrame(rows, columns=["identifier", "first_name", "last_name", "auth_name", "initials"]
                 ).to_csv(path, index=False)


def write_fixture(root: Path, with_metadata: bool = True, cors: tuple[str, ...] | None = None,
                  full_names: str = "column") -> Settings:
    """``full_names``: "column" (metadata has full_name), "raw" (only authors.csv has them) or "none"."""
    ids = [a[0] for a in AUTHORS]
    matrix = np.eye(len(ids), 4, dtype=np.float32) + 0.1
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(root / "emb.npy", matrix)
    (root / "map.json").write_text(json.dumps({"model_name": "test/minilm", "index_to_author_id": ids}),
                                   encoding="utf-8")
    rows = AUTHORS + [NOT_EMBEDDED]
    meta_frame = pd.DataFrame(rows, columns=["author_id", "author_name", "institution", "low_visibility",
                                             "article_count"])
    if full_names == "column":
        meta_frame["full_name"] = [compose_full_name(*RAW_NAMES[r[0]]) if r[0] in RAW_NAMES else None for r in rows]
    meta_frame.to_csv(root / "meta.csv", index=False)
    write_raw_authors(root / "authors.csv")
    pd.DataFrame({"author_id": [r[0] for r in rows], "author_name": [r[1] for r in rows],
                  "article_count": [r[4] for r in rows], "concatenated_profile_text": "x"}
                 ).to_csv(root / "profiles.csv", index=False)
    meta = root / ("meta.csv" if with_metadata else "missing.csv")
    extra = {} if cors is None else {"cors_origins": cors}
    if full_names == "raw":
        extra["authors_path"] = root / "authors.csv"
    return Settings(root / "emb.npy", root / "map.json", meta, root / "profiles.csv", **extra)


class AuthorsApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.client_cm = TestClient(create_app(write_fixture(Path(cls._tmp.name))))
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        cls._tmp.cleanup()

    def search(self, q, **params):
        return self.client.get("/api/v1/authors", params={"q": q, **params})

    def ids(self, q, **params):
        r = self.search(q, **params)
        self.assertEqual(r.status_code, 200, r.text)
        return [a["author_id"] for a in r.json()["results"]]

    def test_accent_and_case_insensitive(self):
        self.assertEqual(self.ids("CARRION"), ["20000000001", "10040712400"])  # more articles first (same tier)
        self.assertEqual(self.ids("carrión v"), ["10040712400"])
        self.assertEqual(self.ids("perez"), ["20000000002", "20000000001"])  # prefix tier beats word tier

    def test_response_fields(self):
        body = self.search("lavoie").json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["query"], "lavoie")
        self.assertEqual(body["results"][0], {"author_id": "15842091200", "name": "Lavoie C.",
                                              "institution": "university of galapagos", "low_visibility": True,
                                              "article_count": 4, "full_name": None})
        body = self.search("pablo piedrahita").json()
        self.assertEqual(body["results"][0]["full_name"], "Piedrahita, Paolo")
        self.assertEqual(body["results"][0]["name"], "Piedrahita P.")

    def test_first_name_last_name_finds_scopus_author(self):
        # "Pablo Piedrahita" -> Scopus "Piedrahita P." (full name Paolo): typo/initial match, any order
        self.assertEqual(self.ids("Pablo Piedrahita")[0], "37026882700")
        self.assertEqual(self.ids("piedrahita pablo")[0], "37026882700")
        self.assertEqual(self.ids("Paolo Piedrahita")[0], "37026882700")
        self.assertEqual(self.ids("Piedrahita, Paolo")[0], "37026882700")

    def test_multi_token_query_needs_a_surname(self):
        # "Quezada, Pablo A." matches "pablo" exactly and "piedrahita" only via the initial "P." of
        # "Quezada P.A."; no surname word matched, so it is not a hit
        self.assertNotIn("20000000005", self.ids("Pablo Piedrahita"))
        self.assertNotIn("20000000006", self.ids("Piedrahita P."))
        self.assertEqual(self.ids("pablo quezada"), ["20000000005"])

    def test_scopus_format_still_works(self):
        self.assertEqual(self.ids("Piedrahita P.")[0], "37026882700")
        self.assertEqual(self.ids("Carrion-Perez M."), ["20000000001"])
        self.assertEqual(self.ids("Lavoie C."), ["15842091200"])

    def test_partial_query(self):
        self.assertEqual(self.ids("piedra")[:1], ["37026882700"])  # word prefix; more articles first
        self.assertEqual(set(self.ids("piedra")), {"37026882700", "57222268863", "24290235400"})
        self.assertEqual(self.ids("pao piedr"), ["37026882700"])
        self.assertEqual(self.ids("dalton cadena"), ["57222268863"])

    def test_case_insensitive_full_name(self):
        self.assertEqual(self.ids("PABLO PIEDRAHITA")[0], "37026882700")
        self.assertEqual(self.ids("víctor CARRIÓN"), ["10040712400"])

    def test_first_name_alone_does_not_match_initials(self):
        # "pablo" must not return every author with initial P: exact first name, then the one-edit typo
        self.assertEqual(self.ids("pablo"), ["20000000005", "37026882700"])
        self.assertEqual(self.ids("zoila"), [])

    def test_exact_scopus_id(self):
        self.assertEqual(self.ids("10040712400"), ["10040712400"])
        self.assertEqual(self.ids(" 15842091200 "), ["15842091200"])
        self.assertEqual(self.ids("100407"), [])  # ids match exactly, not by prefix

    def test_only_embedded_authors(self):
        self.assertNotIn("99999999999", self.ids("fantasma"))
        self.assertEqual(self.ids("99999999999"), [])

    def test_limit(self):
        self.assertEqual(len(self.ids("ar", limit=1)), 1)
        self.assertEqual(self.search("carrion", limit=0).status_code, 422)
        self.assertEqual(self.search("carrion", limit=51).status_code, 422)

    def test_too_short_or_missing(self):
        self.assertEqual(self.search("a").status_code, 422)
        self.assertEqual(self.search("  -  ").status_code, 422)
        self.assertEqual(self.client.get("/api/v1/authors").status_code, 422)

    def test_no_match(self):
        r = self.search("zzzz")
        self.assertEqual(r.json(), {"query": "zzzz", "count": 0, "results": []})

    def test_cors_preflight_default_origin(self):
        r = self.client.options("/api/v1/recommend", headers={
            "Origin": "http://localhost:4200", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "http://localhost:4200")
        r = self.client.get(
            "/api/v1/authors", params={"q": "lavoie"}, headers={"Origin": "http://127.0.0.1:4200"})
        self.assertEqual(r.headers["access-control-allow-origin"], "http://127.0.0.1:4200")

    def test_cors_rejects_other_origin(self):
        r = self.client.get("/api/v1/authors", params={"q": "lavoie"}, headers={"Origin": "http://evil.test"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("access-control-allow-origin", r.headers)


class CorsSettingsTest(unittest.TestCase):
    def test_origins_parsing(self):
        self.assertEqual(_origins(None), ("http://localhost:4200", "http://127.0.0.1:4200"))
        self.assertEqual(_origins("http://a:1/, http://b:2 ,"), ("http://a:1", "http://b:2"))
        self.assertEqual(_origins(""), ())

    def test_cors_disabled(self):
        with tempfile.TemporaryDirectory() as tmp, \
                TestClient(create_app(write_fixture(Path(tmp), cors=()))) as client:
            r = client.get("/api/v1/authors", params={"q": "lavoie"}, headers={"Origin": "http://localhost:4200"})
            self.assertEqual(r.status_code, 200)
            self.assertNotIn("access-control-allow-origin", r.headers)


class FullNameSourcesTest(unittest.TestCase):
    def _first(self, full_names, q="pablo piedrahita"):
        with tempfile.TemporaryDirectory() as tmp, \
                TestClient(create_app(write_fixture(Path(tmp), full_names=full_names))) as client:
            return client.get("/api/v1/authors", params={"q": q}).json()["results"][0]

    def test_full_names_from_raw_authors_csv(self):
        hit = self._first("raw")
        self.assertEqual((hit["author_id"], hit["full_name"]), ("37026882700", "Piedrahita, Paolo"))

    def test_without_full_names_initials_still_match(self):
        hit = self._first("none")  # old author_metadata.csv, no authors.csv: "pablo" ~ initial "P."
        self.assertEqual((hit["author_id"], hit["full_name"]), ("37026882700", None))

    def test_read_full_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_authors(Path(tmp) / "a.csv")
            names = read_full_names(Path(tmp) / "a.csv", {"37026882700", "15842091200"})
            self.assertEqual(names, {"37026882700": "Piedrahita, Paolo", "15842091200": "Lavoie"})
        self.assertIsNone(compose_full_name("Ana", None))


class SearchFallbackTest(unittest.TestCase):
    def test_profiles_fallback_has_names_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp, \
                TestClient(create_app(write_fixture(Path(tmp), with_metadata=False))) as client:
            body = client.get("/api/v1/authors", params={"q": "lavoie"}).json()
            self.assertEqual(body["results"][0]["institution"], "UNKNOWN")
            self.assertEqual(body["results"][0]["article_count"], 4)


class _Embeddings:
    def __init__(self, ids):
        self._rows = {a: i for i, a in enumerate(ids)}

    def row_of(self, author_id):
        return self._rows.get(author_id)


class _Metadata:
    def __init__(self, summaries):
        self._s = summaries

    def summaries(self):
        return self._s

    def get(self, author_id):
        return ResearcherProfile(author_id, None)


class SearchAuthorsUseCaseTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_text("  Ñúñez-Carrión,  Á. "), "nunez carrion a")
        self.assertEqual(normalize_text(None), "")

    def test_word_score_and_edit_distance(self):
        self.assertTrue(within_one_edit("pablo", "paolo"))
        self.assertTrue(within_one_edit("perez", "pereza"))
        self.assertTrue(within_one_edit("ortzi", "ortiz"))  # adjacent swap
        self.assertFalse(within_one_edit("pablo", "pedro"))
        self.assertGreater(word_score("piedrahita", "piedrahita"), word_score("piedra", "piedrahita"))
        self.assertGreater(word_score("piedra", "piedrahita"), word_score("pablo", "paolo"))
        self.assertGreater(word_score("pablo", "paolo"), word_score("pablo", "p"))
        self.assertEqual(word_score("pablo", "maria"), 0)

    def test_more_matching_tokens_rank_first(self):
        uc = SearchAuthors(_Embeddings(["1", "2"]), _Metadata([
            AuthorSummary("1", "Piedrahita P.", article_count=99),
            AuthorSummary("2", "Piedrahita P.", article_count=1, full_name="Piedrahita, Pablo")]))
        self.assertEqual([a.author_id for a in uc.execute(AuthorSearchQuery("pablo piedrahita"))], ["2", "1"])
        self.assertEqual([a.author_id for a in uc.execute(AuthorSearchQuery("piedrahita"))], ["1", "2"])

    def test_tokens_need_distinct_words(self):
        uc = SearchAuthors(_Embeddings(["1"]), _Metadata([AuthorSummary("1", "Perez A.")]))
        self.assertEqual(uc.execute(AuthorSearchQuery("perez perez")), [])

    def test_embedded_without_metadata_found_by_id(self):
        uc = SearchAuthors(_Embeddings(["1", "2"]), _Metadata([AuthorSummary("1", "Ana B.")]))
        self.assertEqual(len(uc), 1)
        self.assertEqual(uc.execute(AuthorSearchQuery("2")), [AuthorSummary("2", None)])
        self.assertEqual(uc.execute(AuthorSearchQuery("ana")), [AuthorSummary("1", "Ana B.")])

    def test_exact_name_first_and_validation(self):
        uc = SearchAuthors(_Embeddings(["1", "2"]), _Metadata([
            AuthorSummary("1", "Ana Bel", article_count=9), AuthorSummary("2", "Ana", article_count=1)]))
        self.assertEqual([a.author_id for a in uc.execute(AuthorSearchQuery("ana"))], ["2", "1"])
        with self.assertRaises(AuthorSearchError):
            uc.execute(AuthorSearchQuery("ana", limit=0))


class CsvSummariesTest(unittest.TestCase):
    def test_summaries_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = write_fixture(Path(tmp))
            repo = CsvMetadataRepository(s.metadata_path, s.profiles_path)
            by_id = {a.author_id: a for a in repo.summaries()}
            self.assertEqual(by_id["10040712400"].article_count, 13)
            self.assertTrue(by_id["15842091200"].low_visibility)


if __name__ == "__main__":
    unittest.main()
