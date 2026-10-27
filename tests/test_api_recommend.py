"""API tests with synthetic artifacts (no real data needed)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from infrastructure.api.main import create_app
from infrastructure.api.settings import Settings

N_AUTHORS = 40
DIM = 8


def write_fixture(root: Path) -> Settings:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(N_AUTHORS, DIM)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [f"A{i:03d}" for i in range(N_AUTHORS)]
    np.save(root / "emb.npy", matrix)
    (root / "map.json").write_text(
        json.dumps({"model_name": "test/minilm", "embedding_dim": DIM, "index_to_author_id": ids}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "author_id": ids,
            "author_name": [f"Author {i}" for i in range(N_AUTHORS)],
            "institution": [f"inst{i % 4}" if i % 7 else "UNKNOWN" for i in range(N_AUTHORS)],
            "low_visibility": [i % 3 == 0 for i in range(N_AUTHORS)],
        }
    ).to_csv(root / "meta.csv", index=False)
    return Settings(root / "emb.npy", root / "map.json", root / "meta.csv", root / "missing.csv")


class RecommendApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.client_cm = TestClient(create_app(write_fixture(Path(cls._tmp.name))))
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        cls._tmp.cleanup()

    def post(self, **payload):
        return self.client.post("/api/v1/recommend", json=payload)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["authors"], N_AUTHORS)
        self.assertEqual(body["model"], "test/minilm")

    def test_happy_path(self):
        r = self.post(author_ids=["A001", "A002"], top_k=10, k_candidates=30)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([t["author_id"] for t in body["team"]], ["A001", "A002"])
        self.assertEqual([p["rank"] for p in body["recommendations"]], list(range(1, 11)))
        first = body["recommendations"][0]
        for key in ("author_id", "name", "institution", "low_visibility", "relevance_score", "final_score"):
            self.assertIn(key, first)
        self.assertIn("gini_simpson", body["fairness_metrics"])
        self.assertEqual(body["parameters"]["rerank"]["max_per_institution"], 3)

    def test_top_k_respected_and_cap(self):
        body = self.post(author_ids=["A005"], top_k=7).json()
        self.assertEqual(len(body["recommendations"]), 7)
        insts = [p["institution"] for p in body["recommendations"] if p["institution"] != "UNKNOWN"]
        self.assertTrue(all(insts.count(i) <= 3 for i in insts))

    def test_team_excluded(self):
        team = ["A003", "A010", "A020"]
        body = self.post(author_ids=team, top_k=30, k_candidates=37).json()
        ids = [p["author_id"] for p in body["recommendations"]]
        self.assertEqual(len(ids), 30)
        self.assertFalse(set(team) & set(ids))

    def test_baseline_order_when_fairness_disabled(self):
        body = self.post(author_ids=["A001"], top_k=5, fairness={"enabled": False}).json()
        scores = [p["relevance_score"] for p in body["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([p["baseline_rank"] for p in body["recommendations"]], [1, 2, 3, 4, 5])

    def test_unknown_ids(self):
        r = self.post(author_ids=["A001", "NOPE"])
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["unknown_author_ids"], ["NOPE"])

    def test_empty_list(self):
        self.assertEqual(self.post(author_ids=[]).status_code, 422)
        self.assertEqual(self.post(author_ids=["  "]).status_code, 422)

    def test_invalid_params(self):
        self.assertEqual(self.post(author_ids=["A001"], top_k=0).status_code, 422)
        self.assertEqual(self.post(author_ids=["A001"], top_k=50, k_candidates=10).status_code, 422)


if __name__ == "__main__":
    unittest.main()
