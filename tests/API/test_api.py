"""Tests for the embedding API endpoints."""

import pytest
from fastapi.testclient import TestClient

from embedding_pfa_case.app import app

EMBED_URL = "/embed"
SIMILARITY_URL = "/similarity"
HEALTH_URL = "/health"


@pytest.fixture(scope="module")
def client():
    """Test client that triggers the full lifespan (model loading)."""
    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get(HEALTH_URL)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# Happy-path embedding
# ---------------------------------------------------------------------------


class TestEmbedHappyPath:
    def test_single_text_query(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["hello world"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_texts"] == 1
        assert body["dim"] == 1024
        assert len(body["embeddings"]) == 1
        assert len(body["embeddings"][0]) == 1024

    def test_multiple_texts(self, client: TestClient) -> None:
        texts = ["first text", "second text", "third text"]
        resp = client.post(EMBED_URL, json={"texts": texts})
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_texts"] == 3
        assert len(body["embeddings"]) == 3

    def test_passage_input_type(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["some document"], "input_type": "passage"})
        assert resp.status_code == 200
        assert resp.json()["num_texts"] == 1

    def test_default_input_type_is_query(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["test"]})
        assert resp.status_code == 200

    def test_danish_text(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["Hvad er meningen med livet?"]})
        assert resp.status_code == 200
        assert resp.json()["num_texts"] == 1

    def test_danish_characters(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["rød grød med fløde", "Ål i Ærø"]})
        assert resp.status_code == 200
        assert resp.json()["num_texts"] == 2


# ---------------------------------------------------------------------------
# Input validation — 422 Unprocessable Entity
# ---------------------------------------------------------------------------


class TestEmbedValidation:
    def test_empty_texts_list(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": []})
        assert resp.status_code == 422

    def test_missing_texts_field(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={})
        assert resp.status_code == 422

    def test_empty_string_in_texts(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": [""]})
        assert resp.status_code == 422

    def test_invalid_input_type(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["hello"], "input_type": "invalid"})
        assert resp.status_code == 422

    def test_text_exceeding_max_length(self, client: TestClient) -> None:
        long_text = "a" * 513
        resp = client.post(EMBED_URL, json={"texts": [long_text]})
        assert resp.status_code == 422

    def test_text_with_disallowed_characters(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": ["hello <script>alert(1)</script>"]})
        assert resp.status_code == 422

    def test_non_string_in_texts(self, client: TestClient) -> None:
        resp = client.post(EMBED_URL, json={"texts": [123]})
        assert resp.status_code == 422

    def test_too_many_texts(self, client: TestClient) -> None:
        texts = ["text"] * 65
        resp = client.post(EMBED_URL, json={"texts": texts})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Similarity endpoint — happy path
# ---------------------------------------------------------------------------


class TestSimilarityHappyPath:
    def test_single_query_single_passage(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"queries": ["hello"], "passages": ["world"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_queries"] == 1
        assert body["num_passages"] == 1
        assert len(body["scores"]) == 1
        assert len(body["scores"][0]) == 1

    def test_multiple_queries_and_passages(self, client: TestClient) -> None:
        resp = client.post(
            SIMILARITY_URL,
            json={"queries": ["first query", "second query"], "passages": ["passage a", "passage b", "passage c"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_queries"] == 2
        assert body["num_passages"] == 3
        assert len(body["scores"]) == 2
        assert all(len(row) == 3 for row in body["scores"])

    def test_matched_query_scores_higher(self, client: TestClient) -> None:
        """A protein query should score higher against a protein passage than an unrelated one."""
        resp = client.post(
            SIMILARITY_URL,
            json={
                "queries": ["how much protein should a female eat"],
                "passages": [
                    "The CDC recommends 46 grams of protein per day for women.",
                    "It is raining outside today.",
                ],
            },
        )
        assert resp.status_code == 200
        scores = resp.json()["scores"]
        assert scores[0][0] > scores[0][1]


# ---------------------------------------------------------------------------
# Similarity endpoint — validation
# ---------------------------------------------------------------------------


class TestSimilarityValidation:
    def test_missing_queries(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"passages": ["text"]})
        assert resp.status_code == 422

    def test_missing_passages(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"queries": ["text"]})
        assert resp.status_code == 422

    def test_empty_queries_list(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"queries": [], "passages": ["text"]})
        assert resp.status_code == 422

    def test_empty_passages_list(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"queries": ["text"], "passages": []})
        assert resp.status_code == 422

    def test_disallowed_characters_in_query(self, client: TestClient) -> None:
        resp = client.post(SIMILARITY_URL, json={"queries": ["<bad>"], "passages": ["text"]})
        assert resp.status_code == 422
