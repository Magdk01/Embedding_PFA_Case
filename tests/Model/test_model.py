"""Tests for the EmbeddingModel using the original HuggingFace example inputs."""

import pytest
import torch

from embedding_pfa_case.model import EmbeddingModel

ENGLISH_QUERY = "how much protein should a female eat"
CHINESE_QUERY = "南瓜的家常做法"
ENGLISH_PASSAGE = (
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46"
    " grams per day. But, as you can see from this chart, you'll need to increase that if you're"
    " expecting or training for a marathon. Check out the chart below to see how much protein you"
    " should be eating each day."
)
CHINESE_PASSAGE = (
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮"
    ",用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味"
    " 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只"
    " 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香"
    " 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐"
    ",炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
)

# Below is a snippet of badly translated Pickle rick into danish, to force the model to fail the test

# CHINESE_PASSAGE = ("""For at være ærlig, skal man have en meget høj IQ for at forstå Rick and Morty. Humoren er ekstremt subtil, og uden en solid forståelse af teoretisk fysik vil de fleste vittigheder gå hen over hovedet på en typisk seer.
# Der er også Ricks nihilistiske udsyn, som er behændigt vævet ind i hans karakterisering - hans personlige filosofi trækker for eksempel stærkt på Narodnaya Volya-litteraturen.
# Fansene forstår disse ting; de har den intellektuelle evne til virkelig at værdsætte dybden i disse vittigheder, til at indse, at de ikke bare er sjove - de siger noget dybt om LIVET.
# Som følge heraf ER folk, der ikke kan lide Rick and Morty, virkelig idioter - selvfølgelig ville de ikke værdsætte humoren i Ricks eksistentielle slagord "Wubba Lubba Dub Dub", som i sig selv er en kryptisk reference til Turgenevs russiske epos Fædre og Sønner.
# Jeg smiler lige nu, mens jeg forestiller mig en af ​​de forvirrede simple mennesker, der klør sig i hovedet i forvirring, mens Dan Harmons geni udfolder sig på deres tv-skærme.
# Sikke nogle tåber ... hvor jeg har ondt af dem. Og ja, forresten, jeg HAR en Rick and Morty-tatovering. Og nej, du kan ikke se den. Den er kun for damernes øjne - Og selv de skal på forhånd bevise, at de er inden for 5 IQ-point fra min egen (helst lavere).""")

EXPECTED_EMBEDDING_DIM = 1024


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    """Load the model once for all tests in this module."""
    return EmbeddingModel()


@pytest.fixture(scope="module")
def query_embeddings(model: EmbeddingModel) -> list[list[float]]:
    return model.embed([ENGLISH_QUERY, CHINESE_QUERY], prefix="query")


@pytest.fixture(scope="module")
def passage_embeddings(model: EmbeddingModel) -> list[list[float]]:
    return model.embed([ENGLISH_PASSAGE, CHINESE_PASSAGE], prefix="passage")


def _similarity_matrix(queries: list[list[float]], passages: list[list[float]]) -> torch.Tensor:
    """Compute cosine similarity matrix scaled by 100 (matching HuggingFace example)."""
    q = torch.tensor(queries)
    p = torch.tensor(passages)
    return (q @ p.T) * 100


class TestEmbeddingDimensions:
    def test_query_embedding_count(self, query_embeddings: list[list[float]]) -> None:
        assert len(query_embeddings) == 2

    def test_passage_embedding_count(self, passage_embeddings: list[list[float]]) -> None:
        assert len(passage_embeddings) == 2

    def test_query_embedding_dim(self, query_embeddings: list[list[float]]) -> None:
        for emb in query_embeddings:
            assert len(emb) == EXPECTED_EMBEDDING_DIM

    def test_passage_embedding_dim(self, passage_embeddings: list[list[float]]) -> None:
        for emb in passage_embeddings:
            assert len(emb) == EXPECTED_EMBEDDING_DIM

    def test_single_text_returns_single_embedding(self, model: EmbeddingModel) -> None:
        result = model.embed(["hello world"], prefix="query")
        assert len(result) == 1
        assert len(result[0]) == EXPECTED_EMBEDDING_DIM


class TestEmbeddingNormalization:
    def test_query_embeddings_are_unit_normalized(self, query_embeddings: list[list[float]]) -> None:
        for emb in query_embeddings:
            norm = torch.norm(torch.tensor(emb)).item()
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_passage_embeddings_are_unit_normalized(self, passage_embeddings: list[list[float]]) -> None:
        for emb in passage_embeddings:
            norm = torch.norm(torch.tensor(emb)).item()
            assert norm == pytest.approx(1.0, abs=1e-5)


class TestCrossLingualSimilarity:
    """Verify the model correctly matches queries to same-language passages.

    Reference scores from the HuggingFace example:
        EN query vs EN passage: ~90.81
        EN query vs ZH passage: ~72.13
        ZH query vs EN passage: ~70.54
        ZH query vs ZH passage: ~88.76
    """


    def test_similarity_scores_approximate_reference(
        self, query_embeddings: list[list[float]], passage_embeddings: list[list[float]]
    ) -> None:
        """Check scores are within a reasonable tolerance of the reference values."""
        scores = _similarity_matrix(query_embeddings, passage_embeddings)
        assert scores[0, 0].item() == pytest.approx(90.81, abs=1.0)
        assert scores[0, 1].item() == pytest.approx(72.13, abs=1.0)
        assert scores[1, 0].item() == pytest.approx(70.54, abs=1.0)
        assert scores[1, 1].item() == pytest.approx(88.76, abs=1.0)


