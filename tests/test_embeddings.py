import os

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from script import embeddings
from script.embeddings import create_event_documents


def test_create_event_documents_creates_one_document_per_csv_row():
    events = pd.DataFrame(
        [
            {"uid": "1", "title_fr": "Concert", "location_city": "Nantes"},
            {"uid": "2", "title_fr": "Atelier", "location_city": "Nantes"},
        ]
    )

    documents = create_event_documents(events)

    assert len(documents) == len(events)
    assert "Concert" in documents[0].page_content
    assert documents[0].metadata["uid"] == "1"
    assert "Atelier" in documents[1].page_content
    assert documents[1].metadata["uid"] == "2"


def test_build_vector_store_creates_faiss_index(tmp_path, monkeypatch):
    class FakeEmbeddings:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "MistralAIEmbeddings", FakeEmbeddings)
    documents = [Document(page_content="Concert à Nantes", metadata={"uid": "1"})]

    vector_store = embeddings.build_vector_store(documents, tmp_path)

    assert vector_store.index.ntotal == 1
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index.pkl").exists()


def get_mistral_embeddings():
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        pytest.skip("La clé MISTRAL_API_KEY n'est pas définie.")
    return MistralAIEmbeddings(model="mistral-embed")


@pytest.mark.skipif(
    os.getenv("RUN_MISTRAL_INTEGRATION") != "1",
    reason="Définir RUN_MISTRAL_INTEGRATION=1 pour appeler l'API Mistral.",
)
def test_mistral_returns_embedding_vector():
    vector = get_mistral_embeddings().embed_query("Un événement culturel à Nantes")

    assert isinstance(vector, list)
    assert vector
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.skipif(
    os.getenv("RUN_MISTRAL_INTEGRATION") != "1",
    reason="Définir RUN_MISTRAL_INTEGRATION=1 pour appeler l'API Mistral.",
)
def test_mistral_embeddings_have_consistent_dimensions():
    vectors = get_mistral_embeddings().embed_documents(
        ["Un concert à Nantes", "Une exposition à Nantes"]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
