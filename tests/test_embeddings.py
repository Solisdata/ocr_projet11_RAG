import os

import pandas as pd
import pytest
from langchain_core.documents import Document

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
        def embed_documents(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    monkeypatch.setattr(embeddings, "create_embeddings", FakeEmbeddings)
    documents = [Document(page_content="Concert a Nantes", metadata={"uid": "1"})]

    vector_store = embeddings.build_vector_store(documents, tmp_path)

    assert vector_store.index.ntotal == 1
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index.pkl").exists()


def get_ollama_embeddings():
    try:
        return embeddings.create_embeddings()
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to call the local Ollama service.",
)
def test_ollama_returns_embedding_vector():
    vector = get_ollama_embeddings().embed_query("Un evenement culturel a Nantes")

    assert isinstance(vector, list)
    assert vector
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to call the local Ollama service.",
)
def test_ollama_embeddings_have_consistent_dimensions():
    vectors = get_ollama_embeddings().embed_documents(
        ["Un concert a Nantes", "Une exposition a Nantes"]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
