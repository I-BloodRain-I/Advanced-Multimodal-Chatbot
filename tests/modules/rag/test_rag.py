import pytest
import numpy as np
from unittest.mock import Mock, create_autospec

from modules.rag.rag import RAG
from modules.rag.vector_db.base import VectorDatabaseBase
from core.entities.types import DocumentChunk, RagDocument, Message


@pytest.fixture
def mock_vector_db():
    return create_autospec(VectorDatabaseBase)


@pytest.fixture
def rag_instance(mock_vector_db):
    return RAG(vector_db=mock_vector_db, n_extracted_docs=2, prompt_format="{context}\n{prompt}")


def test_validate_prompt_format_valid(rag_instance):
    rag_instance._validate_prompt_format()  # should not raise


def test_validate_prompt_format_invalid():
    rag = RAG(vector_db=Mock(), prompt_format="invalid template")
    with pytest.raises(ValueError, match="prompt_format must include '{context}' and '{prompt}'"):
        rag._validate_prompt_format()


def test_add_documents_calls_vector_db(rag_instance, mock_vector_db):
    documents = [Mock(spec=RagDocument)]
    rag_instance.add_documents(documents)
    mock_vector_db.add_documents.assert_called_once_with(documents)


@pytest.mark.parametrize("input_type", ["list", "ndarray", "tensor"])
def test_normalize_embeddings_supported_types(rag_instance, input_type):
    if input_type == "list":
        embeddings = [[0.1, 0.2, 0.3]]
    elif input_type == "ndarray":
        embeddings = np.array([[0.1, 0.2, 0.3]])
    else:
        import torch
        embeddings = torch.tensor([[0.1, 0.2, 0.3]])

    result = rag_instance._normalize_embeddings(embeddings)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)


def test_normalize_embeddings_invalid_type(rag_instance):
    with pytest.raises(ValueError):
        rag_instance._normalize_embeddings("invalid")


def test_extract_similar_docs_normalizes_and_queries(rag_instance, mock_vector_db):
    embeddings = np.array([[0.1, 0.2, 0.3]])
    expected = [[Mock(spec=DocumentChunk)]]
    mock_vector_db.search.return_value = expected

    result = rag_instance.extract_similar_docs(embeddings)
    assert result == expected
    mock_vector_db.search.assert_called_once()


def test_add_context_injects_context(rag_instance, mock_vector_db):
    # Setup
    message = Message(role="user", content="Hello?")
    messages_batch = [[message]]
    embeddings = np.array([[0.1, 0.2, 0.3]])
    chunks = [DocumentChunk(document_id="doc1", content="Relevant info", embeddings=None)]

    mock_vector_db.search.return_value = [[chunks[0]]]

    rag_instance.add_context(messages_batch, embeddings)

    updated_message = messages_batch[0][-1]
    assert "Relevant info" in updated_message.content
    assert "Hello?" in updated_message.content


def test_add_context_logs_and_continues_on_empty_history(rag_instance, mock_vector_db, caplog):
    embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_vector_db.search.return_value = [[]]
    messages_batch = [[]]  # empty history

    with caplog.at_level("WARNING"):
        rag_instance.add_context(messages_batch, embeddings)
        assert "Empty conversation history encountered" in caplog.text