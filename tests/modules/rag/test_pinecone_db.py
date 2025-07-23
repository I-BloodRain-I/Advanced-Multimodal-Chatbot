import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from modules.rag.vector_db.pinecone_db import PineconeDatabase
from core.entities.types import DocumentChunk, RagDocument

@pytest.fixture
def sample_documents():
    return [
        RagDocument(
            id="doc1",
            chunks=[
                DocumentChunk(
                    document_id="chunk1",
                    content="Chunk content 1",
                    embeddings=np.ones(768)
                ),
                DocumentChunk(
                    document_id="chunk2",
                    content="Chunk content 2",
                    embeddings=np.zeros(768)
                )
            ],
            metadata={"language": "en"}
        )
    ]

@pytest.fixture
def pinecone_mock():
    with patch("modules.rag.vector_db.pinecone_db.Pinecone") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.list_indexes.return_value = []
        mock_instance.create_index.return_value = None
        mock_instance.Index.return_value = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance

def test_init_pinecone_database(pinecone_mock):
    db = PineconeDatabase(index_name="test-index")
    assert db.index_name == "test-index"
    assert db.dimension == 768
    pinecone_mock.list_indexes.assert_called_once()

def test_create_index_existing(pinecone_mock):
    pinecone_mock.list_indexes.return_value = [{'name': 'test-index'}]
    db = PineconeDatabase(index_name="test-index")
    pinecone_mock.Index.assert_called_with("test-index")

def test_add_documents_success(pinecone_mock, sample_documents):
    db = PineconeDatabase(index_name="test-index")
    db.index = MagicMock()
    print(sample_documents)
    db.add_documents(sample_documents)
    db.index.upsert.assert_called_once()
    assert db.index.upsert.call_args[0][0][0]['id'] == "doc1_chunk_0"

def test_search_success(pinecone_mock):
    db = PineconeDatabase(index_name="test-index")
    db.index = MagicMock()
    mock_response = {
        'matches': [
            {'metadata': {'doc_id': 'doc1', 'text': 'Matched text'}}
        ]
    }
    db.index.query.return_value = mock_response
    query = np.ones((1, 768))
    results = db.search(query_embeddings=query)
    assert len(results) == 1
    assert isinstance(results[0][0], DocumentChunk)
    assert results[0][0].content == "Matched text"

def test_search_invalid_dimension(pinecone_mock):
    db = PineconeDatabase(index_name="test-index")
    with pytest.raises(ValueError):
        db.search(query_embeddings=np.ones((1, 128)))  # wrong dimension

def test_normalize_embeddings_valid(pinecone_mock):
    db = PineconeDatabase(index_name="test-index")
    vec = np.ones(768)
    norm = db._normalize_embeddings(vec)
    assert norm.shape == (1, 768)

def test_normalize_embeddings_invalid(pinecone_mock):
    db = PineconeDatabase(index_name="test-index")
    with pytest.raises(ValueError):
        db._normalize_embeddings(np.ones((10, 100, 3)))  # 3D array not allowed