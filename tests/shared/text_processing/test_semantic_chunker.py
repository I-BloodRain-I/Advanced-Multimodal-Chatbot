import logging
import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch
from shared.text_processing import SemanticChunker

TEST_MODEL_NAME = 'test-model'
TEST_DEVICE = 'cpu'
TEST_CHUNK_LIMIT = 100
TEST_SIM_THRESHOLD = 0.75
TEST_MIN_TOKENS = 20
TEST_BATCH_SIZE = 32

@pytest.fixture
def mock_sentence_transformer():
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.encode.return_value = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.8, 0.9, 0.1]
    ])
    return mock_model

@pytest.fixture
def mock_tokenizer():
    mock_tokenizer = Mock()
    return mock_tokenizer

@pytest.fixture
def mock_chunker(mock_sentence_transformer, mock_tokenizer):
    with patch('shared.text_processing.semantic_chunker.SentenceTransformer', return_value=mock_sentence_transformer), \
         patch('shared.text_processing.semantic_chunker.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        return SemanticChunker(
            model_name=TEST_MODEL_NAME,
            chunk_token_limit=TEST_CHUNK_LIMIT,
            similarity_threshold=TEST_SIM_THRESHOLD,
            min_chunk_tokens=TEST_MIN_TOKENS,
            device_name=TEST_DEVICE
        )

def test_init_with_valid_params():
    with patch('shared.text_processing.semantic_chunker.SentenceTransformer') as mock_st, \
         patch('shared.text_processing.semantic_chunker.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        
        chunker = SemanticChunker(
            model_name=TEST_MODEL_NAME,
            chunk_token_limit=TEST_CHUNK_LIMIT,
            similarity_threshold=TEST_SIM_THRESHOLD,
            min_chunk_tokens=TEST_MIN_TOKENS,
            device_name=TEST_DEVICE
        )
        
        assert chunker.chunk_limit == TEST_CHUNK_LIMIT
        assert chunker.sim_threshold == TEST_SIM_THRESHOLD
        assert chunker.min_tokens == TEST_MIN_TOKENS
        assert chunker.device == TEST_DEVICE

def test_init_with_invalid_chunk_limit():
    with patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        with pytest.raises(ValueError, match="chunk_token_limit must be positive"):
            SemanticChunker(chunk_token_limit=0)

def test_init_with_invalid_similarity_threshold():
    with patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        with pytest.raises(ValueError, match="similarity_threshold must be between -1 and 1"):
            SemanticChunker(similarity_threshold=2.0)

def test_init_with_invalid_min_tokens():
    with patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        with pytest.raises(ValueError, match="min_chunk_tokens must be non-negative"):
            SemanticChunker(min_chunk_tokens=-1)

def test_init_with_min_tokens_greater_than_limit():
    with patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        with pytest.raises(ValueError, match="min_chunk_tokens .* must be less than chunk_token_limit"):
            SemanticChunker(chunk_token_limit=50, min_chunk_tokens=60)

def test_split_empty_text(mock_chunker):
    result = mock_chunker.split("")
    assert result == []

def test_split_whitespace_only(mock_chunker):
    result = mock_chunker.split("   \n\t  ")
    assert result == []

@patch('shared.text_processing.semantic_chunker.nltk.tokenize.sent_tokenize')
def test_split_sentences(mock_sent_tokenize, mock_chunker):
    mock_sent_tokenize.return_value = ["First sentence.", "Second sentence.", "Third sentence."]
    
    sentences = mock_chunker._split_sentences("Some text")
    
    assert sentences == ["First sentence.", "Second sentence.", "Third sentence."]
    mock_sent_tokenize.assert_called_once_with("Some text")

def test_count_tokens(mock_chunker):
    mock_chunker.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    count = mock_chunker._count_tokens("test text")
    
    assert count == 5
    mock_chunker.tokenizer.encode.assert_called_once_with("test text", add_special_tokens=False)

def test_embed(mock_chunker):
    mock_chunker.batch_size = TEST_BATCH_SIZE
    sentences = ["sentence1", "sentence2"]
    
    result = mock_chunker._embed(sentences)
    
    mock_chunker.model.encode.assert_called_once_with(
        sentences, 
        batch_size=TEST_BATCH_SIZE, 
        show_progress_bar=False
    )
    assert isinstance(result, np.ndarray)

def test_append_chunk(mock_chunker):
    container = []
    chunk = ["sentence1", "sentence2"]
    
    mock_chunker._append_chunk(container, chunk)
    
    assert container == ["sentence1 sentence2"]

def test_append_empty_chunk(mock_chunker):
    container = []
    chunk = []
    
    mock_chunker._append_chunk(container, chunk)
    
    assert container == []

def test_merge_small_empty_chunks(mock_chunker):
    result = mock_chunker._merge_small([])
    assert result == []

def test_merge_small_no_merging_needed(mock_chunker):
    chunks = ["chunk1 with enough tokens", "chunk2 with enough tokens"]
    mock_chunker._count_tokens = Mock(return_value=25)
    
    result = mock_chunker._merge_small(chunks)
    
    assert result == chunks

def test_merge_small_merges_small_chunk(mock_chunker):
    chunks = ["large chunk", "small"]
    mock_chunker._count_tokens = Mock(side_effect=[25, 5, 30])
    
    result = mock_chunker._merge_small(chunks)
    
    assert result == ["large chunk small"]

def test_merge_small_preserves_when_too_large(mock_chunker):
    chunks = ["large chunk", "small"]
    mock_chunker._count_tokens = Mock(side_effect=[50, 105, 155])
    
    result = mock_chunker._merge_small(chunks)
    
    assert result == ["large chunk", "small"]

@patch('shared.text_processing.semantic_chunker.nltk.tokenize.sent_tokenize')
def test_greedy_group_single_sentence(mock_sent_tokenize, mock_chunker):
    sentences = ["Single sentence."]
    embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_chunker._count_tokens = Mock(return_value=10)
    
    result = mock_chunker._greedy_group(sentences, embeddings)
    
    assert result == ["Single sentence."]

@patch('shared.text_processing.semantic_chunker.nltk.tokenize.sent_tokenize')
def test_greedy_group_oversized_sentence(mock_sent_tokenize, mock_chunker):
    sentences = ["Very long sentence that exceeds limit."]
    embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_chunker._count_tokens = Mock(return_value=150)
    
    result = mock_chunker._greedy_group(sentences, embeddings)
    
    assert result == ["Very long sentence that exceeds limit."]

@patch('shared.text_processing.semantic_chunker.F.cosine_similarity')
def test_greedy_group_high_similarity(mock_cosine_sim, mock_chunker):
    sentences = ["First sentence.", "Second sentence."]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    mock_chunker._count_tokens = Mock(return_value=10)
    mock_cosine_sim.return_value = torch.tensor([0.9])
    
    result = mock_chunker._greedy_group(sentences, embeddings)
    
    assert result == ["First sentence. Second sentence."]

@patch('shared.text_processing.semantic_chunker.F.cosine_similarity')
def test_greedy_group_low_similarity(mock_cosine_sim, mock_chunker):
    sentences = ["First sentence.", "Second sentence."]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    mock_chunker._count_tokens = Mock(return_value=10)
    mock_cosine_sim.return_value = torch.tensor([0.3])
    
    result = mock_chunker._greedy_group(sentences, embeddings)
    
    assert result == ["First sentence.", "Second sentence."]

@patch('shared.text_processing.semantic_chunker.F.cosine_similarity')
def test_greedy_group_token_limit_exceeded(mock_cosine_sim, mock_chunker):
    sentences = ["First sentence.", "Second sentence."]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    mock_chunker._count_tokens = Mock(return_value=60)
    mock_cosine_sim.return_value = torch.tensor([0.9])
    
    result = mock_chunker._greedy_group(sentences, embeddings)
    
    assert result == ["First sentence.", "Second sentence."]

@patch('shared.text_processing.semantic_chunker.nltk.tokenize.sent_tokenize')
def test_split_integration(mock_sent_tokenize, mock_chunker):
    mock_sent_tokenize.return_value = ["First sentence.", "Second sentence."]
    mock_chunker._count_tokens = Mock(return_value=10)
    
    result = mock_chunker.split("Some input text.")
    
    assert len(result) > 0
    mock_sent_tokenize.assert_called_once_with("Some input text.")

def test_load_model_failure():
    with patch('shared.text_processing.semantic_chunker.SentenceTransformer', side_effect=Exception("Model load failed")), \
         patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        
        with pytest.raises(Exception, match="Model load failed"):
            SemanticChunker()

def test_load_tokenizer_failure():
    with patch('shared.text_processing.semantic_chunker.SentenceTransformer') as mock_st, \
         patch('shared.text_processing.semantic_chunker.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer load failed")), \
         patch('shared.text_processing.semantic_chunker.get_torch_device', return_value=TEST_DEVICE):
        
        with pytest.raises(Exception, match="Tokenizer load failed"):
            SemanticChunker()