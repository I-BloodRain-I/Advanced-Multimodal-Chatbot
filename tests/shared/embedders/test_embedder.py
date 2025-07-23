import pytest
import torch
from unittest.mock import Mock, patch
from sentence_transformers import SentenceTransformer
import numpy as np

from shared.embedders import Embedder

TEST_MODEL_NAME = 'test-model'
TEST_DEVICE = 'cpu'

@pytest.fixture(autouse=True)
def reset_singleton():
    Embedder._instance = None
    yield
    Embedder._instance = None

@pytest.fixture()
def mock_sentence_transformer():
    with patch('shared.embedders.embedder.SentenceTransformer') as mock_st:
        mock_model = Mock(spec=SentenceTransformer)
        mock_model.eval.return_value.requires_grad_ = Mock(return_value=mock_model)
        mock_st.return_value = mock_model
        yield mock_st, mock_model

def test_singleton_behavior():
    instance1 = Embedder()
    instance2 = Embedder()
    assert instance1 is instance2
    
    instance3 = Embedder(model_name='model1', device_name='cpu')
    instance4 = Embedder(model_name='model2', device_name='cuda')
    assert instance3 is instance4
    assert instance3.model is instance4.model

def test_singleton_thread_safety_and_multiple_instances(mock_sentence_transformer):
    instances = []
    for i in range(10):
        instances.append(Embedder(model_name=f'model_{i}'))
    
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance

def test_initialization_and_model_setup(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    
    embedder = Embedder(model_name=TEST_MODEL_NAME, device_name=TEST_DEVICE)
    
    mock_st.assert_called_once_with(TEST_MODEL_NAME, device=torch.device(TEST_DEVICE))
    mock_model.eval.assert_called_once()
    mock_model.eval().requires_grad_.assert_called_once_with(False)
    assert embedder._initialized is True
    assert hasattr(embedder, '_initialized')

def test_initialization_idempotency(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    
    embedder1 = Embedder()
    original_model = embedder1.model
    
    embedder2 = Embedder(model_name='different-model')
    
    assert embedder1 is embedder2
    assert embedder2.model is original_model
    assert mock_st.call_count == 1
    assert embedder2._initialized is True

def test_load_model_success_and_eval_mode(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    
    embedder = Embedder()
    result = embedder._load_model(TEST_MODEL_NAME)
    
    assert result == mock_model
    mock_model.eval.assert_called()
    mock_model.eval().requires_grad_.assert_called_with(False)

@patch('shared.embedders.embedder.logger')
def test_model_initialization_failures(mock_logger, mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_st.side_effect = Exception("Model load failed")
    
    with pytest.raises(Exception, match="Model load failed"):
        Embedder()
        
    mock_logger.error.assert_called_once()
    
    mock_st.side_effect = None
    embedder2 = Embedder()
    assert embedder2._initialized is True

def test_extract_embeddings_basic_inputs(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_model.encode.return_value = mock_embeddings
    
    embedder = Embedder(device_name=TEST_DEVICE)
    
    with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
        result1 = embedder.extract_embeddings("test prompt")
        result2 = embedder.extract_embeddings("")
        
    mock_model.encode.assert_any_call(
        "test prompt",
        batch_size=32,
        convert_to_tensor=False,
        device=torch.device(TEST_DEVICE),
        show_progress_bar=False
    )
    mock_model.encode.assert_any_call(
        "",
        batch_size=32,
        convert_to_tensor=False,
        device=torch.device(TEST_DEVICE),
        show_progress_bar=False
    )
    assert result1 is mock_embeddings
    assert result2 is mock_embeddings

def test_extract_embeddings_list_inputs(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_model.encode.return_value = mock_embeddings
    
    embedder = Embedder(device_name=TEST_DEVICE)
    
    prompts = ["prompt1", "prompt2"]
    empty_list = []
    mixed_prompts = ["valid_string", "", "another_string"]
    
    with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
        result1 = embedder.extract_embeddings(prompts)
        result2 = embedder.extract_embeddings(empty_list)
        result3 = embedder.extract_embeddings(mixed_prompts)
    
    for call_args in [prompts, empty_list, mixed_prompts]:
        mock_model.encode.assert_any_call(
            call_args,
            batch_size=32,
            convert_to_tensor=False,
            device=torch.device(TEST_DEVICE),
            show_progress_bar=False
        )
    assert result1 is mock_embeddings

def test_extract_embeddings_custom_parameters(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_embeddings = torch.tensor([[0.1, 0.2]])
    mock_model.encode.return_value = mock_embeddings
    
    embedder = Embedder(device_name=TEST_DEVICE)
    with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
        result = embedder.extract_embeddings(
            "test",
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True
        )
    
    mock_model.encode.assert_called_once_with(
        "test",
        batch_size=64,
        convert_to_tensor=True,
        device=torch.device(TEST_DEVICE),
        show_progress_bar=True
    )
    assert result is mock_embeddings


def test_extract_embeddings_large_scale_and_special_chars(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_embeddings = np.random.rand(1000, 768)
    mock_model.encode.return_value = mock_embeddings
    
    embedder = Embedder(device_name=TEST_DEVICE)
    
    large_prompts = [f"prompt_{i}" for i in range(1000)]
    special_prompts = [
        "Hello 🌍",
        "Привет мир", 
        "Special chars: @#$%^&*()",
        "New\nLine\tTab",
        ""
    ]
    
    with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
        result1 = embedder.extract_embeddings(large_prompts, batch_size=1024)
        result2 = embedder.extract_embeddings(special_prompts)
    
    mock_model.encode.assert_any_call(
        large_prompts,
        batch_size=1024,
        convert_to_tensor=False,
        device=torch.device(TEST_DEVICE),
        show_progress_bar=False
    )
    mock_model.encode.assert_any_call(
        special_prompts,
        batch_size=32,
        convert_to_tensor=False,
        device=torch.device(TEST_DEVICE),
        show_progress_bar=False
    )
    assert result1.shape == (1000, 768)

def test_extract_embeddings_multiple_calls_and_errors(mock_sentence_transformer):
    mock_st, mock_model = mock_sentence_transformer
    mock_embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_model.encode.return_value = mock_embeddings
    
    embedder = Embedder(device_name=TEST_DEVICE)
    
    with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
        result1 = embedder.extract_embeddings("prompt1")
        result2 = embedder.extract_embeddings("prompt2")
        result3 = embedder.extract_embeddings(["prompt3", "prompt4"])
    
    assert mock_model.encode.call_count == 3
    assert np.array_equal(result1, mock_embeddings)
    assert np.array_equal(result2, mock_embeddings)
    assert np.array_equal(result3, mock_embeddings)
    
    mock_model.encode.side_effect = RuntimeError("Encoding failed")
    with pytest.raises(RuntimeError, match="Encoding failed"):
        with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
            embedder.extract_embeddings("test prompt")

def test_extract_embeddings_not_implemented_error(mock_sentence_transformer):
    embedder = Embedder()
    embedder.model = "not a SentenceTransformer"
    
    with pytest.raises(NotImplementedError, match="Code for the model type"):
        with patch('shared.embedders.embedder.SentenceTransformer', SentenceTransformer):
            embedder.extract_embeddings("test")
