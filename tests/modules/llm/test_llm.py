import json
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from core.entities.types import Message
from modules.llm import LLM

# Test configuration constants
TEST_MODEL_NAME = "gpt2"  # GPT-2 tiny model
TEST_DTYPE = "fp32"  # Safe dtype for testing
TEST_MAX_NEW_TOKENS = 50
TEST_TEMPERATURE = 0.7
TEST_DEVICE = "cpu"  # Use CPU for consistent testing


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    LLM._instance = None
    yield
    LLM._instance = None


@pytest.fixture
def mock_messages():
    """Provide test messages."""
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]


@pytest.fixture
def mock_messages_batch():
    """Provide test messages batch."""
    return [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi there!")]
    ]


def test_singleton_pattern():
    """Test that LLM follows singleton pattern."""
    with patch.object(LLM, '_load_model'), \
         patch.object(LLM, '_load_tokenizer'), \
         patch.object(LLM, '_save_model_and_tokenizer'), \
         patch('modules.llm.TextStreamer'), \
         patch('shared.config.Config.get', return_value='models'):
        
        llm1 = LLM(
            model_name=TEST_MODEL_NAME,
            dtype=TEST_DTYPE,
            max_new_tokens=TEST_MAX_NEW_TOKENS,
            temperature=TEST_TEMPERATURE,
            device_name=TEST_DEVICE
        )
        llm2 = LLM()
        
        assert llm1 is llm2
        assert id(llm1) == id(llm2)


def test_singleton_initialization_once():
    """Test that singleton is initialized only once."""
    with patch.object(LLM, '_load_model') as mock_load_model, \
         patch.object(LLM, '_load_tokenizer') as mock_load_tokenizer, \
         patch.object(LLM, '_save_model_and_tokenizer') as mock_save, \
         patch('modules.llm.TextStreamer'), \
         patch('shared.config.Config.get', return_value='models'):
        
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([torch.tensor([42], device=TEST_DEVICE)]))
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        
        llm1 = LLM(model_name=TEST_MODEL_NAME, dtype=TEST_DTYPE)
        llm2 = LLM()
        
        assert mock_load_model.call_count == 1
        assert mock_load_tokenizer.call_count == 1
        assert mock_save.call_count == 1


@patch('shared.utils.get_torch_device')
@patch('shared.config.Config.get')
def test_init_success(mock_config_get, mock_get_device):
    """Test successful LLM initialization."""
    mock_config_get.return_value = "/tmp/models"
    mock_get_device.return_value = torch.device("cpu")
    
    with patch.object(LLM, '_load_model') as mock_load_model, \
         patch.object(LLM, '_load_tokenizer') as mock_load_tokenizer, \
         patch.object(LLM, '_save_model_and_tokenizer') as mock_save, \
         patch('modules.llm.TextStreamer') as mock_streamer:
        
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([torch.tensor([42], device=TEST_DEVICE)]))
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        
        llm = LLM(
            model_name=TEST_MODEL_NAME,
            dtype=TEST_DTYPE,
            max_new_tokens=TEST_MAX_NEW_TOKENS,
            temperature=TEST_TEMPERATURE,
            device_name=TEST_DEVICE
        )
        
        assert llm.max_new_tokens == TEST_MAX_NEW_TOKENS
        assert llm.temperature == TEST_TEMPERATURE
        assert llm.model is mock_model
        assert llm.tokenizer is mock_tokenizer
        assert llm._initialized is True


@pytest.mark.parametrize("dtype,exists,expect_exception,expected_key", [
    ("fp32", False, False, "torch_dtype"),
    ("int8", False, False, "quantization_config"),
    ("invalid_dtype", False, True, None),
    ("fp32", True, False, "local_files_only"),
])
def test_load_model_variants(dtype, exists, expect_exception, expected_key):
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch.object(Path, 'exists', return_value=exists):

        llm = LLM.__new__(LLM)
        llm.device = torch.device("cpu")
        llm._models_dir = Path("/tmp/models")

        if expect_exception:
            with pytest.raises(Exception, match="Invalid dtype in config"):
                llm._load_model(TEST_MODEL_NAME, dtype)
        else:
            mock_model = Mock()
            mock_from_pretrained.return_value = mock_model
            result = llm._load_model(TEST_MODEL_NAME, dtype)

            call_args = mock_from_pretrained.call_args[1]
            assert expected_key in call_args


@pytest.mark.parametrize("pad_token,eos_token,exists,should_fail", [
    ("<pad>", "<eos>", False, False),
    (None, "<eos>", False, False),
    ("<pad>", "<eos>", True, False),
    ("<pad>", "<eos>", False, True), 
])
def test_load_tokenizer_variants(pad_token, eos_token, exists, should_fail):
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_from_pretrained, \
         patch.object(Path, 'exists', return_value=exists):
        
        if should_fail:
            mock_from_pretrained.side_effect = Exception("Tokenizer error")
        else:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = pad_token
            mock_tokenizer.eos_token = eos_token
            mock_from_pretrained.return_value = mock_tokenizer

        llm = LLM.__new__(LLM)
        llm._models_dir = Path("/tmp/models")

        if should_fail:
            with pytest.raises(Exception, match="Tokenizer error"):
                llm._load_tokenizer(TEST_MODEL_NAME)
        else:
            tokenizer = llm._load_tokenizer(TEST_MODEL_NAME)
            assert tokenizer.eos_token == eos_token
            if pad_token is None:
                assert tokenizer.pad_token == eos_token


@pytest.mark.parametrize("exists,should_save", [(False, True), (True, False)])
def test_save_model_and_tokenizer_behavior(exists, should_save):
    with patch.object(Path, 'exists', return_value=exists):
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        llm = LLM.__new__(LLM)
        llm.model = mock_model
        llm.tokenizer = mock_tokenizer
        llm._models_dir = Path("/tmp/models")

        llm._save_model_and_tokenizer(TEST_MODEL_NAME)

        if should_save:
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
        else:
            mock_model.save_pretrained.assert_not_called()
            mock_tokenizer.save_pretrained.assert_not_called()


def test_save_model_removes_quantization_config():
    """Test that quantization_config is removed from config.json."""
    config_data = {
        "model_type": "gpt2",
        "quantization_config": {"some": "config"},
        "other": "data"
    }
    
    m = mock_open(read_data=json.dumps(config_data))
    
    with patch.object(Path, 'exists', side_effect=[False, True]), \
         patch('builtins.open', m):
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        llm = LLM.__new__(LLM)
        llm.model = mock_model
        llm.tokenizer = mock_tokenizer
        llm._models_dir = Path("/tmp/models")
        
        llm._save_model_and_tokenizer(TEST_MODEL_NAME)
        
        written_str = ''.join(call.args[0] for call in m().write.call_args_list)
        if written_str:
            written_data = json.loads(written_str)
            assert "quantization_config" not in written_data
            assert written_data["model_type"] == "gpt2"
            assert written_data["other"] == "data"


def test_save_model_and_tokenizer_failure():
    """Test saving model and tokenizer with failure."""
    with patch.object(Path, 'exists') as mock_exists:
        
        mock_exists.return_value = False
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model.save_pretrained.side_effect = Exception("Save error")
        
        llm = LLM.__new__(LLM)
        llm.model = mock_model
        llm.tokenizer = mock_tokenizer
        llm._models_dir = Path("/tmp/models")
        
        llm._save_model_and_tokenizer(TEST_MODEL_NAME)


@patch('shared.text_processing.prompt_transformer.PromptTransformer.format_messages_to_str')
def test_generate_success(mock_format_messages, mock_messages):
    """Test successful text generation."""
    mock_format_messages.return_value = "formatted prompt"
    
    mock_tokenizer = Mock()
    mock_model = Mock()
    
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "Generated response"
    
    mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
    
    llm = LLM.__new__(LLM)
    llm.device = torch.device("cpu")
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.temperature = TEST_TEMPERATURE
    llm.max_new_tokens = TEST_MAX_NEW_TOKENS
    
    result = llm.generate(mock_messages)
    
    assert result == "Generated response"
    mock_format_messages.assert_called_once_with(mock_messages, mock_tokenizer)
    mock_model.generate.assert_called_once()


@patch('shared.text_processing.prompt_transformer.PromptTransformer.format_messages_to_str')
def test_generate_return_tokens(mock_format_messages, mock_messages):
    """Test generation with return_tokens=True."""
    mock_format_messages.return_value = "formatted prompt"
    
    mock_tokenizer = Mock()
    mock_model = Mock()
    
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    
    generated_tokens = torch.tensor([4, 5])
    mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
    
    llm = LLM.__new__(LLM)
    llm.device = torch.device("cpu")
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.temperature = TEST_TEMPERATURE
    llm.max_new_tokens = TEST_MAX_NEW_TOKENS
    
    result = llm.generate(mock_messages, return_tokens=True)
    
    assert torch.equal(result, generated_tokens)


@pytest.mark.parametrize("stage,error", [
    ("format", "Format error"),
    ("tokenization", "Tokenization error"),
    ("generation", "Generation error")
])
@patch('shared.text_processing.prompt_transformer.PromptTransformer.format_messages_to_str')
def test_generate_failures(mock_format_messages, stage, error, mock_messages):
    if stage == "format":
        mock_format_messages.side_effect = Exception(error)
    else:
        mock_format_messages.return_value = "prompt"

    mock_tokenizer = Mock()
    mock_model = Mock()

    if stage == "tokenization":
        mock_tokenizer.side_effect = Exception(error)
    elif stage == "generation":
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_model.generate.side_effect = Exception(error)

    llm = LLM.__new__(LLM)
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.device = torch.device("cpu")
    llm.temperature = TEST_TEMPERATURE
    llm.max_new_tokens = TEST_MAX_NEW_TOKENS

    with pytest.raises(Exception, match=error):
        llm.generate(mock_messages)


@patch('shared.text_processing.prompt_transformer.PromptTransformer.format_messages_to_str')
def test_generate_with_different_parameters(mock_format_messages, mock_messages):
    """Test generation with different parameters."""
    mock_format_messages.return_value = "formatted prompt"
    
    mock_tokenizer = Mock()
    mock_model = Mock()
    
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "Generated response"
    mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
    
    llm = LLM.__new__(LLM)
    llm.device = torch.device("cpu")
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.temperature = 0.9
    llm.max_new_tokens = 100
    
    llm.generate(mock_messages)
    
    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs['temperature'] == 0.9
    assert call_kwargs['max_new_tokens'] == 100
    assert call_kwargs['do_sample'] is True


def test_generate_batch_success(mock_messages_batch):
    """Test successful batch generation."""
    mock_tokenizer = Mock()
    mock_tokenizer.batch_decode.return_value = ["Response 1", "Response 2"]
    
    llm = LLM.__new__(LLM)
    llm.tokenizer = mock_tokenizer
    
    with patch.object(llm, 'generate') as mock_generate:
        mock_generate.side_effect = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6])
        ]
        
        result = llm.generate_batch(mock_messages_batch)
        
        assert result == ["Response 1", "Response 2"]
        assert mock_generate.call_count == 2
        
        for call in mock_generate.call_args_list:
            assert call[1]['return_tokens'] is True


def test_generate_batch_empty():
    """Test batch generation with empty batch."""
    mock_tokenizer = Mock()
    mock_tokenizer.batch_decode.return_value = []
    
    llm = LLM.__new__(LLM)
    llm.tokenizer = mock_tokenizer
    
    with patch.object(llm, 'generate') as mock_generate:
        mock_generate.return_value = []
        
        result = llm.generate_batch([])
        
        assert result == []
        mock_generate.assert_not_called()


def test_generate_batch_single_item():
    """Test batch generation with single item."""
    mock_tokenizer = Mock()
    mock_tokenizer.batch_decode.return_value = ["Single response"]
    
    single_batch = [[{"role": "user", "content": "Hello"}]]
    
    llm = LLM.__new__(LLM)
    llm.tokenizer = mock_tokenizer
    
    with patch.object(llm, 'generate') as mock_generate:
        mock_generate.return_value = torch.tensor([1, 2, 3])
        
        result = llm.generate_batch(single_batch)
        
        assert result == ["Single response"]
        assert mock_generate.call_count == 1


def test_generate_stream(mock_messages, mock_messages_batch):
    """Test streaming generation."""
    mock_streamer = Mock()
    mock_generator = Mock()
    mock_streamer.generate_stream.return_value = mock_generator
    mock_streamer.generate_stream_batch.return_value = [mock_generator, mock_generator]
    
    llm = LLM.__new__(LLM)
    llm._streamer = mock_streamer
    
    single_result = llm.generate_stream(mock_messages)
    batch_result = llm.generate_stream_batch(mock_messages_batch)
    
    assert single_result is mock_generator
    mock_streamer.generate_stream.assert_called_once_with(mock_messages)

    assert batch_result == [mock_generator, mock_generator]
    mock_streamer.generate_stream_batch.assert_called_once_with(mock_messages_batch)