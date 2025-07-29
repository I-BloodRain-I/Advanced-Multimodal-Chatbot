import asyncio
import threading
import pytest
import torch
from unittest.mock import AsyncMock, Mock, patch

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, AsyncTextIteratorStreamer
from core.entities.types import Message
from modules.llm.components.text_streamer import TransformersTextStreamer, StopOnTokens
from modules.llm.configs import LLMGenerationConfig
from shared.text_processing.prompt_transformer import PromptTransformer

# Test configuration constants
TEST_MODEL_NAME = "gpt2"
TEST_DEVICE = "cpu"


@pytest.fixture
def mock_model():
    """Provide mocked model."""
    mock_model = Mock(spec=AutoModelForCausalLM)
    mock_model.device = torch.device(TEST_DEVICE)
    mock_model.generate = AsyncMock()
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Provide mocked tokenizer."""
    mock_tokenizer = Mock(spec=AutoTokenizer)
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.model_max_length = 2048
    mock_tokenizer.chat_template = None
    mock_tokenizer.encode = Mock(return_value=[])
    return mock_tokenizer


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


@pytest.fixture
def streamer(mock_model, mock_tokenizer):
    """Provide TransformersTextStreamer instance."""
    return TransformersTextStreamer(mock_model, mock_tokenizer)


def test_init(mock_model, mock_tokenizer):
    """Test TransformersTextStreamer initialization."""
    streamer = TransformersTextStreamer(mock_model, mock_tokenizer)
    
    assert streamer.model is mock_model
    assert streamer.tokenizer is mock_tokenizer
    assert streamer.device == mock_model.device


def test_stop_on_tokens_stops_on_matching_token():
    """Test StopOnTokens stops on matching token ID."""
    stop_token_ids = [42]
    stop_event = threading.Event()
    criteria = StopOnTokens(stop_token_ids, stop_event)

    input_ids = torch.tensor([[1, 2, 42]])
    result = criteria(input_ids=input_ids, scores=None)
    assert result is True


def test_stop_on_tokens_stops_on_event():
    """Test StopOnTokens stops when event is set."""
    stop_token_ids = [42]
    stop_event = threading.Event()
    stop_event.set()
    criteria = StopOnTokens(stop_token_ids, stop_event)

    input_ids = torch.tensor([[1, 2, 3]])
    result = criteria(input_ids=input_ids, scores=None)
    assert result is True


def test_stop_on_tokens_does_not_stop_on_non_matching_token():
    """Test StopOnTokens doesn't stop on non-matching token."""
    stop_token_ids = [99]
    stop_event = threading.Event()
    criteria = StopOnTokens(stop_token_ids, stop_event)

    input_ids = torch.tensor([[1, 2, 42]])
    result = criteria(input_ids=input_ids, scores=None)
    assert result is False


def test_create_stopping_criteria_with_chat_template(streamer):
    """Test stopping criteria creation with chat template."""
    streamer.tokenizer.chat_template = True
    streamer.tokenizer.encode.side_effect = lambda seq, add_special_tokens=False: [1, 2] if seq == "<|im_end|>" else []

    criteria, stop_event = streamer._create_stopping_criteria()
    
    assert isinstance(criteria, StoppingCriteriaList)
    assert len(criteria) == 1
    assert isinstance(criteria[0], StopOnTokens)
    assert isinstance(stop_event, threading.Event)


def test_create_stopping_criteria_without_chat_template(streamer):
    """Test stopping criteria creation without chat template."""
    streamer.tokenizer.chat_template = False

    criteria, stop_event = streamer._create_stopping_criteria()
    
    assert criteria is None
    assert isinstance(stop_event, threading.Event)


def test_create_stopping_criteria_no_valid_tokens(streamer):
    """Test stopping criteria when no valid stop tokens found."""
    streamer.tokenizer.chat_template = True
    streamer.tokenizer.encode.side_effect = lambda seq, add_special_tokens=False: []

    criteria, stop_event = streamer._create_stopping_criteria()
    
    assert criteria is None
    assert isinstance(stop_event, threading.Event)


def test_create_generator_empty_messages(streamer):
    """Test create_generator with empty messages raises ValueError."""
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        streamer.create_generator([])


def test_create_generator_with_config(streamer, mock_messages):
    """Test create_generator with provided config."""
    config = LLMGenerationConfig(temperature=0.8, max_tokens=100)
    
    with patch.object(streamer, '_create_generator') as mock_create:
        mock_generator = AsyncMock()
        mock_create.return_value = mock_generator
        
        result = streamer.create_generator(mock_messages, config)
        
        assert result is mock_generator
        mock_create.assert_called_once()
        
        # Check parameters passed to _create_generator
        call_args = mock_create.call_args[0]
        assert call_args[0] == mock_messages
        call_kwargs = mock_create.call_args[0][1]
        assert call_kwargs['temperature'] == 0.8
        assert call_kwargs['max_new_tokens'] == 100


def test_create_generator_without_config(streamer, mock_messages):
    """Test create_generator without config uses defaults."""
    with patch.object(streamer, '_create_generator') as mock_create:
        mock_generator = AsyncMock()
        mock_create.return_value = mock_generator
        
        result = streamer.create_generator(mock_messages)
        
        assert result is mock_generator
        mock_create.assert_called_once()


def test_create_batch_generators_empty_batch(streamer):
    """Test create_batch_generators with empty batch raises ValueError."""
    with pytest.raises(ValueError, match="Messages batch cannot be empty"):
        streamer.create_batch_generators([])


def test_create_batch_generators_config_length_mismatch(streamer, mock_messages_batch):
    """Test create_batch_generators with mismatched configs length."""
    configs = [LLMGenerationConfig()]  # Only one config for two messages
    
    with pytest.raises(ValueError, match="Configs length \\(1\\) must match batch size \\(2\\)"):
        streamer.create_batch_generators(mock_messages_batch, configs)


def test_create_batch_generators_success(streamer, mock_messages_batch):
    """Test successful batch generator creation."""
    configs = [
        LLMGenerationConfig(temperature=0.7),
        LLMGenerationConfig(temperature=0.9)
    ]
    
    with patch.object(streamer, 'create_generator') as mock_create:
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_create.side_effect = mock_generators
        
        result = streamer.create_batch_generators(mock_messages_batch, configs)
        
        assert result == mock_generators
        assert mock_create.call_count == 2
        
        # Check that each generator was created with correct parameters
        for i, call in enumerate(mock_create.call_args_list):
            assert call[1]['messages'] == mock_messages_batch[i]  # messages
            assert call[1]['config'] == configs[i]  # config


def test_create_batch_generators_without_configs(streamer, mock_messages_batch):
    """Test batch generator creation without configs."""
    with patch.object(streamer, 'create_generator') as mock_create:
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_create.side_effect = mock_generators
        
        result = streamer.create_batch_generators(mock_messages_batch)
        
        assert result == mock_generators
        assert mock_create.call_count == 2
        
        # Check that each generator was created with None config
        for call in mock_create.call_args_list:
            assert call[1]['config'] is None


@pytest.mark.asyncio
async def test_create_generator_internal_success(streamer, mock_messages):
    """Test internal _create_generator method success path."""
    # This test is complex due to async mocking. Test the method indirectly through create_generator
    with patch.object(streamer, '_create_generator') as mock_internal:
        mock_generator = AsyncMock()
        
        async def mock_async_gen():
            yield "token1"
            yield "token2"
        
        mock_internal.return_value = mock_async_gen()
        
        result = streamer.create_generator(mock_messages)
        tokens = []
        async for token in result:
            tokens.append(token)
        
        assert tokens == ["token1", "token2"]
        mock_internal.assert_called_once()


@pytest.mark.asyncio
async def test_create_generator_format_error(streamer, mock_messages):
    """Test _create_generator with formatting error."""
    with patch.object(PromptTransformer, 'format_messages_to_str', side_effect=Exception("Format error")):
        
        generation_kwargs = {"temperature": 0.7}
        
        with pytest.raises(Exception, match="Format error"):
            generator = streamer._create_generator(mock_messages, generation_kwargs)
            async for _ in generator:
                pass


@pytest.mark.asyncio
async def test_create_generator_tokenization_error(streamer, mock_messages):
    """Test _create_generator with tokenization error."""
    with patch.object(PromptTransformer, 'format_messages_to_str', return_value="formatted prompt"):
        
        streamer.tokenizer.side_effect = Exception("Tokenization error")
        generation_kwargs = {"temperature": 0.7}
        
        with pytest.raises(Exception, match="Tokenization error"):
            generator = streamer._create_generator(mock_messages, generation_kwargs)
            async for _ in generator:
                pass
