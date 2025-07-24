import threading
import pytest
import torch
from unittest.mock import AsyncMock, Mock, patch

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from core.entities.types import Message
from modules.llm.text_streamer import TextStreamer, StopOnTokens
from shared.text_processing.prompt_transformer import PromptTransformer

# Test configuration constants
TEST_MODEL_NAME = "gpt2"  # GPT-2 tiny model
TEST_MAX_NEW_TOKENS = 50
TEST_TEMPERATURE = 0.7
TEST_DEVICE = "cpu"  # Use CPU for consistent testing


@pytest.fixture
def mock_llm_components():
    """Provide mocked LLM components."""
    mock_model = Mock()
    mock_model.parameters = Mock(return_value=iter([torch.tensor([42], device=TEST_DEVICE)]))
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    
    return {
        'model': mock_model,
        'tokenizer': mock_tokenizer
    }


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


def test_stop_on_tokens_stops_on_matching_token():
    stop_token_ids = [42]
    criteria = StopOnTokens(stop_token_ids, None)

    input_ids = torch.tensor([[1, 2, 42]])
    result = criteria(input_ids=input_ids, scores=None)
    assert result is True


def test_stop_on_tokens_does_not_stop_on_non_matching_token():
    stop_token_ids = [99]
    criteria = StopOnTokens(stop_token_ids, None)

    input_ids = torch.tensor([[1, 2, 42]])
    result = criteria(input_ids=input_ids, scores=None)
    assert result is False


def test_create_stopping_criteria_returns_criteria(mock_llm_components):
    tokenizer = mock_llm_components['tokenizer']
    tokenizer.chat_template = True
    tokenizer.encode.side_effect = lambda seq, add_special_tokens=False: [1, 2] if seq == "<|im_end|>" else []

    streamer = TextStreamer(
        model=mock_llm_components['model'],
        tokenizer=tokenizer,
        max_new_tokens=TEST_MAX_NEW_TOKENS,
        temperature=TEST_TEMPERATURE
    )

    criteria, stop_event = streamer._create_stopping_criteria()
    assert isinstance(criteria, StoppingCriteriaList)
    assert isinstance(criteria[0], StopOnTokens)
    assert isinstance(stop_event, threading.Event)


def test_create_stopping_criteria_returns_none_without_chat_template(mock_llm_components):
    tokenizer = mock_llm_components['tokenizer']
    tokenizer.chat_template = False

    streamer = TextStreamer(
        model=mock_llm_components['model'],
        tokenizer=tokenizer,
        max_new_tokens=TEST_MAX_NEW_TOKENS,
        temperature=TEST_TEMPERATURE
    )

    assert streamer._create_stopping_criteria()[0] is None


@pytest.mark.asyncio
async def test_generate_stream_yields_tokens(mock_messages):
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

    streamer = TextStreamer(model=model, tokenizer=tokenizer, max_new_tokens=3)

    with patch.object(PromptTransformer, "format_messages_to_str", return_value="Hello world"):
        gen = streamer.generate_stream(mock_messages)
        tokens = [token async for token in gen]
        
        assert all([isinstance(token, str) for token in tokens])
        assert len(tokens) <= 3


@pytest.mark.asyncio
async def test_generate_stream_batch_creates_generators(mock_llm_components, mock_messages_batch):
    tokenizer = mock_llm_components['tokenizer']
    model = mock_llm_components['model']

    streamer = TextStreamer(model=model, tokenizer=tokenizer)

    with patch.object(streamer, "_create_stream_generator", return_value=AsyncMock()) as mock_gen:
        generators = streamer.generate_stream_batch(mock_messages_batch)
        assert len(generators) == len(mock_messages_batch)
        assert all(call == mock_gen.return_value for call in generators)


def test_generate_stream_batch_empty_input(mock_llm_components):
    streamer = TextStreamer(
        model=mock_llm_components['model'],
        tokenizer=mock_llm_components['tokenizer']
    )
    assert streamer.generate_stream_batch([]) == []
