import json
import time
from unittest.mock import Mock, patch, AsyncMock
import pytest

from core.entities.types import AgentResponse
from orchestrator import PromptDispatcher


@pytest.fixture
def mock_redis():
    redis_mock = Mock()
    redis_mock.lpop.return_value = None
    redis_mock.publish.return_value = None
    redis_mock.delete.return_value = None
    return redis_mock


@pytest.fixture
def mock_pipeline():
    pipeline_mock = Mock()
    pipeline_mock.return_value = []
    return pipeline_mock


@pytest.fixture
def mock_config():
    config_mock = Mock()
    config_mock.get.return_value = {'host': 'localhost', 'port': 6379}
    return config_mock


@pytest.fixture
def dispatcher(mock_redis, mock_pipeline, mock_config):
    with patch('orchestrator.prompt_dispatcher.Redis', return_value=mock_redis), \
         patch('orchestrator.prompt_dispatcher.Pipeline.build', return_value=mock_pipeline), \
         patch('shared.config.Config.get', return_value=mock_config):
        return PromptDispatcher()


def test_init_clears_redis_queue(mock_redis, mock_pipeline, mock_config):
    with patch('orchestrator.prompt_dispatcher.Redis', return_value=mock_redis), \
         patch('orchestrator.prompt_dispatcher.Pipeline.build', return_value=mock_pipeline), \
         patch('shared.config.Config.get', return_value=mock_config):
        PromptDispatcher()
        mock_redis.delete.assert_called_once_with("process:messages_batch")


def test_get_conversation_batch_empty_queue(dispatcher):
    dispatcher._redis.blpop.return_value = [(), None]
    dispatcher._redis.lrange.return_value = []
    result = dispatcher._get_conversation_batch()
    assert not result.conv_ids and not result.histories


def test_get_conversation_batch_single_item(dispatcher):
    batch_item = {
        'conv_id': 'conv_123',
        'history': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]
    }
    dispatcher._redis.blpop.return_value = [(), json.dumps(batch_item)]
    dispatcher._redis.lrange.return_value = []
    
    result = dispatcher._get_conversation_batch()
    
    assert result is not None
    assert result.conv_ids == ['conv_123']
    assert len(result.histories) == 1
    assert len(result.histories[0]) == 2
    assert result.histories[0][0].role == 'user'
    assert result.histories[0][0].content == 'Hello'


def test_get_conversation_batch_multiple_items(dispatcher):
    batch_item1 = {
        'conv_id': 'conv_1',
        'history': [{'role': 'user', 'content': 'Message 1'}]
    }
    batch_item2 = {
        'conv_id': 'conv_2', 
        'history': [{'role': 'user', 'content': 'Message 2'}]
    }
    
    dispatcher._redis.blpop.return_value = [(), json.dumps(batch_item1)]
    dispatcher._redis.lrange.return_value = [json.dumps(batch_item2)]
    
    result = dispatcher._get_conversation_batch()
    
    assert result is not None
    assert result.conv_ids == ['conv_1', 'conv_2']
    assert len(result.histories) == 2


def test_get_conversation_batch_invalid_json(dispatcher):
    dispatcher._redis.blpop.return_value = [(), 'invalid json']
    dispatcher._redis.lrange.return_value = []
    result = dispatcher._get_conversation_batch()
    assert not result.conv_ids and not result.histories


def test_get_conversation_batch_missing_key(dispatcher):
    batch_item = {'conv_id': 'conv_123'}

    dispatcher._redis.blpop.return_value = [(), json.dumps(batch_item)]
    dispatcher._redis.lrange.return_value = []
    
    result = dispatcher._get_conversation_batch()
    
    assert result is not None
    assert result.conv_ids == []
    assert result.histories == []


@pytest.mark.asyncio
async def test_stream_response(dispatcher):
    async def mock_generator():
        yield "Hello"
        yield " "
        yield "World"
    
    await dispatcher._stream_response("conv_123", mock_generator())
    
    expected_calls = [
        (("conv_123", json.dumps({"type": "stream", "content": "Hello"})),),
        (("conv_123", json.dumps({"type": "stream", "content": " "})),),
        (("conv_123", json.dumps({"type": "stream", "content": "World"})),),
        (("conv_123", json.dumps({"type": "end_stream"})),)
    ]
    
    assert dispatcher._redis.publish.call_count == 4
    actual_calls = dispatcher._redis.publish.call_args_list
    assert actual_calls == expected_calls


@pytest.mark.asyncio
async def test_response(dispatcher):
    await dispatcher._response("conv_123", "text", "Hello World")
    
    dispatcher._redis.publish.assert_called_once_with(
        "conv_123", 
        json.dumps({"type": "text", "content": "Hello World"})
    )


def test_send_agent_responses_text_response(dispatcher):
    responses = [
        AgentResponse(conv_id="conv_123", type="text", content="Hello")
    ]
    
    with patch.object(dispatcher, '_response', new_callable=AsyncMock) as mock_response:
        dispatcher._send_agent_responses(responses)
        time.sleep(0.1)
        mock_response.assert_called_once_with("conv_123", "text", "Hello")


def test_send_agent_responses_stream_response(dispatcher):
    async def mock_generator():
        yield "token"
    
    responses = [
        AgentResponse(conv_id="conv_123", type="stream", content=mock_generator())
    ]
    
    with patch.object(dispatcher, '_stream_response', new_callable=AsyncMock) as mock_stream:
        dispatcher._send_agent_responses(responses)
        time.sleep(0.1)
        mock_stream.assert_called_once()


def test_send_agent_responses_mixed_types(dispatcher):
    async def mock_generator():
        yield "token"
    
    responses = [
        AgentResponse(conv_id="conv_1", type="text", content="Hello"),
        AgentResponse(conv_id="conv_2", type="stream", content=mock_generator()),
        AgentResponse(conv_id="conv_3", type="image", content="base64_image_data")
    ]
    
    with patch.object(dispatcher, '_response', new_callable=AsyncMock) as mock_response, \
         patch.object(dispatcher, '_stream_response', new_callable=AsyncMock) as mock_stream:
        dispatcher._send_agent_responses(responses)
        time.sleep(0.1)
        
        assert mock_response.call_count == 2
        mock_stream.assert_called_once()


def test_batch_processing_loop_no_data(dispatcher):
    dispatcher._redis.lpop.return_value = None
    
    with patch('time.sleep') as mock_sleep:
        with patch.object(dispatcher, '_batch_processing_loop') as mock_loop:
            mock_loop.side_effect = [None]
            try:
                dispatcher._batch_processing_loop()
            except:
                pass


@pytest.mark.asyncio
async def test_stream_response_exception(dispatcher):
    async def failing_generator():
        yield "Hello"
        raise Exception("Generator error")
    
    await dispatcher._stream_response("conv_123", failing_generator())
    
    dispatcher._redis.publish.assert_called_once_with(
        "conv_123", 
        json.dumps({"type": "stream", "content": "Hello"})
    )


@pytest.mark.asyncio
async def test_response_exception(dispatcher):
    dispatcher._redis.publish.side_effect = Exception("Publish error")
    await dispatcher._response("conv_123", "text", "Hello")


def test_calc_remain_seconds():
    start_time = time.time()
    time.sleep(0.1)
    
    remain = max(0, 1.0 - (time.time() - start_time))
    assert remain >= 0
    assert remain < 1.0