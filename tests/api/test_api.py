import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import WebSocket, WebSocketDisconnect

from api.manager import ConnectionManager
from api.stream_subscriber import StreamSubscriber
from api.utils import create_app, start_server, store_prompt, load_listener


@pytest.fixture(autouse=True)
def reset_singleton():
    ConnectionManager._instance = None
    yield
    ConnectionManager._instance = None

@pytest.fixture
def connection_manager():
    ConnectionManager._instance = None
    return ConnectionManager()

@pytest.fixture
def mock_websocket():
    websocket = AsyncMock(spec=WebSocket)
    type(websocket).__bool__ = lambda self: True
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.send_bytes = AsyncMock()  
    return websocket


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.connect = AsyncMock()
    redis.close = AsyncMock()
    redis.rpush = AsyncMock()
    redis.pubsub = Mock()
    return redis


def test_connection_manager_singleton():
    ConnectionManager._instance = None
    manager1 = ConnectionManager()
    manager2 = ConnectionManager()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_connection_manager_connect(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    await connection_manager.connect(mock_websocket, conv_id)
    
    mock_websocket.accept.assert_called_once()
    assert conv_id in connection_manager.active_connections
    assert connection_manager.active_connections[conv_id] is mock_websocket


def test_connection_manager_disconnect(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    connection_manager.active_connections[conv_id] = mock_websocket
    
    connection_manager.disconnect(conv_id)
    
    assert conv_id not in connection_manager.active_connections


def test_connection_manager_disconnect_nonexistent(connection_manager):
    conv_id = "nonexistent_conv"
    connection_manager.disconnect(conv_id)
    assert conv_id not in connection_manager.active_connections


@pytest.mark.asyncio
async def test_connection_manager_send_text_message(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    message = "Hello, World!"
    connection_manager.active_connections[conv_id] = mock_websocket
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is True
    mock_websocket.send_text.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_connection_manager_send_bytes_message(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    message = b"Binary data"
    connection_manager.active_connections[conv_id] = mock_websocket
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is True
    mock_websocket.send_bytes.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_connection_manager_send_message_no_connection(connection_manager):
    conv_id = "nonexistent_conv"
    message = "Hello, World!"
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is False


@pytest.mark.asyncio
async def test_connection_manager_send_message_exception(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    message = "Hello, World!"
    connection_manager.active_connections[conv_id] = mock_websocket
    mock_websocket.send_text.side_effect = Exception("Connection error")
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is False
    assert conv_id not in connection_manager.active_connections


def test_stream_subscriber_init():
    manager = Mock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    
    assert subscriber.manager is manager
    assert subscriber.tasks == {}


def test_stream_subscriber_init_default_manager():
    ConnectionManager._instance = None
    subscriber = StreamSubscriber(host="localhost", port=6379)
    
    assert isinstance(subscriber.manager, ConnectionManager)
    ConnectionManager._instance = None


@pytest.mark.asyncio
async def test_stream_subscriber_subscribe_new(mock_redis):
    async def async_gen():
        if False:
            yield

    manager = Mock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    subscriber.redis = mock_redis
    
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.listen = Mock(return_value=async_gen())
    # mock_pubsub.listen.return_value = []
    mock_redis.pubsub.return_value = mock_pubsub
    
    conv_id = "test_conv_123"
    await subscriber.subscribe(conv_id)
    
    mock_pubsub.subscribe.assert_called_once_with(conv_id)
    assert conv_id in subscriber.tasks


@pytest.mark.asyncio
async def test_stream_subscriber_subscribe_already_exists():
    manager = Mock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    
    conv_id = "test_conv_123"
    mock_task = Mock()
    subscriber.tasks[conv_id] = mock_task
    
    await subscriber.subscribe(conv_id)
    
    assert subscriber.tasks[conv_id] is mock_task


def test_stream_subscriber_unsubscribe():
    manager = Mock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    
    conv_id = "test_conv_123"
    mock_task = Mock()
    subscriber.tasks[conv_id] = mock_task
    
    subscriber.unsubscribe(conv_id)
    
    mock_task.cancel.assert_called_once()
    assert conv_id not in subscriber.tasks


def test_stream_subscriber_unsubscribe_nonexistent():
    manager = Mock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    
    conv_id = "nonexistent_conv"
    subscriber.unsubscribe(conv_id)
    
    assert conv_id not in subscriber.tasks


@patch('api.utils.Config')
def test_create_app(mock_config):
    mock_config.return_value.get.return_value.get.side_effect = lambda key: {
        'host': 'localhost',
        'port': 6379
    }[key]
    
    app = create_app()
    
    assert app is not None
    assert hasattr(app, 'state')


@patch('api.utils.uvicorn')
def test_start_server(mock_uvicorn):
    app = Mock()
    host = "localhost"
    port = 8000
    
    start_server(app, host, port)
    
    mock_uvicorn.run.assert_called_once_with(app, host=host, port=port)


@patch('api.utils.uvicorn')
def test_start_server_with_kwargs(mock_uvicorn):
    app = Mock()
    host = "localhost"
    port = 8000
    reload = True
    
    start_server(app, host, port, reload=reload)
    
    mock_uvicorn.run.assert_called_once_with(app, host=host, port=port, reload=reload)


@patch('api.utils.Config')
def test_load_listener(mock_config):
    mock_config.return_value.get.return_value.get.side_effect = lambda key: {
        'host': 'localhost',
        'port': 6379
    }[key]
    
    manager = Mock()
    listener = load_listener(manager)
    
    assert isinstance(listener, StreamSubscriber)
    assert listener.manager is manager


@patch('api.utils.Config')
def test_load_listener_default_manager(mock_config):
    mock_config.return_value.get.return_value.get.side_effect = lambda key: {
        'host': 'localhost',
        'port': 6379
    }[key]
    
    ConnectionManager._instance = None
    listener = load_listener()
    
    assert isinstance(listener, StreamSubscriber)
    assert isinstance(listener.manager, ConnectionManager)
    ConnectionManager._instance = None


@pytest.mark.asyncio
async def test_store_prompt_success(mock_redis):
    conv_id = "test_conv_123"
    prompt_data = {"messages": [{"role": "user", "content": "Hello"}]}
    
    await store_prompt(mock_redis, conv_id, prompt_data)
    
    expected_json = json.dumps(prompt_data)
    mock_redis.rpush.assert_called_once_with("process:messages_batch", expected_json)


@pytest.mark.asyncio
async def test_store_prompt_exception(mock_redis):
    conv_id = "test_conv_123"
    prompt_data = {"messages": [{"role": "user", "content": "Hello"}]}
    mock_redis.rpush.side_effect = Exception("Redis error")
    
    await store_prompt(mock_redis, conv_id, prompt_data)
    
    mock_redis.rpush.assert_called_once()


@pytest.mark.asyncio
@patch('api.server.manager', new_callable=AsyncMock)
@patch('api.server.listener', new_callable=AsyncMock)
async def test_websocket_endpoint_normal_flow(mock_listener, mock_manager):
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = [
        '{"messages": [{"role": "user", "content": "Hello"}]}',
        asyncio.CancelledError()
    ]
    mock_websocket.app.state.redis = AsyncMock()
    
    conv_id = "test_conv_123"
    
    from api.server import websocket_endpoint
    
    with pytest.raises(asyncio.CancelledError):
        await websocket_endpoint(mock_websocket, conv_id)
    
    mock_manager.connect.assert_called_once_with(mock_websocket, conv_id)
    mock_listener.subscribe.assert_called_once_with(conv_id)


@pytest.mark.asyncio
@patch('api.server.manager', new_callable=AsyncMock)
@patch('api.server.listener', new_callable=AsyncMock)
async def test_websocket_endpoint_invalid_json(mock_listener, mock_manager):
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = [
        'invalid json',
        asyncio.CancelledError()
    ]
    mock_websocket.app.state.redis = AsyncMock()
    
    conv_id = "test_conv_123"
    
    from api.server import websocket_endpoint
    
    with pytest.raises(asyncio.CancelledError):
        await websocket_endpoint(mock_websocket, conv_id)
    
    mock_manager.connect.assert_called_once_with(mock_websocket, conv_id)


@pytest.mark.asyncio
@patch('api.server.manager')
@patch('api.server.listener')
async def test_websocket_endpoint_websocket_disconnect(mock_listener, mock_manager):
    mock_manager.connect = AsyncMock()
    mock_manager.disconnect = Mock()  
    mock_listener.subscribe = AsyncMock()
    mock_listener.unsubscribe = Mock()

    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    mock_websocket.app.state.redis = AsyncMock()

    conv_id = "test_conv_123"

    from api.server import websocket_endpoint

    await websocket_endpoint(mock_websocket, conv_id)

    mock_manager.connect.assert_called_once_with(mock_websocket, conv_id)
    mock_manager.disconnect.assert_called_once_with(conv_id)
    mock_listener.unsubscribe.assert_called_once_with(conv_id)


@pytest.mark.asyncio
@patch('api.server.manager')
@patch('api.server.listener')
async def test_websocket_endpoint_unexpected_exception(mock_listener, mock_manager):
    mock_manager.connect = AsyncMock()
    mock_manager.disconnect = Mock()  
    mock_listener.subscribe = AsyncMock()
    mock_listener.unsubscribe = Mock()

    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    mock_websocket.app.state.redis = AsyncMock()

    
    conv_id = "test_conv_123"
    
    from api.server import websocket_endpoint
    
    await websocket_endpoint(mock_websocket, conv_id)
    
    mock_manager.connect.assert_called_once_with(mock_websocket, conv_id)
    mock_manager.disconnect.assert_called_once_with(conv_id)
    mock_listener.unsubscribe.assert_called_once_with(conv_id)


@pytest.mark.asyncio
async def test_stream_subscriber_forward_message(mock_redis):
    manager = AsyncMock()
    subscriber = StreamSubscriber(manager=manager, host="localhost", port=6379)
    subscriber.redis = mock_redis
    
    mock_pubsub = AsyncMock()
    mock_message = {
        'type': 'message',
        'data': b'test message'
    }

    async def async_gen():
        yield mock_message

    mock_pubsub.listen = lambda: async_gen()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.close = AsyncMock()
    
    mock_redis.pubsub.return_value = mock_pubsub
    
    conv_id = "test_conv_123"
    await subscriber.subscribe(conv_id)
    
    await asyncio.sleep(0.1)
    
    mock_pubsub.subscribe.assert_called_once_with(conv_id)


def test_connection_manager_active_connections_isolation():
    ConnectionManager._instance = None
    manager1 = ConnectionManager()
    manager1.active_connections['test1'] = Mock()
    
    manager2 = ConnectionManager()
    assert 'test1' in manager2.active_connections
    
    ConnectionManager._instance = None


@pytest.mark.asyncio
async def test_connection_manager_send_message_empty_string(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    message = ""
    connection_manager.active_connections[conv_id] = mock_websocket
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is True
    mock_websocket.send_text.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_connection_manager_send_message_empty_bytes(connection_manager, mock_websocket):
    conv_id = "test_conv_123"
    message = b""
    connection_manager.active_connections[conv_id] = mock_websocket
    
    result = await connection_manager.send_message(message, conv_id)
    
    assert result is True
    mock_websocket.send_bytes.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_store_prompt_complex_data(mock_redis):
    conv_id = "test_conv_123"
    prompt_data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ],
        "metadata": {
            "timestamp": "2025-01-01T00:00:00Z",
            "user_id": "user123"
        }
    }
    
    await store_prompt(mock_redis, conv_id, prompt_data)
    
    expected_json = json.dumps(prompt_data)
    mock_redis.rpush.assert_called_once_with("process:messages_batch", expected_json)