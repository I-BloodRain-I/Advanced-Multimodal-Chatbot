import pytest
from unittest.mock import patch, AsyncMock
import redis.asyncio as redis

from shared.cache import AsyncRedis


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    AsyncRedis._instance = None
    yield
    AsyncRedis._instance = None

@pytest.fixture
def mock_redis():
    with patch('redis.asyncio.Redis') as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


def test_singleton_pattern():
    instance1 = AsyncRedis()
    instance2 = AsyncRedis()
    assert instance1 is instance2


def test_singleton_no_reinitialization():
    instance1 = AsyncRedis(host='host1', port=1111)
    instance2 = AsyncRedis(host='host2', port=2222)
    
    assert instance1 is instance2
    assert instance1.host == 'host1'
    assert instance1.port == 1111


def test_init():
    redis_instance = AsyncRedis(host='test_host', port=1234, decode_responses=False)
    
    assert redis_instance.host == 'test_host'
    assert redis_instance.port == 1234
    assert redis_instance.decode_responses is False
    assert redis_instance._redis is None


@pytest.mark.asyncio
async def test_connect_success(mock_redis):
    redis_instance = AsyncRedis()
    await redis_instance.connect()
    
    assert redis_instance._redis is not None
    mock_redis.ping.assert_called_once()


@pytest.mark.asyncio
async def test_connect_already_connected(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    await redis_instance.connect()
    
    mock_redis.ping.assert_not_called()


@pytest.mark.asyncio
async def test_connect_error():
    with patch('redis.asyncio.Redis') as mock_redis:
        mock_client = AsyncMock()
        mock_client.ping.side_effect = redis.RedisError("Connection failed")
        mock_redis.return_value = mock_client
        
        redis_instance = AsyncRedis()
        
        with pytest.raises(redis.RedisError):
            await redis_instance.connect()


@pytest.mark.asyncio
async def test_set_success(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.set('key', 'value', 3600)
    
    assert result is True
    mock_redis.set.assert_called_once_with(name='key', value='value', ex=3600)


@pytest.mark.asyncio
async def test_set_without_expiration(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.set('key', 'value')
    
    assert result is True
    mock_redis.set.assert_called_once_with(name='key', value='value', ex=None)


@pytest.mark.asyncio
async def test_set_error(mock_redis):
    mock_redis.set.side_effect = redis.RedisError("Set failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.set('key', 'value')
    
    assert result is False


@pytest.mark.asyncio
async def test_get_success(mock_redis):
    mock_redis.get.return_value = 'value'
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.get('key')
    
    assert result == 'value'
    mock_redis.get.assert_called_once_with(name='key')


@pytest.mark.asyncio
async def test_get_not_found(mock_redis):
    mock_redis.get.return_value = None
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.get('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_get_empty_string_returns_none(mock_redis):
    mock_redis.get.return_value = ""
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.get('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_get_error(mock_redis):
    mock_redis.get.side_effect = redis.RedisError("Get failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.get('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_lpop_success(mock_redis):
    mock_redis.lpop.return_value = 'value'
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.lpop('key')
    
    assert result == 'value'
    mock_redis.lpop.assert_called_once_with(name='key', count=None)


@pytest.mark.asyncio
async def test_lpop_with_count(mock_redis):
    mock_redis.lpop.return_value = ['val1', 'val2']
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.lpop('key', 2)
    
    assert result == ['val1', 'val2']
    mock_redis.lpop.assert_called_once_with(name='key', count=2)


@pytest.mark.asyncio
async def test_lpop_empty(mock_redis):
    mock_redis.lpop.return_value = None
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.lpop('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_lpop_empty_string_returns_none(mock_redis):
    mock_redis.lpop.return_value = ""
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.lpop('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_lpop_error(mock_redis):
    mock_redis.lpop.side_effect = redis.RedisError("Lpop failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.lpop('key')
    
    assert result is None


@pytest.mark.asyncio
async def test_rpush_success(mock_redis):
    mock_redis.rpush.return_value = 1
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.rpush('key', 'value')
    
    assert result is True
    mock_redis.rpush.assert_called_once_with('key', 'value')


@pytest.mark.asyncio
async def test_rpush_zero_return(mock_redis):
    mock_redis.rpush.return_value = 0
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.rpush('key', 'value')
    
    assert result is False


@pytest.mark.asyncio
async def test_rpush_error(mock_redis):
    mock_redis.rpush.side_effect = redis.RedisError("Rpush failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.rpush('key', 'value')
    
    assert result is False


@pytest.mark.asyncio
async def test_append_success(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.append('key', 'value')
    
    assert result is True
    mock_redis.append.assert_called_once_with(key='key', value='value')


@pytest.mark.asyncio
async def test_append_error(mock_redis):
    mock_redis.append.side_effect = redis.RedisError("Append failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.append('key', 'value')
    
    assert result is False


@pytest.mark.asyncio
async def test_delete_success(mock_redis):
    mock_redis.delete.return_value = 1
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.delete('key')
    
    assert result is True
    mock_redis.delete.assert_called_once_with('key')


@pytest.mark.asyncio
async def test_delete_not_found(mock_redis):
    mock_redis.delete.return_value = 0
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.delete('key')
    
    assert result is False


@pytest.mark.asyncio
async def test_delete_error(mock_redis):
    mock_redis.delete.side_effect = redis.RedisError("Delete failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.delete('key')
    
    assert result is False


@pytest.mark.asyncio
async def test_exists_true(mock_redis):
    mock_redis.exists.return_value = 1
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.exists('key')
    
    assert result is True
    mock_redis.exists.assert_called_once_with('key')


@pytest.mark.asyncio
async def test_exists_false(mock_redis):
    mock_redis.exists.return_value = 0
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.exists('key')
    
    assert result is False


@pytest.mark.asyncio
async def test_exists_error(mock_redis):
    mock_redis.exists.side_effect = redis.RedisError("Exists failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    result = await redis_instance.exists('key')
    
    assert result is False


@pytest.mark.asyncio
async def test_flush_all_success(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    await redis_instance.flush_all()
    
    mock_redis.flushall.assert_called_once()


@pytest.mark.asyncio
async def test_flush_all_error(mock_redis):
    mock_redis.flushall.side_effect = redis.RedisError("Flush failed")
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    await redis_instance.flush_all()
    
    mock_redis.flushall.assert_called_once()


@pytest.mark.asyncio
async def test_close(mock_redis):
    redis_instance = AsyncRedis()
    redis_instance._redis = mock_redis
    
    await redis_instance.close()
    
    mock_redis.aclose.assert_called_once()