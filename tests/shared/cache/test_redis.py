import pytest
from unittest.mock import Mock, patch
import redis
from redis.client import PubSub

from shared.cache.redis import Redis


@pytest.fixture(autouse=True)
def reset_singleton():
    Redis._instance = None
    yield
    Redis._instance = None

def test_singleton_pattern():
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        
        instance1 = Redis()
        instance2 = Redis()
        assert instance1 is instance2


def test_singleton_no_reinitialization():
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        
        instance1 = Redis(host='host1', port=1111)
        instance2 = Redis(host='host2', port=2222)
        
        assert instance1 is instance2
        assert instance1.host == 'host1'
        assert instance1.port == 1111


@patch('redis.Redis')
def test_init_success(mock_redis):
    mock_client = Mock()
    mock_redis.return_value = mock_client
    
    redis_instance = Redis(host='test_host', port=1234, db=0, decode_responses=False)
    
    mock_redis.assert_called_once_with(host='test_host', port=1234, db=0, decode_responses=False)
    mock_client.ping.assert_called_once()
    assert redis_instance.host == 'test_host'
    assert redis_instance.port == 1234


@patch('redis.Redis')
def test_init_connection_error(mock_redis):
    mock_redis.side_effect = redis.ConnectionError("Connection failed")
    
    with pytest.raises(redis.ConnectionError):
        Redis()


@patch('redis.Redis')
def test_init_ping_error(mock_redis):
    mock_client = Mock()
    mock_client.ping.side_effect = redis.RedisError("Ping failed")
    mock_redis.return_value = mock_client
    
    with pytest.raises(redis.RedisError):
        Redis()


def test_set_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.set('key', 'value', 3600)
        
        assert result is True
        mock_client.set.assert_called_once_with(name='key', value='value', ex=3600)


def test_set_without_expiration():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.set('key', 'value')
        
        assert result is True
        mock_client.set.assert_called_once_with(name='key', value='value', ex=None)


def test_set_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.set.side_effect = redis.RedisError("Set failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.set('key', 'value')
        
        assert result is False


def test_get_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = 'value'
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.get('key')
        
        assert result == 'value'
        mock_client.get.assert_called_once_with(name='key')


def test_get_not_found():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.get('key')
        
        assert result is None


def test_get_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = redis.RedisError("Get failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.get('key')
        
        assert result is None


def test_lpop_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.return_value = 'value'
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.lpop('key')
        
        assert result == 'value'
        mock_client.lpop.assert_called_once_with(name='key', count=None)


def test_lpop_with_count():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.return_value = ['val1', 'val2']
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.lpop('key', 2)
        
        assert result == ['val1', 'val2']
        mock_client.lpop.assert_called_once_with(name='key', count=2)


def test_lpop_empty():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.return_value = None
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.lpop('key')
        
        assert result is None


def test_lpop_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.side_effect = redis.RedisError("Lpop failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.lpop('key')
        
        assert result is None


def test_rpush_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.rpush.return_value = 1
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.rpush('key', 'value')
        
        assert result is True
        mock_client.rpush.assert_called_once_with('key', 'value')


def test_rpush_zero_return():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.rpush.return_value = 0
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.rpush('key', 'value')
        
        assert result is False


def test_rpush_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.rpush.side_effect = redis.RedisError("Rpush failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.rpush('key', 'value')
        
        assert result is False


def test_append_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.append.return_value = 5
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.append('key', 'value')
        
        assert result == 5
        mock_client.append.assert_called_once_with(key='key', value='value')


def test_append_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.append.side_effect = redis.RedisError("Append failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.append('key', 'value')
        
        assert result is False


def test_delete_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 1
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.delete('key')
        
        assert result is True
        mock_client.delete.assert_called_once_with('key')


def test_delete_not_found():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 0
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.delete('key')
        
        assert result is False


def test_delete_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.delete.side_effect = redis.RedisError("Delete failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.delete('key')
        
        assert result is False


def test_exists_true():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.exists.return_value = 1
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.exists('key')
        
        assert result is True
        mock_client.exists.assert_called_once_with('key')


def test_exists_false():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.exists.return_value = 0
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.exists('key')
        
        assert result is False


def test_exists_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.exists.side_effect = redis.RedisError("Exists failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.exists('key')
        
        assert result is False


def test_flush_all_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.flushall.return_value = True
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.flush_all()
        
        assert result is True
        mock_client.flushall.assert_called_once()


def test_flush_all_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.flushall.side_effect = redis.RedisError("Flush failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.flush_all()
        
        assert result is False


def test_close():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        redis_instance.close()
        
        mock_client.close.assert_called_once()


def test_pubsub_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_pubsub = Mock(spec=PubSub)
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.pubsub()
        
        assert result is mock_pubsub
        mock_client.pubsub.assert_called_once()


def test_pubsub_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.pubsub.side_effect = redis.RedisError("PubSub failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.pubsub()
        
        assert result is None


def test_publish_success():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.publish.return_value = 5
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.publish('channel', 'message')
        
        assert result == 5
        mock_client.publish.assert_called_once_with(channel='channel', message='message')


def test_publish_error():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.publish.side_effect = redis.RedisError("Publish failed")
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.publish('channel', 'message')
        
        assert result == 0


def test_get_empty_string_returns_none():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = ""
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.get('key')
        
        assert result is None


def test_lpop_empty_string_returns_none():
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpop.return_value = ""
        mock_redis.return_value = mock_client
        
        redis_instance = Redis()
        result = redis_instance.lpop('key')
        
        assert result is None