from typing import Optional, Any
import logging

import redis
from redis.client import PubSub

logger = logging.getLogger(__name__)

class Redis:
    """
    Singleton wrapper around the redis-py client to simplify Redis operations.

    This class manages a single Redis connection instance and provides utility
    methods to interact with Redis.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Ensure only one instance is ever created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 decode_responses: bool = True):
        # Prevent reinitialization
        if self._initialized:
            return

        self.host = host
        self.port = port
        try:
            self._redis = redis.Redis(host=self.host, port=self.port, decode_responses=decode_responses)
            self._redis.ping()  # Test the connection
        except Exception as e:
            logger.error(f"Redis connection error: host={self.host}, port={self.port}: {e}", exc_info=True)
            raise
        self._initialized = True

    def set(self, key: str, value: Any, expire_in_sec: Optional[int] = None) -> bool:
        """
        Set a value for a given key in Redis, with optional expiration.

        Args:
            key (str): The Redis key.
            value (Any): The value to store.
            expire_in_sec (Optional[int]): Optional expiration time in seconds.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return self._redis.set(name=key, value=value, ex=expire_in_sec)
        except redis.RedisError as e:
            logger.error(f"Failed to set key '{key}' in Redis: {e}", exc_info=True)
            return False

    def lpop(self, key: str, count: Optional[int] = None) -> Optional[Any]:
        """
        Pop one or more elements from the left of a Redis list.

        Args:
            key (str): The Redis list key.
            count (Optional[int]): Number of elements to pop.

        Returns:
            Optional[Any]: The popped element(s), or None on failure.
        """
        try:
            value = self._redis.lpop(name=key, count=count)
            if value:
                return value
        except redis.RedisError as e:
            logger.error(f"Failed to lpop key '{key}' in Redis: {e}", exc_info=True)
            return None

    def rpush(self, key: str, value: Any) -> bool:
        """
        Push a value to the right end of a Redis list.

        Args:
            key (str): The Redis list key.
            value (Any): The value to append.

        Returns:
            bool: True if at least one element was pushed, False otherwise.
        """
        try:
            return self._redis.rpush(key, value) > 0
        except redis.RedisError as e:
            logger.error(f"Failed to rpush key '{key}' in Redis: {e}", exc_info=True)
            return False

    def append(self, key: str, value: str) -> bool:
        """
        Append a string value to an existing Redis key.

        Args:
            key (str): The Redis key.
            value (str): The string to append.

        Returns:
            bool: True if the operation succeeded, False otherwise.
        """
        try:
            return self._redis.append(key=key, value=value)
        except redis.RedisError as e:
            logger.error(f"Failed to append {value} to '{key}' in Redis: {e}", exc_info=True)
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with a given key.

        Args:
            key (str): The Redis key.

        Returns:
            Optional[Any]: The retrieved value, or None if not found or on error.
        """
        try:
            value = self._redis.get(name=key)
            if value:
                return value
        except redis.RedisError as e:
            logger.error(f"Failed to get key '{key}' from Redis: {e}", exc_info=True)
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key (str): The Redis key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        try:
            return self._redis.delete(key) > 0
        except redis.RedisError as e:
            logger.error(f"Failed to delete key '{key}' from Redis: {e}", exc_info=True)
            return False

    def exists(self, key: str) -> bool:
        """
        Check if a given key exists in Redis.

        Args:
            key (str): The Redis key.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        try:
            return self._redis.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Failed to check existence of key '{key}' in Redis: {e}", exc_info=True)
            return False

    def flush_all(self) -> bool:
        """
        Delete all keys from all Redis databases.

        Returns:
            bool: True if the flush was successful, False otherwise.
        """
        try:
            self._redis.flushall()
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to flush all Redis data: {e}", exc_info=True)
            return False

    def close(self):
        """
        Close the Redis connection.
        """
        self._redis.close()

    def pubsub(self) -> Optional[PubSub]:
        """
        Create a PubSub instance for subscribing to channels.

        Returns:
            Optional[PubSub]: A PubSub object, or None if creation failed.
        """
        try:
            return self._redis.pubsub()
        except redis.RedisError as e:
            logger.error(f"Failed to create PubSub: {e}", exc_info=True)
            return None

    def publish(self, channel: str, message: str) -> int:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish to.
            message (str): The message content.

        Returns:
            int: Number of clients that received the message.
        """
        try:
            return self._redis.publish(channel=channel, message=message)
        except redis.RedisError as e:
            logger.error(f"Failed to publish to channel '{channel}': {e}", exc_info=True)
            return 0