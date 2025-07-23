from typing import Optional, Any
import logging

import redis.asyncio as redis

from shared.cache.redis import Redis

logger = logging.getLogger(__name__)

class AsyncRedis(Redis):
    """
    Asynchronous Redis client that extends a base synchronous Redis class.
    """
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 decode_responses: bool = True):       
        if self._initialized:
            return  # avoid reinitialization

        self.host = host
        self.port = port
        self.decode_responses = decode_responses
        self._redis: Optional[redis.Redis] = None
        self._initialized = True

    async def connect(self):
        """
        Establish an asynchronous Redis connection if not already connected.

        Raises:
            Exception: If connection fails.
        """
        if self._redis is not None:
            return  # Already connected

        try:
            self._redis = redis.Redis(host=self.host, port=self.port, decode_responses=self.decode_responses)
            await self._redis.ping()     # Test connectivity
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis connection error: host={self.host}, port={self.port}: {e}", exc_info=True)
            raise

    async def set(self, key: str, value: Any, expire_in_sec: Optional[int] = None) -> bool:
        try:
            await self._redis.set(name=key, value=value, ex=expire_in_sec)
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to set key '{key}' in Redis: {e}", exc_info=True)
            return False
        
    async def lpop(self, key: str, count: Optional[int] = None) -> Optional[Any]:
        try:
            value = await self._redis.lpop(name=key, count=count)
            if value:
                return value
        except redis.RedisError as e:
            logger.error(f"Failed to lpop key '{key}' in Redis: {e}", exc_info=True)
            return None

    async def rpush(self, key: str, value: Any) -> bool:
        try:
            return await self._redis.rpush(key, value) > 0
        except redis.RedisError as e:
            logger.error(f"Failed to rpush key '{key}' in Redis: {e}", exc_info=True)
            return False

    async def append(self, key: str, value: str) -> bool:
        try:
            await self._redis.append(key=key, value=value)
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to append {value} to '{key}' in Redis: {e}", exc_info=True)
            return False

    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self._redis.get(name=key)
            if value:
                return value
        except redis.RedisError as e:
            logger.error(f"Failed to get key '{key}' from Redis: {e}", exc_info=True)
            return None

    async def delete(self, key: str) -> bool:
        try:
            res = await self._redis.delete(key)
            return res > 0
        except redis.RedisError as e:
            logger.error(f"Failed to delete key '{key}' from Redis: {e}", exc_info=True)
            return False

    async def exists(self, key: str) -> bool:
        try:
            res = await self._redis.exists(key)
            return res > 0
        except redis.RedisError as e:
            logger.error(f"Failed to check existence of key '{key}' in Redis: {e}", exc_info=True)
            return False

    async def flush_all(self) -> None:
        try:
            await self._redis.flushall()
        except redis.RedisError as e:
            logger.error(f"Failed to flush all Redis data: {e}", exc_info=True)

    async def close(self) -> None:
        await self._redis.aclose()