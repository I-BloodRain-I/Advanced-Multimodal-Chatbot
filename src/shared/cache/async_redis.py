"""
Asynchronous Redis client for non-blocking operations.

This module provides an AsyncRedis class that extends the base Redis
functionality with asynchronous operations using redis.asyncio. It supports
all the same operations as the synchronous Redis client but with async/await
syntax for improved performance in asynchronous contexts.

The AsyncRedis client is particularly useful for pub/sub operations and
high-throughput scenarios where blocking operations would impact performance.
"""

from typing import List, Optional, Any, Tuple, Union
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
                 db: int = 0,
                 decode_responses: bool = True):       
        if self._initialized:
            return  # avoid reinitialization

        self.host = host
        self.port = port
        self.db = db
        self.decode_responses = decode_responses
        try:
            self._redis = redis.Redis(host=self.host, port=self.port, db=db, decode_responses=decode_responses)
            # asyncio.run(self._redis.ping())  # Test the connection
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis connection error: host={self.host}, port={self.port}: {e}", exc_info=True)
            raise
        self._initialized = True

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

    async def blpop(self, keys: Union[str, List[str]], timeout: int = 0) -> Optional[Tuple[str, Any]]:
        try:
            result = await self._redis.blpop(keys=keys, timeout=timeout)
            if result:
                return result
        except redis.RedisError as e:
            logger.error(f"Failed to blpop keys '{keys}' in Redis: {e}", exc_info=True)
            return None

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> Optional[List[Any]]:
        try:
            value = await self._redis.lrange(name=key, start=start, end=end)
            if value:
                return value
        except redis.RedisError as e:
            logger.error(f"Failed to lrange key '{key}' in Redis: {e}", exc_info=True)
            return None

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