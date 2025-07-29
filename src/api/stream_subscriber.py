"""
Redis pub/sub streaming subscriber for WebSocket message forwarding.

Handles Redis pub/sub subscriptions for conversation channels and forwards
streaming messages to the appropriate WebSocket connections via the ConnectionManager.
Provides unified message listening and automatic subscription management.
"""
import asyncio
from typing import Dict, Optional
import logging

from api.manager import ConnectionManager
from shared.cache import AsyncRedis

logger = logging.getLogger(__name__)


class StreamSubscriber:
    """
    Handles Redis pub/sub message forwarding to WebSocket clients.

    Subscribes to a Redis channel for each conversation and forwards streamed
    messages to the appropriate WebSocket connection via the ConnectionManager.
    """
    def __init__(self, 
                 manager: Optional[ConnectionManager] = None, 
                 host: str = 'localhost', 
                 port: int = 6379,
                 db: int = 0):
        self.manager = manager or ConnectionManager()
        self.redis = AsyncRedis(host=host, port=port, db=db, decode_responses=False)
        self.pubsub = self.redis.pubsub()
        self.tasks: Dict[str, asyncio.Task] = {}
        self.listener_task = None

    async def _start_listener(self):
        """Starts the unified message listener if not already running."""
        if self.listener_task is None:
            self.listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        """Listens for all messages and forwards them to appropriate channels."""
        try:
            async for msg in self.pubsub.listen():
                if msg['type'] == 'message':
                    conv_id = msg['channel'].decode() if isinstance(msg['channel'], bytes) else msg['channel']
                    await self.manager.send_message(msg['data'], conv_id)
        except Exception as e:
            logger.error(f"Pubsub listener error: {e}", exc_info=True)

    async def subscribe(self, conv_id: str):
        """
        Subscribe to a Redis channel for a given conversation ID.
        
        Args:
            conv_id: The conversation identifier used as the Redis channel.
        """
        if conv_id in self.tasks:
            return

        await self.pubsub.subscribe(conv_id)
        await self._start_listener()
        
        # Create empty task for tracking
        self.tasks[conv_id] = asyncio.create_task(asyncio.sleep(0))
        logger.info(f"Subscribed to: {conv_id}")

    async def unsubscribe(self, conv_id: str):
        """
        Unsubscribe from a Redis channel for a specific conversation.
        
        Args:
            conv_id: The conversation identifier to unsubscribe from.
        """
        if conv_id not in self.tasks:
            return

        await self.pubsub.unsubscribe(conv_id)
        self.tasks[conv_id].cancel()
        del self.tasks[conv_id]
        logger.info(f"Unsubscribed from: {conv_id}")

        # Stop listener if no active subscriptions
        if not self.tasks and self.listener_task:
            self.listener_task.cancel()
            self.listener_task = None

    async def close(self):
        """
        Close all subscriptions and cleanup resources.
        
        Cancels all active tasks and closes the Redis pub/sub connection.
        """
        if self.listener_task:
            self.listener_task.cancel()
        await self.pubsub.close()
        self.tasks.clear()