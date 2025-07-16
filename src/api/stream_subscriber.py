import asyncio
import json
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

    END_MESSAGE = "[[END]]"

    def __init__(self, 
                 manager: Optional[ConnectionManager] = None, 
                 host: str = 'localhost', 
                 port: int = 6379):
        self.manager = manager or ConnectionManager()
        self.redis = AsyncRedis(host=host, port=port, decode_response=True)
        self.tasks: Dict[str, asyncio.Task] = {}

    async def subscribe(self, conv_id: str):
        """
        Start listening to a Redis pub/sub channel for a given conversation ID.

        If already subscribed, this is a no-op. Messages are streamed to the
        corresponding WebSocket client managed by `ConnectionManager`.

        Args:
            conv_id (str): The conversation identifier used as the Redis channel.
        """
        if conv_id in self.tasks:
            logger.info(f"Already subscribed to channel: {conv_id}")
            return

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(conv_id)
        logger.info(f"Subscribed to Redis channel: {conv_id}")

        async def forward(manager: ConnectionManager):
            try:
                async for msg in pubsub.listen():
                    if msg['type'] == 'message':

                        content = msg['data']
                        if isinstance(content, bytes):
                            content = content.decode()

                        # [[END]] indicates end of streaming
                        if content == self.END_MESSAGE:
                            await manager.send_message(
                                json.dumps({"type": "end"}), conv_id
                            )
                            break
                        else:
                            # Forward content to WebSocket client
                            await manager.send_message(
                                json.dumps({"type": "stream", "content": content}), conv_id
                            )
            except asyncio.CancelledError:
                logger.info(f"Forwarding task cancelled for conv_id: {conv_id}")
            except Exception as e:
                logger.error(f"Error forwarding messages for conv_id: {conv_id}: {e}", exc_info=True)
            finally:
                try:
                    await pubsub.unsubscribe(conv_id)
                    await pubsub.close()
                    logger.info(f"Unsubscribed and closed Redis pubsub for conv_id: {conv_id}")
                except Exception as e:
                    logger.error(f"Failed to clean up pubsub for conv_id: {conv_id}: {e}", exc_info=True)

        # Start background task to forward messages
        task = asyncio.create_task(forward(self.manager))
        self.tasks[conv_id] = task

    def unsubscribe(self, conv_id: str):
        """
        Stop listening to the Redis channel for a specific conversation.

        Args:
            conv_id (str): The conversation identifier to unsubscribe from.
        """
        task = self.tasks.get(conv_id)
        if task:
            task.cancel()
            del self.tasks[conv_id]
