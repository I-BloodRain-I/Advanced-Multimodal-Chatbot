import asyncio
import threading
import logging
import time
import json
from typing import AsyncGenerator, Optional, List, Dict

from common.utils import require_env_var
from core.entities.types import ConversationBatch, Message, AgentResponse
from orchestrator.pipeline import Pipeline
from shared.cache.redis import Redis

logger = logging.getLogger(__name__)


class PromptDispatcher:
    """
    Acts as a mediator between the frontend and the backend processing pipeline.

    Incoming requests from the frontend are written to a Redis queue by the server.
    This class periodically polls Redis for new conversation batches, processes them
    through the pipeline, and streams back generated responses to the server.

    It enables asynchronous, batched handling of user prompts, improving throughput and stability.

    Args:
        sleep_seconds (float): Time to wait between polling attempts when no new data is available.
    """
    def __init__(self, sleep_seconds: float = 1.0):
        self.sleep_seconds = sleep_seconds
        self.pipeline = Pipeline.build()
        self._load_redis()

    def start_loop(self):
        """
        Starts the infinite loop that polls Redis for incoming conversation batches.

        This is the main entry point for the dispatcher and is typically called once
        when the background processing service is launched.
        """
        logger.info("PromptDispatcher loop started.")
        self._batch_processing_loop()

    def _load_redis(self):
        """
        Initializes the Redis client and clears any stale message batches from the queue to ensure
        clean startup without processing outdated requests.
        """
        self._redis = Redis(host=require_env_var('REDIS_HOST'), port=int(require_env_var('REDIS_PORT')))
        self._redis.delete("process:messages_batch")    # Clear any unprocessed batches

    def _batch_processing_loop(self):
        """
        Main loop that continuously checks Redis for new prompts, processes them using the pipeline,
        and streams the results back asynchronously.

        If no data is available, it sleeps for the configured interval before checking again.
        """
        def _calc_remain_seconds(sleep_seconds: int, start_time: float):
            # Adjust sleep duration so the loop maintains a steady interval,
            # compensating for time already spent processing
            return max(0, sleep_seconds - (time.time() - start_time))

        while True:
            start_t = time.time()

            try:
                batch = self._get_conversation_batch()
                if batch is None:
                    # No data yet, wait and try again
                    time.sleep(_calc_remain_seconds(self.sleep_seconds, start_t))
                    continue

                logger.info("Starting pipeline execution")
                responses = self.pipeline(batch)
                self._send_agent_responses(responses)

            except Exception as e:
                logger.error(f"Error in batch processing loop.", exc_info=True)

            # Sleep just enough to maintain loop frequency
            time.sleep(_calc_remain_seconds(self.sleep_seconds, start_t))
    
    def _get_conversation_batch(self) -> Optional[ConversationBatch]:
        """
        Retrieves and parses all pending conversation batches from Redis.

        Each item is expected to contain a conversation ID and a list of message history entries.
        Invalid or malformed entries are logged and skipped.

        Returns:
            Optional[ConversationBatch]: A batch of conversation IDs and their histories,
                                         or None if the queue is empty.
        """
        batch_data: List[Dict[str, List[Dict[str, str]]]] = []

        while True:
            item = self._redis.lpop("process:messages_batch")
            if item is None:
                break
            try:
                batch_data.append(json.loads(item))
            except Exception:
                 logger.error(f"Failed to decode Redis batch item: {item}.", exc_info=True)

        if not batch_data:
            return None

        conv_ids, histories = [], []
        for conv_info in batch_data:
            try:
                if 'conv_id' not in conv_info or 'history' not in conv_info:
                    raise KeyError()

                conv_ids.append(conv_info['conv_id'])
                histories.append([Message(role=msg['role'], content=msg['content']) for msg in conv_info['history']])
            except KeyError as e:
                logger.error(f"Missing expected key in conversation info.", exc_info=True)

        return ConversationBatch(conv_ids=conv_ids, histories=histories)
    
    def _send_agent_responses(self, responses: List[AgentResponse]):
        """
        Sends a list of agent-generated responses back to the frontend via Redis.

        This method launches a background thread that streams each response asynchronously.
        """
        def _send_async(responses):
            async def _async_main():
                try:
                    tasks = []
                    for resp in responses:
                        if resp.type == 'stream':
                            # Stream tokens for streaming response
                            tasks.append(self._stream_response(resp.conv_id, resp.content))
                        else:
                            # Send full message or image in one go
                            tasks.append(self._response(resp.conv_id, resp.type, resp.content))
                    await asyncio.gather(*tasks)
                except Exception as e:
                    logger.error(f"Failed to launch async agent responses.", exc_info=True)
            
            # Run in the event loop
            asyncio.run(_async_main())

        # Run in background
        threading.Thread(target=_send_async, args=(responses,), daemon=True).start()

    async def _stream_response(self, conv_id: str, generator: AsyncGenerator[str, None]):
        """
        Streams an individual response token-by-token over Redis to the client.

        This uses Redis pub/sub, where the conversation ID serves as the channel.
        Once all tokens are sent, a special end-of-stream message is published.

        Args:
            conv_id (str): Unique conversation ID used as the Redis channel.
            generator (AsyncGenerator[str, None]): Asynchronous generator producing response tokens.
        """
        try:
            async for token in generator:
                # Stream each token
                self._redis.publish(conv_id, json.dumps({"type": "stream", "content": token}))
            # Signal end of stream
            self._redis.publish(conv_id, json.dumps({"type": "end_stream"}))      
        except Exception as e: 
            logger.error(f"Error streaming response for conversation {conv_id}", exc_info=True)

    async def _response(self, conv_id: str, type: str, content: str):
        """
        Sends a complete response (text or image) to the client via Redis.

        Args:
            conv_id (str): Unique conversation ID to publish to.
            type (str): Response type ("text", "image", etc.).
            content (str): The actual content to send.
        """
        try:
            self._redis.publish(conv_id, json.dumps({"type": type, "content": content}))
        except Exception as e:
            logger.error(f"Error sending response for conversation {conv_id}", exc_info=True)
