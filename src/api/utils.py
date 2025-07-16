from contextlib import asynccontextmanager
import logging
from typing import Dict, List, Optional
import json

from fastapi import FastAPI

from api.manager import ConnectionManager
from api.stream_subscriber import StreamSubscriber
from shared.cache import AsyncRedis
from shared.config import Config

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure a FastAPI application with Redis lifecycle hooks.

    Initializes a Redis client at startup and attaches it to the app state.
    Ensures clean Redis shutdown on app exit.

    Returns:
        FastAPI: A configured FastAPI application instance.
    """
    cfg = Config().get('cache.redis')

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis = AsyncRedis(host=cfg.get('host'), port=cfg.get('port'), decode_responses=False)
        await redis.connect()
        logger.info("Redis connection established.")

        # Make Redis available via app.state
        app.state.redis = redis
        yield  # Application runs here
        
         # Disconnect Redis cleanly on shutdown
        await redis.close()
        logger.info("Redis client closed.")

    app = FastAPI(lifespan=lifespan)
    return app


def start_server(app: FastAPI, host: str, port: int, **kwargs):
    """
    Start the FastAPI application using Uvicorn.

    Args:
        app (FastAPI): The FastAPI application to serve.
        host (str): Host IP or name to bind the server to.
        port (int): Port number to listen on.
        **kwargs: Optional additional arguments to pass to uvicorn.run.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port, **kwargs)


def load_listener(manager: Optional[ConnectionManager] = None) -> StreamSubscriber:
    """
    Create and return a Redis stream subscriber for handling message broadcasting.

    Args:
        manager (Optional[ConnectionManager]): The connection manager for WebSocket clients.
            If None, a new instance will be used.

    Returns:
        StreamSubscriber: An instance set up to listen to Redis pub/sub streams.
    """
    cfg = Config().get('cache.redis')
    return StreamSubscriber(manager=manager, host=cfg.get('host'), port=cfg.get('port'))


async def store_prompt(redis: AsyncRedis, conv_id: str, prompt_data: Dict[str, List[Dict[str, str]]]):
    """
    Store incoming prompt data into Redis for processing.

    Pushes the serialized JSON payload onto the Redis list under the "process:messages_batch" key.

    Args:
        redis (AsyncRedis): The Redis client to use for storing the data.
        conv_id (str): The conversation ID associated with the prompt.
        prompt_data (Dict[str, List[Dict[str, str]]]): The prompt content to store.
    """
    try:
        await redis.rpush("process:messages_batch", json.dumps(prompt_data))
    except Exception as e:
        logger.error(f"Failed to store prompt for {conv_id}: {e}", exc_info=True)