from contextlib import asynccontextmanager
import logging
from typing import Dict, List, Optional
import json

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.manager import ConnectionManager
from api.stream_subscriber import StreamSubscriber
from common.utils import require_env_var
from shared.cache import AsyncRedis
from shared.config import Config

logger = logging.getLogger(__name__)


def create_app(mount_path: str = '/static') -> FastAPI:
    """
    Create and configure a FastAPI application with Redis lifecycle hooks.

    Initializes a Redis client at startup and attaches it to the app state.
    Ensures clean Redis shutdown on app exit.

    Args:
        mount_path: Path to mount static files directory.

    Returns:
        A configured FastAPI application instance.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis = AsyncRedis(host=require_env_var('REDIS_HOST'), 
                           port=int(require_env_var('REDIS_PORT')), 
                           decode_responses=False)
        logger.info("Redis connection established.")

        # Make Redis available via app.state
        app.state.redis = redis
        yield  # Application runs here
        
        # Disconnect Redis cleanly on shutdown
        await redis.close()
        logger.info("Redis client closed.")

    app = FastAPI(lifespan=lifespan)
    app.mount(mount_path, StaticFiles(directory=Config().get('webapp_dir')), name=mount_path)

    return app


def start_server(app: FastAPI, host: str, port: int, **kwargs):
    """
    Start the FastAPI application using Uvicorn.

    Args:
        app: The FastAPI application to serve.
        host: Host IP or name to bind the server to.
        port: Port number to listen on.
        **kwargs: Optional additional arguments to pass to uvicorn.run.
    """
    uvicorn.run(app, host=host, port=port, **kwargs)


def load_listener(manager: Optional[ConnectionManager] = None) -> StreamSubscriber:
    """
    Create and return a Redis stream subscriber for handling message broadcasting.

    Args:
        manager: The connection manager for WebSocket clients.
                 If None, a new instance will be used.

    Returns:
        An instance set up to listen to Redis pub/sub streams.
    """
    return StreamSubscriber(manager=manager, 
                            host=require_env_var('REDIS_HOST'), 
                            port=int(require_env_var('REDIS_PORT')))


async def store_prompt(redis: AsyncRedis, conv_id: str, prompt_data: Dict[str, List[Dict[str, str]]]):
    """
    Store incoming prompt data into Redis for processing.

    Pushes the serialized JSON payload onto the Redis list under the "process:messages_batch" key.

    Args:
        redis: The Redis client to use for storing the data.
        conv_id: The conversation ID associated with the prompt.
        prompt_data: The prompt content to store.
        
    Raises:
        Exception: If storing the prompt data fails.
    """
    try:
        await redis.rpush("process:messages_batch", json.dumps(prompt_data))
    except Exception as e:
        logger.error(f"Failed to store prompt for {conv_id}: {e}", exc_info=True)