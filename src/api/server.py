import logging
import json

from fastapi import WebSocket, WebSocketDisconnect

from common.logging import setup_logging
setup_logging()

from shared.config import Config
from .manager import ConnectionManager
from .utils import create_app, start_server, store_prompt, load_listener

logger = logging.getLogger(__name__)

# Instantiate a singleton connection manager to handle active WebSocket connections
manager = ConnectionManager()

# Load the Redis stream listener tied to the manager
listener = load_listener(manager)

# Create the FastAPI app with Redis integration on startup
app = create_app()


@app.websocket("/ws/{conv_id}")
async def websocket_endpoint(websocket: WebSocket, conv_id: str):
    """
    Handle incoming WebSocket connections for a specific conversation ID.

    Establishes a connection, subscribes to Redis messages for that conversation,
    and continuously receives prompt data from the client to store in Redis.

    Args:
        websocket (WebSocket): The WebSocket connection to the client.
        conv_id (str): Unique identifier for the conversation.
    """
    await manager.connect(websocket, conv_id)
    await listener.subscribe(conv_id)            # Start listening to Redis pub/sub for this conv_id
    redis = websocket.app.state.redis            # Access the Redis client from app state

    try:
        while True:
            # Receive JSON with conv_id and prompt
            data = await websocket.receive_text()
            try:
                prompt_data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {conv_id}: {data}")
                continue

            # Store the parsed prompt data into Redis for downstream processing
            await store_prompt(redis, conv_id, prompt_data)

    except WebSocketDisconnect:
        manager.disconnect(conv_id)
        listener.unsubscribe(conv_id)

    except Exception:
        logger.error(f"Unexpected error in WebSocket handler for {conv_id}", exc_info=True)
        manager.disconnect(conv_id)
        listener.unsubscribe(conv_id)

if __name__ == "__main__":
    server_cfg = Config().get('server')
    start_server(app, server_cfg.get('host'), server_cfg.get('port'))
