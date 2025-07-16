from typing import Dict
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Singleton class that manages active WebSocket connections.

    Allows adding, removing, and sending messages to WebSocket clients
    identified by a conversation ID.
    """
    _instance = None

    def __new__(cls):
        # Enforce singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.active_connections = {}
        return cls._instance

    async def connect(self, websocket: WebSocket, conv_id: str):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket (WebSocket): The WebSocket connection to register.
            conv_id (str): Unique identifier for the conversation/client.
        """
        await websocket.accept()
        self.active_connections[conv_id] = websocket
        logger.info(f"WebSocket connection established for conv_id: {conv_id}")
    
    def disconnect(self, conv_id: str):
        """
        Remove a WebSocket connection from the active registry.

        Args:
            conv_id (str): The conversation ID whose connection should be removed.
        """
        if conv_id in self.active_connections:
            del self.active_connections[conv_id]
            logger.info(f"WebSocket connection removed for conv_id: {conv_id}")
    
    async def send_message(self, message: str, conv_id: str) -> bool:
        """
        Send a message to the WebSocket client associated with the given conv_id.

        Args:
            message (str): The message to send.
            conv_id (str): The recipient's conversation ID.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        websocket = self.active_connections.get(conv_id)
        if not websocket:
            logger.warning(f"Attempted to send message to inactive conv_id: {conv_id}")
            return False
        
        try:
            await websocket.send_text(message)
            return True
        except Exception as e:
            # Remove faulty connection if sending fails
            logger.error(f"Failed to send message to conv_id: {conv_id}: {e}", exc_info=True)
            self.disconnect(conv_id)
            return False