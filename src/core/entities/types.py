"""
Core type definitions for the TensorAlix Agent AI system.

This module defines the fundamental data structures and type aliases used throughout
the system for representing conversations, messages, documents, and AI responses.
It includes Pydantic models for data validation and serialization.
"""

from typing import AsyncGenerator, TypeAlias, Union, Literal, Dict, Optional, List, Any

from numpy.typing import NDArray
from torch import Tensor
from pydantic import BaseModel, ConfigDict

from .enums import TaskType

# Alias for all types of embeddings
EmbeddingArray: TypeAlias = Union[List[float], List[List[float]], NDArray, Tensor]


class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role: The sender of the message.
        content: The textual content of the message.
    """
    role: Literal["system", "user", "assistant"]
    content: str


# Alias for a list of messages in a conversation
MessageHistory: TypeAlias = List[Message]


class TaskBatch(BaseModel):
    """
    Represents a batch of tasks to be processed, including conversation histories and their embeddings.

    Attributes:
        task: The type of task being performed (e.g., IMAGE_GEN, TEXT_GEN).
        conv_ids: Identifiers for each conversation.
        histories: The message histories for each conversation.
        embeddings: Precomputed embeddings corresponding to the conversations.
    """
    task: TaskType
    conv_ids: List[str]
    histories: List[List[Message]]
    embeddings: List[Optional[Any]]


class Conversation(BaseModel):
    """
    Represents a single conversation and its message history.

    Attributes:
        conv_id: Unique identifier for the conversation.
        history: The ordered list of messages in the conversation.
    """
    conv_id: str
    history: List[Message]


class ConversationBatch(BaseModel):
    """
    Represents a batch of conversations.

    Attributes:
        conv_ids: Identifiers for each conversation.
        histories: Message histories for each conversation.
    """
    conv_ids: List[str]
    histories: List[List[Message]]


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document with associated embeddings.

    Attributes:
        document_id: ID of the document this chunk belongs to.
        content: The text content of the chunk.
        embeddings: Vector representation of the chunk content.
    """
    document_id: str
    content: str
    embeddings: Optional[Union[Tensor, NDArray]]

    # Allow non-standard types like Tensor and NDArray
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RagDocument(BaseModel):
    """
    Represents a full document composed of multiple chunks, along with metadata.

    Attributes:
        id: Unique identifier for the document.
        chunks: List of content chunks making up the document.
        metadata: Arbitrary metadata associated with the document.
    """
    id: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]


class AgentResponse(BaseModel):
    """
    Represents a response from the agent, which could be text, a stream, or an image.

    Attributes:
        conv_id: ID of the conversation the response belongs to.
        type: Type of the response content.
        content: The response content itself, either as a string or async stream.
    """
    conv_id: str
    type: Literal['stream', 'text', 'image']
    content: Union[str, AsyncGenerator[str, None]]

    # Allow non-standard types like AsyncGenerator
    model_config = ConfigDict(arbitrary_types_allowed=True)