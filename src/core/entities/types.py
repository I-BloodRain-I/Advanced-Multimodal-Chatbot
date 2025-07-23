from typing import AsyncGenerator, Union, Literal, Dict, Optional, List, Any

from numpy.typing import NDArray
from torch import Tensor
from pydantic import BaseModel, ConfigDict

from .enums import TaskType

# Alias for all types of embeddings
class EmbeddingArray: TypeAlias = Union[List[float], List[List[float]], NDArray, Tensor]

class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Args:
        role (Literal["system", "user", "assistant"]): The sender of the message.
        content (str): The textual content of the message.
    """
    role: Literal["system", "user", "assistant"]
    content: str

# Alias for a list of messages in a conversation
class MessageHistory: TypeAlias = List[Message]

class TaskBatch(BaseModel):
    """
    Represents a batch of tasks to be processed, including conversation histories and their embeddings.

    Args:
        task (TaskType): The type of task being performed (e.g., IMAGE_GEN, WEBSEARCH).
        conv_ids (List[str]): Identifiers for each conversation.
        histories (List[List[Message]]): The message histories for each conversation.
        embeddings (List[Optional[Any]]): Precomputed embeddings corresponding to the conversations.
    """
    task: TaskType
    conv_ids: List[str]
    histories: List[List[Message]]
    embeddings: List[Optional[Any]]

class Conversation(BaseModel):
    """
    Represents a single conversation and its message history.

    Args:
        conv_id (str): Unique identifier for the conversation.
        history (List[Message]): The ordered list of messages in the conversation.
    """
    conv_id: str
    history: List[Message]

class ConversationBatch(BaseModel):
    """
    Represents a batch of conversations.

    Args:
        conv_ids (List[str]): Identifiers for each conversation.
        histories (List[List[Message]]): Message histories for each conversation.
    """
    conv_ids: List[str]
    histories: List[List[Message]]

class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document with associated embeddings.

    Args:
        document_id (str): ID of the document this chunk belongs to.
        content (str): The text content of the chunk.
        embeddings (Union[Tensor, NDArray]): Vector representation of the chunk content.
    """
    document_id: str
    content: str
    embeddings: Optional[Union[Tensor, NDArray]]

    # Allow non-standard types like Tensor and NDArray
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RagDocument(BaseModel):
    """
    Represents a full document composed of multiple chunks, along with metadata.

    Args:
        id (str): Unique identifier for the document.
        chunks (List[DocumentChunk]): List of content chunks making up the document.
        metadata (Dict[str, Any]): Arbitrary metadata associated with the document.
    """
    id: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]

class AgentResponse(BaseModel):
    """
    Represents a response from the agent, which could be text, a stream, or an image.

    Args:
        conv_id (str): ID of the conversation the response belongs to.
        type (Literal['stream', 'text', 'image']): Type of the response content.
        content (Union[str, AsyncGenerator[str, None]]): The response content itself, either as a string or async stream.
    """
    conv_id: str
    type: Literal['stream', 'text', 'image']
    content: Union[str, AsyncGenerator[str, None]]

    # Allow non-standard types like AsyncGenerator
    model_config = ConfigDict(arbitrary_types_allowed=True)