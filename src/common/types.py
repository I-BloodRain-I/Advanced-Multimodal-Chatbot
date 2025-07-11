from typing import TypedDict, NamedTuple, Literal, Dict, Any

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class RagDocument(NamedTuple):
    content: str
    metadata: Dict[str, Any]