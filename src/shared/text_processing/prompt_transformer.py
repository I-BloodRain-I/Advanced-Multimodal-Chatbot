"""
Message transformation utilities for chat format conversion.

This module provides the PromptTransformer class with static methods for
converting between different message formats used throughout the system.
It handles message history concatenation, prompt string formatting, and
various message filtering and extraction operations.

The transformer supports both manual prompt formatting and tokenizer-based
chat templates for optimal compatibility with different language models.
"""

from typing import Dict, Optional, List

from transformers import AutoTokenizer
from core.entities import Message
from core.entities.types import MessageHistory

class PromptTransformer:
    """
    Utility class for transforming chat messages into different formats suitable for model input.
    Includes methods to concatenate new messages with chat histories and to format messages as prompt strings.
    """

    @classmethod
    def concat_messages_with_histories(cls,
                                       message: Message,
                                       history: MessageHistory) -> MessageHistory:
        """
        Appends a single user message to its corresponding history.

        Args:
            message: The latest message to append.
            history: The chat history.

        Returns:
            Updated history with the new message added.
        """
        if message:
            history.append(message)
        return history

    @classmethod
    def concat_messages_with_histories_batch(cls,
                                             messages_batch: MessageHistory,
                                             histories_batch: List[MessageHistory]) -> List[MessageHistory]:
        """
        Appends a batch of user messages to their corresponding chat histories.

        Args:
            messages_batch: List of latest messages from each conversation.
            histories_batch: List of message histories for each conversation.

        Returns:
            Updated list of message histories with the new messages added.

        Raises:
            ValueError: If the number of messages does not match the number of histories.
        """
        if len(messages_batch) != len(histories_batch):
            raise ValueError(f"Mismatch: {len(messages_batch)} messages vs {len(histories_batch)} histories")

        # Append each message to its respective history
        return [cls.concat_messages_with_histories(message, history)
                for message, history in zip(messages_batch, histories_batch)]

    @classmethod
    def format_messages_to_str(cls, 
                               messages: MessageHistory, 
                               tokenizer: Optional[AutoTokenizer] = None) -> str:
        """
        Converts a list of messages into a formatted string prompt, optionally using a tokenizer's chat template.

        Args:
            messages: The messages to convert into a prompt.
            tokenizer: Optional tokenizer for chat template formatting.

        Returns:
            A prompt string representing the sequence of messages.
        """
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            # Use the tokenizer's built-in chat formatting if available
            messages_ = cls.convert_messages_to_dict([messages])[0]
            return tokenizer.apply_chat_template(messages_, 
                                                tokenize=False, 
                                                add_special_tokens=True,
                                                add_generation_prompt=True)
        # Fallback: manually construct prompt string by role
        prompt_parts = []
        for message in messages or []:
            role = message.role
            content = message.content

            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")

        return "\n".join(prompt_parts)

    @classmethod
    def format_messages_to_str_batch(cls, 
                                     messages_batch: List[MessageHistory], 
                                     tokenizer: Optional[AutoTokenizer] = None) -> List[str]:
        """
        Converts a batch of message lists into a list of formatted prompt strings.

        Args:
            messages_batch: A batch of message sequences.
            tokenizer: Optional tokenizer for chat template formatting.

        Returns:
            A list of prompt strings, one for each batch of messages.
        """
        return [cls.format_messages_to_str(message, tokenizer) for message in messages_batch]
    
    @classmethod
    def get_messages_by_role(cls, messages_batch: List[MessageHistory], role: str) -> List[MessageHistory]:
        """
        Filters a batch of message histories, returning only messages with the specified role.

        Args:
            messages_batch: A list of message histories, where each history is a list of Message objects.
            role: The role to filter by ('user', 'assistant', or 'system').

        Returns:
            A list of message histories with only messages matching the given role.
        """
        if role not in ['user', 'assistant', 'system']:
            raise
        return [[message for message in messages if message.role == role] 
                for messages in messages_batch]

    @classmethod
    def get_content_from_messages(cls, messages_batch: List[MessageHistory], separator: str = '\n'):
        """
        Extracts and concatenates message content from a batch of message histories.

        Args:
            messages_batch: A list of message histories (already filtered if needed).
            separator: String used to join message contents (default is newline).

        Returns:
            A list of concatenated message contents per conversation.
        """
        return [separator.join([message.content for message in messages])
                for messages in messages_batch]

    @classmethod
    def convert_messages_to_dict(cls, messages_batch: List[MessageHistory]) -> List[List[Dict[str, str]]]:
        """
        Converts a batch of message histories into lists of dictionaries for serialization.

        Args:
            messages_batch: A batch of message sequences.

        Returns:
            A nested list of dictionaries representing each message.
        """
        return [[messages.model_dump() for messages in histories] for histories in messages_batch]