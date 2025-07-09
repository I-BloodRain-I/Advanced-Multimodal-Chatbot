from typing import List
from chatbot.types import Message

def concat_messages_with_histories(message: Message,
                                   history: List[Message]) -> List[Message]:
    """
    Appends a single user message to its corresponding history.

    Args:
        message (Message): The latest message to append.
        history (List[Message]): The chat history so far.

    Returns:
        List[Message]: Updated history with the new message added.
    """
    if message:
        history.append(message)
    else:
        history.append(history)

def concat_messages_with_histories_batch(messages_batch: List[Message],
                                         histories_batch: List[List[Message]]) -> List[List[Message]]:
    """
    Appends a batch of user messages to their corresponding chat histories.

    Args:
        messages_batch (List[Message]): List of latest messages from each conversation.
        histories_batch (List[List[Message]]): List of message histories for each conversation.

    Returns:
        List[List[Message]]: Updated list of message histories with the new messages added.

    Raises:
        ValueError: If the number of messages does not match the number of histories.
    """
    if len(messages_batch) != len(histories_batch):
         raise ValueError(f"Mismatch: {len(messages_batch)} messages vs {len(histories_batch)} histories")

    return [concat_messages_with_histories(message, history)
            for message, history in zip(messages_batch, histories_batch)]

def format_messages_to_str(messages: List[Message], tokenizer = None) -> str:
    """
    Converts a list of messages into a formatted string prompt, optionally using a tokenizer's chat template.

    Args:
        messages (List[Message]): The messages to convert into a prompt.
        tokenizer (Optional): Optional tokenizer with a chat template to format the messages.

    Returns:
        str: A prompt string representing the sequence of messages.
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        
        return tokenizer.apply_chat_template(messages, 
                                             tokenize=False, 
                                             add_special_tokens=True,
                                             add_generation_prompt=True)
    prompt_parts = []
    for message in messages or []:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "system":
            prompt_parts.append(f"System: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")

    return "\n".join(prompt_parts)

def format_messages_to_str_batch(messages_batch: List[List[Message]], tokenizer = None) -> List[str]:
    """
    Converts a batch of message lists into a list of formatted prompt strings.

    Args:
        messages_batch (List[List[Message]]): A batch of message sequences.
        tokenizer (Optional): Optional tokenizer with a chat template to format the messages.

    Returns:
        List[str]: A list of prompt strings, one for each batch of messages.
    """
    return [format_messages_to_str(message, tokenizer) for message in messages_batch]