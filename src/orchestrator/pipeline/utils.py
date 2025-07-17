from typing import Dict, List, Optional
from collections import defaultdict
import logging

from core.entities import TaskType, Message, TaskBatch
from core.entities.types import EmbeddingArray

logger = logging.getLogger(__name__)


def group_conversations_by_task(
        messages: List[List[Message]],
        task_types: List[TaskType], 
        conv_ids: List[str],
        embeddings: Optional[EmbeddingArray] = None) -> Dict[TaskType, TaskBatch]:
    """
    Groups (conversation_id, prompt, embedding) tuples into batches based on their task type.

    Args:
        messages (List[List[Message]]): A list of message histories, each a list of Message objects.
        task_types (List[TaskType]): List of task types corresponding to each conversation.
        conv_ids (List[str]): List of unique conversation IDs.
        embeddings (Optional[EmbeddingArray]): Optional list of embeddings associated with each message.
                                               If not provided, defaults to None.

    Returns:
        Dict[TaskType, TaskBatch]: A dictionary mapping each task type to its corresponding TaskBatch object.
    """

    # Validate that all input lists are the same length
    if not (len(messages) == len(task_types) == len(conv_ids)):
        logger.error(
            f"Input list lengths mismatch: messages({len(messages)}), "
            f"task_types({len(task_types)}), conv_ids({len(conv_ids)}),", 
            exc_info=True)
        raise ValueError("All input lists must be of the same length.")

    # If embeddings aren't provided, fill with None for consistency
    if embeddings is None:
        embeddings = [None] * len(conv_ids)

    # Group each entry into a bucket by task type
    batches = defaultdict(lambda: {"conv_ids": [], "histories": [], "embeddings": []})

    for conv_id, history, task_type, embed in zip(conv_ids, messages, task_types, embeddings):
        batch = batches[task_type]
        batch["conv_ids"].append(conv_id)
        batch["histories"].append(history)
        batch["embeddings"].append(embed)

    return {
        task: TaskBatch(
            task=task, 
            conv_ids=batch["conv_ids"],
            histories=batch["histories"],
            embeddings=batch["embeddings"]
        ) for task, batch in batches.items()
    }