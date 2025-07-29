"""
Task routing module for conversation classification.

This module provides the Router class that uses a trained neural network classifier
to determine what type of AI task should be performed for each conversation based
on embedding representations of user prompts.
"""

from typing import List, Optional, Tuple

from numpy import ndarray

import torch
from core.entities import EmbeddingArray, TaskType
from .model import TaskClassifier


class Router:
    """
    Routes conversations to task types using a trained classifier model.

    This class takes conversation embeddings and predicts the most likely task type
    using a `TaskClassifier` neural network.

    Args:
        model: A trained classifier model for task routing.
    """
    def __init__(self, model: TaskClassifier):
        self.model = model
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def route(self, 
               embeddings: EmbeddingArray, 
               return_probs = False) -> Tuple[List[TaskType], Optional[torch.Tensor]]:
        """
        Routes embeddings to task types using the classifier.

        Args:
            embeddings: A batch of embeddings representing prompts or conversations.
            return_probs: If True, also returns the confidence scores (max softmax probs) for each prediction. Defaults to False.

        Returns:
            - List of predicted TaskType enums.
            - Optionally, tensor of softmax probabilities for the chosen class.
        """
        embeddings = self._normalize_embeddings(embeddings).to(self.device)

        output = self.model(embeddings)
        # Softmax to get probabilities, then take the max along class dimension
        probs, indexes = torch.nn.functional.softmax(output, dim=-1).max(dim=-1)
        # Convert predicted class indices to TaskType enums
        task_types = [TaskType(task_id) for task_id in indexes.cpu().numpy()]
        if return_probs:
            return task_types, probs 
        else:
            return task_types, None
        
    def _normalize_embeddings(self, embeddings: EmbeddingArray) -> torch.Tensor:
        """Converts input embeddings to a PyTorch tensor if needed."""
        if isinstance(embeddings, torch.Tensor):
            return embeddings.type(torch.float)
        if isinstance(embeddings, ndarray):
            return torch.from_numpy(embeddings).type(torch.float)
        elif not isinstance(embeddings, torch.Tensor):
            return torch.tensor(embeddings, dtype=torch.float)
        