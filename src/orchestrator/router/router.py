import torch
from typing import List, Optional, Tuple, Union

from core.entities.types import EmbeddingArray

from .model import TaskClassifier
from numpy import ndarray

from core.entities import TaskType

class Router:
    """
    Routes conversations to task types using a trained classifier model.

    This class takes conversation embeddings and predicts the most likely task type
    using a `TaskClassifier` neural network.

    Args:
        model (TaskClassifier): A trained classifier model for task routing.
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
            embeddings (EmbeddingArray): A batch of embeddings representing prompts or conversations.
            return_probs (bool): If True, also returns the confidence scores (max softmax probs) for each prediction.

        Returns:
            Tuple[List[TaskType], Optional[torch.Tensor]]: 
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
        
    def _normalize_embeddings(self, embeddings: EmbeddingArray):
        """Converts input embeddings to a PyTorch tensor if needed."""
        if isinstance(embeddings, ndarray):
            return torch.from_numpy(embeddings)
        elif not isinstance(embeddings, torch.Tensor):
            return torch.tensor(embeddings)