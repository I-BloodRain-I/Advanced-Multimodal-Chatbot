"""
Text embedding generation module.

Provides a singleton Embedder class that uses SentenceTransformer models to convert
text into dense vector embeddings. Supports batch processing, GPU acceleration,
and different output formats (NumPy arrays or PyTorch tensors).
"""
from typing import Union, List
import logging

import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from shared.utils import get_torch_device

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton class for embedding text using a SentenceTransformer model.
    
    This class lazily loads a SentenceTransformer model and provides a method
    for extracting embeddings from text prompts.

    Args:
        model_name: Name of the pretrained SentenceTransformer model to load.
        device_name: Torch device identifier (e.g., 'cuda' or 'cpu').
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Ensures only one instance of Embedder exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 device_name: str = 'cuda'):
        # Prevent reinitialization on repeated instantiations
        if self._initialized:
            return
        
        self.device = get_torch_device(device_name)
        self.model = self._load_model(model_name)
        self._initialized = True

    def _load_model(self, model_name: str) -> SentenceTransformer:        
        """Loads and initializes a SentenceTransformer model in evaluation mode."""
        try:
            model = SentenceTransformer(model_name, device=self.device)
            model.eval().requires_grad_(False)
            return model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {model_name}", exc_info=True)
            raise

    def extract_embeddings(self, 
                           prompt: Union[str, List[str]], 
                           batch_size: int = 32,
                           show_progress_bar: bool = False,
                           convert_to_tensor: bool = False) -> Union[NDArray, torch.Tensor]:
        """
        Encodes a text prompt or list of prompts into dense vector embeddings.

        Args:
            prompt: The input text(s) to embed.
            batch_size: Batch size for processing inputs.
            show_progress_bar: Whether to display a progress bar during encoding.
            convert_to_tensor: If True, returns a torch.Tensor instead of a NumPy array.

        Returns:
            The resulting embeddings as NumPy array or torch.Tensor.
            
        Raises:
            NotImplementedError: If model type is not supported.
        """
        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(prompt, 
                                     batch_size=batch_size,
                                     convert_to_tensor=convert_to_tensor, 
                                     device=self.device,
                                     show_progress_bar=show_progress_bar)
        else:
            raise NotImplementedError(f"Code for the model type {type(self.model)} is not implemented")