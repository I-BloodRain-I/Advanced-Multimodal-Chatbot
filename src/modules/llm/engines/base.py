"""
Abstract base class for LLM inference engines.

This module defines the LLMEngineBase abstract class that provides a unified
interface for different language model inference backends such as vLLM,
Transformers, and TensorRT-LLM. It standardizes model loading, tokenization,
and text generation across different implementations.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.entities.types import MessageHistory
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig
from shared.config import Config

class LLMEngineBase(ABC):
    """
    Abstract base class defining the interface for text generation engines.
    Provides a unified API for different LLM inference backends (vLLM, transformers, TensorRT-LLM).
    Concrete implementations handle framework-specific model loading, tokenization, and generation.
    """
    _MODELS_DIR = None
    
    def __init__(self, logger = None):
        super().__init__()
        self.__logger = logger or logging.getLogger(self.__class__.__name__)
    
    @classmethod
    def get_models_dir(cls) -> Path:
        """Get the models directory path, initializing it if needed."""
        if cls._MODELS_DIR is None:
            models_dir = Config.get('models_dir', 'models')
            cls._MODELS_DIR = Path(models_dir)
        return cls._MODELS_DIR

    @classmethod
    @abstractmethod
    def from_config(cls, config: LLMEngineConfig) -> 'LLMEngineBase':
        """Load and initialize the engine instance from `LLMEngineConfig` configuration"""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls) -> Any:
        """
        Load and initialize the language model instance.
        Implementation varies by backend: vLLM AsyncLLMEngine, HuggingFace AutoModelForCausalLM, etc.
        """
        pass

    @classmethod
    @abstractmethod
    def load_tokenizer(cls) -> Any:
        """
        Load the tokenizer for text-to-token encoding/decoding.
        Handles special tokens (BOS, EOS, padding) and vocabulary mapping.
        """
        pass

    def _validate_batch_parameters(self, messages_batch: List[MessageHistory], **kwargs) -> Dict[str, Any]:
        """
        Validate and filter batch parameters to ensure consistency.
        
        Args:
            messages_batch: List of message histories for batch processing
            **kwargs: Batch parameters where each value should be a list
            
        Returns:
            Validated parameters with consistent list lengths
        """
        if not messages_batch:
            error_txt = "Messages batch cannot be empty"
            self.__logger.error(error_txt, exc_info=True)
            raise ValueError(error_txt)
        
        batch_size = len(messages_batch)
        validated_params = {}
        
        for param_name, param_values in kwargs.items():
            # Skip non-list parameters
            if not isinstance(param_values, (list, set, tuple)):
                continue
            
            # Ensure parameter list matches batch size
            if len(param_values) != batch_size:
                self.__logger.warning(f"Parameter '{param_name}' length ({len(param_values)}) "
                                      f"doesn't match batch size ({batch_size}). Skipping.")
                continue
            
            validated_params[param_name] = param_values
        
        return validated_params

    @abstractmethod
    def create_generator(self, 
                         messages: MessageHistory, 
                         config: LLMGenerationConfig) -> AsyncGenerator[str, None]:
        """
        Asynchronous streaming text generation.
        Yields partial responses as tokens are generated, reducing perceived latency.
        """
        pass

    @abstractmethod
    def create_batch_generators(self, 
                                messages_batch: List[MessageHistory], 
                                configs: Optional[List[LLMGenerationConfig]]) -> List[AsyncGenerator[str, None]]:
        """
        Batch streaming generation for multiple conversations.
        Returns independent async generators for concurrent streaming.
        """
        pass