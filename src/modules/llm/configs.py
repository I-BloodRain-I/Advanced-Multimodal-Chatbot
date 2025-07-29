"""
Configuration classes for LLM engine and generation settings.

This module provides configuration classes that encapsulate settings for
language model initialization and text generation. These configurations
provide a clean interface for managing model parameters and generation
settings across different engines and inference backends.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

from core.entities.enums import ModelDType


@dataclass
class LLMGenerationConfig:
    """
    Configuration class for text generation parameters.
    
    This class encapsulates all the key parameters that control the behavior
    of the text generation process, making it easy to create reusable configs
    and pass them between different generation calls.

    Attributes:
        temperature: Controls randomness in text generation (0.0-2.0). Lower=focused, higher=creative. Default: 0.8
        top_p: Nucleus sampling parameter (0.0-1.0). Controls token diversity. Default: 0.9  
        max_tokens: Maximum tokens to generate. Default: 1024
        priority: Request priority for queue management (0-10). Higher=more priority. Default: 0
        **kwargs: Additional parameters stored in extra_data
    """
    
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 1024
    priority: int = 0
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, temperature: float = 0.8, top_p: float = 0.9, max_tokens: int = 1024, priority: int = 0, **kwargs):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.priority = priority
        self.extra_data = kwargs
    
    @property
    def max_new_tokens(self) -> int:
        """Alias for max_tokens to maintain compatibility"""
        return self.max_tokens
    

@dataclass
class LLMEngineConfig:
    """
    Configuration class for LLM engine initialization and model loading.
    
    This class manages the core settings required to initialize and configure
    a language model engine, including model selection and inference parameters.
    Used by the LLM module to set up the appropriate model backend.

    Attributes:
        model_name: Name or path of the language model to use
        dtype: Data type for model inference (e.g. AUTO, FLOAT16, BFLOAT16). Default: AUTO
        **kwargs: Additional engine-specific parameters stored in extra_data
    """
    
    model_name: str
    dtype: ModelDType = ModelDType.AUTO
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, model_name: str, dtype: ModelDType = ModelDType.AUTO, **kwargs):
        self.model_name = model_name
        self.dtype = dtype
        self.extra_data = kwargs