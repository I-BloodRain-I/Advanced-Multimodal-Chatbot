"""
Large Language Model (LLM) management module.

Provides a singleton LLM class that handles language model loading, caching, and text generation
using multiple backend engines (Transformers, vLLM). Supports streaming generation, conversation
management, and automatic model caching for improved performance.
"""
import json
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Union
import logging

from core.entities.enums import ModelDType
from core.entities.types import MessageHistory
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig
from modules.llm.engines import TransformersEngine, VLLMEngine
from shared.config import Config

logger = logging.getLogger(__name__)


class LLM:
    """
    Singleton class for managing a causal language model (LLM) with support for
    multiple inference engines, configurable generation parameters, and streaming generation interfaces.
    
    Args:
        engine_name: The inference engine to use.
        engine_config: Configuration object containing model settings.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, engine_name: Literal['transformers', 'vllm'], engine_config: LLMEngineConfig):
        if self._initialized:
            return # Avoid reinitializing 

        self.config = engine_config
        try:
            self._cache_model(engine_config.model_name)  # Save for future offline use
            self.engine = self.load_engine(engine_name, engine_config)
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        self._initialized = True    
    
    def _cache_model(self, model_name: str):
        """
        Saves the model and tokenizer to disk if not already saved.

        Removes quantization_config from config.json to allow flexibility on reload.
        """
        try:
            models_dir = Path(Config.get('models_dir'))
            model_path = models_dir / model_name
            if not model_path.exists():
                engine_config = LLMEngineConfig(model_name=model_name, dtype=ModelDType.FLOAT16, device='cpu')
                engine = TransformersEngine.from_config(engine_config)

                engine.model.save_pretrained(str(model_path), safe_serialization=True)
                engine.tokenizer.save_pretrained(str(model_path), safe_serialization=True)
                del engine

                # Remove quantization_config so it can be overridden during loading
                config_path = model_path / 'config.json'  
                if config_path.exists():
                    with open(config_path, 'r+') as f:
                        cfg = json.load(f)
                        if 'quantization_config' in cfg:
                            cfg.pop('quantization_config')
                            f.seek(0)
                            json.dump(cfg, f, indent=4)
                            f.truncate()
        except Exception:
            logger.error("Failed to cache model", exc_info=True)
            raise

    @classmethod
    def load_engine(cls, name: str, config: LLMEngineConfig) -> Union[TransformersEngine, VLLMEngine]:
        if name == 'transformers':
            return TransformersEngine.from_config(config)
        elif name == 'vllm':
            return VLLMEngine.from_config(config)
        else:
            error_msg = f"Engine name must be either: [transformers, vllm]"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def generate(self, 
                 messages: MessageHistory,
                 config: Optional[LLMGenerationConfig] = None,
                 **kwargs) -> AsyncGenerator[str, None]:
        """
        Create a streaming text generator for a single conversation.
        
        Args:
            messages: Conversation history to use as prompt context
            config: Optional generation configuration object
            
        Returns:
            Stream of generated text tokens
            
        Raises:
            ValueError: If messages is empty or invalid
            Exception: If model generation fails
        """
        if config is None:
            generation_args = Config.get('llm.generation')
            config = LLMGenerationConfig(**generation_args)
        return self.engine.create_generator(messages, config, **kwargs)

    def generate_batch(self, 
                       messages_batch: List[MessageHistory], 
                       configs: Optional[List[LLMGenerationConfig]] = None,
                       **kwargs) -> List[AsyncGenerator[str, None]]:
        """
        Create multiple streaming text generators for batch processing.
        
        Args:
            messages_batch: List of conversation histories to process
            configs: Optional list of generation configs (one per message history)
            
        Returns:
            List of text stream generators
            
        Raises:
            ValueError: If input validation fails
            Exception: If any generator creation fails
        """
        if configs is None:
            generation_args = Config.get('llm.generation')
            configs = [LLMGenerationConfig(**generation_args)] * len(messages_batch)
        return self.engine.create_batch_generators(messages_batch, configs, **kwargs)