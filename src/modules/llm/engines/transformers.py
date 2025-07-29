"""
Transformers-based LLM engine implementation for the TensorAlix Agent AI system.

This module provides a concrete implementation of the LLMEngineBase interface
using HuggingFace Transformers library. It handles model loading with various
quantization options, tokenizer management, and streaming text generation.

The engine supports different data types including INT4/INT8 quantization
via BitsAndBytes and standard PyTorch data types (float16, float32, bfloat16).
"""

from typing import AsyncGenerator, List, Optional
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.entities.enums import ModelDType
from core.entities.types import MessageHistory
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig
from modules.llm.engines import LLMEngineBase
from modules.llm.components import TransformersTextStreamer
from shared.utils import get_torch_device, get_bitsandbytes_config_for_dtype, to_torch_dtype

logger = logging.getLogger(__name__)


class TransformersEngine(LLMEngineBase):
    """
    HuggingFace Transformers-based implementation of LLM engine.
    
    This class provides text generation capabilities using HuggingFace
    Transformers library with support for various quantization schemes,
    streaming generation, and batch processing.
    
    Args:
        model: Pre-loaded HuggingFace causal language model
        tokenizer: Pre-loaded tokenizer for text encoding/decoding
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        super().__init__(logger=logger)
        self.model = model
        self.tokenizer = tokenizer
        self._streamer = TransformersTextStreamer(model, tokenizer)
        self._device = model.device
        logger.info("TransformerEngine loaded successfully.")
    
    @classmethod
    def from_config(cls, config: LLMEngineConfig) -> 'TransformersEngine':
        extra_data = config.extra_data
        try:
            device = get_torch_device(extra_data.get('device', 'cuda'))
            model = cls.load_model(config.model_name, config.dtype, device)
            tokenizer = cls.load_tokenizer(config.model_name)
            return TransformersEngine(model, tokenizer)
        except Exception:
            logger.error(f"Failed to initialize LLM", exc_info=True)
            raise

    @classmethod
    def load_model(cls, model_name: str, dtype: ModelDType, device: torch.device) -> AutoModelForCausalLM:
        """
        Loads the model with support for dtype-based quantization and device mapping.
        """
        quant_config = None
        torch_dtype = None
        if dtype in [ModelDType.INT4, ModelDType.INT8]:
            quant_config = get_bitsandbytes_config_for_dtype(dtype)
        else:
            torch_dtype = to_torch_dtype(dtype)

        try:
            model_path = cls.get_models_dir() / model_name
            if model_path.exists():
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), 
                    quantization_config=quant_config,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    local_files_only=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    quantization_config=quant_config,
                    torch_dtype=torch_dtype,
                    device_map=device
                )
            model.eval()
            return model
        except Exception:
            logger.error(f"Error loading model {model_name}", exc_info=True)
            raise

    @classmethod
    def load_tokenizer(cls, tokenizer_name: str) -> AutoTokenizer:
        """
        Loads the tokenizer and ensures it has a pad token.
        """
        try:
            tokenizer_path = cls.get_models_dir() / tokenizer_name
            if tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token; using eos_token as pad_token.")
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            logger.error(f"Error loading tokenizer from path {tokenizer_name}", exc_info=True)
            raise
        
    def create_generator(self, 
                         messages: MessageHistory,
                         config: Optional[LLMGenerationConfig] = None,
                         **kwargs) -> AsyncGenerator[str, None]:
        """
        Create a streaming text generator for a single conversation.
        
        Refer to :meth:`TransformersTextStreamer.create_generator` for more details.
        """
        return self._streamer.create_generator(messages, config, **kwargs)

    def create_batch_generators(self, 
                                messages_batch: List[MessageHistory], 
                                configs: Optional[List[LLMGenerationConfig]] = None,
                                **kwargs) -> List[AsyncGenerator[str, None]]:
        """
        Create multiple streaming text generators for batch processing.
        
        Refer to :meth:`TransformersTextStreamer.create_batch_generators` for more details.
        """
        # Validate and filter batch parameters
        validated_batch_params = self._validate_batch_parameters(messages_batch, **kwargs)
        return self._streamer.create_batch_generators(messages_batch, configs, **validated_batch_params)