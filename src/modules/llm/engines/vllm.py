"""
vLLM-based LLM engine implementation for high-performance inference.

This module provides a concrete implementation of the LLMEngineBase interface
using the vLLM library for optimized large language model inference. It supports
asynchronous generation, efficient batching, and streaming text output with
improved throughput compared to standard transformers.

The engine handles model loading, request queuing, and token streaming while
leveraging vLLM's optimizations for production deployments.
"""

from typing import AsyncGenerator, List, Optional
import logging
import uuid

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams

from core.entities.enums import ModelDType
from core.entities.types import MessageHistory
from modules.llm.configs import LLMGenerationConfig, LLMEngineConfig
from shared.text_processing.prompt_transformer import PromptTransformer
from modules.llm.engines.base import LLMEngineBase

logger = logging.getLogger(__name__)


class VLLMEngine(LLMEngineBase):
    def __init__(self, model: AsyncLLMEngine, tokenizer: AutoTokenizer):
        super().__init__(logger=logger)
        self.model = model
        self.tokenizer = tokenizer
        logger.info("VLLMEngine loaded successfully.")
    
    @classmethod
    def from_config(cls, config: LLMEngineConfig) -> 'VLLMEngine':
        try:
            model = cls.load_model(config.model_name, config.dtype, **config.extra_data)
            tokenizer = cls.load_tokenizer(config.model_name)
            return VLLMEngine(model=model, tokenizer=tokenizer)
        except Exception:
            logger.error(f"Failed to initialize VLLMEngine")
            raise

    @classmethod
    def load_model(cls, model_name: str, dtype: ModelDType, **kwargs) -> AsyncLLMEngine:
        try:
            model_path = cls.get_models_dir() / model_name
            model_name = str(model_path) if model_path.exists() else model_name
            
            # Map our ModelDType enum to VLLM's expected dtype strings
            dtype_mapping = {
                ModelDType.AUTO: "auto",
                ModelDType.INT4: "auto",  # VLLM doesn't support int4 directly
                ModelDType.INT8: "auto",  # VLLM doesn't support int8 directly  
                ModelDType.BFLOAT16: "bfloat16",
                ModelDType.FLOAT16: "float16", 
                ModelDType.FLOAT32: "float32"
            }
            
            vllm_dtype = dtype_mapping.get(dtype, "auto")
            model_args = AsyncEngineArgs(model=model_name, dtype=vllm_dtype, **kwargs)
            model = AsyncLLMEngine.from_engine_args(model_args)
            return model
        except Exception:
            logger.error(f"Error loading model {model_name}")
            raise

    @classmethod
    def load_tokenizer(cls, tokenizer_name: str):
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
        except Exception as e:
            logger.error(f"Error loading tokenizer from path {tokenizer_name}: {e}")
            raise
      
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        """Create sampling parameters for text generation."""
        if 'max_new_tokens' in kwargs:
            kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        return SamplingParams(**kwargs)

    async def _extract_token_stream(self, generator: AsyncGenerator[RequestOutput, None]) -> AsyncGenerator[str, None]:
        """
        Extract and yield new tokens from the model output stream.
        
        This method maintains state to track cumulative text and only yields
        newly generated tokens, avoiding duplication.
        """
        cumulative_text = ""

        try:
            async for output in generator:
                # Extract the full generated text from the first output
                current_full_text = output.outputs[0].text
                
                # Calculate the new token by comparing with previous cumulative text
                new_token = current_full_text[len(cumulative_text):]
                
                # Update cumulative text for next iteration
                cumulative_text = current_full_text
                
                # Yield only the new token
                if new_token:  # Only yield if there's actually new content
                    yield new_token
                    
        except (IndexError, AttributeError):
            logger.error(f"Failed to extract tokens from generator output")
            raise

    def create_generator(self, 
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
        if not messages:
            error_txt = "Messages cannot be empty"
            logger.error(error_txt)
            raise ValueError(error_txt)
        
        # Use provided config or create default
        generation_config = config or LLMGenerationConfig()
        
        # Create params for new generator
        config_dict = {
            'temperature': generation_config.temperature,
            'top_p': generation_config.top_p,
            'max_new_tokens': generation_config.max_new_tokens,
        }
        final_params = generation_config.extra_data.copy()
        final_params.update(config_dict)
        
        try:
            prompt = PromptTransformer.format_messages_to_str(messages, self.tokenizer)
            sampling_params = self._create_sampling_params(**final_params)
            request_id = str(uuid.uuid4())
            
            vllm_generator = self.model.generate(
                prompt, 
                sampling_params=sampling_params,
                request_id=request_id,
                priority=generation_config.priority
            )
            
            # Return normalized token stream
            return self._extract_token_stream(vllm_generator)
            
        except Exception:
            logger.error("Failed to create generator")
            raise
    
    def create_batch_generators(self, 
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
        if not messages_batch:
            error_txt = "Messages batch cannot be empty"
            logger.error(error_txt)
            raise ValueError(error_txt)
        
        batch_size = len(messages_batch)
        
        # Validate configs list if provided
        if configs and len(configs) != batch_size:
            error_txt = f"Configs length ({len(configs)}) must match batch size ({batch_size})"
            logger.error(error_txt)
            raise ValueError(error_txt)
        
        # Validate and filter batch parameters
        validated_batch_params = self._validate_batch_parameters(messages_batch, **kwargs)
        
        generators = []
        
        for i, messages in enumerate(messages_batch):
            try:
                # Get config for this generation (if provided)
                config = configs[i] if configs else None
                
                # Extract parameters for this specific generation
                generation_kwargs = {}
                for param_name, param_values in validated_batch_params.items():
                    param_value = param_values[i]
                    if param_value is not None:  # Only include non-None values
                        # Convert plural parameter names to singular
                        singular_param_name = param_name.rstrip('s')
                        generation_kwargs[singular_param_name] = param_value
                
                # Create generator for this message history
                generator = self.create_generator(
                    messages=messages,
                    config=config,
                    **generation_kwargs
                )
                generators.append(generator)
                
            except Exception:
                logger.error(f"Failed to create generator for batch item {i}")
                raise
        
        return generators