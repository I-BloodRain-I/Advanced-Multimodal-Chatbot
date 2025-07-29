"""
Text streaming component for asynchronous token-by-token generation.

This module provides the TransformersTextStreamer class that handles streaming
text generation using HuggingFace Transformers. It supports custom stopping
criteria, asynchronous token generation, and proper resource cleanup for
real-time streaming responses.

The streamer manages tokenization, generation parameters, and token-level
streaming while handling various stopping conditions and error scenarios.
"""

import asyncio
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AsyncTextIteratorStreamer,
    StoppingCriteria, 
    StoppingCriteriaList
)

from core.entities.types import MessageHistory
from modules.llm.configs import LLMGenerationConfig
from shared.text_processing.prompt_transformer import PromptTransformer

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria to stop generation when specific tokens are encountered"""

    def __init__(self, stop_token_ids: List[int], stop_event: threading.Event):
        """
        Args:
            stop_token_ids: List of token IDs that trigger stopping.
            stop_event: Threading event to signal when generation should stop.
        """
        super().__init__()
        self.stop_token_ids = stop_token_ids
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs):
        """Return True if the last token matches a stop token ID or stop event is set."""
        # Check stop event first (for manual cancellation)
        if self.stop_event and self.stop_event.is_set():
            return True
    
        # Check if the last generated token is in our stop tokens
        return input_ids[0][-1].item() in self.stop_token_ids

class TransformersTextStreamer:
    """
    The text streamer for asynchronous token-by-token generation

    Args:
        model: The causal language model.
        tokenizer: The tokenizer to use.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        logger.info(f"TextStreamer initialized on device={self.device}")

    def _create_stopping_criteria(self) -> Tuple[Optional[StoppingCriteriaList], threading.Event]:
        """
        Creats a stopping criteria list based on predefined stop sequences.

        Returns:
            A list of stopping criteria or None if no valid sequences are found. 
            And threading event to signal when generation should stop.
        """
        logger.debug("Creating stopping criteria")

        stop_sequences = ["<|im_end|>", "<|endoftext|>", "<|end|>"]
        stop_token_ids = []

        stop_event = threading.Event()
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:    
            for seq in stop_sequences:
                try:
                    # Encode without special tokens to get the actual token IDs
                    token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
                    if token_ids:
                        stop_token_ids.extend(token_ids)
                        logger.debug(f"Added stop sequence '{seq}' with token IDs: {token_ids}")
                except Exception as e:
                    # Skip sequences that can't be encoded
                    logger.debug(f"Failed to encode stop sequence '{seq}': {e}")
                    continue
            
            if stop_token_ids:
                # Remove duplicates while preserving order
                stop_token_ids = list(dict.fromkeys(stop_token_ids))
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids, stop_event)])
                logger.debug(f"Created stopping criteria with {len(stop_token_ids)} stop tokens: {stop_token_ids}")
                return stopping_criteria, stop_event
            else:
                logger.debug("No valid stop tokens found, no stopping criteria created")
        else:
            logger.debug("No chat template found, no stopping criteria created")
        
        return None, stop_event

    async def _create_generator(self, messages: MessageHistory, generation_kwargs: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Asynchronously generates tokens one-by-one for a message sequence.

        Yields:
            Next generated token.
        """
        try:    
            formatted_prompt = PromptTransformer.format_messages_to_str(messages, self.tokenizer)
            logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
        except Exception as e:
            logger.error(f"Failed to format messages: {e}")
            raise
        
        try:
            encoded_inputs = self.tokenizer(
                formatted_prompt, 
                padding=False,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
        
            input_ids = encoded_inputs["input_ids"].to(self.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.device)

            logger.debug(f"Tokenized input - input_ids shape: {input_ids.shape}, "
                         f"attention_mask shape: {attention_mask.shape}")
        except Exception as e:
            logger.error(f"Failed to tokenize input: {e}")
            raise

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        })
        
        stopping_criteria, stop_event = self._create_stopping_criteria()
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        
        logger.debug(f"Generation parameters: max_new_tokens={generation_kwargs['max_new_tokens']}, "
                     f"temperature={generation_kwargs['temperature']}, stopping_criteria={stopping_criteria is not None}")

        # Start generation using asyncio.to_thread
        generation_task = asyncio.create_task(
            asyncio.to_thread(self.model.generate, **generation_kwargs)
        )
        
        logger.debug("Generation task started")
        
        token_count = 0
        try:
            async for token in streamer:
                if token:
                    token_count += 1
                    yield token
            
            # Wait for generation to complete
            await generation_task
            
        except Exception as e:
            logger.error(f"Error during token streaming: {e}")
            generation_task.cancel()
            raise

        finally:
            stop_event.set()
            streamer.end()
            if not generation_task.done():
                generation_task.cancel()
            logger.debug(f"Generation completed. Total tokens generated: {token_count}")

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
        logger.info(f"Starting stream generation for {len(messages)} messages")

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
            'max_new_tokens': generation_config.max_new_tokens
        }
        final_params = generation_config.extra_data.copy()
        final_params.update(config_dict)

        try:
            return self._create_generator(messages, final_params)
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
        
        generators = []
        
        for i, messages in enumerate(messages_batch):
            try:
                # Get config for this generation (if provided)
                config = configs[i] if configs else None
                
                # Extract parameters for this specific generation
                generation_kwargs = {}
                for param_name, param_values in kwargs.items():
                    param_value = param_values[i]
                    if param_value is not None:  # Only include non-None values
                        generation_kwargs[param_name] = param_value
                
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