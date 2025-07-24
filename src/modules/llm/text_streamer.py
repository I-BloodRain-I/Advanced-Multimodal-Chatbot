import asyncio
import threading
from typing import AsyncGenerator, List, Optional, Tuple
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
from shared.text_processing.prompt_transformer import PromptTransformer

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria to stop generation when specific tokens are encountered"""

    def __init__(self, stop_token_ids: List[int], stop_event: threading.Event):
        """
        Args:
            stop_token_ids (List[int]): List of token IDs that trigger stopping.
            stop_event (threading.Event): Threading event to signal when generation should stop.
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

class TextStreamer:
    """
    The text streamer for asynchronous token-by-token generation

    Args:
        model (AutoModelForCausalLM): The causal language model.
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
    """
    def __init__(self, 
                 model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.debug(f"TextStreamer initialized with max_new_tokens={self.max_new_tokens}, "
                     f"temperature={self.temperature}, device={self.device}")

    def _create_stopping_criteria(self) -> Tuple[Optional[StoppingCriteriaList], threading.Event]:
        """
        Creats a stopping criteria list based on predefined stop sequences.

        Returns:
            Tuple[Optional[StoppingCriteriaList], threading.Event]: 
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

    async def _create_stream_generator(self, messages: MessageHistory) -> AsyncGenerator[str, None]:
        """
        Asynchronously generates tokens one-by-one for a message sequence.

        Yields:
            str: Next generated token.
        """
        try:    
            formatted_prompt = PromptTransformer.format_messages_to_str(messages, self.tokenizer)
            logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
        except Exception as e:
            logger.error(f"Failed to format messages: {e}", exc_info=True)
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
            logger.error(f"Failed to tokenize input: {e}", exc_info=True)
            raise

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        stopping_criteria, stop_event = self._create_stopping_criteria()
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        
        logger.debug(f"Generation parameters: max_new_tokens={self.max_new_tokens}, "
                     f"temperature={self.temperature}, stopping_criteria={stopping_criteria is not None}")

        # Start generation using asyncio.to_thread
        generation_task = asyncio.create_task(
            asyncio.to_thread(self.model.generate, **generation_kwargs)
        )
        
        logger.debug("Generation task started")
        
        token_count = 0
        try:
            async for token in streamer:
                if token:
                    if token_count == self.max_new_tokens:
                        break
                    token_count += 1
                    yield token
            
            # Wait for generation to complete
            await generation_task
            
        except Exception as e:
            logger.error(f"Error during token streaming: {e}", exc_info=True)
            generation_task.cancel()
            raise

        finally:
            stop_event.set()
            streamer.end()
            if not generation_task.done():
                generation_task.cancel()
            logger.debug(f"Generation completed. Total tokens generated: {token_count}")


    def generate_stream(self, messages: MessageHistory) -> AsyncGenerator[str, None]:
        """
        Starts streaming generation for a list of messages.

        Args:
            messages (MessageHistory): A list of user-assistant messages.

        Returns:
            AsyncGenerator[str, None]: Token stream generator.
        """
        logger.info(f"Starting stream generation for {len(messages)} messages")

        if not messages:
            logger.warning("Empty messages provided")
        
        try:
            return self._create_stream_generator(messages)
        except Exception as e:
            logger.error(f"Failed to generate stream: {e}", exc_info=True)
            raise

    def generate_stream_batch(self, messages_batch: List[MessageHistory]) -> List[AsyncGenerator[str, None]]:
        """
        Starts batch streaming generation for multiple sets of messages.

        Args:
            messages_batch (List[MessageHistory]): List of message sequences.

        Returns:
            List[AsyncGenerator[str, None]]: List of token stream generators.
        """
        logger.info(f"Starting batch stream generation for {len(messages_batch)} message batches")

        if not messages_batch:
            logger.warning("Empty messages batch provided")
            return []
            
        try:
            generators = []
            for i, messages in enumerate(messages_batch):
                logger.debug(f"Processing batch item {i+1}/{len(messages_batch)} with {len(messages)} messages")
                generators.append(self._create_stream_generator(messages))
            return generators
        
        except Exception as e:  
            logger.error(f"Failed to generate batch streams: {e}", exc_info=True)
            raise