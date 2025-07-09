from typing import List, Generator, Optional
import threading
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextIteratorStreamer, 
    StoppingCriteria, 
    StoppingCriteriaList
)

from config import Config
from .utils import format_messages_to_str
from chatbot.types import Message

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria to stop generation when specific tokens are encountered"""
    def __init__(self, stop_token_ids: List[int]):
        """
        Args:
            stop_token_ids (List[int]): List of token IDs that trigger stopping.
        """
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        """
        Args:
            input_ids (torch.Tensor): Input token IDs during generation.
            scores (torch.Tensor): Scores for the current step (unused).

        Returns:
            bool: True if last token matches a stop token ID, False otherwise.
        """
        # Check if the last generated token is in our stop tokens
        return input_ids[0][-1].item() in self.stop_token_ids

class TextStreamer:
    def __init__(self, 
                 model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer,
                 max_new_tokens: int = None,
                 temperature: float = None,
                 device: torch.device = None):
        """
        Args:
            model (AutoModelForCausalLM): The causal language model.
            tokenizer (AutoTokenizer): Tokenizer associated with the model.
            max_new_tokens (Optional[int]): Max tokens to generate. Defaults to config value.
            temperature (Optional[float]): Sampling temperature. Defaults to config value.
            device (Optional[torch.device]): Device to run generation on. Defaults to model's device.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = Config()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device if device else next(model.parameters()).device

        # Use config values as fallback if parameters not provided
        self._max_new_tokens = max_new_tokens if max_new_tokens is not None else self._config.get('llm').get('max_new_tokens')
        self._temperature = temperature if temperature is not None else self._config.get('llm').get('temperature')

        self._logger.debug(f"TextStreamer initialized with max_new_tokens={self._max_new_tokens}, "
                           f"temperature={self._temperature}, device={self._device}")

    def _create_stopping_criteria(self) -> Optional[StoppingCriteriaList]:
        """
        Creats a stopping criteria list based on predefined stop sequences.

        Returns:
            Optional[StoppingCriteriaList]: A list of stopping criteria or None if no valid sequences are found.
        """
        self._logger.debug("Creating stopping criteria")

        stop_sequences = ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"]
        stop_token_ids = []

        if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:    
            for seq in stop_sequences:
                try:
                    # Encode without special tokens to get the actual token IDs
                    token_ids = self._tokenizer.encode(seq, add_special_tokens=False)
                    if token_ids:
                        stop_token_ids.extend(token_ids)
                        self._logger.debug(f"Added stop sequence '{seq}' with token IDs: {token_ids}")
                except Exception as e:
                    # Skip sequences that can't be encoded
                    self._logger.debug(f"Failed to encode stop sequence '{seq}': {e}")
                    continue
            
            if stop_token_ids:
                # Remove duplicates while preserving order
                stop_token_ids = list(dict.fromkeys(stop_token_ids))
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
                self._logger.debug(f"Created stopping criteria with {len(stop_token_ids)} stop tokens: {stop_token_ids}")
                return stopping_criteria
            else:
                self._logger.debug("No valid stop tokens found, no stopping criteria created")
        else:
            self._logger.debug("No chat template found, no stopping criteria created")
        
        return None

    def _create_stream_generator(self, messages: List[Message]) -> Generator[str, None, None]:
        """
        Creates a generator to stream model output token-by-token.

        Args:
            messages (List[Message]): List of message objects to format and send to the model.

        Returns:
            Generator[str, None, None]: A generator yielding string tokens from the model.
        """

        # Apply chat template if available
        try:
            formatted_prompt = format_messages_to_str(messages, self._tokenizer)
            self._logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
        except Exception as e:
            self._logger.error(f"Failed to format messages: {e}", exc_info=True)
            raise
        
        # Tokenize the prompt
        try:
            encoded_inputs = self._tokenizer(
                formatted_prompt, 
                padding=False,
                truncation=True,
                max_length=self._tokenizer.model_max_length,
                return_tensors="pt"
            )
        
            input_ids = encoded_inputs["input_ids"].to(self._device)
            attention_mask = encoded_inputs["attention_mask"].to(self._device)

            self._logger.debug(f"Tokenized input - input_ids shape: {input_ids.shape}, "
                               f"attention_mask shape: {attention_mask.shape}")
        except Exception as e:
            self._logger.error(f"Failed to tokenize input: {e}", exc_info=True)
            raise

        # Create the streamer
        streamer = TextIteratorStreamer(self._tokenizer, 
                                        skip_prompt=True, 
                                        skip_special_tokens=True)
        # Prepare generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self._max_new_tokens,
            "do_sample": True,
            "temperature": self._temperature,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Add stopping criteria for chat models
        stopping_criteria = self._create_stopping_criteria()
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        
        self._logger.debug(f"Generation parameters: max_new_tokens={self._max_new_tokens}, "
                           f"temperature={self._temperature}, stopping_criteria={stopping_criteria is not None}")

        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=self._model.generate, 
            kwargs=generation_kwargs
        )
        generation_thread.start()
        self._logger.debug("Generation thread started")
        
        # Stream tokens as they're generated
        token_count = 0
        try:
            for token in streamer:
                if token:  # Filter out empty tokens
                    token_count += 1
                    yield token
        except Exception as e:
            self._logger.error(f"Error during token streaming: {e}", exc_info=True)
            raise
        finally:
            # Ensure thread completes properly
            generation_thread.join(timeout=60)
            if generation_thread.is_alive():
                self._logger.warning("Generation thread did not finish in expected time")
            self._logger.debug(f"Generation completed. Total tokens generated: {token_count}")

    def generate_stream(self, messages: List[Message]) -> Generator[str, None, None]:
        """
        Starts streaming generation for a list of messages.

        Args:
            messages (List[Message]): A list of user-assistant messages.

        Returns:
            Generator[str, None, None]: A generator yielding model-generated tokens.
        """
        self._logger.info(f"Starting stream generation for {len(messages)} messages")

        if not messages:
            self._logger.warning("Empty messages provided")
            return iter([])
        
        try:
            return self._create_stream_generator(messages)
        except Exception as e:
            self._logger.error(f"Failed to generate stream: {e}")
            raise

    def generate_stream_batch(self, messages_batch: List[List[Message]]) -> List[Generator[str, None, None]]:
        """
        Starts batch streaming generation for multiple sets of messages.

        Args:
            messages_batch (List[List[Message]]): List of message sequences.

        Returns:
            List[Generator[str, None, None]]: List of generators, one for each message batch.
        """
        self._logger.info(f"Starting batch stream generation for {len(messages_batch)} message batches")

        if not messages_batch:
            self._logger.warning("Empty messages batch provided")
            return []
            
        try:
            generators = []
            for i, messages in enumerate(messages_batch):
                self._logger.debug(f"Processing batch item {i+1}/{len(messages_batch)} with {len(messages)} messages")
                generators.append(self.generate_stream(messages))
            return generators
        
        except Exception as e:
            self._logger.error(f"Failed to generate batch streams: {e}", exc_info=True)
            raise