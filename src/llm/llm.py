from typing import List, Generator, Union
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import Config
from chatbot.utils import get_torch_device
from chatbot.types import Message
from .text_streamer import TextStreamer
from .utils import *

class LLM:
    """
    Singleton class for managing a causal language model instance with support for
    quantization, configurable generation parameters, and streaming output.
    """

    _instance = None
    _logger = None
    _model = None
    _tokenizer = None
    _config = Config()
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                llm_config = cls._config.get('llm')
                cls._logger = logging.getLogger(cls.__name__)
                cls._device = get_torch_device(llm_config.get('device'))
                cls._model = cls._load_model()
                cls._tokenizer = cls._load_tokenizer()
                cls._logger.info("LLM model and tokenizer loaded successfully.")
            except Exception as e:
                cls._logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
                raise
        return cls._instance

    def __init__(self, max_new_tokens: int = None, temperature: float = None):
        """
        Args:
            max_new_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
        """
        llm_config = self._config.get('llm')
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else llm_config.get('max_new_tokens')
        self.temperature = temperature if temperature is not None else llm_config.get('temperature')

        self._streamer = TextStreamer(model=self._model,
                                      tokenizer=self._tokenizer,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      device=self._device)
    
    @classmethod
    def _load_model(cls) -> AutoModelForCausalLM:
        """
        Loads the model from a configured path with support for dtype-based quantization.

        Returns:
            AutoModelForCausalLM: The initialized transformer model.
        """
        llm_config = cls._config.get('llm')
        model_dtype_str = llm_config.get('dtype')
        model_path = llm_config.get('model_path')

        quant_config = None
        torch_dtype = None

        # Configure quantization and dtype based on user config
        if model_dtype_str == 'int4':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
                bnb_4bit_use_double_quant=True,        
                bnb_4bit_quant_type="nf4"              
            )
        elif model_dtype_str == 'int8':
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,           
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif model_dtype_str == 'bf16':
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif model_dtype_str == 'fp16':
            torch_dtype = torch.float16
        elif model_dtype_str == 'fp32':
            torch_dtype = torch.float32
        else:
            raise Exception(f"Invalid dtype in config: {model_dtype_str}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=quant_config,
                torch_dtype=torch_dtype,
                device_map=cls._device
            )
            model.eval()
            return model
        except Exception as e:
            cls._logger.error(f"Error loading model from path {model_path}: {e}", exc_info=True)
            raise

    @classmethod
    def _load_tokenizer(cls) -> AutoTokenizer:
        """
        Loads the tokenizer and ensures it has a pad token.

        Returns:
            AutoTokenizer: The initialized tokenizer.
        """
        model_path = cls._config.get('llm').get('model_path')
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                cls._logger.warning("Tokenizer has no pad_token; using eos_token as pad_token.")
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            cls._logger.error(f"Error loading tokenizer from path {model_path}: {e}", exc_info=True)
            raise

    def generate(self, messages: List[Message], return_tokens: bool = False) -> Union[str, torch.Tensor]:
        """
        Generates a text response from a list of messages.

        Args:
            messages (List[Message]): List of user/assistant messages forming the prompt.
            return_tokens (bool): If True, return raw token tensor instead of decoded string.

        Returns:
            Union[str, torch.Tensor]: The generated text or token tensor.
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
            encoded_inputs = self._tokenizer(formatted_prompt,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True
            )

            input_ids = encoded_inputs["input_ids"].to(self._device)
            attention_mask = encoded_inputs["attention_mask"].to(self._device)

            self._logger.debug(f"Tokenized input - input_ids shape: {input_ids.shape}, "
                               f"attention_mask shape: {attention_mask.shape}")
        except Exception as e:
            self._logger.error(f"Failed to tokenize input: {e}", exc_info=True)
            raise

        # Generate new tokens
        try:
            self._logger.debug("Generation starting")
            tokens = self._model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        do_sample=True,
                                        temperature=self.temperature,
                                        pad_token_id=self._tokenizer.pad_token_id,
                                        eos_token_id=self._tokenizer.eos_token_id,
                                        max_new_tokens=self.max_new_tokens)[0]
            generated_tokens = tokens[input_ids.shape[1]:]
            self._logger.debug(f"Generation completed. Total tokens generated: {len(generated_tokens)}")
        except Exception as e:
            self._logger.error(f"Error during token generation: {e}", exc_info=True)
            raise

        if not return_tokens:
            return self._tokenizer.decode(generated_tokens, 
                                          clean_up_tokenization_spaces=True, 
                                          skip_special_tokens=True)
        else:
            return tokens

    def generate_batch(self, messages_batch: List[List[Message]]) -> List[str]:
        """
        Generates responses for a batch of message sequences.

        Args:
            messages_batch (List[List[Message]]): List of message sequences.

        Returns:
            List[str]: List of generated responses.
        """
        outputs_ids = [self.generate(messages, return_tokens=True) for messages in messages_batch] 
        return self._tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        
    def generate_stream(self, messages: List[Message]) -> Generator[str, None, None]:
        """
        Streams text generation output for a single message sequence.

        Args:
            messages (List[Message]): List of messages forming the prompt.

        Returns:
            Generator[str, None, None]: Token stream generator.
        """
        return self._streamer.generate_stream(messages)

    def generate_stream_batch(self, messages_batch: List[List[Message]]) -> List[Generator[str, None, None]]:
        """
        Streams outputs for a batch of message sequences.

        Args:
            messages_batch (List[List[Message]]): List of message sequences.

        Returns:
            List[Generator[str, None, None]]: List of token stream generators.
        """
        return self._streamer.generate_stream_batch(messages_batch)