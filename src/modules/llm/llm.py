import json
from pathlib import Path
from typing import AsyncGenerator, List, Union
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from core.entities.types import MessageHistory
from shared.config import Config
from shared.text_processing.prompt_transformer import PromptTransformer
from shared.utils import get_torch_device
from .text_streamer import TextStreamer

logger = logging.getLogger(__name__)


class LLM:
    """
    Singleton class for managing a causal language model (LLM) with support for
    quantized loading, configurable generation parameters, and both standard and
    streaming generation interfaces.
    
    Args:
        model_name (str): Hugging Face model identifier or local path to load the model.
        dtype (str): Desired precision format. Supported values: 'int4', 'int8', 'bf16', 'fp16', 'fp32'.
        max_new_tokens (int): Maximum number of tokens to generate per response.
        temperature (float): Sampling temperature to control output randomness.
        device_name (str): Preferred device for inference, e.g., 'cuda', 'cpu'.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, 
                 model_name: str = 'meta-llama/Llama-3.2-3B-Instruct',
                 dtype: str = 'int8',
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7,
                 device_name: str = 'cuda'):
        if self._initialized:
            return # Avoid reinitializing 

        self._models_dir = Path(Config.get('models_dir'))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        try:
            self.device = get_torch_device(device_name)
            self.model = self._load_model(model_name, dtype)
            self.tokenizer = self._load_tokenizer(model_name)
            self._save_model_and_tokenizer(model_name)  # Save for future offline use
            logger.info("LLM model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise

        self._streamer = TextStreamer(model=self.model,
                                      tokenizer=self.tokenizer,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature)
        self._initialized = True
    
    def _load_model(self, model_name: str, model_dtype_str: str) -> AutoModelForCausalLM:
        """
        Loads the model with support for dtype-based quantization and device mapping.
        """

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
            model_path = self._models_dir / model_name
            if model_path.exists():
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), 
                    quantization_config=quant_config,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    local_files_only=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    quantization_config=quant_config,
                    torch_dtype=torch_dtype,
                    device_map=self.device
                )
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model from path {model_name}: {e}", exc_info=True)
            raise

    def _load_tokenizer(self, tokenizer_name: str) -> AutoTokenizer:
        """
        Loads the tokenizer and ensures it has a pad token.
        """
        try:
            tokenizer_path = self._models_dir / tokenizer_name
            if tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token; using eos_token as pad_token.")
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer from path {tokenizer_name}: {e}", exc_info=True)
            raise

    def _save_model_and_tokenizer(self, model_name: str):
        """
        Saves the model and tokenizer to disk if not already saved.

        Removes quantization_config from config.json to allow flexibility on reload.
        """
        try:
            model_path = self._models_dir / model_name
            if not model_path.exists():
                self.model.save_pretrained(str(model_path), safe_serialization=True)
                self.tokenizer.save_pretrained(str(model_path), safe_serialization=True)

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
            logger.error("Failed to save model and tokenizer.", exc_info=True)

    def generate(self, messages: MessageHistory, return_tokens: bool = False) -> Union[str, torch.Tensor]:
        """
        Generates a text response from a list of messages.

        Args:
            messages (MessageHistory): List of user/assistant messages forming the prompt.
            return_tokens (bool): If True, return raw token tensor instead of decoded string.

        Returns:
            Union[str, torch.Tensor]: The generated text or token tensor.
        """

        # Apply chat template if available
        try:
            formatted_prompt = PromptTransformer.format_messages_to_str(messages, self.tokenizer)
            logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
        except Exception as e:
            logger.error(f"Failed to format messages: {e}", exc_info=True)
            raise
        
        # Tokenize the prompt
        try:
            encoded_inputs = self.tokenizer(formatted_prompt,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True
            )

            input_ids = encoded_inputs["input_ids"].to(self.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.device)

            logger.debug(f"Tokenized input - input_ids shape: {input_ids.shape}, "
                               f"attention_mask shape: {attention_mask.shape}")
        except Exception as e:
            logger.error(f"Failed to tokenize input: {e}", exc_info=True)
            raise

        # Generate new tokens
        try:
            logger.debug("Generation starting")
            tokens = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         do_sample=True,
                                         temperature=self.temperature,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id,
                                         max_new_tokens=self.max_new_tokens)[0]
            generated_tokens = tokens[input_ids.shape[1]:]
            logger.debug(f"Generation completed. Total tokens generated: {len(generated_tokens)}")
        except Exception as e:
            logger.error(f"Error during token generation: {e}", exc_info=True)
            raise

        if not return_tokens:
            return self.tokenizer.decode(generated_tokens, 
                                         clean_up_tokenization_spaces=True, 
                                         skip_special_tokens=True)
        else:
            return generated_tokens

    def generate_batch(self, messages_batch: List[MessageHistory]) -> List[str]:
        """
        Generates responses for a batch of message sequences.

        Args:
            messages_batch (List[MessageHistory]): List of message sequences.

        Returns:
            List[str]: List of generated responses.
        """
        outputs_ids = [self.generate(messages, return_tokens=True) for messages in messages_batch] 
        return self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        
    def generate_stream(self, messages: MessageHistory) -> AsyncGenerator[str, None]:
        """
        Streams text generation output for a single message sequence.

        Args:
            messages (MessageHistory): List of messages forming the prompt.

        Returns:
            AsyncGenerator[str, None]: Token stream generator.
        """
        return self._streamer.generate_stream(messages)

    def generate_stream_batch(self, messages_batch: List[MessageHistory]) -> List[AsyncGenerator[str, None]]:
        """
        Streams outputs for a batch of message sequences.

        Args:
            messages_batch (List[MessageHistory]): List of message sequences.

        Returns:
            List[AsyncGenerator[str, None]]: List of token stream generators.
        """
        return self._streamer.generate_stream_batch(messages_batch)