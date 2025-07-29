from typing import Any, Union
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from core.entities.enums import ModelDType
from shared.config import Config

def get_torch_device(device: str) -> torch.device:
    """
    Returns a valid PyTorch device based on user preference and availability.

    Args:
        device: A string indicating the preferred device, e.g., 'cpu' or 'cuda'.

    Returns:
        A torch.device object corresponding to the chosen or available device.
    """
    if isinstance(device, torch.device):
        return device
    # Use 'cpu' if explicitly requested, otherwise prefer 'cuda' if available
    return torch.device(device) if device == 'cpu' else (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )

def load_model_checkpoint(checkpoint_path: str, device: Union[str, torch.device]) -> Any:
    """
    Loads a PyTorch model checkpoint from a specified path, ensuring it's loaded onto the correct device.

    Args:
        checkpoint_path: Relative path to the model checkpoint file.
        device: The device to map the checkpoint to ('cpu', 'cuda', or torch.device).

    Returns:
        The loaded PyTorch model or state dictionary.
    
    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    # Resolve the full path to the checkpoint based on configured models directory
    models_dir = Path(Config().get('models_dir'))
    path = models_dir / checkpoint_path

    if not path.exists():
        raise FileNotFoundError(f"The checkpoint not found at: {path}")

    device = device if isinstance(device, torch.device) else get_torch_device(device)

    # Load the checkpoint onto the specified device
    return torch.load(path, map_location=device)

def str_to_model_dtype(dtype_str: str) -> ModelDType:
    """
    Convert a string representation to ModelDType enum.
    
    Args:
        dtype_str: String representation of the dtype (e.g., 'fp16', 'float16', 'auto')
        
    Returns:
        The corresponding ModelDType enum value
        
    Raises:
        ValueError: If the string is not a valid dtype representation
    """
    dtype_str = dtype_str.lower().strip()
    
    if dtype_str in ['auto']:
        return ModelDType.AUTO
    elif dtype_str in ['int4', '4bit']:
        return ModelDType.INT4
    elif dtype_str in ['int8', '8bit']:
        return ModelDType.INT8
    elif dtype_str in ['bfloat16', 'bf16']:
        return ModelDType.BFLOAT16
    elif dtype_str in ['float16', 'fp16', 'half']:
        return ModelDType.FLOAT16
    elif dtype_str in ['float32', 'fp32', 'full']:
        return ModelDType.FLOAT32
    else:
        raise ValueError(f"Invalid dtype string: '{dtype_str}'. Supported: auto, int4, int8, bfloat16, float16, float32")

def to_torch_dtype(dtype) -> torch.dtype:
    """
    Convert a ModelDType enum or string to the corresponding PyTorch tensor dtype.
    
    Args:
        dtype: The model data type to convert (ModelDType enum or string)
        
    Returns:
        The corresponding PyTorch tensor data type
        
    Raises:
        Exception: If the dtype is not supported
    """
    # Convert string to ModelDType enum if needed
    if isinstance(dtype, str):
        dtype = str_to_model_dtype(dtype)
    
    if dtype == ModelDType.INT8:
        return torch.int8
    elif dtype in [ModelDType.BFLOAT16, ModelDType.AUTO]:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype == ModelDType.FLOAT16:
        return torch.float16
    elif dtype == ModelDType.FLOAT32:
        return torch.float32
    else:
        raise Exception(f"Invalid dtype: {dtype.value}. Supported: [AUTO, INT8, BFLOAT16, FLOAT16, FLOAT32]")

def get_bitsandbytes_config_for_dtype(dtype: ModelDType) -> BitsAndBytesConfig:
    """
    Returns a BitsAndBytesConfig based on the provided model data type.

    Args:
        dtype: The target quantization data type (INT4 or INT8).

    Returns:
        Configuration for bitsandbytes loading.

    Raises:
        ValueError: If the dtype is unsupported.
    """
    if dtype == ModelDType.INT4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
            bnb_4bit_use_double_quant=True,        
            bnb_4bit_quant_type="nf4"              
        )
    elif dtype == ModelDType.INT8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,           
            llm_int8_enable_fp32_cpu_offload=True
        )
    else:
        raise Exception(f"Invalid dtype: {dtype.value}. Supported: [INT4, INT8]")