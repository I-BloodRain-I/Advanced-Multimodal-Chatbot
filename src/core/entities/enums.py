"""
Core entity enumerations for the TensorAlix Agent AI system.

This module defines enumeration classes used throughout the system for
standardized task types and model data types.
"""

from enum import Enum


class TaskType(Enum):
    """
    Enumeration of supported AI task types in the system.
    
    Attributes:
        WEB_SEARCH: Web search and information retrieval tasks
        IMAGE_GEN: Image generation and manipulation tasks  
        TEXT_GEN: Text generation and language model tasks
    """
    WEB_SEARCH = 0
    IMAGE_GEN  = 1
    TEXT_GEN   = 2


class ModelDType(Enum):
    """
    Enumeration of supported model data types for quantization and precision control.
    
    Attributes:
        AUTO: Automatic data type selection
        INT4: 4-bit integer quantization
        INT8: 8-bit integer quantization
        BFLOAT16: Brain floating point 16-bit precision
        FLOAT16: Half precision floating point (16-bit)
        FLOAT32: Single precision floating point (32-bit)
    """
    AUTO     = 0
    INT4     = 1
    INT8     = 2
    BFLOAT16 = 3
    FLOAT16  = 4
    FLOAT32  = 5