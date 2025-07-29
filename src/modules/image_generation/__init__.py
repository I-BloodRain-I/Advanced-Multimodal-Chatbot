"""
Image Generation module for the TensorAlix Agent AI system.

This module provides text-to-image generation capabilities using diffusion models.
It supports configurable pipelines, optional refinement stages, and various
generation parameters for creating high-quality images from textual prompts.

Classes:
    ImageGenerator: Main singleton class for image generation operations
"""

from .image_generator import ImageGenerator, DEFAULT_NEGATIVE_PROMPTS
from .utils import get_scheduler