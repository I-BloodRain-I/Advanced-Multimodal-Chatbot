import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from diffusers import (
    AutoPipelineForText2Image, 
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.utils import logging as diff_logging
diff_logging.disable_progress_bar()  # Disable progress bar for cleaner logs

import torch

from modules.image_generation.utils import get_scheduler
from shared.config import Config
from shared.utils import get_torch_device

logger = logging.getLogger(__name__)

# Default negative prompts to discourage undesired visual artifacts
DEFAULT_NEGATIVE_PROMPTS = """
(deformed, bad anatomy, extra limbs, poorly drawn hands, bad feet, 
blurry, low quality, jpeg artifacts, watermark, text, bad proportions, 
unnatural pose, bad lighting)""".replace('\n', '')


class ImageGenerator:
    """
    Singleton class for generating images using a text-to-image pipeline.

    Loads a model pipeline (optionally with a refiner) and generates images
    based on prompts with configurable parameters.

    Args:
        model_name (str): Name or path of the model to load.
        device (Union[str, torch.device]): Torch device (e.g., 'cuda', 'cpu').
        dtype (str): Precision type to use ('fp16' or 'fp32').
        scheduler_type (str): Scheduler type for diffusion process.
        use_refiner (bool): Whether to apply an additional refinement stage.
        refiner_name (str): Name of the refiner model to load if applicable.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, 
                 model_name: str, 
                 device: Union[str, torch.device] = 'cuda',
                 dtype: str = 'fp16',
                 scheduler_type: str = 'euler_ancestral',
                 use_refiner: bool = False,
                 refiner_name: str = 'stabilityai/stable-diffusion-xl-refiner-1.0'):
        if self._initialized:
            return
            
        self._models_dir = Path(Config.get('models_dir'))
        self._torch_dtype = torch.half if dtype =='fp16' else torch.float
        self._scheduler_type = scheduler_type
        
        self.device = get_torch_device(device)
        self.pipeline = self._load_pipeline(model_name, AutoPipelineForText2Image)
        self.refiner = None if not use_refiner else (
            self._load_pipeline(refiner_name, StableDiffusionXLImg2ImgPipeline)
        )

        self._initialized = True

    def _load_pipeline(self, model_name: str, pipeline_class) -> Any:
        "Loads and configures a model pipeline, downloading if needed."
        model_path = self._models_dir / model_name 

        if model_path.exists(): 
            # Load from local cache if available
            pipeline = pipeline_class.from_pretrained(
                str(model_path),
                torch_dtype=self._torch_dtype,
                use_safetensors=True,
                local_files_only=True
            )
        else:
            # Download and save to disk if not cached
            pipeline = pipeline_class.from_pretrained(
                model_name,
                torch_dtype=self._torch_dtype,
                use_safetensors=True
            )
            pipeline.save_pretrained(str(model_path), safe_serialization=True)

        pipeline.scheduler = get_scheduler(pipeline, self._scheduler_type)                 
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        # if hasattr(pipeline, "unet"):
        #     pipeline.unet = torch.compile(pipeline.unet)
        # if hasattr(pipeline, "vae"):
        #     pipeline.vae = torch.compile(pipeline.vae)
        # if hasattr(pipeline, "text_encoder"):
        #     pipeline.text_encoder = torch.compile(pipeline.text_encoder)

        return pipeline

    def generate(self, 
                 prompts: List[str],
                 negative_prompts: Optional[List[str]] = None, 
                 width: int = 512, 
                 height: int = 512,
                 guidance_scale: float = 3.0,
                 num_inference_steps: int = 30) -> List[torch.Tensor]:
        """
        Generates images from a list of prompts.

        Args:
            prompts (List[str]): List of textual prompts for image generation.
            negative_prompts (Optional[List[str]]): Prompts to suppress unwanted features.
            width (int): Width of the output image in pixels.
            height (int): Height of the output image in pixels.
            guidance_scale (float): How strongly the model should follow the prompt.
            num_inference_steps (int): Number of steps in the diffusion process.

        Returns:
            List[torch.Tensor]: List of generated image tensors.
        """
        
        if negative_prompts:
            if len(prompts) != len(negative_prompts):
                error_text = f"Mismatch between prompt and negative prompt lengths: {len(prompts)} != {len(negative_prompts)}"
                logger.error(error_text, exc_info=True)
                raise ValueError(error_text)
        else:
            # Apply default negative prompt if none provided
            negative_prompts = [DEFAULT_NEGATIVE_PROMPTS] * len(prompts)
            
        try:
            base_output = self.pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="latent" if self.refiner else 'pt'
            )

            # If refiner is available, further refine the output
            if self.refiner:
                final_output = self.refiner(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    image=base_output.images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="pt"
                )
                return final_output.images

            return base_output.images

        except Exception:
            logger.error("Image generation failed.", exc_info=True)
            raise