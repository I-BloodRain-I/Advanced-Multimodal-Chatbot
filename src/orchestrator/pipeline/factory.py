"""
Pipeline factory module for creating configured Pipeline instances.

This module provides the PipelineFactory class which handles the complex
initialization and configuration of Pipeline instances using application
configuration settings. It loads and configures all necessary components
including embedders, routers, LLM engines, RAG systems, and image generators.
"""

import logging
from typing import Any, Dict

from core.entities.enums import ModelDType
from modules.llm.configs import LLMEngineConfig
from modules.rag.utils import load_vector_db
from orchestrator.pipeline.pipeline import Pipeline
from orchestrator.pipeline.utils import convert_numeric_strings
from orchestrator.router.model import TaskClassifier
from shared.config import Config
from shared.utils import get_torch_device, load_model_checkpoint, str_to_model_dtype

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory class for building a configured instance of the `Pipeline` class.
    
    This class uses the application configuration to prepare and inject various
    components into a pipeline, such as LLM, RAG, embedder, router, etc.
    """

    @classmethod
    def build(cls) -> Pipeline:
        """
        Constructs a Pipeline instance using configuration values.

        Returns:
            A fully configured pipeline instance ready to use.
        """
        pipeline_args = {}
        cfg = Config()

        cls._load_embedder_args(cfg, pipeline_args)
        cls._load_router_args(cfg, pipeline_args)
        cls._load_llm_args(cfg, pipeline_args)
        cls._load_rag_args(cfg, pipeline_args)
        cls._load_img_gen_args(cfg, pipeline_args)

        return Pipeline(**pipeline_args)

    @classmethod
    def _load_embedder_args(cls, cfg: Config, args: Dict[str, Any]):
        """
        Load embedder configuration arguments from config and add to pipeline args.
        """
        cfg_embed = cfg.get('embedder')
        args.update({
            'embed_model_name': convert_numeric_strings(cfg_embed.get('model_name')),
            'embed_device_name': convert_numeric_strings(cfg_embed.get('device_name')),
        })

    @classmethod
    def _load_router_args(cls, cfg: Config, args: Dict[str, Any]):
        """
        Load task classifier/router configuration and initialize the model.
        """
        cfg_router = cfg.get('task_classifier')

        device = get_torch_device(convert_numeric_strings(cfg_router.get('device_name')))
        classifier = TaskClassifier(
            embed_dim=convert_numeric_strings(cfg_router.get('embed_dim')),
            n_classes=convert_numeric_strings(cfg_router.get('n_classes'))).to(device)

        weights = load_model_checkpoint(convert_numeric_strings(cfg_router.get('model_path')), device)['weights']
        classifier.load_state_dict(weights)
        classifier.eval()

        args.update({'router_model': classifier})
    
    @classmethod
    def _load_llm_args(cls, cfg: Config, args: Dict[str, Any]):
        """
        Load LLM engine configuration and create engine config object.
        """
        cfg_llm = cfg.get('llm')

        engine_name = convert_numeric_strings(cfg_llm.get('engine'))
        engine_args = dict(model_name=convert_numeric_strings(cfg_llm.get('model_name')), 
                          dtype=ModelDType[cfg_llm.get('dtype').upper()])

        if f"{engine_name}_engine" in cfg_llm:
            engine_specific_config = cfg_llm.get(f"{engine_name}_engine")
            # Convert numeric strings in engine-specific config
            converted_config = {k: convert_numeric_strings(v) for k, v in engine_specific_config.items()}
            engine_args.update(converted_config)
            engine_config = LLMEngineConfig(**engine_args)
        else:
            error_msg = f"Engine name must be either: [transformers, vllm]"
            logger.error(error_msg)
            raise ValueError(error_msg)

        args.update({
            'llm_engine_name': engine_name,
            'llm_engine_config': engine_config
        })
        
    @classmethod
    def _load_rag_args(cls, cfg: Config, args: Dict[str, Any]):
        """
        Load RAG (Retrieval-Augmented Generation) configuration arguments.
        """
        cfg_rag = cfg.get('rag')
        vector_db = load_vector_db(convert_numeric_strings(cfg_rag.get('default_db')))
        args.update({
            'rag_vector_db': vector_db,
            'rag_n_extracted_docs': convert_numeric_strings(cfg_rag.get('n_extracted_docs')),
            'rag_prompt_format': convert_numeric_strings(cfg_rag.get('prompt_format')),
        })

    @classmethod
    def _load_img_gen_args(cls, cfg: Config, args: Dict[str, Any]):
        """
        Load image generation configuration arguments.
        """
        cfg_img = cfg.get('image_generator')
        args.update({
            'img_model_name': convert_numeric_strings(cfg_img.get('model_name')),
            'img_device_name': convert_numeric_strings(cfg_img.get('device_name')),
            'img_dtype': str_to_model_dtype(cfg_img.get('dtype')) if cfg_img.get('dtype') else ModelDType.FLOAT16,
            'img_scheduler_type': convert_numeric_strings(cfg_img.get('scheduler_type')),
            'img_use_refiner': convert_numeric_strings(cfg_img.get('use_refiner')),
            'img_refiner_name': convert_numeric_strings(cfg_img.get('refiner_name'))
        })
