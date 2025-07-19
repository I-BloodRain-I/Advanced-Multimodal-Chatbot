from typing import Any, Dict

from modules.rag.utils import load_vector_db
from orchestrator.pipeline.pipeline import Pipeline
from orchestrator.router.model import TaskClassifier
from shared.config import Config
from shared.utils import get_torch_device, load_model_checkpoint

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
            Pipeline: A fully configured pipeline instance ready to use.
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
        cfg_embed = cfg.get('embedder')
        args.update({
            'embed_model_name': cfg_embed.get('model_name'),
            'embed_device_name': cfg_embed.get('device_name'),
        })

    @classmethod
    def _load_router_args(cls, cfg: Config, args: Dict[str, Any]):
        cfg_router = cfg.get('task_classifier')

        device = get_torch_device(cfg_router.get('device_name'))
        classifier = TaskClassifier(
            embed_dim=cfg_router.get('embed_dim'),
            n_classes=cfg_router.get('n_classes')).to(device)

        weights = load_model_checkpoint(cfg_router.get('model_path'), device)['weights']
        classifier.load_state_dict(weights)
        classifier.eval()

        args.update({'router_model': classifier})
    
    @classmethod
    def _load_llm_args(cls, cfg: Config, args: Dict[str, Any]):
        cfg_llm = cfg.get('llm')
        args.update({
            'llm_model_name': cfg_llm.get('model_name'),
            'llm_dtype': cfg_llm.get('dtype'),
            'llm_max_new_tokens': cfg_llm.get('max_new_tokens'),
            'llm_temperature': cfg_llm.get('temperature'),
            'llm_device_name': cfg_llm.get('device_name'),
            'llm_stream_output': cfg_llm.get('stream_output'),
        })
        
    @classmethod
    def _load_rag_args(cls, cfg: Config, args: Dict[str, Any]):
        cfg_rag = cfg.get('rag')
        vector_db = load_vector_db(cfg_rag.get('default_db'))
        args.update({
            'rag_vector_db': vector_db,
            'rag_n_extracted_docs': cfg_rag.get('n_extracted_docs'),
            'rag_prompt_format': cfg_rag.get('prompt_format'),
        })

    @classmethod
    def _load_img_gen_args(cls, cfg: Config, args: Dict[str, Any]):
        cfg_img = cfg.get('image_generator')
        args.update({
            'img_model_name': cfg_img.get('model_name'),
            'img_device_name': cfg_img.get('device_name'),
            'img_dtype': cfg_img.get('dtype'),
            'img_scheduler_type': cfg_img.get('scheduler_type'),
            'img_use_refiner': cfg_img.get('use_refiner'),
            'img_refiner_name': cfg_img.get('refiner_name')
        })
