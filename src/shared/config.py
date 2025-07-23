import yaml
from pathlib import Path
from typing import Dict, Any
import logging

from common.utils import require_env_var

logger = logging.getLogger(__name__)


class Config:
    """
    Singleton configuration class that loads and manages application settings from a YAML file.

    This class ensures a single configuration instance is used across the application.
    It loads from a YAML file specified by environment variables and provides methods
    to access, modify, validate, and persist configuration data.
    """
    _instance = None
    _config_data: Dict[str, Any] = None
    _path: Path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._path = cls._get_config_path()
            cls._load_config()
        return cls._instance

    @staticmethod
    def _get_config_path() -> Path:
        return Path(require_env_var("CONFIG_PATH"))

    @classmethod
    def _load_config(cls):
        if not cls._path.exists():
            logger.warning("The config.yaml not found.")
            cls._initialize_default_config()
        else:
            with open(cls._path, 'r', encoding='utf-8') as f:
                cls._config_data = yaml.safe_load(f) or {}
        cls._validate_config()

    @classmethod
    def _initialize_default_config(cls):
        cls._config_data = {
            "data_dir":    "data",
            "logging_dir": "logs",
            "models_dir":  "models",
            "embedder": {
                "model_name": "all-mpnet-base-v2",
                "device_name": "cuda"
            },
            "llm": {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "dtype": "int8",
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "device_name": "cuda",
                "stream_output": True
            },
            "image_generator": {
                "model_name": "dreamlike-art/dreamlike-photoreal-2.0",
                "device_name": "cuda",
                "dtype": "fp16",
                "scheduler_type": "euler_ancestral",
                "use_refiner": False,
                "refiner_name": "stabilityai/stable-diffusion-xl-refiner-1.0"
            },
            "semantic_chunker": {
                "model_name": "all-mpnet-base-v2",
                "device_name": "cuda",
                "chunk_token_limit": 380,
                "similarity_threshold": 0.75,
                "min_chunk_tokens": 120,
                "batch_size": 64,
                "show_progress": False
            },
            "rag": {
                "n_extracted_docs": 3,
                "default_db": "pinecone",
                "prompt_format": "{context}\\n{prompt}",
                "vector_db": {
                    "faiss": {
                        "dir": "vector_db",
                        "dimension": 768,
                        "index_type": "auto",
                        "device": "cuda",
                        "flat_index_type": {
                            "max_docs": 50000
                        },
                        "ivf_index_type": {
                            "max_clusters": 8192,
                            "max_probe": 128,
                            "scaling_factor": 0.5,
                            "max_docs": 500000
                        },
                        "ivfpq_index_type": {
                            "max_clusters": 4096,
                            "max_probe": 64,
                            "scaling_factor": 0.4,
                            "n_subvectors": 64,
                            "n_bits": 8
                        }
                    },
                    "pinecone": {
                        "index_name": "ai-agent",
                        "dimension": 768,
                        "metric": "cosine",
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            },
            "cache": {
                "redis": {
                    "host": "redis",
                    "port": 6379
                }
            },
            "dispatcher": {
                "sleep_seconds": 1.0
            },
            "task_classifier": {
                "device_name": "cuda",
                "embed_dim": 768,
                "model_path": "task_classifier.pth",
                "n_classes": 3
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }

        cls.save()

    @classmethod
    def _validate_config(cls):
        required_keys = [
            "data_dir", "logging_dir", "models_dir", "image_generator",
            "embedder", "llm", "semantic_chunker", "rag", "cache", 
            "dispatcher", "task_classifier", "server"
        ]
        for key in required_keys:
            if key not in cls._config_data:
                raise ValueError(f"Missing required config key: {key}")

    @classmethod
    def get(cls, key: str, default = None) -> Any:
        keys  = key.split('.')
        cfg = cls._config_data
        for key in keys:
            if isinstance(cfg, dict) and key in cfg:
                cfg = cfg[key]
            else:
                return default
        return cfg

    @classmethod
    def set(cls, key: str, value: Any):
        keys = key.split('.')
        cfg = cls._config_data
        for key in keys[:-1]:
            if key not in cfg or not isinstance(cfg[key], dict):
                cfg[key] = {}
            cfg = cfg[key]
        cfg[keys[-1]] = value
        logger.info(f"Value for {key} was set")

    @classmethod
    def update(cls, obj: Dict[str, Any]):
        cls._config_data.update(obj)
        logger.info("Configuration successfully updated with new values.")

    @classmethod
    def reload(cls):
        cls._load_config()
        logger.info("Configuration reloaded from file.")

    @classmethod
    def save(cls):
        with open(cls._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cls._config_data, f)
        logger.info(f"Configuration saved to: {cls._path}")

    @classmethod
    def as_dict(cls):
        return cls._config_data.copy()