import yaml
from pathlib import Path
from typing import Dict, Any
import logging

from utils.validation import require_env_var

class Config:
    _logger = None
    _instance: "Config" = None
    _config_data: Dict[str, Any] = None
    _path: Path = None

    def __new__(cls):
        if cls._instance is None:
            cls._logger = logging.getLogger(cls.__name__)
            cls._instance = super().__new__(cls)
            cls._path = cls._get_config_path()
            cls._load_config()
        return cls._instance

    @staticmethod
    def _get_config_path() -> Path:
        root_dir = Path(require_env_var("ROOT_DIR"))
        return root_dir / require_env_var("CONFIG_PATH")

    @classmethod
    def _load_config(cls):
        if not cls._path.exists():
            cls._logger.warning("The config.yaml not found.")
            cls._initialize_default_config()
        else:
            with open(cls._path, 'r', encoding='utf-8') as f:
                cls._config_data = yaml.safe_load(f) or {}
        cls._validate_config()

    @classmethod
    def _initialize_default_config(cls):
        root_dir = Path(require_env_var("ROOT_DIR"))

        cls._config_data = {
            "root_dir":   root_dir,
            "data_dir":   root_dir / "data",
            "logger_dir": root_dir / "logs",
            "models_dir": root_dir / "models",
            "output_dir": root_dir / "output",
            "embedder": {
                "device": "cuda",
                "model_path": "all-mpnet-base-v2"
            },
            "task_classifier": {
                "device": "cuda",
                "embed_dim": 768,
                "model_path": "task_classifier.pth",
                "n_classes": 3
            },
            "image_gen": {
                "device": "cuda",
                "model_path": ""
            },
            "websearch": {
                "websites": "",
            }
        }
        with open(cls._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cls._config_data, f)
        cls._logger.info("Created new config.yaml.")

    @classmethod
    def _validate_config(cls):
        required_keys = ["root_dir", "data_dir", "logger_dir", "models_dir", "output_dir",
                         "embedder", "task_classifier", "image_gen", "websearch"]
        for key in required_keys:
            if key not in cls._config_data:
                raise ValueError(f"Missing required config key: {key}")

    def get(self, key: str, default = None):
        return self._config_data.get(key, default)

    def set(self, key: str, value: Any):
        self._config_data[key] = value
        self._logger.info(f"Value in {key} was changed")

    def update(self, obj: Dict[str, Any]):
        self._config_data.update(obj)
        self._logger.info(f"Config was updated")

    def reload(self):
        self._load_config()
        self._logger.info(f"Config was reloaded")

    def save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config_data, f)
        self._logger.info(f"Config was saved")

    def as_dict(self):
        return self._config_data.copy()