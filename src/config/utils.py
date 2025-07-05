import yaml
import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Loads the YAML configuration from the path defined by the CONFIG_PATH environment variable.
    """
    path = os.getenv("CONFIG_PATH")
    if path is None:
        raise EnvironmentError("CONFIG_PATH environment variable is not set.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at path: {path}")

    with open(path) as f:
        return yaml.safe_load(f)
    
def save_config(cfg: Dict[str, Any]):
    """
    Saves the given configuration dictionary to disk at CONFIG_PATH.
    """
    path = os.getenv("CONFIG_PATH")
    with open(path, 'w') as f:
        f.write(yaml.safe_dump(cfg))
    
def update_config(key: str, value: Any):
    """
    Updates a specific key in the configuration and persists it to disk.
    """
    cfg = get_config()
    cfg[key] = value
    save_config(cfg)