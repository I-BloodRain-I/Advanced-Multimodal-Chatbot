import yaml
import os
from typing import Dict, Any

def setup():
    """
    Set global paths for access from the directory nesting level.
    """
    ROOT_DIR = os.getenv('ROOT_DIR')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    LOGGER_DIR = os.path.join(ROOT_DIR, 'logs')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

    cfg = get_config()
    cfg['root_dir'] = ROOT_DIR
    cfg['data_dir'] = DATA_DIR
    cfg['models_dir'] = MODELS_DIR
    cfg['logger_dir'] = LOGGER_DIR
    cfg['output_dir'] = OUTPUT_DIR
    save_config(cfg)

def get_config() -> Dict[str, Any]:
    """
    Loads the YAML configuration from the path defined by the CONFIG_PATH environment variable.
    """
    path = os.getenv("CONFIG_PATH")
    root_dir = os.getenv("ROOT_DIR")
    if path is None:
        raise EnvironmentError("CONFIG_PATH environment variable is not set.")
    if root_dir is None:
        raise EnvironmentError("ROOT_DIR environment variable is not set.")
    if not os.path.exists(root_dir + path):
        raise FileNotFoundError(f"Config file not found at path: {root_dir + path}")

    with open(root_dir + path) as f:
        return yaml.safe_load(f)
    
def save_config(cfg: Dict[str, Any]):
    """
    Saves the given configuration dictionary to disk at CONFIG_PATH.
    """
    path = os.getenv("CONFIG_PATH")
    root_dir = os.getenv("ROOT_DIR")
    with open(root_dir + path, 'w') as f:
        f.write(yaml.safe_dump(cfg))
    
def update_config(key: str, value: Any):
    """
    Updates a specific key in the configuration and persists it to disk.
    """
    cfg = get_config()
    cfg[key] = value
    save_config(cfg)