import logging
from datetime import datetime
import os
import sys
from typing import List, Optional

def require_env_var(key: str) -> str:
    """
    Ensures that a required environment variable is set.

    Args:
        key (str): The name of the environment variable to check.

    Returns:
        str: The value of the environment variable.

    Raises:
        EnvironmentError: If the environment variable is not set.
    """
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"{key} environment variable is not set.")
    return value

def setup_logging(level: int = logging.DEBUG, disable_logger_names: Optional[List[str]] = None):
    log_format = logging.Formatter("[%(asctime)s %(name)s] [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Create a log file named by the current date
    from shared.config import Config
    cfg = Config()
    log_dir = cfg.get('logging_dir')
    current_date = datetime.now().strftime("%m-%d-%Y")  # MM-DD-YYYY
    log_file_path = os.path.join(log_dir, f"{current_date}.log")
    os.makedirs(log_dir, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Disable loggers
    if disable_logger_names:
        for logger_name in disable_logger_names:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

def load_dotenv_vars():
    from dotenv import load_dotenv
    load_dotenv()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.extend([
        os.path.join(root_dir, os.getenv("CONFIG_PATH", 'config.yaml')),
        os.path.join(root_dir, os.getenv("PYTHONPATH", 'src'))
    ])