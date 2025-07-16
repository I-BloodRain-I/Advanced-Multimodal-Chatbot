import logging
from datetime import datetime
import os

def setup_logging(level: int = logging.DEBUG):
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