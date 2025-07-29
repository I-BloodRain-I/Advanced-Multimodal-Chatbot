"""
Main entry point for the TensorAlix Agent AI system.

This module initializes the application environment, sets up logging configuration,
and starts the main processing loop. It loads environment variables, configures
logging with specified exclusions, and launches the PromptDispatcher to handle
incoming AI requests from the Redis queue.

The system runs continuously, processing batches of conversations through
the AI pipeline until manually stopped.
"""

from src.common.utils import setup_logging, load_dotenv_vars
load_dotenv_vars()
setup_logging(disable_logger_names=["urllib3.connectionpool", "PIL.Image", 
                                    "httpcore", "filelock"])

from orchestrator.prompt_dispatcher import PromptDispatcher

if __name__ == "__main__":
    PromptDispatcher().start_loop()