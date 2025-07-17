from src.common.utils import setup_logging, load_dotenv_vars
load_dotenv_vars()
setup_logging(disable_logger_names=["urllib3.connectionpool", "PIL.Image", "httpcore"])

from orchestrator.prompt_dispatcher import PromptDispatcher

if __name__ == "__main__":
    PromptDispatcher().start_loop()