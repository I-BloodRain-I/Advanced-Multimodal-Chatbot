import os

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