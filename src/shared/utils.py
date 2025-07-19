from typing import Any, Union
from pathlib import Path
import torch
from shared.config import Config

def get_torch_device(device: str) -> torch.device:
    """
    Returns a valid PyTorch device based on user preference and availability.

    Args:
        device (str): A string indicating the preferred device, e.g., 'cpu' or 'cuda'.

    Returns:
        A torch.device object corresponding to the chosen or available device.
    """
    if isinstance(device, torch.device):
        return device
    # Use 'cpu' if explicitly requested, otherwise prefer 'cuda' if available
    return torch.device(device) if device == 'cpu' else (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )

def load_model_checkpoint(checkpoint_path: str, device: Union[str, torch.device]) -> Any:
    """
    Loads a PyTorch model checkpoint from a specified path, ensuring it's loaded onto the correct device.

    Args:
        checkpoint_path (str): Relative path to the model checkpoint file.
        device (Union[str, torch.device]): The device to map the checkpoint to ('cpu', 'cuda', or torch.device).

    Returns:
        Any: The loaded PyTorch model or state dictionary.
    
    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    # Resolve the full path to the checkpoint based on configured models directory
    models_dir = Path(Config().get('models_dir'))
    path = models_dir / checkpoint_path

    if not path.exists():
        raise FileNotFoundError(f"The checkpoint not found at: {path}")

    device = device if isinstance(device, torch.device) else get_torch_device(device)

    # Load the checkpoint onto the specified device
    return torch.load(path, map_location=device)