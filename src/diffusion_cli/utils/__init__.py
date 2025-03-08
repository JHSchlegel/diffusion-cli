"""
This module contains utility functions for setting up diffusers pipe and saving images.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
from typing import Optional, Dict, Any
from PIL import Image
import torch
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path


# =========================================================================== #
#                             Pipeline Utilities                              #
# =========================================================================== #
def get_default_config() -> Dict[str, Any]:
    """Load the default configuration from YAML.

    Returns:
        Dict[str, Any]: configuration dictionary
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml"
    )
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_device() -> torch.device:
    """Set up and return the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device: Device to use
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int, optional): Random seed. If None, a random seed is generated.

    Returns:
        int: The seed that was set
    """
    if seed is None:
        # generate a random 32-bit integer seed
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def save_image(
    image: Image.Image,
    output_dir: str,
    filename_prefix: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Save the generated image

    Args:
        image (Image.Image): Generated image to save
        output_dir (str): Directory to save the image
        filename_prefix (Optional[str], optional): Prefix for image saving.
            Defaults to None.
        prompt (Optional[str], optional): Used prompt for saving the image.
            Defaults to None.

    Returns:
        str: filepath of the saved image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_bit = ""
    if prompt:
        # create a slug from the prompt (first 30 chars, alphanumeric only)
        prompt = prompt.replace(" ", "_")
        prompt_bit = "".join(c for c in prompt[:30] if c.isalnum() or c == "_")

    prefix = f"{filename_prefix}" if filename_prefix else ""
    filename = f"{prefix}_{timestamp}_{prompt_bit}.png"

    filepath = output_dir / filename
    image.save(filepath)

    return filepath
