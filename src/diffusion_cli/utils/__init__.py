"""
This module contains utility functions for setting up diffusers pipe and
saving images.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image


# =========================================================================== #
#                             Pipeline Utilities                              #
# =========================================================================== #
def get_default_config() -> Any:
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
) -> Path:
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
    path_output_dir: Path = Path(output_dir)
    path_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_bit: str = ""
    if prompt is not None:
        # create a slug from the prompt (first 30 chars, alphanumeric only)
        prompt = prompt.replace(" ", "_")
        prompt_bit = "".join(c for c in prompt[:30] if c.isalnum() or c == "_")

    prefix: str = f"{filename_prefix}" if filename_prefix else ""
    filename: str = f"{prefix}_{timestamp}_{prompt_bit}.png"

    filepath: Path = path_output_dir / filename
    image.save(filepath)

    return filepath
