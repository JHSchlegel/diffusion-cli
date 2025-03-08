"""
diffusion-cli: A command line interface for image generation with
diffusion models
"""

__version__ = "0.0.1"

from .models import DiffusionModel
from .utils import save_image, set_seed, setup_device
