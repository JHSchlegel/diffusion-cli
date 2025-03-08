"""
This module contains a class for abstraction of image generation using
different diffusion models.
"""

from typing import Optional, Tuple, Union

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from PIL import Image

from .utils import set_seed, setup_device


# =========================================================================== #
#                            Diffusion Model                                  #
# =========================================================================== #
class DiffusionModel:
    """Class for generating images using huggingface diffusion models."""

    def __init__(
        self, model_id: str = "stabilityai/stable-diffusion-2"
    ) -> None:
        """
        Initialize the diffusion model.

        Args:
            model_id (str): Hugging Face model ID
        """
        self.model_id = model_id
        self.pipe: Union[DiffusionPipeline, None] = None
        self.device = setup_device()

    def _load_model(self) -> None:
        """
        Load the diffusion model pipeline.
        """
        # use float16 for faster inference
        torch_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # load pipeline:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable safety checker for speed
            use_safetensors=True,
        )

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)

        # Enable memory optimization if using CUDA:
        if self.device.type == "cuda":
            self.pipe.enable_attention_slicing()

    def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Image.Image, int]:
        """Generate an image using the diffusion model.

        Args:
            prompt (str): Prompt for image generation
            width (int):
            height (int): Height of the generated image
            num_inference_steps (int):
            guidance_scale (float): Guidance scale
            seed (int): Random seed for reproducibility
            negative_prompt (str): Negative prompt for image generation

        Returns:
            Tuple[Image.Image, int]: Generated image and seed

        Args:
            prompt (str): Prompt for image generation
            width (int, optional): Width of the generated image.
                Defaults to 512.
            height (int, optional): Height of the generated image.
                Defaults to 512.
            num_inference_steps (int, optional):  Number of inference steps.
                Defaults to 30.
            guidance_scale (float, optional): Guidance scale for how strictly
                it should adhere to prompt. Defaults to 7.5.
            seed (Optional[int], optional): Random seed for reproducibility.
                Defaults to None.
            negative_prompt (Optional[str], optional): Negative prompt for
                image generation. Defaults to None.

        Returns:
            Tuple[Image.Image, int]:  Generated image and seed
        """
        if self.pipe is None:
            self._load_model()
            assert self.pipe is not None, "Model not loaded"

        actual_seed = set_seed(seed)
        output = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )

        return output.images[0], actual_seed
