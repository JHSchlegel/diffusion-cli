"""
Module containing functionalities to run the diffusion model from
the command line.
"""

# =========================================================================== #
#                                Packages and Presets                         #
# =========================================================================== #
import argparse
import sys

from .models import DiffusionModel
from .utils import get_default_config, save_image


# =========================================================================== #
#                                Argument Parser                              #
# =========================================================================== #
def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Parser for the CLI
    """
    config = get_default_config()

    # use default model unless specified otherwise
    default_model = next(
        (model["id"] for model in config["models"] if model.get("default")),
        config["models"][0]["id"],
    )

    # create the parser
    parser = argparse.ArgumentParser(
        description="Generate images using diffusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -------------------------------------------------------------------------
    # Add parser arguments
    # -------------------------------------------------------------------------
    parser.add_argument(
        "prompt", type=str, help="Text prompt for image generation"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_model,
        choices=[model["id"] for model in config["models"]],
        help="Diffusion model to use",
    )

    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=config["parameters"]["width"],
        help="Image width",
    )

    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=config["parameters"]["height"],
        help="Image height",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=config["parameters"]["num_inference_steps"],
        help="Number of inference steps",
    )

    parser.add_argument(
        "--guidance",
        "-g",
        type=float,
        default=config["parameters"]["guidance_scale"],
        help="Guidance scale",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config["parameters"]["seed"],
        help="Random seed (for reproducibility)",
    )

    parser.add_argument(
        "--negative-prompt",
        "-n",
        type=str,
        default=None,
        help="Negative prompt (what you don't want in the image)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=config["parameters"]["output_dir"],
        help="Directory to save generated images",
    )

    parser.add_argument(
        "--filename-prefix",
        "-f",
        type=str,
        default=config["parameters"]["filename_prefix"],
        help="Prefix for generated image filenames",
    )

    return parser


def main() -> None:
    """
    Main function to run the CLI.
    """
    parser = create_parser()
    args = parser.parse_args()

    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Using model🤖: '{args.model}'")

    try:
        model = DiffusionModel(args.model)
        print("Loading model and generating image...")
        print(
            """
            This may take a while ⏳ for the first run as models \
            are downloaded...
            """
        )

        image, actual_seed = model.generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )

        filepath = save_image(
            image=image,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
            prompt=args.prompt,
        )

        print(f"✅ Image generated successfully with seed: {actual_seed}")
        print(f"📁 Saved to: {filepath}")

    except Exception as e:
        print(f"❌ Error generating image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
