"""
Inference script for preference-guided diffusion steering.

This script loads a trained steering module and generates images based on
text prompts with controllable preference steering.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preference_guided_diffusion_steering.models.model import PreferenceGuidedDiffusionModel
from preference_guided_diffusion_steering.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images with preference-guided diffusion steering"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_checkpoint.pt",
        help="Path to trained steering module checkpoint"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["A beautiful sunset over the ocean"],
        help="Text prompts for image generation"
    )
    parser.add_argument(
        "--preferences",
        type=int,
        nargs="+",
        default=None,
        help="Preference values (1=preferred, 0=dispreferred) for each prompt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()


def generate_images(
    model: PreferenceGuidedDiffusionModel,
    prompts: List[str],
    preferences: List[int],
    output_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42
) -> List[Image.Image]:
    """
    Generate images using the preference-guided model.

    Args:
        model: Trained preference-guided diffusion model
        prompts: List of text prompts
        preferences: List of preference values (1=preferred, 0=dispreferred)
        output_dir: Directory to save images
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed

    Returns:
        List of generated PIL images
    """
    logger.info(f"Generating {len(prompts)} images...")

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    images = model.generate_images(
        prompts=prompts,
        preferences=preferences,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # Save images
    for i, (prompt, image) in enumerate(zip(prompts, images)):
        # Create safe filename from prompt
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')

        preference_str = "preferred" if preferences[i] == 1 else "dispreferred"
        filename = f"{i:03d}_{safe_prompt}_{preference_str}.png"
        filepath = os.path.join(output_dir, filename)

        image.save(filepath)
        logger.info(f"Saved image to {filepath}")

    return images


def main():
    """Main inference function."""
    args = parse_args()

    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Set preferences (default to preferred if not specified)
    if args.preferences is None:
        preferences = [1] * len(args.prompts)
    else:
        preferences = args.preferences
        if len(preferences) != len(args.prompts):
            raise ValueError(
                f"Number of preferences ({len(preferences)}) must match "
                f"number of prompts ({len(args.prompts)})"
            )

    logger.info("Loading model...")
    model = PreferenceGuidedDiffusionModel(
        base_model_path=config["model"]["base_model_path"],
        steering_config=config["model"]["steering_config"],
        device=args.device
    )

    # Load trained steering module
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_steering_module(args.checkpoint)
    else:
        logger.warning(f"Checkpoint not found at {args.checkpoint}. Using untrained model.")

    # Generate images
    logger.info(f"Generating images with preferences: {preferences}")
    for i, (prompt, pref) in enumerate(zip(args.prompts, preferences)):
        pref_str = "preferred" if pref == 1 else "dispreferred"
        logger.info(f"  {i+1}. [{pref_str}] {prompt}")

    images = generate_images(
        model=model,
        prompts=args.prompts,
        preferences=preferences,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    logger.info(f"Successfully generated {len(images)} images")
    logger.info(f"Images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
