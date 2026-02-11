#!/usr/bin/env python3
"""
Training script for preference-guided diffusion steering.

This script trains a lightweight steering module that guides text-to-image diffusion
models toward human-preferred aesthetic and semantic qualities using UltraFeedback
preference data.
"""

import argparse
import logging
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preference_guided_diffusion_steering.models.model import PreferenceGuidedDiffusionModel
from preference_guided_diffusion_steering.training.trainer import PreferenceTrainer
from preference_guided_diffusion_steering.data.loader import create_data_loaders
from preference_guided_diffusion_steering.utils.config import load_config, validate_config_completeness
from preference_guided_diffusion_steering.evaluation.metrics import PreferenceMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train preference-guided diffusion steering model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, auto-detect"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory from config"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, more logging)"
    )

    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for debugging)"
    )

    return parser.parse_args()


def setup_device(config_device: str, override_device: str = None) -> str:
    """Setup and validate device configuration."""
    if override_device:
        device = override_device
    else:
        device = config_device

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.info("Using CPU device")

    return device


def create_model(config: dict, device: str) -> PreferenceGuidedDiffusionModel:
    """Create and initialize the preference-guided diffusion model."""
    logger.info("Initializing preference-guided diffusion model...")

    try:
        model = PreferenceGuidedDiffusionModel(
            base_model_path=config["model"]["base_model_path"],
            steering_config=config["model"]["steering_config"],
            device=device,
            enable_cpu_offload=config["model"]["enable_cpu_offload"]
        )

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in model.steering_module.parameters() if p.requires_grad
        )

        logger.info(f"Model initialized successfully")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Base model: {config['model']['base_model_path']}")

        return model

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def create_data_loaders_with_config(config: dict, debug: bool = False) -> dict:
    """Create data loaders for training and validation."""
    logger.info("Creating data loaders...")

    # Adjust config for debug mode
    if debug:
        config["data"]["max_samples"] = min(config["data"].get("max_samples", 100), 100)
        config["data"]["caption_max_samples"] = min(config["data"].get("caption_max_samples", 50), 50)
        config["data"]["num_workers"] = 0  # Disable multiprocessing in debug mode
        logger.info("Debug mode: Using smaller dataset")

    try:
        data_loaders = create_data_loaders(
            config["data"],
            splits=["train", "validation"] if not debug else ["train"]
        )

        logger.info(f"Created data loaders for splits: {list(data_loaders.keys())}")
        for split, loader in data_loaders.items():
            logger.info(f"{split} dataset size: {len(loader)}")

        return data_loaders

    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        # Create minimal synthetic data loaders for development
        logger.info("Creating synthetic data loaders for development...")
        return create_synthetic_data_loaders(config)


def create_synthetic_data_loaders(config: dict) -> dict:
    """Create synthetic data loaders for development/testing."""
    from preference_guided_diffusion_steering.data.loader import UltraFeedbackDataset, PreferenceDataLoader

    # Create small synthetic datasets with force_synthetic=True to ensure synthetic data generation
    train_dataset = UltraFeedbackDataset(
        max_samples=20,
        min_rating_diff=0.5,
        seed=config["seed"],
        force_synthetic=True
    )

    val_dataset = UltraFeedbackDataset(
        max_samples=10,
        min_rating_diff=0.5,
        seed=config["seed"] + 1,
        force_synthetic=True
    )

    train_loader = PreferenceDataLoader(
        preference_dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0
    )

    val_loader = PreferenceDataLoader(
        preference_dataset=val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    return {"train": train_loader, "validation": val_loader}


def main():
    """Main training function."""
    args = parse_arguments()

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        config = load_config(args.config)

        # Validate configuration
        if not validate_config_completeness(config):
            logger.error("Configuration validation failed")
            sys.exit(1)

        # Apply command line overrides
        if args.device:
            config.set("model.device", args.device)
        if args.batch_size:
            config.set("training.batch_size", args.batch_size)
            config.set("data.batch_size", args.batch_size)
        if args.learning_rate:
            config.set("training.optimizer.learning_rate", args.learning_rate)
        if args.num_epochs:
            config.set("training.num_epochs", args.num_epochs)
        if args.checkpoint_dir:
            config.set("training.checkpoint_dir", args.checkpoint_dir)
        if args.disable_mlflow:
            config.set("training.use_mlflow", False)
        if args.max_samples:
            config.set("data.max_samples", args.max_samples)

        # Set up device
        device = setup_device(config.get("model.device"), args.device)
        config.set("model.device", device)

        # Set random seed
        set_seed(config.get("seed"))
        logger.info(f"Set random seed to {config.get('seed')}")

        # Create directories
        checkpoint_dir = Path(config.get("training.checkpoint_dir"))
        checkpoint_dir.mkdir(exist_ok=True)

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Create model
        model = create_model(config.to_dict(), device)

        # Create data loaders
        data_loaders = create_data_loaders_with_config(config.to_dict(), args.debug)

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = PreferenceTrainer(
            model=model,
            config=config.to_dict(),
            device=device
        )

        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                logger.info(f"Resuming training from {args.resume}")
                trainer.load_checkpoint(args.resume)
            else:
                logger.error(f"Resume checkpoint not found: {args.resume}")
                sys.exit(1)

        # Start training
        logger.info("Starting training...")
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {config.get('training.num_epochs')}")
        logger.info(f"  - Batch size: {config.get('training.batch_size')}")
        logger.info(f"  - Learning rate: {config.get('training.optimizer.learning_rate')}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - MLflow: {'enabled' if config.get('training.use_mlflow') else 'disabled'}")

        # Train the model
        train_loader = data_loaders["train"]
        val_loader = data_loaders.get("validation", None)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get("training.num_epochs")
        )

        # Save final model
        final_model_path = checkpoint_dir / "final_steering_module.pt"
        model.save_steering_module(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        # Save training history
        import json
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy types for JSON serialization
            history_serializable = {}
            for key, values in history.items():
                if isinstance(values, list):
                    history_serializable[key] = [float(v) for v in values]
                else:
                    history_serializable[key] = float(values) if hasattr(values, 'item') else values

            json.dump(history_serializable, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # Run basic evaluation
        logger.info("Running post-training evaluation...")
        try:
            metrics = PreferenceMetrics(device=device)

            # Create some test data for evaluation
            test_prompts = [
                "A beautiful sunset over the ocean",
                "A cute cat playing with yarn",
                "Modern architecture in the city",
                "Abstract art with vibrant colors"
            ]

            # Generate images with different preferences
            preferred_images = model.generate_images(
                test_prompts,
                preferences=[1] * len(test_prompts),
                num_inference_steps=20  # Faster for evaluation
            )

            dispreferred_images = model.generate_images(
                test_prompts,
                preferences=[0] * len(test_prompts),
                num_inference_steps=20
            )

            # Evaluate preferences
            eval_results = metrics.evaluate_human_preference_simulation(
                preferred_images, dispreferred_images, test_prompts
            )

            logger.info("Evaluation results:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    logger.info(f"  - {key}: {value:.4f}")
                else:
                    logger.info(f"  - {key}: {value}")

            # Save evaluation results
            eval_path = results_dir / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                # Convert numpy types for JSON serialization
                eval_serializable = {}
                for key, value in eval_results.items():
                    if hasattr(value, 'tolist'):
                        eval_serializable[key] = value.tolist()
                    elif hasattr(value, 'item'):
                        eval_serializable[key] = value.item()
                    else:
                        eval_serializable[key] = value

                json.dump(eval_serializable, f, indent=2)
            logger.info(f"Evaluation results saved to {eval_path}")

        except Exception as e:
            logger.warning(f"Post-training evaluation failed: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Model checkpoints saved in: {checkpoint_dir}")
        logger.info(f"Results saved in: {results_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()