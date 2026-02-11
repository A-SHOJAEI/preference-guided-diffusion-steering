"""
Configuration management utilities.

This module provides utilities for loading, saving, and managing
configuration files for the preference-guided diffusion steering system.
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for preference-guided diffusion steering.

    This class handles loading and merging configuration files,
    environment variable overrides, and configuration validation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config_data = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config_data = yaml.safe_load(f)

            logger.info(f"Configuration loaded from {config_path}")

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Validate configuration
            self._validate_config()

        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            self.config_data = self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            "PGDS_MODEL_PATH": ("model", "base_model_path"),
            "PGDS_DEVICE": ("model", "device"),
            "PGDS_BATCH_SIZE": ("training", "batch_size"),
            "PGDS_LEARNING_RATE": ("training", "optimizer", "learning_rate"),
            "PGDS_NUM_EPOCHS": ("training", "num_epochs"),
            "PGDS_CHECKPOINT_DIR": ("training", "checkpoint_dir"),
            "PGDS_USE_MLFLOW": ("training", "use_mlflow"),
            "PGDS_EXPERIMENT_NAME": ("training", "experiment_name"),
            "PGDS_SEED": ("seed")
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string

                # Set nested configuration
                self._set_nested_config(config_path, value)

    def _set_nested_config(self, path: Union[str, tuple], value: Any) -> None:
        """Set nested configuration value."""
        if isinstance(path, str):
            path = (path,)

        current = self.config_data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[path[-1]] = value

    def _validate_config(self) -> None:
        """Validate configuration values."""
        required_sections = ["model", "training", "data"]

        for section in required_sections:
            if section not in self.config_data:
                logger.warning(f"Missing configuration section: {section}")
                self.config_data[section] = {}

        # Validate specific values
        training_config = self.config_data.get("training", {})

        # Batch size validation
        batch_size = training_config.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.warning(f"Invalid batch size: {batch_size}, using default: 32")
            training_config["batch_size"] = 32

        # Learning rate validation
        optimizer_config = training_config.get("optimizer", {})
        learning_rate = optimizer_config.get("learning_rate", 0.001)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            logger.warning(f"Invalid learning rate: {learning_rate}, using default: 0.001")
            optimizer_config["learning_rate"] = 0.001

        # Number of epochs validation
        num_epochs = training_config.get("num_epochs", 100)
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            logger.warning(f"Invalid num_epochs: {num_epochs}, using default: 100")
            training_config["num_epochs"] = 100

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "seed": 42,
            "model": {
                "base_model_path": "runwayml/stable-diffusion-v1-5",
                "device": "cuda",
                "enable_cpu_offload": False,
                "steering_config": {
                    "hidden_dim": 512,
                    "num_layers": 3,
                    "dropout_rate": 0.1,
                    "activation": "gelu",
                    "use_time_embedding": True,
                    "preference_dim": 256
                }
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 100,
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999]
                },
                "scheduler": {
                    "name": "cosine",
                    "T_max": 100,
                    "eta_min": 0.00001
                },
                "loss": {
                    "margin": 1.0,
                    "preference_weight": 1.0,
                    "guidance_regularization": 0.01,
                    "consistency_weight": 0.1
                },
                "checkpoint_dir": "checkpoints",
                "early_stopping_patience": 20,
                "min_delta": 0.001,
                "max_grad_norm": 1.0,
                "log_steps": 100,
                "use_mlflow": True,
                "experiment_name": "preference-guided-diffusion",
                "mlflow_tracking_uri": "file:./mlruns"
            },
            "data": {
                "preference_dataset": "openbmb/UltraFeedback",
                "caption_dataset": "conceptual_captions",
                "max_samples": None,
                "caption_max_samples": 10000,
                "min_rating_diff": 1.0,
                "batch_size": 32,
                "num_workers": 4,
                "preference_ratio": 0.7,
                "use_caption_dataset": False,
                "cache_dir": None,
                "data_processor": {
                    "tokenizer_name": "openai/clip-vit-base-patch32",
                    "max_text_length": 77,
                    "text_augmentation": True
                }
            },
            "evaluation": {
                "clip_model_name": "openai/clip-vit-base-patch32",
                "batch_size": 32,
                "metrics": {
                    "human_preference_win_rate": 0.65,
                    "clip_score_improvement": 0.08,
                    "fid_degradation_max": 5.0,
                    "preference_prediction_accuracy": 0.74,
                    "steering_latency_overhead_ms": 50
                }
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self.config_data

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config_data

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def update(self, other_config: Dict[str, Any]) -> None:
        """
        Update configuration with another dictionary.

        Args:
            other_config: Dictionary to merge into configuration
        """
        self._deep_update(self.config_data, other_config)

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def save(self, output_path: str) -> None:
        """
        Save configuration to file.

        Args:
            output_path: Path to save configuration
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(
                self.config_data,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=True
            )

        logger.info(f"Configuration saved to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config_data.copy()

    def __getitem__(self, key: str) -> Any:
        """Get configuration item."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration item."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return self.get(key) is not None


def load_config(config_path: str) -> Config:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)


def save_config(config: Union[Config, Dict[str, Any]], output_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration object or dictionary
        output_path: Path to save configuration
    """
    if isinstance(config, Config):
        config.save(output_path)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=True
            )

        logger.info(f"Configuration saved to {output_path}")


def merge_configs(*configs: Union[Config, Dict[str, Any]]) -> Config:
    """
    Merge multiple configurations.

    Args:
        *configs: Configuration objects or dictionaries to merge

    Returns:
        Merged configuration
    """
    merged = Config()

    for config in configs:
        if isinstance(config, Config):
            config_dict = config.to_dict()
        else:
            config_dict = config

        merged.update(config_dict)

    return merged


def create_experiment_config(
    base_config: Union[Config, Dict[str, Any]],
    experiment_params: Dict[str, Any]
) -> Config:
    """
    Create experiment-specific configuration.

    Args:
        base_config: Base configuration
        experiment_params: Experiment-specific parameters

    Returns:
        Experiment configuration
    """
    if isinstance(base_config, Config):
        exp_config = Config()
        exp_config.update(base_config.to_dict())
    else:
        exp_config = Config()
        exp_config.update(base_config)

    exp_config.update(experiment_params)

    return exp_config


def validate_config_completeness(config: Config) -> bool:
    """
    Validate that configuration contains all required fields.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is complete
    """
    required_fields = [
        "model.base_model_path",
        "model.device",
        "training.batch_size",
        "training.num_epochs",
        "training.optimizer.learning_rate",
        "data.preference_dataset"
    ]

    missing_fields = []
    for field in required_fields:
        if config.get(field) is None:
            missing_fields.append(field)

    if missing_fields:
        logger.error(f"Missing required configuration fields: {missing_fields}")
        return False

    return True