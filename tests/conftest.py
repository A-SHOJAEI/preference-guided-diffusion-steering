"""
Pytest configuration and fixtures for preference-guided diffusion steering tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preference_guided_diffusion_steering.models.model import (
    SteeringModule, PreferenceGuidedDiffusionModel
)
from preference_guided_diffusion_steering.data.loader import (
    UltraFeedbackDataset, PreferenceDataLoader
)
from preference_guided_diffusion_steering.data.preprocessing import (
    PreferenceDataProcessor, TextImagePairProcessor
)
from preference_guided_diffusion_steering.training.trainer import PreferenceTrainer
from preference_guided_diffusion_steering.evaluation.metrics import PreferenceMetrics
from preference_guided_diffusion_steering.utils.config import Config


@pytest.fixture(scope="session")
def device():
    """Test device fixture."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "seed": 42,
        "model": {
            "base_model_path": "runwayml/stable-diffusion-v1-5",
            "device": "cpu",  # Use CPU for tests
            "enable_cpu_offload": False,
            "steering_config": {
                "text_embed_dim": 768,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "activation": "gelu",
                "use_time_embedding": True,
                "preference_dim": 128
            }
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 2,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.01
            },
            "scheduler": {
                "name": "cosine",
                "T_max": 10,
                "eta_min": 0.00001
            },
            "loss": {
                "margin": 1.0,
                "preference_weight": 1.0,
                "guidance_regularization": 0.01,
                "consistency_weight": 0.1
            },
            "checkpoint_dir": "test_checkpoints",
            "early_stopping_patience": 5,
            "max_grad_norm": 1.0,
            "log_steps": 10,
            "use_mlflow": False
        },
        "data": {
            "max_samples": 20,
            "min_rating_diff": 0.5,
            "batch_size": 4,
            "num_workers": 0,  # No multiprocessing for tests
            "preference_ratio": 0.7,
            "use_caption_dataset": False,
            "data_processor": {
                "tokenizer_name": "openai/clip-vit-base-patch32",
                "max_text_length": 77,
                "text_augmentation": False  # Disable for consistent tests
            }
        }
    }


@pytest.fixture
def steering_module(test_config, device):
    """Steering module fixture."""
    config = test_config["model"]["steering_config"]
    module = SteeringModule(**config)
    return module.to(device)


@pytest.fixture
def sample_text_embeddings(device):
    """Sample text embeddings fixture."""
    batch_size = 2
    seq_len = 77
    embed_dim = 768
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


@pytest.fixture
def sample_timesteps(device):
    """Sample timesteps fixture."""
    batch_size = 2
    return torch.randint(0, 1000, (batch_size,), device=device)


@pytest.fixture
def sample_preferences(device):
    """Sample preferences fixture."""
    batch_size = 2
    return torch.randint(0, 2, (batch_size,), device=device)


@pytest.fixture
def sample_prompts():
    """Sample text prompts fixture."""
    return [
        "A beautiful sunset over the ocean",
        "A cute cat playing with yarn",
        "Modern architecture in the city",
        "Abstract art with vibrant colors"
    ]


@pytest.fixture
def sample_images():
    """Sample PIL images fixture."""
    images = []
    for i in range(4):
        # Create random colored images
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))
    return images


@pytest.fixture
def preference_dataset(test_config):
    """Preference dataset fixture."""
    return UltraFeedbackDataset(
        max_samples=test_config["data"]["max_samples"],
        min_rating_diff=test_config["data"]["min_rating_diff"],
        seed=test_config["seed"]
    )


@pytest.fixture
def preference_data_processor(test_config):
    """Preference data processor fixture."""
    config = test_config["data"]["data_processor"]
    return PreferenceDataProcessor(**config)


@pytest.fixture
def sample_preference_pairs():
    """Sample preference pairs fixture."""
    return [
        {
            "prompt": "A beautiful landscape",
            "preferred_response": "A stunning landscape with mountains",
            "dispreferred_response": "A landscape",
            "preferred_label": 1,
            "dispreferred_label": 0,
            "rating_diff": 2.0
        },
        {
            "prompt": "A portrait of a person",
            "preferred_response": "A detailed portrait of a person",
            "dispreferred_response": "A blurry portrait",
            "preferred_label": 1,
            "dispreferred_label": 0,
            "rating_diff": 1.5
        }
    ]


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary checkpoint directory fixture."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config_object(test_config):
    """Config object fixture."""
    config = Config()
    config.update(test_config)
    return config


@pytest.fixture
def preference_metrics(device):
    """Preference metrics fixture."""
    return PreferenceMetrics(device=device)


@pytest.fixture
def mock_evaluation_data(sample_images, sample_prompts):
    """Mock evaluation data fixture."""
    return {
        "preferred_images": sample_images[:2],
        "dispreferred_images": sample_images[2:4],
        "original_images": sample_images[:2],
        "steered_images": sample_images[2:4],
        "prompts": sample_prompts[:2],
        "predicted_preferences": np.array([1, 0]),
        "true_preferences": np.array([1, 1]),
        "preference_scores": np.array([0.8, 0.3]),
        "steering_latencies": [45.2, 52.1]
    }


# Test utilities
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_properties(tensor: torch.Tensor, device: str, dtype: torch.dtype = None):
    """Assert tensor device and dtype."""
    assert tensor.device.type == device, f"Expected device {device}, got {tensor.device}"
    if dtype is not None:
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"


def create_mock_batch(batch_size: int = 4) -> Dict[str, Any]:
    """Create mock training batch."""
    prompts = [f"Test prompt {i}" for i in range(batch_size)]
    return {
        "prompts": prompts,
        "preferred_labels": torch.ones(batch_size, dtype=torch.long),
        "dispreferred_labels": torch.zeros(batch_size, dtype=torch.long),
        "rating_diffs": torch.ones(batch_size) * 2.0,
        "data_type": "preference"
    }


def assert_model_output_valid(output: torch.Tensor, input_shape: tuple):
    """Assert model output is valid."""
    assert torch.isfinite(output).all(), "Model output contains non-finite values"
    assert output.shape == input_shape, f"Output shape {output.shape} doesn't match input {input_shape}"


def create_synthetic_preference_data(num_samples: int = 10) -> List[Dict[str, Any]]:
    """Create synthetic preference data for testing."""
    data = []
    for i in range(num_samples):
        data.append({
            "prompt": f"Test prompt {i}",
            "preferred_response": f"High quality response {i}",
            "dispreferred_response": f"Low quality response {i}",
            "preferred_label": 1,
            "dispreferred_label": 0,
            "rating_diff": np.random.uniform(1.0, 3.0)
        })
    return data


# Skip marks for tests requiring external dependencies
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

requires_mlflow = pytest.mark.skipif(
    True,  # Skip MLflow tests by default to avoid dependencies
    reason="MLflow not available or disabled for testing"
)

requires_clip = pytest.mark.skipif(
    True,  # Skip CLIP tests by default
    reason="CLIP models not available or disabled for testing"
)

requires_diffusion_model = pytest.mark.skipif(
    True,  # Skip full diffusion model tests by default
    reason="Diffusion models not available or disabled for testing"
)