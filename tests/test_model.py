"""
Tests for model components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from preference_guided_diffusion_steering.models.model import (
    SteeringModule, PreferenceGuidedDiffusionModel
)

from .conftest import (
    assert_tensor_shape, assert_tensor_properties, assert_model_output_valid,
    requires_diffusion_model, requires_gpu
)


class TestSteeringModule:
    """Test steering module functionality."""

    def test_module_initialization(self, test_config):
        """Test steering module initialization."""
        config = test_config["model"]["steering_config"]
        module = SteeringModule(**config)

        assert module.text_embed_dim == config["text_embed_dim"]
        assert module.hidden_dim == config["hidden_dim"]
        assert module.num_layers == config["num_layers"]
        assert module.use_time_embedding == config["use_time_embedding"]

        # Check that layers are properly initialized
        assert hasattr(module, 'text_proj')
        assert hasattr(module, 'transformer')
        assert hasattr(module, 'output_proj')
        assert hasattr(module, 'guidance_scale')

        # Check guidance scale is learnable parameter
        assert isinstance(module.guidance_scale, torch.nn.Parameter)

    def test_module_forward_basic(self, steering_module, sample_text_embeddings, device):
        """Test basic forward pass."""
        batch_size, seq_len, embed_dim = sample_text_embeddings.shape

        output = steering_module(sample_text_embeddings)

        # Check output shape and properties
        assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
        assert_tensor_properties(output, "cpu")  # Tests run on CPU
        assert torch.isfinite(output).all()

        # Check output is bounded (due to tanh activation)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)

    def test_module_forward_with_timesteps(
        self, steering_module, sample_text_embeddings, sample_timesteps, device
    ):
        """Test forward pass with timestep conditioning."""
        output = steering_module(
            text_embeddings=sample_text_embeddings,
            timesteps=sample_timesteps.to("cpu")
        )

        batch_size, seq_len, embed_dim = sample_text_embeddings.shape
        assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
        assert torch.isfinite(output).all()

    def test_module_forward_with_preferences(
        self, steering_module, sample_text_embeddings, sample_preferences, device
    ):
        """Test forward pass with preference conditioning."""
        output = steering_module(
            text_embeddings=sample_text_embeddings,
            preferences=sample_preferences.to("cpu")
        )

        batch_size, seq_len, embed_dim = sample_text_embeddings.shape
        assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
        assert torch.isfinite(output).all()

    def test_module_forward_full_conditioning(
        self, steering_module, sample_text_embeddings, sample_timesteps,
        sample_preferences, device
    ):
        """Test forward pass with all conditioning inputs."""
        device_str = "cpu"  # Tests run on CPU

        output = steering_module(
            text_embeddings=sample_text_embeddings,
            timesteps=sample_timesteps.to(device_str),
            preferences=sample_preferences.to(device_str)
        )

        batch_size, seq_len, embed_dim = sample_text_embeddings.shape
        assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
        assert torch.isfinite(output).all()

    def test_guidance_scale_parameter(self, steering_module):
        """Test guidance scale parameter functionality."""
        # Initial value should be around 0.1
        initial_scale = steering_module.get_guidance_scale()
        assert isinstance(initial_scale, float)
        assert 0.0 <= abs(initial_scale) <= 1.0  # Should be reasonable

        # Should be trainable
        assert steering_module.guidance_scale.requires_grad

        # Test manual update
        with torch.no_grad():
            steering_module.guidance_scale.fill_(0.5)

        assert abs(steering_module.get_guidance_scale() - 0.5) < 1e-6

    def test_module_different_batch_sizes(self, steering_module, device):
        """Test module with different batch sizes."""
        device_str = "cpu"

        batch_sizes = [1, 4, 8]
        seq_len, embed_dim = 77, 768

        for batch_size in batch_sizes:
            embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device_str)
            output = steering_module(embeddings)

            assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
            assert torch.isfinite(output).all()

    def test_module_gradient_flow(self, steering_module, sample_text_embeddings, device):
        """Test gradient flow through the module."""
        sample_text_embeddings.requires_grad_(True)

        output = steering_module(sample_text_embeddings)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert sample_text_embeddings.grad is not None
        assert torch.isfinite(sample_text_embeddings.grad).all()

        # Check module parameters have gradients
        for param in steering_module.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_module_weight_initialization(self, test_config):
        """Test weight initialization."""
        config = test_config["model"]["steering_config"]
        module = SteeringModule(**config)

        # Check that weights are initialized (not zero)
        for name, param in module.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param, torch.zeros_like(param))
            elif 'bias' in name and param is not None:
                # Biases should be initialized to zero
                assert torch.allclose(param, torch.zeros_like(param))


class TestPreferenceGuidedDiffusionModel:
    """Test preference-guided diffusion model functionality."""

    @requires_diffusion_model
    def test_model_initialization(self, test_config):
        """Test model initialization."""
        config = test_config["model"]
        model = PreferenceGuidedDiffusionModel(
            base_model_path=config["base_model_path"],
            steering_config=config["steering_config"],
            device=config["device"]
        )

        assert hasattr(model, 'pipeline')
        assert hasattr(model, 'steering_module')
        assert model.device == config["device"]

        # Check that base model parameters are frozen
        for param in model.pipeline.unet.parameters():
            assert not param.requires_grad

    def test_mock_model_initialization(self, test_config):
        """Test model initialization with mocked diffusion pipeline."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Mock pipeline components
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.tokenizer.model_max_length = 77
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            assert hasattr(model, 'steering_module')
            assert isinstance(model.steering_module, SteeringModule)

    def test_encode_prompts_mock(self, test_config, sample_prompts):
        """Test prompt encoding with mocked components."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Setup mocks
            mock_instance = Mock()
            mock_tokenizer = Mock()
            mock_text_encoder = Mock()

            # Mock tokenizer
            mock_tokenizer.return_value.input_ids = torch.randint(0, 1000, (len(sample_prompts), 77))
            mock_tokenizer.model_max_length = 77

            # Mock text encoder
            mock_embeddings = torch.randn(len(sample_prompts), 77, 768)
            mock_text_encoder.return_value = [mock_embeddings]

            # Setup pipeline mock
            mock_instance.tokenizer = mock_tokenizer
            mock_instance.text_encoder = mock_text_encoder
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []

            mock_pipeline.from_pretrained.return_value = mock_instance

            # Create model and test
            config = test_config["model"]
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            positive_embeddings, negative_embeddings = model.encode_prompts(sample_prompts[:2])

            assert positive_embeddings.shape[0] == 2
            assert negative_embeddings.shape[0] == 2
            assert positive_embeddings.shape[-1] == 768  # embedding dimension

    def test_apply_preference_steering(self, test_config, device):
        """Test preference steering application."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Setup mocks
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            # Test steering
            batch_size = 2
            text_embeddings = torch.randn(batch_size, 77, 768)
            timesteps = torch.randint(0, 1000, (batch_size,))
            preferences = torch.randint(0, 2, (batch_size,))

            steered_embeddings = model.apply_preference_steering(
                text_embeddings, timesteps, preferences
            )

            assert steered_embeddings.shape == text_embeddings.shape
            assert torch.isfinite(steered_embeddings).all()

            # Steered embeddings should be different from original
            assert not torch.allclose(steered_embeddings, text_embeddings, atol=1e-6)

    def test_save_load_steering_module(self, test_config, temp_checkpoint_dir):
        """Test saving and loading steering module."""
        import os

        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Setup mocks
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            # Save steering module
            save_path = os.path.join(temp_checkpoint_dir, "steering_module.pt")
            model.save_steering_module(save_path)

            assert os.path.exists(save_path)

            # Modify model parameters
            with torch.no_grad():
                model.steering_module.guidance_scale.fill_(0.8)

            original_scale = model.steering_module.get_guidance_scale()

            # Load steering module
            model.load_steering_module(save_path)

            # Parameters should be restored
            restored_scale = model.steering_module.get_guidance_scale()
            assert abs(restored_scale - original_scale) > 0.1  # Should be different


class TestModelIntegration:
    """Integration tests for model components."""

    def test_steering_module_with_different_configs(self):
        """Test steering module with various configurations."""
        configs = [
            {"hidden_dim": 256, "num_layers": 2, "dropout_rate": 0.0},
            {"hidden_dim": 512, "num_layers": 4, "dropout_rate": 0.2},
            {"hidden_dim": 128, "num_layers": 1, "use_time_embedding": False}
        ]

        batch_size, seq_len, embed_dim = 2, 77, 768

        for config in configs:
            module = SteeringModule(text_embed_dim=embed_dim, **config)
            embeddings = torch.randn(batch_size, seq_len, embed_dim)

            output = module(embeddings)

            assert output.shape == embeddings.shape
            assert torch.isfinite(output).all()

    def test_model_memory_efficiency(self, test_config):
        """Test model memory efficiency."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]

            # Test CPU offload option
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"],
                enable_cpu_offload=True
            )

            assert hasattr(model, 'steering_module')

    def test_model_parameter_count(self, test_config):
        """Test model parameter counting."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]
            model = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            # Count trainable parameters
            trainable_params = sum(
                p.numel() for p in model.steering_module.parameters() if p.requires_grad
            )

            # Should have reasonable number of parameters
            assert 10_000 < trainable_params < 10_000_000  # Between 10K and 10M

    def test_model_reproducibility(self, test_config):
        """Test model reproducibility with seeds."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            config = test_config["model"]

            # Set seed and create model
            torch.manual_seed(42)
            model1 = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            # Reset seed and create another model
            torch.manual_seed(42)
            model2 = PreferenceGuidedDiffusionModel(
                base_model_path=config["base_model_path"],
                steering_config=config["steering_config"],
                device=config["device"]
            )

            # Models should have identical parameters
            for (name1, param1), (name2, param2) in zip(
                model1.steering_module.named_parameters(),
                model2.steering_module.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2, atol=1e-6)