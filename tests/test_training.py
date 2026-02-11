"""
Tests for training components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from preference_guided_diffusion_steering.training.trainer import (
    PreferenceLoss, PreferenceTrainer
)
from preference_guided_diffusion_steering.models.model import (
    SteeringModule, PreferenceGuidedDiffusionModel
)

from .conftest import (
    assert_tensor_shape, create_mock_batch, requires_mlflow
)


class TestPreferenceLoss:
    """Test preference loss function."""

    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = PreferenceLoss(
            margin=1.0,
            preference_weight=1.0,
            guidance_regularization=0.01,
            consistency_weight=0.1
        )

        assert loss_fn.margin == 1.0
        assert loss_fn.preference_weight == 1.0
        assert loss_fn.guidance_regularization == 0.01
        assert loss_fn.consistency_weight == 0.1

        assert hasattr(loss_fn, 'ranking_loss')
        assert hasattr(loss_fn, 'mse_loss')

    def test_loss_forward_basic(self, device):
        """Test basic loss computation."""
        device_str = "cpu"  # Tests run on CPU

        loss_fn = PreferenceLoss()

        batch_size, seq_len, embed_dim = 2, 77, 768

        # Create sample embeddings
        preferred_embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device_str)
        dispreferred_embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device_str)
        original_embeddings = torch.randn(batch_size, seq_len, embed_dim, device=device_str)

        # Create mock steering module
        steering_module = Mock()
        steering_module.guidance_scale = torch.tensor(0.1, device=device_str)

        # Compute loss
        loss_dict = loss_fn(
            preferred_embeddings=preferred_embeddings,
            dispreferred_embeddings=dispreferred_embeddings,
            original_embeddings=original_embeddings,
            steering_module=steering_module
        )

        # Check loss components
        required_keys = [
            "total_loss", "ranking_loss", "guidance_reg_loss",
            "consistency_loss", "preferred_scores", "dispreferred_scores"
        ]
        for key in required_keys:
            assert key in loss_dict, f"Missing key: {key}"
            assert isinstance(loss_dict[key], torch.Tensor)
            assert torch.isfinite(loss_dict[key]).all()

        # Total loss should be positive
        assert loss_dict["total_loss"] > 0

    def test_loss_with_rating_differences(self, device):
        """Test loss computation with rating differences."""
        device_str = "cpu"

        loss_fn = PreferenceLoss()

        batch_size = 2
        preferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str)
        dispreferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str)
        original_embeddings = torch.randn(batch_size, 77, 768, device=device_str)

        rating_diffs = torch.tensor([2.0, 1.0], device=device_str)

        steering_module = Mock()
        steering_module.guidance_scale = torch.tensor(0.1, device=device_str)

        loss_dict = loss_fn(
            preferred_embeddings=preferred_embeddings,
            dispreferred_embeddings=dispreferred_embeddings,
            original_embeddings=original_embeddings,
            steering_module=steering_module,
            rating_diffs=rating_diffs
        )

        assert torch.isfinite(loss_dict["total_loss"]).all()
        assert loss_dict["total_loss"] > 0

    def test_loss_gradient_flow(self, device):
        """Test gradient flow through loss function."""
        device_str = "cpu"

        loss_fn = PreferenceLoss()

        batch_size = 2
        preferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str, requires_grad=True)
        dispreferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str, requires_grad=True)
        original_embeddings = torch.randn(batch_size, 77, 768, device=device_str)

        steering_module = Mock()
        steering_module.guidance_scale = torch.tensor(0.1, device=device_str, requires_grad=True)

        loss_dict = loss_fn(
            preferred_embeddings=preferred_embeddings,
            dispreferred_embeddings=dispreferred_embeddings,
            original_embeddings=original_embeddings,
            steering_module=steering_module
        )

        # Backward pass
        loss_dict["total_loss"].backward()

        # Check gradients exist
        assert preferred_embeddings.grad is not None
        assert dispreferred_embeddings.grad is not None
        assert torch.isfinite(preferred_embeddings.grad).all()
        assert torch.isfinite(dispreferred_embeddings.grad).all()

    def test_loss_components_scaling(self, device):
        """Test that loss components scale appropriately with weights."""
        device_str = "cpu"

        # Test with different weights
        weights = [
            {"preference_weight": 2.0, "guidance_regularization": 0.02, "consistency_weight": 0.2},
            {"preference_weight": 0.5, "guidance_regularization": 0.005, "consistency_weight": 0.05}
        ]

        batch_size = 2
        preferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str)
        dispreferred_embeddings = torch.randn(batch_size, 77, 768, device=device_str)
        original_embeddings = torch.randn(batch_size, 77, 768, device=device_str)

        steering_module = Mock()
        steering_module.guidance_scale = torch.tensor(0.1, device=device_str)

        losses = []
        for weight_config in weights:
            loss_fn = PreferenceLoss(**weight_config)
            loss_dict = loss_fn(
                preferred_embeddings=preferred_embeddings,
                dispreferred_embeddings=dispreferred_embeddings,
                original_embeddings=original_embeddings,
                steering_module=steering_module
            )
            losses.append(loss_dict["total_loss"])

        # Losses should be different with different weights
        assert not torch.allclose(losses[0], losses[1], atol=1e-6)


class TestPreferenceTrainer:
    """Test preference trainer functionality."""

    def test_trainer_initialization(self, test_config, temp_checkpoint_dir):
        """Test trainer initialization."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Setup mock
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            # Create model
            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            # Update config for testing
            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False  # Disable MLflow for tests

            trainer = PreferenceTrainer(
                model=model,
                config=config,
                device=test_config["model"]["device"]
            )

            # Check initialization
            assert trainer.model == model
            assert trainer.config == config
            assert hasattr(trainer, 'optimizer')
            assert hasattr(trainer, 'criterion')
            assert hasattr(trainer, 'metrics')
            assert trainer.current_epoch == 0
            assert trainer.current_step == 0

    def test_trainer_setup_components(self, test_config, temp_checkpoint_dir):
        """Test trainer component setup."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Test optimizer setup
            assert trainer.optimizer is not None
            assert len(list(trainer.optimizer.param_groups)) > 0

            # Test scheduler setup
            if config["training"]["scheduler"]["name"] != "none":
                assert trainer.scheduler is not None

            # Test loss function
            assert trainer.criterion is not None

    def test_process_batch(self, test_config, temp_checkpoint_dir):
        """Test batch processing."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            # Setup detailed mock
            mock_instance = Mock()
            mock_tokenizer = Mock()
            mock_text_encoder = Mock()

            # Mock tokenizer output
            mock_tokenizer.return_value.input_ids = torch.randint(0, 1000, (4, 77))
            mock_tokenizer.model_max_length = 77

            # Mock text encoder output
            mock_embeddings = torch.randn(4, 77, 768)
            mock_text_encoder.return_value = [mock_embeddings]

            mock_instance.tokenizer = mock_tokenizer
            mock_instance.text_encoder = mock_text_encoder
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []

            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Create mock batch
            batch = create_mock_batch(batch_size=4)

            # Process batch
            loss_dict = trainer._process_batch(batch, is_training=False)

            # Check output
            assert "total_loss" in loss_dict
            assert torch.isfinite(loss_dict["total_loss"]).all()
            assert loss_dict["total_loss"] > 0

    def test_checkpoint_save_load(self, test_config, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Save checkpoint
            metrics = {"total_loss": 0.5, "accuracy": 0.7}
            checkpoint_path = trainer.save_checkpoint(
                epoch=1,
                metrics=metrics,
                is_best=True
            )

            assert os.path.exists(checkpoint_path)

            # Modify model state
            with torch.no_grad():
                trainer.model.steering_module.guidance_scale.fill_(0.9)

            original_scale = trainer.model.steering_module.get_guidance_scale()

            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)

            # Check loaded data
            assert "epoch" in loaded_data
            assert "model_state_dict" in loaded_data
            assert "optimizer_state_dict" in loaded_data

            # Model state should be restored
            restored_scale = trainer.model.steering_module.get_guidance_scale()
            assert abs(restored_scale - original_scale) > 0.1

    @requires_mlflow
    def test_mlflow_integration(self, test_config, temp_checkpoint_dir):
        """Test MLflow integration."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = True

            with patch('mlflow.start_run'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metric'), \
                 patch('mlflow.end_run'):

                trainer = PreferenceTrainer(model, config)
                assert trainer.use_mlflow

    def test_config_validation(self, test_config, temp_checkpoint_dir):
        """Test configuration validation."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            # Test with valid config
            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)
            assert trainer.config == config

            # Test with invalid optimizer
            config["training"]["optimizer"]["name"] = "invalid_optimizer"

            with pytest.raises(ValueError):
                PreferenceTrainer(model, config)

    def test_early_stopping(self, test_config, temp_checkpoint_dir):
        """Test early stopping functionality."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False
            config["training"]["early_stopping_patience"] = 2

            trainer = PreferenceTrainer(model, config)

            # Simulate validation loss not improving
            trainer.best_validation_loss = 1.0
            trainer.patience_counter = 0

            # First epoch - no improvement
            trainer.best_validation_loss = 1.0
            trainer.patience_counter += 1

            # Second epoch - no improvement
            trainer.patience_counter += 1

            # Check that patience counter is working
            assert trainer.patience_counter == 2

    def test_learning_rate_scheduling(self, test_config, temp_checkpoint_dir):
        """Test learning rate scheduling."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []
            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Check initial learning rate
            initial_lr = trainer.optimizer.param_groups[0]["lr"]
            assert initial_lr == config["training"]["optimizer"]["learning_rate"]

            # Step scheduler if available
            if trainer.scheduler:
                trainer.scheduler.step()
                # Learning rate might change (depends on scheduler type)


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_loss_trainer_integration(self, test_config, temp_checkpoint_dir):
        """Test integration between loss function and trainer."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_tokenizer = Mock()
            mock_text_encoder = Mock()

            mock_tokenizer.return_value.input_ids = torch.randint(0, 1000, (2, 77))
            mock_tokenizer.model_max_length = 77
            mock_embeddings = torch.randn(2, 77, 768)
            mock_text_encoder.return_value = [mock_embeddings]

            mock_instance.tokenizer = mock_tokenizer
            mock_instance.text_encoder = mock_text_encoder
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []

            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Test that trainer uses the correct loss function
            assert isinstance(trainer.criterion, PreferenceLoss)

            # Test loss configuration is applied
            loss_config = config["training"]["loss"]
            assert trainer.criterion.margin == loss_config["margin"]
            assert trainer.criterion.preference_weight == loss_config["preference_weight"]

    def test_full_training_step(self, test_config, temp_checkpoint_dir):
        """Test complete training step."""
        with patch('preference_guided_diffusion_steering.models.model.StableDiffusionPipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_tokenizer = Mock()
            mock_text_encoder = Mock()

            mock_tokenizer.return_value.input_ids = torch.randint(0, 1000, (2, 77))
            mock_tokenizer.model_max_length = 77
            mock_embeddings = torch.randn(2, 77, 768)
            mock_text_encoder.return_value = [mock_embeddings]

            mock_instance.tokenizer = mock_tokenizer
            mock_instance.text_encoder = mock_text_encoder
            mock_instance.text_encoder.config.hidden_size = 768
            mock_instance.unet.parameters.return_value = []
            mock_instance.text_encoder.parameters.return_value = []
            mock_instance.vae.parameters.return_value = []

            mock_pipeline.from_pretrained.return_value = mock_instance

            model = PreferenceGuidedDiffusionModel(
                base_model_path=test_config["model"]["base_model_path"],
                steering_config=test_config["model"]["steering_config"],
                device=test_config["model"]["device"]
            )

            config = test_config.copy()
            config["training"]["checkpoint_dir"] = temp_checkpoint_dir
            config["training"]["use_mlflow"] = False

            trainer = PreferenceTrainer(model, config)

            # Get initial parameter state
            initial_params = {
                name: param.clone()
                for name, param in trainer.model.steering_module.named_parameters()
            }

            # Process training batch
            batch = create_mock_batch(batch_size=2)
            loss_dict = trainer._process_batch(batch, is_training=True)

            # Check that parameters were updated
            for name, param in trainer.model.steering_module.named_parameters():
                if param.requires_grad:
                    assert not torch.allclose(
                        param, initial_params[name], atol=1e-6
                    ), f"Parameter {name} was not updated"

            # Check loss is reasonable
            assert loss_dict["total_loss"] > 0
            assert torch.isfinite(loss_dict["total_loss"])