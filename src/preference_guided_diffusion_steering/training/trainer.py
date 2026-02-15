"""
Training pipeline for preference-guided diffusion steering.

This module implements the main training loop with preference learning,
model checkpointing, early stopping, and MLflow tracking integration.
"""

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import time
from pathlib import Path

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from ..models.model import PreferenceGuidedDiffusionModel, SteeringModule
from ..data.loader import PreferenceDataLoader
from ..data.preprocessing import PreferenceDataProcessor
from ..evaluation.metrics import PreferenceMetrics

logger = logging.getLogger(__name__)


class PreferenceLoss(nn.Module):
    """
    Custom loss function for preference learning that combines
    preference ranking loss with regularization terms.
    """

    def __init__(
        self,
        margin: float = 1.0,
        preference_weight: float = 1.0,
        guidance_regularization: float = 0.01,
        consistency_weight: float = 0.1
    ):
        """
        Initialize preference loss.

        Args:
            margin: Margin for ranking loss
            preference_weight: Weight for preference ranking loss
            guidance_regularization: Weight for guidance scale regularization
            consistency_weight: Weight for consistency regularization
        """
        super().__init__()
        self.margin = margin
        self.preference_weight = preference_weight
        self.guidance_regularization = guidance_regularization
        self.consistency_weight = consistency_weight

        # Loss functions
        # Use reduction='none' so we can apply per-sample weighting by rating_diffs,
        # then manually average. When rating_diffs is not provided, we just call .mean().
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        preferred_embeddings: torch.Tensor,
        dispreferred_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        steering_module: SteeringModule,
        rating_diffs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute preference loss.

        Args:
            preferred_embeddings: Embeddings for preferred examples
            dispreferred_embeddings: Embeddings for dispreferred examples
            original_embeddings: Original text embeddings
            steering_module: The steering module for regularization
            rating_diffs: Preference rating differences

        Returns:
            Dictionary containing individual loss components
        """
        batch_size = preferred_embeddings.size(0)

        # Compute preference scores (similarity to original embeddings)
        preferred_scores = torch.cosine_similarity(
            preferred_embeddings.view(batch_size, -1),
            original_embeddings.view(batch_size, -1),
            dim=1
        )

        dispreferred_scores = torch.cosine_similarity(
            dispreferred_embeddings.view(batch_size, -1),
            original_embeddings.view(batch_size, -1),
            dim=1
        )

        # Ranking loss - preferred should have higher scores
        targets = torch.ones(batch_size, device=preferred_embeddings.device)

        # Compute per-sample ranking loss (reduction='none')
        per_sample_ranking_loss = self.ranking_loss(
            preferred_scores, dispreferred_scores, targets
        )

        if rating_diffs is not None:
            # Weight the per-sample loss by normalized rating differences
            weights = torch.clamp(rating_diffs / rating_diffs.max(), min=0.1, max=1.0)
            ranking_loss = (per_sample_ranking_loss * weights).mean()
        else:
            ranking_loss = per_sample_ranking_loss.mean()

        # Guidance scale regularization - prevent too large guidance scales
        guidance_reg_loss = self.guidance_regularization * torch.abs(
            steering_module.guidance_scale
        )

        # Consistency regularization - encourage smooth guidance modifications
        preferred_guidance = preferred_embeddings - original_embeddings
        dispreferred_guidance = dispreferred_embeddings - original_embeddings

        # L2 regularization on guidance modifications
        consistency_loss = self.consistency_weight * (
            torch.norm(preferred_guidance, p=2, dim=-1).mean() +
            torch.norm(dispreferred_guidance, p=2, dim=-1).mean()
        )

        # Total loss
        total_loss = (
            self.preference_weight * ranking_loss +
            guidance_reg_loss +
            consistency_loss
        )

        return {
            "total_loss": total_loss,
            "ranking_loss": ranking_loss,
            "guidance_reg_loss": guidance_reg_loss,
            "consistency_loss": consistency_loss,
            "preferred_scores": preferred_scores,
            "dispreferred_scores": dispreferred_scores
        }


class PreferenceTrainer:
    """
    Main trainer class for preference-guided diffusion steering.

    Handles the complete training pipeline including data loading,
    model training, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: PreferenceGuidedDiffusionModel,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the preference trainer.

        Args:
            model: Preference-guided diffusion model
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_data_processor()
        self._setup_metrics()
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_validation_loss = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info("Preference trainer initialized successfully")

    def _setup_optimizer(self) -> None:
        """Setup optimizer for training."""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_name = optimizer_config.get("name", "adamw")
        learning_rate = optimizer_config.get("learning_rate", 0.001)
        weight_decay = optimizer_config.get("weight_decay", 0.01)

        # Only train steering module parameters
        trainable_params = self.model.steering_module.parameters()

        if optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Initialized {optimizer_name} optimizer with lr={learning_rate}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get("scheduler", {})
        scheduler_name = scheduler_config.get("name", "cosine")

        if scheduler_name.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 0.00001)
            )
        elif scheduler_name.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                verbose=True
            )
        else:
            self.scheduler = None

        if self.scheduler:
            logger.info(f"Initialized {scheduler_name} scheduler")

    def _setup_loss_function(self) -> None:
        """Setup loss function."""
        loss_config = self.config.get("loss", {})

        self.criterion = PreferenceLoss(
            margin=loss_config.get("margin", 1.0),
            preference_weight=loss_config.get("preference_weight", 1.0),
            guidance_regularization=loss_config.get("guidance_regularization", 0.01),
            consistency_weight=loss_config.get("consistency_weight", 0.1)
        ).to(self.device)

        logger.info("Initialized preference loss function")

    def _setup_data_processor(self) -> None:
        """Setup data processor."""
        processor_config = self.config.get("data_processor", {})

        self.data_processor = PreferenceDataProcessor(
            tokenizer_name=processor_config.get("tokenizer_name", "openai/clip-vit-base-patch32"),
            max_text_length=processor_config.get("max_text_length", 77),
            text_augmentation=processor_config.get("text_augmentation", True),
            seed=self.config.get("seed", 42)
        )

        logger.info("Initialized data processor")

    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.metrics = PreferenceMetrics(device=self.device)
        logger.info("Initialized evaluation metrics")

    def _setup_logging(self) -> None:
        """Setup MLflow logging."""
        self.use_mlflow = MLFLOW_AVAILABLE and self.config.get("use_mlflow", True)

        if self.use_mlflow:
            try:
                # Set MLflow tracking URI
                tracking_uri = self.config.get("mlflow_tracking_uri", "file:./mlruns")
                mlflow.set_tracking_uri(tracking_uri)

                # Start MLflow run
                experiment_name = self.config.get("experiment_name", "preference-guided-diffusion")
                mlflow.set_experiment(experiment_name)

                mlflow.start_run(
                    run_name=f"training_{int(time.time())}"
                )

                # Log configuration
                mlflow.log_params(self._flatten_config(self.config))

                logger.info("MLflow logging initialized")

            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self.use_mlflow = False
        else:
            logger.info("MLflow logging disabled")

    def _flatten_config(self, config: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration for MLflow logging."""
        flattened = {}
        for key, value in config.items():
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, f"{prefix}{key}_"))
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened

    def train_epoch(
        self,
        train_loader: PreferenceDataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.model.steering_module.train()

        total_loss = 0.0
        total_ranking_loss = 0.0
        total_guidance_reg = 0.0
        total_consistency_loss = 0.0
        num_batches = 0

        # Create progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                # Process batch
                loss_dict = self._process_batch(batch, is_training=True)

                # Update metrics
                total_loss += loss_dict["total_loss"].item()
                total_ranking_loss += loss_dict["ranking_loss"].item()
                total_guidance_reg += loss_dict["guidance_reg_loss"].item()
                total_consistency_loss += loss_dict["consistency_loss"].item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss'].item():.4f}",
                    "rank": f"{loss_dict['ranking_loss'].item():.4f}",
                    "guide": f"{loss_dict['guidance_reg_loss'].item():.4f}"
                })

                # Log step metrics
                self.current_step += 1
                if self.current_step % self.config.get("log_steps", 100) == 0:
                    self._log_step_metrics(loss_dict, "train")

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}", exc_info=True)
                if num_batches == 0 and batch_idx >= 2:
                    # If the first 3 batches all fail, raise the error instead of
                    # silently continuing with zero metrics for the entire epoch
                    raise RuntimeError(
                        f"First {batch_idx + 1} training batches all failed. "
                        f"Last error: {e}"
                    ) from e
                continue

        if num_batches == 0:
            logger.error(
                "All training batches failed! Metrics will be zero. "
                "Check error messages above for details."
            )

        # Calculate epoch averages
        avg_metrics = {
            "total_loss": total_loss / max(num_batches, 1),
            "ranking_loss": total_ranking_loss / max(num_batches, 1),
            "guidance_reg_loss": total_guidance_reg / max(num_batches, 1),
            "consistency_loss": total_consistency_loss / max(num_batches, 1),
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }

        return avg_metrics

    def validate_epoch(
        self,
        val_loader: PreferenceDataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.model.steering_module.eval()

        total_loss = 0.0
        total_ranking_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Validation {epoch}",
                leave=False
            )

            for batch_idx, batch in enumerate(pbar):
                try:
                    # Process batch
                    loss_dict = self._process_batch(batch, is_training=False)

                    # Update metrics
                    total_loss += loss_dict["total_loss"].item()
                    total_ranking_loss += loss_dict["ranking_loss"].item()

                    # Calculate per-sample accuracy (preferred > dispreferred)
                    pref_scores = loss_dict["preferred_scores"]
                    dispref_scores = loss_dict["dispreferred_scores"]
                    accuracy = (pref_scores > dispref_scores).float().mean().item()
                    total_accuracy += accuracy

                    num_batches += 1

                    # Update progress bar
                    pbar.set_postfix({
                        "val_loss": f"{loss_dict['total_loss'].item():.4f}",
                        "acc": f"{accuracy:.3f}"
                    })

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}", exc_info=True)
                    if num_batches == 0 and batch_idx >= 2:
                        raise RuntimeError(
                            f"First {batch_idx + 1} validation batches all failed. "
                            f"Last error: {e}"
                        ) from e
                    continue

        if num_batches == 0:
            logger.error(
                "All validation batches failed! Metrics will be zero. "
                "Check error messages above for details."
            )

        # Calculate averages
        avg_metrics = {
            "total_loss": total_loss / max(num_batches, 1),
            "ranking_loss": total_ranking_loss / max(num_batches, 1),
            "accuracy": total_accuracy / max(num_batches, 1)
        }

        return avg_metrics

    def _process_batch(
        self,
        batch: Dict[str, Any],
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single batch.

        Args:
            batch: Batch data
            is_training: Whether in training mode

        Returns:
            Dictionary of loss components
        """
        # Extract prompts and labels
        prompts = batch["prompts"]

        # Handle different batch structures
        if "preferred_labels" in batch and "dispreferred_labels" in batch:
            preferred_labels = batch["preferred_labels"].to(self.device)
            dispreferred_labels = batch["dispreferred_labels"].to(self.device)
            rating_diffs = batch.get("rating_diffs", None)
            if rating_diffs is not None:
                rating_diffs = rating_diffs.to(self.device)
        else:
            # Create synthetic preference pairs from caption data
            labels = batch["labels"].to(self.device)
            batch_size = len(prompts)
            preferred_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
            dispreferred_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            rating_diffs = None

        # Encode prompts
        positive_embeddings, _ = self.model.encode_prompts(prompts)

        # Use the full batch - each prompt is used for BOTH preferred and
        # dispreferred steering (same prompt, different preference signals)
        batch_size = positive_embeddings.size(0)

        # Create timesteps (simulate diffusion timesteps)
        timesteps = torch.randint(
            0, 1000, (batch_size,), device=self.device
        )

        # Apply steering for preferred examples (same prompts, preference=1)
        preferred_steered = self.model.apply_preference_steering(
            positive_embeddings,
            timesteps,
            preferred_labels[:batch_size]
        )

        # Apply steering for dispreferred examples (same prompts, preference=0)
        dispreferred_steered = self.model.apply_preference_steering(
            positive_embeddings,
            timesteps,
            dispreferred_labels[:batch_size]
        )

        # Compute loss
        loss_dict = self.criterion(
            preferred_steered,
            dispreferred_steered,
            positive_embeddings,  # Original embeddings (same prompts for both)
            self.model.steering_module,
            rating_diffs
        )

        # Backward pass if training
        if is_training:
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()

            # Gradient clipping
            max_grad_norm = self.config.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(
                self.model.steering_module.parameters(),
                max_grad_norm
            )

            self.optimizer.step()

        return loss_dict

    def _log_step_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        phase: str
    ) -> None:
        """Log step-level metrics."""
        if self.use_mlflow:
            try:
                for key, value in metrics.items():
                    if torch.is_tensor(value):
                        # For multi-element tensors (e.g. per-sample scores), log the mean
                        if value.numel() > 1:
                            value = value.mean().item()
                        else:
                            value = value.item()
                    mlflow.log_metric(f"{phase}_{key}_step", value, step=self.current_step)
            except Exception as e:
                logger.warning(f"MLflow step logging failed: {e}")

    def _log_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """Log epoch-level metrics."""
        # Console logging
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total_loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.3f}, "
            f"LR: {train_metrics['learning_rate']:.6f}"
        )

        # MLflow logging
        if self.use_mlflow:
            try:
                # Log training metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value, step=epoch)

                # Log validation metrics
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val_{key}", value, step=epoch)

            except Exception as e:
                logger.warning(f"MLflow epoch logging failed: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.steering_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

            # Log model to MLflow
            if self.use_mlflow:
                try:
                    mlflow.pytorch.log_model(
                        self.model.steering_module,
                        "steering_module",
                        registered_model_name="preference_steering_module"
                    )
                except Exception as e:
                    logger.warning(f"MLflow model logging failed: {e}")

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.steering_module.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Update training state
        self.current_epoch = checkpoint["epoch"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def train(
        self,
        train_loader: PreferenceDataLoader,
        val_loader: Optional[PreferenceDataLoader] = None,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs

        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }

        # Early stopping configuration
        early_stopping_patience = self.config.get("early_stopping_patience", 20)
        min_delta = self.config.get("min_delta", 0.001)

        try:
            for epoch in range(self.current_epoch, num_epochs):
                epoch_start_time = time.time()

                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)

                # Validation phase
                if val_loader:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                else:
                    val_metrics = {"total_loss": 0.0, "accuracy": 0.0}

                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["total_loss"])
                    else:
                        self.scheduler.step()

                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics, epoch)

                # Update history
                history["train_loss"].append(train_metrics["total_loss"])
                history["val_loss"].append(val_metrics["total_loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["learning_rate"].append(train_metrics["learning_rate"])

                # Check for best model
                current_val_loss = val_metrics["total_loss"]
                is_best = current_val_loss < self.best_validation_loss

                if is_best:
                    self.best_validation_loss = current_val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)

                # Early stopping check
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Update current epoch
                self.current_epoch = epoch + 1

                # Log epoch time
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            # End MLflow run
            if self.use_mlflow:
                try:
                    mlflow.end_run()
                except Exception as e:
                    logger.warning(f"MLflow run end failed: {e}")

        logger.info("Training completed successfully")
        return history