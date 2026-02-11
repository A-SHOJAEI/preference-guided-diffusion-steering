"""
Tests for data loading and preprocessing modules.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any

from preference_guided_diffusion_steering.data.loader import (
    UltraFeedbackDataset, ConceptualCaptionsDataset, PreferenceDataLoader
)
from preference_guided_diffusion_steering.data.preprocessing import (
    PreferenceDataProcessor, TextImagePairProcessor
)

from .conftest import (
    assert_tensor_shape, create_synthetic_preference_data, create_mock_batch
)


class TestUltraFeedbackDataset:
    """Test UltraFeedback dataset functionality."""

    def test_dataset_initialization(self, test_config):
        """Test dataset initialization."""
        dataset = UltraFeedbackDataset(
            max_samples=test_config["data"]["max_samples"],
            min_rating_diff=test_config["data"]["min_rating_diff"],
            seed=test_config["seed"]
        )

        assert len(dataset) > 0
        assert hasattr(dataset, 'preference_pairs')
        assert isinstance(dataset.preference_pairs, list)

    def test_dataset_getitem(self, preference_dataset):
        """Test dataset item retrieval."""
        if len(preference_dataset) == 0:
            pytest.skip("No data available in dataset")

        item = preference_dataset[0]

        required_keys = ["prompt", "preferred_label", "dispreferred_label"]
        for key in required_keys:
            assert key in item, f"Missing key: {key}"

        assert isinstance(item["prompt"], str)
        assert len(item["prompt"]) > 0
        assert item["preferred_label"] in [0, 1]
        assert item["dispreferred_label"] in [0, 1]
        assert item["preferred_label"] != item["dispreferred_label"]

    def test_get_prompts(self, preference_dataset):
        """Test getting unique prompts."""
        prompts = preference_dataset.get_prompts()

        assert isinstance(prompts, list)
        assert all(isinstance(p, str) for p in prompts)
        assert len(set(prompts)) == len(prompts)  # All unique

    def test_preference_pair_creation(self):
        """Test preference pair creation logic."""
        # Create dataset with synthetic data
        dataset = UltraFeedbackDataset(max_samples=10, seed=42)

        # Check that preference pairs were created
        assert len(dataset.preference_pairs) > 0

        # Verify structure of preference pairs
        for pair in dataset.preference_pairs:
            assert "prompt" in pair
            assert "preferred_label" in pair
            assert "dispreferred_label" in pair
            assert pair["preferred_label"] != pair["dispreferred_label"]


class TestConceptualCaptionsDataset:
    """Test Conceptual Captions dataset functionality."""

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = ConceptualCaptionsDataset(
            max_samples=10,
            image_size=(256, 256)
        )

        assert len(dataset) > 0
        assert hasattr(dataset, 'raw_dataset')

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = ConceptualCaptionsDataset(max_samples=5)

        if len(dataset) == 0:
            pytest.skip("No data available in dataset")

        item = dataset[0]

        assert "caption" in item
        assert "image_url" in item
        assert isinstance(item["caption"], str)
        assert len(item["caption"]) > 0

    def test_get_captions(self):
        """Test getting all captions."""
        dataset = ConceptualCaptionsDataset(max_samples=5)
        captions = dataset.get_captions()

        assert isinstance(captions, list)
        assert len(captions) == len(dataset)
        assert all(isinstance(c, str) for c in captions)


class TestPreferenceDataLoader:
    """Test preference data loader functionality."""

    def test_data_loader_initialization(self, preference_dataset, test_config):
        """Test data loader initialization."""
        data_loader = PreferenceDataLoader(
            preference_dataset=preference_dataset,
            batch_size=test_config["data"]["batch_size"],
            shuffle=False,
            num_workers=0
        )

        assert data_loader.preference_dataset == preference_dataset
        assert data_loader.batch_size == test_config["data"]["batch_size"]
        assert hasattr(data_loader, 'preference_loader')

    def test_collate_preference_batch(self, preference_dataset, test_config):
        """Test preference batch collation."""
        data_loader = PreferenceDataLoader(
            preference_dataset=preference_dataset,
            batch_size=test_config["data"]["batch_size"],
            num_workers=0
        )

        # Create sample batch data
        batch_data = [preference_dataset[i] for i in range(min(2, len(preference_dataset)))]

        if not batch_data:
            pytest.skip("No batch data available")

        collated = data_loader._collate_preference_batch(batch_data)

        assert "prompts" in collated
        assert "preferred_labels" in collated
        assert "dispreferred_labels" in collated
        assert "data_type" in collated

        assert len(collated["prompts"]) == len(batch_data)
        assert collated["preferred_labels"].shape[0] == len(batch_data)
        assert collated["data_type"] == "preference"

    def test_data_loader_iteration(self, preference_dataset, test_config):
        """Test data loader iteration."""
        data_loader = PreferenceDataLoader(
            preference_dataset=preference_dataset,
            batch_size=2,
            num_workers=0
        )

        # Test iteration
        batches_seen = 0
        for batch in data_loader:
            assert "prompts" in batch
            assert isinstance(batch["prompts"], list)

            batches_seen += 1
            if batches_seen >= 2:  # Limit test iterations
                break

        assert batches_seen > 0


class TestPreferenceDataProcessor:
    """Test preference data processor functionality."""

    def test_processor_initialization(self, test_config):
        """Test processor initialization."""
        config = test_config["data"]["data_processor"]
        processor = PreferenceDataProcessor(**config)

        assert processor.max_text_length == config["max_text_length"]
        assert processor.text_augmentation == config["text_augmentation"]
        assert hasattr(processor, 'tokenizer')

    def test_text_normalization(self, preference_data_processor):
        """Test text normalization."""
        test_cases = [
            "hello world",
            "  extra   spaces  ",
            "no punctuation",
            "UPPERCASE TEXT",
            "text with special chars @#$%"
        ]

        for text in test_cases:
            normalized = preference_data_processor.normalize_text(text)

            assert isinstance(normalized, str)
            assert normalized.strip() == normalized  # No leading/trailing spaces
            assert normalized.endswith(('.', '!', '?'))  # Proper punctuation
            if normalized:
                assert normalized[0].isupper()  # Proper capitalization

    def test_text_augmentation(self, preference_data_processor):
        """Test text augmentation."""
        original_text = "A beautiful landscape"
        augmented_texts = preference_data_processor.augment_text(original_text, num_augmentations=2)

        assert isinstance(augmented_texts, list)
        assert len(augmented_texts) >= 1  # At least includes original
        assert original_text in augmented_texts  # Original should be included

        # If augmentation is enabled, should have more variants
        if preference_data_processor.text_augmentation:
            assert len(augmented_texts) > 1

    def test_tokenization(self, preference_data_processor):
        """Test text tokenization."""
        texts = ["A cat", "A dog running", "Beautiful sunset over ocean"]
        tokenized = preference_data_processor.tokenize_texts(texts)

        assert isinstance(tokenized, torch.Tensor)
        assert tokenized.shape[0] == len(texts)
        assert tokenized.shape[1] == preference_data_processor.max_text_length
        assert tokenized.dtype == torch.long

    def test_preference_batch_processing(self, preference_data_processor, sample_preference_pairs):
        """Test processing of preference batches."""
        processed = preference_data_processor.process_preference_batch(sample_preference_pairs)

        required_keys = ["tokenized_prompts", "preferred_labels", "dispreferred_labels"]
        for key in required_keys:
            assert key in processed, f"Missing key: {key}"

        assert isinstance(processed["tokenized_prompts"], torch.Tensor)
        assert isinstance(processed["preferred_labels"], torch.Tensor)
        assert isinstance(processed["dispreferred_labels"], torch.Tensor)
        assert processed["batch_size"] > 0

    def test_synthetic_preference_creation(self, preference_data_processor):
        """Test synthetic preference pair creation."""
        captions = ["A cat", "A dog", "A bird"]
        synthetic_pairs = preference_data_processor.create_synthetic_preferences(
            captions, num_pairs_per_caption=1
        )

        assert isinstance(synthetic_pairs, list)
        assert len(synthetic_pairs) > 0

        for pair in synthetic_pairs:
            assert "prompt" in pair
            assert "preferred_label" in pair
            assert "dispreferred_label" in pair
            assert "synthetic" in pair
            assert pair["synthetic"] is True

    def test_preference_pair_validation(self, preference_data_processor):
        """Test preference pair validation."""
        # Valid pair
        valid_pair = {
            "prompt": "A beautiful sunset",
            "preferred_label": 1,
            "dispreferred_label": 0,
            "rating_diff": 2.0
        }
        assert preference_data_processor.validate_preference_pair(valid_pair)

        # Invalid pairs
        invalid_pairs = [
            {},  # Empty
            {"prompt": "test"},  # Missing labels
            {"prompt": "", "preferred_label": 1, "dispreferred_label": 0},  # Empty prompt
            {"prompt": "test", "preferred_label": 1, "dispreferred_label": 1},  # Same labels
            {"prompt": "test", "preferred_label": "invalid", "dispreferred_label": 0},  # Invalid label type
        ]

        for invalid_pair in invalid_pairs:
            assert not preference_data_processor.validate_preference_pair(invalid_pair)


class TestTextImagePairProcessor:
    """Test text-image pair processor functionality."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = TextImagePairProcessor(
            image_size=(512, 512),
            augmentation_prob=0.3,
            quality_degradation_prob=0.5
        )

        assert processor.image_size == (512, 512)
        assert processor.augmentation_prob == 0.3
        assert processor.quality_degradation_prob == 0.5
        assert hasattr(processor, 'base_transform')
        assert hasattr(processor, 'degradation_transforms')

    def test_image_processing(self, sample_images):
        """Test image processing functionality."""
        processor = TextImagePairProcessor(image_size=(256, 256))

        if not sample_images:
            pytest.skip("No sample images available")

        # Test basic processing
        processed = processor.process_image(
            sample_images[0],
            apply_degradation=False,
            apply_augmentation=False
        )

        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (3, 256, 256)  # CHW format
        assert torch.isfinite(processed).all()

    def test_quality_degradation(self, sample_images):
        """Test quality degradation functions."""
        if not sample_images:
            pytest.skip("No sample images available")

        processor = TextImagePairProcessor()
        original_image = sample_images[0]

        # Test each degradation function
        degradation_functions = [
            processor._add_gaussian_noise,
            processor._add_blur,
            processor._adjust_brightness,
            processor._reduce_saturation
        ]

        for degradation_func in degradation_functions:
            try:
                degraded = degradation_func(original_image)
                assert degraded.size == original_image.size
            except Exception as e:
                pytest.fail(f"Degradation function {degradation_func.__name__} failed: {e}")

    def test_preference_pair_creation(self, sample_images):
        """Test preference pair creation from image-caption pair."""
        if not sample_images:
            pytest.skip("No sample images available")

        processor = TextImagePairProcessor()
        caption = "A test image"

        pair = processor.create_preference_pair(sample_images[0], caption)

        assert "caption" in pair
        assert "preferred_image" in pair
        assert "dispreferred_image" in pair
        assert "preferred_label" in pair
        assert "dispreferred_label" in pair

        assert pair["caption"] == caption
        assert isinstance(pair["preferred_image"], torch.Tensor)
        assert isinstance(pair["dispreferred_image"], torch.Tensor)
        assert pair["preferred_label"] == 1
        assert pair["dispreferred_label"] == 0


class TestDataIntegration:
    """Integration tests for data components."""

    def test_end_to_end_data_pipeline(self, test_config):
        """Test complete data pipeline."""
        # Create dataset
        dataset = UltraFeedbackDataset(
            max_samples=5,
            seed=test_config["seed"]
        )

        # Create data loader
        data_loader = PreferenceDataLoader(
            preference_dataset=dataset,
            batch_size=2,
            num_workers=0
        )

        # Create processor
        processor_config = test_config["data"]["data_processor"]
        processor = PreferenceDataProcessor(**processor_config)

        # Test pipeline
        for batch in data_loader:
            # Process batch
            processed = processor.process_preference_batch([
                {"prompt": prompt, "preferred_label": 1, "dispreferred_label": 0, "rating_diff": 1.0}
                for prompt in batch["prompts"]
            ])

            # Verify processed batch
            assert "tokenized_prompts" in processed
            assert "preferred_labels" in processed
            assert processed["batch_size"] > 0

            break  # Only test one batch

    def test_data_consistency(self, preference_dataset, test_config):
        """Test data consistency across multiple accesses."""
        if len(preference_dataset) == 0:
            pytest.skip("No data available in dataset")

        # Access same item multiple times
        item1 = preference_dataset[0]
        item2 = preference_dataset[0]

        # Should be identical
        assert item1["prompt"] == item2["prompt"]
        assert item1["preferred_label"] == item2["preferred_label"]
        assert item1["dispreferred_label"] == item2["dispreferred_label"]

    def test_batch_size_handling(self, preference_dataset):
        """Test different batch sizes."""
        if len(preference_dataset) == 0:
            pytest.skip("No data available in dataset")

        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            data_loader = PreferenceDataLoader(
                preference_dataset=preference_dataset,
                batch_size=batch_size,
                num_workers=0
            )

            for batch in data_loader:
                assert len(batch["prompts"]) <= batch_size
                break  # Only test first batch