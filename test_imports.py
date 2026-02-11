#!/usr/bin/env python3
"""
Simple test to verify all imports work correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test basic configuration
        from preference_guided_diffusion_steering.utils.config import Config, load_config
        print("✓ Config module imports successful")

        # Test data modules
        from preference_guided_diffusion_steering.data.loader import UltraFeedbackDataset
        from preference_guided_diffusion_steering.data.preprocessing import PreferenceDataProcessor
        print("✓ Data modules import successful")

        # Test model modules
        from preference_guided_diffusion_steering.models.model import SteeringModule
        print("✓ Model modules import successful")

        # Test evaluation modules
        from preference_guided_diffusion_steering.evaluation.metrics import PreferenceMetrics
        print("✓ Evaluation modules import successful")

        # Test training modules
        from preference_guided_diffusion_steering.training.trainer import PreferenceLoss
        print("✓ Training modules import successful")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from preference_guided_diffusion_steering.utils.config import load_config

        # Test loading default config
        config = load_config("configs/default.yaml")
        print("✓ Configuration loading successful")

        # Test key configuration sections
        assert "model" in config.to_dict()
        assert "training" in config.to_dict()
        assert "data" in config.to_dict()
        print("✓ Configuration validation successful")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring external dependencies."""
    try:
        from preference_guided_diffusion_steering.models.model import SteeringModule
        from preference_guided_diffusion_steering.data.loader import UltraFeedbackDataset
        from preference_guided_diffusion_steering.data.preprocessing import PreferenceDataProcessor

        # Test steering module creation
        steering_module = SteeringModule(
            text_embed_dim=768,
            hidden_dim=512,
            num_layers=3
        )
        print("✓ Steering module creation successful")

        # Test dataset creation (will use synthetic data)
        dataset = UltraFeedbackDataset(max_samples=5, seed=42)
        print(f"✓ Dataset creation successful ({len(dataset)} samples)")

        # Test data processor
        processor = PreferenceDataProcessor(
            max_text_length=77,
            text_augmentation=True,
            seed=42
        )
        print("✓ Data processor creation successful")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Preference-Guided Diffusion Steering Project")
    print("=" * 60)

    success_count = 0
    total_tests = 3

    # Test imports
    if test_imports():
        success_count += 1

    print()

    # Test configuration
    if test_config_loading():
        success_count += 1

    print()

    # Test basic functionality
    if test_basic_functionality():
        success_count += 1

    print()
    print("=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! The project is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)