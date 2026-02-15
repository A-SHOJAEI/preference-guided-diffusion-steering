"""
Preference-Guided Diffusion Steering

A lightweight steering module that guides text-to-image diffusion models toward
human-preferred aesthetic and semantic qualities using UltraFeedback preference data.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

# Conditional imports to handle missing dependencies gracefully
__all__ = []

try:
    from .models.model import PreferenceGuidedDiffusionModel, SteeringModule
    __all__.extend(["PreferenceGuidedDiffusionModel", "SteeringModule"])
except ImportError:
    pass

try:
    from .training.trainer import PreferenceTrainer
    __all__.extend(["PreferenceTrainer"])
except ImportError:
    pass

try:
    from .evaluation.metrics import PreferenceMetrics
    __all__.extend(["PreferenceMetrics"])
except ImportError:
    pass