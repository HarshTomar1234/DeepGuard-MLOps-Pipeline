"""
Features Module for DeepGuard MLOps Pipeline.

This module contains feature engineering functionality including:
- Data augmentation
- TensorFlow dataset creation
- Class weight calculation
"""

from .feature_engineering import FeatureEngineering

__all__ = [
    "FeatureEngineering",
]
