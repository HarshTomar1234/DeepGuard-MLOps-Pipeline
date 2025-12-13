"""
Model Module for DeepGuard MLOps Pipeline.

This module contains all model-related functionality including:
- Model Building: CNN architectures and training
- Model Evaluation: Metrics computation and visualization
- Model Registration: MLflow/DagsHub integration and inference
"""

from .model_building import ModelBuilder
from .model_evaluation import ModelEvaluator
from .register_model import ModelRegistry

__all__ = [
    "ModelBuilder",
    "ModelEvaluator",
    "ModelRegistry",
]
