"""
Data Module for DeepGuard MLOps Pipeline.

This module contains data-related functionality including:
- Data Ingestion: Downloading and organizing datasets
- Data Preprocessing: Loading, resizing, and normalizing images
"""

from .data_ingestion import DataIngestion
from .data_preprocessing import DataPreprocessing

__all__ = [
    "DataIngestion",
    "DataPreprocessing",
]
