"""
Feature Engineering Module for DeepGuard MLOps Pipeline.

This module handles data augmentation and feature extraction
for improving model training.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get logger
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Handles feature engineering and data augmentation.
    
    Responsible for:
    - Data augmentation for training
    - Creating data generators for efficient batch processing
    - Feature extraction (if using transfer learning)
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize FeatureEngineering with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        
        # Get configuration values
        data_config = self.config.get("data", {})
        aug_config = self.config.get("augmentation", {})
        
        self.processed_data_dir = Path(data_config.get("processed_dir", "data/processed"))
        self.features_dir = Path(data_config.get("features_dir", "data/features"))
        
        # Augmentation parameters
        self.rotation_range = aug_config.get("rotation_range", 20)
        self.width_shift_range = aug_config.get("width_shift_range", 0.2)
        self.height_shift_range = aug_config.get("height_shift_range", 0.2)
        self.horizontal_flip = aug_config.get("horizontal_flip", True)
        self.vertical_flip = aug_config.get("vertical_flip", False)
        self.zoom_range = aug_config.get("zoom_range", 0.2)
        self.fill_mode = aug_config.get("fill_mode", "nearest")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from numpy files.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info(f"Loading processed data from {self.processed_data_dir}")
        
        X_train = np.load(self.processed_data_dir / "X_train.npy")
        y_train = np.load(self.processed_data_dir / "y_train.npy")
        X_val = np.load(self.processed_data_dir / "X_val.npy")
        y_val = np.load(self.processed_data_dir / "y_val.npy")
        X_test = np.load(self.processed_data_dir / "X_test.npy")
        y_test = np.load(self.processed_data_dir / "y_test.npy")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    def create_train_generator(self) -> ImageDataGenerator:
        """
        Create an ImageDataGenerator for training with augmentation.
        
        Returns:
            Configured ImageDataGenerator for training
        """
        logger.info("Creating training data generator with augmentation")
        
        train_datagen = ImageDataGenerator(
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            zoom_range=self.zoom_range,
            fill_mode=self.fill_mode
        )
        
        return train_datagen
        
    def create_validation_generator(self) -> ImageDataGenerator:
        """
        Create an ImageDataGenerator for validation (no augmentation).
        
        Returns:
            Configured ImageDataGenerator for validation
        """
        logger.info("Creating validation data generator (no augmentation)")
        
        # No augmentation for validation/test
        val_datagen = ImageDataGenerator()
        
        return val_datagen
        
    def create_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for efficient training.
        
        Args:
            X: Image data
            y: Labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow Dataset object
        """
        logger.info(f"Creating TF dataset with batch_size={batch_size}, shuffle={shuffle}, augment={augment}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
            
        # Apply augmentation if requested
        if augment:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to a single image.
        
        Args:
            image: Input image tensor
            label: Image label
            
        Returns:
            Augmented image and label
        """
        # Random flip
        if self.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        if self.vertical_flip:
            image = tf.image.random_flip_up_down(image)
            
        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Clip values to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
        
    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        logger.info("Calculating class weights...")
        
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        
        class_weights = {}
        for cls, count in zip(unique, counts):
            # Inverse frequency weighting
            weight = total / (len(unique) * count)
            class_weights[int(cls)] = weight
            logger.info(f"Class {cls}: count={count}, weight={weight:.4f}")
            
        return class_weights
        
    def get_data_statistics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "image_shape": X_train.shape[1:],
            "train_class_distribution": {
                int(k): int(v) for k, v in 
                zip(*np.unique(y_train, return_counts=True))
            },
            "val_class_distribution": {
                int(k): int(v) for k, v in 
                zip(*np.unique(y_val, return_counts=True))
            },
            "test_class_distribution": {
                int(k): int(v) for k, v in 
                zip(*np.unique(y_test, return_counts=True))
            },
        }
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Training samples: {stats['train_samples']}")
        logger.info(f"  Validation samples: {stats['val_samples']}")
        logger.info(f"  Test samples: {stats['test_samples']}")
        logger.info(f"  Image shape: {stats['image_shape']}")
        
        return stats
        
    def save_features(
        self,
        class_weights: Dict[int, float],
        statistics: Dict[str, Any]
    ) -> Path:
        """
        Save feature engineering outputs for DVC tracking.
        
        Args:
            class_weights: Calculated class weights
            statistics: Dataset statistics
            
        Returns:
            Path to features directory
        """
        import json
        
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save class weights
        with open(self.features_dir / "class_weights.json", 'w') as f:
            json.dump(class_weights, f, indent=2)
            
        # Save statistics
        with open(self.features_dir / "data_statistics.json", 'w') as f:
            # Convert numpy types to Python types
            clean_stats = {}
            for k, v in statistics.items():
                if isinstance(v, tuple):
                    clean_stats[k] = list(v)
                elif isinstance(v, dict):
                    clean_stats[k] = {str(kk): int(vv) for kk, vv in v.items()}
                else:
                    clean_stats[k] = v
            json.dump(clean_stats, f, indent=2)
            
        # Save augmentation config
        aug_config = {
            "rotation_range": self.rotation_range,
            "width_shift_range": self.width_shift_range,
            "height_shift_range": self.height_shift_range,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "zoom_range": self.zoom_range,
            "fill_mode": self.fill_mode
        }
        with open(self.features_dir / "augmentation_config.json", 'w') as f:
            json.dump(aug_config, f, indent=2)
            
        logger.info(f"Feature outputs saved to {self.features_dir}")
        
        return self.features_dir
        
    def run(self) -> Dict[str, Any]:
        """
        Execute the full feature engineering pipeline.
        
        Returns:
            Dictionary containing data generators and statistics
        """
        logger.info("=" * 50)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Load processed data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_processed_data()
        
        # Step 2: Get statistics
        stats = self.get_data_statistics(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Step 3: Calculate class weights
        class_weights = self.get_class_weights(y_train)
        
        # Step 4: Create data generators
        train_datagen = self.create_train_generator()
        val_datagen = self.create_validation_generator()
        
        # Step 5: Save outputs for DVC tracking
        self.save_features(class_weights, stats)
        
        logger.info("Feature Engineering Pipeline completed successfully!")
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_datagen": train_datagen,
            "val_datagen": val_datagen,
            "class_weights": class_weights,
            "statistics": stats
        }


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the feature engineering pipeline
    fe = FeatureEngineering()
    result = fe.run()
    print(f"\nData ready for training!")
    print(f"Training samples: {len(result['X_train'])}")
    print(f"Validation samples: {len(result['X_val'])}")
    print(f"Test samples: {len(result['X_test'])}")
