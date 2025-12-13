"""
Data Preprocessing Module for DeepGuard MLOps Pipeline.

This module handles image preprocessing, normalization, and 
preparing data for model training.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

# Get logger
logger = logging.getLogger(__name__)


class DataPreprocessing:
    """
    Handles data preprocessing for image classification.
    
    Responsible for:
    - Loading images from disk
    - Resizing to target dimensions
    - Normalizing pixel values
    - Creating train/validation/test splits
    - Saving processed data as numpy arrays
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize DataPreprocessing with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        
        # Get configuration values with defaults
        data_config = self.config.get("data", {})
        preprocess_config = self.config.get("preprocessing", {})
        
        self.raw_data_dir = Path(data_config.get("raw_dir", "data/raw"))
        self.processed_data_dir = Path(data_config.get("processed_dir", "data/processed"))
        
        self.image_size = preprocess_config.get("image_size", 128)
        self.sample_size = preprocess_config.get("sample_size", None)  # None = use all data
        self.validation_split = preprocess_config.get("validation_split", 0.2)
        self.random_state = preprocess_config.get("random_state", 42)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load a single image and preprocess it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array or None if loading fails
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize
                img = img.resize((self.image_size, self.image_size))
                
                # Convert to numpy array and normalize
                img_array = np.array(img) / 255.0
                
                return img_array.astype(np.float32)
                
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
            
    def load_images_from_directory(self, directory: Path, label: int) -> Tuple[list, list]:
        """
        Load all images from a directory with a given label.
        
        Args:
            directory: Path to the directory containing images
            label: Label to assign to all images (0=REAL, 1=FAKE)
            
        Returns:
            Tuple of (images list, labels list)
        """
        images = []
        labels = []
        
        # Get all image files
        image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
        
        # Sample if specified
        if self.sample_size and len(image_files) > self.sample_size // 2:
            np.random.seed(self.random_state)
            image_files = list(np.random.choice(image_files, size=self.sample_size // 2, replace=False))
            
        logger.info(f"Loading {len(image_files)} images from {directory.name}")
        
        for img_path in image_files:
            img_array = self.load_and_preprocess_image(img_path)
            if img_array is not None:
                images.append(img_array)
                labels.append(label)
                
        return images, labels
        
    def load_dataset(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a complete dataset split.
        
        Args:
            split: Dataset split to load ("train" or "test")
            
        Returns:
            Tuple of (images array, labels array)
        """
        split_dir = self.raw_data_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        all_images = []
        all_labels = []
        
        # Load REAL images (label = 0)
        real_dir = split_dir / "REAL"
        if real_dir.exists():
            images, labels = self.load_images_from_directory(real_dir, label=0)
            all_images.extend(images)
            all_labels.extend(labels)
            
        # Load FAKE images (label = 1)
        fake_dir = split_dir / "FAKE"
        if fake_dir.exists():
            images, labels = self.load_images_from_directory(fake_dir, label=1)
            all_images.extend(images)
            all_labels.extend(labels)
            
        return np.array(all_images), np.array(all_labels)
        
    def create_validation_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a validation split from training data.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        return train_test_split(
            X_train, y_train,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y_train
        )
        
    def save_processed_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """
        Save processed data as numpy arrays.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
        """
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        np.save(self.processed_data_dir / "X_train.npy", X_train)
        np.save(self.processed_data_dir / "y_train.npy", y_train)
        np.save(self.processed_data_dir / "X_val.npy", X_val)
        np.save(self.processed_data_dir / "y_val.npy", y_val)
        np.save(self.processed_data_dir / "X_test.npy", X_test)
        np.save(self.processed_data_dir / "y_test.npy", y_test)
        
        logger.info(f"Saved processed data to {self.processed_data_dir}")
        
    def run(self) -> Path:
        """
        Execute the full data preprocessing pipeline.
        
        Returns:
            Path to the processed data directory
        """
        logger.info("=" * 50)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Load training data
        logger.info("Loading training data...")
        X_train_full, y_train_full = self.load_dataset("train")
        logger.info(f"Loaded training data: {X_train_full.shape}")
        
        # Step 2: Load test data
        logger.info("Loading test data...")
        X_test, y_test = self.load_dataset("test")
        logger.info(f"Loaded test data: {X_test.shape}")
        
        # Step 3: Create validation split
        logger.info("Creating validation split...")
        X_train, X_val, y_train, y_val = self.create_validation_split(
            X_train_full, y_train_full
        )
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        # Step 4: Save processed data
        self.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test)
        
        logger.info("Data Preprocessing Pipeline completed successfully!")
        
        return self.processed_data_dir


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the data preprocessing pipeline
    preprocessing = DataPreprocessing()
    preprocessing.run()
