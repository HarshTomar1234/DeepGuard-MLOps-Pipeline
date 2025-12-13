"""
Data Ingestion Module for DeepGuard MLOps Pipeline.

This module handles downloading, extracting, and organizing the 
AI-generated vs real images dataset from various sources.
"""

import os
import zipfile
import logging
import shutil
from pathlib import Path

import yaml
import kagglehub

# Get logger
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data ingestion from Kaggle and other sources.
    
    Responsible for:
    - Downloading datasets from Kaggle
    - Extracting and organizing data
    - Validating downloaded data
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        self.raw_data_dir = Path(self.config.get("data", {}).get("raw_dir", "data/raw"))
        self.dataset_name = self.config.get("data", {}).get("dataset_name", "birdy654/cifake-real-and-ai-generated-synthetic-images")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def download_dataset(self) -> Path:
        """
        Download the dataset from Kaggle using kagglehub.
        
        Returns:
            Path to the downloaded dataset directory
        """
        logger.info(f"Starting download of dataset: {self.dataset_name}")
        
        try:
            # Download dataset using kagglehub
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to: {dataset_path}")
            return Path(dataset_path)
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
            
    def organize_data(self, source_path: Path) -> Path:
        """
        Organize downloaded data into the project's raw data directory.
        
        Args:
            source_path: Path where the dataset was downloaded
            
        Returns:
            Path to organized raw data directory
        """
        logger.info(f"Organizing data from {source_path} to {self.raw_data_dir}")
        
        # Create raw data directory
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the data structure
        for item in source_path.iterdir():
            dest = self.raw_data_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                logger.info(f"Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest)
                logger.info(f"Copied file: {item.name}")
                
        return self.raw_data_dir
        
    def validate_data(self) -> dict:
        """
        Validate that the data was downloaded and organized correctly.
        
        Returns:
            Dictionary with validation results including file counts
        """
        logger.info("Validating downloaded data...")
        
        validation_results = {
            "is_valid": True,
            "directories": {},
            "total_images": 0,
            "errors": []
        }
        
        # Check for expected directories
        expected_dirs = ["train", "test"]
        expected_classes = ["REAL", "FAKE"]
        
        for dir_name in expected_dirs:
            dir_path = self.raw_data_dir / dir_name
            if not dir_path.exists():
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Missing directory: {dir_name}")
                continue
                
            dir_info = {"classes": {}}
            for class_name in expected_classes:
                class_path = dir_path / class_name
                if class_path.exists():
                    image_count = len(list(class_path.glob("*.png"))) + len(list(class_path.glob("*.jpg")))
                    dir_info["classes"][class_name] = image_count
                    validation_results["total_images"] += image_count
                else:
                    validation_results["errors"].append(f"Missing class directory: {dir_name}/{class_name}")
                    
            validation_results["directories"][dir_name] = dir_info
            
        logger.info(f"Validation complete. Total images: {validation_results['total_images']}")
        if validation_results["errors"]:
            for error in validation_results["errors"]:
                logger.warning(error)
                
        return validation_results
        
    def run(self) -> Path:
        """
        Execute the full data ingestion pipeline.
        
        Returns:
            Path to the raw data directory
        """
        logger.info("=" * 50)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Download dataset
        source_path = self.download_dataset()
        
        # Step 2: Organize data
        self.organize_data(source_path)
        
        # Step 3: Validate data
        validation = self.validate_data()
        
        if not validation["is_valid"]:
            logger.error("Data validation failed!")
            raise ValueError("Data ingestion validation failed")
            
        logger.info("Data Ingestion Pipeline completed successfully!")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        
        return self.raw_data_dir


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the data ingestion pipeline
    ingestion = DataIngestion()
    ingestion.run()
