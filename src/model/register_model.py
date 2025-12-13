"""
Model Registration Module for DeepGuard MLOps Pipeline.

This module handles model registration with MLflow/DagsHub
and provides prediction utilities for multiple image formats.
"""

import io
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import yaml
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# MLflow imports
import mlflow
import mlflow.keras
import dagshub

# Get logger
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', 
    '.tiff', '.tif', '.webp', '.ico'
}


class ModelRegistry:
    """
    Handles model registration with MLflow/DagsHub.
    
    Responsible for:
    - Registering models to MLflow model registry
    - Managing model versions and stages
    - Loading registered models for inference
    - Making predictions on various image formats
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize ModelRegistry with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        
        # Get configuration values
        mlflow_config = self.config.get("mlflow", {})
        model_config = self.config.get("model", {})
        
        self.dagshub_username = mlflow_config.get("dagshub_username", "")
        self.dagshub_repo = mlflow_config.get("dagshub_repo", "DeepGuard-MLOps-Pipeline")
        self.experiment_name = mlflow_config.get("experiment_name", "DeepGuard-Model-Training")
        self.model_name = mlflow_config.get("registered_model_name", "DeepGuard-Classifier")
        
        self.input_shape = tuple(model_config.get("input_shape", [128, 128, 3]))
        self.models_dir = Path(self.config.get("outputs", {}).get("models_dir", "models"))
        
        self._mlflow_initialized = False
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def initialize_mlflow(self) -> None:
        """Initialize MLflow tracking with DagsHub."""
        if self._mlflow_initialized:
            return
            
        if self.dagshub_username:
            logger.info(f"Initializing DagsHub MLflow for {self.dagshub_username}/{self.dagshub_repo}")
            
            try:
                dagshub.init(
                    repo_owner=self.dagshub_username,
                    repo_name=self.dagshub_repo,
                    mlflow=True
                )
                
                mlflow.set_tracking_uri(
                    f"https://dagshub.com/{self.dagshub_username}/{self.dagshub_repo}.mlflow"
                )
                
                mlflow.set_experiment(self.experiment_name)
                
                self._mlflow_initialized = True
                logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
                
            except Exception as e:
                logger.error(f"Failed to initialize DagsHub: {e}")
                raise
        else:
            logger.warning("DagsHub username not configured. Using local MLflow.")
            self._mlflow_initialized = True
            
    def log_and_register_model(
        self,
        model: keras.Model,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Log model to MLflow and register it in the model registry.
        
        Args:
            model: Trained Keras model
            metrics: Dictionary of evaluation metrics
            params: Optional dictionary of model parameters
            artifacts: Optional dictionary of artifact paths to log
            run_name: Optional name for the MLflow run
            
        Returns:
            MLflow run ID
        """
        self.initialize_mlflow()
        
        run_name = run_name or f"{model.name}_v1"
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            # Log parameters
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
                    
            # Log model parameters
            mlflow.log_param("model_name", model.name)
            mlflow.log_param("total_params", model.count_params())
            mlflow.log_param("input_shape", str(self.input_shape))
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path)
                        
            # Log the model
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name=self.model_name
            )
            
            logger.info(f"✅ Model logged to MLflow!")
            logger.info(f"   Run ID: {run_id}")
            logger.info(f"   Registered as: {self.model_name}")
            
            # Print DagsHub links
            if self.dagshub_username:
                base_url = f"https://dagshub.com/{self.dagshub_username}/{self.dagshub_repo}.mlflow"
                print(f"\n🏃 View run {run_name} at: {base_url}/#/experiments/1/runs/{run_id}")
                print(f"🧪 View experiment at: {base_url}/#/experiments/1")
                
            return run_id
            
    def load_registered_model(
        self,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ) -> keras.Model:
        """
        Load a model from the MLflow registry.
        
        Args:
            version: Specific model version (latest if None)
            stage: Model stage ('Production', 'Staging', etc.)
            
        Returns:
            Loaded Keras model
        """
        self.initialize_mlflow()
        
        if version:
            model_uri = f"models:/{self.model_name}/{version}"
        elif stage:
            model_uri = f"models:/{self.model_name}/{stage}"
        else:
            model_uri = f"models:/{self.model_name}/latest"
            
        logger.info(f"Loading model from {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        
        return model
        
    def load_local_model(self, model_path: Optional[Path] = None) -> keras.Model:
        """
        Load a model from local disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model
        """
        if model_path is None:
            model_files = list(self.models_dir.glob("*_final.keras")) + \
                          list(self.models_dir.glob("*_best.keras"))
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
        logger.info(f"Loading model from {model_path}")
        return keras.models.load_model(model_path)
        
    def preprocess_image(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Preprocess an image for model prediction.
        Supports multiple input formats.
        
        Args:
            image: Image in various formats:
                - File path (str or Path)
                - Raw bytes
                - Numpy array
                - PIL Image
                
        Returns:
            Preprocessed image array ready for prediction
        """
        target_size = (self.input_shape[0], self.input_shape[1])
        
        # Handle different input types
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {path.suffix}. Supported: {SUPPORTED_FORMATS}")
            img = Image.open(path)
            
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            
        elif isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = image / 255.0
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            return image.astype(np.float32)
            
        elif isinstance(image, Image.Image):
            img = image
            
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
        
    def predict(
        self,
        model: keras.Model,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Make a prediction on a single image.
        
        Args:
            model: Trained Keras model
            image: Image in any supported format
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        probability = float(model.predict(img_array, verbose=0)[0][0])
        
        # Classify
        is_fake = probability >= threshold
        label = "FAKE" if is_fake else "REAL"
        confidence = probability if is_fake else (1 - probability)
        
        return {
            "label": label,
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_probability": probability,
            "real_probability": 1 - probability
        }
        
    def predict_batch(
        self,
        model: keras.Model,
        images: List[Union[str, Path, bytes, np.ndarray, Image.Image]],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images.
        
        Args:
            model: Trained Keras model
            images: List of images in any supported format
            threshold: Classification threshold
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img in images:
            try:
                result = self.predict(model, img, threshold)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                results.append({"error": str(e)})
                
        return results
        
    def transition_model_stage(
        self,
        version: int,
        stage: str = "Production"
    ) -> None:
        """
        Transition a model version to a new stage.
        
        Args:
            version: Model version number
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        self.initialize_mlflow()
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Model {self.model_name} v{version} transitioned to {stage}")
        
    def run(
        self,
        model: Optional[keras.Model] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_path: Optional[Path] = None
    ) -> str:
        """
        Execute the model registration pipeline.
        
        Args:
            model: Trained model (will load if None)
            metrics: Evaluation metrics (will load if None)
            model_path: Path to model if loading
            
        Returns:
            MLflow run ID
        """
        logger.info("=" * 50)
        logger.info("Starting Model Registration Pipeline")
        logger.info("=" * 50)
        
        # Load model if not provided
        if model is None:
            model = self.load_local_model(model_path)
            
        # Load metrics if not provided
        if metrics is None:
            reports_dir = Path(self.config.get("outputs", {}).get("reports_dir", "reports"))
            metrics_file = reports_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                logger.warning("No metrics file found. Using empty metrics.")
                metrics = {}
                
        # Register model
        run_id = self.log_and_register_model(
            model=model,
            metrics=metrics,
            params={
                "architecture": model.name,
                "input_shape": str(self.input_shape),
            },
            artifacts={
                "evaluation_plot": str(reports_dir / "figures" / f"{model.name}_evaluation.png")
            } if 'reports_dir' in dir() else None
        )
        
        logger.info("Model Registration Pipeline completed successfully!")
        
        return run_id


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run registration
    registry = ModelRegistry()
    run_id = registry.run()
    
    print(f"\n✅ Model registered successfully!")
    print(f"   Run ID: {run_id}")
