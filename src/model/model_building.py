"""
Model Building Module for DeepGuard MLOps Pipeline.

This module contains CNN model architectures for detecting
AI-generated vs real images.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Get logger
logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Builds and trains CNN models for image classification.
    
    Supports multiple architectures:
    - SimpleCNN: Basic 3-layer CNN for baseline
    - DeeperCNN: Deeper architecture with batch normalization
    - EfficientStyleCNN: Efficient architecture with separable convolutions
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize ModelBuilder with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        
        # Get configuration values
        model_config = self.config.get("model", {})
        training_config = self.config.get("training", {})
        
        self.input_shape = tuple(model_config.get("input_shape", [128, 128, 3]))
        self.architecture = model_config.get("architecture", "DeeperCNN")
        
        self.epochs = training_config.get("epochs", 20)
        self.batch_size = training_config.get("batch_size", 32)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 5)
        
        self.models_dir = Path(self.config.get("outputs", {}).get("models_dir", "models"))
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def build_simple_cnn(self, name: str = "SimpleCNN") -> keras.Model:
        """
        Build a simple CNN architecture for baseline.
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building {name} architecture...")
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        return model
        
    def build_deeper_cnn(self, name: str = "DeeperCNN") -> keras.Model:
        """
        Build a deeper CNN with batch normalization and regularization.
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building {name} architecture...")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        return model
        
    def build_efficient_style_cnn(self, name: str = "EfficientStyleCNN") -> keras.Model:
        """
        Build an efficient CNN using separable convolutions.
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building {name} architecture...")
        
        model = models.Sequential([
            # Stem
            layers.Conv2D(32, (3, 3), strides=2, padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            
            # MBConv-style blocks
            layers.SeparableConv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D((2, 2)),
            
            layers.SeparableConv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D((2, 2)),
            
            layers.SeparableConv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            
            # Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='swish'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        return model
        
    def build_model(self, architecture: Optional[str] = None) -> keras.Model:
        """
        Build a model based on the specified architecture.
        
        Args:
            architecture: Architecture name (uses config default if None)
            
        Returns:
            Compiled Keras model
        """
        arch = architecture or self.architecture
        
        if arch == "SimpleCNN":
            model = self.build_simple_cnn()
        elif arch == "DeeperCNN":
            model = self.build_deeper_cnn()
        elif arch == "EfficientStyleCNN":
            model = self.build_efficient_style_cnn()
        else:
            logger.warning(f"Unknown architecture {arch}, using DeeperCNN")
            model = self.build_deeper_cnn()
            
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info(f"Model compiled with {model.count_params():,} parameters")
        
        return model
        
    def get_callbacks(self, model_name: str) -> list:
        """
        Get training callbacks.
        
        Args:
            model_name: Name for the model checkpoint file
            
        Returns:
            List of Keras callbacks
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.models_dir / f"{model_name}_best.keras"),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
        
    def train(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict[int, float]] = None
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Compiled Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history
        """
        logger.info(f"Training {model.name} for {self.epochs} epochs...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        callbacks = self.get_callbacks(model.name)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info(f"Training completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        
        return history
        
    def save_model(self, model: keras.Model, filename: Optional[str] = None) -> Path:
        """
        Save the trained model.
        
        Args:
            model: Trained Keras model
            filename: Optional filename (uses model.name if not provided)
            
        Returns:
            Path to saved model
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        name = filename or f"{model.name}_final.keras"
        save_path = self.models_dir / name
        
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path
        
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History, Path]:
        """
        Execute the full model building and training pipeline.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            class_weights: Optional class weights
            
        Returns:
            Tuple of (trained model, training history, model save path)
        """
        logger.info("=" * 50)
        logger.info("Starting Model Building Pipeline")
        logger.info("=" * 50)
        
        # Build model
        model = self.build_model()
        model.summary(print_fn=logger.info)
        
        # Train model
        history = self.train(model, X_train, y_train, X_val, y_val, class_weights)
        
        # Save model
        save_path = self.save_model(model)
        
        logger.info("Model Building Pipeline completed successfully!")
        
        return model, history, save_path


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load data and run
    from src.data import FeatureEngineering
    
    fe = FeatureEngineering()
    data = fe.run()
    
    builder = ModelBuilder()
    model, history, save_path = builder.run(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['class_weights']
    )
