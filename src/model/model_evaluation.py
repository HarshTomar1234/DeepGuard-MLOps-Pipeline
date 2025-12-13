"""
Model Evaluation Module for DeepGuard MLOps Pipeline.

This module handles model evaluation, metrics computation,
and visualization generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow import keras

# Get logger
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models and generates metrics/visualizations.
    
    Responsible for:
    - Computing classification metrics
    - Generating confusion matrix and ROC curve plots
    - Saving metrics to JSON for DVC tracking
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize ModelEvaluator with configuration.
        
        Args:
            config_path: Path to the params.yaml configuration file
        """
        self.config = self._load_config(config_path)
        
        # Get configuration values
        output_config = self.config.get("outputs", {})
        
        self.models_dir = Path(output_config.get("models_dir", "models"))
        self.reports_dir = Path(output_config.get("reports_dir", "reports"))
        self.figures_dir = self.reports_dir / "figures"
        self.metrics_file = output_config.get("metrics_file", "metrics.json")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from params.yaml."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
            
    def load_model(self, model_path: Optional[Path] = None) -> keras.Model:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model
        """
        if model_path is None:
            # Find the latest model
            model_files = list(self.models_dir.glob("*_final.keras")) + \
                          list(self.models_dir.glob("*_best.keras"))
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)
        
        return model
        
    def predict(
        self,
        model: keras.Model,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the model.
        
        Args:
            model: Trained Keras model
            X: Input images
            threshold: Classification threshold
            
        Returns:
            Tuple of (class predictions, probability predictions)
        """
        logger.info(f"Making predictions on {len(X)} samples...")
        
        y_pred_proba = model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return y_pred, y_pred_proba
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted class labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Computing evaluation metrics...")
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        }
        
        # Log metrics
        logger.info("Evaluation Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
            
        return metrics
        
    def generate_confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Path:
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for the plot title
            
        Returns:
            Path to saved figure
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE']
        )
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        save_path = self.figures_dir / f"{model_name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        
        return save_path
        
    def generate_roc_curve_plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model"
    ) -> Path:
        """
        Generate and save ROC curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name for the plot title
            
        Returns:
            Path to saved figure
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_path = self.figures_dir / f"{model_name}_roc_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
        
        return save_path
        
    def generate_combined_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model"
    ) -> Path:
        """
        Generate combined evaluation plot with confusion matrix and ROC curve.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name for the plot
            
        Returns:
            Path to saved figure
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE']
        )
        axes[0].set_title(f'{model_name} - Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'{model_name} - ROC Curve')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        save_path = self.figures_dir / f"{model_name}_evaluation.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Combined evaluation plot saved to {save_path}")
        
        return save_path
        
    def save_metrics(self, metrics: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save metrics to JSON file for DVC tracking.
        
        Args:
            metrics: Dictionary of metrics
            filename: Optional filename (uses config default if None)
            
        Returns:
            Path to saved metrics file
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = self.reports_dir / (filename or self.metrics_file)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {save_path}")
        
        return save_path
        
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=['REAL', 'FAKE'],
            digits=4
        )
        
        logger.info(f"\nClassification Report:\n{report}")
        
        return report
        
    def run(
        self,
        model: Optional[keras.Model] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        model_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.
        
        Args:
            model: Trained model (will load from disk if None)
            X_test, y_test: Test data
            model_path: Path to model file if loading
            
        Returns:
            Dictionary containing metrics and file paths
        """
        logger.info("=" * 50)
        logger.info("Starting Model Evaluation Pipeline")
        logger.info("=" * 50)
        
        # Load model if not provided
        if model is None:
            model = self.load_model(model_path)
            
        model_name = model.name if hasattr(model, 'name') else "Model"
        
        # Load test data if not provided
        if X_test is None or y_test is None:
            processed_dir = Path(self.config.get("data", {}).get("processed_dir", "data/processed"))
            X_test = np.load(processed_dir / "X_test.npy")
            y_test = np.load(processed_dir / "y_test.npy")
            
        # Make predictions
        y_pred, y_pred_proba = self.predict(model, X_test)
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate visualizations
        cm_path = self.generate_confusion_matrix_plot(y_test, y_pred, model_name)
        roc_path = self.generate_roc_curve_plot(y_test, y_pred_proba, model_name)
        combined_path = self.generate_combined_plot(y_test, y_pred, y_pred_proba, model_name)
        
        # Generate classification report
        report = self.generate_classification_report(y_test, y_pred)
        
        # Save metrics
        metrics_path = self.save_metrics(metrics)
        
        # Save experiment info for model registration stage
        # This file is required by the DVC pipeline for register_model.py
        experiment_info = {
            "model_name": model_name,
            "model_path": str(self.models_dir / f"{model_name}_final.keras"),
            "metrics": metrics,
            "figures": {
                "confusion_matrix": str(cm_path),
                "roc_curve": str(roc_path),
                "combined_plot": str(combined_path)
            }
        }
        experiment_info_path = self.reports_dir / "experiment_info.json"
        with open(experiment_info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        logger.info(f"Experiment info saved to {experiment_info_path}")
        
        logger.info("Model Evaluation Pipeline completed successfully!")
        
        return {
            "metrics": metrics,
            "metrics_path": metrics_path,
            "confusion_matrix_path": cm_path,
            "roc_curve_path": roc_path,
            "combined_plot_path": combined_path,
            "classification_report": report,
            "experiment_info_path": experiment_info_path
        }


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run evaluation
    evaluator = ModelEvaluator()
    results = evaluator.run()
    
    print("\n" + "=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
