"""
Training module for NYC Taxi Trip Duration prediction.
Uses TaxiDurationTrainer class.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for model training."""
    # Data paths
    features_path: str = "data/processed/features.parquet"
    labels_path: str = "data/processed/labels.parquet"
    
    # Model parameters
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2  # Of training data
    
    # MLflow settings
    mlflow_experiment: str = "NYC-Taxi-Trip-ML"
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Output paths
    model_output_dir: str = "models"
    metrics_output_dir: str = "reports"
    best_model_filename: str = "best_model.joblib"
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metrics_output_dir).mkdir(parents=True, exist_ok=True)


def load_training_data(features_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features and labels for training.
    """
    logger.info(f"Loading features from {features_path}")
    features = pd.read_parquet(features_path)
    
    logger.info(f"Loading labels from {labels_path}")
    labels = pd.read_parquet(labels_path)
    
    # Assuming labels are in a column called 'trip_duration'
    if isinstance(labels, pd.DataFrame):
        if 'trip_duration' in labels.columns:
            labels = labels['trip_duration']
        else:
            labels = labels.iloc[:, 0]
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    return features, labels


def split_data(features: pd.DataFrame, labels: pd.Series, config: TrainConfig) -> Tuple:
    """
    Split data into train, validation, and test sets.
    """
    # First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    # Second split: train and validation from remaining data
    val_size = config.validation_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=config.random_state
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """
    Main training function using TaxiDurationTrainer.
    """
    # Load configuration
    if config is None:
        config = TrainConfig()
    
    logger.info("Starting NYC Taxi Trip Duration training")
    logger.info(f"Configuration: {asdict(config)}")
    
    # 1. Load data
    features, labels = load_training_data(config.features_path, config.labels_path)
    
    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels, config)
    
    # 3. Initialize and train using TaxiDurationTrainer
    from .taxi_trainer import TaxiDurationTrainer
    
    # Configure MLflow
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Create trainer
    trainer = TaxiDurationTrainer(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_tracking_uri
    )
    
    # Train models
    logger.info("Training models with TaxiDurationTrainer...")
    best_model = trainer.train_and_select(
        X_train, y_train,
        X_val, y_val
    )
    
    # 4. Save model
    model_path = Path(config.model_output_dir) / config.best_model_filename
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 5. Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = Path(config.metrics_output_dir) / f"metrics_{timestamp}.json"
    
    metrics_data = {
        "timestamp": timestamp,
        "config": asdict(config),
        "data_summary": {
            "total_samples": len(features),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features": len(features.columns)
        },
        "model_info": {
            "type": "TaxiDurationTrainer",
            "path": str(model_path),
            "features": list(features.columns)
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # 6. Return results
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    }


def load_config_from_file(config_path: str) -> TrainConfig:
    """
    Load training configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainConfig(**config_dict)


if __name__ == "__main__":
    """Entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NYC Taxi Trip Duration model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--features", type=str, help="Path to features file")
    parser.add_argument("--labels", type=str, help="Path to labels file")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = TrainConfig()
    
    # Override config
    if args.features:
        config.features_path = args.features
    if args.labels:
        config.labels_path = args.labels
    
    # Run training
    results = train_model(config)
    print(f"Training completed. Model saved to: {results['model_path']}")
