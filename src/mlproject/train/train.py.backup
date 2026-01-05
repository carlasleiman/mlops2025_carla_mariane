"""
Training module for NYC Taxi Trip Duration prediction.
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
    model_type: str = "random_forest"
    random_state: int = 42
    test_size: float = 0.2
    
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
    
    # Assuming labels are in a column called 'target'
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    return features, labels


def train_model(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """
    Main training function.
    """
    # Load configuration
    if config is None:
        config = TrainConfig()
    
    logger.info("Starting model training")
    
    # 1. Load data
    features, labels = load_training_data(config.features_path, config.labels_path)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # 3. Create and train model (simplified - we'll add your ModelTrainer later)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=config.random_state,
        n_jobs=-1
    )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # 4. Save model
    model_path = Path(config.model_output_dir) / config.best_model_filename
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 5. Return simple results
    return {
        "model_path": str(model_path),
        "train_samples": len(X_train),
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
    
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = TrainConfig()
    
    # Run training
    results = train_model(config)
    print(f"Training completed. Model saved to: {results['model_path']}")
