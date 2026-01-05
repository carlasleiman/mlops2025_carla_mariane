#!/usr/bin/env python3
"""
CLI script for training models.
Matches the project requirement: uv run train
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from mlproject.train import train_model, load_config_from_file
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ML model for NYC Taxi Trip Duration")
    parser.add_argument("--config", "-c", type=str, default="configs/train_config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--features", "-f", type=str, 
                       help="Override features path from config")
    parser.add_argument("--labels", "-l", type=str,
                       help="Override labels path from config")
    parser.add_argument("--output-dir", "-o", type=str,
                       help="Override output directory from config")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow logging")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config_from_file(args.config)
        
        # Override with command line arguments
        if args.features:
            config.features_path = args.features
        if args.labels:
            config.labels_path = args.labels
        if args.output_dir:
            config.model_output_dir = args.output_dir
        
        # Run training
        results = train_model(config)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Model saved to: {results['model_path']}")
        print(f"Train samples: {results['train_samples']}")
        print(f"Test samples: {results['test_samples']}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure to run preprocessing and feature engineering first!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
