#!/usr/bin/env python3
"""
Training script using Carla's TaxiDurationTrainer.
Author: Carla Sleiman
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

print("=" * 60)
print("TAXI DURATION TRAINING SCRIPT")
print("Using: TaxiDurationTrainer (Carla's version)")
print("=" * 60)

print("\nThis script will:")
print("1. Load Kaggle NYC Taxi data")
print("2. Preprocess using Preprocessor class")
print("3. Train 5 models with TaxiDurationTrainer")
print("4. Select best model automatically")
print("5. Save model and log to MLflow")

print("\nTo run with sample data:")
print("python scripts/train_with_sample.py")

print("\nTo run with real Kaggle data:")
print("1. Place train.csv in data/raw/ folder")
print("2. Run: python scripts/train_with_real_data.py")
