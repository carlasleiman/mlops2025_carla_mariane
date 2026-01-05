#!/usr/bin/env python3
"""
Train with sample data (no Kaggle data needed).
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from mlproject.preprocess import Preprocessor
from mlproject.train import TaxiDurationTrainer
from sklearn.model_selection import train_test_split

print("Creating sample data...")
np.random.seed(42)

# Simple sample data
data = {
    'pickup_datetime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00'],
    'pickup_latitude': [40.7128, 40.7138, 40.7148],
    'pickup_longitude': [-74.0060, -74.0070, -74.0080],
    'dropoff_latitude': [40.7580, 40.7590, 40.7600],
    'dropoff_longitude': [-73.9855, -73.9865, -73.9875],
    'trip_duration': [600, 1200, 900],
    'passenger_count': [1, 2, 3]
}

df = pd.DataFrame(data)

# Preprocess
processor = Preprocessor()
df_processed = processor.run(df, is_train=True)

# Train
features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
if 'pickup_hour' in df_processed.columns:
    features.append('pickup_hour')

X = df_processed[features]
y = df_processed['trip_duration']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

trainer = TaxiDurationTrainer(optimize_for="mae")
best_model, best_name = trainer.train_models(X_train, y_train, X_val, y_val)

print(f"✓ Training complete! Best model: {best_name}")
print("Note: This uses sample data. For real training, use Kaggle dataset.")
