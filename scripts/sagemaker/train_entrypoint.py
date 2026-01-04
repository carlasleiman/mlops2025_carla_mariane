#!/usr/bin/env python
"""
SageMaker training entry point for NYC Taxi Trip Duration.
This script runs in SageMaker training container.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

def load_data(data_dir):
    """Load training data from SageMaker input directory."""
    train_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(train_path)
    return df

def preprocess_data(df):
    """Preprocess the data."""
    # Convert datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Encode categorical
    df['vendor_id'] = pd.factorize(df['vendor_id'])[0]
    df['store_and_fwd_flag'] = pd.factorize(df['store_and_fwd_flag'])[0]
    
    return df

def train(args):
    """Train the model."""
    print("?? Starting training in SageMaker...")
    
    # Load data
    data_dir = args.data_dir
    df = load_data(data_dir)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Prepare features and target
    feature_cols = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'hour', 'day_of_week', 
                   'month', 'is_weekend', 'distance_km']
    
    X = df[feature_cols]
    y = df['trip_duration']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )
    
    # Train model
    print(f"Training RandomForest with n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"? Training completed!")
    print(f"   Validation MAE: {mae:.2f}")
    print(f"   Validation RMSE: {rmse:.2f}")
    
    # Save model
    model_dir = args.model_dir
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"?? Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'model_type': 'RandomForestRegressor',
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': args.random_state
        }
    }
    
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print("?? Training pipeline completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    
    args = parser.parse_args()
    train(args)
