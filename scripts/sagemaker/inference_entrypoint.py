#!/usr/bin/env python
"""
SageMaker inference entry point for NYC Taxi Trip Duration.
This script runs in SageMaker inference container.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == 'text/csv':
        # Read CSV from string
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions."""
    # Preprocess input data (same as training)
    df = input_data.copy()
    
    # Convert datetime
    if 'pickup_datetime' in df.columns:
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
    
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
        df['distance_km'] = haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
    
    # Encode categorical
    if 'vendor_id' in df.columns:
        df['vendor_id'] = pd.factorize(df['vendor_id'])[0]
    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'] = pd.factorize(df['store_and_fwd_flag'])[0]
    
    # Select features (must match training)
    feature_cols = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'hour', 'day_of_week', 
                   'month', 'is_weekend', 'distance_km']
    
    # Keep only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols]
    
    # Make predictions
    predictions = model.predict(X)
    return predictions

def output_fn(prediction, accept):
    """Format predictions for output."""
    if accept == 'text/csv':
        # Return as CSV
        output = pd.DataFrame({'prediction': prediction})
        return output.to_csv(index=False), accept
    elif accept == 'application/json':
        # Return as JSON
        output = {'predictions': prediction.tolist()}
        return json.dumps(output), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
