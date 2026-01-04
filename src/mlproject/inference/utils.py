"""
Utility functions for inference operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import joblib

def load_model(model_path: str):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to model file (.pkl, .joblib)
        
    Returns:
        Loaded model object
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_path.suffix == '.joblib':
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

def validate_input_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that input data has required columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        True if validation passes, False otherwise
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"? Missing required columns: {missing}")
        return False
    
    # Check for NaN values in required columns
    for col in required_columns:
        if df[col].isna().any():
            print(f"??  Column '{col}' contains NaN values")
    
    return True

def calculate_metrics(predictions: np.ndarray, 
                     actuals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate metrics for predictions.
    
    Args:
        predictions: Array of predicted values
        actuals: Optional array of actual values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'count': len(predictions),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'median': float(np.median(predictions))
    }
    
    if actuals is not None and len(actuals) == len(predictions):
        errors = predictions - actuals
        metrics.update({
            'mae': float(np.mean(np.abs(errors))),
            'mse': float(np.mean(errors ** 2)),
            'rmse': float(np.sqrt(metrics['mse']))
        })
    
    return metrics

def prepare_output(predictions: pd.DataFrame, 
                   include_id: bool = True,
                   include_features: bool = False) -> pd.DataFrame:
    """
    Prepare output DataFrame with predictions.
    
    Args:
        predictions: DataFrame with predictions
        include_id: Whether to include ID column
        include_features: Whether to include input features
        
    Returns:
        Formatted output DataFrame
    """
    output_cols = []
    
    # Add ID if available
    if include_id and 'id' in predictions.columns:
        output_cols.append('id')
    
    # Add prediction
    output_cols.append('prediction')
    
    # Add timestamp
    if 'prediction_timestamp' in predictions.columns:
        output_cols.append('prediction_timestamp')
    
    # Add features if requested
    if include_features:
        feature_cols = ['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                       'dropoff_latitude', 'dropoff_longitude']
        for col in feature_cols:
            if col in predictions.columns:
                output_cols.append(col)
    
    return predictions[output_cols]
