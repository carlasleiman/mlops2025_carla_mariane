import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import sys
import pickle
import joblib

# Add src to path for imports
sys.path.append('src')

class InferencePipeline:
    """
    Handles batch inference for a trained model:
    - Applies feature engineering
    - Generates predictions
    - Saves predictions to CSV
    """

    def __init__(self, model=None, model_path: str = None):
        """
        Initialize inference pipeline.
        
        Args:
            model: Pre-loaded model object (optional)
            model_path: Path to saved model file (.pkl or .joblib)
        """
        if model is not None:
            self.model = model
            print("? Using provided model")
        elif model_path is not None:
            self.model = self._load_model(model_path)
            print(f"? Loaded model from {model_path}")
        else:
            raise ValueError("Must provide either 'model' or 'model_path'")

    def _load_model(self, model_path: str):
        """Load model from file."""
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

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw data.
        This should match the features used during training.
        """
        df_features = df.copy()
        
        # Convert datetime
        if 'pickup_datetime' in df_features.columns:
            df_features['pickup_datetime'] = pd.to_datetime(df_features['pickup_datetime'])
            df_features['hour'] = df_features['pickup_datetime'].dt.hour
            df_features['day_of_week'] = df_features['pickup_datetime'].dt.dayofweek
            df_features['month'] = df_features['pickup_datetime'].dt.month
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate distance
        if all(col in df_features.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                                      'dropoff_latitude', 'dropoff_longitude']):
            df_features['distance_km'] = self._haversine_distance(
                df_features['pickup_latitude'], df_features['pickup_longitude'],
                df_features['dropoff_latitude'], df_features['dropoff_longitude']
            )
        
        # Encode categorical variables
        categorical_cols = ['vendor_id', 'store_and_fwd_flag']
        for col in categorical_cols:
            if col in df_features.columns:
                df_features[col] = pd.factorize(df_features[col])[0]
        
        # Select numeric columns for prediction
        numeric_cols = df_features.select_dtypes(include=['number']).columns
        return df_features[numeric_cols]

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in km."""
        R = 6371  # Earth radius in km
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    def run(
        self,
        input_data: pd.DataFrame,
        output_path: Optional[str] = None,
        include_features: bool = False
    ) -> pd.DataFrame:
        """
        Run batch inference on input data.
        
        Args:
            input_data: DataFrame with input data
            output_path: Path to save predictions (optional)
            include_features: Whether to include input features in output
            
        Returns:
            DataFrame with predictions
        """
        print(f"?? Processing {len(input_data)} records...")
        
        # Extract features
        X = self._extract_features(input_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create output DataFrame
        result = input_data.copy()
        result['prediction'] = predictions
        result['prediction_timestamp'] = datetime.now()
        
        # Save if output path provided
        if output_path:
            self._save_predictions(result, output_path)
        
        return result

    def _save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """Save predictions to CSV file."""
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename if it's a directory
        if output_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"predictions_{timestamp}.csv"
        elif output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        
        predictions.to_csv(output_path, index=False)
        print(f"?? Predictions saved to {output_path}")
