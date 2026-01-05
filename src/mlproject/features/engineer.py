class FeatureEngineer:
    def transform(self, df, fit=False, is_train=True):
        print(f"[FeatureEngineer] Transforming data (fit={fit}, is_train={is_train})...")
        
        # Make a copy
        df_transformed = df.copy()
        
        # Try to import Mariane's functions
        try:
            from .distance import haversine_distance
            if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                                 'dropoff_latitude', 'dropoff_longitude']):
                df_transformed['haversine_distance'] = haversine_distance(
                    df['pickup_latitude'], df['pickup_longitude'],
                    df['dropoff_latitude'], df['dropoff_longitude']
                )
        except:
            print("Warning: Could not import distance functions")
        
        try:
            from .time_features import extract_time_features
            if 'pickup_datetime' in df.columns:
                df_transformed = extract_time_features(df_transformed, 'pickup_datetime')
        except:
            print("Warning: Could not import time features functions")
        
        # Separate features and target
        if is_train and 'trip_duration' in df_transformed.columns:
            X = df_transformed.drop('trip_duration', axis=1)
            y = df_transformed['trip_duration']
            feature_names = X.columns.tolist()
            print(f"[FeatureEngineer] Transformation complete. X: {X.shape}, y: {y.shape}")
            return X, y, feature_names
        else:
            X = df_transformed
            feature_names = X.columns.tolist()
            print(f"[FeatureEngineer] Transformation complete. X: {X.shape}")
            return X, None, feature_names
