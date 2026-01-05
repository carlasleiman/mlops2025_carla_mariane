#!/usr/bin/env python3
"""
Preprocessing script for NYC Taxi Trip Duration dataset.
Required by MLOps project structure.

Usage:
    python scripts/preprocess.py --input data/train.csv --output data/cleaned_train.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess NYC Taxi data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def load_data(file_path):
    """Load CSV file."""
    print(f"📂 Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_data(df, verbose=False):
    """
    Clean the NYC Taxi dataset.
    
    Steps:
    1. Remove rows with missing coordinates
    2. Handle missing passenger counts
    3. Convert datetime columns
    4. Remove invalid coordinates
    """
    df_clean = df.copy()
    
    # 1. Handle missing coordinates
    coord_cols = ['pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude']
    
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=coord_cols)
    removed = initial_rows - len(df_clean)
    if verbose and removed > 0:
        print(f"   Removed {removed} rows with missing coordinates")
    
    # 2. Fill missing passenger counts
    if 'passenger_count' in df_clean.columns:
        df_clean['passenger_count'] = df_clean['passenger_count'].fillna(1)
        if verbose:
            missing_filled = df['passenger_count'].isna().sum()
            if missing_filled > 0:
                print(f"   Filled {missing_filled} missing passenger counts with 1")
    
    # 3. Convert datetime columns
    datetime_cols = ['pickup_datetime', 'dropoff_datetime']
    for col in datetime_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col])
            if verbose:
                print(f"   Converted {col} to datetime")
    
    # 4. Basic coordinate validation (NYC area)
    nyc_lat_range = (40.5, 41.0)
    nyc_lon_range = (-74.3, -73.7)
    
    mask = (
        df_clean['pickup_latitude'].between(*nyc_lat_range) &
        df_clean['pickup_longitude'].between(*nyc_lon_range) &
        df_clean['dropoff_latitude'].between(*nyc_lat_range) &
        df_clean['dropoff_longitude'].between(*nyc_lon_range)
    )
    
    invalid_rows = len(df_clean) - mask.sum()
    df_clean = df_clean[mask]
    
    if verbose and invalid_rows > 0:
        print(f"   Removed {invalid_rows} rows with coordinates outside NYC area")
    
    print(f"✅ Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

def save_data(df, output_path):
    """Save processed data to CSV."""
    print(f"💾 Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"   Saved {len(df)} rows")

def main():
    """Main preprocessing pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 STARTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    try:
        # 1. Load
        df = load_data(args.input)
        
        # 2. Clean
        df_clean = clean_data(df, verbose=args.verbose)
        
        # 3. Save
        save_data(df_clean, args.output)
        
        print("=" * 60)
        print("🎉 PREPROCESSING COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
