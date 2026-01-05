#!/usr/bin/env python
"""
Batch inference script for NYC Taxi Trip Duration prediction.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from mlproject.inference import InferencePipeline, validate_input_data

def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference for NYC Taxi Trip Duration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file (.pkl or .joblib)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save predictions CSV")
    
    # Optional arguments
    parser.add_argument("--validate", action="store_true",
                       help="Validate input data before inference")
    parser.add_argument("--include-features", action="store_true",
                       help="Include input features in output")
    
    args = parser.parse_args()
    
    print("?? Starting batch inference...")
    
    try:
        # Load input data
        print(f"?? Loading data from {args.input}")
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate input data
        required_columns = [
            'pickup_datetime',
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]
        
        if args.validate:
            print("?? Validating input data...")
            if not validate_input_data(df, required_columns):
                print("? Input validation failed")
                sys.exit(1)
            print("? Input data validation passed")
        
        # Create inference pipeline
        print(f"?? Loading model from {args.model}")
        pipeline = InferencePipeline(model_path=args.model)
        
        # Run inference
        print("? Running inference...")
        predictions = pipeline.run(
            input_data=df,
            output_path=args.output,
            include_features=args.include_features
        )
        
        # Print summary
        print(f"\n?? Inference completed successfully!")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Mean prediction: {predictions['prediction'].mean():.2f}")
        print(f"   Min prediction: {predictions['prediction'].min():.2f}")
        print(f"   Max prediction: {predictions['prediction'].max():.2f}")
        print(f"   Predictions saved to: {args.output}")
        
    except Exception as e:
        print(f"? Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
