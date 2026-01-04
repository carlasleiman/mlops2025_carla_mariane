import pandas as pd
import argparse
from pathlib import Path
import sys

# Add src to path to import our modules
sys.path.append("src")

from mlproject.features.distance import haversine_km
from mlproject.features.time_features import extract_time_features
from mlproject.features.encoding import encode_categorical

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for NYC Taxi Trip Duration")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned data CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save feature dataset CSV")
    args = parser.parse_args()

    # Load cleaned data
    df = pd.read_csv(args.input)
    print(f"? Loaded {len(df)} rows from {args.input}")

    # 1. Add distance feature (Haversine)
    print("?? Calculating distances...")
    df["distance_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    # 2. Extract time features
    print("?? Extracting time features...")
    df = extract_time_features(df, "pickup_datetime")

    # 3. Encode categorical variables
    print("?? Encoding categorical variables...")
    df = encode_categorical(df, ["vendor_id", "store_and_fwd_flag"])

    # Save feature dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"? Features saved to {args.output}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
