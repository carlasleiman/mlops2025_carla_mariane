"""
Tests for inference module.
"""

import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.append("src")

from mlproject.inference import InferencePipeline, load_model, validate_input_data


def test_inference_pipeline():
    """Test InferencePipeline with mock data."""

    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.ones(X.shape[0]) * 1000

    # Create sample data
    data = {
        "pickup_datetime": ["2025-03-18 10:30:00", "2025-03-18 11:15:00"],
        "pickup_latitude": [40.7, 40.75],
        "pickup_longitude": [-74.0, -73.98],
        "dropoff_latitude": [40.8, 40.8],
        "dropoff_longitude": [-74.0, -73.95],
        "vendor_id": [1, 2],
        "store_and_fwd_flag": ["N", "Y"],
    }
    df = pd.DataFrame(data)

    # Create pipeline with mock model
    model = MockModel()
    pipeline = InferencePipeline(model=model)

    # Run inference
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "predictions.csv"
        result = pipeline.run(df, output_path=str(output_path))

        # Check results
        assert len(result) == 2
        assert "prediction" in result.columns
        assert "prediction_timestamp" in result.columns
        assert output_path.exists()

        print("? InferencePipeline test passed")


def test_validate_input_data():
    """Test input data validation."""

    # Valid data
    valid_data = pd.DataFrame(
        {
            "pickup_datetime": ["2025-03-18 10:30:00"],
            "pickup_latitude": [40.7],
            "pickup_longitude": [-74.0],
            "dropoff_latitude": [40.8],
            "dropoff_longitude": [-74.0],
        }
    )

    # Invalid data (missing column)
    invalid_data = pd.DataFrame(
        {
            "pickup_datetime": ["2025-03-18 10:30:00"],
            "pickup_latitude": [40.7],
            # Missing pickup_longitude
            "dropoff_latitude": [40.8],
            "dropoff_longitude": [-74.0],
        }
    )

    required = [
        "pickup_datetime",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    # Test valid data
    assert validate_input_data(valid_data, required) == True

    # Test invalid data
    assert validate_input_data(invalid_data, required) == False

    print("? validate_input_data test passed")


def run_all_tests():
    """Run all inference tests."""
    print("?? Running inference module tests...")

    test_inference_pipeline()
    test_validate_input_data()

    print("\n?? All inference tests passed!")


if __name__ == "__main__":
    run_all_tests()
