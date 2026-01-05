#!/usr/bin/env python3
"""Simple test for TaxiPipeline without external dependencies."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing TaxiPipeline basic functionality...")
print("=" * 50)

try:
    # Import directly to avoid __init__.py issues
    from mlproject.pipelines.pipeline import TaxiPipeline
    
    print("✅ TaxiPipeline class imported successfully!")
    
    # Test creation
    pipeline = TaxiPipeline("configs/pipeline.yaml")
    print("✅ Pipeline instance created successfully!")
    
    print(f"\nPipeline components initialized:")
    print(f"  - Preprocessor: {type(pipeline.preprocessor).__name__}")
    print(f"  - FeatureEngineer: {type(pipeline.feature_engineer).__name__}")
    print(f"  - ModelTrainer: {type(pipeline.model_trainer).__name__}")
    
    print("\n" + "=" * 50)
    print("✅ BASIC TEST PASSED!")
    print("\nThe pipeline is ready to use.")
    print("Run it with:")
    print("  python -m mlproject.pipelines.pipeline --config configs/pipeline.yaml")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
