#!/usr/bin/env python
"""
Script to run SageMaker training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from mlproject.pipelines import SageMakerOrchestrator

def main():
    parser = argparse.ArgumentParser(
        description="Run SageMaker training pipeline for NYC Taxi Trip Duration"
    )
    
    parser.add_argument("--config", type=str, default="configs/sagemaker_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--name", type=str, default=None,
                       help="Custom pipeline name (optional)")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for pipeline completion")
    
    args = parser.parse_args()
    
    print("?? Initializing SageMaker training pipeline...")
    
    try:
        # Initialize orchestrator
        orchestrator = SageMakerOrchestrator(config_path=args.config)
        
        # Run training pipeline
        result = orchestrator.run_training_pipeline(
            pipeline_name=args.name,
            wait=args.wait
        )
        
        print(f"\n? Pipeline Execution Details:")
        print(f"   ARN: {result['execution_arn']}")
        print(f"   Status: {result['status']}")
        print(f"   Steps: {len(result['steps'])}")
        
        if not args.wait:
            print("\n??  Pipeline started in background.")
            print(f"   Use this ARN to check status: {result['execution_arn']}")
            
    except Exception as e:
        print(f"? Error running training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
