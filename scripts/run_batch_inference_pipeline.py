#!/usr/bin/env python
"""
Script to run SageMaker batch inference pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from mlproject.pipelines import SageMakerOrchestrator

def main():
    parser = argparse.ArgumentParser(
        description="Run SageMaker batch inference pipeline for NYC Taxi Trip Duration"
    )
    
    parser.add_argument("--config", type=str, default="configs/sagemaker_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-path", type=str, required=True,
                       help="S3 path to model artifact (e.g., s3://bucket/models/model.tar.gz)")
    parser.add_argument("--name", type=str, default=None,
                       help="Custom pipeline name (optional)")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for pipeline completion")
    
    args = parser.parse_args()
    
    print("?? Initializing SageMaker inference pipeline...")
    
    try:
        # Initialize orchestrator
        orchestrator = SageMakerOrchestrator(config_path=args.config)
        
        # Run inference pipeline
        result = orchestrator.run_inference_pipeline(
            model_artifact_path=args.model_path,
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
        print(f"? Error running inference pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
