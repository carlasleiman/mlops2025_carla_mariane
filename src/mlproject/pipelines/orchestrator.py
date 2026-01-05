"""
SageMaker Pipeline Orchestrator for NYC Taxi Trip Duration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime

from .sagemaker_train import SageMakerTrainingPipeline
from .sagemaker_inference import SageMakerInferencePipeline

class SageMakerOrchestrator:
    """
    Orchestrator class to manage SageMaker pipelines.
    """
    
    def __init__(self, config_path: str = "configs/sagemaker_config.yaml"):
        """
        Initialize SageMaker orchestrator.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize pipelines
        self.training_pipeline = SageMakerTrainingPipeline(
            role_arn=self.config["role_arn"],
            bucket_name=self.config["bucket_name"],
            region=self.config.get("region", "us-east-1"),
            instance_type=self.config.get("instance_type", "ml.m5.xlarge"),
            instance_count=self.config.get("instance_count", 1)
        )
        
        self.inference_pipeline = SageMakerInferencePipeline(
            role_arn=self.config["role_arn"],
            bucket_name=self.config["bucket_name"],
            region=self.config.get("region", "us-east-1"),
            instance_type=self.config.get("instance_type", "ml.m5.xlarge"),
            instance_count=self.config.get("instance_count", 1)
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        """
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Create default config template
            default_config = {
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "bucket_name": "mlops-nyc-taxi-bucket",
                "region": "us-east-1",
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "training": {
                    "entry_point": "train_entrypoint.py",
                    "source_dir": "scripts/sagemaker",
                    "framework_version": "1.0-1",
                    "hyperparameters": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42
                    }
                },
                "inference": {
                    "entry_point": "inference_entrypoint.py",
                    "source_dir": "scripts/sagemaker",
                    "framework_version": "1.0-1",
                    "model_name": "NYCTaxiModel"
                }
            }
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            print(f"??  Created default config at {config_path}")
            return default_config
        
        # Load existing config
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_training_pipeline(self, 
                            wait: bool = True,
                            pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run training pipeline.
        """
        if pipeline_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            pipeline_name = f"NYCTaxiTraining-{timestamp}"
        
        print(f"?? Starting training pipeline: {pipeline_name}")
        
        result = self.training_pipeline.run_pipeline(
            pipeline_name=pipeline_name,
            wait=wait
        )
        
        print(f"? Training pipeline completed with status: {result['status']}")
        return result
    
    def run_inference_pipeline(self,
                              model_artifact_path: str,
                              wait: bool = True,
                              pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference pipeline.
        """
        if pipeline_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            pipeline_name = f"NYCTaxiInference-{timestamp}"
        
        print(f"?? Starting inference pipeline: {pipeline_name}")
        
        result = self.inference_pipeline.run_pipeline(
            model_name=self.config["inference"]["model_name"],
            model_artifact_path=model_artifact_path,
            wait=wait
        )
        
        print(f"? Inference pipeline completed with status: {result['status']}")
        return result
    
    def get_pipeline_status(self, execution_arn: str) -> Dict[str, Any]:
        """
        Get status of a pipeline execution.
        """
        import boto3
        
        client = boto3.client('sagemaker', region_name=self.config["region"])
        response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        
        return {
            "status": response["PipelineExecutionStatus"],
            "start_time": response.get("CreationTime"),
            "end_time": response.get("LastModifiedTime"),
            "arn": execution_arn
        }
