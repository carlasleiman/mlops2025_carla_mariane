"""
SageMaker Training Pipeline for NYC Taxi Trip Duration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput

class SageMakerTrainingPipeline:
    """
    Class to create and manage SageMaker training pipeline.
    """
    
    def __init__(self, 
                 role_arn: str,
                 bucket_name: str,
                 region: str = "us-east-1",
                 instance_type: str = "ml.m5.xlarge",
                 instance_count: int = 1):
        """
        Initialize SageMaker training pipeline.
        
        Args:
            role_arn: IAM role ARN for SageMaker
            bucket_name: S3 bucket name for data and artifacts
            region: AWS region
            instance_type: SageMaker instance type
            instance_count: Number of instances
        """
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region = region
        self.instance_type = instance_type
        self.instance_count = instance_count
        
        # Initialize SageMaker session
        self.boto_session = boto3.Session(region_name=region)
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.boto_session,
            default_bucket=bucket_name
        )
        
    def create_training_step(self,
                            entry_point: str = "train_entrypoint.py",
                            source_dir: str = "scripts/sagemaker",
                            framework_version: str = "1.0-1",
                            hyperparameters: Optional[Dict[str, Any]] = None,
                            input_data_path: str = "s3://{}/data/train",
                            output_model_path: str = "s3://{}/models") -> TrainingStep:
        """
        Create SageMaker training step.
        """
        if hyperparameters is None:
            hyperparameters = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        
        # Format S3 paths
        input_path = input_data_path.format(self.bucket_name)
        output_path = output_model_path.format(self.bucket_name)
        
        # Create SKLearn estimator
        sklearn_estimator = SKLearn(
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version=framework_version,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role_arn,
            hyperparameters=hyperparameters,
            output_path=output_path,
            sagemaker_session=self.sagemaker_session,
            code_location=output_path
        )
        
        # Create training input
        train_input = TrainingInput(
            s3_data=input_path,
            content_type="text/csv"
        )
        
        # Create training step
        training_step = TrainingStep(
            name="NYCTaxiTrainingStep",
            estimator=sklearn_estimator,
            inputs={"train": train_input}
        )
        
        return training_step
    
    def create_pipeline(self,
                       pipeline_name: str = "NYCTaxiTrainingPipeline",
                       pipeline_description: str = "Training pipeline for NYC Taxi Trip Duration") -> Pipeline:
        """
        Create complete SageMaker training pipeline.
        """
        training_step = self.create_training_step()
        
        pipeline = Pipeline(
            name=pipeline_name,
            steps=[training_step],
            sagemaker_session=self.sagemaker_session
        )
        
        return pipeline
    
    def run_pipeline(self, 
                    pipeline_name: str = "NYCTaxiTrainingPipeline",
                    wait: bool = True) -> Dict[str, Any]:
        """
        Run the SageMaker training pipeline.
        """
        pipeline = self.create_pipeline(pipeline_name)
        
        # Upload and run pipeline
        pipeline.upsert(role_arn=self.role_arn)
        execution = pipeline.start()
        
        if wait:
            execution.wait()
            
        execution_steps = execution.list_steps()
        
        return {
            "execution_arn": execution.arn,
            "status": execution.describe()["PipelineExecutionStatus"],
            "steps": execution_steps
        }
