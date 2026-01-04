"""
SageMaker Inference Pipeline for NYC Taxi Trip Duration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CreateModelStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput

class SageMakerInferencePipeline:
    """
    Class to create and manage SageMaker batch inference pipeline.
    """
    
    def __init__(self,
                 role_arn: str,
                 bucket_name: str,
                 region: str = "us-east-1",
                 instance_type: str = "ml.m5.xlarge",
                 instance_count: int = 1):
        """
        Initialize SageMaker inference pipeline.
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
    
    def create_model_step(self,
                         model_name: str,
                         model_artifact_path: str,
                         entry_point: str = "inference_entrypoint.py",
                         source_dir: str = "scripts/sagemaker",
                         framework_version: str = "1.0-1") -> ModelStep:
        """
        Create SageMaker model step.
        """
        # Create SKLearn model
        sklearn_model = SKLearnModel(
            model_data=model_artifact_path,
            role=self.role_arn,
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version=framework_version,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create model step
        model_step = ModelStep(
            name=f"{model_name}ModelStep",
            step_args=sklearn_model.create(
                instance_type=self.instance_type,
                accelerator_type=None
            )
        )
        
        return model_step
    
    def create_transform_step(self,
                             model_name: str,
                             transformer_name: str,
                             input_data_path: str,
                             output_data_path: str,
                             batch_strategy: str = "MultiRecord",
                             max_payload: int = 6,
                             split_type: str = "Line") -> TransformStep:
        """
        Create SageMaker batch transform step.
        """
        # Format S3 paths
        input_path = input_data_path.format(self.bucket_name)
        output_path = output_data_path.format(self.bucket_name)
        
        # Create transformer
        transformer = Transformer(
            model_name=model_name,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            output_path=output_path,
            strategy=batch_strategy,
            assemble_with=split_type,
            max_payload=max_payload,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create transform input
        transform_input = TransformInput(
            data=input_path,
            content_type="text/csv"
        )
        
        # Create transform step
        transform_step = TransformStep(
            name=f"{transformer_name}TransformStep",
            transformer=transformer,
            inputs=transform_input
        )
        
        return transform_step
    
    def create_pipeline(self,
                       model_name: str = "NYCTaxiModel",
                       model_artifact_path: str = "s3://{}/models/model.tar.gz",
                       input_data_path: str = "s3://{}/data/test",
                       output_data_path: str = "s3://{}/predictions",
                       pipeline_name: str = "NYCTaxiInferencePipeline",
                       pipeline_description: str = "Batch inference pipeline for NYC Taxi Trip Duration") -> Pipeline:
        """
        Create complete SageMaker inference pipeline.
        """
        # Create model step
        model_step = self.create_model_step(
            model_name=model_name,
            model_artifact_path=model_artifact_path.format(self.bucket_name)
        )
        
        # Create transform step
        transform_step = self.create_transform_step(
            model_name=model_name,
            transformer_name="NYCTaxiTransformer",
            input_data_path=input_data_path,
            output_data_path=output_data_path
        )
        
        # Create pipeline with dependency
        pipeline = Pipeline(
            name=pipeline_name,
            steps=[model_step, transform_step],
            sagemaker_session=self.sagemaker_session
        )
        
        return pipeline
    
    def run_pipeline(self,
                    model_name: str = "NYCTaxiModel",
                    model_artifact_path: str = "s3://{}/models/model.tar.gz",
                    wait: bool = True) -> Dict[str, Any]:
        """
        Run the SageMaker inference pipeline.
        """
        pipeline = self.create_pipeline(
            model_name=model_name,
            model_artifact_path=model_artifact_path
        )
        
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
