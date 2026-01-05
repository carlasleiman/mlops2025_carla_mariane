"""
Pipeline module for NYC Taxi project.
"""

from .pipeline import TaxiPipeline, main

__all__ = ['TaxiPipeline', 'main']

# Optional imports - don't fail if SageMaker dependencies are missing
try:
    from .sagemaker_train import SageMakerTrainingPipeline
    from .sagemaker_inference import SageMakerInferencePipeline
    __all__.extend(['SageMakerTrainingPipeline', 'SageMakerInferencePipeline'])
    print("[INFO] SageMaker pipelines available")
except ImportError as e:
    # Don't print error to avoid confusion
    pass
