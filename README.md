# mlops2025_carla_mariane
End-to-End ML Project - NYC Taxi Trip Duration Prediction
🚖 NYC Taxi Trip Duration Prediction - MLOps Project
Task: Predict NYC taxi trip duration (regression)
Dataset: NYC Taxi Trip Duration | Kaggle
Team: Carla & Mariane
Course: MLOps Course - USJ
Repository: mlops2025_carla_mariane

mlops2025_carla_mariane/

├── .github/workflows/          # CI/CD pipelines
├── src/mlproject/              # Source code package
│   ├── data/                   # Data utilities
│   ├── preprocess/             # Preprocessing modules
│   ├── features/               # Feature engineering
│   ├── train/                  # Training modules
│   ├── inference/              # Inference modules
│   ├── pipelines/              # Pipeline orchestration
│   ├── utils/                  # Utility functions
│   └── __init__.py
├── scripts/                    # Pipeline scripts
│   ├── preprocess.py           # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   ├── train.py                # Model training
│   └── batch_inference.py      # Batch inference
├── configs/                    # Configuration files
├── tests/                      # Test suite
├── notebooks/                  # Exploratory analysis
├── outputs/                    # Prediction outputs
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container setup
├── pyproject.toml              # Package configuration
├── uv.lock                     # Locked dependencies
└── README.md                   # Project documentation

👥 Team Contributions
Carla Sleiman:

CI/CD Pipeline: GitHub Actions workflow, automation

Containerization: Dockerfile, docker-compose setup

Preprocessing: Data cleaning, missing value handling, outlier detection

Training Pipeline: Model training, hyperparameter tuning

Configuration: OmegaConf setup, project configuration

Packaging: src/ layout, pyproject.toml, dependency management

Mariane:

Feature Engineering: Time-based features, distance calculations (Haversine)

Inference Pipeline: Batch prediction system, output generation

Cloud Deployment: AWS SageMaker pipelines, S3 integration

Testing Suite: Feature and inference tests

Documentation: README, pipeline documentation

Code Quality: Linting, pre-commit hooks
