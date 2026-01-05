🚖 NYC Taxi Trip Duration Prediction - MLOps Project

**Task:** Predict NYC taxi trip duration (regression)  
**Dataset:** [NYC Taxi Trip Duration | Kaggle](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)  
**Team:** Carla & Mariane  
**Course:** MLOps Course - USJ  
**Repository:** https://github.com/carlasleiman/mlops2025_carla_mariane
```
## 📁 Project Structure
mlops2025_carla_mariane/
├── src/mlproject/
│ ├── data/ # Data utilities
│ ├── preprocess/ # Preprocessing modules (Carla)
│ ├── features/ # Feature engineering (Mariane)
│ ├── train/ # Training modules (Carla)
│ ├── inference/ # Inference modules (Mariane)
│ ├── pipelines/ # Pipeline orchestration (Carla)
│ ├── utils/ # Utility functions (Mariane)
│ └── init.py
├── scripts/
│ ├── preprocess.py # Data preprocessing (Carla)
│ ├── feature_engineering.py # Feature engineering (Mariane)
│ ├── train.py # Model training (Carla)
│ └── batch_inference.py # Batch inference (Mariane)
├── configs/ # Configuration files (Carla)
├── tests/ # Test suite (Both)
├── Dockerfile # Docker setup (Carla)
├── docker-compose.yml # Multi-container (Carla)
├── pyproject.toml # Package config (Carla)
├── uv.lock # Locked dependencies (Carla)
└── README.md # Documentation (Carla)
```
text

## 🚀 Quick Start
### 1. Setup Environment
```bash
git clone https://github.com/carlasleiman/mlops2025_carla_mariane.git
cd mlops2025_carla_mariane
uv sync
2. Run Pipeline
bash
# Full training pipeline
uv run train

# Generate predictions
uv run inference
3. Individual Stages
bash
uv run python scripts/preprocess.py
uv run python scripts/feature_engineering.py
uv run python scripts/train.py
uv run python scripts/batch_inference.py
🐳 Docker Deployment
Build and Run
bash
docker build -t mlops-taxi .
docker-compose run app train
docker-compose run app inference
☁️ AWS SageMaker Deployment
Training Pipeline (Mariane)
bash
python scripts/run_training_pipeline.py \
  --role-arn <your-arn> \
  --bucket <your-bucket> \
  --prefix mlops-project
Inference Pipeline (Mariane)
bash
python scripts/run_batch_inference_pipeline.py \
  --model-path s3://<path>/models/ \
  --input-data s3://<path>/test.csv
📊 Model Selection & Evaluation
Evaluation Metric: Root Mean Squared Error (RMSE)

Justification:

Sensitive to large errors (important for trip duration)

Same units as target (seconds)

Standard for regression tasks

Models Evaluated:

Random Forest Regressor

XGBoost Regressor

Results:

Model	Validation RMSE	Training Time
Random Forest	345.2 seconds	2.1 minutes
XGBoost	321.8 seconds	1.8 minutes
Final Model: XGBoost Regressor

Lower RMSE (321.8s vs 345.2s)

Faster training & inference

Better generalization

👥 Team Responsibilities
Carla
Preprocessing: Data cleaning, missing values, outliers

Training: Model training, hyperparameter tuning

Docker: Dockerfile, docker-compose setup

CI/Setup: GitHub Actions workflow, project structure

Configuration: OmegaConf config management

Packaging: src/ layout, pyproject.toml

Documentation: README.md

Mariane
Features: Feature engineering, distance calculations

Inference: Batch prediction pipeline

SageMaker: AWS training & inference pipelines

S3 Integration: Cloud storage setup

Testing: Feature and inference tests

🔧 Key Commands
Testing
bash
uv run pytest tests/ -v
CI Pipeline (GitHub Actions)
Runs on push/pull request

Installs dependencies with uv

Runs test suite

Validates preprocessing

Configuration
Uses OmegaConf (configs/train.yaml)

Centralized settings for training/inference

Easy parameter tuning

✅ Requirements Met
Mandatory:
Git workflow with feature branches + PRs

uv dependency management

src/ layout Python packaging

Docker & docker-compose

AWS SageMaker pipelines

Complete ML pipeline

CLI: uv run train & uv run inference

Best Practices:
Reproducible environment

Configuration management

Testing suite

Modular structure

Clear team contributions

Course: MLOps - USJ
Status: ✅ Complete & Production-Ready
Last Updated: January 2025
