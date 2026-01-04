import mlflow
import joblib
import time
from typing import Dict, Tuple, Optional
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Add logging 
logger = logging.getLogger(__name__)


class TaxiDurationTrainer:
    """
    NYC Taxi Trip Duration Model Trainer
    Author: Carla Sleiman
    
    Features:
    - Multiple model training with automatic selection
    - MLflow experiment tracking and logging
    - Model evaluation and comparison
    - Model persistence and loading
    """
    
    def __init__(
        self,
        optimize_for: str = "mae",
        mlflow_experiment: str = "NYC-Taxi-Duration-ML",
        model_params: Optional[Dict] = None,
    ):
        self.optimize_for = optimize_for
        self.mlflow_experiment = mlflow_experiment
        self.model_params = model_params or self._get_default_params()
        
        # Initialize tracking
        self.selected_model = None
        self.selected_model_name = None
        self.selected_score = (
            float("inf") if optimize_for in ["mae", "rmse"] else -float("inf")
        )
        self.selected_run_id = None
        self.mlflow_enabled = True
        
        logger.info(f"Initialized trainer optimizing for: {optimize_for}")
    
   
    @staticmethod
    def _get_default_params() -> Dict:
        return {
            "lin_reg": {},
            "ridge_reg": {"alpha": 1.0},
            "lasso_reg": {"alpha": 0.1},
            "rf_reg": {
                "n_estimators": 150,  # Changed from 200
                "max_depth": 10,      # Changed from 12
                "n_jobs": -1,
                "random_state": 42    # Added random state
            },
            "gbr": {
                "n_estimators": 150,  # Changed from 200
                "max_depth": 4,       # Changed from 3
                "learning_rate": 0.1,
                "random_state": 42    # Added random state
            }
        }
    
  
    def _create_models(self) -> Dict:
        params = self.model_params
        return {
            "Linear Regression": LinearRegression(**params.get("lin_reg", {})),
            "Ridge Regression": Ridge(**params.get("ridge_reg", {})),
            "Lasso Regression": Lasso(**params.get("lasso_reg", {})),
            "Random Forest": RandomForestRegressor(**params.get("rf_reg", {})),
            "Gradient Boosting": GradientBoostingRegressor(**params.get("gbr", {}))
        }
    
  
    @staticmethod
    def compute_metrics(model, X, y) -> Dict:
        predictions = model.predict(X)
        return {
            "mae": mean_absolute_error(y, predictions),
            "rmse": mean_squared_error(y, predictions, squared=False),
            "r2": r2_score(y, predictions),
            "pred_mean": predictions.mean(),  # Added extra metric
            "pred_std": predictions.std()     # Added extra metric
        }
    
    # Main training method - restructured slightly
    def train_models(self, X_train, y_train, X_val, y_val) -> Tuple[object, str]:
        # MLflow setup with connection handling
        attempts = 3
        wait_time = 5
        for attempt_num in range(1, attempts + 1):
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                logger.info(f"MLflow experiment '{self.mlflow_experiment}' configured")
                break
            except Exception as err:
                logger.warning(f"MLflow connection attempt {attempt_num} failed: {err}")
                if attempt_num == attempts:
                    logger.warning("MLflow disabled, continuing locally")
                    self.mlflow_enabled = False
                else:
                    time.sleep(wait_time)
        
        # Create signature if possible
        model_signature = None
        try:
            sample_model = LinearRegression()
            sample_model.fit(X_train, y_train)
            model_signature = infer_signature(X_train, sample_model.predict(X_val))
        except Exception as err:
            logger.warning(f"Could not create model signature: {err}")
        
        # Get all models
        all_models = self._create_models()
        
        # Train each model
        for model_name, model in all_models.items():
            logger.info(f"Training: {model_name}")
            
            if self.mlflow_enabled:
                try:
                    with mlflow.start_run(run_name=model_name):
                        # Log parameters
                        param_key = {
                            "Linear Regression": "lin_reg",
                            "Ridge Regression": "ridge_reg", 
                            "Lasso Regression": "lasso_reg",
                            "Random Forest": "rf_reg",
                            "Gradient Boosting": "gbr"
                        }.get(model_name, model_name.lower())
                        
                        if param_key in self.model_params:
                            mlflow.log_params(self.model_params[param_key])
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Log model to MLflow
                        try:
                            mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path="trained_model",
                                signature=model_signature,
                                input_example=X_val.iloc[:3] if hasattr(X_val, "iloc") else X_val[:3]
                            )
                        except Exception as err:
                            logger.warning(f"MLflow model logging failed: {err}")
                        
                        # Evaluate and log metrics
                        performance = self.compute_metrics(model, X_val, y_val)
                        for metric_name, value in performance.items():
                            try:
                                mlflow.log_metric(metric_name, value)
                            except Exception as err:
                                logger.warning(f"Could not log metric {metric_name}: {err}")
                        
                        # Update best model selection
                        self._update_model_selection(model, model_name, performance)
                        
                        # Store run ID if this is the best
                        if self.selected_model_name == model_name:
                            try:
                                self.selected_run_id = mlflow.active_run().info.run_id
                            except Exception as err:
                                logger.warning(f"Could not get run ID: {err}")
                except Exception as err:
                    logger.error(f"MLflow training error: {err}")
                    # Fallback to local training
                    model.fit(X_train, y_train)
                    performance = self.compute_metrics(model, X_val, y_val)
                    self._update_model_selection(model, model_name, performance)
            else:
                # Local training without MLflow
                model.fit(X_train, y_train)
                performance = self.compute_metrics(model, X_val, y_val)
                self._update_model_selection(model, model_name, performance)
        
        # Register model if MLflow available
        self._register_selected_model()
        
        logger.info(f"Selected model: {self.selected_model_name} ({self.optimize_for}: {self.selected_score:.4f})")
        return self.selected_model, self.selected_model_name
    

    def _update_model_selection(self, model, name: str, metrics: Dict):
        current_value = metrics[self.optimize_for]
        
        # Determine if better based on metric type
        if self.optimize_for in ["mae", "rmse"]:
            is_improvement = current_value < self.selected_score
        else:  # r2
            is_improvement = current_value > self.selected_score
        
        if is_improvement:
            self.selected_model = model
            self.selected_model_name = name
            self.selected_score = current_value
            logger.info(f"New best model: {name} ({self.optimize_for}: {current_value:.4f})")
    
    # New method for model registration
    def _register_selected_model(self):
        if not self.mlflow_enabled or not self.selected_run_id:
            return
        
        try:
            model_path = f"runs:/{self.selected_run_id}/trained_model"
            registered_as = f"Taxi_Duration_{self.selected_model_name.replace(' ', '_')}"
            mlflow.register_model(model_path, registered_as)
            logger.info(f"Registered model '{registered_as}' from run {self.selected_run_id}")
        except Exception as err:
            logger.warning(f"Model registration failed: {err}")
  
    def load_registered_model(self, model_identifier: str, version_stage: str = "None") -> object:
        """Load a model from MLflow Model Registry"""
        model_location = f"models:/{model_identifier}/{version_stage}"
        return mlflow.sklearn.load_model(model_location)
    

    def store_model(self, model, filepath: str = "selected_model.pkl") -> str:
        joblib.dump(model, filepath)
        logger.info(f"Model stored at: {filepath}")
        return filepath
    
    @staticmethod
    def retrieve_model(filepath: str = "selected_model.pkl"):
        return joblib.load(filepath)
