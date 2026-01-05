from datetime import datetime
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to Python path for absolute imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Try absolute imports first
    from src.mlproject.preprocess.advanced import Preprocessor
    from src.mlproject.features.engineer import FeatureEngineer
    from src.mlproject.train.model_trainer import ModelTrainer
    from src.mlproject.inference.pipeline import InferencePipeline
    print("[INFO] Using absolute imports")
except ImportError:
    # Fallback to relative imports
    from ..preprocess.advanced import Preprocessor
    from ..features.engineer import FeatureEngineer
    from ..train.model_trainer import ModelTrainer
    from ..inference.pipeline import InferencePipeline
    print("[INFO] Using relative imports")

# Try to import OmegaConf, but provide fallback
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    print("[WARNING] OmegaConf not installed. Using simple config dictionary.")
    OMEGACONF_AVAILABLE = False
    OmegaConf = None


class TaxiPipeline:
    """
    Full ML pipeline for NYC Taxi Trip Duration:
    - Preprocessing
    - Feature engineering
    - Model training
    - Batch inference
    """

    def __init__(self, config_file: str = None):
        if config_file:
            print(f"[INFO] Loading config from {config_file}...")
            if OMEGACONF_AVAILABLE:
                self.cfg = OmegaConf.load(config_file)
            else:
                # Simple config fallback
                import yaml
                with open(config_file, 'r') as f:
                    self.cfg = yaml.safe_load(f)
        else:
            print("[INFO] Using default configuration")
            self.cfg = {
                'paths': {
                    'train_csv': 'data/train.csv',
                    'test_csv': 'data/test.csv',
                    'artifact_dir': 'models/',
                    'output_dir': 'outputs/'
                },
                'train': {
                    'metric': 'rmse',
                    'test_size': 0.2,
                    'seed': 42
                }
            }
        
        # Initialize components
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Get metric from config
        if isinstance(self.cfg, dict):
            metric = self.cfg.get('train', {}).get('metric', 'rmse')
        else:
            metric = self.cfg.train.metric
        
        self.model_trainer = ModelTrainer(metric=metric)
        self.inference = None
        
        print("[INFO] Pipeline initialized.")

    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV file."""
        print(f"[INFO] Loading data from {path}...")
        
        # Handle relative paths
        if not os.path.isabs(path):
            project_root = Path(__file__).parent.parent.parent.parent
            path = os.path.join(project_root, path)
        
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Run preprocessing."""
        print(f"[INFO] Preprocessing {'training' if is_train else 'validation/test'} data...")
        
        try:
            df_processed = self.preprocessor.run(df, is_train=is_train)
            print(f"[INFO] Preprocessing done. Data shape: {df_processed.shape}")
            return df_processed
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            # Fallback: basic cleaning
            df_processed = df.copy().dropna()
            print(f"[INFO] Using fallback preprocessing. Shape: {df_processed.shape}")
            return df_processed

    def feature_engineering(self, df: pd.DataFrame, fit: bool = False):
        """Create features."""
        print(f"[INFO] Applying feature engineering (fit={fit})...")
        
        try:
            X, y, _ = self.feature_engineer.transform(df, fit=fit, is_train=True)
            print(f"[INFO] Feature engineering done. Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
        except Exception as e:
            print(f"[ERROR] Feature engineering failed: {e}")
            # Fallback: use all columns except target
            if 'trip_duration' in df.columns:
                X = df.drop('trip_duration', axis=1)
                y = df['trip_duration']
            else:
                X = df
                y = None
            print(f"[INFO] Using fallback features. X: {X.shape}")
            return X, y

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train model."""
        print("[INFO] Training model...")
        
        try:
            model, best_model_name = self.model_trainer.train(X_train, y_train, X_valid, y_valid)
            self.inference = InferencePipeline(model=model)
            print(f"[INFO] Training complete. Best model: {best_model_name}")
            return model, best_model_name
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            # Fallback: simple model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            self.inference = InferencePipeline(model=model)
            print("[INFO] Using fallback RandomForest model")
            return model, "random_forest_fallback"

    def batch_inference(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        """Run batch inference."""
        if self.inference is None:
            raise RuntimeError("No trained model for inference")
        
        print("[INFO] Running batch inference...")
        
        try:
            output_df = self.inference.run(df, fit=True, is_train=False)
            print(f"[INFO] Batch inference complete. Output shape: {output_df.shape}")
            
            if save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
                
                output_file = os.path.join(save_path, f"predictions_{timestamp}.csv")
                output_df.to_csv(output_file, index=False)
                print(f"[INFO] Predictions saved to {output_file}")
            
            return output_df
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            raise

    def run(self):
        """Run the complete pipeline."""
        print("=" * 60)
        print("[INFO] STARTING FULL PIPELINE EXECUTION")
        print("=" * 60)
        
        try:
            # Get paths from config
            if isinstance(self.cfg, dict):
                train_path = self.cfg['paths']['train_csv']
                test_path = self.cfg['paths']['test_csv']
                artifact_dir = self.cfg['paths']['artifact_dir']
                output_dir = self.cfg['paths']['output_dir']
                test_size = self.cfg['train']['test_size']
                random_state = self.cfg['train']['seed']
            else:
                train_path = self.cfg.paths.train_csv
                test_path = self.cfg.paths.test_csv
                artifact_dir = self.cfg.paths.artifact_dir
                output_dir = self.cfg.paths.output_dir
                test_size = self.cfg.train.test_size
                random_state = self.cfg.train.seed

            # 1. Load data
            print("\n[STEP 1] LOADING DATA")
            train_df = self.load_data(train_path)
            test_df = self.load_data(test_path)

            # 2. Split train/valid
            print("\n[STEP 2] SPLITTING DATA")
            train_split, valid_split = train_test_split(
                train_df,
                test_size=test_size,
                random_state=random_state
            )
            print(f"   Train split: {len(train_split)} rows")
            print(f"   Validation split: {len(valid_split)} rows")
            print(f"   Test data: {len(test_df)} rows")

            # 3. Preprocess
            print("\n[STEP 3] PREPROCESSING")
            train_processed = self.preprocess(train_split, is_train=True)
            valid_processed = self.preprocess(valid_split, is_train=True)
            test_processed = self.preprocess(test_df, is_train=False)

            # 4. Feature engineering
            print("\n[STEP 4] FEATURE ENGINEERING")
            X_train, y_train = self.feature_engineering(train_processed, fit=True)
            X_valid, y_valid = self.feature_engineering(valid_processed, fit=False)
            X_test, _ = self.feature_engineering(test_processed, fit=False)

            # 5. Training
            print("\n[STEP 5] MODEL TRAINING")
            model, best_model_name = self.train(X_train, y_train, X_valid, y_valid)

            # 6. Save model
            print("\n[STEP 6] SAVING MODEL")
            os.makedirs(artifact_dir, exist_ok=True)
            
            # Check if save_model is a class method or static method
            try:
                self.model_trainer.save_model(model, os.path.join(artifact_dir, "best_model.pkl"))
            except TypeError:
                # If it's a static method
                ModelTrainer.save_model(model, os.path.join(artifact_dir, "best_model.pkl"))
            
            print(f"[INFO] Model saved to {artifact_dir}best_model.pkl")
            print(f"[INFO] Model name: NYC_Taxi_{best_model_name}")

            # 7. Batch inference
            print("\n[STEP 7] BATCH INFERENCE")
            os.makedirs(output_dir, exist_ok=True)
            output_df = self.batch_inference(X_test, save_path=output_dir)

            print("=" * 60)
            print("[INFO] PIPELINE RUN COMPLETE!")
            print(f"   Best model: {best_model_name}")
            print(f"   Predictions saved to: {output_dir}")
            print("=" * 60)
            
            return model, best_model_name, output_df
            
        except Exception as e:
            print("=" * 60)
            print("[ERROR] PIPELINE FAILED!")
            print(f"Error: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NYC Taxi Pipeline')
    parser.add_argument('--config', type=str, default='configs/pipeline.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    pipeline = TaxiPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
