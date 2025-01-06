from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import numpy as np
from typing import Dict, Tuple
import pandas as pd
from config import Config

class WineQualityModel:
    def __init__(self):
        self.model = LinearRegression(fit_intercept=True, n_jobs=-1)
        self.config = Config()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions)
        }
    
    def log_mlflow(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> None:
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        
        with mlflow.start_run():
            self.train(X_train, y_train)
            
            train_metrics = self.evaluate(X_train, y_train)
            test_metrics = self.evaluate(X_test, y_test)
            
            mlflow.log_metrics({
                'train_rmse': train_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_r2': train_metrics['r2'],
                'test_r2': test_metrics['r2']
            })
            
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("features", self.config.FEATURE_COLUMNS)