from mlflow.tracking import MlflowClient
import mlflow
from config import Config

class ModelDeployment:
    def __init__(self):
        self.client = MlflowClient()
        self.config = Config()
    
    def register_best_model(self) -> None:
        runs = mlflow.search_runs(
            experiment_names=[self.config.EXPERIMENT_NAME]
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found")
        
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri, 
            self.config.MODEL_NAME
        )
        
        self.client.transition_model_version_stage(
            name=self.config.MODEL_NAME,
            version=registered_model.version,
            stage="Production"
        )
    
    def load_production_model(self):
        return mlflow.sklearn.load_model(
            f"models:/{self.config.MODEL_NAME}/Production"
        )