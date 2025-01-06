from preprocessing import DataPreprocessor
from modeling import WineQualityModel
from deployment import ModelDeployment
from helpers import save_pickle, create_directories
from config import Config

def train_pipeline():
    # Initialize
    config = Config()
    create_directories(config)
    preprocessor = DataPreprocessor()
    model = WineQualityModel()
    
    # Data preprocessing
    df = preprocessor.load_data()
    train_df, test_df = preprocessor.split_data(df)
    
    X_train = preprocessor.prepare_features(train_df, fit=True)
    y_train = train_df[config.TARGET_COLUMN]
    
    X_test = preprocessor.prepare_features(test_df)
    y_test = test_df[config.TARGET_COLUMN]
    
    # Model training and logging
    model.log_mlflow(X_train, X_test, y_train, y_test)
    
    # Save artifacts
    save_pickle(preprocessor.scaler, config.SCALER_PATH)
    save_pickle(model.model, config.MODEL_PATH)
    
    return model, preprocessor

def deploy_pipeline():
    deployment = ModelDeployment()
    deployment.register_best_model()
    return deployment.load_production_model()