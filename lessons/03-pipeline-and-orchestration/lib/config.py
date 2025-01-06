from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    
    # Data
    RAW_DATA_PATH = DATA_DIR / "winequalityN.csv"
    TRAIN_DATA_PATH = DATA_DIR / "train.csv"
    TEST_DATA_PATH = DATA_DIR / "test.csv"
    
    # Model
    MODEL_PATH = MODEL_DIR / "wine_model.pkl"
    SCALER_PATH = MODEL_DIR / "wine_scaler.pkl"
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # MLflow
    EXPERIMENT_NAME = "wine-quality-prediction"
    MODEL_NAME = "wine_quality_predictor"
    
    # Features
    TARGET_COLUMN = "quality"
    FEATURE_COLUMNS = [
        "type", "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
    ]