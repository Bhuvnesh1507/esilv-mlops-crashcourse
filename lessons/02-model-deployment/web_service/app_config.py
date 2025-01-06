# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR = f"local_models/wine_scaler__v{MODEL_VERSION}.pkl"
PATH_TO_MODEL = f"local_models/wine_model__v{MODEL_VERSION}.pkl"
FEATURE_COLUMNS = [
    "type", "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]
 
# MISC
APP_TITLE = "WineQualityPredictionApp"
APP_DESCRIPTION = (
    "A simple API to predict wine quality scores (1-10) "
    "based on physicochemical properties of the wine."
)
APP_VERSION = "0.0.1"