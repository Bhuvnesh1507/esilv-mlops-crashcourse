from typing import List
 
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
 
from lib.models import WineData
from lib.preprocessing import preprocess_wine_data
 
 
def run_inference(input_data: List[WineData], scaler: StandardScaler, model: BaseEstimator) -> np.ndarray:
    """Run inference on a list of wine data inputs.
 
    Args:
        input_data (List[WineData]): List of wine data points to run inference on.
        scaler (StandardScaler): The fitted StandardScaler object.
        model (BaseEstimator): The fitted model object.
 
    Returns:
        np.ndarray: The predicted wine quality scores.
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    df = preprocess_wine_data(df)
    X = scaler.transform(df)
    predictions = model.predict(X)
    logger.info(f"Predicted wine quality scores:\n{predictions}")
    return predictions