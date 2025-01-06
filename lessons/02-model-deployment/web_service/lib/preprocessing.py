from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
from app_config import FEATURE_COLUMNS
 
 
def preprocess_wine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess wine data by converting type to numeric and handling missing values
    """
    df = df.copy()
    # Rename columns to match input format
    column_mapping = {
        'fixed_acidity': 'fixed acidity',
        'volatile_acidity': 'volatile acidity', 
        'citric_acid': 'citric acid',
        'residual_sugar': 'residual sugar',
        'free_sulfur_dioxide': 'free sulfur dioxide',
        'total_sulfur_dioxide': 'total sulfur dioxide'
    }
    df = df.rename(columns=column_mapping)
    df['type'] = (df['type'] == 'red').astype(int)
    df = df.fillna(df.mean())
    return df[FEATURE_COLUMNS]