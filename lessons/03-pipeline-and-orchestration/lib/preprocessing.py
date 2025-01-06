import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.config = Config()
    
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.RAW_DATA_PATH)
        return self.clean_data(df)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna()
        df = df.drop_duplicates()
        df['type'] = (df['type'] == 'red').astype(int)
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        return train_df, test_df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = df[self.config.FEATURE_COLUMNS]
        if fit:
            return pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.config.FEATURE_COLUMNS
            )
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=self.config.FEATURE_COLUMNS
        )