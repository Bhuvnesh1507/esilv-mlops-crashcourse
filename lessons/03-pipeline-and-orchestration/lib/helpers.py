import pickle
from pathlib import Path
from typing import Any

def save_pickle(obj: Any, path: Path) -> None:
    """Save object to pickle file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: Path) -> Any:
    """Load object from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_directories(config) -> None:
    """Create necessary directories"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)