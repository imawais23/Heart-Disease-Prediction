import pandas as pd
from pathlib import Path

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the dataset from the given path.
    """
    if data_path.exists():
        print(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure the file exists.")
