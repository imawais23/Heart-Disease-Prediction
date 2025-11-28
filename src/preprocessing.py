import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Standardize column names
    - Drop irrelevant columns
    - Handle missing values (median for numeric, mode for categorical)
    """
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Drop irrelevant columns
    drop_cols = ['id', 'dataset']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
    # Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median)
                print(f"Filled missing values in {col} with median = {median}")
            else:
                mode = df[col].mode()[0]
                df[col] = df[col].fillna(mode)
                print(f"Filled missing values in {col} with mode = {mode}")
            
    return df

def prepare_features(df: pd.DataFrame, out_dir: Path):
    """
    Prepare features for modeling:
    - Identify target
    - One-hot encode categorical variables
    - Split into train/test
    - Scale features
    """
    # Target detection
    if 'target' in df.columns:
        y = df['target'].astype(int)
        # Convert to binary if multiclass (0=No Disease, >0=Disease)
        if y.nunique() > 2:
             print("Detected multiclass target. Converting to binary (0 vs >0).")
             y = (y > 0).astype(int)
        X = df.drop(columns=['target'])
    elif 'heartdisease' in df.columns:
        y = df['heartdisease'].astype(int)
        X = df.drop(columns=['heartdisease'])
    else:
        # Assume last column
        y = df.iloc[:,-1].astype(int)
        if y.nunique() > 2:
             print("Detected multiclass target in last column. Converting to binary (0 vs >0).")
             y = (y > 0).astype(int)
        X = df.iloc[:,:-1]
        
    # One-hot encoding
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)
        
    # Split
    # Check if we can stratify
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        print("Warning: Some classes have fewer than 2 samples. Disabling stratification.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, out_dir / "scaler.joblib")
    print(f"Saved scaler to {out_dir}/scaler.joblib")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
