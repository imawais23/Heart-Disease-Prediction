import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def perform_eda(df: pd.DataFrame, out_dir: Path):
    """
    Perform Exploratory Data Analysis and save plots.
    """
    print("Performing EDA...")
    
    # Data Description
    stats = df.describe(include='all').transpose()
    stats.to_csv(out_dir / "data_description.csv")
    print(f"Saved data description to {out_dir}/data_description.csv")
    
    # Numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Histograms
    for col in numeric:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png")
        plt.close()
        
    print(f"Saved histograms to {out_dir}/")
    
    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df[numeric].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png")
    plt.close()
    print(f"Saved correlation matrix to {out_dir}/correlation_matrix.png")
    
    # Pairplot (subset if too many columns)
    if len(numeric) <= 10:
        sns.pairplot(df[numeric])
        plt.savefig(out_dir / "pairplot.png")
        plt.close()
        print(f"Saved pairplot to {out_dir}/pairplot.png")
