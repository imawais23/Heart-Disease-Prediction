import sys
from pathlib import Path

# Add the current directory to sys.path to ensure src can be imported
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from src.data_loader import load_data
from src.preprocessing import clean_data, prepare_features
from src.eda import perform_eda
from src.model import train_logistic_regression, train_decision_tree
from src.evaluation import evaluate_model, plot_roc_curve, save_feature_importance

def main():
    # Setup paths
    DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"
    OUT_DIR = ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    
    print("Starting Heart Disease Project Pipeline...")
    
    # 1. Load Data
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Dataset shape: {df.shape}")
    
    # 2. Clean Data
    df = clean_data(df)
    
    # 3. EDA
    perform_eda(df, OUT_DIR)
    
    # 4. Preprocessing & Feature Engineering
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = prepare_features(df, OUT_DIR)
    
    # 5. Model Training & Evaluation
    results = {}
    
    # Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train, OUT_DIR)
    results['Logistic Regression'] = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression", OUT_DIR)
    save_feature_importance(lr_model, feature_names, "Logistic Regression", OUT_DIR)
    
    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train, OUT_DIR)
    results['Decision Tree'] = evaluate_model(dt_model, X_test, y_test, "Decision Tree", OUT_DIR)
    save_feature_importance(dt_model, feature_names, "Decision Tree", OUT_DIR)
    
    # 6. Compare Models
    plot_roc_curve(results, OUT_DIR)
    
    # Save Summary
    with open(OUT_DIR / "results_summary.txt", "w") as f:
        for name, res in results.items():
            f.write(f"{name} - Accuracy: {res['acc']:.4f}, AUC: {res['auc']:.4f}\n")
            
    print("Pipeline completed successfully. Check outputs/ directory.")

if __name__ == "__main__":
    main()
