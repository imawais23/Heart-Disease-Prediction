import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import pandas as pd
from pathlib import Path

def evaluate_model(model, X_test, y_test, model_name: str, out_dir: Path):
    """
    Evaluate a model and return metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} -- Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}")
    
    # Save Confusion Matrix
    plt.figure()
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    return {
        'model': model,
        'acc': acc,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'cm': cm
    }

def plot_roc_curve(results: dict, out_dir: Path):
    """
    Plot ROC curve comparison.
    """
    plt.figure()
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} AUC={res['auc']:.3f}")
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_compare.png")
    plt.close()
    print(f"Saved ROC comparison to {out_dir}/roc_compare.png")

def save_feature_importance(model, feature_names, model_name: str, out_dir: Path):
    """
    Save feature importance if applicable.
    """
    if hasattr(model, "coef_"):
        # Linear models
        coef = model.coef_.ravel()
        fi = pd.DataFrame({"feature": feature_names, "coefficient": coef})
        fi['abs_coeff'] = fi['coefficient'].abs()
        fi = fi.sort_values(by='abs_coeff', ascending=False)
    elif hasattr(model, "feature_importances_"):
        # Tree models
        fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        fi = fi.sort_values(by='importance', ascending=False)
    else:
        print(f"Model {model_name} does not expose feature importance.")
        return

    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
    fi.to_csv(out_dir / filename, index=False)
    print(f"Saved feature importance to {out_dir}/{filename}")
