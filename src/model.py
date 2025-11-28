from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

def train_logistic_regression(X_train, y_train, out_dir: Path):
    """
    Train Logistic Regression with Hyperparameter Tuning.
    """
    print("Training Logistic Regression...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    
    # Adjust CV for small datasets
    n_samples = len(X_train)
    cv = 5 if n_samples >= 5 else 2
    
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=cv, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Logistic Regression Params: {grid.best_params_}")
    
    joblib.dump(best_model, out_dir / "model_logistic.joblib")
    print(f"Saved Logistic Regression model to {out_dir}/model_logistic.joblib")
    
    return best_model

def train_decision_tree(X_train, y_train, out_dir: Path):
    """
    Train Decision Tree with Hyperparameter Tuning.
    """
    print("Training Decision Tree...")
    param_grid = {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]}
    
    # Adjust CV for small datasets
    n_samples = len(X_train)
    cv = 5 if n_samples >= 5 else 2
    
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Decision Tree Params: {grid.best_params_}")
    
    joblib.dump(best_model, out_dir / "model_tree.joblib")
    print(f"Saved Decision Tree model to {out_dir}/model_tree.joblib")
    
    return best_model
