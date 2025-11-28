# Heart Disease Prediction Project

## 1. Task Objective
The primary objective of this project is to develop a machine learning model capable of predicting the risk of heart disease in patients based on their clinical health data. This involves:
- Cleaning and preprocessing real-world medical data.
- Performing Exploratory Data Analysis (EDA) to identify trends.
- Training and evaluating classification models.
- Deploying the solution as an interactive web application.

## 2. Dataset Used
- **Name**: UCI Heart Disease Dataset (Cleveland)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Description**: The dataset contains 303 instances with 14 attributes, including age, sex, chest pain type, blood pressure, and cholesterol levels. The target variable indicates the presence of heart disease (0 = No Disease, 1-4 = Disease).

## 3. Models Applied
Two classification algorithms were implemented and compared:
1.  **Logistic Regression**: A linear model chosen for its interpretability and efficiency in binary classification.
2.  **Decision Tree Classifier**: A non-linear model used to capture complex decision boundaries and interactions between features.

Both models were optimized using **GridSearchCV** to find the best hyperparameters.

## 4. Key Results and Findings
The models were evaluated using Accuracy and ROC-AUC scores.

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| **Logistic Regression** | **83.15%** | **0.8989** |
| Decision Tree | 77.72% | 0.8535 |

**Key Findings:**
- **Logistic Regression outperformed the Decision Tree**, achieving a higher accuracy and AUC score.
- **Top Risk Factors**: Feature importance analysis revealed that **Chest Pain Type (cp)**, **Max Heart Rate (thalch)**, and **ST Depression (oldpeak)** are the most significant predictors of heart disease.
- **Data Quality**: Imputing missing values with the median (for numeric) and mode (for categorical) significantly improved model stability.

---

## Project Contents
- `heart_disease_prediction.ipynb` - **Jupyter Notebook** containing the full analysis (Submission File).
- `app.py` - **Streamlit Web App** for interactive predictions.
- `main.py` - Main pipeline script (Data Loading -> Preprocessing -> Modeling -> Evaluation).
- `predict_demo.py` - Demo script to load saved models and make predictions.
- `data/heart_disease_uci.csv` - The raw dataset.
- `outputs/` - Directory containing saved models (`.joblib`), plots, and metrics.

## How to Run
### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Pipeline
Train the models and generate results:
```bash
python3 main.py
```

### 3. Run the Web App (Deployment)
Launch the interactive dashboard:
```bash
streamlit run app.py
```
