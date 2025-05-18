# Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to churn using multiple machine learning models and tracks experiments with MLflow.

## 🔍 Problem Statement

Given a dataset of bank customer records, predict the probability of customer churn (exiting the bank). The goal is to identify patterns that signal potential churn to enable proactive retention strategies.

## 📁 Dataset

The dataset is from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data) and contains features such as:
- CreditScore, Geography, Gender, Age, Tenure
- Balance, NumOfProducts, HasCrCard, IsActiveMember
- EstimatedSalary, and Exited (target)

## 🧠 Models Used

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost Classifier**

## ⚙️ Workflow

1. Preprocess data:
   - Downsample to handle class imbalance
   - Encode categorical variables
   - Scale numeric features

2. Train models and evaluate using:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix

3. Log experiments and artifacts with MLflow:
   - Models
   - Metrics
   - Parameters
   - Preprocessor and plots

## 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run training:
    ```bash
    python train.py
    ```

3. Launch MLflow UI:
    ```bash
    mlflow ui
    ```
    Visit: http://127.0.0.1:5000 to track experiments.

## 📂 Project Structure
.
├── dataset/
│ └── Churn_Modelling.csv
├── src
│ └── data_utils.py
│ └── mlflow_utils.py
│ └── model_utils.py
│ └── train.py
│ └── __init__.py
├── requirements.txt
├── README.md
