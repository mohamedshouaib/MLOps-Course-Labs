import os
os.environ["LOGNAME"] = "Shouaib"

import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split

from data_utils import preprocess_data, rebalance_data
from model_utils import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    plot_confusion_matrix,
)
from mlflow_utils import log_metrics_and_artifacts

def main():
    os.makedirs("artifacts", exist_ok=True)
    mlflow.set_tracking_uri("http://172.17.0.1:5000")
    mlflow.set_experiment("Churn Prediction")

    df = pd.read_csv("/media/shouaib/Work/2025/ITI/31_MLOps/Lab01/MLOps-Course-Labs/dataset/Churn_Modelling.csv")

    X, y = preprocess_data(df)
    X_bal, y_bal = rebalance_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    lr_model, lr_report, lr_matrix = train_logistic_regression(X_train, y_train, X_test, y_test)
    lr_fig = plot_confusion_matrix(lr_matrix, "Logistic Regression")
    log_metrics_and_artifacts(lr_model, "logistic", lr_report, lr_fig, X_test[:5], y_test[:5])

    rf_model, rf_report, rf_matrix = train_random_forest(X_train, y_train, X_test, y_test)
    rf_fig = plot_confusion_matrix(rf_matrix, "Random Forest")
    log_metrics_and_artifacts(rf_model, "random_forest", rf_report, rf_fig, X_test[:5], y_test[:5])
    
    xgb_model, xgb_report, xgb_matrix = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_fig = plot_confusion_matrix(xgb_matrix, "XGBoost")
    log_metrics_and_artifacts(xgb_model, "xgboost", xgb_report, xgb_fig, X_test[:5], y_test[:5])

if __name__ == "__main__":
    main()
