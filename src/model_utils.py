from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds)
    return model, report, matrix

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds)
    return model, report, matrix

def train_xgboost(X_train, y_train, X_test, y_test):
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    
    model = XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False  
    )
    model.fit(X_train_np, y_train_np)
    preds = model.predict(X_test_np)
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds)
    return model, report, matrix

def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig_path = f"artifacts/{title.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path
