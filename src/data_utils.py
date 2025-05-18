import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def preprocess_data(df):
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def rebalance_data(X, y):
    df = pd.DataFrame(X)
    df["Exited"] = y.values
    majority = df[df.Exited == 0]
    minority = df[df.Exited == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced_df = pd.concat([majority, minority_upsampled])
    X_balanced = balanced_df.drop("Exited", axis=1)
    y_balanced = balanced_df["Exited"]
    return X_balanced, y_balanced
