import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "customerID" in cat_cols: cat_cols.remove("customerID")
    if "Churn" in cat_cols: cat_cols.remove("Churn")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    df = preprocess(df)
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    mlflow.set_experiment("Customer-Churn-Prediction")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("models", exist_ok=True)
        model_path = "models/random_forest.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "random_forest_model")

        print("[INFO] Classification Report:")
        print(classification_report(y_val, preds))
