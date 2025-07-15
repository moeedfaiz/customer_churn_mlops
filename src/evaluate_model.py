import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
    df = pd.read_csv("data/processed/test.csv")
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    df = preprocess(df)
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    model = joblib.load("models/random_forest.pkl")
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    print("[INFO] Accuracy:", acc)
    print("[INFO] F1 Score:", f1)
    print("[INFO] Classification Report:\n", classification_report(y, preds))

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
