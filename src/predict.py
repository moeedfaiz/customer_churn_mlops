import pandas as pd
import joblib
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
    input_path = "data/processed/manualtest.csv"         # You can manually place new samples here
    output_path = "data/predictions.csv"
    
    df = pd.read_csv(input_path)
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    customer_ids = df["customerID"]
    df = preprocess(df)

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)
    
    df = df.drop("customerID", axis=1)
    
    model = joblib.load("models/random_forest.pkl")
    predictions = model.predict(df)

    output_df = pd.DataFrame({
        "customerID": customer_ids,
        "Churn_Predicted": ["Yes" if pred == 1 else "No" for pred in predictions]
    })

    output_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")
