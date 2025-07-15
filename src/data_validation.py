import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

def validate_data(df: pd.DataFrame):
    schema = DataFrameSchema({
        "customerID": Column(str),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.ge(0)),
        "PhoneService": Column(str),
        "MultipleLines": Column(str),
        "InternetService": Column(str),
        "OnlineSecurity": Column(str),
        "OnlineBackup": Column(str),
        "DeviceProtection": Column(str),
        "TechSupport": Column(str),
        "StreamingTV": Column(str),
        "StreamingMovies": Column(str),
        "Contract": Column(str),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str),
        "MonthlyCharges": Column(float, Check.ge(0)),
        "TotalCharges": Column(float, Check.ge(0)),  # ✅ Fixed type
        "Churn": Column(str, Check.isin(["Yes", "No"]))
    })
    return schema.validate(df)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")

    # Clean TotalCharges column: remove blanks and convert to float
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    validated_df = validate_data(df)
    print("[INFO] Data validation passed.")
