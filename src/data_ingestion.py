# customer_churn_mlops/src/data_ingestion.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(file_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("[INFO] Data split into train and test sets.")

if __name__ == "__main__":
    RAW_DATA_PATH = "C:/Users/DELL/Desktop/customer_churn_mlops/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "data/processed"
    load_and_split_data(RAW_DATA_PATH, OUTPUT_DIR)

