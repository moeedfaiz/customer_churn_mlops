# src/serve_api.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# ======= Load model =======
model_path = "models/random_forest.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at path: {model_path}")

model = joblib.load(model_path)

# ======= Define input schema =======
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ======= Root route =======
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running!"}

# ======= Prediction route =======
@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # NOTE: Replace this with your actual preprocessing pipeline
    input_df_encoded = pd.get_dummies(input_df)

    # Align columns with training
    model_columns = model.feature_names_in_
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df_encoded)[0]
    return {"churn_prediction": "Yes" if prediction == 1 else "No"}

# ======= Run app =======
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
