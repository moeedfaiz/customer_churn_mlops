import pandas as pd
import joblib
import gradio as gr

# Load the trained model
model = joblib.load("models/random_forest.pkl")

def predict_churn(gender, senior, partner, dependents, tenure, phone, multiple, internet, online_sec,
                  online_backup, device_protect, tech_support, streaming_tv, streaming_movies,
                  contract, paperless, payment, monthly, total):

    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }])

    # Label encode like before
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype("category").cat.codes

    prediction = model.predict(input_df)[0]
    return "Churn" if prediction == 1 else "No Churn"

# UI with Gradio
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["0", "1"], label="Senior Citizen"),
        gr.Radio(["Yes", "No"], label="Partner"),
        gr.Radio(["Yes", "No"], label="Dependents"),
        gr.Number(label="Tenure (months)"),
        gr.Radio(["Yes", "No"], label="Phone Service"),
        gr.Textbox(label="Multiple Lines"),
        gr.Textbox(label="Internet Service"),
        gr.Textbox(label="Online Security"),
        gr.Textbox(label="Online Backup"),
        gr.Textbox(label="Device Protection"),
        gr.Textbox(label="Tech Support"),
        gr.Textbox(label="Streaming TV"),
        gr.Textbox(label="Streaming Movies"),
        gr.Textbox(label="Contract"),
        gr.Radio(["Yes", "No"], label="Paperless Billing"),
        gr.Textbox(label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs="text",
    title="Customer Churn Predictor"
)

if __name__ == "__main__":
    demo.launch()
