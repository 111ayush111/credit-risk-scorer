from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

app = FastAPI(title="Explainable Credit Risk Scorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Applicant(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_history_years: int
    existing_loans: int
    employment_years: int
    debt_to_income: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Credit Risk Scorer API is running"}


@app.post("/predict")
def predict(applicant: Applicant):
    input_dict = applicant.model_dump()
    input_df = pd.DataFrame([input_dict])[features]

    scaled_input = scaler.transform(input_df)
    probability = model.predict_proba(scaled_input)[0][1]
    prediction = "High Risk" if probability > 0.5 else "Low Risk"

    contributions = model.coef_[0] * scaled_input[0]
    explanation = []
    for feat, contrib in zip(features, contributions):
        direction = "increases risk" if contrib > 0 else "reduces risk"
        explanation.append({
            "feature": feat,
            "effect": direction,
            "impact": round(float(abs(contrib)), 3)
        })
    explanation.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "risk_probability": round(float(probability), 3),
        "prediction": prediction,
        "top_reasons": explanation[:3]
    }