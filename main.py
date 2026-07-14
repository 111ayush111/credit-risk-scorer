# import os
# from google import genai
# # for gemni

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import joblib

# # ----------------------------------------------------
# try:
#     gemini_client = genai.Client()
# except Exception as e:
#     print("Warning: Gemini Client could not initialize. Check GEMINI_API_KEY environment variable.", e)
#     gemini_client = None

# # --------------------------------------------------------

# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")
# features = joblib.load("features.pkl")

# app = FastAPI(title="Explainable Credit Risk Scorer")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class Applicant(BaseModel):
#     age: int
#     income: float
#     loan_amount: float
#     credit_history_years: int
#     existing_loans: int
#     employment_years: int
#     debt_to_income: float


# @app.get("/")
# def health_check():
#     return {"status": "ok", "message": "Credit Risk Scorer API is running"}


# @app.post("/predict")
# def predict(applicant: Applicant):
#     input_dict = applicant.model_dump()
#     input_df = pd.DataFrame([input_dict])[features]

#     scaled_input = scaler.transform(input_df)
#     probability = model.predict_proba(scaled_input)[0][1]
#     prediction = "High Risk" if probability > 0.5 else "Low Risk"

#     contributions = model.coef_[0] * scaled_input[0]
#     explanation = []
#     for feat, contrib in zip(features, contributions):
#         direction = "increases risk" if contrib > 0 else "reduces risk"
#         explanation.append({
#             "feature": feat,
#             "effect": direction,
#             "impact": round(float(abs(contrib)), 3)
#         })
#     explanation.sort(key=lambda x: x["impact"], reverse=True)
#     top_reasons = explanation[:3]
#     explanation = top_reasons
    
# # ---------------------------------------------------------------
#     gemini_recommendation = "Gemini integration is not configured."
#     if gemini_client:
#         prompt = f"""
#         You are an expert financial and credit risk advisor. 
#         A customer applied for a loan. Here are the details:
#         - Age: {applicant.age}
#         - Monthly Income: {applicant.income} INR
#         - Requested Loan Amount: {applicant.loan_amount} INR
#         - Credit History: {applicant.credit_history_years} years
#         - Existing Loans: {applicant.existing_loans}
#         - Employment: {applicant.employment_years} years
#         - Debt to Income Ratio: {applicant.debt_to_income}
        
#         Our machine learning model classified this applicant as: {prediction} (Probability of default: {round(probability*100, 2)}%).
#         The mathematical reasons for this decision are:
#         1. {explanation[0]['feature']} ({explanation[0]['effect']})
#         2. {explanation[1]['feature']} ({explanation[1]['effect']})
        
#         Write a very short, polite, and constructive recommendation (3-4 sentences maximum) in Hinglish.
#         If they are 'High Risk', tell them what 1 or 2 specific steps they can take to improve their chances (e.g., reduce existing loans, increase down payment).
#         If they are 'Low Risk', congratulate them and give a quick tip on maintaining their good credit score.
#         """
#         try:
#             response = gemini_client.models.generate_content(
#                 model="gemini-2.5-flash",
#                 contents=prompt
#             )
#             gemini_recommendation = response.text
#         except Exception as api_err:
#             gemini_recommendation = f"Failed to fetch Gemini insights: {str(api_err)}"

#     return {
#         "risk_probability": round(float(probability), 3),
#         "prediction": prediction,
#         "top_reasons": explanation[:3],
#         "gemini_recommendation": gemini_recommendation  # <-- Ye key add karo
#     }
# # ---------------------------------------------------------------


#     # return {
#     #     "risk_probability": round(float(probability), 3),
#     #     "prediction": prediction,
#     #     "top_reasons": explanation[:3]
#     # }


import os
from google import genai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# ----------------------------------------------------
try:
    gemini_client = genai.Client()
except Exception as e:
    print("Warning: Gemini Client could not initialize. Check GEMINI_API_KEY environment variable.", e)
    gemini_client = None

# --------------------------------------------------------

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
    
    # Sorting and Slicing safely
    explanation.sort(key=lambda x: x["impact"], reverse=True)
    top_reasons = explanation[:3]
    
    # SAFE INDEXING: Fallback content setup agar list empty ho toh handle karne ke liye
    reason_1 = top_reasons[0]['feature'] + " (" + top_reasons[0]['effect'] + ")" if len(top_reasons) > 0 else "N/A"
    reason_2 = top_reasons[1]['feature'] + " (" + top_reasons[1]['effect'] + ")" if len(top_reasons) > 1 else "N/A"
    
# ---------------------------------------------------------------
    gemini_recommendation = "Gemini integration is not configured."
    if gemini_client:
        prompt = f"""
        You are an expert financial and credit risk advisor. 
        A customer applied for a loan. Here are the details:
        - Age: {applicant.age}
        - Monthly Income: {applicant.income} INR
        - Requested Loan Amount: {applicant.loan_amount} INR
        - Credit History: {applicant.credit_history_years} years
        - Existing Loans: {applicant.existing_loans}
        - Employment: {applicant.employment_years} years
        - Debt to Income Ratio: {applicant.debt_to_income}
        
        Our machine learning model classified this applicant as: {prediction} (Probability of default: {round(probability*100, 2)}%).
        The mathematical reasons for this decision are:
        1. {reason_1}
        2. {reason_2}
        
        Write a very short, polite, and constructive recommendation (3-4 sentences maximum) in Hinglish.
        If they are 'High Risk', tell them what 1 or 2 specific steps they can take to improve their chances (e.g., reduce existing loans, increase down payment).
        If they are 'Low Risk', congratulate them and give a quick tip on maintaining their good credit score.
        """
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            gemini_recommendation = response.text
        except Exception as api_err:
            gemini_recommendation = f"Failed to fetch Gemini insights: {str(api_err)}"

    return {
        "risk_probability": round(float(probability), 3),
        "prediction": prediction,
        "top_reasons": top_reasons, # <-- Clean variable reference pass karo
        "gemini_recommendation": gemini_recommendation
    }