import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

age = np.random.randint(21, 65, n_samples)
income = np.random.normal(50000, 20000, n_samples).clip(10000, 200000)
loan_amount = np.random.normal(15000, 8000, n_samples).clip(1000, 60000)
credit_history_years = np.random.randint(0, 25, n_samples)
existing_loans = np.random.randint(0, 5, n_samples)
employment_years = np.random.randint(0, 30, n_samples)
debt_to_income = (loan_amount / income).clip(0, 3)

risk_score = (
    -0.00002 * income
    + 0.00008 * loan_amount
    - 0.05 * credit_history_years
    + 0.3 * existing_loans
    - 0.04 * employment_years
    + 2.5 * debt_to_income
    - 0.01 * age
    + np.random.normal(0, 0.5, n_samples)
)

threshold = np.percentile(risk_score, 70)
default = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.round(2),
    "loan_amount": loan_amount.round(2),
    "credit_history_years": credit_history_years,
    "existing_loans": existing_loans,
    "employment_years": employment_years,
    "debt_to_income": debt_to_income.round(3),
    "default": default
})

df.to_csv("credit_data.csv", index=False)
print("Dataset saved:", df.shape)
print(df["default"].value_counts(normalize=True))