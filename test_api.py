from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Health Check (Smoke testing taaki pata chale server up hai).
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# Happy Path (Jahan maine check kiya ki sahi input par API contract validation aur output structure perfectly maintain ho raha hai ya nahi).
def test_predict_returns_valid_structure():
    payload = {
        "age": 30,
        "income": 45000,
        "loan_amount": 12000,
        "credit_history_years": 5,
        "existing_loans": 1,
        "employment_years": 3,
        "debt_to_income": 0.27
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "risk_probability" in data
    assert "prediction" in data
    assert data["prediction"] in ["High Risk", "Low Risk"]
    assert 0 <= data["risk_probability"] <= 1
    assert len(data["top_reasons"]) == 3

# Negative Testing (Jahan bad payload bhej kar yeh verify kiya ki Pydantic core validation sahi se hit ho rahi hai aur API gracefully 422 error throw kar rahi hai bina crash hue).
def test_predict_missing_field_returns_422():
    payload = {"age": 30, "income": 45000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    