from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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


def test_predict_missing_field_returns_422():
    payload = {"age": 30, "income": 45000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    