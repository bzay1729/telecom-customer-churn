from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")


    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 448.75,        
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "decision_threshold" in data
    assert "prediction" in data

    assert 0 <= data["churn_probability"] <= 1
    assert data["decision_threshold"] == 0.35
    assert data["prediction"] in [
        "Churn",
        "No Churn"
    ]

def test_invalid_customer():
    invalid_customer = {
        "gender": "Invalid",
        "SeniorCitizen":5,
    }

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422